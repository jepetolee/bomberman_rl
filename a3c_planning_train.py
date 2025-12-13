#!/usr/bin/env python3
"""
A3C Training with Planning (Dyna-Q style)
==========================================

Architecture:
  - A3C structure with multiple workers
  - Each worker collects experiences with recurrent z states
  - Planning: Generate simulated experiences using environment model
  - Recurrent z management:
    * Initial state: zero vector
    * Subsequent states: previous inference vector
    * Planning: z flows through inference only (no gradient updates)

Key Features:
1. Recurrent latent state (z) management across timesteps
2. Planning generates simulated experiences using environment model
3. Planning z is detached (no gradient updates during planning)
4. Real experiences update z with gradients

Usage:
    python a3c_planning_train.py --num-workers 4 --total-rounds 50000 --planning-steps 100
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import queue
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

# Disable audio
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value, Lock
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
from agent_code.ppo_agent.models.environment_model import EnvironmentModel, create_env_model
from agent_code.ppo_agent.dyna_planning import DynaPlanner, VisitedStatesBuffer
from agent_code.ppo_agent.train_deepsup import train_policy_deepsup
from config.load_config import load_config, get_model_config


# ============== Configuration ==============

@dataclass
class PlanningA3CConfig:
    """Configuration for A3C with Planning"""
    num_workers: int = 4
    total_rounds: int = 50000
    rounds_per_batch: int = 5
    sync_interval: int = 40  # Sync every N batches
    results_dir: str = 'results/planning'
    model_path: str = 'ppo_model.pt'
    
    # Planning parameters
    planning_steps: int = 100  # Number of planning steps per batch
    planning_batch_size: int = 32
    use_planning: bool = True
    env_model_path: Optional[str] = None  # Path to trained environment model
    
    # Recurrent z management
    reset_z_on_round_start: bool = True  # Reset z to zero at round start
    
    # Training parameters
    learning_rate: float = 3e-4
    value_weight: float = 0.5
    real_weight: float = 1.0
    sim_weight: float = 0.5


# ============== Shared State ==============

class PlanningSharedState:
    """Shared state for A3C with Planning"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.lock = Lock()
        self.global_round = Value('i', 0)
        self.total_score = Value('f', 0.0)
        self.total_kills = Value('i', 0)
        self.total_deaths = Value('i', 0)
        self.total_coins = Value('i', 0)
        
        # Worker stats queues
        self.stats_queue = Queue()
        self.log_queue = Queue()
    
    def increment_round(self):
        with self.global_round.get_lock():
            self.global_round.value += 1
            return self.global_round.value
    
    def add_stats(self, score: float, kills: int, deaths: int, coins: int):
        with self.lock:
            self.total_score.value += score
            self.total_kills.value += kills
            self.total_deaths.value += deaths
            self.total_coins.value += coins
    
    def get_stats(self) -> Dict:
        return {
            "rounds": self.global_round.value,
            "score": self.total_score.value,
            "kills": self.total_kills.value,
            "deaths": self.total_deaths.value,
            "coins": self.total_coins.value,
        }


# ============== Model Creation ==============

def create_model(model_path: str, device: torch.device) -> nn.Module:
    """Create or load model"""
    config = load_config()
    model_config = get_model_config(config)
    
    # Create hybrid model
    model = PolicyValueViT_TRM_Hybrid(
        in_channels=model_config.get('in_channels', 10),
        num_actions=model_config.get('num_actions', 6),
        img_size=tuple(model_config.get('img_size', [17, 17])),
        embed_dim=model_config.get('embed_dim', 256),
        vit_depth=model_config.get('vit_depth', 2),
        vit_heads=model_config.get('vit_heads', 4),
        vit_mlp_ratio=model_config.get('vit_mlp_ratio', 4.0),
        vit_patch_size=model_config.get('vit_patch_size', 1),
        trm_n_latent=model_config.get('trm_n_latent', 4),
        trm_mlp_ratio=model_config.get('trm_mlp_ratio', 4.0),
        trm_drop=model_config.get('trm_drop', 0.0),
        trm_patch_size=model_config.get('trm_patch_size', 2),
        trm_patch_stride=model_config.get('trm_patch_stride', 1),
        use_ema=model_config.get('use_ema', True),
        ema_decay=model_config.get('ema_decay', 0.999),
    ).to(device)
    
    # Load if exists
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}, using fresh model")
    else:
        print(f"Creating new model (no existing model at {model_path})")
    
    return model


# ============== Planning Functions ==============

def plan_with_recurrent_z(
    model: PolicyValueViT_TRM_Hybrid,
    planner: DynaPlanner,
    n_planning_steps: int,
    batch_size: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, int, float, torch.Tensor, torch.Tensor]]:
    """
    Generate simulated experiences through planning with recurrent z
    
    Key: During planning, z flows through inference only (detached, no gradients)
    
    Returns:
        List of (state, action, reward, next_state, z_next) tuples
        where z_next is the inference vector (detached) for next planning step
    """
    if len(planner.visited_states) == 0:
        return []
    
    simulated_experiences = []
    
    # Initialize z for planning (zero vector for first state)
    B = batch_size
    z_planning = torch.zeros(B, model.embed_dim, device=device)
    
    # Process in batches
    for step in range(0, n_planning_steps, batch_size):
        batch_steps = min(batch_size, n_planning_steps - step)
        batch_states = []
        batch_actions = []
        batch_z_prev = []
        
        # Sample states and actions
        for i in range(batch_steps):
            try:
                state, state_hash = planner.visited_states.sample_state()
                action = planner.visited_states.sample_action_for_state(state_hash)
                batch_states.append(state)
                batch_actions.append(action)
                
                # Use z from previous planning step (detached)
                # For first batch, use zero; for subsequent, use z_planning
                if step == 0 and i == 0:
                    z_prev = torch.zeros(1, model.embed_dim, device=device)
                else:
                    z_prev = z_planning[i:i+1].detach()  # Detached: no gradient
                batch_z_prev.append(z_prev)
            except (ValueError, KeyError):
                continue
        
        if len(batch_states) == 0:
            continue
        
        # Stack batches
        state_batch = torch.stack(batch_states).to(device)  # [B, C, H, W]
        action_batch = torch.tensor(batch_actions, device=device)  # [B]
        z_prev_batch = torch.cat(batch_z_prev, dim=0)  # [B, embed_dim]
        
        # Predict next state and reward using environment model
        with torch.no_grad():
            next_state_pred, reward_pred = planner.env_model(state_batch, action_batch)
        
        # Forward through policy to get z_next (inference only, detached)
        model.eval()
        with torch.no_grad():
            # Use forward_with_z to get z_next from inference
            _, _, z_next = model.forward_with_z(
                next_state_pred,
                z_prev=z_prev_batch,
            )
            # z_next is already detached (no gradients during planning)
        
        # Store experiences with z_next for next planning step
        for i in range(len(batch_states)):
            simulated_experiences.append((
                batch_states[i].cpu(),
                batch_actions[i],
                float(reward_pred[i].item()),
                next_state_pred[i].cpu(),
                z_next[i:i+1].detach().cpu(),  # Detached z for next planning step
            ))
        
        # Update z_planning for next batch (detached)
        z_planning = z_next.detach()
    
    return simulated_experiences


# ============== Worker Process ==============

def worker_loop(
    worker_id: int,
    global_counter: Value,
    global_lock: Lock,
    result_queue: Queue,
    weight_queue: Queue,
    config: Dict,
    stop_flag: Value,
    shared_state: PlanningSharedState,
):
    """
    Worker process with Planning support
    
    Each worker:
    1. Collects real experiences with recurrent z
    2. Performs planning with detached z (inference only)
    3. Updates model using both real and simulated experiences
    """
    import subprocess
    import warnings
    
    warnings.filterwarnings('ignore')
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Worker {worker_id}] Started on {device}")
    
    rounds_per_batch = config.get('rounds_per_batch', 5)
    total_rounds = config.get('total_rounds', 50000)
    results_dir = config.get('results_dir', 'results/planning')
    results_dir = os.path.abspath(results_dir)
    sync_interval = config.get('sync_interval', 40)
    
    # Model paths
    local_model_path = os.path.join(results_dir, f'worker_{worker_id}_model.pt')
    global_model_path = config['model_path']
    
    # Planning configuration
    use_planning = config.get('use_planning', True)
    planning_steps = config.get('planning_steps', 100)
    planning_batch_size = config.get('planning_batch_size', 32)
    env_model_path = config.get('env_model_path', None)
    
    # Training parameters
    learning_rate = config.get('learning_rate', 3e-4)
    value_weight = config.get('value_weight', 0.5)
    real_weight = config.get('real_weight', 1.0)
    sim_weight = config.get('sim_weight', 0.5)
    
    # Set environment variables
    num_workers = config.get('num_workers', 1)
    os.environ['A3C_NUM_WORKERS'] = str(num_workers)
    os.environ['A3C_RESULTS_DIR'] = results_dir
    os.environ['BOMBER_USE_TRM'] = '1'  # Enable TRM recurrent mode
    
    # Create local model
    local_model = create_model(global_model_path, device)
    torch.save(local_model.state_dict(), local_model_path)
    
    # Create environment model and planner (if planning enabled)
    env_model = None
    planner = None
    if use_planning and env_model_path and os.path.exists(env_model_path):
        try:
            env_model = create_env_model(env_model_path, device)
            visited_states = VisitedStatesBuffer(max_size=10000)
            planner = DynaPlanner(env_model, visited_states, device=device)
            print(f"[Worker {worker_id}] Planning enabled with env model: {env_model_path}")
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to load env model: {e}, planning disabled")
            use_planning = False
    else:
        print(f"[Worker {worker_id}] Planning disabled (env_model_path={env_model_path})")
    
    # Create optimizer
    optimizer = optim.AdamW(local_model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    
    local_updates = 0
    global_round_file = os.path.join(results_dir, 'global_round_count.txt')
    
    while not stop_flag.value:
        with global_lock:
            current_round = global_counter.value
            if current_round >= total_rounds:
                break
        
        # Get opponents (simplified - can be extended with curriculum)
        agent_list = ['ppo_agent', 'ppo_agent', 'random_agent', 'peaceful_agent']
        
        batch_rounds = min(rounds_per_batch, total_rounds - current_round)
        
        # Set environment variables for subprocess
        worker_env = os.environ.copy()
        worker_env['PPO_MODEL_PATH'] = local_model_path
        worker_env['A3C_RESULTS_DIR'] = results_dir
        worker_env['A3C_NUM_WORKERS'] = str(num_workers)
        worker_env['BOMBER_USE_TRM'] = '1'  # Enable TRM recurrent mode
        
        output_file = os.path.join(
            results_dir,
            f"worker_{worker_id}_{int(time.time()*1000)}.json"
        )
        
        cmd = [
            sys.executable, "main.py", "play",
            "--agents"] + agent_list + [
            "--train", "2",
            "--no-gui",
            "--n-rounds", str(batch_rounds),
            "--save-stats", output_file,
            "--silence-errors",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=worker_env,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Reload local model (updated by PPO during game)
            if os.path.exists(local_model_path):
                try:
                    local_model.load_state_dict(
                        torch.load(local_model_path, map_location=device, weights_only=True)
                    )
                    local_updates += 1
                except Exception as e:
                    print(f"[Worker {worker_id}] Failed to reload model: {e}")
            
            # Perform planning and update (if enabled)
            if use_planning and planner is not None:
                try:
                    # Generate simulated experiences with recurrent z
                    simulated_experiences = plan_with_recurrent_z(
                        local_model,
                        planner,
                        planning_steps,
                        planning_batch_size,
                        device,
                    )
                    
                    if len(simulated_experiences) > 0:
                        # Extract batches
                        sim_states = torch.stack([exp[0] for exp in simulated_experiences]).to(device)
                        sim_actions = torch.tensor([exp[1] for exp in simulated_experiences], device=device)
                        sim_rewards = torch.tensor([exp[2] for exp in simulated_experiences], device=device)
                        sim_next_states = torch.stack([exp[3] for exp in simulated_experiences]).to(device)
                        
                        # Train on simulated experiences
                        # Note: For simulated experiences, we use z from planning (detached)
                        # Real experiences will be handled by PPO callback with proper z management
                        from agent_code.ppo_agent.train_deepsup import train_policy_deepsup
                        
                        # Simple training on simulated experiences
                        # (In practice, you might want to integrate this with PPO updates)
                        train_policy_deepsup(
                            local_model,
                            sim_states,
                            sim_actions,
                            optimizer,
                            n_sup=1,  # No DeepSupervision for simulated
                            strategy="last",
                            device=device,
                            rewards=sim_rewards,
                            train_value=True,
                            value_weight=value_weight,
                        )
                        
                        print(f"[Worker {worker_id}] Planning: {len(simulated_experiences)} simulated experiences")
                    
                except Exception as e:
                    print(f"[Worker {worker_id}] Planning failed: {e}")
            
            # Save updated model
            torch.save(local_model.state_dict(), local_model_path)
            
            # Update global counter
            with global_lock:
                new_round = global_counter.value + batch_rounds
                global_counter.value = min(new_round, total_rounds)
            
            # Send weights to main for averaging (if sync interval reached)
            if local_updates % sync_interval == 0:
                try:
                    weight_queue.put((worker_id, local_model_path))
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            print(f"[Worker {worker_id}] Game timeout")
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
    
    print(f"[Worker {worker_id}] Finished")


# ============== Main Training Loop ==============

def run_planning_a3c_training(config: PlanningA3CConfig):
    """Main A3C training function with Planning"""
    print("="*70)
    print("A3C Training with Planning (Dyna-Q style)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Total Rounds: {config.total_rounds}")
    print(f"  Planning: {'Enabled' if config.use_planning else 'Disabled'}")
    if config.use_planning:
        print(f"  Planning Steps: {config.planning_steps}")
        print(f"  Env Model: {config.env_model_path}")
    print(f"  Sync Interval: {config.sync_interval} batches")
    print()
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.model_path) or '.', exist_ok=True)
    
    # Initialize shared state
    shared_state = PlanningSharedState(config.model_path)
    stop_flag = Value('i', 0)
    
    # Prepare config dict
    config_dict = {
        'num_workers': config.num_workers,
        'total_rounds': config.total_rounds,
        'rounds_per_batch': config.rounds_per_batch,
        'sync_interval': config.sync_interval,
        'results_dir': config.results_dir,
        'model_path': config.model_path,
        'use_planning': config.use_planning,
        'planning_steps': config.planning_steps,
        'planning_batch_size': config.planning_batch_size,
        'env_model_path': config.env_model_path,
        'learning_rate': config.learning_rate,
        'value_weight': config.value_weight,
        'real_weight': config.real_weight,
        'sim_weight': config.sim_weight,
    }
    
    # Start worker processes
    workers = []
    result_queue = Queue()
    weight_queue = Queue()
    global_counter = Value('i', 0)
    global_lock = Lock()
    
    print(f"Starting {config.num_workers} worker processes...")
    print()
    
    start_time = time.time()
    
    for i in range(config.num_workers):
        p = Process(
            target=worker_loop,
            args=(i, global_counter, global_lock, result_queue, weight_queue, config_dict, stop_flag, shared_state)
        )
        p.start()
        workers.append(p)
        time.sleep(0.5)
    
    print("="*70)
    print("Training in progress... (Ctrl+C to stop)")
    print("="*70)
    print()
    
    try:
        while global_counter.value < config.total_rounds:
            time.sleep(5)
            
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                print("All workers finished.")
                break
            
            # Log progress
            if global_counter.value % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {global_counter.value}/{config.total_rounds} rounds ({elapsed:.1f}s)")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Stop workers
    stop_flag.value = 1
    
    for w in workers:
        w.join(timeout=10)
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Final rounds: {global_counter.value}")


# ============== Main ==============

if __name__ == '__main__':
    from dataclasses import dataclass
    
    parser = argparse.ArgumentParser(description='A3C Training with Planning')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--total-rounds', type=int, default=50000, help='Total training rounds')
    parser.add_argument('--rounds-per-batch', type=int, default=5, help='Rounds per batch')
    parser.add_argument('--sync-interval', type=int, default=40, help='Sync interval (batches)')
    parser.add_argument('--results-dir', type=str, default='results/planning', help='Results directory')
    parser.add_argument('--model-path', type=str, default='ppo_model.pt', help='Model path')
    parser.add_argument('--planning-steps', type=int, default=100, help='Number of planning steps')
    parser.add_argument('--planning-batch-size', type=int, default=32, help='Planning batch size')
    parser.add_argument('--env-model-path', type=str, default=None, help='Path to environment model')
    parser.add_argument('--no-planning', action='store_true', help='Disable planning')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--value-weight', type=float, default=0.5, help='Value loss weight')
    parser.add_argument('--real-weight', type=float, default=1.0, help='Real experience weight')
    parser.add_argument('--sim-weight', type=float, default=0.5, help='Simulated experience weight')
    
    args = parser.parse_args()
    
    config = PlanningA3CConfig(
        num_workers=args.num_workers,
        total_rounds=args.total_rounds,
        rounds_per_batch=args.rounds_per_batch,
        sync_interval=args.sync_interval,
        results_dir=args.results_dir,
        model_path=args.model_path,
        planning_steps=args.planning_steps,
        planning_batch_size=args.planning_batch_size,
        use_planning=not args.no_planning,
        env_model_path=args.env_model_path,
        learning_rate=args.learning_rate,
        value_weight=args.value_weight,
        real_weight=args.real_weight,
        sim_weight=args.sim_weight,
    )
    
    run_planning_a3c_training(config)

