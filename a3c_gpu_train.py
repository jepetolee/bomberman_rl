#!/usr/bin/env python3
"""
A3C GPU Training with Weight Averaging (Federated Learning Style)
==================================================================

Architecture:
  - Each worker has its own local model
  - Workers independently collect experiences and update locally
  - Periodically: collect weights from all workers → average → distribute

Features:
1. Weight Averaging: Collect and average model weights (not gradients)
2. Adaptive Curriculum: Progress when win rate threshold is met
3. Self-Play: Same model Team vs Same model Team (2v2)
4. Best Model Only: Save only when performance improves

Usage:
    python a3c_gpu_train.py --num-workers 4 --total-rounds 50000 --sync-interval 10
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

# Disable audio (ALSA/PulseAudio) - not needed for training
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value, Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============== Adaptive Curriculum Configuration ==============

# Agents that support team play (one agent controls 2 bombers)
TEAM_PLAY_AGENTS = {
    'aggressive_teacher_agent',
    'team_teacher_agent',
}

CURRICULUM_STAGES = [
    # (stage_name, opponent_pool, win_rate_threshold_to_advance)
    ("Stage 1: Easy", ['random_agent', 'peaceful_agent'], 0.60),
    ("Stage 2: Medium", ['peaceful_agent', 'coin_collector_agent'], 0.65),
    ("Stage 3: Hard", ['coin_collector_agent', 'rule_based_agent'], 0.70),
    ("Stage 4: Expert", ['team_teacher_agent', 'aggressive_teacher_agent'], 0.75),
    # Stage 5 is self-play - handled separately
]

SELF_PLAY_STAGE = "Stage 5: Self-Play (Team vs Team)"


class AdaptiveCurriculum:
    """
    Adaptive curriculum that progresses based on win rate.
    
    Structure:
      - PPO Team (agents 0,1) vs Opponent Team (agents 2,3)
      - Self-Play: PPO Team vs PPO Team (same model, 2v2)
    """
    
    def __init__(self, eval_window: int = 200):
        self.eval_window = eval_window
        self.current_stage = 0
        self.total_stages = len(CURRICULUM_STAGES)
        
        # Win tracking: 1 = win, 0 = loss/draw
        self.recent_results = deque(maxlen=eval_window)
        
        # Stats
        self.stage_rounds = 0
        self.total_rounds = 0
        self.stage_start_time = time.time()
    
    def add_result(self, ppo_team_score: float, opponent_team_score: float):
        """
        Record a match result.
        Win = PPO Team (agents 0,1) scored higher than Opponent Team (agents 2,3)
        """
        win = 1 if ppo_team_score > opponent_team_score else 0
        self.recent_results.append(win)
        self.stage_rounds += 1
        self.total_rounds += 1
    
    def get_win_rate(self) -> float:
        if len(self.recent_results) < 10:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)
    
    def should_advance(self) -> bool:
        """Check if we should advance to next stage."""
        if self.current_stage >= self.total_stages:
            return False  # Already in self-play
        
        if len(self.recent_results) < self.eval_window // 2:
            return False  # Not enough data
        
        win_rate = self.get_win_rate()
        threshold = CURRICULUM_STAGES[self.current_stage][2]
        
        return win_rate >= threshold
    
    def advance_stage(self):
        """Advance to next stage."""
        old_stage = self.current_stage
        self.current_stage += 1
        self.recent_results.clear()
        
        elapsed = time.time() - self.stage_start_time
        print(f"\n{'='*60}")
        print(f"🎯 STAGE ADVANCEMENT!")
        print(f"   {CURRICULUM_STAGES[old_stage][0]} → ", end="")
        if self.current_stage < self.total_stages:
            print(f"{CURRICULUM_STAGES[self.current_stage][0]}")
        else:
            print(f"{SELF_PLAY_STAGE}")
        print(f"   Rounds in stage: {self.stage_rounds}")
        print(f"   Time in stage: {elapsed/60:.1f} min")
        print(f"{'='*60}\n")
        
        self.stage_rounds = 0
        self.stage_start_time = time.time()
    
    def is_selfplay(self) -> bool:
        return self.current_stage >= self.total_stages
    
    def get_opponents(self) -> Tuple[str, str, str, List[int]]:
        """
        Get opponents for current stage with random agent assignment.
        Delegates to helper function for consistency.
        """
        return get_opponents_for_stage(self.current_stage, self.is_selfplay())
    
    def get_status(self) -> str:
        """Get current status string."""
        win_rate = self.get_win_rate()
        
        if self.is_selfplay():
            return f"{SELF_PLAY_STAGE} | Rounds: {self.stage_rounds}"
        
        stage_name, _, threshold = CURRICULUM_STAGES[self.current_stage]
        progress = min(1.0, win_rate / threshold) if threshold > 0 else 1.0
        bar_len = 20
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        return (f"{stage_name} | WR: {win_rate:.1%} [{bar}] {threshold:.0%} | "
                f"Rounds: {self.stage_rounds}")


# ============== Performance Tracking ==============

class PerformanceTracker:
    """Track recent performance and determine best model."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)
        self.kills = deque(maxlen=window_size)
        self.deaths = deque(maxlen=window_size)
        self.best_score = float('-inf')
        self.best_round = 0
    
    def add(self, score: float, kills: int, deaths: int):
        self.scores.append(score)
        self.kills.append(kills)
        self.deaths.append(deaths)
    
    def get_recent_avg(self) -> Dict:
        if not self.scores:
            return {'score': 0, 'kills': 0, 'deaths': 0, 'performance': 0}
        
        avg_score = sum(self.scores) / len(self.scores)
        avg_kills = sum(self.kills) / len(self.kills)
        avg_deaths = sum(self.deaths) / len(self.deaths)
        performance = avg_score + avg_kills * 3 - avg_deaths  # Kills weighted more
        
        return {
            'score': avg_score,
            'kills': avg_kills,
            'deaths': avg_deaths,
            'performance': performance
        }
    
    def is_best(self, current_round: int) -> bool:
        if len(self.scores) < self.window_size // 4:
            return False
        
        stats = self.get_recent_avg()
        if stats['performance'] > self.best_score:
            self.best_score = stats['performance']
            self.best_round = current_round
            return True
        return False


# ============== Model Creation ==============

def create_model(model_path: str, device: torch.device):
    """Create model (local or global)."""
    # Prefer building the model from our YAML config so the architecture matches
    # `policy_phase2.pt` (Hybrid ViT+TRM) when BOMBER_FROZEN_VIT / BOMBER_USE_TRM is enabled.
    # Fallback to env-var ViT-only model if YAML (pyyaml) isn't available.
    model = None
    try:
        from config.load_config import load_config, create_model_from_config  # type: ignore
        cfg_path = os.environ.get("BOMBER_CONFIG_PATH", "config/trm_config.yaml")
        cfg = load_config(cfg_path)
        # strict_yaml=True: YAML 설정을 엄격하게 따르고, 기본값을 사용하지 않음
        model = create_model_from_config(cfg, device=device, strict_yaml=True)
        try:
            model_type = cfg.get("model", {}).get("type", "unknown")
            print(f"[Model] Config type={model_type}, class={model.__class__.__name__}")
        except Exception:
            pass
    except Exception as e:
        # Fallback: use env vars or defaults matching YAML (embed_dim=512, depth=6, heads=8)
        from agent_code.ppo_agent.models.vit import PolicyValueViT
        import settings as s

        mixer = os.environ.get("BOMBER_VIT_MIXER", "attn")
        # Increased defaults for better performance (ViT Only)
        embed_dim = int(os.environ.get("BOMBER_VIT_DIM", "512"))
        depth = int(os.environ.get("BOMBER_VIT_DEPTH", "6"))
        num_heads = int(os.environ.get("BOMBER_VIT_HEADS", "8"))

        # NOTE: Our current features are 10 channels; keep this aligned.
        model = PolicyValueViT(
            in_channels=10,
            num_actions=6,
            img_size=(s.COLS, s.ROWS),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            patch_size=1,
            use_cls_token=False,
            mixer=mixer,
        )

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint strictly: {e}. Starting from scratch.")
            missing_keys, unexpected_keys = ["load_failed"], []

        # If any mismatch, ignore checkpoint to avoid wrong architecture loading
        if missing_keys or unexpected_keys:
            print(f"[WARNING] Checkpoint ignored due to mismatch. Missing={len(missing_keys)}, Unexpected={len(unexpected_keys)}")
            model = create_model_from_config(cfg, device=device, strict_yaml=True)
        else:
            try:
                print(f"[Model] Loaded checkpoint '{model_path}' as {model.__class__.__name__}")
            except Exception:
                pass

    model = model.to(device)
    return model


def try_init_distributed():
    """Initialize torch.distributed if launched under torchrun."""
    if dist.is_available() and not dist.is_initialized():
        required_env = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
        if all(k in os.environ for k in required_env):
            dist.init_process_group(backend="nccl", init_method="env://")
    return dist.is_initialized()


def dist_allreduce_model(model: nn.Module):
    """All-reduce model parameters and buffers, then average."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    for p in model.parameters():
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
        # Only divide if floating/complex; ints keep rank0 value
        if p.data.dtype.is_floating_point or p.data.is_complex():
            p.data /= world_size
        else:
            dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.all_reduce(b.data, op=dist.ReduceOp.SUM)
        if b.data.dtype.is_floating_point or b.data.is_complex():
            b.data /= world_size
        else:
            dist.broadcast(b.data, src=0)


def average_model_weights(weight_dicts: List[Dict], global_weights: Optional[Dict] = None, 
                          global_weight: float = 0.1) -> Dict:
    """
    Average model weights from multiple workers, optionally combined with global model.
    
    Args:
        weight_dicts: List of state_dict from each worker
        global_weights: Optional global model weights to combine
        global_weight: Weight for global model (0.0 = only workers, 1.0 = only global)
                       Default 0.1 means: 0.9 * avg_workers + 0.1 * global
    
    Returns:
        Averaged state_dict (optionally combined with global)
    """
    if not weight_dicts:
        return global_weights if global_weights else {}
    
    # Average worker weights
    avg_worker_weights = {}
    for key in weight_dicts[0].keys():
        # Stack all weights for this parameter
        stacked = torch.stack([w[key].float() for w in weight_dicts])
        # Average
        avg_worker_weights[key] = stacked.mean(dim=0)
    
    # Combine with global model if provided
    if global_weights is not None and global_weight > 0.0:
        # Ensure global weights live on CPU to match worker weights (which are sent CPU)
        global_weights = {k: v.cpu() for k, v in global_weights.items()}
        combined_weights = {}
        worker_weight = 1.0 - global_weight
        for key in avg_worker_weights.keys():
            if key in global_weights:
                # Weighted combination: (1-α) * workers + α * global
                combined_weights[key] = (
                    worker_weight * avg_worker_weights[key] + 
                    global_weight * global_weights[key]
                )
            else:
                combined_weights[key] = avg_worker_weights[key]
        return combined_weights
    
    return avg_worker_weights


# ============== Worker Process ==============

def get_opponents_for_stage(stage: int, is_selfplay: bool) -> Tuple[str, str, str, List[int]]:
    """
    Get opponents for a given stage with random agent assignment.
    Helper function that can be called from worker processes.
    """
    if is_selfplay:
        agent_order = [0, 1, 2, 3]
        random.shuffle(agent_order)
        return 'ppo_agent', 'ppo_agent', SELF_PLAY_STAGE, agent_order
    
    stage_name, opponent_pool, _ = CURRICULUM_STAGES[stage]
    
    # Check if any opponent supports team play
    team_play_available = [a for a in opponent_pool if a in TEAM_PLAY_AGENTS]
    
    if team_play_available and random.random() < 0.5:
        # Use team-play agent: one agent controls both positions
        opp_agent = random.choice(team_play_available)
        opp1 = opp_agent
        opp2 = opp_agent  # Same agent controls both
    else:
        # Use two separate agents
        opp1 = random.choice(opponent_pool)
        opp2 = random.choice(opponent_pool)
    
    # Randomize agent positions to prevent fixed policy learning
    agent_order = [0, 1, 2, 3]
    random.shuffle(agent_order)
    
    return opp1, opp2, stage_name, agent_order


def worker_loop(
    worker_id: int,
    global_counter: Value,
    global_lock: Lock,
    result_queue: Queue,
    weight_queue: Queue,  # For sending weights to main
    config: Dict,
    stop_flag: Value,
    stage_info: Value
):
    """
    Worker process with independent local model and weight averaging.
    
    Each worker:
    1. Has its own local model
    2. Collects experiences independently
    3. Updates local model via PPO
    4. Periodically sends weights to main for averaging
    """
    import subprocess
    import warnings
    
    # Suppress warnings in worker processes
    warnings.filterwarnings('ignore')
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Worker {worker_id}] Started")
    
    rounds_per_batch = config.get('rounds_per_batch', 5)
    total_rounds = config.get('total_rounds', 50000)
    results_dir = config.get('results_dir', 'results/adaptive')
    # Convert to absolute path to avoid working directory issues
    results_dir = os.path.abspath(results_dir)
    sync_interval = config.get('sync_interval', 40)  # Sync every N batches (lower = more frequent)
    
    # Each worker has its own local model file
    local_model_path = os.path.join(results_dir, f'worker_{worker_id}_model.pt')
    global_model_path = config['model_path']
    
    # Set worker count and results dir in environment for epsilon decay calculation
    # This allows each worker to properly scale its completed_rounds and find global round file
    num_workers = config.get('num_workers', 1)
    os.environ['A3C_NUM_WORKERS'] = str(num_workers)
    os.environ['A3C_RESULTS_DIR'] = results_dir  # Pass results_dir so PPO can find global_round_count.txt
    
    # Create local model
    local_model = create_model(global_model_path, device)
    torch.save(local_model.state_dict(), local_model_path)
    
    local_updates = 0
    last_sync_mtime = 0  # Track when we last synced
    
    # File to share global round count for epsilon decay
    global_round_file = os.path.join(results_dir, 'global_round_count.txt')
    
    while not stop_flag.value:
        with global_lock:
            current_round = global_counter.value
            current_stage = stage_info.value
            
            if current_round >= total_rounds:
                break
        
        # Get opponents from curriculum
        is_selfplay = current_stage >= len(CURRICULUM_STAGES)
        opp1, opp2, stage_name, agent_order = get_opponents_for_stage(current_stage, is_selfplay)
        
        # Build agent list.
        #
        # IMPORTANT:
        # `main.py play --train N` marks the *first N agents* in `--agents` as trainable.
        # If we shuffle across that boundary, opponents like `random_agent` / `peaceful_agent`
        # can land in the first slots and the engine will try to import
        # `agent_code.<opponent>.train` (which doesn't exist) → subprocess crash and no stats file.
        #
        # So we keep PPO agents in the first 2 positions, and only randomize the opponents order.
        shuffled_agents = ['ppo_agent', 'ppo_agent']
        opps = [opp1, opp2]
        random.shuffle(opps)
        shuffled_agents += opps
        
        batch_rounds = min(rounds_per_batch, total_rounds - current_round)
        
        # Set environment variables for subprocess - MUST pass via env parameter
        # os.environ changes don't propagate to subprocess automatically
        # IMPORTANT: Set PPO_MODEL_PATH to match the global model path for initial load
        # The subprocess will load from global_model_path first, then use local_model_path for saving
        worker_env = os.environ.copy()
        worker_env['PPO_MODEL_PATH'] = global_model_path  # Use global model path for initial load
        worker_env['A3C_RESULTS_DIR'] = results_dir  # Pass results_dir so PPO can save stats here
        worker_env['A3C_NUM_WORKERS'] = str(num_workers)  # Also pass worker count for epsilon
        # Pass dist rank/world_size to child so env_model sync can be coordinated by files
        if dist.is_available() and dist.is_initialized():
            worker_env['A3C_RANK'] = str(rank)
            worker_env['A3C_WORLD_SIZE'] = str(world_size)
        
        output_file = os.path.join(
            results_dir,
            f"worker_{worker_id}_{int(time.time()*1000)}.json"
        )
        
        cmd = [
            sys.executable, "main.py", "play",
            "--agents"] + shuffled_agents + [
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
                env=worker_env,  # Pass environment variables to subprocess
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Debug: Log subprocess errors if any
            if result.returncode != 0:
                if local_updates == 0 or local_updates % 50 == 0:  # Log first time and periodically
                    print(f"[Worker {worker_id}] ❌ Subprocess error (returncode={result.returncode}):")
                    print(f"[Worker {worker_id}]   Command: {' '.join(cmd)}")
                    if result.stderr:
                        print(f"[Worker {worker_id}]   stderr (full):\n{result.stderr}")
                    if result.stdout:
                        print(f"[Worker {worker_id}]   stdout (last 1000 chars):\n{result.stdout[-1000:]}")
            
            # Debug: Check if output file was created (always log first time, then periodically)
            if not os.path.exists(output_file):
                if local_updates == 0 or local_updates % 50 == 0:
                    print(f"[Worker {worker_id}] ⚠️  Output file not created: {output_file}")
                    print(f"[Worker {worker_id}]   Subprocess returncode: {result.returncode}")
                    if result.stderr:
                        # Print full stderr to see complete error
                        print(f"[Worker {worker_id}]   stderr (full):\n{result.stderr}")
                    if result.stdout:
                        print(f"[Worker {worker_id}]   stdout (last 500 chars):\n{result.stdout[-500:]}")
            else:
                # Debug: Check file content (always log first time, then periodically)
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    by_agent = data.get("by_agent", {})
                    if not by_agent:
                        if local_updates == 0 or local_updates % 50 == 0:
                            print(f"[Worker {worker_id}] ⚠️  Output file exists but 'by_agent' is empty")
                            print(f"[Worker {worker_id}]   File keys: {list(data.keys())}")
                except Exception as e:
                    if local_updates == 0 or local_updates % 50 == 0:
                        print(f"[Worker {worker_id}] ⚠️  Error reading output file: {e}")
            
            # Local model was updated by PPO during game execution
            # Reload it to get latest weights
            if os.path.exists(local_model_path):
                try:
                    local_model.load_state_dict(
                        torch.load(local_model_path, map_location=device, weights_only=True)
                    )
                    local_updates += 1
                    if local_updates % 20 == 0:  # Log every 20 updates
                        print(f"[Worker {worker_id}] Model updated (total: {local_updates})")
                except Exception as e:
                    print(f"[Worker {worker_id}] Failed to reload local model: {e}")
            
            # Try to get teacher stats from file (game subprocess saves it)
            # Check multiple possible locations
            teacher_stats = None
            try:
                # List of possible file locations (in order of likelihood)
                possible_paths = []
                
                # 1. Global model path directory
                model_dir = os.path.dirname(config['model_path']) if os.path.dirname(config['model_path']) else '.'
                possible_paths.append(os.path.join(model_dir, 'teacher_stats.json'))
                
                # 2. Local model directory (where PPO_MODEL_PATH points)
                local_model_dir = os.path.dirname(local_model_path) if os.path.dirname(local_model_path) else '.'
                possible_paths.append(os.path.join(local_model_dir, 'teacher_stats.json'))
                
                # 3. Results directory (where worker files are)
                possible_paths.append(os.path.join(results_dir, 'teacher_stats.json'))
                
                # 4. Current working directory
                possible_paths.append('teacher_stats.json')
                
                # Try to read from any location
                for file_path in possible_paths:
                    if os.path.exists(file_path):
                        # Check file modification time to ensure it's recent
                        file_mtime = os.path.getmtime(file_path)
                        current_time = time.time()
                        # Only use stats if file was updated in last 120 seconds (more lenient)
                        if current_time - file_mtime < 120:
                            with open(file_path, 'r') as f:
                                teacher_stats = json.load(f)
                            # Debug: log when we successfully read stats (first few times)
                            if local_updates <= 5:
                                print(f"[Worker {worker_id}] ✓ Read teacher stats: {teacher_stats.get('usage', 0):.1f}% from {file_path}")
                            break
            except Exception as e:
                # Only log errors first few times to avoid spam
                if local_updates <= 5:
                    print(f"[Worker {worker_id}] ✗ Teacher stats error: {e}")
                pass
            
            batch_stats = {
                'team_a_score': 0, 'team_b_score': 0,
                'kills': 0, 'deaths': 0,
                'opponents': f"{opp1},{opp2}",
                'stage': current_stage,
                'is_selfplay': is_selfplay,
            }
            
            # Always try to read output_file (even if it doesn't exist, we'll handle it)
            file_read_success = False
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    by_agent = data.get("by_agent", {})
                    by_round = data.get("by_round", {})
                    
                    # Debug: Log first few times to see what we're getting
                    if local_updates <= 3:
                        print(f"[Worker {worker_id}] Debug: by_agent keys: {list(by_agent.keys())}")
                        if by_agent:
                            sample_agent = list(by_agent.keys())[0]
                            print(f"[Worker {worker_id}] Debug: {sample_agent} stats: {by_agent[sample_agent]}")
                        print(f"[Worker {worker_id}] Debug: by_round count: {len(by_round)}")
                    
                    # Aggregate kills from by_round (kills are per-round, not per-agent in JSON)
                    total_kills = 0
                    for round_data in by_round.values():
                        total_kills += round_data.get("kills", 0)
                    
                    # Calculate team scores and deaths
                    ppo_score = 0
                    opp_score = 0
                    ppo_deaths = 0
                    for agent_name, stats in by_agent.items():
                        score = stats.get("score", 0)
                        suicides = stats.get("suicides", 0)  # deaths = suicides in JSON
                        
                        if agent_name.startswith("ppo_agent"):
                            batch_stats['team_a_score'] += score
                            batch_stats['deaths'] += suicides  # Add suicides as deaths
                            ppo_score += score
                            ppo_deaths += suicides
                        else:
                            batch_stats['team_b_score'] += score
                            opp_score += score
                    
                    # Assign kills: if ppo team scored more, they likely got the kills
                    # If kills exist and ppo team won (or tied with score > 0), assign kills to ppo
                    if total_kills > 0:
                        if ppo_score >= opp_score and ppo_score > 0:
                            # PPO team won or tied, assign kills to them
                            batch_stats['kills'] += total_kills
                        # If opponents won, don't assign kills to ppo (they got killed)
                    
                    file_read_success = True
                    
                    # Debug: Log stats summary (always log first time, then periodically)
                    if local_updates == 0 or local_updates % 50 == 0:
                        print(f"[Worker {worker_id}] Debug: batch_stats = {batch_stats}")
                        print(f"[Worker {worker_id}] Debug: ppo_score={ppo_score}, opp_score={opp_score}, ppo_deaths={ppo_deaths}, total_kills={total_kills}")
                except Exception as e:
                    if local_updates == 0 or local_updates % 50 == 0:
                        print(f"[Worker {worker_id}] Error parsing output_file: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Debug: Log if file was not read successfully
            if not file_read_success and (local_updates == 0 or local_updates % 50 == 0):
                print(f"[Worker {worker_id}] ⚠️  Could not read output_file, batch_stats remains: {batch_stats}")
            
            # Clean up output file if it exists
            try:
                if os.path.exists(output_file):
                    os.remove(output_file)
            except:
                pass
            
            with global_lock:
                global_counter.value += batch_rounds
                new_count = global_counter.value
                # Write global round count to file for epsilon decay
                try:
                    with open(global_round_file, 'w') as f:
                        f.write(str(new_count))
                except:
                    pass
            
            result_queue.put({
                'type': 'progress',
                'worker_id': worker_id,
                'round': new_count,
                'stats': batch_stats,
                'batch_rounds': batch_rounds,
                'teacher_stats': teacher_stats,  # Include teacher stats
            })
            
            # Periodically send weights for averaging (non-blocking)
            if local_updates % sync_interval == 0:
                # Convert to CPU to reduce memory and avoid serialization issues
                cpu_weights = {k: v.cpu() for k, v in local_model.state_dict().items()}
                weight_queue.put({
                    'worker_id': worker_id,
                    'weights': cpu_weights,
                    'updates': local_updates,
                })
                if local_updates % (sync_interval * 5) == 0:  # Log every 5 sync attempts
                    print(f"[Worker {worker_id}] Sent weights (update {local_updates})")
                # Don't wait for sync - continue immediately
                # Main will process weights asynchronously
                # Worker will get updated model on next sync cycle
                
                # Try to reload global model if it was updated (non-blocking check)
                if os.path.exists(global_model_path):
                    try:
                        # Check file modification time to see if it was updated
                        current_mtime = os.path.getmtime(global_model_path)
                        if current_mtime > last_sync_mtime:
                            global_weights = torch.load(global_model_path, map_location=device, weights_only=True)
                            missing, unexpected = local_model.load_state_dict(global_weights, strict=False)
                            # Only save if load was mostly successful (less than 50% keys missing)
                            if len(missing) < len(local_model.state_dict()) * 0.5:
                                torch.save(local_model.state_dict(), local_model_path)
                                last_sync_mtime = current_mtime
                    except Exception as e:
                        pass  # Silently ignore - will try again next time
            
            # Also check for updated global model periodically (not just on sync)
            # Avoid division by zero: if sync_interval is 1, check every update
            check_interval = max(1, sync_interval // 2) if sync_interval > 1 else 1
            if local_updates % check_interval == 0 and os.path.exists(global_model_path):
                try:
                    current_mtime = os.path.getmtime(global_model_path)
                    if current_mtime > last_sync_mtime:
                        global_weights = torch.load(global_model_path, map_location=device, weights_only=True)
                        missing, unexpected = local_model.load_state_dict(global_weights, strict=False)
                        # Only save if load was mostly successful (less than 50% keys missing)
                        if len(missing) < len(local_model.state_dict()) * 0.5:
                            torch.save(local_model.state_dict(), local_model_path)
                            last_sync_mtime = current_mtime
                except Exception:
                    pass
        
        except subprocess.TimeoutExpired:
            print(f"[Worker {worker_id}] Timeout, retrying...")
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            time.sleep(1)
    
    result_queue.put({
        'type': 'final',
        'worker_id': worker_id,
    })
    
    print(f"[Worker {worker_id}] Finished")


# ============== Main Training ==============

def run_adaptive_training(config: Dict):
    """Main training with weight averaging."""
    is_dist = try_init_distributed()
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_dist else 0
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    if rank == 0:
        print("="*70)
        print("A3C GPU: Weight Averaging (Federated Learning Style)")
        print("="*70)
        print()
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()
        
        print("Architecture:")
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  Each Worker: Independent Local Model              │")
        print("  │  - Collects experiences                            │")
        print("  │  - Updates locally via PPO                         │")
        print("  │  - Sends weights periodically                      │")
        print("  │                                                     │")
        print("  │  Main Process: Weight Averaging                    │")
        print("  │  - Collects weights from all workers              │")
        print("  │  - Averages weights → Global Model                 │")
        print("  │  - Distributes to workers                          │")
        print("  └─────────────────────────────────────────────────────┘")
        print()
        
        print("Curriculum Stages (advance when Team A win rate exceeds threshold):")
        for i, (name, opponents, threshold) in enumerate(CURRICULUM_STAGES):
            print(f"  {i+1}. {name}: Team A vs {opponents} → {threshold:.0%}")
        print(f"  5. {SELF_PLAY_STAGE}")
        print()
        
        print(f"Configuration:")
        print(f"  Workers: {config['num_workers']} (world_size={world_size})")
        print(f"  Max Rounds: {config['total_rounds']}")
        print(f"  Sync Interval: {config['sync_interval']} batches")
        print(f"  Eval Window: {config['eval_window']} rounds")
        print()
    
    # Create directories - ensure absolute path
    config['results_dir'] = os.path.abspath(config['results_dir'])
    if is_dist:
        # Avoid file collisions across ranks
        config['results_dir'] = os.path.join(config['results_dir'], f"rank{rank}")
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Create global model
    if rank == 0:
        print("Creating global model...")
    global_model = create_model(config['model_path'], device)
    if rank == 0:
        print(f"  Parameters: {sum(p.numel() for p in global_model.parameters()):,}")
        print()
    
    # Trackers
    curriculum = AdaptiveCurriculum(eval_window=config['eval_window'])
    perf_tracker = PerformanceTracker(window_size=config['eval_window'])
    
    # Multiprocessing
    mp.set_start_method('spawn', force=True)
    
    global_counter = Value('i', 0)
    global_lock = Lock()
    stop_flag = Value('b', False)
    stage_info = Value('i', 0)
    result_queue = Queue()
    weight_queue = Queue()
    
    # Start workers
    workers = []
    if rank == 0:
        print(f"Starting {config['num_workers']} workers...")
    
    start_time = time.time()
    
    for i in range(config['num_workers']):
        p = Process(
            target=worker_loop,
            args=(i, global_counter, global_lock, result_queue, 
                  weight_queue, config, stop_flag, stage_info)
        )
        p.start()
        workers.append(p)
        time.sleep(0.2)
    
    print()
    print("="*70)
    print("Training in progress... (Ctrl+C to stop)")
    print("="*70)
    print()
    
    total_stats = {'score': 0, 'kills': 0, 'deaths': 0}
    last_log_round = 0
    last_status_check = time.time()
    saves_count = 0
    sync_count = 0
    # Buffer incoming worker weights between loops so we don't drop them
    pending_weights = {}
    # Require at most the available workers (default 2) to sync; override via config
    min_workers_for_sync = max(
        1,
        min(config['num_workers'], config.get('min_workers_for_sync', 2))
    )
    
    try:
        while True:
            with global_lock:
                current_round = global_counter.value
            
            if current_round >= config['total_rounds']:
                break
            
            # Periodic status check (every 10 seconds) even if no messages
            current_time = time.time()
            if current_time - last_status_check >= 10.0:
                last_status_check = current_time
                alive = sum(1 for w in workers if w.is_alive())
                
                # Check if round is actually progressing
                if current_round == last_log_round and current_round > 0:
                    print(f"[WARNING] No progress detected! Round stuck at {current_round}")
                    print(f"  Active workers: {alive}/{config['num_workers']}")
                    print(f"  Queue size: {result_queue.qsize()}")
                    print(f"  Weight queue size: {weight_queue.qsize()}")
                    # Workers are now non-blocking, so no need to unblock
                
                if current_round > last_log_round:
                    # Force log even if no new messages
                    recent = perf_tracker.get_recent_avg()
                    status = curriculum.get_status()
                    print(f"[Round {current_round}/{config['total_rounds']}] (Status check)")
                    print(f"  {status}")
                    print(f"  Score={recent['score']:.2f} Kills={recent['kills']:.2f} Deaths={recent['deaths']:.2f}")
                    print(f"  Active workers: {alive}/{config['num_workers']}")
                    print(f"  Weight queue: {weight_queue.qsize()} pending, Syncs: {sync_count}")
                    print()
                    last_log_round = current_round
            
            # Collect weights from workers for averaging
            # Check more frequently to avoid worker timeouts
            collected_weights = {}
            while not weight_queue.empty():
                try:
                    msg = weight_queue.get_nowait()
                    collected_weights[msg['worker_id']] = msg['weights']
                except queue.Empty:
                    break

            # Keep collected weights across loops so we can reach the threshold
            if collected_weights:
                pending_weights.update(collected_weights)
            
            # Average weights if we have enough workers' updates
            if 0 < len(pending_weights) < min_workers_for_sync:
                if not hasattr(run_adaptive_training, '_last_weight_log') or \
                   time.time() - getattr(run_adaptive_training, '_last_weight_log', 0) > 30:
                    print(f"[Weight Queue] {len(pending_weights)}/{min_workers_for_sync} workers (waiting for more...)")
                    setattr(run_adaptive_training, '_last_weight_log', time.time())
            
            if len(pending_weights) >= min_workers_for_sync:
                sync_count += 1
                weight_list = list(pending_weights.values())
                
                # Get current global model weights for combination
                current_global_weights = global_model.state_dict()
                
                # Average worker weights and combine with global model
                # Formula: 0.9 * avg_workers + 0.1 * global (more weight to workers)
                global_weight_ratio = config.get('global_weight_ratio', 0.1)
                avg_weights = average_model_weights(
                    weight_list, 
                    global_weights=current_global_weights,
                    global_weight=global_weight_ratio
                )
                
                # Update global model
                global_model.load_state_dict(avg_weights)
                
                # Distributed all-reduce to average across torchrun ranks (method B)
                dist_allreduce_model(global_model)
                
                if rank == 0:
                    torch.save(global_model.state_dict(), config['model_path'])
                    print(f"[Sync {sync_count}] ★ Averaged {len(weight_list)}/{config['num_workers']} workers "
                          f"+ {global_weight_ratio:.0%} global model (dist world_size={world_size})")
                    print(f"    Global model updated and saved!")
                
                # Clear collected weights after processing
                pending_weights.clear()
                
                # No need to signal workers - they check file mtime asynchronously
            
            # Process result messages
            # Use shorter timeout to check weight queue more frequently
            try:
                msg = result_queue.get(timeout=1)
                
                if msg['type'] == 'progress':
                    stats = msg['stats']
                    round_num = msg['round']
                    batch_rounds = msg['batch_rounds']
                    teacher_stats = msg.get('teacher_stats')  # Get teacher stats from worker
                    
                    # Debug: Log first few progress messages to see what we're getting
                    if round_num <= 100 or round_num % 500 == 0:
                        print(f"[Debug Round {round_num}] stats from worker: {stats}")
                    
                    # Update totals
                    total_stats['score'] += stats['team_a_score']
                    total_stats['kills'] += stats['kills']
                    total_stats['deaths'] += stats['deaths']
                    
                    # Track performance
                    perf_tracker.add(stats['team_a_score'], stats['kills'], stats['deaths'])
                    
                    # Track wins for curriculum
                    for _ in range(batch_rounds):
                        curriculum.add_result(
                            stats['team_a_score'] / max(1, batch_rounds),
                            stats['team_b_score'] / max(1, batch_rounds)
                        )
                    
                    # Check stage advancement
                    if curriculum.should_advance():
                        curriculum.advance_stage()
                        with global_lock:
                            stage_info.value = curriculum.current_stage
                    
                    # Log progress every 50 rounds
                    if round_num - last_log_round >= 50:
                        last_log_round = round_num
                        recent = perf_tracker.get_recent_avg()
                        status = curriculum.get_status()
                        
                        print(f"[Round {round_num}/{config['total_rounds']}]")
                        print(f"  {status}")
                        if stats.get('is_selfplay'):
                            print(f"  Team A vs Team B (same model)")
                        print(f"  Score={recent['score']:.2f} Kills={recent['kills']:.2f} Deaths={recent['deaths']:.2f}")
                        
                        # Show teacher model usage stats if available from worker
                        if teacher_stats:
                            print(f"  📚 Teacher: {teacher_stats['usage']:.1f}% used "
                                  f"(eps={teacher_stats['epsilon']:.3f}, invalid={teacher_stats['invalid_rate']:.1f}%)")
                        else:
                            # Show that we're looking for stats but didn't find them
                            if round_num <= 500:  # Only show first few times
                                print(f"  📚 Teacher stats: Not available yet (may need more rounds)")
                        
                        if perf_tracker.is_best(round_num):
                            saves_count += 1
                            torch.save(global_model.state_dict(), config['model_path'])
                            print(f"  ★ NEW BEST! Perf={recent['performance']:.2f} (saved)")
                        print()
                
                elif msg['type'] == 'final':
                    print(f"[Worker {msg.get('worker_id', '?')}] Finished")
                    
            except queue.Empty:
                # Check worker status periodically
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    print("All workers finished!")
                    break
                # Continue loop - workers are still alive
            
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        stop_flag.value = True
    
    # Cleanup
    stop_flag.value = True
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    
    elapsed = time.time() - start_time
    final_round = global_counter.value
    
    # Final save
    if perf_tracker.is_best(final_round):
        torch.save(global_model.state_dict(), config['model_path'])
        saves_count += 1
    
    # Summary
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print()
    print(f"Total Rounds: {final_round}")
    print(f"Elapsed: {elapsed/60:.1f} min ({final_round/elapsed:.2f} rounds/sec)")
    print()
    print(f"Final Stage: {curriculum.get_status()}")
    print(f"Weight Syncs: {sync_count}")
    print(f"Best Model Saves: {saves_count}")
    print()
    print(f"Stats: Score={total_stats['score']}, Kills={total_stats['kills']}, Deaths={total_stats['deaths']}")
    print(f"Model: {config['model_path']}")
    
    # Save results
    results = {
        'config': config,
        'final_round': final_round,
        'elapsed_seconds': elapsed,
        'final_stage': curriculum.current_stage,
        'weight_syncs': sync_count,
        'total_stats': total_stats,
        'best_performance': perf_tracker.best_score,
    }
    
    if rank == 0:
        results_file = os.path.join(config['results_dir'], 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="A3C: Weight Averaging (Federated Learning Style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
  - Each worker has independent local model
  - Workers update locally, send weights periodically
  - Main averages weights and distributes

Examples:
    python a3c_gpu_train.py --num-workers 4 --total-rounds 50000 --sync-interval 10
    python a3c_gpu_train.py --num-workers 8 --total-rounds 200000 --sync-interval 5
        """
    )
    
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--total-rounds", type=int, default=50000)
    parser.add_argument("--rounds-per-batch", type=int, default=5)
    parser.add_argument("--sync-interval", type=int, default=5,
                        help="Sync weights every N batches (lower = more frequent)")
    parser.add_argument("--global-weight-ratio", type=float, default=0.2,
                        help="Weight ratio for global model in averaging (0.0-1.0). "
                             "0.2 means: 0.8 * workers + 0.2 * global")
    parser.add_argument("--eval-window", type=int, default=200,
                        help="Window for win rate calculation")
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default="agent_code/ppo_agent/ppo_model.pt",
        help="Path to the (global) model checkpoint used for initialization and saving.",
    )
    parser.add_argument(
        "--results-dir",
        dest="results_dir",
        type=str,
        default=None,
        help="Directory to write worker stats and checkpoints (default: results/weight_avg_<timestamp>)",
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config = {
        'num_workers': args.num_workers,
        'total_rounds': args.total_rounds,
        'rounds_per_batch': args.rounds_per_batch,
        'sync_interval': args.sync_interval,
        'global_weight_ratio': args.global_weight_ratio,
        'eval_window': args.eval_window,
        'model_path': args.model_path,
        'results_dir': args.results_dir or f'results/weight_avg_{timestamp}',
    }
    
    run_adaptive_training(config)


if __name__ == "__main__":
    main()
