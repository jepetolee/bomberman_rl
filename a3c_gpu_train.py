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
    from agent_code.ppo_agent.models.vit import PolicyValueViT
    import settings as s
    
    mixer = os.environ.get("BOMBER_VIT_MIXER", "attn")
    embed_dim = int(os.environ.get("BOMBER_VIT_DIM", "64"))
    depth = int(os.environ.get("BOMBER_VIT_DEPTH", "2"))
    num_heads = int(os.environ.get("BOMBER_VIT_HEADS", "4"))
    
    model = PolicyValueViT(
        in_channels=9,
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
            model.load_state_dict(state_dict)
        except Exception as e:
            pass
    
    model = model.to(device)
    return model


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
        
        # Build agent list
        agent_list = ['ppo_agent', 'ppo_agent', opp1, opp2]
        shuffled_agents = [agent_list[i] for i in agent_order]
        
        batch_rounds = min(rounds_per_batch, total_rounds - current_round)
        
        # Set environment variables for subprocess - MUST pass via env parameter
        # os.environ changes don't propagate to subprocess automatically
        worker_env = os.environ.copy()
        worker_env['PPO_MODEL_PATH'] = local_model_path
        worker_env['A3C_RESULTS_DIR'] = results_dir  # Pass results_dir so PPO can save stats here
        worker_env['A3C_NUM_WORKERS'] = str(num_workers)  # Also pass worker count for epsilon
        
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
            
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                by_agent = data.get("by_agent", {})
                for agent_name, stats in by_agent.items():
                    score = stats.get("score", 0)
                    kills = stats.get("kills", 0)
                    deaths = stats.get("suicides", 0)
                    
                    if agent_name.startswith("ppo_agent"):
                        batch_stats['team_a_score'] += score
                        batch_stats['kills'] += kills
                        batch_stats['deaths'] += deaths
                    else:
                        batch_stats['team_b_score'] += score
                
                try:
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
                            local_model.load_state_dict(global_weights)
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
                        local_model.load_state_dict(global_weights)
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
    print("="*70)
    print("A3C GPU: Weight Averaging (Federated Learning Style)")
    print("="*70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print(f"  Workers: {config['num_workers']}")
    print(f"  Max Rounds: {config['total_rounds']}")
    print(f"  Sync Interval: {config['sync_interval']} batches")
    print(f"  Eval Window: {config['eval_window']} rounds")
    print()
    
    # Create directories - ensure absolute path
    config['results_dir'] = os.path.abspath(config['results_dir'])
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Create global model
    print("Creating global model...")
    global_model = create_model(config['model_path'], device)
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
            
            # Average weights if we have enough (at least 10 workers)
            # Wait for sufficient workers to ensure stable averaging
            min_workers_for_sync = 5
            if len(collected_weights) > 0:
                # Log when we have weights but not enough
                if len(collected_weights) < min_workers_for_sync:
                    if not hasattr(run_adaptive_training, '_last_weight_log') or \
                       time.time() - getattr(run_adaptive_training, '_last_weight_log', 0) > 30:
                        print(f"[Weight Queue] {len(collected_weights)}/{min_workers_for_sync} workers (waiting for more...)")
                        setattr(run_adaptive_training, '_last_weight_log', time.time())
            
            if len(collected_weights) >= min_workers_for_sync:
                sync_count += 1
                weight_list = list(collected_weights.values())
                
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
                torch.save(global_model.state_dict(), config['model_path'])
                
                print(f"[Sync {sync_count}] ★ Averaged {len(collected_weights)}/{config['num_workers']} workers "
                      f"+ {global_weight_ratio:.0%} global model")
                print(f"    Global model updated and saved!")
                
                # Clear collected weights after processing
                collected_weights.clear()
                
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
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config = {
        'num_workers': args.num_workers,
        'total_rounds': args.total_rounds,
        'rounds_per_batch': args.rounds_per_batch,
        'sync_interval': args.sync_interval,
        'global_weight_ratio': args.global_weight_ratio,
        'eval_window': args.eval_window,
        'model_path': 'agent_code/ppo_agent/ppo_model.pt',
        'results_dir': f'results/weight_avg_{timestamp}',
    }
    
    run_adaptive_training(config)


if __name__ == "__main__":
    main()
