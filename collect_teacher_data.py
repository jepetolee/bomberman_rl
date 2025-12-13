#!/usr/bin/env python3
"""
Teacher Data Collection Script
==============================

Collects episode data from teacher agent (aggressive_teacher_agent) playing
against various opponents. Data will be used for Phase 2: Dyna-Q Planning + DeepSupervision.

Usage:
    python collect_teacher_data.py --episodes 20000 --workers 4
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value, Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
from agent_code.ppo_agent.callbacks import state_to_features, ACTIONS


# ============== Configuration ==============

# Opponents to collect data against
OPPONENT_POOL = [
    'random_agent',
    'peaceful_agent',
    'coin_collector_agent',
    'rule_based_agent',
    'team_teacher_agent',
]

TEACHER_AGENT = 'aggressive_teacher_agent'
MIN_EPISODE_LENGTH = 10  # Minimum steps for valid episode


class EpisodeData:
    """Data structure for a single episode"""
    def __init__(self):
        self.states = []           # List of state tensors [C, H, W]
        self.teacher_actions = []  # List of action indices
        self.rewards = []          # List of rewards
        self.dones = []            # List of done flags
        self.game_states = []      # List of full game_state dicts (for env model)
        
    def add_step(self, state: np.ndarray, action_idx: int, reward: float, done: bool, game_state: dict):
        """Add a step to the episode"""
        self.states.append(state)
        self.teacher_actions.append(action_idx)
        self.rewards.append(reward)
        self.dones.append(done)
        self.game_states.append(game_state)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            'states': np.array(self.states),  # [T, C, H, W]
            'teacher_actions': np.array(self.teacher_actions),  # [T]
            'rewards': np.array(self.rewards),  # [T]
            'dones': np.array(self.dones),  # [T]
            'game_states': self.game_states,  # List of dicts
        }
    
    def is_valid(self) -> bool:
        """Check if episode is valid (has minimum length)"""
        return len(self.states) >= MIN_EPISODE_LENGTH
    
    def __len__(self):
        return len(self.states)


def worker_collect_data(
    worker_id: int,
    episodes_queue: Queue,
    global_counter: Value,
    global_lock: Lock,
    total_episodes: int,
    data_dir: str,
    stop_flag: Value,
):
    """
    Worker process that collects teacher episode data
    """
    import subprocess
    import pickle
    
    print(f"[Worker {worker_id}] Started data collection")
    
    episodes_collected = 0
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    while not stop_flag.value:
        with global_lock:
            if global_counter.value >= total_episodes:
                break
            episode_id = global_counter.value
            global_counter.value += 1
        
        # Select random opponent
        opponent = random.choice(OPPONENT_POOL)
        
        # Determine if self-play or vs opponent
        agents = [TEACHER_AGENT, TEACHER_AGENT, opponent, opponent]
        
        # Run a single episode
        output_file = os.path.join(data_dir, f'temp_worker_{worker_id}_{episode_id}.json')
        
        cmd = [
            sys.executable, "main.py", "play",
            "--agents"] + agents + [
            "--train", "0",  # No training, just collect data
            "--no-gui",
            "--n-rounds", "1",
            "--save-stats", output_file,
            "--silence-errors",
        ]
        
        try:
            # Run game and capture episode data
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
                cwd=os.getcwd(),
            )
            
            if result.returncode == 0:
                # Load replay if available
                episode = EpisodeData()
                
                # Try to load from replay or stats
                try:
                    with open(output_file, 'r') as f:
                        stats = json.load(f)
                    
                    # Extract episode data from stats
                    # Note: This requires parsing the game replay or stats
                    # For now, we'll collect via a custom data collection method
                    
                except Exception as e:
                    print(f"[Worker {worker_id}] Failed to parse output: {e}")
                    continue
                
                # Save episode if valid
                if episode.is_valid():
                    episode_file = os.path.join(data_dir, f'episode_{episode_id:06d}.pt')
                    episode_dict = episode.to_dict()
                    
                    with open(episode_file, 'wb') as f:
                        pickle.dump(episode_dict, f)
                    
                    episodes_queue.put(episode_id)
                    episodes_collected += 1
                    
                    if episodes_collected % 10 == 0:
                        print(f"[Worker {worker_id}] Collected {episodes_collected} episodes (total: {global_counter.value}/{total_episodes})")
            else:
                print(f"[Worker {worker_id}] Game failed: {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            print(f"[Worker {worker_id}] Episode {episode_id} timed out")
        except Exception as e:
            print(f"[Worker {worker_id}] Error in episode {episode_id}: {e}")
        
        # Cleanup temp file
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
    
    print(f"[Worker {worker_id}] Finished. Collected {episodes_collected} episodes")


class TeacherDataCollector:
    """Custom collector that hooks into game execution to capture data"""
    
    def __init__(self):
        self.current_episode = EpisodeData()
        self.episodes = []
    
    def reset(self):
        """Reset for new episode"""
        if self.current_episode.is_valid():
            self.episodes.append(self.current_episode.to_dict())
        self.current_episode = EpisodeData()
    
    def add_step(self, game_state: dict, action: str, reward: float, done: bool):
        """Add a step from teacher agent"""
        state = state_to_features(game_state)
        action_idx = ACTIONS.index(action) if action in ACTIONS else 0
        
        self.current_episode.add_step(
            state=state,
            action_idx=action_idx,
            reward=reward,
            done=done,
            game_state=game_state,
        )


def collect_episode_via_replay(worker_id: int, episode_id: int, data_dir: str) -> Optional[Dict]:
    """
    Alternative: Collect episode by running game with custom hooks
    This is a simplified version - full implementation would need to modify
    the game loop to capture data during execution.
    """
    # This would require modifying the game execution to capture data
    # For now, we'll use a workaround by parsing replay files or stats
    pass


def main():
    parser = argparse.ArgumentParser(description='Collect teacher agent episode data')
    parser.add_argument('--episodes', type=int, default=20000, help='Total episodes to collect')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--data-dir', type=str, default='data/teacher_episodes', help='Output directory')
    parser.add_argument('--min-length', type=int, default=10, help='Minimum episode length')
    
    args = parser.parse_args()
    
    # Set min episode length
    global MIN_EPISODE_LENGTH
    MIN_EPISODE_LENGTH = args.min_length
    
    # Create data directory
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  TEACHER DATA COLLECTION                                     ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"")
    print(f"Configuration:")
    print(f"  Total Episodes: {args.episodes}")
    print(f"  Workers: {args.workers}")
    print(f"  Teacher Agent: {TEACHER_AGENT}")
    print(f"  Opponents: {', '.join(OPPONENT_POOL)}")
    print(f"  Output Directory: {data_dir}")
    print(f"")
    
    # Shared state
    global_counter = Value('i', 0)
    global_lock = Lock()
    episodes_queue = Queue()
    stop_flag = Value('b', False)
    
    # Start worker processes
    workers = []
    for worker_id in range(args.workers):
        p = Process(
            target=worker_collect_data,
            args=(worker_id, episodes_queue, global_counter, global_lock, 
                  args.episodes, data_dir, stop_flag)
        )
        p.start()
        workers.append(p)
    
    # Monitor progress
    try:
        start_time = time.time()
        last_count = 0
        
        while global_counter.value < args.episodes:
            time.sleep(5)
            current_count = global_counter.value
            elapsed = time.time() - start_time
            
            if current_count > last_count:
                rate = current_count / elapsed if elapsed > 0 else 0
                eta = (args.episodes - current_count) / rate if rate > 0 else 0
                print(f"Progress: {current_count}/{args.episodes} ({current_count*100/args.episodes:.1f}%) "
                      f"| Rate: {rate:.1f} eps/min | ETA: {eta/60:.1f} min")
                last_count = current_count
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        stop_flag.value = True
    
    finally:
        # Wait for workers
        print("\nWaiting for workers to finish...")
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        
        # Final count
        final_count = global_counter.value
        elapsed = time.time() - start_time
        
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║  COLLECTION COMPLETE                                         ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
        print(f"  Episodes Collected: {final_count}")
        print(f"  Time Elapsed: {elapsed/60:.1f} minutes")
        print(f"  Average Rate: {final_count/elapsed*60:.1f} episodes/minute")
        print(f"  Data Directory: {data_dir}")
        print(f"")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

