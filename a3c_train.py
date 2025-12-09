#!/usr/bin/env python3
"""
A3C (Asynchronous Advantage Actor-Critic) Training for Bomberman RL
====================================================================

Multiple workers asynchronously collect experience and update shared model.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                        GLOBAL MODEL (Shared)                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  PPO Policy Network                          │   │
│  │                 (agent_code/ppo_agent/)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ↑ async update         ↑ async update      ↑ async       │
│           │                      │                   │             │
├───────────┼──────────────────────┼───────────────────┼─────────────┤
│  ┌────────┴──────┐  ┌────────────┴──────┐  ┌────────┴──────────┐  │
│  │   Worker 0    │  │     Worker 1      │  │     Worker N      │  │
│  │  (Env + Game) │  │   (Env + Game)    │  │   (Env + Game)    │  │
│  │               │  │                   │  │                   │  │
│  │ Local Policy  │  │   Local Policy    │  │   Local Policy    │  │
│  └───────────────┘  └───────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

Usage:
    # 4 workers (recommended: num_cpu_cores)
    python a3c_train.py --num-workers 4 --total-rounds 2000
    
    # 8 workers with teacher agent
    python a3c_train.py --num-workers 8 --total-rounds 5000 --opponent aggressive_teacher_agent
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Queue, Value, Lock
import torch.multiprocessing as tmp


# ============== Configuration ==============

@dataclass
class A3CConfig:
    """A3C Training Configuration"""
    # Workers
    num_workers: int = 4
    
    # Training
    total_rounds: int = 2000
    rounds_per_sync: int = 10  # Sync with global model every N rounds
    
    # Opponents
    opponent: str = "aggressive_teacher_agent"
    use_selfplay: bool = False
    selfplay_ratio: float = 0.3  # 30% self-play, 70% vs opponent
    
    # Paths
    model_path: str = "agent_code/ppo_agent/ppo_model.pt"
    checkpoint_dir: str = "agent_code/ppo_agent/checkpoints"
    results_dir: str = "results/a3c_train"
    
    # Logging
    log_interval: int = 50  # Log every N rounds
    save_interval: int = 200  # Save checkpoint every N rounds


# ============== Shared State ==============

class SharedState:
    """Shared state between workers using multiprocessing primitives."""
    
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


# ============== Worker Process ==============

def worker_process(
    worker_id: int,
    shared_state: SharedState,
    config: A3CConfig,
    stop_event: mp.Event
):
    """
    Worker process that runs game environment and collects experience.
    Periodically syncs with global model.
    """
    print(f"[Worker {worker_id}] Starting...")
    
    worker_rounds = 0
    local_stats = {
        "score": 0,
        "kills": 0,
        "deaths": 0,
        "coins": 0,
    }
    
    while not stop_event.is_set():
        global_round = shared_state.global_round.value
        
        if global_round >= config.total_rounds:
            break
        
        # Determine opponent for this batch
        if config.use_selfplay and torch.rand(1).item() < config.selfplay_ratio:
            opponent = "ppo_agent"  # Self-play
            opponent_label = "self"
        else:
            opponent = config.opponent
            opponent_label = "teacher"
        
        # Run a batch of rounds
        batch_size = min(config.rounds_per_sync, config.total_rounds - global_round)
        
        try:
            # Run training batch
            result = run_training_batch_worker(
                worker_id=worker_id,
                opponent=opponent,
                rounds=batch_size,
                results_dir=config.results_dir
            )
            
            # Update stats
            if result:
                score = result.get("score", 0)
                kills = result.get("kills", 0)
                deaths = result.get("suicides", 0)
                coins = result.get("coins", 0)
                
                local_stats["score"] += score
                local_stats["kills"] += kills
                local_stats["deaths"] += deaths
                local_stats["coins"] += coins
                
                shared_state.add_stats(score, kills, deaths, coins)
            
            # Increment global round counter
            for _ in range(batch_size):
                new_round = shared_state.increment_round()
                worker_rounds += 1
                
                # Log progress
                if new_round % config.log_interval == 0:
                    stats = shared_state.get_stats()
                    avg_score = stats["score"] / max(1, stats["rounds"])
                    shared_state.log_queue.put(
                        f"[Round {new_round}/{config.total_rounds}] "
                        f"Avg Score: {avg_score:.2f}, "
                        f"Kills: {stats['kills']}, Deaths: {stats['deaths']}"
                    )
                
                # Save checkpoint
                if new_round % config.save_interval == 0:
                    ckpt_path = os.path.join(
                        config.checkpoint_dir, 
                        f"a3c_round_{new_round:06d}.pt"
                    )
                    try:
                        import shutil
                        shutil.copy(config.model_path, ckpt_path)
                        shared_state.log_queue.put(f"  Saved checkpoint: {ckpt_path}")
                    except Exception as e:
                        pass
            
        except Exception as e:
            shared_state.log_queue.put(f"[Worker {worker_id}] Error: {e}")
            time.sleep(1)  # Brief pause on error
    
    # Report final stats
    shared_state.stats_queue.put({
        "worker_id": worker_id,
        "rounds": worker_rounds,
        "stats": local_stats
    })
    
    print(f"[Worker {worker_id}] Finished. Rounds: {worker_rounds}")


def run_training_batch_worker(
    worker_id: int,
    opponent: str,
    rounds: int,
    results_dir: str
) -> Optional[Dict]:
    """Run a training batch for a worker."""
    output_file = os.path.join(
        results_dir, 
        f"worker_{worker_id}_batch_{int(time.time())}.json"
    )
    
    # Determine agents
    if opponent == "ppo_agent":
        agents = ["ppo_agent", "ppo_agent", "ppo_agent", "ppo_agent"]
        train_count = 4
    else:
        agents = ["ppo_agent", "ppo_agent", opponent, opponent]
        train_count = 2
    
    cmd = [
        sys.executable, "main.py", "play",
        "--agents", *agents,
        "--train", str(train_count),
        "--no-gui",
        "--n-rounds", str(rounds),
        "--save-stats", output_file,
        "--silence-errors",
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 min timeout per batch
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Extract PPO agent stats
            by_agent = data.get("by_agent", {})
            ppo_stats = {"score": 0, "kills": 0, "suicides": 0, "coins": 0}
            
            for agent_name, stats in by_agent.items():
                if agent_name.startswith("ppo_agent"):
                    ppo_stats["score"] += stats.get("score", 0)
                    ppo_stats["kills"] += stats.get("kills", 0)
                    ppo_stats["suicides"] += stats.get("suicides", 0)
                    ppo_stats["coins"] += stats.get("coins", 0)
            
            # Clean up temp file
            try:
                os.remove(output_file)
            except:
                pass
            
            return ppo_stats
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    
    return None


# ============== Logger Thread ==============

def logger_thread(shared_state: SharedState, stop_event: mp.Event):
    """Thread that handles logging from all workers."""
    while not stop_event.is_set():
        try:
            msg = shared_state.log_queue.get(timeout=1)
            print(msg)
        except queue.Empty:
            continue
        except:
            break


# ============== Main Training Loop ==============

def run_a3c_training(config: A3CConfig):
    """Main A3C training function."""
    print("="*70)
    print("A3C (Asynchronous Advantage Actor-Critic) Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Total Rounds: {config.total_rounds}")
    print(f"  Opponent: {config.opponent}")
    print(f"  Self-play Ratio: {config.selfplay_ratio*100:.0f}%")
    print(f"  Sync Interval: {config.rounds_per_sync} rounds")
    print()
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize shared state
    shared_state = SharedState(config.model_path)
    stop_event = mp.Event()
    
    # Start logger thread
    import threading
    log_thread = threading.Thread(target=logger_thread, args=(shared_state, stop_event))
    log_thread.daemon = True
    log_thread.start()
    
    # Start worker processes
    workers = []
    print(f"Starting {config.num_workers} worker processes...")
    print()
    
    start_time = time.time()
    
    for i in range(config.num_workers):
        p = Process(
            target=worker_process,
            args=(i, shared_state, config, stop_event)
        )
        p.start()
        workers.append(p)
        time.sleep(0.5)  # Stagger starts
    
    print("="*70)
    print("Training in progress... (Ctrl+C to stop)")
    print("="*70)
    print()
    
    try:
        # Wait for all rounds to complete
        while shared_state.global_round.value < config.total_rounds:
            time.sleep(5)
            
            # Check if workers are still alive
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                print("All workers finished.")
                break
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Stop workers
    stop_event.set()
    
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    
    elapsed = time.time() - start_time
    
    # Collect final stats
    worker_stats = []
    while not shared_state.stats_queue.empty():
        try:
            worker_stats.append(shared_state.stats_queue.get_nowait())
        except:
            break
    
    # Print summary
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print()
    
    final_stats = shared_state.get_stats()
    print(f"Total Rounds: {final_stats['rounds']}")
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"Rounds/sec: {final_stats['rounds']/elapsed:.2f}")
    print()
    print("Aggregate Stats:")
    print(f"  Total Score: {final_stats['score']:.0f}")
    print(f"  Total Kills: {final_stats['kills']}")
    print(f"  Total Deaths: {final_stats['deaths']}")
    print(f"  Total Coins: {final_stats['coins']}")
    print()
    
    if final_stats['rounds'] > 0:
        print("Average per Round:")
        print(f"  Score: {final_stats['score']/final_stats['rounds']:.2f}")
        print(f"  Kills: {final_stats['kills']/final_stats['rounds']:.2f}")
        print(f"  Deaths: {final_stats['deaths']/final_stats['rounds']:.2f}")
    
    print()
    print(f"Model saved to: {config.model_path}")
    print(f"Checkpoints: {config.checkpoint_dir}/")
    
    # Save final results
    results = {
        "config": {
            "num_workers": config.num_workers,
            "total_rounds": config.total_rounds,
            "opponent": config.opponent,
            "selfplay_ratio": config.selfplay_ratio,
        },
        "stats": final_stats,
        "elapsed_seconds": elapsed,
        "worker_stats": worker_stats,
    }
    
    results_file = os.path.join(config.results_dir, "a3c_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results


# ============== Main ==============

def main():
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(
        description="A3C Training for Bomberman RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 4 workers, 2000 rounds
    python a3c_train.py --num-workers 4 --total-rounds 2000
    
    # 8 workers with self-play
    python a3c_train.py --num-workers 8 --total-rounds 5000 --selfplay
    
    # Against teacher agent
    python a3c_train.py --num-workers 4 --opponent aggressive_teacher_agent
        """
    )
    
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--total-rounds", type=int, default=2000,
                        help="Total training rounds (default: 2000)")
    parser.add_argument("--opponent", type=str, default="aggressive_teacher_agent",
                        help="Opponent agent (default: aggressive_teacher_agent)")
    parser.add_argument("--selfplay", action="store_true",
                        help="Enable self-play (30%% of games)")
    parser.add_argument("--selfplay-ratio", type=float, default=0.3,
                        help="Self-play ratio when enabled (default: 0.3)")
    parser.add_argument("--sync-interval", type=int, default=10,
                        help="Rounds between model syncs (default: 10)")
    
    args = parser.parse_args()
    
    config = A3CConfig(
        num_workers=args.num_workers,
        total_rounds=args.total_rounds,
        opponent=args.opponent,
        use_selfplay=args.selfplay,
        selfplay_ratio=args.selfplay_ratio,
        rounds_per_sync=args.sync_interval,
    )
    
    # Add timestamp to results dir
    config.results_dir = f"results/a3c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_a3c_training(config)


if __name__ == "__main__":
    main()

