#!/usr/bin/env python3
"""
Self-Play Training System for Bomberman RL
===========================================

A3C-style distributed learning + AlphaZero-style self-play

Phase 1: Train vs aggressive_teacher_agent until win_rate >= threshold
Phase 2: Self-play - compete against past versions of itself

Usage:
    # Single process
    python self_play_train.py
    
    # Distributed (4 workers)
    python self_play_train.py --num-workers 4
    
    # Custom settings
    python self_play_train.py --teacher-rounds 1000 --selfplay-rounds 2000 --win-threshold 0.25
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


# ============== Configuration ==============

DEFAULT_CONFIG = {
    # Phase 1: Teacher training
    "teacher_rounds": 1000,
    "teacher_agent": "aggressive_teacher_agent",
    "win_threshold": 0.25,  # 25% win rate to move to self-play
    "eval_interval": 100,   # Evaluate every N rounds
    
    # Phase 2: Self-play
    "selfplay_rounds": 2000,
    "selfplay_checkpoint_interval": 200,  # Save checkpoint every N rounds
    "selfplay_opponent_pool_size": 5,     # Keep last N checkpoints as opponents
    "selfplay_latest_prob": 0.5,          # Probability to play against latest version
    
    # Distributed settings
    "num_workers": 1,
    "rounds_per_worker": 50,
    
    # Paths
    "checkpoint_dir": "agent_code/ppo_agent/checkpoints",
    "model_path": "agent_code/ppo_agent/ppo_model.pt",
    "results_dir": "results/self_play",
}


# ============== Utility Functions ==============

def run_training_batch(
    agents: List[str],
    train_count: int,
    rounds: int,
    output_file: str,
    extra_args: List[str] = None
) -> Dict:
    """Run a training batch and return results."""
    cmd = [
        sys.executable, "main.py", "play",
        "--agents", *agents,
        "--train", str(train_count),
        "--no-gui",
        "--n-rounds", str(rounds),
        "--save-stats", output_file,
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Training batch failed: {result.stderr}")
        return {}
    
    # Load results
    try:
        with open(output_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load results: {e}")
        return {}


def calculate_win_rate(results: Dict, ppo_agent_prefix: str = "ppo_agent") -> float:
    """Calculate win rate for PPO agents."""
    if not results:
        return 0.0
    
    by_agent = results.get("by_agent", {})
    
    ppo_score = 0
    opponent_score = 0
    
    for agent_name, stats in by_agent.items():
        score = stats.get("score", 0)
        if agent_name.startswith(ppo_agent_prefix):
            ppo_score += score
        else:
            opponent_score += score
    
    total = ppo_score + opponent_score
    if total == 0:
        return 0.0
    
    return ppo_score / total


def calculate_detailed_stats(results: Dict, ppo_agent_prefix: str = "ppo_agent") -> Dict:
    """Calculate detailed stats for PPO agents."""
    by_agent = results.get("by_agent", {})
    
    stats = {
        "total_score": 0,
        "total_kills": 0,
        "total_coins": 0,
        "total_suicides": 0,
        "rounds": 0,
    }
    
    for agent_name, agent_stats in by_agent.items():
        if agent_name.startswith(ppo_agent_prefix):
            stats["total_score"] += agent_stats.get("score", 0)
            stats["total_kills"] += agent_stats.get("kills", 0)
            stats["total_coins"] += agent_stats.get("coins", 0)
            stats["total_suicides"] += agent_stats.get("suicides", 0)
            stats["rounds"] = max(stats["rounds"], agent_stats.get("rounds", 0))
    
    return stats


def save_checkpoint(checkpoint_dir: str, model_path: str, checkpoint_name: str) -> str:
    """Save a checkpoint of the current model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
    shutil.copy(model_path, checkpoint_path)
    return checkpoint_path


def get_checkpoint_list(checkpoint_dir: str) -> List[str]:
    """Get list of available checkpoints sorted by creation time."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pt'):
            path = os.path.join(checkpoint_dir, f)
            checkpoints.append((path, os.path.getmtime(path)))
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in checkpoints]


def create_selfplay_agent(checkpoint_path: str, agent_name: str = "selfplay_opponent") -> str:
    """Create a temporary agent that uses a specific checkpoint."""
    # Create temporary agent directory
    agent_dir = f"agent_code/{agent_name}"
    os.makedirs(agent_dir, exist_ok=True)
    
    # Copy checkpoint as the agent's model
    shutil.copy(checkpoint_path, os.path.join(agent_dir, "ppo_model.pt"))
    
    # Create callbacks.py that loads this model
    callbacks_code = f'''
"""Self-play opponent agent (auto-generated)"""
from agent_code.ppo_agent.callbacks import setup, act, state_to_features, ACTIONS

# This agent uses the same code as ppo_agent but with a different model checkpoint
'''
    
    with open(os.path.join(agent_dir, "callbacks.py"), 'w') as f:
        f.write(callbacks_code)
    
    # Create empty __init__.py
    Path(os.path.join(agent_dir, "__init__.py")).touch()
    
    return agent_name


# ============== Phase 1: Teacher Training ==============

def run_teacher_phase(config: Dict) -> Tuple[bool, Dict]:
    """
    Phase 1: Train against teacher agent until win threshold is reached.
    Returns (success, final_stats)
    """
    print("\n" + "="*60)
    print("PHASE 1: TEACHER TRAINING")
    print("="*60)
    print(f"Target: {config['win_threshold']*100:.0f}% win rate vs {config['teacher_agent']}")
    print(f"Max rounds: {config['teacher_rounds']}")
    print()
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    total_rounds = 0
    best_win_rate = 0.0
    all_stats = []
    
    while total_rounds < config['teacher_rounds']:
        batch_rounds = min(config['eval_interval'], config['teacher_rounds'] - total_rounds)
        
        print(f"Training batch: rounds {total_rounds + 1} to {total_rounds + batch_rounds}...")
        
        output_file = os.path.join(
            config['results_dir'], 
            f"teacher_batch_{total_rounds:05d}.json"
        )
        
        results = run_training_batch(
            agents=["ppo_agent", "ppo_agent", config['teacher_agent'], config['teacher_agent']],
            train_count=2,
            rounds=batch_rounds,
            output_file=output_file
        )
        
        total_rounds += batch_rounds
        
        # Calculate stats
        win_rate = calculate_win_rate(results)
        stats = calculate_detailed_stats(results)
        stats['win_rate'] = win_rate
        stats['total_rounds'] = total_rounds
        all_stats.append(stats)
        
        best_win_rate = max(best_win_rate, win_rate)
        
        print(f"  Rounds: {total_rounds}/{config['teacher_rounds']}")
        print(f"  Win rate: {win_rate*100:.1f}% (best: {best_win_rate*100:.1f}%)")
        print(f"  Kills: {stats['total_kills']}, Suicides: {stats['total_suicides']}")
        print()
        
        # Check if threshold reached
        if win_rate >= config['win_threshold']:
            print(f"✓ Win threshold {config['win_threshold']*100:.0f}% reached!")
            
            # Save checkpoint before moving to self-play
            ckpt_name = f"pre_selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_checkpoint(config['checkpoint_dir'], config['model_path'], ckpt_name)
            print(f"  Saved checkpoint: {ckpt_name}")
            
            return True, {"all_stats": all_stats, "final_win_rate": win_rate}
    
    print(f"✗ Did not reach win threshold after {total_rounds} rounds")
    print(f"  Best win rate: {best_win_rate*100:.1f}%")
    print("  Continuing to self-play anyway...")
    
    # Save checkpoint anyway
    ckpt_name = f"pre_selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_checkpoint(config['checkpoint_dir'], config['model_path'], ckpt_name)
    
    return False, {"all_stats": all_stats, "final_win_rate": best_win_rate}


# ============== Phase 2: Self-Play ==============

def run_selfplay_phase(config: Dict) -> Dict:
    """
    Phase 2: Self-play training - compete against past versions.
    """
    print("\n" + "="*60)
    print("PHASE 2: SELF-PLAY (AlphaZero Style)")
    print("="*60)
    print(f"Rounds: {config['selfplay_rounds']}")
    print(f"Checkpoint interval: {config['selfplay_checkpoint_interval']}")
    print(f"Opponent pool size: {config['selfplay_opponent_pool_size']}")
    print()
    
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    total_rounds = 0
    all_stats = []
    generation = 0
    
    # Save initial checkpoint as generation 0
    gen0_name = f"gen_{generation:04d}"
    save_checkpoint(config['checkpoint_dir'], config['model_path'], gen0_name)
    print(f"Saved initial generation: {gen0_name}")
    
    while total_rounds < config['selfplay_rounds']:
        batch_rounds = min(
            config['selfplay_checkpoint_interval'], 
            config['selfplay_rounds'] - total_rounds
        )
        
        # Get available checkpoints
        checkpoints = get_checkpoint_list(config['checkpoint_dir'])
        
        if len(checkpoints) < 2:
            # Not enough checkpoints, train against self (latest)
            print(f"\nSelf-play batch (vs self): rounds {total_rounds + 1} to {total_rounds + batch_rounds}")
            
            output_file = os.path.join(
                config['results_dir'], 
                f"selfplay_batch_{total_rounds:05d}.json"
            )
            
            results = run_training_batch(
                agents=["ppo_agent", "ppo_agent", "ppo_agent", "ppo_agent"],
                train_count=4,  # All 4 are training
                rounds=batch_rounds,
                output_file=output_file
            )
        else:
            # Select opponent from pool
            pool = checkpoints[:config['selfplay_opponent_pool_size']]
            
            # Probability-based selection: prefer recent versions
            if random.random() < config['selfplay_latest_prob'] and len(pool) > 1:
                opponent_ckpt = pool[0]  # Latest
            else:
                opponent_ckpt = random.choice(pool)
            
            opponent_name = Path(opponent_ckpt).stem
            print(f"\nSelf-play batch (vs {opponent_name}): rounds {total_rounds + 1} to {total_rounds + batch_rounds}")
            
            # Create temporary opponent agent
            temp_agent = create_selfplay_agent(opponent_ckpt, "selfplay_opponent")
            
            output_file = os.path.join(
                config['results_dir'], 
                f"selfplay_batch_{total_rounds:05d}.json"
            )
            
            results = run_training_batch(
                agents=["ppo_agent", "ppo_agent", temp_agent, temp_agent],
                train_count=2,
                rounds=batch_rounds,
                output_file=output_file
            )
        
        total_rounds += batch_rounds
        
        # Calculate stats
        win_rate = calculate_win_rate(results)
        stats = calculate_detailed_stats(results)
        stats['win_rate'] = win_rate
        stats['total_rounds'] = total_rounds
        stats['generation'] = generation
        all_stats.append(stats)
        
        print(f"  Rounds: {total_rounds}/{config['selfplay_rounds']}")
        print(f"  Win rate vs opponent: {win_rate*100:.1f}%")
        print(f"  Kills: {stats['total_kills']}, Suicides: {stats['total_suicides']}")
        
        # Save new generation checkpoint
        if total_rounds % config['selfplay_checkpoint_interval'] == 0:
            generation += 1
            gen_name = f"gen_{generation:04d}"
            save_checkpoint(config['checkpoint_dir'], config['model_path'], gen_name)
            print(f"  ✓ Saved generation {generation}: {gen_name}")
        
        print()
    
    # Final checkpoint
    final_name = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_checkpoint(config['checkpoint_dir'], config['model_path'], final_name)
    print(f"Saved final model: {final_name}")
    
    return {"all_stats": all_stats, "generations": generation}


# ============== Distributed Training ==============

def run_worker(worker_id: int, config: Dict) -> Dict:
    """Run a training worker (for distributed training)."""
    print(f"Worker {worker_id} starting...")
    
    # Each worker runs a subset of rounds
    worker_config = config.copy()
    worker_config['teacher_rounds'] = config['rounds_per_worker']
    worker_config['selfplay_rounds'] = config['rounds_per_worker']
    worker_config['results_dir'] = os.path.join(config['results_dir'], f"worker_{worker_id}")
    
    # Phase 1
    success, phase1_stats = run_teacher_phase(worker_config)
    
    # Phase 2
    phase2_stats = run_selfplay_phase(worker_config)
    
    return {
        "worker_id": worker_id,
        "phase1": phase1_stats,
        "phase2": phase2_stats
    }


def run_distributed(config: Dict) -> Dict:
    """Run distributed training with multiple workers."""
    print("\n" + "="*60)
    print(f"DISTRIBUTED TRAINING ({config['num_workers']} workers)")
    print("="*60)
    print()
    
    results = []
    
    if config['num_workers'] == 1:
        # Single process
        result = run_worker(0, config)
        results.append(result)
    else:
        # Multi-process
        with ProcessPoolExecutor(max_workers=config['num_workers']) as executor:
            futures = {
                executor.submit(run_worker, i, config): i 
                for i in range(config['num_workers'])
            }
            
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Worker {worker_id} completed")
                except Exception as e:
                    print(f"Worker {worker_id} failed: {e}")
    
    return {"workers": results}


# ============== Evaluation ==============

def run_final_evaluation(config: Dict) -> Dict:
    """Evaluate the final model against various opponents."""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    print()
    
    opponents = [
        ("rule_based_agent", "rule_based_agent"),
        (config['teacher_agent'], config['teacher_agent']),
        ("rule_based_agent", config['teacher_agent']),
    ]
    
    results = {}
    
    for opp1, opp2 in opponents:
        opponent_name = f"{opp1}_vs_{opp2}"
        print(f"Evaluating vs {opponent_name}...")
        
        output_file = os.path.join(config['results_dir'], f"eval_{opponent_name}.json")
        
        eval_results = run_training_batch(
            agents=["ppo_agent", "ppo_agent", opp1, opp2],
            train_count=0,  # No training, just evaluation
            rounds=50,
            output_file=output_file
        )
        
        win_rate = calculate_win_rate(eval_results)
        stats = calculate_detailed_stats(eval_results)
        
        results[opponent_name] = {
            "win_rate": win_rate,
            "stats": stats
        }
        
        print(f"  Win rate: {win_rate*100:.1f}%")
        print(f"  Kills: {stats['total_kills']}, Suicides: {stats['total_suicides']}")
        print()
    
    return results


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="Self-Play Training System (A3C + AlphaZero style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python self_play_train.py
    
    # Longer training with lower threshold
    python self_play_train.py --teacher-rounds 2000 --selfplay-rounds 3000 --win-threshold 0.20
    
    # Distributed training (4 workers)
    python self_play_train.py --num-workers 4
    
    # Skip teacher phase (direct self-play)
    python self_play_train.py --skip-teacher
        """
    )
    
    parser.add_argument("--teacher-rounds", type=int, default=DEFAULT_CONFIG['teacher_rounds'],
                        help="Rounds for teacher training phase")
    parser.add_argument("--selfplay-rounds", type=int, default=DEFAULT_CONFIG['selfplay_rounds'],
                        help="Rounds for self-play phase")
    parser.add_argument("--win-threshold", type=float, default=DEFAULT_CONFIG['win_threshold'],
                        help="Win rate threshold to move to self-play (0.0-1.0)")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG['num_workers'],
                        help="Number of parallel workers")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher phase and go directly to self-play")
    parser.add_argument("--skip-selfplay", action="store_true",
                        help="Skip self-play phase")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing model")
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config['teacher_rounds'] = args.teacher_rounds
    config['selfplay_rounds'] = args.selfplay_rounds
    config['win_threshold'] = args.win_threshold
    config['num_workers'] = args.num_workers
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['results_dir'] = f"results/self_play_{timestamp}"
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['results_dir'], "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*60)
    print("SELF-PLAY TRAINING SYSTEM")
    print("A3C-style + AlphaZero-style Self-Play")
    print("="*60)
    print(f"\nResults directory: {config['results_dir']}")
    print(f"Checkpoints: {config['checkpoint_dir']}")
    print()
    
    all_results = {"config": config}
    
    try:
        if args.eval_only:
            # Only evaluation
            eval_results = run_final_evaluation(config)
            all_results['evaluation'] = eval_results
        else:
            # Phase 1: Teacher training
            if not args.skip_teacher:
                success, phase1_results = run_teacher_phase(config)
                all_results['phase1'] = phase1_results
            
            # Phase 2: Self-play
            if not args.skip_selfplay:
                phase2_results = run_selfplay_phase(config)
                all_results['phase2'] = phase2_results
            
            # Final evaluation
            eval_results = run_final_evaluation(config)
            all_results['evaluation'] = eval_results
        
        # Save all results
        with open(os.path.join(config['results_dir'], "all_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {config['results_dir']}")
        print(f"Model saved to: {config['model_path']}")
        print(f"Checkpoints: {config['checkpoint_dir']}")
        print("\nFinal evaluation:")
        for opponent, results in all_results.get('evaluation', {}).items():
            print(f"  vs {opponent}: {results['win_rate']*100:.1f}% win rate")
        
        # Generate plots
        print("\nGenerating plots...")
        try:
            subprocess.run([
                sys.executable, "plot_results.py",
                os.path.join(config['results_dir'], "all_results.json")
            ], capture_output=True)
            print("  ✓ Plots generated")
        except:
            print("  (Plot generation skipped)")
        
        print("\n" + "="*60)
        print("Done! Check the results directory for details.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial results saved to: {config['results_dir']}")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()

