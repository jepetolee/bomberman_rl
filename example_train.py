#!/usr/bin/env python3
"""
Example Training Script for Bomberman RL
교사 모델 기반 PPO 학습 예제

Usage:
    python example_train.py --mode basic
    python example_train.py --mode advanced
    python example_train.py --mode progressive
"""

import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print(f"✓ {description} - SUCCESS")
    else:
        print(f"✗ {description} - FAILED")
        return False
    
    return True


def train_basic():
    """Basic training: PPO vs rule_based agents."""
    print("\n" + "="*60)
    print("BASIC TRAINING MODE")
    print("="*60)
    
    # Train
    if not run_command([
        'python3', 'main.py', 'play',
        '--agents', 'ppo_agent', 'rule_based_agent', 'rule_based_agent', 'rule_based_agent',
        '--train', '1',
        '--no-gui',
        '--n-rounds', '200',
        '--save-stats', 'results/example_basic.json'
    ], "Training PPO agent (200 rounds vs rule_based)"):
        return
    
    # Plot
    run_command([
        'python3', 'plot_results.py',
        'results/example_basic.json'
    ], "Generating training charts")
    
    print("\n✓ Basic training complete!")
    print("Check results/example_basic_*.png for visualizations")


def train_advanced():
    """Advanced training: PPO vs aggressive teacher agents."""
    print("\n" + "="*60)
    print("ADVANCED TRAINING MODE")
    print("="*60)
    
    # Train
    if not run_command([
        'python3', 'main.py', 'play',
        '--agents', 'ppo_agent', 'ppo_agent', 
        'aggressive_teacher_agent', 'aggressive_teacher_agent',
        '--train', '2',
        '--no-gui',
        '--n-rounds', '300',
        '--save-stats', 'results/example_advanced.json'
    ], "Training PPO agents (300 rounds vs teacher agents)"):
        return
    
    # Plot
    run_command([
        'python3', 'plot_results.py',
        'results/example_advanced.json',
        '--include', 'ppo_agent',
        '--rolling', '30'
    ], "Generating training charts")
    
    print("\n✓ Advanced training complete!")
    print("Check results/example_advanced_*.png for visualizations")


def train_progressive():
    """Progressive training: Gradually increasing difficulty with teacher agent as final boss."""
    print("\n" + "="*60)
    print("PROGRESSIVE TRAINING MODE (Teacher Agent as Final Boss)")
    print("="*60)
    
    # Train
    if not run_command([
        'python3', 'main.py', 'play',
        '--agents', 'ppo_agent', 'ppo_agent',
        '--train', '2',
        '--dynamic-opponents',
        '--opponent-pool', 'peaceful_agent', 'coin_collector_agent', 'random_agent',
        '--rb-agent', 'aggressive_teacher_agent',
        '--rb-prob-start', '0.05',
        '--rb-prob-end', '0.7',
        '--n-rounds', '500',
        '--no-gui',
        '--save-stats', 'results/example_progressive_all.json'
    ], "Progressive training (500 rounds, 5%->70% teacher agent probability)"):
        return
    
    # Plot
    run_command([
        'python3', 'plot_results.py',
        'results/example_progressive_all.json',
        '--include', 'ppo_agent',
        '--rolling', '50'
    ], "Generating training charts")
    
    # Analyze
    run_command([
        'python3', 'analyze_matchups.py',
        'results/example_progressive_all.json'
    ], "Analyzing matchup statistics")
    
    print("\n✓ Progressive training complete!")
    print("Check results/example_progressive_*.png for visualizations")


def evaluate():
    """Evaluate trained agent against various opponents."""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # Evaluate
    if not run_command([
        'python3', 'main.py', 'play',
        '--agents', 'ppo_agent', 'ppo_agent',
        'rule_based_agent', 'aggressive_teacher_agent',
        '--no-gui',
        '--n-rounds', '50',
        '--save-stats', 'results/example_eval.json'
    ], "Evaluating trained PPO agent (50 rounds)"):
        return
    
    # Plot
    run_command([
        'python3', 'plot_results.py',
        'results/example_eval.json'
    ], "Generating evaluation charts")
    
    print("\n✓ Evaluation complete!")
    print("Check results/example_eval_*.png for visualizations")


def test_teacher():
    """Test the aggressive teacher agent."""
    print("\n" + "="*60)
    print("TEACHER AGENT TEST")
    print("="*60)
    
    # Test
    if not run_command([
        'python3', 'main.py', 'play',
        '--agents', 'aggressive_teacher_agent', 'aggressive_teacher_agent',
        'rule_based_agent', 'rule_based_agent',
        '--no-gui',
        '--n-rounds', '30',
        '--save-stats', 'results/example_teacher.json'
    ], "Testing aggressive teacher agent (30 rounds)"):
        return
    
    # Plot
    run_command([
        'python3', 'plot_results.py',
        'results/example_teacher.json'
    ], "Generating test charts")
    
    print("\n✓ Teacher test complete!")
    print("Check results/example_teacher_*.png for visualizations")


def main():
    parser = argparse.ArgumentParser(
        description="Example training script for Bomberman RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  basic       - PPO vs rule_based (200 rounds) - Good for beginners
  advanced    - PPO vs teacher agents (300 rounds) - More challenging
  progressive - Adaptive difficulty (500 rounds) - Best results
  evaluate    - Test trained agent (50 rounds)
  teacher     - Test teacher agent performance

Examples:
  python example_train.py --mode basic
  python example_train.py --mode progressive
  python example_train.py --mode evaluate
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['basic', 'advanced', 'progressive', 'evaluate', 'teacher'],
        default='basic',
        help='Training mode to run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Bomberman RL - Example Training Script")
    print("="*60)
    
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    if args.mode == 'basic':
        train_basic()
    elif args.mode == 'advanced':
        train_advanced()
    elif args.mode == 'progressive':
        train_progressive()
    elif args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'teacher':
        test_teacher()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nModel saved at: agent_code/ppo_agent/ppo_model.pt")
    print("\nNext steps:")
    print("  1. Review charts in results/ directory")
    print("  2. Test with GUI: python3 main.py play --agents ppo_agent aggressive_teacher_agent")
    print("  3. Continue training or try different mode\n")


if __name__ == '__main__':
    main()

