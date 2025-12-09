#!/usr/bin/env python3
"""
Model Evaluation Script
=======================

Evaluate trained PPO model against various opponents and generate performance reports.

Usage:
    python evaluate_model.py [--model-path MODEL_PATH] [--rounds N] [--opponents OPP1,OPP2]
    
Examples:
    # Default evaluation (vs all opponents)
    python evaluate_model.py
    
    # Custom model and rounds
    python evaluate_model.py --model-path agent_code/ppo_agent/ppo_model.pt --rounds 100
    
    # Specific opponents
    python evaluate_model.py --opponents aggressive_teacher_agent,rule_based_agent
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


# ============== Configuration ==============

DEFAULT_OPPONENTS = [
    ('random_agent', 'random_agent'),
    ('peaceful_agent', 'peaceful_agent'),
    ('coin_collector_agent', 'coin_collector_agent'),
    ('rule_based_agent', 'rule_based_agent'),
    ('aggressive_teacher_agent', 'aggressive_teacher_agent'),
    ('team_teacher_agent', 'team_teacher_agent'),
    ('rule_based_agent', 'aggressive_teacher_agent'),  # Mixed
]

DEFAULT_ROUNDS = 50
DEFAULT_MODEL_PATH = 'agent_code/ppo_agent/ppo_model.pt'


# ============== Evaluation Functions ==============

def run_evaluation(
    model_path: str,
    opp1: str,
    opp2: str,
    rounds: int,
    output_file: str
) -> Dict:
    """Run evaluation match and return stats."""
    print(f"  Evaluating vs {opp1} + {opp2} ({rounds} rounds)...", end=' ', flush=True)
    
    cmd = [
        sys.executable, 'main.py', 'play',
        '--agents', 'ppo_agent', 'ppo_agent', opp1, opp2,
        '--no-gui',
        '--n-rounds', str(rounds),
        '--save-stats', output_file,
        '--silence-errors',
    ]
    
    # Set model path for PPO agent
    env = os.environ.copy()
    env['PPO_MODEL_PATH'] = model_path
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            print(f"✗ Error (code {result.returncode})")
            return None
        
        if not os.path.exists(output_file):
            print("✗ No output file")
            return None
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Extract PPO team stats
        ppo_score = 0
        ppo_kills = 0
        ppo_deaths = 0
        opp_score = 0
        opp_kills = 0
        opp_deaths = 0
        
        by_agent = data.get('by_agent', {})
        for agent_name, stats in by_agent.items():
            score = stats.get('score', 0)
            kills = stats.get('kills', 0)
            deaths = stats.get('suicides', 0)
            
            if agent_name.startswith('ppo_agent'):
                ppo_score += score
                ppo_kills += kills
                ppo_deaths += deaths
            else:
                opp_score += score
                opp_kills += kills
                opp_deaths += deaths
        
        # Calculate win rate
        wins = 1 if ppo_score > opp_score else 0
        
        result_stats = {
            'opponent': f"{opp1}+{opp2}",
            'rounds': rounds,
            'ppo_score': ppo_score,
            'ppo_kills': ppo_kills,
            'ppo_deaths': ppo_deaths,
            'opp_score': opp_score,
            'opp_kills': opp_kills,
            'opp_deaths': opp_deaths,
            'win': wins,
            'score_diff': ppo_score - opp_score,
            'kill_diff': ppo_kills - opp_kills,
        }
        
        print(f"✓ Win: {wins}, Score: {ppo_score:.1f} vs {opp_score:.1f}, "
              f"Kills: {ppo_kills} vs {opp_kills}")
        
        return result_stats
        
    except subprocess.TimeoutExpired:
        print("✗ Timeout")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def evaluate_all(
    model_path: str,
    opponents: List[Tuple[str, str]],
    rounds: int,
    results_dir: str
) -> Dict:
    """Evaluate against all opponents."""
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    total_wins = 0
    total_rounds = 0
    
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_path}")
    print(f"{'='*70}\n")
    
    for i, (opp1, opp2) in enumerate(opponents, 1):
        output_file = os.path.join(results_dir, f'eval_{i}_{opp1}_{opp2}.json')
        
        stats = run_evaluation(model_path, opp1, opp2, rounds, output_file)
        
        if stats:
            results[f"{opp1}+{opp2}"] = stats
            total_wins += stats['win']
            total_rounds += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}\n")
    
    if total_rounds > 0:
        overall_win_rate = (total_wins / total_rounds) * 100
        
        print(f"Overall Win Rate: {overall_win_rate:.1f}% ({total_wins}/{total_rounds} matchups)")
        print(f"\nDetailed Results:")
        print(f"{'Opponent':<40} {'Win':<6} {'Score Diff':<12} {'Kill Diff':<12}")
        print(f"{'-'*70}")
        
        for opp_name, stats in results.items():
            win_str = "✓" if stats['win'] else "✗"
            print(f"{opp_name:<40} {win_str:<6} {stats['score_diff']:>+10.1f}  {stats['kill_diff']:>+10}")
        
        # Save summary
        summary = {
            'model_path': model_path,
            'evaluation_date': datetime.now().isoformat(),
            'total_matchups': total_rounds,
            'total_wins': total_wins,
            'overall_win_rate': overall_win_rate,
            'rounds_per_matchup': rounds,
            'results': results
        }
        
        summary_file = os.path.join(results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
        return summary
    
    return None


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO model against various opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default evaluation
    python evaluate_model.py
    
    # Custom model and rounds
    python evaluate_model.py --model-path agent_code/ppo_agent/ppo_model.pt --rounds 100
    
    # Specific opponents
    python evaluate_model.py --opponents aggressive_teacher_agent,rule_based_agent
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to PPO model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=DEFAULT_ROUNDS,
        help=f'Number of rounds per matchup (default: {DEFAULT_ROUNDS})'
    )
    
    parser.add_argument(
        '--opponents',
        type=str,
        help='Comma-separated opponent pair (e.g., "aggressive_teacher_agent,rule_based_agent")'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/evaluations',
        help='Directory to save evaluation results (default: results/evaluations)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Parse opponents
    if args.opponents:
        opp_parts = args.opponents.split(',')
        if len(opp_parts) != 2:
            print("✗ Error: --opponents must be in format 'opp1,opp2'")
            sys.exit(1)
        opponents = [(opp_parts[0].strip(), opp_parts[1].strip())]
    else:
        opponents = DEFAULT_OPPONENTS
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.results_dir, f'eval_{timestamp}')
    
    # Run evaluation
    summary = evaluate_all(args.model_path, opponents, args.rounds, results_dir)
    
    if summary:
        print(f"\n{'='*70}")
        print("Evaluation Complete!")
        print(f"{'='*70}")
        print(f"\nResults saved to: {results_dir}/")
        print(f"Summary: {results_dir}/evaluation_summary.json")
        print("\nNext steps:")
        print("  1. Review evaluation_summary.json for detailed stats")
        print("  2. Compare with previous evaluations")
        print("  3. Use plot_results.py to visualize results")
    else:
        print("\n✗ Evaluation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

