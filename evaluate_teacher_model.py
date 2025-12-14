#!/usr/bin/env python3
"""
Teacher Model Evaluation Script
================================
교사 모델(aggressive_teacher_agent)의 성능을 모든 다른 에이전트와 비교 평가하고,
반복적으로 개선하여 최고 성능을 달성합니다.

Usage:
    python evaluate_teacher_model.py [--rounds N] [--iterations M]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np


# ============== Configuration ==============

ALL_AGENTS = [
    'random_agent',
    'peaceful_agent',
    'coin_collector_agent',
    'rule_based_agent',
    'team_teacher_agent',
    'aggressive_teacher_agent',  # 평가 대상
]

TEACHER_AGENT = 'aggressive_teacher_agent'

DEFAULT_ROUNDS = 50  # 각 매치당 라운드 수
DEFAULT_ITERATIONS = 10  # 최대 반복 개선 횟수


# ============== Evaluation Functions ==============

def run_match(
    team1_agents: List[str],
    team2_agents: List[str],
    rounds: int,
    output_file: str
) -> Optional[Dict]:
    """두 팀 간의 매치를 실행하고 통계를 반환합니다."""
    agents = team1_agents + team2_agents
    
    cmd = [
        sys.executable, 'main.py', 'play',
        '--agents', *agents,
        '--no-gui',
        '--n-rounds', str(rounds),
        '--save-stats', output_file,
        '--silence-errors',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10분 타임아웃
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            print(f"    ✗ Error (code {result.returncode})")
            if result.stderr:
                print(f"      stderr: {result.stderr[:200]}")
            return None
        
        if not os.path.exists(output_file):
            print(f"    ✗ Output file not found: {output_file}")
            return None
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # 팀별 통계 계산
        team1_stats = {
            'score': 0,
            'kills': 0,
            'deaths': 0,
            'suicides': 0,
        }
        team2_stats = {
            'score': 0,
            'kills': 0,
            'deaths': 0,
            'suicides': 0,
        }
        
        by_agent = data.get('by_agent', {})
        for agent_name, stats in by_agent.items():
            score = stats.get('score', 0)
            kills = stats.get('kills', 0)
            deaths = stats.get('suicides', 0)
            
            # 팀 분류
            is_team1 = any(agent_name.startswith(agent.split('_')[0]) for agent in team1_agents)
            
            if is_team1:
                team1_stats['score'] += score
                team1_stats['kills'] += kills
                team1_stats['deaths'] += deaths
                team1_stats['suicides'] += deaths
            else:
                team2_stats['score'] += score
                team2_stats['kills'] += kills
                team2_stats['deaths'] += deaths
                team2_stats['suicides'] += deaths
        
        return {
            'team1': team1_stats,
            'team2': team2_stats,
            'team1_win': team1_stats['score'] > team2_stats['score'],
            'score_diff': team1_stats['score'] - team2_stats['score'],
        }
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def evaluate_teacher_against_all(
    teacher_agent: str,
    opponents: List[str],
    rounds: int,
    results_dir: str
) -> Dict[str, Dict]:
    """교사 모델을 모든 상대와 비교 평가합니다."""
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Evaluating Teacher Agent: {teacher_agent}")
    print(f"{'='*70}\n")
    
    # 2 vs 2 매치
    for opponent in opponents:
        if opponent == teacher_agent:
            continue
        
        print(f"  vs {opponent}:")
        print(f"    Match: {teacher_agent} x2 vs {opponent} x2")
        
        output_file = os.path.join(results_dir, f'match_{teacher_agent}_vs_{opponent}.json')
        
        match_result = run_match(
            team1_agents=[teacher_agent, teacher_agent],
            team2_agents=[opponent, opponent],
            rounds=rounds,
            output_file=output_file
        )
        
        if match_result:
            win_rate = 1.0 if match_result['team1_win'] else 0.0
            results[opponent] = {
                'win_rate': win_rate,
                'score_diff': match_result['score_diff'],
                'teacher_kills': match_result['team1']['kills'],
                'teacher_deaths': match_result['team1']['deaths'],
                'opponent_kills': match_result['team2']['kills'],
                'opponent_deaths': match_result['team2']['deaths'],
                'details': match_result,
            }
            
            win_str = "✓ WIN" if match_result['team1_win'] else "✗ LOSS"
            print(f"    {win_str}: Score {match_result['team1']['score']:.1f} vs {match_result['team2']['score']:.1f}, "
                  f"Kills {match_result['team1']['kills']} vs {match_result['team2']['kills']}, "
                  f"Deaths {match_result['team1']['deaths']} vs {match_result['team2']['deaths']}")
        else:
            print(f"    ✗ Failed to get results")
            results[opponent] = None
    
    return results


def calculate_overall_performance(results: Dict[str, Dict]) -> Dict:
    """전체 성능 지표를 계산합니다."""
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return {
            'win_rate': 0.0,
            'avg_score_diff': 0.0,
            'total_wins': 0,
            'total_matches': 0,
            'avg_kills': 0.0,
            'avg_deaths': 0.0,
            'kill_death_ratio': 0.0,
        }
    
    total_wins = sum(1 for r in valid_results.values() if r['win_rate'] > 0.5)
    avg_score_diff = np.mean([r['score_diff'] for r in valid_results.values()])
    avg_kills = np.mean([r['teacher_kills'] for r in valid_results.values()])
    avg_deaths = np.mean([r['teacher_deaths'] for r in valid_results.values()])
    kdr = avg_kills / avg_deaths if avg_deaths > 0 else avg_kills
    
    return {
        'win_rate': total_wins / len(valid_results),
        'avg_score_diff': avg_score_diff,
        'total_wins': total_wins,
        'total_matches': len(valid_results),
        'avg_kills': avg_kills,
        'avg_deaths': avg_deaths,
        'kill_death_ratio': kdr,
    }


def print_performance_summary(performance: Dict, results: Dict[str, Dict]):
    """성능 요약을 출력합니다."""
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Overall Win Rate: {performance['win_rate']*100:.1f}% ({performance['total_wins']}/{performance['total_matches']})")
    print(f"Average Score Difference: {performance['avg_score_diff']:+.1f}")
    print(f"Average Kills: {performance['avg_kills']:.1f}")
    print(f"Average Deaths: {performance['avg_deaths']:.1f}")
    print(f"Kill/Death Ratio: {performance['kill_death_ratio']:.2f}")
    
    print(f"\nDetailed Results:")
    print(f"{'Opponent':<30} {'Win':<6} {'Score Diff':<12} {'K/D':<12} {'Deaths':<8}")
    print(f"{'-'*70}")
    
    for opponent, result in sorted(results.items()):
        if result is None:
            continue
        win_str = "✓" if result['win_rate'] > 0.5 else "✗"
        kdr = result['teacher_kills'] / result['teacher_deaths'] if result['teacher_deaths'] > 0 else result['teacher_kills']
        print(f"{opponent:<30} {win_str:<6} {result['score_diff']:>+10.1f}  "
              f"{kdr:>10.2f}  {result['teacher_deaths']:>6}")


def identify_issues(results: Dict[str, Dict], performance: Dict) -> List[str]:
    """성능 문제점을 파악합니다."""
    issues = []
    
    # 높은 자살률 체크
    if performance['avg_deaths'] > performance['avg_kills'] * 1.5:
        issues.append(f"높은 자살률: Deaths({performance['avg_deaths']:.1f}) > Kills({performance['avg_kills']:.1f}) * 1.5")
    
    # 낮은 승률 체크
    if performance['win_rate'] < 0.5:
        issues.append(f"낮은 승률: {performance['win_rate']*100:.1f}% < 50%")
    
    # 특정 상대에게 약한 경우
    for opponent, result in results.items():
        if result is None:
            continue
        if result['win_rate'] < 0.3 and result['score_diff'] < -50:
            issues.append(f"{opponent}에게 약함: 승률 {result['win_rate']*100:.1f}%, 점수 차이 {result['score_diff']:.1f}")
    
    # 높은 사망률
    if performance['kill_death_ratio'] < 0.5:
        issues.append(f"낮은 K/D 비율: {performance['kill_death_ratio']:.2f} < 0.5")
    
    return issues


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and iteratively improve teacher model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=DEFAULT_ROUNDS,
        help=f'Number of rounds per matchup (default: {DEFAULT_ROUNDS})'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f'Maximum number of improvement iterations (default: {DEFAULT_ITERATIONS})'
    )
    
    parser.add_argument(
        '--teacher-agent',
        type=str,
        default=TEACHER_AGENT,
        help=f'Teacher agent to evaluate (default: {TEACHER_AGENT})'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/teacher_evaluation',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_results_dir = os.path.join(args.results_dir, f'iter_{timestamp}')
    
    best_performance = None
    best_iteration = 0
    
    print(f"\n{'='*70}")
    print(f"Teacher Model Evaluation & Improvement")
    print(f"{'='*70}")
    print(f"Teacher Agent: {args.teacher_agent}")
    print(f"Rounds per match: {args.rounds}")
    print(f"Max iterations: {args.iterations}")
    print(f"Results directory: {base_results_dir}")
    print(f"{'='*70}\n")
    
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{args.iterations}")
        print(f"{'='*70}\n")
        
        iteration_dir = os.path.join(base_results_dir, f'iteration_{iteration}')
        
        # 평가 실행
        results = evaluate_teacher_against_all(
            teacher_agent=args.teacher_agent,
            opponents=ALL_AGENTS,
            rounds=args.rounds,
            results_dir=iteration_dir
        )
        
        # 성능 계산
        performance = calculate_overall_performance(results)
        
        # 결과 출력
        print_performance_summary(performance, results)
        
        # 문제점 파악
        issues = identify_issues(results, performance)
        if issues:
            print(f"\nIssues identified:")
            for issue in issues:
                print(f"  - {issue}")
        
        # 최고 성능 업데이트
        if best_performance is None or performance['win_rate'] > best_performance['win_rate']:
            best_performance = performance
            best_iteration = iteration
            print(f"\n✓ New best performance! Win rate: {performance['win_rate']*100:.1f}%")
        
        # 결과 저장
        summary_file = os.path.join(iteration_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'iteration': iteration,
                'performance': performance,
                'results': results,
                'issues': issues,
            }, f, indent=2)
        
        # 최고 성능 달성 시 중단 고려
        if performance['win_rate'] >= 1.0 and performance['kill_death_ratio'] > 1.0:
            print(f"\n✓ Perfect performance achieved! Stopping iterations.")
            break
        
        # 수정 필요 시 사용자에게 알림
        if iteration < args.iterations:
            print(f"\n⚠ Next iteration will test improvements.")
            print(f"  Please modify {args.teacher_agent} based on the identified issues.")
            input("  Press Enter to continue to next iteration, or Ctrl+C to stop...")
    
    # 최종 요약
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    print(f"Best performance at iteration {best_iteration}:")
    if best_performance:
        print(f"  Win Rate: {best_performance['win_rate']*100:.1f}%")
        print(f"  Avg Score Diff: {best_performance['avg_score_diff']:+.1f}")
        print(f"  K/D Ratio: {best_performance['kill_death_ratio']:.2f}")
    print(f"\nAll results saved to: {base_results_dir}")


if __name__ == '__main__':
    main()
