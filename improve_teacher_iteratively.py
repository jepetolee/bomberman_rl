#!/usr/bin/env python3
"""
Iterative Teacher Model Improvement Script
==========================================
교사 모델을 반복적으로 평가하고 개선하는 자동화 스크립트입니다.
각 반복에서 성능을 측정하고, 이전 최고 성능과 비교하여 개선 여부를 확인합니다.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# 평가 스크립트를 임포트하여 사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_evaluation(rounds: int = 30) -> Optional[Dict]:
    """교사 모델 평가를 실행하고 결과를 반환합니다."""
    from evaluate_teacher_model import (
        evaluate_teacher_against_all,
        calculate_overall_performance,
        ALL_AGENTS,
        TEACHER_AGENT
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/teacher_evaluation/auto_{timestamp}'
    
    print(f"\n{'='*70}")
    print(f"Running Evaluation ({rounds} rounds per match)")
    print(f"{'='*70}\n")
    
    results = evaluate_teacher_against_all(
        teacher_agent=TEACHER_AGENT,
        opponents=ALL_AGENTS,
        rounds=rounds,
        results_dir=results_dir
    )
    
    performance = calculate_overall_performance(results)
    
    return {
        'performance': performance,
        'results': results,
        'results_dir': results_dir,
    }


def print_comparison(current: Dict, previous: Optional[Dict] = None):
    """현재 성능과 이전 성능을 비교 출력합니다."""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    p = current['performance']
    print(f"Current Performance:")
    print(f"  Win Rate: {p['win_rate']*100:.1f}% ({p['total_wins']}/{p['total_matches']})")
    print(f"  Avg Score Diff: {p['avg_score_diff']:+.1f}")
    print(f"  K/D Ratio: {p['kill_death_ratio']:.2f}")
    print(f"  Avg Kills: {p['avg_kills']:.1f}")
    print(f"  Avg Deaths: {p['avg_deaths']:.1f}")
    
    if previous:
        prev_p = previous['performance']
        print(f"\nPrevious Best:")
        print(f"  Win Rate: {prev_p['win_rate']*100:.1f}% ({prev_p['total_wins']}/{prev_p['total_matches']})")
        print(f"  Avg Score Diff: {prev_p['avg_score_diff']:+.1f}")
        print(f"  K/D Ratio: {prev_p['kill_death_ratio']:.2f}")
        
        win_rate_improvement = p['win_rate'] - prev_p['win_rate']
        score_improvement = p['avg_score_diff'] - prev_p['avg_score_diff']
        kdr_improvement = p['kill_death_ratio'] - prev_p['kill_death_ratio']
        
        print(f"\nImprovement:")
        print(f"  Win Rate: {win_rate_improvement*100:+.1f}%")
        print(f"  Score Diff: {score_improvement:+.1f}")
        print(f"  K/D Ratio: {kdr_improvement:+.2f}")
        
        if p['win_rate'] > prev_p['win_rate']:
            print(f"\n✓ Improved! Win rate increased by {win_rate_improvement*100:.1f}%")
        elif p['win_rate'] < prev_p['win_rate']:
            print(f"\n✗ Regressed. Win rate decreased by {abs(win_rate_improvement*100):.1f}%")
        else:
            print(f"\n= No change in win rate")


def main():
    print(f"\n{'='*70}")
    print("ITERATIVE TEACHER MODEL IMPROVEMENT")
    print(f"{'='*70}")
    print(f"This script will evaluate the teacher model and help identify")
    print(f"areas for improvement. After each evaluation, you can modify")
    print(f"the agent code and run this script again to see improvements.\n")
    
    best_performance = None
    iteration = 1
    
    while True:
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}\n")
        
        # 평가 실행
        current_result = run_evaluation(rounds=30)
        
        if not current_result:
            print("✗ Evaluation failed!")
            break
        
        # 비교 출력
        print_comparison(current_result, best_performance)
        
        # 최고 성능 업데이트
        if best_performance is None:
            best_performance = current_result
            print(f"\n✓ This is the first evaluation - saved as baseline")
        else:
            if current_result['performance']['win_rate'] > best_performance['performance']['win_rate']:
                best_performance = current_result
                print(f"\n✓ New best performance! Updated baseline.")
            else:
                print(f"\n⚠ Performance did not improve. Consider reverting changes or trying different approach.")
        
        # 결과 저장
        summary_file = f"results/teacher_evaluation/best_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump({
                'iteration': iteration,
                'current': current_result,
                'best': best_performance,
            }, f, indent=2)
        
        # 사용자에게 다음 단계 안내
        print(f"\n{'='*70}")
        response = input("\nContinue to next iteration? (y/n): ").strip().lower()
        if response != 'y':
            break
        
        print("\n⚠ Please modify the agent code before continuing.")
        input("Press Enter when ready to evaluate again...")
        
        iteration += 1
    
    # 최종 요약
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    if best_performance:
        p = best_performance['performance']
        print(f"Best Performance Achieved:")
        print(f"  Win Rate: {p['win_rate']*100:.1f}% ({p['total_wins']}/{p['total_matches']})")
        print(f"  Avg Score Diff: {p['avg_score_diff']:+.1f}")
        print(f"  K/D Ratio: {p['kill_death_ratio']:.2f}")
        print(f"\nResults saved to: {best_performance['results_dir']}")


if __name__ == '__main__':
    main()
