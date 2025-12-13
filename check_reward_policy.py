#!/usr/bin/env python3
"""
보상 정책 확인 스크립트
=====================

학습된 모델이 어떤 보상 정책을 따르는지 분석합니다.

1. 모델의 행동 분포 분석
2. 보상 함수와의 일치도 확인
3. 전략 패턴 분석 (공격적/방어적/수집 중심)

Usage:
    python check_reward_policy.py --model-path ppo_model.pt --rounds 100
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
from config.load_config import load_config, get_model_config
from agent_code.ppo_agent.features import state_to_features


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """모델 로드"""
    config = load_config()
    model_config = get_model_config(config)
    
    model = PolicyValueViT_TRM_Hybrid(
        in_channels=model_config.get('in_channels', 11),
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
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✓ 모델 로드: {model_path}")
    else:
        print(f"✗ 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    model.eval()
    return model


def analyze_action_distribution(
    model: nn.Module,
    num_samples: int = 1000,
    device: torch.device = None
) -> Dict:
    """모델의 행동 분포 분석"""
    if device is None:
        device = next(model.parameters()).device
    
    # 랜덤 상태 생성 (실제 게임 상태와 유사하게)
    action_counts = Counter()
    action_probs_sum = np.zeros(6)
    value_sum = 0.0
    value_count = 0
    
    print(f"행동 분포 분석 중... ({num_samples} 샘플)")
    
    with torch.no_grad():
        for i in range(num_samples):
            # 랜덤 상태 생성 (10 channels, 17x17)
            random_state = torch.randn(1, 10, 17, 17).to(device)
            
            # Forward pass
            if hasattr(model, 'forward_with_z'):
                logits, value, _ = model.forward_with_z(random_state, z_prev=None)
            else:
                logits, value = model(random_state)
            
            # 행동 선택 (확률 분포에서 샘플링)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            action_probs_sum += probs
            
            # 가장 높은 확률의 행동
            action_idx = int(torch.argmax(logits, dim=-1).item())
            action_counts[ACTIONS[action_idx]] += 1
            
            value_sum += float(value.item())
            value_count += 1
    
    avg_probs = action_probs_sum / num_samples
    avg_value = value_sum / value_count if value_count > 0 else 0.0
    
    return {
        'action_counts': dict(action_counts),
        'action_probs': {ACTIONS[i]: float(avg_probs[i]) for i in range(6)},
        'avg_value': avg_value,
    }


def check_reward_alignment(
    model_path: str,
    rounds: int = 50
) -> Dict:
    """실제 게임에서 보상 정책과의 일치도 확인"""
    print(f"\n실제 게임 평가 중... ({rounds} 라운드)")
    
    # 평가 실행
    eval_dir = "results/reward_policy_check"
    os.makedirs(eval_dir, exist_ok=True)
    
    output_file = os.path.join(eval_dir, "policy_check.json")
    
    cmd = [
        sys.executable, 'main.py', 'play',
        '--agents', 'ppo_agent', 'ppo_agent', 'random_agent', 'random_agent',
        '--no-gui',
        '--n-rounds', str(rounds),
        '--save-stats', output_file,
        '--silence-errors',
    ]
    
    env = os.environ.copy()
    env['PPO_MODEL_PATH'] = model_path
    env['BOMBER_USE_TRM'] = '1'
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0 or not os.path.exists(output_file):
            print("✗ 게임 실행 실패")
            return None
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # PPO 에이전트 통계 추출
        ppo_stats = {}
        total_kills = 0
        total_deaths = 0
        total_coins = 0
        total_bombs = 0
        
        by_agent = data.get('by_agent', {})
        for agent_name, stats in by_agent.items():
            if agent_name.startswith('ppo_agent'):
                total_kills += stats.get('kills', 0)
                total_deaths += stats.get('suicides', 0) + stats.get('got_killed', 0)
                total_coins += stats.get('coins_collected', 0)
                total_bombs += stats.get('bombs_dropped', 0)
        
        ppo_stats = {
            'kills': total_kills,
            'deaths': total_deaths,
            'coins': total_coins,
            'bombs': total_bombs,
            'kill_death_ratio': total_kills / max(total_deaths, 1),
        }
        
        return ppo_stats
        
    except Exception as e:
        print(f"✗ 평가 실패: {e}")
        return None


def analyze_strategy(action_dist: Dict, game_stats: Dict = None) -> Dict:
    """전략 패턴 분석"""
    strategy = {
        'aggressive': False,  # 킬 중심
        'defensive': False,   # 생존 중심
        'collector': False,   # 코인 수집 중심
        'bomber': False,      # 폭탄 사용 중심
    }
    
    # 행동 분포 기반 분석
    bomb_prob = action_dist['action_probs'].get('BOMB', 0.0)
    wait_prob = action_dist['action_probs'].get('WAIT', 0.0)
    move_probs = sum(action_dist['action_probs'].get(a, 0.0) for a in ['UP', 'DOWN', 'LEFT', 'RIGHT'])
    
    if bomb_prob > 0.15:  # 15% 이상 폭탄 사용
        strategy['bomber'] = True
        strategy['aggressive'] = True
    
    if wait_prob < 0.1:  # 대기 적음
        strategy['aggressive'] = True
    
    if move_probs > 0.6:  # 이동 빈번
        strategy['collector'] = True
    
    # 게임 통계 기반 분석
    if game_stats:
        kdr = game_stats.get('kill_death_ratio', 0.0)
        if kdr > 1.5:
            strategy['aggressive'] = True
        elif kdr < 0.5:
            strategy['defensive'] = True
        
        if game_stats.get('coins', 0) > 50:  # 많은 코인 수집
            strategy['collector'] = True
    
    return strategy


def print_report(
    action_dist: Dict,
    game_stats: Dict = None,
    strategy: Dict = None
):
    """보고서 출력"""
    print("\n" + "="*70)
    print("보상 정책 분석 보고서")
    print("="*70)
    
    print("\n📊 행동 분포:")
    print("-" * 70)
    for action, prob in sorted(action_dist['action_probs'].items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        print(f"  {action:<6} {prob*100:>5.1f}% {bar}")
    
    print(f"\n  평균 가치 예측: {action_dist['avg_value']:.2f}")
    
    if game_stats:
        print("\n🎮 게임 통계:")
        print("-" * 70)
        print(f"  킬: {game_stats['kills']}")
        print(f"  사망: {game_stats['deaths']}")
        print(f"  킬/사망 비율: {game_stats['kill_death_ratio']:.2f}")
        print(f"  코인 수집: {game_stats['coins']}")
        print(f"  폭탄 사용: {game_stats['bombs']}")
    
    if strategy:
        print("\n🎯 전략 패턴:")
        print("-" * 70)
        if strategy['aggressive']:
            print("  ✓ 공격적 전략 (킬 중심)")
        if strategy['defensive']:
            print("  ✓ 방어적 전략 (생존 중심)")
        if strategy['collector']:
            print("  ✓ 수집 중심 전략")
        if strategy['bomber']:
            print("  ✓ 폭탄 활용 전략")
        
        if not any(strategy.values()):
            print("  ⚠ 명확한 전략 패턴이 감지되지 않음")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='보상 정책 확인')
    parser.add_argument('--model-path', type=str, default='ppo_model.pt', help='모델 경로')
    parser.add_argument('--rounds', type=int, default=50, help='평가 라운드 수')
    parser.add_argument('--samples', type=int, default=1000, help='행동 분포 분석 샘플 수')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 모델 로드
    model = load_model(args.model_path, device)
    
    # 행동 분포 분석
    action_dist = analyze_action_distribution(model, num_samples=args.samples, device=device)
    
    # 실제 게임 평가
    game_stats = check_reward_alignment(args.model_path, rounds=args.rounds)
    
    # 전략 분석
    strategy = analyze_strategy(action_dist, game_stats)
    
    # 보고서 출력
    print_report(action_dist, game_stats, strategy)


if __name__ == '__main__':
    main()

