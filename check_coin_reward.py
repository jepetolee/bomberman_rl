#!/usr/bin/env python3
"""
코인 보상 정책 확인 스크립트
==========================

COIN_FOUND 이벤트와 BFS 맵이 보상에 어떻게 적용되는지 확인합니다.

1. COIN_FOUND 이벤트 발생 조건 확인
2. 적의 코인 BFS 맵 생성 확인
3. 보상 적용 방식 확인
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
from agent_code.ppo_agent.callbacks import state_to_features, _team_tag


def analyze_coin_found_event():
    """COIN_FOUND 이벤트 발생 조건 분석"""
    print("="*70)
    print("COIN_FOUND 이벤트 분석")
    print("="*70)
    
    print("\n1. COIN_FOUND 이벤트:")
    print("   - 게임 엔진에서 코인을 발견했을 때 발생")
    print("   - 보상: 0.1 (train.py의 _reward_from_events)")
    print("   - 이벤트는 게임 엔진에서 자동으로 생성됨")
    
    print("\n2. 현재 보상 정책:")
    print("   - COIN_FOUND: 0.1 (낮은 보상)")
    print("   - COIN_COLLECTED: 1.0 (수집 시 높은 보상)")
    print("   - KILLED_OPPONENT: 50.0 (킬이 훨씬 높은 보상)")
    
    print("\n3. 특징:")
    print("   - COIN_FOUND는 코인을 '발견'했을 때 (아직 수집 전)")
    print("   - COIN_COLLECTED는 코인을 '수집'했을 때")
    print("   - 발견만으로는 낮은 보상, 수집해야 높은 보상")


def analyze_coin_features(game_state: dict) -> Dict:
    """게임 상태에서 코인 관련 특징 분석"""
    if game_state is None:
        return {}
    
    field = game_state['field']
    coins = game_state['coins']
    self_info = game_state['self']
    others = game_state['others']
    
    self_name, _, _, (sx, sy) = self_info
    self_tag = _team_tag(self_name)
    
    # 코인 분류
    all_coins = set(coins)
    enemy_coins = []
    teammate_coins = []
    neutral_coins = []
    
    # 적과 아군 위치
    enemy_positions = []
    teammate_positions = []
    
    for other_name, _, _, (ox, oy) in others:
        tag = _team_tag(other_name)
        if tag == self_tag:
            teammate_positions.append((ox, oy))
        else:
            enemy_positions.append((ox, oy))
    
    # 코인을 적/아군 근처로 분류 (간단한 휴리스틱)
    for cx, cy in coins:
        # 가장 가까운 적/아군 찾기
        min_enemy_dist = min([abs(cx - ex) + abs(cy - ey) for ex, ey in enemy_positions], default=float('inf'))
        min_teammate_dist = min([abs(cx - tx) + abs(cy - ty) for tx, ty in teammate_positions], default=float('inf'))
        
        if min_enemy_dist < min_teammate_dist and min_enemy_dist < 5:
            enemy_coins.append((cx, cy))
        elif min_teammate_dist < 5:
            teammate_coins.append((cx, cy))
        else:
            neutral_coins.append((cx, cy))
    
    return {
        'total_coins': len(coins),
        'enemy_coins': enemy_coins,
        'teammate_coins': teammate_coins,
        'neutral_coins': neutral_coins,
        'enemy_positions': enemy_positions,
        'teammate_positions': teammate_positions,
        'self_position': (sx, sy),
    }


def create_bfs_map(field: np.ndarray, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> np.ndarray:
    """
    BFS를 사용하여 시작점에서 목표까지의 거리 맵 생성
    
    Returns:
        distance_map: [H, W] 각 위치까지의 최단 거리 (도달 불가능하면 -1)
    """
    from collections import deque
    
    H, W = field.shape
    distance_map = np.full((H, W), -1, dtype=np.float32)
    
    if len(targets) == 0:
        return distance_map
    
    # BFS
    queue = deque([start])
    distance_map[start] = 0.0
    visited = {start}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        current_dist = distance_map[x, y]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            
            if (nx, ny) in visited:
                continue
            
            # 벽이나 크레이트는 통과 불가
            if field[nx, ny] != 0:  # 0 = free space
                continue
            
            distance_map[nx, ny] = current_dist + 1.0
            visited.add((nx, ny))
            queue.append((nx, ny))
    
    return distance_map


def analyze_coin_bfs_reward(game_state: dict) -> Dict:
    """코인 BFS 맵과 보상의 관계 분석"""
    if game_state is None:
        return {}
    
    field = game_state['field']
    coins = game_state['coins']
    self_info = game_state['self']
    others = game_state['others']
    
    self_name, _, _, (sx, sy) = self_info
    self_tag = _team_tag(self_name)
    
    # 적 위치 찾기
    enemy_positions = []
    for other_name, _, _, (ox, oy) in others:
        tag = _team_tag(other_name)
        if tag != self_tag:
            enemy_positions.append((ox, oy))
    
    # 적의 코인 찾기 (적 근처의 코인)
    enemy_coins = []
    for cx, cy in coins:
        min_enemy_dist = min([abs(cx - ex) + abs(cy - ey) for ex, ey in enemy_positions], default=float('inf'))
        if min_enemy_dist < 5:  # 적으로부터 5칸 이내
            enemy_coins.append((cx, cy))
    
    # BFS 맵 생성 (자신의 위치에서 적의 코인까지)
    bfs_map = create_bfs_map(field, (sx, sy), enemy_coins)
    
    # 특징 분석
    coin_distances = []
    for cx, cy in enemy_coins:
        dist = bfs_map[cx, cy]
        if dist >= 0:
            coin_distances.append(dist)
    
    return {
        'enemy_coins': enemy_coins,
        'bfs_map': bfs_map,
        'coin_distances': coin_distances,
        'min_distance': min(coin_distances) if coin_distances else -1,
        'avg_distance': np.mean(coin_distances) if coin_distances else -1,
    }


def check_reward_application():
    """보상 적용 방식 확인"""
    print("\n" + "="*70)
    print("보상 적용 방식 확인")
    print("="*70)
    
    print("\n1. 현재 보상 구조:")
    print("   - COIN_FOUND: 0.1 (코인 발견 시)")
    print("   - COIN_COLLECTED: 1.0 (코인 수집 시)")
    print("   - 거리 기반 shaping reward: 없음")
    
    print("\n2. BFS 맵 활용 가능성:")
    print("   - 현재: BFS 맵이 보상에 직접 적용되지 않음")
    print("   - 제안: 적의 코인까지의 거리가 가까워질수록 보상 증가")
    print("   - 예: reward += 0.1 * (1.0 / (distance + 1))")
    
    print("\n3. 특징 맵에서의 코인 표현:")
    print("   - grid[3, x, y] = 1.0 (모든 코인, 적/아군 구분 없음)")
    print("   - 적의 코인과 아군의 코인을 구분하지 않음")
    print("   - BFS 맵은 특징 맵에 포함되지 않음")
    
    print("\n4. 개선 제안:")
    print("   - 적의 코인까지의 BFS 거리를 특징 맵에 추가 (grid[10])")
    print("   - 거리 기반 shaping reward 추가")
    print("   - 적의 코인에 더 높은 보상 부여")


def main():
    parser = argparse.ArgumentParser(description='코인 보상 정책 확인')
    parser.add_argument('--check-events', action='store_true', help='이벤트 분석')
    parser.add_argument('--check-bfs', action='store_true', help='BFS 맵 분석')
    parser.add_argument('--check-reward', action='store_true', help='보상 적용 확인')
    
    args = parser.parse_args()
    
    if args.check_events or not any([args.check_bfs, args.check_reward]):
        analyze_coin_found_event()
    
    if args.check_bfs:
        print("\n" + "="*70)
        print("BFS 맵 분석")
        print("="*70)
        print("\nBFS 맵은 현재 특징 맵에 포함되지 않습니다.")
        print("코인 위치는 grid[3]에만 표시되며, 거리 정보는 없습니다.")
    
    if args.check_reward or not any([args.check_events, args.check_bfs]):
        check_reward_application()
    
    print("\n" + "="*70)
    print("요약")
    print("="*70)
    print("\n1. COIN_FOUND: 게임 엔진에서 자동 생성, 보상 0.1")
    print("2. BFS 맵: 현재 특징 맵에 포함되지 않음")
    print("3. 보상: 거리 기반 shaping reward 없음")
    print("4. 개선: BFS 거리를 특징 맵에 추가하고 shaping reward 적용 권장")


if __name__ == '__main__':
    main()

