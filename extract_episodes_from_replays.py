#!/usr/bin/env python3
"""
Extract Episode Data from Replay Files
=======================================

Replay 파일들을 로드하여 teacher agent의 episode 데이터를 추출합니다.
실제 강화학습 환경의 보상점수(GAE 포함)를 계산하여 저장합니다.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from collections import deque
import numpy as np
import torch

# Disable audio
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
import events as e
from agent_code.ppo_agent.callbacks import state_to_features, ACTIONS
from replay import ReplayWorld
from agents import Agent, SequentialAgentBackend
from environment import BombeRLeWorld

# Import reward calculation functions from PPO training
try:
    from agent_code.ppo_agent.train import (
        _reward_from_events,
        _get_distance_to_enemy_coins,
        get_distance_to_nearest_enemy,
        _compute_gae
    )
except ImportError:
    # Fallback if import fails
    print("Warning: Could not import reward functions from PPO training")
    _reward_from_events = None
    _get_distance_to_enemy_coins = None
    get_distance_to_nearest_enemy = None
    _compute_gae = None


def _estimate_events_from_state_change(
    old_game_state: dict,
    action: str,
    new_game_state: dict,
    agent
) -> List[str]:
    """
    Estimate game events from state changes (for reward calculation)
    
    This is a simplified event estimation - in a real replay we might have
    actual events stored, but here we estimate from state differences.
    """
    events = []
    
    if old_game_state is None or new_game_state is None:
        return events
    
    # Get scores to detect coin collection
    old_self = old_game_state.get('self', (None, 0, 0, (0, 0)))
    new_self = new_game_state.get('self', (None, 0, 0, (0, 0)))
    old_score = old_self[1] if len(old_self) > 1 else 0
    new_score = new_self[1] if len(new_self) > 1 else 0
    
    # Coin collected
    if new_score > old_score:
        score_diff = new_score - old_score
        # Assume score difference is from coins (simplified)
        for _ in range(int(score_diff)):
            events.append(e.COIN_COLLECTED)
    
    # Movement events
    if action == 'UP':
        events.append(e.MOVED_UP)
    elif action == 'DOWN':
        events.append(e.MOVED_DOWN)
    elif action == 'LEFT':
        events.append(e.MOVED_LEFT)
    elif action == 'RIGHT':
        events.append(e.MOVED_RIGHT)
    elif action == 'WAIT':
        events.append(e.WAITED)
    elif action == 'BOMB':
        events.append(e.BOMB_DROPPED)
    
    # Check for death
    old_self_pos = old_self[3] if len(old_self) > 3 else (0, 0)
    new_self_pos = new_self[3] if len(new_self) > 3 else (0, 0)
    
    # Check if agent is still alive in new state
    new_others = new_game_state.get('others', [])
    old_others = old_game_state.get('others', [])
    
    # If agent disappeared from active agents, they died
    old_names = [o[0] for o in old_others if len(o) > 0]
    new_names = [o[0] for o in new_others if len(o) > 0]
    old_self_name = old_self[0] if len(old_self) > 0 else None
    
    if old_self_name and old_self_name in old_names and old_self_name not in new_names:
        # Agent died - check if suicide or got killed
        # Simplified: assume got killed (could be improved)
        events.append(e.GOT_KILLED)
    
    # Check for opponent kills (simplified - check if opponents disappeared)
    # This is a rough estimate
    old_enemy_names = [o[0] for o in old_others if len(o) > 0 and o[0] != old_self_name]
    new_enemy_names = [o[0] for o in new_others if len(o) > 0 and o[0] != old_self_name]
    
    for enemy_name in old_enemy_names:
        if enemy_name not in new_enemy_names:
            # Enemy disappeared - might be a kill
            events.append(e.KILLED_OPPONENT)
    
    return events


def extract_episode_from_replay(replay_path: str, teacher_agent_prefix: str = "aggressive_teacher_agent") -> Optional[Dict]:
    """
    Replay 파일에서 teacher agent의 episode 데이터 추출 (deprecated - use extract_from_replay_world)
    """
    return None


def extract_from_replay_world(replay_path: str, teacher_agent_prefix: str = "aggressive_teacher_agent") -> Optional[Dict]:
    """
    ReplayWorld를 사용해서 state 추출하고 실제 보상점수(GAE 포함) 계산
    """
    if _reward_from_events is None:
        print("Error: Reward functions not available. Cannot compute actual rewards.")
        return None
    
    try:
        # Load replay to get basic info
        with open(replay_path, 'rb') as f:
            replay_data = pickle.load(f)
        
        # Find teacher agent names
        teacher_names = [name for name in replay_data['actions'].keys() if name.startswith(teacher_agent_prefix)]
        if not teacher_names:
            return None
        
        teacher_name = teacher_names[0]
        teacher_actions = replay_data['actions'][teacher_name]
        n_steps = replay_data.get('n_steps', len(teacher_actions))
        
        # Create ReplayWorld and simulate game
        args = SimpleNamespace(
            no_gui=True,
            fps=None,
            turn_based=False,
            update_interval=0.1,
            save_replay=False,
            replay=replay_path,
            make_video=False,
            continue_without_training=True,
            log_dir=os.path.dirname(os.path.abspath(__file__)) + "/logs",
            save_stats=False,
            match_name=None,
            seed=None,
            silence_errors=True,
            scenario="classic"
        )
        
        replay_world = ReplayWorld(args)
        replay_world.new_round()
        
        states = []
        actions = []
        game_states = []
        rewards_raw = []  # Raw rewards before GAE
        dones = []
        values = []  # Placeholder for value estimates (for GAE)
        
        step = 0
        max_iterations = n_steps + 10
        prev_game_state = None
        prev_action = None
        
        while replay_world.running and step < n_steps and step < max_iterations:
            # Find teacher agent
            teacher_agent = None
            for agent in replay_world.active_agents:
                if agent.name == teacher_name:
                    teacher_agent = agent
                    break
            
            if teacher_agent is None or teacher_agent.dead:
                break
            
            # Get game state BEFORE action is executed (at current step)
            game_state = replay_world.get_state_for_agent(teacher_agent)
            if game_state is None:
                break
            
            # Convert to feature representation
            state = state_to_features(game_state)
            
            # Get action for this step
            if step < len(teacher_actions):
                action_str = teacher_actions[step]
                action_idx = ACTIONS.index(action_str) if action_str in ACTIONS else 0
            else:
                break
            
            # Calculate reward based on game state changes (actual PPO reward)
            reward = 0.0
            
            if prev_game_state is not None:
                # Estimate events from state changes
                events = _estimate_events_from_state_change(
                    prev_game_state, prev_action, game_state, teacher_agent
                )
                
                # Calculate base reward from events (same as PPO _reward_from_events)
                # Create mock self object with required attributes
                mock_self = SimpleNamespace()
                # _reward_from_events uses self only for suicide penalty scheduling
                # For teacher data collection, we use default penalty
                reward = _reward_from_events(mock_self, events)
                
                # Add shaping rewards (same as PPO game_events_occurred)
                dist_old = get_distance_to_nearest_enemy(prev_game_state)
                dist_new = get_distance_to_nearest_enemy(game_state)
                
                if dist_old != float('inf') and dist_new != float('inf'):
                    if dist_new < dist_old:
                        reward += 0.5  # Approaching reward
                    elif dist_new > dist_old:
                        reward += -0.2  # Retreating penalty
                
                # Coin distance shaping
                coin_dist_old = _get_distance_to_enemy_coins(prev_game_state)
                coin_dist_new = _get_distance_to_enemy_coins(game_state)
                
                if coin_dist_old != float('inf') and coin_dist_new != float('inf'):
                    if coin_dist_new < coin_dist_old:
                        coin_shaping_reward = 0.1 * (1.0 / (coin_dist_new + 1.0))
                        reward += coin_shaping_reward
                    elif coin_dist_new > coin_dist_old:
                        reward += -0.05
            
            # Store data
            states.append(state.copy() if isinstance(state, np.ndarray) else state)
            actions.append(action_idx)
            game_state_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) 
                             for k, v in game_state.items()}
            game_states.append(game_state_copy)
            rewards_raw.append(reward)
            dones.append(False)
            values.append(0.0)  # Placeholder - actual values would require model
            
            # Update for next iteration
            prev_game_state = game_state_copy
            prev_action = action_str
            
            # Advance game step
            replay_world.do_step()
            step += 1
            
            if replay_world.time_to_stop() or not replay_world.running:
                break
        
        # Final done - add final reward (win/loss)
        if len(dones) > 0:
            dones[-1] = True
            
            # Add win/loss reward for final step (same as PPO end_of_round)
            if len(game_states) > 0:
                final_game_state = game_states[-1]
                try:
                    self_name, self_score, _, _ = final_game_state.get('self', (None, 0, 0, (0, 0)))
                    others_data = final_game_state.get('others', [])
                    
                    self_tag = str(self_name).split('_')[0] if self_name else ""
                    our_team_score = self_score
                    enemy_team_score = 0
                    
                    for other_name, other_score, _, _ in others_data:
                        other_tag = str(other_name).split('_')[0] if other_name else ""
                        if other_tag == self_tag:
                            our_team_score += other_score
                        else:
                            enemy_team_score += other_score
                    
                    # Add win/loss reward (same as PPO)
                    if our_team_score > enemy_team_score:
                        rewards_raw[-1] += 100.0  # Win bonus
                    elif our_team_score < enemy_team_score:
                        rewards_raw[-1] -= 50.0  # Loss penalty
                except Exception:
                    pass
        
        if len(states) < 10:
            return None
        
        # Compute GAE (Generalized Advantage Estimation) for actual reward values
        # Use PPO training defaults
        gamma = 0.99
        gae_lambda = 0.95
        
        # Compute GAE to get actual returns (reward + advantage)
        adv, returns = _compute_gae(
            rewards_raw,
            values,  # Zero-initialized values
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        
        # Returns contain the actual reward values (with GAE)
        actual_rewards = returns.numpy() if isinstance(returns, torch.Tensor) else np.array(returns)
        advantages = adv.numpy() if isinstance(adv, torch.Tensor) else np.array(adv)
        
        return {
            'states': np.array(states),
            'teacher_actions': np.array(actions),
            'rewards': actual_rewards,  # GAE-computed returns (actual reward values)
            'rewards_raw': np.array(rewards_raw),  # Raw rewards before GAE
            'advantages': advantages,  # GAE advantages
            'dones': np.array(dones),
            'game_states': game_states,
            'gae_params': {
                'gamma': gamma,
                'gae_lambda': gae_lambda,
            }
        }
        
    except Exception as e:
        print(f"Error extracting from replay world {replay_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract episode data from replay files with actual rewards')
    parser.add_argument('--replay-dir', type=str, default='replays', help='Directory containing replay files')
    parser.add_argument('--output-dir', type=str, default='data/teacher_episodes', help='Output directory for episodes')
    parser.add_argument('--teacher-prefix', type=str, default='aggressive_teacher_agent', help='Teacher agent name prefix')
    
    args = parser.parse_args()
    
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all replay files
    replay_files = list(replay_dir.glob('*.pt'))
    
    if not replay_files:
        print(f"No replay files found in {replay_dir}")
        return
    
    print(f"Found {len(replay_files)} replay files")
    print(f"Extracting episodes with actual rewards (GAE) to {output_dir}")
    
    episode_count = 0
    for replay_file in replay_files:
        episode_data = extract_from_replay_world(str(replay_file), args.teacher_prefix)
        
        if episode_data is not None:
            episode_file = output_dir / f'episode_{episode_count:06d}.pt'
            with open(episode_file, 'wb') as f:
                pickle.dump(episode_data, f)
            episode_count += 1
            
            if episode_count % 10 == 0:
                print(f"Extracted {episode_count} episodes...")
                if episode_count <= 10:
                    # Show sample reward stats
                    print(f"  Sample rewards - Raw: mean={episode_data['rewards_raw'].mean():.2f}, "
                          f"GAE Returns: mean={episode_data['rewards'].mean():.2f}")
    
    print(f"\nExtracted {episode_count} episodes from {len(replay_files)} replay files")
    print(f"Rewards are computed using actual PPO reward function with GAE")


if __name__ == '__main__':
    main()
