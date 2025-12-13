#!/usr/bin/env python3
"""
Extract Episode Data from Replay Files
=======================================

Replay 파일들을 로드하여 teacher agent의 episode 데이터를 추출합니다.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import numpy as np
import torch

# Disable audio
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
from agent_code.ppo_agent.callbacks import state_to_features, ACTIONS
from replay import ReplayWorld
from agents import Agent, SequentialAgentBackend
from environment import BombeRLeWorld


def extract_episode_from_replay(replay_path: str, teacher_agent_prefix: str = "aggressive_teacher_agent") -> Optional[Dict]:
    """
    Replay 파일에서 teacher agent의 episode 데이터 추출
    
    Args:
        replay_path: Replay 파일 경로 (.pt)
        teacher_agent_prefix: Teacher agent 이름 prefix
    
    Returns:
        Episode data dict or None if extraction failed
    """
    try:
        # Load replay
        with open(replay_path, 'rb') as f:
            replay = pickle.load(f)
        
        # Find teacher agent names
        teacher_names = [name for name in replay['actions'].keys() if name.startswith(teacher_agent_prefix)]
        if not teacher_names:
            return None
        
        # Use first teacher agent
        teacher_name = teacher_names[0]
        teacher_actions = replay['actions'][teacher_name]
        
        # Reconstruct states by replaying the game
        # We need to recreate the game state step by step
        # This function is deprecated, use extract_from_replay_world instead
        return None
        
        # Create a minimal world to extract states
        world = BombeRLeWorld(args, [])
        
        # Initialize from replay
        world.arena = np.array(replay['arena'])
        world.round = replay['round']
        world.step = 0
        
        # Extract episode data
        states = []
        actions = []
        game_states = []
        
        # Replay each step
        n_steps = replay.get('n_steps', len(teacher_actions))
        
        # For each step, we need to reconstruct the game state
        # This is complex, so we'll use a simpler approach:
        # Load replay and extract state-action pairs
        
        # Simpler approach: Create ReplayWorld and extract states
        # But ReplayWorld doesn't expose states easily...
        
        # Alternative: Direct extraction from replay
        # We'll need to reconstruct states manually
        
        # For now, return None and use a different approach
        return None
        
    except Exception as e:
        print(f"Error extracting from {replay_path}: {e}")
        return None


def extract_from_replay_world(replay_path: str, teacher_agent_prefix: str = "aggressive_teacher_agent") -> Optional[Dict]:
    """
    ReplayWorld를 사용해서 state 추출
    """
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
        # Use SimpleNamespace (similar to argparse.Namespace) instead of WorldArgs namedtuple
        args = SimpleNamespace(
            no_gui=True,
            fps=None,  # Not used in ReplayWorld
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
        rewards = []
        dones = []
        
        step = 0
        max_iterations = n_steps + 10  # Safety limit
        
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
            
            # Store data
            states.append(state.copy() if isinstance(state, np.ndarray) else state)
            actions.append(action_idx)
            # Store a copy of game_state dict
            game_state_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) 
                             for k, v in game_state.items()}
            game_states.append(game_state_copy)
            rewards.append(0.0)  # Reward will be computed later if needed
            dones.append(False)
            
            # Advance game step using do_step (which handles everything)
            replay_world.do_step()
            step += 1
            
            # Check if should stop
            if replay_world.time_to_stop() or not replay_world.running:
                break
        
        # Final done
        if len(dones) > 0:
            dones[-1] = True
        
        if len(states) < 10:  # Minimum episode length
            return None
        
        return {
            'states': np.array(states),
            'teacher_actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'game_states': game_states,
        }
        
    except Exception as e:
        print(f"Error extracting from replay world {replay_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract episode data from replay files')
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
    print(f"Extracting episodes to {output_dir}")
    
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
    
    print(f"\nExtracted {episode_count} episodes from {len(replay_files)} replay files")


if __name__ == '__main__':
    main()

