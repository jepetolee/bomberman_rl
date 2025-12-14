#!/usr/bin/env python3
"""
Phase 2 Training: Dyna-Q Planning + DeepSupervision
====================================================

1. Train environment model on collected teacher data
2. Use Dyna-Q planning to generate simulated experiences
3. Train Policy Network with DeepSupervision on real + simulated data

Usage:
    python train_phase2.py --train-env-model
    python train_phase2.py --train-policy --use-planning
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
from config.load_config import load_config, get_phase2_config, get_model_config, create_model_from_config, apply_config_to_env
from agent_code.ppo_agent.models.environment_model import EnvironmentModel, create_env_model
from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM
from agent_code.ppo_agent.dyna_planning import DynaPlanner, VisitedStatesBuffer
from agent_code.ppo_agent.train_deepsup import train_policy_deepsup, train_with_simulated_experiences, create_policy_optimizer
from agent_code.ppo_agent.callbacks import ACTIONS


class TeacherDataset(Dataset):
    """Dataset for teacher episode data"""
    def __init__(self, data_dir: str, max_episodes: int = None):
        self.data_dir = Path(data_dir)
        self.episode_files = sorted(self.data_dir.glob('episode_*.pt'))
        
        if max_episodes is not None:
            self.episode_files = self.episode_files[:max_episodes]
        
        print(f"Loaded {len(self.episode_files)} episodes from {data_dir}")
    
    def __len__(self):
        return len(self.episode_files)
    
    def __getitem__(self, idx):
        with open(self.episode_files[idx], 'rb') as f:
            episode = pickle.load(f)
        return episode


def load_all_episodes(data_dir: str, max_episodes: int = None) -> List[Dict]:
    """Load all episodes into memory"""
    dataset = TeacherDataset(data_dir, max_episodes)
    episodes = []
    
    for i in range(len(dataset)):
        episodes.append(dataset[i])
    
    return episodes


def extract_transitions(episodes: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract (state, action, reward, next_state) transitions from episodes
    
    Returns:
        states: [N, C, H, W]
        actions: [N]
        rewards: [N]
        next_states: [N, C, H, W]
    """
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    
    for episode in episodes:
        states = episode['states']  # [T, C, H, W]
        actions = episode['teacher_actions']  # [T]
        rewards = episode['rewards']  # [T]
        
        # Create transitions: (s_t, a_t, r_t, s_{t+1})
        for t in range(len(states) - 1):
            all_states.append(states[t])
            all_actions.append(actions[t])
            all_rewards.append(rewards[t])
            all_next_states.append(states[t + 1])
    
    # Convert to tensors
    states_tensor = torch.from_numpy(np.array(all_states)).float()
    actions_tensor = torch.from_numpy(np.array(all_actions)).long()
    rewards_tensor = torch.from_numpy(np.array(all_rewards)).float()
    next_states_tensor = torch.from_numpy(np.array(all_next_states)).float()
    
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor


def train_env_model(
    data_dir: str,
    model_path: str,
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device = None,
):
    """Train environment model on teacher data"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training environment model on device: {device}")
    
    # Load episodes
    episodes = load_all_episodes(data_dir)
    if len(episodes) == 0:
        raise ValueError(f"No episodes found in {data_dir}")
    
    # Extract transitions
    states, actions, rewards, next_states = extract_transitions(episodes)
    
    print(f"Loaded {len(states)} transitions")
    
    # Create model
    model = EnvironmentModel(
        state_channels=9,
        state_height=s.ROWS,
        state_width=s.COLS,
        num_actions=len(ACTIONS),
        hidden_dim=256,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_state = nn.MSELoss()
    criterion_reward = nn.MSELoss()
    
    # Data loader
    dataset = torch.utils.data.TensorDataset(states, actions, rewards, next_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_state_loss = 0.0
        total_reward_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_actions, batch_rewards, batch_next_states in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_rewards = batch_rewards.to(device)
            batch_next_states = batch_next_states.to(device)
            
            # Forward
            next_state_pred, reward_pred = model(batch_states, batch_actions)
            
            # Losses
            state_loss = criterion_state(next_state_pred, batch_next_states)
            reward_loss = criterion_reward(reward_pred, batch_rewards)
            loss = state_loss + reward_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            num_batches += 1
        
        avg_state_loss = total_state_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"State Loss: {avg_state_loss:.4f} | "
                  f"Reward Loss: {avg_reward_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved environment model to {model_path}")


def train_policy_with_planning(
    data_dir: str,
    env_model_path: Optional[str],
    policy_model_path: str,
    num_epochs: int = 100,
    batch_size: int = 64,
    n_planning_steps: int = 100,
    n_sup: int = 16,
    lr: float = 1e-4,
    strategy: str = "last",
    train_value: bool = True,
    value_weight: float = 0.5,
    real_weight: float = 1.0,
    sim_weight: float = 0.5,
    config_path: str = "config/trm_config.yaml",
    device: torch.device = None,
):
    """Train policy with Dyna-Q planning + DeepSupervision"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training policy with planning on device: {device}")
    
    # Load config for model creation
    from config.load_config import load_config as load_config_func, create_model_from_config
    config = load_config_func(config_path)
    
    # Load episodes
    episodes = load_all_episodes(data_dir)
    if len(episodes) == 0:
        raise ValueError(f"No episodes found in {data_dir}")
    
    # Load environment model (if planning is enabled)
    env_model = None
    if env_model_path is not None:
        env_model = create_env_model(
            state_channels=10,  # Updated to match state_to_features output
            state_height=s.ROWS,
            state_width=s.COLS,
            num_actions=len(ACTIONS),
            model_path=env_model_path,
            device=device,
        )
    
    # Create Dyna planner (if planning is enabled)
    planner = None
    if env_model is not None:
        from agent_code.ppo_agent.dyna_planning import DynaPlanner
        visited_states = VisitedStatesBuffer(max_size=10000)
        planner = DynaPlanner(env_model, visited_states, device=device)
        
        # Populate visited states from real episodes
        print("Populating visited states buffer...")
        for episode in episodes:
            states = episode['states']
            actions = episode['teacher_actions']
            for state, action in zip(states, actions):
                state_tensor = torch.from_numpy(state).float()
                planner.add_experience(state_tensor, int(action))
        
        print(f"Visited states buffer size: {len(visited_states)}")
    
    # Create policy model from config
    # strict_yaml=True: YAML 설정을 엄격하게 따르고, 기본값을 사용하지 않음
    model = create_model_from_config(config, device=device, strict_yaml=True)
    
    # Check model type to determine if DeepSupervision is used
    from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
    from agent_code.ppo_agent.models.vit import PolicyValueViT
    from agent_code.ppo_agent.models.efficient_gtrxl import PolicyValueEfficientGTrXL
    is_vit_only = isinstance(model, PolicyValueViT)
    is_efficient_gtrxl = isinstance(model, PolicyValueEfficientGTrXL)
    is_hybrid = isinstance(model, PolicyValueViT_TRM_Hybrid)
    use_deepsup = not (is_vit_only or is_efficient_gtrxl)  # Only TRM models use DeepSupervision
    
    optimizer = create_policy_optimizer(model, lr=lr)
    
    # Extract real experiences (including rewards)
    # Limit number of episodes to avoid OOM (sample randomly if too many)
    max_episodes_for_training = 1000  # Limit to avoid OOM
    if len(episodes) > max_episodes_for_training:
        import random
        episodes = random.sample(episodes, max_episodes_for_training)
        print(f"Sampled {max_episodes_for_training} episodes from {len(episodes) + max_episodes_for_training} total")
    
    states, actions, rewards, _ = extract_transitions(episodes)
    # Keep on CPU, move to device only during batch processing
    # states = states.to(device)  # Don't move all at once
    # actions = actions.to(device)
    # rewards = rewards.to(device) if train_value else None
    
    # Training loop
    model.train()
    
    # Calculate number of batches per epoch
    num_real_samples = len(states)
    num_batches_per_epoch = max(1, num_real_samples // batch_size)
    
    print(f"Training configuration:")
    print(f"  - Model type: {model.__class__.__name__}")
    print(f"  - Total real samples: {num_real_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Batches per epoch: {num_batches_per_epoch}")
    if use_deepsup:
        print(f"  - DeepSupervision steps (n_sup): {n_sup} (TRM 모델용)")
    else:
        print(f"  - DeepSupervision: 사용 안 함 (표준 지도학습, n_sup 파라미터 무시)")
    print(f"  - Strategy: {strategy}")
    
    for epoch in range(num_epochs):
        epoch_real_policy_loss = 0.0
        epoch_real_value_loss = 0.0
        epoch_sim_policy_loss = 0.0
        epoch_sim_value_loss = 0.0
        num_batches_processed = 0
        
        # Generate simulated experiences via planning (if enabled)
        simulated_experiences = []
        if planner is not None and n_planning_steps > 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}: Generating {n_planning_steps} simulated experiences...")
            simulated_experiences = planner.plan(n_planning_steps=n_planning_steps)
            print(f"  Generated {len(simulated_experiences)} simulated experiences")
        
        if len(simulated_experiences) > 0:
            sim_states_all = torch.stack([s for s, _, _, _ in simulated_experiences])  # Keep on CPU for now
            sim_actions_all = torch.tensor([a for _, a, _, _ in simulated_experiences])
            sim_rewards_all = torch.tensor([r for _, _, r, _ in simulated_experiences]) if train_value else None
        else:
            sim_states_all = torch.empty(0, 10, s.ROWS, s.COLS)
            sim_actions_all = torch.empty(0, dtype=torch.long)
            sim_rewards_all = None
        
        # Shuffle data for this epoch
        real_indices = torch.randperm(num_real_samples)
        
        # Process batches
        for batch_idx in range(num_batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_real_samples)
            
            # Get real batch
            batch_real_indices = real_indices[start_idx:end_idx]
            real_batch_states = states[batch_real_indices].to(device)
            real_batch_actions = actions[batch_real_indices].to(device)
            real_batch_rewards = rewards[batch_real_indices].to(device) if rewards is not None else None
            
            # Get simulated batch (if available)
            if len(sim_states_all) > 0:
                num_sim_in_batch = min(batch_size - len(batch_real_indices), len(sim_states_all))
                sim_batch_indices = torch.randperm(len(sim_states_all))[:num_sim_in_batch]
                sim_batch_states = sim_states_all[sim_batch_indices].to(device)
                sim_batch_actions = sim_actions_all[sim_batch_indices].to(device)
                sim_batch_rewards = sim_rewards_all[sim_batch_indices].to(device) if sim_rewards_all is not None else None
            else:
                sim_batch_states = torch.empty(0, 10, s.ROWS, s.COLS).to(device)
                sim_batch_actions = torch.empty(0, dtype=torch.long).to(device)
                sim_batch_rewards = None
            
            # Train with DeepSupervision (this will perform n_sup iterations internally)
            real_policy_loss, real_value_loss, sim_policy_loss, sim_value_loss = train_with_simulated_experiences(
                model=model,
                real_states=real_batch_states,
                real_actions=real_batch_actions,
                simulated_states=sim_batch_states,
                simulated_actions=sim_batch_actions,
                optimizer=optimizer,
                real_rewards=real_batch_rewards,
                sim_rewards=sim_batch_rewards,
                train_value=train_value,
                value_weight=value_weight,
                real_weight=real_weight,
                sim_weight=sim_weight,
                n_sup=n_sup,
                strategy=strategy,
                device=device,
            )
            
            epoch_real_policy_loss += real_policy_loss
            epoch_real_value_loss += real_value_loss
            epoch_sim_policy_loss += sim_policy_loss
            epoch_sim_value_loss += sim_value_loss
            num_batches_processed += 1
            
            # Print batch progress every 10 batches or at the end
            if (batch_idx + 1) % max(1, num_batches_per_epoch // 10) == 0 or batch_idx == num_batches_per_epoch - 1:
                print(f"  Batch {batch_idx+1}/{num_batches_per_epoch} | "
                      f"Real P: {real_policy_loss:.4f}, Real V: {real_value_loss:.4f} | "
                      f"Sim P: {sim_policy_loss:.4f}, Sim V: {sim_value_loss:.4f}")
        
        # Average losses for the epoch
        epoch_real_policy_loss /= num_batches_processed
        epoch_real_value_loss /= num_batches_processed
        epoch_sim_policy_loss /= num_batches_processed
        epoch_sim_value_loss /= num_batches_processed
        
        print(f"Epoch {epoch+1}/{num_epochs} Summary | "
              f"Avg Real Policy Loss: {epoch_real_policy_loss:.4f}, Avg Real Value Loss: {epoch_real_value_loss:.4f} | "
              f"Avg Sim Policy Loss: {epoch_sim_policy_loss:.4f}, Avg Sim Value Loss: {epoch_sim_value_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(policy_model_path), exist_ok=True)
    torch.save(model.state_dict(), policy_model_path)
    print(f"Saved policy model to {policy_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Dyna-Q Planning + DeepSupervision')
    parser.add_argument('--config', type=str, default='config/trm_config.yaml', help='Path to YAML config file')
    parser.add_argument('--train-env-model', action='store_true', help='Train environment model')
    parser.add_argument('--train-policy', action='store_true', help='Train policy with DeepSupervision')
    parser.add_argument('--use-planning', action='store_true', help='Use Dyna-Q planning (requires trained env model)')
    parser.add_argument('--data-dir', type=str, default=None, help='Directory with teacher episodes (overrides config)')
    parser.add_argument('--env-model-path', type=str, default=None, help='Path to environment model (overrides config)')
    parser.add_argument('--policy-model-path', type=str, default=None, help='Path to save policy model (overrides config)')
    parser.add_argument('--num-epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--n-sup', type=int, default=None, help='Number of supervision steps (overrides config)')
    parser.add_argument('--planning-steps', type=int, default=None, help='Number of planning steps per epoch (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--train-value', action='store_true', default=None, help='Train value network with rewards (overrides config)')
    
    args = parser.parse_args()
    
    # Load YAML configuration
    config = load_config(args.config)
    phase2_cfg = get_phase2_config(config)
    model_cfg = get_model_config(config)
    
    # Apply config to environment variables
    apply_config_to_env(config)
    
    # Use command line args if provided, otherwise use config
    data_dir = args.data_dir or phase2_cfg.get('data_dir', 'data/teacher_episodes')
    env_model_path = args.env_model_path or phase2_cfg.get('env_model', {}).get('model_path', 'data/env_models/env_model.pt')
    policy_model_path = args.policy_model_path or phase2_cfg.get('deepsupervision', {}).get('policy_model_path', 'ppo_model.pt')
    
    # DeepSupervision config
    deepsup_cfg = phase2_cfg.get('deepsupervision', {})
    num_epochs = args.num_epochs or deepsup_cfg.get('num_epochs', 100)
    batch_size = args.batch_size or deepsup_cfg.get('batch_size', 64)
    n_sup = args.n_sup or model_cfg.get('trm', {}).get('n_sup', 16)
    # Ensure lr is a float (YAML might read scientific notation as string)
    lr = float(args.lr if args.lr is not None else deepsup_cfg.get('learning_rate', 1e-4))
    strategy = deepsup_cfg.get('strategy', 'last')
    train_value = args.train_value if args.train_value is not None else deepsup_cfg.get('train_value', True)
    value_weight = float(deepsup_cfg.get('value_weight', 0.5))
    real_weight = float(deepsup_cfg.get('real_experience_weight', 1.0))
    sim_weight = float(deepsup_cfg.get('sim_experience_weight', 0.5))
    
    # Planning config
    planning_cfg = phase2_cfg.get('planning', {})
    n_planning_steps = args.planning_steps or planning_cfg.get('n_planning_steps', 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.train_env_model:
        env_model_cfg = phase2_cfg.get('env_model', {})
        train_env_model(
            data_dir=data_dir,
            model_path=env_model_path,
            num_epochs=env_model_cfg.get('num_epochs', 50),
            batch_size=env_model_cfg.get('batch_size', 128),
            lr=float(env_model_cfg.get('learning_rate', 1e-3)),
        )
    
    if args.train_policy:
        use_planning = args.use_planning or planning_cfg.get('enabled', False)
        train_policy_with_planning(
            data_dir=data_dir,
            env_model_path=env_model_path if use_planning else None,
            policy_model_path=policy_model_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_planning_steps=n_planning_steps if use_planning else 0,
            n_sup=n_sup,
            lr=lr,
            strategy=strategy,
            train_value=train_value,
            value_weight=value_weight,
            real_weight=real_weight,
            sim_weight=sim_weight,
            config_path=args.config,  # Pass config path
        )


if __name__ == '__main__':
    main()

