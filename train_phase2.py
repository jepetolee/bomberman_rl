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
from typing import List, Dict, Tuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
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
    env_model_path: str,
    policy_model_path: str,
    num_epochs: int = 100,
    batch_size: int = 64,
    n_planning_steps: int = 100,
    n_sup: int = 16,
    lr: float = 1e-4,
    device: torch.device = None,
):
    """Train policy with Dyna-Q planning + DeepSupervision"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training policy with planning on device: {device}")
    
    # Load episodes
    episodes = load_all_episodes(data_dir)
    if len(episodes) == 0:
        raise ValueError(f"No episodes found in {data_dir}")
    
    # Load environment model
    env_model = create_env_model(
        state_channels=9,
        state_height=s.ROWS,
        state_width=s.COLS,
        num_actions=len(ACTIONS),
        model_path=env_model_path,
        device=device,
    )
    
    # Create Dyna planner
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
    
    # Create policy model
    embed_dim = int(os.environ.get("BOMBER_VIT_DIM", 64))
    z_dim = int(os.environ.get("BOMBER_TRM_Z_DIM", str(embed_dim)))
    n_latent = int(os.environ.get("BOMBER_TRM_N", "6"))
    T = int(os.environ.get("BOMBER_TRM_T", "3"))
    
    model = PolicyValueViT_TRM(
        in_channels=9,
        num_actions=len(ACTIONS),
        img_size=(s.COLS, s.ROWS),
        embed_dim=embed_dim,
        z_dim=z_dim,
        n_latent=n_latent,
        n_sup=n_sup,
        T=T,
    ).to(device)
    
    optimizer = create_policy_optimizer(model, lr=lr)
    
    # Extract real experiences
    states, actions, _, _ = extract_transitions(episodes)
    states = states.to(device)
    actions = actions.to(device)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Generate simulated experiences via planning
        print(f"Epoch {epoch+1}/{num_epochs}: Generating simulated experiences...")
        simulated_experiences = planner.plan(n_planning_steps=n_planning_steps)
        
        if len(simulated_experiences) > 0:
            sim_states = torch.stack([s for s, _, _, _ in simulated_experiences]).to(device)
            sim_actions = torch.tensor([a for _, a, _, _ in simulated_experiences], device=device)
        else:
            sim_states = torch.empty(0, 9, s.ROWS, s.COLS).to(device)
            sim_actions = torch.empty(0, dtype=torch.long).to(device)
        
        # Sample batches
        num_real = min(batch_size // 2, len(states))
        num_sim = batch_size - num_real
        
        real_indices = torch.randperm(len(states))[:num_real]
        real_batch_states = states[real_indices]
        real_batch_actions = actions[real_indices]
        
        if len(sim_states) > 0:
            sim_indices = torch.randperm(len(sim_states))[:num_sim]
            sim_batch_states = sim_states[sim_indices]
            sim_batch_actions = sim_actions[sim_indices]
        else:
            sim_batch_states = torch.empty(0, 9, s.ROWS, s.COLS).to(device)
            sim_batch_actions = torch.empty(0, dtype=torch.long).to(device)
        
        # Train with DeepSupervision
        real_loss, sim_loss = train_with_simulated_experiences(
            model=model,
            real_states=real_batch_states,
            real_actions=real_batch_actions,
            simulated_states=sim_batch_states,
            simulated_actions=sim_batch_actions,
            optimizer=optimizer,
            n_sup=n_sup,
            strategy="last",
            real_weight=1.0,
            sim_weight=0.5,
            device=device,
        )
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Real Loss: {real_loss:.4f} | "
                  f"Sim Loss: {sim_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(policy_model_path), exist_ok=True)
    torch.save(model.state_dict(), policy_model_path)
    print(f"Saved policy model to {policy_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Dyna-Q + DeepSupervision Training')
    parser.add_argument('--train-env-model', action='store_true', help='Train environment model')
    parser.add_argument('--train-policy', action='store_true', help='Train policy with planning')
    parser.add_argument('--data-dir', type=str, default='data/teacher_episodes', help='Teacher data directory')
    parser.add_argument('--env-model-path', type=str, default='data/env_models/env_model.pt', help='Environment model path')
    parser.add_argument('--policy-model-path', type=str, default='data/policy_models/policy_phase2.pt', help='Policy model path')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--planning-steps', type=int, default=100, help='Number of planning steps')
    parser.add_argument('--n-sup', type=int, default=16, help='Number of supervision steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.train_env_model:
        train_env_model(
            data_dir=args.data_dir,
            model_path=args.env_model_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=device,
        )
    
    if args.train_policy:
        train_policy_with_planning(
            data_dir=args.data_dir,
            env_model_path=args.env_model_path,
            policy_model_path=args.policy_model_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            n_planning_steps=args.planning_steps,
            n_sup=args.n_sup,
            lr=args.lr,
            device=device,
        )


if __name__ == '__main__':
    main()

