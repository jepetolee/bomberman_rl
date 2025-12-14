#!/usr/bin/env python3
"""
Single Agent + Planning Training (Stage-based, Self-Play)
==========================================================

Architecture:
  - Single process, single model
  - Sequential: Play games → Collect experiences → PPO update → Planning update
  - Stage-based curriculum progression
  - Self-Play: Agent vs Agent (same model) for final stage
  - TRM-focused: Only TRM parameters are trained (ViT frozen)

Environment Model (env_model):
  - Predicts (next_state, reward) from (state, action)
  - Trained in Phase 2 on teacher data
  - Used for Dyna-Q planning to generate simulated experiences
  - Path: --env-model-path (e.g., data/env_models/env_model.pt)

Usage:
    python a2c_planning_train.py --total-rounds 50000 --planning-episodes 5
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

# Disable audio
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as s
from config.load_config import load_config, create_model_from_config
from agent_code.ppo_agent.models.environment_model import create_env_model, EnvironmentModel
from agent_code.ppo_agent.dyna_planning import DynaPlanner, VisitedStatesBuffer


# ============== Policy Model as Environment Model Wrapper ==============

class PolicyAsEnvModel(nn.Module):
    """
    Wrapper to use policy model as environment model.
    Policy model is copied from teacher forcing model and both are updated.
    """
    def __init__(self, policy_model: nn.Module):
        super().__init__()
        self.policy_model = policy_model
        
        # Add head to predict next_state and reward from policy features
        # Use policy model's embed_dim
        if hasattr(policy_model, 'embed_dim'):
            self.embed_dim = policy_model.embed_dim
        elif hasattr(policy_model, 'vit') and hasattr(policy_model.vit, 'embed_dim'):
            self.embed_dim = policy_model.vit.embed_dim
        else:
            self.embed_dim = 256  # Default
        
        # Head to predict next_state (same shape as input)
        self.next_state_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, 10 * s.ROWS * s.COLS),  # 10 channels * H * W
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Head to predict reward
        self.reward_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 1),
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict (next_state, reward) from (state, action)
        
        Env model은 TRM을 추론에 사용하지 않음 (파라미터는 유지하되 사용 안함).
        ViT features만 사용하여 학습과 추론 수행.
        ViT + Policy head + Value head는 학습됨.
        
        Args:
            state: [B, C, H, W] current state
            action: [B] action indices
        Returns:
            next_state_pred: [B, C, H, W] predicted next state
            reward_pred: [B] predicted reward
        """
        B = state.shape[0]
        
        # Get features from policy model
        # TRM은 사용하지 않음 (파라미터는 유지하되 추론에 작용 안함)
        from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
        if isinstance(self.policy_model, PolicyValueViT_TRM_Hybrid):
            # Hybrid model: use ViT forward_features directly (TRM excluded)
            features = self.policy_model.vit.forward_features(state)  # [B, embed_dim]
        elif hasattr(self.policy_model, 'forward_features'):
            # ViT-based model
            features = self.policy_model.forward_features(state)  # [B, embed_dim]
        elif hasattr(self.policy_model, 'vit'):
            # Hybrid model (fallback)
            features = self.policy_model.vit.forward_features(state)  # [B, embed_dim]
        else:
            # Fallback: use forward and extract features
            with torch.no_grad():
                logits, value = self.policy_model(state)
            # Use value as feature (simple approximation)
            features = value.unsqueeze(-1) if value.dim() == 1 else value
            if features.shape[-1] != self.embed_dim:
                # Project to embed_dim if needed
                if not hasattr(self, '_proj'):
                    self._proj = nn.Linear(features.shape[-1], self.embed_dim).to(features.device)
                features = self._proj(features)
        
        # Incorporate action information
        # Simple: add action embedding to features
        if not hasattr(self, 'action_embed'):
            self.action_embed = nn.Embedding(6, self.embed_dim).to(features.device)
        action_emb = self.action_embed(action)  # [B, embed_dim]
        combined_features = features + action_emb  # [B, embed_dim]
        
        # Predict next_state
        next_state_flat = self.next_state_head(combined_features)  # [B, 10*H*W]
        next_state_pred = next_state_flat.view(B, 10, s.ROWS, s.COLS)  # [B, 10, H, W]
        
        # Predict reward
        reward_pred = self.reward_head(combined_features).squeeze(-1)  # [B]
        
        return next_state_pred, reward_pred
from agent_code.ppo_agent.train_frozen_vit import setup_frozen_vit_optimizer, _ppo_update_frozen_vit
from agent_code.ppo_agent.callbacks import SHARED


# ============== Stage-based Curriculum ==============

CURRICULUM_STAGES = [
    # (stage_name, opponent_pool, rounds_per_stage, win_rate_threshold)
    ("Stage 1: Easy", ['random_agent', 'peaceful_agent'], 5000, 0.60),
    ("Stage 2: Medium", ['peaceful_agent', 'coin_collector_agent'], 5000, 0.65),
    ("Stage 3: Hard", ['coin_collector_agent', 'rule_based_agent'], 5000, 0.70),
    ("Stage 4: Expert", ['team_teacher_agent', 'aggressive_teacher_agent'], 5000, 0.75),
    # Stage 5 is self-play - handled separately
]

SELF_PLAY_STAGE = ("Stage 5: Self-Play", None, 10000, 0.0)  # No threshold for self-play


class StageTracker:
    """Track progress through curriculum stages"""
    
    def __init__(self):
        self.current_stage = 0
        self.stage_rounds = 0
        self.stage_wins = 0
        self.stage_start_time = time.time()
    
    def get_stage_info(self) -> Tuple[str, List[str], int, float]:
        """Get current stage info"""
        if self.current_stage < len(CURRICULUM_STAGES):
            return CURRICULUM_STAGES[self.current_stage]
        else:
            return SELF_PLAY_STAGE
    
    def is_selfplay(self) -> bool:
        return self.current_stage >= len(CURRICULUM_STAGES)
    
    def add_result(self, won: bool):
        """Record a match result"""
        self.stage_rounds += 1
        if won:
            self.stage_wins += 1
    
    def get_win_rate(self) -> float:
        if self.stage_rounds == 0:
            return 0.0
        return self.stage_wins / self.stage_rounds
    
    def should_advance(self) -> bool:
        """Check if should advance to next stage"""
        if self.is_selfplay():
            return False
        
        stage_name, opponent_pool, rounds_per_stage, threshold = self.get_stage_info()
        
        # Need minimum rounds and win rate threshold
        if self.stage_rounds < rounds_per_stage // 2:
            return False
        
        return self.get_win_rate() >= threshold
    
    def advance_stage(self):
        """Advance to next stage"""
        old_stage = self.current_stage
        self.current_stage += 1
        self.stage_rounds = 0
        self.stage_wins = 0
        
        elapsed = time.time() - self.stage_start_time
        print(f"\n{'='*60}")
        print(f"🎯 STAGE ADVANCEMENT!")
        if old_stage < len(CURRICULUM_STAGES):
            print(f"   {CURRICULUM_STAGES[old_stage][0]} → ", end="")
        if self.current_stage < len(CURRICULUM_STAGES):
            print(f"{CURRICULUM_STAGES[self.current_stage][0]}")
        else:
            print(f"{SELF_PLAY_STAGE[0]}")
        print(f"   Rounds in stage: {self.stage_rounds}")
        print(f"   Time in stage: {elapsed/60:.1f} min")
        print(f"{'='*60}\n")
        
        self.stage_start_time = time.time()
    
    def get_opponents(self) -> Tuple[str, str]:
        """Get opponents for current stage"""
        if self.is_selfplay():
            return 'ppo_agent', 'ppo_agent'  # Self-play
        
        stage_name, opponent_pool, _, _ = self.get_stage_info()
        opp1 = random.choice(opponent_pool)
        opp2 = random.choice(opponent_pool)
        return opp1, opp2


# ============== Model Creation ==============

def create_model(model_path: str, device: torch.device):
    """Create model from config"""
    try:
        from config.load_config import load_config, create_model_from_config
        cfg_path = os.environ.get("BOMBER_CONFIG_PATH", "config/trm_config.yaml")
        cfg = load_config(cfg_path)
        # strict_yaml=True: YAML 설정을 엄격하게 따르고, 기본값을 사용하지 않음
        model = create_model_from_config(cfg, device=device, strict_yaml=True)
    except Exception as e:
        print(f"Failed to load from config: {e}, using fallback")
        from agent_code.ppo_agent.models.vit import PolicyValueViT
        embed_dim = int(os.environ.get("BOMBER_VIT_DIM", "256"))
        depth = int(os.environ.get("BOMBER_VIT_DEPTH", "2"))
        num_heads = int(os.environ.get("BOMBER_VIT_HEADS", "4"))
        
        model = PolicyValueViT(
            in_channels=10,
            num_actions=6,
            img_size=(s.ROWS, s.COLS),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            patch_size=1,
            use_cls_token=False,
            mixer="attn",
        ).to(device)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {model_path}")
        except Exception:
            print(f"Failed to load model, using fresh initialization")
    
    return model


# ============== Environment Model Training ==============

def update_env_model(
    env_model: nn.Module,
    env_optimizer: optim.Optimizer,
    experience_buffer: List[Tuple],
    batch_size: int = 64,
    num_updates: int = 10,
    device: torch.device = None,
):
    """
    Update environment model using collected real experiences.
    
    Env model 학습 범위:
    - ViT 백본: 학습 ✓
    - Policy head: 학습 ✓
    - Value head: 학습 ✓
    - Env heads (next_state_head, reward_head): 학습 ✓
    - TRM: frozen ✗ (절대 학습 안함, 추론에도 사용 안함)
    
    Args:
        experience_buffer: List of (state, action, reward, next_state) tuples
        batch_size: Batch size for training
        num_updates: Number of gradient steps
    """
    if len(experience_buffer) < batch_size:
        return
    
    env_model.train()
    
    # Ensure TRM is frozen during env_model training
    from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
    if isinstance(env_model.policy_model, PolicyValueViT_TRM_Hybrid):
        # Freeze TRM (절대 학습 안함)
        for param in env_model.policy_model.trm_net.parameters():
            param.requires_grad = False
    
    criterion_state = nn.MSELoss()
    criterion_reward = nn.MSELoss()
    
    # Sample random batches
    for _ in range(num_updates):
        batch = random.sample(experience_buffer, min(batch_size, len(experience_buffer)))
        
        states = torch.stack([s for s, _, _, _ in batch]).to(device)  # [B, C, H, W]
        actions = torch.tensor([a for _, a, _, _ in batch], dtype=torch.long, device=device)  # [B]
        rewards = torch.tensor([r for _, _, r, _ in batch], dtype=torch.float32, device=device)  # [B]
        next_states = torch.stack([ns for _, _, _, ns in batch]).to(device)  # [B, C, H, W]
        
        # Forward pass (TRM은 사용 안함, ViT features만 사용)
        next_state_pred, reward_pred = env_model(states, actions)
        
        # Losses
        state_loss = criterion_state(next_state_pred, next_states)
        reward_loss = criterion_reward(reward_pred, rewards)
        loss = state_loss + reward_loss
        
        # Backward
        env_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients only for trainable parameters (TRM 제외)
        trainable_params = []
        if isinstance(env_model.policy_model, PolicyValueViT_TRM_Hybrid):
            trainable_params.extend(list(env_model.policy_model.vit.parameters()))
            trainable_params.extend(list(env_model.policy_model.pi_head.parameters()))
            trainable_params.extend(list(env_model.policy_model.v_head.parameters()))
        trainable_params.extend(list(env_model.next_state_head.parameters()))
        trainable_params.extend(list(env_model.reward_head.parameters()))
        if hasattr(env_model, 'action_embed'):
            trainable_params.extend(list(env_model.action_embed.parameters()))
        
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        env_optimizer.step()
    
    env_model.eval()


# ============== Planning Integration ==============

def run_planning_updates(
    model: nn.Module,
    planner: DynaPlanner,
    n_episodes: int,
    horizon: int,
    device: torch.device,
    model_path: str,
):
    """Run N planning episodes and update model"""
    if planner is None or len(planner.visited_states) < 200:
        return
    
    model.train()
    
    for ep in range(n_episodes):
        # Build simulated rollout
        sim_states = []
        sim_actions = []
        sim_logps = []
        sim_values = []
        sim_rewards = []
        sim_dones = []
        
        try:
            start_state, _ = planner.visited_states.sample_state()
        except Exception:
            return
        
        state = start_state.unsqueeze(0).to(device)
        z_prev = None  # Non-recurrent: always start from zero
        
        model.eval()
        for t in range(horizon):
            with torch.no_grad():
                if hasattr(model, 'forward_with_z'):
                    logits, value, z_new = model.forward_with_z(state, z_prev=None)
                else:
                    logits, value = model(state)
                    z_new = None
                
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                
                # World model step (use updated env_model)
                next_state_pred, reward_pred = planner.env_model(state, action)
            
            sim_states.append(state.squeeze(0).detach().cpu())
            sim_actions.append(int(action.item()))
            sim_logps.append(float(logp.item()))
            sim_values.append(float(value.squeeze(0).item()))
            sim_rewards.append(float(reward_pred.squeeze(0).item()))
            sim_dones.append(False)
            
            state = next_state_pred.detach()
        
        sim_dones[-1] = True
        
        # Swap buffers and run PPO update with DeepSupervision
        # Planning updates use DeepSupervision: each (state, action) pair goes through
        # n_sup recursive reasoning steps, with loss computed at each step
        SHARED.buf_states = sim_states
        SHARED.buf_actions = sim_actions
        SHARED.buf_logps = sim_logps
        SHARED.buf_values = sim_values
        SHARED.buf_rewards = sim_rewards
        SHARED.buf_dones = sim_dones
        
        # Use DeepSupervision for Planning updates
        _ppo_update_frozen_vit(use_deep_supervision=True)
    
    # Save updated model
    try:
        torch.save(model.state_dict(), model_path)
    except Exception:
        pass


# ============== Main Training Loop ==============

def run_training(config: Dict):
    """
    Single process training loop:
    1. Play games (subprocess)
    2. Collect experiences (from SHARED buffers)
    3. PPO update
    4. Planning update (if enabled)
    5. Next round
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Single Agent + Planning Training (TRM-focused)")
    print("="*60)
    print(f"  Total Rounds: {config['total_rounds']}")
    print(f"  Rounds per Batch: {config['rounds_per_batch']}")
    print(f"  Planning Episodes: {config['planning_episodes']}")
    print(f"  Planning Horizon: {config['planning_horizon']}")
    print()
    
    results_dir = os.path.abspath(config['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = os.path.join(results_dir, 'model.pt')
    
    # Create model
    print("Creating model...")
    model = create_model(config.get('initial_model_path', None) or model_path, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (TRM-only)
    SHARED.policy = model
    SHARED.model_path = model_path
    setup_frozen_vit_optimizer()
    print()
    
    # Setup planning if enabled
    planner = None
    env_model = None
    env_optimizer = None
    env_model_path = config.get('env_model_path')
    experience_buffer = []  # Store (state, action, reward, next_state) tuples
    max_experience_buffer_size = 10000  # Keep last N experiences
    
    if config.get('use_planning', True):
        print(f"Environment Model (env_model):")
        print(f"  - Purpose: Predicts (next_state, reward) from (state, action)")
        print(f"  - Structure: Copied from teacher forcing model (phase2 pretrained)")
        print(f"  - Training: Real-time updates from collected experiences")
        print(f"  - Usage: Dyna-Q planning to generate simulated experiences")
        
        # Copy teacher forcing model (phase2 pretrained) as env_model
        try:
            import copy
            # Deep copy the policy model to use as base for env_model
            env_policy_model = copy.deepcopy(model)
            env_policy_model.train()  # Start in train mode for updates
            
            # Wrap policy model as environment model
            env_model = PolicyAsEnvModel(env_policy_model).to(device)
            env_model.train()  # Start in train mode for updates
            
            # Create optimizer for env_model: ViT + Policy head + Value head만 학습, TRM은 frozen
            from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
            if isinstance(env_policy_model, PolicyValueViT_TRM_Hybrid):
                # Collect trainable parameters for env_model
                env_trainable_params = []
                
                # ViT 백본 학습
                env_trainable_params.extend(list(env_model.policy_model.vit.parameters()))
                
                # Policy head 학습
                env_trainable_params.extend(list(env_model.policy_model.pi_head.parameters()))
                
                # Value head 학습
                env_trainable_params.extend(list(env_model.policy_model.v_head.parameters()))
                
                # Env model's own heads 학습
                env_trainable_params.extend(list(env_model.next_state_head.parameters()))
                env_trainable_params.extend(list(env_model.reward_head.parameters()))
                if hasattr(env_model, 'action_embed'):
                    env_trainable_params.extend(list(env_model.action_embed.parameters()))
                
                # TRM은 frozen (학습 안함)
                for param in env_model.policy_model.trm_net.parameters():
                    param.requires_grad = False
                
                # Create optimizer with only trainable parameters
                env_optimizer = optim.Adam(env_trainable_params, lr=1e-3)
                print(f"  ✓ Env Model Optimizer: ViT + Policy head + Value head + Env heads only")
                print(f"    (TRM frozen: {sum(p.numel() for p in env_model.policy_model.trm_net.parameters()):,} params)")
            else:
                # Fallback: use all parameters
                env_optimizer = optim.Adam(env_model.parameters(), lr=1e-3)
            
            visited_states = VisitedStatesBuffer(max_size=10000)
            planner = DynaPlanner(env_model, visited_states, device=device)
            print(f"  ✓ Created by copying teacher forcing model (will update in real-time)")
            print(f"  ✓ Both policy model and env_model will be updated jointly")
            
            # If initial env_model_path provided, try to load it (for compatibility)
            if env_model_path and os.path.exists(env_model_path):
                try:
                    # Try to load as EnvironmentModel first (old format)
                    old_env_model = create_env_model(
                        state_channels=10,
                        state_height=s.ROWS,
                        state_width=s.COLS,
                        num_actions=6,
                        model_path=env_model_path,
                        device=device,
                    )
                    print(f"  ℹ️  Note: env_model_path provided but using copied policy model instead")
                except Exception:
                    pass
        except Exception as e:
            print(f"  ⚠️  Failed to create env_model from policy model: {e}")
            import traceback
            traceback.print_exc()
            config['use_planning'] = False
    else:
        print("  Planning disabled")
    print()
    
    # Set environment variables for subprocess
    train_env = os.environ.copy()
    train_env['PPO_MODEL_PATH'] = model_path
    train_env['BOMBER_FROZEN_VIT'] = '1'  # TRM-only training
    train_env['BOMBER_USE_TRM'] = '1'
    train_env['BOMBER_TRM_RECURRENT'] = '0'  # Non-recurrent
    
    # Training loop
    stage_tracker = StageTracker()
    total_rounds = 0
    last_log_round = 0
    
    print(f"Starting training...\n")
    start_time = time.time()
    
    try:
        while total_rounds < config['total_rounds']:
            # Get opponents for current stage
            opp1, opp2 = stage_tracker.get_opponents()
            
            # Build agent list (ppo_agent always first two for training)
            agent_list = ['ppo_agent', 'ppo_agent', opp1, opp2]
            
            output_file = os.path.join(
                results_dir,
                f"round_{total_rounds}_{int(time.time()*1000)}.json"
            )
            
            # Play games
            cmd = [
                sys.executable, "main.py", "play",
                "--agents"] + agent_list + [
                "--train", "2",
                "--no-gui",
                "--n-rounds", str(config['rounds_per_batch']),
                "--save-stats", output_file,
                "--silence-errors",
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=train_env,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                # Parse results
                batch_stats = {
                    'team_a_score': 0, 'team_b_score': 0,
                    'kills': 0, 'deaths': 0,
                    'opponents': f"{opp1},{opp2}",
                    'stage': stage_tracker.current_stage,
                    'is_selfplay': stage_tracker.is_selfplay(),
                }
                
                if result.returncode == 0 and os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                        
                        by_agent = data.get("by_agent", {})
                        by_round = data.get("by_round", {})
                        
                        # Aggregate stats
                        total_kills = sum(r.get("kills", 0) for r in by_round.values())
                        ppo_score = 0
                        opp_score = 0
                        
                        for agent_name, stats in by_agent.items():
                            score = stats.get("score", 0)
                            if agent_name.startswith("ppo_agent"):
                                batch_stats['team_a_score'] += score
                                batch_stats['deaths'] += stats.get("suicides", 0)
                                ppo_score += score
                            else:
                                batch_stats['team_b_score'] += score
                                opp_score += score
                        
                        if total_kills > 0 and ppo_score >= opp_score and ppo_score > 0:
                            batch_stats['kills'] = total_kills
                        
                        # Add visited states to planner (simplified - would need replay parsing)
                        # For now, we'll rely on states collected during actual gameplay
                        
                    except Exception:
                        pass
                    finally:
                        try:
                            os.remove(output_file)
                        except:
                            pass
                
                # Extract real experiences for env_model training
                if env_model is not None and len(SHARED.buf_states) > 1:
                    # Extract transitions: (state, action, reward, next_state)
                    states = SHARED.buf_states
                    actions = SHARED.buf_actions
                    rewards = SHARED.buf_rewards
                    
                    for i in range(len(states) - 1):
                        state = states[i]
                        action = actions[i]
                        reward = rewards[i]
                        next_state = states[i + 1]
                        
                        # Add to experience buffer
                        experience_buffer.append((state, action, reward, next_state))
                        
                        # Also add to visited_states for planning
                        if planner is not None:
                            planner.visited_states.add(state, action)
                    
                    # Keep buffer size manageable
                    if len(experience_buffer) > max_experience_buffer_size:
                        experience_buffer = experience_buffer[-max_experience_buffer_size:]
                
                # Real PPO update (from SHARED buffers filled during gameplay)
                if len(SHARED.buf_states) > 0:
                    model.train()
                    _ppo_update_frozen_vit()
                    model.eval()
                    torch.save(model.state_dict(), model_path)
                
                # Update env_model with real experiences (real-time learning)
                # Both policy model (base) and env_model (wrapper) are updated
                if env_model is not None and env_optimizer is not None and len(experience_buffer) >= 64:
                    update_env_model(
                        env_model,
                        env_optimizer,
                        experience_buffer,
                        batch_size=64,
                        num_updates=5,  # 5 gradient steps per round
                        device=device,
                    )
                    # Save updated env_model periodically (includes policy base)
                    if total_rounds % 100 == 0:
                        env_model_path_updated = os.path.join(results_dir, 'env_model_updated.pt')
                        torch.save(env_model.state_dict(), env_model_path_updated)
                        # Also save the policy base model separately
                        env_policy_path = os.path.join(results_dir, 'env_policy_base.pt')
                        torch.save(env_model.policy_model.state_dict(), env_policy_path)
                
                # Planning updates (if enabled, using updated env_model)
                if planner is not None and config.get('use_planning', True) and config.get('planning_episodes', 0) > 0:
                    # Ensure env_model is in eval mode for planning
                    env_model.eval()
                    run_planning_updates(
                        model,
                        planner,
                        config['planning_episodes'],
                        config['planning_horizon'],
                        device,
                        model_path,
                    )
                    # Switch back to train mode for next update
                    env_model.train()
                
                # Track progress
                total_rounds += config['rounds_per_batch']
                won = batch_stats['team_a_score'] > batch_stats['team_b_score']
                stage_tracker.add_result(won)
                
                # Check stage advancement
                if stage_tracker.should_advance():
                    stage_tracker.advance_stage()
                
                # Log progress
                if total_rounds - last_log_round >= 50:
                    last_log_round = total_rounds
                    stage_name, _, _, _ = stage_tracker.get_stage_info()
                    wr = stage_tracker.get_win_rate()
                    
                    print(f"[Round {total_rounds}/{config['total_rounds']}]")
                    print(f"  {stage_name} | WR: {wr*100:.1f}%")
                    print(f"  Score={batch_stats['team_a_score']:.2f} Kills={batch_stats['kills']:.2f} Deaths={batch_stats['deaths']:.2f}")
                    print()
                
            except Exception as e:
                print(f"[Error] Round {total_rounds}: {e}")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Save final model
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved final model to {model_path}")
        print(f"Total training time: {(time.time() - start_time)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description='Single Agent + Planning Training')
    parser.add_argument('--total-rounds', type=int, default=50000, help='Total training rounds')
    parser.add_argument('--rounds-per-batch', type=int, default=5, help='Rounds per batch')
    parser.add_argument('--planning-episodes', type=int, default=5, help='Planning episodes per round')
    parser.add_argument('--planning-horizon', type=int, default=8, help='Planning rollout horizon')
    parser.add_argument('--env-model-path', type=str, default=None, help='Path to environment model')
    parser.add_argument('--initial-model-path', type=str, default=None, help='Path to initial model (phase2.pt)')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    
    args = parser.parse_args()
    
    # Create results directory
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"results/single_agent_{timestamp}"
    
    config = {
        'total_rounds': args.total_rounds,
        'rounds_per_batch': args.rounds_per_batch,
        'planning_episodes': args.planning_episodes,
        'planning_horizon': args.planning_horizon,
        'env_model_path': args.env_model_path,
        'use_planning': args.env_model_path is not None,
        'results_dir': args.results_dir,
        'initial_model_path': args.initial_model_path,
    }
    
    run_training(config)


if __name__ == '__main__':
    main()
