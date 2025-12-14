from typing import List, Tuple
import os
import json
from pathlib import Path
import numpy as np
from collections import deque
import settings as s

import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import state_to_features, ACTIONS, SHARED

# Optional: Dyna-Q planning in Phase 3 (env model + simulated rollouts)
from .models.environment_model import create_env_model
from .dyna_planning import VisitedStatesBuffer


class PolicyAsEnvModel(nn.Module):
    """
    Wrapper to use policy model as environment model.
    Policy model is copied from teacher forcing model and both are updated.
    This is the "coach" that learns from the "player's" actions.
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
        # For EfficientGTrXL: use forward_features (CNN + GTrXL features)
        # For TRM Hybrid: use ViT features only (TRM not used in env_model)
        from .models.vit_trm import PolicyValueViT_TRM_Hybrid
        from .models.efficient_gtrxl import PolicyValueEfficientGTrXL
        
        if isinstance(self.policy_model, PolicyValueEfficientGTrXL):
            # EfficientGTrXL: use forward_features (CNN + GTrXL, all learned)
            features = self.policy_model.forward_features(state)  # [B, embed_dim]
        elif isinstance(self.policy_model, PolicyValueViT_TRM_Hybrid):
            # Hybrid model: ViT features만 사용 (TRM은 사용 안함)
            # ViT features (학습됨)
            features = self.policy_model.vit.forward_features(state)  # [B, embed_dim]
            # TRM은 사용하지 않음 (파라미터는 policy_phase2.pt에서 유지되지만 추론에 작용 안함)
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

def get_distance_to_nearest_enemy(game_state):
    if game_state is None:
        return float('inf')

    field = game_state['field']
    _, _, _, (sx, sy) = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']

    self_name = game_state['self'][0]
    self_team = self_name.split('_')[0] if self_name else ""
    
    enemy_positions = []
    for o_n, o_s, o_b, (ox, oy) in others:
        other_team = o_n.split('_')[0] if o_n else ""
        if other_team != self_team:
            enemy_positions.append((ox, oy))

    if not enemy_positions:
        return float('inf')

    dist_map = np.full((s.COLS, s.ROWS), float('inf'))
    queue = deque()

    for ex, ey in enemy_positions:
        dist_map[ex, ey] = 0
        queue.append((ex, ey, 0))

    bomb_locs = { (bx, by) for (bx, by), t in bombs }

    while queue:
        x, y, d = queue.popleft()

        if x == sx and y == sy:
            return d

        next_dist = d + 1
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                if dist_map[nx, ny] == float('inf') and field[nx, ny] == 0 and (nx, ny) not in bomb_locs:
                    dist_map[nx, ny] = next_dist
                    queue.append((nx, ny, next_dist))
    
    return float('inf')

def setup_training(self):
    # Share hyperparameters and optimizer across agent instances
    SHARED.gamma = 0.99
    SHARED.gae_lambda = 0.95
    SHARED.clip_range = 0.2
    SHARED.ent_coef = 0.3            # Higher entropy for more exploration (was 0.2)
    SHARED.vf_coef = 0.5
    SHARED.max_grad_norm = 0.5
    SHARED.update_epochs = 6         # More update epochs for better learning (was 4)
    SHARED.batch_size = 128          # Smaller batch for more frequent updates (was 256)
    SHARED.learning_rate = 5e-4      # Slightly higher learning rate (was 3e-4)

    if SHARED.optimizer is None:
        # Check if frozen ViT mode is enabled
        use_frozen_vit = os.environ.get("BOMBER_FROZEN_VIT", "0") == "1"
        
        if use_frozen_vit:
            # Setup optimizer for frozen ViT (ViT 제외, Value Network와 TRM만)
            from .train_frozen_vit import setup_frozen_vit_optimizer
            setup_frozen_vit_optimizer()
        else:
            # Original optimizer (all parameters)
            SHARED.optimizer = optim.Adam(SHARED.policy.parameters(), lr=SHARED.learning_rate)

    # ----------------------------
    # Phase 3 Dyna-Q planning setup (optional, controlled by env vars)
    # ----------------------------
    if not hasattr(SHARED, "_planning_initialized"):
        SHARED._planning_initialized = True
        SHARED.use_planning = os.environ.get("BOMBER_USE_PLANNING", "0") == "1"
        try:
            SHARED.planning_episodes_per_round = int(os.environ.get("BOMBER_PLANNING_EPISODES", "0"))
        except Exception:
            SHARED.planning_episodes_per_round = 0
        try:
            SHARED.planning_horizon = int(os.environ.get("BOMBER_PLANNING_HORIZON", "8"))
        except Exception:
            SHARED.planning_horizon = 8
        try:
            SHARED.planning_min_buffer = int(os.environ.get("BOMBER_PLANNING_MIN_BUFFER", "200"))
        except Exception:
            SHARED.planning_min_buffer = 200

        SHARED.env_model = None
        SHARED.env_optimizer = None
        SHARED.visited_states = None
        SHARED.env_experience_buffer = []  # Store (state, action, reward, next_state) for env_model training
        SHARED.max_env_buffer_size = 10000  # Keep last N experiences

        if SHARED.use_planning:
            # Create env_model by copying policy model (teacher forcing model)
            # This is the "coach" that learns from the "player's" actions
            # Env model: ViT + Policy head + Value head 학습, TRM은 frozen
            import copy
            try:
                # Deep copy policy model as base for env_model
                env_policy_base = copy.deepcopy(SHARED.policy)
                env_policy_base.train()  # Start in train mode for updates
                
                # Wrap policy model as environment model
                SHARED.env_model = PolicyAsEnvModel(env_policy_base).to(SHARED.device)
                SHARED.env_model.train()  # Start in train mode for updates
                
                # Setup env_model optimizer
                from .models.vit_trm import PolicyValueViT_TRM_Hybrid
                from .models.efficient_gtrxl import PolicyValueEfficientGTrXL
                
                if isinstance(env_policy_base, PolicyValueEfficientGTrXL):
                    # EfficientGTrXL: 모든 파라미터 학습 (CNN backbone + GTrXL + heads)
                    # Env model heads만 추가로 학습
                    env_trainable_params = list(SHARED.env_model.policy_model.parameters())
                    env_trainable_params.extend(list(SHARED.env_model.next_state_head.parameters()))
                    env_trainable_params.extend(list(SHARED.env_model.reward_head.parameters()))
                    if hasattr(SHARED.env_model, 'action_embed'):
                        env_trainable_params.extend(list(SHARED.env_model.action_embed.parameters()))
                    SHARED.env_optimizer = optim.Adam(env_trainable_params, lr=1e-3)
                    
                    try:
                        self.logger.info(
                            f"[Env Model] EfficientGTrXL: All parameters trainable "
                            f"({sum(p.numel() for p in env_trainable_params):,} params)"
                        )
                    except Exception:
                        pass
                elif isinstance(env_policy_base, PolicyValueViT_TRM_Hybrid):
                    # Collect trainable parameters for env_model
                    env_trainable_params = []
                    
                    # ViT 백본 학습
                    env_trainable_params.extend(list(SHARED.env_model.policy_model.vit.parameters()))
                    
                    # Policy head 학습
                    env_trainable_params.extend(list(SHARED.env_model.policy_model.pi_head.parameters()))
                    
                    # Value head 학습
                    env_trainable_params.extend(list(SHARED.env_model.policy_model.v_head.parameters()))
                    
                    # Env model's own heads 학습
                    env_trainable_params.extend(list(SHARED.env_model.next_state_head.parameters()))
                    env_trainable_params.extend(list(SHARED.env_model.reward_head.parameters()))
                    if hasattr(SHARED.env_model, 'action_embed'):
                        env_trainable_params.extend(list(SHARED.env_model.action_embed.parameters()))
                    
                    # TRM은 frozen (학습 안함)
                    for param in SHARED.env_model.policy_model.trm_net.parameters():
                        param.requires_grad = False
                    
                    # Create optimizer with only trainable parameters
                    SHARED.env_optimizer = optim.Adam(env_trainable_params, lr=1e-3)
                    
                    try:
                        self.logger.info(
                            f"[Env Model] Trainable: ViT + Policy head + Value head + Env heads "
                            f"({sum(p.numel() for p in env_trainable_params):,} params)"
                        )
                        self.logger.info(
                            f"[Env Model] Frozen: TRM "
                            f"({sum(p.numel() for p in SHARED.env_model.policy_model.trm_net.parameters()):,} params)"
                        )
                    except Exception:
                        pass
                else:
                    # Fallback: use all parameters
                    SHARED.env_optimizer = optim.Adam(SHARED.env_model.parameters(), lr=1e-3)
                
                SHARED.visited_states = VisitedStatesBuffer(max_size=10000)
                
                try:
                    self.logger.info(
                        f"[Planning] Enabled: episodes_per_round={SHARED.planning_episodes_per_round}, "
                        f"horizon={SHARED.planning_horizon}, env_model=copied from policy"
                    )
                    self.logger.info(
                        f"[Planning] Policy: TRM만 학습 | Env Model: ViT+Heads만 학습 (TRM frozen)"
                    )
                except Exception:
                    pass
            except Exception as e:
                try:
                    self.logger.warning(f"[Planning] Failed to create env_model from policy: {e}")
                    import traceback
                    traceback.print_exc()
                except Exception:
                    pass
                SHARED.use_planning = False


def _maybe_add_to_visited(state_tensor_cpu: torch.Tensor, action_idx: int):
    """Record real (state, action) into visited buffer for planning."""
    try:
        if getattr(SHARED, "use_planning", False) and getattr(SHARED, "visited_states", None) is not None:
            # state_tensor_cpu: [C,H,W] on CPU
            SHARED.visited_states.add(state_tensor_cpu, int(action_idx))
    except Exception:
        pass


def _update_env_model():
    """
    Update environment model using collected real experiences.
    This is the "coach" learning from the "player's" actions.
    
    Env model은 스스로 업데이트됨 (자체 optimizer 사용).
    
    Env model 학습 범위:
    - ViT 백본: 학습 ✓
    - Policy head: 학습 ✓
    - Value head: 학습 ✓
    - Env heads (next_state_head, reward_head): 학습 ✓
    - TRM: frozen ✗ (절대 학습 안함, 추론에도 사용 안함)
    
    Note: TRM 파라미터는 policy_phase2.pt에서 유지되지만, 
          학습과 추론 모두에 작용하지 않음 (ViT features만 사용).
    """
    if not getattr(SHARED, "use_planning", False):
        return
    if SHARED.env_model is None or SHARED.env_optimizer is None:
        return
    if len(SHARED.env_experience_buffer) < 64:
        return
    
    import random
    SHARED.env_model.train()
    
    # Ensure TRM is frozen during env_model training (for TRM Hybrid only)
    # EfficientGTrXL은 모든 파라미터가 학습됨
    from .models.vit_trm import PolicyValueViT_TRM_Hybrid
    if isinstance(SHARED.env_model.policy_model, PolicyValueViT_TRM_Hybrid):
        # Freeze TRM (절대 학습 안함)
        for param in SHARED.env_model.policy_model.trm_net.parameters():
            param.requires_grad = False
    
    criterion_state = nn.MSELoss()
    criterion_reward = nn.MSELoss()
    
    # Sample random batches and update
    for _ in range(5):  # 5 gradient steps per round
        batch = random.sample(SHARED.env_experience_buffer, min(64, len(SHARED.env_experience_buffer)))
        
        states = torch.stack([s for s, _, _, _ in batch]).to(SHARED.device)  # [B, C, H, W]
        actions = torch.tensor([a for _, a, _, _ in batch], dtype=torch.long, device=SHARED.device)  # [B]
        rewards = torch.tensor([r for _, _, r, _ in batch], dtype=torch.float32, device=SHARED.device)  # [B]
        next_states = torch.stack([ns for _, _, _, ns in batch]).to(SHARED.device)  # [B, C, H, W]
        
        # Forward pass (TRM은 사용 안함, ViT features만 사용)
        # TRM 파라미터는 policy_phase2.pt에서 유지되지만 추론에 작용 안함
        next_state_pred, reward_pred = SHARED.env_model(states, actions)
        
        # Losses
        state_loss = criterion_state(next_state_pred, next_states)
        reward_loss = criterion_reward(reward_pred, rewards)
        loss = state_loss + reward_loss
        
        # Backward (Env 모델이 스스로 업데이트)
        SHARED.env_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients only for trainable parameters
        trainable_params = []
        from .models.vit_trm import PolicyValueViT_TRM_Hybrid
        from .models.efficient_gtrxl import PolicyValueEfficientGTrXL
        if isinstance(SHARED.env_model.policy_model, PolicyValueEfficientGTrXL):
            # EfficientGTrXL: 모든 파라미터 학습 가능
            trainable_params.extend(list(SHARED.env_model.policy_model.parameters()))
        elif isinstance(SHARED.env_model.policy_model, PolicyValueViT_TRM_Hybrid):
            # TRM Hybrid: ViT + heads만 학습 (TRM은 frozen)
            trainable_params.extend(list(SHARED.env_model.policy_model.vit.parameters()))
            trainable_params.extend(list(SHARED.env_model.policy_model.pi_head.parameters()))
            trainable_params.extend(list(SHARED.env_model.policy_model.v_head.parameters()))
        trainable_params.extend(list(SHARED.env_model.next_state_head.parameters()))
        trainable_params.extend(list(SHARED.env_model.reward_head.parameters()))
        if hasattr(SHARED.env_model, 'action_embed'):
            trainable_params.extend(list(SHARED.env_model.action_embed.parameters()))
        
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        SHARED.env_optimizer.step()
    
    SHARED.env_model.eval()  # Switch back to eval for planning


def _run_planning_updates(logger=None):
    """
    After a real round update, run N simulated planning rollouts with Deep Supervision.
    
    Deep Supervision in Planning:
    - For each planning step, apply multiple recursive reasoning steps (like TRM's n_sup)
    - Model improves its prediction through recursive reasoning before making decisions
    - Similar to DeepSupervision training but applied during planning rollouts
    
    Non-recurrent mode: z is always re-initialized to zero (z_prev=None) each step.
    """
    if not getattr(SHARED, "use_planning", False):
        return
    if SHARED.env_model is None or SHARED.visited_states is None:
        return
    if SHARED.planning_episodes_per_round <= 0:
        return
    if len(SHARED.visited_states) < SHARED.planning_min_buffer:
        return

    device = SHARED.device
    model = SHARED.policy
    env_model = SHARED.env_model
    
    # Get n_sup for Deep Supervision (ViT Only and EfficientGTrXL models do not use DeepSupervision)
    # Use model's n_sup if available, otherwise default to 4
    from .models.vit_trm import PolicyValueViT_TRM_Hybrid
    from .models.vit import PolicyValueViT
    from .models.efficient_gtrxl import PolicyValueEfficientGTrXL
    if isinstance(model, PolicyValueViT) or isinstance(model, PolicyValueEfficientGTrXL):
        # ViT Only or EfficientGTrXL: no DeepSupervision in planning (use standard forward)
        use_planning_deepsup = False
        planning_n_sup = 1
    elif isinstance(model, PolicyValueViT_TRM_Hybrid):
        planning_n_sup = getattr(model, 'trm_n_sup', 4)  # Use TRM's n_sup for deep supervision
        use_planning_deepsup = True
    else:
        planning_n_sup = 4  # Default (for TRM models)
        use_planning_deepsup = True

    for ep in range(SHARED.planning_episodes_per_round):
        # Build a simulated rollout buffers (then run PPO once)
        sim_states = []
        sim_actions = []
        sim_logps = []
        sim_values = []
        sim_rewards = []
        sim_dones = []

        # Sample a start state from visited buffer
        try:
            start_state, _ = SHARED.visited_states.sample_state()
        except Exception:
            return

        state = start_state.unsqueeze(0).to(device)  # [1,C,H,W]

        model.eval()
        for t in range(max(1, SHARED.planning_horizon)):
            with torch.no_grad():
                # Deep Supervision in Planning (only for TRM models)
                if use_planning_deepsup:
                    # Apply Deep Supervision: multiple recursive steps to improve prediction
                    z_prev = None  # Always start from zero (non-recurrent)
                    for sup_step in range(planning_n_sup):
                        # Policy inference with recursive reasoning (TRM)
                        logits, value, z_new = model.forward_with_z(state, z_prev=z_prev)
                        # Use improved z for next supervision step
                        z_prev = z_new.detach()  # Detach to prevent long BPTT chains
                else:
                    # ViT Only or EfficientGTrXL: standard forward pass (no recursion)
                    # EfficientGTrXL supports memory but we don't use it in planning (simpler)
                    if isinstance(model, PolicyValueEfficientGTrXL):
                        logits, value = model(state, memory=None)
                    else:
                        logits, value = model(state)
                
                # Final prediction
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                # World model step (Env model uses ViT only, no TRM)
                next_state_pred, reward_pred = env_model(state, action)

            # Store transition (detach world model outputs)
            sim_states.append(state.squeeze(0).detach().cpu())
            sim_actions.append(int(action.item()))
            sim_logps.append(float(logp.item()))
            sim_values.append(float(value.squeeze(0).item()))
            sim_rewards.append(float(reward_pred.squeeze(0).item()))
            sim_dones.append(False)

            # Advance
            state = next_state_pred.detach()

        # Mark terminal at end of simulated rollout
        sim_dones[-1] = True

        # Swap buffers into SHARED and run PPO update with DeepSupervision
        # Planning updates use DeepSupervision: each (state, action) pair goes through
        # n_sup recursive reasoning steps, with loss computed at each step
        SHARED.buf_states = sim_states
        SHARED.buf_actions = sim_actions
        SHARED.buf_logps = sim_logps
        SHARED.buf_values = sim_values
        SHARED.buf_rewards = sim_rewards
        SHARED.buf_dones = sim_dones
        # No recurrent z buffer in non-recurrent mode
        if hasattr(SHARED, "buf_z_prev"):
            SHARED.buf_z_prev = []

        # Use DeepSupervision for Planning updates (only for TRM models, not ViT Only or EfficientGTrXL)
        # ViT Only and EfficientGTrXL models do not use DeepSupervision
        from .models.vit import PolicyValueViT
        from .models.efficient_gtrxl import PolicyValueEfficientGTrXL
        use_planning_deepsup = not (isinstance(model, PolicyValueViT) or isinstance(model, PolicyValueEfficientGTrXL))
        _ppo_update(use_deep_supervision=use_planning_deepsup)

        # Save updated weights immediately so A3C worker can pick up latest
        try:
            torch.save(SHARED.policy.state_dict(), SHARED.model_path)
        except Exception:
            pass

    try:
        if logger is not None:
            logger.debug(
                f"[Planning] Completed {SHARED.planning_episodes_per_round} planning PPO updates "
                f"with Deep Supervision (n_sup={planning_n_sup})"
            )
    except Exception:
        pass


def _reward_from_events(self, events: List[str]) -> float:
    # Focus on winning, reduce suicide by lowering bomb reward
    game_rewards = {
        # Primary objectives - VERY HIGH rewards for kills
        e.KILLED_OPPONENT: 50.0,      # MASSIVE reward for kills! (was 10.0)
        e.COIN_COLLECTED: 1.0,
        e.CRATE_DESTROYED: 0.3,       # Encourage destruction
        e.SURVIVED_ROUND: 2.0,        # Increased survival reward (was 1.0)
        
        # Reduced bomb reward to prevent reckless bombing/self-kills
        e.BOMB_DROPPED: 0.5,          # Reduced from 5.0 to discourage risky bombs
        e.COIN_FOUND: 0.2,            # 적의 코인 발견 시 더 높은 보상 (was 0.1)
        
        # Stronger penalties for death/suicide (MUCH STRONGER)
        e.KILLED_TEAMMATE: -30.0,
        e.KILLED_SELF: -40.0,         # MUCH STRONGER penalty to strongly discourage suicide (was -20.0)
        e.GOT_KILLED: -30.0,          # MUCH STRONGER penalty to emphasize survival (was -15.0)
        e.INVALID_ACTION: -0.1,       # Stronger penalty for invalid
        
        # Movement - encourage activity
        e.WAITED: -0.2,               # Stronger penalty for waiting (was -0.1)
        e.MOVED_UP: 0.05,             # More reward for movement (was 0.02)
        e.MOVED_DOWN: 0.05,
        e.MOVED_LEFT: 0.05,
        e.MOVED_RIGHT: 0.05,
    }
    # Configure suicide penalty schedule
    try:
        completed = getattr(SHARED, 'completed_rounds', 0)
        suppress_until = getattr(SHARED, 'suppress_suicide_until', 0)
        switch_round = getattr(SHARED, 'suicide_penalty_switch_round', 800)
        high_penalty = getattr(SHARED, 'suicide_penalty_high', -15.0)
        # Default 'after' penalty equals negative magnitude of opponent kill reward
        default_after = abs(game_rewards[e.KILLED_OPPONENT])
        after_penalty = getattr(SHARED, 'suicide_penalty_after', default_after)
        if not (after_penalty == after_penalty):  # NaN check
            after_penalty = default_after

        if completed < suppress_until:
            suicide_penalty = 0.0
        else:
            suicide_penalty = high_penalty if completed < switch_round else after_penalty
        game_rewards[e.KILLED_SELF] = float(suicide_penalty)
    except Exception:
        pass
    reward = 0.0
    for ev in events:
        reward += game_rewards.get(ev, 0.0)
    return float(reward)


def _get_distance_to_enemy_coins(game_state: dict) -> float:
    """적의 코인까지의 최단 거리 반환 (BFS 기반)"""
    if game_state is None:
        return float('inf')
    
    field = game_state['field']
    coins = game_state['coins']
    self_info = game_state['self']
    others = game_state['others']
    
    self_name, _, _, (sx, sy) = self_info
    self_tag = self_name.split('_')[0] if self_name else ""
    
    # 적 위치 찾기
    enemy_positions = []
    for other_name, _, _, (ox, oy) in others:
        tag = other_name.split('_')[0] if other_name else ""
        if tag != self_tag:
            enemy_positions.append((ox, oy))
    
    # 적의 코인 찾기
    enemy_coins = []
    for cx, cy in coins:
        if len(enemy_positions) > 0:
            min_enemy_dist = min([abs(cx - ex) + abs(cy - ey) for ex, ey in enemy_positions])
            if min_enemy_dist < 5:  # 적으로부터 5칸 이내
                enemy_coins.append((cx, cy))
        else:
            enemy_coins.append((cx, cy))
    
    if len(enemy_coins) == 0:
        return float('inf')
    
    # BFS로 최단 거리 계산
    H, W = field.shape
    queue = deque([(sx, sy)])
    visited = {(sx, sy)}
    dist = {(sx, sy): 0}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        current_dist = dist[(x, y)]
        
        # 목표 코인에 도달했는지 확인
        if (x, y) in enemy_coins:
            return float(current_dist)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            
            if (nx, ny) in visited:
                continue
            
            if field[nx, ny] != 0:  # 벽이나 크레이트
                continue
            
            visited.add((nx, ny))
            dist[(nx, ny)] = current_dist + 1
            queue.append((nx, ny))
    
    # 도달 불가능
    return float('inf')


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # We rely on values cached in callbacks.act()
    if self._last_state_tensor is None or self._last_action_idx is None:
        return

    reward = _reward_from_events(self, events)

    if old_game_state is not None and new_game_state is not None:
        dist_old = get_distance_to_nearest_enemy(old_game_state)
        dist_new = get_distance_to_nearest_enemy(new_game_state)
        
        if dist_old != float('inf') and dist_new != float('inf'):
            if dist_new < dist_old:
                shaping_reward = 0.5 # Approaching reward
                reward += shaping_reward
                self.logger.debug(f"Approaching enemy: +{shaping_reward} (dist {dist_old} -> {dist_new})")
            
            elif dist_new > dist_old:
                shaping_penalty = -0.2 # Retreating reward
                reward += shaping_penalty
                self.logger.debug(f"Retreating: {shaping_penalty} (dist {dist_old} -> {dist_new})")
        
        # 적의 코인까지 거리 기반 shaping reward
        coin_dist_old = _get_distance_to_enemy_coins(old_game_state)
        coin_dist_new = _get_distance_to_enemy_coins(new_game_state)
        
        if coin_dist_old != float('inf') and coin_dist_new != float('inf'):
            if coin_dist_new < coin_dist_old:
                # 거리가 가까워질수록 더 높은 보상 (역수 관계)
                coin_shaping_reward = 0.1 * (1.0 / (coin_dist_new + 1.0))  # 거리 기반 보상
                reward += coin_shaping_reward
                self.logger.debug(f"Approaching enemy coin: +{coin_shaping_reward:.3f} (dist {coin_dist_old:.1f} -> {coin_dist_new:.1f})")
            
            elif coin_dist_new > coin_dist_old:
                # 거리가 멀어지면 작은 페널티
                coin_shaping_penalty = -0.05
                reward += coin_shaping_penalty
                self.logger.debug(f"Retreating from enemy coin: {coin_shaping_penalty} (dist {coin_dist_old:.1f} -> {coin_dist_new:.1f})")

    done = False  # terminal handled in end_of_round

    # Store step into shared buffers
    SHARED.buf_states.append(self._last_state_tensor.squeeze(0))  # [C,H,W]
    SHARED.buf_actions.append(self._last_action_idx)
    SHARED.buf_logps.append(self._last_log_prob)
    SHARED.buf_values.append(self._last_value)
    SHARED.buf_rewards.append(reward)
    SHARED.buf_dones.append(done)
    # Optional recurrent z_prev (for recurrent PPO / planning-consistent PPO)
    if hasattr(SHARED, "buf_z_prev"):
        SHARED.buf_z_prev.append(getattr(self, "_last_z_prev", None))

    # Populate visited buffer for planning
    _maybe_add_to_visited(self._last_state_tensor.squeeze(0), self._last_action_idx)
    
    # Collect experience for env_model training: (state, action, reward, next_state)
    # This is the "coach" observing and learning from the "player's" actions
    if getattr(SHARED, "use_planning", False) and len(SHARED.buf_states) > 1:
        try:
            # Extract transitions: (state[i], action[i], reward[i], state[i+1])
            prev_state = SHARED.buf_states[-2]  # Previous state
            prev_action = SHARED.buf_actions[-2]  # Previous action
            prev_reward = SHARED.buf_rewards[-2]  # Reward for that action
            current_state = SHARED.buf_states[-1]  # Next state (result of that action)
            
            # Add to experience buffer
            SHARED.env_experience_buffer.append((prev_state, prev_action, prev_reward, current_state))
            
            # Keep buffer size manageable
            if len(SHARED.env_experience_buffer) > SHARED.max_env_buffer_size:
                SHARED.env_experience_buffer = SHARED.env_experience_buffer[-SHARED.max_env_buffer_size:]
        except Exception:
            pass  # Silently ignore errors in experience collection


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Final reward for last step
    final_reward = _reward_from_events(self, events)
    
    # Add win/loss penalty based on final score comparison
    # This encourages winning and penalizes losing
    if last_game_state is not None:
        try:
            # Get our team's total score
            self_name, self_score, _, _ = last_game_state.get('self', (None, 0, 0, (0, 0)))
            others_data = last_game_state.get('others', [])
            
            # Calculate team scores
            self_tag = str(self_name).split('_')[0] if self_name else ""
            our_team_score = self_score
            enemy_team_score = 0
            
            for other_name, other_score, _, _ in others_data:
                other_tag = str(other_name).split('_')[0] if other_name else ""
                if other_tag == self_tag:
                    our_team_score += other_score
                else:
                    enemy_team_score += other_score
            
            # Add win/loss reward - MUCH HIGHER to emphasize winning
            if our_team_score > enemy_team_score:
                # Win bonus - significantly increased to prioritize winning
                final_reward += 100.0  # Increased from 50.0 to strongly reward winning
            elif our_team_score < enemy_team_score:
                # Loss penalty - significantly increased to discourage losing
                final_reward -= 50.0  # Keep loss penalty at -50.0
            # Draw: no extra reward/penalty
        except Exception:
            pass  # Silently ignore errors in score calculation
    
    if SHARED.buf_rewards:
        # Fold terminal bonus/penalty into the final transition instead of creating a duplicate entry
        SHARED.buf_rewards[-1] += final_reward
        SHARED.buf_dones[-1] = True
    elif self._last_state_tensor is not None and self._last_action_idx is not None:
        # Fallback for edge cases where nothing was buffered (e.g., agent never acted)
        SHARED.buf_states.append(self._last_state_tensor.squeeze(0))
        SHARED.buf_actions.append(self._last_action_idx)
        SHARED.buf_logps.append(self._last_log_prob)
        SHARED.buf_values.append(self._last_value)
        SHARED.buf_rewards.append(final_reward)
        SHARED.buf_dones.append(True)

    # Coordinate update: run once when all our shared instances have finished this round
    round_id = last_game_state['round'] if last_game_state is not None else None
    if round_id is not None:
        count = SHARED.round_done_counts.get(round_id, 0) + 1
        SHARED.round_done_counts[round_id] = count
        if count >= len(SHARED.instance_ids):
            if len(SHARED.buf_states) > 1:
                _ppo_update()
                try:
                    torch.save(SHARED.policy.state_dict(), SHARED.model_path)
                    # Optional periodic checkpoints
                    SHARED.save_round_counter += 1
                    if getattr(SHARED, 'save_every_rounds', 0) and SHARED.save_round_counter % max(1, SHARED.save_every_rounds) == 0:
                        ckpt_dir = SHARED.checkpoint_dir or os.path.join(os.path.dirname(SHARED.model_path) or '.', 'checkpoints')
                        try:
                            os.makedirs(ckpt_dir, exist_ok=True)
                        except Exception:
                            pass
                        ckpt_path = os.path.join(ckpt_dir, f"ckpt_round_{SHARED.save_round_counter:06d}.pt")
                        torch.save(SHARED.policy.state_dict(), ckpt_path)
                        try:
                            self.logger.info(f"Saved PPO checkpoint: {ckpt_path}")
                        except Exception:
                            pass
                except Exception as ex:
                    self.logger.warning(f"Failed to save PPO model: {ex}")

                # Update env_model with real experiences (coach learns from player)
                try:
                    _update_env_model()
                except Exception as ex:
                    try:
                        self.logger.warning(f"[Env Model] Update failed: {ex}")
                    except Exception:
                        pass
                
                # After real update and env_model update, run Dyna-Q planning updates (Phase 3)
                try:
                    _run_planning_updates(self.logger)
                except Exception as ex:
                    try:
                        self.logger.warning(f"[Planning] Failed: {ex}")
                    except Exception:
                        pass
            # One full round completed across all our instances
            try:
                SHARED.completed_rounds += 1
                # Calculate teacher stats every round if we have data (for faster feedback)
                if SHARED.total_action_count > 0:
                    teacher_usage = SHARED.teacher_action_count / SHARED.total_action_count * 100
                    current_eps = SHARED.current_epsilon()
                    invalid_rate = SHARED.teacher_invalid_count / max(1, SHARED.teacher_action_count + SHARED.teacher_invalid_count) * 100
                    # Store in shared state
                    SHARED.last_teacher_stats = {
                        'round': round_id,
                        'usage': teacher_usage,
                        'epsilon': current_eps,
                        'invalid_rate': invalid_rate,
                        'total_actions': SHARED.total_action_count,
                        'teacher_actions': SHARED.teacher_action_count
                    }
                    # Save to file every round (so worker can always read latest)
                    # Save to ONE location only: A3C results_dir if available, otherwise model directory
                    try:
                        # Priority 1: A3C results directory (set by worker) - MUST be absolute path
                        stats_file = None
                        a3c_results_dir = os.environ.get('A3C_RESULTS_DIR', None)
                        if a3c_results_dir:
                            # Convert to absolute path to avoid working directory issues
                            a3c_results_dir = os.path.abspath(a3c_results_dir)
                            stats_file = os.path.join(a3c_results_dir, 'teacher_stats.json')
                        else:
                            # Priority 2: Model directory
                            stats_file = SHARED._stats_file or os.path.join(
                                os.path.dirname(SHARED.model_path) if os.path.dirname(SHARED.model_path) else '.',
                                'teacher_stats.json'
                            )
                            # Also convert to absolute path
                            stats_file = os.path.abspath(stats_file)
                        
                        # Ensure directory exists
                        stats_dir = os.path.dirname(stats_file) or '.'
                        os.makedirs(stats_dir, exist_ok=True)
                        with open(stats_file, 'w') as f:
                            json.dump(SHARED.last_teacher_stats, f)
                        
                        # Log first time to confirm it's working
                        if SHARED.completed_rounds == 1:
                            try:
                                self.logger.info(f"📚 Teacher stats tracking started: {teacher_usage:.1f}% (eps={current_eps:.3f}, total={SHARED.total_action_count})")
                                self.logger.info(f"📚 Stats saved to: {stats_file}")
                            except:
                                pass
                    except Exception as e:
                        try:
                            if SHARED.completed_rounds <= 10:  # Log first few errors
                                self.logger.warning(f"Failed to save teacher stats: {e}")
                        except:
                            pass
            except Exception:
                pass
            # Clear for next round
            SHARED.reset_buffers()
            # Cleanup count to avoid growth
            try:
                del SHARED.round_done_counts[round_id]
            except KeyError:
                pass


def _compute_gae(rewards, values, dones, *, gamma: float, gae_lambda: float):
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if (t == T - 1) else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return torch.tensor(adv, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def _ppo_update(use_deep_supervision=False):
    """
    PPO 업데이트 (모든 파라미터 학습 또는 Frozen ViT 모드)
    
    Args:
        use_deep_supervision: If True, apply DeepSupervision (multiple recursive reasoning steps)
                             for each (state, action) pair. Used in Planning updates.
    """
    # Check if frozen ViT mode is enabled
    use_frozen_vit = os.environ.get("BOMBER_FROZEN_VIT", "0") == "1"
    
    if use_frozen_vit:
        # Use frozen ViT update (ViT 고정, Value Network와 TRM만 학습)
        from .train_frozen_vit import _ppo_update_frozen_vit
        _ppo_update_frozen_vit(use_deep_supervision=use_deep_supervision)
        return
    
    # Original PPO update (all parameters trainable)
    SHARED.policy.train()

    device = SHARED.device

    states = torch.stack(SHARED.buf_states).to(device)           # [T, C, H, W]
    actions = torch.tensor(SHARED.buf_actions, dtype=torch.long, device=device)  # [T]
    old_logps = torch.tensor(SHARED.buf_logps, dtype=torch.float32, device=device)  # [T]
    values = torch.tensor(SHARED.buf_values, dtype=torch.float32, device=device)    # [T]
    rewards = torch.tensor(SHARED.buf_rewards, dtype=torch.float32, device=device)  # [T]
    dones = torch.tensor(SHARED.buf_dones, dtype=torch.bool, device=device)         # [T]

    adv, returns = _compute_gae(
        rewards.tolist(), values.tolist(), dones.tolist(),
        gamma=SHARED.gamma, gae_lambda=SHARED.gae_lambda
    )
    adv = adv.to(device)
    returns = returns.to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    T = states.size(0)
    
    # Check model types for DeepSupervision support
    from .models.vit_trm import PolicyValueViT_TRM, PolicyValueViT_TRM_Hybrid
    from .models.vit import PolicyValueViT
    from .models.efficient_gtrxl import PolicyValueEfficientGTrXL
    is_vit_only = isinstance(SHARED.policy, PolicyValueViT)
    is_efficient_gtrxl = isinstance(SHARED.policy, PolicyValueEfficientGTrXL)
    is_trm_hybrid = isinstance(SHARED.policy, PolicyValueViT_TRM_Hybrid)
    is_trm = isinstance(SHARED.policy, PolicyValueViT_TRM)
    use_recurrent_trm = is_trm and os.environ.get("BOMBER_TRM_RECURRENT", "0") == "1"
    
    # Get n_sup for DeepSupervision (if applicable - ViT Only and EfficientGTrXL do not use DeepSupervision)
    n_sup = 4  # Default
    if use_deep_supervision and not (is_vit_only or is_efficient_gtrxl):
        if is_trm_hybrid:
            n_sup = getattr(SHARED.policy, 'trm_n_sup', 4)
        elif is_trm:
            n_sup = getattr(SHARED.policy, 'n_sup', 4)

    for _ in range(SHARED.update_epochs):
        perm = torch.randperm(T)
        for start in range(0, T, SHARED.batch_size):
            mb_idx = perm[start:start + SHARED.batch_size]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logps = old_logps[mb_idx]
            mb_adv = adv[mb_idx]
            mb_returns = returns[mb_idx]
            mb_dones = dones[mb_idx]

            if use_deep_supervision and (is_trm_hybrid or is_trm) and not (is_vit_only or is_efficient_gtrxl):
                # DeepSupervision: Apply multiple recursive reasoning steps
                # Each supervision step contributes to learning
                step_policy_losses = []
                step_value_losses = []
                
                # Initialize z for each sample (zeros)
                z_prev = None  # Always zero init in non-recurrent mode
                
                # Apply DeepSupervision: n_sup recursive reasoning steps
                for sup_step in range(n_sup):
                    if is_trm_hybrid:
                        # Hybrid model: use forward_with_z
                        logits, values_pred, z_new = SHARED.policy.forward_with_z(
                            mb_states,
                            z_prev=z_prev,
                        )
                    elif is_trm:
                        # Original TRM model: use forward_with_z
                        logits, values_pred, z_new = SHARED.policy.forward_with_z(
                            mb_states,
                            z_prev=z_prev,
                        )
                    else:
                        # Fallback to standard forward
                        logits, values_pred = SHARED.policy(mb_states)
                        z_new = None
                    
                    # Compute loss at each supervision step
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logps = dist.log_prob(mb_actions)
                    
                    ratio = torch.exp(new_logps - mb_old_logps)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - SHARED.clip_range, 1.0 + SHARED.clip_range) * mb_adv
                    step_policy_loss = -torch.min(surr1, surr2).mean()
                    step_policy_losses.append(step_policy_loss)
                    
                    step_value_loss = nn.functional.mse_loss(values_pred, mb_returns)
                    step_value_losses.append(step_value_loss)
                    
                    # Detach z to avoid very long BPTT chains (per-step supervision)
                    if z_new is not None:
                        z_prev = z_new.detach()
                    else:
                        break  # Break if model doesn't support recursive reasoning
                
                # Average losses across all supervision steps
                policy_loss = torch.stack(step_policy_losses).mean()
                value_loss = torch.stack(step_value_losses).mean()
                
                # Entropy (use final prediction)
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()
                
            else:
                # Standard PPO update (no DeepSupervision for ViT Only/EfficientGTrXL, or no DeepSupervision requested)
                # Handle ViT Only, EfficientGTrXL, Hybrid, and original TRM models
                if is_vit_only or is_efficient_gtrxl:
                    # ViT Only or EfficientGTrXL model: standard forward pass
                    if is_efficient_gtrxl:
                        logits, values_pred = SHARED.policy(mb_states, memory=None)
                    else:
                        logits, values_pred = SHARED.policy(mb_states)
                elif is_trm_hybrid:
                    # Hybrid model returns (logits, value, trm_feat)
                    logits, values_pred, _ = SHARED.policy.forward(
                        mb_states,
                        z_prev=None,  # Always zero init in non-recurrent mode
                        use_trm=True,
                        detach_trm=False,  # Enable TRM gradients
                    )
                elif use_recurrent_trm or is_trm:
                    # Original TRM model: use forward_with_z or standard forward
                    if hasattr(SHARED.policy, 'forward_with_z'):
                        logits, values_pred, _ = SHARED.policy.forward_with_z(mb_states, z_prev=None)
                    else:
                        logits, values_pred = SHARED.policy(mb_states)
                else:
                    # Standard ViT model
                    logits, values_pred = SHARED.policy(mb_states)
                
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - SHARED.clip_range, 1.0 + SHARED.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values_pred, mb_returns)

            loss = policy_loss + SHARED.vf_coef * value_loss - SHARED.ent_coef * entropy

            SHARED.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(SHARED.policy.parameters(), SHARED.max_grad_norm)
            SHARED.optimizer.step()

