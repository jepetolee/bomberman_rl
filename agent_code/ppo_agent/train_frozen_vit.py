"""
PPO Training with Frozen ViT
============================

ViT 백본은 고정하고, Value Network와 TRM만 강화학습을 수행합니다.

사용법:
    환경 변수 설정:
    - BOMBER_FROZEN_VIT=1: ViT 고정 모드 활성화
    - PPO_MODEL_PATH=data/policy_models/policy_phase2.pt: 사전 학습 모델 경로
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .train import _compute_gae, SHARED


def _ppo_update_frozen_vit():
    """
    PPO 업데이트 (ViT 고정, Value Network와 TRM만 학습)
    
    핵심:
    - ViT 백본: requires_grad = False (고정)
    - Value Network: requires_grad = True (학습)
    - TRM: requires_grad = True (학습)
    - Policy Head: requires_grad = True (학습, TRM과 연결되어 있으므로)
    """
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
    
    # Check if using hybrid model
    from .models.vit_trm import PolicyValueViT_TRM_Hybrid
    is_hybrid = isinstance(SHARED.policy, PolicyValueViT_TRM_Hybrid)
    
    # Freeze ViT backbone, enable gradients for Value Network and TRM
    if is_hybrid:
        # Freeze ViT backbone
        for param in SHARED.policy.vit.parameters():
            param.requires_grad = False
        
        # Enable gradients for Value Network (hybrid model's own v_head)
        for param in SHARED.policy.v_head.parameters():
            param.requires_grad = True
        
        # Enable gradients for TRM
        for param in SHARED.policy.trm_patch_proj.parameters():
            param.requires_grad = True
        SHARED.policy.trm_pos_embed.requires_grad = True
        for param in SHARED.policy.trm_net.parameters():
            param.requires_grad = True
        
        # Enable gradients for Policy Head (TRM과 연결되어 있으므로)
        for param in SHARED.policy.pi_head.parameters():
            param.requires_grad = True
        
        print(f"[Frozen ViT] ViT 백본 고정, Value Network와 TRM만 학습")
    else:
        # Original TRM model - freeze patch embedding and ViT-like components if any
        # For now, assume all parameters are trainable except we'll handle it differently
        print(f"[Frozen ViT] Original TRM model - all parameters trainable")

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

            # Forward pass
            if is_hybrid:
                # Use TRM during RL (use_trm=True, detach_trm=False)
                logits, values_pred, _ = SHARED.policy.forward(
                    mb_states,
                    z_prev=None,  # Will be initialized to zero
                    use_trm=True,
                    detach_trm=False,  # TRM gradients enabled
                )
            else:
                # Original TRM model
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
            
            # Only clip gradients for trainable parameters
            if is_hybrid:
                # Collect trainable parameters (Value Network + TRM + Policy Head)
                trainable_params = []
                trainable_params.extend(list(SHARED.policy.v_head.parameters()))
                trainable_params.extend(list(SHARED.policy.trm_patch_proj.parameters()))
                trainable_params.append(SHARED.policy.trm_pos_embed)
                trainable_params.extend(list(SHARED.policy.trm_net.parameters()))
                trainable_params.extend(list(SHARED.policy.pi_head.parameters()))
            else:
                trainable_params = list(SHARED.policy.parameters())
            
            nn.utils.clip_grad_norm_(trainable_params, SHARED.max_grad_norm)
            SHARED.optimizer.step()
    
    # Re-enable gradients for ViT (in case needed elsewhere, though frozen during training)
    if is_hybrid:
        for param in SHARED.policy.vit.parameters():
            param.requires_grad = True  # Re-enable but won't be updated due to optimizer


def setup_frozen_vit_optimizer():
    """
    Optimizer 설정 (ViT 제외, Value Network와 TRM만)
    """
    from .models.vit_trm import PolicyValueViT_TRM_Hybrid
    
    if isinstance(SHARED.policy, PolicyValueViT_TRM_Hybrid):
        # Collect only trainable parameters
        trainable_params = []
        
        # Value Network
        trainable_params.extend(list(SHARED.policy.v_head.parameters()))
        
        # TRM components
        trainable_params.extend(list(SHARED.policy.trm_patch_proj.parameters()))
        trainable_params.append(SHARED.policy.trm_pos_embed)
        trainable_params.extend(list(SHARED.policy.trm_net.parameters()))
        
        # Policy Head (TRM과 연결되어 있으므로)
        trainable_params.extend(list(SHARED.policy.pi_head.parameters()))
        
        # Create optimizer with only trainable parameters
        SHARED.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=SHARED.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        print(f"[Frozen ViT Optimizer] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"[Frozen ViT Optimizer] Total parameters: {sum(p.numel() for p in SHARED.policy.parameters()):,}")
        print(f"[Frozen ViT Optimizer] Frozen (ViT): {sum(p.numel() for p in SHARED.policy.vit.parameters()):,}")
    else:
        # Original TRM model - use all parameters
        SHARED.optimizer = torch.optim.AdamW(
            SHARED.policy.parameters(),
            lr=SHARED.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

