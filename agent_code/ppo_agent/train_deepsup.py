"""
DeepSupervision Training for TRM Policy and Value Networks
==========================================================

Trains both Policy and Value Networks using DeepSupervision: same state is processed
N_sup times to improve reasoning before computing policy and value losses.

- Policy Network: learns from teacher actions
- Value Network: learns from teacher rewards (if provided)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

from .models.vit_trm import PolicyValueViT_TRM


def train_policy_deepsup(
    model: PolicyValueViT_TRM,
    states: torch.Tensor,
    actions: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_sup: int = 16,
    strategy: str = "last",  # "last", "all", "weighted"
    rewards: Optional[torch.Tensor] = None,  # Optional rewards for value supervision
    train_value: bool = True,  # Whether to train value network with rewards
    value_weight: float = 0.5,  # Weight for value loss
    device: torch.device = None,
) -> Tuple[float, float]:  # Returns (policy_loss, value_loss)
    """
    Train Policy and Value Networks with DeepSupervision
    
    Args:
        model: TRM model
        states: [B, C, H, W] batch of states
        actions: [B] batch of action indices (teacher actions)
        optimizer: Optimizer for training
        n_sup: Number of supervision steps
        strategy: "last" (only final step), "all" (all steps), "weighted" (weighted average)
        rewards: [B] optional batch of rewards for value supervision
        train_value: Whether to train value network with rewards
        value_weight: Weight for value loss relative to policy loss
        device: Device to use
    
    Returns:
        (policy_loss, value_loss) tuple
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    
    # Enable/disable gradient for value head based on train_value flag
    for param in model.v_head.parameters():
        param.requires_grad = train_value and (rewards is not None)
    
    # Enable gradient for policy head
    for param in model.pi_head.parameters():
        param.requires_grad = True
    
    # Enable gradient for TRM network (needed for reasoning)
    for param in model.trm_net.parameters():
        param.requires_grad = True
    
    B = states.shape[0]
    states = states.to(device)
    actions = actions.to(device)
    
    # Handle rewards
    if rewards is not None:
        rewards = rewards.to(device)
        train_value = train_value and len(rewards) > 0
    else:
        train_value = False
    
    # Initialize z for each sample (zeros for supervised learning)
    z = torch.zeros(B, model.z_dim, device=device)
    
    # Patch embedding (shared across all supervision steps)
    x_embed = model._patch_embed(states)  # [B, N, D]
    x_embed = x_embed + model.pos_embed

    policy_losses = []
    value_losses = []
    
    if strategy == "last":
        # Only compute loss on final supervision step (paper approach)
        # Do T-1 steps without gradient, then 1 with gradient
        with torch.no_grad():
            for _ in range(n_sup - 1):
                z = model._latent_recursion(x_embed, z, model.n_latent)
                # Deep recursion: T-1 no grad + 1 with grad
                z = model._deep_recursion(x_embed, z, model.n_latent, model.T)[0]
        
        # Final step with gradient
        z = model._latent_recursion(x_embed, z, model.n_latent)
        z = model.trm_net(x_embed, z)  # One more step with gradient
        
        logits = model.pi_head(z)
        policy_loss = F.cross_entropy(logits, actions)
        policy_losses.append(policy_loss)
        
        # Value supervision if rewards provided
        if train_value and rewards is not None:
            values = model.v_head(z).squeeze(-1)  # [B]
            value_loss = F.mse_loss(values, rewards)
            value_losses.append(value_loss)
        
    elif strategy == "all":
        # Compute loss on all supervision steps and average
        step_policy_losses = []
        step_value_losses = []
        
        for sup_step in range(n_sup):
            if sup_step == n_sup - 1:
                # Last step: compute with gradient
                z = model._latent_recursion(x_embed, z, model.n_latent)
                logits = model.pi_head(z)
                policy_loss = F.cross_entropy(logits, actions)
                step_policy_losses.append(policy_loss)
                
                if train_value and rewards is not None:
                    values = model.v_head(z).squeeze(-1)
                    value_loss = F.mse_loss(values, rewards)
                    step_value_losses.append(value_loss)
            else:
                # Previous steps: update z (with or without gradient)
                with torch.no_grad():
                    z = model._latent_recursion(x_embed, z, model.n_latent)
        
        policy_loss = torch.stack(step_policy_losses).mean()
        policy_losses.append(policy_loss)
        
        if train_value and len(step_value_losses) > 0:
            value_loss = torch.stack(step_value_losses).mean()
            value_losses.append(value_loss)
        
    elif strategy == "weighted":
        # Weighted average: later steps have higher weight
        step_policy_losses = []
        step_value_losses = []
        weights = []
        
        for sup_step in range(n_sup):
            # Update z
            z = model._latent_recursion(x_embed, z, model.n_latent)
            
            # Compute policy loss
            logits = model.pi_head(z)
            policy_loss = F.cross_entropy(logits, actions)
            step_policy_losses.append(policy_loss)
            
            # Compute value loss if rewards provided
            if train_value and rewards is not None:
                values = model.v_head(z).squeeze(-1)
                value_loss = F.mse_loss(values, rewards)
                step_value_losses.append(value_loss)
            
            # Weight: exponential increase
            weight = 2.0 ** (sup_step - n_sup + 1)  # Last step has weight 1.0
            weights.append(weight)
        
        # Weighted average for policy
        weights_tensor = torch.tensor(weights, device=device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        policy_loss = sum(w * l for w, l in zip(weights_tensor, step_policy_losses))
        policy_losses.append(policy_loss)
        
        # Weighted average for value
        if train_value and len(step_value_losses) > 0:
            value_loss = sum(w * l for w, l in zip(weights_tensor, step_value_losses))
            value_losses.append(value_loss)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Compute total loss and backprop
    if len(policy_losses) == 0:
        return 0.0, 0.0
    
    total_policy_loss = torch.stack(policy_losses).mean()
    
    # Add value loss if available
    total_value_loss = torch.tensor(0.0, device=device)
    if train_value and len(value_losses) > 0:
        total_value_loss = torch.stack(value_losses).mean()
    
    # Combined loss
    total_loss = total_policy_loss + value_weight * total_value_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Re-enable value head gradients (in case it's used elsewhere)
    for param in model.v_head.parameters():
        param.requires_grad = True
    
    return float(total_policy_loss.item()), float(total_value_loss.item())


def train_with_simulated_experiences(
    model: PolicyValueViT_TRM,
    real_states: torch.Tensor,
    real_actions: torch.Tensor,
    simulated_states: torch.Tensor,
    simulated_actions: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_sup: int = 16,
    strategy: str = "last",
    real_weight: float = 1.0,
    sim_weight: float = 0.5,
    real_rewards: Optional[torch.Tensor] = None,
    sim_rewards: Optional[torch.Tensor] = None,
    train_value: bool = True,
    value_weight: float = 0.5,
    device: torch.device = None,
) -> Tuple[float, float, float, float]:  # (real_policy_loss, real_value_loss, sim_policy_loss, sim_value_loss)
    """
    Train with both real and simulated experiences
    
    Args:
        model: TRM model
        real_states: [B1, C, H, W] real states
        real_actions: [B1] real actions
        simulated_states: [B2, C, H, W] simulated states from planning
        simulated_actions: [B2] simulated actions
        optimizer: Optimizer
        n_sup: Number of supervision steps
        strategy: DeepSupervision strategy
        real_weight: Weight for real experience loss
        sim_weight: Weight for simulated experience loss
        real_rewards: [B1] optional real rewards for value supervision
        sim_rewards: [B2] optional simulated rewards for value supervision
        train_value: Whether to train value network
        value_weight: Weight for value loss
        device: Device
    
    Returns:
        (real_policy_loss, real_value_loss, sim_policy_loss, sim_value_loss) tuple
    """
    real_policy_loss, real_value_loss = 0.0, 0.0
    sim_policy_loss, sim_value_loss = 0.0, 0.0
    
    # Train on real experiences
    if len(real_states) > 0:
        real_policy_loss, real_value_loss = train_policy_deepsup(
            model, real_states, real_actions, optimizer,
            n_sup=n_sup, strategy=strategy, 
            rewards=real_rewards, train_value=train_value, value_weight=value_weight,
            device=device
        )
    
    # Train on simulated experiences (with lower weight)
    if len(simulated_states) > 0:
        # Temporarily scale learning rate for simulated data
        original_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = original_lr * sim_weight
        
        sim_policy_loss, sim_value_loss = train_policy_deepsup(
            model, simulated_states, simulated_actions, optimizer,
            n_sup=n_sup, strategy=strategy,
            rewards=sim_rewards, train_value=train_value, value_weight=value_weight,
            device=device
        )
        
        # Restore learning rate
        optimizer.param_groups[0]['lr'] = original_lr
    
    return (real_policy_loss * real_weight, real_value_loss * real_weight,
            sim_policy_loss * sim_weight, sim_value_loss * sim_weight)


def create_policy_optimizer(model: PolicyValueViT_TRM, lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create optimizer for policy network only (excluding value head)
    """
    policy_params = list(model.pi_head.parameters()) + list(model.trm_net.parameters())
    return torch.optim.AdamW(policy_params, lr=lr, betas=(0.9, 0.95))

