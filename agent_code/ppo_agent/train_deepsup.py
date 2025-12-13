"""
DeepSupervision Training for TRM Policy Network
================================================

Trains the Policy Network using DeepSupervision: same state is processed
N_sup times to improve reasoning before computing policy loss.

Only Policy Network is trained (Value Network is excluded).
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
    device: torch.device = None,
) -> float:
    """
    Train Policy Network with DeepSupervision
    
    Args:
        model: TRM model (only policy head will be trained)
        states: [B, C, H, W] batch of states
        actions: [B] batch of action indices (teacher actions)
        optimizer: Optimizer for policy network
        n_sup: Number of supervision steps
        strategy: "last" (only final step), "all" (all steps), "weighted" (weighted average)
        device: Device to use
    
    Returns:
        Average policy loss
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    
    # Disable gradient for value head
    for param in model.v_head.parameters():
        param.requires_grad = False
    
    # Enable gradient for policy head
    for param in model.pi_head.parameters():
        param.requires_grad = True
    
    # Enable gradient for TRM network (needed for reasoning)
    for param in model.trm_net.parameters():
        param.requires_grad = True
    
    B = states.shape[0]
    states = states.to(device)
    actions = actions.to(device)
    
    # Initialize z for each sample (zeros for supervised learning)
    z = torch.zeros(B, model.z_dim, device=device)
    
    # Patch embedding (shared across all supervision steps)
    x_embed = model.patch_embed(states)  # [B, N, D]
    x_embed = x_embed + model.pos_embed
    
    losses = []
    
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
        loss = F.cross_entropy(logits, actions)
        losses.append(loss)
        
    elif strategy == "all":
        # Compute loss on all supervision steps and average
        step_losses = []
        
        for sup_step in range(n_sup):
            if sup_step == n_sup - 1:
                # Last step: compute with gradient
                z = model._latent_recursion(x_embed, z, model.n_latent)
                logits = model.pi_head(z)
                loss = F.cross_entropy(logits, actions)
                step_losses.append(loss)
            else:
                # Previous steps: update z (with or without gradient)
                with torch.no_grad():
                    z = model._latent_recursion(x_embed, z, model.n_latent)
        
        loss = torch.stack(step_losses).mean()
        losses.append(loss)
        
    elif strategy == "weighted":
        # Weighted average: later steps have higher weight
        step_losses = []
        weights = []
        
        for sup_step in range(n_sup):
            # Update z
            z = model._latent_recursion(x_embed, z, model.n_latent)
            
            # Compute loss
            logits = model.pi_head(z)
            step_loss = F.cross_entropy(logits, actions)
            step_losses.append(step_loss)
            
            # Weight: exponential increase
            weight = 2.0 ** (sup_step - n_sup + 1)  # Last step has weight 1.0
            weights.append(weight)
        
        # Weighted average
        weights_tensor = torch.tensor(weights, device=device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        loss = sum(w * l for w, l in zip(weights_tensor, step_losses))
        losses.append(loss)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Re-enable value head gradients (in case it's used elsewhere)
    for param in model.v_head.parameters():
        param.requires_grad = True
    
    return loss.item()


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
    device: torch.device = None,
) -> Tuple[float, float]:
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
        device: Device
    
    Returns:
        (real_loss, sim_loss) tuple
    """
    real_loss = 0.0
    sim_loss = 0.0
    
    # Train on real experiences
    if len(real_states) > 0:
        real_loss = train_policy_deepsup(
            model, real_states, real_actions, optimizer,
            n_sup=n_sup, strategy=strategy, device=device
        )
    
    # Train on simulated experiences (with lower weight)
    if len(simulated_states) > 0:
        # Temporarily scale learning rate for simulated data
        original_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = original_lr * sim_weight
        
        sim_loss = train_policy_deepsup(
            model, simulated_states, simulated_actions, optimizer,
            n_sup=n_sup, strategy=strategy, device=device
        )
        
        # Restore learning rate
        optimizer.param_groups[0]['lr'] = original_lr
    
    return real_loss * real_weight, sim_loss * sim_weight


def create_policy_optimizer(model: PolicyValueViT_TRM, lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create optimizer for policy network only (excluding value head)
    """
    policy_params = list(model.pi_head.parameters()) + list(model.trm_net.parameters())
    return torch.optim.AdamW(policy_params, lr=lr, betas=(0.9, 0.95))

