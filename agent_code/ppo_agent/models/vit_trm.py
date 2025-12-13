import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PolicyValueViT  # For hybrid ViT+TRM model

# PatchEmbed removed - using custom implementation with stride support


class SwiGLU(nn.Module):
    """SwiGLU activation function used in TRM paper"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class TRMRecursiveNet(nn.Module):
    """Tiny 2-layer network for recursive reasoning"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        hidden_dim = int(embed_dim * mlp_ratio)
        
        # Input: x_embed (patches) and z (latent)
        # z has same dimension as embed_dim, so concatenate: [x_agg, z] -> [embed_dim + embed_dim]
        self.input_proj = nn.Linear(embed_dim + embed_dim, embed_dim)
        
        # First layer with SwiGLU
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.swiglu = SwiGLU()
        self.drop1 = nn.Dropout(drop)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(drop)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection for z (z_dim = embed_dim, so identity projection)
        # Keep for compatibility but it's effectively identity when z_dim == embed_dim

    def forward(self, x_embed: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_embed: [B, N, D] patch embeddings where D = embed_dim
            z: [B, embed_dim] latent state (same dimension as embed_dim)
        Returns:
            z_new: [B, embed_dim] updated latent state
        """
        B, N, D = x_embed.shape
        embed_dim = self.embed_dim
        
        # Ensure z has correct shape [B, embed_dim]
        if z.dim() == 2:
            # z is already 2D, check if it's [B, embed_dim]
            if z.shape[0] != B:
                # Batch size mismatch, take first B or repeat if needed
                if z.shape[0] > B:
                    z = z[:B]
                else:
                    # Repeat to match batch size
                    z = z.repeat((B // z.shape[0] + 1, 1))[:B]
            if z.shape[1] != embed_dim:
                # embed_dim mismatch, pad or trim
                if z.shape[1] > embed_dim:
                    z = z[:, :embed_dim]
                else:
                    z = torch.cat([z, torch.zeros(B, embed_dim - z.shape[1], device=z.device, dtype=z.dtype)], dim=1)
        elif z.dim() == 1:
            # z is 1D: could be [B*embed_dim], [embed_dim], or [B]
            if z.shape[0] == B * embed_dim:
                # z is [B*embed_dim], reshape to [B, embed_dim]
                z = z.view(B, embed_dim)
            elif z.shape[0] == embed_dim:
                # z is [embed_dim], expand to [B, embed_dim]
                z = z.unsqueeze(0).expand(B, -1)
            elif z.shape[0] == B:
                # z is [B], but we need [B, embed_dim] - pad with zeros
                z = torch.zeros(B, embed_dim, device=z.device, dtype=z.dtype)
            else:
                # Unknown shape, create zeros
                z = torch.zeros(B, embed_dim, device=z.device, dtype=z.dtype)
        elif z.dim() == 0:
            # z is scalar, create zeros
            z = torch.zeros(B, embed_dim, device=z.device, dtype=z.dtype)
        else:
            # z has too many dimensions, flatten to [B, embed_dim]
            z = z.view(B, -1)
            if z.shape[1] != embed_dim:
                if z.shape[1] > embed_dim:
                    z = z[:, :embed_dim]
                else:
                    z = torch.cat([z, torch.zeros(B, embed_dim - z.shape[1], device=z.device, dtype=z.dtype)], dim=1)
        
        # Aggregate x_embed (mean pooling or use cls-like approach)
        x_agg = x_embed.mean(dim=1)  # [B, embed_dim]
        
        # Concatenate aggregated patches with z
        xz = torch.cat([x_agg, z], dim=-1)  # [B, embed_dim + embed_dim]
        
        # Project to embed_dim
        h = self.input_proj(xz)  # [B, embed_dim]
        
        # Two-layer MLP with SwiGLU
        h = self.fc1(h)  # [B, hidden_dim * 2]
        h = self.swiglu(h)  # [B, hidden_dim]
        h = self.drop1(h)
        h = self.fc2(h)  # [B, embed_dim]
        h = self.drop2(h)
        
        # Residual connection and norm
        h = self.norm(h)
        
        # z_new is just h (since z_dim == embed_dim, no projection needed)
        z_new = h  # [B, embed_dim]
        
        return z_new


class PolicyValueViT_TRM(nn.Module):
    """Vision Transformer with Tiny Recursive Model structure"""
    def __init__(
        self,
        in_channels: int = 8,
        num_actions: int = 6,
        img_size: Tuple[int, int] = (17, 17),
        embed_dim: int = 128,
        n_latent: int = 6,  # Number of latent recursion steps
        n_sup: int = 16,    # Number of supervision steps
        T: int = 3,         # Deep recursion steps (T-1 no_grad + 1 with grad)
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        patch_size: int = 1,
        patch_stride: int = None,  # If None, uses patch_size (non-overlapping)
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # z_dim is always equal to embed_dim
        self.z_dim = embed_dim
        self.n_latent = n_latent
        self.n_sup = n_sup
        self.T = T
        self.use_ema = use_ema
        
        # Patch embedding with smaller patches or overlapping patches
        # If patch_stride < patch_size, creates overlapping patches for more information
        if patch_stride is None:
            patch_stride = patch_size
        
        # Create custom patch embedding that supports stride
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.img_size = img_size
        
        # Calculate number of patches with stride
        h, w = img_size
        self.grid_size = (
            (h - patch_size) // patch_stride + 1,
            (w - patch_size) // patch_stride + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Convolution for patch extraction (stride allows overlapping)
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_stride,
            padding=0
        )
        
        # Positional embedding for patches
        num_tokens = self.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # TRM Recursive Network (z_dim = embed_dim always)
        self.trm_net = TRMRecursiveNet(embed_dim, mlp_ratio=mlp_ratio, drop=drop)
        
        # EMA for stable training (if needed)
        if use_ema:
            self.trm_net_ema = TRMRecursiveNet(embed_dim, mlp_ratio=mlp_ratio, drop=drop)
            for param in self.trm_net_ema.parameters():
                param.requires_grad = False
            self.ema_decay = ema_decay
            self._update_ema()
        
        # Feature aggregation: combine patches and z
        self.aggregate_norm = nn.LayerNorm(embed_dim)
        
        # Policy and Value heads
        hidden = 256
        self.pi_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _update_ema(self):
        """Update EMA model parameters"""
        if not self.use_ema:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.trm_net_ema.parameters(), self.trm_net.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _latent_recursion(self, x_embed: torch.Tensor, z: torch.Tensor, n: int) -> torch.Tensor:
        """Perform n steps of latent recursion"""
        for _ in range(n):
            z = self.trm_net(x_embed, z)
        return z

    def _deep_recursion(self, x_embed: torch.Tensor, z: torch.Tensor, n: int, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deep recursion: T-1 no_grad steps, then 1 with grad
        Returns: (z_final, logits_intermediate)
        """
        # T-1 no gradient steps to improve z
        with torch.no_grad():
            for _ in range(T - 1):
                z = self._latent_recursion(x_embed, z, n)
        
        # Final step with gradients (if training)
        if self.training:
            z = self._latent_recursion(x_embed, z, n)
        else:
            # Inference: use EMA model if available
            if self.use_ema:
                for _ in range(n):
                    z = self.trm_net_ema(x_embed, z)
            else:
                z = self._latent_recursion(x_embed, z, n)
        
        return z

    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches with stride (can be overlapping)
        
        Args:
            x: [B, C, H, W] input image
        Returns:
            x_embed: [B, N, D] patch embeddings where N = num_patches
        """
        # Extract patches using convolution with stride
        x_patches = self.patch_proj(x)  # [B, D, H_patches, W_patches]
        B, D, H_p, W_p = x_patches.shape
        
        # Flatten spatial dimensions
        x_embed = x_patches.flatten(2).transpose(1, 2)  # [B, N, D]
        return x_embed

    def forward_with_z(self, x: torch.Tensor, z_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with recurrent latent state
        
        Args:
            x: [B, C, H, W] input image
            z_prev: [B, embed_dim] previous timestep latent (None means zeros)
        Returns:
            logits: [B, num_actions]
            value: [B]
            z_new: [B, embed_dim] updated latent for next timestep
        """
        B = x.shape[0]
        device = x.device
        
        # Initialize z if not provided (z_dim = embed_dim)
        if z_prev is None:
            z = torch.zeros(B, self.embed_dim, device=device)
        else:
            z = z_prev
        
        # Patch embedding with smaller/overlapping patches
        x_embed = self._patch_embed(x)  # [B, N, D]
        x_embed = x_embed + self.pos_embed
        
        # Initial latent recursion (from previous z)
        z = self._latent_recursion(x_embed, z, self.n_latent)
        
        # Deep supervision: N_sup steps of improvement
        if self.n_sup > 0:
            if self.training:
                # Training: deep supervision with T-step recursion
                for sup_step in range(self.n_sup):
                    z = self._deep_recursion(x_embed, z, self.n_latent, self.T)[0]
            else:
                # Inference: simple recursion
                z = self._latent_recursion(x_embed, z, self.n_sup)
        
        # Update EMA if training
        if self.training and self.use_ema:
            self._update_ema()
        
        # Get final feature from z
        # z is already in the right dimension for heads
        logits = self.pi_head(z)
        value = self.v_head(z).squeeze(-1)
        
        return logits, value, z

    def forward(self, x: torch.Tensor):
        """
        Standard forward (for compatibility, uses z=None)
        """
        logits, value, _ = self.forward_with_z(x, z_prev=None)
        return logits, value


# -------------------------------------------------------------
# Hybrid model: ViT backbone + TRM residual embedding
# -------------------------------------------------------------
class PolicyValueViT_TRM_Hybrid(nn.Module):
    """
    Hybrid model that combines a ViT backbone feature with a TRM residual
    by simple addition before policy/value heads.

    - Pretraining (supervised): set use_trm=False or detach_trm=True to train ViT only
    - RL phase: use_trm=True, detach_trm=False to let TRM learn and modulate embeddings
    """
    def __init__(
        self,
        in_channels: int = 10,
        num_actions: int = 6,
        img_size: Tuple[int, int] = (17, 17),
        embed_dim: int = 256,
        # ViT backbone params
        vit_depth: int = 2,
        vit_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_patch_size: int = 1,
        # TRM params
        trm_n_latent: int = 4,
        trm_mlp_ratio: float = 4.0,
        trm_drop: float = 0.0,
        trm_patch_size: int = 2,
        trm_patch_stride: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.trm_n_latent = trm_n_latent
        self.use_ema = use_ema

        # ViT backbone (will drive actor/critic; TRM adds residual)
        self.vit = PolicyValueViT(
            in_channels=in_channels,
            num_actions=num_actions,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            drop=0.0,
            attn_drop=0.0,
            patch_size=vit_patch_size,
            use_cls_token=False,
            mixer="attn",
        )

        # TRM branch (patch embedding with stride/overlap)
        if trm_patch_stride is None:
            trm_patch_stride = trm_patch_size
        self.trm_patch_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=trm_patch_size,
            stride=trm_patch_stride,
            padding=0,
        )
        h, w = img_size
        grid_size = (
            (h - trm_patch_size) // trm_patch_stride + 1,
            (w - trm_patch_size) // trm_patch_stride + 1,
        )
        num_tokens = grid_size[0] * grid_size[1]
        self.trm_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.trm_pos_embed, std=0.02)

        self.trm_net = TRMRecursiveNet(embed_dim, mlp_ratio=trm_mlp_ratio, drop=trm_drop)
        if use_ema:
            self.trm_net_ema = TRMRecursiveNet(embed_dim, mlp_ratio=trm_mlp_ratio, drop=trm_drop)
            for p in self.trm_net_ema.parameters():
                p.requires_grad = False
            self.ema_decay = ema_decay
            self._update_ema()

        # Shared heads (after fusion)
        hidden = 256
        self.pi_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.v_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _update_ema(self):
        if not self.use_ema:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.trm_net_ema.parameters(), self.trm_net.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _trm_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        x_patches = self.trm_patch_proj(x)  # [B, D, H_p, W_p]
        B, D, H_p, W_p = x_patches.shape
        x_embed = x_patches.flatten(2).transpose(1, 2)  # [B, N, D]
        return x_embed

    def _latent_recursion(self, x_embed: torch.Tensor, z: torch.Tensor, n: int) -> torch.Tensor:
        for _ in range(n):
            z = self.trm_net(x_embed, z)
        return z

    def forward(
        self,
        x: torch.Tensor,
        z_prev: Optional[torch.Tensor] = None,
        use_trm: bool = True,
        detach_trm: bool = False,
    ):
        """
        use_trm=False or detach_trm=True  -> effectively ViT-only (for pretraining)
        use_trm=True, detach_trm=False    -> hybrid (for RL phase)
        """
        device = x.device
        B = x.shape[0]

        # ViT backbone feature
        vit_feat = self.vit.forward_features(x)  # [B, D]

        # TRM residual
        if use_trm:
            x_embed = self._trm_patch_embed(x) + self.trm_pos_embed  # [B, N, D]
            if z_prev is None:
                z = torch.zeros(B, self.embed_dim, device=device)
            else:
                z = z_prev

            z = self._latent_recursion(x_embed, z, self.trm_n_latent)
            trm_feat = z
            if detach_trm:
                trm_feat = trm_feat.detach()
            fused = vit_feat + trm_feat

            if not self.training and self.use_ema:
                # Optional EMA update during eval
                self._update_ema()
        else:
            fused = vit_feat
            trm_feat = z_prev

        logits = self.pi_head(fused)
        value = self.v_head(fused).squeeze(-1)
        return logits, value, trm_feat

