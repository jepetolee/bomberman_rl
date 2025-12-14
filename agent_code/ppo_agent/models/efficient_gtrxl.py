"""
EfficientNetB0-style CNN + Gated Transformer-XL (GTrXL) Architecture
====================================================================

Combines EfficientNetB0-inspired CNN feature extraction with GTrXL for
sequence modeling in RL. Based on "Stabilizing Transformers for RL" (Parisotto et al., 2019).

Architecture:
1. EfficientNetB0-style CNN backbone (MBConv blocks)
2. Feature map → sequence conversion
3. GTrXL (Gated Transformer-XL) for temporal modeling
4. Policy and Value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution block (EfficientNet style)
    
    Expansion → Depthwise → SE → Pointwise (with residual connection)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float = 6.0,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.expanded_channels = int(in_channels * expansion_ratio)
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase (1x1 conv)
        if expansion_ratio != 1.0:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, self.expanded_channels, 1, bias=False),
                nn.BatchNorm2d(self.expanded_channels),
                Swish()
            )
        else:
            self.expand = None
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                self.expanded_channels,
                self.expanded_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=self.expanded_channels,  # Depthwise
                bias=False
            ),
            nn.BatchNorm2d(self.expanded_channels),
            Swish()
        )
        
        # SE block
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SEBlock(self.expanded_channels, reduction=se_channels)
        
        # Pointwise (projection) phase
        self.project = nn.Sequential(
            nn.Conv2d(self.expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        
        # Expansion
        if self.expand is not None:
            x = self.expand(x)
        
        # Depthwise
        x = self.depthwise(x)
        
        # SE
        x = self.se(x)
        
        # Projection
        x = self.project(x)
        x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
        
        return x


class EfficientNetBackbone(nn.Module):
    """
    EfficientNetB0-style CNN backbone for feature extraction
    Adapted for 17x17 input size (Bomberman grid)
    """
    def __init__(
        self,
        in_channels: int = 10,
        base_channels: int = 32,
        width_mult: float = 1.0,  # Width multiplier
        depth_mult: float = 1.0,  # Depth multiplier (not used in simplified version)
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Stem (first conv layer)
        out_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        
        # MBConv blocks (EfficientNetB0 configuration)
        # (expansion, channels, kernel, stride, num_blocks)
        mb_config = [
            (1, 16, 3, 1, 1),   # Stage 1
            (6, 24, 3, 2, 2),   # Stage 2
            (6, 40, 5, 2, 2),   # Stage 3
            (6, 80, 3, 1, 3),   # Stage 4
        ]
        
        blocks = []
        in_ch = out_channels
        
        for expansion, channels, kernel, stride, num_blocks in mb_config:
            out_ch = int(channels * width_mult)
            for i in range(num_blocks):
                blocks.append(
                    MBConvBlock(
                        in_ch,
                        out_ch,
                        expansion_ratio=expansion,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        dropout=dropout
                    )
                )
                in_ch = out_ch
        
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch  # Final output channels
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.stem(x)
        x = self.blocks(x)
        # Output: [B, out_channels, H', W']
        return x


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Relative Position Encoding (Transformer-XL style)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relative position encoding
        self.rel_pos_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.u = nn.Parameter(torch.zeros(num_heads, self.head_dim))  # Content-based key bias
        self.v = nn.Parameter(torch.zeros(num_heads, self.head_dim))  # Position-based key bias
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x, mem: Optional[torch.Tensor] = None, rel_pos: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, T, D] current sequence
            mem: [B, M, D] memory (previous sequence, optional)
            rel_pos: [T, M+T, D] relative position encoding (optional)
        
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        
        # Prepare key/value with memory
        if mem is not None:
            kv_input = torch.cat([mem, x], dim=1)  # [B, M+T, D]
        else:
            kv_input = x
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        K = self.k_proj(kv_input).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M+T, d]
        V = self.v_proj(kv_input).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M+T, d]
        
        # Compute attention scores
        # Content-based attention
        attn_content = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, M+T]
        
        # Relative position attention (if provided)
        if rel_pos is not None:
            # Simplified: use relative position encoding in attention
            # Full implementation would include position-aware key computation
            rel_pos_k = self.rel_pos_proj(rel_pos)  # [T, M+T, D]
            rel_pos_k = rel_pos_k.view(T, -1, self.num_heads, self.head_dim).transpose(0, 2).transpose(1, 2)  # [H, T, M+T, d]
            Q_expanded = Q.unsqueeze(-2)  # [B, H, T, 1, d]
            attn_pos = torch.matmul(Q_expanded, rel_pos_k.transpose(-2, -1)).squeeze(-2) * self.scale  # [B, H, T, M+T]
            attn = attn_content + attn_pos
        else:
            attn = attn_content
        
        # Apply causal mask (if needed)
        if mem is None:
            # Standard causal mask for autoregressive
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        out = self.out_proj(out)
        
        return out


class GatedResidual(nn.Module):
    """
    Gated residual connection (GTrXL style)
    Uses GRU gate to control information flow
    """
    def __init__(self, dim: int):
        super().__init__()
        # Gate: takes (residual, new_output) -> gate_value
        self.gate_proj = nn.Linear(dim * 2, dim, bias=False)
        self.gate_norm = nn.LayerNorm(dim)
    
    def forward(self, residual: torch.Tensor, new_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: [B, T, D] residual connection
            new_output: [B, T, D] new computation output
        
        Returns:
            [B, T, D]
        """
        # Compute gate (similar to GRU)
        gate_input = torch.cat([residual, new_output], dim=-1)  # [B, T, 2D]
        gate = torch.sigmoid(self.gate_norm(self.gate_proj(gate_input)))  # [B, T, D]
        
        # Gated residual: gate * new_output + (1 - gate) * residual
        output = gate * new_output + (1 - gate) * residual
        return output


class GTrXLBlock(nn.Module):
    """
    Gated Transformer-XL Block with Identity Map Reordering
    Based on "Stabilizing Transformers for RL" (Parisotto et al., 2019)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)
        
        # Identity Map Reordering: LayerNorm applied to input only
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = RelativeMultiHeadAttention(embed_dim, num_heads, dropout)
        self.attn_gate = GatedResidual(embed_dim)
        
        # MLP block (also with gated residual)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.mlp_gate = GatedResidual(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence
            mem: [B, M, D] memory (optional)
            rel_pos: [T, M+T, D] relative position (optional)
        
        Returns:
            [B, T, D]
        """
        # Identity Map Reordering: norm input, then add residual
        residual = x
        x = self.attn_norm(x)
        attn_out = self.attn(x, mem=mem, rel_pos=rel_pos)
        x = self.attn_gate(residual, attn_out)
        
        # MLP with gated residual
        residual = x
        x = self.mlp_norm(x)
        mlp_out = self.mlp(x)
        x = self.mlp_gate(residual, mlp_out)
        
        return x


class PolicyValueEfficientGTrXL(nn.Module):
    """
    EfficientNetB0-style CNN + Gated Transformer-XL for RL
    
    Architecture:
    1. EfficientNetB0 CNN backbone (spatial feature extraction)
    2. Feature map → sequence conversion
    3. GTrXL blocks (temporal modeling with memory)
    4. Policy and Value heads
    """
    def __init__(
        self,
        in_channels: int = 10,
        num_actions: int = 6,
        img_size: Tuple[int, int] = (17, 17),
        # CNN parameters
        cnn_base_channels: int = 32,
        cnn_width_mult: float = 1.0,
        # Transformer parameters
        embed_dim: int = 256,
        gtrxl_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # Memory parameters
        memory_size: int = 256,  # Memory length for Transformer-XL
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        
        # EfficientNetB0-style CNN backbone
        self.cnn_backbone = EfficientNetBackbone(
            in_channels=in_channels,
            base_channels=cnn_base_channels,
            width_mult=cnn_width_mult,
            dropout=dropout,
        )
        cnn_out_channels = self.cnn_backbone.out_channels
        
        # Project CNN features to transformer dimension
        # CNN output spatial size depends on strides (17x17 -> ~4x4 after strides)
        # We'll use adaptive pooling to get a fixed-size feature map
        self.cnn_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed 4x4 output
        self.feature_proj = nn.Linear(cnn_out_channels, embed_dim)
        
        # Positional encoding for spatial positions (4x4 = 16 positions)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # GTrXL blocks
        self.gtrxl_blocks = nn.ModuleList([
            GTrXLBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(gtrxl_depth)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Policy and Value heads
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
        
        # Initialize gates to favor identity (for stable training)
        # This is important for GTrXL stability
        self._init_gates()
    
    def _init_gates(self):
        """Initialize gates to favor identity (pass-through) for stable training"""
        for block in self.gtrxl_blocks:
            # Initialize gate to output ~0.5 (balanced)
            nn.init.zeros_(block.attn_gate.gate_proj.weight)
            nn.init.zeros_(block.mlp_gate.gate_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] input image
            memory: [B, M, D] previous memory (optional)
        
        Returns:
            logits: [B, num_actions]
            value: [B, 1]
        """
        B = x.shape[0]
        
        # CNN feature extraction
        cnn_feat = self.cnn_backbone(x)  # [B, C_out, H', W']
        cnn_feat = self.cnn_pool(cnn_feat)  # [B, C_out, 4, 4]
        
        # Convert to sequence: [B, C, 4, 4] -> [B, 16, C]
        B, C, H, W = cnn_feat.shape
        cnn_feat = cnn_feat.view(B, C, -1).transpose(1, 2)  # [B, 16, C]
        
        # Project to transformer dimension
        x_seq = self.feature_proj(cnn_feat)  # [B, 16, embed_dim]
        
        # Add positional encoding
        x_seq = x_seq + self.pos_embed
        
        # GTrXL blocks with memory
        for block in self.gtrxl_blocks:
            x_seq = block(x_seq, mem=memory)
        
        # Final norm
        x_seq = self.final_norm(x_seq)
        
        # Pool sequence to single vector (mean pooling)
        feat = x_seq.mean(dim=1)  # [B, embed_dim]
        
        # Policy and Value
        logits = self.pi_head(feat)  # [B, num_actions]
        value = self.v_head(feat)  # [B, 1]
        
        return logits, value
    
    def forward_features(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return features before policy/value heads (for env_model and other uses)
        
        Args:
            x: [B, C, H, W] input image
            memory: [B, M, D] previous memory (optional)
        
        Returns:
            feat: [B, embed_dim] pooled feature vector
        """
        B = x.shape[0]
        
        # CNN feature extraction
        cnn_feat = self.cnn_backbone(x)  # [B, C_out, H', W']
        cnn_feat = self.cnn_pool(cnn_feat)  # [B, C_out, 4, 4]
        
        # Convert to sequence
        B, C, H, W = cnn_feat.shape
        cnn_feat = cnn_feat.view(B, C, -1).transpose(1, 2)  # [B, 16, C]
        
        # Project to transformer dimension
        x_seq = self.feature_proj(cnn_feat)  # [B, 16, embed_dim]
        
        # Add positional encoding
        x_seq = x_seq + self.pos_embed
        
        # GTrXL blocks with memory
        for block in self.gtrxl_blocks:
            x_seq = block(x_seq, mem=memory)
        
        # Final norm
        x_seq = self.final_norm(x_seq)
        
        # Pool sequence to single vector (mean pooling)
        feat = x_seq.mean(dim=1)  # [B, embed_dim]
        
        return feat
    
    def forward_with_memory(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with memory update (for Transformer-XL style training)
        
        Returns:
            logits: [B, num_actions]
            value: [B, 1]
            new_memory: [B, M, D] updated memory
        """
        B = x.shape[0]
        
        # CNN feature extraction
        cnn_feat = self.cnn_backbone(x)
        cnn_feat = self.cnn_pool(cnn_feat)
        
        # Convert to sequence
        B, C, H, W = cnn_feat.shape
        cnn_feat = cnn_feat.view(B, C, -1).transpose(1, 2)
        x_seq = self.feature_proj(cnn_feat)
        x_seq = x_seq + self.pos_embed
        
        # Store current sequence for memory update
        # For simplicity, we use the current sequence as new memory
        # In full Transformer-XL, this would be more sophisticated
        new_memory = x_seq.detach()  # [B, 16, embed_dim]
        
        # GTrXL blocks
        for block in self.gtrxl_blocks:
            x_seq = block(x_seq, mem=memory)
        
        x_seq = self.final_norm(x_seq)
        feat = x_seq.mean(dim=1)
        
        logits = self.pi_head(feat)
        value = self.v_head(feat)
        
        return logits, value, new_memory

