import math
from typing import Tuple

import torch
import torch.nn as nn

from .attention import TransformerEncoderBlock
from .gdn import GDNEncoderBlock


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 1, img_size: Tuple[int, int] = (17, 17)):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PolicyValueViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        num_actions: int = 6,
        img_size: Tuple[int, int] = (17, 17),
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        patch_size: int = 1,
        use_cls_token: bool = False,
        mixer: str = "attn",  # "gdn" (headed) or "attn"
    ):
        super().__init__()
        self.use_cls = use_cls_token

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size=patch_size, img_size=img_size)
        num_tokens = self.patch_embed.num_patches + (1 if use_cls_token else 0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        blocks = []
        for _ in range(depth):
            if mixer == "attn":
                blocks.append(TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop))
            else:  # gdn headed by default
                blocks.append(GDNEncoderBlock(embed_dim, mlp_ratio=mlp_ratio, drop=drop, num_heads=num_heads, variant="head"))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        x = self.patch_embed(x)
        B, N, D = x.shape

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.use_cls:
            feat = x[:, 0]
        else:
            feat = x.mean(dim=1)

        logits = self.pi_head(feat)
        value = self.v_head(feat).squeeze(-1)
        return logits, value


