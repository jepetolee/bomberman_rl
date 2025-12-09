import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.alpha_proj = nn.Linear(dim, dim)
        self.beta_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        assert C == self.dim

        q = self._l2_norm(F.silu(self.q_proj(x)))           # [B,N,C]
        k = self._l2_norm(F.silu(self.k_proj(x)))           # [B,N,C]
        v = F.silu(self.v_proj(x))                          # [B,N,C]
        alpha = torch.sigmoid(self.alpha_proj(x))           # [B,N,C] in (0,1)
        beta = torch.sigmoid(self.beta_proj(x))             # [B,N,C] in (0,1)

        eye = torch.eye(C, device=x.device).unsqueeze(0).expand(B, C, C)  # [B,C,C]
        S = torch.zeros(B, C, C, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(N):
            qt = q[:, t, :]                                 # [B,C]
            kt = k[:, t, :]                                 # [B,C]
            vt = v[:, t, :]                                 # [B,C]
            at = alpha[:, t, :].mean(dim=-1, keepdim=True)  # [B,1] (scalar gate per token)
            bt = beta[:, t, :].mean(dim=-1, keepdim=True)   # [B,1]

            kkT = torch.einsum('bi,bj->bij', kt, kt)        # [B,C,C]
            decay = eye - bt.view(B, 1, 1) * kkT            # [B,C,C]
            S = torch.bmm(S, at.view(B, 1, 1) * decay) + bt.view(B, 1, 1) * torch.einsum('bi,bj->bij', vt, kt)

            ot = torch.einsum('bi,bij->bj', qt, S.transpose(1, 2))  # [B,C]
            outputs.append(ot)

        O = torch.stack(outputs, dim=1)                     # [B,N,C]
        return self.out_proj(O)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        # Aim to preserve length; for even kernels, conv will overshoot by 1, we trim back to input length
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, groups=channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] -> depthwise conv over N
        B, N, C = x.shape
        x = x.transpose(1, 2)  # [B, C, N]
        y = self.conv(x)       # [B, C, N +/-]
        # Trim or pad to original length N if needed
        if y.size(-1) > N:
            y = y[..., :N]
        elif y.size(-1) < N:
            pad_len = N - y.size(-1)
            y = torch.nn.functional.pad(y, (0, pad_len))
        y = self.act(y)
        y = y.transpose(1, 2)  # [B, N, C]
        return y


class GatedDeltaMixerHeaded(nn.Module):
    def __init__(self, dim: int, num_heads: int, kernel_size: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.a_proj = nn.Linear(dim, num_heads)
        self.b_proj = nn.Linear(dim, num_heads)

        self.q_conv = ShortConv1d(dim, kernel_size)
        self.k_conv = ShortConv1d(dim, kernel_size)
        self.v_conv = ShortConv1d(dim, kernel_size)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self._l2_norm(self.q_conv(F.silu(self.q_proj(x))))
        k = self._l2_norm(self.k_conv(F.silu(self.k_proj(x))))
        v = self.v_conv(F.silu(self.v_proj(x)))
        a = torch.sigmoid(self.a_proj(x))  # [B,N,H]
        b = torch.sigmoid(self.b_proj(x))  # [B,N,H]

        q = q.view(B, N, H, Dh)
        k = k.view(B, N, H, Dh)
        v = v.view(B, N, H, Dh)

        S = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(N):
            qt = q[:, t]                    # [B,H,Dh]
            kt = k[:, t]                    # [B,H,Dh]
            vt = v[:, t]                    # [B,H,Dh]
            at = a[:, t].unsqueeze(-1)      # [B,H,1]
            bt = b[:, t].unsqueeze(-1)      # [B,H,1]

            kkT = torch.einsum('bhd,bhe->bhde', kt, kt)         # [B,H,Dh,Dh]
            decay = torch.eye(Dh, device=x.device).view(1, 1, Dh, Dh) - bt.unsqueeze(-1) * kkT
            S = torch.einsum('bhde,bhef->bhdf', S, at.unsqueeze(-1) * decay) + bt.unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', vt, kt)

            ot = torch.einsum('bhd,bhed->bhe', qt, S.transpose(-2, -1))  # [B,H,Dh]
            outputs.append(ot.reshape(B, C))

        O = torch.stack(outputs, dim=1)  # [B,N,C]
        return self.out_proj(O)


class GDNEncoderBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, num_heads: int = 8, variant: str = "head"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if variant == "head":
            self.mixer = GatedDeltaMixerHeaded(dim, num_heads=num_heads)
        else:
            self.mixer = GatedDeltaMixer(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


