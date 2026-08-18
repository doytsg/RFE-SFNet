"""Lightweight spectral token mixer and FFN blocks for RFE-SFNet.

The default mixer is the concise H6R4+R8 variant: six complete low-rank
spectral heads, all-head softmax routing, and a head-specific rank-8 channel
recombination inside each expert. Experimental router context, routing-stat
hooks, temperature routing, hard sparse masking, and post-mixture value mixing
have been removed.

Use ``token_mixer='no_phase'`` for the default path, or
``token_mixer='self_attention'`` through ``--use_mhsa`` for the MHSA ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_to_multiple(x: int, multiple: int = 8) -> int:
    return max(multiple, int(round(x / multiple) * multiple))


class SpectralGatedFilter1D(nn.Module):
    """Original attention-free real-envelope spectral token mixer."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 128,
        gate_kernel_size: int = 7,
        dropout: float = 0.0,
        init_identity: bool = True,
        num_heads: int = 1,
        freq_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.freq_len = self.max_seq_len // 2 + 1
        envelope = torch.ones(self.freq_len, self.dim, dtype=torch.float32)
        if not init_identity:
            nn.init.trunc_normal_(envelope, std=0.02)
        self.envelope = nn.Parameter(envelope)
        self.dropout = nn.Dropout(dropout)

    def _interpolate_freq_param(self, weight: torch.Tensor, freq_len: int) -> torch.Tensor:
        if weight.shape[0] == freq_len:
            return weight
        return F.interpolate(
            weight.t().unsqueeze(0), size=freq_len, mode="linear", align_corners=False
        ).squeeze(0).t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, but got {channels}.")
        shortcut_dtype = x.dtype
        x_float = x.float()
        x_fft = torch.fft.rfft(x_float, dim=1, norm="ortho")
        envelope = self._interpolate_freq_param(self.envelope, x_fft.shape[1])
        y_fft = x_fft * envelope.to(device=x_fft.device, dtype=x_fft.real.dtype).unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=seq_len, dim=1, norm="ortho")
        return self.dropout(y.to(dtype=shortcut_dtype))



class CompetitiveSpectralMoH1D(nn.Module):
    """No-shared spectral MoH with head-specific channel recombination.

    Each routed head is a complete low-rank real spectral filter. After the
    inverse FFT, every head applies its own rank-R channel recombination before
    all heads are mixed by token-wise softmax router weights.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 128,
        gate_kernel_size: int = 7,
        dropout: float = 0.0,
        init_identity: bool = True,
        num_heads: int = 6,
        rank: int = 4,
        balance_loss_weight: float = 0.003,
        expert_strength: float = 0.5,
        proj_rank: int = 8,
        freq_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.freq_len = self.max_seq_len // 2 + 1
        self.num_heads = int(num_heads)
        self.rank = max(1, int(rank))
        # Kept for command compatibility; all-head routing has no load-balance aux loss.
        self.balance_loss_weight = float(balance_loss_weight)
        self.expert_strength = float(expert_strength)
        self.proj_rank = max(0, int(proj_rank))

        self.expert_freq = nn.Parameter(torch.empty(self.num_heads, self.rank, self.freq_len))
        self.expert_channel = nn.Parameter(torch.empty(self.num_heads, self.rank, self.dim))
        self.head_gain = nn.Parameter(torch.ones(self.num_heads))
        nn.init.trunc_normal_(self.expert_freq, std=0.02)
        nn.init.trunc_normal_(self.expert_channel, std=0.02)

        if self.proj_rank > 0:
            self.out_down = nn.Parameter(torch.empty(self.num_heads, self.dim, self.proj_rank))
            self.out_up = nn.Parameter(torch.empty(self.num_heads, self.proj_rank, self.dim))
            self.out_scale = nn.Parameter(torch.full((self.num_heads,), 0.10))
            nn.init.trunc_normal_(self.out_down, std=0.02)
            nn.init.trunc_normal_(self.out_up, std=0.02)
        else:
            self.register_parameter("out_down", None)
            self.register_parameter("out_up", None)
            self.register_parameter("out_scale", None)

        self.router_norm = nn.LayerNorm(self.dim)
        self.router = nn.Linear(self.dim, self.num_heads, bias=False)
        nn.init.trunc_normal_(self.router.weight, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def _interp_expert_freq(self, weight: torch.Tensor, freq_len: int) -> torch.Tensor:
        if weight.shape[-1] == freq_len:
            return weight
        h, r, f = weight.shape
        return F.interpolate(
            weight.reshape(h * r, 1, f), size=freq_len, mode="linear", align_corners=False
        ).reshape(h, r, freq_len)

    def _expert_envelopes(self, freq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        freq = torch.tanh(self._interp_expert_freq(self.expert_freq, freq_len)).to(device=device, dtype=dtype)
        channel = torch.tanh(self.expert_channel).to(device=device, dtype=dtype)
        low_rank = torch.einsum("hrf,hrc->hfc", freq, channel) / (float(self.rank) ** 0.5)
        gain = self.head_gain.to(device=device, dtype=dtype).view(self.num_heads, 1, 1)
        return 1.0 + self.expert_strength * torch.tanh(gain * low_rank)

    def _project_heads(self, expert_y: torch.Tensor) -> torch.Tensor:
        if self.proj_rank <= 0:
            return expert_y
        out_down = self.out_down.to(device=expert_y.device, dtype=expert_y.dtype)
        out_up = self.out_up.to(device=expert_y.device, dtype=expert_y.dtype)
        z = torch.einsum("blhc,hcp->blhp", expert_y, out_down)
        delta = torch.einsum("blhp,hpc->blhc", z, out_up) / (float(self.proj_rank) ** 0.5)
        scale = torch.tanh(self.out_scale).to(device=expert_y.device, dtype=expert_y.dtype)
        return expert_y + scale.view(1, 1, self.num_heads, 1) * delta

    def _routing_weights(self, x_float: torch.Tensor) -> torch.Tensor:
        scores = self.router(self.router_norm(x_float))
        return torch.softmax(scores, dim=-1)

    def aux_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.router.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, but got {channels}.")
        shortcut_dtype = x.dtype
        x_float = x.float()
        x_fft = torch.fft.rfft(x_float, dim=1, norm="ortho")
        freq_len = x_fft.shape[1]

        envelopes = self._expert_envelopes(freq_len, x_fft.device, x_fft.real.dtype)
        weights = self._routing_weights(x_float).to(dtype=x_fft.real.dtype)
        expert_fft = x_fft.unsqueeze(1) * envelopes.unsqueeze(0)
        expert_y = torch.fft.irfft(expert_fft, n=seq_len, dim=2, norm="ortho")
        expert_y = self._project_heads(expert_y.permute(0, 2, 1, 3))
        y = torch.sum(expert_y * weights.unsqueeze(-1), dim=2)
        return self.dropout(y.to(dtype=shortcut_dtype))


class ConvSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 1.5, dropout: float = 0.1, dw_kernel_size: int = 15) -> None:
        super().__init__()
        hidden = _round_to_multiple(int((2.0 * mlp_ratio * dim) / 3.0), 8)
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size=dw_kernel_size, padding=dw_kernel_size // 2, groups=hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        gate = self.dwconv(gate.transpose(1, 2)).transpose(1, 2)
        x = value * F.silu(gate)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LiBottleneckFFN(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, dropout: float = 0.1, dw_kernel_size: int = 15) -> None:
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size=dw_kernel_size, padding=dw_kernel_size // 2, groups=hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class PeriodicBottleneckFFN(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, dropout: float = 0.1, dw_kernel_size: int = 15, init_freq: float = 1.0) -> None:
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size=dw_kernel_size, padding=dw_kernel_size // 2, groups=hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.full((hidden,), float(init_freq)))
        self.phase = nn.Parameter(torch.zeros(hidden))
        self.gamma = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc1(x)
        z = self.dwconv(z.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(z) + self.gamma * torch.sin(self.alpha * z + self.phase)
        y = self.drop(y)
        y = self.fc2(y)
        return self.drop(y)


class SelfAttentionTokenMixer1D(nn.Module):
    def __init__(self, dim: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim={dim} must be divisible by nhead={nhead}")
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


class DSFBEncoderLayerV2(nn.Module):
    """MetaFormer-style layer with the lean no-phase spectral MoH mixer."""

    def __init__(
        self,
        dim: int,
        seq_len: int = 128,
        mlp_ratio: float = 1.5,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        gate_kernel_size: int = 7,
        ffn_dw_kernel_size: int = 15,
        ffn_type: str = "li_bottleneck",
        ffn_reduction: int = 4,
        token_mixer: str = "no_phase",
        nhead: int = 4,
        dsfb_num_heads: int = 1,
        dsfb_freq_kernel_size: int = 1,
        ffn_periodic_init_freq: float = 1.0,
        moh_num_heads: int = 6,
        moh_rank: int = 4,
        moh_balance_loss_weight: float = 0.003,
        moh_expert_strength: float = 0.5,
        moh_proj_rank: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if token_mixer == "dsfb":
            self.token_mixer = SpectralGatedFilter1D(
                dim=dim, max_seq_len=seq_len, gate_kernel_size=gate_kernel_size,
                dropout=dropout, num_heads=dsfb_num_heads, freq_kernel_size=dsfb_freq_kernel_size,
            )
        elif token_mixer == "self_attention":
            self.token_mixer = SelfAttentionTokenMixer1D(dim=dim, nhead=nhead, dropout=dropout)
        elif token_mixer in ("no_phase", "csmoh_plus", "competitive_moh_dsfb"):
            self.token_mixer = CompetitiveSpectralMoH1D(
                dim=dim, max_seq_len=seq_len, gate_kernel_size=gate_kernel_size,
                dropout=dropout, num_heads=moh_num_heads, rank=moh_rank,
                balance_loss_weight=moh_balance_loss_weight,
                expert_strength=moh_expert_strength,
                proj_rank=moh_proj_rank,
                freq_kernel_size=dsfb_freq_kernel_size,
            )
        else:
            raise ValueError(
                f"Unsupported token_mixer: {token_mixer}. "
                "Use 'no_phase', 'csmoh_plus', 'competitive_moh_dsfb', 'dsfb', or 'self_attention'."
            )

        self.norm2 = nn.LayerNorm(dim)
        if ffn_type == "li_bottleneck":
            self.ffn = LiBottleneckFFN(dim=dim, reduction=ffn_reduction, dropout=dropout, dw_kernel_size=ffn_dw_kernel_size)
        elif ffn_type == "swiglu":
            self.ffn = ConvSwiGLUFFN(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout, dw_kernel_size=ffn_dw_kernel_size)
        elif ffn_type == "periodic":
            self.ffn = PeriodicBottleneckFFN(dim=dim, reduction=ffn_reduction, dropout=dropout, dw_kernel_size=ffn_dw_kernel_size, init_freq=ffn_periodic_init_freq)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def aux_loss(self) -> torch.Tensor:
        fn = getattr(self.token_mixer, "aux_loss", None)
        if callable(fn):
            return fn()
        return torch.zeros((), device=self.norm1.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
