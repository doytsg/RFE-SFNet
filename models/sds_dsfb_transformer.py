"""RFE-SFNet implementation with the lightweight Spectral Filter Mixer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sds_frontend import MatchedConvFrontEnd1D, PlainConvFrontEnd1D, SDSFrontEnd1D
from .refined_dsfb_modules import DSFBEncoderLayerV2


def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(1, max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    if d_model % 2 == 0:
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
    else:
        even_idx = torch.arange(0, d_model, 2, dtype=torch.float32)
        odd_idx = torch.arange(1, d_model, 2, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * torch.exp(even_idx * -(math.log(10000.0) / d_model)))
        pe[0, :, 1::2] = torch.cos(position * torch.exp(odd_idx * -(math.log(10000.0) / d_model)))
    return pe


class RFESFNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        d_model: int = 128,
        num_layers: int = 1,
        max_len: int = 128,
        dropout: float = 0.1,
        mlp_ratio: float = 1.5,
        frontend_dilations=(1, 4, 12),
        token_mixer: str = "no_phase",
        nhead: int = 4,
        dsfb_num_heads: int = 1,
        dsfb_freq_kernel_size: int = 1,
        use_haar_wavelet: bool = True,
        use_identity_branch: bool = True,
        ffn_type: str = "li_bottleneck",
        ffn_periodic_init_freq: float = 1.0,
        simple_last_down: bool = True,
        simple_head: bool = True,
        pos_embedding: str = "none",
        use_sds_frontend: bool = True,
        matched_conv_frontend: bool = False,
        use_sk_fusion: bool = False,
        cross_scale_mode: str = "full",
        wavelet_downsample: str = "haar_lpr",
        moh_num_heads: int = 6,
        moh_rank: int = 4,
        moh_balance_loss_weight: float = 0.003,
        moh_expert_strength: float = 0.5,
        moh_proj_rank: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.token_mixer = token_mixer
        self.pos_embedding_type = pos_embedding

        if use_sds_frontend:
            frontend_kwargs = dict(
                d_model=self.d_model,
                dilations=frontend_dilations,
                dropout=dropout,
                use_haar_wavelet=use_haar_wavelet,
                use_identity_branch=use_identity_branch,
                simple_last_down=simple_last_down,
                use_sk_fusion=use_sk_fusion,
                cross_scale_mode=cross_scale_mode,
                wavelet_downsample=wavelet_downsample,
            )
            self.frontend = SDSFrontEnd1D(**frontend_kwargs)
        elif matched_conv_frontend:
            self.frontend = MatchedConvFrontEnd1D(d_model=self.d_model, dropout=dropout)
        else:
            self.frontend = PlainConvFrontEnd1D(d_model=self.d_model, dropout=dropout)

        if pos_embedding == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        elif pos_embedding == "sinusoidal":
            self.register_buffer("pos_embedding", _build_sinusoidal_pe(self.max_len, self.d_model), persistent=False)
        elif pos_embedding == "none":
            self.pos_embedding = None
        else:
            raise ValueError(f"Unsupported pos_embedding: {pos_embedding}. Use 'learned', 'sinusoidal' or 'none'.")

        self.encoder = nn.ModuleList([
            DSFBEncoderLayerV2(
                dim=self.d_model,
                seq_len=self.max_len,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                token_mixer=token_mixer,
                nhead=nhead,
                dsfb_num_heads=dsfb_num_heads,
                dsfb_freq_kernel_size=dsfb_freq_kernel_size,
                ffn_type=ffn_type,
                ffn_periodic_init_freq=ffn_periodic_init_freq,
                moh_num_heads=moh_num_heads,
                moh_rank=moh_rank,
                moh_balance_loss_weight=moh_balance_loss_weight,
                moh_expert_strength=moh_expert_strength,
                moh_proj_rank=moh_proj_rank,
            )
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        if simple_head:
            self.head = nn.Linear(self.d_model, num_classes)
        else:
            self.head = nn.Sequential(nn.Linear(self.d_model, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def aux_loss(self) -> torch.Tensor:
        losses = [layer.aux_loss() for layer in self.encoder if hasattr(layer, "aux_loss")]
        if not losses:
            return torch.zeros((), device=self.final_norm.weight.device)
        return torch.stack([loss.to(device=self.final_norm.weight.device) for loss in losses]).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        if self.pos_embedding is not None:
            seq_len = x.shape[1]
            if seq_len <= self.max_len:
                pos = self.pos_embedding[:, :seq_len, :]
            else:
                pos = F.interpolate(self.pos_embedding.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
            x = x + pos
        for layer in self.encoder:
            x = layer(x)
        x = self.final_norm(x).mean(dim=1)
        return self.head(x)


SDSDSFBTransformer = RFESFNet


if __name__ == "__main__":
    for mixer in ("no_phase", "self_attention", "dsfb"):
        model = RFESFNet(
            num_classes=10,
            d_model=128,
            num_layers=1,
            token_mixer=mixer,
            pos_embedding="none",
            moh_num_heads=6,
            moh_rank=4,
            moh_proj_rank=8,
        )
        x = torch.randn(2, 1, 2048)
        y = model(x)
        print(mixer, tuple(y.shape), sum(p.numel() for p in model.parameters()), "aux", float(model.aux_loss()))

