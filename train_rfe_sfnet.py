import argparse

import torch

from models.rfe_sfnet import RFESFNet
from train_common import (
    ExperimentConfig,
    add_common_args,
    maybe_report_model_stats,
    run_experiment,
    set_seed,
)

try:
    from thop import clever_format, profile
    THOP_AVAILABLE = True
except ImportError:
    clever_format = None
    profile = None
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Install with 'pip install thop' for FLOPs calculation.")


def parse_dilations(dilations_text: str):
    return tuple(int(d) for d in dilations_text.split(","))


def add_bool_arg(parser, name: str, default: bool, help_text: str):
    parser.add_argument(
        f"--{name}",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="RFE-SFNet for bearing fault diagnosis")
    add_common_args(parser)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=1, help="Number of DSFB encoder layers")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum token length after the front-end")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_ratio", type=float, default=1.5)
    parser.add_argument("--nhead", type=int, default=4, help="Number of heads for self-attention ablation")
    parser.add_argument("--token_mixer", type=str, default="no_phase",
                        choices=["no_phase", "dsfb", "self_attention", "csmoh_plus", "competitive_moh_dsfb"],
                        help="Token mixer type: no_phase (default), dsfb, self_attention, csmoh_plus, or competitive_moh_dsfb")
    parser.add_argument("--use_mhsa", action="store_true",
                        help="Ablation: replace DSFB with standard multi-head self-attention")
    parser.add_argument("--dsfb_num_heads", type=int, default=1,
                        help="Number of spectral heads in DSFB (1 = original, 4 recommended for MH-DSFB)")
    parser.add_argument("--dsfb_freq_kernel_size", type=int, default=1,
                        help="Frequency-axis depthwise conv kernel inside DSFB (1 = disabled, 5 recommended)")
    parser.add_argument("--moh_num_heads", type=int, default=6,
                        help="Number of routed spectral heads for no-phase CSMoH")
    parser.add_argument("--moh_rank", type=int, default=4,
                        help="Low-rank factor count for no-phase CSMoH")
    parser.add_argument("--moh_balance_loss_weight", type=float, default=0.003,
                        help="Kept for old commands; ignored by all-head no-phase CSMoH")
    parser.add_argument("--moh_expert_strength", type=float, default=0.5,
                        help="No-phase CSMoH spectral magnitude expert strength")
    parser.add_argument("--moh_proj_rank", type=int, default=8,
                        help="Head-specific low-rank channel recombination rank for no-phase CSMoH")
    parser.add_argument("--ffn_type", type=str, default="li_bottleneck",
                        choices=["li_bottleneck", "swiglu", "periodic"],
                        help="FFN inside DSFB encoder layer")
    parser.add_argument("--ffn_periodic_init_freq", type=float, default=1.0,
                        help="Initial per-channel frequency for the periodic FFN")
    add_bool_arg(parser, "simple_down4", True,
                 "Use parameter-free AvgPool for the last 128->128 downsample by default")
    add_bool_arg(parser, "simple_head", True,
                 "Use a single Linear(d_model -> num_classes) head by default")
    parser.add_argument("--pos_embedding", type=str, default="none",
                        choices=["learned", "sinusoidal", "none"],
                        help="Positional embedding type")
    parser.add_argument("--no_haar_wavelet", action="store_true",
                        help="Ablation: replace Haar wavelet downsample with a plain strided conv")
    parser.add_argument("--no_identity_branch", action="store_true",
                        help="Ablation: drop the identity (skip) branch inside SDS blocks")
    parser.add_argument("--no_sk_fusion", action="store_true", default=True,
                        help="Replace SKFusion1D with parameter-free SumFusion1D (default)")
    parser.add_argument("--sk_fusion", dest="no_sk_fusion", action="store_false",
                        help="Enable SKFusion1D branch fusion for ablation runs")
    parser.add_argument("--frontend_dilations", type=str, default="1,4,12",
                        help="Comma-separated dilations for RFE-Stem")
    parser.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10,12")
    return parser


def build_model(args):
    frontend_dilations = parse_dilations(args.frontend_dilations)
    token_mixer = "self_attention" if args.use_mhsa else args.token_mixer
    model = RFESFNet(
        num_classes=args.num_classes,
        d_model=args.d_model,
        num_layers=args.n_blocks,
        max_len=args.max_len,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        frontend_dilations=frontend_dilations,
        token_mixer=token_mixer,
        nhead=args.nhead,
        dsfb_num_heads=args.dsfb_num_heads,
        dsfb_freq_kernel_size=args.dsfb_freq_kernel_size,
        moh_num_heads=args.moh_num_heads,
        moh_rank=args.moh_rank,
        moh_balance_loss_weight=args.moh_balance_loss_weight,
        moh_expert_strength=args.moh_expert_strength,
        moh_proj_rank=args.moh_proj_rank,
        use_haar_wavelet=not args.no_haar_wavelet,
        use_identity_branch=not args.no_identity_branch,
        ffn_type=args.ffn_type,
        ffn_periodic_init_freq=args.ffn_periodic_init_freq,
        simple_last_down=args.simple_down4,
        simple_head=args.simple_head,
        pos_embedding=args.pos_embedding,
        use_sk_fusion=not args.no_sk_fusion,
    )
    return model, frontend_dilations, token_mixer


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("RFE-SFNet Training")
    print("=" * 60)

    model, frontend_dilations, token_mixer = build_model(args)
    model = model.to(device)
    print(f"[INFO] RFE-Stem dilations: {frontend_dilations}")
    print(f"[INFO] SF Blocks: {args.n_blocks}, d_model={args.d_model}, mlp_ratio={args.mlp_ratio}")
    print(f"[INFO] token_mixer={token_mixer}, nhead={args.nhead}, dsfb_num_heads={args.dsfb_num_heads}")
    print(f"[INFO] dsfb_freq_kernel_size={args.dsfb_freq_kernel_size}, haar_wavelet={not args.no_haar_wavelet}")
    print(f"[INFO] identity_branch={not args.no_identity_branch}")
    print(f"[INFO] simple_down4={args.simple_down4}, simple_head={args.simple_head}, pos_embedding={args.pos_embedding}")
    print(f"[INFO] sk_fusion={not args.no_sk_fusion}, moh_heads={args.moh_num_heads}, moh_rank={args.moh_rank}, moh_proj_rank={args.moh_proj_rank}")
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    is_mhsa = token_mixer == "self_attention"
    config = ExperimentConfig(
        model_key="rfe_sfnet_mhsa" if is_mhsa else "rfe_sfnet",
        model_display_name="RFE-SFNet-MHSA" if is_mhsa else "RFE-SFNet",
        confusion_title=(
            "Confusion Matrix - RFE-SFNet-MHSA"
            if is_mhsa
            else "Confusion Matrix - RFE-SFNet"
        ),
        noise_plot_title=(
            "RFE-SFNet-MHSA Robustness Under Different Noise Levels"
            if is_mhsa
            else "RFE-SFNet Robustness Under Different Noise Levels"
        ),
        history_title_suffix="RFE-SFNet-MHSA" if is_mhsa else "RFE-SFNet",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

