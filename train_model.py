import argparse

import torch

from models.almformer import ALMformer
from models.cnn_transformer import CNNTransformer
from models.convformer_nse import ConvformerNSE
from models.drsn_cw import DRSN_CW, DRSN_CW_Lite
from models.gtfenet import GTFENET
from models.liconvformer import Liconvformer
from models.mslk_transformer import MSLKTransformer
from models.rfe_sfnet import RFESFNet
from models.tslanet import TSLANet
from models.wdcnn import WDCNN
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


def add_base_subparser(subparsers, name: str, description: str, aliases=None):
    parser = subparsers.add_parser(name, aliases=aliases or [], help=description, description=description)
    add_common_args(parser)
    return parser


def parse_kernel_sizes(kernel_sizes_text: str):
    return [int(k) for k in kernel_sizes_text.split(",")]


def parse_dilations(dilations_text: str):
    return tuple(int(d) for d in dilations_text.split(","))


def add_bool_arg(parser, name: str, default: bool, help_text: str):
    parser.add_argument(
        f"--{name}",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def resolve_tsla_flags(args):
    use_icb = args.use_icb and not args.no_icb
    use_asb = args.use_asb and not args.no_asb
    adaptive_filter = args.adaptive_filter and not args.no_adaptive_filter
    return use_icb, use_asb, adaptive_filter


def print_mslk_mode(args, kernel_sizes):
    print(f"[ABLATION] Multi-Scale Large Kernel sizes: {kernel_sizes}")
    print("[ABLATION] GTG and Soft Thresholding are REMOVED in this version")
    if args.no_gated_sk:
        print("[ABLATION] 浣跨敤鍘熷 MSLK Block (Concat铻嶅悎)")
    else:
        print("[INFO] 浣跨敤 MS-SK Block (DepthwiseConv + SelectiveKernelFusion)")

    if args.use_asb:
        print("[INFO] 浣跨敤 DSFB (鍔ㄦ€侀璋辫瀺鍚堝潡) 鏇夸唬澶氬ご鑷敞鎰忓姏")
        print("[INFO] DSFB features: cross-channel interaction + soft attention + spectral filtering")
        if args.use_icb:
            print("[INFO] 浣跨敤 ICB (浜や簰寮忓嵎绉潡) 鏇夸唬鏍囧噯 FFN")
        else:
            print("[INFO] 浣跨敤鏍囧噯 FFN")
    else:
        print("[INFO] 浣跨敤鏍囧噯 Transformer (澶氬ご鑷敞鎰忓姏)")
        if args.use_icb:
            print("[WARNING] --use_icb only works with --use_asb; ignoring it here")


def run_cnn_transformer(args, device):
    model = CNNTransformer(
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.n_blocks,
        dropout=args.dropout,
    ).to(device)
    config = ExperimentConfig(
        model_key="cnn_transformer",
        model_display_name="CNN-Transformer",
        confusion_title="Confusion Matrix",
        noise_plot_title="Model Robustness Under Different Noise Levels",
    )
    run_experiment(args, model, device, config)


def run_convformer_nse(args, device):
    print("=" * 60)
    print("Convformer-NSE Training Script")
    print("=" * 60)
    model = ConvformerNSE(in_channels=1, num_classes=args.num_classes).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="convformer_nse",
        model_display_name="Convformer-NSE",
        confusion_title="Confusion Matrix - Convformer-NSE",
        noise_plot_title="Convformer-NSE Robustness Under Different Noise Levels",
        history_title_suffix="Convformer-NSE",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


def run_liconvformer(args, device):
    model = Liconvformer(
        None,
        in_channel=1,
        out_channel=args.num_classes,
        drop=args.drop,
        dim=args.dim,
    ).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="liconvformer",
        model_display_name="Liconvformer",
        confusion_title="Confusion Matrix - Liconvformer",
        noise_plot_title="Model Robustness Under Different Noise Levels",
    )
    run_experiment(args, model, device, config)


def run_almformer(args, device):
    print("=" * 60)
    print("ALMformer Training")
    print("=" * 60)
    model = ALMformer(num_classes=args.num_classes, depth=args.depth).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="almformer",
        model_display_name="ALMformer",
        confusion_title="Confusion Matrix - ALMformer",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="ALMformer",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


def run_wdcnn(args, device):
    print("=" * 60)
    print("WDCNN: Wide Deep Convolutional Neural Network")
    print("=" * 60)
    use_dropout = not args.no_dropout
    model = WDCNN(num_classes=args.num_classes, use_dropout=use_dropout, dropout=args.dropout).to(device)
    print(f"\nModel: WDCNN (use_dropout={use_dropout}, dropout={args.dropout})")
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="wdcnn",
        model_display_name="WDCNN",
        confusion_title="Confusion Matrix - WDCNN",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="WDCNN",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


def run_tslanet(args, device):
    print("=" * 60)
    print("TSLANet Training")
    print("=" * 60)
    use_icb, use_asb, adaptive_filter = resolve_tsla_flags(args)
    print(f"[INFO] ICB: {use_icb}, ASB: {use_asb}, Adaptive Filter: {adaptive_filter}")
    model = TSLANet(
        seq_len=args.window_size,
        num_channels=1,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        dropout_rate=args.dropout,
        use_icb=use_icb,
        use_asb=use_asb,
        adaptive_filter=adaptive_filter,
    ).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="tslanet",
        model_display_name="TSLANet",
        confusion_title="Confusion Matrix - TSLANet",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="TSLANet",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


def run_drsn_cw(args, device):
    print("=" * 60)
    print("DRSN-CW Training (Deep Residual Shrinkage Network)")
    print("=" * 60)
    print(f"[INFO] Model: {'DRSN-CW-Lite' if args.lite else 'DRSN-CW'}")
    print(f"[INFO] Base channels: {args.base_channels}")
    if args.lite:
        model = DRSN_CW_Lite(
            num_classes=args.num_classes,
            in_channels=1,
            base_channels=args.base_channels,
            num_blocks=[1, 1, 1, 1],
        ).to(device)
    else:
        model = DRSN_CW(
            num_classes=args.num_classes,
            in_channels=1,
            base_channels=args.base_channels,
            num_blocks=[2, 2, 2, 2],
        ).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="drsn_cw",
        model_display_name="DRSN-CW",
        confusion_title="Confusion Matrix - DRSN-CW",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="DRSN-CW",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


def run_gtfenet(args, device):
    model = GTFENET(num_classes=args.num_classes, input_length=args.window_size).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="gtfenet",
        model_display_name="GTFENET",
        confusion_title="Confusion Matrix - GTFENET",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="GTFENET",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


def run_mslk(args, device):
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)
    model = MSLKTransformer(
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.n_blocks,
        dropout=args.dropout,
        kernel_sizes=kernel_sizes,
        use_gated_sk=not args.no_gated_sk,
        use_asb=args.use_asb,
        use_icb=args.use_icb,
        channel_mixer_groups=args.channel_mixer_groups,
        use_channel_mixer=not args.no_channel_mixer,
    ).to(device)
    print_mslk_mode(args, kernel_sizes)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)
    config = ExperimentConfig(
        model_key="mslk_ablation",
        model_display_name="MSLK Ablation",
        confusion_title="Confusion Matrix - MSLK Ablation (No GTG/Soft-Thresh)",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="MSLK Ablation",
        classification_zero_division=0,
        best_model_filename="best_1_12.pth",
        history_plot_filename="training_history_mslk_ablation.png",
        noise_plot_filename="noise_robustness_mslk_ablation.png",
        confusion_matrix_filename="confusion_matrix_mslk_ablation.png",
        history_csv_template="results/training_history_mslk_snr{train_snr_min}_{train_snr_max}.csv",
    )
    run_experiment(args, model, device, config)


def run_rfe_sfnet(args, device):
    frontend_dilations = parse_dilations(args.frontend_dilations)
    token_mixer = "self_attention" if args.use_mhsa else args.token_mixer
    use_lean_csmoh = token_mixer in ("no_phase", "csmoh_plus", "competitive_moh_dsfb")
    print("=" * 60)
    print("RFE-SFNet Training")
    print("=" * 60)
    print(f"[INFO] RFE-Stem dilations: {frontend_dilations}")
    print(f"[INFO] SF Blocks: {args.n_blocks}, d_model={args.d_model}, mlp_ratio={args.mlp_ratio}")
    print(
        f"[INFO] token_mixer={token_mixer}, nhead={args.nhead}, "
        f"dsfb_num_heads={args.dsfb_num_heads}, moh_heads={args.moh_num_heads}, "
        f"moh_rank={args.moh_rank}"
    )
    if use_lean_csmoh:
        print(
            f"[INFO] no-phase CSMoH expert_strength={args.moh_expert_strength}, "
            f"proj_rank={args.moh_proj_rank}, routing=all-head softmax"
        )
    print(
        f"[INFO] dsfb_freq_kernel_size={args.dsfb_freq_kernel_size}, "
        f"haar_wavelet={not args.no_haar_wavelet}, wavelet_downsample={args.wavelet_downsample}"
    )
    print(f"[INFO] identity_branch={not args.no_identity_branch}")
    print(f"[INFO] ffn_type={args.ffn_type}, ffn_periodic_init_freq={args.ffn_periodic_init_freq}")
    print(f"[INFO] simple_down4={args.simple_down4}, simple_head={args.simple_head}, pos_embedding={args.pos_embedding}")
    print(
        f"[INFO] sds_frontend={not args.no_sds_frontend}, "
        f"matched_conv_frontend={args.matched_conv_frontend}, sk_fusion={not args.no_sk_fusion}"
    )
    print(
        f"[INFO] cross_scale_mode={args.cross_scale_mode}, "
        "sum_fusion=sum, cross_scale_coeff=1, haar_refine=residual_dwconv"
    )

    model_kwargs = dict(
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
        use_haar_wavelet=not args.no_haar_wavelet,
        use_identity_branch=not args.no_identity_branch,
        ffn_type=args.ffn_type,
        ffn_periodic_init_freq=args.ffn_periodic_init_freq,
        simple_last_down=args.simple_down4,
        simple_head=args.simple_head,
        pos_embedding=args.pos_embedding,
        use_sds_frontend=not args.no_sds_frontend,
        matched_conv_frontend=args.matched_conv_frontend,
        use_sk_fusion=not args.no_sk_fusion,
        cross_scale_mode=args.cross_scale_mode,
        wavelet_downsample=args.wavelet_downsample,
        moh_proj_rank=args.moh_proj_rank,
    )
    if use_lean_csmoh:
        model_kwargs.update(
            moh_expert_strength=args.moh_expert_strength,
        )
    model = RFESFNet(**model_kwargs).to(device)
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified training entry for PU-BS-10 and CWRU experiments.",
    )
    subparsers = parser.add_subparsers(dest="model", required=True)

    cnn = add_base_subparser(subparsers, "cnn_transformer", "Train CNN-Transformer baseline", aliases=["cnn"])
    cnn.add_argument("--n_blocks", type=int, default=3)
    cnn.add_argument("--d_model", type=int, default=128)
    cnn.add_argument("--nhead", type=int, default=4)
    cnn.add_argument("--dropout", type=float, default=0.1)
    cnn.set_defaults(run_fn=run_cnn_transformer)

    convformer = add_base_subparser(subparsers, "convformer_nse", "Train Convformer-NSE", aliases=["convformer"])
    convformer.set_defaults(snr_list="-12,-11,-10,-8,-6,-5,-4,-2,0,2,4,6,8,10,12")
    convformer.set_defaults(run_fn=run_convformer_nse)

    liconvformer = add_base_subparser(subparsers, "liconvformer", "Train Liconvformer")
    liconvformer.add_argument("--dim", type=int, default=16,
                              help="Base dim; final embedding is 8*dim. Use 16 to get d_model=128.")
    liconvformer.add_argument("--drop", type=float, default=0.1)
    liconvformer.set_defaults(run_fn=run_liconvformer)

    almformer = add_base_subparser(subparsers, "almformer", "Train ALMformer")
    almformer.add_argument("--depth", type=int, default=4, help="MetaFormer Blocks 鏁伴噺")
    almformer.add_argument("--dropout", type=float, default=0.1)
    almformer.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10")
    almformer.set_defaults(run_fn=run_almformer)

    wdcnn = add_base_subparser(subparsers, "wdcnn", "Train WDCNN")
    wdcnn.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in classifier")
    wdcnn.add_argument("--no_dropout", action="store_true", help="Disable dropout")
    wdcnn.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10,12")
    wdcnn.set_defaults(run_fn=run_wdcnn)

    tslanet = add_base_subparser(subparsers, "tslanet", "Train TSLANet")
    tslanet.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    tslanet.add_argument("--depth", type=int, default=2, help="Number of TSLANet layers")
    tslanet.add_argument("--patch_size", type=int, default=8, help="Patch size")
    tslanet.add_argument("--dropout", type=float, default=0.15)
    tslanet.add_argument("--use_icb", action="store_true", default=True,
                         help="Use ICB (Interactive Convolutional Block)")
    tslanet.add_argument("--no_icb", action="store_true", help="Disable ICB")
    tslanet.add_argument("--use_asb", action="store_true", default=True,
                         help="Use ASB (Adaptive Spectral Block)")
    tslanet.add_argument("--no_asb", action="store_true", help="Disable ASB")
    tslanet.add_argument("--adaptive_filter", action="store_true", default=True,
                         help="Use adaptive filter in ASB")
    tslanet.add_argument("--no_adaptive_filter", action="store_true",
                         help="Disable adaptive filter in ASB")
    tslanet.set_defaults(run_fn=run_tslanet)

    drsn = add_base_subparser(subparsers, "drsn_cw", "Train DRSN-CW", aliases=["drsn"])
    drsn.add_argument("--base_channels", type=int, default=32, help="Base channel count")
    drsn.add_argument("--lite", action="store_true", help="Use the lightweight DRSN-CW model")
    drsn.set_defaults(snr_list="-12,-10,-9,-8,-6,-5,-4,-2,0,2,4,6,8,10")
    drsn.set_defaults(run_fn=run_drsn_cw)

    gtfenet = add_base_subparser(subparsers, "gtfenet", "Train GTFENET")
    gtfenet.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10")
    gtfenet.set_defaults(run_fn=run_gtfenet)

    mslk = add_base_subparser(subparsers, "mslk", "Train custom MSLK model", aliases=["mslk_ablation"])
    mslk.add_argument("--n_blocks", type=int, default=1)
    mslk.add_argument("--d_model", type=int, default=128)
    mslk.add_argument("--nhead", type=int, default=4)
    mslk.add_argument("--dropout", type=float, default=0.1)
    mslk.add_argument("--kernel_sizes", type=str, default="15,31,51",
                      help="Comma-separated kernel sizes for MG-STL blocks")
    mslk.add_argument("--no_gated_sk", action="store_true",
                      help="娑堣瀺瀹為獙锛氱鐢⊿K铻嶅悎锛屼娇鐢ㄥ師濮婥oncat铻嶅悎")
    mslk.add_argument("--use_asb", action="store_true",
                      help="浣跨敤 DSFB (鍔ㄦ€侀璋辫瀺鍚堝潡) 鏇夸唬澶氬ご鑷敞鎰忓姏鏈哄埗")
    mslk.add_argument("--use_icb", action="store_true",
                      help="浣跨敤 ICB (浜や簰寮忓嵎绉潡) 鏇夸唬鏍囧噯 FFN (浠呭湪 use_asb 鏃舵湁鏁?")
    mslk.add_argument("--channel_mixer_groups", type=int, default=4,
                      help="Group count for the Adaptive_Spectral_Block channel mixer")
    mslk.add_argument("--no_channel_mixer", action="store_true",
                      help="娑堣瀺瀹為獙: 绂佺敤 DSFB 涓殑棰戝煙 Channel Mixer")
    mslk.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10,12")
    mslk.set_defaults(run_fn=run_mslk)

    rfe_sfnet = add_base_subparser(
        subparsers,
        "rfesfnet",
        "Train RFE-SFNet",
        aliases=["rfe-sfnet", "rfe_sfnet", "sds_dsfb", "sds_dsfb_transformer", "sds"],
    )
    sds_dsfb = rfe_sfnet
    sds_dsfb.add_argument("--d_model", type=int, default=128)
    sds_dsfb.add_argument("--n_blocks", type=int, default=1, help="Number of SF Blocks")
    sds_dsfb.add_argument("--max_len", type=int, default=128, help="Maximum token length after RFE-Stem")
    sds_dsfb.add_argument("--dropout", type=float, default=0.1)
    sds_dsfb.add_argument("--mlp_ratio", type=float, default=1.5)
    sds_dsfb.add_argument("--nhead", type=int, default=4, help="Number of heads for self-attention ablation")
    sds_dsfb.add_argument("--token_mixer", type=str, default="no_phase",
                          choices=[
                              "no_phase",
                              "dsfb",
                              "self_attention",
                              "csmoh_plus",
                              "competitive_moh_dsfb",
                          ],
                          help="Token mixer type: no_phase (default CSMoH), dsfb baseline, "
                               "self_attention for MHSA ablation, csmoh_plus/competitive_moh_dsfb "
                               "as aliases for the same CSMoH module")
    sds_dsfb.add_argument("--use_mhsa", action="store_true",
                          help="Ablation: replace DSFB with standard multi-head self-attention")
    sds_dsfb.add_argument("--dsfb_num_heads", type=int, default=1,
                          help="Number of spectral heads in DSFB (1 = original, 4 recommended for MH-DSFB)")
    sds_dsfb.add_argument("--dsfb_freq_kernel_size", type=int, default=1,
                          help="Frequency-axis depthwise conv kernel inside DSFB (1 = disabled, 5 recommended)")
    sds_dsfb.add_argument("--moh_num_heads", type=int, default=6,
                          help="Number of routed spectral heads for no-phase CSMoH")
    sds_dsfb.add_argument("--moh_rank", type=int, default=4,
                          help="Low-rank factor count for no-phase CSMoH")
    sds_dsfb.add_argument("--moh_balance_loss_weight", type=float, default=0.003,
                          help="Kept for old commands; ignored by all-head no-phase CSMoH")
    sds_dsfb.add_argument("--moh_expert_strength", type=float, default=0.5,
                          help="No-phase CSMoH spectral magnitude expert strength")
    sds_dsfb.add_argument("--moh_proj_rank", type=int, default=8,
                          help="Head-specific low-rank channel recombination rank for no-phase CSMoH")
    sds_dsfb.add_argument("--ffn_type", type=str, default="li_bottleneck",
                          choices=["li_bottleneck", "swiglu", "periodic"],
                          help="FFN inside DSFB encoder layer: li_bottleneck (default), swiglu, or periodic")
    sds_dsfb.add_argument("--ffn_periodic_init_freq", type=float, default=1.0,
                          help="Initial per-channel frequency (alpha) for the periodic FFN")
    add_bool_arg(
        sds_dsfb,
        "simple_down4",
        True,
        "Use parameter-free AvgPool for the last 128->128 downsample by default",
    )
    add_bool_arg(
        sds_dsfb,
        "simple_head",
        True,
        "Use a single Linear(d_model -> num_classes) head by default",
    )
    sds_dsfb.add_argument("--pos_embedding", type=str, default="none",
                          choices=["learned", "sinusoidal", "none"],
                          help="Positional embedding: 'none' (default, A9 ablation), "
                               "'learned', or 'sinusoidal' (zero params)")
    sds_dsfb.add_argument("--no_haar_wavelet", action="store_true",
                          help="Ablation: replace Haar wavelet downsample with a plain strided conv")
    sds_dsfb.add_argument("--wavelet_downsample", type=str, default="haar_lpr",
                          choices=["haar_lpr"],
                          help="Haar downsampling rule. Only the Haar-LPR variant is retained.")
    sds_dsfb.add_argument("--no_identity_branch", action="store_true",
                          help="Ablation: drop the identity (skip) branch inside SDS blocks")
    sds_dsfb.add_argument("--no_sds_frontend", action="store_true",
                          help="Ablation A1: replace RFE-Stem with a plain stem + strided conv front-end")
    sds_dsfb.add_argument("--matched_conv_frontend", action="store_true",
                          help="Ablation A1B: use a parameter-matched plain Conv front-end when --no_sds_frontend is set")
    sds_dsfb.add_argument(
        "--no_sk_fusion",
        action="store_true",
        default=True,
        help="Replace SKFusion1D with parameter-free SumFusion1D (default)",
    )
    sds_dsfb.add_argument(
        "--sk_fusion",
        dest="no_sk_fusion",
        action="store_false",
        help="Enable SKFusion1D branch fusion for ablation runs",
    )
    sds_dsfb.add_argument("--cross_scale_mode", type=str, default="gap",
                          choices=["full", "gap"],
                          help="Cross-scale coupling between dilated branches: "
                               "'gap' (default, current) adds the time-pooled previous-scale summary with unit "
                               "coefficient; 'full' adds the entire previous-scale tensor.")
    sds_dsfb.add_argument("--frontend_dilations", type=str, default="1,4,12",
                          help="Comma-separated dilations for RFE-Stem")
    sds_dsfb.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10,12")
    sds_dsfb.set_defaults(run_fn=run_rfe_sfnet)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.run_fn(args, device)


if __name__ == "__main__":
    main()

