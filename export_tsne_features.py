import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
    add_noise_snr,
    normalize_batch,
    prepare_dataloaders,
    set_seed,
)
from train_model import build_parser, parse_dilations, parse_kernel_sizes, resolve_tsla_flags


def _parse_args():
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    export_parser = argparse.ArgumentParser(add_help=False)
    export_parser.add_argument("--checkpoint", required=True, help="Path to the trained best .pth file.")
    export_parser.add_argument("--output_prefix", required=True, help="Output prefix without extension.")
    export_parser.add_argument("--export_snr", type=float, default=0.0,
                               help="SNR used when extracting test features. Use none for clean.")
    export_parser.add_argument("--export_noise_type", type=str, default="gaussian",
                               help="Noise type used when extracting test features.")
    export_parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    export_parser.add_argument("--tsne_iter", type=int, default=1000)
    export_args = export_parser.parse_args(unknown)

    for key, value in vars(export_args).items():
        setattr(args, key, value)
    if isinstance(args.export_snr, str) and args.export_snr.lower() == "none":
        args.export_snr = None
    return args


def _build_model(args, device):
    if args.model == "cnn_transformer":
        model = CNNTransformer(
            num_classes=args.num_classes,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.n_blocks,
            dropout=args.dropout,
        )
    elif args.model == "convformer_nse":
        model = ConvformerNSE(in_channels=1, num_classes=args.num_classes)
    elif args.model == "liconvformer":
        model = Liconvformer(None, in_channel=1, out_channel=args.num_classes,
                             drop=args.drop, dim=args.dim)
    elif args.model == "almformer":
        model = ALMformer(num_classes=args.num_classes, depth=args.depth)
    elif args.model == "wdcnn":
        model = WDCNN(num_classes=args.num_classes,
                      use_dropout=not args.no_dropout,
                      dropout=args.dropout)
    elif args.model == "tslanet":
        use_icb, use_asb, adaptive_filter = resolve_tsla_flags(args)
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
        )
    elif args.model == "drsn_cw":
        if args.lite:
            model = DRSN_CW_Lite(
                num_classes=args.num_classes,
                in_channels=1,
                base_channels=args.base_channels,
                num_blocks=[1, 1, 1, 1],
            )
        else:
            model = DRSN_CW(
                num_classes=args.num_classes,
                in_channels=1,
                base_channels=args.base_channels,
                num_blocks=[2, 2, 2, 2],
            )
    elif args.model == "gtfenet":
        model = GTFENET(num_classes=args.num_classes, input_length=args.window_size)
    elif args.model in ("mslk", "mslk_ablation"):
        model = MSLKTransformer(
            num_classes=args.num_classes,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.n_blocks,
            dropout=args.dropout,
            kernel_sizes=parse_kernel_sizes(args.kernel_sizes),
            use_gated_sk=not args.no_gated_sk,
            use_asb=args.use_asb,
            use_icb=args.use_icb,
            channel_mixer_groups=args.channel_mixer_groups,
            use_channel_mixer=not args.no_channel_mixer,
        )
    elif args.model in ("rfesfnet", "rfe-sfnet", "rfe_sfnet", "sds_dsfb", "sds_dsfb_transformer", "sds"):
        token_mixer = "self_attention" if args.use_mhsa else args.token_mixer
        model_kwargs = dict(
            num_classes=args.num_classes,
            d_model=args.d_model,
            num_layers=args.n_blocks,
            max_len=args.max_len,
            dropout=args.dropout,
            mlp_ratio=args.mlp_ratio,
            frontend_dilations=parse_dilations(args.frontend_dilations),
            token_mixer=token_mixer,
            nhead=args.nhead,
            dsfb_num_heads=args.dsfb_num_heads,
            dsfb_freq_kernel_size=args.dsfb_freq_kernel_size,
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
        )
        if token_mixer in ("no_phase", "csmoh_plus", "competitive_moh_dsfb"):
            model_kwargs.update(
                moh_num_heads=args.moh_num_heads,
                moh_rank=args.moh_rank,
                moh_balance_loss_weight=args.moh_balance_loss_weight,
                moh_expert_strength=args.moh_expert_strength,
                moh_proj_rank=args.moh_proj_rank,
            )
        model = RFESFNet(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model.to(device)


def _load_state_dict(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        state = {
            k.removeprefix("module."): v
            for k, v in state.items()
            if not k.endswith(".total_ops")
            and not k.endswith(".total_params")
            and k not in {"total_ops", "total_params"}
        }
    model.load_state_dict(state)


def _find_last_linear(model):
    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No nn.Linear layer found; cannot infer penultimate features.")
    return linear_layers[-1]


@torch.no_grad()
def _collect_features(model, loader, device, snr_db, noise_type):
    layer_name, layer = _find_last_linear(model)
    captured = []

    def hook(_module, inputs, _output):
        feat = inputs[0].detach()
        if feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1)
        captured.append(feat.cpu())

    handle = layer.register_forward_hook(hook)
    model.eval()
    features, labels, preds = [], [], []
    try:
        for x, y in loader:
            x = normalize_batch(x.to(device))
            if snr_db is not None:
                x = add_noise_snr(x, snr_db, noise_type=noise_type)
            captured.clear()
            logits = model(x)
            if not captured:
                raise RuntimeError(f"Feature hook on layer '{layer_name}' did not capture any tensor.")
            features.append(captured[-1])
            labels.append(y.cpu())
            preds.append(torch.argmax(logits, dim=1).cpu())
    finally:
        handle.remove()

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    return layer_name, features, labels, preds


def _write_features_csv(path, features, labels, preds, class_names):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["label", "label_name", "pred", "pred_name", "correct"]
        header.extend(f"f{i}" for i in range(features.shape[1]))
        writer.writerow(header)
        for feat, label, pred in zip(features, labels, preds):
            writer.writerow([
                int(label),
                class_names[int(label)] if int(label) < len(class_names) else str(label),
                int(pred),
                class_names[int(pred)] if int(pred) < len(class_names) else str(pred),
                int(label == pred),
                *[f"{v:.8g}" for v in feat],
            ])


def _write_tsne_csv(path, coords, labels, preds, class_names):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "label_name", "pred", "pred_name", "correct"])
        for xy, label, pred in zip(coords, labels, preds):
            writer.writerow([
                f"{xy[0]:.8g}",
                f"{xy[1]:.8g}",
                int(label),
                class_names[int(label)] if int(label) < len(class_names) else str(label),
                int(pred),
                class_names[int(pred)] if int(pred) < len(class_names) else str(pred),
                int(label == pred),
            ])


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t-SNE] Using device: {device}")
    print(f"[t-SNE] Checkpoint: {args.checkpoint}")
    print(f"[t-SNE] Export noise: type={args.export_noise_type}, snr={args.export_snr}")

    _train_loader, _val_loader, test_loader = prepare_dataloaders(args)
    class_names = getattr(args, "class_names", [str(i) for i in range(args.num_classes)])

    model = _build_model(args, device)
    _load_state_dict(model, args.checkpoint, device)
    layer_name, features, labels, preds = _collect_features(
        model,
        test_loader,
        device,
        snr_db=args.export_snr,
        noise_type=args.export_noise_type,
    )
    acc = float((labels == preds).mean() * 100.0)
    print(f"[t-SNE] Captured layer: {layer_name}")
    print(f"[t-SNE] Feature matrix: {features.shape}, test accuracy={acc:.2f}%")

    scaled = StandardScaler().fit_transform(features)
    perplexity = min(args.tsne_perplexity, max(5.0, (len(scaled) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=args.tsne_iter,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    coords = tsne.fit_transform(scaled)

    features_path = f"{args.output_prefix}_features.csv"
    tsne_path = f"{args.output_prefix}_tsne.csv"
    _write_features_csv(features_path, features, labels, preds, class_names)
    _write_tsne_csv(tsne_path, coords, labels, preds, class_names)
    print(f"[t-SNE] Feature CSV saved to: {features_path}")
    print(f"[t-SNE] Origin-ready t-SNE CSV saved to: {tsne_path}")


if __name__ == "__main__":
    main()

