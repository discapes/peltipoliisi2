#!/usr/bin/env python3
"""
Evaluate a trained TrajectoryGRU checkpoint on held-out FRED sequences.

Computes several trajectory quality metrics (per-step MSE, ADE, FDE) using
the same preprocessing as the training script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_fred_predictor import (  # reuse dataset/model utilities
    SlidingWindowDataset,
    TrajectoryGRU,
    load_samples,
)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return per-sample per-step squared error and L2 error."""
    mse = (preds - targets) ** 2
    l2 = torch.linalg.norm(preds - targets, dim=-1)
    return mse, l2


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    dataset_root = Path(args.dataset_root).resolve()

    if args.sequences:
        seq_ids: Sequence[int] = args.sequences
    else:
        seq_ids = [0, 1, 2, 3]

    ckpt = torch.load(checkpoint, map_location=device)
    input_len = ckpt.get("input_len", args.input_len)
    pred_len = ckpt.get("pred_len", args.pred_len)
    hidden_size = ckpt.get("hidden_size", args.hidden_size)
    num_layers = ckpt.get("num_layers", args.num_layers)
    dropout = ckpt.get("dropout", args.dropout)
    mlp_hidden_mult = ckpt.get("mlp_hidden_mult", args.mlp_hidden_mult)
    layer_norm = ckpt.get("layer_norm", args.layer_norm)

    if (input_len != args.input_len or pred_len != args.pred_len or
            num_layers != args.num_layers or hidden_size != args.hidden_size):
        print("Note: overriding architecture hyperparameters with checkpoint values.")

    samples = load_samples(seq_ids, dataset_root, args.image_width, args.image_height)
    dataset = SlidingWindowDataset(samples,
                                   input_len,
                                   pred_len,
                                   stride=args.stride)
    if len(dataset) == 0:
        raise RuntimeError("Dataset produced zero evaluation windows. "
                           "Check input/pred lengths and available data.")

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))

    model = TrajectoryGRU(hidden_size,
                          pred_len,
                          input_len,
                          num_layers=num_layers,
                          dropout=dropout,
                          mlp_hidden_mult=mlp_hidden_mult,
                          layer_norm=layer_norm).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mse_acc: List[torch.Tensor] = []
    l2_acc: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets = (b.to(device) for b in batch)
            preds = model(inputs)
            mse, l2 = compute_metrics(preds, targets)
            mse_acc.append(mse.cpu())
            l2_acc.append(l2.cpu())

    mse_all = torch.cat(mse_acc)
    l2_all = torch.cat(l2_acc)
    per_step_mse = mse_all.mean().item()
    ade = l2_all.mean().item()
    fde = l2_all[:, -1].mean().item()

    print("Evaluation Summary")
    print("==================")
    print(f"Samples evaluated : {len(dataset)}")
    print(f"Per-step MSE      : {per_step_mse:.6f}")
    print(f"ADE (avg L2)      : {ade:.6f}")
    print(f"FDE (final L2)    : {fde:.6f}")

    if args.save_errors:
        np.savez(args.save_errors,
                 per_step_mse=mse_all.numpy(),
                 per_step_l2=l2_all.numpy())
        print(f"Saved per-sample errors to {args.save_errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained TrajectoryGRU checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/trajectory_gru.pt",
                        help="Path to the .pt checkpoint produced by training.")
    parser.add_argument("--dataset-root", type=str, default="../FRED",
                        help="Path containing sequence folders with coordinates.txt")
    parser.add_argument("--sequences", type=int, nargs="*",
                        help="Sequence IDs to evaluate on. Defaults to [0,1,2,3].")
    parser.add_argument("--input-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp-hidden-mult", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-width", type=float, default=1280.0)
    parser.add_argument("--image-height", type=float, default=720.0)
    parser.add_argument("--save-errors", type=str,
                        help="Optional path to npz file storing per-sample errors.")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Match LayerNorm usage if checkpoint metadata is missing.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU evaluation even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    evaluate(args)


if __name__ == "__main__":
    main()

