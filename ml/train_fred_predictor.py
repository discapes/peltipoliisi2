#!/usr/bin/env python3
"""
Minimal GRU-based trajectory predictor trained on FRED coordinate tracks.

It expects the FRED dataset to be extracted next to this repository:
../FRED/<sequence_id>/coordinates.txt

Example:
  python ml/train_fred_predictor.py --sequences 0 1 2 --val-sequences 3 \
      --input-len 20 --pred-len 10 --epochs 15
"""
from __future__ import annotations

import argparse
import math
import os
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrajectorySample:
    positions: np.ndarray  # (T, 2) normalized xy
    timestamps: np.ndarray  # (T,)


def parse_coordinates_file(path: Path,
                           image_width: float,
                           image_height: float) -> TrajectorySample:
    """Parse FRED coordinates.txt into normalized center positions."""
    timestamps: List[float] = []
    centers: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                time_str, coords_str = line.split(":")
                parts = [float(tok.strip()) for tok in coords_str.split(",")]
                if len(parts) < 4:
                    continue
                x1, y1, x2, y2 = parts[:4]
            except ValueError as e:
                continue
            cx = ((x1 + x2) * 0.5) / image_width
            cy = ((y1 + y2) * 0.5) / image_height
            timestamps.append(float(time_str))
            centers.append((cx, cy))
    if not timestamps:
        raise ValueError(f"No coordinates found in {path}")
    positions = np.asarray(centers, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float32)
    return TrajectorySample(positions=positions, timestamps=times)


class SlidingWindowDataset(Dataset):
    def __init__(self,
                 samples: Sequence[TrajectorySample],
                 input_len: int,
                 pred_len: int,
                 stride: int = 1):
        self.inputs: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        for seq in samples:
            num_steps = seq.positions.shape[0]
            for start in range(0, num_steps - input_len - pred_len, stride):
                inp = seq.positions[start:start + input_len]
                tgt = seq.positions[start + input_len:start + input_len + pred_len]
                # Normalize relative to last input position for stability
                base = inp[-1:]
                self.inputs.append(inp - base)
                self.targets.append(tgt - base)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.inputs[idx]),
                torch.from_numpy(self.targets[idx]))


class TrajectoryGRU(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 pred_len: int,
                 input_len: int,
                 *,
                 num_layers: int,
                 dropout: float,
                 mlp_hidden_mult: float,
                 layer_norm: bool):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=2,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=gru_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size) if layer_norm else None
        mlp_hidden = max(hidden_size, int(hidden_size * mlp_hidden_mult))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, pred_len * 2),
        )
        self.pred_len = pred_len
        self.input_len = input_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.view(-1, self.pred_len, 2)


def export_torchscript(model: TrajectoryGRU, path: Path) -> None:
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.eval()
    scripted = torch.jit.script(cpu_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(path.as_posix())
    print(f"Exported TorchScript model to {path}")


def split_sequences(all_sequences: Sequence[int],
                    holdout_sequences: Sequence[int]) -> Tuple[List[int], List[int]]:
    holdout_set = set(holdout_sequences)
    train = [s for s in all_sequences if s not in holdout_set]
    val = [s for s in all_sequences if s in holdout_set]
    if not train or not val:
        raise ValueError("Provide at least one train and one validation sequence id.")
    return train, val


def load_samples(sequence_ids: Iterable[int],
                 dataset_root: Path,
                 image_width: float,
                 image_height: float) -> List[TrajectorySample]:
    samples: List[TrajectorySample] = []
    for sid in sequence_ids:
        seq_dir = dataset_root / str(sid)
        coord_file = seq_dir / "coordinates.txt"
        if not coord_file.exists():
            raise FileNotFoundError(coord_file)
        sample = parse_coordinates_file(coord_file, image_width, image_height)
        samples.append(sample)
    return samples


def train(args: argparse.Namespace) -> None:
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.cpu and has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        print("Set PYTORCH_ENABLE_MPS_FALLBACK=0 to avoid silent CPU fallback.")
    print(f"Training on device: {device}")
    dataset_root = Path(args.dataset_root).resolve()
    all_seq_ids = args.sequences or [0, 1, 2, 3]
    val_seq_ids = args.val_sequences or [all_seq_ids[-1]]
    train_ids, val_ids = split_sequences(all_seq_ids, val_seq_ids)

    train_samples = load_samples(train_ids, dataset_root, args.image_width, args.image_height)
    val_samples = load_samples(val_ids, dataset_root, args.image_width, args.image_height)

    train_ds = SlidingWindowDataset(train_samples, args.input_len, args.pred_len, stride=args.stride)
    val_ds = SlidingWindowDataset(val_samples, args.input_len, args.pred_len, stride=args.stride)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=pin_mem)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=pin_mem)

    model = TrajectoryGRU(args.hidden_size,
                          args.pred_len,
                          args.input_len,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          mlp_hidden_mult=args.mlp_hidden_mult,
                          layer_norm=args.layer_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=args.scheduler_factor,
                                                           patience=args.scheduler_patience)
    criterion = nn.MSELoss()

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = (b.to(device) for b in batch)
                preds = model(inputs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * inputs.size(0)
        avg_val = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val)
        print(f"Epoch {epoch}: train={avg_train:.6f} val={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            ckpt_path = Path(args.output_dir) / "trajectory_gru.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(),
                        "input_len": args.input_len,
                        "pred_len": args.pred_len,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "mlp_hidden_mult": args.mlp_hidden_mult,
                        "layer_norm": args.layer_norm},
                       ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            export_torchscript(model, Path(args.torchscript_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simple trajectory predictor on FRED.")
    parser.add_argument("--dataset-root", type=str, default="../FRED",
                        help="Path containing sequence folders with coordinates.txt")
    parser.add_argument("--sequences", type=int, nargs="*", default=[0, 1, 2, 3],
                        help="Sequence IDs to load (must exist under dataset root).")
    parser.add_argument("--val-sequences", type=int, nargs="*", default=[3],
                        help="Sequence IDs reserved for validation.")
    parser.add_argument("--input-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp-hidden-mult", type=float, default=2.0,
                        help="Multiplier for the hidden dimension inside the MLP head.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm. Set <=0 to disable clipping.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--image-width", type=float, default=1280.0)
    parser.add_argument("--image-height", type=float, default=720.0)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--torchscript-path", type=str, default="artifacts/trajectory_gru.ts",
                        help="Where to store the TorchScript module alongside the checkpoint.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of worker processes for data loading.")
    parser.add_argument("--threads", type=int, default=0,
                        help="Override torch.set_num_threads (0 keeps PyTorch default).")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Apply LayerNorm to GRU outputs before the head.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU/MPS is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(args)


if __name__ == "__main__":
    main()

