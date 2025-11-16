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
import hashlib
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
import expelliarmus


@dataclass
class TrajectorySample:
    positions: np.ndarray  # (T, 2) normalized xy
    timestamps: np.ndarray  # (T,)
    events: np.ndarray  # (T,) list of event arrays for each timestep
    event_frames: np.ndarray | None = None  # (T, 2, H, W)


def load_event_stream(seq_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load event stream from FRED sequence directory."""
    event_file = seq_dir / "Event" / "events.raw"
    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")

    # Load events using expelliarmus
    wizard = expelliarmus.Wizard(encoding="evt3")  # FRED uses EVT3 format
    events = wizard.read(event_file)

    # Convert timestamps to microseconds (they're already in microseconds in EVT3)
    timestamps = events['t'].astype(np.int64)
    x = events['x'].astype(np.uint16)
    y = events['y'].astype(np.uint16)
    polarity = events['p'].astype(np.uint8)

    # Get sensor dimensions (assume 1280x720 for FRED)
    sensor_width = 1280
    sensor_height = 720

    return timestamps, x, y, polarity, sensor_width, sensor_height


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
    return TrajectorySample(positions=positions, timestamps=times, events=None, event_frames=None)


class SlidingWindowDataset(Dataset):
    def __init__(self,
                 samples: Sequence[TrajectorySample],
                 input_len: int,
                 pred_len: int,
                 stride: int = 1):
        self.samples: Sequence[TrajectorySample] = samples
        self.input_len = input_len
        self.pred_len = pred_len
        self.indices: List[Tuple[int, int]] = []

        for seq_idx, seq in enumerate(samples):
            num_steps = seq.positions.shape[0]
            max_start = num_steps - input_len - pred_len
            if max_start <= 0:
                continue
            for start in range(0, max_start, stride):
                self.indices.append((seq_idx, start))
        if not self.indices:
            raise ValueError("No training windows generated; check sequence lengths or stride.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, start = self.indices[idx]
        seq = self.samples[seq_idx]
        pos = seq.positions[start:start + self.input_len]
        tgt = seq.positions[start + self.input_len:start + self.input_len + self.pred_len]

        base = pos[-1:]
        pos_rel = pos - base
        tgt_rel = tgt - base

        if seq.event_frames is None:
            raise ValueError("Event frames missing for sequence; event conditioning required.")
        event_frames = seq.event_frames[start:start + self.input_len]
        return (torch.from_numpy(pos_rel.astype(np.float32, copy=False)),
                torch.from_numpy(tgt_rel.astype(np.float32, copy=False)),
                torch.from_numpy(event_frames.astype(np.float32, copy=False)))


class EventEncoder(nn.Module):
    """Simple CNN encoder for event frames."""
    def __init__(self, input_channels: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # 1280x720 -> 640x360
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 640x360 -> 320x180
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 320x180 -> 160x90
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 1x1 (MPS-friendly)
            nn.Flatten(),  # -> 64
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)  # (B*T, hidden_dim)
        return features.view(B, T, -1)


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

        self.event_encoder = EventEncoder(hidden_dim=hidden_size // 2)
        position_input_size = 2 + (hidden_size // 2)

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=position_input_size,
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

    def forward(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        # pos: (B, T, 2), events: (B, T, 2, H, W)
        event_features = self.event_encoder(events)
        x = torch.cat([pos, event_features], dim=-1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.layer_norm(last)
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


EVENT_CACHE_VERSION = 1


def create_event_frames(events: np.ndarray,
                        coord_timestamps: np.ndarray,
                        sensor_width: int,
                        sensor_height: int,
                        *,
                        window_us: int,
                        frame_height: int,
                        frame_width: int,
                        dtype: np.dtype = np.float16) -> np.ndarray:
    """Create downsampled event frames for each coordinate timestamp."""
    coord_ts_us = (coord_timestamps * 1e6).astype(np.int64)
    num_frames = coord_ts_us.shape[0]
    frames = np.zeros((num_frames, 2, frame_height, frame_width), dtype=np.float32)

    event_ts = events["t"]
    start_idx = 0
    end_idx = 0
    num_events = len(event_ts)
    x_scale = frame_width / sensor_width
    y_scale = frame_height / sensor_height

    for frame_idx, ts_us in enumerate(coord_ts_us):
        start_bound = ts_us - window_us // 2
        end_bound = ts_us + window_us // 2

        while start_idx < num_events and event_ts[start_idx] < start_bound:
            start_idx += 1
        if end_idx < start_idx:
            end_idx = start_idx
        while end_idx < num_events and event_ts[end_idx] < end_bound:
            end_idx += 1

        window_events = events[start_idx:end_idx]
        if window_events.size == 0:
            continue

        xs = np.floor(window_events["x"].astype(np.float32) * x_scale).astype(np.int32)
        ys = np.floor(window_events["y"].astype(np.float32) * y_scale).astype(np.int32)
        pols = window_events["p"].astype(np.int32)

        valid = (
            (xs >= 0) & (xs < frame_width) &
            (ys >= 0) & (ys < frame_height) &
            (pols >= 0) & (pols < 2)
        )
        if not np.any(valid):
            continue
        xs = xs[valid]
        ys = ys[valid]
        pols = pols[valid]

        np.add.at(frames[frame_idx], (pols, ys, xs), 1.0)

    return frames.astype(dtype, copy=False)


def _hash_array(arr: np.ndarray) -> str:
    hasher = hashlib.sha1()
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _cache_path(cache_dir: Path,
                sequence_id: int,
                window_us: int,
                frame_height: int,
                frame_width: int,
                timestamps_hash: str) -> Path:
    filename = (
        f"seq{sequence_id}_v{EVENT_CACHE_VERSION}_"
        f"win{window_us}_fh{frame_height}_fw{frame_width}_"
        f"{timestamps_hash}.npz"
    )
    return cache_dir / filename


def _load_cached_event_frames(cache_path: Path,
                              expected_len: int,
                              timestamps: np.ndarray) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as data:
            cached_ts = data["timestamps"]
            if cached_ts.shape != timestamps.shape or not np.allclose(cached_ts, timestamps):
                return None
            cached_frames = data["event_frames"]
            if cached_frames.shape[0] != expected_len:
                return None
            return cached_frames
    except Exception:
        return None


def _save_cached_event_frames(cache_path: Path,
                              event_frames: np.ndarray,
                              timestamps: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, event_frames=event_frames, timestamps=timestamps)


def load_samples(sequence_ids: Iterable[int],
                 dataset_root: Path,
                 image_width: float,
                 image_height: float,
                 *,
                 event_window_us: int,
                 event_frame_height: int,
                 event_frame_width: int,
                 cache_dir: Path | None,
                 use_cache: bool) -> List[TrajectorySample]:
    samples: List[TrajectorySample] = []
    for sid in sequence_ids:
        seq_dir = dataset_root / str(sid)
        coord_file = seq_dir / "coordinates.txt"
        if not coord_file.exists():
            raise FileNotFoundError(coord_file)

        # Load coordinates
        sample = parse_coordinates_file(coord_file, image_width, image_height)

        # Load events - required
        event_timestamps, event_x, event_y, event_polarity, sensor_width, sensor_height = load_event_stream(seq_dir)

        # Create structured array for events (similar to expelliarmus output)
        events = np.empty(len(event_timestamps), dtype=[('t', 'i8'), ('x', 'u2'), ('y', 'u2'), ('p', 'u1')])
        events['t'] = event_timestamps
        events['x'] = event_x
        events['y'] = event_y
        events['p'] = event_polarity

        cache_path = None
        if cache_dir and use_cache:
            ts_hash = _hash_array(sample.timestamps.astype(np.float32, copy=False))
            cache_path = _cache_path(cache_dir, sid, event_window_us,
                                     event_frame_height, event_frame_width, ts_hash)
            cached = _load_cached_event_frames(cache_path, sample.positions.shape[0], sample.timestamps)
            if cached is not None:
                sample.event_frames = cached
                print(f"[cache] Loaded event frames for sequence {sid} from {cache_path}")
                samples.append(sample)
                continue

        # Create event frames for each coordinate timestamp
        event_frames = create_event_frames(events,
                                           sample.timestamps,
                                           sensor_width,
                                           sensor_height,
                                           window_us=event_window_us,
                                           frame_height=event_frame_height,
                                           frame_width=event_frame_width)

        sample.events = events
        sample.event_frames = event_frames

        if cache_path and use_cache:
            _save_cached_event_frames(cache_path, sample.event_frames, sample.timestamps)
            print(f"[cache] Saved event frames for sequence {sid} to {cache_path}")

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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    dataset_root = Path(args.dataset_root).resolve()
    cache_dir: Path | None = None
    if not args.no_event_cache:
        cache_dir = Path(args.event_cache_dir).resolve()
    all_seq_ids = args.sequences or [0, 1, 2, 3]
    val_seq_ids = args.val_sequences or [all_seq_ids[-1]]
    train_ids, val_ids = split_sequences(all_seq_ids, val_seq_ids)

    load_kwargs = dict(
        event_window_us=args.event_window_us,
        event_frame_height=args.event_frame_height,
        event_frame_width=args.event_frame_width,
        cache_dir=cache_dir,
        use_cache=not args.no_event_cache,
    )

    train_samples = load_samples(train_ids, dataset_root, args.image_width, args.image_height, **load_kwargs)
    val_samples = load_samples(val_ids, dataset_root, args.image_width, args.image_height, **load_kwargs)

    train_ds = SlidingWindowDataset(train_samples, args.input_len, args.pred_len, stride=args.stride)
    val_ds = SlidingWindowDataset(val_samples, args.input_len, args.pred_len, stride=args.stride)

    pin_mem = device.type == "cuda"
    loader_extras: dict = {}
    if args.workers > 0:
        loader_extras["prefetch_factor"] = max(2, args.prefetch_factor)
        loader_extras["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=pin_mem,
                              **loader_extras)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=pin_mem,
                            **loader_extras)

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

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            pos_inputs, targets, event_inputs = batch
            pos_inputs = pos_inputs.to(device, non_blocking=pin_mem)
            targets = targets.to(device, non_blocking=pin_mem)
            event_inputs = event_inputs.to(device, non_blocking=pin_mem)
            if not use_amp:
                event_inputs = event_inputs.float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=autocast_dtype,
                                enabled=use_amp):
                preds = model(pos_inputs, event_inputs)
                loss = criterion(preds, targets)
            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            total_loss += loss.item() * pos_inputs.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pos_inputs, targets, event_inputs = batch
                pos_inputs = pos_inputs.to(device, non_blocking=pin_mem)
                targets = targets.to(device, non_blocking=pin_mem)
                event_inputs = event_inputs.to(device, non_blocking=pin_mem)
                if not use_amp:
                    event_inputs = event_inputs.float()
                with torch.autocast(device_type=device.type,
                                    dtype=autocast_dtype,
                                    enabled=use_amp):
                    preds = model(pos_inputs, event_inputs)
                    loss = criterion(preds, targets)
                val_loss += loss.item() * pos_inputs.size(0)
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
    parser.add_argument("--event-frame-width", type=int, default=320,
                        help="Width of cached event frames (after downsampling).")
    parser.add_argument("--event-frame-height", type=int, default=180,
                        help="Height of cached event frames (after downsampling).")
    parser.add_argument("--event-window-us", type=int, default=10000,
                        help="Temporal window size (microseconds) for event aggregation.")
    parser.add_argument("--event-cache-dir", type=str, default="artifacts/event_cache",
                        help="Directory used to cache precomputed event frames.")
    parser.add_argument("--no-event-cache", action="store_true",
                        help="Disable event frame caching (always recompute).")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--torchscript-path", type=str, default="artifacts/trajectory_gru.ts",
                        help="Where to store the TorchScript module alongside the checkpoint.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of worker processes for data loading.")
    parser.add_argument("--threads", type=int, default=0,
                        help="Override torch.set_num_threads (0 keeps PyTorch default).")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Apply LayerNorm to GRU outputs before the head.")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Number of batches to prefetch per worker.")
    parser.add_argument("--persistent-workers", action="store_true",
                        help="Keep dataloader workers alive between epochs.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable torch.cuda.amp mixed-precision training when on CUDA.")
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

