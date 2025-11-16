# Event Camera Drone Odometry

This project visualizes DAT event streams, estimates rotor RPMs, and feeds the
resulting rotor-center tracks into a learned trajectory predictor for short-term
motion forecasting.

## Build & Run

```bash
# make sure Torch_DIR points to your LibTorch install (share/cmake/Torch)
meson setup build
meson compile -C build
./build/peltipoliisi2 data/drone_idle.dat [artifacts/trajectory_gru.ts]
```

- Argument 1: `<DAT filepath>` – required `.dat` event file paced in real time.
- Argument 2 (optional): TorchScript trajectory model. Defaults to
  `artifacts/trajectory_gru.ts` or can be supplied via `TRAJECTORY_MODEL`.

## Pipeline overview

1. **Event ingestion** – The reader thread replays DAT files in real time and
   maintains per-pixel event counts inside a sliding 50 ms window.
2. **RPM estimation** – A worker thread snapshots active events, clusters bright
   regions (mlpack DBSCAN), and estimates rotor RPM via FFT-based timestamps.
3. **Rotor tracking** – Bounding boxes receive persistent IDs with short RPM
   histories for HUD display.
4. **Trajectory forecasting** – The fused rotor centroid is pushed through a
   GRU-based predictor (TorchScript) that outputs the next `pred_len` steps,
   visualized as a green path in the renderer.

## Next steps

- Improve the centroid fusion (e.g., lift to 3D using stereo or depth priors).
- Blend multiple tracked targets into the predictor to handle occlusions.
- Add evaluation utilities comparing predicted vs. ground-truth trajectories.

## Learning-based motion prediction & runtime predictor

To experiment with a data-driven predictor using the FRED dataset, extract the
sequences next to this repo (e.g. `../FRED/0`, `../FRED/1`, …) and run:

```bash
cd ml
uv run train-fred-predictor --sequences 0 1 2 --val-sequences 3 \
    --input-len 20 --pred-len 10 --epochs 15
```

The script (`ml/train_fred_predictor.py`) parses each `coordinates.txt`, builds
sliding windows of normalized drone positions, and trains a small GRU to predict
future motion horizons. Whenever validation improves, two artifacts are saved:

- `artifacts/trajectory_gru.pt`: checkpoint with weights/state for continued
  training.
- `artifacts/trajectory_gru.ts`: TorchScript module consumed by the C++ runtime.

Use `--torchscript-path` to override the export destination. The module keeps
its `input_len`/`pred_len` metadata, so the C++ side auto-detects horizon
lengths when loading the file.

### Runtime overlay

When `peltipoliisi2` finds a TorchScript model (either via the optional second
CLI argument or the `TRAJECTORY_MODEL` environment variable), it continuously
feeds the fused rotor centroid into the GRU and overlays:

- **Orange path** – the most recent observation history (length = `input_len`).
- **Green path + dots** – the predicted future positions (length =
  `pred_len`).
- HUD text summarizing the current prediction horizon.

Leave the second CLI argument empty (or unset `TRAJECTORY_MODEL`) to disable the
predictor while keeping the rest of the pipeline intact.
