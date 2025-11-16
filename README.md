# Peltipoliisi2 – Real-Time Event Camera RPM & Motion Analyzer

**SensoFusion challenge submission**

High-speed object motion extraction using event camera streams (DAT format) with clustering and per-rotor RPM estimation. It can track propellers in real time using a C++ multithreading architecure and determine the propeller RPMs separately. We used DBSCAN from mlpack for clustering, and run fourier transforms with FFTW on randomly sampled 3x3 pixel areas from areas where the event density reaches a threshold, to determine the RPM. The dat file reading is paced at realtime speed and the event density matrix is updated continuously as the events come in. The rendering, clustering and fourier transforms are based on data  from a sliding window (deque where the events are "expired" out at the same realtime pace).

[See the demo video here](https://drive.google.com/file/d/1YmCVdrG5gTGRMeDZRMD2vSiJRBLuFc9r/view?usp=sharing)

## Usage:
Prerequisites: 
- g++
- meson
- OpenCV4 
- FFTW3
- mlpack.

Compile and run:
```bash
./run.sh
```

## Features
* Streams a DAT event file paced to original capture time and maintains a 50 ms sliding window of active events.
* Maintains a per-pixel event density matrix updated on event arrival (+1) and expiry (−1) to reflect live spatio‑temporal intensity.
* Randomly reservoir-samples candidate pixels exceeding an event threshold to form analysis points.
* Performs local FFT (FFTW) on microsecond timestamps around sampled pixels to estimate blade-pass frequency and infer RPM (supports configurable blade count).
* Runs DBSCAN (mlpack) on dense coordinates; aggregates RPM samples falling inside cluster boxes → per-rotor RPM.
* Tracks clusters frame-to-frame (IoU matching) assigning stable rotor IDs and producing smoothed RPM time series and mini inline graphs.
* Renders at target 60 FPS independent of event burst load (separate threads for reader, RPM/clustering, and renderer).
* Exposes global RPM statistic (temporal median of sample medians) plus per-rotor stats.
* Multi-threaded architecure:
    1. Event Reader: Parses DAT header, decodes 8-byte records, issues arrival/expiry callbacks respecting real-time pacing, maintains shared deque of active events.
    2. RPM & Clustering Worker: Every 1/60 s snapshots counts + active events, performs sampling → FFT RPM extraction → DBSCAN clustering → overlay publication.
    3. Renderer: Every 1/60 s builds visualization: white pixels where count ≥ threshold, draws cluster boxes, per-rotor history graphs, and textual stats.

**RPM Estimation**
1. For each sampled pixel region (3×3 by default), collect timestamps of events still active (inside 50 ms window).
2. Bin timestamps adaptively (bin width derived from median inter-event gap; size rounded to power-of-two with bounds) → detrend (remove DC) & apply Hann window.
3. Real-to-complex FFT (FFTW_ESTIMATE plan). Magnitude spectrum searched in plausible rotational band (5–5000 Hz).
4. Neighborhood sum scoring across bins stabilizes peak choice; harmonics (1/2 or 1/3 of dominant) accepted if within energy ratio (−4.4 dB equivalent) to prefer fundamentals.
5. Convert blade-pass frequency → rotor RPM (divide by blade count → ×60).
6. Median + short rolling median history smooth transient spikes.

**Clustering and tracking**
* Coordinates: all pixels whose count ≥ threshold in current snapshot.
* DBSCAN groups spatially proximate high-density regions (rotors, moving props).
* Cluster RPM: median (robust) of RPM samples inside bounding box; fallback to N/A if insufficient samples.
* Tracking: IoU-based assignment; rotors expire after configurable unseen frames (default 5). Per-rotor history graph aids stability assessment.


## Source files
| File | Purpose |
|------|---------|
| `meson.build` | Meson build config (C++20, dependencies: OpenCV, FFTW3, mlpack). |
| `src/defs.hpp` | Core types, constants, shared `FrameState`, forward declarations. |
| `src/event_reader.cpp` | DAT parsing, real-time paced streaming, sliding window maintenance. |
| `src/estimate_rpms.cpp` | Sampling, local event extraction, median & temporal smoothing, clustering orchestration. |
| `src/rpm_from_fft.cpp` | FFT-based rotor frequency → RPM estimation with leakage reduction & harmonic handling. |
| `src/cluster_worker.cpp` | DBSCAN clustering + bounding box + per-cluster RPM aggregation. |
| `src/render.cpp` | Visualization (counts mask, cluster boxes, per-rotor RPM history graphs, UI controls). |
| `src/main.cpp` | Thread orchestration & lifecycle. |