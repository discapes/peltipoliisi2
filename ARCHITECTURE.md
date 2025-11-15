# Event Visualizer Architecture

This document explains the structure, threading model, data flow, and design rationale of the event camera DAT file visualizer implemented in this project.

## Overview
The application streams a DAT file containing event camera data and renders a live visualization at a stable 30 FPS. It separates responsibilities across two cooperating threads:

- **Reader Thread**: Decodes events from disk (optionally paced to original capture time) and writes pixels into a shared frame buffer.
- **Render/Main Thread**: Periodically (30 FPS) copies the frame, overlays statistics, displays the window, and clears the buffer for the next interval.

This decoupling keeps UI refresh cadence stable regardless of event burstiness and prevents rendering overhead from throttling event ingestion.

## Key Source Files
- `event_reader.hpp` / `event_reader.cpp`: Streaming interface for DAT files (`stream_dat_events`) and event decoding.
- `peltipoliisi2.cpp`: Application entry point and visualization logic (`run_event_visualizer`, `run_dat_reader`, `render_frame`, `event_pixel_callback`).

## Data Structures
### `Event`
Represents one decoded 8-byte record: timestamp (`t` microseconds), position (`x`, `y`), and binary `polarity`.

### `DatHeaderInfo`
Holds parsed header metadata: width, height, version, textual date, event type and size (validated to be 8 bytes for this format).

### `FrameState`
Shared mutable state between threads:
- `frame` (`cv::Mat`): Accumulation buffer for the current 1/30 s interval.
- `events_total`: Total events processed since start.
- `events_since_clear`: Events accumulated in the current frame interval (presently reset but not displayed; available for future stats).
- `running`: Atomic flag indicating lifecycle termination.
- `mtx`: Mutex guarding compound operations on `frame` (pixel write, resize, copy+clear).

## Thread Responsibilities
### Reader Thread (`run_dat_reader`)
1. Opens the DAT file in binary mode.
2. Parses the ASCII header lines until `% end`, then consumes two metadata bytes (`event_type`, `event_size`).
3. Validates event size (must be 8) and streams the file in chunks (1 MB default), decoding each `RawRecord` into an `Event` via bitfield extraction.
4. Invokes the per-event callback (`event_pixel_callback`) for visualization.
5. In **realtime mode** (default unless the `FAST_EVENTS` environment variable is set), sleeps to align wall-clock time with event timestamps (using first event timestamp as zero baseline).
6. On completion or error, logs summary stats and sets `running = false` to signal the render loop to stop.

### Render/Main Thread (`run_event_visualizer` loop)
1. Initializes a window and an initial frame (default size, later resized if header differs).
2. Spawns the reader thread.
3. Enters a deterministic frame loop targeting 30 FPS using `steady_clock` and `sleep_until` to avoid drift.
4. Each iteration calls `render_frame`:
   - Copy current accumulation buffer to a local `display` image.
   - Clear shared `frame` to black, resetting pixel accumulation.
   - Overlay textual statistics (currently `events_total`).
   - Show the window and process a single key (ESC or `q` triggers shutdown).
5. Loop exits when `running` becomes false or the user quits.
6. Joins the reader thread for orderly shutdown.

## Locking & Concurrency
- **Granularity**: The mutex only protects short-lived operations manipulating the frame contents. Timestamps, pacing logic, file IO, and event decoding run outside the lock.
- **Atomics**: Counters and the `running` flag use relaxed memory ordering (sufficient since they are polled rather than depending on strict happens-before relationships).
- **Contention Minimization**: Pixel writes hold the lock briefly; rendering performs copy+clear under one lock acquisition per frame. This keeps throughput high even for massive event rates.

## Timing & Pacing
- **Rendering cadence**: The main loop calculates the next frame deadline (`next_frame += frame_interval`) and sleeps until that point. Using `sleep_until` instead of `sleep_for` mitigates drift accumulation.
- **Realtime event pacing**: When enabled, the reader thread derives microsecond deltas from the first event timestamp and sleeps until the corresponding wall-clock target. This makes the run duration approximate the original capture span.
- **Fast mode**: Setting `FAST_EVENTS` skips sleeps in the reader thread, filling frames as fast as IO and decoding allow.

## Frame Lifecycle
1. Accumulate events (pixel coloring) into `frame` during the current interval.
2. At frame boundary (render iteration): copy to local `display` then clear the shared `frame`.
3. Display `display` and overlay stats.
4. Begin new accumulation window.

This produces a live strobing view of recent activity rather than a cumulative trace. Removing the clear step would turn the visualization into a persistent density map.

## Event Decoding Details
Each 8-byte record contains two 32-bit little-endian words:
- `t32`: timestamp (microseconds)
- `w32`: packed bitfield (layout inferred from Python reference):
  - bits 13..0: `x` (14 bits)
  - bits 27..14: `y` (14 bits)
  - bits 31..28: polarity nibble (any nonzero → 1)

Conversion is performed in-place without endian swapping on little-endian hosts.

## Why Two Threads?
Separating ingestion and rendering provides:
- Stable, deterministic frame rate independent of event frequency.
- Reduced risk of blocking IO or pacing sleeps during expensive GUI operations.
- Clean separation of concerns (decode vs. present) for maintenance and profiling.
- Avoidance of tying refresh frequency to event arrival (prevents both over-rendering during bursts and apparent freezes during sparse periods).

## Alternatives & Trade-offs
| Approach | Pros | Cons |
|----------|------|------|
| Single thread (callback-driven renders) | Simplifies lifecycle | FPS jitter; GUI stalls under bursts; no updates when idle |
| Timer in same thread as reader | Slightly simpler than two threads | Still couples IO + rendering; pacing + rendering sleeps interact |
| Double-buffering (two `cv::Mat`s) | Eliminates copy+clear contention | More memory; complexity for pointer swaps |
| GPU-based accumulation | Higher performance for massive rates | Additional dependencies & complexity |

Current choice balances simplicity, responsiveness, and extensibility.

## Extensibility Points
- Add per-frame event rate overlay (`events_since_clear`).
- Implement cumulative visualization or decay-based heat map.
- Introduce cancellation check inside `stream_dat_events` for faster abort when user quits.
- Add CLI flags for: FPS target, fast vs realtime mode, logging interval.
- Export event statistics to CSV or JSON for offline analysis.

## Error Handling & Edge Cases
- Non-opening file: logs and aborts early.
- Header parse failure or wrong `event_size`: explicit error and early return.
- Out-of-bounds events: safely skipped.
- Empty file / no events: visualization displays a hint.

## Performance Considerations
- Chunked reading (1 MB) amortizes system calls.
- Minimal allocations: frame reused, buffer reused, tiny objects per event only for decoding.
- Relaxed atomics and short locks keep throughput high.
- Realtime sleeps rely on monotonic `steady_clock` for precision.

## Build & Run
Meson/Ninja build places executable at `build/peltipoliisi2`.

Run (realtime pacing):
```bash
./build/peltipoliisi2
```
Fast mode (no pacing sleeps):
```bash
FAST_EVENTS=1 ./build/peltipoliisi2
```
Optional alternate file:
```bash
./build/peltipoliisi2 data/your_file.dat
```

## Summary
The design separates high-frequency event ingestion from scheduled rendering to ensure smooth visualization and scalable performance. The current modular functions (`run_dat_reader`, `event_pixel_callback`, `render_frame`) clarify responsibilities and provide a solid foundation for future enhancements without entangling timing, IO, and UI concerns.
