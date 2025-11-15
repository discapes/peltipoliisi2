# Event Visualizer Architecture

This document explains the structure, threading model, data flow, and design rationale of the event camera DAT file visualizer implemented in this project.

## Overview
The application streams a DAT file containing event camera data and renders a live visualization at a stable 30 FPS. It separates responsibilities across two cooperating threads:

- **Reader Thread**: Decodes events from disk (paced to original capture time) and increments per-pixel counters in a shared integer matrix.
- **Render/Main Thread**: Periodically (30 FPS) composes a display image by coloring pixels white where the counter at that pixel is greater than or equal to a user-selectable threshold, overlays statistics, and shows the window. Counters are reset every frame so previous locations disappear; you can also press `C` to clear mid-frame.

This decoupling keeps UI refresh cadence stable regardless of event burstiness and prevents rendering overhead from throttling event ingestion.

## Key Source Files
- `event_reader.hpp` / `event_reader.cpp`: Streaming interface for DAT files (`stream_dat_events`) and event decoding.
- `peltipoliisi2.cpp`: Application entry point and visualization logic (`run_dat_reader`, `render_frame`, `event_pixel_callback`).

## Data Structures
### `Event`
Represents one decoded 8-byte record: timestamp (`t` microseconds), position (`x`, `y`), and binary `polarity`.

### `DatHeaderInfo`
Holds parsed header metadata: width, height, version, textual date, event type and size (validated to be 8 bytes for this format).

### `FrameState`
Shared mutable state between threads:
- `frame` (`cv::Mat` 8UC3): Display buffer used by the render thread.
- `counts` (`cv::Mat` 32SC1): Per-pixel event counters incremented by the reader thread and cleared each frame by the render thread.
- `threshold` (`int`): Minimum count to color a pixel white in the display.
- `events_total`: Total events processed since start.
- `events_since_clear`: Events accumulated since the last render pass (internal stat).
- `running`: Atomic flag indicating lifecycle termination.
- `mtx`: Mutex guarding compound operations on `counts`/`frame`.

## Thread Responsibilities
### Reader Thread (`run_dat_reader`)
1. Opens the DAT file in binary mode.
2. Parses the ASCII header lines until `% end`, then consumes two metadata bytes (`event_type`, `event_size`).
3. Validates event size (must be 8) and streams the file in chunks (1 MB default), decoding each `RawRecord` into an `Event` via bitfield extraction.
4. Invokes the per-event callback (`event_pixel_callback`) which increments the counter at the event's `(x, y)` location.
5. Sleeps to align wall-clock time with event timestamps (using first event timestamp as zero baseline).
6. On completion or error, logs summary stats and sets `running = false` to signal the render loop to stop.

### Render/Main Thread (main loop)
1. Initializes a window and an initial frame (default size, later resized if header differs).
2. Spawns the reader thread.
3. Enters a deterministic frame loop targeting 30 FPS using `steady_clock` and `sleep_until` to avoid drift.
4. Each iteration calls `render_frame`:
  - Create a local `display` image.
  - Compute a mask where `counts >= threshold` and set corresponding pixels to white.
  - Clear `counts` so the threshold must be reached again in the next frame.
  - Overlay textual statistics (currently `events_total` and `threshold`).
  - Show the window and process a single key (ESC or `q` quits, `C` clears counters immediately).
5. Loop exits when `running` becomes false or the user quits.
6. Joins the reader thread for orderly shutdown.

## Locking & Concurrency
- **Granularity**: The mutex only protects short-lived operations manipulating the frame contents. Timestamps, pacing logic, file IO, and event decoding run outside the lock.
- **Atomics**: Counters and the `running` flag use relaxed memory ordering (sufficient since they are polled rather than depending on strict happens-before relationships).
- **Contention Minimization**: Pixel writes hold the lock briefly; rendering performs copy+clear under one lock acquisition per frame. This keeps throughput high even for massive event rates.

## Timing & Pacing
- **Rendering cadence**: The main loop calculates the next frame deadline (`next_frame += frame_interval`) and sleeps until that point. Using `sleep_until` instead of `sleep_for` mitigates drift accumulation.
- **Realtime event pacing**: The reader thread derives microsecond deltas from the first event timestamp and sleeps until the corresponding wall-clock target. This makes the run duration approximate the original capture span.

## Frame Lifecycle
1. Accumulate events by incrementing `counts(y, x)` for each event.
2. At render time: build a binary mask (`counts >= threshold`) and set those pixels to white in `display`.
3. Clear `counts` to zero to start a fresh frame.
4. Overlay stats and show.

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
./build/peltipoliisi2 data/your_file.dat [threshold]
```
Examples:
```bash
./build/peltipoliisi2 data/drone_idle.dat           # default threshold (10)
./build/peltipoliisi2 data/drone_idle.dat 25        # use threshold 25
```

## Summary
The design separates high-frequency event ingestion from scheduled rendering to ensure smooth visualization and scalable performance. The current modular functions (`run_dat_reader`, `event_pixel_callback`, `render_frame`) clarify responsibilities and provide a solid foundation for future enhancements without entangling timing, IO, and UI concerns.
