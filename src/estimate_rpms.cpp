#include "defs.hpp"
#include <random>

FrameState::RpmStats estimate_rpms() {
  // Thread-safe RPM stats computation. Takes snapshots under lock then
  // releases the lock for heavier processing. Avoids data races with
  // event_pixel_callback pushing to frame_events concurrently.
  FrameState::RpmStats stats;
  struct SamplePoint { int x; int y; };
  vector<SamplePoint> sample_points;
  vector<FrameEvent> events_snapshot;
  vector<RpmSample> rpm_samples; // precomputed per-point RPMs for clusters
  vector<double> cluster_coords;
  cluster_coords.reserve(4096);
  u64 frame_id_for_workers = 0;
  int counts_rows = 0, counts_cols = 0;

  {
    lock_guard<mutex> lk(fs.mtx);
    if (fs.counts.empty()) return stats;
    counts_rows = fs.counts.rows;
    counts_cols = fs.counts.cols;

    sample_points.reserve(SAMPLE_POINTS);
    static int frame_seed = 0;
    mt19937 rng(static_cast<u32>(frame_seed++) ^ 0x9e3779b9u);
    uniform_real_distribution<double> uni01(0.0, 1.0);

    long long seen = 0;
    for (int y = 0; y < counts_rows; ++y) {
      const int *row = fs.counts.ptr<int>(y);
      for (int x = 0; x < counts_cols; ++x) {
        if (row[x] >= EVENT_COUNT_THRESHOLD) {
          cluster_coords.push_back(static_cast<double>(x));
          cluster_coords.push_back(static_cast<double>(y));
          ++seen;
          if (static_cast<int>(sample_points.size()) < SAMPLE_POINTS) {
            sample_points.push_back({x, y});
          } else {
            long long r = static_cast<long long>(floor(uni01(rng) * seen));
            if (r < SAMPLE_POINTS) sample_points[static_cast<size_t>(r)] = {x, y};
          }
        }
      }
    }
    // Snapshot events from the reader's active deque (windowed events)
  }
  snapshot_active_events(events_snapshot);
  frame_id_for_workers = 0; //fs.frame_index.load(memory_order_relaxed);

  stats.sampled = sample_points.size();
  if (sample_points.empty()) return stats;

  vector<double> rpms;
  rpms.reserve(sample_points.size());
  rpm_samples.reserve(sample_points.size());
  for (const auto &sp : sample_points) {
    const int x0 = max(0, sp.x - SAMPLE_RANGE);
    const int x1 = min(counts_cols - 1, sp.x + SAMPLE_RANGE);
    const int y0 = max(0, sp.y - SAMPLE_RANGE);
    const int y1 = min(counts_rows - 1, sp.y + SAMPLE_RANGE);

    vector<FrameEvent> local;
    local.reserve(64);
    for (const auto &fe : events_snapshot) {
      if (fe.x >= x0 && fe.x <= x1 && fe.y >= y0 && fe.y <= y1) {
        local.push_back(fe);
      }
    }
    if (local.size() >= 16) {
      double rpm = rpm_from_fft(local, NUM_BLADES);
      if (isfinite(rpm) && rpm > 0.0) {
        rpms.push_back(rpm);
        rpm_samples.push_back(RpmSample{sp.x, sp.y, rpm});
      }
    }
  }

  stats.valid = rpms.size();
  if (!rpms.empty()) {
    size_t mid = rpms.size() / 2;
    nth_element(rpms.begin(), rpms.begin() + mid, rpms.end());
    if (rpms.size() % 2 == 1) {
      stats.median = rpms[mid];
    } else {
      double a = *max_element(rpms.begin(), rpms.begin() + mid);
      double b = rpms[mid];
      stats.median = 0.5 * (a + b);
    }
  }
  // Temporal smoothing: maintain a short history and use its median.
  if (isfinite(stats.median) && stats.median > 0.0) {
    lock_guard<mutex> lk(fs.mtx);
    fs.rpm_median_history.push_back(stats.median);
    if (fs.rpm_median_history.size() > fs.rpm_median_history_limit) {
      fs.rpm_median_history.erase(fs.rpm_median_history.begin());
    }
    if (!fs.rpm_median_history.empty()) {
      vector<double> h = fs.rpm_median_history;
      size_t hmid = h.size() / 2;
      nth_element(h.begin(), h.begin() + hmid, h.end());
      if (h.size() % 2 == 1) {
        stats.median = h[hmid];
      } else {
        double ha = *max_element(h.begin(), h.begin() + hmid);
        double hb = h[hmid];
        stats.median = 0.5 * (ha + hb);
      }
    }
  }
  if (!cluster_coords.empty()) {
    // Compute clusters synchronously and publish overlays
    double eps = fs.cluster_eps;
    size_t min_pts = fs.cluster_min_points;
    auto overlays = cluster_worker(move(cluster_coords), move(rpm_samples), eps, min_pts);
    lock_guard<mutex> lk(fs.overlay_mtx);
    fs.overlay_data = std::move(overlays);
    fs.overlay_frame = frame_id_for_workers;
  }

  return stats;
}