// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include "defs.hpp"
#include <random>

constexpr int DEFAULT_W = 1280;
constexpr int DEFAULT_H = 720;
constexpr int CLUSTER_BOX_PADDING = 6;
constexpr int SLIDING_WINDOW_US = 50'000;
constexpr int EVENT_COUNT_THRESHOLD = 10;
const double TARGET_FPS = 30.0;
using steady = chrono::steady_clock;

FrameState fs;

// On each incoming or expiring event: delta=+1 to increment, delta=-1 to decrement.
inline void event_pixel_callback(const FrameEvent &e, int delta) {
  if (e.x >= fs.frame.cols || e.y >= fs.frame.rows) return;
  lock_guard<mutex> lk(fs.mtx);
  // Increment counter; bounds are checked above.
  int &cnt = fs.counts.at<int>(e.y, e.x);
  if (delta > 0) {
    if (cnt < INT32_MAX) ++cnt;
    // Store timestamp and location for this frame (only arrivals)
    fs.frame_events.push_back(FrameEvent{e.t, e.x, e.y, e.polarity});
  } else if (delta < 0) {
    if (cnt > 0) --cnt; // clamp at 0
  }
}

// Thread body: stream DAT events, resize frame after header, log summary.
void run_dat_reader(const string &dat_path) {
  DatHeaderInfo header;
  // Uses default 50ms window; stream emits +1 on arrival and -1 on expiry.
  stream_dat_events(dat_path, event_pixel_callback, &header, SLIDING_WINDOW_US);
  fs.running.store(false);
  fs.cluster_request_cv.notify_all();
}

FrameState::RpmStats compute_rpm_stats_from_counts() {
  // Thread-safe RPM stats computation. Takes snapshots under lock then
  // releases the lock for heavier processing. Avoids data races with
  // event_pixel_callback pushing to frame_events concurrently.
  FrameState::RpmStats stats;
  struct SamplePoint { int x; int y; };
  vector<SamplePoint> sample_points;
  vector<FrameEvent> events_snapshot;
  vector<double> cluster_coords;
  cluster_coords.reserve(4096);
  u64 frame_id_for_workers = 0;
  int counts_rows = 0, counts_cols = 0;

  {
    lock_guard<mutex> lk(fs.mtx);
    if (fs.counts.empty()) return stats;
    counts_rows = fs.counts.rows;
    counts_cols = fs.counts.cols;

    constexpr int K = 10;
    sample_points.reserve(K);
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
          if (static_cast<int>(sample_points.size()) < K) {
            sample_points.push_back({x, y});
          } else {
            long long r = static_cast<long long>(floor(uni01(rng) * seen));
            if (r < K) sample_points[static_cast<size_t>(r)] = {x, y};
          }
        }
      }
    }
    // Snapshot events then clear for next interval.
    events_snapshot = fs.frame_events;
    fs.frame_events.clear();
  }
  frame_id_for_workers = 0; //fs.frame_index.load(memory_order_relaxed);

  stats.sampled = sample_points.size();
  if (sample_points.empty()) return stats;

  vector<double> rpms;
  rpms.reserve(sample_points.size());
  for (const auto &sp : sample_points) {
    const int x0 = max(0, sp.x - 1);
    const int x1 = min(counts_cols - 1, sp.x + 1);
    const int y0 = max(0, sp.y - 1);
    const int y1 = min(counts_rows - 1, sp.y + 1);

    vector<FrameEvent> local;
    local.reserve(64);
    for (const auto &fe : events_snapshot) {
      if (fe.x >= x0 && fe.x <= x1 && fe.y >= y0 && fe.y <= y1) {
        local.push_back(fe);
      }
    }
    if (local.size() >= 16) {
      double rpm = estimate_rpm_from_events(local, 2);
      if (isfinite(rpm) && rpm > 0.0) rpms.push_back(rpm);
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
  if (!cluster_coords.empty()) {
    bool posted_cluster_request = false;
    {
      lock_guard<mutex> req_lk(fs.cluster_request_mtx);
      if (!fs.cluster_request_ready) {
        fs.cluster_request_coords = move(cluster_coords);
        fs.cluster_request_events = move(events_snapshot);
        fs.cluster_request_frame = frame_id_for_workers;
        fs.cluster_request_ready = true;
        posted_cluster_request = true;
      }
    }
    if (posted_cluster_request) fs.cluster_request_cv.notify_one();
  }


  return stats;
}

void rpm_counter_loop() {
  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/10.0));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    auto stats = compute_rpm_stats_from_counts();
    {
      lock_guard<mutex> lk(fs.mtx);
      fs.rpm_stats = stats;
    }
    this_thread::sleep_until(next_frame);
  }
}

// RPM estimation moved to rpm_estimator.cpp/rpm_estimator.hpp
bool render_frame() {
  cv::Mat display;
  cv::Mat counts_snapshot;
  FrameState::RpmStats rpm_stats_copy;

  {
    lock_guard<mutex> lk(fs.mtx);
    // Prepare display buffer
    display.create(fs.frame.rows, fs.frame.cols, CV_8UC3);
    display.setTo(cv::Scalar(0,0,0));

    // Build mask where counts >= threshold and set those pixels to white.
    cv::Mat mask;
    cv::compare(fs.counts, EVENT_COUNT_THRESHOLD, mask, cv::CMP_GE); // mask: 255 where true
    display.setTo(cv::Scalar(255,255,255), mask);
    rpm_stats_copy = fs.rpm_stats; // snapshot under lock
  } // mutex unlocked here


   vector<FrameState::ClusterOverlay> overlays_copy;
  {
    lock_guard<mutex> lk(fs.overlay_mtx);
    overlays_copy = fs.overlay_data;
  }
  for (const auto &overlay : overlays_copy) {
    cv::Rect padded = overlay.box;
    padded.x = max(0, padded.x - CLUSTER_BOX_PADDING);
    padded.y = max(0, padded.y - CLUSTER_BOX_PADDING);
    padded.width = min(display.cols - padded.x,
                            overlay.box.width + 2 * CLUSTER_BOX_PADDING);
    padded.height = min(display.rows - padded.y,
                             overlay.box.height + 2 * CLUSTER_BOX_PADDING);
    cv::rectangle(display, padded, overlay.color, 2);
  }

  int rpm_block_y = 45;
  auto put_line = [&](const string &line,
                      const cv::Scalar &text_color = cv::Scalar(0, 255, 255),
                      const cv::Scalar *marker = nullptr) {
    if (marker) {
      cv::rectangle(display,
                    cv::Point(10, rpm_block_y - 12),
                    cv::Point(22, rpm_block_y - 2),
                    *marker, cv::FILLED);
      cv::putText(display, line, cv::Point(26, rpm_block_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  text_color, 1, cv::LINE_AA);
    } else {
      cv::putText(display, line, cv::Point(10, rpm_block_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55,
                  text_color, 1, cv::LINE_AA);
    }
    rpm_block_y += 18;
  };


  if (!isfinite(rpm_stats_copy.median) || rpm_stats_copy.median <= 0.0) {
    put_line("Global RPM: N/A");
  } else {
    put_line("Global RPM: " + to_string(static_cast<int>(round(rpm_stats_copy.median))) +
             " (" + to_string(rpm_stats_copy.sampled) + " samples)");
  }

  if (overlays_copy.empty()) {
    put_line("Clusters: none");
  } else {
    put_line("Clusters:");
    for (size_t i = 0; i < overlays_copy.size(); ++i) {
      const auto &overlay = overlays_copy[i];
      string line = "#" + to_string(i) + ": ";
      if (isfinite(overlay.rpm)) {
        line += to_string(static_cast<int>(round(overlay.rpm))) + " RPM";
      } else {
        line += "RPM N/A";
      }
      put_line(line, overlay.color, &overlay.color);
    }
  }


  cv::imshow("Events", display);
  int key = cv::waitKey(1);
  if (key == 27 || key == 'q') { fs.running.store(false); return false; }
  if (key == 'c' || key == 'C') {
    lock_guard<mutex> lk(fs.mtx);
    fs.counts.setTo(0);
  }
  return true;
}


int main(int argc, char **argv) {
  // Usage: program <DAT filepath> [threshold]
  if (argc < 2 || argc > 2) {
    cout << "Usage: " << argv[0] << " <DAT filepath>\n";
    return 1;
  }
  string dat_path = argv[1];
  cout << "Event visualizer (30 FPS). File: " << dat_path << "\nESC/q to quit, C to clear.\n";

  fs.frame = cv::Mat(DEFAULT_H, DEFAULT_W, CV_8UC3, cv::Scalar(0,0,0));
  fs.counts = cv::Mat(DEFAULT_H, DEFAULT_W, CV_32SC1, cv::Scalar(0));
  cv::namedWindow("Events", cv::WINDOW_AUTOSIZE);

  thread reader(run_dat_reader, dat_path);
  thread rpm_counter(rpm_counter_loop);
  thread cluster_thread(cluster_worker);

  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/TARGET_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    if (!render_frame()) break;
    this_thread::sleep_until(next_frame);
  }

  fs.running.store(false);
  fs.cluster_request_cv.notify_all();
  if (reader.joinable()) reader.join();
  if (rpm_counter.joinable()) rpm_counter.join();
  if (cluster_thread.joinable()) cluster_thread.join();
  return 0;
}
