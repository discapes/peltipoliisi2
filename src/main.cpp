// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include "defs.hpp"
#include <random>
#include <thread>

constexpr int DEFAULT_W = 1280;
constexpr int DEFAULT_H = 720;
constexpr int CLUSTER_BOX_PADDING = 6;
constexpr int SLIDING_WINDOW_US = 50'000;
constexpr int EVENT_COUNT_THRESHOLD = 10;
constexpr double TARGET_FPS = 60.0;
constexpr double CLUSTER_FPS = 20.0;
constexpr int SAMPLE_POINTS = 200;
constexpr int SAMPLE_RANGE = 1;
constexpr int NUM_BLADES = 2;
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
    // Snapshot events then clear for next interval.
    events_snapshot = fs.frame_events;
    fs.frame_events.clear();
  }
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
      double rpm = estimate_rpm_from_events(local, NUM_BLADES);
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

void update_rotor_tracking(vector<FrameState::ClusterOverlay> &new_overlays) {
  // Simple IoU based tracking
  const double iou_threshold = 0.3;
  vector<bool> matched_new(new_overlays.size(), false);

  lock_guard<mutex> lk(fs.mtx);

  // Increment unseen counters and remove old trackers
  for (auto &rotor : fs.tracked_rotors) {
    rotor.frames_unseen++;
  }
  fs.tracked_rotors.erase(
      remove_if(fs.tracked_rotors.begin(), fs.tracked_rotors.end(),
                [](const auto &r) { return r.frames_unseen > 5; }),
      fs.tracked_rotors.end());

  // Match existing trackers to new overlays
  for (auto &rotor : fs.tracked_rotors) {
    double best_iou = 0;
    int best_match_idx = -1;

    for (size_t i = 0; i < new_overlays.size(); ++i) {
      if (matched_new[i]) continue;
      double iou = (double)(rotor.box & new_overlays[i].box).area() /
                   (double)(rotor.box | new_overlays[i].box).area();
      if (iou > best_iou) {
        best_iou = iou;
        best_match_idx = i;
      }
    }

    if (best_match_idx != -1 && best_iou > iou_threshold) {
      rotor.box = new_overlays[best_match_idx].box;
      if (isfinite(new_overlays[best_match_idx].rpm)) {
        rotor.rpm_history.push_back(new_overlays[best_match_idx].rpm);
        while (rotor.rpm_history.size() > fs.RPM_HISTORY_SIZE) {
          rotor.rpm_history.pop_front();
        }
      }
      rotor.frames_unseen = 0;
      matched_new[best_match_idx] = true;
    }
  }

  // Add new trackers for unmatched overlays
  static const array<cv::Scalar, 8> palette = {
      cv::Scalar(0, 0, 255),   cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),
      cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
      cv::Scalar(128, 255, 0), cv::Scalar(0, 128, 255)};

  for (size_t i = 0; i < new_overlays.size(); ++i) {
    if (!matched_new[i]) {
      FrameState::TrackedRotor new_rotor;
      new_rotor.id = fs.next_rotor_id++;
      new_rotor.box = new_overlays[i].box;
      new_rotor.color = palette[new_rotor.id % palette.size()];
      if (isfinite(new_overlays[i].rpm)) {
        new_rotor.rpm_history.push_back(new_overlays[i].rpm);
      }
      fs.tracked_rotors.push_back(new_rotor);
    }
  }
}


void rpm_counter_loop() {
  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/CLUSTER_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    auto stats = compute_rpm_stats_from_counts();
    {
      lock_guard<mutex> lk(fs.mtx);
      fs.rpm_stats = stats;
    }

    vector<FrameState::ClusterOverlay> overlays_copy;
    {
      lock_guard<mutex> lk(fs.overlay_mtx);
      overlays_copy = fs.overlay_data;
    }
    update_rotor_tracking(overlays_copy);

    this_thread::sleep_until(next_frame);
  }
}

// RPM estimation moved to rpm_estimator.cpp/rpm_estimator.hpp
bool render_frame() {
  cv::Mat display;
  cv::Mat counts_snapshot;
  FrameState::RpmStats rpm_stats_copy;
  vector<FrameState::TrackedRotor> tracked_rotors_copy;

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
    
    // Snapshot tracked rotors for rendering
    tracked_rotors_copy = fs.tracked_rotors;
  } // mutex unlocked here

  for (const auto &rotor : tracked_rotors_copy) {
    cv::Rect padded = rotor.box;
    padded.x = max(0, padded.x - CLUSTER_BOX_PADDING);
    padded.y = max(0, padded.y - CLUSTER_BOX_PADDING);
    padded.width = min(display.cols - padded.x,
                            rotor.box.width + 2 * CLUSTER_BOX_PADDING);
    padded.height = min(display.rows - padded.y,
                             rotor.box.height + 2 * CLUSTER_BOX_PADDING);
    cv::rectangle(display, padded, rotor.color, 2);
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

  if (tracked_rotors_copy.empty()) {
    put_line("Clusters: none");
  } else {
    put_line("Clusters:");
    for (const auto &rotor : tracked_rotors_copy) {
      string line = "#" + to_string(rotor.id) + ": ";
      if (!rotor.rpm_history.empty() && isfinite(rotor.rpm_history.back())) {
        line += to_string(static_cast<int>(round(rotor.rpm_history.back()))) + " RPM";
      } else {
        line += "RPM N/A";
      }
      put_line(line, rotor.color, &rotor.color);

      // Draw RPM history graph in sidebar
      if (rotor.rpm_history.size() > 1) {
        int graph_w = 150;
        int graph_h = 40;
        int text_margin = 5;

        cv::Point graph_origin(10, rpm_block_y);
        cv::Rect graph_rect(graph_origin, cv::Size(graph_w, graph_h));
        cv::rectangle(display, graph_rect, cv::Scalar(100, 100, 100), 1);


        // Find min/max RPM in history for scaling
        double min_rpm = 0;
        double max_rpm = 0;

        for (double rpm : rotor.rpm_history) {
          if (rpm > max_rpm) max_rpm = rpm;
        }

        max_rpm *= 1.5; // Add 20% padding

        double rpm_range = max_rpm - min_rpm;
        if (rpm_range < 500) rpm_range = 500;

        vector<cv::Point> points;
        points.reserve(rotor.rpm_history.size());
        for (size_t i = 0; i < rotor.rpm_history.size(); ++i) {
          int x = graph_origin.x + (int)((double)i / (fs.RPM_HISTORY_SIZE - 1) * graph_w);
          int y = graph_origin.y + graph_h - (int)(((rotor.rpm_history[i] - min_rpm) / max_rpm) * graph_h);
          points.emplace_back(x, y);
        }
        cv::polylines(display, points, false, rotor.color, 1, cv::LINE_AA);

        // Y-axis label and ticks
        cv::putText(display, "RPM", cv::Point(graph_origin.x, graph_origin.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(display, to_string((int)max_rpm), cv::Point(graph_rect.br().x + text_margin, graph_origin.y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(display, to_string((int)min_rpm), cv::Point(graph_rect.br().x + text_margin, graph_origin.y + graph_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        // X-axis label
        cv::putText(display, "Time", cv::Point(graph_origin.x + graph_w / 2 - 10, graph_rect.br().y + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        rpm_block_y += graph_h + 30;
      }
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
  // Background cluster thread removed; clustering is now invoked synchronously.

  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/TARGET_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    if (!render_frame()) break;
    this_thread::sleep_until(next_frame);
  }

  fs.running.store(false);
  if (reader.joinable()) reader.join();
  if (rpm_counter.joinable()) rpm_counter.join();
  return 0;
}
