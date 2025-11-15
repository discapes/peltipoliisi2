// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <fftw3.h>
#include "event_reader.hpp"
#include "rpm_estimator.hpp"

const int DEFAULT_W = 1280;
const int DEFAULT_H = 720;
const double TARGET_FPS = 30.0;
using steady = chrono::steady_clock;

struct FrameState {
  mutex mtx;
  cv::Mat frame; // BGR display buffer
  cv::Mat counts; // CV_32SC1 per-pixel event counters
  int threshold{10};
  // Events collected in the current frame (reset each render)
  using FrameEvent = Event; // reuse Event from event_reader.hpp
  std::vector<FrameEvent> frame_events;
  atomic<bool> running{true};
  struct RpmStats {
    double median = std::numeric_limits<double>::quiet_NaN();
    size_t sampled = 0;
    size_t valid = 0;
  };
  RpmStats rpm_stats{};
} fs;

// On each incoming or expiring event: delta=+1 to increment, delta=-1 to decrement.
inline void event_pixel_callback(const Event &e, int delta) {
  if (e.x >= fs.frame.cols || e.y >= fs.frame.rows) return;
  lock_guard<mutex> lk(fs.mtx);
  // Increment counter; bounds are checked above.
  int &cnt = fs.counts.at<int>(e.y, e.x);
  if (delta > 0) {
    if (cnt < INT32_MAX) ++cnt;
    // Store timestamp and location for this frame (only arrivals)
    fs.frame_events.push_back(FrameState::FrameEvent{e.t, e.x, e.y, e.polarity});
  } else if (delta < 0) {
    if (cnt > 0) --cnt; // clamp at 0
  }
}

// Thread body: stream DAT events, resize frame after header, log summary.
void run_dat_reader(const string &dat_path) {
  DatHeaderInfo header;
  // Uses default 50ms window; stream emits +1 on arrival and -1 on expiry.
  bool ok = stream_dat_events(dat_path, event_pixel_callback, &header, 50'000);
  fs.running.store(false);
}

FrameState::RpmStats compute_rpm_stats_from_counts() {
  FrameState::RpmStats stats;
  if (fs.counts.empty()) return stats;

  struct SamplePoint { int x; int y; };
  std::vector<SamplePoint> sample_points;
  constexpr int K = 100;
  sample_points.reserve(K);
  static int frame_seed = 0;
  std::mt19937 rng(static_cast<uint32_t>(frame_seed++) ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> uni01(0.0, 1.0);

  long long seen = 0;
  for (int y = 0; y < fs.counts.rows; ++y) {
    const int *row = fs.counts.ptr<int>(y);
    for (int x = 0; x < fs.counts.cols; ++x) {
      if (row[x] >= fs.threshold) {
        ++seen;
        if (static_cast<int>(sample_points.size()) < K) {
          sample_points.push_back({x, y});
        } else {
          long long r = static_cast<long long>(std::floor(uni01(rng) * seen));
          if (r < K) sample_points[static_cast<size_t>(r)] = {x, y};
        }
      }
    }
  }

  stats.sampled = sample_points.size();
  if (sample_points.empty()) {
    return stats;
  }

  std::vector<double> rpms;
  rpms.reserve(sample_points.size());
  for (const auto &sp : sample_points) {
    const int x0 = std::max(0, sp.x - 1);
    const int x1 = std::min(fs.counts.cols - 1, sp.x + 1);
    const int y0 = std::max(0, sp.y - 1);
    const int y1 = std::min(fs.counts.rows - 1, sp.y + 1);

    std::vector<FrameState::FrameEvent> local;
    local.reserve(64);
    for (const auto &fe : fs.frame_events) {
      if (fe.x >= x0 && fe.x <= x1 && fe.y >= y0 && fe.y <= y1) {
        local.push_back(fe);
      }
    }
    if (local.size() >= 16) {
      double rpm = estimate_rpm_from_events(local, 2);
      if (std::isfinite(rpm) && rpm > 0.0) rpms.push_back(rpm);
    }
  }
  fs.frame_events.clear();

  stats.valid = rpms.size();
  if (!rpms.empty()) {
    size_t mid = rpms.size() / 2;
    std::nth_element(rpms.begin(), rpms.begin() + mid, rpms.end());
    if (rpms.size() % 2 == 1) {
      stats.median = rpms[mid];
    } else {
      double a = *std::max_element(rpms.begin(), rpms.begin() + mid);
      double b = rpms[mid];
      stats.median = 0.5 * (a + b);
    }
  }

  return stats;
}

void rpm_counter_loop() {
  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/10.0));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    compute_rpm_stats_from_counts();
    this_thread::sleep_until(next_frame);
  }
}

// RPM estimation moved to rpm_estimator.cpp/rpm_estimator.hpp
bool render_frame() {
  cv::Mat display;
  cv::Mat counts_snapshot;
  uint64_t frame_seed = 0;
  int threshold = 0;

  {
    lock_guard<mutex> lk(fs.mtx);
    // Prepare display buffer
    display.create(fs.frame.rows, fs.frame.cols, CV_8UC3);
    display.setTo(cv::Scalar(0,0,0));

    // Build mask where counts >= threshold and set those pixels to white.
    cv::Mat mask;
    cv::compare(fs.counts, fs.threshold, mask, cv::CMP_GE); // mask: 255 where true
    display.setTo(cv::Scalar(255,255,255), mask);
  } // mutex unlocked here


  // RPM overlay in corner: show median of sampled points
  string rpm_text;
  if (!std::isfinite(fs.rpm_stats.median) || fs.rpm_stats.median <= 0.0) {
    rpm_text = "RPM (median of samples): N/A";
  } else {
    rpm_text = "RPM (median of samples): " +
               to_string(static_cast<int>(std::round(fs.rpm_stats.median))) +
               "  (n=" + to_string(fs.rpm_stats.sampled) + "/valid=" +
               to_string(fs.rpm_stats.valid) + ")";
  }
  cv::putText(display, rpm_text, cv::Point(10, 45),
              cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

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
  if (argc < 2 || argc > 3) {
    cout << "Usage: " << argv[0] << " <DAT filepath> [threshold]\n";
    return 1;
  }
  string dat_path = argv[1];
  if (argc == 3) {
    try {
      fs.threshold = max(1, stoi(argv[2]));
    } catch (...) {
      cerr << "Invalid threshold '" << argv[2] << "', using default " << fs.threshold << "\n";
    }
  }
  cout << "Event visualizer (30 FPS). File: " << dat_path << "\nESC/q to quit, C to clear.\n";

  fs.frame = cv::Mat(DEFAULT_H, DEFAULT_W, CV_8UC3, cv::Scalar(0,0,0));
  fs.counts = cv::Mat(DEFAULT_H, DEFAULT_W, CV_32SC1, cv::Scalar(0));
  cv::namedWindow("Events", cv::WINDOW_AUTOSIZE);

  thread reader(run_dat_reader, dat_path);
  thread rpm_counter(rpm_counter_loop);

  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/TARGET_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    if (!render_frame()) break;
    this_thread::sleep_until(next_frame);
  }
  if (reader.joinable()) reader.join();
  if (rpm_counter.joinable()) rpm_counter.join();
  return 0;
}
