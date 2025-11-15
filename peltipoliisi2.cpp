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
  std::atomic<int> cursor_x{-1};
  std::atomic<int> cursor_y{-1};
  std::atomic<u64> frame_index{0};
  // Events collected in the current frame (reset each render)
  using FrameEvent = Event; // reuse Event from event_reader.hpp
  std::vector<FrameEvent> frame_events;
  // Requests to dump timestamps for a given (x,y) at frame end
  struct DumpRequest { int x; int y; };
  std::vector<DumpRequest> dump_requests;
  atomic<bool> running{true};
  atomic<u64> events_total{0};
  atomic<u64> events_since_clear{0};
} fs;

// OpenCV mouse callback: update cursor coordinates (in window coords)
static void on_mouse(int event, int x, int y, int /*flags*/, void* /*userdata*/) {
  if (event == cv::EVENT_MOUSEMOVE) {
    fs.cursor_x.store(x, memory_order_relaxed);
    fs.cursor_y.store(y, memory_order_relaxed);
  } else if (event == cv::EVENT_LBUTTONDOWN) {
    // Record cursor and enqueue a dump request; actual file IO will be done at frame end.
    fs.cursor_x.store(x, memory_order_relaxed);
    fs.cursor_y.store(y, memory_order_relaxed);
    std::lock_guard<std::mutex> lk(fs.mtx);
    fs.dump_requests.push_back(FrameState::DumpRequest{x, y});
  }
}

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
    fs.events_total.fetch_add(1, memory_order_relaxed);
    fs.events_since_clear.fetch_add(1, memory_order_relaxed);
    auto total = fs.events_total.load(memory_order_relaxed);
    if (total % 1000000 == 0) {
      cout << "[DBG] processed events_total=" << total << endl;
    }
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

// RPM estimation moved to rpm_estimator.cpp/rpm_estimator.hpp
bool render_frame(FrameState &fs) {
  cv::Mat display;
  // We will sample up to 100 points that met the threshold and compute RPM per point
  struct SamplePoint { int x; int y; };
  std::vector<SamplePoint> sample_points;
  std::vector<FrameState::FrameEvent> frame_events_copy; // snapshot for RPM calc outside the lock

  {
    lock_guard<mutex> lk(fs.mtx);

    // Prepare display buffer
    display.create(fs.frame.rows, fs.frame.cols, CV_8UC3);
    display.setTo(cv::Scalar(0,0,0));

    // Build mask where counts >= threshold and set those pixels to white.
    cv::Mat mask;
    cv::compare(fs.counts, fs.threshold, mask, cv::CMP_GE); // mask: 255 where true
    display.setTo(cv::Scalar(255,255,255), mask);

    // Sample up to 100 points from pixels that reached the threshold using reservoir sampling
    const int K = 10;
    std::mt19937 rng(static_cast<uint32_t>(fs.frame_index.load(memory_order_relaxed)) ^ 0x9e3779b9u);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    long long seen = 0;
    for (int y = 0; y < mask.rows; ++y) {
      const uchar *row = mask.ptr<uchar>(y);
      for (int x = 0; x < mask.cols; ++x) {
        if (row[x]) {
          ++seen;
          if ((int)sample_points.size() < K) {
            sample_points.push_back({x, y});
          } else {
            // Replace an existing sample with probability K/seen
            // Generate an integer r in [0, seen-1] using floating RNG for simplicity
            // This is sufficient for uniform reservoir sampling.
            long long r = static_cast<long long>(std::floor(uni01(rng) * seen));
            if (r < K) sample_points[(size_t)r] = {x, y};
          }
        }
      }
    }

    // Take a snapshot of frame_events for use outside the lock (we're about to clear)
    frame_events_copy = fs.frame_events;

    // ---- existing dump-to-file logic (unchanged) ----
    for (const auto &req : fs.dump_requests) {
  std::vector<FrameState::FrameEvent> matches;
      int x0 = std::max(0, req.x - 1);
      int x1 = std::min(fs.frame.cols - 1, req.x + 1);
      int y0 = std::max(0, req.y - 1);
      int y1 = std::min(fs.frame.rows - 1, req.y + 1);
      for (const auto &fe : fs.frame_events) {
        if (fe.x >= x0 && fe.x <= x1 && fe.y >= y0 && fe.y <= y1) matches.push_back(fe);
      }
      if (!matches.empty()) {
        std::string fname = "cursor_events_frame" + std::to_string(fs.frame_index.load(memory_order_relaxed)) + "_" + std::to_string(req.x) + "_" + std::to_string(req.y) + "_3x3.csv";
        std::ofstream ofs(fname);
        if (ofs) {
          for (const auto &fe : matches) {
            ofs << fe.t << ' ' << static_cast<int>(fe.polarity) << '\n';
          }
          ofs.close();
          std::cout << "[cursor-events] wrote " << matches.size() << " lines (3x3 around " << req.x << "," << req.y << ") to " << fname << std::endl;
        } else {
          std::cerr << "[cursor-events] failed to open file " << fname << std::endl;
        }
      } else {
        std::cout << "[cursor-events] no events in 3x3 around (" << req.x << "," << req.y << ") this frame" << std::endl;
      }
    }

  // Clear per-frame state for next frame (do not clear counts; sliding window keeps it updated)
    fs.frame_events.clear();
    fs.dump_requests.clear();
    fs.frame_index.fetch_add(1, memory_order_relaxed);
  fs.events_since_clear.store(0, memory_order_relaxed);
  } // mutex unlocked here

  // ---- Compute RPM for each sampled point (using 3x3 neighborhood) and take median ----
  std::vector<double> rpms;
  rpms.reserve(sample_points.size());
  for (const auto &sp : sample_points) {
    // Collect events in 3x3 around the sample point from the snapshot
    const int x0 = std::max(0, sp.x - 1);
    const int x1 = std::min(display.cols - 1, sp.x + 1);
    const int y0 = std::max(0, sp.y - 1);
    const int y1 = std::min(display.rows - 1, sp.y + 1);
  std::vector<FrameState::FrameEvent> local;
    local.reserve(64);
    for (const auto &fe : frame_events_copy) {
      if (fe.x >= x0 && fe.x <= x1 && fe.y >= y0 && fe.y <= y1) {
        local.push_back(fe);
      }
    }
    if (local.size() >= 16) {
  double rpm = estimate_rpm_from_events(local, 2);
      if (std::isfinite(rpm) && rpm > 0.0) rpms.push_back(rpm);
    }
  }

  double median_rpm = std::numeric_limits<double>::quiet_NaN();
  if (!rpms.empty()) {
    size_t mid = rpms.size() / 2;
    std::nth_element(rpms.begin(), rpms.begin() + mid, rpms.end());
    if (rpms.size() % 2 == 1) {
      median_rpm = rpms[mid];
    } else {
      // Even count: average the two middle values
      double a = *std::max_element(rpms.begin(), rpms.begin() + mid);
      double b = rpms[mid];
      median_rpm = 0.5 * (a + b);
    }
  }

  // Overlay text
  u64 total = fs.events_total.load(memory_order_relaxed);
  string text = "events_total=" + to_string(total) +
                "  threshold=" + to_string(fs.threshold) +
                "  [c] clear";
  cv::putText(display, text, cv::Point(10,20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0,255,0), 1, cv::LINE_AA);

  // RPM overlay in corner: show median of sampled points
  string rpm_text;
  if (!std::isfinite(median_rpm) || median_rpm <= 0.0) {
    rpm_text = "RPM (median of samples): N/A";
  } else {
    rpm_text = "RPM (median of samples): " + to_string(static_cast<int>(std::round(median_rpm))) +
               "  (n=" + to_string(sample_points.size()) + "/valid=" + to_string(rpms.size()) + ")";
  }
  cv::putText(display, rpm_text, cv::Point(10, 45),
              cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

  if (total == 0) {
    cv::putText(display, "(no events yet - file reading or pacing)", cv::Point(10,70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,50,200), 1, cv::LINE_AA);
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
  cv::setMouseCallback("Events", on_mouse, nullptr);

  thread reader(run_dat_reader, dat_path);

  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/TARGET_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    if (!render_frame(fs)) break;
    this_thread::sleep_until(next_frame);
  }
  if (reader.joinable()) reader.join();
  return 0;
}
