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
#include <unordered_map>
#include <fftw3.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include "event_reader.hpp"



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
  struct FrameEvent { u32 t; uint16_t x; uint16_t y; uint8_t pol; }; // lightweight record with polarity
  std::vector<FrameEvent> frame_events;
  // Requests to dump timestamps for a given (x,y) at frame end
  struct DumpRequest { int x; int y; };
  std::vector<DumpRequest> dump_requests;
  atomic<bool> running{true};
  atomic<u64> events_total{0};
  atomic<u64> events_since_clear{0};
  double cluster_eps{6.5};
  size_t cluster_min_points{12};
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

// On each incoming event: increment per-pixel counter and update totals.
inline void event_pixel_callback(const Event &e) {
  if (e.x >= fs.frame.cols || e.y >= fs.frame.rows) return;
  lock_guard<mutex> lk(fs.mtx);
  // Increment counter; bounds are checked above.
  int &cnt = fs.counts.at<int>(e.y, e.x);
  // Prevent overflow in very long runs.
  if (cnt < INT32_MAX) ++cnt;
  // Store timestamp and location for this frame
  fs.frame_events.push_back(FrameState::FrameEvent{e.t, e.x, e.y, e.polarity});
  fs.events_total.fetch_add(1, memory_order_relaxed);
  fs.events_since_clear.fetch_add(1, memory_order_relaxed);
  auto total = fs.events_total.load(memory_order_relaxed);
  if (total % 1000000 == 0) {
    cout << "[DBG] processed events_total=" << total << endl;
  }
}

// Thread body: stream DAT events, resize frame after header, log summary.
void run_dat_reader(const string &dat_path) {
  DatHeaderInfo header;
  bool ok = stream_dat_events(dat_path, event_pixel_callback, &header);
  if (!ok) {
    std::cerr << "[reader] failed to stream events from " << dat_path << std::endl;
  }
  fs.running.store(false);
}

// Number of blades on the propeller (adjust if needed)
static const int NUM_BLADES = 2;

// Estimate RPM from timestamps (µs) of events in a small region using FFTW.
// Returns NaN if not enough data or no clear peak.
double estimate_rpm_from_events(const std::vector<FrameState::FrameEvent> &events) {
  const size_t N = events.size();
  if (N < 16) return std::numeric_limits<double>::quiet_NaN(); // not enough events

  // Extract times in seconds, sorted
  std::vector<double> t;
  t.reserve(N);
  for (const auto &e : events) t.push_back(static_cast<double>(e.t) * 1e-6);
  std::sort(t.begin(), t.end());

  const double t0 = t.front();
  const double t_last = t.back();
  const double T_total = t_last - t0;
  if (T_total <= 0.0) return std::numeric_limits<double>::quiet_NaN();

  // Inter-event gaps to estimate a reasonable bin width
  std::vector<double> dt;
  dt.reserve(N - 1);
  for (size_t i = 0; i + 1 < N; ++i) {
    double d = t[i + 1] - t[i];
    if (d > 0.0) dt.push_back(d);
  }
  if (dt.size() < 4) return std::numeric_limits<double>::quiet_NaN();

  // Median of dt
  std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
  double median_dt = dt[dt.size() / 2];

  // Bin width: several times median inter-event gap, but not too small
  double bin_width = std::max(5.0 * median_dt, 1e-5); // >= 10 µs

  int nbins = static_cast<int>(std::ceil(T_total / bin_width));
  if (nbins < 16) nbins = 16;

  // Allocate FFTW input/output
  double *in = (double*)fftw_malloc(sizeof(double) * nbins);
  fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (nbins / 2 + 1));
  if (!in || !out) {
    if (in) fftw_free(in);
    if (out) fftw_free(out);
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Zero-initialize bins
  std::fill(in, in + nbins, 0.0);

  // Histogram: event counts per bin
  for (double ti : t) {
    double rel = ti - t0;  // relative time
    int idx = static_cast<int>(rel / bin_width);
    if (idx >= 0 && idx < nbins) in[idx] += 1.0;
  }

  // Remove DC (mean) component
  double sum = 0.0;
  for (int i = 0; i < nbins; ++i) sum += in[i];
  double mean = sum / nbins;
  for (int i = 0; i < nbins; ++i) in[i] -= mean;

  // Plan & execute FFT
  fftw_plan plan = fftw_plan_dft_r2c_1d(nbins, in, out, FFTW_ESTIMATE);
  fftw_execute(plan);

  // Frequency axis info
  double fs = 1.0 / bin_width;          // sampling frequency (Hz)
  int nfreq = nbins / 2 + 1;

  // Ignore DC (k=0); search within reasonable band
  double f_min = 5.0;      // Hz
  double f_max = 5000.0;   // Hz

  double best_mag = 0.0;
  double best_freq = 0.0;

  for (int k = 1; k < nfreq; ++k) {
    double f = (static_cast<double>(k) * fs) / nbins;
    if (f < f_min || f > f_max) continue;

    double re = out[k][0];
    double im = out[k][1];
    double mag = std::hypot(re, im);

    if (mag > best_mag) {
      best_mag = mag;
      best_freq = f;
    }
  }

  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  if (best_freq <= 0.0) return std::numeric_limits<double>::quiet_NaN();

  // best_freq is blade-pass frequency (Hz)
  double f_rot = best_freq / NUM_BLADES; // rotor frequency
  double rpm = 60.0 * f_rot;
  return rpm;
}
bool render_frame(FrameState &fs) {
  cv::Mat display;
  // We will sample up to 100 points that met the threshold and compute RPM per point
  struct SamplePoint { int x; int y; };
  std::vector<SamplePoint> sample_points;
  std::vector<FrameState::FrameEvent> frame_events_copy; // snapshot for RPM calc outside the lock
  std::vector<cv::Rect> cluster_boxes;
  std::vector<double> cluster_coords;
  cluster_coords.reserve(4096);

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
          cluster_coords.push_back(static_cast<double>(x));
          cluster_coords.push_back(static_cast<double>(y));
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

    if (!cluster_coords.empty()) {
      const size_t point_count = cluster_coords.size() / 2;
      if (point_count >= fs.cluster_min_points) {
        arma::mat data(cluster_coords.data(), 2, point_count, /*copy_aux_mem*/ false, /*strict*/ true);
        mlpack::DBSCAN<> clusterer(fs.cluster_eps, fs.cluster_min_points);
        arma::Row<size_t> assignments;
        clusterer.Cluster(data, assignments);

        struct Bounds {
          int min_x = std::numeric_limits<int>::max();
          int min_y = std::numeric_limits<int>::max();
          int max_x = std::numeric_limits<int>::min();
          int max_y = std::numeric_limits<int>::min();
          bool initialized = false;
        };

        std::unordered_map<size_t, Bounds> boxes;
        boxes.reserve(point_count);

        for (size_t idx = 0; idx < point_count; ++idx) {
          const size_t label = assignments[idx];
          if (label == std::numeric_limits<size_t>::max()) continue; // noise

          const int px = static_cast<int>(cluster_coords[2 * idx]);
          const int py = static_cast<int>(cluster_coords[2 * idx + 1]);
          auto &b = boxes[label];
          if (!b.initialized) {
            b.min_x = b.max_x = px;
            b.min_y = b.max_y = py;
            b.initialized = true;
          } else {
            b.min_x = std::min(b.min_x, px);
            b.min_y = std::min(b.min_y, py);
            b.max_x = std::max(b.max_x, px);
            b.max_y = std::max(b.max_y, py);
          }
        }

        std::vector<cv::Rect> boxes_local;
        boxes_local.reserve(boxes.size());
        for (const auto &entry : boxes) {
          const auto &b = entry.second;
          if (!b.initialized) continue;
          boxes_local.emplace_back(cv::Point(b.min_x, b.min_y),
                                   cv::Point(b.max_x + 1, b.max_y + 1));
        }
        cluster_boxes = std::move(boxes_local);
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
            ofs << fe.t << ' ' << static_cast<int>(fe.pol) << '\n';
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

    // Clear counters and per-frame state for next frame
    fs.counts.setTo(0);
    fs.frame_events.clear();
    fs.dump_requests.clear();
    fs.frame_index.fetch_add(1, memory_order_relaxed);
    fs.events_since_clear.store(0, memory_order_relaxed);
  } // mutex unlocked here

  // ---- Compute RPM for each sampled point (using 3x3 neighborhood) and take median ----
  std::vector<double> rpms;
  rpms.reserve(sample_points.size());
  for (const auto &sp : sample_points) {
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
      double rpm = estimate_rpm_from_events(local);
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

  for (const auto &rect : cluster_boxes) {
    cv::rectangle(display, rect, cv::Scalar(0, 0, 255), 2);
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
