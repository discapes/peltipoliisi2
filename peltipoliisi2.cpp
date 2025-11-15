// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
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
  atomic<bool> running{true};
  atomic<u64> events_total{0};
  atomic<u64> events_since_clear{0};
} fs;

// On each incoming event: increment per-pixel counter and update totals.
inline void event_pixel_callback(const Event &e) {
  if (e.x >= fs.frame.cols || e.y >= fs.frame.rows) return;
  lock_guard<mutex> lk(fs.mtx);
  // Increment counter; bounds are checked above.
  int &cnt = fs.counts.at<int>(e.y, e.x);
  // Prevent overflow in very long runs.
  if (cnt < INT32_MAX) ++cnt;
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
  fs.running.store(false);
}

// Render one frame from counters; overlay text; display; handle controls. Returns false on quit.
bool render_frame(FrameState &fs) {
  cv::Mat display;
  {
    lock_guard<mutex> lk(fs.mtx);
    // Prepare display buffer
    display.create(fs.frame.rows, fs.frame.cols, CV_8UC3);
    display.setTo(cv::Scalar(0,0,0));
    // Build mask where counts >= threshold and set those pixels to white.
    cv::Mat mask;
    cv::compare(fs.counts, fs.threshold, mask, cv::CMP_GE); // mask: 255 where true
    display.setTo(cv::Scalar(255,255,255), mask);
    // Clear counters so threshold must be reached again next frame
    fs.counts.setTo(0);
    // Reset per-frame counter (not currently displayed)
    fs.events_since_clear.store(0, memory_order_relaxed);
  }
  u64 total = fs.events_total.load(memory_order_relaxed);
  string text = "events_total=" + to_string(total) + "  threshold=" + to_string(fs.threshold) + "  [c] clear";
  cv::putText(display, text, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
  if (total == 0) {
    cv::putText(display, "(no events yet - file reading or pacing)", cv::Point(10,45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,50,200), 1, cv::LINE_AA);
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
