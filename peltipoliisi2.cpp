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
    // Before clearing, report count at cursor (if in bounds)
    int cx = fs.cursor_x.load(memory_order_relaxed);
    int cy = fs.cursor_y.load(memory_order_relaxed);
    if (cx >= 0 && cy >= 0 && cy < fs.counts.rows && cx < fs.counts.cols) {
      int cnt = fs.counts.at<int>(cy, cx);
      cout << "frame cursor=(" << cx << "," << cy << ") events=" << cnt << "\n";
      cout.flush();
    }
    // Process any pending dump requests for this frame: write matching timestamps to files.
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
    // Clear counters so threshold must be reached again next frame
    fs.counts.setTo(0);
    // Reset per-frame event vector and dump request queue; then advance frame index
    fs.frame_events.clear();
    fs.dump_requests.clear();
    fs.frame_index.fetch_add(1, memory_order_relaxed);
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
