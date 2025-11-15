// Event visualizer (30 FPS) coloring pixels by polarity.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include "event_reader.hpp"

struct FrameState {
  mutex mtx;
  cv::Mat frame; // BGR
  atomic<bool> running{true};
  atomic<u64> events_total{0};
  atomic<u64> events_since_clear{0};
};

int run_event_visualizer(const string &dat_path) {
  const int DEFAULT_W = 1280;
  const int DEFAULT_H = 720;
  FrameState fs;
  fs.frame = cv::Mat(DEFAULT_H, DEFAULT_W, CV_8UC3, cv::Scalar(0,0,0));
  cv::namedWindow("Events", cv::WINDOW_AUTOSIZE);

  bool realtime = true;
  if (getenv("FAST_EVENTS")) {
    cout << "[INFO] FAST_EVENTS set: disabling realtime pacing for rapid fill." << endl;
    realtime = false;
  }

  thread reader([&fs, dat_path, realtime]() {
    DatHeaderInfo header; u64 count=0; u32 first_ts=0,last_ts=0; double wall_sec=0; u64 span_us=0;
    auto cb = [&fs](const Event &e) {
      if (e.x >= fs.frame.cols || e.y >= fs.frame.rows) return;
      lock_guard<mutex> lk(fs.mtx);
      fs.frame.at<cv::Vec3b>(e.y, e.x) = e.polarity ? cv::Vec3b(0,0,255) : cv::Vec3b(255,0,0);
      fs.events_total.fetch_add(1, memory_order_relaxed);
      fs.events_since_clear.fetch_add(1, memory_order_relaxed);
      auto total = fs.events_total.load(memory_order_relaxed);
      if (total % 1000000 == 0) {
        cout << "[DBG] processed events_total=" << total << endl;
      }
    };
    bool ok = stream_dat_events(dat_path, cb, &header, &count, &first_ts, &last_ts, &wall_sec, &span_us, realtime);
    if (ok) {
      // Reallocate frame to actual dimensions if different
      if ((header.width > 0 && header.height > 0) && (header.width != fs.frame.cols || header.height != fs.frame.rows)) {
        lock_guard<mutex> lk(fs.mtx);
        fs.frame = cv::Mat(header.height, header.width, CV_8UC3, cv::Scalar(0,0,0));
        cout << "[INFO] Resized frame to " << header.width << "x" << header.height << endl;
      }
      cout << "[DAT] header " << header.width << "x" << header.height
           << " version=" << header.version << " date=" << header.date << "\n"
           << "[DAT] event_type=" << header.event_type << " event_size=" << header.event_size << "\n"
           << "[DAT] events=" << count << " span_us=" << span_us << " wall_clock_s=" << wall_sec << "\n";
    } else {
      cout << "[DAT] failed reading file: " << dat_path << "\n";
    }
    fs.running.store(false);
  });

  const double target_fps = 30.0;
  using clock = chrono::steady_clock;
  auto frame_interval = chrono::duration_cast<clock::duration>(chrono::duration<double>(1.0/target_fps));
  auto next_frame = clock::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    cv::Mat display;
    {
      lock_guard<mutex> lk(fs.mtx);
      fs.frame.copyTo(display);
      fs.frame.setTo(cv::Scalar(0,0,0));
      fs.events_since_clear.store(0, memory_order_relaxed);
    }
    // Overlay counters
    u64 total = fs.events_total.load(memory_order_relaxed);
    string text = "events_total=" + to_string(total);
    cv::putText(display, text, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    // If frame is empty (no events this interval) show hint
    if (total == 0) {
      cv::putText(display, "(no events yet - file reading or pacing)", cv::Point(10,45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50,50,200), 1, cv::LINE_AA);
    }
    cv::imshow("Events", display);
    int key = cv::waitKey(1);
    if (key == 27 || key == 'q') { fs.running.store(false); }
    this_thread::sleep_until(next_frame);
  }
  if (reader.joinable()) reader.join();
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 2) {
    cout << argv[0] << " [optional DAT filepath]\n";
    return 1;
  }
  string path = (argc == 2) ? argv[1] : string("data/fan_const_rpm.dat");
  cout << "Event visualizer (30 FPS). File: " << path << "\nESC/q to quit.\n";
  return run_event_visualizer(path);
}
