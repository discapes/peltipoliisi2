#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include "event_reader.hpp"
using namespace std;

// Small struct to hold shared square state and window params
struct SquareState {
  int W = 600;
  int H = 400;
  int S = 50; // side length
  double px = 0.0; // top-left x
  double py = 0.0; // top-left y
  double vx = 200.0; // px/s
  double vy = 150.0; // px/s
  std::mutex mtx;
  std::atomic<bool> running{false};
};

// Initialize the square state to centered position
void init_state(SquareState &s) {
  s.px = (s.W - s.S) / 2.0;
  s.py = (s.H - s.S) / 2.0;
  s.running.store(true);
}

// Simulation thread function: advances physics using real delta time
void sim_thread_func(SquareState &s) {
  using clock = std::chrono::steady_clock;
  auto last = clock::now();
  while (s.running.load()) {
    auto now = clock::now();
    std::chrono::duration<double> elapsed = now - last;
    last = now;
    double dt = elapsed.count();

    {
      std::lock_guard<std::mutex> lk(s.mtx);
      s.px += s.vx * dt;
      s.py += s.vy * dt;

      // bounce: keep square fully inside
      if (s.px <= 0.0) { s.px = 0.0; s.vx = -s.vx; }
      if (s.px + s.S >= s.W) { s.px = s.W - s.S; s.vx = -s.vx; }
      if (s.py <= 0.0) { s.py = 0.0; s.vy = -s.vy; }
      if (s.py + s.S >= s.H) { s.py = s.H - s.S; s.vy = -s.vy; }
    }

    // small sleep to avoid busy loop
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }
}

// Render loop runs at target FPS and handles user input; returns when quitting
int render_loop(SquareState &s, const std::string &window_name = "MyWindow") {
  // Create window and image buffer
  cv::Mat img(s.H, s.W, CV_8UC3);
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  const double target_fps = 30.0;
  using clock = std::chrono::steady_clock;
  const clock::duration frame_time = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / target_fps));
  auto next_frame = clock::now();

  while (s.running.load()) {
    next_frame += frame_time;

    double rx, ry;
    int Sloc;
    {
      std::lock_guard<std::mutex> lk(s.mtx);
      rx = s.px;
      ry = s.py;
      Sloc = s.S;
    }

    // draw
    img.setTo(cv::Scalar(255, 0, 0));
    cv::rectangle(img, cv::Point((int)std::round(rx), (int)std::round(ry)),
                  cv::Point((int)std::round(rx) + Sloc, (int)std::round(ry) + Sloc),
                  cv::Scalar(0, 255, 0), cv::FILLED);

    cv::imshow(window_name, img);

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {
      s.running.store(false);
      break;
    }

    std::this_thread::sleep_until(next_frame);
  }

  return 0;
}

// Run the application: init, start sim thread, render, join and cleanup
int run_app() {
  SquareState s;
  init_state(s);

  std::thread sim(sim_thread_func, std::ref(s));

  int res = render_loop(s, "MyWindow");

  // ensure sim stops and join
  s.running.store(false);
  if (sim.joinable()) sim.join();

  return res;
}

int main(int argc, char **argv) {
  if (argc != 1) {
    std::cout << argv[0] << " takes no arguments.\n";
    return 1;
  }

  std::cout << "initializing...\n";
  std::cout << "Press ESC or 'q' to quit.\n";

  // Kick off DAT reader in a background thread (non-blocking for UI).
  // Assumption: file path is fixed; change as needed.
  const std::string dat_path = "data/fan_const_rpm.dat"; // placeholder
  std::thread dat_reader([dat_path]() {
    DatHeaderInfo header;
    std::uint64_t count = 0;
    std::uint32_t first_ts = 0, last_ts = 0;
    double wall_sec = 0.0;
    std::uint64_t span_us = 0;
    bool ok = stream_dat_events(dat_path,
                                nullptr, // no per-event processing yet
                                &header,
                                &count,
                                &first_ts,
                                &last_ts,
                                &wall_sec,
                                &span_us,
                                true); // realtime pacing enabled
    if (ok) {
      std::cout << "[DAT] Parsed header: width=" << header.width
                << " height=" << header.height
                << " version=" << header.version
                << " date=" << header.date
                << " events=" << count << "\n";
      std::cout << "[DAT] Data timespan: " << span_us << " us (first=" << first_ts
                << " last=" << last_ts << ") wall_clock=" << wall_sec << " s\n";
    } else {
      std::cout << "[DAT] Reader finished with error or file missing (" << dat_path << ").\n";
    }
  });

  int res = run_app();
  if (dat_reader.joinable()) dat_reader.join();
  return res;
}
