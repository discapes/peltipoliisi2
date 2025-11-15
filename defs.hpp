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

#include <array>
#include <condition_variable>
#include <unordered_map>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include "event_reader.hpp"
using namespace std;

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
  double cluster_eps{6.5};
  size_t cluster_min_points{12};
  // Background clustering communication
  std::mutex cluster_request_mtx;
  std::condition_variable cluster_request_cv;
  std::vector<double> cluster_request_coords;
  std::vector<FrameState::FrameEvent> cluster_request_events;
  u64 cluster_request_frame{0};
  bool cluster_request_ready{false};
  std::mutex overlay_mtx;
  struct ClusterOverlay {
    cv::Rect box;
    double rpm = std::numeric_limits<double>::quiet_NaN();
    cv::Scalar color{0, 0, 255};
  };
  std::vector<ClusterOverlay> overlay_data;
  u64 overlay_frame{0};
};
extern FrameState fs;