// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

using u32 = uint32_t;
using u64 = uint64_t;

// Decoded event structure from 8-byte DAT record.
// t: timestamp in microseconds
// x, y: coordinates (14-bit each, stored in lower bits)
// polarity: 0 or 1
struct FrameEvent { u32 t; uint16_t x; uint16_t y; uint8_t polarity; };

struct FrameState {
  mutex mtx;
  cv::Mat frame; // BGR display buffer
  cv::Mat counts; // CV_32SC1 per-pixel event counters
  // Events collected in the current frame (reset each render)
  vector<FrameEvent> frame_events;
  atomic<bool> running{true};
  struct RpmStats {
    double median = numeric_limits<double>::quiet_NaN();
    size_t sampled = 0;
    size_t valid = 0;
  };
  RpmStats rpm_stats{};
  double cluster_eps{6.5};
  size_t cluster_min_points{12};
  // Background clustering communication
  mutex cluster_request_mtx;
  condition_variable cluster_request_cv;
  vector<double> cluster_request_coords;
  vector<FrameEvent> cluster_request_events;
  u64 cluster_request_frame{0};
  bool cluster_request_ready{false};
  mutex overlay_mtx;
  struct ClusterOverlay {
    cv::Rect box;
    double rpm = numeric_limits<double>::quiet_NaN();
    cv::Scalar color{0, 0, 255};
  };
  vector<ClusterOverlay> overlay_data;
  u64 overlay_frame{0};
};
extern FrameState fs;

// Estimate RPM from timestamps (µs) of events in a small region using FFTW.
// - events: vector of Event; only the timestamp field (t, in microseconds) is used
// - num_blades: number of blades on the propeller (defaults to 2)
// Returns NaN if not enough data or no clear peak.
double estimate_rpm_from_events(const vector<FrameEvent> &events, int num_blades = 2);

struct DatHeaderInfo {
    int width=-1, height=-1, version=-1;
    string date;
    int event_type=-1, event_size=-1;
};

// Reads a DAT file and streams events paced to real time.
// Additionally implements a sliding time window (default 50ms):
//  - At each event arrival time, invokes callback(e, +1) to indicate an increment.
//  - When an event leaves the window (t + window_us), invokes callback(e, -1) to indicate a decrement.
// The function sleeps until the earlier of the next arrival or next expiry to maintain pacing.
// Returns true on success; false otherwise.
bool stream_dat_events(const string &path,
                       const function<void(const FrameEvent &, int delta)> &callback,
                       DatHeaderInfo *out_header=nullptr,
                       u32 window_us = 50'000);

void cluster_worker();