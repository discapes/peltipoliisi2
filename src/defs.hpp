// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include <cstdint>
#include <mutex>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <deque>

using namespace std;
using steady = chrono::steady_clock;

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int DEFAULT_W = 1280;
constexpr int DEFAULT_H = 720;
constexpr int SLIDING_WINDOW_US = 50'000;
constexpr int EVENT_COUNT_THRESHOLD = 10;
constexpr double TARGET_FPS = 60.0;
constexpr double CLUSTER_FPS = 60.0;
constexpr int SAMPLE_POINTS = 200;
constexpr int SAMPLE_RANGE = 1;
constexpr int NUM_BLADES = 2;
constexpr int CLUSTER_BOX_PADDING = 6;

// Decoded event structure from 8-byte DAT record.
// t: timestamp in microseconds
// x, y: coordinates (14-bit each, stored in lower bits)
// polarity: 0 or 1
struct FrameEvent
{
  u32 t;
  uint16_t x;
  uint16_t y;
  uint8_t polarity;
};

// A precomputed RPM sample at location (x, y)
struct RpmSample
{
  int x;
  int y;
  double rpm;
};

struct FrameState
{
  mutex mtx;
  cv::Mat counts;
  atomic<bool> running{true};
  struct RpmStats
  {
    double median = numeric_limits<double>::quiet_NaN();
    size_t sampled = 0;
    size_t valid = 0;
  };
  RpmStats rpm_stats{};
  // Keep a short history of global RPM medians to smooth bursts.
  vector<double> rpm_median_history;
  size_t rpm_median_history_limit{7};
  double cluster_eps{6.5};
  size_t cluster_min_points{12};
  // Background clustering communication
  mutex overlay_mtx;
  struct ClusterOverlay
  {
    cv::Rect box;
    double rpm = numeric_limits<double>::quiet_NaN();
    cv::Scalar color{0, 0, 255};
  };
  vector<ClusterOverlay> overlay_data;
  u64 overlay_frame{0};

  // Rotor tracking
  struct TrackedRotor
  {
    int id;
    cv::Rect box;
    cv::Scalar color;
    deque<double> rpm_history;
    int frames_unseen = 0;
  };
  vector<TrackedRotor> tracked_rotors;
  int next_rotor_id = 0;
  static constexpr int RPM_HISTORY_SIZE = 50;
};
extern FrameState fs;

// Estimate RPM from timestamps (µs) of events in a small region using FFTW.
// - events: vector of Event; only the timestamp field (t, in microseconds) is used
// - num_blades: number of blades on the propeller (defaults to 2)
// Returns NaN if not enough data or no clear peak.
double rpm_from_fft(const vector<FrameEvent> &events, int num_blades = 2);

FrameState::RpmStats estimate_rpms();

struct DatHeaderInfo
{
  int width = -1, height = -1, version = -1;
  string date;
  int event_type = -1, event_size = -1;
};

// Reads a DAT file and streams events paced to real time.
// Additionally implements a sliding time window (default 50ms):
//  - At each event arrival time, invokes callback(e, +1) to indicate an increment.
//  - When an event leaves the window (t + window_us), invokes callback(e, -1) to indicate a decrement.
// The function sleeps until the earlier of the next arrival or next expiry to maintain pacing.
// Returns true on success; false otherwise.
bool stream_dat_events(const string &path,
                       const function<void(const FrameEvent &, int delta)> &callback,
                       DatHeaderInfo *out_header,
                       u32 window_us,
                       const atomic<bool> &stop_flag);

// Snapshot the current deque of active events (those within the sliding window)
// maintained by the event reader. Thread-safe copy.
void snapshot_active_events(vector<FrameEvent> &out);

// Compute clusters and overlays from sampled coords and events.
// - coords: packed [x0, y0, x1, y1, ...]
// - events: events for the current interval
// - cluster_eps, cluster_min_points: DBSCAN parameters
// Returns a list of overlays (bounding boxes, colors, optional RPM per cluster).
vector<FrameState::ClusterOverlay> cluster_worker(
    vector<double> coords,
    vector<RpmSample> rpm_samples,
    double cluster_eps,
    size_t cluster_min_points);

void render_thread_loop();


struct TrajectoryPredictorConfig
{
  std::string model_path;
  int input_len = 20;
  int pred_len = 10;
  float image_width = static_cast<float>(DEFAULT_W);
  float image_height = static_cast<float>(DEFAULT_H);
};

class TrajectoryPredictor
{
public:
  bool load(const TrajectoryPredictorConfig &cfg, std::string *error_msg = nullptr);
  void reset();
  void add_observation(const cv::Point2f &pixel);

  std::vector<cv::Point2f> latest_prediction_pixels() const;
  std::vector<cv::Point2f> observation_trace_pixels() const;
  bool ready() const;
  int pred_len() const;

private:
  void run_inference_locked();

  mutable std::mutex mtx_;
  bool loaded_{false};
  int input_len_{0};
  int pred_len_{0};
  float image_width_{static_cast<float>(DEFAULT_W)};
  float image_height_{static_cast<float>(DEFAULT_H)};
  std::deque<cv::Point2f> history_norm_;
  std::deque<cv::Point2f> history_pixels_;
  std::vector<cv::Point2f> last_prediction_pixels_;
  torch::jit::script::Module module_;
};

extern TrajectoryPredictor g_trajectory_predictor;