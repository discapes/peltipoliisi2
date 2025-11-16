// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include "defs.hpp"
#include <random>
#include <thread>



using steady = chrono::steady_clock;

FrameState fs;

// On each incoming or expiring event: delta=+1 to increment, delta=-1 to decrement.
inline void event_pixel_callback(const FrameEvent &e, int delta) {
  if (e.x >= fs.frame.cols || e.y >= fs.frame.rows) return;
  lock_guard<mutex> lk(fs.mtx);
  // Increment counter; bounds are checked above.
  int &cnt = fs.counts.at<int>(e.y, e.x);
  if (delta > 0) {
    if (cnt < INT32_MAX) ++cnt;
  // No longer storing per-event copies; we'll snapshot the active deque instead.
  } else if (delta < 0) {
    if (cnt > 0) --cnt; // clamp at 0
  }
}

// Thread body: stream DAT events, resize frame after header, log summary.
void run_dat_reader(const string &dat_path) {
  DatHeaderInfo header;
  // Uses default 50ms window; stream emits +1 on arrival and -1 on expiry.
  stream_dat_events(dat_path, event_pixel_callback, &header, SLIDING_WINDOW_US);
  fs.running.store(false);
}

void update_rotor_tracking(vector<FrameState::ClusterOverlay> &new_overlays) {
  // Simple IoU based tracking
  const double iou_threshold = 0.3;
  vector<bool> matched_new(new_overlays.size(), false);

  lock_guard<mutex> lk(fs.mtx);

  // Increment unseen counters and remove old trackers
  for (auto &rotor : fs.tracked_rotors) {
    rotor.frames_unseen++;
  }
  fs.tracked_rotors.erase(
      remove_if(fs.tracked_rotors.begin(), fs.tracked_rotors.end(),
                [](const auto &r) { return r.frames_unseen > 5; }),
      fs.tracked_rotors.end());

  // Match existing trackers to new overlays
  for (auto &rotor : fs.tracked_rotors) {
    double best_iou = 0;
    int best_match_idx = -1;

    for (size_t i = 0; i < new_overlays.size(); ++i) {
      if (matched_new[i]) continue;
      double iou = (double)(rotor.box & new_overlays[i].box).area() /
                   (double)(rotor.box | new_overlays[i].box).area();
      if (iou > best_iou) {
        best_iou = iou;
        best_match_idx = i;
      }
    }

    if (best_match_idx != -1 && best_iou > iou_threshold) {
      rotor.box = new_overlays[best_match_idx].box;
      if (isfinite(new_overlays[best_match_idx].rpm)) {
        rotor.rpm_history.push_back(new_overlays[best_match_idx].rpm);
        while (rotor.rpm_history.size() > fs.RPM_HISTORY_SIZE) {
          rotor.rpm_history.pop_front();
        }
      }
      rotor.frames_unseen = 0;
      matched_new[best_match_idx] = true;
    }
  }

  // Add new trackers for unmatched overlays
  static const array<cv::Scalar, 8> palette = {
      cv::Scalar(0, 0, 255),   cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),
      cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
      cv::Scalar(128, 255, 0), cv::Scalar(0, 128, 255)};

  for (size_t i = 0; i < new_overlays.size(); ++i) {
    if (!matched_new[i]) {
      FrameState::TrackedRotor new_rotor;
      new_rotor.id = fs.next_rotor_id++;
      new_rotor.box = new_overlays[i].box;
      new_rotor.color = palette[new_rotor.id % palette.size()];
      if (isfinite(new_overlays[i].rpm)) {
        new_rotor.rpm_history.push_back(new_overlays[i].rpm);
      }
      fs.tracked_rotors.push_back(new_rotor);
    }
  }
}


void rpm_counter_loop() {
  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/CLUSTER_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    auto stats = estimate_rpms();
    {
      lock_guard<mutex> lk(fs.mtx);
      fs.rpm_stats = stats;
    }

    vector<FrameState::ClusterOverlay> overlays_copy;
    {
      lock_guard<mutex> lk(fs.overlay_mtx);
      overlays_copy = fs.overlay_data;
    }
    update_rotor_tracking(overlays_copy);

    this_thread::sleep_until(next_frame);
  }
}


int main(int argc, char **argv) {
  // Usage: program <DAT filepath> [threshold]
  if (argc < 2 || argc > 2) {
    cout << "Usage: " << argv[0] << " <DAT filepath>\n";
    return 1;
  }
  string dat_path = argv[1];
  cout << "Event visualizer (30 FPS). File: " << dat_path << "\nESC/q to quit, C to clear.\n";

  fs.frame = cv::Mat(DEFAULT_H, DEFAULT_W, CV_8UC3, cv::Scalar(0,0,0));
  fs.counts = cv::Mat(DEFAULT_H, DEFAULT_W, CV_32SC1, cv::Scalar(0));
  cv::namedWindow("Events", cv::WINDOW_AUTOSIZE);

  thread reader(run_dat_reader, dat_path);
  thread rpm_counter(rpm_counter_loop);
  // Background cluster thread removed; clustering is now invoked synchronously.

  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0/TARGET_FPS));
  auto next_frame = steady::now();
  while (fs.running.load()) {
    next_frame += frame_interval;
    if (!render_frame()) break;
    this_thread::sleep_until(next_frame);
  }

  fs.running.store(false);
  if (reader.joinable()) reader.join();
  if (rpm_counter.joinable()) rpm_counter.join();
  return 0;
}
