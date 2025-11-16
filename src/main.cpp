// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include "defs.hpp"
#include <array>
#include <cstdlib>
#include <optional>
#include <thread>

FrameState fs{
    .counts = cv::Mat(DEFAULT_H, DEFAULT_W, CV_32SC1, cv::Scalar(0))};

// On each incoming or expiring event: delta=+1 to increment, delta=-1 to decrement.
inline void event_pixel_callback(const FrameEvent &e, int delta)
{
  lock_guard<mutex> lk(fs.mtx);
  int &cnt = fs.counts.at<int>(e.y, e.x);
  cnt += delta;
}

// Thread body: stream DAT events, resize frame after header, log summary.
void run_dat_reader(const string &dat_path)
{
  DatHeaderInfo header;
  // Uses default 50ms window; stream emits +1 on arrival and -1 on expiry.
  stream_dat_events(dat_path, event_pixel_callback, &header, SLIDING_WINDOW_US, fs.running);
  fs.running.store(false);
}

optional<cv::Point2f> compute_rotor_centroid(const vector<FrameState::TrackedRotor> &rotors)
{
  if (rotors.empty())
    return nullopt;

  cv::Point2d weighted_sum(0.0, 0.0);
  double weight_sum = 0.0;

  for (const auto &rotor : rotors)
  {
    const cv::Point2d center(rotor.box.x + rotor.box.width * 0.5,
                             rotor.box.y + rotor.box.height * 0.5);

    double weight = 1.0;
    if (!rotor.rpm_history.empty())
    {
      const double rpm = rotor.rpm_history.back();
      if (isfinite(rpm) && rpm > 0.0)
        weight = rpm;
    }

    weighted_sum += center * weight;
    weight_sum += weight;
  }

  if (weight_sum <= 0.0)
    weight_sum = static_cast<double>(rotors.size());

  const cv::Point2d avg = weighted_sum / weight_sum;
  return cv::Point2f(static_cast<float>(avg.x), static_cast<float>(avg.y));
}

void update_rotor_tracking(vector<FrameState::ClusterOverlay> &new_overlays)
{
  // Simple IoU based tracking
  const double iou_threshold = 0.3;
  vector<bool> matched_new(new_overlays.size(), false);

  lock_guard<mutex> lk(fs.mtx);

  // Increment unseen counters and remove old trackers
  for (auto &rotor : fs.tracked_rotors)
  {
    rotor.frames_unseen++;
  }
  fs.tracked_rotors.erase(
      remove_if(fs.tracked_rotors.begin(), fs.tracked_rotors.end(),
                [](const auto &r)
                { return r.frames_unseen > 5; }),
      fs.tracked_rotors.end());

  // Match existing trackers to new overlays
  for (auto &rotor : fs.tracked_rotors)
  {
    double best_iou = 0;
    int best_match_idx = -1;

    for (size_t i = 0; i < new_overlays.size(); ++i)
    {
      if (matched_new[i])
        continue;
      double iou = (double)(rotor.box & new_overlays[i].box).area() /
                   (double)(rotor.box | new_overlays[i].box).area();
      if (iou > best_iou)
      {
        best_iou = iou;
        best_match_idx = i;
      }
    }

    if (best_match_idx != -1 && best_iou > iou_threshold)
    {
      rotor.box = new_overlays[best_match_idx].box;
      if (isfinite(new_overlays[best_match_idx].rpm))
      {
        rotor.rpm_history.push_back(new_overlays[best_match_idx].rpm);
        while (rotor.rpm_history.size() > fs.RPM_HISTORY_SIZE)
        {
          rotor.rpm_history.pop_front();
        }
      }
      rotor.frames_unseen = 0;
      matched_new[best_match_idx] = true;
    }
  }

  // Add new trackers for unmatched overlays
  static const array<cv::Scalar, 8> palette = {
      cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
      cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
      cv::Scalar(128, 255, 0), cv::Scalar(0, 128, 255)};

  for (size_t i = 0; i < new_overlays.size(); ++i)
  {
    if (!matched_new[i])
    {
      FrameState::TrackedRotor new_rotor;
      new_rotor.id = fs.next_rotor_id++;
      new_rotor.box = new_overlays[i].box;
      new_rotor.color = palette[new_rotor.id % palette.size()];
      if (isfinite(new_overlays[i].rpm))
      {
        new_rotor.rpm_history.push_back(new_overlays[i].rpm);
      }
      fs.tracked_rotors.push_back(new_rotor);
    }
  }
}

void rpm_counter_loop()
{
  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0 / CLUSTER_FPS));
  auto next_frame = steady::now();
  while (fs.running.load())
  {
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

    vector<FrameState::TrackedRotor> tracked_snapshot;
    {
      lock_guard<mutex> lk(fs.mtx);
      tracked_snapshot = fs.tracked_rotors;
    }

    if (auto centroid = compute_rotor_centroid(tracked_snapshot))
    {
      g_trajectory_predictor.add_observation(*centroid);
    }

    this_thread::sleep_until(next_frame);
  }
}

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 3)
  {
    cout << "Usage: " << argv[0] << " <DAT filepath> [trajectory_model.ts]\n";
    return 1;
  }

  string dat_path = argv[1];
  string model_path;
  if (argc == 3)
  {
    model_path = argv[2];
  }
  else if (const char *env = std::getenv("TRAJECTORY_MODEL"))
  {
    model_path = env;
  }
  else
  {
    model_path = "artifacts/trajectory_gru.ts";
  }

  if (!model_path.empty())
  {
    TrajectoryPredictorConfig cfg;
    cfg.model_path = model_path;
    cfg.image_width = static_cast<float>(fs.counts.cols);
    cfg.image_height = static_cast<float>(fs.counts.rows);
    string error_msg;
    if (!g_trajectory_predictor.load(cfg, &error_msg))
    {
      if (!error_msg.empty())
        cerr << "Trajectory predictor disabled: " << error_msg << endl;
    }
  }

  thread reader(run_dat_reader, dat_path);
  thread rpm_counter(rpm_counter_loop);
  render_thread_loop();

  if (reader.joinable())
    reader.join();
  if (rpm_counter.joinable())
    rpm_counter.join();
}
