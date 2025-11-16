#include "defs.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ATen/Functions.h>

TrajectoryPredictor g_trajectory_predictor;

using namespace std;

TrajectoryPredictor::~TrajectoryPredictor()
{
  stop_worker();
}

namespace
{
constexpr int EVENT_TENSOR_WIDTH = 320;
constexpr int EVENT_TENSOR_HEIGHT = 180;

inline cv::Point2f clamp_norm_point(const cv::Point2f &pt)
{
  return cv::Point2f(std::clamp(pt.x, 0.0f, 1.0f), std::clamp(pt.y, 0.0f, 1.0f));
}

torch::Tensor build_event_tensor(const vector<FrameEvent> &events,
                                 int timesteps,
                                 float sensor_width,
                                 float sensor_height)
{
  auto base = torch::zeros({2, EVENT_TENSOR_HEIGHT, EVENT_TENSOR_WIDTH}, torch::kFloat32);
  float scale_x = EVENT_TENSOR_WIDTH / sensor_width;
  float scale_y = EVENT_TENSOR_HEIGHT / sensor_height;

  auto accessor = base.accessor<float, 3>();
  for (const auto &ev : events)
  {
    int x = std::clamp(static_cast<int>(ev.x * scale_x), 0, EVENT_TENSOR_WIDTH - 1);
    int y = std::clamp(static_cast<int>(ev.y * scale_y), 0, EVENT_TENSOR_HEIGHT - 1);
    int channel = ev.polarity ? 1 : 0;
    accessor[channel][y][x] += 1.0f;
  }

  auto repeated = base.unsqueeze(0).repeat({timesteps, 1, 1, 1}); // (T, 2, H, W)
  return repeated.unsqueeze(0); // (1, T, 2, H, W)
}
} // namespace

bool TrajectoryPredictor::load(const TrajectoryPredictorConfig &cfg, std::string *error_msg)
{
  namespace fs = std::filesystem;
  if (cfg.model_path.empty())
  {
    if (error_msg)
      *error_msg = "No TorchScript model path provided.";
    return false;
  }
  if (!fs::exists(cfg.model_path))
  {
    if (error_msg)
      *error_msg = "TorchScript model not found at " + cfg.model_path;
    return false;
  }

  try
  {
    module_ = torch::jit::load(cfg.model_path);
  }
  catch (const c10::Error &err)
  {
    if (error_msg)
      *error_msg = std::string("Failed to load TorchScript model: ") + err.what();
    return false;
  }

  module_.eval();
  module_.to(torch::kCPU);

  bool model_use_events = true;
  if (module_.hasattr("use_events"))
  {
    try
    {
      model_use_events = module_.attr("use_events").toBool();
    }
    catch (const c10::Error &)
    {
    }
  }

  int input_len = cfg.input_len;
  int pred_len = cfg.pred_len;
  if (module_.hasattr("input_len"))
  {
    try
    {
      input_len = static_cast<int>(module_.attr("input_len").toInt());
    }
    catch (const c10::Error &)
    {
    }
  }
  if (module_.hasattr("pred_len"))
  {
    try
    {
      pred_len = static_cast<int>(module_.attr("pred_len").toInt());
    }
    catch (const c10::Error &)
    {
    }
  }

  {
    lock_guard lock(mtx_);
    input_len_ = input_len;
    pred_len_ = pred_len;
    image_width_ = max(cfg.image_width, 1.0f);
    image_height_ = max(cfg.image_height, 1.0f);
    event_window_us_ = cfg.event_window_us;
    model_use_events_ = model_use_events;
    use_events_ = model_use_events;
    pending_inference_ = false;
    history_norm_.clear();
    history_pixels_.clear();
    last_prediction_pixels_.clear();
    loaded_ = true;
  }

  if (error_msg)
    *error_msg = {};
  cout << "Trajectory predictor ready (input_len=" << input_len_
       << ", pred_len=" << pred_len_
       << ", events=" << (use_events_ ? "enabled" : "disabled")
       << ") from " << cfg.model_path << endl;
  ensure_worker();
  return true;
}

void TrajectoryPredictor::reset()
{
  lock_guard lock(mtx_);
  history_norm_.clear();
  history_pixels_.clear();
  last_prediction_pixels_.clear();
  pending_inference_ = false;
}

void TrajectoryPredictor::add_observation(const cv::Point2f &pixel)
{
  lock_guard lock(mtx_);
  const cv::Point2f raw_norm(pixel.x / image_width_, pixel.y / image_height_);
  const cv::Point2f norm(clamp_norm_point(raw_norm));
  history_norm_.push_back(norm);
  history_pixels_.push_back(pixel);
  while (history_norm_.size() > static_cast<size_t>(max(1, input_len_)))
  {
    history_norm_.pop_front();
    history_pixels_.pop_front();
  }
  if (loaded_ && static_cast<int>(history_norm_.size()) == input_len_)
  {
    static int history_log_counter = 0;
    if ((history_log_counter++ % 90) == 0)
    {
      std::ostringstream oss;
      oss << "[trajectory] history (norm) ";
      for (const auto &pt : history_norm_)
      {
        oss << "(" << std::fixed << std::setprecision(3) << pt.x << ","
            << std::fixed << std::setprecision(3) << pt.y << ") ";
      }
      cerr << oss.str() << endl;
    }

    request_inference_locked();
  }
}

std::vector<cv::Point2f> TrajectoryPredictor::latest_prediction_pixels() const
{
  lock_guard lock(mtx_);
  return last_prediction_pixels_;
}

std::vector<cv::Point2f> TrajectoryPredictor::observation_trace_pixels() const
{
  lock_guard lock(mtx_);
  return std::vector<cv::Point2f>(history_pixels_.begin(), history_pixels_.end());
}

bool TrajectoryPredictor::ready() const
{
  lock_guard lock(mtx_);
  return loaded_ && static_cast<int>(history_norm_.size()) == input_len_ && !last_prediction_pixels_.empty();
}

int TrajectoryPredictor::pred_len() const
{
  lock_guard lock(mtx_);
  return pred_len_;
}

void TrajectoryPredictor::request_inference_locked()
{
  if (!loaded_)
    return;
  if (static_cast<int>(history_norm_.size()) != input_len_)
    return;
  pending_inference_ = true;
  cv_.notify_one();
}

void TrajectoryPredictor::inference_thread_loop()
{
  std::vector<cv::Point2f> history_norm_copy;
  std::vector<cv::Point2f> history_pixels_copy;
  std::vector<FrameEvent> active_events;

  while (true)
  {
    float image_width = 0.0f;
    float image_height = 0.0f;
    int input_len = 0;
    int pred_len = 0;
    int event_window_us = 0;
    bool use_events = false;

    {
      std::unique_lock lock(mtx_);
      cv_.wait(lock, [this] {
        return stop_worker_ || (pending_inference_ && loaded_ &&
                                static_cast<int>(history_norm_.size()) == input_len_);
      });
      if (stop_worker_)
        break;
      pending_inference_ = false;
      history_norm_copy.assign(history_norm_.begin(), history_norm_.end());
      history_pixels_copy.assign(history_pixels_.begin(), history_pixels_.end());
      image_width = image_width_;
      image_height = image_height_;
      input_len = input_len_;
      pred_len = pred_len_;
      event_window_us = event_window_us_;
      use_events = use_events_;
    }

    active_events.clear();
    if (use_events)
    {
      snapshot_active_events(active_events);
    }
    auto preds = run_inference(history_norm_copy,
                               history_pixels_copy,
                               active_events,
                               image_width,
                               image_height,
                               input_len,
                               pred_len,
                               use_events);
    if (preds.empty())
      continue;

    std::lock_guard lock(mtx_);
    last_prediction_pixels_ = std::move(preds);
    static int prediction_log_counter = 0;
    if ((prediction_log_counter++ % 90) == 0)
    {
      std::ostringstream oss;
      oss << "[trajectory] predictions ";
      for (const auto &pt : last_prediction_pixels_)
      {
        oss << "(" << std::fixed << std::setprecision(1) << pt.x << ","
            << std::fixed << std::setprecision(1) << pt.y << ") ";
      }
      cerr << oss.str() << endl;
    }
  }
}

std::vector<cv::Point2f> TrajectoryPredictor::run_inference(
    const std::vector<cv::Point2f> &history_norm,
    const std::vector<cv::Point2f> &history_pixels,
    const std::vector<FrameEvent> &active_events,
    float image_width,
    float image_height,
    int input_len,
    int pred_len,
    bool use_events)
{
  std::vector<cv::Point2f> empty;
  if (history_norm.empty() || static_cast<int>(history_norm.size()) != input_len ||
      pred_len <= 0)
  {
    return empty;
  }
  if (history_pixels.size() != history_norm.size())
    return empty;

  torch::NoGradGuard guard;

  torch::Tensor pos_input = torch::zeros({1, input_len, 2}, torch::kFloat32);
  const cv::Point2f base = history_norm.back();
  for (int i = 0; i < input_len; ++i)
  {
    const cv::Point2f &pt = history_norm[i];
    pos_input[0][i][0] = pt.x - base.x;
    pos_input[0][i][1] = pt.y - base.y;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(pos_input);

  if (use_events)
  {
    torch::Tensor event_input = build_event_tensor(
        active_events,
        input_len,
        std::max(1.0f, image_width),
        std::max(1.0f, image_height));
    inputs.emplace_back(event_input);
  }

  torch::Tensor rel;
  try
  {
    auto output = module_.forward(inputs);
    if (!output.isTensor())
      return empty;
    rel = output.toTensor().detach().to(torch::kCPU);
  }
  catch (const c10::Error &err)
  {
    cerr << "Trajectory predictor inference failed: " << err.what() << endl;
    return empty;
  }

  if (rel.dim() == 3 && rel.size(0) == 1)
    rel = rel.squeeze(0);
  if (rel.dim() != 2 || rel.size(0) < pred_len || rel.size(1) != 2)
    return empty;

  std::vector<cv::Point2f> preds;
  preds.reserve(pred_len);
  for (int i = 0; i < pred_len; ++i)
  {
    const float rel_x = rel[i][0].item<float>();
    const float rel_y = rel[i][1].item<float>();
    const float abs_x = std::clamp((base.x + rel_x) * image_width, 0.0f, image_width);
    const float abs_y = std::clamp((base.y + rel_y) * image_height, 0.0f, image_height);
    preds.emplace_back(abs_x, abs_y);
  }
  return preds;
}

void TrajectoryPredictor::ensure_worker()
{
  if (inference_thread_.joinable())
    return;
  stop_worker_ = false;
  inference_thread_ = std::thread(&TrajectoryPredictor::inference_thread_loop, this);
}

void TrajectoryPredictor::stop_worker()
{
  {
    std::lock_guard lock(mtx_);
    stop_worker_ = true;
    cv_.notify_all();
  }
  if (inference_thread_.joinable())
    inference_thread_.join();
  stop_worker_ = false;
  pending_inference_ = false;
}

