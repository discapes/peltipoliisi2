#include "defs.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ATen/Functions.h>

TrajectoryPredictor g_trajectory_predictor;

using namespace std;

namespace
{
inline cv::Point2f clamp_norm_point(const cv::Point2f &pt)
{
  return cv::Point2f(std::clamp(pt.x, 0.0f, 1.0f), std::clamp(pt.y, 0.0f, 1.0f));
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
    history_norm_.clear();
    history_pixels_.clear();
    last_prediction_pixels_.clear();
    loaded_ = true;
  }

  if (error_msg)
    *error_msg = {};
  cout << "Trajectory predictor ready (input_len=" << input_len_
       << ", pred_len=" << pred_len_ << ") from " << cfg.model_path << endl;
  return true;
}

void TrajectoryPredictor::reset()
{
  lock_guard lock(mtx_);
  history_norm_.clear();
  history_pixels_.clear();
  last_prediction_pixels_.clear();
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
    run_inference_locked();
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

void TrajectoryPredictor::run_inference_locked()
{
  if (!loaded_ || input_len_ <= 0 || pred_len_ <= 0)
    return;

  torch::NoGradGuard guard;
  torch::Tensor input = torch::zeros({1, input_len_, 2}, torch::kFloat32);
  const cv::Point2f base = history_norm_.back();
  for (int i = 0; i < input_len_; ++i)
  {
    const cv::Point2f &pt = history_norm_[i];
    input[0][i][0] = pt.x - base.x;
    input[0][i][1] = pt.y - base.y;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(input);

  torch::Tensor rel;
  try
  {
    auto output = module_.forward(inputs);
    if (!output.isTensor())
      return;
    rel = output.toTensor().detach().to(torch::kCPU);
  }
  catch (const c10::Error &err)
  {
    cerr << "Trajectory predictor inference failed: " << err.what() << endl;
    return;
  }

  if (rel.dim() == 3 && rel.size(0) == 1)
    rel = rel.squeeze(0);
  if (rel.dim() != 2 || rel.size(0) < pred_len_ || rel.size(1) != 2)
    return;

  std::vector<cv::Point2f> preds;
  preds.reserve(pred_len_);
  for (int i = 0; i < pred_len_; ++i)
  {
    const float rel_x = rel[i][0].item<float>();
    const float rel_y = rel[i][1].item<float>();
    const float abs_x = std::clamp((base.x + rel_x) * image_width_, 0.0f, image_width_);
    const float abs_y = std::clamp((base.y + rel_y) * image_height_, 0.0f, image_height_);
    preds.emplace_back(abs_x, abs_y);
  }

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

