#include "defs.hpp"
#include <chrono>
#include <cmath>
#include <optional>
#include <thread>

namespace
{
void draw_text_line(cv::Mat &display,
                    int &cursor_y,
                    const std::string &line,
                    const cv::Scalar &text_color = cv::Scalar(0, 255, 255),
                    const cv::Scalar *marker = nullptr)
{
  if (marker)
  {
    cv::rectangle(display,
                  cv::Point(10, cursor_y - 12),
                  cv::Point(22, cursor_y - 2),
                  *marker, cv::FILLED);
    cv::putText(display, line, cv::Point(26, cursor_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                text_color, 1, cv::LINE_AA);
  }
  else
  {
    cv::putText(display, line, cv::Point(10, cursor_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                text_color, 1, cv::LINE_AA);
  }
  cursor_y += 18;
}
}

void render_frame(cv::Mat &frame)
{
  cv::Mat display;
  FrameState::RpmStats rpm_stats_copy;
  std::vector<FrameState::TrackedRotor> tracked_rotors_copy;

  display.create(frame.rows, frame.cols, CV_8UC3);
  display.setTo(cv::Scalar(0, 0, 0));

  cv::Mat mask;
  cv::compare(fs.counts, EVENT_COUNT_THRESHOLD, mask, cv::CMP_GE);
  display.setTo(cv::Scalar(255, 255, 255), mask);
  rpm_stats_copy = fs.rpm_stats;

  tracked_rotors_copy = fs.tracked_rotors;

  for (const auto &rotor : tracked_rotors_copy)
  {
    cv::Rect padded = rotor.box;
    padded.x = max(0, padded.x - CLUSTER_BOX_PADDING);
    padded.y = max(0, padded.y - CLUSTER_BOX_PADDING);
    padded.width = min(display.cols - padded.x,
                       rotor.box.width + 2 * CLUSTER_BOX_PADDING);
    padded.height = min(display.rows - padded.y,
                        rotor.box.height + 2 * CLUSTER_BOX_PADDING);
    cv::rectangle(display, padded, rotor.color, 2);
  }

  auto to_int_points = [](const std::vector<cv::Point2f> &src)
  {
    std::vector<cv::Point> dst;
    dst.reserve(src.size());
    for (const auto &pt : src)
    {
      dst.emplace_back(cvRound(pt.x), cvRound(pt.y));
    }
    return dst;
  };

  auto observed_trace = to_int_points(g_trajectory_predictor.observation_trace_pixels());
  optional<cv::Point> last_observed;
  if (observed_trace.size() >= 2)
  {
    cv::polylines(display, observed_trace, false, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
  }
  if (!observed_trace.empty())
  {
    last_observed = observed_trace.back();
  }
  std::optional<cv::Point2f> avg_prediction_delta;
  auto predicted_trace = to_int_points(g_trajectory_predictor.latest_prediction_pixels());
  if (!predicted_trace.empty())
  {
    vector<cv::Point> prediction_path = predicted_trace;
    if (last_observed)
    {
      prediction_path.insert(prediction_path.begin(), *last_observed);
    }
    if (prediction_path.size() >= 2)
    {
      cv::polylines(display, prediction_path, false, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
      if (last_observed)
      {
        cv::arrowedLine(display, *last_observed, prediction_path.back(),
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.25);
      }
    }
    for (const auto &pt : predicted_trace)
    {
      cv::circle(display, pt, 4, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    const float vis_scale = 3.0f;
    const cv::Point2f base = last_observed ? cv::Point2f(static_cast<float>(last_observed->x),
                                                        static_cast<float>(last_observed->y))
                                           : cv::Point2f(static_cast<float>(prediction_path.front().x),
                                                         static_cast<float>(prediction_path.front().y));
    vector<cv::Point> exaggerated;
    exaggerated.reserve(predicted_trace.size());
    cv::Point2f avg_delta(0.0f, 0.0f);
    for (const auto &pt : predicted_trace)
    {
      cv::Point2f delta(static_cast<float>(pt.x) - base.x,
                        static_cast<float>(pt.y) - base.y);
      avg_delta += delta;
      cv::Point2f scaled = base + delta * vis_scale;
      int sx = std::clamp(static_cast<int>(std::round(scaled.x)), 0, display.cols - 1);
      int sy = std::clamp(static_cast<int>(std::round(scaled.y)), 0, display.rows - 1);
      exaggerated.emplace_back(sx, sy);
    }
    if (!exaggerated.empty())
    {
      cv::polylines(display, exaggerated, false, cv::Scalar(64, 255, 64), 1, cv::LINE_AA);
      for (const auto &pt : exaggerated)
      {
        cv::circle(display, pt, 2, cv::Scalar(64, 255, 64), cv::FILLED);
      }
    }

    if (!predicted_trace.empty())
    {
      avg_delta *= (1.0f / static_cast<float>(predicted_trace.size()));
      avg_prediction_delta = avg_delta;
    }
  }

  int rpm_block_y = 45;

  if (!std::isfinite(rpm_stats_copy.median) || rpm_stats_copy.median <= 0.0)
  {
    draw_text_line(display, rpm_block_y, "Global RPM: N/A");
  }
  else
  {
    draw_text_line(display, rpm_block_y,
                   "Global RPM: " + std::to_string(static_cast<int>(std::round(rpm_stats_copy.median))) +
                       " (" + std::to_string(rpm_stats_copy.sampled) + " samples)");
  }

  if (tracked_rotors_copy.empty())
  {
    draw_text_line(display, rpm_block_y, "Clusters: none");
  }
  else
  {
    draw_text_line(display, rpm_block_y, "Clusters:");
    for (const auto &rotor : tracked_rotors_copy)
    {
      std::string line = "#" + std::to_string(rotor.id) + ": ";
      if (!rotor.rpm_history.empty() && std::isfinite(rotor.rpm_history.back()))
      {
        line += std::to_string(static_cast<int>(std::round(rotor.rpm_history.back()))) + " RPM";
      }
      else
      {
        line += "RPM N/A";
      }
      draw_text_line(display, rpm_block_y, line, rotor.color, &rotor.color);

      if (rotor.rpm_history.size() > 1)
      {
        int graph_w = 150;
        int graph_h = 40;
        int text_margin = 5;

        cv::Point graph_origin(10, rpm_block_y);
        cv::Rect graph_rect(graph_origin, cv::Size(graph_w, graph_h));
        cv::rectangle(display, graph_rect, cv::Scalar(100, 100, 100), 1);

        double min_rpm = 0;
        double max_rpm = 0;
        for (double rpm : rotor.rpm_history)
        {
          if (rpm > max_rpm)
            max_rpm = rpm;
        }
        max_rpm *= 1.5;
        double rpm_range = max_rpm - min_rpm;
        if (rpm_range < 500)
          rpm_range = 500;

        std::vector<cv::Point> points;
        points.reserve(rotor.rpm_history.size());
        for (size_t i = 0; i < rotor.rpm_history.size(); ++i)
        {
          int x = graph_origin.x + static_cast<int>((double)i / (FrameState::RPM_HISTORY_SIZE - 1) * graph_w);
          int y = graph_origin.y + graph_h - static_cast<int>(((rotor.rpm_history[i] - min_rpm) / max_rpm) * graph_h);
          points.emplace_back(x, y);
        }
        cv::polylines(display, points, false, rotor.color, 1, cv::LINE_AA);

        cv::putText(display, "RPM", cv::Point(graph_origin.x, graph_origin.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(display, std::to_string(static_cast<int>(max_rpm)),
                    cv::Point(graph_rect.br().x + text_margin, graph_origin.y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(display, std::to_string(static_cast<int>(min_rpm)),
                    cv::Point(graph_rect.br().x + text_margin, graph_origin.y + graph_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        cv::putText(display, "Time",
                    cv::Point(graph_origin.x + graph_w / 2 - 10, graph_rect.br().y + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        rpm_block_y += graph_h + 30;
      }
    }
  }

  if (!predicted_trace.empty())
  {
    draw_text_line(display, rpm_block_y,
                   "Trajectory horizon: +" + std::to_string(predicted_trace.size()) + " steps");
    if (avg_prediction_delta)
    {
      draw_text_line(display, rpm_block_y,
                     "Δx=" + std::to_string(static_cast<int>(avg_prediction_delta->x)) +
                         " px   Δy=" + std::to_string(static_cast<int>(avg_prediction_delta->y)) + " px");
    }
  }
  else if (!observed_trace.empty())
  {
    draw_text_line(display, rpm_block_y, "Trajectory horizon: collecting context");
  }

  cv::imshow("Events", display);
  int key = cv::waitKey(1);
  if (key == 27 || key == 'q')
  {
    fs.running.store(false);
  }
}

void render_thread_loop()
{
  cv::Mat frame(DEFAULT_H, DEFAULT_W, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::namedWindow("Events", cv::WINDOW_AUTOSIZE);

  auto frame_interval = chrono::duration_cast<steady::duration>(chrono::duration<double>(1.0 / TARGET_FPS));
  auto next_frame = steady::now();
  while (fs.running.load())
  {
    next_frame += frame_interval;
    render_frame(frame);
    this_thread::sleep_until(next_frame);
  }
}
