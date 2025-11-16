#include "defs.hpp"
#include <chrono>

void render_frame(cv::Mat &frame)
{
  cv::Mat display;
  cv::Mat counts_snapshot;
  FrameState::RpmStats rpm_stats_copy;
  vector<FrameState::TrackedRotor> tracked_rotors_copy;

  // Prepare display buffer
  display.create(frame.rows, frame.cols, CV_8UC3);
  display.setTo(cv::Scalar(0, 0, 0));

  // Build mask where counts >= threshold and set those pixels to white.
  cv::Mat mask;
  cv::compare(fs.counts, EVENT_COUNT_THRESHOLD, mask, cv::CMP_GE); // mask: 255 where true
  display.setTo(cv::Scalar(255, 255, 255), mask);
  rpm_stats_copy = fs.rpm_stats; // snapshot under lock

  // Snapshot tracked rotors for rendering
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

  int rpm_block_y = 45;
  auto put_line = [&](const string &line,
                      const cv::Scalar &text_color = cv::Scalar(0, 255, 255),
                      const cv::Scalar *marker = nullptr)
  {
    if (marker)
    {
      cv::rectangle(display,
                    cv::Point(10, rpm_block_y - 12),
                    cv::Point(22, rpm_block_y - 2),
                    *marker, cv::FILLED);
      cv::putText(display, line, cv::Point(26, rpm_block_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  text_color, 1, cv::LINE_AA);
    }
    else
    {
      cv::putText(display, line, cv::Point(10, rpm_block_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55,
                  text_color, 1, cv::LINE_AA);
    }
    rpm_block_y += 18;
  };

  if (!isfinite(rpm_stats_copy.median) || rpm_stats_copy.median <= 0.0)
  {
    put_line("Global RPM: N/A");
  }
  else
  {
    put_line("Global RPM: " + to_string(static_cast<int>(round(rpm_stats_copy.median))) +
             " (" + to_string(rpm_stats_copy.sampled) + " samples)");
  }

  if (tracked_rotors_copy.empty())
  {
    put_line("Clusters: none");
  }
  else
  {
    put_line("Clusters:");
    for (const auto &rotor : tracked_rotors_copy)
    {
      string line = "#" + to_string(rotor.id) + ": ";
      if (!rotor.rpm_history.empty() && isfinite(rotor.rpm_history.back()))
      {
        line += to_string(static_cast<int>(round(rotor.rpm_history.back()))) + " RPM";
      }
      else
      {
        line += "RPM N/A";
      }
      put_line(line, rotor.color, &rotor.color);

      // Draw RPM history graph in sidebar
      if (rotor.rpm_history.size() > 1)
      {
        int graph_w = 150;
        int graph_h = 40;
        int text_margin = 5;

        cv::Point graph_origin(10, rpm_block_y);
        cv::Rect graph_rect(graph_origin, cv::Size(graph_w, graph_h));
        cv::rectangle(display, graph_rect, cv::Scalar(100, 100, 100), 1);

        // Find min/max RPM in history for scaling
        double min_rpm = 0;
        double max_rpm = 0;

        for (double rpm : rotor.rpm_history)
        {
          if (rpm > max_rpm)
            max_rpm = rpm;
        }

        max_rpm *= 1.5; // Add 20% padding

        double rpm_range = max_rpm - min_rpm;
        if (rpm_range < 500)
          rpm_range = 500;

        vector<cv::Point> points;
        points.reserve(rotor.rpm_history.size());
        for (size_t i = 0; i < rotor.rpm_history.size(); ++i)
        {
          int x = graph_origin.x + (int)((double)i / (fs.RPM_HISTORY_SIZE - 1) * graph_w);
          int y = graph_origin.y + graph_h - (int)(((rotor.rpm_history[i] - min_rpm) / max_rpm) * graph_h);
          points.emplace_back(x, y);
        }
        cv::polylines(display, points, false, rotor.color, 1, cv::LINE_AA);

        // Y-axis label and ticks
        cv::putText(display, "RPM", cv::Point(graph_origin.x, graph_origin.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(display, to_string((int)max_rpm), cv::Point(graph_rect.br().x + text_margin, graph_origin.y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(display, to_string((int)min_rpm), cv::Point(graph_rect.br().x + text_margin, graph_origin.y + graph_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        // X-axis label
        cv::putText(display, "Time", cv::Point(graph_origin.x + graph_w / 2 - 10, graph_rect.br().y + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        rpm_block_y += graph_h + 30;
      }
    }
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