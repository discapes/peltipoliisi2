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
#include "defs.hpp"
#include "rpm_estimator.hpp"

void cluster_worker() {
  std::vector<double> coords;
  std::vector<FrameState::FrameEvent> events;
  while (true) {
    u64 frame_id = 0;
    {
      std::unique_lock<std::mutex> lk(fs.cluster_request_mtx);
      fs.cluster_request_cv.wait(lk, [] {
        return fs.cluster_request_ready || !fs.running.load();
      });
      if (!fs.cluster_request_ready && !fs.running.load()) break;
      coords.swap(fs.cluster_request_coords);
      events.swap(fs.cluster_request_events);
      frame_id = fs.cluster_request_frame;
      fs.cluster_request_ready = false;
    }

    std::vector<FrameState::ClusterOverlay> overlays;
    const size_t point_count = coords.size() / 2;
    if (point_count >= fs.cluster_min_points) {
      try {
        arma::mat data(coords.data(), 2, point_count, false, true);
        mlpack::DBSCAN<> clusterer(fs.cluster_eps, fs.cluster_min_points);
        arma::Row<size_t> assignments;
        clusterer.Cluster(data, assignments);

        struct Bounds {
          int min_x = std::numeric_limits<int>::max();
          int min_y = std::numeric_limits<int>::max();
          int max_x = std::numeric_limits<int>::min();
          int max_y = std::numeric_limits<int>::min();
          bool initialized = false;
        };
        std::unordered_map<size_t, Bounds> aggregates;
        aggregates.reserve(point_count);

        for (size_t idx = 0; idx < point_count; ++idx) {
          const size_t label = assignments[idx];
          if (label == std::numeric_limits<size_t>::max()) continue;
          Bounds &b = aggregates[label];
          int px = static_cast<int>(coords[2 * idx]);
          int py = static_cast<int>(coords[2 * idx + 1]);
          if (!b.initialized) {
            b.min_x = b.max_x = px;
            b.min_y = b.max_y = py;
            b.initialized = true;
          } else {
            b.min_x = std::min(b.min_x, px);
            b.min_y = std::min(b.min_y, py);
            b.max_x = std::max(b.max_x, px);
            b.max_y = std::max(b.max_y, py);
          }
        }

        std::vector<std::pair<size_t, Bounds>> clusters;
        clusters.reserve(aggregates.size());
        for (auto &entry : aggregates) {
          if (!entry.second.initialized) continue;
          clusters.emplace_back(entry.first, entry.second);
        }
        std::sort(clusters.begin(), clusters.end(),
                  [](const auto &a, const auto &b) {
                    return a.first < b.first;
                  });

        static const std::array<cv::Scalar, 8> palette = {
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 255),
            cv::Scalar(255, 255, 0),
            cv::Scalar(128, 255, 0),
            cv::Scalar(0, 128, 255)
        };

        overlays.reserve(clusters.size());
        for (size_t idx = 0; idx < clusters.size(); ++idx) {
          const auto &b = clusters[idx].second;
          FrameState::ClusterOverlay overlay;
          overlay.box = cv::Rect(cv::Point(b.min_x, b.min_y),
                                 cv::Point(b.max_x + 1, b.max_y + 1));
          overlay.color = palette[idx % palette.size()];

          std::vector<FrameState::FrameEvent> local;
          local.reserve(128);
          for (const auto &ev : events) {
            if (overlay.box.contains(cv::Point(ev.x, ev.y))) {
              local.push_back(ev);
            }
          }
          if (local.size() >= 16) {
            double rpm = estimate_rpm_from_events(local);
            if (std::isfinite(rpm) && rpm > 0.0) overlay.rpm = rpm;
          }
          overlays.push_back(std::move(overlay));
        }
      } catch (const std::exception &ex) {
        std::cerr << "[cluster-worker] DBSCAN failed: " << ex.what() << std::endl;
      }
    }

    {
      std::lock_guard<std::mutex> lk(fs.overlay_mtx);
      fs.overlay_data = std::move(overlays);
      fs.overlay_frame = frame_id;
    }

    coords.clear();
    events.clear();
  }
}