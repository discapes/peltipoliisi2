// Event visualizer (30 FPS) – color white when a pixel sees >= threshold events.
#include "defs.hpp"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include <mlpack/methods/dbscan/dbscan.hpp>
#pragma GCC diagnostic pop

// Refactored cluster_worker: compute overlays from given coords/events and parameters.
// Returns overlays; does not access or modify global FrameState.
vector<FrameState::ClusterOverlay> cluster_worker(
  vector<double> coords,
  vector<RpmSample> rpm_samples,
  double cluster_eps,
  size_t cluster_min_points) {
  vector<FrameState::ClusterOverlay> overlays;
  const size_t point_count = coords.size() / 2;
  if (point_count < cluster_min_points) return overlays;

  try {
    arma::mat data(coords.data(), 2, point_count, false, true);
    mlpack::DBSCAN<> clusterer(cluster_eps, cluster_min_points);
    arma::Row<size_t> assignments;
    clusterer.Cluster(data, assignments);

    struct Bounds {
      int min_x = numeric_limits<int>::max();
      int min_y = numeric_limits<int>::max();
      int max_x = numeric_limits<int>::min();
      int max_y = numeric_limits<int>::min();
      bool initialized = false;
    };
    unordered_map<size_t, Bounds> aggregates;
    aggregates.reserve(point_count);

    for (size_t idx = 0; idx < point_count; ++idx) {
      const size_t label = assignments[idx];
      if (label == numeric_limits<size_t>::max()) continue;
      Bounds &b = aggregates[label];
      int px = static_cast<int>(coords[2 * idx]);
      int py = static_cast<int>(coords[2 * idx + 1]);
      if (!b.initialized) {
        b.min_x = b.max_x = px;
        b.min_y = b.max_y = py;
        b.initialized = true;
      } else {
        b.min_x = min(b.min_x, px);
        b.min_y = min(b.min_y, py);
        b.max_x = max(b.max_x, px);
        b.max_y = max(b.max_y, py);
      }
    }

    vector<pair<size_t, Bounds>> clusters;
    clusters.reserve(aggregates.size());
    for (auto &entry : aggregates) {
      if (!entry.second.initialized) continue;
      clusters.emplace_back(entry.first, entry.second);
    }
    sort(clusters.begin(), clusters.end(),
         [](const auto &a, const auto &b) { return a.first < b.first; });

    static const array<cv::Scalar, 8> palette = {
        cv::Scalar(0, 0, 255),  cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
        cv::Scalar(128, 255, 0), cv::Scalar(0, 128, 255)};

    overlays.reserve(clusters.size());
    for (size_t idx = 0; idx < clusters.size(); ++idx) {
      const auto &b = clusters[idx].second;
      FrameState::ClusterOverlay overlay;
      overlay.box = cv::Rect(cv::Point(b.min_x, b.min_y),
                             cv::Point(b.max_x + 1, b.max_y + 1));
      overlay.color = palette[idx % palette.size()];

      // Gather precomputed RPMs whose sample location lies inside the cluster box
      vector<double> rpms;
      rpms.reserve(32);
      for (const auto &s : rpm_samples) {
        if (overlay.box.contains(cv::Point(s.x, s.y)) && isfinite(s.rpm) && s.rpm > 0.0) {
          rpms.push_back(s.rpm);
        }
      }
      if (!rpms.empty()) {
        const size_t mid = rpms.size() / 2;
        nth_element(rpms.begin(), rpms.begin() + mid, rpms.end());
        if (rpms.size() % 2 == 1) {
          overlay.rpm = rpms[mid];
        } else {
          double a = *max_element(rpms.begin(), rpms.begin() + mid);
          double b = rpms[mid];
          overlay.rpm = 0.5 * (a + b);
        }
      }
      overlays.push_back(move(overlay));
    }
  } catch (const exception &ex) {
    cerr << "[cluster-worker] DBSCAN failed: " << ex.what() << endl;
  }

  return overlays;
}