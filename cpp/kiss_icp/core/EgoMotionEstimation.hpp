#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>

namespace kiss_icp::core {

std::pair<Eigen::Vector2d, double> fitVelocityProfile(
    const Eigen::Vector3d &direction1, double doppler1,
    const Eigen::Vector3d &direction2, double doppler2);

std::vector<size_t> detectStationaryTargets(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers,
    double ransac_threshold,
    int max_iterations = 100);

Eigen::Vector3d estimateEgoVelocity(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers);

Eigen::Vector3d estimateEgoMotion(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers,
    double ransac_threshold,
    int ransac_iterations = 100);

}  // namespace kiss_icp::core