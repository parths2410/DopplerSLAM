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

Eigen::Vector2d estimateSensorVelocity(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers);

std::pair<double, double> calculateEgoMotion(
    const Eigen::Vector2d &sensor_velocity,
    double sensor_mount_angle,
    double sensor_offset_x,
    double sensor_offset_y);

std::tuple<std::pair<double, double>, std::vector<size_t>> estimateEgoMotion(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers,
    double ransac_threshold,
    int ransac_iterations = 100);

}  // namespace kiss_icp::core