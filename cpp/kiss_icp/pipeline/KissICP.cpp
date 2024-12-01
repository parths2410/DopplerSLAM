// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "KissICP.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/core/EgoMotionEstimation.hpp"

#include <iostream>

namespace kiss_icp::pipeline {

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<Eigen::Vector3d> &directions,
                                                    const std::vector<double> &dopplers,
                                                    const Sophus::SE3d &T_V_S,
                                                    const std::vector<double> &timestamps) {
    const auto &deskew_frame = [&]() -> std::vector<Eigen::Vector3d> {
        if (!config_.deskew) return frame;
        // TODO(Nacho) Add some asserts here to sanitize the timestamps

        //  If not enough poses for the estimation, do not de-skew
        const size_t N = poses().size();
        if (N <= 2) return frame;

        // Estimate linear and angular velocities
        const auto &start_pose = poses_[N - 2];
        const auto &finish_pose = poses_[N - 1];
        return DeSkewScan(frame, timestamps, start_pose, finish_pose);
    }();
    return RegisterFrame(deskew_frame, directions, dopplers, T_V_S);
}


// TODO: Add doppler velocities to the function signature
KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<Eigen::Vector3d> &directions,
                                                    const std::vector<double> &dopplers,
                                                    const Sophus::SE3d &T_V_S) {
    // Preprocess the input cloud
    const auto &[cropped_frame, indexes] = Preprocess(frame, config_.max_range, config_.min_range);

    std::vector<double> cropped_dopplers;
    std::vector<Eigen::Vector3d> cropped_directions;
    cropped_dopplers.reserve(dopplers.size());
    cropped_directions.reserve(directions.size());
    for (const auto &idx : indexes) {
        cropped_dopplers.emplace_back(dopplers[idx]);
        cropped_directions.emplace_back(directions[idx]);
    }

    // Voxelize
    const auto &[source_pair, frame_downsample_pair] = Voxelize(cropped_frame);

    const auto &[source, indices_source] = source_pair;
    const auto &[frame_downsample, indices_frame_downsample] = frame_downsample_pair;

    std::vector<double> dopplers_ds_1;
    std::vector<Eigen::Vector3d> directions_ds_1;
    dopplers_ds_1.reserve(indices_source.size());
    directions_ds_1.reserve(indices_source.size());
    for (const auto &idx : indices_source) {
        dopplers_ds_1.emplace_back(cropped_dopplers[idx]);
        directions_ds_1.emplace_back(cropped_directions[idx]);
    }

    Eigen::Vector3d v_pred = kiss_icp::core::estimateEgoMotion(directions_ds_1, dopplers_ds_1, 0.1, 100);
    // std::cout << "[KissICP::RegisterFrame] ego_velocity = " << ego_velocity.transpose() << std::endl;

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    // Compute initial_guess for ICP
    auto T_pred = GetPredictionModel();
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    const auto pose_pred = last_pose * T_pred;

    // Maintain delta_ts

    double period = 0.05; // TODO: Make this a parameter
    // Run icp
    // TODO: Add doppler velocities to the RegisterFrame function
    const Sophus::SE3d new_pose = kiss_icp::RegisterFrame(source,         //
                                                          local_map_,     //
                                                          dopplers_ds_1,  //
                                                          directions_ds_1,
                                                          v_pred,
                                                          T_pred,         //
                                                          pose_pred,      //
                                                          T_V_S,
                                                          period,          //
                                                          3.0 * sigma,    //
                                                          sigma / 3.0);
    const auto model_deviation = pose_pred.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
    poses_.push_back(new_pose);
    return {frame, source};
}

KissICP::Vector3dVectorTupleWithIndices KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto [frame_downsample, indices_frame_downsample] = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const auto [source, indices_source] = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    
    return {{source, indices_source}, {frame_downsample, indices_frame_downsample}};
}

double KissICP::GetAdaptiveThreshold() {
    if (!HasMoved()) {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d KissICP::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool KissICP::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

}  // namespace kiss_icp::pipeline
