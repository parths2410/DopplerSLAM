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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include <iostream>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;

std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
    const std::vector<Eigen::Vector3d> &source,
    const std::vector<Eigen::Vector3d> &target,
    double kernel) {
    auto compute_jacobian_and_residual = [&](auto i) {
        const Eigen::Vector3d residual = source[i] - target[i];
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto Weight = [&](double residual2) {
                return square(kernel) / square(kernel + residual2);
            };
            auto &[JTJ_private, JTr_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                const double w = Weight(residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });

    return std::make_tuple(JTJ, JTr);
}
}  // namespace

namespace kiss_icp {

Sophus::SE3d RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                           const VoxelHashMap &voxel_map,
                           const std::vector<double> &dopplers,
                           const std::vector<Eigen::Vector3d> &directions,
                           Sophus::SE3d &T_pred,
                           const Sophus::SE3d &pose_pred,
                           const Sophus::SE3d &T_V_S,
                           double period,
                           double max_correspondence_distance,
                           double kernel) {
    (void)T_V_S;
    Eigen::Matrix3d R_V_S = T_V_S.rotationMatrix();
    Eigen::Vector3d r_v_to_s_in_V = T_V_S.translation();
    Eigen::Matrix3d R_S_V = R_V_S.transpose();

    if (voxel_map.Empty()) return pose_pred;

    // std::cout << "[kiss_icp::RegisterFrame] pose_pred = \n" << pose_pred.matrix() << std::endl;
    // std::cout << " ------------ " << std::endl;
    // std::cout << "[kiss_icp::RegisterFrame] T_pred = \n" << T_pred.matrix() << std::endl;
    // std::cout << std::endl;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(pose_pred, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d(); // TODO: Why is T_ICP initialized to the identity?
    // TODO: Add doppler velocities to the ICP loop
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // TODO : State_Vector from T_pred
        Eigen::Vector6d state_vector = T_pred.log();

        // TODO : Find Velcoity (linear and angular) from State_Vector, state_vector / delta_t
        Eigen::Vector3d v_v_in_V = -state_vector.block<3, 1>(0, 0) / period;
        Eigen::Vector3d w_v_in_V = -state_vector.block<3, 1>(3, 0) / period;

        Eigen::Vector3d v_s_in_V = v_v_in_V + w_v_in_V.cross(r_v_to_s_in_V);
        Eigen::Vector3d v_s_in_S = R_S_V * v_s_in_V;
        (void)v_s_in_S;

        // Equation (10)
        const auto &[src_pair, tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance);
        const auto &[src, src_indices] = src_pair;

        auto dplrs_in_S = std::vector<double>(src_indices.size());
        for (std::size_t i = 0; i < src_indices.size(); ++i) {
            dplrs_in_S[i] = dopplers[src_indices[i]];
        }
        auto src_dirs_in_V = std::vector<Eigen::Vector3d>(src_indices.size());
        for (std::size_t i = 0; i < src_indices.size(); ++i) {
            src_dirs_in_V[i] = directions[src_indices[i]];
        }

        // std::cout << "[kiss_icp::RegisterFrame] src_indices.size() = " << src_indices.size() << std::endl;
        // std::cout << "[kiss_icp::RegisterFrame] src.size() = " << src.size() << std::endl;
        // std::cout << "[kiss_icp::RegisterFrame] tgt.size() = " << tgt.size() << std::endl;
        // std::cout << "[kiss_icp::RegisterFrame] dplrs.size() = " << dplrs.size() << std::endl;
        // std::cout << "[kiss_icp::RegisterFrame] src_dirs.size() = " << src_dirs.size() << std::endl;
        // std::cout << std::endl;

        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        T_pred = estimation * T_pred;
        // Termination criteria
        if (dx.norm() < ESTIMATION_THRESHOLD_) break;
    }
    // Spit the final transformation
    return T_icp * pose_pred;
}

}  // namespace kiss_icp
