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
    const std::vector<double> &dopplers_in_S,
    const std::vector<Eigen::Vector3d> &directions_in_V,
    const Sophus::SE3d &T_V_S,
    Eigen::Vector3d v_s_in_S,
    double kernel_geometric,
    double kernel_doppler,
    double lambda_doppler, 
    double period) {
    const Eigen::Matrix3d R_V_S = T_V_S.rotationMatrix();
    const Eigen::Vector3d r_v_to_s_in_V = T_V_S.translation();
    const Eigen::Matrix3d R_S_V = R_V_S.transpose();

    const double lambda_geometric = 1.0 - lambda_doppler;
    const double sqrt_lambda_doppler = std::sqrt(lambda_doppler);
    const double sqrt_lambda_geometric = std::sqrt(lambda_geometric);

    const double sqrt_lambda_doppler_by_dt = sqrt_lambda_doppler / period; 

    double doppler_outlier_threshold = 5; // TODO: replace with variable

    int num_outliers = 0;
    double mean_residual_doppler = 0.0;
    auto compute_jacobian_and_residual = [&](auto i) {
        Eigen::Vector3d residual_geometric = source[i] - target[i];
        residual_geometric = sqrt_lambda_geometric * residual_geometric;
        Eigen::Matrix3_6d J_r_geometric;
        J_r_geometric.block<3, 3>(0, 0) = sqrt_lambda_geometric * Eigen::Matrix3d::Identity();
        J_r_geometric.block<3, 3>(0, 3) = sqrt_lambda_geometric * -1.0 * Sophus::SO3d::hat(source[i]);
        
        Eigen::Vector3d ds_in_V = directions_in_V[i];
        Eigen::Vector3d ds_in_S = R_S_V * ds_in_V;
        const double doppler_pred_in_S = -ds_in_S.dot(v_s_in_S);
        double residual_doppler = dopplers_in_S[i] - doppler_pred_in_S;
        residual_doppler = sqrt_lambda_doppler_by_dt * residual_doppler;
        Eigen::Matrix<double, 1, 6> J_r_doppler;
        J_r_doppler.block<1, 3>(0, 0) = sqrt_lambda_doppler_by_dt * ds_in_V.cross(r_v_to_s_in_V); 
        J_r_doppler.block<1, 3>(0, 3) = sqrt_lambda_doppler_by_dt * -ds_in_V.transpose();

        if (std::abs(residual_doppler) > doppler_outlier_threshold) {
            // std::cout << "[kiss_icp::BuildLinearSystem] doppler outlier detected :: " << std::abs(residual_doppler) << std::endl;
            // residual_geometric = Eigen::Vector3d::Zero();
            residual_doppler = 0.0;
            // J_r_geometric.setZero();
            J_r_doppler.setZero();
            num_outliers++;
        }
        mean_residual_doppler += std::abs(residual_doppler);
        return std::make_tuple(J_r_geometric, residual_geometric, J_r_doppler, residual_doppler);
    };
    ResultTuple result;
    auto Weight = [&](double kernel, double residual2) {
        return square(kernel) / square(kernel + residual2);
    };
    (void)Weight;
    auto TukeyWeight = [&](double kernel, double residual2) {
        const double u = square(kernel);
        if (residual2 > u) return 0.0;
        const double r2 = residual2 / u;
        return square(1.0 - r2) * square(1.0 - r2 * r2);
    };
    (void)TukeyWeight;

    for (size_t i = 0; i < source.size(); ++i) {
        const auto &[J_r_geometric, residual_geometric, J_r_doppler, residual_doppler] = compute_jacobian_and_residual(i);

        const double w_geometric = Weight(kernel_geometric, residual_geometric.squaredNorm());
        const double w_doppler = TukeyWeight(kernel_doppler, std::abs(residual_doppler));
        result.JTJ.noalias() += J_r_geometric.transpose() * w_geometric * J_r_geometric;
        result.JTr.noalias() += J_r_geometric.transpose() * w_geometric * residual_geometric;
        result.JTJ.noalias() += J_r_doppler.transpose() * w_doppler * J_r_doppler;
        result.JTr.noalias() += J_r_doppler.transpose() * w_doppler * residual_doppler;

        // std::cout << "[kiss_icp::BuildLinearSystem] w_geometric = " << w_geometric << " w_doppler = " << w_doppler << std::endl;
    }

    // std::cout << "[kiss_icp::BuildLinearSystem] num_outliers = " << num_outliers << std::endl;
    // std::cout << "[kiss_icp::BuildLinearSystem] mean_residual_doppler = " << mean_residual_doppler / double(source.size()) << std::endl;
    return std::make_tuple(result.JTJ, result.JTr);
}

Eigen::Vector6d TransformMatrix4dToVector6d(const Sophus::SE3d &input) {
    Eigen::Vector6d output;
    Eigen::Matrix3d R = input.rotationMatrix();
    double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    if (!(sy < 1e-6)) {
        output(3) = atan2(R(2, 1), R(2, 2));
        output(4) = atan2(-R(2, 0), sy);
        output(5) = atan2(R(1, 0), R(0, 0));
    } else {
        output(3) = atan2(-R(1, 2), R(1, 1));
        output(4) = atan2(-R(2, 0), sy);
        output(5) = 0;
    }
    output.block<3, 1>(0, 0) = input.translation();
    return output;
}

}  // namespace

namespace kiss_icp {

Sophus::SE3d RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                           const VoxelHashMap &voxel_map,
                           const std::vector<double> &dopplers,
                           const std::vector<Eigen::Vector3d> &directions,
                           Eigen::Vector3d &v_pred,
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

    // std::cout << "[kiss_icp::RegisterFrame] kernel = " << kernel << std::endl;
    // std::cout << "[kiss_icp::RegisterFrame] pose_pred = \n" << pose_pred.matrix() << std::endl;
    // std::cout << " ------------ " << std::endl;
    // std::cout << "[kiss_icp::RegisterFrame] T_pred = \n" << T_pred.matrix() << std::endl;
    // std::cout << std::endl;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(pose_pred, source);

    double lambda_doppler = 0.1;
    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d(); // TODO: Why is T_ICP initialized to the identity?
    // TODO: Add doppler velocities to the ICP loop
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // TODO : State_Vector from T_pred
        Eigen::Vector6d state_vector = TransformMatrix4dToVector6d(T_pred);

        // TODO : Find Velcoity (linear and angular) from State_Vector, state_vector / delta_t
        Eigen::Vector3d v_v_in_V = state_vector.block<3, 1>(0, 0) / period;
        Eigen::Vector3d w_v_in_V = state_vector.block<3, 1>(3, 0) / period;

        Eigen::Vector3d v_s_in_V = v_v_in_V + w_v_in_V.cross(r_v_to_s_in_V);
        Eigen::Vector3d v_s_in_S = R_S_V * v_s_in_V;

        // std::cout << "[kiss_icp::RegisterFrame] v_s_in_S = " << v_s_in_S.transpose() << std::endl;
        // std::cout << "[kiss_icp::RegisterFrame] v_pred = " << v_pred.transpose() << std::endl;
        // std::cout << " ------------ " << std::endl;
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
        
        auto dplrs_pred_in_S = std::vector<double>(src_indices.size());
        for (std::size_t i = 0; i < src_indices.size(); ++i) {
            auto src_dir_in_S = R_S_V * src_dirs_in_V[i];
            dplrs_pred_in_S[i] = -src_dir_in_S.dot(v_s_in_S);
        }

        (void)kernel;
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, dplrs_in_S, src_dirs_in_V, T_V_S, v_s_in_S, kernel, 0.2, lambda_doppler, period);
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
