#include "EgoMotionEstimation.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <limits>

namespace kiss_icp::core {

// Implement the functions here
std::pair<Eigen::Vector2d, double> fitVelocityProfile(
    const Eigen::Vector3d &direction1, double doppler1,
    const Eigen::Vector3d &direction2, double doppler2) {
    // Form the A matrix and b vector for two points
    Eigen::Matrix2d A;
    A << direction1.x(), direction1.y(),
         direction2.x(), direction2.y();
    Eigen::Vector2d b(doppler1, doppler2);

    // Solve for v = [vx, vy] using A * v = b
    Eigen::Vector2d velocity = A.colPivHouseholderQr().solve(b);

    // Calculate the error (residual) for this fit
    double error = (A * velocity - b).norm();

    return {velocity, error};
}

std::vector<size_t> detectStationaryTargets(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers,
    double ransac_threshold,
    int max_iterations) {
    size_t N = directions.size();
    if (N < 2) {
        throw std::invalid_argument("Not enough points for RANSAC.");
    }

    // Initialize variables
    std::vector<size_t> best_inliers;
    double best_error = std::numeric_limits<double>::max();
    std::srand(unsigned(std::time(0))); // Seed for random sampling

    // Perform RANSAC iterations
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Randomly select two targets
        size_t idx1 = std::rand() % N;
        size_t idx2 = std::rand() % N;
        if (idx1 == idx2) continue;

        // Fit a velocity profile using the two points
        auto [velocity, fit_error] = fitVelocityProfile(
            directions[idx1], dopplers[idx1],
            directions[idx2], dopplers[idx2]);

        // Check residuals for all points and count inliers
        std::vector<size_t> inliers;
        double total_error = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double residual = std::abs(directions[i].x() * velocity.x() +
                                       directions[i].y() * velocity.y() - dopplers[i]);
            if (residual < ransac_threshold) {
                inliers.push_back(i);
                total_error += residual;
            }
        }

        // Update the best fit if current inliers are better
        if (inliers.size() > best_inliers.size() || 
            (inliers.size() == best_inliers.size() && total_error < best_error)) {
            best_inliers = inliers;
            best_error = total_error;
        }
    }

    return best_inliers; // Indices of stationary targets
}

Eigen::Vector3d estimateEgoVelocity(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers) {
    
    size_t N = directions.size();
    if (N == 0) {
        throw std::invalid_argument("No stationary targets provided.");
    }

    // Construct the A matrix and b vector
    Eigen::MatrixXd A(N, 2); // A matrix (N x 2)
    Eigen::VectorXd b(N);    // b vector (N x 1)

    for (size_t i = 0; i < N; ++i) {
        A(i, 0) = directions[i].x(); // cos(theta)
        A(i, 1) = directions[i].y(); // sin(theta)
        b(i) = dopplers[i];          // Radial velocity (Doppler measurement)
    }

    // Solve for velocity vector v = [vx, vy] using Least Squares: A * v = b
    Eigen::Vector2d velocity = A.colPivHouseholderQr().solve(b);

    Eigen::Vector3d velocity_3d;
    velocity_3d.block<2, 1>(0, 0) = velocity;
    velocity_3d(2) = 0;
    return velocity_3d; // [vx, vy, 0]
}

Eigen::Vector3d estimateEgoMotion(
    const std::vector<Eigen::Vector3d> &directions,
    const std::vector<double> &dopplers,
    double ransac_threshold,
    int ransac_iterations) {
    // Step 1: Stationary target detection (RANSAC)
    std::vector<size_t> stationary_indices = detectStationaryTargets(
        directions, dopplers, ransac_threshold, ransac_iterations);

    // Filter inputs for stationary targets
    std::vector<Eigen::Vector3d> filtered_directions;
    std::vector<double> filtered_dopplers;
    for (size_t idx : stationary_indices) {
        filtered_directions.push_back(directions[idx]);
        filtered_dopplers.push_back(dopplers[idx]);
    }

    // Step 2: Velocity profile analysis (LSQ)
    Eigen::Vector3d sensor_velocity = estimateEgoVelocity(filtered_directions, filtered_dopplers);

    return sensor_velocity;
}

}  // namespace kiss_icp::core