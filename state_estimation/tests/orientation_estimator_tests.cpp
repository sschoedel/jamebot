#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <state_estimation/OrientationEstimator.hpp>
#include <state_estimation/SimulatedImu.hpp>

namespace {

constexpr double kDegToRad = state_estimation::Quaternion::pi() / 180.0;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool nearVec(
    const state_estimation::Vec3& actual,
    const state_estimation::Vec3& expected,
    double tolerance) {
  return near(actual.x, expected.x, tolerance) && near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

state_estimation::OrientationEstimate initialize(
    state_estimation::OrientationEstimator& estimator,
    state_estimation::SimulatedImu& imu) {
  state_estimation::OrientationEstimate estimate;
  for (int i = 0; i < 800 && !estimate.initialized; ++i) {
    estimate = estimator.step(imu.readSample());
  }
  return estimate;
}

void testInitializesUprightGravity() {
  state_estimation::SimulatedImu imu(0.005);
  state_estimation::OrientationEstimator estimator;

  const auto estimate = initialize(estimator, imu);

  require(estimate.initialized, "upright IMU should initialize");
  require(
      nearVec(estimate.projected_gravity_imu, {0.0, 0.0, -1.0}, 1e-9),
      "upright projected gravity should point down in IMU frame");
}

void testInitializesArbitraryOrientationGravity() {
  state_estimation::SimulatedImu imu(0.005);
  const auto true_q_world_imu =
      state_estimation::Quaternion::fromRollPitchYaw(40.0 * kDegToRad, -25.0 * kDegToRad,
                                                     130.0 * kDegToRad);
  imu.setOrientationWorldImu(true_q_world_imu);

  state_estimation::OrientationEstimator estimator;
  const auto estimate = initialize(estimator, imu);

  const auto expected_gravity_imu =
      true_q_world_imu.inverseRotate(state_estimation::OrientationEstimator::worldGravityDirection())
          .normalized();

  require(estimate.initialized, "arbitrary stationary IMU should initialize");
  require(
      nearVec(estimate.projected_gravity_imu, expected_gravity_imu, 1e-9),
      "projected gravity should match true gravity direction even though yaw is arbitrary");
}

void testEstimatesStartupGyroBias() {
  state_estimation::SimulatedImu imu(0.005);
  const state_estimation::Vec3 bias{0.02, -0.015, 0.004};
  imu.setGyroBiasRadps(bias);

  state_estimation::OrientationEstimator estimator;
  const auto estimate = initialize(estimator, imu);

  require(estimate.initialized, "biased stationary IMU should initialize");
  require(nearVec(estimate.gyro_bias_radps, bias, 1e-12), "startup gyro bias should be mean gyro");
}

void testTracksTiltAfterInitialization() {
  state_estimation::SimulatedImu imu(0.005);
  state_estimation::OrientationEstimatorConfig config;
  config.accel_correction_gain = 3.0;
  state_estimation::OrientationEstimator estimator(config);

  auto estimate = initialize(estimator, imu);
  require(estimate.initialized, "IMU should initialize before motion test");

  imu.setAngularVelocityImuRadps({20.0 * kDegToRad, 0.0, 0.0});
  for (int i = 0; i < 200; ++i) {
    estimate = estimator.step(imu.readSample());
  }

  const auto expected_gravity_imu =
      imu.orientationWorldImu()
          .inverseRotate(state_estimation::OrientationEstimator::worldGravityDirection())
          .normalized();

  require(
      nearVec(estimate.projected_gravity_imu, expected_gravity_imu, 2e-3),
      "projected gravity should track simulated roll motion");
}

void testOutputsBiasCorrectedAngularVelocityInImuFrame() {
  state_estimation::SimulatedImu imu(0.005);
  const state_estimation::Vec3 bias{0.01, -0.02, 0.005};
  const state_estimation::Vec3 angular_velocity{0.0, 0.0, 45.0 * kDegToRad};
  imu.setGyroBiasRadps(bias);

  state_estimation::OrientationEstimator estimator;
  auto estimate = initialize(estimator, imu);
  require(estimate.initialized, "IMU should initialize before angular velocity output test");

  imu.setAngularVelocityImuRadps(angular_velocity);
  estimate = estimator.step(imu.readSample());

  require(
      nearVec(estimate.angular_velocity_imu_radps, angular_velocity, 1e-12),
      "step output should report bias-corrected angular velocity in IMU frame");
}

}  // namespace

int main() {
  testInitializesUprightGravity();
  testInitializesArbitraryOrientationGravity();
  testEstimatesStartupGyroBias();
  testTracksTiltAfterInitialization();
  testOutputsBiasCorrectedAngularVelocityInImuFrame();

  std::cout << "All state_estimation tests passed\n";
  return EXIT_SUCCESS;
}
