#include <iomanip>
#include <iostream>

#include <state_estimation/OrientationEstimator.hpp>
#include <state_estimation/SimulatedImu.hpp>

namespace {

void printVector(const char* name, const state_estimation::Vec3& vector) {
  std::cout << name << " = [" << vector.x << ", " << vector.y << ", " << vector.z << "]\n";
}

void printQuaternion(const char* name, const state_estimation::Quaternion& quaternion) {
  std::cout << name << " = [" << quaternion.w << ", " << quaternion.x << ", " << quaternion.y
            << ", " << quaternion.z << "]\n";
}

}  // namespace

int main() {
  using namespace state_estimation;

  constexpr double deg_to_rad = Quaternion::pi() / 180.0;

  SimulatedImu imu(0.005);
  imu.setOrientationWorldImu(
      Quaternion::fromRollPitchYaw(35.0 * deg_to_rad, -20.0 * deg_to_rad, 120.0 * deg_to_rad));
  imu.setGyroBiasRadps({0.015, -0.01, 0.005});
  imu.setAccelNoiseStddevMps2(0.03);
  imu.setGyroNoiseStddevRadps(0.001);

  OrientationEstimator estimator;

  OrientationEstimate estimate;
  while (!estimate.initialized) {
    estimate = estimator.step(imu.readSample());
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Initialized after " << imu.time() << " seconds\n";
  printQuaternion("q_WI", estimate.q_world_imu);
  printVector("projected_gravity_I", estimate.projected_gravity_imu);
  printVector("angular_velocity_I_radps", estimate.angular_velocity_imu_radps);
  printVector("gyro_bias_radps", estimate.gyro_bias_radps);

  imu.setAngularVelocityImuRadps({0.0, 0.0, 30.0 * deg_to_rad});
  for (int i = 0; i < 200; ++i) {
    estimate = estimator.step(imu.readSample());
  }

  std::cout << "\nAfter one second of relative yaw rotation:\n";
  printQuaternion("q_WI", estimate.q_world_imu);
  printVector("projected_gravity_I", estimate.projected_gravity_imu);
  printVector("angular_velocity_I_radps", estimate.angular_velocity_imu_radps);

  return 0;
}
