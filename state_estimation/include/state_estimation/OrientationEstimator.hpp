#pragma once

#include <cstddef>
#include <string>

#include <state_estimation/ImuTypes.hpp>
#include <state_estimation/Quaternion.hpp>

namespace state_estimation {

struct OrientationEstimatorConfig {
  double gravity_mps2{9.80665};
  double initialization_duration_s{3.0};
  std::size_t minimum_initialization_samples{20};

  double initialization_accel_norm_tolerance_mps2{1.5};
  double initialization_mean_gyro_limit_radps{0.35};

  double accel_correction_gain{2.0};
  double accel_correction_norm_tolerance_mps2{2.0};

  double max_integration_dt_s{0.05};
};

struct OrientationEstimate {
  bool initialized{false};
  Quaternion q_world_imu{Quaternion::identity()};
  Vec3 projected_gravity_imu{0.0, 0.0, -1.0};
  Vec3 angular_velocity_imu_radps{};
  Vec3 gyro_bias_radps{};
  std::size_t initialization_sample_count{0};
  std::string status{"waiting for initialization samples"};
};

class OrientationEstimator {
 public:
  explicit OrientationEstimator(OrientationEstimatorConfig config = {});

  void reset();

  OrientationEstimate step(const ImuSample& sample);
  OrientationEstimate addImuSample(const ImuSample& sample);

  bool initialized() const {
    return initialized_;
  }

  Quaternion orientationWorldImu() const {
    return q_world_imu_;
  }

  Vec3 projectedGravityImu() const;

  Vec3 angularVelocityImuRadps() const {
    return angular_velocity_imu_radps_;
  }

  Vec3 gyroBiasRadps() const {
    return gyro_bias_radps_;
  }

  OrientationEstimate estimate() const;

  static constexpr Vec3 worldGravityDirection() {
    return {0.0, 0.0, -1.0};
  }

 private:
  bool initializeFromAccumulatedSamples();
  void resetInitializationWindow();
  void updateRunningEstimate(const ImuSample& sample);
  bool accelNormIsTrusted(double accel_norm) const;

  OrientationEstimatorConfig config_{};

  bool initialized_{false};
  Quaternion q_world_imu_{Quaternion::identity()};
  Vec3 angular_velocity_imu_radps_{};
  Vec3 gyro_bias_radps_{};
  std::string status_{"waiting for initialization samples"};

  bool have_last_timestamp_{false};
  double last_timestamp_s_{0.0};

  bool have_initialization_start_{false};
  double initialization_start_timestamp_s_{0.0};
  double initialization_last_timestamp_s_{0.0};
  std::size_t initialization_sample_count_{0};
  Vec3 initialization_accel_sum_{};
  Vec3 initialization_gyro_sum_{};
};

}  // namespace state_estimation
