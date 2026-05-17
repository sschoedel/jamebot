#pragma once

#include <cstdint>
#include <random>

#include <state_estimation/ImuTypes.hpp>
#include <state_estimation/Quaternion.hpp>

namespace state_estimation {

class SimulatedImu final : public ImuSource {
 public:
  explicit SimulatedImu(double sample_period_s = 0.005);

  ImuSample readSample() override;
  ImuSample sample(double dt_s);

  void setTime(double timestamp_s);
  void setSamplePeriod(double sample_period_s);
  void setOrientationWorldImu(const Quaternion& q_world_imu);
  void setAngularVelocityImuRadps(const Vec3& angular_velocity_imu_radps);
  void setLinearAccelerationWorldMps2(const Vec3& linear_acceleration_world_mps2);
  void setGyroBiasRadps(const Vec3& gyro_bias_radps);
  void setAccelBiasMps2(const Vec3& accel_bias_mps2);
  void setGyroNoiseStddevRadps(double stddev);
  void setAccelNoiseStddevMps2(double stddev);
  void setMagnetometerEnabled(bool enabled);
  void setMagneticFieldWorldUt(const Vec3& magnetic_field_world_uT);
  void seed(std::uint32_t seed_value);

  double time() const {
    return timestamp_s_;
  }

  Quaternion orientationWorldImu() const {
    return q_world_imu_;
  }

  Vec3 angularVelocityImuRadps() const {
    return angular_velocity_imu_radps_;
  }

 private:
  Vec3 sampleIsotropicNoise(double stddev);

  double sample_period_s_{0.005};
  double timestamp_s_{0.0};
  double gravity_mps2_{9.80665};

  Quaternion q_world_imu_{Quaternion::identity()};
  Vec3 angular_velocity_imu_radps_{};
  Vec3 linear_acceleration_world_mps2_{};
  Vec3 gyro_bias_radps_{};
  Vec3 accel_bias_mps2_{};

  double gyro_noise_stddev_radps_{0.0};
  double accel_noise_stddev_mps2_{0.0};

  bool magnetometer_enabled_{false};
  Vec3 magnetic_field_world_uT_{20.0, 0.0, -45.0};

  std::mt19937 rng_{7};
};

}  // namespace state_estimation
