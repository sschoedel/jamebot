#include <state_estimation/SimulatedImu.hpp>

namespace state_estimation {

SimulatedImu::SimulatedImu(double sample_period_s)
    : sample_period_s_(sample_period_s) {}

ImuSample SimulatedImu::readSample() {
  return sample(sample_period_s_);
}

ImuSample SimulatedImu::sample(double dt_s) {
  if (dt_s > 0.0) {
    q_world_imu_ =
        (q_world_imu_ * Quaternion::fromRotationVector(angular_velocity_imu_radps_ * dt_s))
            .normalized();
    timestamp_s_ += dt_s;
  }

  const Vec3 gravity_world_mps2{0.0, 0.0, -gravity_mps2_};
  const Vec3 specific_force_world_mps2 =
      linear_acceleration_world_mps2_ - gravity_world_mps2;

  ImuSample sample;
  sample.timestamp_s = timestamp_s_;
  sample.accelerometer_mps2 =
      q_world_imu_.inverseRotate(specific_force_world_mps2) + accel_bias_mps2_ +
      sampleIsotropicNoise(accel_noise_stddev_mps2_);
  sample.gyroscope_radps =
      angular_velocity_imu_radps_ + gyro_bias_radps_ +
      sampleIsotropicNoise(gyro_noise_stddev_radps_);

  if (magnetometer_enabled_) {
    sample.magnetometer_uT = q_world_imu_.inverseRotate(magnetic_field_world_uT_);
  }

  return sample;
}

void SimulatedImu::setTime(double timestamp_s) {
  timestamp_s_ = timestamp_s;
}

void SimulatedImu::setSamplePeriod(double sample_period_s) {
  sample_period_s_ = sample_period_s;
}

void SimulatedImu::setOrientationWorldImu(const Quaternion& q_world_imu) {
  q_world_imu_ = q_world_imu.normalized();
}

void SimulatedImu::setAngularVelocityImuRadps(const Vec3& angular_velocity_imu_radps) {
  angular_velocity_imu_radps_ = angular_velocity_imu_radps;
}

void SimulatedImu::setLinearAccelerationWorldMps2(const Vec3& linear_acceleration_world_mps2) {
  linear_acceleration_world_mps2_ = linear_acceleration_world_mps2;
}

void SimulatedImu::setGyroBiasRadps(const Vec3& gyro_bias_radps) {
  gyro_bias_radps_ = gyro_bias_radps;
}

void SimulatedImu::setAccelBiasMps2(const Vec3& accel_bias_mps2) {
  accel_bias_mps2_ = accel_bias_mps2;
}

void SimulatedImu::setGyroNoiseStddevRadps(double stddev) {
  gyro_noise_stddev_radps_ = stddev;
}

void SimulatedImu::setAccelNoiseStddevMps2(double stddev) {
  accel_noise_stddev_mps2_ = stddev;
}

void SimulatedImu::setMagnetometerEnabled(bool enabled) {
  magnetometer_enabled_ = enabled;
}

void SimulatedImu::setMagneticFieldWorldUt(const Vec3& magnetic_field_world_uT) {
  magnetic_field_world_uT_ = magnetic_field_world_uT;
}

void SimulatedImu::seed(std::uint32_t seed_value) {
  rng_.seed(seed_value);
}

Vec3 SimulatedImu::sampleIsotropicNoise(double stddev) {
  if (stddev <= 0.0) {
    return {};
  }
  std::normal_distribution<double> distribution(0.0, stddev);
  return {distribution(rng_), distribution(rng_), distribution(rng_)};
}

}  // namespace state_estimation
