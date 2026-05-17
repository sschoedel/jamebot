#include <state_estimation/OrientationEstimator.hpp>

#include <cmath>
#include <sstream>

namespace state_estimation {

OrientationEstimator::OrientationEstimator(OrientationEstimatorConfig config)
    : config_(config) {
  reset();
}

void OrientationEstimator::reset() {
  initialized_ = false;
  q_world_imu_ = Quaternion::identity();
  angular_velocity_imu_radps_ = {};
  gyro_bias_radps_ = {};
  status_ = "waiting for initialization samples";
  have_last_timestamp_ = false;
  last_timestamp_s_ = 0.0;
  resetInitializationWindow();
}

OrientationEstimate OrientationEstimator::step(const ImuSample& sample) {
  if (!sample.accelerometer_mps2.isFinite() || !sample.gyroscope_radps.isFinite() ||
      !std::isfinite(sample.timestamp_s)) {
    status_ = "rejected non-finite IMU sample";
    return estimate();
  }

  angular_velocity_imu_radps_ = sample.gyroscope_radps - gyro_bias_radps_;

  if (!initialized_) {
    if (!have_initialization_start_) {
      have_initialization_start_ = true;
      initialization_start_timestamp_s_ = sample.timestamp_s;
    }

    initialization_last_timestamp_s_ = sample.timestamp_s;
    initialization_accel_sum_ += sample.accelerometer_mps2;
    initialization_gyro_sum_ += sample.gyroscope_radps;
    ++initialization_sample_count_;

    const double initialization_duration =
        initialization_last_timestamp_s_ - initialization_start_timestamp_s_;

    if (initialization_duration >= config_.initialization_duration_s) {
      initializeFromAccumulatedSamples();
    } else {
      std::ostringstream status;
      status << "initializing: " << initialization_duration << " / "
             << config_.initialization_duration_s << " s";
      status_ = status.str();
    }

    return estimate();
  }

  updateRunningEstimate(sample);
  return estimate();
}

OrientationEstimate OrientationEstimator::addImuSample(const ImuSample& sample) {
  return step(sample);
}

Vec3 OrientationEstimator::projectedGravityImu() const {
  return q_world_imu_.inverseRotate(worldGravityDirection()).normalized();
}

OrientationEstimate OrientationEstimator::estimate() const {
  return OrientationEstimate{
    initialized_,
    q_world_imu_,
    projectedGravityImu(),
    angular_velocity_imu_radps_,
    gyro_bias_radps_,
    initialization_sample_count_,
    status_,
  };
}

bool OrientationEstimator::initializeFromAccumulatedSamples() {
  if (initialization_sample_count_ < config_.minimum_initialization_samples) {
    status_ = "initialization failed: not enough samples";
    resetInitializationWindow();
    return false;
  }

  const Vec3 mean_accel = initialization_accel_sum_ /
                          static_cast<double>(initialization_sample_count_);
  const Vec3 mean_gyro = initialization_gyro_sum_ /
                         static_cast<double>(initialization_sample_count_);

  const double accel_norm = mean_accel.norm();
  if (std::abs(accel_norm - config_.gravity_mps2) >
      config_.initialization_accel_norm_tolerance_mps2) {
    std::ostringstream status;
    status << "initialization failed: mean accel norm " << accel_norm
           << " m/s^2 is outside tolerance";
    status_ = status.str();
    resetInitializationWindow();
    return false;
  }

  if (mean_gyro.norm() > config_.initialization_mean_gyro_limit_radps) {
    std::ostringstream status;
    status << "initialization failed: mean gyro norm " << mean_gyro.norm()
           << " rad/s is outside tolerance";
    status_ = status.str();
    resetInitializationWindow();
    return false;
  }

  const Vec3 measured_gravity_imu = (-mean_accel).normalized();
  q_world_imu_ =
      Quaternion::fromTwoVectors(measured_gravity_imu, worldGravityDirection()).normalized();
  gyro_bias_radps_ = mean_gyro;
  angular_velocity_imu_radps_ = {};
  initialized_ = true;
  have_last_timestamp_ = true;
  last_timestamp_s_ = initialization_last_timestamp_s_;
  status_ = "initialized";
  return true;
}

void OrientationEstimator::resetInitializationWindow() {
  have_initialization_start_ = false;
  initialization_start_timestamp_s_ = 0.0;
  initialization_last_timestamp_s_ = 0.0;
  initialization_sample_count_ = 0;
  initialization_accel_sum_ = {};
  initialization_gyro_sum_ = {};
}

void OrientationEstimator::updateRunningEstimate(const ImuSample& sample) {
  if (!have_last_timestamp_) {
    have_last_timestamp_ = true;
    last_timestamp_s_ = sample.timestamp_s;
    return;
  }

  const double dt_s = sample.timestamp_s - last_timestamp_s_;
  last_timestamp_s_ = sample.timestamp_s;

  if (dt_s <= 0.0) {
    status_ = "skipped sample: timestamp did not advance";
    return;
  }

  if (dt_s > config_.max_integration_dt_s) {
    status_ = "skipped sample: integration timestep too large";
    return;
  }

  angular_velocity_imu_radps_ = sample.gyroscope_radps - gyro_bias_radps_;
  Vec3 integration_angular_velocity = angular_velocity_imu_radps_;

  const double accel_norm = sample.accelerometer_mps2.norm();
  if (accelNormIsTrusted(accel_norm)) {
    const Vec3 measured_gravity_imu = (-sample.accelerometer_mps2).normalized();
    const Vec3 estimated_gravity_imu = projectedGravityImu();
    const Vec3 tilt_error_imu = measured_gravity_imu.cross(estimated_gravity_imu);
    integration_angular_velocity += config_.accel_correction_gain * tilt_error_imu;
    status_ = "running: gyro integration with accelerometer tilt correction";
  } else {
    status_ = "running: gyro integration only, accel norm outside correction gate";
  }

  q_world_imu_ =
      (q_world_imu_ * Quaternion::fromRotationVector(integration_angular_velocity * dt_s))
          .normalized();
}

bool OrientationEstimator::accelNormIsTrusted(double accel_norm) const {
  return std::isfinite(accel_norm) &&
         std::abs(accel_norm - config_.gravity_mps2) <=
             config_.accel_correction_norm_tolerance_mps2;
}

}  // namespace state_estimation
