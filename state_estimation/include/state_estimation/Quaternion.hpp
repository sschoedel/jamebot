#pragma once

#include <algorithm>
#include <cmath>

#include <state_estimation/ImuTypes.hpp>

namespace state_estimation {

class Quaternion {
 public:
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};

  constexpr Quaternion() = default;
  constexpr Quaternion(double w_in, double x_in, double y_in, double z_in)
      : w(w_in), x(x_in), y(y_in), z(z_in) {}

  static constexpr Quaternion identity() {
    return {};
  }

  static Quaternion fromAxisAngle(const Vec3& axis, double angle_rad) {
    const Vec3 unit_axis = axis.normalized();
    const double half_angle = 0.5 * angle_rad;
    const double sin_half_angle = std::sin(half_angle);
    return Quaternion{
      std::cos(half_angle),
      unit_axis.x * sin_half_angle,
      unit_axis.y * sin_half_angle,
      unit_axis.z * sin_half_angle,
    }.normalized();
  }

  static Quaternion fromRotationVector(const Vec3& rotation_vector_rad) {
    const double angle = rotation_vector_rad.norm();
    if (angle < 1e-12) {
      return identity();
    }
    return fromAxisAngle(rotation_vector_rad / angle, angle);
  }

  static Quaternion fromTwoVectors(const Vec3& from, const Vec3& to) {
    const Vec3 from_unit = from.normalized();
    const Vec3 to_unit = to.normalized();

    const double dot = std::clamp(from_unit.dot(to_unit), -1.0, 1.0);
    if (dot > 1.0 - 1e-12) {
      return identity();
    }

    if (dot < -1.0 + 1e-12) {
      Vec3 orthogonal = Vec3{1.0, 0.0, 0.0}.cross(from_unit);
      if (orthogonal.norm() < 1e-6) {
        orthogonal = Vec3{0.0, 1.0, 0.0}.cross(from_unit);
      }
      return fromAxisAngle(orthogonal, pi());
    }

    const Vec3 cross = from_unit.cross(to_unit);
    return Quaternion{1.0 + dot, cross.x, cross.y, cross.z}.normalized();
  }

  static Quaternion fromRollPitchYaw(double roll_rad, double pitch_rad, double yaw_rad) {
    const double cr = std::cos(0.5 * roll_rad);
    const double sr = std::sin(0.5 * roll_rad);
    const double cp = std::cos(0.5 * pitch_rad);
    const double sp = std::sin(0.5 * pitch_rad);
    const double cy = std::cos(0.5 * yaw_rad);
    const double sy = std::sin(0.5 * yaw_rad);

    return Quaternion{
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    }.normalized();
  }

  static constexpr double pi() {
    return 3.141592653589793238462643383279502884;
  }

  double squaredNorm() const {
    return w * w + x * x + y * y + z * z;
  }

  double norm() const {
    return std::sqrt(squaredNorm());
  }

  Quaternion normalized(double epsilon = 1e-12) const {
    const double n = norm();
    if (n <= epsilon) {
      return identity();
    }
    return {w / n, x / n, y / n, z / n};
  }

  Quaternion conjugate() const {
    return {w, -x, -y, -z};
  }

  Quaternion inverse() const {
    const double squared_norm = squaredNorm();
    if (squared_norm <= 1e-12) {
      return identity();
    }
    const Quaternion c = conjugate();
    return {c.w / squared_norm, c.x / squared_norm, c.y / squared_norm, c.z / squared_norm};
  }

  Quaternion operator*(const Quaternion& rhs) const {
    return {
      w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
      w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
      w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
      w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
    };
  }

  Vec3 rotate(const Vec3& vector) const {
    const Quaternion vector_quat{0.0, vector.x, vector.y, vector.z};
    const Quaternion rotated = (*this) * vector_quat * conjugate();
    return {rotated.x, rotated.y, rotated.z};
  }

  Vec3 inverseRotate(const Vec3& vector) const {
    const Quaternion vector_quat{0.0, vector.x, vector.y, vector.z};
    const Quaternion rotated = conjugate() * vector_quat * (*this);
    return {rotated.x, rotated.y, rotated.z};
  }
};

}  // namespace state_estimation

