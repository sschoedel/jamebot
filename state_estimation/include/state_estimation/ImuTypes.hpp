#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

namespace state_estimation {

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  constexpr Vec3() = default;
  constexpr Vec3(double x_in, double y_in, double z_in) : x(x_in), y(y_in), z(z_in) {}

  constexpr Vec3 operator+(const Vec3& rhs) const {
    return {x + rhs.x, y + rhs.y, z + rhs.z};
  }

  constexpr Vec3 operator-(const Vec3& rhs) const {
    return {x - rhs.x, y - rhs.y, z - rhs.z};
  }

  constexpr Vec3 operator-() const {
    return {-x, -y, -z};
  }

  constexpr Vec3 operator*(double scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }

  constexpr Vec3 operator/(double scalar) const {
    return {x / scalar, y / scalar, z / scalar};
  }

  Vec3& operator+=(const Vec3& rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  Vec3& operator-=(const Vec3& rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  Vec3& operator*=(double scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }

  Vec3& operator/=(double scalar) {
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
  }

  constexpr double dot(const Vec3& rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  constexpr Vec3 cross(const Vec3& rhs) const {
    return {
      y * rhs.z - z * rhs.y,
      z * rhs.x - x * rhs.z,
      x * rhs.y - y * rhs.x,
    };
  }

  double squaredNorm() const {
    return dot(*this);
  }

  double norm() const {
    return std::sqrt(squaredNorm());
  }

  Vec3 normalized(double epsilon = 1e-12) const {
    const double n = norm();
    if (n <= epsilon) {
      return {};
    }
    return *this / n;
  }

  bool isFinite() const {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }
};

inline constexpr Vec3 operator*(double scalar, const Vec3& vector) {
  return vector * scalar;
}

struct ImuSample {
  double timestamp_s{0.0};
  Vec3 accelerometer_mps2{};
  Vec3 gyroscope_radps{};
  std::optional<Vec3> magnetometer_uT{};
};

class ImuSource {
 public:
  virtual ~ImuSource() = default;
  virtual ImuSample readSample() = 0;
};

}  // namespace state_estimation

