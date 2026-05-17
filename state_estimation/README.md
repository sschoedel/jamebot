# state_estimation

A small C++17 quaternion orientation estimator for IMU-based robot state estimation.

The estimator assumes the robot is stationary for a short boot window, defaults to
3 seconds, and initializes roll/pitch from gravity. Absolute yaw is intentionally
unobservable without a trusted heading reference, so startup yaw is set by the
chosen gravity-alignment solution instead of a magnetometer.

## Coordinate conventions

- `W`: world frame, z-up.
- `I`: IMU sensor frame.
- `q_WI`: unit quaternion rotating IMU-frame vectors into world-frame vectors.
- World gravity direction is `[0, 0, -1]`.
- `projectedGravityImu()` returns gravity direction in the IMU frame:

```text
g_I = R_WI^T * [0, 0, -1]
```

At rest, an accelerometer measures specific force, so gravity direction in IMU
frame is estimated as:

```text
g_I ~= -normalize(accelerometer_mps2)
```

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the demo:

```sh
./build/state_estimation_demo
```

## API sketch

```cpp
#include <state_estimation/OrientationEstimator.hpp>

state_estimation::OrientationEstimator estimator;

state_estimation::ImuSample sample;
sample.timestamp_s = time_s;
sample.accelerometer_mps2 = {ax, ay, az};
sample.gyroscope_radps = {gx, gy, gz};

const auto estimate = estimator.step(sample);

if (estimate.initialized) {
  const auto q_WI = estimate.q_world_imu;
  const auto gravity_I = estimate.projected_gravity_imu;
  const auto omega_I = estimate.angular_velocity_imu_radps;
}
```

For now, `SimulatedImu` provides programmatically controllable samples. A real
driver can later produce the same `ImuSample` objects and feed them into
`OrientationEstimator::step`.
