"""IMU integration for 2D odometry estimation using GTSAM Pose2.

This module integrates IMU measurements to estimate 2D pose changes (x, y, yaw)
for vehicle trajectory estimation.
"""

import gtsam
import numpy as np
import numpy.typing as npt

from .data import IMUData


class IMUIntegrator:
    """Integrates IMU measurements to estimate 2D pose changes using GTSAM.

    Performs numerical integration of accelerometer and gyroscope data
    to compute relative pose transformations compatible with GTSAM Pose2.
    Only x, y position and yaw angle are tracked for 2D vehicle trajectories.
    """

    def __init__(self, gravity: float = 9.81) -> None:
        """Initialize IMU integrator.

        Args:
            gravity: Magnitude of gravity (default 9.81 m/s^2).
                     Used to remove gravity from vertical accelerometer readings.
        """
        self.gravity = gravity
        self.position = np.zeros(2)  # 2D position (x, y)
        self.velocity = np.zeros(2)  # 2D velocity (vx, vy)
        self.yaw = 0.0  # Yaw angle

    def integrate(
        self,
        imu_measurements: list[IMUData],
        initial_yaw: float = 0.0,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Integrate IMU measurements for 2D motion.

        Args:
            imu_measurements: List of IMU measurements to integrate.
            initial_yaw: Initial yaw angle in radians.

        Returns:
            Tuple of (position [x, y], velocity [vx, vy], yaw).
        """
        self.yaw = initial_yaw
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)

        for i in range(len(imu_measurements) - 1):
            dt = imu_measurements[i + 1].timestamp - imu_measurements[i].timestamp
            self._integrate_step(imu_measurements[i], dt)

        return self.position.copy(), self.velocity.copy(), self.yaw

    def integrate_to_gtsam_pose(
        self,
        imu_measurements: list[IMUData],
        initial_pose: gtsam.Pose2,
    ) -> gtsam.Pose2:
        """Integrate IMU measurements and return GTSAM Pose2.

        Args:
            imu_measurements: List of IMU measurements to integrate.
            initial_pose: Initial pose as GTSAM Pose2.

        Returns:
            Integrated pose as GTSAM Pose2.
        """
        # Get initial state
        initial_yaw = initial_pose.theta()
        initial_position = np.array([initial_pose.x(), initial_pose.y()])

        # Integrate
        delta_position, _velocity, final_yaw = self.integrate(imu_measurements, initial_yaw)

        # Transform delta position from body frame to world frame
        c, s = np.cos(initial_yaw), np.sin(initial_yaw)
        R = np.array([[c, -s], [s, c]])
        world_delta = R @ delta_position

        # Add to initial position
        final_position = initial_position + world_delta

        return gtsam.Pose2(float(final_position[0]), float(final_position[1]), float(final_yaw))

    def _integrate_step(self, imu: IMUData, dt: float) -> None:
        """Perform single integration step for 2D motion.

        Args:
            imu: IMU measurement.
            dt: Time step.
        """
        # Integrate yaw from gyroscope z-axis (rotation around vertical)
        omega_z = imu.angular_velocity[2]  # Yaw rate
        self.yaw += omega_z * dt

        # Get 2D acceleration in body frame (x forward, y left)
        # Remove gravity effect from z-axis accelerometer (not needed for 2D)
        accel_body = imu.linear_acceleration[:2]  # Only x, y

        # Transform to world frame using current yaw
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        accel_world = R @ accel_body

        # Integrate velocity and position
        self.velocity += accel_world * dt
        self.position += self.velocity * dt + 0.5 * accel_world * dt**2
