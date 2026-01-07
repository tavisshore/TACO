"""IMU integration for odometry estimation using GTSAM."""

import gtsam
import numpy as np
import numpy.typing as npt

from .data import IMUData

# Constant for rotation threshold
_ROTATION_EPSILON = 1e-8


class IMUIntegrator:
    """Integrates IMU measurements to estimate pose changes using GTSAM.

    Performs numerical integration of accelerometer and gyroscope data
    to compute relative pose transformations compatible with GTSAM.
    """

    def __init__(self, gravity: npt.NDArray[np.float64]) -> None:
        """Initialize IMU integrator.

        Args:
            gravity: Gravity vector in world frame (typically [0, 0, -9.81]).
        """
        self.gravity = gravity
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)  # Rotation matrix

    def integrate(
        self,
        imu_measurements: list[IMUData],
        initial_orientation: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Integrate IMU measurements.

        Args:
            imu_measurements: List of IMU measurements to integrate.
            initial_orientation: Initial orientation as rotation matrix.

        Returns:
            Tuple of (position, velocity, orientation).
        """
        self.orientation = initial_orientation
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        for i in range(len(imu_measurements) - 1):
            dt = imu_measurements[i + 1].timestamp - imu_measurements[i].timestamp
            self._integrate_step(imu_measurements[i], dt)

        return self.position, self.velocity, self.orientation

    def integrate_to_gtsam_pose(
        self,
        imu_measurements: list[IMUData],
        initial_pose: gtsam.Pose3,
    ) -> gtsam.Pose3:
        """Integrate IMU measurements and return GTSAM Pose3.

        Args:
            imu_measurements: List of IMU measurements to integrate.
            initial_pose: Initial pose as GTSAM Pose3.

        Returns:
            Integrated pose as GTSAM Pose3.
        """
        # Get initial orientation
        initial_R = initial_pose.rotation().matrix()
        initial_t = initial_pose.translation()

        # Integrate
        position, _velocity, orientation = self.integrate(imu_measurements, initial_R)

        # Add initial position (initial_t is already a numpy array from translation())
        final_position = initial_t + position

        # Create GTSAM Pose3
        rot = gtsam.Rot3(orientation)
        point = gtsam.Point3(final_position[0], final_position[1], final_position[2])

        return gtsam.Pose3(rot, point)

    def _integrate_step(self, imu: IMUData, dt: float) -> None:
        """Perform single integration step.

        Args:
            imu: IMU measurement.
            dt: Time step.
        """
        # Integrate angular velocity to update orientation
        omega = imu.angular_velocity
        omega_norm = np.linalg.norm(omega)

        if omega_norm > _ROTATION_EPSILON:
            # Rodrigues' rotation formula
            omega_skew = self._skew_symmetric(omega)
            R_delta = (
                np.eye(3)
                + np.sin(omega_norm * dt) / omega_norm * omega_skew
                + (1 - np.cos(omega_norm * dt)) / (omega_norm**2) * (omega_skew @ omega_skew)
            )
            self.orientation = self.orientation @ R_delta

        # Transform acceleration to world frame and remove gravity
        # IMU measures specific force (includes gravity effect)
        # To get true kinematic acceleration, subtract gravity
        accel_world = self.orientation @ imu.linear_acceleration - self.gravity
        self.velocity += accel_world * dt
        self.position += self.velocity * dt + 0.5 * accel_world * dt**2

    @staticmethod
    def _skew_symmetric(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Create skew-symmetric matrix from vector.

        Args:
            v: 3D vector.

        Returns:
            3x3 skew-symmetric matrix.
        """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
