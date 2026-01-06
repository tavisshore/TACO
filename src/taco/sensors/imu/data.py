"""IMU data representation."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class IMUData:
    """Represents a single IMU measurement.

    Contains accelerometer and gyroscope readings with timestamp.
    """

    timestamp: float
    linear_acceleration: npt.NDArray[np.float64]  # m/s^2 in body frame
    angular_velocity: npt.NDArray[np.float64]  # rad/s in body frame
    acceleration_covariance: npt.NDArray[np.float64]  # 3x3
    gyroscope_covariance: npt.NDArray[np.float64]  # 3x3

    def __post_init__(self) -> None:
        """Validate IMU data."""
        if self.linear_acceleration.shape != (3,):
            raise ValueError("Linear acceleration must be a 3D vector")
        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be a 3D vector")
        if self.acceleration_covariance.shape != (3, 3):
            raise ValueError("Acceleration covariance must be 3x3")
        if self.gyroscope_covariance.shape != (3, 3):
            raise ValueError("Gyroscope covariance must be 3x3")

    @classmethod
    def from_raw(
        cls,
        timestamp: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        accel_cov: float = 0.01,
        gyro_cov: float = 0.001,
    ) -> "IMUData":
        """Create IMU data from raw measurements.

        Args:
            timestamp: Measurement timestamp.
            accel_x: X-axis acceleration (m/s^2).
            accel_y: Y-axis acceleration (m/s^2).
            accel_z: Z-axis acceleration (m/s^2).
            gyro_x: X-axis angular velocity (rad/s).
            gyro_y: Y-axis angular velocity (rad/s).
            gyro_z: Z-axis angular velocity (rad/s).
            accel_cov: Acceleration noise variance.
            gyro_cov: Gyroscope noise variance.

        Returns:
            IMUData instance.
        """
        return cls(
            timestamp=timestamp,
            linear_acceleration=np.array([accel_x, accel_y, accel_z]),
            angular_velocity=np.array([gyro_x, gyro_y, gyro_z]),
            acceleration_covariance=np.eye(3) * accel_cov,
            gyroscope_covariance=np.eye(3) * gyro_cov,
        )
