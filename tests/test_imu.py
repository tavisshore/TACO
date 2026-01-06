"""Tests for IMU data processing."""

import numpy as np
import pytest

from taco.sensors.imu import IMUData, IMUIntegrator


class TestIMUData:
    """Test IMUData class."""

    def test_imu_data_creation(self) -> None:
        """Test creating IMU data."""
        imu = IMUData.from_raw(
            timestamp=0.0,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=-9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
        )

        assert imu.timestamp == 0.0
        assert np.allclose(imu.linear_acceleration, [0.0, 0.0, -9.81])
        assert np.allclose(imu.angular_velocity, [0.0, 0.0, 0.0])

    def test_invalid_acceleration_shape(self) -> None:
        """Test that invalid acceleration shape raises error."""
        with pytest.raises(ValueError, match="Linear acceleration must be a 3D vector"):
            IMUData(
                timestamp=0.0,
                linear_acceleration=np.array([0.0, 0.0]),  # Wrong shape
                angular_velocity=np.zeros(3),
                acceleration_covariance=np.eye(3),
                gyroscope_covariance=np.eye(3),
            )


class TestIMUIntegrator:
    """Test IMU integration."""

    def test_integrator_initialization(self) -> None:
        """Test initializing integrator."""
        gravity = np.array([0.0, 0.0, -9.81])
        integrator = IMUIntegrator(gravity)

        assert np.allclose(integrator.gravity, gravity)
        assert np.allclose(integrator.position, [0.0, 0.0, 0.0])
        assert np.allclose(integrator.velocity, [0.0, 0.0, 0.0])

    def test_stationary_integration(self) -> None:
        """Test integration with stationary sensor."""
        gravity = np.array([0.0, 0.0, -9.81])
        integrator = IMUIntegrator(gravity)

        # Create stationary measurements (only gravity)
        measurements = [
            IMUData.from_raw(0.0, 0.0, 0.0, -9.81, 0.0, 0.0, 0.0),
            IMUData.from_raw(0.1, 0.0, 0.0, -9.81, 0.0, 0.0, 0.0),
            IMUData.from_raw(0.2, 0.0, 0.0, -9.81, 0.0, 0.0, 0.0),
        ]

        position, velocity, orientation = integrator.integrate(
            measurements, initial_orientation=np.eye(3)
        )

        # Should remain near origin with proper gravity compensation
        assert np.linalg.norm(velocity) < 1.0  # Small drift expected
