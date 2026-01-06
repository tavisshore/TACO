"""IMU preintegration using GTSAM.

Implements IMU preintegration using GTSAM's built-in PreintegratedIMUMeasurements.
"""

from typing import Optional

import gtsam
import numpy as np
import numpy.typing as npt

from .data import IMUData


class IMUPreintegrator:
    """Preintegrates IMU measurements between keyframes using GTSAM.

    Uses GTSAM's PreintegratedIMUMeasurements for efficient optimization.
    """

    def __init__(
        self,
        gravity: npt.NDArray[np.float64],
        gyro_noise_sigma: float = 0.0001,
        accel_noise_sigma: float = 0.001,
        gyro_bias_sigma: float = 0.00001,
        accel_bias_sigma: float = 0.0001,
        integration_error_sigma: float = 0.0001,
    ) -> None:
        """Initialize preintegrator with GTSAM parameters.

        Args:
            gravity: Gravity vector in world frame.
            gyro_noise_sigma: Gyroscope noise sigma (rad/s).
            accel_noise_sigma: Accelerometer noise sigma (m/s^2).
            gyro_bias_sigma: Gyroscope bias sigma (rad/s).
            accel_bias_sigma: Accelerometer bias sigma (m/s^2).
            integration_error_sigma: Integration uncertainty sigma.
        """
        self.gravity_magnitude = np.linalg.norm(gravity)

        # Create GTSAM PreintegrationParams
        self.params = gtsam.PreintegrationParams.MakeSharedU(self.gravity_magnitude)

        # Set noise parameters
        kGyroSigma = gyro_noise_sigma
        kAccelSigma = accel_noise_sigma
        kGyroBiasSigma = gyro_bias_sigma
        kAccelBiasSigma = accel_bias_sigma
        kIntegrationSigma = integration_error_sigma

        # Gyroscope and accelerometer covariance
        self.params.setGyroscopeCovariance(np.eye(3) * kGyroSigma**2)
        self.params.setAccelerometerCovariance(np.eye(3) * kAccelSigma**2)
        self.params.setIntegrationCovariance(np.eye(3) * kIntegrationSigma**2)

        # Bias covariance
        self.params.setBiasAccCovariance(np.eye(3) * kAccelBiasSigma**2)
        self.params.setBiasOmegaCovariance(np.eye(3) * kGyroBiasSigma**2)

        # Current preintegrated measurement
        self.pim: gtsam.PreintegratedImuMeasurements
        self._reset_pim()

    def _reset_pim(self, bias: gtsam.imuBias.ConstantBias | None = None) -> None:
        """Reset the preintegrated measurements.

        Args:
            bias: Initial IMU bias.
        """
        if bias is None:
            bias = gtsam.imuBias.ConstantBias()
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, bias)

    def reset(self, bias: gtsam.imuBias.ConstantBias | None = None) -> None:
        """Reset preintegrated measurements.

        Args:
            bias: Initial IMU bias.
        """
        self._reset_pim(bias)

    def integrate_measurement(
        self,
        measured_acceleration: npt.NDArray[np.float64],
        measured_omega: npt.NDArray[np.float64],
        dt: float,
    ) -> None:
        """Integrate a single IMU measurement.

        Args:
            measured_acceleration: Measured acceleration (m/s^2).
            measured_omega: Measured angular velocity (rad/s).
            dt: Time interval (s).
        """
        self.pim.integrateMeasurement(measured_acceleration, measured_omega, dt)

    def integrate_measurements(
        self,
        imu_measurements: list[IMUData],
        bias_accel: npt.NDArray[np.float64] | None = None,
        bias_gyro: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Preintegrate a list of IMU measurements.

        Args:
            imu_measurements: List of IMU measurements.
            bias_accel: Accelerometer bias to use.
            bias_gyro: Gyroscope bias to use.
        """
        # Use default biases if not provided
        if bias_accel is None:
            bias_accel = np.zeros(3)
        if bias_gyro is None:
            bias_gyro = np.zeros(3)

        # Create bias object
        bias = gtsam.imuBias.ConstantBias(bias_accel, bias_gyro)
        self.reset(bias)

        # Integrate all measurements
        for i in range(len(imu_measurements) - 1):
            dt = imu_measurements[i + 1].timestamp - imu_measurements[i].timestamp
            self.integrate_measurement(
                imu_measurements[i].linear_acceleration,
                imu_measurements[i].angular_velocity,
                dt,
            )

    def predict(
        self,
        current_pose: gtsam.Pose3,
        current_velocity: npt.NDArray[np.float64],
    ) -> tuple[gtsam.Pose3, npt.NDArray[np.float64]]:
        """Predict pose and velocity after preintegration.

        Args:
            current_pose: Current pose.
            current_velocity: Current velocity (3D).

        Returns:
            Tuple of (predicted_pose, predicted_velocity).
        """
        current_vel_gtsam = current_velocity.reshape(3)

        predicted_nav_state = self.pim.predict(
            gtsam.NavState(current_pose, current_vel_gtsam),
            self.pim.biasHat(),
        )

        predicted_pose = predicted_nav_state.pose()
        predicted_velocity = predicted_nav_state.velocity()

        return predicted_pose, np.array(
            [predicted_velocity[0], predicted_velocity[1], predicted_velocity[2]]
        )

    def get_delta_pose(self) -> gtsam.Pose3:
        """Get the preintegrated delta pose.

        Returns:
            Delta Pose3.
        """
        return gtsam.Pose3(self.pim.deltaRij(), self.pim.deltaPij())

    def get_preintegrated_measurements(self) -> gtsam.PreintegratedImuMeasurements:
        """Get the GTSAM preintegrated measurements object.

        Returns:
            GTSAM PreintegratedImuMeasurements.
        """
        return self.pim


def create_imu_factor(
    pose_key_i: int,
    vel_key_i: int,
    pose_key_j: int,
    vel_key_j: int,
    bias_key: int,
    pim: gtsam.PreintegratedImuMeasurements,
) -> gtsam.ImuFactor:
    """Create a GTSAM ImuFactor from preintegrated measurements.

    Args:
        pose_key_i: Key for pose at time i.
        vel_key_i: Key for velocity at time i.
        pose_key_j: Key for pose at time j.
        vel_key_j: Key for velocity at time j.
        bias_key: Key for IMU bias.
        pim: Preintegrated IMU measurements.

    Returns:
        GTSAM ImuFactor.
    """
    return gtsam.ImuFactor(
        gtsam.symbol("x", pose_key_i),
        gtsam.symbol("v", vel_key_i),
        gtsam.symbol("x", pose_key_j),
        gtsam.symbol("v", vel_key_j),
        gtsam.symbol("b", bias_key),
        pim,
    )
