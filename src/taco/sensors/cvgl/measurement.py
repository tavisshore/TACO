"""CVGL measurement representation with GTSAM Pose2 support.

This module provides the CVGLMeasurement class for 2D vehicle localization
using image matching (x, y, yaw).
"""

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt


@dataclass
class CVGLMeasurement:
    """Represents a CVGL image localization measurement for 2D poses.

    CVGL provides position-only measurements (x, y) from image matching.
    Heading (yaw) should come from IMU integration, not from visual localization,
    as single-image heading estimation is unreliable.
    """

    timestamp: float
    position: npt.NDArray[np.float64]  # 2D position (x, y) in world frame (UTM or local)
    position_covariance: npt.NDArray[np.float64]  # 2x2 covariance for (x, y)
    coordinates: tuple[float, float]  # Geographic coordinates (lat, lon)
    confidence: float  # Matching confidence [0, 1]
    num_inliers: int  # Number of inlier features
    yaw: float | None = None  # Optional: heading from IMU (not from CVGL)
    image_id: str | None = None  # Reference image identifier

    def __post_init__(self) -> None:
        """Validate measurement data."""
        if self.position.shape != (2,):
            raise ValueError("Position must be a 2D vector (x, y)")

        if self.position_covariance.shape != (2, 2):
            raise ValueError("Position covariance must be 2x2 for (x, y)")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")

    @property
    def position_std(self) -> float:
        """Get position standard deviation (average of x and y std devs).

        For isotropic covariance (equal uncertainty in x and y), this returns
        the standard deviation. For anisotropic covariance, returns the average.

        Returns:
            Position standard deviation in meters.
        """
        # Extract standard deviations from diagonal of covariance matrix
        std_x = np.sqrt(self.position_covariance[0, 0])
        std_y = np.sqrt(self.position_covariance[1, 1])
        return float((std_x + std_y) / 2.0)

    @property
    def covariance(self) -> npt.NDArray[np.float64]:
        """Get 3x3 covariance for Pose2 compatibility.

        Returns 3x3 covariance with position covariance and very high yaw uncertainty
        (effectively a position-only measurement).
        """
        # Create 3x3 covariance with very high yaw uncertainty
        cov_3x3 = np.eye(3) * 1e6  # Default to very high uncertainty
        cov_3x3[:2, :2] = self.position_covariance  # Set position covariance
        cov_3x3[2, 2] = 1e6  # Very high yaw uncertainty (effectively ignored)
        return cov_3x3

    def to_transformation_matrix(self, yaw: float | None = None) -> npt.NDArray[np.float64]:
        """Convert measurement to 3x3 SE(2) transformation matrix.

        Args:
            yaw: Heading angle in radians. If None, uses self.yaw or defaults to 0.

        Returns:
            3x3 homogeneous transformation matrix.
        """
        if yaw is None:
            yaw = self.yaw if self.yaw is not None else 0.0

        c, s = np.cos(yaw), np.sin(yaw)
        T = np.array(
            [
                [c, -s, self.position[0]],
                [s, c, self.position[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        return T

    def to_gtsam_pose(self, yaw: float | None = None) -> gtsam.Pose2:
        """Convert measurement to GTSAM Pose2.

        Note: CVGL provides position-only measurements. The yaw parameter should
        come from the current pose estimate (e.g., from IMU integration).

        Args:
            yaw: Heading angle in radians from IMU. If None, uses self.yaw or defaults to 0.
                For position-only measurements, the yaw uncertainty is very high (see covariance).

        Returns:
            GTSAM Pose2 object with position from CVGL and yaw from IMU.
        """
        if yaw is None:
            yaw = self.yaw if self.yaw is not None else 0.0

        return gtsam.Pose2(float(self.position[0]), float(self.position[1]), float(yaw))

    def get_gtsam_noise_model(self) -> gtsam.noiseModel.Gaussian:
        """Get GTSAM noise model from covariance.

        Returns:
            GTSAM Gaussian noise model.
        """
        return gtsam.noiseModel.Gaussian.Covariance(self.covariance)
