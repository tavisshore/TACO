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

    Contains the estimated global pose (x, y, yaw) from image matching
    and the associated uncertainty.
    """

    timestamp: float
    position: npt.NDArray[np.float64]  # 2D position (x, y) in world frame
    yaw: float  # Yaw angle in radians
    covariance: npt.NDArray[np.float64]  # 3x3 covariance (x, y, theta)
    confidence: float  # Matching confidence [0, 1]
    num_inliers: int  # Number of inlier features
    image_id: str | None = None  # Reference image identifier

    def __post_init__(self) -> None:
        """Validate measurement data."""
        if self.position.shape != (2,):
            raise ValueError("Position must be a 2D vector (x, y)")

        if self.covariance.shape != (3, 3):
            raise ValueError("Covariance must be 3x3 for Pose2")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")

    def to_transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Convert measurement to 3x3 SE(2) transformation matrix.

        Returns:
            3x3 homogeneous transformation matrix.
        """
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        T = np.array(
            [
                [c, -s, self.position[0]],
                [s, c, self.position[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        return T

    def to_gtsam_pose(self) -> gtsam.Pose2:
        """Convert measurement to GTSAM Pose2.

        Returns:
            GTSAM Pose2 object.
        """
        return gtsam.Pose2(float(self.position[0]), float(self.position[1]), float(self.yaw))

    def get_gtsam_noise_model(self) -> gtsam.noiseModel.Gaussian:
        """Get GTSAM noise model from covariance.

        Returns:
            GTSAM Gaussian noise model.
        """
        return gtsam.noiseModel.Gaussian.Covariance(self.covariance)
