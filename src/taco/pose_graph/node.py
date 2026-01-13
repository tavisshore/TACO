"""Pose graph node representation using GTSAM Pose2.

This module provides helper classes for working with GTSAM Pose2 objects
for 2D vehicle trajectory representation (x, y, yaw).
"""

from dataclasses import dataclass

import gtsam
import numpy as np
import numpy.typing as npt


@dataclass
class PoseNode:
    """Represents a 2D pose node for GTSAM integration.

    This is a convenience wrapper that can be converted to/from GTSAM Pose2.
    Stores position (x, y) and yaw angle for vehicle trajectory representation.
    """

    position: npt.NDArray[np.float64]  # 2D position (x, y)
    yaw: float  # Yaw angle in radians
    timestamp: float
    covariance: npt.NDArray[np.float64] | None = None  # 3x3 covariance matrix

    def __post_init__(self) -> None:
        """Validate node data."""
        if self.position.shape != (2,):
            raise ValueError("Position must be a 2D vector (x, y)")

        if self.covariance is not None and self.covariance.shape != (3, 3):
            raise ValueError("Covariance must be a 3x3 matrix for Pose2")

    def to_gtsam_pose(self) -> gtsam.Pose2:
        """Convert to GTSAM Pose2.

        Returns:
            GTSAM Pose2 object.
        """
        return gtsam.Pose2(float(self.position[0]), float(self.position[1]), float(self.yaw))

    @staticmethod
    def from_gtsam_pose(
        pose: gtsam.Pose2,
        timestamp: float,
        covariance: npt.NDArray[np.float64] | None = None,
    ) -> "PoseNode":
        """Create PoseNode from GTSAM Pose2.

        Args:
            pose: GTSAM Pose2 object.
            timestamp: Timestamp for the pose.
            covariance: Optional 3x3 covariance matrix.

        Returns:
            PoseNode instance.
        """
        position = np.array([pose.x(), pose.y()], dtype=np.float64)
        yaw = pose.theta()

        return PoseNode(
            position=position,
            yaw=yaw,
            timestamp=timestamp,
            covariance=covariance,
        )


def create_noise_model_diagonal(
    sigmas: npt.NDArray[np.float64],
) -> gtsam.noiseModel.Diagonal:
    """Create a diagonal noise model from standard deviations.

    Args:
        sigmas: Standard deviations for each dimension (3 for Pose2: x, y, theta).

    Returns:
        GTSAM diagonal noise model.
    """
    vec = np.array([float(s) for s in sigmas], dtype=np.float64)
    return gtsam.noiseModel.Diagonal.Sigmas(vec)


def create_noise_model_gaussian(
    covariance: npt.NDArray[np.float64],
) -> gtsam.noiseModel.Gaussian:
    """Create a Gaussian noise model from covariance.

    Args:
        covariance: Covariance matrix (3x3 for Pose2).

    Returns:
        GTSAM Gaussian noise model.
    """
    return gtsam.noiseModel.Gaussian.Covariance(covariance)


def create_noise_model_isotropic(dim: int, sigma: float) -> gtsam.noiseModel.Isotropic:
    """Create an isotropic noise model.

    Args:
        dim: Dimension of the noise model (3 for Pose2).
        sigma: Standard deviation (same for all dimensions).

    Returns:
        GTSAM isotropic noise model.
    """
    return gtsam.noiseModel.Isotropic.Sigma(dim, sigma)
