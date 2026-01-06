"""Pose graph node representation using GTSAM.

This module provides helper classes for working with GTSAM Pose3 objects.
"""

from dataclasses import dataclass
from typing import Optional

import gtsam
import numpy as np
import numpy.typing as npt


@dataclass
class PoseNode:
    """Represents a pose node for GTSAM integration.

    This is a convenience wrapper that can be converted to/from GTSAM Pose3.
    """

    position: npt.NDArray[np.float64]  # 3D position (x, y, z)
    orientation: npt.NDArray[np.float64]  # Quaternion (w, x, y, z) or rotation matrix
    timestamp: float
    covariance: Optional[npt.NDArray[np.float64]] = None  # 6x6 covariance matrix

    def __post_init__(self) -> None:
        """Validate node data."""
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3D vector")

        # Check if orientation is quaternion or rotation matrix
        if self.orientation.shape == (4,):
            # Normalize quaternion
            self.orientation = self.orientation / np.linalg.norm(self.orientation)
        elif self.orientation.shape != (3, 3):
            raise ValueError("Orientation must be quaternion (4,) or rotation matrix (3, 3)")

    def to_gtsam_pose(self) -> gtsam.Pose3:
        """Convert to GTSAM Pose3.

        Returns:
            GTSAM Pose3 object.
        """
        # Create GTSAM Rot3 from orientation
        if self.orientation.shape == (4,):
            # Use GTSAM's quaternion constructor (w, x, y, z order)
            w, x, y, z = self.orientation
            rot = gtsam.Rot3.Quaternion(float(w), float(x), float(y), float(z))
        else:
            # Ensure rotation matrix is contiguous and float64 for GTSAM
            R = np.ascontiguousarray(self.orientation, dtype=np.float64)
            rot = gtsam.Rot3(R)

        # Create GTSAM Point3
        point = gtsam.Point3(
            float(self.position[0]), float(self.position[1]), float(self.position[2])
        )

        return gtsam.Pose3(rot, point)

    @staticmethod
    def from_gtsam_pose(
        pose: gtsam.Pose3,
        timestamp: float,
        covariance: Optional[npt.NDArray[np.float64]] = None,
    ) -> "PoseNode":
        """Create PoseNode from GTSAM Pose3.

        Args:
            pose: GTSAM Pose3 object.
            timestamp: Timestamp for the pose.
            covariance: Optional 6x6 covariance matrix.

        Returns:
            PoseNode instance.
        """
        position = pose.translation()
        position_array = np.array([position.x(), position.y(), position.z()])

        rotation = pose.rotation().matrix()

        return PoseNode(
            position=position_array,
            orientation=rotation,
            timestamp=timestamp,
            covariance=covariance,
        )

    @staticmethod
    def _quaternion_to_rotation_matrix(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert quaternion to rotation matrix.

        Args:
            q: Quaternion as [w, x, y, z].

        Returns:
            3x3 rotation matrix.
        """
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float64,
        )


def create_noise_model_diagonal(
    sigmas: npt.NDArray[np.float64],
) -> gtsam.noiseModel.Diagonal:
    """Create a diagonal noise model for GTSAM.

    Args:
        sigmas: Standard deviations for each dimension (6D for Pose3).

    Returns:
        GTSAM diagonal noise model.
    """
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def create_noise_model_gaussian(
    covariance: npt.NDArray[np.float64],
) -> gtsam.noiseModel.Gaussian:
    """Create a Gaussian noise model from covariance.

    Args:
        covariance: Covariance matrix (6x6 for Pose3).

    Returns:
        GTSAM Gaussian noise model.
    """
    return gtsam.noiseModel.Gaussian.Covariance(covariance)


def create_noise_model_isotropic(dim: int, sigma: float) -> gtsam.noiseModel.Isotropic:
    """Create an isotropic noise model.

    Args:
        dim: Dimension of the noise model.
        sigma: Standard deviation (same for all dimensions).

    Returns:
        GTSAM isotropic noise model.
    """
    return gtsam.noiseModel.Isotropic.Sigma(dim, sigma)
