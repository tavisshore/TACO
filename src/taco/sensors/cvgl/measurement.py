"""CVGL measurement representation with GTSAM support."""

from dataclasses import dataclass
from typing import Optional

import gtsam
import numpy as np
import numpy.typing as npt


@dataclass
class CVGLMeasurement:
    """Represents a CVGL image localization measurement.

    Contains the estimated global pose from image matching and the
    associated uncertainty.
    """

    timestamp: float
    position: npt.NDArray[np.float64]  # 3D position in world frame
    orientation: npt.NDArray[np.float64]  # Quaternion (w, x, y, z) or rotation matrix
    covariance: npt.NDArray[np.float64]  # 6x6 covariance (position + orientation)
    confidence: float  # Matching confidence [0, 1]
    num_inliers: int  # Number of inlier features
    image_id: Optional[str] = None  # Reference image identifier

    def __post_init__(self) -> None:
        """Validate measurement data."""
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3D vector")

        # Check orientation format
        if self.orientation.shape == (4,):
            # Normalize quaternion
            self.orientation = self.orientation / np.linalg.norm(self.orientation)
        elif self.orientation.shape != (3, 3):
            raise ValueError("Orientation must be quaternion (4,) or rotation matrix (3, 3)")

        if self.covariance.shape != (6, 6):
            raise ValueError("Covariance must be 6x6")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")

    def to_transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Convert measurement to 4x4 transformation matrix.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        T = np.eye(4)
        T[:3, 3] = self.position

        if self.orientation.shape == (4,):
            # Convert quaternion to rotation matrix
            T[:3, :3] = self._quaternion_to_rotation(self.orientation)
        else:
            T[:3, :3] = self.orientation

        return T

    def to_gtsam_pose(self) -> gtsam.Pose3:
        """Convert measurement to GTSAM Pose3.

        Returns:
            GTSAM Pose3 object.
        """
        # Convert orientation to rotation matrix if needed
        if self.orientation.shape == (4,):
            R = self._quaternion_to_rotation(self.orientation)
        else:
            R = self.orientation

        # Create GTSAM Rot3 and Point3
        rot = gtsam.Rot3(R)
        point = gtsam.Point3(self.position[0], self.position[1], self.position[2])

        return gtsam.Pose3(rot, point)

    def get_gtsam_noise_model(self) -> gtsam.noiseModel.Gaussian:
        """Get GTSAM noise model from covariance.

        Returns:
            GTSAM Gaussian noise model.
        """
        return gtsam.noiseModel.Gaussian.Covariance(self.covariance)

    @staticmethod
    def _quaternion_to_rotation(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert quaternion to rotation matrix.

        Args:
            q: Quaternion as [w, x, y, z].

        Returns:
            3x3 rotation matrix.
        """
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ])
