"""Pose graph edge representation for GTSAM.

This module provides helper classes for creating GTSAM factors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import gtsam
import numpy as np
import numpy.typing as npt


class EdgeType(Enum):
    """Type of edge constraint."""

    IMU = "imu"  # IMU odometry constraint
    CVGL = "cvgl"  # CVGL localization constraint
    LOOP_CLOSURE = "loop_closure"  # Loop closure constraint
    PRIOR = "prior"  # Prior constraint


@dataclass
class Edge:
    """Represents a constraint between two poses for GTSAM integration.

    This is a convenience wrapper that can be converted to GTSAM factors.
    """

    from_node_id: int
    to_node_id: int
    relative_transform: npt.NDArray[np.float64]  # 4x4 transformation matrix
    information_matrix: npt.NDArray[np.float64]  # 6x6 information matrix
    edge_type: EdgeType
    measurement_timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate edge data."""
        if self.relative_transform.shape != (4, 4):
            raise ValueError("Relative transform must be a 4x4 matrix")
        if self.information_matrix.shape != (6, 6):
            raise ValueError("Information matrix must be 6x6")

    def to_gtsam_between_factor(self) -> gtsam.BetweenFactorPose3:
        """Convert to GTSAM BetweenFactorPose3.

        Returns:
            GTSAM BetweenFactorPose3.
        """
        # Extract rotation and translation from 4x4 matrix
        R = self.relative_transform[:3, :3]
        t = self.relative_transform[:3, 3]

        # Create GTSAM Pose3
        rot = gtsam.Rot3(R)
        point = gtsam.Point3(t[0], t[1], t[2])
        relative_pose = gtsam.Pose3(rot, point)

        # Create noise model from information matrix
        # Information matrix is inverse of covariance
        covariance = np.linalg.inv(self.information_matrix)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)

        # Create symbols
        from_symbol = gtsam.symbol("x", self.from_node_id)
        to_symbol = gtsam.symbol("x", self.to_node_id)

        return gtsam.BetweenFactorPose3(from_symbol, to_symbol, relative_pose, noise_model)

    @staticmethod
    def from_poses(
        from_node_id: int,
        to_node_id: int,
        from_pose: gtsam.Pose3,
        to_pose: gtsam.Pose3,
        noise_sigmas: npt.NDArray[np.float64],
        edge_type: EdgeType = EdgeType.IMU,
    ) -> "Edge":
        """Create an Edge from two GTSAM poses.

        Args:
            from_node_id: Source node ID.
            to_node_id: Target node ID.
            from_pose: Source pose.
            to_pose: Target pose.
            noise_sigmas: Standard deviations for the 6 DOF.
            edge_type: Type of edge.

        Returns:
            Edge instance.
        """
        # Compute relative transformation
        relative_pose = from_pose.between(to_pose)

        # Convert to 4x4 matrix
        R = relative_pose.rotation().matrix()
        t = relative_pose.translation()
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = [t.x(), t.y(), t.z()]

        # Create information matrix from sigmas
        information = np.diag(1.0 / (noise_sigmas**2))

        return Edge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relative_transform=transform,
            information_matrix=information,
            edge_type=edge_type,
        )


def create_between_factor(
    from_pose_id: int,
    to_pose_id: int,
    relative_pose: gtsam.Pose3,
    noise_model: gtsam.noiseModel.Base,
) -> gtsam.BetweenFactorPose3:
    """Create a GTSAM between factor.

    Args:
        from_pose_id: Source pose ID.
        to_pose_id: Target pose ID.
        relative_pose: Relative transformation.
        noise_model: Noise model.

    Returns:
        GTSAM BetweenFactorPose3.
    """
    from_symbol = gtsam.symbol("x", from_pose_id)
    to_symbol = gtsam.symbol("x", to_pose_id)

    return gtsam.BetweenFactorPose3(from_symbol, to_symbol, relative_pose, noise_model)


def create_prior_factor(
    pose_id: int,
    prior_pose: gtsam.Pose3,
    noise_model: gtsam.noiseModel.Base,
) -> gtsam.PriorFactorPose3:
    """Create a GTSAM prior factor.

    Args:
        pose_id: Pose ID to constrain.
        prior_pose: Prior pose value.
        noise_model: Noise model.

    Returns:
        GTSAM PriorFactorPose3.
    """
    symbol = gtsam.symbol("x", pose_id)
    return gtsam.PriorFactorPose3(symbol, prior_pose, noise_model)
