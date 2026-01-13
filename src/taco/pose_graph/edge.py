"""Pose graph edge representation for GTSAM Pose2.

This module provides helper classes for creating GTSAM factors
for 2D vehicle trajectory optimization.
"""

from dataclasses import dataclass
from enum import Enum

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
    """Represents a constraint between two 2D poses for GTSAM integration.

    This is a convenience wrapper that can be converted to GTSAM factors.
    Uses 3x3 SE(2) transformation matrices for relative poses.
    """

    from_node_id: int
    to_node_id: int
    relative_transform: npt.NDArray[np.float64]  # 3x3 SE(2) transformation matrix
    information_matrix: npt.NDArray[np.float64]  # 3x3 information matrix
    edge_type: EdgeType
    measurement_timestamp: float | None = None

    def __post_init__(self) -> None:
        """Validate edge data."""
        if self.relative_transform.shape != (3, 3):
            raise ValueError("Relative transform must be a 3x3 SE(2) matrix")
        if self.information_matrix.shape != (3, 3):
            raise ValueError("Information matrix must be 3x3 for Pose2")

    def to_gtsam_between_factor(self) -> gtsam.BetweenFactorPose2:
        """Convert to GTSAM BetweenFactorPose2.

        Returns:
            GTSAM BetweenFactorPose2.
        """
        # Extract x, y, theta from 3x3 SE(2) matrix
        x = float(self.relative_transform[0, 2])
        y = float(self.relative_transform[1, 2])
        theta = float(np.arctan2(self.relative_transform[1, 0], self.relative_transform[0, 0]))

        relative_pose = gtsam.Pose2(x, y, theta)

        # Create noise model from information matrix
        covariance = np.linalg.inv(self.information_matrix)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)

        # Create symbols
        from_symbol = gtsam.symbol("x", self.from_node_id)
        to_symbol = gtsam.symbol("x", self.to_node_id)

        return gtsam.BetweenFactorPose2(from_symbol, to_symbol, relative_pose, noise_model)

    @staticmethod
    def from_poses(
        from_node_id: int,
        to_node_id: int,
        from_pose: gtsam.Pose2,
        to_pose: gtsam.Pose2,
        noise_sigmas: npt.NDArray[np.float64],
        edge_type: EdgeType = EdgeType.IMU,
    ) -> "Edge":
        """Create an Edge from two GTSAM Pose2 objects.

        Args:
            from_node_id: Source node ID.
            to_node_id: Target node ID.
            from_pose: Source pose.
            to_pose: Target pose.
            noise_sigmas: Standard deviations for the 3 DOF (x, y, theta).
            edge_type: Type of edge.

        Returns:
            Edge instance.
        """
        # Compute relative transformation
        relative_pose = from_pose.between(to_pose)

        # Convert to 3x3 SE(2) matrix
        c, s = np.cos(relative_pose.theta()), np.sin(relative_pose.theta())
        transform = np.array(
            [
                [c, -s, relative_pose.x()],
                [s, c, relative_pose.y()],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

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
    relative_pose: gtsam.Pose2,
    noise_model: gtsam.noiseModel.Base,
) -> gtsam.BetweenFactorPose2:
    """Create a GTSAM between factor for Pose2.

    Args:
        from_pose_id: Source pose ID.
        to_pose_id: Target pose ID.
        relative_pose: Relative transformation.
        noise_model: Noise model.

    Returns:
        GTSAM BetweenFactorPose2.
    """
    from_symbol = gtsam.symbol("x", from_pose_id)
    to_symbol = gtsam.symbol("x", to_pose_id)

    return gtsam.BetweenFactorPose2(from_symbol, to_symbol, relative_pose, noise_model)


def create_prior_factor(
    pose_id: int,
    prior_pose: gtsam.Pose2,
    noise_model: gtsam.noiseModel.Base,
) -> gtsam.PriorFactorPose2:
    """Create a GTSAM prior factor for Pose2.

    Args:
        pose_id: Pose ID to constrain.
        prior_pose: Prior pose value.
        noise_model: Noise model.

    Returns:
        GTSAM PriorFactorPose2.
    """
    symbol = gtsam.symbol("x", pose_id)
    return gtsam.PriorFactorPose2(symbol, prior_pose, noise_model)
