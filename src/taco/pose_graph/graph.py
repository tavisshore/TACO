"""Pose graph implementation using GTSAM."""

from typing import Dict, List, Optional

import gtsam
import numpy as np
import numpy.typing as npt


class PoseGraph:
    """Main pose graph class for sensor fusion using GTSAM.

    Combines IMU measurements and CVGL image localization data into
    a unified factor graph for optimization.
    """

    def __init__(self) -> None:
        """Initialize an empty pose graph."""
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.current_estimates = gtsam.Values()
        self._current_pose_id = 0
        self._pose_timestamps: Dict[int, float] = {}

    def add_pose_estimate(
        self,
        pose: gtsam.Pose3,
        timestamp: float,
        pose_id: Optional[int] = None,
    ) -> int:
        """Add a pose estimate to the graph.

        Args:
            pose: The 6-DOF pose (GTSAM Pose3).
            timestamp: Timestamp of the pose.
            pose_id: Optional pose identifier. If None, auto-increments.

        Returns:
            The assigned pose ID.
        """
        if pose_id is None:
            pose_id = self._current_pose_id
            self._current_pose_id += 1

        symbol = gtsam.symbol("x", pose_id)
        self.initial_estimates.insert(symbol, pose)
        self._pose_timestamps[pose_id] = timestamp

        return pose_id

    def add_prior_factor(
        self,
        pose_id: int,
        pose: gtsam.Pose3,
        noise_model: gtsam.noiseModel.Base,
    ) -> None:
        """Add a prior factor to fix a pose.

        Args:
            pose_id: The pose identifier.
            pose: The prior pose value.
            noise_model: Noise model for the prior.
        """
        symbol = gtsam.symbol("x", pose_id)
        prior_factor = gtsam.PriorFactorPose3(symbol, pose, noise_model)
        self.graph.add(prior_factor)

    def add_between_factor(
        self,
        from_pose_id: int,
        to_pose_id: int,
        relative_pose: gtsam.Pose3,
        noise_model: gtsam.noiseModel.Base,
    ) -> None:
        """Add a between factor (odometry constraint).

        Args:
            from_pose_id: Source pose ID.
            to_pose_id: Target pose ID.
            relative_pose: Relative transformation from source to target.
            noise_model: Noise model for the measurement.
        """
        from_symbol = gtsam.symbol("x", from_pose_id)
        to_symbol = gtsam.symbol("x", to_pose_id)

        between_factor = gtsam.BetweenFactorPose3(
            from_symbol, to_symbol, relative_pose, noise_model
        )
        self.graph.add(between_factor)

    def add_gps_factor(
        self,
        pose_id: int,
        position: npt.NDArray[np.float64],
        noise_model: gtsam.noiseModel.Base,
    ) -> None:
        """Add a GPS/absolute position factor.

        Args:
            pose_id: The pose identifier.
            position: 3D position measurement.
            noise_model: Noise model for the measurement.
        """
        symbol = gtsam.symbol("x", pose_id)
        point = gtsam.Point3(position[0], position[1], position[2])

        # Use GPSFactor for absolute position measurements
        gps_factor = gtsam.GPSFactor(symbol, point, noise_model)
        self.graph.add(gps_factor)

    def add_pose_factor(
        self,
        pose_id: int,
        pose: gtsam.Pose3,
        noise_model: gtsam.noiseModel.Base,
    ) -> None:
        """Add a full 6-DOF pose factor (e.g., from CVGL).

        Args:
            pose_id: The pose identifier.
            pose: The measured pose.
            noise_model: Noise model for the measurement.
        """
        symbol = gtsam.symbol("x", pose_id)
        pose_factor = gtsam.PriorFactorPose3(symbol, pose, noise_model)
        self.graph.add(pose_factor)

    def optimize(
        self,
        optimizer_type: str = "LevenbergMarquardt",
        max_iterations: int = 100,
    ) -> gtsam.Values:
        """Optimize the pose graph.

        Args:
            optimizer_type: Type of optimizer ("LevenbergMarquardt" or "GaussNewton").
            max_iterations: Maximum number of optimization iterations.

        Returns:
            Optimized values.
        """
        if optimizer_type == "LevenbergMarquardt":
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(max_iterations)
            params.setRelativeErrorTol(1e-5)
            params.setAbsoluteErrorTol(1e-5)
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.initial_estimates, params
            )
        elif optimizer_type == "GaussNewton":
            params = gtsam.GaussNewtonParams()
            params.setMaxIterations(max_iterations)
            params.setRelativeErrorTol(1e-5)
            params.setAbsoluteErrorTol(1e-5)
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimates, params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self.current_estimates = optimizer.optimize()
        return self.current_estimates

    def get_pose(self, pose_id: int) -> Optional[gtsam.Pose3]:
        """Get an optimized pose by ID.

        Args:
            pose_id: The pose identifier.

        Returns:
            The optimized Pose3 if available, None otherwise.
        """
        symbol = gtsam.symbol("x", pose_id)
        if self.current_estimates.exists(symbol):
            return self.current_estimates.atPose3(symbol)
        elif self.initial_estimates.exists(symbol):
            return self.initial_estimates.atPose3(symbol)
        return None

    def get_all_poses(self) -> Dict[int, gtsam.Pose3]:
        """Get all optimized poses.

        Returns:
            Dictionary mapping pose IDs to Pose3 objects.
        """
        poses = {}
        estimates = (
            self.current_estimates
            if self.current_estimates.size() > 0
            else self.initial_estimates
        )

        for pose_id in range(self._current_pose_id):
            symbol = gtsam.symbol("x", pose_id)
            if estimates.exists(symbol):
                poses[pose_id] = estimates.atPose3(symbol)

        return poses

    def get_marginal_covariance(self, pose_id: int) -> npt.NDArray[np.float64]:
        """Get marginal covariance for a pose.

        Args:
            pose_id: The pose identifier.

        Returns:
            6x6 covariance matrix.
        """
        symbol = gtsam.symbol("x", pose_id)
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        return marginals.marginalCovariance(symbol)

    def get_error(self) -> float:
        """Get the current graph error.

        Returns:
            Total graph error.
        """
        estimates = (
            self.current_estimates
            if self.current_estimates.size() > 0
            else self.initial_estimates
        )
        return self.graph.error(estimates)

    def size(self) -> int:
        """Get the number of factors in the graph.

        Returns:
            Number of factors.
        """
        return self.graph.size()
