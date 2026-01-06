"""Pose graph optimization using GTSAM."""

from typing import List, Optional

import gtsam

from .graph import PoseGraph


class GraphOptimizer:
    """Optimizes pose graphs using GTSAM backend solvers.

    Supports Levenberg-Marquardt and Gauss-Newton optimization.
    """

    def __init__(
        self,
        method: str = "LevenbergMarquardt",
        max_iterations: int = 100,
        relative_error_tol: float = 1e-5,
        absolute_error_tol: float = 1e-5,
    ) -> None:
        """Initialize optimizer.

        Args:
            method: Optimization method ("LevenbergMarquardt" or "GaussNewton").
            max_iterations: Maximum number of iterations.
            relative_error_tol: Relative error tolerance for convergence.
            absolute_error_tol: Absolute error tolerance for convergence.
        """
        self.method = method
        self.max_iterations = max_iterations
        self.relative_error_tol = relative_error_tol
        self.absolute_error_tol = absolute_error_tol

    def optimize(
        self,
        graph: PoseGraph,
        fixed_poses: List[int] | None = None,
    ) -> gtsam.Values:
        """Optimize the pose graph.

        Args:
            graph: The pose graph to optimize.
            fixed_poses: List of pose IDs to keep fixed (not used directly,
                        but can be implemented with strong priors).

        Returns:
            The optimized values.
        """
        return graph.optimize(
            optimizer_type=self.method,
            max_iterations=self.max_iterations,
        )

    def optimize_incremental(
        self,
        graph: PoseGraph,
    ) -> gtsam.Values:
        """Perform incremental optimization using iSAM2.

        Args:
            graph: The pose graph to optimize.

        Returns:
            The optimized values.
        """
        # Configure iSAM2
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)

        isam = gtsam.ISAM2(parameters)

        # Add all factors and initial estimates
        isam.update(graph.graph, graph.initial_estimates)

        # Get result
        result = isam.calculateEstimate()
        graph.current_estimates = result

        return result
