"""Camera model utilities for projections and undistortion."""

import numpy as np
import numpy.typing as npt


class CameraModel:
    """Pinhole camera model with optional distortion parameters.

    Provides projection and unprojection operations for 3D-2D transformations.
    """

    def __init__(self, params: dict) -> None:
        """Initialize camera model from parameters.

        Args:
            params: Dictionary containing camera parameters:
                - fx, fy: Focal lengths
                - cx, cy: Principal point
                - width, height: Image dimensions
                - k1, k2, p1, p2, k3: Optional distortion coefficients
        """
        self.fx = params.get("fx", 1.0)
        self.fy = params.get("fy", 1.0)
        self.cx = params.get("cx", 0.0)
        self.cy = params.get("cy", 0.0)
        self.width = params.get("width", 640)
        self.height = params.get("height", 480)

        # Distortion coefficients (default to no distortion)
        self.k1 = params.get("k1", 0.0)
        self.k2 = params.get("k2", 0.0)
        self.p1 = params.get("p1", 0.0)
        self.p2 = params.get("p2", 0.0)
        self.k3 = params.get("k3", 0.0)

    @property
    def intrinsic_matrix(self) -> npt.NDArray[np.float64]:
        """Get 3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

    def project(
        self, points_3d: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project 3D points to 2D image coordinates.

        Args:
            points_3d: Nx3 array of 3D points in camera frame.

        Returns:
            Nx2 array of 2D pixel coordinates.
        """
        # Normalize by depth
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]

        # Apply distortion if present
        if any([self.k1, self.k2, self.p1, self.p2, self.k3]):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
            x_distorted = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            y_distorted = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
            x, y = x_distorted, y_distorted

        # Apply intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        return np.column_stack([u, v])

    def unproject(
        self, points_2d: npt.NDArray[np.float64], depth: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Unproject 2D points to 3D given depth.

        Args:
            points_2d: Nx2 array of 2D pixel coordinates.
            depth: N array of depth values.

        Returns:
            Nx3 array of 3D points in camera frame.
        """
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy

        X = x * depth
        Y = y * depth
        Z = depth

        return np.column_stack([X, Y, Z])
