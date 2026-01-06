"""Tests for camera model module."""

import numpy as np
import pytest

from taco.data.camera_model import CameraModel


class TestCameraModel:
    """Test CameraModel class."""

    @pytest.fixture
    def default_camera(self) -> CameraModel:
        """Create a default camera model for testing."""
        params = {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
        }
        return CameraModel(params)

    @pytest.fixture
    def distorted_camera(self) -> CameraModel:
        """Create a camera model with distortion for testing."""
        params = {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
            "k1": 0.1,
            "k2": 0.01,
            "p1": 0.001,
            "p2": 0.001,
            "k3": 0.0,
        }
        return CameraModel(params)

    def test_camera_initialization(self, default_camera: CameraModel) -> None:
        """Test camera model initialization."""
        assert default_camera.fx == 500.0
        assert default_camera.fy == 500.0
        assert default_camera.cx == 320.0
        assert default_camera.cy == 240.0
        assert default_camera.width == 640
        assert default_camera.height == 480

    def test_camera_default_values(self) -> None:
        """Test camera model with minimal parameters."""
        cam = CameraModel({})
        assert cam.fx == 1.0
        assert cam.fy == 1.0
        assert cam.cx == 0.0
        assert cam.cy == 0.0
        assert cam.k1 == 0.0  # No distortion by default

    def test_intrinsic_matrix(self, default_camera: CameraModel) -> None:
        """Test intrinsic matrix property."""
        K = default_camera.intrinsic_matrix

        assert K.shape == (3, 3)
        assert K[0, 0] == 500.0  # fx
        assert K[1, 1] == 500.0  # fy
        assert K[0, 2] == 320.0  # cx
        assert K[1, 2] == 240.0  # cy
        assert K[2, 2] == 1.0

        # Off-diagonal elements should be zero
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0

    def test_project_point_at_center(self, default_camera: CameraModel) -> None:
        """Test projecting point on optical axis."""
        # Point at (0, 0, 1) should project to principal point
        point = np.array([[0.0, 0.0, 1.0]])
        projected = default_camera.project(point)

        assert projected.shape == (1, 2)
        assert np.allclose(projected[0], [320.0, 240.0])

    def test_project_point_offset(self, default_camera: CameraModel) -> None:
        """Test projecting point offset from optical axis."""
        # Point at (1, 0, 1) should project to cx + fx
        point = np.array([[1.0, 0.0, 1.0]])
        projected = default_camera.project(point)

        assert np.allclose(projected[0, 0], 320.0 + 500.0)  # cx + fx * (x/z)
        assert np.allclose(projected[0, 1], 240.0)  # cy

    def test_project_multiple_points(self, default_camera: CameraModel) -> None:
        """Test projecting multiple points at once."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ])
        projected = default_camera.project(points)

        assert projected.shape == (4, 2)

    def test_project_with_depth(self, default_camera: CameraModel) -> None:
        """Test projecting points at different depths."""
        # Points at different depths but same direction should project to same pixel
        point_near = np.array([[0.5, 0.5, 1.0]])
        point_far = np.array([[1.0, 1.0, 2.0]])

        proj_near = default_camera.project(point_near)
        proj_far = default_camera.project(point_far)

        assert np.allclose(proj_near, proj_far)

    def test_unproject_point_at_center(self, default_camera: CameraModel) -> None:
        """Test unprojecting principal point."""
        pixel = np.array([[320.0, 240.0]])
        depth = np.array([1.0])

        unprojected = default_camera.unproject(pixel, depth)

        assert unprojected.shape == (1, 3)
        assert np.allclose(unprojected[0], [0.0, 0.0, 1.0])

    def test_unproject_point_offset(self, default_camera: CameraModel) -> None:
        """Test unprojecting point offset from principal point."""
        pixel = np.array([[820.0, 240.0]])  # cx + fx = 320 + 500
        depth = np.array([1.0])

        unprojected = default_camera.unproject(pixel, depth)

        assert np.allclose(unprojected[0, 0], 1.0, atol=1e-10)  # x = (u - cx) / fx * z
        assert np.allclose(unprojected[0, 1], 0.0, atol=1e-10)
        assert np.allclose(unprojected[0, 2], 1.0, atol=1e-10)

    def test_project_unproject_roundtrip(self, default_camera: CameraModel) -> None:
        """Test project -> unproject roundtrip."""
        original_points = np.array([
            [0.5, 0.3, 2.0],
            [-0.2, 0.4, 1.5],
            [1.0, -0.5, 3.0],
        ])

        # Project points
        projected = default_camera.project(original_points)

        # Unproject with original depths
        depths = original_points[:, 2]
        recovered = default_camera.unproject(projected, depths)

        assert np.allclose(original_points, recovered, atol=1e-10)

    def test_project_with_distortion(self, distorted_camera: CameraModel) -> None:
        """Test that distortion affects projection."""
        point = np.array([[0.5, 0.5, 1.0]])

        # Project with undistorted camera
        undistorted_cam = CameraModel({
            "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
        })
        proj_undistorted = undistorted_cam.project(point)

        # Project with distorted camera
        proj_distorted = distorted_camera.project(point)

        # Results should be different
        assert not np.allclose(proj_undistorted, proj_distorted)

    def test_unproject_multiple_points(self, default_camera: CameraModel) -> None:
        """Test unprojecting multiple points."""
        pixels = np.array([
            [320.0, 240.0],
            [420.0, 240.0],
            [320.0, 340.0],
        ])
        depths = np.array([1.0, 2.0, 3.0])

        unprojected = default_camera.unproject(pixels, depths)

        assert unprojected.shape == (3, 3)
        assert unprojected[0, 2] == 1.0
        assert unprojected[1, 2] == 2.0
        assert unprojected[2, 2] == 3.0
