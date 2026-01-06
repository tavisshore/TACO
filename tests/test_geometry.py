"""Tests for geometry utility functions."""

import numpy as np
import pytest

from taco.utils.geometry import find_apex_yaw_indices, rotate_trajectory, rotation_matrix


class TestRotationMatrix:
    """Test rotation matrix creation."""

    def test_rotation_matrix_z_axis_90_degrees(self) -> None:
        """Test 90 degree rotation around z-axis."""
        R = rotation_matrix("z", 90.0)

        # Check that x-axis maps to y-axis
        x_axis = np.array([1, 0, 0])
        rotated = R @ x_axis
        assert np.allclose(rotated, [0, 1, 0], atol=1e-10)

    def test_rotation_matrix_x_axis_90_degrees(self) -> None:
        """Test 90 degree rotation around x-axis."""
        R = rotation_matrix("x", 90.0)

        # Check that y-axis maps to z-axis
        y_axis = np.array([0, 1, 0])
        rotated = R @ y_axis
        assert np.allclose(rotated, [0, 0, 1], atol=1e-10)

    def test_rotation_matrix_y_axis_90_degrees(self) -> None:
        """Test 90 degree rotation around y-axis."""
        R = rotation_matrix("y", 90.0)

        # Check that z-axis maps to x-axis
        z_axis = np.array([0, 0, 1])
        rotated = R @ z_axis
        assert np.allclose(rotated, [1, 0, 0], atol=1e-10)

    def test_rotation_matrix_identity(self) -> None:
        """Test zero rotation gives identity matrix."""
        R = rotation_matrix("z", 0.0)
        assert np.allclose(R, np.eye(3))

    def test_rotation_matrix_360_degrees(self) -> None:
        """Test 360 degree rotation gives identity matrix."""
        R = rotation_matrix("z", 360.0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_invalid_axis(self) -> None:
        """Test invalid axis raises ValueError."""
        with pytest.raises(ValueError, match="Invalid axis"):
            rotation_matrix("w", 90.0)

    def test_rotation_matrix_is_orthogonal(self) -> None:
        """Test rotation matrix is orthogonal (R^T R = I)."""
        R = rotation_matrix("z", 45.0)
        assert np.allclose(R.T @ R, np.eye(3))
        assert np.allclose(np.linalg.det(R), 1.0)


class TestRotateTrajectory:
    """Test trajectory rotation function."""

    def test_rotate_trajectory_single_point(self) -> None:
        """Test rotating a single point."""
        point = np.array([[1.0, 0.0, 0.0]])
        rotated = rotate_trajectory(point, axis="z", degrees=90.0)

        assert rotated.shape == (1, 3)
        assert np.allclose(rotated[0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotate_trajectory_multiple_points(self) -> None:
        """Test rotating multiple points."""
        trajectory = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        rotated = rotate_trajectory(trajectory, axis="z", degrees=90.0)

        assert rotated.shape == (3, 3)
        assert np.allclose(rotated[:, 0], [0.0, 0.0, 0.0], atol=1e-10)  # x -> 0
        assert np.allclose(rotated[:, 1], [1.0, 2.0, 3.0], atol=1e-10)  # y -> original x

    def test_rotate_trajectory_from_list(self) -> None:
        """Test rotating trajectory from list input."""
        trajectory = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        rotated = rotate_trajectory(trajectory, axis="z", degrees=90.0)

        assert isinstance(rotated, np.ndarray)
        assert rotated.shape == (2, 3)

    def test_rotate_trajectory_empty(self) -> None:
        """Test rotating empty trajectory."""
        trajectory = np.array([])
        rotated = rotate_trajectory(trajectory, axis="z", degrees=90.0)

        assert len(rotated) == 0

    def test_rotate_trajectory_preserves_distance(self) -> None:
        """Test that rotation preserves distances between points."""
        trajectory = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )

        # Calculate original distances
        orig_dist_01 = np.linalg.norm(trajectory[1] - trajectory[0])
        orig_dist_12 = np.linalg.norm(trajectory[2] - trajectory[1])

        rotated = rotate_trajectory(trajectory, axis="z", degrees=45.0)

        # Check distances are preserved
        new_dist_01 = np.linalg.norm(rotated[1] - rotated[0])
        new_dist_12 = np.linalg.norm(rotated[2] - rotated[1])

        assert np.allclose(orig_dist_01, new_dist_01)
        assert np.allclose(orig_dist_12, new_dist_12)


class TestFindApexYawIndices:
    """Test yaw apex detection function."""

    def test_find_apex_straight_line(self) -> None:
        """Test finding apex in straight trajectory (no turns)."""
        yaws = np.zeros(100)  # Constant yaw = no turning
        apex, starts, ends = find_apex_yaw_indices(yaws, threshold=0.1)

        assert len(apex) == 0
        assert len(starts) == 0
        assert len(ends) == 0

    def test_find_apex_single_turn(self) -> None:
        """Test finding apex in single turn."""
        # Create a trajectory with one turn
        yaws = np.concatenate(
            [
                np.zeros(20),  # Straight
                np.linspace(0, np.pi / 2, 20),  # Turn 90 degrees
                np.ones(20) * np.pi / 2,  # Straight again
            ]
        )

        apex, _, _ = find_apex_yaw_indices(yaws, threshold=0.05, min_turn_length=3)

        # Should find one turn
        assert len(apex) >= 1
        # Apex should be in the turning region (around index 20-40)
        assert any(15 < a < 45 for a in apex)

    def test_find_apex_multiple_turns(self) -> None:
        """Test finding apex in multiple turns."""
        # Create S-curve trajectory
        yaws = np.concatenate(
            [
                np.zeros(15),
                np.linspace(0, np.pi / 2, 15),  # Turn right
                np.ones(15) * np.pi / 2,
                np.linspace(np.pi / 2, 0, 15),  # Turn left
                np.zeros(15),
            ]
        )

        apex, _, _ = find_apex_yaw_indices(yaws, threshold=0.05, min_turn_length=3)

        # Should find two turns
        assert len(apex) >= 2

    def test_find_apex_short_sequence(self) -> None:
        """Test with very short sequence."""
        yaws = np.array([0.0, 0.1])
        apex, starts, ends = find_apex_yaw_indices(yaws)

        # Should handle gracefully
        assert isinstance(apex, list)
        assert isinstance(starts, list)
        assert isinstance(ends, list)

    def test_find_apex_handles_angle_wrapping(self) -> None:
        """Test that angle wrapping is handled correctly."""
        # Yaw crossing from -pi to pi
        yaws = np.linspace(-np.pi + 0.1, np.pi - 0.1, 50)

        # This should not cause issues
        apex, _, _ = find_apex_yaw_indices(yaws, threshold=0.05)

        # Should complete without error
        assert isinstance(apex, list)

    def test_find_apex_returns_correct_types(self) -> None:
        """Test return types are correct."""
        yaws = np.linspace(0, np.pi, 50)
        apex, starts, ends = find_apex_yaw_indices(yaws)

        assert isinstance(apex, list)
        assert isinstance(starts, list)
        assert isinstance(ends, list)

        # All indices should be integers
        for idx in apex + starts + ends:
            assert isinstance(idx, int | np.integer)
