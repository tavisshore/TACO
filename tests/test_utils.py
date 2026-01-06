"""Tests for utility functions."""

import numpy as np

from taco.utils.conversions import (
    pose_to_transform,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_to_pose,
)


class TestConversions:
    """Test coordinate conversion utilities."""

    def test_quaternion_to_rotation_identity(self) -> None:
        """Test identity quaternion to rotation matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)

        assert np.allclose(R, np.eye(3))

    def test_rotation_to_quaternion_identity(self) -> None:
        """Test identity rotation matrix to quaternion."""
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)

        # Should be close to [1, 0, 0, 0] or [-1, 0, 0, 0]
        assert np.allclose(np.abs(q[0]), 1.0)
        assert np.allclose(q[1:], [0.0, 0.0, 0.0])

    def test_quaternion_rotation_roundtrip(self) -> None:
        """Test quaternion -> rotation -> quaternion roundtrip."""
        q_original = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90 deg around x-axis
        R = quaternion_to_rotation_matrix(q_original)
        q_recovered = rotation_matrix_to_quaternion(R)

        # Quaternions q and -q represent the same rotation
        assert np.allclose(q_original, q_recovered) or np.allclose(q_original, -q_recovered)

    def test_pose_to_transform(self) -> None:
        """Test pose to transformation matrix."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        T = pose_to_transform(position, quaternion)

        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], position)
        assert np.allclose(T[:3, :3], np.eye(3))
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_transform_to_pose(self) -> None:
        """Test transformation matrix to pose."""
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]

        position, quaternion = transform_to_pose(T)

        assert np.allclose(position, [1.0, 2.0, 3.0])
        assert quaternion.shape == (4,)

    def test_pose_transform_roundtrip(self) -> None:
        """Test pose -> transform -> pose roundtrip."""
        pos_original = np.array([1.0, 2.0, 3.0])
        quat_original = np.array([0.7071, 0.7071, 0.0, 0.0])

        T = pose_to_transform(pos_original, quat_original)
        pos_recovered, quat_recovered = transform_to_pose(T)

        assert np.allclose(pos_original, pos_recovered)
        # Quaternions q and -q represent the same rotation
        assert np.allclose(quat_original, quat_recovered) or np.allclose(
            quat_original, -quat_recovered
        )
