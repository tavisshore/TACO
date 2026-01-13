"""Coordinate transformation utilities with GTSAM Pose2 support.

This module provides utilities for working with 2D poses (x, y, yaw) using GTSAM Pose2.
"""

import gtsam
import numpy as np
import numpy.typing as npt


def yaw_to_rotation_matrix(yaw: float) -> npt.NDArray[np.float64]:
    """Convert yaw angle to 2D rotation matrix.

    Args:
        yaw: Yaw angle in radians.

    Returns:
        2x2 rotation matrix.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotation_matrix_to_yaw(R: npt.NDArray[np.float64]) -> float:
    """Extract yaw angle from rotation matrix.

    Args:
        R: 2x2 or 3x3 rotation matrix.

    Returns:
        Yaw angle in radians.
    """
    if R.shape in ((2, 2), (3, 3)):
        return float(np.arctan2(R[1, 0], R[0, 0]))
    raise ValueError("Rotation matrix must be 2x2 or 3x3")


def quaternion_to_yaw(q: npt.NDArray[np.float64]) -> float:
    """Extract yaw angle from quaternion.

    Args:
        q: Quaternion as [w, x, y, z].

    Returns:
        Yaw angle in radians.
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    # Yaw (rotation around z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_to_quaternion(yaw: float) -> npt.NDArray[np.float64]:
    """Convert yaw angle to quaternion (rotation around z-axis only).

    Args:
        yaw: Yaw angle in radians.

    Returns:
        Quaternion as [w, x, y, z].
    """
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)


def quaternion_to_rotation_matrix(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion to rotation matrix.

    Args:
        q: Quaternion as [w, x, y, z].

    Returns:
        3x3 rotation matrix.
    """
    q = q / np.linalg.norm(q)  # Normalize
    w, x, y, z = q

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def rotation_matrix_to_quaternion(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Quaternion as [w, x, y, z].
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def transform_to_pose2d(
    T: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    """Extract 2D position and yaw from transformation matrix.

    Args:
        T: 3x3 SE(2) transformation matrix.

    Returns:
        Tuple of (position [x, y], yaw).
    """
    position = T[:2, 2]
    yaw = rotation_matrix_to_yaw(T[:2, :2])
    return position, yaw


def pose2d_to_transform(
    position: npt.NDArray[np.float64],
    yaw: float,
) -> npt.NDArray[np.float64]:
    """Create 3x3 SE(2) transformation matrix from 2D position and yaw.

    Args:
        position: 2D position vector [x, y].
        yaw: Yaw angle in radians.

    Returns:
        3x3 SE(2) transformation matrix.
    """
    T = np.eye(3)
    T[:2, :2] = yaw_to_rotation_matrix(yaw)
    T[:2, 2] = position[:2]
    return T


def gtsam_pose2_to_transform(pose: gtsam.Pose2) -> npt.NDArray[np.float64]:
    """Convert GTSAM Pose2 to 3x3 SE(2) transformation matrix.

    Args:
        pose: GTSAM Pose2 object.

    Returns:
        3x3 SE(2) transformation matrix.
    """
    T = np.eye(3)
    T[:2, :2] = yaw_to_rotation_matrix(pose.theta())
    T[0, 2] = pose.x()
    T[1, 2] = pose.y()
    return T


def transform_to_gtsam_pose2(T: npt.NDArray[np.float64]) -> gtsam.Pose2:
    """Convert 3x3 SE(2) transformation matrix to GTSAM Pose2.

    Args:
        T: 3x3 SE(2) transformation matrix.

    Returns:
        GTSAM Pose2 object.
    """
    x = float(T[0, 2])
    y = float(T[1, 2])
    yaw = rotation_matrix_to_yaw(T[:2, :2])
    return gtsam.Pose2(x, y, yaw)


def numpy_pose_to_gtsam(
    position: npt.NDArray[np.float64],
    yaw: float,
) -> gtsam.Pose2:
    """Convert numpy position and yaw to GTSAM Pose2.

    Args:
        position: 2D position vector [x, y].
        yaw: Yaw angle in radians.

    Returns:
        GTSAM Pose2 object.
    """
    return gtsam.Pose2(float(position[0]), float(position[1]), float(yaw))


def gtsam_pose2_to_numpy(
    pose: gtsam.Pose2,
) -> tuple[npt.NDArray[np.float64], float]:
    """Convert GTSAM Pose2 to numpy arrays.

    Args:
        pose: GTSAM Pose2 object.

    Returns:
        Tuple of (position [x, y], yaw).
    """
    position = np.array([pose.x(), pose.y()], dtype=np.float64)
    return position, pose.theta()
