"""Coordinate transformation utilities with GTSAM support."""

import gtsam
import numpy as np
import numpy.typing as npt


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


def transform_to_pose(
    T: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract position and orientation from transformation matrix.

    Args:
        T: 4x4 transformation matrix.

    Returns:
        Tuple of (position, quaternion).
    """
    position = T[:3, 3]
    rotation = T[:3, :3]
    quaternion = rotation_matrix_to_quaternion(rotation)

    return position, quaternion


def pose_to_transform(
    position: npt.NDArray[np.float64],
    quaternion: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Create transformation matrix from position and orientation.

    Args:
        position: 3D position vector.
        quaternion: Orientation as quaternion [w, x, y, z].

    Returns:
        4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = quaternion_to_rotation_matrix(quaternion)

    return T


def gtsam_pose_to_transform(pose: gtsam.Pose3) -> npt.NDArray[np.float64]:
    """Convert GTSAM Pose3 to 4x4 transformation matrix.

    Args:
        pose: GTSAM Pose3 object.

    Returns:
        4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T


def transform_to_gtsam_pose(T: npt.NDArray[np.float64]) -> gtsam.Pose3:
    """Convert 4x4 transformation matrix to GTSAM Pose3.

    Args:
        T: 4x4 transformation matrix.

    Returns:
        GTSAM Pose3 object.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    rot = gtsam.Rot3(R)
    point = gtsam.Point3(t[0], t[1], t[2])

    return gtsam.Pose3(rot, point)


def numpy_pose_to_gtsam(
    position: npt.NDArray[np.float64],
    orientation: npt.NDArray[np.float64],
) -> gtsam.Pose3:
    """Convert numpy arrays to GTSAM Pose3.

    Args:
        position: 3D position vector.
        orientation: Quaternion [w, x, y, z] or 3x3 rotation matrix.

    Returns:
        GTSAM Pose3 object.
    """
    # Convert orientation to rotation matrix if needed
    if orientation.shape == (4,):
        R = quaternion_to_rotation_matrix(orientation)
    else:
        R = orientation

    rot = gtsam.Rot3(R)
    point = gtsam.Point3(position[0], position[1], position[2])

    return gtsam.Pose3(rot, point)


def gtsam_pose_to_numpy(
    pose: gtsam.Pose3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert GTSAM Pose3 to numpy arrays.

    Args:
        pose: GTSAM Pose3 object.

    Returns:
        Tuple of (position, rotation_matrix).
    """
    position_array = np.asarray(pose.translation())
    rotation_matrix = pose.rotation().matrix()

    return position_array, rotation_matrix
