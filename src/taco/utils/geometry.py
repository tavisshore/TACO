"""Geometry utilities for trajectory transformations."""

from typing import List, Union

import numpy as np
import numpy.typing as npt


def rotate_trajectory(
    trajectory: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    axis: str = "z",
    degrees: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Rotate a trajectory around a specified axis.

    Args:
        trajectory: List of 3D points or Nx3 array representing the trajectory.
        axis: Axis to rotate around ('x', 'y', or 'z').
        degrees: Rotation angle in degrees.

    Returns:
        Rotated trajectory as Nx3 array.
    """
    # Convert to numpy array if needed
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)

    # Handle empty trajectory
    if len(trajectory) == 0:
        return trajectory

    # Ensure 2D array
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)

    # Get rotation matrix
    R = rotation_matrix(axis, degrees)

    # Apply rotation to each point
    rotated = (R @ trajectory.T).T

    return rotated


def rotation_matrix(axis: str, degrees: float) -> npt.NDArray[np.float64]:
    """Create a 3D rotation matrix for rotation around a specified axis.

    Args:
        axis: Axis to rotate around ('x', 'y', or 'z').
        degrees: Rotation angle in degrees.

    Returns:
        3x3 rotation matrix.
    """
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)

    if axis.lower() == "x":
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ])
    if axis.lower() == "y":
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])
    if axis.lower() == "z":
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ])
    raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")


def find_apex_yaw_indices(
    yaws: npt.NDArray[np.float64],
    threshold: float = 0.3,
    min_turn_length: int = 5,
) -> tuple:
    """Find indices where significant yaw changes (turns) occur.

    Identifies turn apex points (maximum yaw change), as well as
    turn start and end indices.

    Args:
        yaws: Array of yaw angles in radians.
        threshold: Minimum yaw rate to consider as turning.
        min_turn_length: Minimum number of frames for a valid turn.

    Returns:
        Tuple of (apex_indices, start_indices, end_indices).
    """
    if len(yaws) < 3:
        return [], [], []

    # Calculate yaw rate (derivative)
    yaw_rate = np.diff(yaws)

    # Handle angle wrapping
    yaw_rate = np.arctan2(np.sin(yaw_rate), np.cos(yaw_rate))

    # Find regions with significant yaw rate
    turning = np.abs(yaw_rate) > threshold

    # Find turn boundaries
    turn_changes = np.diff(turning.astype(int))
    turn_starts = np.where(turn_changes == 1)[0] + 1
    turn_ends = np.where(turn_changes == -1)[0] + 1

    # Handle edge cases
    if turning[0]:
        turn_starts = np.insert(turn_starts, 0, 0)
    if turning[-1]:
        turn_ends = np.append(turn_ends, len(turning))

    # Find apex (maximum yaw rate) for each turn
    apex_indices = []
    valid_starts = []
    valid_ends = []

    for start, end in zip(turn_starts, turn_ends):
        if end - start >= min_turn_length:
            # Find apex (maximum absolute yaw rate) in this turn
            turn_yaw_rates = np.abs(yaw_rate[start:end])
            apex_local = np.argmax(turn_yaw_rates)
            apex_indices.append(start + apex_local)
            valid_starts.append(start)
            valid_ends.append(end)

    return apex_indices, valid_starts, valid_ends
