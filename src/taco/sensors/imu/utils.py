"""IMU utility functions for corner/turn detection.

This module provides functions to detect corner apexes from IMU gyroscope data,
useful for identifying key waypoints in vehicle trajectories.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter


@dataclass
class TurnDetection:
    """Results from turn/corner detection.

    Attributes:
        apex_indices: Indices of turn apex points (maximum yaw rate).
        start_indices: Indices where each turn begins.
        end_indices: Indices where each turn ends.
        turn_directions: Sign of each turn (+1 for left, -1 for right).
        exit_angles: Exit heading after each turn in radians (relative to initial heading).
            This is the direction of travel after completing the turn.
    """

    apex_indices: npt.NDArray[np.int64]
    start_indices: npt.NDArray[np.int64]
    end_indices: npt.NDArray[np.int64]
    entry_angles: npt.NDArray[np.float64]
    exit_angles: npt.NDArray[np.float64]


def detect_corners_from_gyro(
    gyro_z: npt.NDArray[np.float64] | torch.Tensor,
    dt: float | npt.NDArray[np.float64] | torch.Tensor = 0.1,
    initial_heading: float = 0.0,
    min_yaw_rate: float = 0.2,
    min_turn_angle: float = 0.25,
    min_apex_distance: int = 10,
    min_turn_duration: int = 5,
    smooth_window: int = 5,
    edge_threshold: float = 0.08,
    edge_hold_samples: int = 3,
    sign_consistency: float = 0.65,
    peak_prominence: float = 0.05,
    exit_heading_lookahead: int = 20,
    turn_latency: int = 25,
) -> TurnDetection:
    """Detect corner apexes directly from gyroscope yaw rate measurements.

    This function uses the z-axis gyroscope data (yaw rate) to detect turns,
    which is more robust than differentiating noisy heading estimates.

    Args:
        gyro_z: Array of z-axis angular velocity (yaw rate) in rad/s.
            Can be 1D array or tensor of shape (N,) or (N, 1).
        dt: Time step between samples in seconds. Can be scalar or array
            of per-sample time deltas.
        initial_heading: Initial heading at frame 0 in radians. This is added
            to the cumulative yaw to get absolute world heading.
        min_yaw_rate: Minimum peak yaw rate (rad/s) to consider as a turn.
            Default 0.15 rad/s (~8.6 deg/s).
        min_turn_angle: Minimum net heading change (rad) to keep a turn.
            Default 0.25 rad (~14 degrees).
        min_apex_distance: Minimum samples between detected apexes.
            Helps prevent detecting the same turn multiple times.
        min_turn_duration: Minimum samples from turn start to end.
        smooth_window: Window size for smoothing yaw rate before peak detection.
            Set to 1 to disable smoothing.
        edge_threshold: Yaw rate threshold (rad/s) for detecting turn start/end.
        edge_hold_samples: Number of consecutive samples below edge_threshold
            required to confirm turn boundary.
        sign_consistency: Fraction of samples that must match the net turn
            direction. Filters out oscillatory noise.
        peak_prominence: Minimum prominence for peak detection (rad/s).
        exit_heading_lookahead: Number of samples after apex to compute exit
            heading. The exit heading is the heading at apex + this offset.

    Returns:
        TurnDetection dataclass with apex, start, and end indices.
        - apex_indices: Frame indices of turn apexes (peak yaw rate)
        - start_indices: Frame indices where each turn starts
        - end_indices: Frame indices where each turn ends
        - turn_directions: +1 for left turns, -1 for right turns
        - entry_angles: Absolute entry heading before each turn in world frame
        - exit_angles: Absolute exit heading after each turn in world frame

    Example:
        >>> # From KITTI data loader
        >>> gyro_z = data.gyro[:, 2].numpy()  # z-axis angular velocity
        >>> dt = data.dt.numpy()
        >>> initial_yaw = data.yaws[0]  # Initial heading
        >>> turns = detect_corners_from_gyro(gyro_z, dt, initial_heading=initial_yaw)
        >>> print(f"Found {len(turns.apex_indices)} corners")
    """
    # Convert torch tensors to numpy
    if isinstance(gyro_z, torch.Tensor):
        gyro_z = gyro_z.cpu().numpy()
    if isinstance(dt, torch.Tensor):
        dt = dt.cpu().numpy()

    # Flatten if needed
    gyro_z = np.asarray(gyro_z).flatten()
    dt_arr = np.asarray(dt).flatten()

    # Handle scalar dt
    if dt_arr.size == 1:
        dt_scalar = float(dt_arr[0])
        dt_arr = np.full(len(gyro_z), dt_scalar)

    n = len(gyro_z)

    # Smooth the yaw rate to reduce noise
    if smooth_window > 1:
        yaw_rate_smooth = uniform_filter1d(gyro_z, size=smooth_window, mode="nearest")
    else:
        yaw_rate_smooth = gyro_z.copy()

    # Absolute yaw rate for peak detection
    yaw_rate_abs = np.abs(yaw_rate_smooth)

    # Find peaks in absolute yaw rate - these are the apex points
    apex_indices, _ = find_peaks(
        yaw_rate_abs,
        height=min_yaw_rate,
        distance=min_apex_distance,
        prominence=peak_prominence,
    )

    # If apex_indices is within 10 of the end, remove it (incomplete turn)
    apex_indices = apex_indices[apex_indices < n - turn_latency]

    # Integrate yaw rate to get cumulative heading (relative to start)
    cumulative_yaw = np.zeros(n)
    for i in range(1, n):
        # Gyro z is positive for left turns, so subtract to get heading
        cumulative_yaw[i] = cumulative_yaw[i - 1] - yaw_rate_smooth[i] * dt_arr[i]

    # Add initial heading to get absolute world heading
    absolute_heading = cumulative_yaw + initial_heading

    def find_turn_edge(idx: int, step: int) -> int:
        """Find turn start/end by walking until yaw rate drops below threshold."""
        j = idx
        below_count = 0
        while 0 <= j < n:
            if yaw_rate_abs[j] < edge_threshold:
                below_count += 1
                if below_count >= edge_hold_samples:
                    return j - (edge_hold_samples - 1) * abs(step)
            else:
                below_count = 0
            j += step
        return 0 if step < 0 else n - 1

    # Validate each apex and find turn boundaries
    valid_apexes = []
    valid_starts = []
    valid_ends = []
    entry_angles = []
    exit_angles = []

    for apex in apex_indices:
        # Find turn boundaries (used for validation, not for output index)
        start = find_turn_edge(apex - 1, step=-1)
        end = find_turn_edge(apex + 1, step=+1)

        # Ensure valid range
        start = max(0, start)
        end = min(n - 1, end)

        # Duration check
        if end - start + 1 < min_turn_duration:
            continue

        # Calculate net heading change over the turn
        segment_dt = dt_arr[start : end + 1]
        segment_yaw_rate = yaw_rate_smooth[start : end + 1]
        net_angle = np.sum(segment_yaw_rate * segment_dt)

        # Minimum turn angle check
        if np.abs(net_angle) < min_turn_angle:
            continue

        # Sign consistency check - most samples should rotate same direction
        turn_sign = np.sign(net_angle)
        same_sign_count = np.sum(np.sign(segment_yaw_rate) == turn_sign)
        consistency = same_sign_count / len(segment_yaw_rate)
        if consistency < sign_consistency:
            continue

        # Exit heading: absolute heading shortly after the apex
        # This gives the direction of travel as you exit the corner
        exit_idx = min(apex + exit_heading_lookahead, n - 1)
        exit_heading = absolute_heading[exit_idx]

        # Check for overlap with previous turn (based on apex proximity)
        if valid_apexes and apex - valid_apexes[-1] < min_apex_distance:
            # Overlapping turn - keep the one with larger angle change
            if np.abs(net_angle) > np.abs(
                cumulative_yaw[valid_ends[-1]] - cumulative_yaw[valid_starts[-1]]
            ):
                valid_apexes[-1] = apex
                valid_starts[-1] = start
                valid_ends[-1] = end
                entry_angles[-1] = absolute_heading[start]
                exit_angles[-1] = exit_heading
            continue

        valid_apexes.append(apex)
        valid_starts.append(start)
        valid_ends.append(end)
        entry_angles.append(absolute_heading[start])
        exit_angles.append(exit_heading)

    return TurnDetection(
        apex_indices=np.array(valid_apexes, dtype=np.int64),
        start_indices=np.array(valid_starts, dtype=np.int64),
        end_indices=np.array(valid_ends, dtype=np.int64),
        entry_angles=np.array(entry_angles, dtype=np.float64),
        exit_angles=np.array(exit_angles, dtype=np.float64),
    )


def detect_corners_from_yaw(
    yaw_series: npt.NDArray[np.float64] | torch.Tensor,
    dt: float = 0.1,
    initial_heading: float | None = None,
    min_peak_rate: float = 0.35,
    min_turn_angle: float = 0.26,
    min_apex_distance: int = 30,
    min_turn_duration: int = 20,
    smooth_window: int = 5,
    edge_threshold: float = 0.17,
    edge_hold_samples: int = 3,
    sign_consistency: float = 0.7,
    peak_prominence: float = 0.09,
) -> TurnDetection:
    """Detect corner apexes from yaw angle time series.

    This function differentiates the yaw angle to get yaw rate, then
    detects turns. Use detect_corners_from_gyro when raw gyro data is
    available, as it avoids differentiation noise.

    Args:
        yaw_series: Array of yaw angles in radians.
        dt: Time step between samples in seconds.
        initial_heading: Initial heading at frame 0 in radians. If None,
            uses the first value from yaw_series.
        min_peak_rate: Minimum peak yaw rate (rad/s) to consider as a turn.
        min_turn_angle: Minimum net heading change (rad) to keep a turn.
        min_apex_distance: Minimum samples between detected apexes.
        min_turn_duration: Minimum samples from turn start to end.
        smooth_window: Window size for smoothing yaw before differentiation.
        edge_threshold: Yaw rate threshold (rad/s) for detecting turn start/end.
        edge_hold_samples: Samples below threshold to confirm turn boundary.
        sign_consistency: Fraction of samples matching net turn direction.
        peak_prominence: Minimum prominence for peak detection (rad/s).

    Returns:
        TurnDetection dataclass with apex, start, and end indices.
    """
    # Convert torch tensors to numpy
    if isinstance(yaw_series, torch.Tensor):
        yaw_series = yaw_series.cpu().numpy()

    yaw = np.asarray(yaw_series).flatten()

    # Use first yaw value as initial heading if not provided
    if initial_heading is None:
        initial_heading = yaw[0]

    # Unwrap to handle angle discontinuities
    yaw = np.unwrap(yaw)

    # Smooth yaw before differentiating
    if smooth_window > 1 and len(yaw) > smooth_window:
        # Use Savitzky-Golay filter for smoother derivatives
        yaw = savgol_filter(yaw, smooth_window, polyorder=2)

    # Compute yaw rate via differentiation
    yaw_rate = np.gradient(yaw, dt)

    # Now use the gyro-based detection on the derived yaw rate
    return detect_corners_from_gyro(
        gyro_z=yaw_rate,
        dt=dt,
        initial_heading=initial_heading,
        min_yaw_rate=min_peak_rate,
        min_turn_angle=min_turn_angle,
        min_apex_distance=min_apex_distance,
        min_turn_duration=min_turn_duration,
        smooth_window=1,  # Already smoothed
        edge_threshold=edge_threshold,
        edge_hold_samples=edge_hold_samples,
        sign_consistency=sign_consistency,
        peak_prominence=peak_prominence,
    )


def detect_corners_from_kitti(
    kitti_data,
    use_gyro: bool = True,
    **kwargs,
) -> TurnDetection:
    """Convenience function to detect corners from KITTI data loader.

    Args:
        kitti_data: KITTI data loader instance with gyro, dt, and yaws attributes.
        use_gyro: If True, use raw gyroscope data. If False, derive from yaw.
        **kwargs: Additional arguments passed to detection function.
            If 'initial_heading' is not provided, uses kitti_data.yaws[0].

    Returns:
        TurnDetection dataclass with detected corner information.

    Example:
        >>> from taco.data.kitti import Kitti
        >>> data = Kitti(args)
        >>> turns = detect_corners_from_kitti(data)
        >>> print(f"Found {len(turns.apex_indices)} corners at frames: {turns.apex_indices}")
    """
    # Get initial heading from KITTI if not provided
    if "initial_heading" not in kwargs:
        kwargs["initial_heading"] = kitti_data.yaws[0]

    if use_gyro:
        # Use raw gyroscope z-axis data
        gyro_z = kitti_data.gyro[:, 2]
        dt = kitti_data.dt
        return detect_corners_from_gyro(gyro_z, dt, **kwargs)

    # Derive from yaw angles
    yaw = np.array(kitti_data.yaws)
    dt = kitti_data.dt.mean().item() if hasattr(kitti_data.dt, "mean") else 0.1
    return detect_corners_from_yaw(yaw, dt, **kwargs)


def filter_close_corners(
    turns: TurnDetection,
    min_separation: int = 20,
    keep_strategy: str = "largest",
) -> TurnDetection:
    """Filter out corners that are too close together.

    When multiple corners are detected in quick succession (e.g., chicanes),
    this function can merge or filter them based on the chosen strategy.

    Args:
        turns: TurnDetection result to filter.
        min_separation: Minimum samples between apex indices.
        keep_strategy: How to handle close corners:
            - "largest": Keep the corner with largest turn angle
            - "first": Keep the first detected corner
            - "last": Keep the last detected corner

    Returns:
        Filtered TurnDetection with reduced corner count.
    """
    if len(turns.apex_indices) <= 1:
        return turns

    keep_mask = np.ones(len(turns.apex_indices), dtype=bool)
    i = 0

    while i < len(turns.apex_indices) - 1:
        # Find all corners within min_separation of current
        group_end = i + 1
        while (
            group_end < len(turns.apex_indices)
            and turns.apex_indices[group_end] - turns.apex_indices[i] < min_separation
        ):
            group_end += 1

        if group_end > i + 1:
            # Multiple close corners - select one based on strategy
            group_indices = list(range(i, group_end))

            if keep_strategy == "largest":
                best = max(group_indices, key=lambda x: abs(turns.exit_angles[x]))
            elif keep_strategy == "first":
                best = group_indices[0]
            elif keep_strategy == "last":
                best = group_indices[-1]
            else:
                raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

            for idx in group_indices:
                if idx != best:
                    keep_mask[idx] = False

            i = group_end
        else:
            i += 1

    return TurnDetection(
        apex_indices=turns.apex_indices[keep_mask],
        start_indices=turns.start_indices[keep_mask],
        end_indices=turns.end_indices[keep_mask],
        turn_directions=turns.turn_directions[keep_mask],
        exit_angles=turns.exit_angles[keep_mask],
    )
