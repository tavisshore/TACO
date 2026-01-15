"""Plotting functions for 2D trajectory visualization with GTSAM Pose2 support."""

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

from ..pose_graph import PoseGraph


def _convert_latlon_to_meters(
    positions: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert lat/lon positions to local meters coordinate frame.

    Uses the first position as the origin and converts to a local
    East-North-Up (ENU) coordinate frame in meters.

    Args:
        positions: Nx2 array of positions (lon, lat) or (x, y).

    Returns:
        Nx2 array of positions in meters relative to origin.
    """
    if len(positions) == 0:
        return positions

    # Use first position as origin
    origin = positions[0]

    # Earth radius in meters
    earth_radius = 6371000.0

    # Convert to radians
    lat_rad = np.radians(positions[:, 1])
    lon_rad = np.radians(positions[:, 0])
    origin_lat_rad = np.radians(origin[1])
    origin_lon_rad = np.radians(origin[0])

    # Convert to local ENU coordinates (meters)
    # x = East, y = North
    x = earth_radius * (lon_rad - origin_lon_rad) * np.cos(origin_lat_rad)
    y = earth_radius * (lat_rad - origin_lat_rad)

    return np.column_stack([x, y])


def _is_latlon(positions: npt.NDArray[np.float64]) -> bool:
    """Check if positions appear to be lat/lon coordinates.

    Args:
        positions: Nx2 array of positions.

    Returns:
        True if positions appear to be lat/lon (small range, typical GPS values).
    """
    if len(positions) == 0:
        return False
    # Check if values are in typical lat/lon range
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()

    # Lat/lon typically has small ranges (< 1 degree for local trajectories)
    # and values in specific ranges
    x_mean = np.mean(positions[:, 0])
    y_mean = np.mean(positions[:, 1])

    # Check if x (lon) is in valid longitude range and y (lat) in valid latitude range
    is_valid_lon = -180 <= x_mean <= 180
    is_valid_lat = -90 <= y_mean <= 90

    # Check if ranges are small (typical for vehicle trajectories in degrees)
    is_small_range = x_range < 1.0 and y_range < 1.0

    return is_valid_lon and is_valid_lat and is_small_range


def plot_trajectory(
    positions: npt.NDArray[np.float64],
    title: str = "Trajectory",
    show: bool = True,
    convert_latlon: bool = True,
) -> plt.Figure:
    """Plot a 2D trajectory.

    Args:
        positions: Nx2 array of positions (x, y) or (lon, lat).
        title: Plot title.
        show: Whether to display the plot.
        convert_latlon: If True, auto-detect and convert lat/lon to meters.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Check if we need to convert lat/lon to meters
    plot_positions = positions
    xlabel = "X (m)"
    ylabel = "Y (m)"

    if convert_latlon and _is_latlon(positions):
        plot_positions = _convert_latlon_to_meters(positions)
        xlabel = "East (m)"
        ylabel = "North (m)"

    # ax.plot(plot_positions[:, 0], plot_positions[:, 1], "b-", linewidth=2)
    # ax.scatter(plot_positions[0, 0], plot_positions[0, 1], c="g", s=100, label="Start", zorder=5)
    # ax.scatter(plot_positions[-1, 0], plot_positions[-1, 1], c="r", s=100, label="End", zorder=5)

    x = plot_positions[:, 0]
    y = plot_positions[:, 1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="rainbow", linewidth=2)
    lc.set_array(np.linspace(1, 0, len(segments)))

    ax.add_collection(lc)
    ax.autoscale()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    if show:
        plt.show()

    return fig


def plot_pose_graph(
    graph: PoseGraph,
    show_edges: bool = False,
    show_orientation: bool = True,
    title: str = "Pose Graph",
    show: bool = True,
    convert_latlon: bool = True,
) -> plt.Figure:
    """Plot a 2D pose graph using GTSAM Pose2.

    Args:
        graph: The pose graph to visualize.
        show_edges: Whether to draw edges between poses.
        show_orientation: Whether to show yaw orientation arrows.
        title: Plot title.
        show: Whether to display the plot.
        convert_latlon: If True, auto-detect and convert lat/lon to meters.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Get all poses from the graph
    poses_dict = graph.get_all_poses()

    if len(poses_dict) == 0:
        print("No poses to plot")
        return fig

    # Extract positions and yaws
    positions = []
    yaws = []
    for pose_id in sorted(poses_dict.keys()):
        pose = poses_dict[pose_id]
        positions.append([pose.x(), pose.y()])
        yaws.append(pose.theta())

    positions = np.array(positions)
    yaws = np.array(yaws)

    # Check if we need to convert lat/lon to meters
    plot_positions = positions
    xlabel = "X (m)"
    ylabel = "Y (m)"

    if convert_latlon and _is_latlon(positions):
        plot_positions = _convert_latlon_to_meters(positions)
        xlabel = "East (m)"
        ylabel = "North (m)"

    # Calculate appropriate arrow length based on trajectory scale
    trajectory_scale = max(
        plot_positions[:, 0].max() - plot_positions[:, 0].min(),
        plot_positions[:, 1].max() - plot_positions[:, 1].min(),
    )
    arrow_length = trajectory_scale * 0.02  # 2% of trajectory scale

    # Plot nodes
    ax.scatter(plot_positions[:, 0], plot_positions[:, 1], c="b", s=50, label="Poses", zorder=5)

    # Plot trajectory line
    ax.plot(plot_positions[:, 0], plot_positions[:, 1], "b-", alpha=0.5, linewidth=1)

    # Plot orientation arrows
    if show_orientation and arrow_length > 0:
        for _, (pos, yaw) in enumerate(zip(plot_positions, yaws, strict=False)):
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            ax.arrow(
                pos[0],
                pos[1],
                dx,
                dy,
                head_width=arrow_length * 0.3,
                head_length=arrow_length * 0.2,
                fc="blue",
                ec="blue",
                alpha=0.6,
            )

    # Mark start and end
    if len(plot_positions) > 0:
        ax.scatter(
            plot_positions[0, 0], plot_positions[0, 1], c="g", s=100, label="Start", zorder=10
        )
    if len(plot_positions) > 1:
        ax.scatter(
            plot_positions[-1, 0], plot_positions[-1, 1], c="r", s=100, label="End", zorder=10
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    if show:
        plt.show()

    return fig


def plot_gtsam_values(
    values: gtsam.Values,
    pose_keys: list[int],
    show_orientation: bool = True,
    title: str = "GTSAM Trajectory",
    show: bool = True,
    convert_latlon: bool = True,
) -> plt.Figure:
    """Plot GTSAM Values containing Pose2 objects.

    Args:
        values: GTSAM Values object.
        pose_keys: List of pose key indices.
        show_orientation: Whether to show yaw orientation arrows.
        title: Plot title.
        show: Whether to display the plot.
        convert_latlon: If True, auto-detect and convert lat/lon to meters.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    positions = []
    yaws = []
    for key in pose_keys:
        symbol = gtsam.symbol("x", key)
        if values.exists(symbol):
            pose = values.atPose2(symbol)
            positions.append([pose.x(), pose.y()])
            yaws.append(pose.theta())

    if len(positions) == 0:
        print("No poses found")
        return fig

    positions = np.array(positions)
    yaws = np.array(yaws)

    # Check if we need to convert lat/lon to meters
    plot_positions = positions
    xlabel = "X (m)"
    ylabel = "Y (m)"

    if convert_latlon and _is_latlon(positions):
        plot_positions = _convert_latlon_to_meters(positions)
        xlabel = "East (m)"
        ylabel = "North (m)"

    # Calculate appropriate arrow length based on trajectory scale
    trajectory_scale = max(
        plot_positions[:, 0].max() - plot_positions[:, 0].min(),
        plot_positions[:, 1].max() - plot_positions[:, 1].min(),
    )
    arrow_length = trajectory_scale * 0.02  # 2% of trajectory scale

    # Plot trajectory
    ax.plot(plot_positions[:, 0], plot_positions[:, 1], "b-", linewidth=2)
    ax.scatter(plot_positions[:, 0], plot_positions[:, 1], c="b", s=50)

    # Plot orientation arrows
    if show_orientation and arrow_length > 0:
        for pos, yaw in zip(plot_positions, yaws, strict=False):
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            ax.arrow(
                pos[0],
                pos[1],
                dx,
                dy,
                head_width=arrow_length * 0.3,
                head_length=arrow_length * 0.2,
                fc="blue",
                ec="blue",
                alpha=0.6,
            )

    # Mark start and end
    ax.scatter(plot_positions[0, 0], plot_positions[0, 1], c="g", s=100, label="Start", zorder=10)
    ax.scatter(plot_positions[-1, 0], plot_positions[-1, 1], c="r", s=100, label="End", zorder=10)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    if show:
        plt.show()

    return fig
