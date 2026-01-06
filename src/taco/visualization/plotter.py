"""Plotting functions for visualization with GTSAM support."""

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..pose_graph import PoseGraph


def plot_trajectory(
    positions: npt.NDArray[np.float64],
    title: str = "Trajectory",
    show: bool = True,
) -> plt.Figure:
    """Plot a 3D trajectory.

    Args:
        positions: Nx3 array of positions.
        title: Plot title.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="g", s=100, label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c="r", s=100, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig


def plot_pose_graph(
    graph: PoseGraph,
    show_edges: bool = False,
    title: str = "Pose Graph",
    show: bool = True,
) -> plt.Figure:
    """Plot a pose graph in 3D using GTSAM poses.

    Args:
        graph: The pose graph to visualize.
        show_edges: Whether to draw edges (not implemented for GTSAM graphs).
        title: Plot title.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Get all poses from the graph
    poses_dict = graph.get_all_poses()

    if len(poses_dict) == 0:
        print("No poses to plot")
        return fig

    # Extract positions
    positions = []
    for pose_id in sorted(poses_dict.keys()):
        pose = poses_dict[pose_id]
        t = pose.translation()
        positions.append([t.x(), t.y(), t.z()])

    positions = np.array(positions)

    # Plot nodes
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="b", s=50, label="Poses")

    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", alpha=0.5, linewidth=1)

    # Mark start and end
    if len(positions) > 0:
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="g", s=100, label="Start")
    if len(positions) > 1:
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c="r", s=100, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig


def plot_gtsam_values(
    values: gtsam.Values,
    pose_keys: list[int],
    title: str = "GTSAM Trajectory",
    show: bool = True,
) -> plt.Figure:
    """Plot GTSAM Values containing Pose3 objects.

    Args:
        values: GTSAM Values object.
        pose_keys: List of pose key indices.
        title: Plot title.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    positions = []
    for key in pose_keys:
        symbol = gtsam.symbol("x", key)
        if values.exists(symbol):
            pose = values.atPose3(symbol)
            t = pose.translation()
            positions.append([t.x(), t.y(), t.z()])

    if len(positions) == 0:
        print("No poses found")
        return fig

    positions = np.array(positions)

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=2)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="b", s=50)

    # Mark start and end
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="g", s=100, label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c="r", s=100, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig
