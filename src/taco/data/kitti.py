"""KITTI dataset loader for pose graph optimization.

This module provides data loaders for the KITTI odometry and raw datasets,
including IMU data, camera images, and GPS coordinates.
"""

from __future__ import annotations

import glob
import math
import os
from datetime import datetime
from math import asin, atan2, cos, degrees, radians, sin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pykitti
import pypose as pp
import torch
from geopy.distance import geodesic
from haversine import Unit, haversine
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# Optional imports - these may not be available in all environments
try:
    import streetview
except ImportError:
    streetview = None

from taco.data.camera_model import CameraModel
from taco.data.graph_refine import simplify_sharp_turns
from taco.data.sat import download_satmap
from taco.sensors.imu import TurnDetection
from taco.utils.geometry import rotate_trajectory
from taco.visualization.plotter import _convert_latlon_to_meters


def latlon_to_local_meters(
    coords: List[Tuple[float, float]],
    origin: Tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert lat/lon coordinates to local meters coordinate frame.

    Uses the first coordinate (or specified origin) as the origin and converts
    to a local East-North coordinate frame in meters.

    Args:
        coords: List of (lat, lon) tuples.
        origin: Optional (lat, lon) tuple to use as origin. If None, uses first coord.

    Returns:
        Nx2 numpy array of positions in meters (x=East, y=North) relative to origin.
    """
    if len(coords) == 0:
        return np.array([])

    if origin is None:
        origin = coords[0]

    origin_lat, origin_lon = origin

    # Earth radius in meters
    earth_radius = 6371000.0

    # Convert origin to radians
    origin_lat_rad = math.radians(origin_lat)

    positions = []
    for lat, lon in coords:
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        origin_lon_rad = math.radians(origin_lon)

        # Convert to local ENU coordinates (meters)
        # x = East, y = North
        x = earth_radius * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
        y = earth_radius * (lat_rad - origin_lat_rad)
        positions.append([x, y])

    return np.array(positions, dtype=np.float64)


def destination_point(
    lat1: float, lon1: float, lat2: float, lon2: float, distance_m: float
) -> Tuple[float, float]:
    """
    Compute the geographic coordinate `distance_m` meters from (lat1, lon1)
    in the direction of (lat2, lon2).
    """
    # Convert to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    # Compute initial bearing from point1 to point2
    dlon = lon2_rad - lon1_rad
    x = cos(lat2_rad) * sin(dlon)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
    bearing = atan2(x, y)  # radians

    # Earth radius (mean)
    R = 6371000.0
    d = distance_m / R  # angular distance

    lat_dest = asin(sin(lat1_rad) * cos(d) + cos(lat1_rad) * sin(d) * cos(bearing))
    lon_dest = lon1_rad + atan2(
        sin(bearing) * sin(d) * cos(lat1_rad), cos(d) - sin(lat1_rad) * sin(lat_dest)
    )

    return degrees(lat_dest), degrees(lon_dest)


# Is this bad practice?
def enu_yaw_to_compass_cw(yaw_rad):
    """
    Convert KITTI OXTS yaw (ENU, 0 = East, CCW-positive, radians)
    to a north-aligned, clockwise-increasing heading.

    Returns:
        heading_rad in [0, 2π)
        heading_deg in [0, 360)
    """
    heading_rad = (0.5 * np.pi - yaw_rad) % (2 * np.pi)
    # heading_deg = np.degrees(heading_rad)
    return heading_rad


def calculate_bearing(lat1, lon1, lat2, lon2):
    phi_1, lambda_1, phi_2, lambda_2 = map(math.radians, [lat1, lon1, lat2, lon2])
    lambda_d = lambda_2 - lambda_1
    x = math.sin(lambda_d) * math.cos(phi_2)
    y = math.cos(phi_1) * math.sin(phi_2) - math.sin(phi_1) * math.cos(phi_2) * math.cos(lambda_d)
    theta = math.atan2(x, y)
    bearing = (math.degrees(theta) + 360) % 360  # Normalize to [0, 360)
    bearing = np.deg2rad(bearing)  # Convert to radians
    return bearing


def calculate_bearings(graph):
    """Calculate exit bearings from each node to its neighbors.

    For each edge, calculates the bearing (direction) when leaving from each end.
    Uses edge geometry if available with >2 points, otherwise uses direct node positions.

    Args:
        graph: NetworkX graph with node positions (x=lon, y=lat) and optional edge geometry.

    Returns:
        Dict mapping node -> {neighbor: bearing_in_radians}.
    """
    edge_angles = {}

    for u in graph.nodes:
        edge_angles[u] = {}

        # Get node u position
        u_lat, u_lon = graph.nodes[u]["y"], graph.nodes[u]["x"]

        for v in graph.neighbors(u):
            # Get node v position
            v_lat, v_lon = graph.nodes[v]["y"], graph.nodes[v]["x"]

            # Bearing from u to v using node coordinates only
            edge_angles[u][v] = calculate_bearing(u_lat, u_lon, v_lat, v_lon)

    return edge_angles


def angle_difference(angle1: float, angle2: float) -> float:
    """Compute the absolute angular difference between two angles in radians.

    Handles wrap-around correctly (e.g., 0.1 rad and 6.2 rad are ~0.18 rad apart).

    Args:
        angle1: First angle in radians.
        angle2: Second angle in radians.

    Returns:
        Absolute angular difference in [0, π].
    """
    diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def find_matching_node_turns(
    graph: nx.Graph,
    entry_angle: float,
    exit_angle: float,
    angle_tolerance: float = 0.35,
) -> list[tuple[int, int, int, float]]:
    """Find graph nodes where a turn with given entry/exit angles is possible.

    For a turn to match at a node, there must exist:
    - An incoming edge whose bearing matches the entry_angle (reversed, since we
      arrive FROM that direction)
    - An outgoing edge whose bearing matches the exit_angle

    Args:
        graph: NetworkX graph with 'yaws' attribute on nodes containing
            {neighbor_id: bearing_angle} for each edge.
        entry_angle: Heading when entering the turn (absolute, in radians).
        exit_angle: Heading when exiting the turn (absolute, in radians).
        angle_tolerance: Maximum angular difference (radians) to consider a match.
            Default 0.35 rad (~20 degrees).

    Returns:
        List of (node_id, entry_neighbor, exit_neighbor, score) tuples for matching nodes.
        Score is the combined angular error (lower is better).
    """

    matches = []

    for node in graph.nodes():
        yaws = graph.nodes[node].get("yaws", {})
        if len(yaws) < 2:
            # Need at least 2 edges for a turn
            continue

        # For each possible (entry_edge, exit_edge) combination
        for entry_neighbor, entry_bearing in yaws.items():
            incoming_heading = (entry_bearing + np.pi) % (2 * np.pi)
            entry_error = angle_difference(incoming_heading, entry_angle)

            if entry_error > angle_tolerance:
                continue

            for exit_neighbor, exit_bearing in yaws.items():
                if exit_neighbor == entry_neighbor:
                    continue  # Can't exit the way we came

                exit_error = angle_difference(exit_bearing, exit_angle)

                if exit_error > angle_tolerance:
                    continue

                # Combined angular error as score (lower is better)
                score = entry_error + exit_error
                matches.append((node, entry_neighbor, exit_neighbor, score))

    return matches


def narrow_candidates_by_turn_sequence(
    data,
    turns: TurnDetection,
    angle_tolerance: float = 0.35,
    connectivity_check: bool = True,
    max_path_length: int = 10,
) -> list[list[tuple[int, int, int, float]]]:
    """Narrow down candidate nodes using a sequence of detected turns.

    Given a sequence of turns (entry/exit angle pairs), finds graph nodes that
    could correspond to each turn. Optionally filters by connectivity (each
    successive turn node must be reachable from the previous one in the direction
    of the previous exit angle).

    Args:
        graph: NetworkX graph with 'yaws' attribute on nodes.
        turns: TurnDetection object with entry_angles and exit_angles arrays.
        angle_tolerance: Maximum angular difference for a match (radians).
        connectivity_check: If True, filter candidates to only include nodes
            reachable from the previous turn's candidates in the direction of
            the previous exit angle.
        max_path_length: Maximum path length to search for connectivity (default 10).

    Returns:
        List of candidate lists, one per turn. Each candidate list contains
        (node_id, entry_neighbor, exit_neighbor, score) tuples.
        Score is the combined angular error (lower is better).
    """
    if len(turns.entry_angles) != len(turns.exit_angles):
        raise ValueError("entry_angles and exit_angles must have same length")

    graph = data.graph

    all_candidates = []

    for i, (entry, exit_) in enumerate(zip(turns.entry_angles, turns.exit_angles, strict=True)):
        candidates = find_matching_node_turns(graph, entry, exit_, angle_tolerance)

        if connectivity_check and i > 0 and all_candidates[i - 1]:
            # Get previous turn information
            prev_candidates = all_candidates[i - 1]
            prev_exit_angle = turns.exit_angles[i - 1]

            # Build a set of reachable nodes from previous turn exits
            # Only consider paths that generally follow the exit angle direction
            reachable_nodes = set()

            for _, _, prev_exit_neighbor, _ in prev_candidates:
                # Start from the exit neighbor of the previous turn
                # Use BFS to find nodes reachable within max_path_length
                visited = {prev_exit_neighbor}
                queue = [(prev_exit_neighbor, 0)]  # (node, depth)

                while queue:
                    current_node, depth = queue.pop(0)

                    if depth >= max_path_length:
                        continue

                    reachable_nodes.add(current_node)

                    # Explore neighbors that are roughly in the direction of prev_exit_angle
                    current_yaws = graph.nodes[current_node].get("yaws", {})
                    for neighbor, bearing in current_yaws.items():
                        if neighbor not in visited:
                            # Check if this edge is roughly aligned with the exit direction
                            angle_diff = angle_difference(bearing, prev_exit_angle)
                            # Allow edges within ~90 degrees of the exit direction
                            if angle_diff < np.pi / 2:
                                visited.add(neighbor)
                                queue.append((neighbor, depth + 1))

            # Filter candidates to those reachable in the exit direction
            reachable_candidates = []
            for cand in candidates:
                node_id, entry_neighbor, _exit_neighbor, _score = cand
                # Check if the candidate node or its entry neighbor is reachable
                if node_id in reachable_nodes or entry_neighbor in reachable_nodes:
                    reachable_candidates.append(cand)

            candidates = reachable_candidates

        all_candidates.append(candidates)

    return all_candidates


def narrow_candidates_from_turns(
    data: Kitti,
    turns: TurnDetection,
    angle_tolerance: float = np.deg2rad(30),  # radians
    connectivity_check: bool = True,
    verbose: bool = False,
    output_path: str | Path | None = None,
    frame_idx: int | None = None,
) -> list[list[tuple[int, int, int, float]]]:
    """Convenience wrapper to narrow candidates using TurnDetection result.

    Args:
        data: Kitti data instance with graph attribute.
        turns: TurnDetection dataclass with entry_angles and exit_angles.
        angle_tolerance: Maximum angular difference for a match (radians).
        connectivity_check: If True, filter by connectivity between successive turns.
        verbose: If True, save a matplotlib visualization of candidate nodes.
        output_path: Path to save the visualization. Defaults to 'candidate_nodes.png'.
        frame_idx: Frame index for getting ground truth node position.

    Returns:
        List of candidate lists, one per turn. Each candidate is a tuple of
        (node_id, entry_neighbor, exit_neighbor, score) where score is the
        combined angular error (lower is better).

    Example:
        >>> from taco.sensors.imu import detect_corners_from_kitti
        >>> turns = detect_corners_from_kitti(kitti_data)
        >>> candidates = narrow_candidates_from_turns(kitti_data, turns, verbose=True)
        >>> # candidates[i] contains possible nodes for turn i
        >>> for i, cands in enumerate(candidates):
        ...     print(f"Turn {i}: {len(cands)} candidate nodes")
        ...     # Sort by score to get best matches
        ...     sorted_cands = sorted(cands, key=lambda x: x[3])
        ...     if sorted_cands:
        ...         best = sorted_cands[0]
        ...         print(f"  Best match: node {best[0]} with score {best[3]:.3f} rad")
    """
    candidates = narrow_candidates_by_turn_sequence(
        data,
        turns,
        angle_tolerance=angle_tolerance,
        connectivity_check=connectivity_check,
    )

    if verbose:
        plot_candidate_nodes(data, candidates, output_path, frame_idx=frame_idx)

    return candidates


def _get_all_node_positions(data) -> dict:
    """Get node positions from all available graphs."""
    all_nodes = {}
    if hasattr(data, "raw_graph"):
        for node, node_data in data.raw_graph.nodes(data=True):
            all_nodes[node] = (node_data["x"], node_data["y"])
    for node, node_data in data.original_graph.nodes(data=True):
        if node not in all_nodes:
            all_nodes[node] = (node_data["x"], node_data["y"])
    return all_nodes


def _prepare_positions_for_plotting(data, all_nodes: dict) -> dict:
    """Prepare node positions in the correct coordinate system for plotting."""
    ox_pos = {node: (data_pt["x"], data_pt["y"]) for node, data_pt in data.graph.nodes(data=True)}
    for node, coords in all_nodes.items():
        if node not in ox_pos:
            ox_pos[node] = coords
    return ox_pos


def _compute_node_colors_and_sizes(
    candidate_node_ids: list, node_scores: dict, color
) -> tuple[list, list]:
    """Compute node colors and sizes based on scores."""
    scores = list(node_scores.values())
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score > min_score else 1.0

    node_colors = []
    node_sizes = []
    for node_id in candidate_node_ids:
        score = node_scores[node_id]
        normalized = (score - min_score) / score_range
        brightness = 1.0 - (normalized * 0.7)
        base_color = np.array(plt.cm.colors.to_rgb(color))
        bright_color = base_color * brightness + (1 - brightness) * 0.2
        node_colors.append(bright_color)
        node_sizes.append(200 - normalized * 100)
    return node_colors, node_sizes


def _draw_direction_arrow(
    ax,
    node_x: float,
    node_y: float,
    bearing: float,
    arrow_length: float,
    color: str,
    reverse: bool = False,
):
    """Draw a single direction arrow at a node."""
    if reverse:
        bearing = bearing + np.pi
    math_angle = np.pi / 2 - bearing
    dx = arrow_length * np.cos(math_angle)
    dy = arrow_length * np.sin(math_angle)

    if reverse:
        ax.annotate(
            "",
            xy=(node_x, node_y),
            xytext=(node_x - dx, node_y - dy),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5, "alpha": 0.7},
            zorder=10,
        )
    else:
        ax.annotate(
            "",
            xy=(node_x + dx, node_y + dy),
            xytext=(node_x, node_y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5, "alpha": 0.7},
            zorder=10,
        )


def _draw_candidate_arrows(
    ax, turn_candidates: list, node_data: dict, pos: dict, data, arrow_length: float
):
    """Draw entry/exit arrows for all candidates."""
    for node_id, entry_neighbor, exit_neighbor, score in turn_candidates:
        if node_id not in pos:
            continue
        if node_data[node_id][2] != score:
            continue

        node_x, node_y = pos[node_id]
        yaws = data.graph.nodes[node_id].get("yaws", {}) if data.graph.has_node(node_id) else {}

        if entry_neighbor in yaws:
            _draw_direction_arrow(
                ax, node_x, node_y, yaws[entry_neighbor], arrow_length, "blue", reverse=True
            )

        if exit_neighbor in yaws:
            _draw_direction_arrow(
                ax, node_x, node_y, yaws[exit_neighbor], arrow_length, "red", reverse=False
            )


def plot_candidate_nodes(
    data,  # Kitti instance
    candidates: list[list[tuple[int, int, int, float]]],
    output_path: str | Path | None = None,
    frame_idx: int | None = None,
) -> None:
    """Plot the graph with candidate nodes highlighted for each turn.

    Only plots candidates that are present after connectivity filtering,
    showing the connected sequence of turn candidates. Node colors indicate
    their score (angular error), with better matches being brighter.

    Args:
        data: Kitti data instance with graph attribute.
        candidates: List of candidate lists from narrow_candidates_by_turn_sequence.
            Each candidate is (node_id, entry_neighbor, exit_neighbor, score).
        output_path: Path to save the figure. Defaults to 'candidate_nodes.png'.
        frame_idx: Frame index for getting ground truth node position.
    """
    output_path = Path(output_path) if output_path else Path("candidate_nodes.png")

    all_nodes = _get_all_node_positions(data)

    # fig, ax = ox.plot_graph(
    #     data.raw_graph,
    #     figsize=(14, 12),
    #     bgcolor="#FFFFFF",
    #     node_size=10,
    #     node_color="#000000",
    #     edge_color="#444444",
    #     show=False,
    #     close=False,
    # )

    fig, ax = plt.subplots(figsize=(12, 12))
    pos = {node: (data["x"], data["y"]) for node, data in data.graph.nodes(data=True)}
    nx.draw_networkx_edges(data.graph, pos, ax=ax, edge_color="gray", width=1)
    nx.draw_networkx_nodes(data.graph, pos, ax=ax, node_color="red", node_size=10)

    pos = _prepare_positions_for_plotting(data, all_nodes)
    colors = plt.cm.tab10.colors

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    arrow_length = extent * 0.02

    candidates = [candidates[-1]] if candidates else []

    for turn_idx, turn_candidates in enumerate(candidates):
        if not turn_candidates:
            continue

        color = colors[turn_idx % len(colors)]

        node_scores = {}
        node_data = {}
        for node_id, entry_neighbor, exit_neighbor, score in turn_candidates:
            if node_id not in node_scores or score < node_scores[node_id]:
                node_scores[node_id] = score
                node_data[node_id] = (entry_neighbor, exit_neighbor, score)

        candidate_node_ids = list(node_scores.keys())

        if node_scores:
            node_colors, node_sizes = _compute_node_colors_and_sizes(
                candidate_node_ids, node_scores, color
            )

            nx.draw_networkx_nodes(
                data.graph,
                pos,
                nodelist=candidate_node_ids,
                ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

            min_score = min(node_scores.values())
            max_score = max(node_scores.values())
            best_score_deg = np.rad2deg(min_score)
            worst_score_deg = np.rad2deg(max_score)
            ax.plot(
                [],
                [],
                marker="o",
                color=color,
                linestyle="",
                markersize=8,
                label=f"Turn {turn_idx + 1} ({len(candidate_node_ids)} candidates)\n"
                f"Score range: {best_score_deg:.1f}° - {worst_score_deg:.1f}°",
            )
        _draw_candidate_arrows(ax, turn_candidates, node_data, pos, data, arrow_length)

    if frame_idx is not None:
        gt_node = data.get_gt_node(frame_idx)
        if gt_node in pos:
            ax.scatter(
                [pos[gt_node][0]],
                [pos[gt_node][1]],
                c="green",
                s=200,
                marker="*",
                edgecolors="black",
                linewidths=1,
                label="Ground Truth",
                zorder=12,
            )

    ax.plot([], [], color="blue", lw=2, label="Entry direction")
    ax.plot([], [], color="red", lw=2, label="Exit direction")

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (meters)")
    ax.set_ylabel("Latitude (meters)")
    ax.set_title("Road Network Graph with Turn Candidate Nodes\n(Brighter = Better Match)")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved candidate nodes visualization to {output_path}")


# Better way of storing this
RAW_DICT = {
    0: {
        "path": "raw_data/2011_10_03/2011_10_03_drive_0027_sync/oxts/data/",
        "start": 0,
        "end": 4540,
        "date": "2011_10_03",
        "drive": "0027",
    },
    1: {
        "path": "raw_data/2011_10_03/2011_10_03_drive_0042_sync/oxts/data/",
        "start": 0,
        "end": 1100,
        "date": "2011_10_03",
        "drive": "0042",
    },
    2: {
        "path": "raw_data/2011_10_03/2011_10_03_drive_0034_sync/oxts/data/",
        "start": 0,
        "end": 4660,
        "date": "2011_10_03",
        "drive": "0034",
    },
    3: {
        "path": "raw_data/2011_09_26/2011_09_26_drive_0067_sync/oxts/data/",
        "start": 0,
        "end": 800,
        "date": "2011_09_26",
        "drive": "0067",
    },
    4: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0016_sync/oxts/data/",
        "start": 0,
        "end": 270,
        "date": "2011_09_30",
        "drive": "0016",
    },
    5: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0018_sync/oxts/data/",
        "start": 0,
        "end": 2760,
        "date": "2011_09_30",
        "drive": "0018",
    },
    6: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0020_sync/oxts/data/",
        "start": 0,
        "end": 1100,
        "date": "2011_09_30",
        "drive": "0020",
    },
    7: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0027_sync/oxts/data/",
        "start": 0,
        "end": 1100,
        "date": "2011_09_30",
        "drive": "0027",
    },
    8: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0028_sync/oxts/data/",
        "start": 1100,
        "end": 5170,
        "date": "2011_09_30",
        "drive": "0028",
    },
    9: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0033_sync/oxts/data/",
        "start": 0,
        "end": 1590,
        "date": "2011_09_30",
        "drive": "0033",
    },
    10: {
        "path": "raw_data/2011_09_30/2011_09_30_drive_0034_sync/oxts/data/",
        "start": 0,
        "end": 1200,
        "date": "2011_09_30",
        "drive": "0034",
    },
}


class Kitti:
    def __init__(self, args):
        """
        Dataloader for KITTI Visual Odometry Dataset
            http://www.cvlibs.net/datasets/kitti/eval_odometry.php

        Arguments:
            data_path {str}: path to data sequences
            pose_path {str}: path to poses
            sequence {str}: sequence to be tested (default: "00")
        """
        self.data_path = "/scratch/datasets/KITTI/odometry/dataset/sequences_jpg/"
        self.sequence = str(args.sequence).zfill(2)  # Ensure sequence is two digits
        self.camera_id = "0"
        self.frame_id = 0
        self.verbose = args.verbose
        self.output_dir = args.output_dir
        self.debug = args.debug

        # Read ground truth poses
        with open(
            os.path.join("/scratch/datasets/KITTI/odometry/dataset/poses/", self.sequence + ".txt")
        ) as f:
            self.poses = f.readlines()

        # Get frames list
        frames_dir = os.path.join(self.data_path, self.sequence, f"image_{self.camera_id}", "*.jpg")
        self.frames = sorted(glob.glob(frames_dir))

        # Camera Parameters
        self.cam_params = {}
        frame = cv2.imread(self.frames[self.frame_id], 0)
        self.cam_params["width"] = frame.shape[0]
        self.cam_params["height"] = frame.shape[1]
        self.read_intrinsics_param()

        self.cam = CameraModel(params=self.cam_params)
        self.data = pykitti.raw(
            "/scratch/datasets/KITTI/",
            RAW_DICT[args.sequence]["date"],
            RAW_DICT[args.sequence]["drive"],
            dataset="sync",
        )
        seq_range = range(RAW_DICT[args.sequence]["start"], RAW_DICT[args.sequence]["end"])
        self.gt_coords = [
            (self.data.oxts[i].packet.lat, self.data.oxts[i].packet.lon) for i in seq_range
        ]
        self.mid_coord = (
            sum(lat for lat, _ in self.gt_coords) / len(self.gt_coords),
            sum(lon for _, lon in self.gt_coords) / len(self.gt_coords),
        )
        lat0 = self.mid_coord[0]
        # lon0 = self.mid_coord[1]
        EXTRA_LEN = 100  # meters

        # meters per degree at lat0 (more accurate than a flat 111_320)
        phi = math.radians(lat0)
        m_per_deg_lat = (
            111132.92
            - 559.82 * math.cos(2 * phi)
            + 1.175 * math.cos(4 * phi)
            - 0.0023 * math.cos(6 * phi)
        )
        m_per_deg_lon = (
            111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi)
        )

        lat_pad = EXTRA_LEN / m_per_deg_lat
        lon_pad = EXTRA_LEN / m_per_deg_lon

        # Bounding box from points
        lats = [lat for lat, _ in self.gt_coords]
        lons = [lon for _, lon in self.gt_coords]
        self.bbox = (max(lats), min(lats), max(lons), min(lons))  # (north, south, east, west)

        lat_pad = 0
        self.bbox_padded = (
            self.bbox[2] + lon_pad,  # east
            self.bbox[1] - lat_pad,  # south
            self.bbox[0] + lat_pad,  # north
            self.bbox[3] - lon_pad,  # west
        )

        self.seq_len = len(self.data.timestamps) - 1
        self.dt = torch.tensor(
            [
                datetime.timestamp(self.data.timestamps[i + 1])
                - datetime.timestamp(self.data.timestamps[i])
                for i in seq_range
            ]
        )
        self.gyro = torch.tensor(
            [
                [
                    self.data.oxts[i].packet.wx,
                    self.data.oxts[i].packet.wy,
                    self.data.oxts[i].packet.wz,
                ]
                for i in seq_range
            ]
        )
        self.acc = torch.tensor(
            [
                [
                    self.data.oxts[i].packet.ax,
                    self.data.oxts[i].packet.ay,
                    self.data.oxts[i].packet.az,
                ]
                for i in seq_range
            ]
        )

        # IMU Values
        self.gt_yaw = [np.deg2rad(90) - self.data.oxts[i].packet.yaw for i in seq_range]
        self.gt_rot = pp.euler2SO3(
            torch.tensor(
                [
                    [
                        self.data.oxts[i].packet.roll,
                        self.data.oxts[i].packet.pitch,
                        self.data.oxts[i].packet.yaw,
                    ]
                    for i in seq_range
                ],
                dtype=torch.float32,
            )
        )
        self.gt_vel = self.gt_rot @ torch.tensor(
            [
                [
                    self.data.oxts[i].packet.vf,
                    self.data.oxts[i].packet.vl,
                    self.data.oxts[i].packet.vu,
                ]
                for i in seq_range
            ]
        )
        self.gt_pos = torch.tensor(
            rotate_trajectory(
                [self.data.oxts[i].T_w_imu[0:3, 3] for i in seq_range],
                axis="z",
                degrees=90 - np.rad2deg(self.data.oxts[0].packet.yaw),
            )
        )

        # Convert lat/lon ground truth to local meters coordinate frame
        # This provides 2D positions (x, y) in meters relative to the first position
        self.gt_pos_meters = torch.tensor(
            latlon_to_local_meters(self.gt_coords),
            dtype=torch.float32,
        )

        self.duration = 2

        # Setup CVGL data: graph, sat images, etc.
        self.setup_graph()

    def __len__(self):
        return len(self.frames)

    def get_next_data(self):
        """
        Returns:
            frame {ndarray}: image frame at index self.frame_id
            pose {list}: list containing the ground truth pose [x, y, z]
            frame_id {int}: integer representing the frame index
        """
        # Read frame as grayscale
        frame = cv2.imread(self.frames[self.frame_id], 0)
        self.cam_params["width"] = frame.shape[0]
        self.cam_params["height"] = frame.shape[0]

        # Read poses
        pose = self.poses[self.frame_id]
        pose = pose.strip().split()
        pose = [float(pose[3]), float(pose[7]), float(pose[11])]  # coordinates for the left camera
        frame_id = self.frame_id
        yaw = self.data.oxts[self.frame_id].packet.yaw
        # self.frame_id = self.frame_id + 1
        return frame, pose, frame_id, yaw

    def get_nodes(self, frame_idx):
        """
        Takes index of current frame in sequence, get's coord, and return nearest node index from graph.
        """
        coord = self.get_coord(frame_idx)
        lat, lon = coord

        # Find nearest node using haversine distance
        min_dist = float("inf")
        nearest_node = None
        for node, data in self.graph.nodes(data=True):
            node_lat, node_lon = data["y"], data["x"]
            dist = haversine((lat, lon), (node_lat, node_lon), unit=Unit.METERS)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def read_intrinsics_param(self):
        """
        Reads camera intrinsics parameters

        Returns:
            cam_params {dict}: dictionary with focal lenght and principal point
        """
        calib_file = os.path.join(self.data_path, self.sequence, "calib.txt")
        with open(calib_file) as f:
            lines = f.readlines()
            line = lines[int(self.camera_id)].strip().split()
            [fx, cx, fy, cy] = [float(line[1]), float(line[3]), float(line[6]), float(line[7])]

            # focal length of camera
            self.cam_params["fx"] = fx
            self.cam_params["fy"] = fy
            # principal point
            self.cam_params["cx"] = cx
            self.cam_params["cy"] = cy

    def get_colour_img(self, frame=None):
        street = self.data.get_cam2(self.frame_id) if frame is None else self.data.get_cam2(frame)
        street = street.resize((1241, 376))  # Check
        return street

    def get_gt_yaw(self, frame=None):
        return self.gt_yaw[self.frame_id] if frame is None else self.gt_yaw[frame]

    def get_coord(self, frame=None):
        return self.gt_coords[self.frame_id] if frame is None else self.gt_coords[frame]

    def get_start_coord(self):
        return self.gt_coords[0]

    def get_timestamp(self, frame=None):
        return (
            datetime.timestamp(self.data.timestamps[self.frame_id])
            if frame is None
            else datetime.timestamp(self.data.timestamps[frame])
        )

    def get_imu(self, i):
        start_frame = i - self.duration + 1
        end_frame = i + 1

        return {
            "dt": self.dt[start_frame:end_frame].unsqueeze(-1),
            "dt_full": self.dt[:end_frame].unsqueeze(-1),
            "acc": self.acc[start_frame:end_frame],
            "gyro": self.gyro[start_frame:end_frame],
            "gyro_full": self.gyro[:end_frame],
            "gt_pos": self.gt_pos[start_frame + 1 : end_frame + 1].unsqueeze(0),
            "gt_rot": self.gt_rot[start_frame + 1 : end_frame + 1],
            "gt_vel": self.gt_vel[start_frame + 1 : end_frame + 1],
            "init_pos": self.gt_pos[start_frame][None, ...],
            "init_rot": self.gt_rot[start_frame:end_frame],
            "init_vel": self.gt_vel[start_frame][None, ...],
        }

    def get_init_value(self):
        return {
            "pos": self.gt_pos[:1],
            "rot": self.gt_rot[:1],
            "yaw": self.gt_yaw[0],
            "vel": self.gt_vel[:1],
        }

    def get_pos_meters(self, frame: int | None = None) -> np.ndarray:
        """Get position in meters (x, y) relative to start.

        Args:
            frame: Frame index. If None, returns current frame position.

        Returns:
            2D position array [x, y] in meters.
        """
        idx = self.frame_id if frame is None else frame
        return self.gt_pos_meters[idx].numpy()

    def _get_edge_data_for_bearing(self, node, neighbor):
        """Get edge data from original_graph.

        Returns: tuple of (edge_data, is_reversed) where is_reversed indicates
                 if the edge was found in reverse direction (neighbor -> node).
        """
        # For original graph without inserted nodes
        if node in self.original_graph.nodes and neighbor in self.original_graph.nodes:
            if self.original_graph.has_edge(node, neighbor):
                return self.original_graph[node][neighbor]

            if self.original_graph.has_edge(neighbor, node):
                return self.original_graph[neighbor][node]

        if node in self.graph.nodes and neighbor in self.graph.nodes:
            if self.graph.has_edge(node, neighbor):
                return self.graph[node][neighbor]

            if self.graph.has_edge(neighbor, node):
                return self.graph[neighbor][node]

        return None

    def _extract_bearing_from_geometry(self, node, edge_data) -> float | None:
        """Extract bearing from edge geometry using the first segment from the node.

        Args:
            node: The node we're calculating bearing from
            edge_data: The edge geometry data
            is_reversed: If True, the edge geometry is stored neighbor->node and needs reversal
        """
        if not edge_data or "geometry" not in edge_data:
            return None
        geom = edge_data["geometry"]

        if not hasattr(geom, "coords"):
            return None
        coords = list(geom.coords)
        if len(coords) < 2:
            return None

        # Get node position
        node_lat, node_lon = self.graph.nodes[node]["y"], self.graph.nodes[node]["x"]

        # Find which end of the geometry is closest to our node
        first_point = coords[0]
        last_point = coords[-1]
        dist_to_first = (node_lon - first_point[0]) ** 2 + (node_lat - first_point[1]) ** 2
        dist_to_last = (node_lon - last_point[0]) ** 2 + (node_lat - last_point[1]) ** 2

        # If node is closer to the end, reverse coords so node is at the start
        if dist_to_last < dist_to_first:
            coords = coords[::-1]

        if len(coords) > 2:
            # Find the first segment that is at least 3 meters long # Optimise
            accumulated_length = 0.0
            for i in range(1, len(coords)):
                pt1 = coords[i - 1]
                pt2 = coords[i]
                segment_length = haversine((pt1[1], pt1[0]), (pt2[1], pt2[0]), unit=Unit.METERS)
                accumulated_length += segment_length
                if accumulated_length >= 3.0:
                    next_point_lat, next_point_lon = pt2[1], pt2[0]
                    return calculate_bearing(node_lat, node_lon, next_point_lat, next_point_lon)

        # Now node is always at coords[0], use bearing from node to coords[1]
        next_point_lat, next_point_lon = coords[1][1], coords[1][0]
        return calculate_bearing(node_lat, node_lon, next_point_lat, next_point_lon)

    def _calculate_node_bearing(self, node, neighbor) -> float:
        """Calculate bearing from node to neighbor."""
        node_lat, node_lon = self.graph.nodes[node]["y"], self.graph.nodes[node]["x"]

        edge_data = self._get_edge_data_for_bearing(node, neighbor)

        bearing = self._extract_bearing_from_geometry(node, edge_data)

        if bearing is None:
            neighbor_lat = self.graph.nodes[neighbor]["y"]
            neighbor_lon = self.graph.nodes[neighbor]["x"]
            bearing = calculate_bearing(node_lat, node_lon, neighbor_lat, neighbor_lon)
        return bearing

    def _compute_node_yaws(self, node) -> dict:
        """Compute yaw angles for all neighbors of a node."""
        node_yaws = {}
        for neighbor in self.graph.neighbors(node):
            node_yaws[neighbor] = self._calculate_node_bearing(node, neighbor)
        return node_yaws

    def _plot_graph_with_bearings(self):
        """Plot the graph with bearing arrows for debugging."""
        fig, ax = plt.subplots(figsize=(12, 12))
        pos = {node: (data["x"], data["y"]) for node, data in self.graph.nodes(data=True)}
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color="gray", width=1)
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color="red", node_size=10)

        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
        arrow_length = extent * 0.015

        for node, data in self.graph.nodes(data=True):
            node_x, node_y = pos[node]
            yaws = data.get("yaws", {})
            for bearing in yaws.values():
                math_angle = np.pi / 2 - bearing
                dx = arrow_length * np.cos(math_angle)
                dy = arrow_length * np.sin(math_angle)
                ax.annotate(
                    "",
                    xy=(node_x + dx, node_y + dy),
                    xytext=(node_x, node_y),
                    arrowprops={"arrowstyle": "->", "color": "blue", "lw": 0.8, "alpha": 0.6},
                )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"KITTI Sequence {self.sequence} Road Network with Exit Bearings")
        fig.savefig(f"{self.output_dir}/kitti_sequence_{self.sequence}_graph.png", dpi=150)
        plt.close(fig)

    def setup_graph(self):
        g = ox.graph.graph_from_point(
            center_point=self.mid_coord,
            dist=500,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=False,
            custom_filter=None,
        )
        g = ox.projection.project_graph(g, to_latlong=True)
        g.remove_edges_from(nx.selfloop_edges(g))

        if self.debug:
            fig, ax = ox.plot_graph(
                g,
                figsize=(12, 12),
                bgcolor="#FFFFFF",
                node_size=10,
                node_color="#000000",
                edge_color="#444444",
                show=False,
                close=False,
            )
            ax.set_title(f"KITTI Sequence {self.sequence} Raw Road Network Graph")
            fig.savefig(f"{self.output_dir}/kitti_sequence_{self.sequence}_raw_graph.png", dpi=300)
            plt.close(fig)

        self.original_graph = g.copy()

        g = simplify_sharp_turns(g)

        self.graph = nx.Graph()
        for n in g.nodes(data=True):
            self.graph.add_node(n[0], x=n[1]["x"], y=n[1]["y"])
        for start, end in g.edges():
            self.graph.add_edge(start, end)

        node_list = list(self.graph.nodes)
        self.node_coords = {}
        for node in node_list:
            self.node_coords[node] = (
                float(self.graph.nodes[node]["y"]),
                float(self.graph.nodes[node]["x"]),
            )

        for node in tqdm(node_list, "Downloading Junction Data", position=0, total=len(node_list)):
            pos = (
                float(self.graph.nodes[node]["y"]),
                float(self.graph.nodes[node]["x"]),
            )
            sat_image = download_satmap(pos)
            self.graph.nodes[node]["sat_image"] = sat_image
            self.graph.nodes[node]["yaws"] = self._compute_node_yaws(node)

        if self.debug:
            print(
                f"Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges."
            )
            self._plot_graph_with_bearings()

    def download_streetview(self, node_coord, number=5):
        sv_images = streetview.search_panoramas(lat=node_coord[0], lon=node_coord[1])

        new_pan = []
        distances = []
        for pan in sv_images:
            if pan.heading is not None:  # Absolutely necessary
                dist = haversine((pan.lat, pan.lon), node_coord, unit=Unit.METERS)
                if dist < 30:
                    distances.append(dist)
                    new_pan.append(pan)
        # indices of the closest panoramas increasing distance
        distances = sorted(range(len(distances)), key=lambda k: distances[k])

        # At most 10
        if len(distances) > 10:
            distances = distances[:10]

        panos = []
        for pano_idx in distances:
            pano = new_pan[pano_idx]
            if Path(f"pano_temp/{pano.pano_id}.jpg").exists():
                continue
            try:
                image = streetview.get_panorama(
                    pano_id=pano.pano_id,
                    multi_threaded=True,
                )
                if isinstance(image, Image.Image):
                    image.save(f"pano_temp/{pano.pano_id}.jpg", "jpeg")
                    panos.append(pano.pano_id)
                    if len(panos) >= number:
                        return panos
            except Exception:
                print(f"Failed to download panorama {pano.pano_id} at {node_coord}.")
                continue
        return panos

    def get_gt_node(self, frame_idx, lookback_frames: int = 10):
        """Get the graph node that the vehicle has turned through.

        Uses a lookback strategy to find the node closest to an earlier frame,
        which is more robust for identifying the node the vehicle has already
        passed through during a turn.

        Args:
            frame_idx: Current frame index.
            lookback_frames: Number of frames to look back (default 10).

        Returns:
            The nearest node ID to the earlier frame position.
        """
        # Use an earlier frame to find the node we've already passed through
        lookback_idx = max(0, frame_idx - lookback_frames)
        coord = self.get_coord(lookback_idx)
        lat, lon = coord

        # Find nearest node using haversine distance
        min_dist = float("inf")
        nearest_node = None
        for node, data in self.graph.nodes(data=True):
            node_lat, node_lon = data["y"], data["x"]
            dist = haversine((lat, lon), (node_lat, node_lon), unit=Unit.METERS)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node


class KittiCVGL(Dataset):
    def __init__(self, args):
        self.kitti = Kitti(args)
        self.kitti.setup_graph()

        self.img_size = 384
        self.data = self.kitti.data
        self.graph = self.kitti.graph
        self.og_graph = self.kitti.original_graph

        for node in tqdm(
            self.graph.nodes,
            "Downloading Streetview Images",
            position=0,
            total=len(self.graph.nodes),
        ):
            node_pos = (self.graph.nodes[node]["y"], self.graph.nodes[node]["x"])
            panos = self.download_streetview(node_pos, number=5)
            self.og_graph.nodes[node]["streetview"] = panos

    def download_streetview(self, node_coord, number=5):
        sv_images = streetview.search_panoramas(lat=node_coord[0], lon=node_coord[1])
        new_pan = []
        distances = []
        for pan in sv_images:
            if pan.heading is not None:  # Absolutely necessary
                dist = haversine((pan.lat, pan.lon), node_coord, unit=Unit.METERS)
                if dist < 30:
                    distances.append(dist)
                    new_pan.append(pan)

        # indices of the closest panoramas increasing distance
        distances = sorted(range(len(distances)), key=lambda k: distances[k])

        # At most 10
        if len(distances) > 10:
            distances = distances[:10]

        panos = []
        for pano_idx in distances:
            pano = new_pan[pano_idx]
            if Path(f"pano_temp/{pano.pano_id}.jpg").exists():
                continue
            try:
                image = streetview.get_panorama(
                    pano_id=pano.pano_id,
                    multi_threaded=True,
                )
                if isinstance(image, Image.Image):
                    image.save(f"pano_temp/{pano.pano_id}.jpg", "jpeg")
                    panos.append(pano.pano_id)
                    if len(panos) >= number:
                        return panos
            except Exception as e:
                print(f"Failed to download panorama {pano.pano_id} at {node_coord}. Error: {e}")
                continue
        return panos

    def __len__(self):
        return len(self.kitti)

    def __getitem__(self, idx):
        _, pose, frame_id, yaw = self.kitti.get_next_data()
        colour_img = self.kitti.get_colour_img(frame_id)
        return {"frame": colour_img, "pose": pose, "frame_id": frame_id, "yaw": yaw}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KITTI CVGL Dataset")
    parser.add_argument("--sequence", type=int, default=0, help="Sequence number to load (0-10)")
    args = parser.parse_args()

    dataset = KittiCVGL(args)
    print(f"Loaded {len(dataset)} frames from KITTI sequence {args.sequence}.")
