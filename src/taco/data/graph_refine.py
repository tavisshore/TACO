"""Graph refinement utilities for road network processing."""

import math
from typing import Any

import networkx as nx
import numpy as np


def simplify_sharp_turns(graph: nx.MultiDiGraph, angle_threshold: float = 30.0) -> nx.MultiDiGraph:
    """Simplify graph by removing nodes at sharp turns.

    This function processes a road network graph and merges nodes
    where the turn angle exceeds a threshold, creating smoother paths.

    Args:
        graph: NetworkX MultiDiGraph representing road network.
        angle_threshold: Minimum angle (degrees) to consider a turn sharp.

    Returns:
        Simplified graph with sharp turn nodes removed.
    """
    # Work on a copy to avoid modifying the original
    g = graph.copy()

    # Find nodes with exactly 2 edges (potential simplification candidates)
    nodes_to_check = [node for node in g.nodes() if g.degree(node) == 2]

    nodes_to_remove = []

    for node in nodes_to_check:
        # Get neighboring nodes
        neighbors = list(g.neighbors(node))
        if len(neighbors) != 2:
            continue

        n1, n2 = neighbors

        # Get node positions
        try:
            node_pos = (g.nodes[node]["y"], g.nodes[node]["x"])
            n1_pos = (g.nodes[n1]["y"], g.nodes[n1]["x"])
            n2_pos = (g.nodes[n2]["y"], g.nodes[n2]["x"])
        except KeyError:
            continue

        # Calculate angle at this node
        angle = calculate_turn_angle(n1_pos, node_pos, n2_pos)

        # If angle is close to 180 (straight), this node can be simplified
        if abs(180 - angle) < angle_threshold:
            nodes_to_remove.append(node)

    # Remove nodes and reconnect edges
    for node in nodes_to_remove:
        neighbors = list(g.neighbors(node))
        if len(neighbors) == 2:
            n1, n2 = neighbors
            # Add direct edge between neighbors
            if not g.has_edge(n1, n2):
                g.add_edge(n1, n2)
            g.remove_node(node)

    return g


def calculate_turn_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """Calculate the angle at p2 formed by p1-p2-p3.

    Args:
        p1: First point (lat, lon).
        p2: Middle point (lat, lon).
        p3: Third point (lat, lon).

    Returns:
        Angle in degrees (0-180).
    """
    # Convert to vectors
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    # Calculate dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 == 0 or mag2 == 0:
        return 180.0

    # Calculate angle
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    angle = math.degrees(math.acos(cos_angle))

    return angle


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2.

    Args:
        lat1, lon1: First point coordinates.
        lat2, lon2: Second point coordinates.

    Returns:
        Bearing in radians.
    """
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)

    delta_lambda = lambda2 - lambda1

    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    theta = math.atan2(x, y)
    bearing = (math.degrees(theta) + 360) % 360

    return math.radians(bearing)
