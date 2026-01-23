"""Graph refinement utilities for road network processing."""

from __future__ import annotations

import math
import uuid
from copy import deepcopy
from itertools import pairwise
from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point


def _edge_length(ls: LineString) -> float:
    # Length in coordinate units. If your graph is projected (meters), great.
    # If lat/lon, you'll get degrees. Replace with geodesic if you need meters.
    return float(ls.length)


def _wrap180(deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    x = (deg + 180.0) % 360.0 - 180.0
    # Map -180 to +180 for consistency
    return 180.0 if np.isclose(x, -180.0) else x


def _segment_headings(coords: List[Tuple[float, float]]) -> np.ndarray:
    vecs = np.diff(np.asarray(coords, dtype=float), axis=0)
    headings = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    return headings  # length = n_seg


def _turn_deltas(headings: np.ndarray) -> np.ndarray:
    deltas = np.diff(headings)
    wrapped = np.vectorize(_wrap180)(deltas)
    return wrapped  # length = n_seg-1, each at an interior vertex


def _new_node_id(G: nx.Graph):
    # Try to keep node id type consistent if nodes are integers
    try:
        ints = [
            int(n) for n in G.nodes if isinstance(n, int | np.integer | str) and str(n).isdigit()
        ]
        if len(ints) == G.number_of_nodes():
            return max(ints) + 1
    except Exception:
        pass
    return f"apex_{uuid.uuid4().hex[:8]}"


def _line_intersection(p1, p2, p3, p4) -> Tuple[float, float] | None:
    """
    Calculate intersection point of two lines.
    Line 1: p1 -> p2
    Line 2: p3 -> p4
    Returns the intersection point or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:  # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)


def _as_linestring(data) -> LineString | None:
    geom = data.get("geometry", None)
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        # flatten into one by concatenating parts
        coords = []
        for i, part in enumerate(geom.geoms):
            if i == 0:
                coords.extend(list(part.coords))
            else:
                # avoid duplicating the joint node
                part_coords = list(part.coords)
                if coords and part_coords and coords[-1] == part_coords[0]:
                    coords.extend(part_coords[1:])
                else:
                    coords.extend(part_coords)
        return LineString(coords)
    return None


def _edge_coords(u, v, data, G: nx.MultiDiGraph) -> List[Tuple[float, float]]:
    ls = _as_linestring(data)
    if ls is not None:
        # drop duplicate consecutive points
        coords = list(ls.coords)
        dedup = [coords[0]]
        for p in coords[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        return dedup
    # fallback: straight line u->v from node positions
    ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
    vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
    return [(ux, uy), (vx, vy)]


def _group_turn_runs(idxs: np.ndarray, signs: np.ndarray) -> List[Tuple[int, int, float]]:
    """Group consecutive significant deltas into runs of the same sign."""
    groups = []
    run_start = idxs[0]
    run_sign = signs[0]
    prev_i = idxs[0]
    for k in range(1, len(idxs)):
        i = idxs[k]
        s = signs[k]
        contiguous = i == prev_i + 1
        same_sign = s == run_sign
        if not (contiguous and same_sign):
            groups.append((run_start, prev_i, run_sign))
            run_start = i
            run_sign = s
        prev_i = i
    groups.append((run_start, prev_i, run_sign))
    return groups


def _find_apex_indices(
    groups: List[Tuple[int, int, float]], deltas: np.ndarray, min_total_turn_deg: float
) -> Tuple[List[int], List[float]]:
    """For each run, check cumulative turn and pick apex (global delta index)."""
    apex_delta_indices = []
    apex_turns_abs = []
    for start, end, _ in groups:
        run_slice = deltas[start : end + 1]
        total_signed = np.sum(run_slice)
        if abs(total_signed) >= min_total_turn_deg:
            local_apex = int(np.argmax(np.abs(run_slice)))
            apex_idx = start + local_apex
            apex_delta_indices.append(apex_idx)
            apex_turns_abs.append(abs(total_signed))
    return apex_delta_indices, apex_turns_abs


def _merge_close_apexes(
    apex_delta_indices: List[int], apex_turns_abs: List[float], min_apex_vertex_gap: int
) -> List[int]:
    """Merge apexes that are too close (by vertex index distance)."""
    apex_pairs = sorted(zip(apex_delta_indices, apex_turns_abs, strict=False), key=lambda x: x[0])
    merged = []
    for idx, turn_abs in apex_pairs:
        if not merged:
            merged.append([idx, turn_abs])
            continue
        last_idx, last_turn = merged[-1]
        if idx - last_idx <= min_apex_vertex_gap:
            if turn_abs > last_turn:
                merged[-1] = [idx, turn_abs]
        else:
            merged.append([idx, turn_abs])
    return [m[0] for m in merged]


def _calculate_apex_position(
    apex_idx: int, groups: List[Tuple[int, int, float]], coords: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """Calculate proper apex position as intersection of incoming/outgoing lines."""
    run_start_idx: int | None = None
    run_end_idx: int | None = None
    for start, end, _ in groups:
        if start <= apex_idx <= end:
            run_start_idx = start
            run_end_idx = end
            break

    if run_start_idx is None or run_end_idx is None:
        return coords[apex_idx + 1]

    incoming_p1_idx = run_start_idx
    incoming_p2_idx = run_start_idx + 1
    outgoing_p1_idx = run_end_idx + 1
    outgoing_p2_idx = run_end_idx + 2

    if (
        incoming_p1_idx < 0
        or incoming_p2_idx >= len(coords)
        or outgoing_p1_idx >= len(coords)
        or outgoing_p2_idx >= len(coords)
    ):
        return coords[apex_idx + 1]

    p1_in = coords[incoming_p1_idx]
    p2_in = coords[incoming_p2_idx]
    p1_out = coords[outgoing_p1_idx]
    p2_out = coords[outgoing_p2_idx]

    intersection = _line_intersection(p1_in, p2_in, p1_out, p2_out)
    return intersection if intersection is not None else coords[apex_idx + 1]


def _split_geometry_at_apexes(
    coords: List[Tuple[float, float]],
    apex_coord_indices: List[int],
    apex_points: List[Tuple[float, float]],
    chain_nodes: List,
    node_xy: dict,
) -> List[LineString]:
    """Split the original edge geometry at apex points."""
    chain_geoms = []
    split_indices = [0, *apex_coord_indices, len(coords) - 1]

    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]

        segment_coords = coords[start_idx : end_idx + 1].copy()

        if i > 0:
            segment_coords[0] = apex_points[i - 1]
        if i < len(apex_points):
            segment_coords[-1] = apex_points[i]

        if len(segment_coords) >= 2:
            chain_geoms.append(LineString(segment_coords))
        else:
            chain_geoms.append(LineString([node_xy[chain_nodes[i]], node_xy[chain_nodes[i + 1]]]))

    return chain_geoms


def simplify_sharp_turns(
    G: nx.MultiDiGraph,
    *,
    min_total_turn_deg: float = 30.0,
    min_segment_turn_deg: float = 3.0,
    min_apex_vertex_gap: int = 1,
    inplace: bool = False,
    mark_apex_edges: bool = True,
    update_raw_graph: bool = True,
) -> nx.MultiDiGraph:
    """
    Detects one or more sharp, sign-consistent turn runs inside each edge geometry.
    For every qualifying run, inserts a node at the run's apex (max |delta|) and
    replaces the edge with multiple straight segments (u -> apex1 -> apex2 -> ... -> v).

    Args:
        G: Input MultiDiGraph.
        min_total_turn_deg: Minimum cumulative turn angle to detect a corner.
        min_segment_turn_deg: Minimum per-segment turn to consider significant.
        min_apex_vertex_gap: Minimum vertex gap between detected apexes.
        inplace: If True, modify G in place.
        mark_apex_edges: If True, mark created edges with 'apex_simplified' attribute.
        update_raw_graph: If True, also add apex nodes to the raw graph with split geometries.

    Returns:
        Modified graph with apex nodes inserted at sharp turns.
    """
    H = G if inplace else G.copy()
    edges = list(H.edges(keys=True, data=True))

    for u, v, key, data in edges:
        coords = _edge_coords(u, v, data, H)
        if len(coords) < 3:
            continue

        headings = _segment_headings(coords)
        deltas = _turn_deltas(headings)

        mask = np.abs(deltas) >= min_segment_turn_deg
        if not np.any(mask):
            continue

        idxs = np.nonzero(mask)[0]
        signs = np.sign(deltas[mask])

        groups = _group_turn_runs(idxs, signs)
        apex_delta_indices, apex_turns_abs = _find_apex_indices(groups, deltas, min_total_turn_deg)

        if not apex_delta_indices:
            continue

        apex_delta_indices = _merge_close_apexes(
            apex_delta_indices, apex_turns_abs, min_apex_vertex_gap
        )
        apex_coord_indices = [i + 1 for i in apex_delta_indices]

        apex_points = [
            _calculate_apex_position(apex_idx, groups, coords) for apex_idx in apex_delta_indices
        ]

        created_nodes = []
        for pt in apex_points:
            node_id = _new_node_id(H)
            H.add_node(node_id, x=float(pt[0]), y=float(pt[1]))
            created_nodes.append(node_id)

        chain_nodes = [u, *created_nodes, v]
        node_xy = {n: (H.nodes[n]["x"], H.nodes[n]["y"]) for n in chain_nodes}

        chain_geoms = []
        if update_raw_graph and len(created_nodes) > 0 and len(coords) > 2:
            chain_geoms = _split_geometry_at_apexes(
                coords, apex_coord_indices, apex_points, chain_nodes, node_xy
            )

        H.remove_edge(u, v, key)

        base_attrs = {k: val for k, val in data.items() if k != "geometry"}
        if mark_apex_edges:
            base_attrs["apex_simplified"] = True
            base_attrs["apex_count"] = len(created_nodes)

        for i, (a, b) in enumerate(pairwise(chain_nodes)):
            attrs = dict(base_attrs)
            geom = chain_geoms[i] if i < len(chain_geoms) else LineString([node_xy[a], node_xy[b]])
            attrs["geometry"] = geom
            attrs["length"] = _edge_length(geom)
            H.add_edge(a, b, **attrs)

    return H


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
