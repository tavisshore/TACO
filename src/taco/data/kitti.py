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

from ..utils.geometry import rotate_trajectory
from .camera_model import CameraModel
from .graph_refine import simplify_sharp_turns
from .sat import download_satmap


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
    # Calculate exit edges from each node
    edge_angles = {}
    for u, v, data in graph.edges(keys=False, data=True):
        if u not in edge_angles:
            edge_angles[u] = {}
        if v not in edge_angles:
            edge_angles[v] = {}
        # Edge - direction one
        lat1, lon1 = graph.nodes[u]["y"], graph.nodes[u]["x"]
        if "geometry" in data:
            lon2, lat2 = list(data["geometry"].coords)[1]  # 1st vertex after start
        else:
            lat2, lon2 = graph.nodes[v]["y"], graph.nodes[v]["x"]
        bear = calculate_bearing(lat1, lon1, lat2, lon2)
        edge_angles[u][v] = bear
        # Other end
        lat1, lon1 = graph.nodes[v]["y"], graph.nodes[v]["x"]
        if "geometry" in data:
            lon2, lat2 = list(data["geometry"].coords)[-2]  # Last vertex
        else:
            lat2, lon2 = graph.nodes[u]["y"], graph.nodes[u]["x"]
        bear = calculate_bearing(lat1, lon1, lat2, lon2)
        edge_angles[v][u] = bear
    return edge_angles


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
        self.gt_rot = pp.euler2SO3(
            torch.tensor(
                [
                    [self.data.oxts[i].packet.roll, self.data.oxts[i].packet.pitch, np.deg2rad(90)]
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

        # CVGL Compass
        self.yaws = [enu_yaw_to_compass_cw(self.data.oxts[i].packet.yaw) for i in seq_range]

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
        Takes index of current frame in sequence, get's coord, and return nearest node index from graph
        """
        coord = self.get_coord(frame_idx)
        lat, lon = coord
        nearest_node = ox.distance.nearest_nodes(
            self.original_graph, lon, lat
        )  # Note: lon, lat order
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

    def get_yaw(self, frame=None):
        return self.yaws[self.frame_id] if frame is None else self.yaws[frame]

    def get_yaws(self, frames=None):
        if frames is None:
            return self.yaws
        return [self.yaws[frame] for frame in frames]

    def get_coord(self, frame=None):
        return self.gt_coords[self.frame_id] if frame is None else self.gt_coords[frame]

    def get_start_coord(self):
        return self.gt_coords[0]

    def get_start_yaw(self):
        return self.yaws[0]

    def get_timestamp(self, frame=None):
        return (
            datetime.timestamp(self.data.timestamps[self.frame_id])
            if frame is None
            else datetime.timestamp(self.data.timestamps[frame])
        )

    # IMU
    def get_imu(self, i):
        start_frame = i - self.duration + 1
        end_frame = i + 1

        return {
            "dt": self.dt[start_frame:end_frame].unsqueeze(-1),
            "acc": self.acc[start_frame:end_frame],
            "gyro": self.gyro[start_frame:end_frame],
            "gt_pos": self.gt_pos[start_frame + 1 : end_frame + 1].unsqueeze(0),
            "gt_rot": self.gt_rot[start_frame + 1 : end_frame + 1],
            "gt_vel": self.gt_vel[start_frame + 1 : end_frame + 1],
            "init_pos": self.gt_pos[start_frame][None, ...],
            "init_rot": self.gt_rot[start_frame:end_frame],
            "init_vel": self.gt_vel[start_frame][None, ...],
        }

    def imu_init_rot(self):
        return self.yaws[0]  # * 2  # Right - why is it halved??

    def get_init_value(self):
        return {"pos": self.gt_pos[:1], "rot": self.gt_rot[:1], "vel": self.gt_vel[:1]}

    def get_pos_meters(self, frame: int | None = None) -> np.ndarray:
        """Get position in meters (x, y) relative to start.

        Args:
            frame: Frame index. If None, returns current frame position.

        Returns:
            2D position array [x, y] in meters.
        """
        idx = self.frame_id if frame is None else frame
        return self.gt_pos_meters[idx].numpy()

    def setup_graph(self):
        g = ox.graph.graph_from_point(
            center_point=self.mid_coord,
            dist=500,
            dist_type="bbox",
            network_type="drive",
            simplify=True,  # Only keeps junctions as nodes
            retain_all=False,  # Only keep the largest connected component
            truncate_by_edge=False,  # Basically keeps neighbouring nodes from outside the bbox
            custom_filter=None,
        )
        g = ox.projection.project_graph(g, to_latlong=True)
        g.remove_edges_from(nx.selfloop_edges(g))

        g = simplify_sharp_turns(g)

        self.edge_angles = calculate_bearings(g)
        self.original_graph = g
        # Create networkx graph
        self.graph = nx.Graph()
        for n in g.nodes(data=True):
            self.graph.add_node(n[0], x=n[1]["x"], y=n[1]["y"])
        for start, end in g.edges():
            self.graph.add_edge(start, end)
        node_list = list(self.graph.nodes)

        # Download lots of streetviews per satellite image patch to improve CVGL
        for node in tqdm(node_list, "Downloading Junction Data", position=0, total=len(node_list)):
            pos = (
                float(self.graph.nodes[node]["y"]),
                float(self.graph.nodes[node]["x"]),
            )  # (lat, lon)
            sat_image = download_satmap(pos)
            self.graph.nodes[node]["sat_image"] = sat_image  # Now just the path

            self.graph.nodes[node]["yaws"] = {
                n: self.edge_angles[node][n] for n in self.graph.neighbors(node)
            }

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
