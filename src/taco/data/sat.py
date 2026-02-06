"""Satellite map downloading utilities."""

from __future__ import annotations

import hashlib
import math
import random
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
import requests
import utm
from haversine import Unit, haversine
from PIL import Image
from streetview import get_panorama, search_panoramas
from tqdm import tqdm

if TYPE_CHECKING:
    pass

# Disable PIL image size limits
Image.MAX_IMAGE_PIXELS = None


# Cache directory for satellite images
CACHE_DIR = Path("sat_cache")

# Constants for graph dataset construction
DATE_LIMIT = datetime.strptime("2025-01", "%Y-%m")

HEADERS = {
    "cache-control": "max-age=0",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36",
}


def crop_image_only_outside(img, tol=0):
    """
    Removes black spaces in images
    """
    img = np.array(img)
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    img = img[row_start:row_end, col_start:col_end]
    # resize to original size
    img = Image.fromarray(img)
    img = img.resize((n, m))
    return img


def download_tile(url, headers, channels):
    """Download a single map tile from a URL."""
    response = requests.get(url, headers=headers)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)

    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)


def project_with_scale(lat, lon, scale):
    """Mercator projection with scale.

    See: https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
    """
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    zoom: int,
    url: str,
    headers: dict,
    tile_size: int = 256,
    channels: int = 3,
) -> np.ndarray:
    """Download a map image by stitching together tiles from a tile server.

    Args:
        lat1: Latitude of top-left corner
        lon1: Longitude of top-left corner
        lat2: Latitude of bottom-right corner
        lon2: Longitude of bottom-right corner
        zoom: Zoom level
        url: Tile server URL template with {x}, {y}, {z} placeholders
        headers: HTTP headers for requests
        tile_size: Size of each tile in pixels (default: 256)
        channels: Number of image channels (default: 3 for RGB)

    Returns:
        Downloaded image as numpy array
    """
    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)

            if tile is not None:
                # Find the pixel coordinates of the new tile relative to the image
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size

                # Define where the tile will be placed on the image
                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)

                # Define how border tiles will be cropped
                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]

    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        thread = threading.Thread(target=build_row, args=[tile_y])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return img


def image_size(lat1: float, lon1: float, lat2: float, lon2: float, zoom: int, tile_size: int = 256):
    """Calculates the size of an image without downloading it.

    Args:
        lat1: Latitude of top-left corner
        lon1: Longitude of top-left corner
        lat2: Latitude of bottom-right corner
        lon2: Longitude of bottom-right corner
        zoom: Zoom level
        tile_size: Size of each tile in pixels (default: 256)

    Returns:
        Tuple of (width, height) in pixels
    """
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y


def download_satmap(
    coord: Tuple[float, float],
    cache_dir: Path = Path(CACHE_DIR),
    zoom: int = 20,
) -> Path:
    """Download satellite map image for a given coordinate.

    Downloads high quality satellite images as north-aligned tiles.

    Args:
        coord: (latitude, longitude) tuple.
        cache_dir: Directory where images will be saved in satellite/ subdirectory.
        zoom: Zoom level (higher = more detail). Default is 20.

    Returns:
        Path to the downloaded/cached image file.
    """

    image_name = Path(f"{coord[0]}_{coord[1]}_{zoom}.jpg")
    sat_path = cache_dir / "sat"
    if not sat_path.exists():
        sat_path.mkdir(parents=True, exist_ok=True)
    image_path = sat_path / image_name

    # Return cached image if exists
    if image_path.is_file():
        return image_name

    # Convert to UTM coordinates for metric-based bounding box
    lat, lon, zn, zl = utm.from_latlon(coord[0], coord[1])

    # Get satellite images that cover area + 100m each side (50m buffer on all sides)
    lat_min = lat - 50
    lat_max = lat + 50
    lon_min = lon - 50
    lon_max = lon + 50

    # Convert back to lat/lon
    lat_min, lon_min = utm.to_latlon(lat_min, lon_min, zn, zl)
    lat_max, lon_max = utm.to_latlon(lat_max, lon_max, zn, zl)

    tl = (lat_max, lon_min)
    br = (lat_min, lon_max)

    lat1, lon1 = re.findall(r"[+-]?\d*\.\d+|d+", str(tl))
    lat2, lon2 = re.findall(r"[+-]?\d*\.\d+|d+", str(br))
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)

    # Download satellite image from Google Maps tile server
    img = download_image(
        lat1,
        lon1,
        lat2,
        lon2,
        zoom,
        "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        HEADERS,
        256,
        3,
    )

    # Convert BGR to RGB
    img = img[:, :, ::-1]
    img = Image.fromarray(img)
    img = img.resize((512, 512))

    img.save(image_path)
    img.close()

    return image_name


def get_tile_coords(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        zoom: Zoom level.

    Returns:
        (tile_x, tile_y) tuple.
    """
    n = 2**zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = np.radians(lat)
    y = int((1 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2 * n)
    return x, y
