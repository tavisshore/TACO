"""Satellite map downloading utilities."""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Optional PIL import
try:
    from PIL import Image
except ImportError:
    Image = None

# Optional requests import
try:
    import requests
except ImportError:
    requests = None


# Cache directory for satellite images
CACHE_DIR = Path("sat_cache")


def download_satmap(
    coord: Tuple[float, float],
    zoom: int = 18,
    size: Tuple[int, int] = (640, 640),
    cache_dir: Optional[Path] = None,
) -> str:
    """Download satellite map image for a given coordinate.

    Args:
        coord: (latitude, longitude) tuple.
        zoom: Zoom level (higher = more detail).
        size: Image size (width, height).
        cache_dir: Directory to cache images.

    Returns:
        Path to the downloaded/cached image file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create unique filename from coordinates
    lat, lon = coord
    coord_hash = hashlib.md5(f"{lat:.6f},{lon:.6f},{zoom}".encode()).hexdigest()[:12]
    cache_path = cache_dir / f"sat_{coord_hash}.jpg"

    # Return cached image if exists
    if cache_path.exists():
        return str(cache_path)

    # Try to download from tile servers (placeholder implementation)
    # In production, this would use Google Maps, Mapbox, or similar API
    if requests is not None and Image is not None:
        try:
            # This is a placeholder - actual implementation would use a real tile server
            # For now, create a placeholder image
            img = Image.new("RGB", size, color=(100, 120, 100))
            img.save(cache_path, "JPEG")
            return str(cache_path)
        except Exception as e:
            print(f"Failed to download satellite image: {e}")

    # Fallback: create a placeholder image
    if Image is not None:
        img = Image.new("RGB", size, color=(100, 120, 100))
        img.save(cache_path, "JPEG")

    return str(cache_path)


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
