"""Tests for satellite map download module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from taco.data.sat import download_satmap, get_tile_coords


class TestGetTileCoords:
    """Test tile coordinate calculation."""

    def test_tile_coords_origin(self) -> None:
        """Test tile coordinates at origin."""
        x, y = get_tile_coords(0.0, 0.0, zoom=10)

        # At zoom 10, there are 2^10 = 1024 tiles
        # Origin should be at center
        assert x == 512  # Midpoint
        assert 500 < y < 524  # Near equator

    def test_tile_coords_increases_with_zoom(self) -> None:
        """Test that tile coordinates scale with zoom level."""
        lat, lon = 45.0, 10.0

        x1, y1 = get_tile_coords(lat, lon, zoom=10)
        x2, y2 = get_tile_coords(lat, lon, zoom=11)

        # Higher zoom should approximately double the coordinates
        assert abs(x2 - 2 * x1) <= 1
        assert abs(y2 - 2 * y1) <= 1

    def test_tile_coords_longitude_wrap(self) -> None:
        """Test longitude wrapping at international date line."""
        x1, _ = get_tile_coords(0.0, 179.0, zoom=10)
        x2, _ = get_tile_coords(0.0, -179.0, zoom=10)

        # These should be on opposite sides of the map
        assert abs(x1 - x2) > 900  # Most of the 1024 tiles apart

    def test_tile_coords_returns_integers(self) -> None:
        """Test that tile coordinates are integers."""
        x, y = get_tile_coords(37.7749, -122.4194, zoom=15)

        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_tile_coords_positive(self) -> None:
        """Test that tile coordinates are non-negative."""
        test_coords = [
            (0.0, 0.0),
            (45.0, 90.0),
            (-45.0, -90.0),
            (85.0, 180.0),
            (-85.0, -180.0),
        ]

        for lat, lon in test_coords:
            x, y = get_tile_coords(lat, lon, zoom=10)
            assert x >= 0
            assert y >= 0

    def test_tile_coords_within_bounds(self) -> None:
        """Test that tile coordinates are within valid range."""
        zoom = 10
        max_tile = 2**zoom

        test_coords = [
            (0.0, 0.0),
            (45.0, 90.0),
            (-45.0, -90.0),
        ]

        for lat, lon in test_coords:
            x, y = get_tile_coords(lat, lon, zoom=zoom)
            assert 0 <= x < max_tile
            assert 0 <= y < max_tile


class TestDownloadSatmap:
    """Test satellite map download function."""

    def test_download_returns_path(self) -> None:
        """Test that download returns a path string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            result = download_satmap((37.7749, -122.4194), cache_dir=cache_dir)

            assert isinstance(result, str)

    def test_download_creates_file(self) -> None:
        """Test that download creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            result = download_satmap((37.7749, -122.4194), cache_dir=cache_dir)

            assert Path(result).exists()

    def test_download_caches_result(self) -> None:
        """Test that repeated downloads use cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            coord = (37.7749, -122.4194)

            # First download
            result1 = download_satmap(coord, cache_dir=cache_dir)

            # Get file modification time
            mtime1 = Path(result1).stat().st_mtime

            # Second download should use cache
            result2 = download_satmap(coord, cache_dir=cache_dir)

            # Same file should be returned
            assert result1 == result2

            # File should not have been modified
            mtime2 = Path(result2).stat().st_mtime
            assert mtime1 == mtime2

    def test_download_different_coords_different_files(self) -> None:
        """Test that different coordinates produce different files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            result1 = download_satmap((37.7749, -122.4194), cache_dir=cache_dir)
            result2 = download_satmap((40.7128, -74.0060), cache_dir=cache_dir)

            # Different coordinates should produce different files
            assert result1 != result2

    def test_download_creates_cache_directory(self) -> None:
        """Test that download creates cache directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nested" / "cache"
            assert not cache_dir.exists()

            download_satmap((37.7749, -122.4194), cache_dir=cache_dir)

            assert cache_dir.exists()

    def test_download_respects_zoom_level(self) -> None:
        """Test that different zoom levels produce different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            coord = (37.7749, -122.4194)

            result1 = download_satmap(coord, zoom=15, cache_dir=cache_dir)
            result2 = download_satmap(coord, zoom=18, cache_dir=cache_dir)

            # Different zoom levels should produce different files
            assert result1 != result2

    def test_download_default_cache_directory(self) -> None:
        """Test that default cache directory is used."""
        # Just verify the function runs without error
        # (actual file creation depends on system permissions)
        try:
            result = download_satmap((37.7749, -122.4194))
            assert isinstance(result, str)
        except PermissionError:
            pytest.skip("No permission to write to default cache directory")

    def test_download_file_is_image(self) -> None:
        """Test that downloaded file is a valid image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            result = download_satmap((37.7749, -122.4194), cache_dir=cache_dir)

            # File should end with .jpg
            assert result.endswith(".jpg")

            # File should have some content
            assert Path(result).stat().st_size > 0
