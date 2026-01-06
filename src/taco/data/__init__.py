"""Data loading modules for various datasets."""

from .camera_model import CameraModel

__all__ = ["CameraModel"]

# KITTI data loader - optional imports (require pykitti, pypose, osmnx, etc.)
try:
    from .kitti import Kitti, KittiCVGL

    __all__.extend(["Kitti", "KittiCVGL"])
except ImportError:
    # KITTI dependencies not available
    pass
