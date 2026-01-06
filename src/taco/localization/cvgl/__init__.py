"""CVGL (Computer Vision Global Localization) module."""

from .localizer import CVGLLocalizer
from .measurement import CVGLMeasurement

__all__ = [
    "CVGLLocalizer",
    "CVGLMeasurement",
]

# Optional imports (require PyTorch)
try:
    from .dataset import ImageDatabaseDataset, TripletDataset
    from .model import ImageRetrievalModel

    __all__.extend([
        "ImageRetrievalModel",
        "TripletDataset",
        "ImageDatabaseDataset",
    ])
except ImportError:
    # PyTorch not available, skip model imports
    pass
