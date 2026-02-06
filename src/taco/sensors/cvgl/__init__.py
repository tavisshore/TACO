"""CVGL (Computer Vision Global Localization) module."""

from .callbacks import ProgressiveAugmentationCallback
from .cvusa import (
    CVUSADataset,
    CVUSADatasetConfig,
    ProgressiveAugmentation,
    create_default_augmentations,
    create_synchronized_augmentations,
)
from .kitti import KITTIValDataset, KITTIValDatasetConfig
from .localizer import CVGLLocalizer
from .measurement import CVGLMeasurement
from .model import (
    ImageRetrievalModel,
    ImageRetrievalModelConfig,
    create_convnext_encoder,
    create_sample4geo_encoder,
)

__all__ = [
    "CVGLLocalizer",
    "CVGLMeasurement",
    "CVUSADataset",
    "CVUSADatasetConfig",
    "ImageRetrievalModel",
    "ImageRetrievalModelConfig",
    "create_sample4geo_encoder",
    "create_convnext_encoder",
    "KITTIValDataset",
    "KITTIValDatasetConfig",
    "ProgressiveAugmentation",
    "ProgressiveAugmentationCallback",
    "create_default_augmentations",
    "create_synchronized_augmentations",
]
