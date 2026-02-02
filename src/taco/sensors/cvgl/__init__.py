"""CVGL (Computer Vision Global Localization) module."""

from .cvusa import CVUSADataset, CVUSADatasetConfig
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
]
