"""Sensor data processing modules."""

from .imu import IMUData, IMUIntegrator, IMUPreintegrator

__all__ = ["IMUData", "IMUIntegrator", "IMUPreintegrator"]

# CVGL sensor - optional imports (require PyTorch)
try:
    from .cvgl import (
        CVGLLocalizer,
        CVGLMeasurement,
        ImageDatabaseDataset,
        ImageRetrievalModel,
        TripletDataset,
    )

    __all__.extend([
        "CVGLLocalizer",
        "CVGLMeasurement",
        "ImageRetrievalModel",
        "TripletDataset",
        "ImageDatabaseDataset",
    ])
except ImportError:
    # PyTorch not available, CVGL sensor not available
    pass
