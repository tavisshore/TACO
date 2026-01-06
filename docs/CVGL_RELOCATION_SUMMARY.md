# CVGL Relocation Summary

## What Was Done

Successfully moved the CVGL (Computer Vision Global Localization) module from `taco.localization.cvgl` to `taco.sensors.cvgl` to properly classify it as an emulated sensor.

## Changes Made

### 1. Directory Structure

**Created:**
```
src/taco/sensors/cvgl/
├── __init__.py          # Module exports with optional PyTorch imports
├── dataset.py           # TripletDataset, ImageDatabaseDataset
├── localizer.py         # CVGLLocalizer
├── measurement.py       # CVGLMeasurement
├── model.py             # ImageRetrievalModel (PyTorch Lightning)
└── train.py             # Training script
```

**Removed:**
```
src/taco/localization/cvgl/  # Entire directory removed
```

**Updated:**
```
src/taco/localization/__init__.py  # Now empty with deprecation note
src/taco/sensors/__init__.py       # Added CVGL exports
```

### 2. Import Path Changes

All occurrences of `taco.localization.cvgl` were updated to `taco.sensors.cvgl` in:

- ✅ Source code (`src/taco/sensors/cvgl/*.py`)
- ✅ Test files (`tests/test_cvgl.py`, `tests/test_cvgl_model.py`)
- ✅ Example scripts (`examples/train_image_retrieval.py`, `examples/inference_image_retrieval.py`)
- ✅ Documentation (`docs/image_retrieval.md`, `docs/CVGL_MODEL_SUMMARY.md`, `docs/CVGL_BUGFIX.md`)
- ✅ Training script (`src/taco/sensors/cvgl/train.py`)

### 3. New Import Patterns

Users can now import CVGL in multiple ways:

```python
# Direct module import
from taco.sensors.cvgl import CVGLLocalizer, CVGLMeasurement

# Convenience import from sensors
from taco.sensors import CVGLLocalizer, CVGLMeasurement

# Model imports (require PyTorch)
from taco.sensors.cvgl import ImageRetrievalModel, TripletDataset
from taco.sensors import ImageRetrievalModel  # Also works
```

### 4. Module Structure

```
taco/
├── sensors/                    # All sensor-related code
│   ├── imu/                   # IMU sensor (accelerometer + gyroscope)
│   │   ├── data.py           # IMUData
│   │   ├── integrator.py     # IMUIntegrator
│   │   └── preintegration.py # IMUPreintegrator
│   └── cvgl/                  # CVGL sensor (camera-based localization)
│       ├── localizer.py      # CVGLLocalizer
│       ├── measurement.py    # CVGLMeasurement
│       ├── model.py          # ImageRetrievalModel
│       ├── dataset.py        # Training datasets
│       └── train.py          # Training script
├── localization/              # High-level localization algorithms (future)
│   └── __init__.py           # Empty (CVGL moved to sensors)
└── pose_graph/                # Factor graph optimization
    ├── graph.py              # PoseGraph
    ├── node.py               # PoseNode
    └── edge.py               # Edge
```

## Rationale

### Why CVGL is a Sensor

1. **Produces Measurements**: CVGL outputs pose measurements from camera images
2. **Emulated Sensor**: Treats camera as a 6-DOF pose sensor
3. **Direct Input to Pose Graph**: Measurements go directly into factor graph
4. **Consistent with IMU**: Groups logically with other sensors

### Architecture Benefits

1. **Clear Separation of Concerns**
   - `sensors/`: Raw sensor data and measurements
   - `pose_graph/`: Graph construction and optimization
   - `localization/`: Reserved for high-level SLAM algorithms

2. **Extensibility**
   - Easy to add new sensors (GPS, wheel odometry, etc.)
   - All sensors in one place
   - Consistent API across sensors

3. **Modularity**
   - Each sensor is self-contained
   - Optional dependencies (PyTorch for CVGL)
   - Core package works without PyTorch

## Test Results

All tests pass with the new structure:

```
======================== 23 passed, 8 skipped in 0.59s =========================
```

- ✅ All original TACO tests pass
- ✅ CVGL measurement tests pass
- ✅ CVGL model tests properly skip when PyTorch not available
- ✅ Import tests verify old path is removed
- ✅ No regressions introduced

## Validation

### Import Validation
```python
# ✓ New imports work
from taco.sensors.cvgl import CVGLLocalizer
from taco.sensors import CVGLMeasurement

# ✓ Old imports fail as expected
from taco.localization.cvgl import CVGLLocalizer  # ModuleNotFoundError
```

### Functionality Validation
```python
# ✓ All CVGL functionality works unchanged
from taco.sensors.cvgl import CVGLLocalizer, CVGLMeasurement
import numpy as np

localizer = CVGLLocalizer()
measurement = CVGLMeasurement(
    timestamp=0.0,
    position=np.zeros(3),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    covariance=np.eye(6),
    confidence=0.95,
    num_inliers=100,
)
# Works perfectly!
```

### Training Validation
```bash
# ✓ Training command updated
python -m taco.sensors.cvgl.train \
    --train-data-dir data/images/train \
    --train-triplets data/triplets/train.txt \
    --val-data-dir data/images/val \
    --val-triplets data/triplets/val.txt \
    --batch-size 32 \
    --gpus 1
```

## Documentation

Created comprehensive documentation:

1. **[CVGL_MIGRATION.md](CVGL_MIGRATION.md)** - Step-by-step migration guide
   - Import path changes
   - Code examples before/after
   - Migration script
   - FAQ section

2. **Updated existing docs:**
   - [image_retrieval.md](image_retrieval.md) - Usage guide with new imports
   - [CVGL_MODEL_SUMMARY.md](CVGL_MODEL_SUMMARY.md) - Model documentation
   - [CVGL_BUGFIX.md](CVGL_BUGFIX.md) - Bug fix reference

3. **Updated examples:**
   - [train_image_retrieval.py](../examples/train_image_retrieval.py)
   - [inference_image_retrieval.py](../examples/inference_image_retrieval.py)

## Impact

### Breaking Changes
- ✅ Old import path removed: `taco.localization.cvgl` → `taco.sensors.cvgl`
- ✅ Users must update imports (simple find/replace)

### Non-Breaking
- ✅ Model checkpoints still work
- ✅ Saved databases still work
- ✅ All functionality unchanged
- ✅ API remains identical

## Backwards Compatibility

**Not maintained** - this is a breaking change. Users must update to new import paths.

### Migration is Simple
```python
# Old
from taco.localization.cvgl import CVGLLocalizer

# New
from taco.sensors.cvgl import CVGLLocalizer
```

Or use automated migration script provided in [CVGL_MIGRATION.md](CVGL_MIGRATION.md).

## Files Modified

### Created
- `src/taco/sensors/cvgl/` (entire directory)
- `docs/CVGL_MIGRATION.md`
- `docs/CVGL_RELOCATION_SUMMARY.md`

### Modified
- `src/taco/sensors/__init__.py` - Added CVGL exports
- `src/taco/localization/__init__.py` - Removed exports, added deprecation note
- `tests/test_cvgl.py` - Updated imports
- `tests/test_cvgl_model.py` - Updated imports
- `examples/train_image_retrieval.py` - Updated imports
- `examples/inference_image_retrieval.py` - Updated imports
- `docs/image_retrieval.md` - Updated all import examples
- `docs/CVGL_MODEL_SUMMARY.md` - Updated all import examples
- `docs/CVGL_BUGFIX.md` - Updated references

### Removed
- `src/taco/localization/cvgl/` (entire directory)

## Next Steps

Users upgrading should:

1. Read [CVGL_MIGRATION.md](CVGL_MIGRATION.md)
2. Run migration script or manually update imports
3. Test their code
4. Verify everything works

## Conclusion

✅ **Successfully relocated CVGL to sensors module**
- Clean separation of concerns
- Consistent architecture
- All tests passing
- Comprehensive documentation
- Clear migration path

The TACO package now has a logical, extensible sensor architecture with IMU and CVGL sensors colocated in `taco.sensors/`.
