# CVGL Migration Guide

## Overview

The CVGL (Computer Vision Global Localization) module has been moved from `taco.localization.cvgl` to `taco.sensors.cvgl` to better reflect its role as an emulated sensor in the TACO system.

## What Changed

### Module Location

**Before:**
```python
from taco.localization.cvgl import CVGLLocalizer, CVGLMeasurement
from taco.localization.cvgl import ImageRetrievalModel, TripletDataset
```

**After:**
```python
from taco.sensors.cvgl import CVGLLocalizer, CVGLMeasurement
from taco.sensors.cvgl import ImageRetrievalModel, TripletDataset
```

### Directory Structure

**Before:**
```
src/taco/
├── localization/
│   └── cvgl/
│       ├── __init__.py
│       ├── localizer.py
│       ├── measurement.py
│       ├── model.py
│       ├── dataset.py
│       └── train.py
├── sensors/
│   └── imu/
│       └── ...
```

**After:**
```
src/taco/
├── localization/
│   └── __init__.py  (empty, with deprecation notice)
├── sensors/
│   ├── imu/
│   │   └── ...
│   └── cvgl/
│       ├── __init__.py
│       ├── localizer.py
│       ├── measurement.py
│       ├── model.py
│       ├── dataset.py
│       └── train.py
```

## Migration Steps

### 1. Update Import Statements

Find and replace all occurrences of `taco.localization.cvgl` with `taco.sensors.cvgl`:

```bash
# Using sed (Linux/Mac)
find . -type f -name "*.py" -exec sed -i 's/taco\.localization\.cvgl/taco.sensors.cvgl/g' {} +

# Or manually update each file
```

### 2. Update Training Commands

**Before:**
```bash
python -m taco.localization.cvgl.train \
    --train-data-dir data/images/train \
    ...
```

**After:**
```bash
python -m taco.sensors.cvgl.train \
    --train-data-dir data/images/train \
    ...
```

### 3. Update Code Examples

#### Basic Usage

**Before:**
```python
from taco.localization.cvgl import CVGLLocalizer, CVGLMeasurement

localizer = CVGLLocalizer()
measurement = localizer.localize(image, timestamp=0.0)
```

**After:**
```python
from taco.sensors.cvgl import CVGLLocalizer, CVGLMeasurement

localizer = CVGLLocalizer()
measurement = localizer.localize(image, timestamp=0.0)
```

#### Model Training

**Before:**
```python
from taco.localization.cvgl import ImageRetrievalModel, TripletDataset
import lightning as L

model = ImageRetrievalModel(embedding_dim=512)
dataset = TripletDataset(...)
```

**After:**
```python
from taco.sensors.cvgl import ImageRetrievalModel, TripletDataset
import lightning as L

model = ImageRetrievalModel(embedding_dim=512)
dataset = TripletDataset(...)
```

### 4. Update Test Imports

**Before:**
```python
from taco.localization.cvgl import CVGLMeasurement
```

**After:**
```python
from taco.sensors.cvgl import CVGLMeasurement
```

## Convenience Imports

You can now import CVGL components directly from `taco.sensors`:

```python
# New convenience imports
from taco.sensors import CVGLLocalizer, CVGLMeasurement
from taco.sensors import ImageRetrievalModel  # Requires PyTorch

# Or import from specific module
from taco.sensors.cvgl import CVGLLocalizer
```

## Why This Change?

### 1. **Conceptual Clarity**
   - CVGL provides sensor measurements (pose estimates from images)
   - It's an "emulated sensor" - treats camera as a 6-DOF pose sensor
   - Groups naturally with other sensors like IMU

### 2. **Consistent Architecture**
   ```
   taco.sensors/
   ├── imu/          # IMU sensor (accelerometer + gyroscope)
   ├── cvgl/         # CVGL sensor (camera-based localization)
   └── ...           # Future sensors (GPS, wheel odometry, etc.)
   ```

### 3. **Cleaner Separation**
   - `sensors/`: Raw sensor data and measurements
   - `localization/`: High-level localization algorithms (future)
   - `pose_graph/`: Graph optimization and fusion

## Backward Compatibility

### Deprecation Notice

The old `taco.localization.cvgl` module has been removed. Code using the old import path will fail with:

```python
ImportError: cannot import name 'CVGLLocalizer' from 'taco.localization'
```

### Migration Script

Use this script to automatically update your code:

```python
#!/usr/bin/env python3
"""Migrate CVGL imports from localization to sensors."""

import sys
from pathlib import Path

def migrate_file(file_path: Path) -> bool:
    """Migrate a single Python file."""
    content = file_path.read_text()
    original = content

    # Replace imports
    content = content.replace(
        'from taco.localization.cvgl import',
        'from taco.sensors.cvgl import'
    )
    content = content.replace(
        'import taco.localization.cvgl',
        'import taco.sensors.cvgl'
    )

    if content != original:
        file_path.write_text(content)
        return True
    return False

def main():
    """Migrate all Python files in current directory."""
    updated = []

    for py_file in Path('.').rglob('*.py'):
        if migrate_file(py_file):
            updated.append(py_file)

    if updated:
        print(f"✓ Updated {len(updated)} files:")
        for f in updated:
            print(f"  - {f}")
    else:
        print("✓ No files needed updating")

if __name__ == '__main__':
    main()
```

Save as `migrate_cvgl.py` and run:
```bash
python migrate_cvgl.py
```

## Documentation Updates

All documentation has been updated:
- ✅ [image_retrieval.md](image_retrieval.md) - Usage examples updated
- ✅ [CVGL_MODEL_SUMMARY.md](CVGL_MODEL_SUMMARY.md) - Import paths updated
- ✅ [CVGL_BUGFIX.md](CVGL_BUGFIX.md) - Reference updated
- ✅ Example scripts updated
- ✅ Test files updated

## FAQ

### Q: Will my existing checkpoints still work?

**A:** Yes! Model checkpoints are not affected by import path changes. You can load old checkpoints with the new import:

```python
from taco.sensors.cvgl import ImageRetrievalModel

model = ImageRetrievalModel.load_from_checkpoint("old_checkpoint.ckpt")
```

### Q: Do I need to retrain my models?

**A:** No. Only the import paths changed, not the model architecture or weights.

### Q: What about saved databases?

**A:** Saved numpy databases (`.npz` files with embeddings) work unchanged.

### Q: Can I still use the old import path?

**A:** No. The old module has been removed. You must update to the new import path.

## Getting Help

If you encounter issues during migration:

1. Check this guide for common patterns
2. Review the updated examples in `examples/`
3. Look at the updated tests in `tests/`
4. Open an issue on GitHub if you find a bug

## Summary

| Aspect | Change |
|--------|--------|
| Import Path | `taco.localization.cvgl` → `taco.sensors.cvgl` |
| CLI Commands | `python -m taco.localization.cvgl.train` → `python -m taco.sensors.cvgl.train` |
| Functionality | No changes - all features work identically |
| Checkpoints | Compatible - no retraining needed |
| Tests | Updated to use new path |
| Documentation | Updated throughout |

The migration is straightforward - simply update import paths and everything else continues to work!
