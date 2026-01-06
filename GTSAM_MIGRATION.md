# GTSAM Migration Guide

This document describes the migration of TACO to use GTSAM (Georgia Tech Smoothing and Mapping) as the backend optimization library.

## Overview

TACO now uses GTSAM for all pose graph optimization operations. GTSAM provides:
- State-of-the-art nonlinear optimization (Levenberg-Marquardt, Gauss-Newton)
- Efficient IMU preintegration
- Incremental smoothing (iSAM2)
- Robust factor graph infrastructure

## Key Changes

### 1. Pose Graph Module (`src/taco/pose_graph/`)

#### `graph.py`
- **Before**: Custom graph structure with nodes and edges
- **After**: Wraps `gtsam.NonlinearFactorGraph` and `gtsam.Values`
- **New Methods**:
  - `add_pose_estimate()` - Add poses using `gtsam.Pose3`
  - `add_prior_factor()` - Add prior constraints
  - `add_between_factor()` - Add odometry constraints
  - `add_gps_factor()` - Add position-only measurements
  - `add_pose_factor()` - Add full 6-DOF pose measurements
  - `optimize()` - Optimize using GTSAM optimizers
  - `get_marginal_covariance()` - Get uncertainty estimates

#### `node.py`
- **New**: `to_gtsam_pose()` method converts `PoseNode` to `gtsam.Pose3`
- **New**: `from_gtsam_pose()` static method creates `PoseNode` from `gtsam.Pose3`
- **New Helper Functions**:
  - `create_noise_model_diagonal()` - Create diagonal noise models
  - `create_noise_model_gaussian()` - Create Gaussian noise models
  - `create_noise_model_isotropic()` - Create isotropic noise models

#### `edge.py`
- **New**: `to_gtsam_between_factor()` method converts edges to GTSAM factors
- **New**: `from_poses()` creates edges from GTSAM poses
- **New Helper Functions**:
  - `create_between_factor()` - Create GTSAM between factors
  - `create_prior_factor()` - Create GTSAM prior factors

#### `optimizer.py`
- **Before**: Placeholder implementation
- **After**: Full GTSAM optimization support
  - `optimize()` - Batch optimization
  - `optimize_incremental()` - iSAM2 incremental optimization

### 2. IMU Module (`src/taco/sensors/imu/`)

#### `integrator.py`
- **New**: `integrate_to_gtsam_pose()` - Returns `gtsam.Pose3` directly
- **Backward Compatible**: Original numpy-based `integrate()` method still available

#### `preintegration.py`
- **Complete Rewrite**: Now uses `gtsam.PreintegratedImuMeasurements`
- **New Methods**:
  - `integrate_measurements()` - Preintegrate IMU data
  - `predict()` - Predict pose/velocity after preintegration
  - `get_preintegrated_measurements()` - Get GTSAM PIM object
- **New Function**: `create_imu_factor()` - Create GTSAM IMU factors

### 3. CVGL Localization (`src/taco/localization/cvgl/`)

#### `measurement.py`
- **New**: `to_gtsam_pose()` - Convert measurement to `gtsam.Pose3`
- **New**: `get_gtsam_noise_model()` - Get GTSAM noise model from covariance

### 4. Utilities (`src/taco/utils/conversions.py`)

**New GTSAM Conversion Functions**:
- `gtsam_pose_to_transform()` - Convert `gtsam.Pose3` to 4x4 matrix
- `transform_to_gtsam_pose()` - Convert 4x4 matrix to `gtsam.Pose3`
- `numpy_pose_to_gtsam()` - Convert numpy arrays to `gtsam.Pose3`
- `gtsam_pose_to_numpy()` - Convert `gtsam.Pose3` to numpy arrays

### 5. Visualization (`src/taco/visualization/plotter.py`)

#### `plot_pose_graph()`
- **Updated**: Now works with GTSAM poses from the graph
- Extracts positions from `gtsam.Pose3` objects

#### `plot_gtsam_values()` (New)
- Directly plots `gtsam.Values` objects
- Useful for visualizing optimization results

## Usage Examples

### Basic Pose Graph

```python
import gtsam
import numpy as np
from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.utils.conversions import numpy_pose_to_gtsam

# Create graph
graph = PoseGraph()

# Add initial pose
initial_pose = numpy_pose_to_gtsam(
    position=np.zeros(3),
    orientation=np.array([1, 0, 0, 0])
)
pose_id = graph.add_pose_estimate(initial_pose, timestamp=0.0)

# Add prior
prior_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))
graph.add_prior_factor(pose_id, initial_pose, prior_noise)

# Optimize
result = graph.optimize()
```

### IMU Preintegration

```python
from taco.sensors.imu import IMUPreintegrator, IMUData

# Create preintegrator
gravity = np.array([0, 0, -9.81])
preintegrator = IMUPreintegrator(gravity)

# Integrate measurements
preintegrator.integrate_measurements(imu_data_list)

# Predict next state
next_pose, next_vel = preintegrator.predict(current_pose, current_vel)

# Add IMU factor to graph
from taco.sensors.imu.preintegration import create_imu_factor
factor = create_imu_factor(
    pose_key_i=0,
    vel_key_i=0,
    pose_key_j=1,
    vel_key_j=1,
    bias_key=0,
    pim=preintegrator.get_preintegrated_measurements()
)
```

### CVGL Integration

```python
from taco.localization.cvgl import CVGLMeasurement

# Get CVGL measurement
measurement = localizer.localize(image, timestamp, camera_intrinsics)

# Add to graph
if measurement.confidence > 0.8:
    cvgl_pose = measurement.to_gtsam_pose()
    noise = measurement.get_gtsam_noise_model()
    graph.add_pose_factor(pose_id, cvgl_pose, noise)
```

## Migration Checklist

If you have existing code using the old API:

- [ ] Replace manual graph construction with `PoseGraph` GTSAM methods
- [ ] Convert pose representations to `gtsam.Pose3`
- [ ] Update IMU preintegration to use GTSAM's implementation
- [ ] Add noise models using helper functions
- [ ] Update optimization calls to use `graph.optimize()`
- [ ] Update visualization calls for GTSAM compatibility

## Dependencies

**New Dependency**: `gtsam>=4.2.0`

Install with:
```bash
pip install gtsam
```

Or install TACO with all dependencies:
```bash
pip install -e ".[dev]"
```

## Performance Benefits

1. **Faster Optimization**: GTSAM's optimized C++ backend
2. **Better Accuracy**: State-of-the-art optimization algorithms
3. **Incremental Updates**: iSAM2 for online SLAM
4. **Robust Estimation**: Built-in robust cost functions
5. **Marginal Covariances**: Efficient uncertainty quantification

## Backward Compatibility

The following are still available for backward compatibility:
- `PoseNode` class (can be converted to/from `gtsam.Pose3`)
- `Edge` class (can be converted to GTSAM factors)
- Numpy-based IMU integration (alongside GTSAM version)

## References

- [GTSAM Documentation](https://gtsam.org/)
- [GTSAM Python Tutorial](https://github.com/borglab/gtsam/tree/develop/python)
- [IMU Preintegration Paper](https://rpg.ifi.uzh.ch/docs/RSS15_Forster.pdf)
