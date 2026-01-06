# TACO: Trajectory Aligning Cross-view Optimisation

[![CI](https://github.com/tavisshore/taco/workflows/CI/badge.svg)](https://github.com/tavisshore/taco/actions)
[![codecov](https://codecov.io/gh/tavisshore/taco/branch/main/graph/badge.svg)](https://codecov.io/gh/tavisshore/taco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![PyPI version](https://badge.fury.io/py/taco.svg)](https://badge.fury.io/py/taco) -->
<!-- [![Python versions](https://img.shields.io/pypi/pyversions/taco.svg)](https://pypi.org/project/taco/) -->

**TACO** is a Python library for pose graph optimization that fuses **IMU (Inertial Measurement Unit)** data with **CVGL (Computer Vision Global Localization)** image-based measurements for robust 6-DOF pose estimation.

Built on [**GTSAM (Georgia Tech Smoothing and Mapping)**](https://gtsam.org/), TACO provides a high-level interface for creating and optimizing pose graphs with modern optimization algorithms.

## Features

### Core Functionality
- **GTSAM-Based Optimization**: Leverage state-of-the-art factor graph optimization
- **IMU Preintegration**: Efficient IMU factor integration using GTSAM's preintegration
- **Pose Graph Construction**: Build complex factor graphs with multiple sensor modalities
- **CVGL Localization**: Image-based global localization with uncertainty quantification
- **Sensor Fusion**: Seamlessly combine odometry and absolute pose measurements
- **iSAM2 Support**: Incremental smoothing and mapping for online SLAM

### Development
- Modern Python packaging with `pyproject.toml`
- Comprehensive testing with pytest
- Code quality tools (ruff, black, mypy)
- Pre-commit hooks
- CI/CD with GitHub Actions
- Full API documentation with Sphinx
- Type hints throughout

## Installation

```bash
pip install taco
```

For development:

```bash
git clone https://github.com/tavisshore/taco.git
cd taco
pip install -e ".[dev,docs]"
pre-commit install
```

## Quick Start

### Basic Example with GTSAM

```python
import numpy as np
import gtsam
from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.utils.conversions import numpy_pose_to_gtsam
from taco.visualization import plot_pose_graph

# Create a pose graph
graph = PoseGraph()

# Create initial pose using GTSAM
initial_pose = numpy_pose_to_gtsam(
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
)

# Add pose with prior factor
pose_id = graph.add_pose_estimate(initial_pose, timestamp=0.0)
prior_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))
graph.add_prior_factor(pose_id, initial_pose, prior_noise)

# Add odometry between poses
next_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.0, 0.0))
next_id = graph.add_pose_estimate(next_pose, timestamp=1.0)

relative_pose = initial_pose.between(next_pose)
odometry_noise = create_noise_model_diagonal(np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
graph.add_between_factor(pose_id, next_id, relative_pose, odometry_noise)

# Optimize
result = graph.optimize()

# Visualize
plot_pose_graph(graph)
```

### IMU Preintegration with GTSAM

```python
import gtsam
from taco.sensors.imu import IMUData, IMUPreintegrator

# Create IMU measurements
imu_data = [
    IMUData.from_raw(0.0, 0.0, 0.0, -9.81, 0.0, 0.0, 0.1),
    IMUData.from_raw(0.1, 0.0, 0.0, -9.81, 0.0, 0.0, 0.1),
    # ... more measurements
]

# Preintegrate using GTSAM
gravity = np.array([0.0, 0.0, -9.81])
preintegrator = IMUPreintegrator(gravity)
preintegrator.integrate_measurements(imu_data)

# Predict next pose
current_pose = gtsam.Pose3()  # Current pose
current_velocity = np.zeros(3)  # Current velocity
predicted_pose, predicted_velocity = preintegrator.predict(current_pose, current_velocity)

# Get preintegrated measurements for factor graph
pim = preintegrator.get_preintegrated_measurements()
```

### CVGL Localization with GTSAM

```python
from pathlib import Path
from taco.localization.cvgl import CVGLLocalizer

# Initialize localizer with map
localizer = CVGLLocalizer(map_path=Path("path/to/map.db"))

# Localize an image
image = load_image("query.jpg")  # your image loading code
measurement = localizer.localize(
    image,
    timestamp=0.0,
    camera_intrinsics=K
)

if measurement and measurement.confidence > 0.8:
    # Convert to GTSAM types
    cvgl_pose = measurement.to_gtsam_pose()
    noise_model = measurement.get_gtsam_noise_model()

    # Add to pose graph
    graph.add_pose_factor(pose_id, cvgl_pose, noise_model)
    print(f"Added CVGL measurement with confidence: {measurement.confidence}")
```

## Development

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=taco --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Building Documentation

```bash
cd docs
make html
```

## Project Structure

```
taco/
├── src/taco/
│   ├── pose_graph/          # Core pose graph implementation
│   │   ├── graph.py         # Main PoseGraph class
│   │   ├── node.py          # Pose node representation
│   │   ├── edge.py          # Edge constraints
│   │   └── optimizer.py     # Graph optimization algorithms
│   ├── sensors/
│   │   └── imu/             # IMU data processing
│   │       ├── data.py      # IMU measurement representation
│   │       ├── integrator.py    # IMU integration
│   │       └── preintegration.py # IMU preintegration
│   ├── localization/
│   │   └── cvgl/            # CVGL image localization
│   │       ├── localizer.py # Localization engine
│   │       └── measurement.py # CVGL measurement
│   ├── utils/               # Utility functions
│   │   ├── conversions.py   # Coordinate transformations
│   │   └── io.py            # Data I/O
│   └── visualization/       # Plotting and visualization
│       └── plotter.py
├── tests/                   # Comprehensive test suite
│   ├── test_pose_graph.py
│   ├── test_imu.py
│   ├── test_cvgl.py
│   └── test_utils.py
├── docs/                    # Sphinx documentation
├── .github/workflows/       # CI/CD pipelines
└── scripts/                 # Development scripts
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with modern Python packaging standards
- Uses the latest development tools and best practices
- IMU preintegration based on Forster et al. (RSS 2015)
