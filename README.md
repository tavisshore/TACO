# TACO: Trajectory Aligning Cross-view Optimisation
[![CI](https://github.com/tavisshore/taco/workflows/CI/badge.svg)](https://github.com/tavisshore/taco/actions)
[![codecov](https://codecov.io/gh/tavisshore/taco/branch/main/graph/badge.svg)](https://codecov.io/gh/tavisshore/taco)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![PyPI version](https://badge.fury.io/py/taco.svg)](https://badge.fury.io/py/taco) -->
<!-- [![Python versions](https://img.shields.io/pypi/pyversions/taco.svg)](https://pypi.org/project/taco/) -->

**TACO** is a Python library for pose graph optimisation that fuses **IMU (Inertial Measurement Unit)** data with **CVGL (Cross-View Geo-Localisation)** image-based measurements for robust 6-DOF pose estimation.

Built on [GTSAM (Georgia Tech Smoothing and Mapping)](https://gtsam.org/).

<!-- ## Features
### Core Functionality
- **GTSAM-Based Optimisation**: Leverage state-of-the-art factor graph optimisation
- **IMU Preintegration**: Efficient IMU factor integration using GTSAM's preintegration
- **Pose Graph Construction**: Build complex factor graphs with multiple sensor modalities
- **CVGL Localisation**: Image-based global localisation with uncertainty quantification
- **Sensor Fusion**: Seamlessly combine odometry and absolute pose measurements
- **iSAM2 Support**: Incremental smoothing and mapping for online SLAM -->

## Installation
```bash
pip install taco
```

For development:
```bash
git clone https://github.com/tavisshore/taco.git
cd taco
pip install -e ".[dev]"
pre-commit install
```

## Quick Start
### TODO

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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
