"""Tests for CVGL localization."""

import numpy as np
import pytest

from taco.localization.cvgl import CVGLMeasurement


class TestCVGLMeasurement:
    """Test CVGLMeasurement class."""

    def test_measurement_creation(self) -> None:
        """Test creating a CVGL measurement."""
        measurement = CVGLMeasurement(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(6),
            confidence=0.95,
            num_inliers=100,
        )

        assert measurement.timestamp == 0.0
        assert np.allclose(measurement.position, [1.0, 2.0, 3.0])
        assert measurement.confidence == 0.95
        assert measurement.num_inliers == 100

    def test_to_transformation_matrix(self) -> None:
        """Test conversion to transformation matrix."""
        measurement = CVGLMeasurement(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            covariance=np.eye(6),
            confidence=0.95,
            num_inliers=100,
        )

        T = measurement.to_transformation_matrix()

        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
        assert np.allclose(T[:3, :3], np.eye(3))  # Identity rotation

    def test_invalid_confidence(self) -> None:
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            CVGLMeasurement(
                timestamp=0.0,
                position=np.zeros(3),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                covariance=np.eye(6),
                confidence=1.5,  # Invalid
                num_inliers=100,
            )
