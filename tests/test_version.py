"""Test package version."""

import taco


def test_version():
    """Test that version is defined."""
    assert hasattr(taco, "__version__")
    assert isinstance(taco.__version__, str)
