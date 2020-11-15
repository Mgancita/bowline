"""Test data_preprocessors.__init__."""

from data_preprocessor import __version__


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"
