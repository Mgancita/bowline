"""Test bowline.__init__."""

from bowline import __version__


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"
