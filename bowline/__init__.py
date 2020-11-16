"""bowline namespace."""

from importlib_metadata import version

from .preprocessors import StandardPreprocessor

try:
    __version__ = version(__package__)
except:  # noqa: E722
    __version__ = "0.0.0"

__all__ = ["StandardPreprocessor", "__version__"]
