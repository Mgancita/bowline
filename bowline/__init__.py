"""bowline namespace."""

from importlib_metadata import version

from .preprocessors import StandardPreprocessor

__version__ = version(__package__)

__all__ = ["StandardPreprocessor", "__version__"]
