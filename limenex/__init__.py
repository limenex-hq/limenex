"""Limenex — agentic AI execution governance layer."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("limenex")
except PackageNotFoundError:
    # Package is not installed (e.g. running directly from source tree
    # without `pip install -e .`). Fallback only.
    __version__ = "0.0.0+unknown"

__all__: list[str] = []
