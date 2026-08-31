"""Cube Bench: a benchmark for spatial reasoning on the 3x3 Rubik's Cube."""

from __future__ import annotations

# Keep the package version aligned with pyproject.toml.
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("cube-bench")
except PackageNotFoundError:
    __version__ = "0+unknown"
