"""Cube Bench: a benchmark for spatial reasoning on the 3x3 Rubik's Cube."""

from __future__ import annotations

# Keep version single-sourced from pyproject.toml
try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.10+
except Exception:  # pragma: no cover
    # Only needed for very old Pythons; your project requires 3.10+
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("cube-bench")  # ← use the *distribution* name (with hyphen)
except PackageNotFoundError:  # not installed (e.g., running from source without pip -e .)
    __version__ = "0+unknown"
