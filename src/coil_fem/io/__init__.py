"""I/O helpers for coil-fem (gmsh full-body export, etc.)."""

from .gmsh import to_full_body
from .readable import volume_weighted_summary

__all__ = ["to_full_body", "volume_weighted_summary"]
