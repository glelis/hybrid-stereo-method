"""Hybrid stereo method integration module.

This module combines multifocus stereo and photometric stereo techniques
for improved 3D reconstruction.
"""

from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
    integrate_slopes_to_height,
)

__all__ = [
    "IntegrateRecursiveConfig",
    "integrate_normals_to_height",
    "integrate_slopes_to_height",
]
