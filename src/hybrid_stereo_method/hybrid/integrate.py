"""Surface integration from normal maps using recursive multigrid solver.

This module provides a Python interface to the C-based gus_integrate_recursive
tool, which computes height maps from normal/slope maps using an iterative
multigrid approach.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_stereo_method.infrastructure.io.image_io import (
    convert_image_array_to_fni,
    read_fni_to_image_array,
)

logger = logging.getLogger(__name__)

# Default path to the C executable (relative to project root)
# Path: src/hybrid_stereo_method/hybrid/integrate.py 
#   -> hybrid (1) -> hybrid_stereo_method (2) -> src (3) -> project_root (4)
# Actually: integrate.py.parent = hybrid, hybrid.parent = hybrid_stereo_method, 
#           hybrid_stereo_method.parent = src, src.parent = project_root
_MODULE_DIR = Path(__file__).parent  # hybrid/
_PROJECT_ROOT = _MODULE_DIR.parent.parent.parent  # src -> project_root
DEFAULT_EXECUTABLE = _PROJECT_ROOT / "csrc" / "integrate_recursive" / "gus_integrate_recursive"


@dataclass
class IntegrateRecursiveConfig:
    """Configuration for the recursive integration solver.
    
    Attributes:
        initial_method: Initial guess method - "zero", "hints", or "reference"
        initial_noise: Random perturbation magnitude for initial guess (0.0 = none)
        max_level: Maximum recursion level (default 30)
        max_iter: Maximum iterations per level
        conv_tol: Convergence tolerance
        sort_sys: Whether to sort equations by weight
        verbose: Enable verbose output
        report_step: Frequency for debug output (0 = disabled)
        slopes_scale: Scale factors for X/Y slopes (default (1.0, 1.0))
    """
    initial_method: str = "zero"
    initial_noise: float = 0.0
    max_level: int = 30
    max_iter: int = 100000
    conv_tol: float = 0.0000005
    sort_sys: bool = False
    verbose: bool = False
    report_step: int = 0
    slopes_scale: tuple[float, float] = field(default_factory=lambda: (1.0, 1.0))


def integrate_normals_to_height(
    normal_map: np.ndarray,
    output_dir: str | Path,
    output_prefix: str = "result",
    config: IntegrateRecursiveConfig | None = None,
    executable_path: str | Path | None = None,
    hints_map: np.ndarray | None = None,
    hints_weight: float = 0.0,
    reference_map: np.ndarray | None = None,
    hints_fni_path: str | Path | None = None,
    reference_fni_path: str | Path | None = None,
) -> np.ndarray:
    """Integrate a normal map to compute a height map.
    
    Uses the recursive multigrid C solver to compute surface heights
    from surface normals via gradient integration.
    
    Args:
        normal_map: Normal map array with shape (H, W, 3) or (H, W, 4).
            Channels 0-2 are Nx, Ny, Nz components. Channel 3 (optional)
            is the reliability weight.
        output_dir: Directory for output files.
        output_prefix: Prefix for output file names.
        config: Solver configuration. Uses defaults if None.
        executable_path: Path to gus_integrate_recursive binary.
            Uses default location if None.
        hints_map: Optional independent height estimate (H+1, W+1, 1-2 channels).
        hints_weight: Weight for hints map (0.0 to 1.0).
        reference_map: Optional reference height map for error analysis.
    
    Returns:
        Height map as numpy array with shape (H+1, W+1).
    
    Raises:
        FileNotFoundError: If the executable is not found.
        RuntimeError: If the integration fails.
    """
    if config is None:
        config = IntegrateRecursiveConfig()
    
    if executable_path is None:
        executable_path = DEFAULT_EXECUTABLE
    
    executable_path = Path(executable_path)
    if not executable_path.exists():
        raise FileNotFoundError(
            f"Integration executable not found at: {executable_path}\n"
            "Please build it with: cd csrc/integrate_recursive && make"
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write normal map to FNI format
    normal_fni_path = output_dir / f"{output_prefix}_normals.fni"
    convert_image_array_to_fni(normal_map, normal_fni_path)
    logger.info(f"Wrote normal map to: {normal_fni_path}")
    
    # Build command line arguments
    cmd = [
        str(executable_path),
        "-normals", str(normal_fni_path),
    ]
    
    # Add scale if not default
    if config.slopes_scale != (1.0, 1.0):
        cmd.extend(["scale", str(config.slopes_scale[0]), str(config.slopes_scale[1])])
    
    # Add hints map if provided
    if hints_fni_path is not None:
        cmd.extend(["-hints", str(hints_fni_path), str(hints_weight)])
        logger.info(f"Using existing hints map at: {hints_fni_path}")
    elif hints_map is not None:
        hints_fni_path = output_dir / f"{output_prefix}_hints.fni"
        convert_image_array_to_fni(hints_map, hints_fni_path)
        cmd.extend(["-hints", str(hints_fni_path), str(hints_weight)])
        logger.info(f"Wrote hints map to: {hints_fni_path}")
    
    # Add reference map if provided  
    if reference_fni_path is not None:
        cmd.extend(["-reference", str(reference_fni_path)])
        logger.info(f"Using existing reference map at: {reference_fni_path}")
    elif reference_map is not None:
        reference_fni_path = output_dir / f"{output_prefix}_reference.fni"
        convert_image_array_to_fni(reference_map, reference_fni_path)
        cmd.extend(["-reference", str(reference_fni_path)])
        logger.info(f"Wrote reference map to: {reference_fni_path}")
    
    # Add solver parameters
    cmd.extend([
        "-initial", config.initial_method, str(config.initial_noise),
        "-maxLevel", str(config.max_level),
        "-maxIter", str(config.max_iter),
        "-convTol", str(config.conv_tol),
    ])
    
    if config.sort_sys:
        cmd.extend(["-sortSys", "T"])
    
    if config.verbose:
        cmd.append("-verbose")
    
    if config.report_step > 0:
        cmd.extend(["-reportStep", str(config.report_step)])
    
    # Output prefix (mandatory)
    out_prefix = output_dir / output_prefix
    cmd.extend(["-outPrefix", str(out_prefix)])
    
    logger.info(f"Running integration: {' '.join(cmd)}")
    
    # Execute the C program
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(output_dir),
        )
        if config.verbose:
            logger.debug(f"stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Integration failed with return code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        logger.error(f"stdout: {e.stdout}")
        raise RuntimeError(f"Integration failed: {e.stderr}") from e
    
    # Read the output height map
    # Output is written to {PREFIX}-00-end-Z.fni for level 0 final result
    height_fni_path = output_dir / f"{output_prefix}-00-end-Z.fni"
    
    if not height_fni_path.exists():
        # Try alternative naming
        height_fni_path = output_dir / f"{output_prefix}-ini-Z.fni"
        if not height_fni_path.exists():
            raise RuntimeError(
                f"Height map output not found at: {height_fni_path}\n"
                f"Check output directory: {output_dir}"
            )
    
    height_map = read_fni_to_image_array(height_fni_path)
    logger.info(f"Read height map from: {height_fni_path}, shape: {height_map.shape}")
    
    # Return only the height channel (first channel)
    if len(height_map.shape) == 3:
        return height_map[:, :, 0]
    return height_map


def integrate_slopes_to_height(
    slope_map: np.ndarray,
    output_dir: str | Path,
    output_prefix: str = "result",
    config: IntegrateRecursiveConfig | None = None,
    executable_path: str | Path | None = None,
    hints_map: np.ndarray | None = None,
    hints_weight: float = 0.0,
    reference_map: np.ndarray | None = None,
    reference_fni_path: str | Path | None = None,
) -> np.ndarray:
    """Integrate a slope map to compute a height map.
    
    Similar to integrate_normals_to_height but accepts slopes (gradients)
    directly instead of normals.
    
    Args:
        slope_map: Slope map array with shape (H, W, 2) or (H, W, 3).
            Channel 0 is dZ/dX, channel 1 is dZ/dY. Channel 2 (optional)
            is the reliability weight.
        output_dir: Directory for output files.
        output_prefix: Prefix for output file names.
        config: Solver configuration.
        executable_path: Path to gus_integrate_recursive binary.
        hints_map: Optional independent height estimate.
        hints_weight: Weight for hints map.
        reference_map: Optional reference height map.
    
    Returns:
        Height map as numpy array with shape (H+1, W+1).
    """
    if config is None:
        config = IntegrateRecursiveConfig()
    
    if executable_path is None:
        executable_path = DEFAULT_EXECUTABLE
    
    executable_path = Path(executable_path)
    if not executable_path.exists():
        raise FileNotFoundError(f"Integration executable not found at: {executable_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write slope map to FNI format
    slope_fni_path = output_dir / f"{output_prefix}_slopes.fni"
    convert_image_array_to_fni(slope_map, slope_fni_path)
    logger.info(f"Wrote slope map to: {slope_fni_path}")
    
    # Build command - same as normals but use -slopes
    cmd = [
        str(executable_path),
        "-slopes", str(slope_fni_path),
    ]
    
    if config.slopes_scale != (1.0, 1.0):
        cmd.extend(["scale", str(config.slopes_scale[0]), str(config.slopes_scale[1])])
    
    if hints_map is not None:
        hints_fni_path = output_dir / f"{output_prefix}_hints.fni"
        convert_image_array_to_fni(hints_map, hints_fni_path)
        cmd.extend(["-hints", str(hints_fni_path), str(hints_weight)])
    
    if reference_fni_path is not None:
        cmd.extend(["-reference", str(reference_fni_path)])
    elif reference_map is not None:
        reference_fni_path = output_dir / f"{output_prefix}_reference.fni"
        convert_image_array_to_fni(reference_map, reference_fni_path)
        cmd.extend(["-reference", str(reference_fni_path)])
    
    cmd.extend([
        "-initial", config.initial_method, str(config.initial_noise),
        "-maxLevel", str(config.max_level),
        "-maxIter", str(config.max_iter),
        "-convTol", str(config.conv_tol),
    ])
    
    if config.sort_sys:
        cmd.extend(["-sortSys", "T"])
    
    if config.verbose:
        cmd.append("-verbose")
    
    if config.report_step > 0:
        cmd.extend(["-reportStep", str(config.report_step)])
    
    out_prefix = output_dir / output_prefix
    cmd.extend(["-outPrefix", str(out_prefix)])
    
    logger.info(f"Running integration: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(output_dir),
        )
        if config.verbose:
            logger.debug(f"stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Integration failed: {e.stderr}")
        raise RuntimeError(f"Integration failed: {e.stderr}") from e
    
    height_fni_path = output_dir / f"{output_prefix}-00-end-Z.fni"
    if not height_fni_path.exists():
        height_fni_path = output_dir / f"{output_prefix}-ini-Z.fni"
        if not height_fni_path.exists():
            raise RuntimeError(f"Height map not found at: {height_fni_path}")
    
    height_map = read_fni_to_image_array(height_fni_path)
    
    if len(height_map.shape) == 3:
        return height_map[:, :, 0]
    return height_map
