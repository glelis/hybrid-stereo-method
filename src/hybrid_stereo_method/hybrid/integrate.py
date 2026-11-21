"""Surface integration from normal maps using recursive multigrid solver.

This module provides a Python interface to the C-based gus_integrate_recursive
tool, which computes height maps from normal/slope maps using an iterative
multigrid approach.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_stereo_method.infrastructure.io.image_io import (
    convert_image_array_to_fni,
    read_fni_to_image_array,
    save_image,
)

logger = logging.getLogger(__name__)

# Default path to the C executable: integrate.py -> hybrid/ -> hybrid_stereo_method/
# -> src/ -> project root.
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
        report_step: Write the intermediate height maps every ``report_step``
            iterations of each level, plus every one of the first
            ``report_step`` iterations (C solver ``-reportStep``; 0 = only the
            start/end of each level). They land in ``{prefix}-iters/level-LL/``
            (see organize_integration_outputs).
        slopes_scale: Scale factors for X/Y slopes. The default (1.0, 1.0) is
            right for normals in the numpy image frame (y = row index, growing
            down). Normals in a Y-up frame (lights.npy of the Stolfi generators,
            e.g. melon24) need sy = -1; the hybrid pipeline takes the signs from
            ``hybrid.integration.slopes_scale`` and multiplies them by
            ``pixel_size`` (see hybrid.main.build_integration_config).
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


# Per-level files the C solver writes flat in output_dir: {PREFIX}-{LL}-{beg|end}-{TAG}.{ext}
_LEVEL_FILE_RE = re.compile(r"^-(\d{2})-(beg|end)-(.+)$")
# Per-iteration files it writes in {PREFIX}-iters/ (prefix "it-" + "-{LL}"): it--{LL}-{ITER:09d}-{TAG}.{ext}
_ITER_FILE_RE = re.compile(r"^it--(\d{2})-(\d{9})-(.+)$")


def organize_integration_outputs(output_dir: str | Path, output_prefix: str) -> Path:
    """Move the solver's per-level/per-iteration files into ``{prefix}-iters/``
    and export every height map ``Z`` as ``.npy`` and ``.png``.

    The C solver writes the per-level maps (``{prefix}-LL-beg-*``,
    ``{prefix}-LL-end-*``) and the initial guess (``{prefix}-ini-Z.fni``) flat
    in ``output_dir``, and the intermediate iterations (only with
    ``-reportStep`` > 0) in ``{prefix}-iters/it--LL-NNNNNNNNN-*``. Resulting
    layout::

        {prefix}-iters/
            ini-Z.{fni,npy,png}
            level-LL/
                beg-{G,H,R,SW}.fni, beg-S.txt     (system/data of level LL)
                beg-Z.{fni,npy,png}               (start of level LL)
                it-NNNNNNNNN-Z.{fni,npy,png}      (iteration NNNNNNNNN)
                end-Z.{fni,npy,png}               (converged level LL)

    The ``.npy`` holds the height channel (channel 0) unchanged; the ``.png``
    is min-max normalized for visualization only.

    Returns:
        Path to the ``{prefix}-iters`` directory.
    """
    output_dir = Path(output_dir)
    iters_dir = output_dir / f"{output_prefix}-iters"
    iters_dir.mkdir(parents=True, exist_ok=True)

    moves: list[tuple[Path, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_file() or not path.name.startswith(output_prefix):
            continue
        rest = path.name[len(output_prefix):]
        if rest == "-ini-Z.fni":
            moves.append((path, iters_dir / "ini-Z.fni"))
        elif m := _LEVEL_FILE_RE.match(rest):
            level, stage, tail = m.groups()
            moves.append((path, iters_dir / f"level-{level}" / f"{stage}-{tail}"))
    for path in iters_dir.iterdir():
        if path.is_file() and (m := _ITER_FILE_RE.match(path.name)):
            level, iteration, tail = m.groups()
            moves.append((path, iters_dir / f"level-{level}" / f"it-{iteration}-{tail}"))

    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dest)

    n_exported = 0
    for fni_path in sorted(iters_dir.rglob("*-Z.fni")):
        z = read_fni_to_image_array(fni_path)
        if z.ndim == 3:
            z = z[:, :, 0]
        np.save(fni_path.with_suffix(".npy"), z)
        save_image(fni_path.parent, fni_path.with_suffix(".png").name, np.nan_to_num(z))
        n_exported += 1
    logger.info(
        f"Organized integration outputs in {iters_dir} ({len(moves)} files moved, "
        f"{n_exported} height maps exported to .npy/.png)"
    )
    return iters_dir


def _read_final_height_and_organize(output_dir: Path, output_prefix: str) -> np.ndarray:
    """Read the final level-0 height map, then tidy up the solver outputs."""
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

    organize_integration_outputs(output_dir, output_prefix)

    # Return only the height channel (first channel)
    if len(height_map.shape) == 3:
        return height_map[:, :, 0]
    return height_map


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
    
    # Resolved now: subprocess.run(cwd=output_dir) would look up a relative
    # executable inside output_dir, not from the caller's CWD.
    executable_path = Path(executable_path).resolve()
    if not executable_path.exists():
        raise FileNotFoundError(
            f"Integration executable not found at: {executable_path}\n"
            "Please build it with: cd csrc/integrate_recursive && make"
        )
    
    # Absolute paths: the binary runs with cwd=output_dir, so a relative path
    # (e.g. output 'data/results/...' from the YAML) would be resolved from
    # inside output_dir and the binary would fail to read its inputs.
    output_dir = Path(output_dir).resolve()
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
        hints_fni_path = Path(hints_fni_path).resolve()
        cmd.extend(["-hints", str(hints_fni_path), str(hints_weight)])
        logger.info(f"Using existing hints map at: {hints_fni_path}")
    elif hints_map is not None:
        hints_fni_path = output_dir / f"{output_prefix}_hints.fni"
        convert_image_array_to_fni(hints_map, hints_fni_path)
        cmd.extend(["-hints", str(hints_fni_path), str(hints_weight)])
        logger.info(f"Wrote hints map to: {hints_fni_path}")
    
    # Add reference map if provided  
    if reference_map is not None:
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
    
    # Read the output height map ({PREFIX}-00-end-Z.fni, level 0 final result)
    # and move the per-level/per-iteration files into {PREFIX}-iters/.
    return _read_final_height_and_organize(output_dir, output_prefix)


def integrate_slopes_to_height(
    slope_map: np.ndarray,
    output_dir: str | Path,
    output_prefix: str = "result",
    config: IntegrateRecursiveConfig | None = None,
    executable_path: str | Path | None = None,
    hints_map: np.ndarray | None = None,
    hints_weight: float = 0.0,
    reference_map: np.ndarray | None = None,
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
    
    # Resolved now: subprocess.run(cwd=output_dir) would look up a relative
    # executable inside output_dir, not from the caller's CWD.
    executable_path = Path(executable_path).resolve()
    if not executable_path.exists():
        raise FileNotFoundError(f"Integration executable not found at: {executable_path}")
    
    # Absolute paths: the binary runs with cwd=output_dir, so a relative path
    # (e.g. output 'data/results/...' from the YAML) would be resolved from
    # inside output_dir and the binary would fail to read its inputs.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # The C solver requires a 3-channel slope map (dZ/dX, dZ/dY, weight):
    # a 2-channel map aborts it ("slope map {G} must have 3 channels").
    slope_map = np.asarray(slope_map)
    if slope_map.ndim == 3 and slope_map.shape[2] == 2:
        slope_map = np.concatenate(
            (slope_map, np.ones(slope_map.shape[:2] + (1,), dtype=slope_map.dtype)), axis=-1
        )
    elif slope_map.ndim != 3 or slope_map.shape[2] != 3:
        raise ValueError(f"slope_map must have shape (H, W, 2) or (H, W, 3), got {slope_map.shape}")

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
    
    if reference_map is not None:
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
    
    return _read_final_height_and_organize(output_dir, output_prefix)
