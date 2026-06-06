"""Conversion of cell-centered multifocus depth maps to the vertex grid the
C integrator expects for hints (INT-05)."""

from __future__ import annotations

import numpy as np


def cell_to_vertex_grid(z_cells: np.ndarray, w_cells: np.ndarray) -> np.ndarray:
    """Convert cell-centered (H, W) height+weight maps to a vertex grid
    (H+1, W+1, 2) by confidence-weighted averaging of the up-to-4 adjacent
    cells of each vertex.

    Fixes INT-05: the C solver demands hints with the height-map (vertex)
    dimensions; feeding the raw (H, W) cell grid made it expand the map with
    ``float_image_expand_by_one``, shifting every hint by half a cell. For a
    linear field, averaging the adjacent cell centers interpolates exactly at
    the vertex position — no shift.

    Cells with non-finite height get weight 0; vertices with no valid adjacent
    cell get height NaN and weight 0 (excluded by the solver).
    """
    if z_cells.shape != w_cells.shape or z_cells.ndim != 2:
        raise ValueError("z_cells and w_cells must be 2-D arrays of equal shape")
    h, w = z_cells.shape
    w_eff = np.where(np.isfinite(z_cells), w_cells, 0.0)
    z_eff = np.where(np.isfinite(z_cells), np.nan_to_num(z_cells), 0.0)

    zw_pad = np.zeros((h + 2, w + 2))
    w_pad = np.zeros((h + 2, w + 2))
    zw_pad[1:-1, 1:-1] = z_eff * w_eff
    w_pad[1:-1, 1:-1] = w_eff

    zw = zw_pad[:-1, :-1] + zw_pad[:-1, 1:] + zw_pad[1:, :-1] + zw_pad[1:, 1:]
    ww = w_pad[:-1, :-1] + w_pad[:-1, 1:] + w_pad[1:, :-1] + w_pad[1:, 1:]

    out = np.full((h + 1, w + 1, 2), np.nan)
    valid = ww > 0
    out[..., 0][valid] = zw[valid] / ww[valid]
    out[..., 1] = np.where(valid, ww / 4.0, 0.0)
    return out
