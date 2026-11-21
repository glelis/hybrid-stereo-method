"""Convenção de orientação do integrador: domo sintético integra para domo.

Regressão da investigação 2026-06-06 (F1/H2). Constrói as normais analíticas
de z(x, y) = h0 - (r/R)^2 (domo, +z para fora da superfície, frame de imagem
y-down do numpy) e exige que a altura integrada tenha o centro ACIMA da borda
e correlação fortemente positiva com o z analítico.
"""

import numpy as np
import pytest

from hybrid_stereo_method.evaluation.loaders import vertex_to_cell
from hybrid_stereo_method.evaluation.metrics import pearson_r
from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(), reason="binário C não compilado"
)


def dome_normals(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(normais (h, w, 3) no frame de imagem y-down, z analítico (h, w))."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy, scale = (w - 1) / 2.0, (h - 1) / 2.0, 8.0 / min(h, w)
    z = -(((xx - cx) * scale) ** 2 + ((yy - cy) * scale) ** 2)
    dzdx = -2.0 * (xx - cx) * scale**2
    dzdy = -2.0 * (yy - cy) * scale**2  # y = índice de linha, crescendo para baixo
    n = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n, z


def test_dome_integrates_to_dome(tmp_path):
    normals, z_true = dome_normals(32, 32)
    height = integrate_normals_to_height(
        normal_map=normals.astype(np.float32),
        output_dir=tmp_path,
        output_prefix="dome",
        config=IntegrateRecursiveConfig(),
    )
    h_cell = vertex_to_cell(height)
    center = h_cell[12:20, 12:20].mean()
    rim = np.concatenate([h_cell[0], h_cell[-1], h_cell[:, 0], h_cell[:, -1]]).mean()
    pearson = pearson_r(h_cell, z_true)
    print(f"\ndome: center={center:.4f}, rim={rim:.4f}, pearson={pearson:.4f}")
    assert center > rim, "domo integrou de cabeça para baixo (H2)"
    assert pearson > 0.95
