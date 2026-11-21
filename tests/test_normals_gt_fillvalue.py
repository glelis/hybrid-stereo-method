"""load_normals_gt exclui o fill-value (-1/√3,-1/√3,-1/√3) do foreground.

Investigação 2026-06-06 (C6): o fill-value tem norma exatamente 1 e passava o
teste |norma-1|<=tol, poluindo o erro angular. Normais de superfície visível
têm nz>0; o fill-value tem nz<0 -> excluído.
"""

import numpy as np

from hybrid_stereo_method.evaluation.loaders import load_normals_gt


def _encode_normal(nx, ny, nz, scale=65535):
    """(nx,ny,nz) em [-1,1] -> trio RGB inteiro como sNrm grava."""
    return [int(round((c + 1.0) / 2.0 * scale)) for c in (nx, ny, nz)]


def test_fillvalue_excluded_from_foreground(tmp_path):
    import cv2

    s = 1.0 / np.sqrt(3.0)
    front = _encode_normal(0.0, 0.0, 1.0)          # normal frontal válida (nz=1)
    fill = _encode_normal(-s, -s, -s)              # fill-value (nz<0)
    # imagem 1x2: col 0 = válida, col 1 = fill. sNrm é gravado em RGB; cv2 espera BGR.
    rgb = np.array([[front, fill]], dtype=np.uint16)        # (1,2,3) RGB
    bgr = rgb[..., ::-1]
    cv2.imwrite(str(tmp_path / "sNrm.png"), bgr)

    n_unit, fg = load_normals_gt(tmp_path)
    assert fg.shape == (1, 2)
    assert fg[0, 0]            # frontal é foreground
    assert not fg[0, 1]       # fill-value excluído
    assert np.isnan(n_unit[0, 1]).all()
