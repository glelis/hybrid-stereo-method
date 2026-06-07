"""Estima pixel_size: fator afim entre altura sem-hints (px) e zMos (z_foc).

pixel_size = 1/a, onde zMos ~ a*height_px + b no objeto (CONV-4: height_zfoc =
height_px * pixel_size). Se F4 indicou relação INVERSA (depth = -height), o
sinal de a já captura isso — reporte e use |1/a| com a observação de sinal.
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_normals_gt,
    load_zmos,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.metrics import affine_fit_rmse
from hybrid_stereo_method.infrastructure.io.image_io import read_fni_to_image_array
from scripts.investigation.common import OUT, RAW, RESULTS, save_json


def main() -> None:
    h_px = read_fni_to_image_array(OUT / "a1_hints" / "no_hints" / "a1-00-end-Z.fni")
    h_px = h_px[..., 0] if h_px.ndim == 3 else h_px
    h_cell = vertex_to_cell(h_px)
    zmos = load_zmos(RESULTS)
    _n, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    m = fg & np.isfinite(zmos) & np.isfinite(h_cell)
    _rmse, (a, b) = affine_fit_rmse(h_cell[m], zmos[m])
    report = {"a": a, "b": b, "pixel_size_estimate": 1.0 / a if a != 0 else None,
              "sign_note": "a<0 => relação inversa height<->depth; ver F4"}
    print(f"  zMos ~ {a:+.4g}*h_px {b:+.4g}  ->  pixel_size ~ {report['pixel_size_estimate']}")
    save_json("pixel_size_estimate.json", report)


if __name__ == "__main__":
    main()
