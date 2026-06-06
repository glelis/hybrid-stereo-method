"""F1 — Integrador isolado: normais GT (sNrm) -> altura, vs hAvg (H2).

Se alguma orientação der pearson >> 0: integrador + convenção OK (H2 refutada
no integrador; a orientação vencedora é a convenção do pipeline).
Se ambas derem ruim: bug no integrador/conversão normais->slopes.
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from scripts.investigation.common import OUT, RAW, RESULTS, height_metrics_vs_gt, save_json


def main() -> None:
    n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    report: dict = {"foreground_fraction": float(fg.mean())}

    for label, flip in (("y_as_is", False), ("y_flipped", True)):
        n = n_gt.copy()
        if flip:
            n[..., 1] *= -1.0
        # NaN (fundo) -> normal vertical com peso 0: o solver ignora por peso.
        weight = fg.astype(np.float64)
        n = np.where(np.isfinite(n), n, 0.0)
        n[~fg] = (0.0, 0.0, 1.0)
        n4 = np.concatenate([n, weight[..., None]], axis=-1)

        out_dir = OUT / "f1_gt_normals" / label
        height = integrate_normals_to_height(
            normal_map=n4.astype(np.float32),
            output_dir=out_dir,
            output_prefix="gtn",
            config=IntegrateRecursiveConfig(),  # zero initial, sem hints/reference
        )
        report[label] = height_metrics_vs_gt(height)
        print(f"  {label}: pearson={report[label]['pearson_r']:+.3f} "
              f"rmse_affine={report[label]['rmse_affine']:.0f} a={report[label]['a']:+.3g}")

    save_json("f1_integrate_gt_normals.json", report)


if __name__ == "__main__":
    main()
