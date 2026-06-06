"""F4 — Onde o multifocus erra (H5): objeto x fundo, correlação por região."""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_normals_gt,
    load_shrp_z_gt,
    load_zmos,
)
from hybrid_stereo_method.evaluation.metrics import pearson_r
from scripts.investigation.common import RAW, RESULTS, save_json, summarize


def main() -> None:
    zmos = load_zmos(RESULTS)
    sharp = find_sharp_dir(RESULTS, RAW)
    gt_h = load_height_gt(sharp)
    _n_gt, fg = load_normals_gt(sharp)
    z_gt, z_vals = load_shrp_z_gt(RAW)
    step = float(np.median(np.diff(sorted(z_vals))))
    err_frames = np.abs(zmos - z_gt) / step

    def region(mask: np.ndarray, label: str) -> dict:
        m = mask & np.isfinite(zmos) & np.isfinite(gt_h)
        return {
            "err_frames": summarize(f"erro de foco em frames ({label})", err_frames[m]),
            "pearson_zmos_vs_havg": pearson_r(zmos[m], gt_h[m]),
            "pearson_zmos_vs_zshrp": pearson_r(zmos[m], z_gt[m]),
            "n_pixels": int(m.sum()),
        }

    report = {
        "object": region(fg, "objeto"),
        "background": region(~fg, "fundo"),
        "all": region(np.ones_like(fg, dtype=bool), "tudo"),
    }
    for k in ("object", "background", "all"):
        print(f"  {k}: r(zMos,hAvg)={report[k]['pearson_zmos_vs_havg']:+.3f} "
              f"r(zMos,z_shrp)={report[k]['pearson_zmos_vs_zshrp']:+.3f}")
    save_json("f4_multifocus_spatial.json", report)


if __name__ == "__main__":
    main()
