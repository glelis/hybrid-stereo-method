"""F2 — Erro angular das normais por região (objeto x fundo) e sobreposição
com o penhasco do run original (H1).
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.evaluation.metrics import angular_error_deg
from scripts.investigation.common import RAW, RESULTS, cliff_mask_cell, save_json, summarize


def main() -> None:
    n_est = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")[..., :3]
    confidence = np.load(RESULTS / "photometric_stereo" / "confidence.npy")
    n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    ae = angular_error_deg(n_est, n_gt)  # y_as_is: orientação vencedora da avaliação
    cliff = cliff_mask_cell()
    bg = ~fg

    report = {
        "foreground_fraction": float(fg.mean()),
        "cliff_fraction": float(cliff.mean()),
        "cliff_in_background_fraction": float((cliff & bg).sum() / max(cliff.sum(), 1)),
        "angular_error_object": summarize("erro angular (objeto)", ae[fg]),
        "angular_error_background": summarize("erro angular (fundo)", ae[bg]),
        "angular_error_cliff": summarize("erro angular (penhasco)", ae[cliff]),
        "confidence_object": summarize("confiança (objeto)", confidence[fg]),
        "confidence_background": summarize("confiança (fundo)", confidence[bg]),
        "confidence_cliff": summarize("confiança (penhasco)", confidence[cliff]),
        "est_nan_fraction_background": float(np.isnan(n_est[bg]).any(axis=-1).mean()),
        "est_nan_fraction_object": float(np.isnan(n_est[fg]).any(axis=-1).mean()),
    }
    save_json("f2_normals_regions.json", report)


if __name__ == "__main__":
    main()
