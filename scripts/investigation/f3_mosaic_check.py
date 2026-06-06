"""F3 — Mosaicos: o PSNR 2-4 dB é saturação do PNG ou mosaico errado (H3)?

Compara, por luz: (a) sMos.png (uint8, possivelmente saturado) e
(b) sMos.fni * 255 (float, unidades da fonte 0-65535) contra o sVal GT
interno de cada L* (uint16), ambos reescalados para 0-255.
"""

from __future__ import annotations

import cv2
import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_smos_pairs
from hybrid_stereo_method.evaluation.metrics import pearson_r, psnr, ssim
from hybrid_stereo_method.infrastructure.io.image_io import read_fni_to_image_array
from scripts.investigation.common import RAW, RESULTS, save_json

SOURCE_MAX = 65535.0  # stack é uint16 (log do run: max_all 65535.0)


def main() -> None:
    report: dict = {"per_light": {}}
    pairs = find_smos_pairs(RESULTS, RAW)
    assert pairs, "nenhum par sMos/sVal encontrado — find_smos_pairs mudou?"
    for light, smos_png_path, sval_path in pairs:
        png = cv2.imread(str(smos_png_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
        gt16 = cv2.imread(str(sval_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
        gt255 = gt16 * (255.0 / SOURCE_MAX)
        fni = read_fni_to_image_array(smos_png_path.parent / "sMos.fni")
        est_src = np.asarray(fni, dtype=np.float64) * 255.0  # desfaz o /255 da escrita
        est255 = est_src * (255.0 / SOURCE_MAX)
        if est255.shape != gt255.shape:  # FNI pode vir (H, W, C) vs GT (H, W) ou vice-versa
            if est255.ndim == 3 and gt255.ndim == 3 and est255.shape[-1] != gt255.shape[-1]:
                est255, gt255 = est255.mean(-1), gt255.mean(-1)
            elif est255.ndim != gt255.ndim:
                est255 = est255.mean(-1) if est255.ndim == 3 else est255
                gt255 = gt255.mean(-1) if gt255.ndim == 3 else gt255
        report["per_light"][light] = {
            "png_saturated_fraction": float((png >= 255).mean()),
            "png_psnr_vs_gt": psnr(png.astype(np.float64), gt255),
            "fni_psnr_vs_gt": psnr(est255, gt255),
            "fni_ssim_vs_gt": ssim(est255, gt255),
            "fni_pearson_vs_gt": pearson_r(est255, gt255),
            "fni_src_range": [float(est_src.min()), float(est_src.max())],
        }
        r = report["per_light"][light]
        print(f"  {light}: png_sat={r['png_saturated_fraction']:.1%} "
              f"png_psnr={r['png_psnr_vs_gt']:.1f}dB fni_psnr={r['fni_psnr_vs_gt']:.1f}dB "
              f"fni_ssim={r['fni_ssim_vs_gt']:.3f}")

    save_json("f3_mosaic_check.json", report)


if __name__ == "__main__":
    main()
