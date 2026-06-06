"""F0 — Sanidade do GT e da comparação da avaliação (H6).

Perguntas:
1. O GT sharp/ da raiz corresponde às imagens do stack (melon14)?
2. Shapes/dtypes de todos os artefatos GT e estimados são compatíveis?
3. zMos vs hAvg: qual o sinal natural da relação?
"""

from __future__ import annotations

import cv2
import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_height_map,
    load_normals_gt,
    load_zmos,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.metrics import pearson_r
from scripts.investigation.common import RAW, RESULTS, STACK_ROOT, save_json, summarize


def gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64).mean(axis=-1) if img.ndim == 3 else img.astype(np.float64)


def main() -> None:
    report: dict = {}

    # 1. dtypes/shapes do GT da raiz
    sharp = find_sharp_dir(RESULTS, RAW)
    print(f"sharp dir: {sharp}")
    for name in ("hAvg.png", "hDev.png", "sNrm.png", "sVal.png", "shrp.png"):
        img = cv2.imread(str(sharp / name), cv2.IMREAD_UNCHANGED)
        report[name] = (
            {"shape": list(img.shape), "dtype": str(img.dtype),
             "min": float(img.min()), "max": float(img.max())}
            if img is not None else "ausente/ilegível"
        )
        print(f"  {name}: {report[name]}")

    # 2. correspondência GT <-> imagens (melon14?)
    root_sval = gray(cv2.imread(str(sharp / "sVal.png"), cv2.IMREAD_UNCHANGED))
    l000_sval = gray(cv2.imread(str(STACK_ROOT / "L000" / "sharp" / "sVal.png"),
                                cv2.IMREAD_UNCHANGED))
    zf_imgs = sorted((STACK_ROOT / "L000").glob("zf*/*sVal.png"))
    assert zf_imgs, f"nenhum sVal.png nas pastas zf de {STACK_ROOT / 'L000'}"
    stack_mean = np.mean(
        [gray(cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) for p in zf_imgs], axis=0
    )
    report["gt_vs_images"] = {
        "r_rootSval_vs_L000sharpSval": pearson_r(root_sval, l000_sval),
        "r_rootSval_vs_L000stackMean": pearson_r(root_sval, stack_mean),
        "n_zf_frames_L000": len(zf_imgs),
    }
    print(f"  GT<->imagens: {report['gt_vs_images']}")

    # 3. shapes da comparação da avaliação + sinal zMos vs hAvg
    gt_h = load_height_gt(sharp)
    zmos = load_zmos(RESULTS)
    h_cell = vertex_to_cell(load_height_map(RESULTS))
    report["shapes"] = {"hAvg": list(gt_h.shape), "zMos": list(zmos.shape),
                       "height_cell": list(h_cell.shape)}
    valid = np.isfinite(zmos) & np.isfinite(gt_h)
    report["zmos_vs_havg"] = {"pearson": pearson_r(zmos[valid], gt_h[valid])}
    report["stats"] = [summarize("hAvg", gt_h), summarize("zMos", zmos),
                       summarize("height_cell", h_cell)]
    n_gt, fg = load_normals_gt(sharp)
    report["snrm_foreground_fraction"] = float(fg.mean())
    print(f"  zMos vs hAvg pearson: {report['zmos_vs_havg']['pearson']:.3f}; "
          f"sNrm foreground: {report['snrm_foreground_fraction']:.1%}")

    save_json("f0_gt_sanity.json", report)


if __name__ == "__main__":
    main()
