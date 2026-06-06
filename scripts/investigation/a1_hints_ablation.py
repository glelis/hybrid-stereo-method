"""A1 — Ablação dos hints: quanto da degradação vem dos hints incomensuráveis (H4)?

(a) sem hints           -> efeito puro das normais estimadas
(b) com hints (weight .1) -> deve reproduzir ~ o run original (pearson ~ -0.20)
(c) sem hints, sem confiança -> papel do canal de peso
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from scripts.investigation.common import (
    CLIFF_THRESHOLD,
    OUT,
    RESULTS,
    height_metrics_vs_gt,
    save_json,
    summarize,
)


def main() -> None:
    normals = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")
    confidence = np.load(RESULTS / "photometric_stereo" / "confidence.npy")
    n4 = np.concatenate([normals, confidence[..., None].astype(normals.dtype)], axis=-1)
    hints_fni = RESULTS / "integration" / "hints_vertex.fni"

    cases = {
        "no_hints": dict(normal_map=n4, hints_fni_path=None, hints_weight=0.0),
        "with_hints_w0.1": dict(normal_map=n4, hints_fni_path=hints_fni, hints_weight=0.1),
        "no_hints_no_conf": dict(normal_map=normals, hints_fni_path=None, hints_weight=0.0),
    }
    report: dict = {}
    for label, kw in cases.items():
        height = integrate_normals_to_height(
            output_dir=OUT / "a1_hints" / label,
            output_prefix="a1",
            config=IntegrateRecursiveConfig(),
            **kw,
        )
        m = height_metrics_vs_gt(height)
        m["cliff_fraction"] = float((height < CLIFF_THRESHOLD).mean())
        m["height_stats"] = summarize(label, height)
        report[label] = m
        print(f"  {label}: pearson={m['pearson_r']:+.3f} cliff={m['cliff_fraction']:.1%}")

    save_json("a1_hints_ablation.json", report)


if __name__ == "__main__":
    main()
