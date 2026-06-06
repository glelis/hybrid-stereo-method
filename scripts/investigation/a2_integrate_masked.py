"""A2 — Se o fundo sai da integração, o penhasco some? (prova causal de H1)

Máscara derivada do foreground do sNrm GT — só para diagnóstico.
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from scripts.investigation.common import (
    CLIFF_THRESHOLD,
    OUT,
    RAW,
    RESULTS,
    height_metrics_vs_gt,
    save_json,
    summarize,
)


def main() -> None:
    normals = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")
    confidence = np.load(RESULTS / "photometric_stereo" / "confidence.npy")
    _n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))

    weight = confidence * fg.astype(confidence.dtype)  # fundo -> peso 0
    n = np.where(np.isfinite(normals), normals, 0.0)
    n4 = np.concatenate([n, weight[..., None].astype(n.dtype)], axis=-1)

    height = integrate_normals_to_height(
        normal_map=n4,
        output_dir=OUT / "a2_masked",
        output_prefix="a2",
        config=IntegrateRecursiveConfig(),  # sem hints: isola o efeito da máscara
    )
    m = height_metrics_vs_gt(height)
    m["cliff_fraction"] = float((height < CLIFF_THRESHOLD).mean())
    m["height_stats"] = summarize("altura mascarada", height)
    # métricas restritas ao objeto (fundo integrado sem dados não é informativo)
    from hybrid_stereo_method.evaluation.evaluators import evaluate_height
    from hybrid_stereo_method.evaluation.loaders import load_height_gt, vertex_to_cell

    gt = load_height_gt(find_sharp_dir(RESULTS, RAW))
    obj = evaluate_height(vertex_to_cell(height), gt, gt_valid=fg)
    m["object_only"] = {k: v for k, v in obj.items() if not k.startswith("_")}
    print(f"  mascarado: pearson(obj)={m['object_only']['pearson_r']:+.3f} "
          f"cliff={m['cliff_fraction']:.1%}")
    save_json("a2_integrate_masked.json", m)


if __name__ == "__main__":
    main()
