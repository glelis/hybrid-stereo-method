"""Caminhos e helpers comuns dos scripts de investigação do run 20260606_1126.

Investigação: docs/superpowers/specs/2026-06-06-height-map-flattening-investigation-design.md
Uso: python -m scripts.investigation.<script>  (a partir da raiz do repo)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_stereo_method.evaluation.evaluators import evaluate_height
from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_height_map,
    vertex_to_cell,
)

RESULTS = Path(
    "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo/"
    "20260606_1126_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50"
)
RAW = Path(
    "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/"
    "2025-03-08-stQ-melon24-amb0.00-glo0 (2).50"
)
STACK_ROOT = RAW / "stQ-melon14-amb0.00-glo0.50" / "0512x0384-hs01-kr10"
OUT = RESULTS / "investigation"

CLIFF_THRESHOLD = -200.0  # altura abaixo da qual um pixel pertence ao penhasco


def save_json(name: str, payload: dict[str, Any]) -> Path:
    """Grava payload em OUT/name (cria a pasta) e devolve o caminho."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    print(f"[saved] {path}")
    return path


def height_metrics_vs_gt(height_vertex: np.ndarray) -> dict[str, Any]:
    """Altura vertex-grid (H+1, W+1) → métricas afins vs hAvg (mesma régua da avaliação)."""
    gt = load_height_gt(find_sharp_dir(RESULTS, RAW))
    result = evaluate_height(vertex_to_cell(height_vertex), gt)
    return {k: v for k, v in result.items() if not k.startswith("_")}


def cliff_mask_cell() -> np.ndarray:
    """Máscara cell-grid (H, W) do penhasco do run original (altura < CLIFF_THRESHOLD)."""
    return vertex_to_cell(load_height_map(RESULTS)) < CLIFF_THRESHOLD


def summarize(label: str, arr: np.ndarray) -> dict[str, Any]:
    """Percentis/estatísticas de um array (ignora NaN) com rótulo, p/ JSON e console."""
    v = np.asarray(arr, dtype=np.float64)
    v = v[np.isfinite(v)]
    stats = {
        "label": label,
        "n": int(v.size),
        "min": float(v.min()) if v.size else None,
        "p01": float(np.percentile(v, 1)) if v.size else None,
        "p50": float(np.percentile(v, 50)) if v.size else None,
        "p99": float(np.percentile(v, 99)) if v.size else None,
        "max": float(v.max()) if v.size else None,
        "mean": float(v.mean()) if v.size else None,
        "std": float(v.std()) if v.size else None,
    }
    print(f"  {label}: min={stats['min']:.3g} p50={stats['p50']:.3g} "
          f"max={stats['max']:.3g} std={stats['std']:.3g}" if v.size else f"  {label}: vazio")
    return stats
