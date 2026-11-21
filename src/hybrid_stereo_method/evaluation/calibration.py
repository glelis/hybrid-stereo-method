"""Calibração da integração por dataset: sinal de ``sy`` e ``pixel_size``.

Mede, a partir do ground truth do próprio dataset, os dois parâmetros de
``hybrid.integration`` que dependem da geometria da cena/câmera:

- **sinal de sy** (``slopes_scale[1]``): integra as normais do GT (sNrm) com
  sy = +1 e sy = -1 e fica com o sinal cuja altura integrada correlaciona com a
  altura do GT (hAvg). O sinal errado lê toda inclinação "para cima" como "para
  baixo" (um domo vira sela).
- **pixel_size**: a integração devolve alturas em "pixels"; os hints (zMos)
  estão em unidades de ``z_foc``. A pilha ``shrp`` (plano de foco mais nítido por
  pixel) dá uma altura do GT em unidades de ``z_foc``; o fator do ajuste linear
  ``z_shrp ≈ pixel_size * Z_int + b`` é o tamanho do pixel nessas unidades.
  Uma segunda estimativa independente passa pelo hAvg:
  ``pixel_size = (hAvg/Z_int) / (hAvg/z_shrp)``.

Opcionalmente (``results_dir``) repete o teste de sinal com as normais
estimadas pelo pipeline, sem hints.

Uso: ``python scripts/calibrate_dataset.py --data_dir <dataset> [--results_dir <run>]``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_normals_gt,
    load_shrp_z_gt,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.metrics import pearson_r
from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)

# Diferença mínima de Pearson entre os dois sinais para a decisão ser confiável.
SIGN_MARGIN = 0.2
# Discrepância relativa máxima aceitável entre as duas estimativas de pixel_size.
PIXEL_SIZE_AGREEMENT = 0.15


def fit_line(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Mínimos quadrados y ≈ a*x + b sobre ``mask``; devolve (a, b)."""
    m = mask & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        raise ValueError("pixels insuficientes para o ajuste linear")
    design = np.vstack([x[m], np.ones(int(m.sum()))]).T
    a, b = np.linalg.lstsq(design, y[m], rcond=None)[0]
    return float(a), float(b)


def normals_with_weight(normals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(H, W, 3) + máscara → (H, W, 4) para o C: inválidos viram (0, 0, 1) com peso 0."""
    n = np.nan_to_num(np.asarray(normals, dtype=np.float32), nan=0.0)
    n[~valid] = (0.0, 0.0, 1.0)
    return np.concatenate([n, valid.astype(np.float32)[..., None]], axis=-1)


def integrate_cells(
    normals4: np.ndarray, sy: float, workdir: Path, prefix: str, max_iter: int = 10000
) -> np.ndarray:
    """Integra com slopes_scale=(1, sy) e sem hints; devolve a altura na grade de células."""
    config = IntegrateRecursiveConfig(slopes_scale=(1.0, float(sy)), max_iter=max_iter)
    z = integrate_normals_to_height(normals4, workdir, prefix, config=config)
    return vertex_to_cell(z)


def sign_test(
    normals4: np.ndarray, height_gt: np.ndarray, mask: np.ndarray, workdir: Path, tag: str
) -> dict[str, Any]:
    """Pearson(altura integrada, hAvg) para sy = +1 e -1; escolhe o sinal vencedor."""
    out: dict[str, Any] = {"heights": {}}
    for sy in (1.0, -1.0):
        zc = integrate_cells(normals4, sy, workdir / f"{tag}_sy{sy:+.0f}", "z")
        out["heights"][sy] = zc
        out[f"pearson_sy{sy:+.0f}"] = pearson_r(zc[mask], height_gt[mask])
    plus, minus = out["pearson_sy+1"], out["pearson_sy-1"]
    out["sy"] = -1.0 if minus > plus else 1.0
    out["confident"] = abs(minus - plus) >= SIGN_MARGIN
    return out


def estimate_pixel_size(
    z_int: np.ndarray, z_shrp: np.ndarray, height_gt: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Duas estimativas independentes de pixel_size (unidades de z_foc por pixel)."""
    direct, offset = fit_line(z_int, z_shrp, mask)
    h_per_zint, _ = fit_line(z_int, height_gt, mask)
    h_per_zfoc, _ = fit_line(z_shrp, height_gt, mask)
    via_havg = h_per_zint / h_per_zfoc
    estimates = [abs(direct), abs(via_havg)]
    value = float(np.mean(estimates))
    return {
        "pixel_size_direct": direct,
        "pixel_size_via_havg": via_havg,
        "pixel_size": value,
        "offset_direct": offset,
        "pearson_zint_zshrp": pearson_r(z_int[mask], z_shrp[mask]),
        # sinal negativo: z_foc cresce no sentido oposto à altura do GT
        "zfoc_axis_inverted": direct < 0,
        "agreement": abs(estimates[0] - estimates[1]) / value <= PIXEL_SIZE_AGREEMENT,
    }


def calibrate(
    data_dir: str | Path, workdir: str | Path, results_dir: str | Path | None = None
) -> dict[str, Any]:
    """Roda a calibração completa sobre um dataset (e, opcionalmente, um run)."""
    workdir = Path(workdir)
    sharp = find_sharp_dir(results_dir or data_dir, data_dir)
    height_gt = load_height_gt(sharp)
    normals_gt, fg = load_normals_gt(sharp)
    logging.info("GT de %s: %d px de primeiro plano", sharp, int(fg.sum()))

    report: dict[str, Any] = {"data_dir": str(data_dir), "sharp_dir": str(sharp)}
    gt_sign = sign_test(normals_with_weight(normals_gt, fg), height_gt, fg, workdir, "gt")
    report["sign_gt"] = {k: v for k, v in gt_sign.items() if k != "heights"}

    z_shrp, z_vals = load_shrp_z_gt(data_dir)
    report["z_foc"] = z_vals
    z_int = gt_sign["heights"][gt_sign["sy"]]
    report["pixel_size"] = estimate_pixel_size(z_int, z_shrp, height_gt, fg)

    if results_dir is not None:
        run = Path(results_dir) / "photometric_stereo"
        normals = np.load(run / "normal_map.npy")
        valid = np.isfinite(normals).all(axis=-1)
        n4 = normals_with_weight(normals, valid)
        conf_path = run / "confidence_map.npy"
        if conf_path.exists():
            n4[..., 3] *= np.nan_to_num(np.load(conf_path)).astype(np.float32)
        mask = fg & valid
        run_sign = sign_test(n4, height_gt, mask, workdir, "run")
        report["sign_pipeline"] = {k: v for k, v in run_sign.items() if k != "heights"}
    return report


def format_report(report: dict[str, Any]) -> str:
    """Relatório legível + trecho de YAML para colar em hybrid.integration."""
    s, p = report["sign_gt"], report["pixel_size"]
    lines = [
        f"Dataset: {report['data_dir']}",
        f"GT:      {report['sharp_dir']}",
        "",
        "1) Sinal de sy — Pearson(altura integrada das normais do GT, hAvg)",
        f"   sy = +1: {s['pearson_sy+1']:.4f}",
        f"   sy = -1: {s['pearson_sy-1']:.4f}",
        f"   → sy = {s['sy']:+.0f}" + ("" if s["confident"] else "  (INCERTO: diferença pequena)"),
    ]
    if "sign_pipeline" in report:
        r = report["sign_pipeline"]
        lines += [
            "   Normais do pipeline (sem hints):",
            f"   sy = +1: {r['pearson_sy+1']:.4f}   sy = -1: {r['pearson_sy-1']:.4f}"
            f"   → sy = {r['sy']:+.0f}",
        ]
        if r["sy"] != s["sy"]:
            lines.append("   ATENÇÃO: normais do pipeline e do GT preferem sinais diferentes")
    lines += [
        "",
        "2) pixel_size — unidades de z_foc por pixel",
        f"   z_foc do dataset: {report['z_foc'][0]} … {report['z_foc'][-1]} "
        f"({len(report['z_foc'])} planos)",
        f"   direto  (z_shrp ≈ a·Z_int + b): {p['pixel_size_direct']:.3f}"
        f"   (Pearson {p['pearson_zint_zshrp']:.4f})",
        f"   via hAvg:                        {p['pixel_size_via_havg']:.3f}",
        f"   → pixel_size ≈ {p['pixel_size']:.2f}"
        + ("" if p["agreement"] else "  (INCERTO: as duas estimativas divergem > 15%)"),
    ]
    if p["zfoc_axis_inverted"]:
        lines.append("   ATENÇÃO: z_foc cresce no sentido oposto à altura do GT")
    lines += [
        "",
        "Trecho para configs/hb_experiment.yaml (hybrid.integration):",
        f"    pixel_size: {p['pixel_size']:.2f}",
        f"    slopes_scale: [1.0, {s['sy']:.1f}]",
    ]
    return "\n".join(lines)
