"""Avaliação automatizada dos resultados do pipeline híbrido.

Uso standalone:
    python -m hybrid_stereo_method.evaluation.main --results_dir <pasta> [--data_dir <pasta>]

Uso como hook (hybrid/main.py):
    run_evaluation(output_path, data_dir=..., config=parameters.get("evaluation"))

Cada etapa é avaliada de forma independente: artefato ou GT ausente gera
status "skipped: <motivo>" e as demais etapas seguem. O CLI retorna exit
code != 0 apenas se NENHUMA etapa pôde ser avaliada.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hybrid_stereo_method.evaluation.evaluators import (
    evaluate_focus_selection,
    evaluate_height,
    evaluate_hybrid_gain,
    evaluate_mosaics,
    evaluate_normals,
)
from hybrid_stereo_method.evaluation.loaders import (
    MissingArtifactError,
    find_sharp_dir,
    find_smos_pairs,
    load_hdev_mask,
    load_height_gt,
    load_height_map,
    load_normal_map,
    load_normals_gt,
    load_shrp_z_gt,
    load_zmos,
    resolve_data_dir,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.report import (
    save_error_map,
    write_metrics_json,
    write_report_md,
)

ERROR_MAP_FILES = {
    "multifocus_depth": "multifocus_depth_error.png",
    "focus_selection": "multifocus_focus_selection_error.png",
    "photometric_normals": "photometric_angular_error.png",
    "integration_height": "integration_height_error.png",
}


def _package_version() -> str:
    try:
        return version("hybrid-stereo-method")
    except PackageNotFoundError:
        return "unknown"


def _to_255_scale(img: np.ndarray) -> np.ndarray:
    """Imagem inteira (8 ou 16 bits) → float64 na escala 0-255 (PSNR/SSIM comuns)."""
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float64) * (255.0 / float(np.iinfo(img.dtype).max))
    return img.astype(np.float64)


# Exceções de artefato ilegível/corrompido que viram "skipped" por etapa
# (spec: "avaliação parcial é melhor que nenhuma"; nunca exceção não tratada).
_LOAD_ERRORS = (MissingArtifactError, ValueError, OSError)


def run_evaluation(
    results_dir: str | Path,
    data_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Avalia os resultados em results_dir contra o ground truth disponível.

    Escreve metrics.json, report.md e mapas de erro em <results_dir>/evaluation/
    e retorna o dict de métricas (com os arrays "_error_map" já removidos das
    saídas serializadas, mas presentes no dict retornado).
    """
    config = config or {}
    results_dir = Path(results_dir)
    out_dir = results_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(results_dir, data_dir)

    metrics: dict[str, Any] = {
        "meta": {
            "results_dir": str(results_dir),
            "data_dir": str(data_dir) if data_dir is not None else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "package_version": _package_version(),
            "config": dict(config),
        }
    }

    # --- ground truth canônico de altura/normais (sharp/) -------------------
    gt_height: np.ndarray | None = None
    gt_valid: np.ndarray | None = None
    sharp_dir: Path | None = None
    gt_reason = ""
    try:
        sharp_dir = find_sharp_dir(results_dir, data_dir)
        gt_height = load_height_gt(sharp_dir)
        if config.get("hdev_mask", False):
            gt_valid = load_hdev_mask(sharp_dir, float(config.get("hdev_threshold", 0.1)))
            if gt_valid is None:
                logging.warning("hdev_mask habilitado, mas hDev.png ausente — sem máscara")
    except _LOAD_ERRORS as exc:
        gt_reason = str(exc)
        logging.warning("GT de altura indisponível: %s", gt_reason)

    # --- 1. multifocus: profundidade ----------------------------------------
    zmos: np.ndarray | None = None
    try:
        zmos = load_zmos(results_dir)
    except _LOAD_ERRORS as exc:
        metrics["multifocus_depth"] = {"status": f"skipped: {exc}"}
    if zmos is not None:
        if gt_height is None:
            metrics["multifocus_depth"] = {"status": f"skipped: {gt_reason}"}
        else:
            metrics["multifocus_depth"] = evaluate_height(zmos, gt_height)
            if gt_valid is not None:
                metrics["multifocus_depth_hdev_masked"] = evaluate_height(zmos, gt_height, gt_valid)

    # --- 2. multifocus: seleção de foco -------------------------------------
    if zmos is None:
        metrics["focus_selection"] = {"status": "skipped: zMos.fni indisponível"}
    elif data_dir is None:
        metrics["focus_selection"] = {
            "status": "skipped: data_dir não informado (pilha shrp vive no dataset)"
        }
    else:
        try:
            z_gt, z_vals = load_shrp_z_gt(data_dir)
            metrics["focus_selection"] = evaluate_focus_selection(zmos, z_gt, z_vals)
        except _LOAD_ERRORS as exc:
            metrics["focus_selection"] = {"status": f"skipped: {exc}"}

    # --- 3. multifocus: mosaicos por luz ------------------------------------
    if data_dir is None:
        metrics["mosaics"] = {
            "status": "skipped: data_dir não informado (sVal por luz vive no dataset)"
        }
    else:
        pairs = find_smos_pairs(results_dir, data_dir)
        loaded_pairs = []
        for light, est_path, gt_path in pairs:
            est = cv2.imread(str(est_path), cv2.IMREAD_UNCHANGED)
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if est is None or gt_img is None:
                logging.warning("mosaico ilegível para %s — pulado", light)
                continue
            loaded_pairs.append((light, _to_255_scale(est), _to_255_scale(gt_img)))
        metrics["mosaics"] = evaluate_mosaics(loaded_pairs)

    # --- 4. fotométrico: normais --------------------------------------------
    try:
        n_est = load_normal_map(results_dir)
        if sharp_dir is None:
            metrics["photometric_normals"] = {
                "status": f"skipped: pasta sharp/ indisponível ({gt_reason})"
            }
        else:
            n_gt, fg = load_normals_gt(sharp_dir)
            metrics["photometric_normals"] = evaluate_normals(n_est, n_gt, fg)
    except _LOAD_ERRORS as exc:
        metrics["photometric_normals"] = {"status": f"skipped: {exc}"}

    # --- 5. integração: altura final ----------------------------------------
    height_cell: np.ndarray | None = None
    try:
        height_vertex = load_height_map(results_dir)
        if height_vertex.ndim == 3:
            height_vertex = height_vertex[..., 0]
        height_cell = vertex_to_cell(height_vertex)
    except _LOAD_ERRORS as exc:
        metrics["integration_height"] = {"status": f"skipped: {exc}"}
    if height_cell is not None:
        if gt_height is None:
            metrics["integration_height"] = {"status": f"skipped: {gt_reason}"}
        else:
            metrics["integration_height"] = evaluate_height(height_cell, gt_height)
            if gt_valid is not None:
                metrics["integration_height_hdev_masked"] = evaluate_height(
                    height_cell, gt_height, gt_valid
                )

    # --- 6. síntese: ganho do híbrido ----------------------------------------
    if zmos is None or height_cell is None or gt_height is None:
        metrics["hybrid_gain"] = {
            "status": "skipped: requer zMos, height_map e GT de altura simultaneamente"
        }
    else:
        metrics["hybrid_gain"] = evaluate_hybrid_gain(zmos, height_cell, gt_height, gt_valid)

    # --- saídas ---------------------------------------------------------------
    error_map_files: dict[str, str] = {}
    for stage, fname in ERROR_MAP_FILES.items():
        block = metrics.get(stage, {})
        emap = block.get("_error_map")
        if emap is not None:
            save_error_map(emap, out_dir / fname, title=stage)
            error_map_files[stage] = fname

    write_metrics_json(metrics, out_dir)
    write_report_md(metrics, error_map_files, out_dir)

    n_ok = sum(1 for k, v in metrics.items() if k != "meta" and v.get("status") == "ok")
    logging.info("Avaliação concluída: %d etapa(s) ok — saídas em %s", n_ok, out_dir)
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliação automatizada dos resultados do pipeline híbrido."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Pasta de resultados de um experimento (timestamped).",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Pasta do dataset de entrada (para GT por luz e pilha shrp). "
        "Se omitida, tenta o parameters.yaml salvo no resultado.",
    )
    parser.add_argument(
        "--hdev_mask",
        action="store_true",
        help="Também reporta métricas excluindo pixels de GT incerto (hDev.png).",
    )
    parser.add_argument("--hdev_threshold", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "evaluation.log"),
        ],
    )

    metrics = run_evaluation(
        args.results_dir,
        data_dir=args.data_dir,
        config={"hdev_mask": args.hdev_mask, "hdev_threshold": args.hdev_threshold},
    )
    n_ok = sum(1 for k, v in metrics.items() if k != "meta" and v.get("status") == "ok")
    if n_ok == 0:
        logging.error("Nenhuma etapa pôde ser avaliada.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
