"""Avaliadores por etapa: recebem arrays, devolvem dict de métricas.

Contrato comum dos retornos:
- "status": "ok" ou "skipped: <motivo>";
- chaves iniciadas por "_" carregam arrays (mapas de erro) e são removidas
  pela serialização JSON em report.py;
- "valid_fraction" e "low_validity" (< 1% de pixels válidos) sempre presentes
  quando status == "ok".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_stereo_method.evaluation.metrics import (
    affine_fit_rmse,
    angular_error_deg,
    pearson_r,
    psnr,
    ssim,
)

LOW_VALIDITY_THRESHOLD = 0.01


def _skipped(reason: str) -> dict[str, Any]:
    return {"status": f"skipped: {reason}"}


def evaluate_height(
    est: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray | None = None
) -> dict[str, Any]:
    """Métricas afim-invariantes entre um mapa de altura estimado e o GT.

    est/gt em cell-grid (H, W); NaN no estimado é inválido; gt_valid é uma
    máscara opcional de confiabilidade do GT (ex.: derivada de hDev.png).
    O fit é gt ≈ a*est + b; o mapa de erro é |(a*est + b) - gt| (NaN fora
    da máscara), nas unidades do GT.
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if est.shape != gt.shape:
        return _skipped(f"shapes incompatíveis: estimado {est.shape} vs GT {gt.shape}")
    valid = np.isfinite(est) & np.isfinite(gt)
    if gt_valid is not None:
        valid &= gt_valid
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _skipped("nenhum pixel válido em comum")
    rmse, (a, b) = affine_fit_rmse(est[valid], gt[valid])
    error_map = np.where(valid, np.abs(a * est + b - gt), np.nan)
    valid_fraction = n_valid / est.size
    return {
        "status": "ok",
        "rmse_affine": rmse,
        "mae_affine": float(np.nanmean(error_map)),
        "a": a,
        "b": b,
        "pearson_r": pearson_r(est[valid], gt[valid]),
        "gt_std": float(gt[valid].std()),
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
        "_error_map": error_map,
    }


def evaluate_focus_selection(
    zmos: np.ndarray, z_gt: np.ndarray, z_vals: list[float]
) -> dict[str, Any]:
    """Erro de seleção de foco: zMos (z_foc) vs z_gt = z_foc[argmax(shrp)].

    Unidades já comensuráveis (sem fit afim). O erro em frames divide pelo
    passo focal (mediana dos deltas de z_vals). "Acerto exato" = erro <= 0.5
    frame (zMos é sub-pixel; 0.5 arredonda para o frame GT).
    """
    zmos = np.asarray(zmos, dtype=np.float64)
    z_gt = np.asarray(z_gt, dtype=np.float64)
    if zmos.shape != z_gt.shape:
        return _skipped(f"shapes incompatíveis: zMos {zmos.shape} vs z_gt {z_gt.shape}")
    step = float(np.median(np.diff(sorted(z_vals)))) if len(z_vals) > 1 else 1.0
    valid = np.isfinite(zmos) & np.isfinite(z_gt)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _skipped("nenhum pixel válido")
    error_frames = np.where(valid, np.abs(zmos - z_gt) / step, np.nan)
    e = error_frames[valid]
    valid_fraction = n_valid / zmos.size
    return {
        "status": "ok",
        "step_z": step,
        "n_frames": len(z_vals),
        "median_err_frames": float(np.median(e)),
        "mean_err_frames": float(np.mean(e)),
        "p90_err_frames": float(np.percentile(e, 90)),
        "exact_match_pct": float(np.mean(e <= 0.5) * 100.0),
        "within_1_frame_pct": float(np.mean(e <= 1.0) * 100.0),
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
        "_error_map": error_frames,
    }


def evaluate_mosaics(mosaics: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """PSNR/SSIM por luz entre sMos estimado e sVal GT, + agregados.

    `mosaics` é uma lista (luz, imagem_estimada, imagem_gt), ambas uint8 e na
    MESMA ordem de canais (BGR do cv2 nos dois lados — comparação consistente).
    """
    if not mosaics:
        return _skipped("nenhum par sMos/sVal encontrado")
    per_light: dict[str, dict[str, Any]] = {}
    for light, est, gt in mosaics:
        if est.shape != gt.shape:
            per_light[light] = {
                "status": f"skipped: shapes incompatíveis {est.shape} vs {gt.shape}"
            }
            continue
        per_light[light] = {
            "status": "ok",
            "psnr": psnr(est, gt),
            "ssim": ssim(est, gt),
        }
    oks = {k: v for k, v in per_light.items() if v["status"] == "ok"}
    if not oks:
        return {"status": "skipped: nenhum par com shapes compatíveis", "per_light": per_light}
    ssims = {k: v["ssim"] for k, v in oks.items()}
    return {
        "status": "ok",
        "per_light": per_light,
        "n_lights": len(oks),
        "psnr_mean": float(np.mean([v["psnr"] for v in oks.values()])),
        "psnr_min": float(np.min([v["psnr"] for v in oks.values()])),
        "ssim_mean": float(np.mean(list(ssims.values()))),
        "ssim_min": float(np.min(list(ssims.values()))),
        "worst_light": min(ssims, key=lambda k: ssims[k]),
    }


def evaluate_normals(
    n_est: np.ndarray, n_gt: np.ndarray, gt_foreground: np.ndarray
) -> dict[str, Any]:
    """Erro angular (graus) das normais estimadas vs GT, nas duas orientações de y.

    CONV-2: o frame de y do GT (sNrm de render y-up vs numpy y-down) é
    desconhecido a priori; o erro é computado com o GT como está ("y_as_is")
    e com ny invertido ("y_flipped"); a orientação de menor erro médio vira
    "winning_orientation" e seu mapa de erro é exposto em "_error_map".
    """
    n_est = np.asarray(n_est, dtype=np.float64)[..., :3]
    n_gt = np.asarray(n_gt, dtype=np.float64)
    if n_est.shape[:2] != n_gt.shape[:2]:
        return _skipped(f"shapes incompatíveis: estimado {n_est.shape[:2]} vs GT {n_gt.shape[:2]}")
    results: dict[str, dict[str, Any]] = {}
    error_maps: dict[str, np.ndarray] = {}
    for label, flip in (("y_as_is", False), ("y_flipped", True)):
        gt = n_gt.copy()
        if flip:
            gt[..., 1] *= -1.0
        ae = angular_error_deg(n_est, gt)
        valid = np.isfinite(ae) & gt_foreground
        n_valid = int(valid.sum())
        if n_valid == 0:
            return _skipped("nenhum pixel válido (estimado NaN ou GT sem frente)")
        vals = ae[valid]
        valid_fraction = n_valid / ae.size
        results[label] = {
            "mean_deg": float(vals.mean()),
            "median_deg": float(np.median(vals)),
            "p95_deg": float(np.percentile(vals, 95)),
            "valid_fraction": valid_fraction,
        }
        error_maps[label] = np.where(valid, ae, np.nan)
    winner = min(results, key=lambda k: results[k]["mean_deg"])
    valid_fraction = results[winner]["valid_fraction"]
    return {
        "status": "ok",
        "winning_orientation": winner,
        "y_as_is": results["y_as_is"],
        "y_flipped": results["y_flipped"],
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
        "_error_map": error_maps[winner],
    }


def evaluate_hybrid_gain(
    zmos: np.ndarray,
    height_cell: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Ganho do híbrido: RMSE afim do multifocus vs do resultado final na
    INTERSEÇÃO das máscaras de pixels válidos (comparação justa).

    gain = rmse_multifocus / rmse_final (> 1 = a combinação melhorou).
    """
    zmos = np.asarray(zmos, dtype=np.float64)
    height_cell = np.asarray(height_cell, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if zmos.shape != gt.shape or height_cell.shape != gt.shape:
        return _skipped(
            f"shapes incompatíveis: zMos {zmos.shape}, altura {height_cell.shape}, GT {gt.shape}"
        )
    valid = np.isfinite(zmos) & np.isfinite(height_cell) & np.isfinite(gt)
    if gt_valid is not None:
        valid &= gt_valid
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _skipped("nenhum pixel válido em comum")
    rmse_mf, _ = affine_fit_rmse(zmos[valid], gt[valid])
    rmse_final, _ = affine_fit_rmse(height_cell[valid], gt[valid])
    valid_fraction = n_valid / gt.size
    return {
        "status": "ok",
        "rmse_multifocus": rmse_mf,
        "rmse_final": rmse_final,
        "gain": rmse_mf / rmse_final if rmse_final > 0.0 else float("inf"),
        "pearson_multifocus": pearson_r(zmos[valid], gt[valid]),
        "pearson_final": pearson_r(height_cell[valid], gt[valid]),
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
    }
