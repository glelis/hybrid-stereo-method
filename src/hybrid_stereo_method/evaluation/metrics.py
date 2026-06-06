"""Métricas puras de avaliação (arrays → números). Sem I/O.

Origem única das métricas usadas pela avaliação automatizada e pelos testes
(`tests/synthetic_utils.py` reexporta `affine_fit_rmse` daqui).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def affine_fit_rmse(est: np.ndarray, gt: np.ndarray) -> tuple[float, tuple[float, float]]:
    """RMSE de gt vs (a*est + b) com a, b ótimos por mínimos quadrados.

    Atenção: o fit afim absorve escala global, offset E INVERSÃO DE SINAL —
    use-o para medir forma; convenções de sinal são decididas por testes
    dedicados (ver auditoria CONV-1). Retorna (rmse, (a, b)).
    """
    est_flat = np.asarray(est, dtype=np.float64).ravel()
    gt_flat = np.asarray(gt, dtype=np.float64).ravel()
    a = np.stack([est_flat, np.ones_like(est_flat)], axis=1)
    coef, *_ = np.linalg.lstsq(a, gt_flat, rcond=None)
    rmse = float(np.sqrt(np.mean((a @ coef - gt_flat) ** 2)))
    return rmse, (float(coef[0]), float(coef[1]))


def pearson_r(est: np.ndarray, gt: np.ndarray) -> float:
    """Correlação de Pearson; NaN se alguma das séries for constante."""
    est_flat = np.asarray(est, dtype=np.float64).ravel()
    gt_flat = np.asarray(gt, dtype=np.float64).ravel()
    if est_flat.std() == 0.0 or gt_flat.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(est_flat, gt_flat)[0, 1])


def angular_error_deg(n_est: np.ndarray, n_gt: np.ndarray) -> np.ndarray:
    """Erro angular por pixel, em graus, entre campos de normais (H, W, 3).

    As entradas são renormalizadas; o produto interno é clampado a [-1, 1]
    para não gerar NaN por arredondamento. NaN nas entradas propaga para o
    pixel correspondente do resultado.
    """
    a = np.asarray(n_est, dtype=np.float64)
    b = np.asarray(n_gt, dtype=np.float64)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = np.sum(a * b, axis=-1) / (norm_a * norm_b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def psnr(est: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    """PSNR em dB (inf para imagens idênticas)."""
    return float(peak_signal_noise_ratio(gt, est, data_range=data_range))


def ssim(est: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    """SSIM em [-1, 1]; imagens coloridas usam channel_axis=-1."""
    kwargs: dict = {"channel_axis": -1} if est.ndim == 3 else {}
    return float(structural_similarity(gt, est, data_range=data_range, **kwargs))
