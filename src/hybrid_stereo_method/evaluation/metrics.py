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


def detrended_pearson_rmse(
    est: np.ndarray, gt: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Pearson e RMSE após remover um tilt 2D (plano em x, y) — investigação
    2026-06-06 (C6).

    Integração não-ancorada injeta uma rampa 2D espúria que o fit afim 1D
    (``affine_fit_rmse``) não remove. Aqui:
    - ``pearson``: correlação PARCIAL de est e gt dado (x, y) — correlação dos
      resíduos de cada um após regredir contra o plano [x, y, 1]. NaN se algum
      resíduo for constante.
    - ``rmse``: resíduo do ajuste conjunto ``gt ≈ a*est + bx*x + by*y + c``,
      nas unidades do GT.

    Todas as entradas são 1-D (pixels válidos já selecionados) e do mesmo tamanho.
    """
    est = np.asarray(est, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    plane = np.stack([x, y, np.ones_like(x)], axis=1)

    def _residual(v: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(plane, v, rcond=None)
        return v - plane @ coef

    est_r = _residual(est)
    gt_r = _residual(gt)
    # Detecta resíduo numericamente constante (pode ter ruído FP mesmo quando
    # est é exatamente um plano): compara std com a escala de est (relativo).
    est_scale = float(np.abs(est).mean()) or 1.0
    gt_scale = float(np.abs(gt).mean()) or 1.0
    if est_r.std() < est_scale * 1e-10 or gt_r.std() < gt_scale * 1e-10:
        pearson = float("nan")
    else:
        pearson = pearson_r(est_r, gt_r)

    design = np.stack([est, x, y, np.ones_like(est)], axis=1)
    coef, *_ = np.linalg.lstsq(design, gt, rcond=None)
    rmse = float(np.sqrt(np.mean((design @ coef - gt) ** 2)))
    return pearson, rmse


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
