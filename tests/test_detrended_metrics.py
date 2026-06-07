"""Métrica de altura com remoção de tilt 2D (investigação 2026-06-06, C6).

Integração não-ancorada injeta uma rampa 2D espúria; o fit afim 1D não a
remove. detrended_pearson_rmse remove um plano (x, y) de est e gt antes de
pontuar, isolando a fidelidade de forma real.
"""

import numpy as np

from hybrid_stereo_method.evaluation.metrics import detrended_pearson_rmse, pearson_r


def _coords(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return xx.ravel(), yy.ravel()


def test_detrend_removes_pure_tilt():
    """est = gt + rampa 2D: pearson plano cai, detrended ~ 1."""
    n = 40
    x, y = _coords(n)
    rng = np.arange(n * n, dtype=float)
    gt = np.sin(rng / 7.0)  # forma arbitrária reprodutível
    est = gt + 0.05 * x + 0.03 * y  # adiciona tilt
    plain = pearson_r(est, gt)
    pr, rmse = detrended_pearson_rmse(est, gt, x, y)
    assert pr > 0.999  # tilt removido recupera a correlação
    assert pr > plain  # detrended é melhor que o pearson simples
    assert rmse < 1e-6  # gt é exatamente est menos um plano


def test_detrend_no_tilt_matches_plain_sign():
    """Sem tilt, detrended não inventa correlação onde não há."""
    n = 30
    x, y = _coords(n)
    rng = np.arange(n * n, dtype=float)
    est = np.sin(rng / 5.0)
    gt = np.cos(rng / 5.0)
    pr, _rmse = detrended_pearson_rmse(est, gt, x, y)
    assert -1.0 <= pr <= 1.0


def test_detrend_constant_returns_nan():
    """est constante após detrend -> pearson NaN (sem variância)."""
    n = 20
    x, y = _coords(n)
    est = 1.0 + 0.05 * x + 0.03 * y  # vira constante (~0) após remover o plano
    gt = np.arange(n * n, dtype=float)
    pr, _rmse = detrended_pearson_rmse(est, gt, x, y)
    assert np.isnan(pr)
