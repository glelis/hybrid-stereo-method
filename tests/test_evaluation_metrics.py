"""Testes unitários das métricas puras de avaliação (casos analíticos)."""
import numpy as np
import pytest

from hybrid_stereo_method.evaluation.metrics import (
    affine_fit_rmse,
    angular_error_deg,
    pearson_r,
    psnr,
    ssim,
)


def test_affine_fit_rmse_recovers_affine_transform():
    rng = np.random.default_rng(0)
    est = rng.uniform(0.0, 10.0, (32, 32))
    gt = 2.0 * est + 3.0
    rmse, (a, b) = affine_fit_rmse(est, gt)
    assert rmse == pytest.approx(0.0, abs=1e-9)
    assert a == pytest.approx(2.0, abs=1e-9)
    assert b == pytest.approx(3.0, abs=1e-9)


def test_affine_fit_rmse_absorbs_sign_inversion():
    est = np.linspace(0.0, 1.0, 100)
    rmse, (a, _) = affine_fit_rmse(est, -est)
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert a == pytest.approx(-1.0, abs=1e-12)


def test_pearson_r_perfect_and_constant():
    x = np.arange(50, dtype=float)
    assert pearson_r(x, 3.0 * x + 1.0) == pytest.approx(1.0)
    assert pearson_r(x, -x) == pytest.approx(-1.0)
    assert np.isnan(pearson_r(x, np.ones_like(x)))  # gt constante: r indefinido


def test_angular_error_deg_known_rotation():
    # campo (2, 2, 3): identidade vs rotação de 30° em torno de x aplicada a +z
    n = np.zeros((2, 2, 3))
    n[..., 2] = 1.0  # todos +z
    rot = n.copy()
    rot[..., 1] = np.sin(np.deg2rad(30.0))
    rot[..., 2] = np.cos(np.deg2rad(30.0))
    ae = angular_error_deg(n, rot)
    assert ae.shape == (2, 2)
    assert np.allclose(ae, 30.0, atol=1e-9)


def test_angular_error_deg_identical_is_zero_and_clips_rounding():
    rng = np.random.default_rng(1)
    n = rng.normal(size=(8, 8, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ae = angular_error_deg(n, n)
    assert np.allclose(ae, 0.0, atol=1e-6)  # arccos clampado: sem NaN por arredondamento


def test_angular_error_deg_nan_propagates():
    n = np.zeros((2, 2, 3))
    n[..., 2] = 1.0
    bad = n.copy()
    bad[0, 0, :] = np.nan
    ae = angular_error_deg(bad, n)
    assert np.isnan(ae[0, 0])
    assert ae[1, 1] == pytest.approx(0.0, abs=1e-9)


def test_psnr_identical_is_inf_and_noisy_is_finite():
    img = (np.random.default_rng(2).uniform(0, 255, (32, 32))).astype(np.uint8)
    assert np.isinf(psnr(img, img))
    noisy = np.clip(img.astype(int) + 10, 0, 255).astype(np.uint8)
    assert 20.0 < psnr(noisy, img) < 40.0


def test_ssim_identical_is_one_color_and_gray():
    gray = (np.random.default_rng(3).uniform(0, 255, (32, 32))).astype(np.uint8)
    color = np.stack([gray] * 3, axis=-1)
    assert ssim(gray, gray) == pytest.approx(1.0)
    assert ssim(color, color) == pytest.approx(1.0)
