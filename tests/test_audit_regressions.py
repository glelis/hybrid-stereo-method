"""Regressões das correções da auditoria 2026-06-09 que não tinham teste.

Cada teste cita o achado do relatório
docs/superpowers/reports/2026-06-09-23-38-relatorio-correcoes-bugs.md.
"""

import numpy as np

from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d
from hybrid_stereo_method.multifocus.indicators.wavelet import calculate_wavelet_focus_indicator
from hybrid_stereo_method.multifocus.math_utils import quadratic_interpolation
from hybrid_stereo_method.photometric.wps import estimate_normals_argmax_lstsq_robust


def test_quadratic_interpolation_exact_on_parabola():
    """Hann com a = π/(0.5(m+1)): pesos positivos dentro da janela; uma
    parábola exata deve ser reproduzida em qualquer k_fuzzy fracionário."""
    k = np.arange(9, dtype=float)
    val = 10.0 - (k - 4.3) ** 2
    for k_fuzzy in (2.2, 4.3, 4.6, 6.9):
        assert abs(quadratic_interpolation(val, k_fuzzy) - (10.0 - (k_fuzzy - 4.3) ** 2)) < 1e-6


def test_wavelet_indicator_has_image_resolution_and_tracks_sharpness():
    rng = np.random.default_rng(0)
    sharp = rng.random((64, 48)).astype(np.float32)
    blurred = np.full_like(sharp, sharp.mean())
    fi_sharp = calculate_wavelet_focus_indicator(sharp)
    assert fi_sharp.shape == sharp.shape
    assert fi_sharp.mean() > calculate_wavelet_focus_indicator(blurred).mean()


def test_argmax_degenerate_cases_share_marker_and_clamp():
    n = 7
    # foco nulo → marcador degenerado (n/2, confiança 0)
    assert compute_argmax_fuzzy_1d(np.zeros(n), (0, 0), False, "") == (n / 2, 0)
    # curva convexa → mesmo marcador
    convex = (np.arange(n) - 3.0) ** 2 + 1.0
    k, conf = compute_argmax_fuzzy_1d(convex, (0, 0), False, "")
    assert (k, conf) == (n / 2, 0)
    # pico no último frame → vértice limitado a [0, n-1]
    rising = np.arange(n, dtype=float) ** 2
    k, _ = compute_argmax_fuzzy_1d(rising, (0, 0), False, "")
    assert 0.0 <= k <= n - 1


def test_wps_albedo_is_norm_before_normalization():
    lights = np.array([[0, 0, 1], [0.5, 0, 0.866], [0, 0.5, 0.866], [-0.5, 0, 0.866]], float)
    n = np.array([0.2, -0.1, 1.0]) / np.linalg.norm([0.2, -0.1, 1.0])
    albedo = 150.0
    images = [np.full((2, 2), albedo * max(lights[i] @ n, 0.0)) for i in range(len(lights))]
    normals, alb, conf, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
    np.testing.assert_allclose(normals[0, 0], n, atol=1e-6)
    np.testing.assert_allclose(alb[0, 0], albedo, rtol=1e-6)
    assert 0 < conf[0, 0] <= 1
