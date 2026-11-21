"""Calibração de sy e pixel_size (scripts/calibrate_dataset.py) em dados com resposta conhecida."""

import numpy as np
import pytest

from hybrid_stereo_method.evaluation.calibration import (
    estimate_pixel_size,
    fit_line,
    normals_with_weight,
    sign_test,
)
from hybrid_stereo_method.hybrid.integrate import DEFAULT_EXECUTABLE

needs_binary = pytest.mark.skipif(not DEFAULT_EXECUTABLE.exists(), reason="binário C não compilado")

SIZE, PIX = 48, 2.0


def scene():
    """Altura em unidades de z_foc: rampa em y + bump fora do centro (assimétrica em y)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    z = 0.6 * yy + 25.0 * np.exp(-(((xx - 30) ** 2 + (yy - 15) ** 2) / (2 * 6.0**2)))
    return z


def normals_y_up(z, pixel_size):
    """Normais no frame Y-up (convenção Stolfi): ny com sinal trocado em relação ao numpy."""
    dzdy, dzdx = np.gradient(z)  # por pixel
    gx, gy = dzdx / pixel_size, dzdy / pixel_size  # inclinações físicas
    n = np.stack([-gx, +gy, np.ones_like(z)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def test_fit_line_recovers_slope_and_offset():
    x = np.linspace(0, 10, 50)
    a, b = fit_line(x, 3.0 * x - 2.0, np.ones_like(x, bool))
    assert a == pytest.approx(3.0) and b == pytest.approx(-2.0)


def test_estimate_pixel_size_without_binary():
    rng = np.random.default_rng(0)
    z_int = rng.random((20, 20)) * 50  # altura "em pixels"
    z_shrp = PIX * z_int + 7.0  # mesma superfície em unidades de z_foc
    h = 300.0 * z_shrp  # hAvg em níveis de cinza
    out = estimate_pixel_size(z_int, z_shrp, h, np.ones_like(z_int, bool))
    assert out["pixel_size_direct"] == pytest.approx(PIX)
    assert out["pixel_size_via_havg"] == pytest.approx(PIX)
    assert out["agreement"] and not out["zfoc_axis_inverted"]


@needs_binary
def test_sign_and_pixel_size_on_synthetic_y_up_scene(tmp_path):
    z = scene()
    valid = np.ones_like(z, bool)
    n4 = normals_with_weight(normals_y_up(z, PIX), valid)

    sign = sign_test(n4, z, valid, tmp_path, "synth")
    assert sign["sy"] == -1.0 and sign["confident"], sign

    steps = np.arange(0, z.max() + 10, 5.0)  # pilha de foco com passo 5
    z_shrp = steps[np.abs(z[..., None] - steps).argmin(-1)]
    out = estimate_pixel_size(sign["heights"][-1.0], z_shrp, 100.0 * z, valid)
    # ~5% de erro de discretização (gradiente por diferenças finitas no bump):
    # suficiente para escolher a escala, não é uma medida de precisão fina.
    assert out["pixel_size_via_havg"] == pytest.approx(PIX, rel=0.1)
    assert out["pixel_size_direct"] == pytest.approx(PIX, rel=0.1)
