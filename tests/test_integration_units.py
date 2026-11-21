"""INT-04 / CONV-4: hints em unidades de `z_foc` vs alturas integradas em pixel.

Os slopes que o C integra (dZ/dX = -nx/nz) são adimensionais e somados sobre
células de 1 px, então `Z` sai em "altura por pixel". Os hints (zMos absoluto)
estão em unidades de `z_foc`. `hybrid.integration.pixel_size` reconcilia as
escalas: `build_integration_config` multiplica os sinais de `slopes_scale`
(sy = -1 para datasets Y-up) por `pixel_size`.

Portado (e adaptado à convenção de sinais da main) da branch
worktree-vectorized-snuggling-harp, commit 9324de1.
"""

import logging

import numpy as np
import pytest
from synthetic_utils import affine_fit_rmse

from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    integrate_normals_to_height,
    integrate_slopes_to_height,
)
from hybrid_stereo_method.hybrid.main import build_integration_config

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

# Normais sintéticas no frame de imagem do numpy (y para baixo) → sinais (1, 1).
NUMPY_FRAME = {"slopes_scale": [1.0, 1.0]}
SIZE, AX, AY, PIX = 32, 0.05, 0.02, 2.5


def constant_normals(size: int, ax: float, ay: float) -> np.ndarray:
    n = np.array([-ax, -ay, 1.0]) / np.sqrt(1.0 + ax**2 + ay**2)
    return np.broadcast_to(n, (size, size, 3)).copy()


def physical_ramp_vertices(size: int, ax: float, ay: float, pixel_size: float) -> np.ndarray:
    y, x = np.mgrid[0 : size + 1, 0 : size + 1].astype(np.float64)
    return pixel_size * (ax * x + ay * y)


# --- build_integration_config (sem binário) ---------------------------------


def test_pixel_size_multiplies_axis_signs():
    cfg = build_integration_config({"pixel_size": 2.5, "slopes_scale": [1.0, -1.0]}, debug=False)
    assert cfg.slopes_scale == (2.5, -2.5)


def test_default_signs_are_y_up_without_pixel_size():
    assert build_integration_config({}, debug=False).slopes_scale == (1.0, -1.0)


def test_rejects_non_positive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size"):
        build_integration_config({"pixel_size": 0}, debug=False)


def test_config_threads_solver_params():
    params = {"initial_noise": 0.1, "max_level": 5, "max_iter": 123, "conv_tol": 1e-3}
    cfg = build_integration_config(params, debug=True)
    assert (cfg.initial_noise, cfg.max_level, cfg.max_iter, cfg.conv_tol) == (0.1, 5, 123, 1e-3)
    assert cfg.verbose is True


def test_warns_only_when_hints_used_without_pixel_size(caplog):
    with caplog.at_level(logging.WARNING):
        build_integration_config({"use_hints": True}, debug=False)
    assert any("pixel_size" in m and "INT-04" in m for m in caplog.messages)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        build_integration_config({"use_hints": True, "pixel_size": 2.5}, debug=False)
        build_integration_config({}, debug=False)
    assert not any("pixel_size" in m for m in caplog.messages)


def test_initial_method_hints_requires_use_hints():
    assert build_integration_config({}, debug=False).initial_method == "zero"
    with pytest.raises(ValueError, match="use_hints"):
        build_integration_config({"initial_method": "hints"}, debug=False)


# --- comportamento físico via binário C -------------------------------------


@needs_binary
def test_pixel_size_yields_heights_in_zfoc_units(tmp_path):
    gt = physical_ramp_vertices(SIZE, AX, AY, PIX)
    cfg = build_integration_config({**NUMPY_FRAME, "pixel_size": PIX}, debug=False)
    z = integrate_normals_to_height(constant_normals(SIZE, AX, AY), tmp_path, "p", config=cfg)
    rmse, (a, _b) = affine_fit_rmse(z, gt)
    assert rmse < 0.05 * gt.std()
    assert abs(a - 1.0) < 0.02, f"altura fora das unidades de z_foc: a={a:.4f}"


@needs_binary
def test_without_pixel_size_heights_are_biased_by_pixel_size(tmp_path):
    gt = physical_ramp_vertices(SIZE, AX, AY, PIX)
    cfg = build_integration_config(NUMPY_FRAME, debug=False)
    z = integrate_normals_to_height(constant_normals(SIZE, AX, AY), tmp_path, "u", config=cfg)
    _rmse, (a, _b) = affine_fit_rmse(z, gt)
    assert abs(a - PIX) < 0.02 * PIX


@needs_binary
def test_dominant_hints_in_zfoc_units_agree_with_scaled_slopes(tmp_path):
    """Hints em z_foc e slopes escalados descrevem a MESMA superfície: com
    peso de hints dominante não há tensão entre os termos. Com o BUG-006
    corrigido (szero só sem hints) os hints também ancoram o nível absoluto."""
    gt = physical_ramp_vertices(SIZE, AX, AY, PIX)
    hints = np.stack([gt, np.ones_like(gt)], axis=-1)
    cfg = build_integration_config(
        {**NUMPY_FRAME, "pixel_size": PIX, "use_hints": True}, debug=False
    )
    z = integrate_normals_to_height(
        constant_normals(SIZE, AX, AY), tmp_path, "h", config=cfg,
        hints_map=hints, hints_weight=1000.0,
    )
    rms = float(np.sqrt(((z - gt) ** 2).mean()))
    assert rms < 0.02 * gt.std(), f"hints e slopes inconsistentes: rms={rms:.4f}"


@needs_binary
def test_two_channel_slope_map_gets_unit_weight(tmp_path):
    """O C exige 3 canais em -slopes; um mapa (H, W, 2) abortava o binário."""
    slopes = np.zeros((16, 16, 2), dtype=np.float32)
    slopes[..., 0] = 0.1
    z = integrate_slopes_to_height(
        slopes, tmp_path, "s", config=build_integration_config(NUMPY_FRAME, debug=False)
    )
    assert z.shape == (17, 17)
    assert np.all(np.diff(z, axis=1) > 0), "rampa em X não recuperada"


@needs_binary
def test_relative_output_dir_and_hints_path(tmp_path, monkeypatch):
    """O binário roda com cwd=output_dir: caminhos relativos (output do YAML
    'data/results/...') faziam o C falhar em 'reading the normal map {N}'."""
    monkeypatch.chdir(tmp_path)
    gt = physical_ramp_vertices(SIZE, AX, AY, 1.0)
    from hybrid_stereo_method.infrastructure.io.image_io import convert_image_array_to_fni

    (tmp_path / "rel").mkdir()
    convert_image_array_to_fni(np.stack([gt, np.ones_like(gt)], -1), "rel/hints.fni")
    cfg = build_integration_config({**NUMPY_FRAME, "use_hints": True}, debug=False)
    z = integrate_normals_to_height(
        constant_normals(SIZE, AX, AY), "rel/integration", "h", config=cfg,
        hints_fni_path="rel/hints.fni", hints_weight=0.1,
    )
    assert z.shape == (SIZE + 1, SIZE + 1)
