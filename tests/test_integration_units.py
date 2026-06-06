"""INT-04 / CONV-4: hints em unidades físicas de `z_foc` vs alturas em unidades de pixel.

Os slopes que o C integra são adimensionais (dZdX = -nx/nz, rise físico por run
físico) e são somados sobre células de 1 px, então `Z` sai em "altura-por-pixel".
Os hints (`zMos_with_confidence.fni`) estão em unidades físicas de `z_foc`. O
parâmetro `hybrid.integration.pixel_size` (tamanho lateral de 1 pixel nas mesmas
unidades de `z_foc`) reconcilia as escalas: `slopes_scale = (pixel_size,
pixel_size)` faz o C multiplicar os slopes pelo passo físico do pixel, e `Z` sai
em unidades físicas de `z_foc` — comensurável com os hints, que entram sem escala.

Estrutura:
- testes do helper `build_integration_config` (sem binário C);
- testes físicos com o binário (rampa em unidades físicas, hints em z_foc).
"""

import logging

import numpy as np
import pytest
from synthetic_utils import affine_fit_rmse

from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    integrate_normals_to_height,
)
from hybrid_stereo_method.hybrid.main import build_integration_config

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

# Rampa física: slopes adimensionais (dz físico / dx físico) e pixel de tamanho
# físico PIX (mesmas unidades de z_foc). Altura física no vértice (x, y) em px:
# Z_phys = PIX * (AX*x + AY*y), pois cada passo de pixel avança PIX em X físico.
SIZE, AX, AY, PIX = 32, 0.05, 0.02, 2.5


def constant_normals(size: int, ax: float, ay: float) -> np.ndarray:
    """Normais unitárias constantes do plano físico z = ax*X + ay*Y."""
    n = np.array([-ax, -ay, 1.0]) / np.sqrt(1.0 + ax**2 + ay**2)
    return np.broadcast_to(n, (size, size, 3)).copy()


def physical_ramp_vertices(size: int, ax: float, ay: float, pixel_size: float) -> np.ndarray:
    """Altura física na grade de vértices (size+1, size+1)."""
    y, x = np.mgrid[0 : size + 1, 0 : size + 1].astype(np.float64)
    return pixel_size * (ax * x + ay * y)


# ---------------------------------------------------------------------------
# build_integration_config (sem binário)
# ---------------------------------------------------------------------------


def test_config_derives_slopes_scale_from_pixel_size():
    config = build_integration_config({"pixel_size": 2.5}, debug=False)
    assert config.slopes_scale == (2.5, 2.5)


def test_config_defaults_to_unit_scale_without_pixel_size():
    config = build_integration_config({}, debug=False)
    assert config.slopes_scale == (1.0, 1.0)


def test_config_threads_solver_params():
    params = {
        "pixel_size": 2.0,
        "initial_method": "zero",
        "initial_noise": 0.1,
        "max_level": 5,
        "max_iter": 123,
        "conv_tol": 1e-3,
    }
    config = build_integration_config(params, debug=True)
    assert config.initial_method == "zero"
    assert config.initial_noise == 0.1
    assert config.max_level == 5
    assert config.max_iter == 123
    assert config.conv_tol == 1e-3
    assert config.verbose is True


def test_warns_when_hints_used_without_pixel_size(caplog):
    with caplog.at_level(logging.WARNING):
        build_integration_config({"use_hints": True}, debug=False)
    assert any("pixel_size" in m and "INT-04" in m for m in caplog.messages), (
        f"warning de escala incomensurável ausente: {caplog.messages}"
    )


def test_no_warning_when_hints_used_with_pixel_size(caplog):
    with caplog.at_level(logging.WARNING):
        build_integration_config({"use_hints": True, "pixel_size": 2.5}, debug=False)
    assert not any("pixel_size" in m for m in caplog.messages)


def test_no_warning_without_hints(caplog):
    with caplog.at_level(logging.WARNING):
        build_integration_config({}, debug=False)
    assert not any("pixel_size" in m for m in caplog.messages)


def test_initial_method_defaults_to_zero_and_hints_requires_use_hints():
    """INT-01: default era 'hints' mesmo sem -hints, e o binário aborta
    (demand H!=NULL). Default correto: 'zero'; 'hints' exige use_hints."""
    from hybrid_stereo_method.hybrid.main import build_integration_config

    cfg = build_integration_config({}, debug=False)
    assert cfg.initial_method == "zero"

    import pytest

    with pytest.raises(ValueError, match="use_hints"):
        build_integration_config({"initial_method": "hints", "use_hints": False}, debug=False)

    cfg = build_integration_config(
        {"initial_method": "hints", "use_hints": True, "pixel_size": 1.0}, debug=False
    )
    assert cfg.initial_method == "hints"


# ---------------------------------------------------------------------------
# Comportamento físico via binário C
# ---------------------------------------------------------------------------


@needs_binary
@pytest.mark.slow
def test_pixel_size_yields_heights_in_physical_units(tmp_path):
    """Com slopes_scale = (PIX, PIX), a altura integrada sai em unidades
    físicas de z_foc: fit afim contra a rampa física dá a ≈ 1."""
    normals = constant_normals(SIZE, AX, AY)
    gt_phys = physical_ramp_vertices(SIZE, AX, AY, PIX)

    config = build_integration_config({"pixel_size": PIX, "initial_method": "zero"}, debug=False)
    z = integrate_normals_to_height(normals, tmp_path, "phys", config=config)

    rmse, (a, b) = affine_fit_rmse(z, gt_phys)
    print(f"\npixel_size={PIX}: a={a:.4f}, b={b:.4f}, rmse={rmse:.4f}, std(gt)={gt_phys.std():.4f}")
    assert rmse < 0.05 * gt_phys.std(), f"forma não recuperada: rmse={rmse:.4f}"
    assert abs(a - 1.0) < 0.02, f"altura não está em unidades físicas: a={a:.4f} (esperado ~1)"


@needs_binary
@pytest.mark.slow
def test_default_unit_scale_biases_heights_by_pixel_size(tmp_path):
    """Evidência do defeito INT-04/CONV-4: sem pixel_size, Z fica em unidades
    de pixel e o fit contra a rampa física dá a ≈ PIX (viés de escala)."""
    normals = constant_normals(SIZE, AX, AY)
    gt_phys = physical_ramp_vertices(SIZE, AX, AY, PIX)

    config = build_integration_config({"initial_method": "zero"}, debug=False)  # default (1, 1)
    z = integrate_normals_to_height(normals, tmp_path, "pix", config=config)

    rmse, (a, b) = affine_fit_rmse(z, gt_phys)
    print(f"\ndefault scale: a={a:.4f} (PIX={PIX}), b={b:.4f}, rmse={rmse:.4f}")
    assert rmse < 0.05 * gt_phys.std()
    assert abs(a - PIX) < 0.02 * PIX, f"esperado viés a ≈ PIX={PIX} com escala default: a={a:.4f}"


@needs_binary
@pytest.mark.slow
def test_hints_in_zfoc_units_commensurable_with_scaled_slopes(tmp_path):
    """Hints em z_foc + slopes escalados por pixel_size descrevem a MESMA
    superfície: com peso de hints dominante a solução converge para a rampa
    física sem tensão entre os dois termos (a menos da constante global —
    o solver C força soma zero: ``szero = TRUE`` hard-coded em
    pst_integrate_iterative.c:75, então hints NÃO ancoram o nível absoluto)."""
    normals = constant_normals(SIZE, AX, AY)
    gt_phys = physical_ramp_vertices(SIZE, AX, AY, PIX)

    hints = np.zeros((SIZE + 1, SIZE + 1, 2))
    hints[..., 0] = gt_phys  # alturas em unidades físicas de z_foc
    hints[..., 1] = 1.0  # confiança máxima

    config = build_integration_config({"pixel_size": PIX, "use_hints": True}, debug=False)
    z = integrate_normals_to_height(
        normals, tmp_path, "hinted", config=config, hints_map=hints, hints_weight=1000.0
    )

    # o solver remove a média global (szero): compara formas demeaned
    assert abs(z.mean()) < 1e-3, f"esperado Z de média zero (szero no C): mean={z.mean():.4f}"
    err = (z - z.mean()) - (gt_phys - gt_phys.mean())
    rms = float(np.sqrt((err**2).mean()))
    print(f"\nhinted w=1000: demeaned rms err={rms:.4f}, std(gt)={gt_phys.std():.4f}")
    assert rms < 0.02 * gt_phys.std(), (
        f"hints e slopes escalados deveriam ser consistentes: rms={rms:.4f}"
    )


@needs_binary
@pytest.mark.slow
def test_hints_at_default_weight_do_not_distort_commensurable_solution(tmp_path):
    """Com unidades comensuráveis, o peso de hints moderado (ordem do default
    0.1 do config) não muda a escala da solução: a permanece ~1."""
    normals = constant_normals(SIZE, AX, AY)
    gt_phys = physical_ramp_vertices(SIZE, AX, AY, PIX)

    hints = np.zeros((SIZE + 1, SIZE + 1, 2))
    hints[..., 0] = gt_phys
    hints[..., 1] = 1.0

    config = build_integration_config({"pixel_size": PIX, "use_hints": True}, debug=False)
    z = integrate_normals_to_height(
        normals, tmp_path, "hinted_w05", config=config, hints_map=hints, hints_weight=0.5
    )

    rmse, (a, b) = affine_fit_rmse(z, gt_phys)
    print(f"\nhinted w=0.5: a={a:.4f}, rmse={rmse:.4f}, std(gt)={gt_phys.std():.4f}")
    # tensão numérica do solver multigrid permite ~7% de desvio; o que importa
    # é a escala não ir para PIX (incomensurável puro daria a ≈ 2.5)
    assert 0.9 < a < 1.1, f"escala distorcida com hints comensuráveis: a={a:.4f}"
    assert rmse < 0.2 * gt_phys.std()


@needs_binary
@pytest.mark.slow
def test_integrator_accepts_confidence_weight_channel(tmp_path):
    """PS-06/INT-03: normal map (H,W,4) com canal 3 = confiança; pixels com
    peso 0 (sombra) não devem contaminar nem crashar a integração."""
    from synthetic_utils import gaussian_bump, normals_from_height

    size = 32
    z = gaussian_bump(size, amplitude=4.0)
    n = normals_from_height(z)
    conf = np.ones((size, size), dtype=np.float64)
    n4 = np.concatenate([n, conf[..., None]], axis=-1)
    # zona "sombreada": NaN nas normais + confiança 0 (o que o wps produz)
    n4[10:14, 10:14, :3] = np.nan
    n4[10:14, 10:14, 3] = 0.0

    out = integrate_normals_to_height(n4, tmp_path, "w4")
    assert out.shape == (size + 1, size + 1)
    assert np.isfinite(out).all()


def test_missing_end_z_raises_instead_of_returning_initial_guess(tmp_path):
    """INT-02: se o solver 'sucede' sem escrever -00-end-Z.fni, devolver o chute
    inicial (-ini-Z.fni) silenciosamente mascara a falha — deve levantar."""
    from hybrid_stereo_method.hybrid.integrate import integrate_slopes_to_height

    fake = tmp_path / "fake_solver.sh"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    # escreve um -ini-Z.fni que o fallback antigo devolveria
    slopes = np.zeros((4, 4, 3))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x-ini-Z.fni").write_text(
        "begin float_image_t (format of 2006-03-25)\nNC = 1\nNX = 1\nNY = 1\n"
        "    0     0 +0.0000000e+00\n\nend float_image_t\n"
    )
    with pytest.raises(RuntimeError, match="end-Z"):
        integrate_slopes_to_height(slopes, out, "x", executable_path=fake)


# ---------------------------------------------------------------------------
# INT-05: cell-to-vertex grid conversion
# ---------------------------------------------------------------------------


def test_cell_to_vertex_grid_no_half_cell_shift():
    """INT-05: o C exigia hints (H+1,W+1) e expandia a grade de células com
    deslocamento de meia célula. A conversão correta média as até 4 células
    adjacentes: numa rampa linear o vértice v vale exatamente a média dos
    centros vizinhos (sem shift)."""
    from hybrid_stereo_method.hybrid.hints import cell_to_vertex_grid

    H = W = 6
    j = np.mgrid[0:H, 0:W][1].astype(np.float64)
    z = 2.0 * j  # rampa em x, valores nos CENTROS das células
    w = np.ones_like(z)
    out = cell_to_vertex_grid(z, w)
    assert out.shape == (H + 1, W + 1, 2)
    # vértices interiores: média das células j-1 e j -> 2*(j-0.5)
    np.testing.assert_allclose(out[1:-1, 3, 0], 2.0 * (3 - 0.5))
    # borda: só a célula disponível
    np.testing.assert_allclose(out[1:-1, 0, 0], 0.0)
    # pesos válidos em toda parte
    assert (out[..., 1] > 0).all()


def test_cell_to_vertex_grid_nan_cells_get_zero_weight():
    from hybrid_stereo_method.hybrid.hints import cell_to_vertex_grid

    z = np.ones((4, 4))
    z[1, 1] = np.nan
    w = np.ones((4, 4))
    out = cell_to_vertex_grid(z, w)
    assert np.isfinite(out[..., 0][out[..., 1] > 0]).all()


# ---------------------------------------------------------------------------
# INT-06: reference_scale converte uint8 reference para unidades de altura
# ---------------------------------------------------------------------------


def test_reference_scale_is_emitted_in_command(tmp_path, monkeypatch):
    """INT-06: hAvg.png é uint8 (0-255) e Z sai em unidades físicas; o C aceita
    '-reference R scale S' para tornar a comparação comensurável."""
    import subprocess

    from hybrid_stereo_method.hybrid.integrate import (
        IntegrateRecursiveConfig,
        integrate_slopes_to_height,
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise subprocess.CalledProcessError(1, cmd, stderr="stop-after-capture")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = np.zeros((5, 5))
    cfg = IntegrateRecursiveConfig(reference_scale=0.05)
    with pytest.raises(RuntimeError):
        integrate_slopes_to_height(
            np.zeros((4, 4, 3)), tmp_path, "r", config=cfg, reference_map=ref,
            executable_path="/bin/true",
        )
    cmd = captured["cmd"]
    i = cmd.index("-reference")
    assert cmd[i + 2 : i + 4] == ["scale", "0.05"], cmd


def test_reference_scale_not_emitted_when_default(tmp_path, monkeypatch):
    """INT-06: quando reference_scale == 1.0 (default), o argumento 'scale' NÃO
    é emitido (mantém compatibilidade retroativa)."""
    import subprocess

    from hybrid_stereo_method.hybrid.integrate import (
        IntegrateRecursiveConfig,
        integrate_slopes_to_height,
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise subprocess.CalledProcessError(1, cmd, stderr="stop-after-capture")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = np.zeros((5, 5))
    cfg = IntegrateRecursiveConfig()  # reference_scale=1.0 default
    with pytest.raises(RuntimeError):
        integrate_slopes_to_height(
            np.zeros((4, 4, 3)), tmp_path, "r2", config=cfg, reference_map=ref,
            executable_path="/bin/true",
        )
    cmd = captured["cmd"]
    i = cmd.index("-reference")
    # next token after path should NOT be "scale"
    assert cmd[i + 2] != "scale", f"scale should not be emitted for default 1.0: {cmd}"


def test_build_integration_config_threads_reference_scale():
    """INT-06: build_integration_config passa reference_scale ao config."""
    config = build_integration_config({"reference_scale": 0.5}, debug=False)
    assert config.reference_scale == 0.5


def test_build_integration_config_reference_scale_default():
    """INT-06: sem reference_scale no YAML, o default é 1.0."""
    config = build_integration_config({}, debug=False)
    assert config.reference_scale == 1.0


def test_warns_when_use_reference_without_reference_scale(caplog):
    """INT-06: use_reference=True sem reference_scale gera warning informativo."""
    with caplog.at_level(logging.WARNING):
        build_integration_config({"use_reference": True}, debug=False)
    assert any("reference_scale" in m and "INT-06" in m for m in caplog.messages), (
        f"warning INT-06 ausente: {caplog.messages}"
    )


def test_no_warning_when_use_reference_with_reference_scale(caplog):
    """INT-06: use_reference=True COM reference_scale não gera warning."""
    with caplog.at_level(logging.WARNING):
        build_integration_config({"use_reference": True, "reference_scale": 0.431}, debug=False)
    assert not any("INT-06" in m for m in caplog.messages)


def test_no_warning_without_use_reference(caplog):
    """INT-06: sem use_reference, nenhum warning de reference_scale."""
    with caplog.at_level(logging.WARNING):
        build_integration_config({}, debug=False)
    assert not any("reference_scale" in m for m in caplog.messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
