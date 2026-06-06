# tests/test_multifocus_synthetic.py
"""Fase 3.1 — multifocus com pilha de foco sintética.

depth está em UNIDADES DE ÍNDICE de frame (z_foc = 0..n-1), então iSel é
diretamente comparável ao ground truth. Falha = achado MF-xx.
"""

import numpy as np
import pytest
from synthetic_utils import defocus_stack, gaussian_bump, texture

from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy
from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator
from hybrid_stereo_method.multifocus.mosaic import mosaic


def _run_multifocus(depth, size, n_frames, seed=3):
    z_foc = list(range(n_frames))
    sharp = 255.0 * texture(size, seed=seed)
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)
    fi = focus_indicator(
        stack,
        "laplacian",
        laplacian_kernel_size=5,
        radius=None,
        square=True,
        smooth=True,
        spatial_median_filter=False,
        zero_border=False,
    )
    iSel, wSel = compute_argmax_fuzzy(fi, False, "", {"r_max": 2})
    return iSel, wSel


def test_recovers_tilted_plane_depth():
    size, n_frames = 64, 9
    x = np.mgrid[0:size, 0:size][1].astype(np.float64)
    depth = 1.5 + 5.0 * x / (size - 1)  # rampa em x: 1.5 .. 6.5 (índices de frame)
    iSel, _ = _run_multifocus(depth, size, n_frames)

    interior = (slice(8, -8), slice(8, -8))  # evita efeitos de borda do filtro
    err = np.abs(iSel[interior] - depth[interior])
    print(
        f"\nplano: mediana|iSel-z| = {np.median(err):.3f} frames, p90 = {np.percentile(err, 90):.3f}"
    )
    assert np.median(err) < 0.5, f"erro sub-frame não atingido: mediana {np.median(err):.2f}"


def test_recovers_bump_depth():
    size, n_frames = 64, 9
    depth = 2.0 + gaussian_bump(size, amplitude=4.0)  # 2 .. 6
    iSel, _ = _run_multifocus(depth, size, n_frames)

    interior = (slice(8, -8), slice(8, -8))
    err = np.abs(iSel[interior] - depth[interior])
    print(f"\nbump: mediana|iSel-z| = {np.median(err):.3f} frames")
    assert np.median(err) < 0.5


def test_confidence_is_scale_invariant_goodness_of_fit():
    """MF-07 (correção 2026-06-05): a confiança agora é o R² (ponderado) do ajuste
    parabólico local — invariante à escala da curva de foco, em [0,1], com
    significado uniforme entre pixels/imagens.

    MUDANÇA DE SEMÂNTICA DELIBERADA: o teste anterior
    (``test_textureless_region_gets_zero_confidence``) afirmava que a região SEM
    textura recebia confiança distintamente MENOR que a texturizada. Isso valia
    para a métrica antiga ``|A|/fnoc`` (curvatura/amplitude), que media força do
    pico — separação medida 0.38 (sem=0.33, com=0.87). O R² é uma medida de
    QUALIDADE DE AJUSTE, não de força do pico: uma curva suave de vazamento de
    desfoque (região lisa) ajusta uma parábola TÃO bem quanto um pico real, então
    R² NÃO separa textura de ausência de textura (valores novos: sem=0.79,
    com=0.70). Isto está alinhado com a ressalva já registrada no audit-note
    MF-07: o canal de confiança não é (e não era) um detector validado de regiões
    sem textura. A detecção de "sem sinal" continua coberta pelo ramo
    ``focus_values[k_max] == 0`` -> conf 0 (ver test_confidence_r2.py).

    Este teste agora fixa a propriedade central da correção: invariância de escala
    e faixa [0,1] no caminho 2D real (com indicador de foco).
    """
    size, n_frames = 64, 9
    depth = np.full((size, size), 4.0)
    z_foc = list(range(n_frames))
    sharp = 255.0 * texture(size, seed=5)
    sharp[24:40, 24:40] = 128.0  # quadrado plano, sem textura
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)

    def _conf(stk):
        fi = focus_indicator(
            stk,
            "laplacian",
            laplacian_kernel_size=5,
            radius=None,
            square=True,
            smooth=True,
            spatial_median_filter=False,
            zero_border=False,
        )
        return compute_argmax_fuzzy(fi, False, "", {"r_max": 2})

    iSel, wSel = _conf(stack)
    # mesma cena com brilho global ×0.1: confiança (R²) deve ser idêntica
    _, wSel_scaled = _conf(stack * 0.1)

    flat = wSel[28:36, 28:36]
    textured = wSel[4:16, 4:16]
    print(
        f"\n[MF-07] conf média (R²): sem textura = {flat.mean():.4f}, com textura = {textured.mean():.4f}"
    )
    print("[MF-07] valores antigos (|A|/fnoc): sem=0.33, com=0.87 (separação 0.38)")
    print(f"iSel na região sem textura: mediana = {np.median(iSel[28:36, 28:36]):.2f} (gt = 4.0)")

    # faixa [0,1] sem re-stretch global
    assert wSel.min() >= 0.0 and wSel.max() <= 1.0
    # invariância de escala no caminho 2D (até ruído numérico do indicador de foco)
    assert np.allclose(wSel, wSel_scaled, atol=1e-6), (
        "confiança R² não é invariante à escala global"
    )


def test_zero_peak_returns_nan_not_middle_frame():
    """MF-03: pico de foco nulo é indecidível — deve virar NaN/conf 0, não n/2."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d

    k, conf = compute_argmax_fuzzy_1d(np.zeros(9), [0, 0], {"r_max": 2})
    assert conf == 0
    assert np.isnan(k), f"pico nulo devolveu k={k} em vez de NaN (MF-03)"


def test_mosaic_masks_zero_confidence_pixels():
    """MF-04: pixels com confiança 0 não podem entrar no zMos como profundidade
    válida — viram NaN; o sMos usa o frame mais próximo (precisa de valor)."""
    n, h, w = 5, 4, 4
    stack = np.random.default_rng(0).uniform(0, 255, (n, h, w, 3))
    z_foc = [10.0, 20.0, 30.0, 40.0, 50.0]
    iSel = np.full((h, w), 2.0)
    iSel[0, 0] = np.nan  # MF-03: pixel indecidível
    wSel = np.ones((h, w))
    wSel[1, 1] = 0.0  # confiança zero

    sMos, zMos = mosaic(iSel, stack, z_foc, "linear_interpolation", wSel=wSel)

    assert np.isnan(zMos[0, 0]) and np.isnan(zMos[1, 1])
    assert np.isfinite(sMos).all(), "sMos deve sempre ter valor (consumido pelo PS)"
    assert zMos[2, 2] == 30.0
