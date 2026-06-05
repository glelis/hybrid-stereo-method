# tests/test_multifocus_synthetic.py
"""Fase 3.1 — multifocus com pilha de foco sintética.

depth está em UNIDADES DE ÍNDICE de frame (z_foc = 0..n-1), então iSel é
diretamente comparável ao ground truth. Falha = achado MF-xx.
"""
import numpy as np

from synthetic_utils import defocus_stack, gaussian_bump, texture

from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy
from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator


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
    print(f"\nplano: mediana|iSel-z| = {np.median(err):.3f} frames, p90 = {np.percentile(err, 90):.3f}")
    assert np.median(err) < 0.5, f"erro sub-frame não atingido: mediana {np.median(err):.2f}"


def test_recovers_bump_depth():
    size, n_frames = 64, 9
    depth = 2.0 + gaussian_bump(size, amplitude=4.0)  # 2 .. 6
    iSel, _ = _run_multifocus(depth, size, n_frames)

    interior = (slice(8, -8), slice(8, -8))
    err = np.abs(iSel[interior] - depth[interior])
    print(f"\nbump: mediana|iSel-z| = {np.median(err):.3f} frames")
    assert np.median(err) < 0.5


def test_textureless_region_gets_zero_confidence():
    """Região central SEM textura: a profundidade lá é indecidível; o método deve
    sinalizar confiança ~0 (e o que iSel devolve lá é documentado — lead MF do
    'return n/2')."""
    size, n_frames = 64, 9
    depth = np.full((size, size), 4.0)
    z_foc = list(range(n_frames))
    sharp = 255.0 * texture(size, seed=5)
    sharp[24:40, 24:40] = 128.0  # quadrado plano, sem textura
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)
    fi = focus_indicator(
        stack, "laplacian", laplacian_kernel_size=5, radius=None,
        square=True, smooth=True, spatial_median_filter=False, zero_border=False,
    )
    iSel, wSel = compute_argmax_fuzzy(fi, False, "", {"r_max": 2})

    flat = wSel[28:36, 28:36]       # miolo da região sem textura
    textured = wSel[4:16, 4:16]
    print(f"\nconf média: sem textura = {flat.mean():.4f}, com textura = {textured.mean():.4f}")
    print(f"iSel na região sem textura: mediana = {np.median(iSel[28:36, 28:36]):.2f} (gt = 4.0)")
    assert flat.mean() < 0.5 * textured.mean(), (
        "confiança em região sem textura não é distintamente menor"
    )
