# tests/test_confidence_r2.py
"""MF-07 regression tests — confiança = R² do ajuste local (escala-invariante, [0,1]).

A correção MF-07 substitui ``conf = |A|/fnoc`` (dependente da escala global de
normalização do stack) pelo R² do ajuste parabólico local, que é invariante a
ganho e offset da curva de foco e tem significado uniforme em [0,1]. Também
remove o ``normalize(wSel)`` global do wrapper 2D, que reesticava uma escala já
absoluta.
"""

import numpy as np

from hybrid_stereo_method.multifocus.argmax_fuzzy import (
    compute_argmax_fuzzy,
    compute_argmax_fuzzy_1d,
)


def _clean_peak(n=9, peak=4, amp=10.0, offset=0.0):
    x = np.arange(n, dtype=np.float64)
    # parábola côncava com vértice em `peak`
    y = amp - 0.5 * (x - peak) ** 2
    y = np.clip(y, 0.0, None)
    return y + offset


# --------------------------------------------------------------------------- #
# Scale invariance — the MF-07 regression test
# --------------------------------------------------------------------------- #
def test_confidence_scale_invariant_gain():
    """Mesma curva de foco ×1.0 e ×0.01 -> mesma confiança e mesmo k_fuzzy.

    Pré-correção isto FALHA: conf = |A|/fnoc escala com o ganho da curva.
    """
    base = _clean_peak()
    k1, c1 = compute_argmax_fuzzy_1d(base, [0, 0], {"r_max": 2})
    k2, c2 = compute_argmax_fuzzy_1d(base * 0.01, [0, 0], {"r_max": 2})

    assert np.isclose(c1, c2, atol=1e-9), f"conf não invariante a ganho: {c1} vs {c2}"
    assert np.isclose(k1, k2, atol=1e-9), f"k_fuzzy não invariante a ganho: {k1} vs {k2}"


def test_confidence_scale_invariant_offset():
    """Offset aditivo na curva -> mesma confiança (R² é invariante a offset)."""
    base = _clean_peak()
    k1, c1 = compute_argmax_fuzzy_1d(base, [0, 0], {"r_max": 2})
    k2, c2 = compute_argmax_fuzzy_1d(base + 100.0, [0, 0], {"r_max": 2})

    assert np.isclose(c1, c2, atol=1e-9), f"conf não invariante a offset: {c1} vs {c2}"
    assert np.isclose(k1, k2, atol=1e-9), f"k_fuzzy não invariante a offset: {k1} vs {k2}"


def test_confidence_scale_invariant_gain_and_offset():
    base = _clean_peak()
    _, c1 = compute_argmax_fuzzy_1d(base, [0, 0], {"r_max": 2})
    _, c2 = compute_argmax_fuzzy_1d(base * 0.005 + 50.0, [0, 0], {"r_max": 2})
    assert np.isclose(c1, c2, atol=1e-9), f"conf não invariante a ganho+offset: {c1} vs {c2}"


# --------------------------------------------------------------------------- #
# Range [0, 1]
# --------------------------------------------------------------------------- #
def test_confidence_in_unit_range():
    curves = [
        _clean_peak(),
        _clean_peak(amp=3.0),
        _clean_peak() + np.random.RandomState(0).normal(0, 0.5, 9),
    ]
    for y in curves:
        _, c = compute_argmax_fuzzy_1d(y, [0, 0], {"r_max": 2})
        assert 0.0 <= c <= 1.0, f"conf fora de [0,1]: {c}"


def test_clean_peak_high_confidence():
    _, c = compute_argmax_fuzzy_1d(_clean_peak(), [0, 0], {"r_max": 2})
    assert c > 0.95, f"pico limpo deveria ter conf alta, obteve {c}"


def test_flat_window_zero_confidence():
    """Janela plana (sem pico decidível) -> conf 0 via ss_tot ~ 0 ou rejeição."""
    # valores não-nulos mas planos na janela ajustada
    y = np.full(9, 5.0)
    y[4] = 5.0  # totalmente plano
    _, c = compute_argmax_fuzzy_1d(y, [0, 0], {"r_max": 2})
    assert c == 0.0, f"janela plana deveria ter conf 0, obteve {c}"


# --------------------------------------------------------------------------- #
# Ordering — clean > noisy
# --------------------------------------------------------------------------- #
def test_clean_peak_beats_noisy_peak():
    rng = np.random.RandomState(42)
    clean = _clean_peak(amp=10.0)
    noisy = clean + rng.normal(0, 3.0, clean.shape)
    _, c_clean = compute_argmax_fuzzy_1d(clean, [0, 0], {"r_max": 2})
    _, c_noisy = compute_argmax_fuzzy_1d(noisy, [0, 0], {"r_max": 2})
    assert c_clean > c_noisy, f"limpo {c_clean} deveria > ruidoso {c_noisy}"


# --------------------------------------------------------------------------- #
# Rejection paths preserved
# --------------------------------------------------------------------------- #
def test_convex_curve_zero_confidence():
    """Curva convexa (A > 0) -> conf 0 (rejeição preservada)."""
    x = np.arange(9, dtype=np.float64)
    y = 0.5 * (x - 4) ** 2 + 1.0  # convexa
    _, c = compute_argmax_fuzzy_1d(y, [0, 0], {"r_max": 2})
    assert c == 0.0, f"curva convexa deveria ter conf 0, obteve {c}"


def test_all_zero_returns_zero_confidence():
    """focus_values[k_max] == 0 -> (NaN, 0) — MF-03: pico nulo é indecidível.

    Comportamento anterior (antes de MF-03): retornava (n/2, 0), propagando um
    índice plausível sem base observacional. Agora retorna NaN para que o mosaic
    possa mascarar o pixel via confiança 0 em vez de inventar n/2.
    """
    y = np.zeros(9, dtype=np.float64)
    k, c = compute_argmax_fuzzy_1d(y, [0, 0], {"r_max": 2})
    assert c == 0.0
    assert np.isnan(k), f"pico nulo devolveu k={k} em vez de NaN (MF-03)"


# --------------------------------------------------------------------------- #
# 2D wrapper — no global re-stretch
# --------------------------------------------------------------------------- #
def test_wrapper_no_global_restretch():
    """wSel em [0,1] SEM reesticar globalmente: se a melhor confiança real é
    ~0.8, o máximo de wSel NÃO deve ser artificialmente 1.0.

    Pin do remoção do normalize() global no wrapper 2D.
    """
    n, h, w = 9, 4, 4
    rng = np.random.RandomState(7)
    stack = np.zeros((n, h, w), dtype=np.float64)
    # mesma curva ruidosa (R² < 1) em todos os pixels -> melhor conf < 1
    for i in range(h):
        for j in range(w):
            noisy = _clean_peak(amp=10.0) + rng.normal(0, 1.0, n)
            stack[:, i, j] = np.clip(noisy, 0.0, None)

    _, wSel = compute_argmax_fuzzy(stack, False, "", {"r_max": 2})

    assert wSel.min() >= 0.0 and wSel.max() <= 1.0, "wSel fora de [0,1]"
    # nenhum R² atinge 1.0 com este ruído -> sem re-stretch, max < ~0.999
    assert wSel.max() < 0.999, (
        f"wSel.max() == {wSel.max()} sugere re-stretch global (normalize não removido)"
    )
