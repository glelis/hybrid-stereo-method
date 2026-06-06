"""Testes das correções do ajuste parabólico do argmax difuso (MF-05/MF-06/MF-11)."""

import numpy as np


def test_peak_index_is_true_argmax():
    """MF-05: a janela soma-de-3 podia escolher índice fora do pico verdadeiro
    (fv=[0,0,0,9,0,5,6,5,0,0] -> max-sum dá 6; argmax verdadeiro é 3)."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import find_peak_index

    assert find_peak_index(np.array([0, 0, 0, 9, 0, 5, 6, 5, 0, 0])) == 3
    # empate exato: desempata pela vizinhança com mais suporte
    assert find_peak_index(np.array([0, 5, 0, 0, 4, 5, 4, 0])) == 5


def test_parabola_vertex_unbiased_by_value_weights():
    """MF-06: ponderar a regressão pelos próprios valores de foco enviesa o
    vértice para o frame de maior valor. Curva gaussiana com pico em 4.3:
    o ajuste deve achar o vértice perto de 4.3 (não puxado para 4)."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d

    x = np.arange(9, dtype=float)
    fv = np.exp(-((x - 4.3) ** 2) / (2 * 1.2**2))
    k, conf = compute_argmax_fuzzy_1d(fv, [0, 0], {"r_max": 2})
    assert conf > 0
    assert abs(k - 4.3) < 0.1, f"vértice enviesado: {k:.3f} (gt 4.3) — MF-06"


def test_vertex_outside_stack_clamps_to_n_minus_1_with_zero_conf():
    """MF-11: clamp era min(n, k) — índice n não existe; e vértice extrapolado
    (fora de [0, n-1]) significa pico não-bracketado -> conf 0."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d

    fv = np.array([0.0, 0.05, 0.1, 0.3, 0.7, 1.0])  # acelerada: k_raw=7.5 > n-1=5
    k, conf = compute_argmax_fuzzy_1d(fv, [0, 0], {"r_max": 2})
    assert k <= len(fv) - 1, f"k={k} excede o índice máximo válido {len(fv)-1}"
    assert conf == 0, "vértice extrapolado deve ter confiança 0"
