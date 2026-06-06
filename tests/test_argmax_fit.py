"""Testes das correções do ajuste parabólico do argmax difuso (MF-05/MF-06/MF-11)."""

import numpy as np


def test_peak_index_is_true_argmax():
    """MF-05: a janela soma-de-3 podia escolher índice fora do pico verdadeiro
    (fv=[0,0,0,9,0,5,6,5,0,0] -> max-sum dá 6; argmax verdadeiro é 3)."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import find_peak_index

    assert find_peak_index(np.array([0, 0, 0, 9, 0, 5, 6, 5, 0, 0])) == 3
    # empate exato: desempata pela vizinhança com mais suporte
    assert find_peak_index(np.array([0, 5, 0, 0, 4, 5, 4, 0])) == 5
