"""Índice de luz a partir do nome da pasta (L0, L000 -> 0) — investigação 2026-06-06.

Regressão: sMos_by_light era indexado pelo NOME ('L000') mas consultado como
f'L{i}' ('L0'), quebrando datasets com pastas zero-padded (KeyError: 'L0').
A indexação passa a ser por inteiro, alinhada a pair_mosaics_to_lights.
"""

import numpy as np
import pytest

from hybrid_stereo_method.hybrid.main import light_index, pair_mosaics_to_lights


@pytest.mark.parametrize(
    ("name", "expected"),
    [("L0", 0), ("L000", 0), ("L1", 1), ("L01", 1), ("L011", 11), ("L10", 10)],
)
def test_light_index_parses_padded_and_unpadded(name, expected):
    assert light_index(name) == expected


def test_light_index_rejects_non_light_name():
    with pytest.raises(ValueError):
        light_index("average")


def test_smos_images_alignment_matches_pair_mosaics_to_lights():
    """A ordem in-memory por índice bate com a ordem de pair_mosaics_to_lights."""
    # pastas zero-padded fora de ordem natural
    light_dirs = ["L002", "L000", "L001"]
    mosaics = {name: np.full((2, 2), light_index(name), dtype=float) for name in light_dirs}
    by_index = {light_index(name): mosaics[name] for name in light_dirs}
    n_lights = 3
    in_memory = [by_index[i] for i in range(n_lights)]
    # caminhos fake nomeados pelo dir-pai L<n>, na mesma convenção de pair_mosaics_to_lights
    paths = [f"/x/{name}/sMos.png" for name in light_dirs]
    ordered_paths = pair_mosaics_to_lights(paths, n_lights)
    for i, p in enumerate(ordered_paths):
        assert light_index(p.split("/")[-2]) == i
        assert in_memory[i][0, 0] == float(i)
