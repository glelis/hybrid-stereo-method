"""Layout dos artefatos da integração: tudo por nível/iteração em ``{prefix}-iters/``.

O C escreve os mapas por nível (``{prefix}-LL-beg/end-*``) soltos no diretório
de saída e as iterações em ``{prefix}-iters/it--LL-NNNNNNNNN-*``;
``organize_integration_outputs`` move tudo para ``{prefix}-iters/level-LL/`` e
exporta cada mapa Z em .npy (canal de altura intacto) e .png.
"""

import numpy as np
import pytest

from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from hybrid_stereo_method.hybrid.main import build_integration_config
from hybrid_stereo_method.infrastructure.io.image_io import read_fni_to_image_array

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)


def test_report_step_from_yaml():
    assert build_integration_config({"report_step": 5}, debug=False).report_step == 5
    assert build_integration_config({}, debug=False).report_step == 0


def test_rejects_negative_report_step():
    with pytest.raises(ValueError, match="report_step"):
        build_integration_config({"report_step": -1}, debug=False)


@needs_binary
def test_outputs_organized_under_iters_dir(tmp_path):
    y, x = np.mgrid[0:16, 0:16].astype(np.float64)
    ax, ay = 0.03 * np.cos(x / 3), 0.02 * np.sin(y / 4)
    normals = np.stack([-ax, -ay, np.ones_like(ax)], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    cfg = IntegrateRecursiveConfig(max_level=1, max_iter=50, report_step=3)

    z = integrate_normals_to_height(normals, tmp_path, "h", config=cfg)

    # Nada por nível fica solto no diretório de saída.
    loose = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert loose == ["h_normals.fni"]

    iters = tmp_path / "h-iters"
    assert {p.name for p in iters.iterdir() if p.is_dir()} == {"level-00", "level-01"}
    assert not [p for p in iters.iterdir() if p.name.startswith("it--")]
    for ext in ("fni", "npy", "png"):
        assert (iters / f"ini-Z.{ext}").exists()
        for level in ("level-00", "level-01"):
            assert (iters / level / f"beg-Z.{ext}").exists()
            assert (iters / level / f"end-Z.{ext}").exists()

    # Iterações intermediárias (report_step=3 → 1, 2, 3, 6, ...) em fni/npy/png.
    it_npy = sorted((iters / "level-00").glob("it-*-Z.npy"))
    assert it_npy, "nenhuma iteração intermediária exportada"
    for npy in it_npy:
        assert npy.with_suffix(".fni").exists() and npy.with_suffix(".png").exists()

    # O .npy guarda o canal de altura sem alteração; end do nível 0 = resultado.
    end = np.load(iters / "level-00" / "end-Z.npy")
    np.testing.assert_array_equal(end, z)
    np.testing.assert_array_equal(
        end, read_fni_to_image_array(iters / "level-00" / "end-Z.fni")[:, :, 0]
    )
