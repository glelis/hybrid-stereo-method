"""Testes dos loaders de artefatos e ground truth."""
import cv2
import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height

from hybrid_stereo_method.evaluation.loaders import (
    MissingArtifactError,
    find_sharp_dir,
    load_hdev_mask,
    load_height_gt,
    load_normals_gt,
)


@pytest.fixture
def sharp_dir(tmp_path):
    """Pasta sharp/ sintética: hAvg uint16, sNrm RGB codificado, hDev uint16."""
    z = gaussian_bump(16, amplitude=6.0)
    normals = normals_from_height(z)
    d = tmp_path / "sharp"
    d.mkdir()
    h16 = ((z - z.min()) / (z.max() - z.min()) * 65535.0).round().astype(np.uint16)
    cv2.imwrite(str(d / "hAvg.png"), h16)
    rgb = np.clip((normals + 1.0) / 2.0 * 255.0, 0, 255).round().astype(np.uint8)
    rgb[0, 0] = 0  # pixel de fundo: preto decodifica para (-1,-1,-1), norma 1.73
    cv2.imwrite(str(d / "sNrm.png"), rgb[..., ::-1])  # cv2 grava BGR
    dev = np.zeros((16, 16), np.uint16)
    dev[0, :] = 65535  # primeira linha: GT incerto
    cv2.imwrite(str(d / "hDev.png"), dev)
    return d, z, normals


def test_find_sharp_dir_prefers_results_then_data(tmp_path):
    results = tmp_path / "results"
    data = tmp_path / "data"
    (data / "sharp").mkdir(parents=True)
    results.mkdir()
    assert find_sharp_dir(results, data) == data / "sharp"
    (results / "sharp").mkdir()
    assert find_sharp_dir(results, data) == results / "sharp"
    with pytest.raises(MissingArtifactError):
        find_sharp_dir(tmp_path / "nada", None)


def test_load_height_gt_reads_uint16_as_float(sharp_dir):
    d, z, _ = sharp_dir
    gt = load_height_gt(d)
    assert gt.shape == (16, 16)
    assert gt.dtype == np.float64
    assert gt.max() == pytest.approx(65535.0)
    # forma preservada: correlação alta com o z original
    assert np.corrcoef(gt.ravel(), z.ravel())[0, 1] > 0.999


def test_load_normals_gt_decodes_and_masks_background(sharp_dir):
    d, _, normals = sharp_dir
    n, fg = load_normals_gt(d)
    assert n.shape == (16, 16, 3)
    assert not fg[0, 0]  # pixel preto (fundo) excluído
    assert fg[8, 8]
    # decodificação fiel a ~1/255 de quantização
    err = np.linalg.norm(n[fg] - normals[fg], axis=-1)
    assert err.max() < 0.02
    # normais retornadas são unitárias nos pixels de frente
    assert np.allclose(np.linalg.norm(n[fg], axis=-1), 1.0, atol=1e-9)


def test_load_hdev_mask_thresholds_normalized_range(sharp_dir):
    d, _, _ = sharp_dir
    mask = load_hdev_mask(d, threshold=0.1)
    assert mask is not None
    assert not mask[0, :].any()  # linha com hDev no máximo do range: excluída
    assert mask[1:, :].all()


def test_load_hdev_mask_returns_none_when_absent(tmp_path):
    d = tmp_path / "sharp"
    d.mkdir()
    assert load_hdev_mask(d, threshold=0.1) is None
