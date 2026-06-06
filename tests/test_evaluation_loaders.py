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


# --- artefatos do pipeline -------------------------------------------------
from hybrid_stereo_method.evaluation.loaders import (  # noqa: E402
    find_smos_pairs,
    load_height_map,
    load_normal_map,
    load_shrp_z_gt,
    load_zmos,
    resolve_data_dir,
    vertex_to_cell,
)
from hybrid_stereo_method.infrastructure.io.image_io import (  # noqa: E402
    convert_image_array_to_fni,
)


def test_load_zmos_preserves_nan(tmp_path):
    mf = tmp_path / "multifocus_stereo" / "average"
    mf.mkdir(parents=True)
    z = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    z[0, 0] = np.nan
    convert_image_array_to_fni(z, mf / "zMos.fni")
    out = load_zmos(tmp_path)
    assert out.shape == (3, 4)
    assert np.isnan(out[0, 0])
    assert out[2, 3] == pytest.approx(11.0)
    with pytest.raises(MissingArtifactError):
        load_zmos(tmp_path / "nada")


def test_load_normal_and_height_maps(tmp_path):
    (tmp_path / "photometric_stereo").mkdir()
    (tmp_path / "integration").mkdir()
    n = np.zeros((4, 4, 3), np.float32)
    np.save(tmp_path / "photometric_stereo" / "normal_map.npy", n)
    h = np.ones((5, 5), np.float32)
    np.save(tmp_path / "integration" / "height_map.npy", h)
    assert load_normal_map(tmp_path).shape == (4, 4, 3)
    assert load_height_map(tmp_path).shape == (5, 5)
    with pytest.raises(MissingArtifactError):
        load_normal_map(tmp_path / "nada")


def test_vertex_to_cell_averages_four_corners():
    v = np.array([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
    c = vertex_to_cell(v)
    assert c.shape == (2, 2)
    assert c[0, 0] == pytest.approx((0 + 2 + 2 + 4) / 4.0)
    assert c[1, 1] == pytest.approx((4 + 6 + 6 + 8) / 4.0)


def test_load_shrp_z_gt_parses_zf_names_and_argmax(tmp_path):
    # nomes reais: zf015.0000-df020.0000; nitidez máxima no plano mais próximo
    data = tmp_path / "data"
    z_vals = [15.0, 25.0, 35.0]
    depth = np.full((4, 4), 25.0)
    depth[0, 0] = 15.0
    depth[3, 3] = 35.0
    for zv in z_vals:
        d = data / "L000" / f"zf{zv:08.4f}-df020.0000"
        d.mkdir(parents=True)
        shrp = (np.exp(-np.abs(depth - zv)) * 65535.0).astype(np.uint16)
        cv2.imwrite(str(d / "shrp.png"), shrp)
    z_gt, vals = load_shrp_z_gt(data)
    assert vals == z_vals
    assert z_gt[0, 0] == pytest.approx(15.0)
    assert z_gt[1, 1] == pytest.approx(25.0)
    assert z_gt[3, 3] == pytest.approx(35.0)
    with pytest.raises(MissingArtifactError):
        load_shrp_z_gt(tmp_path / "vazio")


def test_find_smos_pairs_matches_lights(tmp_path):
    results = tmp_path / "results"
    data = tmp_path / "data"
    img = np.zeros((4, 4, 3), np.uint8)
    for light in ("L000", "L001"):
        d = results / "multifocus_stereo" / light
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "sMos.png"), img)
        g = data / light / "sharp"
        g.mkdir(parents=True)
        cv2.imwrite(str(g / "sVal.png"), img)
    # luz sem GT correspondente: fica de fora dos pares
    extra = results / "multifocus_stereo" / "L002"
    extra.mkdir(parents=True)
    cv2.imwrite(str(extra / "sMos.png"), img)
    pairs = find_smos_pairs(results, data)
    assert [p[0] for p in pairs] == ["L000", "L001"]
    assert all(p[1].exists() and p[2].exists() for p in pairs)
    assert find_smos_pairs(results, None) == []


def test_resolve_data_dir_precedence(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    data = tmp_path / "input" / "folder"
    data.mkdir(parents=True)
    # 1) argumento explícito vence
    assert resolve_data_dir(results, data) == data
    # 2) parameters.yaml salvo pelo pipeline
    (results / "parameters.yaml").write_text(
        "experiment:\n  paths:\n"
        f"    input: '{tmp_path / 'input'}'\n"
        "    data_folder: 'folder'\n"
    )
    assert resolve_data_dir(results, None) == data
    # 3) sem nada: None
    assert resolve_data_dir(tmp_path, None) is None
