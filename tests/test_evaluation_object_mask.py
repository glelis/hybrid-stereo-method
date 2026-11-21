"""Métricas no objeto: altura avaliada só na máscara de frente do GT (sNrm)."""
import cv2
import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height

from hybrid_stereo_method.evaluation.main import run_evaluation
from hybrid_stereo_method.infrastructure.io.image_io import convert_image_array_to_fni

SIZE = 32
OBJECT_KEYS = ("multifocus_depth_object", "integration_height_object", "hybrid_gain_object")


def _make_run(root, write_snrm=True):
    """Objeto = disco central com altura gaussiana; fundo = lixo aleatório.

    O estimado (zMos e height_map) é igual ao GT no objeto e ruído
    independente no fundo, então só a métrica no objeto deve ser ~1.
    """
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    obj = (yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2 < (SIZE * 0.4) ** 2
    z = gaussian_bump(SIZE, amplitude=6.0)
    gt = np.where(obj, z, rng.uniform(z.min(), z.max(), z.shape))
    est = np.where(obj, z, rng.uniform(z.min(), z.max(), z.shape))

    results = root / "results"
    sharp = results / "sharp"
    sharp.mkdir(parents=True)
    h16 = ((gt - gt.min()) / (gt.max() - gt.min()) * 65535.0).round().astype(np.uint16)
    cv2.imwrite(str(sharp / "hAvg.png"), h16)
    if write_snrm:
        normals = normals_from_height(z)
        rgb16 = np.clip((normals + 1.0) / 2.0 * 65535.0, 0, 65535).round().astype(np.uint16)
        rgb16[~obj] = 0  # decodifica para (-1,-1,-1): |n| != 1 -> fundo
        cv2.imwrite(str(sharp / "sNrm.png"), rgb16[..., ::-1])

    mf = results / "multifocus_stereo" / "average"
    mf.mkdir(parents=True)
    convert_image_array_to_fni(est, mf / "zMos.fni")
    integ = results / "integration"
    integ.mkdir()
    np.save(integ / "height_map.npy", np.pad(est, ((0, 1), (0, 1)), mode="edge").astype(np.float32))
    return results, obj


@pytest.fixture
def run_dir(tmp_path):
    return _make_run(tmp_path)


def test_object_metrics_present(run_dir):
    results, _ = run_dir
    metrics = run_evaluation(results, data_dir=None)
    for key in OBJECT_KEYS:
        assert metrics[key]["status"] == "ok", (key, metrics[key])


def test_object_pearson_ignores_background(run_dir):
    results, _ = run_dir
    metrics = run_evaluation(results, data_dir=None)
    # zMos é cell-grid: no objeto o estimado é exatamente o GT
    assert metrics["multifocus_depth_object"]["pearson_r"] > 0.999
    assert metrics["multifocus_depth"]["pearson_r"] < 0.9
    # height_map é vertex-grid: a média dos 4 vértices mistura o ruído do
    # fundo nas células da borda do objeto, então o teto fica abaixo de 1
    assert metrics["integration_height_object"]["pearson_r"] > 0.95
    assert metrics["integration_height"]["pearson_r"] < 0.9


def test_object_mask_excludes_scale_band(tmp_path):
    # pasta "0032x0026" => nominal H=26 numa imagem de 32 linhas: faixa de 6
    results, obj = _make_run(tmp_path / "0032x0026-hs01")
    metrics = run_evaluation(results, data_dir=None)
    assert metrics["meta"]["scale_band_rows"] == 6
    expected = obj[6:].sum() / obj[6:].size
    assert metrics["meta"]["object_fraction"] == pytest.approx(expected)


def test_object_mask_disabled(run_dir):
    results, _ = run_dir
    metrics = run_evaluation(results, data_dir=None, config={"object_mask": False})
    assert not any(key in metrics for key in OBJECT_KEYS)


def test_object_mask_without_snrm(tmp_path):
    results, _ = _make_run(tmp_path, write_snrm=False)
    metrics = run_evaluation(results, data_dir=None)
    for key in OBJECT_KEYS:
        assert metrics[key]["status"].startswith("skipped"), (key, metrics[key])
    assert metrics["integration_height"]["status"] == "ok"


def test_report_has_object_table(run_dir):
    results, _ = run_dir
    run_evaluation(results, data_dir=None)
    report = (results / "evaluation" / "report.md").read_text()
    assert "## Objeto vs imagem toda" in report
    assert "| integração (altura final) |" in report
