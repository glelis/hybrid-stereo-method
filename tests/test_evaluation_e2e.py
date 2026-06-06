"""Integração: pasta de resultados sintética completa → run_evaluation → saídas."""
import json

import cv2
import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height, texture

from hybrid_stereo_method.evaluation.main import run_evaluation
from hybrid_stereo_method.infrastructure.io.image_io import convert_image_array_to_fni

SIZE = 32


@pytest.fixture
def synthetic_run(tmp_path):
    """Resultados + dataset sintéticos coerentes (mesma superfície gaussiana).

    GT: sharp/ com hAvg uint16 e sNrm codificado; dataset com L*/sharp/sVal.png
    e pilha zf*/shrp.png. Artefatos: zMos = z (com 1 NaN), normal_map = normais
    exatas, height_map = z em vertex-grid (pad de borda).
    """
    z = gaussian_bump(SIZE, amplitude=6.0)
    normals = normals_from_height(z)
    results = tmp_path / "results"
    data = tmp_path / "data"

    sharp = results / "sharp"
    sharp.mkdir(parents=True)
    h16 = ((z - z.min()) / (z.max() - z.min()) * 65535.0).round().astype(np.uint16)
    cv2.imwrite(str(sharp / "hAvg.png"), h16)
    rgb = np.clip((normals + 1.0) / 2.0 * 255.0, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(sharp / "sNrm.png"), rgb[..., ::-1])  # BGR

    mf = results / "multifocus_stereo" / "average"
    mf.mkdir(parents=True)
    zmos = z.copy()
    zmos[0, 0] = np.nan
    convert_image_array_to_fni(zmos, mf / "zMos.fni")

    tex = (texture(SIZE) * 255.0).astype(np.uint8)
    tex3 = cv2.merge([tex, tex, tex])
    for li in range(2):
        light = f"L{li:03d}"
        d = results / "multifocus_stereo" / light
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "sMos.png"), tex3)
        g = data / light / "sharp"
        g.mkdir(parents=True)
        cv2.imwrite(str(g / "sVal.png"), tex3)  # idêntico → SSIM 1

    z_vals = [1.0, 3.0, 5.0, 7.0]
    for zv in z_vals:
        d = data / "L000" / f"zf{zv:08.4f}-df020.0000"
        d.mkdir(parents=True)
        shrp = (np.exp(-np.abs(z - zv)) * 65535.0).astype(np.uint16)
        cv2.imwrite(str(d / "shrp.png"), shrp)

    ps = results / "photometric_stereo"
    ps.mkdir()
    np.save(ps / "normal_map.npy", normals.astype(np.float32))
    np.save(ps / "confidence.npy", np.ones((SIZE, SIZE), np.float32))

    integ = results / "integration"
    integ.mkdir()
    z_vertex = np.pad(z, ((0, 1), (0, 1)), mode="edge")
    np.save(integ / "height_map.npy", z_vertex.astype(np.float32))

    return results, data, z


def test_run_evaluation_full(synthetic_run):
    results, data, z = synthetic_run
    metrics = run_evaluation(results, data_dir=data)

    out = results / "evaluation"
    assert (out / "metrics.json").exists()
    assert (out / "report.md").exists()
    loaded = json.loads((out / "metrics.json").read_text())

    for stage in (
        "multifocus_depth",
        "focus_selection",
        "mosaics",
        "photometric_normals",
        "integration_height",
        "hybrid_gain",
    ):
        assert loaded[stage]["status"] == "ok", f"{stage}: {loaded[stage]['status']}"

    # multifocus: zMos == z (a menos da escala uint16 do GT) → o resíduo do fit
    # afim é só o arredondamento da codificação (uniforme ±0.5 nível de cinza,
    # RMS ≈ 0.29) → RMSE < 1 nível de cinza
    assert loaded["multifocus_depth"]["rmse_affine"] < 1.0
    assert loaded["multifocus_depth"]["pearson_r"] > 0.999
    # seleção de foco: zmos == z → erro <= meio passo focal → 100% dentro de 1 frame
    assert loaded["focus_selection"]["within_1_frame_pct"] == pytest.approx(100.0)
    # mosaicos idênticos → SSIM 1, PSNR inf (serializado como string)
    assert loaded["mosaics"]["ssim_mean"] == pytest.approx(1.0)
    assert loaded["mosaics"]["psnr_mean"] == "inf"
    # normais exatas (a menos da quantização de 8 bits do sNrm) → erro < 1°
    assert loaded["photometric_normals"]["winning_orientation"] == "y_as_is"
    assert loaded["photometric_normals"]["y_as_is"]["mean_deg"] < 1.0
    # integração: o vertex→cell introduz meio pixel de shift + suavização
    # (o fit afim não absorve shift); num bump com |∇z| até ~0.5/px isso fica
    # bem abaixo de 25% do std do GT, mas não de 10% — threshold com folga
    ih = loaded["integration_height"]
    assert ih["rmse_affine"] < 0.25 * ih["gt_std"]
    assert ih["pearson_r"] > 0.99
    # mapas de erro salvos
    assert (out / "multifocus_depth_error.png").exists()
    assert (out / "multifocus_focus_selection_error.png").exists()
    assert (out / "photometric_angular_error.png").exists()
    assert (out / "integration_height_error.png").exists()
    # dict retornado é o mesmo conteúdo serializado
    assert metrics["multifocus_depth"]["status"] == "ok"


def test_run_evaluation_skips_missing_stage(synthetic_run, tmp_path):
    results, data, _ = synthetic_run
    import shutil

    shutil.rmtree(results / "photometric_stereo")
    metrics = run_evaluation(results, data_dir=data)
    assert metrics["photometric_normals"]["status"].startswith("skipped")
    assert metrics["multifocus_depth"]["status"] == "ok"  # demais etapas seguem


def test_run_evaluation_without_data_dir_skips_dataset_dependent(synthetic_run):
    results, _, _ = synthetic_run
    metrics = run_evaluation(results, data_dir=None)
    assert metrics["focus_selection"]["status"].startswith("skipped")
    assert metrics["mosaics"]["status"].startswith("skipped")
    assert metrics["multifocus_depth"]["status"] == "ok"  # GT local (sharp copiado)


def test_run_evaluation_no_gt_at_all(tmp_path):
    empty = tmp_path / "vazio"
    empty.mkdir()
    metrics = run_evaluation(empty, data_dir=None)
    statuses = [v["status"] for k, v in metrics.items() if k != "meta"]
    assert statuses and all(s.startswith("skipped") for s in statuses)
