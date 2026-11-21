"""Avaliação: mosaico lido do sMos.npy na escala do GT e guarda de z_foc divergente."""

import cv2
import numpy as np
import yaml

from hybrid_stereo_method.evaluation.loaders import (
    find_smos_pairs,
    load_configured_z_foc,
    load_smos_pair,
)


def _layout(tmp_path, with_npy=True):
    rng = np.random.default_rng(0)
    gt16 = (rng.random((20, 24)) * 30000 + 1000).astype(np.uint16)  # 16 bits, faixa estreita
    data = tmp_path / "data"
    (data / "L000" / "sharp").mkdir(parents=True)
    cv2.imwrite(str(data / "L000" / "sharp" / "sVal.png"), gt16)
    d = tmp_path / "results" / "multifocus_stereo" / "L000"
    d.mkdir(parents=True)
    stretched = cv2.normalize(gt16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(d / "sMos.png"), stretched)
    if with_npy:
        np.save(d / "sMos.npy", gt16.astype(np.float32))
    return tmp_path / "results", data, gt16


def test_pairs_prefer_lossless_npy(tmp_path):
    results, data, _ = _layout(tmp_path)
    [(light, est, _gt)] = find_smos_pairs(results, data)
    assert light == "L000" and est.name == "sMos.npy"


def test_identical_mosaic_is_identical_on_the_same_scale(tmp_path):
    """Um sMos.npy igual ao GT deve dar erro zero; o PNG min-max não daria."""
    results, data, gt16 = _layout(tmp_path)
    [(_l, est_path, gt_path)] = find_smos_pairs(results, data)
    est, gt = load_smos_pair(est_path, gt_path)
    np.testing.assert_allclose(est, gt)
    np.testing.assert_allclose(gt, gt16 * (255.0 / 65535.0))

    est_png, gt_png = load_smos_pair(est_path.with_suffix(".png"), gt_path)
    assert np.abs(est_png - gt_png).max() > 10  # o esticamento aparece como erro


def test_falls_back_to_png_for_old_runs(tmp_path):
    results, data, _ = _layout(tmp_path, with_npy=False)
    [(_l, est, _gt)] = find_smos_pairs(results, data)
    assert est.name == "sMos.png"


def test_configured_z_foc_read_from_saved_parameters(tmp_path):
    assert load_configured_z_foc(tmp_path) is None
    params = {"multifocus": {"parameters": {"z_foc": [7.2, 32.8]}}}
    (tmp_path / "parameters.yaml").write_text(yaml.safe_dump(params))
    assert load_configured_z_foc(tmp_path) == [7.2, 32.8]


def test_hdev_mask_reports_excluded_fraction():
    from hybrid_stereo_method.evaluation.evaluators import evaluate_height

    gt = np.arange(100, dtype=float).reshape(10, 10)
    mask = np.ones_like(gt, dtype=bool)
    mask[:2] = False
    out = evaluate_height(gt * 2.0, gt, mask)
    assert out["gt_mask_excluded_fraction"] == 0.2
    assert "gt_mask_excluded_fraction" not in evaluate_height(gt * 2.0, gt)


def test_unknown_focus_indicator_raises_value_error():
    import pytest

    from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator

    with pytest.raises(ValueError, match="Unknown focus indicator"):
        focus_indicator(np.zeros((1, 8, 8), np.float32), "sobel")


def test_relative_input_path_resolved_from_results_ancestors(tmp_path, monkeypatch):
    from hybrid_stereo_method.evaluation.loaders import resolve_data_dir

    repo = tmp_path / "repo"
    data = repo / "data" / "raw" / "ds"
    data.mkdir(parents=True)
    results = repo / "data" / "results" / "run1"
    results.mkdir(parents=True)
    params = {"experiment": {"paths": {"input": "data/raw/", "data_folder": "ds"}}}
    (results / "parameters.yaml").write_text(yaml.safe_dump(params))
    monkeypatch.chdir(tmp_path)  # CWD fora da raiz do repo
    assert resolve_data_dir(results, None).resolve() == data.resolve()
