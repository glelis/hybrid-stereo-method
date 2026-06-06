"""Testes dos avaliadores por etapa (arrays → dict de métricas)."""

import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height

from hybrid_stereo_method.evaluation.evaluators import (
    evaluate_focus_selection,
    evaluate_height,
    evaluate_hybrid_gain,
    evaluate_mosaics,
    evaluate_normals,
)


def test_evaluate_height_affine_invariant_and_masks_nan():
    z = gaussian_bump(32, amplitude=6.0)
    est = 2.0 * z + 3.0  # afim do GT → erro ~0
    est[0, 0] = np.nan
    out = evaluate_height(est, z)
    assert out["status"] == "ok"
    assert out["rmse_affine"] == pytest.approx(0.0, abs=1e-9)
    assert out["a"] == pytest.approx(0.5, abs=1e-9)  # fit é gt ≈ a*est + b
    assert out["pearson_r"] == pytest.approx(1.0)
    assert out["valid_fraction"] == pytest.approx(1023 / 1024)
    assert out["low_validity"] is False
    assert np.isnan(out["_error_map"][0, 0])
    assert out["_error_map"][16, 16] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_height_shape_mismatch_and_empty():
    z = np.zeros((4, 4))
    assert evaluate_height(np.zeros((5, 4)), z)["status"].startswith("skipped")
    assert evaluate_height(np.full((4, 4), np.nan), z)["status"].startswith("skipped")


def test_evaluate_height_low_validity_flag():
    z = np.random.default_rng(0).uniform(size=(20, 20))
    est = np.full_like(z, np.nan)
    est[0, 0] = z[0, 0]
    est[0, 1] = z[0, 1]
    out = evaluate_height(est, z)  # 2/400 = 0.5% < 1%
    assert out["status"] == "ok"
    assert out["low_validity"] is True


def test_evaluate_focus_selection_frames_and_percentages():
    z_vals = [10.0, 20.0, 30.0]
    z_gt = np.full((4, 4), 20.0)
    zmos = z_gt.copy()
    zmos[0, 0] = 30.0  # 1 frame de erro
    zmos[0, 1] = np.nan
    out = evaluate_focus_selection(zmos, z_gt, z_vals)
    assert out["status"] == "ok"
    assert out["step_z"] == pytest.approx(10.0)
    assert out["median_err_frames"] == pytest.approx(0.0)
    assert out["exact_match_pct"] == pytest.approx(14 / 15 * 100)
    assert out["within_1_frame_pct"] == pytest.approx(100.0)
    assert out["valid_fraction"] == pytest.approx(15 / 16)


def test_evaluate_mosaics_per_light_and_aggregates():
    rng = np.random.default_rng(1)
    gt = rng.uniform(0, 255, (16, 16, 3)).astype(np.uint8)
    noisy = np.clip(gt.astype(int) + 12, 0, 255).astype(np.uint8)
    out = evaluate_mosaics([("L000", gt, gt), ("L001", noisy, gt)])
    assert out["status"] == "ok"
    assert out["per_light"]["L000"]["ssim"] == pytest.approx(1.0)
    assert out["per_light"]["L001"]["ssim"] < 1.0
    assert out["ssim_min"] == out["per_light"]["L001"]["ssim"]
    assert out["worst_light"] == "L001"
    assert evaluate_mosaics([])["status"].startswith("skipped")


def test_evaluate_normals_dual_orientation_resolves_y_frame():
    z = gaussian_bump(24, amplitude=5.0)
    n = normals_from_height(z)
    n_gt_yup = n.copy()
    n_gt_yup[..., 1] *= -1.0  # GT num frame y-up (POV-Ray)
    fg = np.ones(z.shape, dtype=bool)
    out = evaluate_normals(n, n_gt_yup, fg)
    assert out["status"] == "ok"
    assert out["winning_orientation"] == "y_flipped"
    assert out["y_flipped"]["mean_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["y_as_is"]["mean_deg"] > 1.0
    assert out["_error_map"].shape == z.shape


def test_evaluate_normals_masks_background_and_nan():
    z = gaussian_bump(16, amplitude=5.0)
    n = normals_from_height(z)
    est = n.copy()
    est[0, 0, :] = np.nan
    fg = np.ones(z.shape, dtype=bool)
    fg[0, 1] = False
    out = evaluate_normals(est, n, fg)
    assert out["status"] == "ok"
    assert out["winning_orientation"] == "y_as_is"
    assert np.isnan(out["_error_map"][0, 0])
    assert np.isnan(out["_error_map"][0, 1])
    expected_valid = (16 * 16 - 2) / (16 * 16)
    assert out["y_as_is"]["valid_fraction"] == pytest.approx(expected_valid)


def test_evaluate_hybrid_gain_on_common_mask():
    z = gaussian_bump(32, amplitude=6.0)
    rng = np.random.default_rng(2)
    mf = z + rng.normal(0.0, 0.5, z.shape)  # multifocus ruidoso
    final = z + rng.normal(0.0, 0.05, z.shape)  # híbrido melhor
    mf[0, 0] = np.nan
    out = evaluate_hybrid_gain(mf, final, z)
    assert out["status"] == "ok"
    assert out["gain"] > 1.0
    assert out["rmse_final"] < out["rmse_multifocus"]
    assert out["valid_fraction"] == pytest.approx(1023 / 1024)
    assert evaluate_hybrid_gain(np.zeros((3, 3)), final, z)["status"].startswith("skipped")
