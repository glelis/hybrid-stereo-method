"""Testes da serialização JSON, mapas de erro e report.md."""
import json

import numpy as np

from hybrid_stereo_method.evaluation.report import (
    json_safe,
    save_error_map,
    write_metrics_json,
    write_report_md,
)


def test_json_safe_strips_arrays_and_handles_nonfinite():
    metrics = {
        "stage": {
            "status": "ok",
            "rmse": np.float64(1.5),
            "count": np.int64(7),
            "psnr": float("inf"),
            "r": float("nan"),
            "_error_map": np.zeros((4, 4)),
            "nested": {"ssim": np.float32(0.5), "_mask": np.ones(3)},
        }
    }
    out = json_safe(metrics)
    assert out["stage"]["rmse"] == 1.5
    assert out["stage"]["count"] == 7
    assert out["stage"]["psnr"] == "inf"
    assert out["stage"]["r"] == "nan"
    assert "_error_map" not in out["stage"]
    assert "_mask" not in out["stage"]["nested"]
    assert out["stage"]["nested"]["ssim"] == 0.5
    json.dumps(out)  # serializável de ponta a ponta


def test_write_metrics_json_roundtrip(tmp_path):
    path = write_metrics_json({"meta": {"x": 1}, "s": {"status": "ok"}}, tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["s"]["status"] == "ok"


def test_save_error_map_writes_png_with_nan(tmp_path):
    err = np.random.default_rng(0).uniform(0.0, 2.0, (16, 16))
    err[0, 0] = np.nan
    path = save_error_map(err, tmp_path / "err.png", title="teste")
    assert path.exists()
    assert path.stat().st_size > 0


def test_write_report_md_renders_all_sections(tmp_path):
    metrics = {
        "meta": {"results_dir": "/r", "data_dir": "/d", "timestamp": "t", "package_version": "v"},
        "multifocus_depth": {
            "status": "ok", "rmse_affine": 0.1, "mae_affine": 0.05, "a": 1.0, "b": 0.0,
            "pearson_r": 0.99, "gt_std": 1.0, "valid_fraction": 0.98, "low_validity": False,
        },
        "focus_selection": {"status": "skipped: sem data_dir"},
        "mosaics": {
            "status": "ok", "n_lights": 1, "psnr_mean": 30.0, "psnr_min": 30.0,
            "ssim_mean": 0.9, "ssim_min": 0.9, "worst_light": "L000",
            "per_light": {"L000": {"status": "ok", "psnr": 30.0, "ssim": 0.9}},
        },
        "photometric_normals": {
            "status": "ok", "winning_orientation": "y_flipped",
            "y_as_is": {"mean_deg": 20.0, "median_deg": 19.0, "p95_deg": 30.0,
                        "valid_fraction": 0.97},
            "y_flipped": {"mean_deg": 2.0, "median_deg": 1.5, "p95_deg": 5.0,
                          "valid_fraction": 0.97},
            "valid_fraction": 0.97, "low_validity": False,
        },
        "integration_height": {
            "status": "ok", "rmse_affine": 0.05, "mae_affine": 0.02, "a": 1.0, "b": 0.0,
            "pearson_r": 0.999, "gt_std": 1.0, "valid_fraction": 0.99, "low_validity": False,
        },
        "hybrid_gain": {
            "status": "ok", "rmse_multifocus": 0.1, "rmse_final": 0.05, "gain": 2.0,
            "pearson_multifocus": 0.99, "pearson_final": 0.999,
            "valid_fraction": 0.97, "low_validity": False,
        },
    }
    error_maps = {"multifocus_depth": "multifocus_depth_error.png"}
    path = write_report_md(metrics, error_maps, tmp_path)
    text = path.read_text()
    assert "## Multifocus — profundidade" in text
    assert "skipped: sem data_dir" in text
    assert "L000" in text
    assert "y_flipped" in text  # veredito CONV-2
    assert "2.0" in text  # ganho
    assert "![multifocus_depth](multifocus_depth_error.png)" in text
