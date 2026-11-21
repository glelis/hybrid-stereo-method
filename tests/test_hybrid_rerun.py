"""Reexecução parcial do pipeline a partir de um run existente (sweep v2)."""
import copy
import json
import os

import numpy as np
import pytest
import yaml

from hybrid_stereo_method.hybrid.integrate import DEFAULT_EXECUTABLE
from hybrid_stereo_method.hybrid.rerun import check_upstream_compatible

BASE = {
    "experiment": {"paths": {"input": "/raw", "data_folder": "ds", "output": "/out"}},
    "multifocus": {
        "parameters": {"z_foc": [1.0, 2.0, 3.0], "interpolation": "linear_interpolation"},
        "focus_measure": {"method": "fourier", "parameters": {"radius": 0.1}},
    },
    "photometric": {"solver": {"shadow_threshold": 1e-3}},
    "hybrid": {"integration": {"hints_weight": 0.1}},
}


def _variant(path, value):
    cfg = copy.deepcopy(BASE)
    node = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return cfg


def test_check_upstream_identical_ok():
    assert check_upstream_compatible(copy.deepcopy(BASE), BASE, "integration") == []


def test_check_upstream_detects_multifocus_diff():
    new = _variant("multifocus.focus_measure.parameters.radius", 0.05)
    diffs = check_upstream_compatible(new, BASE, "photometric")
    assert len(diffs) == 1 and "radius" in diffs[0]


def test_check_upstream_photometric_ignored_for_photometric_stage():
    new = _variant("photometric.solver.shadow_threshold", 1e-2)
    assert check_upstream_compatible(new, BASE, "photometric") == []
    assert len(check_upstream_compatible(new, BASE, "integration")) == 1


def test_check_upstream_ignores_downstream_and_output():
    new = _variant("hybrid.integration.hints_weight", 0.3)
    new["experiment"]["paths"]["output"] = "/elsewhere"
    assert check_upstream_compatible(new, BASE, "integration") == []


def test_check_upstream_detects_dataset_diff():
    new = _variant("experiment.paths.data_folder", "other")
    assert len(check_upstream_compatible(new, BASE, "integration")) == 1


def test_zfoc_auto_matches_resolved_list():
    new = _variant("multifocus.parameters.z_foc", "auto")
    assert check_upstream_compatible(new, BASE, "integration") == []


# --- e2e: pipeline completo + rerun da integração -----------------------------

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)


@pytest.fixture
def full_run(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import hybrid_stereo_method.photometric.main_wps as main_wps_mod
    from test_e2e_hybrid import _build_dataset

    for name in ("disp_normalmap", "disp_channels", "disp_channels_3d"):
        monkeypatch.setattr(main_wps_mod, name, lambda **kw: None)
    from hybrid_stereo_method.hybrid.main import main as hybrid_main

    raw = tmp_path / "raw"
    _, z_foc = _build_dataset(raw)
    parameters = {
        "experiment": {
            "type": "hybrid",
            "paths": {"input": str(raw), "data_folder": "synth", "output": str(tmp_path / "results")},
            "settings": {"debug": False, "gabaritos": False},
        },
        "multifocus": {
            "focus_measure": {
                "method": "laplacian",
                "parameters": {"kernel_size": 5, "radius": None},
                "preprocessing": {
                    "square": True,
                    "smooth": True,
                    "spatial_median_filter": False,
                    "zero_border": False,
                },
            },
            "optimization": {"r_max": 2},
            "parameters": {"z_foc": z_foc, "interpolation": "linear_interpolation"},
        },
        "photometric": {
            "solver": {"epsilon": 1e-6, "shadow_threshold": 1e-3, "outlier_threshold_multiplier": 3}
        },
        "hybrid": {
            "integration": {
                "initial_method": "zero",
                "use_hints": False,
                "max_iter": 20000,
                "conv_tol": 5e-7,
            }
        },
        "evaluation": {"enabled": True},
    }
    hybrid_main(copy.deepcopy(parameters))
    (source,) = list((tmp_path / "results").glob("*_synth"))
    param_file = tmp_path / "rerun.yaml"
    parameters["experiment"]["paths"]["output"] = str(tmp_path / "rerun")
    param_file.write_text(yaml.safe_dump(parameters))
    return source, param_file


@needs_binary
@pytest.mark.slow
def test_rerun_integration_matches_full_run(full_run):
    from hybrid_stereo_method.hybrid.rerun import main as rerun_main

    source, param_file = full_run
    new = rerun_main(
        ["--source_run", str(source), "--from_stage", "integration", "--param_file", str(param_file)]
    )
    h_src = np.load(source / "integration" / "height_map.npy")
    h_new = np.load(os.path.join(new, "integration", "height_map.npy"))
    assert np.allclose(h_src, h_new, atol=1e-6)
    m_src = json.loads((source / "evaluation" / "metrics.json").read_text())
    m_new = json.loads(open(os.path.join(new, "evaluation", "metrics.json")).read())
    assert m_new["integration_height"]["pearson_r"] == pytest.approx(
        m_src["integration_height"]["pearson_r"], abs=1e-9
    )
    saved = yaml.safe_load(open(os.path.join(new, "parameters.yaml")))
    assert saved["rerun"] == {"source_run": str(source.resolve()), "from_stage": "integration"}


@needs_binary
@pytest.mark.slow
def test_rerun_creates_symlinks_not_copies(full_run):
    from hybrid_stereo_method.hybrid.rerun import main as rerun_main

    source, param_file = full_run
    new = rerun_main(
        ["--source_run", str(source), "--from_stage", "photometric", "--param_file", str(param_file)]
    )
    assert os.path.islink(os.path.join(new, "multifocus_stereo"))
    assert not os.path.islink(os.path.join(new, "photometric_stereo"))
    assert os.path.exists(os.path.join(new, "photometric_stereo", "normal_map.npy"))
    assert os.path.exists(os.path.join(new, "evaluation", "metrics.json"))


@needs_binary
@pytest.mark.slow
def test_rerun_incompatible_config_aborts(full_run, capsys):
    from hybrid_stereo_method.hybrid.rerun import main as rerun_main

    source, param_file = full_run
    cfg = yaml.safe_load(param_file.read_text())
    cfg["multifocus"]["optimization"]["r_max"] = 4
    param_file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit) as exc:
        rerun_main(
            ["--source_run", str(source), "--from_stage", "integration", "--param_file", str(param_file)]
        )
    assert exc.value.code == 2
    assert "r_max" in capsys.readouterr().err
