"""Framework do sweep v2 (configs/sweep_v2): gerador, ranking, poda, runner."""
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

SWEEP = Path(__file__).resolve().parents[1] / "configs" / "sweep_v2"
sys.path.insert(0, str(SWEEP))

import run as sweep_run  # noqa: E402
import stages  # noqa: E402
import summarize  # noqa: E402

DATASETS = {
    "d1": {"input": "/raw/hybrid", "data_folder": "ds1", "pixel_size": 1.8, "role": "dev"},
    "dz": {"input": "/raw/hybrid", "data_folder": "dsz", "pixel_size": None, "role": "test"},
}


@pytest.fixture
def base():
    return yaml.safe_load((SWEEP / "base.yaml").read_text())


# --- gerador -----------------------------------------------------------------


def test_grid_count():
    assert len(stages.grid({"a": [1, 2, 3], "b": [True, False]})) == 6


def test_ofat_count():
    assert len(stages.ofat({"a": [1, 2, 3], "b": [True, False]}, center={})) == 5


def test_ofat_excludes_center_duplicates():
    deltas = stages.ofat({"a": [1, 2, 3]}, center={"a": 2})
    assert deltas == [{"a": 1}, {"a": 3}]


def test_materialize_unknown_key_raises(base, tmp_path):
    spec = stages.RunSpec("A1", "d1", "x", {"multifocus.nope": 1}, "full")
    with pytest.raises(KeyError):
        stages.materialize(spec, DATASETS, base, out_root=tmp_path)


def test_materialize_sets_dataset_fields(base, tmp_path):
    spec = stages.RunSpec("A1", "d1", "x", {"multifocus.focus_measure.parameters.radius": 0.03}, "full")
    path = stages.materialize(spec, DATASETS, base, out_root=tmp_path)
    cfg = yaml.safe_load(path.read_text())
    assert cfg["hybrid"]["integration"]["pixel_size"] == 1.8
    assert cfg["multifocus"]["parameters"]["z_foc"] == "auto"
    assert cfg["hybrid"]["integration"]["report_step"] == 0
    assert cfg["experiment"]["paths"]["data_folder"] == "ds1"
    assert cfg["multifocus"]["focus_measure"]["parameters"]["radius"] == 0.03
    assert cfg["experiment"]["paths"]["output"].endswith(os.path.join("A1", "d1", "x"))


def test_materialize_rejects_null_pixel_size(base, tmp_path):
    spec = stages.RunSpec("S0", "dz", "base", {}, "full")
    with pytest.raises(ValueError, match="pixel_size"):
        stages.materialize(spec, DATASETS, base, out_root=tmp_path)


def test_config_name_deterministic_and_unique():
    specs = stages.STAGES["A1"]("d1", stages.StageContext(DATASETS, centers={}))
    names = [s.name for s in specs]
    assert len(names) == len(set(names)) == 89
    again = [s.name for s in stages.STAGES["A1"]("d1", stages.StageContext(DATASETS, centers={}))]
    assert names == again
    assert all(len(n) <= 80 for n in names)


def test_config_name_readable():
    name = stages.config_name(
        {
            "multifocus.focus_measure.method": "fourier",
            "multifocus.focus_measure.parameters.radius": 0.05,
            "multifocus.focus_measure.preprocessing.smooth": True,
        }
    )
    assert name == "fourier_r0.05_sm1"
    assert stages.config_name({}) == "center"


# --- ranking -----------------------------------------------------------------


def _row(name, r, rmse, n_delta=1, status="ok", dataset="d1"):
    return {
        "dataset": dataset,
        "name": name,
        "status": status,
        "n_delta": n_delta,
        "integration_height_object.pearson_r": r,
        "integration_height_object.rmse_affine": rmse,
    }


def test_rank_tie_rule():
    ranked = summarize.rank_configs(
        [_row("a", 0.900, 10.0), _row("b", 0.897, 10.1), _row("c", 0.880, 10.0)]
    )
    assert [r["name"] for r in ranked] == ["a", "b", "c"]
    assert [r["tie_with_best"] for r in ranked] == [True, True, False]


def test_rank_tie_requires_rmse_close():
    ranked = summarize.rank_configs([_row("a", 0.900, 10.0), _row("b", 0.899, 11.0)])
    assert ranked[1]["tie_with_best"] is False


def test_pick_winner_prefers_fewer_overrides():
    ranked = summarize.rank_configs([_row("a", 0.900, 10.0, n_delta=3), _row("b", 0.898, 10.0, n_delta=1)])
    assert summarize.pick_winner(ranked)["name"] == "b"


def test_missing_metrics_excluded_from_ranking():
    rows = [_row("a", 0.9, 10.0), _row("b", None, None, status="missing")]
    ranked = summarize.rank_configs(rows)
    assert [r["name"] for r in ranked] == ["a"]


def test_aggregate_min_constraint():
    rows = [
        _row("A", 0.98, 1.0, dataset="d1"),
        _row("A", 0.82, 1.0, dataset="d2"),  # mean 0.90, min 0.82
        _row("B", 0.90, 1.0, dataset="d1"),
        _row("B", 0.88, 1.0, dataset="d2"),  # mean 0.89, min 0.88
    ]
    agg = summarize.aggregate(rows, ["d1", "d2"])
    assert agg[0]["name"] == "B"
    assert agg[0]["chosen"] is True


def test_aggregate_skips_configs_missing_a_dataset():
    rows = [_row("A", 0.9, 1.0, dataset="d1"), _row("B", 0.9, 1.0, dataset="d1"), _row("B", 0.9, 1.0, dataset="d2")]
    assert [a["name"] for a in summarize.aggregate(rows, ["d1", "d2"])] == ["B"]


def test_collect_rows_reads_spec_and_metrics(tmp_path):
    cfg = tmp_path / "A1" / "d1" / "cfgA"
    run = cfg / "20261121_1200_ds1"
    (run / "evaluation").mkdir(parents=True)
    (cfg / "spec.json").write_text(json.dumps({"name": "cfgA", "dataset": "d1", "delta": {"k": 1}, "overrides": {"k": 1}}))
    (run / "evaluation" / "metrics.json").write_text(
        json.dumps({"integration_height_object": {"status": "ok", "pearson_r": 0.9, "rmse_affine": 3.0}})
    )
    missing = tmp_path / "A1" / "d1" / "cfgB"
    missing.mkdir()
    (missing / "spec.json").write_text(json.dumps({"name": "cfgB", "dataset": "d1", "delta": {}, "overrides": {}}))
    rows = summarize.collect_rows(tmp_path, "A1", ["d1"])
    by = {r["name"]: r for r in rows}
    assert by["cfgA"]["integration_height_object.pearson_r"] == 0.9
    assert by["cfgA"]["n_delta"] == 1
    assert by["cfgA"]["run_dir"] == str(run)
    assert by["cfgB"]["status"] == "missing"


# --- poda e runner -----------------------------------------------------------


def _fake_run(root):
    run = root / "run"
    for light in ("L000", "average"):
        d = run / "multifocus_stereo" / light
        d.mkdir(parents=True)
        for f in ("sMos.npy", "sMos.png", "zMos.png", "focus_00.png", "iSel.png",
                  "iSel.fni", "wSel.fni", "sMos.fni", "zMos.fni", "zMos_with_confidence.fni"):
            (d / f).write_bytes(b"x")
    hi = run / "integration" / "height-iters" / "level-00"
    hi.mkdir(parents=True)
    (hi / "z.fni").write_bytes(b"x")
    (run / "integration" / "height_map.npy").write_bytes(b"x")
    (run / "photometric_stereo").mkdir()
    (run / "photometric_stereo" / "normal_map.npy").write_bytes(b"x")
    (run / "evaluation").mkdir()
    (run / "evaluation" / "metrics.json").write_text("{}")
    (run / "parameters.yaml").write_text("{}")
    return run


def test_prune_keeps_required_files(tmp_path):
    run = _fake_run(tmp_path)
    sweep_run.prune(run)
    mf = run / "multifocus_stereo"
    for keep in (
        mf / "L000" / "sMos.npy",
        mf / "L000" / "sMos.png",
        mf / "average" / "zMos.png",
        mf / "average" / "zMos_with_confidence.fni",
        run / "integration" / "height_map.npy",
        run / "photometric_stereo" / "normal_map.npy",
        run / "evaluation" / "metrics.json",
        run / "parameters.yaml",
    ):
        assert keep.exists(), keep
    assert not (mf / "L000" / "focus_00.png").exists()
    assert not (mf / "average" / "iSel.png").exists()
    assert not (run / "integration" / "height-iters").exists()
    # FNIs: só os do average que a avaliação (zMos) e os hints leem ficam
    assert (mf / "average" / "zMos.fni").exists()
    for gone in ("iSel.fni", "wSel.fni", "sMos.fni"):
        assert not (mf / "average" / gone).exists(), gone
    for gone in ("iSel.fni", "wSel.fni", "sMos.fni", "zMos.fni", "zMos_with_confidence.fni"):
        assert not (mf / "L000" / gone).exists(), gone


def test_prune_does_not_follow_symlinks(tmp_path):
    source = _fake_run(tmp_path / "src")
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    os.symlink(source / "multifocus_stereo", rerun / "multifocus_stereo")
    sweep_run.prune(rerun)
    assert (source / "multifocus_stereo" / "L000" / "focus_00.png").exists()


def test_latest_run_with_metrics(tmp_path):
    cfg = tmp_path / "cfg"
    old = cfg / "20261121_1200_ds"
    new = cfg / "20261121_1300_ds"
    (old / "evaluation").mkdir(parents=True)
    (old / "evaluation" / "metrics.json").write_text("{}")
    new.mkdir()  # run interrompido: sem metrics
    assert sweep_run.latest_done_run(cfg) == old
    assert sweep_run.latest_done_run(tmp_path / "nope") is None


def test_runner_skips_done(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(stages, "RESULTS", tmp_path)
    monkeypatch.setattr(stages, "load_datasets", lambda: DATASETS)
    done = tmp_path / "S0" / "d1" / "base" / "20261121_1200_ds1" / "evaluation"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text("{}")
    sweep_run.main(["--stage", "S0", "--datasets", "d1", "--dry-run"])
    out = capsys.readouterr().out
    assert "skip  S0/d1/base" in out
    assert "run   S0/d1/legacy15" in out


def test_c3_uses_a2_sigma_run_with_b_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "RESULTS", tmp_path)
    a2run = tmp_path / "A2" / "d1" / "c0_ss3.5" / "20261122_0800_ds1"
    (a2run / "evaluation").mkdir(parents=True)
    (a2run / "evaluation" / "metrics.json").write_text("{}")
    pre = "multifocus.focus_measure.preprocessing"
    solver = "photometric.solver.shadow_threshold"
    centers = {
        "A1": {"top": [{"name": "x", "overrides": {f"{pre}.smooth": True}, "run_dir": "/a1/x"},
                       {"name": "y", "overrides": {f"{pre}.smooth": True}, "run_dir": "/a1/y"}]},
        "A2": {"winner": {"overrides": {f"{pre}.smooth_sigma": 3.5}}},
        "C1": {"winner": {"overrides": {solver: 0.05}}},
    }
    monkeypatch.setattr(stages, "SOLVER", "photometric.solver")
    specs = stages.stage_c3("d1", stages.StageContext(DATASETS, centers))
    assert [s.name for s in specs] == ["a1top0", "a1top1"]
    assert specs[0].source_run == str(a2run)
    assert specs[0].overrides[f"{pre}.smooth_sigma"] == 3.5
    assert specs[0].overrides[solver] == 0.05
    assert specs[1].source_run == "/a1/y"  # sem run de A2: cai no de A1
    assert f"{pre}.smooth_sigma" not in specs[1].overrides


def test_vfinal_reruns_v_candidates_with_hw1(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "RESULTS", tmp_path)
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "candidates.yaml").write_text("lap9_ss3.5:\n  multifocus.focus_measure.parameters.kernel_size: 9\n")
    monkeypatch.setattr(stages, "SWEEP_DIR", sweep)
    run = tmp_path / "V" / "d1" / "lap9_ss3.5" / "20261122_1000_ds1"
    (run / "evaluation").mkdir(parents=True)
    (run / "evaluation" / "metrics.json").write_text("{}")
    specs = stages.stage_vfinal("d1", stages.StageContext(DATASETS, {}))
    assert [s.name for s in specs] == ["lap9_ss3.5_hw1"]  # candidatos sem run de V são pulados
    s = specs[0]
    assert s.mode == "integration" and s.source_run == str(run)
    assert s.overrides["hybrid.integration.hints_weight"] == 1.0
    assert s.overrides["multifocus.focus_measure.parameters.kernel_size"] == 9
