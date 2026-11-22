"""Executa um estágio do sweep v2 em paralelo, de forma idempotente.

Uso:
    python configs/sweep_v2/run.py --stage A1 --datasets d1,d2 -j 3 --prune
    python configs/sweep_v2/run.py --stage S0 --datasets d1 --dry-run

Pula specs cujo run mais recente já tem evaluation/metrics.json (reexecutar o
comando retoma um estágio interrompido). Cada run grava spec.json na pasta da
config (lido por summarize.py) e uma linha em data/results/sweep_v2/manifest.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stages  # noqa: E402

# PNGs de multifocus_stereo/<luz|average>/ que ficam (visualização leve).
_KEEP_PREFIXES = ("sMos", "zMos")
# FNIs que a avaliação (average/zMos.fni) e os hints (average/zMos_with_confidence.fni)
# leem. O fotométrico lê os sMos.npy por luz, que nunca são apagados.
_KEEP_AVERAGE_FNI = ("zMos.fni", "zMos_with_confidence.fni")


latest_done_run = stages.latest_done_run  # implementação única (stages.py)


def prune(run_dir: Path) -> int:
    """Apaga o que é pesado e não é lido depois; nunca segue symlinks.

    Remove integration/height-iters/, os PNGs de multifocus_stereo/*/ exceto
    sMos*/zMos* e os FNIs exceto average/zMos*.fni. Devolve os bytes liberados.
    """
    run_dir = Path(run_dir)
    freed = 0
    iters = run_dir / "integration" / "height-iters"
    if iters.is_dir() and not iters.is_symlink():
        freed += sum(f.stat().st_size for f in iters.rglob("*") if f.is_file())
        shutil.rmtree(iters)
    mf = run_dir / "multifocus_stereo"
    if mf.is_symlink() or not mf.is_dir():
        return freed
    for root, dirs, files in os.walk(mf, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            if name.endswith(".png"):
                if name.startswith(_KEEP_PREFIXES):
                    continue
            elif name.endswith(".fni"):
                if os.path.basename(root) == "average" and name in _KEEP_AVERAGE_FNI:
                    continue
            else:
                continue
            freed += os.path.getsize(path)
            os.remove(path)
    return freed


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=stages.ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _command(spec: stages.RunSpec, cfg_path: Path) -> list[str]:
    if spec.mode == "full":
        return [sys.executable, "-m", "hybrid_stereo_method.hybrid.main", "--param_file", str(cfg_path)]
    return [sys.executable, "-m", "hybrid_stereo_method.hybrid.rerun", "--source_run", spec.source_run,
            "--from_stage", spec.mode, "--param_file", str(cfg_path)]


def _execute(spec: stages.RunSpec, cfg_path: Path, threads: int, do_prune: bool) -> dict:
    out = spec.out_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(
        {"stage": spec.stage, "dataset": spec.dataset, "name": spec.name, "mode": spec.mode,
         "source_run": spec.source_run, "delta": spec.delta, "overrides": spec.overrides},
        indent=1, default=str))
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    env["MPLBACKEND"] = "Agg"
    log = out / f"_run_{datetime.now():%Y%m%d_%H%M%S}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(_command(spec, cfg_path), cwd=stages.ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, check=False)
    run = latest_done_run(out)
    ok = proc.returncode == 0 and run is not None
    freed = prune(run) if (ok and do_prune) else 0
    return {"stage": spec.stage, "dataset": spec.dataset, "name": spec.name,
            "run_dir": str(run) if run else None, "status": "ok" if ok else "failed",
            "returncode": proc.returncode, "seconds": round(time.time() - t0, 1),
            "freed_mb": round(freed / 2**20, 1), "log": str(log)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=sorted(stages.STAGES))
    ap.add_argument("--datasets", required=True, help="ids separados por vírgula (datasets.yaml)")
    ap.add_argument("-j", "--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="threads BLAS/OMP por run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--force-new", action="store_true", help="ignora o skip e cria runs novos")
    ap.add_argument("--only", default=None, help="roda só specs cujo nome contém este texto")
    ap.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                    help="override extra para todos os specs (valor em YAML)")
    args = ap.parse_args(argv)

    datasets = stages.load_datasets()
    base = stages.load_base()
    extra = {}
    for item in args.set:
        key, _, value = item.partition("=")
        extra[key] = yaml.safe_load(value)

    todo: list[tuple[stages.RunSpec, Path]] = []
    for ds in args.datasets.split(","):
        if ds not in datasets:
            raise SystemExit(f"dataset desconhecido: {ds}")
        ctx = stages.StageContext(datasets, stages.load_centers(ds))
        for spec in stages.STAGES[args.stage](ds, ctx):
            if args.only and args.only not in spec.name:
                continue
            if extra:
                spec.overrides = {**spec.overrides, **extra}
                spec.delta = {**spec.delta, **extra}
                # pasta própria: sem isso o run original contaria como feito (skip) ou,
                # com --force-new, o spec.json da config seria sobrescrito
                spec.name = f"{spec.name}_{stages.config_name(extra)}"
            label = f"{spec.stage}/{spec.dataset}/{spec.name}"
            if not args.force_new and latest_done_run(spec.out_dir()) is not None:
                print(f"skip  {label}")
                continue
            print(f"run   {label}  [{spec.mode}]")
            if args.dry_run:
                continue
            todo.append((spec, stages.materialize(spec, datasets, base)))
    print(f"{len(todo)} runs a executar")
    if args.dry_run or not todo:
        return

    manifest = stages.RESULTS / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_execute, s, p, args.threads, args.prune): s for s, p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            rec["git_sha"] = sha
            with open(manifest, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{i}/{len(todo)}] {rec['status']:6} {rec['stage']}/{rec['dataset']}/{rec['name']} "
                  f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
