"""Tabela, ranking e promoção de um estágio do sweep v2.

Uso:
    python configs/sweep_v2/summarize.py --stage A1 --datasets d1,d2
    python configs/sweep_v2/summarize.py --stage A1 --datasets d1,d2 --promote --top 3
    python configs/sweep_v2/summarize.py --stage S0 --datasets d1 --promote --pick base
    python configs/sweep_v2/summarize.py --stage V --datasets d1,d2,d3,d4 --aggregate

Métrica primária: integration_height_object.pearson_r (desempate: rmse_affine).
Empate: |Δr| < tie_r E Δrmse relativo < tie_rmse; no empate vence quem tem
menos overrides em relação ao centro (n_delta), depois a maior r.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stages  # noqa: E402
from run import latest_done_run  # noqa: E402

PRIMARY = "integration_height_object.pearson_r"
TIEBREAK = "integration_height_object.rmse_affine"
METRICS = [
    PRIMARY,
    TIEBREAK,
    "integration_height_object.pearson_detrended",
    "integration_height.pearson_r",
    "integration_height.rmse_affine",
    "multifocus_depth_object.pearson_r",
    "multifocus_depth.pearson_r",
    "focus_selection.median_err_frames",
    "focus_selection.within_1_frame_pct",
    "photometric_normals.mean_deg",
    "photometric_normals.p95_deg",
    "hybrid_gain_object.gain",
    "meta.object_fraction",
]
DEFAULT_OUT = stages.ROOT / "docs" / "relatorios" / "sweep_v2"


def _get(metrics: dict, dotted: str) -> Any:
    block, _, key = dotted.partition(".")
    b = metrics.get(block, {})
    if block == "photometric_normals" and b.get("status") == "ok":
        b = b.get(b.get("winning_orientation", ""), {})
    elif block != "meta" and b.get("status") != "ok":
        return None
    v = b.get(key)
    return v if isinstance(v, (int, float)) else None


def collect_rows(root: Path, stage: str, datasets: list[str]) -> list[dict]:
    rows = []
    for ds in datasets:
        d = Path(root) / stage / ds
        if not d.is_dir():
            continue
        for cfg_dir in sorted(p for p in d.iterdir() if (p / "spec.json").exists()):
            spec = json.loads((cfg_dir / "spec.json").read_text())
            row: dict[str, Any] = {"dataset": ds, "name": spec["name"], "n_delta": len(spec.get("delta", {})),
                                   "delta": spec.get("delta", {}), "overrides": spec.get("overrides", {})}
            run = latest_done_run(cfg_dir)
            if run is None:
                row.update(status="missing", run_dir=None)
            else:
                m = json.loads((run / "evaluation" / "metrics.json").read_text())
                row.update({k: _get(m, k) for k in METRICS})
                row["run_dir"] = str(run)
                row["status"] = "ok" if row[PRIMARY] is not None else "no-metric"
            rows.append(row)
    return rows


def rank_configs(rows: list[dict], metric: str = PRIMARY, tie_r: float = 0.005,
                 tie_rmse: float = 0.02) -> list[dict]:
    ok = [dict(r) for r in rows if r.get("status") == "ok" and r.get(metric) is not None]
    ok.sort(key=lambda r: (-r[metric], r.get(TIEBREAK) or 0.0))
    if not ok:
        return ok
    best = ok[0]
    for r in ok:
        close_r = best[metric] - r[metric] < tie_r
        b_rmse, r_rmse = best.get(TIEBREAK), r.get(TIEBREAK)
        close_rmse = b_rmse is None or r_rmse is None or abs(r_rmse - b_rmse) / max(b_rmse, 1e-12) < tie_rmse
        r["tie_with_best"] = bool(close_r and close_rmse)
    return ok


def pick_winner(ranked: list[dict], metric: str = PRIMARY) -> dict:
    tied = [r for r in ranked if r["tie_with_best"]]
    return min(tied, key=lambda r: (r["n_delta"], -r[metric]))


def aggregate(rows: list[dict], datasets: list[str], metric: str = PRIMARY, slack: float = 0.01) -> list[dict]:
    """Por config presente em todos os datasets: mean_r, min_r, mean_rank; escolha
    = maior mean_r com min_r >= (melhor min_r) - slack."""
    ranks: dict[tuple[str, str], int] = {}
    for ds in datasets:
        for i, r in enumerate(rank_configs([r for r in rows if r["dataset"] == ds], metric), 1):
            ranks[(ds, r["name"])] = i
    by_name: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("status") == "ok" and r.get(metric) is not None:
            by_name.setdefault(r["name"], {})[r["dataset"]] = r[metric]
    out = []
    for name, vals in by_name.items():
        if not all(ds in vals for ds in datasets):
            continue
        v = [vals[ds] for ds in datasets]
        out.append({"name": name, "mean_r": sum(v) / len(v), "min_r": min(v),
                    "mean_rank": sum(ranks[(ds, name)] for ds in datasets) / len(datasets),
                    **{f"r_{ds}": vals[ds] for ds in datasets}})
    if not out:
        return out
    floor = max(a["min_r"] for a in out) - slack
    for a in out:
        a["eligible"] = a["min_r"] >= floor
    out.sort(key=lambda a: (not a["eligible"], -a["mean_r"]))
    for i, a in enumerate(out):
        a["chosen"] = i == 0
    return out


def _center_rows(stage: str, ds: str) -> list[dict]:
    """Linhas de referência: o(s) centro(s) do estágio, lidos dos runs promovidos."""
    prev = {"S0conv": ["S0"], "A2": ["A1"], "B": ["A2"], "C1": ["B"], "C2": ["C1"], "C3": ["C1"],
            "Vhints": ["V"]}.get(stage, [])
    centers = stages.load_centers(ds)
    out = []
    for p in prev:
        c = centers.get(p)
        if not c:
            continue
        items = c.get("top") if stage == "A2" else [c["winner"]]
        for i, item in enumerate(items or []):
            run = Path(item["run_dir"]) if item.get("run_dir") else None
            if run is None or not (run / "evaluation" / "metrics.json").exists():
                continue
            m = json.loads((run / "evaluation" / "metrics.json").read_text())
            row = {"dataset": ds, "name": f"_center{i if stage == 'A2' else ''}:{item['name']}", "n_delta": 0,
                   "delta": {}, "overrides": item["overrides"], "run_dir": str(run), "status": "ok",
                   **{k: _get(m, k) for k in METRICS}}
            out.append(row)
    return out


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_outputs(stage: str, rows: list[dict], ranked_by_ds: dict[str, list[dict]],
                  agg: list[dict] | None, out: Path, top_n: int = 10) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "name", "status", "n_delta", *METRICS, "delta", "run_dir"]
    with open(out / f"{stage}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "delta": json.dumps(r.get("delta", {}), default=str)})

    lines = [f"# Sweep v2 — estágio {stage}", "", f"Gerado em {datetime.now():%Y-%m-%d %H:%M}.",
             f"Métrica primária: `{PRIMARY}`; ✓ = empatado com o melhor.", ""]
    short = [PRIMARY, TIEBREAK, "integration_height.pearson_r", "multifocus_depth_object.pearson_r",
             "focus_selection.median_err_frames", "photometric_normals.mean_deg"]
    head = ["#", "config", "r obj", "rmse obj", "r img", "r mf obj", "foco med (fr)", "normais (°)", "Δr centro", "empate"]
    for ds, ranked in ranked_by_ds.items():
        n_rows = [r for r in rows if r["dataset"] == ds]
        missing = [r["name"] for r in n_rows if r["status"] != "ok"]
        center = next((r for r in ranked if r["name"].startswith("_center")), None)
        lines += [f"## {ds}", "", f"{len(ranked)} configs com métrica; {len(missing)} sem: "
                  f"{', '.join(missing) if missing else '—'}", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        shown = ranked[:top_n] + ([center] if center and center not in ranked[:top_n] else [])
        for r in shown:
            i = ranked.index(r) + 1
            d = (r[PRIMARY] - center[PRIMARY]) if center else None
            vals = [_fmt(r.get(k), 2 if "frames" in k or "deg" in k else 4) for k in short]
            lines.append(f"| {i} | `{r['name']}` | " + " | ".join(vals)
                         + f" | {_fmt(d) if d is not None else '—'} | {_fmt(r['tie_with_best'])} |")
        lines.append("")
    if agg:
        dss = [k[2:] for k in agg[0] if k.startswith("r_")]
        lines += ["## Agregado multi-dataset", "",
                  "| config | " + " | ".join(f"r {d}" for d in dss) + " | média | mín | rank médio | elegível | escolhida |",
                  "|" + "---|" * (len(dss) + 6)]
        for a in agg:
            lines.append(f"| `{a['name']}` | " + " | ".join(_fmt(a[f'r_{d}']) for d in dss)
                         + f" | {_fmt(a['mean_r'])} | {_fmt(a['min_r'])} | {a['mean_rank']:.1f} "
                         f"| {_fmt(a['eligible'])} | {_fmt(a['chosen'])} |")
        lines.append("")
    (out / f"{stage}.md").write_text("\n".join(lines))


def _item(r: dict) -> dict:
    return {"name": r["name"].split(":", 1)[-1], "overrides": r["overrides"], "run_dir": r["run_dir"],
            PRIMARY: r[PRIMARY]}


def promote(stage: str, ds: str, ranked: list[dict], top: int | None, pick: str | None) -> Path:
    if pick:
        matches = [r for r in ranked if r["name"].split(":", 1)[-1] == pick]
        if not matches:
            raise SystemExit(f"{ds}: config '{pick}' não está no ranking de {stage}")
        winner, how = matches[0], "manual (--pick)"
    else:
        winner, how = pick_winner(ranked), "regra de empate"
    data: dict[str, Any] = {"stage": stage, "promoted_at": datetime.now().isoformat(timespec="seconds"),
                            "how": how, "winner": _item(winner)}
    if top:
        seen, items = set(), []
        for r in ranked:
            key = json.dumps(r["overrides"], sort_keys=True, default=str)
            if key in seen or r["name"].startswith("_center"):
                continue
            seen.add(key)
            items.append(_item(r))
            if len(items) == top:
                break
        data["top"] = items
    path = stages.CENTERS / ds / f"{stage}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--pick", default=None, help="promove esta config em vez do vencedor da regra")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--tie-r", type=float, default=0.005)
    args = ap.parse_args(argv)

    dss = args.datasets.split(",")
    rows = collect_rows(stages.RESULTS, args.stage, dss)
    for ds in dss:
        rows += _center_rows(args.stage, ds)
    ranked_by_ds = {ds: rank_configs([r for r in rows if r["dataset"] == ds], tie_r=args.tie_r) for ds in dss}
    agg = aggregate([r for r in rows if not r["name"].startswith("_center")], dss) if args.aggregate else None
    write_outputs(args.stage, rows, ranked_by_ds, agg, args.out)
    print(f"escrito {args.out / (args.stage + '.md')}")
    for ds, ranked in ranked_by_ds.items():
        if not ranked:
            print(f"{ds}: nenhuma config com métrica")
            continue
        w = pick_winner(ranked)
        print(f"{ds}: melhor {ranked[0]['name']} r={ranked[0][PRIMARY]:.4f} | vencedor (empate) {w['name']} r={w[PRIMARY]:.4f}")
        if args.promote:
            print(f"  promovido → {promote(args.stage, ds, ranked, args.top, args.pick)}")


if __name__ == "__main__":
    main()
