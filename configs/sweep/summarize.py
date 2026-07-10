"""Tabela comparativa das configs do sweep (lê evaluation/metrics.json de cada run).

Uso:  python configs/sweep/summarize.py [pasta-do-sweep]
Padrão: data/results/hybrid_sweep. Usa o run mais recente de cada config.
"""
import json
import sys
from pathlib import Path

DEFAULT_SWEEP_DIR = Path(__file__).resolve().parents[2] / "data" / "results" / "hybrid_sweep"


def latest_metrics(config_dir: Path) -> dict | None:
    runs = sorted([d for d in config_dir.iterdir() if d.is_dir()], reverse=True)
    for run in runs:
        f = run / "evaluation" / "metrics.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def fmt(value, spec=".3f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def main() -> None:
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEP_DIR
    rows = []
    for config_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        m = latest_metrics(config_dir)
        if m is None:
            rows.append((config_dir.name, None))
            continue
        mf = m.get("multifocus_depth", {})
        fs = m.get("focus_selection", {})
        ih = m.get("integration_height", {})
        gain = m.get("hybrid_gain", {})
        rows.append(
            (
                config_dir.name,
                {
                    "mf_r": mf.get("pearson_r"),
                    "mf_rmse": mf.get("rmse_affine"),
                    "foc_med": fs.get("median_err_frames"),
                    "foc_1fr": fs.get("within_1_frame_pct"),
                    "h_r": ih.get("pearson_r"),
                    "h_rmse": ih.get("rmse_affine"),
                    "gain": gain.get("gain") if isinstance(gain, dict) else None,
                },
            )
        )

    print(
        "| config | mf Pearson r | mf RMSE | foco mediano (fr) | foco ±1fr (%) "
        "| altura Pearson r | altura RMSE | ganho |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, r in rows:
        if r is None:
            print(f"| {name} | sem metrics.json | | | | | | |")
            continue
        print(
            f"| {name} | {fmt(r['mf_r'])} | {fmt(r['mf_rmse'], '.0f')} "
            f"| {fmt(r['foc_med'], '.2f')} | {fmt(r['foc_1fr'], '.1f')} "
            f"| {fmt(r['h_r'])} | {fmt(r['h_rmse'], '.0f')} | {fmt(r['gain'], '.2f')} |"
        )

    scored = [(r["h_r"], name) for name, r in rows if r and isinstance(r.get("h_r"), float)]
    if scored:
        best_r, best_name = max(scored)
        print(f"\nMelhor altura final (Pearson r): **{best_name}** ({best_r:.3f})")


if __name__ == "__main__":
    main()
