"""Saídas da avaliação: metrics.json, mapas de erro PNG e report.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg precisa vir antes do pyplot)


def json_safe(obj: Any) -> Any:
    """Converte recursivamente para tipos serializáveis em JSON.

    - chaves iniciadas por "_" (arrays internos, ex.: mapas de erro) são removidas;
    - escalares numpy viram float/int nativos;
    - floats não-finitos viram strings ("inf", "-inf", "nan") — JSON não os aceita.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else str(f)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_metrics_json(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / "metrics.json"
    path.write_text(json.dumps(json_safe(metrics), indent=2, ensure_ascii=False))
    return path


def save_error_map(error_map: np.ndarray, path: str | Path, title: str = "") -> Path:
    """Salva um mapa de erro como PNG com colormap e colorbar.

    Escala de cor: [0, p99 dos valores finitos] — robusta a outliers.
    NaN (pixels inválidos) aparece na cor de fundo do matplotlib.
    """
    data = np.asarray(error_map, dtype=np.float64)
    finite = data[np.isfinite(data)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    h, w = data.shape
    fig, ax = plt.subplots(figsize=(6.0, max(2.0, 6.0 * h / w)))
    im = ax.imshow(data, cmap="inferno", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return Path(path)


def _fmt(value: Any, nd: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{nd}f}" if np.isfinite(value) else str(value)
    return str(value)


def _height_section(title: str, block: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    if block["status"] != "ok":
        lines += [f"**{block['status']}**", ""]
        return lines
    lines += [
        "| métrica | valor |",
        "|---|---|",
        f"| RMSE (pós-fit afim) | {_fmt(block['rmse_affine'])} |",
        f"| MAE (pós-fit afim) | {_fmt(block['mae_affine'])} |",
        f"| a, b (gt ≈ a·est + b) | {_fmt(block['a'])}, {_fmt(block['b'])} |",
        f"| Pearson r | {_fmt(block['pearson_r'])} |",
        f"| std do GT (contexto p/ RMSE) | {_fmt(block['gt_std'])} |",
        f"| fração de pixels válidos | {_fmt(block['valid_fraction'])} |",
    ]
    if block.get("low_validity"):
        lines.append("| **ALERTA** | menos de 1% de pixels válidos |")
    lines.append("")
    return lines


def write_report_md(
    metrics: dict[str, Any], error_map_files: dict[str, str], out_dir: str | Path
) -> Path:
    """Gera report.md consolidado. `error_map_files` mapeia chave da etapa →
    nome do arquivo PNG (relativo a out_dir) para embutir as imagens."""
    meta = metrics.get("meta", {})
    lines: list[str] = [
        "# Relatório de avaliação automática",
        "",
        f"- **results_dir:** `{meta.get('results_dir', '?')}`",
        f"- **data_dir:** `{meta.get('data_dir', '—')}`",
        f"- **quando:** {meta.get('timestamp', '?')}",
        f"- **versão do pacote:** {meta.get('package_version', '?')}",
        "",
    ]

    if "multifocus_depth" in metrics:
        lines += _height_section(
            "Multifocus — profundidade (zMos vs hAvg)", metrics["multifocus_depth"]
        )
    if "multifocus_depth_hdev_masked" in metrics:
        lines += _height_section(
            "Multifocus — profundidade (com máscara hDev)",
            metrics["multifocus_depth_hdev_masked"],
        )

    fs = metrics.get("focus_selection")
    if fs is not None:
        lines += ["## Multifocus — seleção de foco (zMos vs argmax shrp)", ""]
        if fs["status"] != "ok":
            lines += [f"**{fs['status']}**", ""]
        else:
            lines += [
                "| métrica | valor |",
                "|---|---|",
                f"| erro mediano (frames) | {_fmt(fs['median_err_frames'])} |",
                f"| erro médio (frames) | {_fmt(fs['mean_err_frames'])} |",
                f"| p90 (frames) | {_fmt(fs['p90_err_frames'])} |",
                f"| acerto exato (≤0.5 frame) | {_fmt(fs['exact_match_pct'], 1)}% |",
                f"| dentro de ±1 frame | {_fmt(fs['within_1_frame_pct'], 1)}% |",
                f"| fração de pixels válidos | {_fmt(fs['valid_fraction'])} |",
                "",
            ]

    mo = metrics.get("mosaics")
    if mo is not None:
        lines += ["## Multifocus — mosaicos por luz (sMos vs sVal)", ""]
        if mo["status"] != "ok":
            lines += [f"**{mo['status']}**", ""]
        else:
            lines += ["| luz | PSNR (dB) | SSIM |", "|---|---|---|"]
            for light, v in sorted(mo["per_light"].items()):
                if v["status"] == "ok":
                    lines.append(f"| {light} | {_fmt(v['psnr'], 2)} | {_fmt(v['ssim'])} |")
                else:
                    lines.append(f"| {light} | {v['status']} | — |")
            lines += [
                f"| **média** | {_fmt(mo['psnr_mean'], 2)} | {_fmt(mo['ssim_mean'])} |",
                f"| **mínimo** | {_fmt(mo['psnr_min'], 2)} | {_fmt(mo['ssim_min'])} |",
                "",
                f"Pior luz (menor SSIM): **{mo['worst_light']}** — degrada a entrada "
                "do fotométrico.",
                "",
            ]

    ph = metrics.get("photometric_normals")
    if ph is not None:
        lines += ["## Fotométrico — normais (normal_map vs sNrm)", ""]
        if ph["status"] != "ok":
            lines += [f"**{ph['status']}**", ""]
        else:
            lines += [
                "| orientação do GT | erro médio (°) | mediano (°) | p95 (°) |",
                "|---|---|---|---|",
            ]
            for label in ("y_as_is", "y_flipped"):
                v = ph[label]
                marker = " **(vencedora)**" if label == ph["winning_orientation"] else ""
                lines.append(
                    f"| {label}{marker} | {_fmt(v['mean_deg'], 2)} | "
                    f"{_fmt(v['median_deg'], 2)} | {_fmt(v['p95_deg'], 2)} |"
                )
            lines += [
                "",
                f"**Veredito CONV-2:** a orientação `{ph['winning_orientation']}` do GT "
                "casa melhor com as normais estimadas — use-a para decidir "
                "`photometric.flip_lights_y` deste dataset.",
                "",
            ]

    if "integration_height" in metrics:
        lines += _height_section(
            "Integração — altura final (height_map vs hAvg)", metrics["integration_height"]
        )
    if "integration_height_hdev_masked" in metrics:
        lines += _height_section(
            "Integração — altura final (com máscara hDev)",
            metrics["integration_height_hdev_masked"],
        )

    hg = metrics.get("hybrid_gain")
    if hg is not None:
        lines += ["## Síntese — ganho do híbrido", ""]
        if hg["status"] != "ok":
            lines += [f"**{hg['status']}**", ""]
        else:
            lines += [
                "| | multifocus sozinho | resultado final |",
                "|---|---|---|",
                f"| RMSE afim | {_fmt(hg['rmse_multifocus'])} | {_fmt(hg['rmse_final'])} |",
                f"| Pearson r | {_fmt(hg['pearson_multifocus'])} | "
                f"{_fmt(hg['pearson_final'])} |",
                "",
                f"**Ganho** (RMSE_multifocus / RMSE_final): **{_fmt(hg['gain'], 2)}** "
                "(> 1 = a combinação melhorou; máscara comum aos dois mapas).",
                "",
            ]

    if error_map_files:
        lines += ["## Mapas de erro", ""]
        for stage, fname in sorted(error_map_files.items()):
            lines += [f"### {stage}", "", f"![{stage}]({fname})", ""]

    path = Path(out_dir) / "report.md"
    path.write_text("\n".join(lines))
    return path
