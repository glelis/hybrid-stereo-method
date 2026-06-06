"""Descoberta e carregamento de artefatos do pipeline e ground truth.

Convenções de caminho (ver spec 2026-06-06-automated-evaluation-design.md):
- GT canônico em <results_dir>/sharp (copiado pelo pipeline) ou <data_dir>/sharp;
- artefatos por etapa em multifocus_stereo/, photometric_stereo/, integration/.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from hybrid_stereo_method.infrastructure.io.image_io import (
    read_fni_to_image_array,
    read_yaml_parameters,
)


class MissingArtifactError(FileNotFoundError):
    """Artefato ou ground truth esperado não encontrado/ilegível."""


def _read_png(path: Path, what: str) -> np.ndarray:
    """cv2.imread IMREAD_UNCHANGED (preserva uint16; cores vêm em BGR)."""
    if not path.exists():
        raise MissingArtifactError(f"{what} não encontrado: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise MissingArtifactError(f"{what} ilegível: {path}")
    return img


def find_sharp_dir(results_dir: str | Path, data_dir: str | Path | None) -> Path:
    """GT canônico: <results_dir>/sharp (copiado pelo pipeline) ou <data_dir>/sharp."""
    for base in (results_dir, data_dir):
        if base is not None and (Path(base) / "sharp").is_dir():
            return Path(base) / "sharp"
    raise MissingArtifactError(
        f"pasta sharp/ (ground truth) não encontrada em {results_dir} nem em {data_dir}"
    )


def load_height_gt(sharp_dir: str | Path) -> np.ndarray:
    """hAvg.png (uint16 ou uint8) → float64 (H, W)."""
    img = _read_png(Path(sharp_dir) / "hAvg.png", "hAvg.png (altura GT)")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float64)


def load_hdev_mask(sharp_dir: str | Path, threshold: float) -> np.ndarray | None:
    """Máscara de confiabilidade do GT a partir de hDev.png (None se ausente).

    True = pixel confiável. hDev é normalizado ao próprio range [0, 1];
    pixels com valor normalizado > threshold são excluídos.
    """
    path = Path(sharp_dir) / "hDev.png"
    if not path.exists():
        return None
    dev = _read_png(path, "hDev.png").astype(np.float64)
    if dev.ndim == 3:
        dev = dev[..., 0]
    span = dev.max() - dev.min()
    if span == 0.0:
        return np.ones(dev.shape, dtype=bool)
    return (dev - dev.min()) / span <= threshold


def load_normals_gt(
    sharp_dir: str | Path, norm_tolerance: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Decodifica sNrm.png → (normais unitárias (H, W, 3) float64, máscara de frente).

    Codificação: n = (v/max_do_dtype)*2 - 1 por canal RGB = (nx, ny, nz) — suporta PNG de
    8 ou 16 bits (o GT real é 16 bits). cv2 lê BGR, então os canais são reordenados.
    Pixels válidos têm norma decodificada ≈ 1; fundo (ex.: preto → (-1,-1,-1), norma 1.73)
    cai fora da janela |norma - 1| <= norm_tolerance e vira NaN na saída.
    """
    img = _read_png(Path(sharp_dir) / "sNrm.png", "sNrm.png (normais GT)")
    if img.ndim != 3 or img.shape[-1] < 3:
        raise MissingArtifactError(f"sNrm.png não é uma imagem de 3 canais: shape {img.shape}")
    rgb = img[..., 2::-1].astype(np.float64)  # BGR(A) → RGB
    scale = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    n = rgb / scale * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1)
    foreground = np.abs(norm - 1.0) <= norm_tolerance
    safe_norm = np.where(norm == 0.0, 1.0, norm)
    n_unit = np.where(foreground[..., None], n / safe_norm[..., None], np.nan)
    return n_unit, foreground


# --- artefatos do pipeline ---------------------------------------------------

_LIGHT_RE = re.compile(r"L\d+")
_ZF_RE = re.compile(r"zf(\d+(?:\.\d+)?)")


def load_zmos(results_dir: str | Path) -> np.ndarray:
    """zMos.fni do multifocus (unidades z_foc, NaN = pixel inválido) → float64."""
    path = Path(results_dir) / "multifocus_stereo" / "average" / "zMos.fni"
    if not path.exists():
        raise MissingArtifactError(f"zMos.fni (profundidade multifocus) não encontrado: {path}")
    zmos = read_fni_to_image_array(path).astype(np.float64)
    if zmos.ndim == 3:  # simetria com load_height_gt/load_height_map (1º canal)
        zmos = zmos[..., 0]
    return zmos


def load_normal_map(results_dir: str | Path) -> np.ndarray:
    """normal_map.npy do fotométrico ((H, W, 3), pode conter NaN)."""
    path = Path(results_dir) / "photometric_stereo" / "normal_map.npy"
    if not path.exists():
        raise MissingArtifactError(f"normal_map.npy (fotométrico) não encontrado: {path}")
    return np.load(path)


def load_height_map(results_dir: str | Path) -> np.ndarray:
    """height_map.npy da integração (vertex-grid (H+1, W+1))."""
    path = Path(results_dir) / "integration" / "height_map.npy"
    if not path.exists():
        raise MissingArtifactError(f"height_map.npy (integração) não encontrado: {path}")
    return np.load(path)


def vertex_to_cell(z_vertex: np.ndarray) -> np.ndarray:
    """Vertex-grid (H+1, W+1) → cell-grid (H, W): média dos 4 vértices da célula.

    A altura integrada vive nos vértices (INT-05); o GT (hAvg.png) é cell-grid.
    """
    z = np.asarray(z_vertex, dtype=np.float64)
    return 0.25 * (z[:-1, :-1] + z[:-1, 1:] + z[1:, :-1] + z[1:, 1:])


def load_shrp_z_gt(data_dir: str | Path) -> tuple[np.ndarray, list[float]]:
    """GT de seleção de foco: z_gt = z_foc[argmax(pilha shrp)] por pixel.

    Os valores de z_foc são parseados dos NOMES das pastas zf* (ex.:
    "zf045.0000-df020.0000" → 45.0) — sem dependência de config. Usa a luz
    (pasta-mãe) com mais planos zf*/shrp.png. Retorna (z_gt (H, W) float64,
    lista ordenada dos z parseados).
    """
    by_parent: dict[Path, list[tuple[float, Path]]] = {}
    for root, _dirs, files in os.walk(data_dir):
        m = _ZF_RE.match(Path(root).name)
        if m and "shrp.png" in files:
            by_parent.setdefault(Path(root).parent, []).append(
                (float(m.group(1)), Path(root) / "shrp.png")
            )
    if not by_parent:
        raise MissingArtifactError(f"nenhuma pasta zf*/shrp.png encontrada sob {data_dir}")
    parent = max(by_parent, key=lambda p: len(by_parent[p]))
    pairs = sorted(by_parent[parent])
    z_vals = [z for z, _ in pairs]
    frames = []
    for _z, path in pairs:
        img = _read_png(path, "shrp.png").astype(np.float64)
        if img.ndim == 3:
            img = img.mean(axis=-1)
        frames.append(img)
    stack = np.stack(frames)
    z_gt = np.asarray(z_vals, dtype=np.float64)[np.argmax(stack, axis=0)]
    logging.info("GT de seleção de foco: %d planos zf de %s", len(z_vals), parent)
    return z_gt, z_vals


def find_smos_pairs(
    results_dir: str | Path, data_dir: str | Path | None
) -> list[tuple[str, Path, Path]]:
    """Pares (luz, sMos.png estimado, sVal.png GT) para as luzes presentes nos dois lados."""
    mf = Path(results_dir) / "multifocus_stereo"
    if data_dir is None or not mf.is_dir():
        return []
    gt_by_light: dict[str, Path] = {}
    for root, _dirs, files in os.walk(data_dir):
        r = Path(root)
        if r.name == "sharp" and _LIGHT_RE.fullmatch(r.parent.name) and "sVal.png" in files:
            gt_by_light[r.parent.name] = r / "sVal.png"
    pairs = []
    for d in sorted(p for p in mf.iterdir() if p.is_dir() and _LIGHT_RE.fullmatch(p.name)):
        smos = d / "sMos.png"
        if smos.exists() and d.name in gt_by_light:
            pairs.append((d.name, smos, gt_by_light[d.name]))
    return pairs


def resolve_data_dir(results_dir: str | Path, data_dir: str | Path | None) -> Path | None:
    """Pasta do dataset: argumento explícito > parameters.yaml salvo no resultado > None."""
    if data_dir is not None:
        return Path(data_dir)
    params_path = Path(results_dir) / "parameters.yaml"
    if params_path.exists():
        try:
            params = read_yaml_parameters(params_path)
            paths = params["experiment"]["paths"]
            candidate = Path(paths["input"]) / paths["data_folder"]
            if candidate.is_dir():
                return candidate
            logging.warning("data_dir do parameters.yaml não existe: %s", candidate)
        except (KeyError, TypeError, yaml.YAMLError) as exc:
            logging.warning("parameters.yaml sem experiment.paths utilizável: %s", exc)
    return None
