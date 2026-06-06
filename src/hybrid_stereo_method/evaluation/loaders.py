"""Descoberta e carregamento de artefatos do pipeline e ground truth.

Convenções de caminho (ver spec 2026-06-06-automated-evaluation-design.md):
- GT canônico em <results_dir>/sharp (copiado pelo pipeline) ou <data_dir>/sharp;
- artefatos por etapa em multifocus_stereo/, photometric_stereo/, integration/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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

    Codificação: n = (v/255)*2 - 1 por canal RGB = (nx, ny, nz); cv2 lê BGR,
    então os canais são reordenados. Pixels válidos têm norma decodificada ≈ 1;
    fundo (ex.: preto → (-1,-1,-1), norma 1.73) cai fora da janela
    |norma - 1| <= norm_tolerance e vira NaN na saída.
    """
    img = _read_png(Path(sharp_dir) / "sNrm.png", "sNrm.png (normais GT)")
    if img.ndim != 3 or img.shape[-1] < 3:
        raise MissingArtifactError(f"sNrm.png não é uma imagem de 3 canais: shape {img.shape}")
    rgb = img[..., 2::-1].astype(np.float64)  # BGR(A) → RGB
    n = rgb / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1)
    foreground = np.abs(norm - 1.0) <= norm_tolerance
    safe_norm = np.where(norm == 0.0, 1.0, norm)
    n_unit = np.where(foreground[..., None], n / safe_norm[..., None], np.nan)
    return n_unit, foreground
