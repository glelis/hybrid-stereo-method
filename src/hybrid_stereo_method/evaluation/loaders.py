"""Descoberta e carregamento de artefatos do pipeline e ground truth.

Convenções de caminho (ver spec 2026-06-06-automated-evaluation-design.md):
- GT canônico em <results_dir>/sharp (se alguém o copiar para lá), <data_dir>/sharp
  ou, no layout hybrid_stereo_new, <data_dir>/.../L###/sharp. O pipeline NÃO copia
  sharp/: a avaliação standalone precisa de --data_dir ou de um parameters.yaml
  cujo experiment.paths aponte para o dataset;
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
    """GT canônico de altura/normais (hAvg/sNrm/hDev).

    1. <results_dir>/sharp ou <data_dir>/sharp — pasta global no topo (estrutura antiga);
    2. fallback (estrutura hybrid_stereo_new): sharp/ aninhado por-luz em L###/sharp/.
       hAvg/sNrm/hDev são globais da cena (idênticos entre luzes), então usa a PRIMEIRA
       luz (menor L###). sVal.png, que varia por luz, é lido à parte por find_smos_pairs.
    """
    for base in (results_dir, data_dir):
        if base is not None and (Path(base) / "sharp").is_dir():
            return Path(base) / "sharp"
    if data_dir is not None:
        nested = sorted(
            Path(root)
            for root, _dirs, files in os.walk(data_dir)
            if Path(root).name == "sharp"
            and _LIGHT_RE.fullmatch(Path(root).parent.name)
            and "hAvg.png" in files
        )
        if nested:
            logging.info("sharp/ (GT global) via fallback per-luz — primeira luz: %s", nested[0])
            return nested[0]
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
    # C6 (investigação 2026-06-06): exige nz>0 (normal voltada à câmera). O
    # fill-value (-1/√3,-1/√3,-1/√3) tem norma 1 (passava o teste) mas nz<0 —
    # excluí-lo evita poluir o erro angular com normais traseiras espúrias.
    foreground = (np.abs(norm - 1.0) <= norm_tolerance) & (n[..., 2] > 0.0)
    safe_norm = np.where(norm == 0.0, 1.0, norm)
    n_unit = np.where(foreground[..., None], n / safe_norm[..., None], np.nan)
    return n_unit, foreground


_SIZE_RE = re.compile(r"(\d+)x(\d+)")


def scale_band_rows(sharp_dir: str | Path, image_height: int) -> int:
    """Número de linhas da faixa de escala de cinza no topo dos PNGs da libmultifok.

    A libmultifok grava imagens W×(H+38): a pasta da cena diz o tamanho nominal
    (ex.: ``0512x0384-hs01-kr07``) e as linhas excedentes, no topo, são uma
    escala de cinza que não pertence à cena. Ela decodifica como "normais"
    válidas e alturas de 0 a 65535, então precisa sair de qualquer ajuste.
    Devolve ``image_height - H`` do primeiro ancestral com padrão WxH no nome,
    ou 0 se não houver tal pasta ou a imagem não for mais alta que o nominal.
    """
    for part in Path(sharp_dir).resolve().parts[::-1]:
        m = _SIZE_RE.match(part)
        if m:
            return max(0, int(image_height) - int(m.group(2)))
    return 0


# --- artefatos do pipeline ---------------------------------------------------

# Pasta de luz: L000, ou com sufixo (stF-melon24 de 2025-02-09: L000-amb0.00-glo0.00).
_LIGHT_RE = re.compile(r"L\d+(?:-.+)?")
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
    """Pares (luz, sMos estimado, sVal.png GT) para as luzes presentes nos dois lados.

    Prefere o ``sMos.npy`` lossless (radiometria original da pilha); cai no
    ``sMos.png`` só para runs antigos — esse PNG é esticado por min-max por
    imagem e não é comparável radiometricamente ao GT (ver load_smos_pair).
    """
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
        if d.name not in gt_by_light:
            continue
        for candidate in ("sMos.npy", "sMos.png"):
            smos = d / candidate
            if smos.exists():
                pairs.append((d.name, smos, gt_by_light[d.name]))
                break
    return pairs


def _gray(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=-1) if img.ndim == 3 else img


def load_smos_pair(est_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega (sMos estimado, sVal GT) na MESMA escala 0-255 (float64, cinza).

    O GT é inteiro (8 ou 16 bits) e vai para 0-255 dividindo pelo máximo do
    dtype. O ``sMos.npy`` está na radiometria da pilha de entrada — a mesma do
    GT —, então recebe exatamente o mesmo fator. Um ``sMos.png`` legado (min-max
    por imagem) é usado como está, com warning: PSNR/SSIM nesse caso medem
    também o esticamento.
    """
    gt = _read_png(gt_path, "sVal.png")
    if not np.issubdtype(gt.dtype, np.integer):
        raise ValueError(f"sVal.png com dtype inesperado {gt.dtype}: {gt_path}")
    factor = 255.0 / float(np.iinfo(gt.dtype).max)
    gt_255 = _gray(gt.astype(np.float64)) * factor

    if est_path.suffix == ".npy":
        est = np.load(est_path).astype(np.float64)
        est_255 = _gray(est) * factor
    else:
        logging.warning(
            "%s: só há sMos.png (min-max por imagem) — métricas de mosaico não são "
            "radiometricamente comparáveis ao GT",
            est_path,
        )
        est = _read_png(est_path, "sMos.png")
        est_255 = _gray(est.astype(np.float64)) * (255.0 / float(np.iinfo(est.dtype).max))
    return est_255, gt_255


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
            # Caminhos relativos no YAML (ex.: 'data/raw/...') são relativos à
            # raiz do repo de onde o pipeline rodou, não ao CWD da avaliação:
            # tenta cada ancestral da pasta de resultados.
            if not candidate.is_absolute():
                for ancestor in Path(results_dir).resolve().parents:
                    if (ancestor / candidate).is_dir():
                        return ancestor / candidate
            logging.warning("data_dir do parameters.yaml não existe: %s", candidate)
        except (KeyError, TypeError, yaml.YAMLError) as exc:
            logging.warning("parameters.yaml sem experiment.paths utilizável: %s", exc)
    return None


def load_configured_z_foc(results_dir: str | Path) -> list[float] | None:
    """z_foc usado no run (do parameters.yaml salvo), ou None se indisponível."""
    params_path = Path(results_dir) / "parameters.yaml"
    if not params_path.exists():
        return None
    try:
        z_foc = read_yaml_parameters(params_path)["multifocus"]["parameters"]["z_foc"]
        return [float(z) for z in z_foc]
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        return None
