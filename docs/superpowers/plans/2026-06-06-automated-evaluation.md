# Automated Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feature de avaliação automatizada que compara as saídas de cada etapa do pipeline híbrido (multifocus, fotométrico, integração) contra o ground truth do dataset, conforme `docs/superpowers/specs/2026-06-06-automated-evaluation-design.md`.

**Architecture:** Novo subpacote `src/hybrid_stereo_method/evaluation/` com camadas separadas: métricas puras (`metrics.py`), I/O e descoberta de artefatos (`loaders.py`), avaliadores por etapa (`evaluators.py`), geração de saídas (`report.py`) e orquestração + CLI (`main.py`). Hook opcional em `hybrid/main.py`. Tudo TDD com dados sintéticos de `tests/synthetic_utils.py`.

**Tech Stack:** Python 3, numpy, opencv (cv2), scikit-image (PSNR/SSIM), matplotlib (mapas de erro), PyYAML, pytest. Sem dependências novas.

---

## Contexto essencial do codebase (leia antes de começar)

- **Layout da pasta de resultados** (criada por `hybrid/main.py`):
  - `sharp/` — GT copiado do dataset (`hAvg.png` uint16, `sNrm.png` uint8 RGB, `hDev.png` uint16, ...)
  - `multifocus_stereo/average/zMos.fni` — profundidade multifocus **crua** em unidades z_foc, com NaN em pixels inválidos (FNI float)
  - `multifocus_stereo/L<n>/sMos.png` — mosaico all-in-focus por luz (uint8, radiometria preservada). `<n>` pode ser `L0` ou `L000` conforme o dataset
  - `photometric_stereo/normal_map.npy` — normais (H, W, 3) float, pode conter NaN; `confidence.npy` (H, W)
  - `integration/height_map.npy` — altura final em **vertex-grid (H+1, W+1)** (INT-05)
- **Layout do dataset** (`data_dir`): `L<n>/zf<valor>-.../{sVal,shrp,...}.png` (pilha de foco + nitidez GT por plano), `L<n>/sharp/sVal.png` (all-in-focus GT por luz), `sharp/` no topo (GT canônico), `lights.npy`
- **FNI**: ler com `read_fni_to_image_array` de `infrastructure/io/image_io.py` (retorna float32; NaN preservado)
- **cv2.imread com `IMREAD_UNCHANGED`** preserva uint16, mas imagens coloridas vêm em **BGR** — decodificação do `sNrm.png` exige reordenar para RGB antes de mapear (R,G,B)→(nx,ny,nz)
- **Codificação do sNrm.png**: `n = (v/255)*2 - 1` por canal; pixels válidos têm norma ≈ 1; fundo decodifica para vetores de norma ≠ 1 (ex.: preto → (-1,-1,-1), norma 1.73) — máscara de frente = `|norma - 1| < tol`
- **`iSel.fni` é salvo NORMALIZADO** (`multifocus/main.py:180`) — nunca usá-lo para comparação; `zMos.fni` é o artefato cru equivalente
- **Convenção de normais**: `n = (-dz/dx, -dz/dy, 1)/|.|`, y cresce para baixo (frame numpy); ver `tests/synthetic_utils.py` cabeçalho
- Lint: `ruff check src/` e `ruff format src/` (line-length 100); tipos: `mypy src/` (somente `src/`)
- Testes: `pytest tests/<arquivo> -v` (pytest já configurado com `--cov`)

## File Structure

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/hybrid_stereo_method/evaluation/__init__.py` | Create | marcador de pacote |
| `src/hybrid_stereo_method/evaluation/metrics.py` | Create | métricas puras (arrays → números), sem I/O |
| `src/hybrid_stereo_method/evaluation/loaders.py` | Create | descoberta/carregamento de artefatos e GT |
| `src/hybrid_stereo_method/evaluation/evaluators.py` | Create | um avaliador por etapa (arrays → dict) |
| `src/hybrid_stereo_method/evaluation/report.py` | Create | metrics.json + mapas de erro PNG + report.md |
| `src/hybrid_stereo_method/evaluation/main.py` | Create | `run_evaluation()` + CLI |
| `tests/test_evaluation_metrics.py` | Create | testes unitários das métricas |
| `tests/test_evaluation_loaders.py` | Create | testes unitários dos loaders |
| `tests/test_evaluation_evaluators.py` | Create | testes unitários dos avaliadores |
| `tests/test_evaluation_report.py` | Create | testes unitários do relatório |
| `tests/test_evaluation_e2e.py` | Create | integração: pasta de resultados sintética → run_evaluation |
| `tests/synthetic_utils.py` | Modify | `affine_fit_rmse` passa a reexportar de `evaluation.metrics` |
| `src/hybrid_stereo_method/hybrid/main.py` | Modify | salvar `parameters.yaml`; hook `run_evaluation` |
| `tests/test_e2e_hybrid.py` | Modify | habilitar avaliação no E2E e validar saídas |
| `configs/hb_experiment.yaml` | Modify | seção `evaluation` |
| `pyproject.toml` | Modify | script `evaluate-results` |
| `CLAUDE.md` | Modify | documentar o novo entry point |

---

### Task 1: Métricas puras (`metrics.py`)

**Files:**
- Create: `src/hybrid_stereo_method/evaluation/__init__.py`
- Create: `src/hybrid_stereo_method/evaluation/metrics.py`
- Modify: `tests/synthetic_utils.py:85-97` (reexport)
- Test: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation_metrics.py
"""Testes unitários das métricas puras de avaliação (casos analíticos)."""
import numpy as np
import pytest

from hybrid_stereo_method.evaluation.metrics import (
    affine_fit_rmse,
    angular_error_deg,
    pearson_r,
    psnr,
    ssim,
)


def test_affine_fit_rmse_recovers_affine_transform():
    rng = np.random.default_rng(0)
    est = rng.uniform(0.0, 10.0, (32, 32))
    gt = 2.0 * est + 3.0
    rmse, (a, b) = affine_fit_rmse(est, gt)
    assert rmse == pytest.approx(0.0, abs=1e-9)
    assert a == pytest.approx(2.0, abs=1e-9)
    assert b == pytest.approx(3.0, abs=1e-9)


def test_affine_fit_rmse_absorbs_sign_inversion():
    est = np.linspace(0.0, 1.0, 100)
    rmse, (a, _) = affine_fit_rmse(est, -est)
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert a == pytest.approx(-1.0, abs=1e-12)


def test_pearson_r_perfect_and_constant():
    x = np.arange(50, dtype=float)
    assert pearson_r(x, 3.0 * x + 1.0) == pytest.approx(1.0)
    assert pearson_r(x, -x) == pytest.approx(-1.0)
    assert np.isnan(pearson_r(x, np.ones_like(x)))  # gt constante: r indefinido


def test_angular_error_deg_known_rotation():
    # campo (2, 2, 3): identidade vs rotação de 30° em torno de x aplicada a +z
    n = np.zeros((2, 2, 3))
    n[..., 2] = 1.0  # todos +z
    rot = n.copy()
    rot[..., 1] = np.sin(np.deg2rad(30.0))
    rot[..., 2] = np.cos(np.deg2rad(30.0))
    ae = angular_error_deg(n, rot)
    assert ae.shape == (2, 2)
    assert np.allclose(ae, 30.0, atol=1e-9)


def test_angular_error_deg_identical_is_zero_and_clips_rounding():
    rng = np.random.default_rng(1)
    n = rng.normal(size=(8, 8, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ae = angular_error_deg(n, n)
    assert np.allclose(ae, 0.0, atol=1e-6)  # arccos clampado: sem NaN por arredondamento


def test_angular_error_deg_nan_propagates():
    n = np.zeros((2, 2, 3))
    n[..., 2] = 1.0
    bad = n.copy()
    bad[0, 0, :] = np.nan
    ae = angular_error_deg(bad, n)
    assert np.isnan(ae[0, 0])
    assert ae[1, 1] == pytest.approx(0.0, abs=1e-9)


def test_psnr_identical_is_inf_and_noisy_is_finite():
    img = (np.random.default_rng(2).uniform(0, 255, (32, 32))).astype(np.uint8)
    assert np.isinf(psnr(img, img))
    noisy = np.clip(img.astype(int) + 10, 0, 255).astype(np.uint8)
    assert 20.0 < psnr(noisy, img) < 40.0


def test_ssim_identical_is_one_color_and_gray():
    gray = (np.random.default_rng(3).uniform(0, 255, (32, 32))).astype(np.uint8)
    color = np.stack([gray] * 3, axis=-1)
    assert ssim(gray, gray) == pytest.approx(1.0)
    assert ssim(color, color) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hybrid_stereo_method.evaluation'`

- [ ] **Step 3: Write the implementation**

```python
# src/hybrid_stereo_method/evaluation/__init__.py
"""Avaliação automatizada dos resultados do pipeline híbrido contra ground truth."""
```

```python
# src/hybrid_stereo_method/evaluation/metrics.py
"""Métricas puras de avaliação (arrays → números). Sem I/O.

Origem única das métricas usadas pela avaliação automatizada e pelos testes
(`tests/synthetic_utils.py` reexporta `affine_fit_rmse` daqui).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def affine_fit_rmse(est: np.ndarray, gt: np.ndarray) -> tuple[float, tuple[float, float]]:
    """RMSE de gt vs (a*est + b) com a, b ótimos por mínimos quadrados.

    Atenção: o fit afim absorve escala global, offset E INVERSÃO DE SINAL —
    use-o para medir forma; convenções de sinal são decididas por testes
    dedicados (ver auditoria CONV-1). Retorna (rmse, (a, b)).
    """
    est_flat = np.asarray(est, dtype=np.float64).ravel()
    gt_flat = np.asarray(gt, dtype=np.float64).ravel()
    a = np.stack([est_flat, np.ones_like(est_flat)], axis=1)
    coef, *_ = np.linalg.lstsq(a, gt_flat, rcond=None)
    rmse = float(np.sqrt(np.mean((a @ coef - gt_flat) ** 2)))
    return rmse, (float(coef[0]), float(coef[1]))


def pearson_r(est: np.ndarray, gt: np.ndarray) -> float:
    """Correlação de Pearson; NaN se alguma das séries for constante."""
    est_flat = np.asarray(est, dtype=np.float64).ravel()
    gt_flat = np.asarray(gt, dtype=np.float64).ravel()
    if est_flat.std() == 0.0 or gt_flat.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(est_flat, gt_flat)[0, 1])


def angular_error_deg(n_est: np.ndarray, n_gt: np.ndarray) -> np.ndarray:
    """Erro angular por pixel, em graus, entre campos de normais (H, W, 3).

    As entradas são renormalizadas; o produto interno é clampado a [-1, 1]
    para não gerar NaN por arredondamento. NaN nas entradas propaga para o
    pixel correspondente do resultado.
    """
    a = np.asarray(n_est, dtype=np.float64)
    b = np.asarray(n_gt, dtype=np.float64)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = np.sum(a * b, axis=-1) / (norm_a * norm_b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def psnr(est: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    """PSNR em dB (inf para imagens idênticas)."""
    return float(peak_signal_noise_ratio(gt, est, data_range=data_range))


def ssim(est: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    """SSIM em [-1, 1]; imagens coloridas usam channel_axis=-1."""
    kwargs: dict = {"channel_axis": -1} if est.ndim == 3 else {}
    return float(structural_similarity(gt, est, data_range=data_range, **kwargs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_metrics.py -v`
Expected: 8 PASS. (`psnr` de imagens idênticas emite um warning do skimage — esperado.)

- [ ] **Step 5: Reexport em `tests/synthetic_utils.py` (DRY)**

Em `tests/synthetic_utils.py`, remova a função `affine_fit_rmse` (linhas 85–97) e adicione no bloco de imports do topo (logo após `import numpy as np`):

```python
# affine_fit_rmse foi promovida ao pacote (origem única); reexport para os
# testes existentes que importam daqui.
from hybrid_stereo_method.evaluation.metrics import affine_fit_rmse  # noqa: F401
```

- [ ] **Step 6: Run the full test suite to catch regressions**

Run: `pytest tests/ -x -q -m "not slow"`
Expected: PASS (nenhum teste existente quebra; os que importam `affine_fit_rmse` de `synthetic_utils` continuam funcionando)

- [ ] **Step 7: Lint, types, commit**

Run: `ruff check src/ tests/test_evaluation_metrics.py && ruff format --check src/hybrid_stereo_method/evaluation/ && mypy src/`
Expected: sem erros novos (mypy pode já reportar erros pré-existentes em outros módulos — só não introduza novos)

```bash
git add src/hybrid_stereo_method/evaluation/ tests/test_evaluation_metrics.py tests/synthetic_utils.py
git commit -m "feat(eval): pure metrics module; affine_fit_rmse promoted from test utils"
```

---

### Task 2: Loaders de ground truth (`loaders.py`, parte 1)

**Files:**
- Create: `src/hybrid_stereo_method/evaluation/loaders.py`
- Test: `tests/test_evaluation_loaders.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation_loaders.py
"""Testes dos loaders de artefatos e ground truth."""
import cv2
import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height

from hybrid_stereo_method.evaluation.loaders import (
    MissingArtifactError,
    find_sharp_dir,
    load_hdev_mask,
    load_height_gt,
    load_normals_gt,
)


@pytest.fixture
def sharp_dir(tmp_path):
    """Pasta sharp/ sintética: hAvg uint16, sNrm RGB codificado, hDev uint16."""
    z = gaussian_bump(16, amplitude=6.0)
    normals = normals_from_height(z)
    d = tmp_path / "sharp"
    d.mkdir()
    h16 = ((z - z.min()) / (z.max() - z.min()) * 65535.0).round().astype(np.uint16)
    cv2.imwrite(str(d / "hAvg.png"), h16)
    rgb = np.clip((normals + 1.0) / 2.0 * 255.0, 0, 255).round().astype(np.uint8)
    rgb[0, 0] = 0  # pixel de fundo: preto decodifica para (-1,-1,-1), norma 1.73
    cv2.imwrite(str(d / "sNrm.png"), rgb[..., ::-1])  # cv2 grava BGR
    dev = np.zeros((16, 16), np.uint16)
    dev[0, :] = 65535  # primeira linha: GT incerto
    cv2.imwrite(str(d / "hDev.png"), dev)
    return d, z, normals


def test_find_sharp_dir_prefers_results_then_data(tmp_path):
    results = tmp_path / "results"
    data = tmp_path / "data"
    (data / "sharp").mkdir(parents=True)
    results.mkdir()
    assert find_sharp_dir(results, data) == data / "sharp"
    (results / "sharp").mkdir()
    assert find_sharp_dir(results, data) == results / "sharp"
    with pytest.raises(MissingArtifactError):
        find_sharp_dir(tmp_path / "nada", None)


def test_load_height_gt_reads_uint16_as_float(sharp_dir):
    d, z, _ = sharp_dir
    gt = load_height_gt(d)
    assert gt.shape == (16, 16)
    assert gt.dtype == np.float64
    assert gt.max() == pytest.approx(65535.0)
    # forma preservada: correlação alta com o z original
    assert np.corrcoef(gt.ravel(), z.ravel())[0, 1] > 0.999


def test_load_normals_gt_decodes_and_masks_background(sharp_dir):
    d, _, normals = sharp_dir
    n, fg = load_normals_gt(d)
    assert n.shape == (16, 16, 3)
    assert not fg[0, 0]  # pixel preto (fundo) excluído
    assert fg[8, 8]
    # decodificação fiel a ~1/255 de quantização
    err = np.linalg.norm(n[fg] - normals[fg], axis=-1)
    assert err.max() < 0.02
    # normais retornadas são unitárias nos pixels de frente
    assert np.allclose(np.linalg.norm(n[fg], axis=-1), 1.0, atol=1e-9)


def test_load_hdev_mask_thresholds_normalized_range(sharp_dir):
    d, _, _ = sharp_dir
    mask = load_hdev_mask(d, threshold=0.1)
    assert mask is not None
    assert not mask[0, :].any()  # linha com hDev no máximo do range: excluída
    assert mask[1:, :].all()


def test_load_hdev_mask_returns_none_when_absent(tmp_path):
    d = tmp_path / "sharp"
    d.mkdir()
    assert load_hdev_mask(d, threshold=0.1) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` em `evaluation.loaders`

- [ ] **Step 3: Write the implementation**

```python
# src/hybrid_stereo_method/evaluation/loaders.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_loaders.py -v`
Expected: 5 PASS

- [ ] **Step 5: Lint, types, commit**

Run: `ruff check src/ tests/test_evaluation_loaders.py && mypy src/`
Expected: sem erros novos

```bash
git add src/hybrid_stereo_method/evaluation/loaders.py tests/test_evaluation_loaders.py
git commit -m "feat(eval): ground-truth loaders (hAvg uint16, sNrm decode, hDev mask)"
```

---

### Task 3: Loaders de artefatos do pipeline (`loaders.py`, parte 2)

**Files:**
- Modify: `src/hybrid_stereo_method/evaluation/loaders.py` (acrescentar funções)
- Test: `tests/test_evaluation_loaders.py` (acrescentar testes)

- [ ] **Step 1: Write the failing tests** (acrescentar ao final de `tests/test_evaluation_loaders.py`)

```python
# --- artefatos do pipeline -------------------------------------------------
from hybrid_stereo_method.evaluation.loaders import (  # noqa: E402
    find_smos_pairs,
    load_height_map,
    load_normal_map,
    load_shrp_z_gt,
    load_zmos,
    resolve_data_dir,
    vertex_to_cell,
)
from hybrid_stereo_method.infrastructure.io.image_io import (  # noqa: E402
    convert_image_array_to_fni,
)


def test_load_zmos_preserves_nan(tmp_path):
    mf = tmp_path / "multifocus_stereo" / "average"
    mf.mkdir(parents=True)
    z = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    z[0, 0] = np.nan
    convert_image_array_to_fni(z, mf / "zMos.fni")
    out = load_zmos(tmp_path)
    assert out.shape == (3, 4)
    assert np.isnan(out[0, 0])
    assert out[2, 3] == pytest.approx(11.0)
    with pytest.raises(MissingArtifactError):
        load_zmos(tmp_path / "nada")


def test_load_normal_and_height_maps(tmp_path):
    (tmp_path / "photometric_stereo").mkdir()
    (tmp_path / "integration").mkdir()
    n = np.zeros((4, 4, 3), np.float32)
    np.save(tmp_path / "photometric_stereo" / "normal_map.npy", n)
    h = np.ones((5, 5), np.float32)
    np.save(tmp_path / "integration" / "height_map.npy", h)
    assert load_normal_map(tmp_path).shape == (4, 4, 3)
    assert load_height_map(tmp_path).shape == (5, 5)
    with pytest.raises(MissingArtifactError):
        load_normal_map(tmp_path / "nada")


def test_vertex_to_cell_averages_four_corners():
    v = np.array([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
    c = vertex_to_cell(v)
    assert c.shape == (2, 2)
    assert c[0, 0] == pytest.approx((0 + 2 + 2 + 4) / 4.0)
    assert c[1, 1] == pytest.approx((4 + 6 + 6 + 8) / 4.0)


def test_load_shrp_z_gt_parses_zf_names_and_argmax(tmp_path):
    # nomes reais: zf015.0000-df020.0000; nitidez máxima no plano mais próximo
    data = tmp_path / "data"
    z_vals = [15.0, 25.0, 35.0]
    depth = np.full((4, 4), 25.0)
    depth[0, 0] = 15.0
    depth[3, 3] = 35.0
    for zv in z_vals:
        d = data / "L000" / f"zf{zv:08.4f}-df020.0000"
        d.mkdir(parents=True)
        shrp = (np.exp(-np.abs(depth - zv)) * 65535.0).astype(np.uint16)
        cv2.imwrite(str(d / "shrp.png"), shrp)
    z_gt, vals = load_shrp_z_gt(data)
    assert vals == z_vals
    assert z_gt[0, 0] == pytest.approx(15.0)
    assert z_gt[1, 1] == pytest.approx(25.0)
    assert z_gt[3, 3] == pytest.approx(35.0)
    with pytest.raises(MissingArtifactError):
        load_shrp_z_gt(tmp_path / "vazio")


def test_find_smos_pairs_matches_lights(tmp_path):
    results = tmp_path / "results"
    data = tmp_path / "data"
    img = np.zeros((4, 4, 3), np.uint8)
    for light in ("L000", "L001"):
        d = results / "multifocus_stereo" / light
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "sMos.png"), img)
        g = data / light / "sharp"
        g.mkdir(parents=True)
        cv2.imwrite(str(g / "sVal.png"), img)
    # luz sem GT correspondente: fica de fora dos pares
    extra = results / "multifocus_stereo" / "L002"
    extra.mkdir(parents=True)
    cv2.imwrite(str(extra / "sMos.png"), img)
    pairs = find_smos_pairs(results, data)
    assert [p[0] for p in pairs] == ["L000", "L001"]
    assert all(p[1].exists() and p[2].exists() for p in pairs)
    assert find_smos_pairs(results, None) == []


def test_resolve_data_dir_precedence(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    data = tmp_path / "input" / "folder"
    data.mkdir(parents=True)
    # 1) argumento explícito vence
    assert resolve_data_dir(results, data) == data
    # 2) parameters.yaml salvo pelo pipeline
    (results / "parameters.yaml").write_text(
        "experiment:\n  paths:\n"
        f"    input: '{tmp_path / 'input'}'\n"
        "    data_folder: 'folder'\n"
    )
    assert resolve_data_dir(results, None) == data
    # 3) sem nada: None
    assert resolve_data_dir(tmp_path, None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_loaders.py -v`
Expected: os 5 testes da Task 2 PASSAM; os 6 novos FALHAM com `ImportError`

- [ ] **Step 3: Write the implementation**

3a. Substitua o bloco de imports no topo de `loaders.py` por (as funções novas precisam de `logging`, `os`, `re` e dos helpers de FNI/YAML):

```python
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np

from hybrid_stereo_method.infrastructure.io.image_io import (
    read_fni_to_image_array,
    read_yaml_parameters,
)
```

3b. Acrescente ao final de `loaders.py`:

```python
# --- artefatos do pipeline ---------------------------------------------------

_LIGHT_RE = re.compile(r"L\d+")
_ZF_RE = re.compile(r"zf(\d+(?:\.\d+)?)")


def load_zmos(results_dir: str | Path) -> np.ndarray:
    """zMos.fni do multifocus (unidades z_foc, NaN = pixel inválido) → float64."""
    path = Path(results_dir) / "multifocus_stereo" / "average" / "zMos.fni"
    if not path.exists():
        raise MissingArtifactError(f"zMos.fni (profundidade multifocus) não encontrado: {path}")
    return read_fni_to_image_array(path).astype(np.float64)


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
        except (KeyError, TypeError) as exc:
            logging.warning("parameters.yaml sem experiment.paths utilizável: %s", exc)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_loaders.py -v`
Expected: 11 PASS

- [ ] **Step 5: Lint, types, commit**

Run: `ruff check src/ tests/test_evaluation_loaders.py && mypy src/`
Expected: sem erros novos

```bash
git add src/hybrid_stereo_method/evaluation/loaders.py tests/test_evaluation_loaders.py
git commit -m "feat(eval): pipeline artifact loaders (zMos, normals, height, shrp GT, sMos pairs)"
```

---

### Task 4: Avaliadores por etapa (`evaluators.py`)

**Files:**
- Create: `src/hybrid_stereo_method/evaluation/evaluators.py`
- Test: `tests/test_evaluation_evaluators.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation_evaluators.py
"""Testes dos avaliadores por etapa (arrays → dict de métricas)."""
import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height

from hybrid_stereo_method.evaluation.evaluators import (
    evaluate_focus_selection,
    evaluate_height,
    evaluate_hybrid_gain,
    evaluate_mosaics,
    evaluate_normals,
)


def test_evaluate_height_affine_invariant_and_masks_nan():
    z = gaussian_bump(32, amplitude=6.0)
    est = 2.0 * z + 3.0  # afim do GT → erro ~0
    est[0, 0] = np.nan
    out = evaluate_height(est, z)
    assert out["status"] == "ok"
    assert out["rmse_affine"] == pytest.approx(0.0, abs=1e-9)
    assert out["a"] == pytest.approx(0.5, abs=1e-9)  # fit é gt ≈ a*est + b
    assert out["pearson_r"] == pytest.approx(1.0)
    assert out["valid_fraction"] == pytest.approx(1023 / 1024)
    assert out["low_validity"] is False
    assert np.isnan(out["_error_map"][0, 0])
    assert out["_error_map"][16, 16] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_height_shape_mismatch_and_empty():
    z = np.zeros((4, 4))
    assert evaluate_height(np.zeros((5, 4)), z)["status"].startswith("skipped")
    assert evaluate_height(np.full((4, 4), np.nan), z)["status"].startswith("skipped")


def test_evaluate_height_low_validity_flag():
    z = np.random.default_rng(0).uniform(size=(20, 20))
    est = np.full_like(z, np.nan)
    est[0, 0] = z[0, 0]
    est[0, 1] = z[0, 1]
    out = evaluate_height(est, z)  # 2/400 = 0.5% < 1%
    assert out["status"] == "ok"
    assert out["low_validity"] is True


def test_evaluate_focus_selection_frames_and_percentages():
    z_vals = [10.0, 20.0, 30.0]
    z_gt = np.full((4, 4), 20.0)
    zmos = z_gt.copy()
    zmos[0, 0] = 30.0  # 1 frame de erro
    zmos[0, 1] = np.nan
    out = evaluate_focus_selection(zmos, z_gt, z_vals)
    assert out["status"] == "ok"
    assert out["step_z"] == pytest.approx(10.0)
    assert out["median_err_frames"] == pytest.approx(0.0)
    assert out["exact_match_pct"] == pytest.approx(14 / 15 * 100)
    assert out["within_1_frame_pct"] == pytest.approx(100.0)
    assert out["valid_fraction"] == pytest.approx(15 / 16)


def test_evaluate_mosaics_per_light_and_aggregates():
    rng = np.random.default_rng(1)
    gt = rng.uniform(0, 255, (16, 16, 3)).astype(np.uint8)
    noisy = np.clip(gt.astype(int) + 12, 0, 255).astype(np.uint8)
    out = evaluate_mosaics([("L000", gt, gt), ("L001", noisy, gt)])
    assert out["status"] == "ok"
    assert out["per_light"]["L000"]["ssim"] == pytest.approx(1.0)
    assert out["per_light"]["L001"]["ssim"] < 1.0
    assert out["ssim_min"] == out["per_light"]["L001"]["ssim"]
    assert out["worst_light"] == "L001"
    assert evaluate_mosaics([])["status"].startswith("skipped")


def test_evaluate_normals_dual_orientation_resolves_y_frame():
    z = gaussian_bump(24, amplitude=5.0)
    n = normals_from_height(z)
    n_gt_yup = n.copy()
    n_gt_yup[..., 1] *= -1.0  # GT num frame y-up (POV-Ray)
    fg = np.ones(z.shape, dtype=bool)
    out = evaluate_normals(n, n_gt_yup, fg)
    assert out["status"] == "ok"
    assert out["winning_orientation"] == "y_flipped"
    assert out["y_flipped"]["mean_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["y_as_is"]["mean_deg"] > 1.0
    assert out["_error_map"].shape == z.shape


def test_evaluate_normals_masks_background_and_nan():
    z = gaussian_bump(16, amplitude=5.0)
    n = normals_from_height(z)
    est = n.copy()
    est[0, 0, :] = np.nan
    fg = np.ones(z.shape, dtype=bool)
    fg[0, 1] = False
    out = evaluate_normals(est, n, fg)
    assert out["status"] == "ok"
    assert out["winning_orientation"] == "y_as_is"
    assert np.isnan(out["_error_map"][0, 0])
    assert np.isnan(out["_error_map"][0, 1])
    expected_valid = (16 * 16 - 2) / (16 * 16)
    assert out["y_as_is"]["valid_fraction"] == pytest.approx(expected_valid)


def test_evaluate_hybrid_gain_on_common_mask():
    z = gaussian_bump(32, amplitude=6.0)
    rng = np.random.default_rng(2)
    mf = z + rng.normal(0.0, 0.5, z.shape)  # multifocus ruidoso
    final = z + rng.normal(0.0, 0.05, z.shape)  # híbrido melhor
    mf[0, 0] = np.nan
    out = evaluate_hybrid_gain(mf, final, z)
    assert out["status"] == "ok"
    assert out["gain"] > 1.0
    assert out["rmse_final"] < out["rmse_multifocus"]
    assert out["valid_fraction"] == pytest.approx(1023 / 1024)
    assert evaluate_hybrid_gain(np.zeros((3, 3)), final, z)["status"].startswith("skipped")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_evaluators.py -v`
Expected: FAIL — `ModuleNotFoundError` em `evaluation.evaluators`

- [ ] **Step 3: Write the implementation**

```python
# src/hybrid_stereo_method/evaluation/evaluators.py
"""Avaliadores por etapa: recebem arrays, devolvem dict de métricas.

Contrato comum dos retornos:
- "status": "ok" ou "skipped: <motivo>";
- chaves iniciadas por "_" carregam arrays (mapas de erro) e são removidas
  pela serialização JSON em report.py;
- "valid_fraction" e "low_validity" (< 1% de pixels válidos) sempre presentes
  quando status == "ok".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_stereo_method.evaluation.metrics import (
    affine_fit_rmse,
    angular_error_deg,
    pearson_r,
    psnr,
    ssim,
)

LOW_VALIDITY_THRESHOLD = 0.01


def _skipped(reason: str) -> dict[str, Any]:
    return {"status": f"skipped: {reason}"}


def evaluate_height(
    est: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray | None = None
) -> dict[str, Any]:
    """Métricas afim-invariantes entre um mapa de altura estimado e o GT.

    est/gt em cell-grid (H, W); NaN no estimado é inválido; gt_valid é uma
    máscara opcional de confiabilidade do GT (ex.: derivada de hDev.png).
    O fit é gt ≈ a*est + b; o mapa de erro é |(a*est + b) - gt| (NaN fora
    da máscara), nas unidades do GT.
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if est.shape != gt.shape:
        return _skipped(f"shapes incompatíveis: estimado {est.shape} vs GT {gt.shape}")
    valid = np.isfinite(est) & np.isfinite(gt)
    if gt_valid is not None:
        valid &= gt_valid
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _skipped("nenhum pixel válido em comum")
    rmse, (a, b) = affine_fit_rmse(est[valid], gt[valid])
    error_map = np.where(valid, np.abs(a * est + b - gt), np.nan)
    valid_fraction = n_valid / est.size
    return {
        "status": "ok",
        "rmse_affine": rmse,
        "mae_affine": float(np.nanmean(error_map)),
        "a": a,
        "b": b,
        "pearson_r": pearson_r(est[valid], gt[valid]),
        "gt_std": float(gt[valid].std()),
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
        "_error_map": error_map,
    }


def evaluate_focus_selection(
    zmos: np.ndarray, z_gt: np.ndarray, z_vals: list[float]
) -> dict[str, Any]:
    """Erro de seleção de foco: zMos (z_foc) vs z_gt = z_foc[argmax(shrp)].

    Unidades já comensuráveis (sem fit afim). O erro em frames divide pelo
    passo focal (mediana dos deltas de z_vals). "Acerto exato" = erro <= 0.5
    frame (zMos é sub-pixel; 0.5 arredonda para o frame GT).
    """
    zmos = np.asarray(zmos, dtype=np.float64)
    z_gt = np.asarray(z_gt, dtype=np.float64)
    if zmos.shape != z_gt.shape:
        return _skipped(f"shapes incompatíveis: zMos {zmos.shape} vs z_gt {z_gt.shape}")
    step = float(np.median(np.diff(sorted(z_vals)))) if len(z_vals) > 1 else 1.0
    valid = np.isfinite(zmos) & np.isfinite(z_gt)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _skipped("nenhum pixel válido")
    error_frames = np.where(valid, np.abs(zmos - z_gt) / step, np.nan)
    e = error_frames[valid]
    valid_fraction = n_valid / zmos.size
    return {
        "status": "ok",
        "step_z": step,
        "n_frames": len(z_vals),
        "median_err_frames": float(np.median(e)),
        "mean_err_frames": float(np.mean(e)),
        "p90_err_frames": float(np.percentile(e, 90)),
        "exact_match_pct": float(np.mean(e <= 0.5) * 100.0),
        "within_1_frame_pct": float(np.mean(e <= 1.0) * 100.0),
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
        "_error_map": error_frames,
    }


def evaluate_mosaics(mosaics: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """PSNR/SSIM por luz entre sMos estimado e sVal GT, + agregados.

    `mosaics` é uma lista (luz, imagem_estimada, imagem_gt), ambas uint8 e na
    MESMA ordem de canais (BGR do cv2 nos dois lados — comparação consistente).
    """
    if not mosaics:
        return _skipped("nenhum par sMos/sVal encontrado")
    per_light: dict[str, dict[str, Any]] = {}
    for light, est, gt in mosaics:
        if est.shape != gt.shape:
            per_light[light] = {
                "status": f"skipped: shapes incompatíveis {est.shape} vs {gt.shape}"
            }
            continue
        per_light[light] = {
            "status": "ok",
            "psnr": psnr(est, gt),
            "ssim": ssim(est, gt),
        }
    oks = {k: v for k, v in per_light.items() if v["status"] == "ok"}
    if not oks:
        return {"status": "skipped: nenhum par com shapes compatíveis", "per_light": per_light}
    ssims = {k: v["ssim"] for k, v in oks.items()}
    return {
        "status": "ok",
        "per_light": per_light,
        "n_lights": len(oks),
        "psnr_mean": float(np.mean([v["psnr"] for v in oks.values()])),
        "psnr_min": float(np.min([v["psnr"] for v in oks.values()])),
        "ssim_mean": float(np.mean(list(ssims.values()))),
        "ssim_min": float(np.min(list(ssims.values()))),
        "worst_light": min(ssims, key=lambda k: ssims[k]),
    }


def evaluate_normals(
    n_est: np.ndarray, n_gt: np.ndarray, gt_foreground: np.ndarray
) -> dict[str, Any]:
    """Erro angular (graus) das normais estimadas vs GT, nas duas orientações de y.

    CONV-2: o frame de y do GT (sNrm de render y-up vs numpy y-down) é
    desconhecido a priori; o erro é computado com o GT como está ("y_as_is")
    e com ny invertido ("y_flipped"); a orientação de menor erro médio vira
    "winning_orientation" e seu mapa de erro é exposto em "_error_map".
    """
    n_est = np.asarray(n_est, dtype=np.float64)[..., :3]
    n_gt = np.asarray(n_gt, dtype=np.float64)
    if n_est.shape[:2] != n_gt.shape[:2]:
        return _skipped(
            f"shapes incompatíveis: estimado {n_est.shape[:2]} vs GT {n_gt.shape[:2]}"
        )
    results: dict[str, dict[str, Any]] = {}
    error_maps: dict[str, np.ndarray] = {}
    for label, flip in (("y_as_is", False), ("y_flipped", True)):
        gt = n_gt.copy()
        if flip:
            gt[..., 1] *= -1.0
        ae = angular_error_deg(n_est, gt)
        valid = np.isfinite(ae) & gt_foreground
        n_valid = int(valid.sum())
        if n_valid == 0:
            return _skipped("nenhum pixel válido (estimado NaN ou GT sem frente)")
        vals = ae[valid]
        valid_fraction = n_valid / ae.size
        results[label] = {
            "mean_deg": float(vals.mean()),
            "median_deg": float(np.median(vals)),
            "p95_deg": float(np.percentile(vals, 95)),
            "valid_fraction": valid_fraction,
        }
        error_maps[label] = np.where(valid, ae, np.nan)
    winner = min(results, key=lambda k: results[k]["mean_deg"])
    valid_fraction = results[winner]["valid_fraction"]
    return {
        "status": "ok",
        "winning_orientation": winner,
        "y_as_is": results["y_as_is"],
        "y_flipped": results["y_flipped"],
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
        "_error_map": error_maps[winner],
    }


def evaluate_hybrid_gain(
    zmos: np.ndarray,
    height_cell: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Ganho do híbrido: RMSE afim do multifocus vs do resultado final na
    INTERSEÇÃO das máscaras de pixels válidos (comparação justa).

    gain = rmse_multifocus / rmse_final (> 1 = a combinação melhorou).
    """
    zmos = np.asarray(zmos, dtype=np.float64)
    height_cell = np.asarray(height_cell, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if zmos.shape != gt.shape or height_cell.shape != gt.shape:
        return _skipped(
            f"shapes incompatíveis: zMos {zmos.shape}, altura {height_cell.shape}, "
            f"GT {gt.shape}"
        )
    valid = np.isfinite(zmos) & np.isfinite(height_cell) & np.isfinite(gt)
    if gt_valid is not None:
        valid &= gt_valid
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _skipped("nenhum pixel válido em comum")
    rmse_mf, _ = affine_fit_rmse(zmos[valid], gt[valid])
    rmse_final, _ = affine_fit_rmse(height_cell[valid], gt[valid])
    valid_fraction = n_valid / gt.size
    return {
        "status": "ok",
        "rmse_multifocus": rmse_mf,
        "rmse_final": rmse_final,
        "gain": rmse_mf / rmse_final if rmse_final > 0.0 else float("inf"),
        "pearson_multifocus": pearson_r(zmos[valid], gt[valid]),
        "pearson_final": pearson_r(height_cell[valid], gt[valid]),
        "valid_fraction": valid_fraction,
        "low_validity": bool(valid_fraction < LOW_VALIDITY_THRESHOLD),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_evaluators.py -v`
Expected: 8 PASS

- [ ] **Step 5: Lint, types, commit**

Run: `ruff check src/ tests/test_evaluation_evaluators.py && mypy src/`
Expected: sem erros novos

```bash
git add src/hybrid_stereo_method/evaluation/evaluators.py tests/test_evaluation_evaluators.py
git commit -m "feat(eval): per-stage evaluators (height, focus selection, mosaics, normals, gain)"
```

---

### Task 5: Relatório e serialização (`report.py`)

**Files:**
- Create: `src/hybrid_stereo_method/evaluation/report.py`
- Test: `tests/test_evaluation_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation_report.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_report.py -v`
Expected: FAIL — `ModuleNotFoundError` em `evaluation.report`

- [ ] **Step 3: Write the implementation**

```python
# src/hybrid_stereo_method/evaluation/report.py
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
        lines += _height_section("Multifocus — profundidade (zMos vs hAvg)",
                                 metrics["multifocus_depth"])
    if "multifocus_depth_hdev_masked" in metrics:
        lines += _height_section("Multifocus — profundidade (com máscara hDev)",
                                 metrics["multifocus_depth_hdev_masked"])

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
        lines += _height_section("Integração — altura final (height_map vs hAvg)",
                                 metrics["integration_height"])
    if "integration_height_hdev_masked" in metrics:
        lines += _height_section("Integração — altura final (com máscara hDev)",
                                 metrics["integration_height_hdev_masked"])

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_report.py -v`
Expected: 4 PASS

- [ ] **Step 5: Lint, types, commit**

Run: `ruff check src/ tests/test_evaluation_report.py && mypy src/`
Expected: sem erros novos (nota: o `matplotlib.use("Agg")` antes do import do pyplot exige o `# noqa: E402` já incluído)

```bash
git add src/hybrid_stereo_method/evaluation/report.py tests/test_evaluation_report.py
git commit -m "feat(eval): report outputs (metrics.json, error-map PNGs, report.md)"
```

---

### Task 6: Orquestração e CLI (`main.py`) + teste de integração

**Files:**
- Create: `src/hybrid_stereo_method/evaluation/main.py`
- Test: `tests/test_evaluation_e2e.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluation_e2e.py
"""Integração: pasta de resultados sintética completa → run_evaluation → saídas."""
import json

import cv2
import numpy as np
import pytest
from synthetic_utils import gaussian_bump, normals_from_height, texture

from hybrid_stereo_method.evaluation.main import run_evaluation
from hybrid_stereo_method.infrastructure.io.image_io import convert_image_array_to_fni

SIZE = 32


@pytest.fixture
def synthetic_run(tmp_path):
    """Resultados + dataset sintéticos coerentes (mesma superfície gaussiana).

    GT: sharp/ com hAvg uint16 e sNrm codificado; dataset com L*/sharp/sVal.png
    e pilha zf*/shrp.png. Artefatos: zMos = z (com 1 NaN), normal_map = normais
    exatas, height_map = z em vertex-grid (pad de borda).
    """
    z = gaussian_bump(SIZE, amplitude=6.0)
    normals = normals_from_height(z)
    results = tmp_path / "results"
    data = tmp_path / "data"

    sharp = results / "sharp"
    sharp.mkdir(parents=True)
    h16 = ((z - z.min()) / (z.max() - z.min()) * 65535.0).round().astype(np.uint16)
    cv2.imwrite(str(sharp / "hAvg.png"), h16)
    rgb = np.clip((normals + 1.0) / 2.0 * 255.0, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(sharp / "sNrm.png"), rgb[..., ::-1])  # BGR

    mf = results / "multifocus_stereo" / "average"
    mf.mkdir(parents=True)
    zmos = z.copy()
    zmos[0, 0] = np.nan
    convert_image_array_to_fni(zmos, mf / "zMos.fni")

    tex = (texture(SIZE) * 255.0).astype(np.uint8)
    tex3 = cv2.merge([tex, tex, tex])
    for li in range(2):
        light = f"L{li:03d}"
        d = results / "multifocus_stereo" / light
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "sMos.png"), tex3)
        g = data / light / "sharp"
        g.mkdir(parents=True)
        cv2.imwrite(str(g / "sVal.png"), tex3)  # idêntico → SSIM 1

    z_vals = [1.0, 3.0, 5.0, 7.0]
    for zv in z_vals:
        d = data / "L000" / f"zf{zv:08.4f}-df020.0000"
        d.mkdir(parents=True)
        shrp = (np.exp(-np.abs(z - zv)) * 65535.0).astype(np.uint16)
        cv2.imwrite(str(d / "shrp.png"), shrp)

    ps = results / "photometric_stereo"
    ps.mkdir()
    np.save(ps / "normal_map.npy", normals.astype(np.float32))
    np.save(ps / "confidence.npy", np.ones((SIZE, SIZE), np.float32))

    integ = results / "integration"
    integ.mkdir()
    z_vertex = np.pad(z, ((0, 1), (0, 1)), mode="edge")
    np.save(integ / "height_map.npy", z_vertex.astype(np.float32))

    return results, data, z


def test_run_evaluation_full(synthetic_run):
    results, data, z = synthetic_run
    metrics = run_evaluation(results, data_dir=data)

    out = results / "evaluation"
    assert (out / "metrics.json").exists()
    assert (out / "report.md").exists()
    loaded = json.loads((out / "metrics.json").read_text())

    for stage in (
        "multifocus_depth", "focus_selection", "mosaics",
        "photometric_normals", "integration_height", "hybrid_gain",
    ):
        assert loaded[stage]["status"] == "ok", f"{stage}: {loaded[stage]['status']}"

    # multifocus: zMos == z (a menos da escala uint16 do GT) → o resíduo do fit
    # afim é só o arredondamento da codificação (uniforme ±0.5 nível de cinza,
    # RMS ≈ 0.29) → RMSE < 1 nível de cinza
    assert loaded["multifocus_depth"]["rmse_affine"] < 1.0
    assert loaded["multifocus_depth"]["pearson_r"] > 0.999
    # seleção de foco: zmos == z → erro <= meio passo focal → 100% dentro de 1 frame
    assert loaded["focus_selection"]["within_1_frame_pct"] == pytest.approx(100.0)
    # mosaicos idênticos → SSIM 1, PSNR inf (serializado como string)
    assert loaded["mosaics"]["ssim_mean"] == pytest.approx(1.0)
    assert loaded["mosaics"]["psnr_mean"] == "inf"
    # normais exatas (a menos da quantização de 8 bits do sNrm) → erro < 1°
    assert loaded["photometric_normals"]["winning_orientation"] == "y_as_is"
    assert loaded["photometric_normals"]["y_as_is"]["mean_deg"] < 1.0
    # integração: o vertex→cell introduz meio pixel de shift + suavização
    # (o fit afim não absorve shift); num bump com |∇z| até ~0.5/px isso fica
    # bem abaixo de 25% do std do GT, mas não de 10% — threshold com folga
    ih = loaded["integration_height"]
    assert ih["rmse_affine"] < 0.25 * ih["gt_std"]
    assert ih["pearson_r"] > 0.99
    # mapas de erro salvos
    assert (out / "multifocus_depth_error.png").exists()
    assert (out / "multifocus_focus_selection_error.png").exists()
    assert (out / "photometric_angular_error.png").exists()
    assert (out / "integration_height_error.png").exists()
    # dict retornado é o mesmo conteúdo serializado
    assert metrics["multifocus_depth"]["status"] == "ok"


def test_run_evaluation_skips_missing_stage(synthetic_run, tmp_path):
    results, data, _ = synthetic_run
    import shutil

    shutil.rmtree(results / "photometric_stereo")
    metrics = run_evaluation(results, data_dir=data)
    assert metrics["photometric_normals"]["status"].startswith("skipped")
    assert metrics["multifocus_depth"]["status"] == "ok"  # demais etapas seguem


def test_run_evaluation_without_data_dir_skips_dataset_dependent(synthetic_run):
    results, _, _ = synthetic_run
    metrics = run_evaluation(results, data_dir=None)
    assert metrics["focus_selection"]["status"].startswith("skipped")
    assert metrics["mosaics"]["status"].startswith("skipped")
    assert metrics["multifocus_depth"]["status"] == "ok"  # GT local (sharp copiado)


def test_run_evaluation_no_gt_at_all(tmp_path):
    empty = tmp_path / "vazio"
    empty.mkdir()
    metrics = run_evaluation(empty, data_dir=None)
    statuses = [v["status"] for k, v in metrics.items() if k != "meta"]
    assert statuses and all(s.startswith("skipped") for s in statuses)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_e2e.py -v`
Expected: FAIL — `ModuleNotFoundError` em `evaluation.main`

- [ ] **Step 3: Write the implementation**

```python
# src/hybrid_stereo_method/evaluation/main.py
"""Avaliação automatizada dos resultados do pipeline híbrido.

Uso standalone:
    python -m hybrid_stereo_method.evaluation.main --results_dir <pasta> [--data_dir <pasta>]

Uso como hook (hybrid/main.py):
    run_evaluation(output_path, data_dir=..., config=parameters.get("evaluation"))

Cada etapa é avaliada de forma independente: artefato ou GT ausente gera
status "skipped: <motivo>" e as demais etapas seguem. O CLI retorna exit
code != 0 apenas se NENHUMA etapa pôde ser avaliada.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hybrid_stereo_method.evaluation.evaluators import (
    evaluate_focus_selection,
    evaluate_height,
    evaluate_hybrid_gain,
    evaluate_mosaics,
    evaluate_normals,
)
from hybrid_stereo_method.evaluation.loaders import (
    MissingArtifactError,
    find_sharp_dir,
    find_smos_pairs,
    load_hdev_mask,
    load_height_gt,
    load_height_map,
    load_normal_map,
    load_normals_gt,
    load_shrp_z_gt,
    load_zmos,
    resolve_data_dir,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.report import (
    save_error_map,
    write_metrics_json,
    write_report_md,
)

ERROR_MAP_FILES = {
    "multifocus_depth": "multifocus_depth_error.png",
    "focus_selection": "multifocus_focus_selection_error.png",
    "photometric_normals": "photometric_angular_error.png",
    "integration_height": "integration_height_error.png",
}


def _package_version() -> str:
    try:
        return version("hybrid-stereo-method")
    except PackageNotFoundError:
        return "unknown"


def run_evaluation(
    results_dir: str | Path,
    data_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Avalia os resultados em results_dir contra o ground truth disponível.

    Escreve metrics.json, report.md e mapas de erro em <results_dir>/evaluation/
    e retorna o dict de métricas (com os arrays "_error_map" já removidos das
    saídas serializadas, mas presentes no dict retornado).
    """
    config = config or {}
    results_dir = Path(results_dir)
    out_dir = results_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(results_dir, data_dir)

    metrics: dict[str, Any] = {
        "meta": {
            "results_dir": str(results_dir),
            "data_dir": str(data_dir) if data_dir is not None else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "package_version": _package_version(),
            "config": dict(config),
        }
    }

    # --- ground truth canônico de altura/normais (sharp/) -------------------
    gt_height: np.ndarray | None = None
    gt_valid: np.ndarray | None = None
    sharp_dir: Path | None = None
    gt_reason = ""
    try:
        sharp_dir = find_sharp_dir(results_dir, data_dir)
        gt_height = load_height_gt(sharp_dir)
        if config.get("hdev_mask", False):
            gt_valid = load_hdev_mask(sharp_dir, float(config.get("hdev_threshold", 0.1)))
            if gt_valid is None:
                logging.warning("hdev_mask habilitado, mas hDev.png ausente — sem máscara")
    except MissingArtifactError as exc:
        gt_reason = str(exc)
        logging.warning("GT de altura indisponível: %s", gt_reason)

    # --- 1. multifocus: profundidade ----------------------------------------
    zmos: np.ndarray | None = None
    try:
        zmos = load_zmos(results_dir)
    except MissingArtifactError as exc:
        metrics["multifocus_depth"] = {"status": f"skipped: {exc}"}
    if zmos is not None:
        if gt_height is None:
            metrics["multifocus_depth"] = {"status": f"skipped: {gt_reason}"}
        else:
            metrics["multifocus_depth"] = evaluate_height(zmos, gt_height)
            if gt_valid is not None:
                metrics["multifocus_depth_hdev_masked"] = evaluate_height(
                    zmos, gt_height, gt_valid
                )

    # --- 2. multifocus: seleção de foco -------------------------------------
    if zmos is None:
        metrics["focus_selection"] = {"status": "skipped: zMos.fni indisponível"}
    elif data_dir is None:
        metrics["focus_selection"] = {
            "status": "skipped: data_dir não informado (pilha shrp vive no dataset)"
        }
    else:
        try:
            z_gt, z_vals = load_shrp_z_gt(data_dir)
            metrics["focus_selection"] = evaluate_focus_selection(zmos, z_gt, z_vals)
        except MissingArtifactError as exc:
            metrics["focus_selection"] = {"status": f"skipped: {exc}"}

    # --- 3. multifocus: mosaicos por luz ------------------------------------
    if data_dir is None:
        metrics["mosaics"] = {
            "status": "skipped: data_dir não informado (sVal por luz vive no dataset)"
        }
    else:
        pairs = find_smos_pairs(results_dir, data_dir)
        loaded_pairs = []
        for light, est_path, gt_path in pairs:
            est = cv2.imread(str(est_path), cv2.IMREAD_UNCHANGED)
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if est is None or gt_img is None:
                logging.warning("mosaico ilegível para %s — pulado", light)
                continue
            loaded_pairs.append((light, est, gt_img))
        metrics["mosaics"] = evaluate_mosaics(loaded_pairs)

    # --- 4. fotométrico: normais --------------------------------------------
    try:
        n_est = load_normal_map(results_dir)
        if sharp_dir is None:
            metrics["photometric_normals"] = {
                "status": f"skipped: pasta sharp/ indisponível ({gt_reason})"
            }
        else:
            n_gt, fg = load_normals_gt(sharp_dir)
            metrics["photometric_normals"] = evaluate_normals(n_est, n_gt, fg)
    except MissingArtifactError as exc:
        metrics["photometric_normals"] = {"status": f"skipped: {exc}"}

    # --- 5. integração: altura final ----------------------------------------
    height_cell: np.ndarray | None = None
    try:
        height_vertex = load_height_map(results_dir)
        if height_vertex.ndim == 3:
            height_vertex = height_vertex[..., 0]
        height_cell = vertex_to_cell(height_vertex)
    except MissingArtifactError as exc:
        metrics["integration_height"] = {"status": f"skipped: {exc}"}
    if height_cell is not None:
        if gt_height is None:
            metrics["integration_height"] = {"status": f"skipped: {gt_reason}"}
        else:
            metrics["integration_height"] = evaluate_height(height_cell, gt_height)
            if gt_valid is not None:
                metrics["integration_height_hdev_masked"] = evaluate_height(
                    height_cell, gt_height, gt_valid
                )

    # --- 6. síntese: ganho do híbrido ----------------------------------------
    if zmos is None or height_cell is None or gt_height is None:
        metrics["hybrid_gain"] = {
            "status": "skipped: requer zMos, height_map e GT de altura simultaneamente"
        }
    else:
        metrics["hybrid_gain"] = evaluate_hybrid_gain(zmos, height_cell, gt_height, gt_valid)

    # --- saídas ---------------------------------------------------------------
    error_map_files: dict[str, str] = {}
    for stage, fname in ERROR_MAP_FILES.items():
        block = metrics.get(stage, {})
        emap = block.get("_error_map")
        if emap is not None:
            save_error_map(emap, out_dir / fname, title=stage)
            error_map_files[stage] = fname

    write_metrics_json(metrics, out_dir)
    write_report_md(metrics, error_map_files, out_dir)

    n_ok = sum(
        1 for k, v in metrics.items() if k != "meta" and v.get("status") == "ok"
    )
    logging.info(
        "Avaliação concluída: %d etapa(s) ok — saídas em %s", n_ok, out_dir
    )
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliação automatizada dos resultados do pipeline híbrido."
    )
    parser.add_argument(
        "--results_dir", required=True,
        help="Pasta de resultados de um experimento (timestamped).",
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Pasta do dataset de entrada (para GT por luz e pilha shrp). "
        "Se omitida, tenta o parameters.yaml salvo no resultado.",
    )
    parser.add_argument(
        "--hdev_mask", action="store_true",
        help="Também reporta métricas excluindo pixels de GT incerto (hDev.png).",
    )
    parser.add_argument("--hdev_threshold", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "evaluation.log"),
        ],
    )

    metrics = run_evaluation(
        args.results_dir,
        data_dir=args.data_dir,
        config={"hdev_mask": args.hdev_mask, "hdev_threshold": args.hdev_threshold},
    )
    n_ok = sum(1 for k, v in metrics.items() if k != "meta" and v.get("status") == "ok")
    if n_ok == 0:
        logging.error("Nenhuma etapa pôde ser avaliada.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation_e2e.py -v`
Expected: 4 PASS

- [ ] **Step 5: Smoke test do CLI sobre dados reais (se a pasta existir)**

Se houver algum resultado real em `data/results/hybrid_stereo/` com a estrutura nova (multifocus_stereo/ etc.), rode:

```bash
python -m hybrid_stereo_method.evaluation.main \
  --results_dir "<uma pasta de resultados>" \
  --data_dir "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/2025-03-08-stQ-melon24-amb0.00-glo0 (2).50"
```

Expected: roda sem traceback; etapas com artefatos presentes ficam "ok", as demais "skipped" com motivo claro. (Se não houver resultado real compatível, pule este passo — o teste sintético cobre o fluxo.)

- [ ] **Step 6: Lint, types, commit**

Run: `ruff check src/ tests/test_evaluation_e2e.py && mypy src/`
Expected: sem erros novos

```bash
git add src/hybrid_stereo_method/evaluation/main.py tests/test_evaluation_e2e.py
git commit -m "feat(eval): run_evaluation orchestration + CLI (--results_dir/--data_dir)"
```

---

### Task 7: Hook no pipeline, config, entry point e docs

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (salvar `parameters.yaml`; hook ao final)
- Modify: `tests/test_e2e_hybrid.py` (habilitar avaliação e validar saídas)
- Modify: `configs/hb_experiment.yaml` (seção `evaluation`)
- Modify: `pyproject.toml` (script `evaluate-results`)
- Modify: `CLAUDE.md` (documentar entry point)

- [ ] **Step 1: Write the failing test** — em `tests/test_e2e_hybrid.py`, dentro de `test_hybrid_pipeline_end_to_end`, acrescente ao dict `parameters` (após a chave `"hybrid"`):

```python
        "evaluation": {"enabled": True},
```

e ao FINAL do teste (após o `assert np.isfinite(rmse)`):

```python
    # Hook de avaliação automática (spec 2026-06-06): parameters.yaml salvo e
    # evaluation/ gerada. O dataset sintético tem sharp/hAvg.png mas não sNrm,
    # L*/sharp/sVal nem zf*/shrp.png → essas etapas devem ser "skipped" sem
    # derrubar o pipeline.
    import json

    assert (out_dirs[0] / "parameters.yaml").exists(), "config resolvido não foi salvo"
    eval_json = out_dirs[0] / "evaluation" / "metrics.json"
    assert eval_json.exists(), "hook de avaliação não gerou metrics.json"
    eval_metrics = json.loads(eval_json.read_text())
    assert eval_metrics["multifocus_depth"]["status"] == "ok"
    assert eval_metrics["integration_height"]["status"] == "ok"
    assert eval_metrics["photometric_normals"]["status"].startswith("skipped")
    assert (out_dirs[0] / "evaluation" / "report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_e2e_hybrid.py -v -m slow`
Expected: FAIL no novo assert de `parameters.yaml` (o pipeline ainda não salva o config nem chama a avaliação). Requer o binário C compilado (`cd csrc/integrate_recursive && make`); se o binário não existir o teste é skipped — compile antes.

- [ ] **Step 3: Implement the hook em `src/hybrid_stereo_method/hybrid/main.py`**

3a. Adicione `import yaml` ao bloco de imports do topo (após `import shutil`):

```python
import yaml
```

3b. Logo APÓS a linha `log_parameters(parameters)` (linha ~215), salve o config resolvido (ANTES de o dict receber arrays numpy como `filtered_images`):

```python
    # Salva o config resolvido para reuso (ex.: avaliação standalone descobre o
    # data_dir sozinha). Salvo ANTES de o dict ser mutado com arrays numpy.
    with open(os.path.join(output_path, "parameters.yaml"), "w") as f:
        yaml.safe_dump(parameters, f, sort_keys=False, allow_unicode=True)
    logging.info("Resolved parameters saved to: %s", os.path.join(output_path, "parameters.yaml"))
```

3c. Ao FINAL de `main()`, imediatamente ANTES do bloco `logging.info("=" * 60)` / `"Hybrid stereo pipeline complete!"`:

```python
    # =========================================================================
    # Step 4 (opcional): Avaliação automática contra ground truth
    # =========================================================================
    eval_config = parameters.get("evaluation") or {}
    if eval_config.get("enabled", False):
        logging.info("=" * 60)
        logging.info("STEP 4: Automated Evaluation")
        logging.info("=" * 60)
        try:
            from hybrid_stereo_method.evaluation.main import run_evaluation

            run_evaluation(
                output_path,
                data_dir=os.path.join(input_path, data_foldername),
                config=eval_config,
            )
        except Exception:
            # Falha na avaliação NUNCA derruba um experimento que já produziu
            # resultados (spec 2026-06-06): registre e siga.
            logging.exception("Avaliação automática falhou — resultados preservados")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_e2e_hybrid.py -v -m slow`
Expected: PASS (1 teste, lento)

- [ ] **Step 5: Config, entry point e docs**

5a. Em `configs/hb_experiment.yaml`, acrescente ao final do arquivo:

```yaml

evaluation:
  # Avaliação automática contra o ground truth ao final do pipeline
  # (também disponível standalone:
  #  python -m hybrid_stereo_method.evaluation.main --results_dir <pasta>)
  enabled: true

  # Também reportar métricas de altura excluindo pixels onde o próprio GT é
  # incerto (hDev.png alto). Threshold = fração do range do hDev acima da
  # qual o pixel é excluído.
  hdev_mask: false
  hdev_threshold: 0.1
```

5b. Em `pyproject.toml`, na seção `[project.scripts]`, acrescente:

```toml
evaluate-results = "hybrid_stereo_method.evaluation.main:cli"
```

5c. Em `CLAUDE.md`, na seção "Running Experiments", após o bloco de comandos existente, acrescente:

```markdown
Avaliação automatizada contra ground truth (standalone, sobre qualquer resultado já gerado;
também roda como hook ao final do pipeline híbrido se `evaluation.enabled: true` no YAML):

```bash
python -m hybrid_stereo_method.evaluation.main --results_dir <pasta_de_resultados> [--data_dir <pasta_do_dataset>]
```

Saídas em `<results_dir>/evaluation/`: `metrics.json`, `report.md` e mapas de erro PNG.
```

- [ ] **Step 6: Full verification**

Run: `pytest tests/ -q && ruff check src/ && ruff format --check src/hybrid_stereo_method/evaluation/ && mypy src/`
Expected: todos os testes PASS (incluindo os lentos, com o binário C compilado); lint/format limpos; mypy sem erros novos

- [ ] **Step 7: Commit**

```bash
git add src/hybrid_stereo_method/hybrid/main.py tests/test_e2e_hybrid.py configs/hb_experiment.yaml pyproject.toml CLAUDE.md
git commit -m "feat(eval): pipeline hook + resolved parameters.yaml + config/docs/entry point"
```

---

## Self-review (cobertura da spec)

| Requisito da spec | Task |
|---|---|
| CLI standalone `--results_dir`/`--data_dir` | 6 |
| Hook opcional + `evaluation.enabled` | 7 |
| `parameters.yaml` salvo pelo pipeline; precedência do data_dir | 3 (resolve_data_dir), 7 |
| Métricas afim-invariantes (RMSE, a, b, Pearson, MAE) | 1, 4 |
| zMos vs hAvg + mapa de erro | 3, 4, 6 |
| Seleção de foco vs pilha shrp (z parseado dos nomes zf*) | 3, 4 |
| sMos vs sVal por luz (PSNR/SSIM + pior luz) | 3, 4 |
| Normais vs sNrm decodificado, dupla orientação y (CONV-2) | 2, 4 |
| height_map vertex→cell + comparação final | 3, 4, 6 |
| Síntese do ganho na interseção das máscaras | 4, 6 |
| `metrics.json` (status/skipped, rastreabilidade, meta) | 5, 6 |
| `report.md` consolidado + mapas de erro PNG | 5, 6 |
| Skips graciosos, shapes incompatíveis, low_validity, exit code | 4, 5, 6 |
| Máscara hDev opcional (com e sem) | 2, 6, 7 |
| Reexport de affine_fit_rmse (DRY) | 1 |
| Testes unitários + integração sintética | 1–6 |
| Fora de escopo documentado (selected-pixels, devE, etc.) | — (spec) |
