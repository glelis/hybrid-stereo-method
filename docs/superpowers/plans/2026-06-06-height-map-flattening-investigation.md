# Height-Map Flattening Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Localizar e corrigir as causas do mapa de altura final degradado do pipeline híbrido (anti-correlação com GT, penhasco de −455 no fundo), validando com a avaliação automática.

**Architecture:** Bisseção reversa do pipeline (spec `docs/superpowers/specs/2026-06-06-height-map-flattening-investigation-design.md`): scripts de diagnóstico read-only (F0–F4) + ablações causais (A1–A4) → checkpoint com o usuário → correções gated por veredicto → re-run de validação. Scripts vivem em `scripts/investigation/`, reusam os loaders/métricas de `hybrid_stereo_method.evaluation` e escrevem saídas em `<results_dir>/investigation/`.

**Tech Stack:** Python (numpy, cv2), loaders/métricas do pacote `hybrid_stereo_method.evaluation`, solver C `gus_integrate_recursive`, pytest para os fixes.

---

## Contexto: evidência já coletada (exploração do brainstorming)

Run de referência: `data/results/hybrid_stereo/20260606_1126_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/`

| Fato | Valor | Implicação |
|------|-------|-----------|
| Altura final: objeto em ~[18, 68]; ~7% dos pixels ≈ −455 (banda no topo, linhas 0–14 inteiras; até linha ~363 em colunas) | `height_map.npy` (423×513 vertex-grid) | "Outlier" é um penhasco regional coerente, não um spike |
| `integration_height.pearson_r = −0.20`, slope afim `a = −35` | metrics.json | Anti-correlação global → suspeita de sinal/convenção (H2) |
| `photometric_normals`: median 2.3°, mean 10.7°, p95 47°, valid 92.5% (y_as_is vence) | metrics.json | **Normais BOAS no objeto** — cauda p95 e 7.5% inválidos no fundo (H1) |
| Mosaicos PSNR 1.5–4 dB, SSIM ~0.18 | metrics.json | sMos.png saturado: `save_image(normalize=False)` faz `np.clip(img, 0, 255)` mas o stack é uint16 (0–65535) → PNG quase todo branco. PS usa floats em memória (PS-07), por isso as normais escapam |
| `sMos.fni` escrito como `sMos_light / 255.0` | hybrid/main.py:334 | Escala errada p/ dados 16-bit (valores 0–257, não 0–1) |
| `pixel_size` ausente no YAML; `use_hints: True`, `hints_weight: 0.1` | log do run | Alturas em unidades de pixel × hints em z_foc (15–125): incomensuráveis (INT-04/CONV-4) — warning já existe no código (H4) |
| `multifocus_depth.pearson_r = 0.27`, RMSE afim ≈ 17 795 (gt_std 18 468) | metrics.json | Multifocus quase não-informativo (H5) |
| `mask: False`, sem `mask.png` no dataset | log/dataset | Fundo entra na estimativa de normais e na integração (H1) |
| GT `sharp/` da raiz é do **melon14** (nome da pasta externa desatualizado) | confirmado pelo usuário | GT correto; F0 ainda verifica barato |
| Run usou `focus_measure.method: fourier`; o YAML atual diz `laplacian` | log vs configs/hb_experiment.yaml | Config foi editado após o run — re-run de validação precisa restaurar `fourier` para comparação justa |

Hipóteses H1–H6 e fases F0–F4/A1–A4: ver a spec. Mapeamento tarefas→fases indicado em cada tarefa.

## Convenções deste plano

- **Diretório de trabalho:** raiz do worktree (`.claude/worktrees/vectorized-snuggling-harp`). Todos os comandos rodam de lá.
- Caminhos do dataset/resultados são **absolutos e contêm espaços e parênteses** — sempre entre aspas no shell.
- Cada script imprime um resumo legível E grava um JSON em `<results_dir>/investigation/`.
- Tarefas marcadas **[GATE: …]** só executam se a condição (veredicto de fase anterior) for verdadeira; caso contrário marque como pulada no relatório e siga.
- Veredictos por hipótese: `confirmada` / `refutada` / `inconclusiva` — sempre com o número que sustenta.

---

### Task 0: Setup — binário C, pasta de scripts, smoke test

**Files:**
- Create: `scripts/investigation/__init__.py` (vazio)
- Create: `scripts/investigation/common.py`

- [ ] **Step 1: Compilar o solver C no worktree**

```bash
cd csrc/integrate_recursive && make && cd ../..
ls -la csrc/integrate_recursive/gus_integrate_recursive
```

Expected: binário existe e é executável. Se o make falhar, PARE e reporte — nada de F1/A* funciona sem ele.

- [ ] **Step 2: Criar `scripts/investigation/common.py`** (caminhos do run + helpers compartilhados)

```python
"""Caminhos e helpers comuns dos scripts de investigação do run 20260606_1126.

Investigação: docs/superpowers/specs/2026-06-06-height-map-flattening-investigation-design.md
Uso: python -m scripts.investigation.<script>  (a partir da raiz do repo)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_stereo_method.evaluation.evaluators import evaluate_height
from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_height_map,
    vertex_to_cell,
)

RESULTS = Path(
    "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo/"
    "20260606_1126_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50"
)
RAW = Path(
    "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/"
    "2025-03-08-stQ-melon24-amb0.00-glo0 (2).50"
)
STACK_ROOT = RAW / "stQ-melon14-amb0.00-glo0.50" / "0512x0384-hs01-kr10"
OUT = RESULTS / "investigation"

CLIFF_THRESHOLD = -200.0  # altura abaixo da qual um pixel pertence ao penhasco


def save_json(name: str, payload: dict[str, Any]) -> Path:
    """Grava payload em OUT/name (cria a pasta) e devolve o caminho."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    print(f"[saved] {path}")
    return path


def height_metrics_vs_gt(height_vertex: np.ndarray) -> dict[str, Any]:
    """Altura vertex-grid (H+1, W+1) → métricas afins vs hAvg (mesma régua da avaliação)."""
    gt = load_height_gt(find_sharp_dir(RESULTS, RAW))
    result = evaluate_height(vertex_to_cell(height_vertex), gt)
    return {k: v for k, v in result.items() if not k.startswith("_")}


def cliff_mask_cell() -> np.ndarray:
    """Máscara cell-grid (H, W) do penhasco do run original (altura < CLIFF_THRESHOLD)."""
    return vertex_to_cell(load_height_map(RESULTS)) < CLIFF_THRESHOLD


def summarize(label: str, arr: np.ndarray) -> dict[str, Any]:
    """Percentis/estatísticas de um array (ignora NaN) com rótulo, p/ JSON e console."""
    v = np.asarray(arr, dtype=np.float64)
    v = v[np.isfinite(v)]
    stats = {
        "label": label,
        "n": int(v.size),
        "min": float(v.min()) if v.size else None,
        "p01": float(np.percentile(v, 1)) if v.size else None,
        "p50": float(np.percentile(v, 50)) if v.size else None,
        "p99": float(np.percentile(v, 99)) if v.size else None,
        "max": float(v.max()) if v.size else None,
        "mean": float(v.mean()) if v.size else None,
        "std": float(v.std()) if v.size else None,
    }
    print(f"  {label}: min={stats['min']:.3g} p50={stats['p50']:.3g} "
          f"max={stats['max']:.3g} std={stats['std']:.3g}" if v.size else f"  {label}: vazio")
    return stats
```

- [ ] **Step 3: Criar os `__init__.py` vazios e smoke-test dos imports**

```bash
mkdir -p scripts/investigation
touch scripts/__init__.py scripts/investigation/__init__.py
python -c "
from scripts.investigation.common import RESULTS, RAW, STACK_ROOT, cliff_mask_cell
assert RESULTS.is_dir() and RAW.is_dir() and STACK_ROOT.is_dir()
m = cliff_mask_cell()
print('cliff pixels:', int(m.sum()), 'shape:', m.shape)
"
```

Expected: `cliff pixels: ~14000–15000, shape: (422, 512)` (na exploração: 14 770 pixels < −200).

- [ ] **Step 4: Commit**

```bash
git add scripts/investigation/
git commit -m "chore(investigation): common paths/helpers for height-map flattening investigation"
```

---

### Task 1: F0 — Sanidade do GT e da avaliação (testa H6, confirma melon14)

**Files:**
- Create: `scripts/investigation/f0_gt_sanity.py`

- [ ] **Step 1: Escrever o script**

```python
"""F0 — Sanidade do GT e da comparação da avaliação (H6).

Perguntas:
1. O GT sharp/ da raiz corresponde às imagens do stack (melon14)?
2. Shapes/dtypes de todos os artefatos GT e estimados são compatíveis?
3. zMos vs hAvg: qual o sinal natural da relação?
"""

from __future__ import annotations

import cv2
import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_height_map,
    load_normals_gt,
    load_zmos,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.metrics import pearson_r
from scripts.investigation.common import RAW, RESULTS, STACK_ROOT, save_json, summarize


def gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64).mean(axis=-1) if img.ndim == 3 else img.astype(np.float64)


def main() -> None:
    report: dict = {}

    # 1. dtypes/shapes do GT da raiz
    sharp = find_sharp_dir(RESULTS, RAW)
    print(f"sharp dir: {sharp}")
    for name in ("hAvg.png", "hDev.png", "sNrm.png", "sVal.png", "shrp.png"):
        img = cv2.imread(str(sharp / name), cv2.IMREAD_UNCHANGED)
        report[name] = (
            {"shape": list(img.shape), "dtype": str(img.dtype),
             "min": float(img.min()), "max": float(img.max())}
            if img is not None else "ausente/ilegível"
        )
        print(f"  {name}: {report[name]}")

    # 2. correspondência GT <-> imagens (melon14?)
    root_sval = gray(cv2.imread(str(sharp / "sVal.png"), cv2.IMREAD_UNCHANGED))
    l000_sval = gray(cv2.imread(str(STACK_ROOT / "L000" / "sharp" / "sVal.png"),
                                cv2.IMREAD_UNCHANGED))
    zf_imgs = sorted((STACK_ROOT / "L000").glob("zf*/*sVal.png"))
    assert zf_imgs, f"nenhum sVal.png nas pastas zf de {STACK_ROOT / 'L000'}"
    stack_mean = np.mean(
        [gray(cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) for p in zf_imgs], axis=0
    )
    report["gt_vs_images"] = {
        "r_rootSval_vs_L000sharpSval": pearson_r(root_sval, l000_sval),
        "r_rootSval_vs_L000stackMean": pearson_r(root_sval, stack_mean),
        "n_zf_frames_L000": len(zf_imgs),
    }
    print(f"  GT<->imagens: {report['gt_vs_images']}")

    # 3. shapes da comparação da avaliação + sinal zMos vs hAvg
    gt_h = load_height_gt(sharp)
    zmos = load_zmos(RESULTS)
    h_cell = vertex_to_cell(load_height_map(RESULTS))
    report["shapes"] = {"hAvg": list(gt_h.shape), "zMos": list(zmos.shape),
                       "height_cell": list(h_cell.shape)}
    valid = np.isfinite(zmos) & np.isfinite(gt_h)
    report["zmos_vs_havg"] = {"pearson": pearson_r(zmos[valid], gt_h[valid])}
    report["stats"] = [summarize("hAvg", gt_h), summarize("zMos", zmos),
                       summarize("height_cell", h_cell)]
    n_gt, fg = load_normals_gt(sharp)
    report["snrm_foreground_fraction"] = float(fg.mean())
    print(f"  zMos vs hAvg pearson: {report['zmos_vs_havg']['pearson']:.3f}; "
          f"sNrm foreground: {report['snrm_foreground_fraction']:.1%}")

    save_json("f0_gt_sanity.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e registrar veredicto**

```bash
python -m scripts.investigation.f0_gt_sanity
```

Expected/decisão:
- `r_rootSval_vs_L000stackMean` **> 0.5** → GT corresponde às imagens (melon14 confirmado); H6 segue só na parte de alinhamento. **< 0.2** → GT NÃO corresponde — PARE e reporte ao usuário (contradiz a resposta dele; a investigação muda de rumo).
- shapes todos (422, 512) → alinhamento da avaliação OK.
- Sinal de `zmos_vs_havg.pearson`: anote — se a relação física natural for inversa (depth vs height), o valor "saudável" seria fortemente negativo; 0.27 fraco é problema de qualquer forma (H5).

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/f0_gt_sanity.py
git commit -m "feat(investigation): F0 GT/eval sanity check (H6)"
```

---

### Task 2: F1 — Integrador isolado com normais do GT (testa H2)

**Files:**
- Create: `scripts/investigation/f1_integrate_gt_normals.py`

- [ ] **Step 1: Escrever o script** — integra as normais GT nas DUAS orientações de y; quem produzir pearson fortemente positivo vs hAvg revela a convenção correta e isenta (ou incrimina) o integrador.

```python
"""F1 — Integrador isolado: normais GT (sNrm) -> altura, vs hAvg (H2).

Se alguma orientação der pearson >> 0: integrador + convenção OK (H2 refutada
no integrador; a orientação vencedora é a convenção do pipeline).
Se ambas derem ruim: bug no integrador/conversão normais->slopes.
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from scripts.investigation.common import OUT, RAW, RESULTS, height_metrics_vs_gt, save_json


def main() -> None:
    n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    report: dict = {"foreground_fraction": float(fg.mean())}

    for label, flip in (("y_as_is", False), ("y_flipped", True)):
        n = n_gt.copy()
        if flip:
            n[..., 1] *= -1.0
        # NaN (fundo) -> normal vertical com peso 0: o solver ignora por peso.
        weight = fg.astype(np.float64)
        n = np.where(np.isfinite(n), n, 0.0)
        n[~fg] = (0.0, 0.0, 1.0)
        n4 = np.concatenate([n, weight[..., None]], axis=-1)

        out_dir = OUT / "f1_gt_normals" / label
        height = integrate_normals_to_height(
            normal_map=n4.astype(np.float32),
            output_dir=out_dir,
            output_prefix="gtn",
            config=IntegrateRecursiveConfig(),  # zero initial, sem hints/reference
        )
        report[label] = height_metrics_vs_gt(height)
        print(f"  {label}: pearson={report[label]['pearson_r']:+.3f} "
              f"rmse_affine={report[label]['rmse_affine']:.0f} a={report[label]['a']:+.3g}")

    save_json("f1_integrate_gt_normals.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e aplicar a tabela de decisão**

```bash
python -m scripts.investigation.f1_integrate_gt_normals
```

Tabela de decisão (registre o veredicto no JSON do relatório da Task 9):

| Resultado | Veredicto |
|-----------|-----------|
| Uma orientação com pearson ≥ +0.8 | Integrador OK; H2 refutada para o integrador; anotar orientação vencedora (compare com `winning_orientation: y_as_is` das normais estimadas — divergência indica inversão na fronteira PS→integração) |
| Ambas com \|pearson\| < 0.5 | Integrador/conversão suspeitos — abrir diagnóstico do C (`csrc/integrate_recursive`), comparar com integração simples por Frankot-Chellappa em numpy antes de mexer no C |
| Alguma com pearson ≤ −0.8 | Sinal invertido em convenção conhecida — H2 confirmada com local evidente |

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/f1_integrate_gt_normals.py
git commit -m "feat(investigation): F1 integrate GT normals, both y orientations (H2)"
```

---

### Task 3: A1 — Normais estimadas, com × sem hints (testa H4)

**Files:**
- Create: `scripts/investigation/a1_hints_ablation.py`

- [ ] **Step 1: Escrever o script** — três integrações com as normais ESTIMADAS do run: (a) sem hints; (b) com hints originais (reproduz o run); (c) sem hints e sem peso de confiança. Compara métricas e estatísticas do penhasco.

```python
"""A1 — Ablação dos hints: quanto da degradação vem dos hints incomensuráveis (H4)?

(a) sem hints           -> efeito puro das normais estimadas
(b) com hints (weight .1) -> deve reproduzir ~ o run original (pearson ~ -0.20)
(c) sem hints, sem confiança -> papel do canal de peso
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from scripts.investigation.common import (
    CLIFF_THRESHOLD,
    OUT,
    RESULTS,
    height_metrics_vs_gt,
    save_json,
    summarize,
)


def main() -> None:
    normals = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")
    confidence = np.load(RESULTS / "photometric_stereo" / "confidence.npy")
    n4 = np.concatenate([normals, confidence[..., None].astype(normals.dtype)], axis=-1)
    hints_fni = RESULTS / "integration" / "hints_vertex.fni"

    cases = {
        "no_hints": dict(normal_map=n4, hints_fni_path=None, hints_weight=0.0),
        "with_hints_w0.1": dict(normal_map=n4, hints_fni_path=hints_fni, hints_weight=0.1),
        "no_hints_no_conf": dict(normal_map=normals, hints_fni_path=None, hints_weight=0.0),
    }
    report: dict = {}
    for label, kw in cases.items():
        height = integrate_normals_to_height(
            output_dir=OUT / "a1_hints" / label,
            output_prefix="a1",
            config=IntegrateRecursiveConfig(),
            **kw,
        )
        m = height_metrics_vs_gt(height)
        m["cliff_fraction"] = float((height < CLIFF_THRESHOLD).mean())
        m["height_stats"] = summarize(label, height)
        report[label] = m
        print(f"  {label}: pearson={m['pearson_r']:+.3f} cliff={m['cliff_fraction']:.1%}")

    save_json("a1_hints_ablation.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e interpretar**

```bash
python -m scripts.investigation.a1_hints_ablation
```

Interpretação:
- `with_hints_w0.1` deve sair próximo do run original (pearson ≈ −0.20) — confirma que a reprodução é fiel (se MUITO diferente, investigue o que mais mudou antes de seguir).
- `no_hints` muito melhor que `with_hints` → **H4 confirmada** (hints incomensuráveis degradam).
- Penhasco presente mesmo em `no_hints` → o penhasco vem das normais de fundo (reforça H1), não dos hints.

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/a1_hints_ablation.py
git commit -m "feat(investigation): A1 hints ablation (H4)"
```

---

### Task 4: F2 — Análise regional das normais × penhasco (testa H1)

**Files:**
- Create: `scripts/investigation/f2_normals_regions.py`

- [ ] **Step 1: Escrever o script**

```python
"""F2 — Erro angular das normais por região (objeto x fundo) e sobreposição
com o penhasco do run original (H1).
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.evaluation.metrics import angular_error_deg
from scripts.investigation.common import RAW, RESULTS, cliff_mask_cell, save_json, summarize


def main() -> None:
    n_est = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")[..., :3]
    confidence = np.load(RESULTS / "photometric_stereo" / "confidence.npy")
    n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    ae = angular_error_deg(n_est, n_gt)  # y_as_is: orientação vencedora da avaliação
    cliff = cliff_mask_cell()
    bg = ~fg

    report = {
        "foreground_fraction": float(fg.mean()),
        "cliff_fraction": float(cliff.mean()),
        "cliff_in_background_fraction": float((cliff & bg).sum() / max(cliff.sum(), 1)),
        "angular_error_object": summarize("erro angular (objeto)", ae[fg]),
        "angular_error_background": summarize("erro angular (fundo)", ae[bg]),
        "angular_error_cliff": summarize("erro angular (penhasco)", ae[cliff]),
        "confidence_object": summarize("confiança (objeto)", confidence[fg]),
        "confidence_background": summarize("confiança (fundo)", confidence[bg]),
        "confidence_cliff": summarize("confiança (penhasco)", confidence[cliff]),
        "est_nan_fraction_background": float(np.isnan(n_est[bg]).any(axis=-1).mean()),
        "est_nan_fraction_object": float(np.isnan(n_est[fg]).any(axis=-1).mean()),
    }
    save_json("f2_normals_regions.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e interpretar**

```bash
python -m scripts.investigation.f2_normals_regions
```

Interpretação:
- `cliff_in_background_fraction` alto (> 0.8) + erro angular/confiança do fundo muito piores que do objeto → **H1 confirmada**: o penhasco é o fundo sem máscara.
- Confiança ALTA no fundo apesar de erro angular alto → o canal de peso não protege a integração — anote: a correção precisa de máscara explícita, confiança não basta.
- Nota: o GT angular no fundo é NaN (sNrm de fundo é inválido); os stats de fundo virão de poucos pixels ou NaN — use principalmente `confidence_*` e `est_nan_*` para caracterizar o fundo.

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/f2_normals_regions.py
git commit -m "feat(investigation): F2 regional normal-error vs cliff overlay (H1)"
```

---

### Task 5: F3 — Mosaicos: saturação do PNG × conteúdo real (testa H3)

**Files:**
- Create: `scripts/investigation/f3_mosaic_check.py`

- [ ] **Step 1: Escrever o script** — separa "mosaico estruturalmente errado" de "PNG de visualização saturado". O conteúdo real está em `sMos.fni` (fonte: float dividido por 255 — para dados 16-bit, multiplica-se de volta por 255 para obter unidades da fonte).

```python
"""F3 — Mosaicos: o PSNR 2-4 dB é saturação do PNG ou mosaico errado (H3)?

Compara, por luz: (a) sMos.png (uint8, possivelmente saturado) e
(b) sMos.fni * 255 (float, unidades da fonte 0-65535) contra o sVal GT
interno de cada L* (uint16), ambos reescalados para 0-255.
"""

from __future__ import annotations

import cv2
import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_smos_pairs
from hybrid_stereo_method.evaluation.metrics import pearson_r, psnr, ssim
from hybrid_stereo_method.infrastructure.io.image_io import read_fni_to_image_array
from scripts.investigation.common import RAW, RESULTS, save_json

SOURCE_MAX = 65535.0  # stack é uint16 (log do run: max_all 65535.0)


def main() -> None:
    report: dict = {"per_light": {}}
    pairs = find_smos_pairs(RESULTS, RAW)
    assert pairs, "nenhum par sMos/sVal encontrado — find_smos_pairs mudou?"
    for light, smos_png_path, sval_path in pairs:
        png = cv2.imread(str(smos_png_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
        gt16 = cv2.imread(str(sval_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
        gt255 = gt16 * (255.0 / SOURCE_MAX)
        fni = read_fni_to_image_array(smos_png_path.parent / "sMos.fni")
        est_src = np.asarray(fni, dtype=np.float64) * 255.0  # desfaz o /255 da escrita
        est255 = est_src * (255.0 / SOURCE_MAX)
        if est255.shape != gt255.shape:  # FNI pode vir (H, W, C) vs GT (H, W) ou vice-versa
            if est255.ndim == 3 and gt255.ndim == 3 and est255.shape[-1] != gt255.shape[-1]:
                est255, gt255 = est255.mean(-1), gt255.mean(-1)
            elif est255.ndim != gt255.ndim:
                est255 = est255.mean(-1) if est255.ndim == 3 else est255
                gt255 = gt255.mean(-1) if gt255.ndim == 3 else gt255
        report["per_light"][light] = {
            "png_saturated_fraction": float((png >= 255).mean()),
            "png_psnr_vs_gt": psnr(png.astype(np.float64), gt255),
            "fni_psnr_vs_gt": psnr(est255, gt255),
            "fni_ssim_vs_gt": ssim(est255, gt255),
            "fni_pearson_vs_gt": pearson_r(est255, gt255),
            "fni_src_range": [float(est_src.min()), float(est_src.max())],
        }
        r = report["per_light"][light]
        print(f"  {light}: png_sat={r['png_saturated_fraction']:.1%} "
              f"png_psnr={r['png_psnr_vs_gt']:.1f}dB fni_psnr={r['fni_psnr_vs_gt']:.1f}dB "
              f"fni_ssim={r['fni_ssim_vs_gt']:.3f}")

    save_json("f3_mosaic_check.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e interpretar**

```bash
python -m scripts.investigation.f3_mosaic_check
```

Interpretação:
- `png_saturated_fraction` ≈ 1.0 e `fni_psnr` ≥ ~15 dB / `fni_ssim` ≥ ~0.7 → **H3 refutada como bug estrutural**; o problema é o EXPORT do PNG (clip uint8 de dados 16-bit) — métrica de mosaico da avaliação era artefato. Gate da Task 10 abre.
- `fni_psnr` TAMBÉM baixo → mosaico estruturalmente ruim — H3 confirmada; investigar `mosaic()`/iSel antes de qualquer fix de export.

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/f3_mosaic_check.py
git commit -m "feat(investigation): F3 mosaic PNG-saturation vs real content (H3)"
```

---

### Task 6: F4 — Multifocus: distribuição espacial do erro (testa H5)

**Files:**
- Create: `scripts/investigation/f4_multifocus_spatial.py`

- [ ] **Step 1: Escrever o script**

```python
"""F4 — Onde o multifocus erra (H5): objeto x fundo, correlação por região."""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_height_gt,
    load_normals_gt,
    load_shrp_z_gt,
    load_zmos,
)
from hybrid_stereo_method.evaluation.metrics import pearson_r
from scripts.investigation.common import RAW, RESULTS, save_json, summarize


def main() -> None:
    zmos = load_zmos(RESULTS)
    sharp = find_sharp_dir(RESULTS, RAW)
    gt_h = load_height_gt(sharp)
    _n_gt, fg = load_normals_gt(sharp)
    z_gt, z_vals = load_shrp_z_gt(RAW)
    step = float(np.median(np.diff(sorted(z_vals))))
    err_frames = np.abs(zmos - z_gt) / step

    def region(mask: np.ndarray, label: str) -> dict:
        m = mask & np.isfinite(zmos) & np.isfinite(gt_h)
        return {
            "err_frames": summarize(f"erro de foco em frames ({label})", err_frames[m]),
            "pearson_zmos_vs_havg": pearson_r(zmos[m], gt_h[m]),
            "pearson_zmos_vs_zshrp": pearson_r(zmos[m], z_gt[m]),
            "n_pixels": int(m.sum()),
        }

    report = {
        "object": region(fg, "objeto"),
        "background": region(~fg, "fundo"),
        "all": region(np.ones_like(fg, dtype=bool), "tudo"),
    }
    for k in ("object", "background", "all"):
        print(f"  {k}: r(zMos,hAvg)={report[k]['pearson_zmos_vs_havg']:+.3f} "
              f"r(zMos,z_shrp)={report[k]['pearson_zmos_vs_zshrp']:+.3f}")
    save_json("f4_multifocus_spatial.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e interpretar**

```bash
python -m scripts.investigation.f4_multifocus_spatial
```

Interpretação:
- Erro concentrado no fundo, objeto com `pearson_zmos_vs_zshrp` alto → multifocus OK no objeto; H5 vira "limitação esperada em região sem textura" (sem fix de algoritmo; máscara resolve).
- Erro alto TAMBÉM no objeto → H5 confirmada como problema real — candidatos: focus measure (run usou `fourier` com `radius: 0.1`), pré-processamento desligado. Fix vira ajuste de config testado por re-run (não mexer no algoritmo nesta investigação).
- O sinal de `pearson_zmos_vs_havg` no objeto também responde a pergunta de convenção depth↔height para os hints (alimenta a Task 12).

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/f4_multifocus_spatial.py
git commit -m "feat(investigation): F4 multifocus spatial error split (H5)"
```

---

### Task 7: A2 — Integração com fundo mascarado (causal p/ H1)

**Files:**
- Create: `scripts/investigation/a2_integrate_masked.py`

- [ ] **Step 1: Escrever o script** — zera o peso do fundo (máscara = foreground do sNrm GT; uso DIAGNÓSTICO apenas — prova causal, não é a correção de produção).

```python
"""A2 — Se o fundo sai da integração, o penhasco some? (prova causal de H1)

Máscara derivada do foreground do sNrm GT — só para diagnóstico.
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from scripts.investigation.common import (
    CLIFF_THRESHOLD,
    OUT,
    RAW,
    RESULTS,
    height_metrics_vs_gt,
    save_json,
    summarize,
)


def main() -> None:
    normals = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")
    confidence = np.load(RESULTS / "photometric_stereo" / "confidence.npy")
    _n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))

    weight = confidence * fg.astype(confidence.dtype)  # fundo -> peso 0
    n = np.where(np.isfinite(normals), normals, 0.0)
    n4 = np.concatenate([n, weight[..., None].astype(n.dtype)], axis=-1)

    height = integrate_normals_to_height(
        normal_map=n4,
        output_dir=OUT / "a2_masked",
        output_prefix="a2",
        config=IntegrateRecursiveConfig(),  # sem hints: isola o efeito da máscara
    )
    m = height_metrics_vs_gt(height)
    m["cliff_fraction"] = float((height < CLIFF_THRESHOLD).mean())
    m["height_stats"] = summarize("altura mascarada", height)
    # métricas restritas ao objeto (fundo integrado sem dados não é informativo)
    from hybrid_stereo_method.evaluation.evaluators import evaluate_height
    from hybrid_stereo_method.evaluation.loaders import load_height_gt, vertex_to_cell

    gt = load_height_gt(find_sharp_dir(RESULTS, RAW))
    obj = evaluate_height(vertex_to_cell(height), gt, gt_valid=fg)
    m["object_only"] = {k: v for k, v in obj.items() if not k.startswith("_")}
    print(f"  mascarado: pearson(obj)={m['object_only']['pearson_r']:+.3f} "
          f"cliff={m['cliff_fraction']:.1%}")
    save_json("a2_integrate_masked.json", m)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e interpretar**

```bash
python -m scripts.investigation.a2_integrate_masked
```

Interpretação: penhasco some (cliff_fraction ≈ 0) e `object_only.pearson_r` salta para positivo forte → **H1 confirmada causalmente**; gate da Task 13 abre. Se o pearson do objeto continuar ruim mesmo mascarado, o problema principal está nas normais do objeto ou convenção (volte ao resultado de F1).

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/a2_integrate_masked.py
git commit -m "feat(investigation): A2 masked-background integration (H1 causal proof)"
```

---

### Task 8: A4 — Normais a partir dos sharp GT (contribuição dos mosaicos)

**Files:**
- Create: `scripts/investigation/a4_normals_from_sharp.py`

- [ ] **Step 1: Escrever o script** — roda o estimador WPS sobre os `sVal` sharp por luz do GT (entrada "perfeita") e compara o erro angular com o das normais do pipeline (entrada = mosaicos). Fecha o sanduíche com F1: integrador puro × estimador puro.

```python
"""A4 — Estimador WPS com entrada perfeita (sharp GT por luz):
quanto do erro das normais vem dos mosaicos vs do estimador?
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.evaluation.metrics import angular_error_deg
from hybrid_stereo_method.infrastructure.io.image_io import read_image, read_yaml_parameters
from hybrid_stereo_method.infrastructure.utils import convert_to_grayscale
from hybrid_stereo_method.photometric.main_wps import reconcile_lights
from hybrid_stereo_method.photometric.wps import estimate_normals_argmax_lstsq_robust
from scripts.investigation.common import RAW, RESULTS, STACK_ROOT, save_json, summarize


def main() -> None:
    params = read_yaml_parameters("configs/hb_experiment.yaml")
    wps_params = params["photometric"]["solver"]
    lights = np.load(RAW / "lights.npy")
    lights = reconcile_lights(
        lights, flip_y=bool(params["photometric"].get("flip_lights_y", False))
    )

    images = []
    for i in range(lights.shape[0]):
        img = read_image(STACK_ROOT / f"L{i:03d}" / "sharp" / "sVal.png")
        images.append(convert_to_grayscale(np.asarray(img, dtype=np.float64)))

    normals_sharp, _albedo, conf, _sel = estimate_normals_argmax_lstsq_robust(
        images, lights, wps_params
    )

    n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    n_mosaic = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")[..., :3]
    ae_sharp = angular_error_deg(normals_sharp[..., :3], n_gt)
    ae_mosaic = angular_error_deg(n_mosaic, n_gt)

    report = {
        "input_intensity_range": [float(min(i.min() for i in images)),
                                  float(max(i.max() for i in images))],
        "wps_thresholds": {k: wps_params[k] for k in
                           ("shadow_absolute_threshold", "saturation_threshold")
                           if k in wps_params},
        "angular_error_from_sharp_object": summarize("erro (sharp GT, objeto)", ae_sharp[fg]),
        "angular_error_from_mosaic_object": summarize("erro (mosaico, objeto)", ae_mosaic[fg]),
    }
    save_json("a4_normals_from_sharp.json", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar e interpretar**

```bash
python -m scripts.investigation.a4_normals_from_sharp
```

Interpretação:
- Erro (sharp) ≈ erro (mosaico) → mosaicos NÃO degradam as normais (consistente com F3 "saturação só no PNG").
- Erro (sharp) muito menor → a cadeia multifocus→mosaico degrada as normais de verdade; volte a F3/F4 para o elo exato.
- ATENÇÃO ao `input_intensity_range` vs `wps_thresholds`: as imagens são 16-bit (0–65535) e `saturation_threshold: 250` está em unidades 8-bit — se o estimador descartar quase tudo por "saturação", isso é um achado próprio (registre como hipótese nova H7: thresholds do solver em escala errada para dados 16-bit; o run original tem o MESMO problema, então compare maçãs com maçãs).

- [ ] **Step 3: Commit**

```bash
git add scripts/investigation/a4_normals_from_sharp.py
git commit -m "feat(investigation): A4 WPS on per-light sharp GT (mosaic contribution)"
```

---

### Task 9: CHECKPOINT — consolidar veredictos e revisar com o usuário **[GATE: parar e esperar aprovação]**

**Files:**
- Create: `docs/superpowers/investigations/2026-06-06-height-map-flattening-investigation.md` (versão "diagnóstico")

- [ ] **Step 1: Escrever a versão diagnóstico do relatório** com: tabela hipótese → veredicto (confirmada/refutada/inconclusiva) → evidência (número + JSON de origem em `<results_dir>/investigation/`), para H1–H6 (+H7 se A4 a levantou); lista de correções propostas, cada uma ligada ao seu gate (Tasks 10–13) e à evidência.

- [ ] **Step 2: Commit e apresentar ao usuário**

```bash
git add docs/superpowers/investigations/2026-06-06-height-map-flattening-investigation.md
git commit -m "docs(investigation): diagnostic verdicts for height-map flattening (H1-H6)"
```

Apresente a tabela de veredictos e a lista de fixes propostos. **NÃO prossiga para as Tasks 10–13 sem aprovação explícita do usuário sobre QUAIS fixes aplicar** (em especial a estratégia de máscara da Task 13, que tem decisão de produto: mask.png do dataset vs máscara derivada por intensidade).

---

### Task 10: Fix — export dos mosaicos dtype-aware **[GATE: F3 mostrou png_saturated_fraction ≈ 1 e fni_ssim alto]**

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (bloco que salva `sMos.png`/`sMos.fni`, ~linhas 329–336)
- Test: `tests/test_mosaic_export_scale.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
"""Export de mosaico dtype-aware: PNG/FNI escalados pelo max do dtype da fonte.

Regressão da investigação 2026-06-06 (F3): sMos.png era np.clip(float16bit, 0, 255)
-> imagem ~toda branca; sMos.fni era /255 -> range 0-257 para dados 16-bit.
"""

import cv2
import numpy as np

from hybrid_stereo_method.hybrid.main import mosaic_export_scale


def test_export_scale_uint16_source():
    assert mosaic_export_scale(np.uint16) == 65535.0


def test_export_scale_uint8_source():
    assert mosaic_export_scale(np.uint8) == 255.0


def test_export_scale_float_source_defaults_to_255():
    # fonte float: assume já em escala 0-255 (comportamento legado)
    assert mosaic_export_scale(np.float64) == 255.0


def test_16bit_mosaic_png_is_not_saturated(tmp_path):
    from hybrid_stereo_method.infrastructure.io.image_io import save_image

    smos = np.full((8, 8, 3), 32768.0)  # cinza médio em unidades 16-bit
    scale = mosaic_export_scale(np.uint16)
    save_image(str(tmp_path), "sMos.png", smos * (255.0 / scale), normalize=False)
    png = cv2.imread(str(tmp_path / "sMos.png"), cv2.IMREAD_UNCHANGED)
    assert 120 <= png.mean() <= 135  # cinza médio, não branco saturado
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
pytest tests/test_mosaic_export_scale.py -v
```

Expected: FAIL — `ImportError: cannot import name 'mosaic_export_scale'`.

- [ ] **Step 3: Implementar** — em `src/hybrid_stereo_method/hybrid/main.py`, adicionar a função (perto dos outros helpers, após `pair_mosaics_to_lights`):

```python
def mosaic_export_scale(source_dtype) -> float:
    """Valor máximo da escala radiométrica da fonte, para exportar mosaicos.

    Os mosaicos float herdam as unidades do stack de entrada (ex.: 0-65535
    para sVal.png de 16 bits). PNG (uint8) e FNI (0-1) precisam ser
    reescalados por ESTE máximo — usar 255 fixo satura dados 16-bit
    (investigação 2026-06-06, F3/H3).
    """
    dt = np.dtype(source_dtype)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 255.0
```

E no loop por luz, substituir:

```python
        save_image(output_path_multifocus, "sMos.png", sMos_light, normalize=False)
        convert_image_array_to_fni(
            sMos_light / 255.0, os.path.join(output_path_multifocus, "sMos.fni")
        )
```

por:

```python
        # F3/H3 (investigação 2026-06-06): escalar pelo max do dtype da FONTE —
        # sMos_light está nas unidades do stack (0-65535 p/ 16-bit); 255 fixo
        # saturava o PNG (avaliação de mosaico virava artefato) e deixava o
        # FNI em 0-257 em vez de 0-1.
        export_scale = mosaic_export_scale(image_stack.dtype)
        save_image(
            output_path_multifocus,
            "sMos.png",
            sMos_light * (255.0 / export_scale),
            normalize=False,
        )
        convert_image_array_to_fni(
            sMos_light / export_scale, os.path.join(output_path_multifocus, "sMos.fni")
        )
```

- [ ] **Step 4: Rodar os testes**

```bash
pytest tests/test_mosaic_export_scale.py -v && ruff check src/ tests/ && ruff format --check src/ tests/
```

Expected: 4 PASS; ruff limpo (rode `ruff format` se reclamar).

- [ ] **Step 5: Commit**

```bash
git add src/hybrid_stereo_method/hybrid/main.py tests/test_mosaic_export_scale.py
git commit -m "fix(hybrid): dtype-aware mosaic export — 16-bit stacks saturated sMos.png (F3/H3)"
```

---

### Task 11: Fix — convenção de sinal/orientação **[GATE: F1 confirmou H2]**

**Files:**
- Test: `tests/test_integration_orientation.py`
- Modify: o local exato depende do achado de F1 — candidatos, em ordem: fronteira PS→integração em `src/hybrid_stereo_method/hybrid/main.py` (sinal de ny ao montar `normal_map`), `photometric.flip_lights_y` no YAML, conversão normais→slopes no C (`csrc/integrate_recursive`). F1 + a comparação com `winning_orientation` das normais estimadas apontam qual.

- [ ] **Step 1: Escrever o teste de regressão da convenção (falha antes do fix se H2 confirmada)** — um domo sintético cujas normais DEVEM integrar para um domo com centro mais alto que a borda:

```python
"""Convenção de orientação do integrador: domo sintético integra para domo.

Regressão da investigação 2026-06-06 (F1/H2). Constrói as normais analíticas
de z(x, y) = h0 - (r/R)^2 (domo, +z para fora da superfície, frame de imagem
y-down do numpy) e exige que a altura integrada tenha o centro ACIMA da borda
e correlação fortemente positiva com o z analítico.
"""

import numpy as np
import pytest

from hybrid_stereo_method.evaluation.loaders import vertex_to_cell
from hybrid_stereo_method.evaluation.metrics import pearson_r
from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(), reason="binário C não compilado"
)


def dome_normals(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(normais (h, w, 3) no frame de imagem y-down, z analítico (h, w))."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy, scale = (w - 1) / 2.0, (h - 1) / 2.0, 8.0 / min(h, w)
    z = -(((xx - cx) * scale) ** 2 + ((yy - cy) * scale) ** 2)
    dzdx = -2.0 * (xx - cx) * scale**2
    dzdy = -2.0 * (yy - cy) * scale**2  # y = índice de linha, crescendo para baixo
    n = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n, z


def test_dome_integrates_to_dome(tmp_path):
    normals, z_true = dome_normals(32, 32)
    height = integrate_normals_to_height(
        normal_map=normals.astype(np.float32),
        output_dir=tmp_path,
        output_prefix="dome",
        config=IntegrateRecursiveConfig(),
    )
    h_cell = vertex_to_cell(height)
    center = h_cell[12:20, 12:20].mean()
    rim = np.concatenate([h_cell[0], h_cell[-1], h_cell[:, 0], h_cell[:, -1]]).mean()
    assert center > rim, "domo integrou de cabeça para baixo (H2)"
    assert pearson_r(h_cell, z_true) > 0.95
```

- [ ] **Step 2: Rodar**

```bash
pytest tests/test_integration_orientation.py -v
```

- Se **PASSAR** com o código atual: a convenção interna está certa; o problema de F1 estava noutro lugar (ex.: GT sNrm em frame y-up — i.e. dado, não código). Commit só do teste como regressão e registre no relatório.
- Se **FALHAR**: H2 confirmada no código. Vá ao Step 3.

- [ ] **Step 3: [somente se Step 2 falhou] Aplicar o fix mínimo no local apontado por F1** — ex.: se a inversão está na fronteira PS→integração, negar ny ao montar o normal_map em `hybrid/main.py` imediatamente após o `np.load`:

```python
        normal_map = np.load(normal_map_path)
        # H2 (investigação 2026-06-06, F1): o estimador WPS produz ny no frame
        # y-up das luzes; o integrador C espera o frame de imagem y-down —
        # negar ny aqui alinha as duas convenções (regressão:
        # tests/test_integration_orientation.py).
        normal_map = normal_map.copy()
        normal_map[..., 1] *= -1.0
```

(Este snippet é o candidato mais provável; se F1 apontar outro local — lights, C — aplicar lá o equivalente mínimo e documentar no commit. NÃO aplicar em dois lugares: uma inversão, um local.)

- [ ] **Step 4: Rodar TODOS os testes**

```bash
pytest && ruff check src/ tests/
```

Expected: tudo PASS (inclusive o teste do domo).

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_orientation.py src/ csrc/
git commit -m "fix(hybrid): align y-axis convention at PS->integration boundary (F1/H2)"
```

---

### Task 12: Fix — comensurabilidade dos hints **[GATE: A1 confirmou H4 E o multifocus tem sinal utilizável no objeto (F4)]**

**Files:**
- Create: `scripts/investigation/estimate_pixel_size.py`
- Modify: `configs/hb_experiment.yaml` (descomentar/definir `pixel_size`; rever `hints_weight`)

- [ ] **Step 1: Estimar `pixel_size` empiricamente** — coeficiente afim entre a altura integrada SEM hints (unidades de pixel, da ablação A1) e o zMos (z_foc) na região de objeto:

```python
"""Estima pixel_size: fator afim entre altura sem-hints (px) e zMos (z_foc).

pixel_size = 1/a, onde zMos ~ a*height_px + b no objeto (CONV-4: height_zfoc =
height_px * pixel_size). Se F4 indicou relação INVERSA (depth = -height), o
sinal de a já captura isso — reporte e use |1/a| com a observação de sinal.
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import (
    find_sharp_dir,
    load_normals_gt,
    load_zmos,
    vertex_to_cell,
)
from hybrid_stereo_method.evaluation.metrics import affine_fit_rmse
from hybrid_stereo_method.infrastructure.io.image_io import read_fni_to_image_array
from scripts.investigation.common import OUT, RAW, RESULTS, save_json


def main() -> None:
    h_px = read_fni_to_image_array(OUT / "a1_hints" / "no_hints" / "a1-00-end-Z.fni")
    h_px = h_px[..., 0] if h_px.ndim == 3 else h_px
    h_cell = vertex_to_cell(h_px)
    zmos = load_zmos(RESULTS)
    _n, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    m = fg & np.isfinite(zmos) & np.isfinite(h_cell)
    _rmse, (a, b) = affine_fit_rmse(h_cell[m], zmos[m])
    report = {"a": a, "b": b, "pixel_size_estimate": 1.0 / a if a != 0 else None,
              "sign_note": "a<0 => relação inversa height<->depth; ver F4"}
    print(f"  zMos ~ {a:+.4g}*h_px {b:+.4g}  ->  pixel_size ~ {report['pixel_size_estimate']}")
    save_json("pixel_size_estimate.json", report)


if __name__ == "__main__":
    main()
```

```bash
python -m scripts.investigation.estimate_pixel_size
```

- [ ] **Step 2: Atualizar `configs/hb_experiment.yaml`** — descomentar `pixel_size` com o valor estimado (arredondado a 2–3 algarismos significativos) e comentário apontando para `investigation/pixel_size_estimate.json`. Se F4 mostrou relação inversa (zMos é profundidade, não altura), os hints estão no sentido errado — nesse caso NÃO basta pixel_size: registre no relatório e mantenha `use_hints: false` até existir conversão depth→height (anotar como trabalho futuro), em vez de alimentar hints anti-correlacionados.

- [ ] **Step 3: Commit**

```bash
git add configs/hb_experiment.yaml scripts/investigation/estimate_pixel_size.py
git commit -m "fix(config): commensurable hints — empirical pixel_size (A1/H4, CONV-4)"
```

---

### Task 13: Fix — máscara de fundo na integração **[GATE: A2 confirmou H1; estratégia aprovada pelo usuário no checkpoint]**

**Files:**
- Create: `src/hybrid_stereo_method/hybrid/background_mask.py`
- Test: `tests/test_background_mask.py`
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (Step 3, antes de `integrate_normals_to_height`)
- Modify: `configs/hb_experiment.yaml` (nova chave `hybrid.integration.background_mask`)

A estratégia default proposta (a confirmar no checkpoint): usar `mask.png` do dataset quando existir; senão, derivar máscara por intensidade dos mosaicos (fundo = pixels escuros em TODAS as luzes). GT nunca é usado pelo método.

- [ ] **Step 1: Escrever os testes que falham**

```python
"""Máscara de fundo para a integração (H1, investigação 2026-06-06).

Fundo sem máscara => normais degeneradas entram na integração e criam o
penhasco regional. derive_background_mask: mask.png explícita > derivação por
intensidade (escuro em todas as luzes) > None (sem máscara).
"""

import cv2
import numpy as np

from hybrid_stereo_method.hybrid.background_mask import derive_background_mask


def test_explicit_mask_png_wins(tmp_path):
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    fg = derive_background_mask(tmp_path, mosaics=None, intensity_quantile=0.1)
    assert fg.dtype == bool and fg.shape == (6, 6)
    assert fg[3, 3] and not fg[0, 0]


def test_intensity_fallback_dark_everywhere_is_background(tmp_path):
    bright = np.full((6, 6), 200.0)
    bright[0, :] = 1.0  # linha 0 escura em todas as luzes -> fundo
    mosaics = [bright.copy(), bright.copy()]
    fg = derive_background_mask(tmp_path, mosaics=mosaics, intensity_quantile=0.1)
    assert not fg[0].any() and fg[3].all()


def test_no_mask_no_mosaics_returns_none(tmp_path):
    assert derive_background_mask(tmp_path, mosaics=None) is None
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
pytest tests/test_background_mask.py -v
```

Expected: FAIL — `ModuleNotFoundError: ... background_mask`.

- [ ] **Step 3: Implementar `src/hybrid_stereo_method/hybrid/background_mask.py`**

```python
"""Máscara de fundo para a integração (H1, investigação 2026-06-06).

O fundo de cena sem textura produz normais degeneradas com confiança alta;
integrá-las cria rampas/penhascos regionais que achatam o relevo do objeto.
Prioridade: mask.png do dataset > derivação por intensidade > None.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def derive_background_mask(
    data_dir: str | Path,
    mosaics: list[np.ndarray] | None,
    intensity_quantile: float = 0.1,
) -> np.ndarray | None:
    """Máscara booleana (H, W): True = objeto (integra), False = fundo (peso 0).

    1. <data_dir>/mask.png existe -> pixels > 0 são objeto.
    2. mosaics fornecidos -> fundo = pixels cujo MÁXIMO entre as luzes fica
       abaixo do quantil ``intensity_quantile`` do máximo global (escuro em
       todas as luzes não tem sinal fotométrico utilizável).
    3. Caso contrário -> None (sem máscara; comportamento atual preservado).
    """
    mask_path = Path(data_dir) / "mask.png"
    if mask_path.exists():
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img > 0
    if mosaics:
        stack = np.stack([np.asarray(m, dtype=np.float64) for m in mosaics])
        if stack.ndim == 4:  # (n, H, W, C) -> intensidade
            stack = stack.mean(axis=-1)
        peak = stack.max(axis=0)  # melhor caso por pixel entre as luzes
        return peak > np.quantile(peak, intensity_quantile)
    return None
```

- [ ] **Step 4: Rodar os testes**

```bash
pytest tests/test_background_mask.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Ligar no pipeline** — em `src/hybrid_stereo_method/hybrid/main.py`, logo após anexar a confiança ao `normal_map` (bloco PS-06/INT-03), inserir:

```python
        # H1 (investigação 2026-06-06): fundo sem máscara tem normais
        # degeneradas com confiança alta — zera o peso do fundo antes da
        # integração (A2 provou que isso elimina o penhasco regional).
        if integration_params.get("background_mask", False):
            from hybrid_stereo_method.hybrid.background_mask import derive_background_mask

            fg_mask = derive_background_mask(
                os.path.join(input_path, data_foldername),
                mosaics=[sMos_by_light[f"L{i}"] for i in range(n_lights)],
                intensity_quantile=float(
                    integration_params.get("background_mask_quantile", 0.1)
                ),
            )
            if fg_mask is not None and normal_map.shape[-1] == 4:
                normal_map[..., 3] *= fg_mask.astype(normal_map.dtype)
                logging.info(
                    "Background mask applied: %.1f%% of pixels masked out",
                    100.0 * float((~fg_mask).mean()),
                )
            else:
                logging.warning("background_mask enabled but no mask derivable")
```

Nota: `integration_params` é definido algumas linhas ABAIXO no código atual — mover a linha `integration_params = parameters.get("hybrid", {}).get("integration", {})` para antes deste bloco.

E em `configs/hb_experiment.yaml`, na seção `hybrid.integration`:

```yaml
    # Zera o peso do fundo na integração (H1, investigação 2026-06-06):
    # usa mask.png do dataset se existir; senão deriva por intensidade
    # (fundo = escuro em todas as luzes).
    background_mask: true
    background_mask_quantile: 0.1
```

- [ ] **Step 6: Rodar tudo**

```bash
pytest && ruff check src/ tests/ && mypy src/hybrid_stereo_method/hybrid/background_mask.py
```

Expected: PASS/limpo.

- [ ] **Step 7: Commit**

```bash
git add src/hybrid_stereo_method/hybrid/ tests/test_background_mask.py configs/hb_experiment.yaml
git commit -m "feat(hybrid): background mask zeroes integration weights (A2/H1)"
```

---

### Task 14: Validação — re-run completo + avaliação antes×depois

- [ ] **Step 1: Restaurar paridade de config com o run original** — o run usou `focus_measure.method: fourier`; o YAML atual diz `laplacian`. Editar `configs/hb_experiment.yaml` → `method: 'fourier'` (a investigação não mexeu no focus measure; mudá-lo agora contaminaria o antes×depois). Registrar no relatório qualquer outra diferença entre o `log` do run original e o YAML final (diff dos parâmetros logados).

- [ ] **Step 2: Re-run do pipeline híbrido**

```bash
python -m hybrid_stereo_method.multifocus.main --param_file configs/hb_experiment.yaml 2>/dev/null; \
python -m hybrid_stereo_method.hybrid.main --param_file configs/hb_experiment.yaml
```

(Só o segundo comando importa — o primeiro está aí como lembrete de que NÃO é necessário: o hybrid roda as três etapas. A avaliação roda automaticamente no final, `evaluation.enabled: true`.)

Expected: novo diretório `data/results/hybrid_stereo/<timestamp>_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/` com `evaluation/metrics.json`.

- [ ] **Step 3: Comparar antes×depois contra os critérios da spec**

```bash
python - <<'EOF'
import json
from pathlib import Path

before = json.load(open(
    "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo/"
    "20260606_1126_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/evaluation/metrics.json"))
results_root = Path("/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo")
latest = max((d for d in results_root.iterdir()
              if d.is_dir() and d.name > "20260606_1126" and (d / "evaluation/metrics.json").exists()),
             key=lambda d: d.name)
after = json.load(open(latest / "evaluation/metrics.json"))
print(f"after-run: {latest.name}")
for stage in ("multifocus_depth", "focus_selection", "mosaics",
              "photometric_normals", "integration_height", "hybrid_gain"):
    print(f"== {stage}\n  antes : {before.get(stage)}\n  depois: {after.get(stage)}")
ih = after["integration_height"]
gt_std = ih["gt_std"]
print(f"\nCRITÉRIOS spec: pearson_r={ih['pearson_r']:+.3f} (alvo >= +0.8) | "
      f"rmse/gt_std={ih['rmse_affine'] / gt_std:.2f} (alvo <= 0.50)")
EOF
```

Critérios de aceite (spec): `integration_height.pearson_r ≥ +0.8`; `rmse_affine ≤ 0.5 × gt_std`; penhasco eliminado ou confinado à região mascarada (verifique `height_map.npy` novo com o critério < −200 fora da máscara). Se algum alvo falhar: cada gap deve ser explicado por limitação intrínseca documentada com a ablação correspondente (spec) — senão, há causa corrigível restante: volte à fase F* pertinente.

- [ ] **Step 4: Commit do ajuste de config**

```bash
git add configs/hb_experiment.yaml
git commit -m "chore(config): restore fourier focus measure for fair before/after validation"
```

---

### Task 15: Relatório final + encerramento

**Files:**
- Modify: `docs/superpowers/investigations/2026-06-06-height-map-flattening-investigation.md`

- [ ] **Step 1: Completar o relatório** com: (a) tabela final hipótese→veredicto→evidência→correção (com hash do commit de cada fix); (b) tabela antes×depois de TODAS as métricas; (c) status dos critérios de aceite da spec, com explicação de gaps por limitação documentada, se houver; (d) seção "trabalho futuro" (itens anotados nas tasks: ex. conversão depth→height para hints, thresholds do WPS em escala 16-bit se H7 surgiu, ferramentas de diagnóstico reutilizáveis).

- [ ] **Step 2: Verificação final do repo**

```bash
pytest && ruff check src/ tests/ && ruff format --check src/ tests/
```

Expected: tudo PASS/limpo.

- [ ] **Step 3: Commit final**

```bash
git add docs/superpowers/investigations/
git commit -m "docs(investigation): final report — height-map flattening root causes and fixes"
```
