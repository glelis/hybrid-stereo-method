# Method Audit Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir todos os achados remanescentes do relatório `docs/superpowers/reports/2026-06-04-method-audit.md` (42 ativos − 7 já corrigidos: MF-01, MF-02, MF-07, MF-09, MF-12, INT-04, CONV-4), mais uma regressão descoberta na preparação deste plano (REG-01).

**Architecture:** Um achado (ou grupo coeso de achados no mesmo trecho) por tarefa, em ordem de severidade (crítico → alto → médio → baixo), com TDD: teste que falha no estado atual → correção mínima → teste passa → commit `fix(<ID>): ...` → commit `docs(<ID>): pin correction SHA`. Os 4 xfail-evidência existentes viram testes verdes (remover o marker faz parte da correção). A consolidação final re-mede a baseline E2E e atualiza o relatório.

**Tech Stack:** Python 3 (numpy, opencv, pytest), binário C `gus_integrate_recursive` (gcc/make só na Tarefa 26).

---

## Protocolo (vale para todas as tarefas)

- **Ambiente:** o worktree NÃO tem editable install próprio — rodar sempre `PYTHONPATH=src pytest ...` a partir da raiz do worktree.
- **Suíte rápida:** `PYTHONPATH=src pytest -m "not slow" -q` após cada tarefa (deve ficar verde, fora xfails ainda não corrigidos).
- **Suíte completa (lenta):** após Tarefas 2, 18 e 27: `PYTHONPATH=src pytest -q`.
- **Commit docs (pin SHA):** após cada commit `fix(<ID>): ...`:
  1. `SHA=$(git rev-parse --short HEAD)`
  2. No relatório `docs/superpowers/reports/2026-06-04-method-audit.md`, no bloco do achado, acrescentar a linha:
     `- **Correção aplicada:** \`$SHA\` (2026-06-05) — <1 frase do que mudou e teste que pina>.`
  3. Na nota correspondente em `docs/superpowers/reports/audit-notes/0X-*.md`, mudar o status para `corrigido ($SHA)`.
  4. `git add docs/superpowers/reports/ && git commit -m "docs(<ID>): pin correction SHA to $SHA"`
- **Regra de teste-evidência:** os testes xfail existentes são evidência; ao corrigir, REMOVER o marker (XPASS strict falharia). Se um teste existente quebrar por mudança de comportamento *intencional*, atualizar o teste citando o ID — nunca afrouxar limiar sem justificar no commit.
- **Lint:** `ruff check src/ tests/ && ruff format src/ tests/` antes de cada commit.

---

## Estrutura de arquivos

| Arquivo | Mudança |
|---|---|
| `src/hybrid_stereo_method/hybrid/main.py` | MF-14, REG-01, CONV-6, INT-01, PS-06, PS-07, MF-04, INT-05, INT-06 |
| `src/hybrid_stereo_method/hybrid/integrate.py` | INT-02, INT-06, INT-08 |
| `src/hybrid_stereo_method/hybrid/hints.py` (novo) | INT-05 (célula→vértice) |
| `src/hybrid_stereo_method/photometric/wps.py` | PS-01..PS-05, PS-10 |
| `src/hybrid_stereo_method/photometric/main_wps.py` | PS-06, PS-07, PS-08, CONV-2 |
| `src/hybrid_stereo_method/photometric/visualization.py` | PS-11 |
| `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py` | MF-03, MF-05, MF-06, MF-11 |
| `src/hybrid_stereo_method/multifocus/mosaic.py` | MF-03, MF-04 |
| `src/hybrid_stereo_method/multifocus/main.py` | MF-04 (wSel cru), MF-13 |
| `src/hybrid_stereo_method/multifocus/indicators/applicator.py` | MF-08, MF-10 |
| `src/hybrid_stereo_method/multifocus/indicators/non_linear_res.py` | MF-10 (WLS) |
| `src/hybrid_stereo_method/multifocus/image_alignment.py` | IO-06 |
| `src/hybrid_stereo_method/infrastructure/io/image_io.py` | IO-01, IO-02, IO-03, IO-04 (docstring) |
| `src/hybrid_stereo_method/infrastructure/utils.py` | PS-09 |
| `csrc/integrate_recursive/gus_integrate_recursive.c` + binário | INT-07, build |
| `configs/hb_experiment.yaml` | chaves novas (PS-02, PS-08, INT-06, CONV-2) |
| `tests/` | novos testes por tarefa; remoção dos 4 xfail; `test_average_float_path.py` import |

---

### Task 1: Reparar a baseline da suíte (import quebrado)

**Files:**
- Modify: `tests/test_average_float_path.py`

- [ ] **Step 1: Reproduzir a falha**

Run: `PYTHONPATH=src pytest tests/test_average_float_path.py -q`
Expected: 1 failed — `ModuleNotFoundError: No module named 'tests.synthetic_utils'`

- [ ] **Step 2: Corrigir o import**

Em `tests/test_average_float_path.py`, dentro de `_build_small_dataset`, trocar:

```python
    from tests.synthetic_utils import (
```
por:
```python
    from synthetic_utils import (
```
(pytest insere `tests/` no sys.path; os demais testes importam `synthetic_utils` sem prefixo.)

- [ ] **Step 3: Verificar**

Run: `PYTHONPATH=src pytest tests/test_average_float_path.py -q`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_average_float_path.py
git commit -m "test: fix synthetic_utils import to match pytest rootdir convention"
```

---

### Task 2: MF-14 (crítico) — detecção de diretórios de luz em qualquer profundidade

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py`
- Modify: `tests/test_e2e_hybrid.py` (remover xfail; remover variante workaround)
- Test: `tests/test_hybrid_path_selection.py` (adicionar unit test)

- [ ] **Step 1: Escrever o teste unitário que falha**

Em `tests/test_hybrid_path_selection.py` adicionar:

```python
def test_collect_light_dirs_finds_lights_at_any_depth(tmp_path):
    """MF-14: num layout limpo L<n>/zf<m>/sVal.png o pai imediato é sempre zf*,
    então a detecção por pai imediato devolve vazio. A detecção correta acha o
    componente L<n> em QUALQUER posição do caminho relativo ao dataset."""
    from hybrid_stereo_method.hybrid.main import collect_light_dirs

    data = tmp_path / "synth"
    files = []
    for li in range(3):
        for k in range(2):
            d = data / f"L{li}" / f"zf{k}"
            d.mkdir(parents=True)
            f = d / "sVal.png"
            f.write_bytes(b"")
            files.append(str(f))
    (data / "lights.npy").write_bytes(b"")
    files.append(str(data / "lights.npy"))
    # detritos que NÃO são luzes: prefixo L sem dígito, L<n> no nome do dataset
    (data / "Lixo").mkdir()
    f = data / "Lixo" / "x.png"
    f.write_bytes(b"")
    files.append(str(f))

    assert collect_light_dirs(files, str(data)) == ["L0", "L1", "L2"]


def test_collect_light_dirs_natural_order(tmp_path):
    from hybrid_stereo_method.hybrid.main import collect_light_dirs

    data = tmp_path / "d"
    files = []
    for name in ["L10", "L2", "L1"]:
        d = data / name / "zf0"
        d.mkdir(parents=True)
        f = d / "sVal.png"
        f.write_bytes(b"")
        files.append(str(f))
    assert collect_light_dirs(files, str(data)) == ["L1", "L2", "L10"]
```

Run: `PYTHONPATH=src pytest tests/test_hybrid_path_selection.py -q`
Expected: FAIL — `ImportError: cannot import name 'collect_light_dirs'`

- [ ] **Step 2: Implementar `collect_light_dirs`**

Em `src/hybrid_stereo_method/hybrid/main.py`, adicionar `import re` no topo e a função (após `collect_dirs_with_prefix`):

```python
def collect_light_dirs(files: list[str], data_path: str | Path) -> list[str]:
    """Return unique ``L<n>`` directory components found at ANY depth under
    ``data_path``, in natural order.

    Fixes MF-14: the previous detection looked only at the immediate parent of
    each file, which in the documented layout ``L<n>/zf<m>/sVal.png`` is always
    a ``zf*`` directory — so no lights were detected on clean datasets and the
    pipeline aborted in the photometric step.
    """
    lights: set[str] = set()
    for f in files:
        try:
            parts = Path(f).relative_to(data_path).parts
        except ValueError:
            parts = Path(f).parts
        for part in parts[:-1]:  # exclude the filename itself
            if re.fullmatch(r"L\d+", part):
                lights.add(part)
    return natsorted(lights)
```

E no `main()`, trocar:

```python
    light_directories = collect_dirs_with_prefix(input_files_path, prefix="L")
```
por:
```python
    light_directories = collect_light_dirs(
        input_files_path, os.path.join(input_path, data_foldername)
    )
```
(ajustar o comentário acima da linha para citar MF-14).

- [ ] **Step 3: Rodar o unit test**

Run: `PYTHONPATH=src pytest tests/test_hybrid_path_selection.py -q`
Expected: PASS

- [ ] **Step 4: Virar o E2E limpo de xfail para verde**

Em `tests/test_e2e_hybrid.py`:
1. Remover o decorator `@_mf14_xfail` de `test_hybrid_pipeline_end_to_end` e a definição `_mf14_xfail` (atualizar o comentário: MF-14 corrigido nesta tarefa).
2. Remover a variante `test_hybrid_pipeline_end_to_end_with_workaround` inteira (redundante: o teste limpo agora mede a baseline) e qualquer helper exclusivo dela (ex.: criação de `marker.txt`).
3. Em `tests/test_average_float_path.py`, remover a criação dos `marker.txt` (workaround MF-14) do `_build_small_dataset`.

Run: `PYTHONPATH=src pytest tests/test_e2e_hybrid.py tests/test_average_float_path.py -v -s`
Expected: PASS, com a linha `=== BASELINE E2E ===` impressa (registrar os números — vão para a Task 27).

- [ ] **Step 5: Commit + pin**

```bash
git add src/hybrid_stereo_method/hybrid/main.py tests/
git commit -m "fix(MF-14): detect L<n> light dirs at any path depth, not just immediate parent"
SHA=$(git rev-parse --short HEAD)
# atualizar relatório (MF-14 → corrigido, SHA) e audit-notes/01-multifocus.md
git add docs/superpowers/reports/
git commit -m "docs(MF-14): pin correction SHA to $SHA"
```

---

### Task 3: REG-01 — regressão radiométrica do `sMos.png` (normalize=True)

Descoberto na preparação deste plano: o commit `aa772ed` trocou `normalize=False` → `normalize=True` no save do `sMos.png`, contradizendo o próprio comentário e reintroduzindo o stretch por-luz que o commit `4d7316b` tinha removido. Registrar como achado novo REG-01 no relatório (seção 2.4 ou nova subseção "Regressões"), severidade alta, confirmado.

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py`
- Modify: `tests/test_e2e_hybrid.py`

- [ ] **Step 1: Teste de radiometria no E2E**

Em `tests/test_e2e_hybrid.py::test_hybrid_pipeline_end_to_end`, após a checagem do `height_map.npy`, adicionar:

```python
    # REG-01: com normalize=True cada sMos.png é esticado a [0,255] e TODOS os
    # mosaicos têm max==255; com normalize=False (correto) o máximo de cada luz
    # preserva a radiometria (< 255, pois albedo<=230 no dataset sintético).
    smos_paths = sorted(out_dirs[0].glob("multifocus_stereo/L*/sMos.png"))
    assert len(smos_paths) == N_LIGHTS
    maxima = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED).max() for p in smos_paths]
    assert all(m < 255 for m in maxima), (
        f"sMos.png re-esticado por luz (max={maxima}): regressão REG-01 ativa"
    )
```

Run: `PYTHONPATH=src pytest tests/test_e2e_hybrid.py -q`
Expected: FAIL no novo assert (todos os máximos == 255).

- [ ] **Step 2: Restaurar `normalize=False`**

Em `src/hybrid_stereo_method/hybrid/main.py`, trocar:

```python
        save_image(output_path_multifocus, "sMos.png", sMos_light, normalize=True)
```
por:
```python
        save_image(output_path_multifocus, "sMos.png", sMos_light, normalize=False)
```

- [ ] **Step 3: Rodar**

Run: `PYTHONPATH=src pytest tests/test_e2e_hybrid.py -q`
Expected: PASS

- [ ] **Step 4: Commit + pin (criar o bloco REG-01 no relatório no commit docs)**

```bash
git add src/hybrid_stereo_method/hybrid/main.py tests/test_e2e_hybrid.py
git commit -m "fix(REG-01): restore normalize=False for per-light sMos.png (regression from aa772ed)"
SHA=$(git rev-parse --short HEAD)
git add docs/superpowers/reports/
git commit -m "docs(REG-01): record sMos normalize regression and pin fix to $SHA"
```

---

### Task 4: CONV-6 — pareamento luz↔mosaico por chave `L<n>`, não por posição

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py`
- Test: `tests/test_hybrid_path_selection.py`

- [ ] **Step 1: Teste que falha**

```python
def test_pair_mosaics_to_lights_orders_by_index_and_validates():
    from hybrid_stereo_method.hybrid.main import pair_mosaics_to_lights

    paths = ["/out/L10/sMos.png", "/out/L0/sMos.png", "/out/L2/sMos.png", "/out/L1/sMos.png"]
    # OK quando os índices são exatamente 0..n-1 — devolve em ordem de índice
    import pytest

    with pytest.raises(ValueError, match="do not match"):
        pair_mosaics_to_lights(paths, n_lights=4)  # tem L10, falta L3

    paths_ok = ["/out/L2/sMos.png", "/out/L0/sMos.png", "/out/L1/sMos.png"]
    assert pair_mosaics_to_lights(paths_ok, n_lights=3) == [
        "/out/L0/sMos.png",
        "/out/L1/sMos.png",
        "/out/L2/sMos.png",
    ]

    with pytest.raises(ValueError, match="no L<n> parent"):
        pair_mosaics_to_lights(["/out/average/sMos.png"], n_lights=1)
```

Run: `PYTHONPATH=src pytest tests/test_hybrid_path_selection.py -q` → FAIL (ImportError)

- [ ] **Step 2: Implementar e usar**

Em `hybrid/main.py`:

```python
def pair_mosaics_to_lights(mosaic_paths: list[str], n_lights: int) -> list[str]:
    """Order per-light mosaics by their ``L<n>`` index and verify the indices
    are exactly ``0..n_lights-1`` (one mosaic per lights.npy row).

    Fixes CONV-6: pairing was positional (natsorted paths vs lights.npy rows)
    with only a count check; a missing/extra light dir could silently shift
    every light↔mosaic association.
    """
    indexed: dict[int, str] = {}
    for p in mosaic_paths:
        m = re.fullmatch(r"L(\d+)", os.path.basename(os.path.dirname(p)))
        if m is None:
            raise ValueError(f"Mosaic path has no L<n> parent directory: {p}")
        idx = int(m.group(1))
        if idx in indexed:
            raise ValueError(f"Duplicate mosaic for light L{idx}: {p} and {indexed[idx]}")
        indexed[idx] = p
    if set(indexed) != set(range(n_lights)):
        raise ValueError(
            f"Mosaic light indices {sorted(indexed)} do not match lights.npy "
            f"rows 0..{n_lights - 1}: every row must have exactly one L<n> mosaic"
        )
    return [indexed[i] for i in range(n_lights)]
```

No Step 2 do `main()`, trocar a construção de `sMos_path_list` para validar contra as luzes:

```python
    output_files = find_all_files(output_path)
    mosaic_paths = [
        file
        for file in output_files
        if os.path.basename(file) == "sMos.png"
        and os.path.basename(os.path.dirname(file)) != "average"
    ]
    parameters["lights_path"] = [file for file in input_files_path if "lights.npy" in file][0]
    n_lights = int(np.load(parameters["lights_path"]).shape[0])
    # CONV-6: pair light<n> -> lights.npy row n by KEY, not by sort position
    parameters["sMos_path_list"] = pair_mosaics_to_lights(mosaic_paths, n_lights)
```
(remover a atribuição duplicada de `lights_path` que existia mais abaixo).

- [ ] **Step 3: Rodar** `PYTHONPATH=src pytest tests/test_hybrid_path_selection.py -m "not slow" -q` → PASS; depois suíte rápida.

- [ ] **Step 4: Commit + pin** (`fix(CONV-6): pair light mosaics to lights.npy rows by L<n> key`)

---

### Task 5: INT-01 — default de `initial_method` e validação hints

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (função `build_integration_config`)
- Test: `tests/test_integration_units.py`

- [ ] **Step 1: Teste que falha**

```python
def test_initial_method_defaults_to_zero_and_hints_requires_use_hints():
    """INT-01: default era 'hints' mesmo sem -hints, e o binário aborta
    (demand H!=NULL). Default correto: 'zero'; 'hints' exige use_hints."""
    from hybrid_stereo_method.hybrid.main import build_integration_config

    cfg = build_integration_config({}, debug=False)
    assert cfg.initial_method == "zero"

    import pytest

    with pytest.raises(ValueError, match="use_hints"):
        build_integration_config({"initial_method": "hints", "use_hints": False}, debug=False)

    cfg = build_integration_config(
        {"initial_method": "hints", "use_hints": True, "pixel_size": 1.0}, debug=False
    )
    assert cfg.initial_method == "hints"
```

Run → FAIL (default atual é "hints" e não há validação).

- [ ] **Step 2: Corrigir em `build_integration_config`**

```python
    initial_method = integration_params.get("initial_method", "zero")
    if initial_method == "hints" and not integration_params.get("use_hints", False):
        raise ValueError(
            "hybrid.integration.initial_method='hints' requires use_hints: true — "
            "without a hints map the C solver aborts (INT-01)."
        )
```
e usar `initial_method=initial_method` no construtor (alinha com o default `"zero"` do dataclass e do binário).

- [ ] **Step 3: Rodar** o teste → PASS; suíte rápida.

- [ ] **Step 4: Commit + pin** (`fix(INT-01): default initial_method to zero; validate hints requires use_hints`)

---

### Task 6: PS-02 — limiares absoluto de sombra e de saturação

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/wps.py` (`estimate_normals_argmax_lstsq_robust`)
- Modify: `configs/hb_experiment.yaml`, `configs/wps_experiment.yaml` (se tiver seção solver)
- Test: `tests/test_photometric_synthetic.py`

- [ ] **Step 1: Teste que falha (reproduz a verificação do revisor da auditoria)**

```python
def test_wps_rejects_8bit_floor_shadows():
    """PS-02: com o piso de 8 bits (sombras viram 1/255 em vez de 0), o limiar
    relativo 1e-3 não rejeita nada e o erro angular sobe para ~3.6° médio.
    Com limiar ABSOLUTO de sombra, os pixels sombreados são rejeitados e o
    erro volta a < 0.5°."""
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=10.0, sigma_frac=0.15))
    lights = ring_lights(5, tilt_deg=75.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]
    floor = 255.0 / 255.0  # menor valor não-nulo de um sensor 8 bits, escala 0-255
    images = [np.maximum(img, floor) for img in images]

    normals, _, _, _ = estimate_normals_argmax_lstsq_robust(
        images, lights, {"shadow_absolute_threshold": 2.0}
    )
    valid = np.isfinite(normals).all(axis=-1)
    assert valid.any()
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\npiso 8-bit + limiar absoluto: erro médio = {ang.mean():.3f}°")
    assert ang.mean() < 0.5, f"sombras de piso 8-bit não rejeitadas: {ang.mean():.2f}°"


def test_wps_rejects_saturated_measurements():
    """PS-02: medições saturadas (>= saturation_threshold) devem sair do lstsq."""
    size = 32
    n_gt = normals_from_height(gaussian_bump(size, amplitude=5.0))
    lights = ring_lights(8, tilt_deg=30.0)
    images = [render_lambertian(n_gt, light, albedo=300.0) for light in lights]
    images_sat = [np.minimum(img, 255.0) for img in images]  # clipe do sensor

    normals, _, _, _ = estimate_normals_argmax_lstsq_robust(
        images_sat, lights, {"saturation_threshold": 250.0}
    )
    valid = np.isfinite(normals).all(axis=-1)
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\nsaturação tratada: erro médio = {ang.mean():.3f}°")
    assert ang.mean() < 1.0
```

Run → FAIL (chaves não existem; erro alto).

- [ ] **Step 2: Implementar em `wps.py`**

No início da função:

```python
    shadow_threshold = wps_params.get("shadow_threshold", 1e-3)
    # PS-02: o limiar relativo é inócuo abaixo do piso de 8 bits (1/255 ≈ 3.9e-3
    # > 1e-3 sempre que houver sinal). Limiar ABSOLUTO em radiância linear
    # rejeita sombras reais; limiar superior rejeita medições saturadas.
    shadow_absolute = wps_params.get("shadow_absolute_threshold")
    saturation_threshold = wps_params.get("saturation_threshold")
```

Na seleção de pixels válidos:

```python
            valid_indices = pixel_values / v_max > shadow_threshold
            if shadow_absolute is not None:
                valid_indices &= pixel_values >= shadow_absolute
            if saturation_threshold is not None:
                valid_indices &= pixel_values <= saturation_threshold
```

Em `configs/hb_experiment.yaml`, seção `photometric.solver`, adicionar (com comentários):

```yaml
    # Absolute shadow threshold in the same intensity units as the input
    # images (0-255 for 8-bit data). Measurements below it are treated as
    # shadow and excluded (PS-02). ~2/255 rejects the 8-bit noise floor.
    shadow_absolute_threshold: 2.0

    # Upper threshold: measurements at/above it are treated as saturated and
    # excluded (PS-02). 250 leaves margin below the 255 clip.
    saturation_threshold: 250.0
```

- [ ] **Step 3: Rodar** `PYTHONPATH=src pytest tests/test_photometric_synthetic.py -q -s` → novos PASS, antigos inalterados.

- [ ] **Step 4: Commit + pin** (`fix(PS-02): absolute shadow and saturation thresholds for robust WPS`)

---

### Task 7: PS-06 + INT-03 — confiança do PS como peso do integrador

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/main_wps.py` (salvar `confidence.npy`)
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (integrar mapa (H,W,4))
- Test: `tests/test_integration_units.py`

O lado C já aceita normal map de 4 canais com canal 3 = peso (`pst_normal_map_to_slope_map`, `demand((NC==3)||(NC==4))`) e NaN→peso 0 via backstop. Falta o Python emitir o peso.

- [ ] **Step 1: Teste que falha (binário, marcado slow)**

```python
@needs_binary
@pytest.mark.slow
def test_integrator_accepts_confidence_weight_channel(tmp_path):
    """PS-06/INT-03: normal map (H,W,4) com canal 3 = confiança; pixels com
    peso 0 (sombra) não devem contaminar nem crashar a integração."""
    from synthetic_utils import gaussian_bump, normals_from_height

    from hybrid_stereo_method.hybrid.integrate import integrate_normals_to_height

    size = 32
    z = gaussian_bump(size, amplitude=4.0)
    n = normals_from_height(z)
    conf = np.ones((size, size), dtype=np.float64)
    n4 = np.concatenate([n, conf[..., None]], axis=-1)
    # zona "sombreada": NaN nas normais + confiança 0 (o que o wps produz)
    n4[10:14, 10:14, :3] = np.nan
    n4[10:14, 10:14, 3] = 0.0

    out = integrate_normals_to_height(n4, tmp_path, "w4")
    assert out.shape == (size + 1, size + 1)
    assert np.isfinite(out).all()
```

Run → provavelmente já PASSA no C (que aceita 4 canais); se passar, é o pino do contrato. O que FALTA (e falha) é o pipeline usar esse caminho — Step 2 cobre.

- [ ] **Step 2: Pipeline emite e consome a confiança**

Em `main_wps.py`, após `np.save(normal_map_path, normals)`:

```python
    # PS-06/INT-03: persist the confidence so the integrator can use it as the
    # per-pixel weight channel (shadowed pixels carry weight 0 explicitly).
    confidence_path = os.path.join(output_path, "confidence.npy")
    np.save(confidence_path, confidence)
```

Em `hybrid/main.py` Step 3, após carregar `normal_map`:

```python
        # PS-06/INT-03: attach the photometric confidence as the weight channel
        # (H,W,4) so shadowed/degenerate pixels (NaN normals, confidence 0) are
        # excluded by weight instead of relying only on the C NaN backstop.
        confidence_path = os.path.join(
            parameters["output_path_photometric"], "confidence.npy"
        )
        if os.path.exists(confidence_path):
            confidence = np.load(confidence_path)
            normal_map = np.concatenate(
                [normal_map, confidence[..., None].astype(normal_map.dtype)], axis=-1
            )
        else:
            logging.warning(
                "confidence.npy not found — integrating normals without weight channel"
            )
```

- [ ] **Step 3: Rodar** `PYTHONPATH=src pytest tests/test_integration_units.py tests/test_e2e_hybrid.py -q` → PASS.

- [ ] **Step 4: Commit + pin** (`fix(PS-06,INT-03): photometric confidence becomes integrator weight channel`)

---

### Task 8: PS-07 — PS consome mosaicos float em memória (fecha o caminho de dados de IO-04/IO-05/CONV-5)

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (lista in-memory)
- Modify: `src/hybrid_stereo_method/photometric/main_wps.py` (preferir in-memory)
- Test: `tests/test_average_float_path.py` (mesmo padrão do MF-12)

- [ ] **Step 1: Teste que falha**

Em `tests/test_average_float_path.py` adicionar (reutilizando `_build_small_dataset` e o padrão de spy existente):

```python
@needs_binary
def test_photometric_receives_float_mosaics_in_memory(tmp_path, monkeypatch):
    """PS-07: o PS deve receber os mosaicos float em memória (sMos_images);
    o sMos.png uint8 vira só visualização."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import hybrid_stereo_method.photometric.main_wps as main_wps_mod
    from hybrid_stereo_method.hybrid.main import main as hybrid_main

    monkeypatch.setattr(main_wps_mod, "disp_normalmap", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels_3d", lambda **kw: None)

    captured = {}
    real_estimator = main_wps_mod.estimate_normals_argmax_lstsq_robust

    def spy_estimator(images, lights, params):
        captured["dtypes"] = [np.asarray(img).dtype for img in images]
        return real_estimator(images, lights, params)

    monkeypatch.setattr(
        main_wps_mod, "estimate_normals_argmax_lstsq_robust", spy_estimator
    )

    parameters = _build_parameters(tmp_path)  # helper já existente no arquivo
    hybrid_main(parameters)

    assert captured, "estimator não foi chamado"
    assert all(dt.kind == "f" for dt in captured["dtypes"]), (
        f"PS recebeu dtypes {captured['dtypes']} — caminho uint8/PNG ainda ativo (PS-07)"
    )
```

(Se o helper de parâmetros do arquivo tiver outro nome, usar o existente.)

Run → FAIL (dtypes uint8, lidos do PNG).

- [ ] **Step 2: hybrid/main.py acumula floats por luz**

No laço de luzes, acumular num dict por índice (consistente com CONV-6):

```python
    sMos_by_light: dict[str, np.ndarray] = {}
    for light_dir in light_directories:
        ...
        sMos_light, _ = mosaic(iSel_avg, image_stack, zFoc, interpolation_type)
        sMos_by_light[light_dir] = sMos_light  # float64 (H,W,3) — PS-07
        ...
```

E após o pareamento CONV-6 do Step 2 (que devolve a ordem L0..L(n-1)):

```python
    # PS-07: hand the float mosaics to the PS in lights.npy row order, bypassing
    # the uint8 PNG round-trip (sMos.png stays as visualization only).
    parameters["sMos_images"] = [
        sMos_by_light[f"L{i}"] for i in range(n_lights)
    ]
```

- [ ] **Step 3: main_wps.py prefere in-memory**

No branch hybrid de carregamento:

```python
    if experiment_type == "hybrid":
        if parameters.get("sMos_images") is not None:
            logging.info("... Using in-memory float mosaics (sMos_images) — PS-07 ...")
            images = [np.asarray(img, dtype=np.float64) for img in parameters["sMos_images"]]
        else:
            images = read_images(parameters.get("sMos_path_list"))
```

- [ ] **Step 4: Rodar** o teste novo + `tests/test_e2e_hybrid.py` (baseline pode melhorar — registrar números impressos) → PASS.

- [ ] **Step 5: Commit + pin** — no commit docs, além de PS-07, atualizar IO-04, IO-05 e CONV-5 no relatório: o caminho de dados não passa mais por PNG em nenhum estágio (médias: MF-12; mosaicos: este commit); PNGs restantes são visualização. (`fix(PS-07): feed photometric stereo with float mosaics in memory`)

---

### Task 9: PS-08 — linearização gamma opcional (fecha CONV-5)

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/main_wps.py`
- Modify: `configs/hb_experiment.yaml`
- Test: `tests/test_photometric_synthetic.py`

- [ ] **Step 1: Teste que falha**

```python
def test_linearize_gamma_helper():
    """PS-08: o modelo Lambertiano exige intensidades lineares; se a aquisição
    for gamma-encoded, decodificar com I_lin = 255*(I/255)^gamma."""
    from hybrid_stereo_method.photometric.main_wps import linearize_intensities

    img = np.array([[0.0, 127.5, 255.0]])
    out = linearize_intensities(img, gamma=2.2)
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[0, 2], 255.0)
    np.testing.assert_allclose(out[0, 1], 255.0 * (0.5 ** 2.2), rtol=1e-12)
    # gamma=1.0 é a identidade (default: dados assumidos lineares)
    np.testing.assert_allclose(linearize_intensities(img, gamma=1.0), img)
```

Run → FAIL (função não existe).

- [ ] **Step 2: Implementar**

Em `main_wps.py`:

```python
def linearize_intensities(img: np.ndarray, gamma: float) -> np.ndarray:
    """Decode gamma-encoded intensities to linear radiance (PS-08).

    The Lambertian model I = rho * (L . n) requires LINEAR intensities. The
    pipeline assumes the input ``sVal.png`` are linear (gamma=1.0, default);
    if the acquisition applied a gamma curve, set ``photometric.parameters.gamma``
    to decode: I_lin = 255 * (I/255)^gamma.
    """
    if gamma == 1.0:
        return img
    img = np.asarray(img, dtype=np.float64)
    return 255.0 * np.power(np.clip(img, 0.0, None) / 255.0, gamma)
```

E após a conversão para grayscale:

```python
    gamma = float(parameters.get("photometric", {}).get("parameters", {}).get("gamma", 1.0))
    images = [linearize_intensities(img, gamma) for img in images]
```

Em `configs/hb_experiment.yaml`, seção `photometric.parameters`:

```yaml
    # Gamma of the acquisition. The Lambertian model requires LINEAR
    # intensities (PS-08/CONV-5); 1.0 (default) means the sVal.png are already
    # linear. If they are sRGB/gamma-encoded, set the encoding gamma (e.g. 2.2)
    # to decode before solving.
    gamma: 1.0
```

- [ ] **Step 3: Rodar** → PASS. **Step 4: Commit + pin** (`fix(PS-08): optional gamma linearization; document linearity premise`) — no docs, marcar CONV-5 como fechado (3 pontos: MF-12, PS-07, PS-08).

---

### Task 10: MF-03 + MF-04 — invalidez propagada (NaN) e máscara de confiança no mosaico

**Files:**
- Modify: `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py` (`return n/2, 0` → NaN)
- Modify: `src/hybrid_stereo_method/multifocus/mosaic.py` (parâmetro `wSel`)
- Modify: `src/hybrid_stereo_method/multifocus/main.py` (passar wSel; wSel cru no FNI; viz com nan_to_num)
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (passar `wSel_avg` ao mosaico por luz)
- Test: `tests/test_multifocus_synthetic.py`

- [ ] **Step 1: Testes que falham**

```python
def test_zero_peak_returns_nan_not_middle_frame():
    """MF-03: pico de foco nulo é indecidível — deve virar NaN/conf 0, não n/2."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d

    k, conf = compute_argmax_fuzzy_1d(np.zeros(9), [0, 0], {"r_max": 2})
    assert conf == 0
    assert np.isnan(k), f"pico nulo devolveu k={k} em vez de NaN (MF-03)"


def test_mosaic_masks_zero_confidence_pixels():
    """MF-04: pixels com confiança 0 não podem entrar no zMos como profundidade
    válida — viram NaN; o sMos usa o frame mais próximo (precisa de valor)."""
    from hybrid_stereo_method.multifocus.mosaic import mosaic

    n, h, w = 5, 4, 4
    stack = np.random.default_rng(0).uniform(0, 255, (n, h, w, 3))
    z_foc = [10.0, 20.0, 30.0, 40.0, 50.0]
    iSel = np.full((h, w), 2.0)
    iSel[0, 0] = np.nan  # MF-03: pixel indecidível
    wSel = np.ones((h, w))
    wSel[1, 1] = 0.0  # confiança zero

    sMos, zMos = mosaic(iSel, stack, z_foc, "linear_interpolation", wSel=wSel)

    assert np.isnan(zMos[0, 0]) and np.isnan(zMos[1, 1])
    assert np.isfinite(sMos).all(), "sMos deve sempre ter valor (consumido pelo PS)"
    assert zMos[2, 2] == 30.0
```

Run → FAIL.

- [ ] **Step 2: `argmax_fuzzy.py`**

Trocar:
```python
    if focus_values[k_max] == 0:
        return n / 2, 0
```
por:
```python
    if focus_values[k_max] == 0:
        # MF-03: pico nulo = profundidade indecidível. NaN propaga a invalidez;
        # o consumidor (mosaic) mascara via confiança 0 em vez de inventar n/2.
        return np.nan, 0
```

- [ ] **Step 3: `mosaic.py`**

Nova assinatura e máscara (preservando todo o resto):

```python
def mosaic(
    iSel,
    image_stack: np.array,
    zFoc: list,
    interpolation_type: str,
    wSel: np.ndarray | None = None,
    min_confidence: float = 0.0,
):
```
Docstring: acrescentar que `wSel`/`min_confidence` (MF-04) mascaram pixels de
confiança ≤ limiar: `zMos` recebe NaN (inválido explícito; o peso 0 em
`zMos_with_confidence` exclui o hint no C), `sMos` recebe o frame mais próximo
(o PS precisa de valor em todo pixel).

No corpo do laço, antes do bloco de interpolação:

```python
            k_fuzzy = iSel[i, j]
            invalid = not np.isfinite(k_fuzzy) or (
                wSel is not None and wSel[i, j] <= min_confidence
            )
            if invalid:
                K_indice = (
                    n_frames // 2
                    if not np.isfinite(k_fuzzy)
                    else min(max(int(k_fuzzy), 0), n_frames - 1)
                )
                zMos[i, j] = np.nan
                sMos[i, j, :] = image_stack[K_indice, i, j, :]
                continue
```

- [ ] **Step 4: Chamadas e exports**

Em `multifocus/main.py`:
- `sMos, zMos = mosaic(iSel, image_stack, zFoc, interpolation_type, wSel=wSel)`
- visualizações com NaN: `save_image(output_path, "iSel.png", np.nan_to_num(iSel, nan=0.0))` e `convert_image_array_to_fni(normalize(np.nan_to_num(iSel, nan=0.0)), ...)`; idem `zMos.png`.
- MF-07 follow-through: `zMos_with_confidence = np.stack((zMos, wSel), axis=-1)` (sem `normalize(wSel)` — R² já é absoluto em [0,1]; re-esticar destruiria a escala). Idem no `wSel.fni`: usar `wSel` cru.
- onde `zMos` tem NaN, garantir peso 0: `wSel = np.where(np.isfinite(zMos), wSel, 0.0)` antes do stack.

Em `hybrid/main.py` (laço de luzes):
```python
        sMos_light, _ = mosaic(iSel_avg, image_stack, zFoc, interpolation_type, wSel=wSel_avg)
```

- [ ] **Step 5: Rodar** novos testes + `tests/test_multifocus_synthetic.py tests/test_confidence_r2.py tests/test_e2e_hybrid.py` → PASS (se algum teste existente pinar `normalize(wSel)`, atualizar citando MF-04/MF-07).

- [ ] **Step 6: Dois commits + pins**

```bash
git add src/hybrid_stereo_method/multifocus/argmax_fuzzy.py tests/test_multifocus_synthetic.py
git commit -m "fix(MF-03): undecidable focus peak returns NaN instead of middle frame"
# pin docs MF-03
git add -A src tests
git commit -m "fix(MF-04): mosaic masks zero-confidence pixels; raw R2 exported as weight"
# pin docs MF-04
```

---

### Task 11: MF-05 + MF-06 + MF-11 — correções do ajuste parabólico (3 commits)

**Files:**
- Modify: `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py`
- Test: `tests/test_confidence_r2.py` (ou novo `tests/test_argmax_fit.py`)

- [ ] **Step 1 (MF-05): teste que falha**

```python
def test_peak_index_is_true_argmax():
    """MF-05: a janela soma-de-3 podia escolher índice fora do pico verdadeiro
    (fv=[0,0,0,9,0,5,6,5,0,0] -> max-sum dá 6; argmax verdadeiro é 3)."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import find_peak_index

    assert find_peak_index(np.array([0, 0, 0, 9, 0, 5, 6, 5, 0, 0])) == 3
    # empate exato: desempata pela vizinhança com mais suporte
    assert find_peak_index(np.array([0, 5, 0, 0, 4, 5, 4, 0])) == 5
```

- [ ] **Step 2 (MF-05): implementar `find_peak_index` e usar**

```python
def find_peak_index(focus_values) -> int:
    """Índice do pico verdadeiro (argmax), com desempate pela vizinhança.

    Substitui ``find_index_of_max_sum`` (MF-05): a soma-de-3 é um passa-baixa
    que favorece platôs largos e pode escolher uma janela que nem contém o
    argmax. Empates exatos são desfeitos pela maior soma dos vizinhos.
    """
    fv = np.asarray(focus_values, dtype=np.float64)
    if fv.size < 3:
        raise ValueError(f"focus_values must contain at least 3 frames, got {fv.size}")
    candidates = np.flatnonzero(fv == fv.max())
    if candidates.size == 1:
        return int(candidates[0])
    padded = np.pad(fv, 1, mode="edge")
    support = padded[candidates] + padded[candidates + 1] + padded[candidates + 2]
    return int(candidates[np.argmax(support)])
```

Em `compute_argmax_fuzzy_1d`: `k_max = find_peak_index(focus_values)`. Remover `find_index_of_max_sum` (era usado só ali); `grep -rn "find_index_of_max_sum" src tests` e atualizar referências de teste, citando MF-05.

Commit: `fix(MF-05): true argmax (tie-broken) replaces 3-sum window for peak seed`+pin.

- [ ] **Step 3 (MF-06): teste que falha**

```python
def test_parabola_vertex_unbiased_by_value_weights():
    """MF-06: ponderar a regressão pelos próprios valores de foco enviesa o
    vértice para o frame de maior valor. Curva gaussiana com pico em 4.3:
    o ajuste deve achar o vértice perto de 4.3 (não puxado para 4)."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d

    x = np.arange(9, dtype=float)
    fv = np.exp(-((x - 4.3) ** 2) / (2 * 1.2**2))
    k, conf = compute_argmax_fuzzy_1d(fv, [0, 0], {"r_max": 2})
    assert conf > 0
    assert abs(k - 4.3) < 0.1, f"vértice enviesado: {k:.3f} (gt 4.3) — MF-06"
```

(Registrar no commit o valor obtido ANTES da correção, para o relatório.)

- [ ] **Step 4 (MF-06): remover os pesos do ajuste**

- `A, B, C = tuple(np.polyfit(x_list, y_list, 2))` dentro de um único try/except `LinAlgError` → fallback `return k_max, 0`.
- Remover `calculate_weights` e `w_list` (inclusive da linha/header do CSV de debug — `grep -rn "calculate_weights\|w_list" src tests` e atualizar).
- O R² (MF-07) vira não ponderado: `w_arr` deixa de existir; `y_bar = float(np.mean(y_arr))`; `ss_res = float(np.sum((y_arr - y_hat) ** 2))`; `ss_tot = float(np.sum((y_arr - y_bar) ** 2))` (atualizar o comentário MF-07 do bloco).
- Rodar também `tests/test_confidence_r2.py` e `tests/test_multifocus_synthetic.py`: se mudarem valores pinados de R², atualizar citando MF-06 e registrar os novos valores.

Commit: `fix(MF-06): unweighted parabola fit removes value-weight vertex bias`+pin.

- [ ] **Step 5 (MF-11): teste que falha**

```python
def test_vertex_outside_stack_clamps_to_n_minus_1_with_zero_conf():
    """MF-11: clamp era min(n, k) — índice n não existe; e vértice extrapolado
    (fora de [0, n-1]) significa pico não-bracketado -> conf 0."""
    from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy_1d

    fv = np.array([0.1, 0.2, 0.4, 0.7, 0.95, 1.0])  # crescente: vértice além do fim
    k, conf = compute_argmax_fuzzy_1d(fv, [0, 0], {"r_max": 2})
    assert k <= len(fv) - 1, f"k={k} excede o índice máximo válido {len(fv)-1}"
    assert conf == 0, "vértice extrapolado deve ter confiança 0"
```

- [ ] **Step 6 (MF-11): implementar**

No ramo côncavo:

```python
    else:  # calcula o ponto de maximo da funcao
        k_raw = -B / (2 * A)
        if k_raw < 0 or k_raw > n - 1:
            # MF-11: vértice fora do stack — pico não bracketado pelos frames.
            # Clampa ao índice VÁLIDO máximo (n-1, não n) e zera a confiança.
            k_fuzzy = max(0.0, min(float(n - 1), k_raw))
            conf = 0
            fnoc = 0
        else:
            k_fuzzy = k_raw
            fnoc = -(B**2) / (4 * A) + C
            if fnoc < 0:
                conf = 0
            else:
                ...  # bloco R² existente (não ponderado, do Step 4)
```

Commit: `fix(MF-11): clamp vertex to n-1 and zero confidence on extrapolation`+pin.

- [ ] **Step 7: Rodar** `PYTHONPATH=src pytest tests/ -m "not slow" -q` → verde.

---

### Task 12: MF-08 — piso explícito da normalização do stack de foco

**Files:**
- Modify: `src/hybrid_stereo_method/multifocus/indicators/applicator.py`
- Test: `tests/test_multifocus_synthetic.py`

- [ ] **Step 1: Teste que falha**

```python
def test_focus_indicator_normalization_floor_and_degenerate_stack(caplog):
    """MF-08: o piso pós-clip deve ser deslocado a 0 explicitamente (a guarda
    min_val<0 nunca dispara em indicadores |.|>=0), e stack constante (max==0)
    deve avisar e devolver zeros, não dividir silenciosamente."""
    import logging

    from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator

    rng = np.random.default_rng(0)
    stack = rng.uniform(10.0, 255.0, (3, 16, 16))
    fi = focus_indicator(stack, "laplacian", laplacian_kernel_size=5)
    assert fi.min() == 0.0, f"piso não deslocado a 0: min={fi.min()} (MF-08)"
    assert fi.max() == 1.0

    flat = np.full((3, 16, 16), 7.0)  # sem gradiente -> indicador todo zero
    with caplog.at_level(logging.WARNING):
        fi0 = focus_indicator(flat, "laplacian", laplacian_kernel_size=5)
    assert (fi0 == 0).all()
    assert any("no focus signal" in r.message for r in caplog.records)
```

- [ ] **Step 2: Implementar**

Substituir o bloco de normalização (linhas ~83-93) por:

```python
    p1 = np.percentile(focus_indicator_stack, 1)
    focus_indicator_stack = np.clip(focus_indicator_stack, p1, np.inf)

    # MF-08: floor explícito — desloca o mínimo global do stack para 0 (a antiga
    # guarda `if min_val < 0` nunca disparava: indicadores são magnitudes >= 0,
    # então o piso ficava em p1/max). O deslocamento é afim e uniforme entre
    # frames: não altera o argmax nem o vértice da parábola por pixel.
    focus_indicator_stack = focus_indicator_stack - np.min(focus_indicator_stack)

    max_val = np.max(focus_indicator_stack)
    if max_val > 0:
        focus_indicator_stack = focus_indicator_stack / max_val
    else:
        logging.warning(
            "focus indicator stack is constant — no focus signal; returning zeros (MF-08)"
        )
```

- [ ] **Step 3: Rodar** novos + `tests/test_multifocus_synthetic.py` (valores pinados podem mudar marginalmente — atualizar citando MF-08 se preciso) → PASS. **Step 4: Commit + pin** (`fix(MF-08): explicit normalization floor; warn on constant focus stack`).

---

### Task 13: MF-13 — aviso de `z_foc` não-uniforme

**Files:**
- Modify: `src/hybrid_stereo_method/multifocus/main.py`
- Test: `tests/test_multifocus_synthetic.py`

- [ ] **Step 1: Teste (helper puro, sem rodar o main inteiro)**

```python
def test_warn_nonuniform_z_foc(caplog):
    """MF-13: ajuste parabólico em índice + conversão índice->z só é exato com
    espaçamento uniforme; espaçamento não-uniforme deve gerar aviso explícito."""
    import logging

    from hybrid_stereo_method.multifocus.main import check_z_foc_uniformity

    with caplog.at_level(logging.WARNING):
        check_z_foc_uniformity([0.0, 10.0, 20.0, 35.0])
    assert any("non-uniform" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        check_z_foc_uniformity([0.0, 10.0, 20.0, 30.0])
    assert not caplog.records
```

- [ ] **Step 2: Implementar e chamar**

Em `multifocus/main.py`:

```python
def check_z_foc_uniformity(z_foc: list[float]) -> None:
    """Warn when z_foc spacing is non-uniform (MF-13): the sub-frame parabola is
    fitted in INDEX space and converted to z afterwards, which is only exact when
    the index->z map is affine (uniform spacing)."""
    steps = np.diff(np.asarray(z_foc, dtype=np.float64))
    if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        logging.warning(
            "z_foc spacing is non-uniform (steps %s): the parabolic sub-frame fit "
            "is performed in index space and is only exact for uniform spacing "
            "(MF-13) — sub-frame depths may be biased between planes.",
            np.round(steps, 6).tolist(),
        )
```
Chamar logo após a validação de tamanho do `zFoc` (linha ~143): `check_z_foc_uniformity(zFoc)`.

- [ ] **Step 3: Rodar → PASS. Step 4: Commit + pin** (`fix(MF-13): warn when z_foc spacing breaks index-space parabola premise`).

---

### Task 14: MF-10 — `non_linear_res` integrado, com WLS e despacho com erro

**Files:**
- Modify: `src/hybrid_stereo_method/multifocus/indicators/applicator.py`
- Modify: `src/hybrid_stereo_method/multifocus/indicators/non_linear_res.py`
- Test: `tests/test_multifocus_synthetic.py`

- [ ] **Step 1: Testes que falham**

```python
def test_focus_indicator_unknown_type_raises():
    """MF-10: tipo desconhecido caía em UnboundLocalError silencioso."""
    from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator

    with pytest.raises(ValueError, match="Unknown focus_indicator_type"):
        focus_indicator(np.zeros((2, 8, 8)), "tipografado")


def test_non_linear_res_dispatch_and_plane_residual():
    """MF-10: 'non_linear_res' despachável; num plano perfeito o resíduo do
    ajuste de plano é ~0 (o WLS usa os MESMOS pesos W do resíduo)."""
    from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator

    y, x = np.mgrid[0:16, 0:16].astype(np.float64)
    plane = 3.0 * x - 2.0 * y + 5.0
    fi = focus_indicator(np.stack([plane, plane]), "non_linear_res")
    assert fi.shape == (2, 16, 16)
    assert np.allclose(fi, 0.0, atol=1e-8), "plano perfeito deve ter resíduo ~0"
```

- [ ] **Step 2: Implementar**

`applicator.py` — no despacho:

```python
        elif focus_indicator_type == "non_linear_res":
            focus_indicator = calcular_indicador_foco(img)

        else:
            raise ValueError(f"Unknown focus_indicator_type: {focus_indicator_type!r}")
```
(com `from hybrid_stereo_method.multifocus.indicators.non_linear_res import calcular_indicador_foco` no topo).

`non_linear_res.py` — WLS com os pesos da máscara (MF-10):

```python
            # MF-10: o ajuste do plano usa os MESMOS pesos W do resíduo final
            # (WLS via reescala por sqrt(w)); antes o lstsq era não ponderado.
            w = mascara_pesos.flatten().astype(np.float64)
            sw = np.sqrt(w)
            coeficientes, _, _, _ = lstsq(X * sw[:, None], intensidades * sw)
```
(substituindo a chamada `lstsq(X, intensidades)`).

- [ ] **Step 3: Rodar → PASS. Step 4: Commit + pin** (`fix(MF-10): wire non_linear_res into dispatch with WLS; raise on unknown indicator`).

---

### Task 15: INT-02 — remover o fallback silencioso para `-ini-Z.fni`

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/integrate.py`
- Test: `tests/test_integration_units.py`

- [ ] **Step 1: Teste que falha**

```python
def test_missing_end_z_raises_instead_of_returning_initial_guess(tmp_path):
    """INT-02: se o solver 'sucede' sem escrever -00-end-Z.fni, devolver o chute
    inicial (-ini-Z.fni) silenciosamente mascara a falha — deve levantar."""
    from hybrid_stereo_method.hybrid.integrate import integrate_slopes_to_height

    fake = tmp_path / "fake_solver.sh"
    fake.write_text("#!/bin/sh\ncp /dev/null \"$PWD/dummy\"\nexit 0\n")
    fake.chmod(0o755)
    # escreve um -ini-Z.fni que o fallback antigo devolveria
    slopes = np.zeros((4, 4, 3))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x-ini-Z.fni").write_text(
        "begin float_image_t (format of 2006-03-25)\nNC = 1\nNX = 1\nNY = 1\n"
        "    0     0 +0.0000000e+00\n\nend float_image_t\n"
    )
    with pytest.raises(RuntimeError, match="end-Z"):
        integrate_slopes_to_height(slopes, out, "x", executable_path=fake)
```

- [ ] **Step 2: Implementar**

Em `_run_integration`, substituir o bloco de fallback por:

```python
    # Read the output height map written by the solver at level 0.
    # INT-02: no fallback to {prefix}-ini-Z.fni — that file is the INITIAL
    # GUESS (zero or raw hints); silently returning it would disguise a failed
    # integration as a result.
    height_fni_path = output_dir / f"{output_prefix}-00-end-Z.fni"
    if not height_fni_path.exists():
        raise RuntimeError(
            f"Integration finished without writing the final height map "
            f"({height_fni_path.name}) — refusing to fall back to the initial "
            f"guess. Check solver output in: {output_dir}"
        )
```

- [ ] **Step 3: Rodar** o novo + `tests/test_integration_units.py tests/test_convention_integration.py` → PASS. **Step 4: Commit + pin** (`fix(INT-02): missing end-Z output raises instead of returning initial guess`).

---

### Task 16: INT-05 — hints em grade de vértices (H+1, W+1)

**Files:**
- Create: `src/hybrid_stereo_method/hybrid/hints.py`
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (gera hints de vértice a partir de `zMos_avg`/`wSel_avg`)
- Test: `tests/test_integration_units.py`

- [ ] **Step 1: Teste que falha**

```python
def test_cell_to_vertex_grid_no_half_cell_shift():
    """INT-05: o C exigia hints (H+1,W+1) e expandia a grade de células com
    deslocamento de meia célula. A conversão correta média as até 4 células
    adjacentes: numa rampa linear o vértice v vale exatamente a média dos
    centros vizinhos (sem shift)."""
    from hybrid_stereo_method.hybrid.hints import cell_to_vertex_grid

    H = W = 6
    j = np.mgrid[0:H, 0:W][1].astype(np.float64)
    z = 2.0 * j  # rampa em x, valores nos CENTROS das células
    w = np.ones_like(z)
    out = cell_to_vertex_grid(z, w)
    assert out.shape == (H + 1, W + 1, 2)
    # vértices interiores: média das células j-1 e j -> 2*(j-0.5)
    np.testing.assert_allclose(out[1:-1, 3, 0], 2.0 * (3 - 0.5))
    # borda: só a célula disponível
    np.testing.assert_allclose(out[1:-1, 0, 0], 0.0)
    # pesos válidos em toda parte
    assert (out[..., 1] > 0).all()


def test_cell_to_vertex_grid_nan_cells_get_zero_weight():
    from hybrid_stereo_method.hybrid.hints import cell_to_vertex_grid

    z = np.ones((4, 4))
    z[1, 1] = np.nan
    w = np.ones((4, 4))
    out = cell_to_vertex_grid(z, w)
    assert np.isfinite(out[..., 0][out[..., 1] > 0]).all()
```

- [ ] **Step 2: Implementar `hybrid/hints.py`**

```python
"""Conversion of cell-centered multifocus depth maps to the vertex grid the
C integrator expects for hints (INT-05)."""

from __future__ import annotations

import numpy as np


def cell_to_vertex_grid(z_cells: np.ndarray, w_cells: np.ndarray) -> np.ndarray:
    """Convert cell-centered (H, W) height+weight maps to a vertex grid
    (H+1, W+1, 2) by confidence-weighted averaging of the up-to-4 adjacent
    cells of each vertex.

    Fixes INT-05: the C solver demands hints with the height-map (vertex)
    dimensions; feeding the raw (H, W) cell grid made it expand the map with
    ``float_image_expand_by_one``, shifting every hint by half a cell. For a
    linear field, averaging the adjacent cell centers interpolates exactly at
    the vertex position — no shift.

    Cells with non-finite height get weight 0; vertices with no valid adjacent
    cell get height NaN and weight 0 (excluded by the solver).
    """
    if z_cells.shape != w_cells.shape or z_cells.ndim != 2:
        raise ValueError("z_cells and w_cells must be 2-D arrays of equal shape")
    h, w = z_cells.shape
    w_eff = np.where(np.isfinite(z_cells), w_cells, 0.0)
    z_eff = np.where(np.isfinite(z_cells), np.nan_to_num(z_cells), 0.0)

    zw_pad = np.zeros((h + 2, w + 2))
    w_pad = np.zeros((h + 2, w + 2))
    zw_pad[1:-1, 1:-1] = z_eff * w_eff
    w_pad[1:-1, 1:-1] = w_eff

    zw = zw_pad[:-1, :-1] + zw_pad[:-1, 1:] + zw_pad[1:, :-1] + zw_pad[1:, 1:]
    ww = w_pad[:-1, :-1] + w_pad[:-1, 1:] + w_pad[1:, :-1] + w_pad[1:, 1:]

    out = np.full((h + 1, w + 1, 2), np.nan)
    valid = ww > 0
    out[..., 0][valid] = zw[valid] / ww[valid]
    out[..., 1] = np.where(valid, ww / 4.0, 0.0)
    return out
```

- [ ] **Step 3: Usar no `hybrid/main.py`**

No bloco `use_hints`, substituir a localização do arquivo de células por:

```python
        if integration_params.get("use_hints", False):
            # INT-05: build VERTEX-grid hints (H+1, W+1, 2) from the cell-grid
            # multifocus outputs, instead of letting the C side expand the cell
            # grid with a half-cell shift.
            from hybrid_stereo_method.hybrid.hints import cell_to_vertex_grid

            wSel_eff = np.where(np.isfinite(zMos_avg), wSel_avg, 0.0)
            hints_vertex = cell_to_vertex_grid(zMos_avg, wSel_eff)
            os.makedirs(integration_output, exist_ok=True)
            hints_fni_path = os.path.join(integration_output, "hints_vertex.fni")
            convert_image_array_to_fni(hints_vertex, hints_fni_path)
            logging.info(f"Wrote vertex-grid hints map: {hints_fni_path}")
```
(o import pode ir para o topo do arquivo; alinhar a docstring `(H+1, W+1)` de `integrate.py` que agora é verdadeira).

- [ ] **Step 4: Rodar** novos + e2e com `use_hints` se houver teste (test_integration_units cobre INT-04 com hints — rodar) → PASS. **Step 5: Commit + pin** (`fix(INT-05): vertex-grid hints built cell->vertex without half-cell shift`).

---

### Task 17: INT-06 — `reference_scale` para o mapa de referência

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/integrate.py` (config + emissão `scale`)
- Modify: `src/hybrid_stereo_method/hybrid/main.py` (config YAML + aviso)
- Modify: `configs/hb_experiment.yaml`
- Test: `tests/test_integration_units.py`

- [ ] **Step 1: Teste que falha (constrói o cmd sem executar)**

```python
def test_reference_scale_is_emitted_in_command(tmp_path, monkeypatch):
    """INT-06: hAvg.png é uint8 (0-255) e Z sai em unidades físicas; o C aceita
    '-reference R scale S' para tornar a comparação comensurável."""
    import subprocess

    from hybrid_stereo_method.hybrid.integrate import (
        IntegrateRecursiveConfig,
        integrate_slopes_to_height,
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise subprocess.CalledProcessError(1, cmd, stderr="stop-after-capture")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = np.zeros((5, 5))
    cfg = IntegrateRecursiveConfig(reference_scale=0.05)
    with pytest.raises(RuntimeError):
        integrate_slopes_to_height(
            np.zeros((4, 4, 3)), tmp_path, "r", config=cfg, reference_map=ref,
            executable_path="/bin/true",
        )
    cmd = captured["cmd"]
    i = cmd.index("-reference")
    assert cmd[i + 2 : i + 4] == ["scale", "0.05"], cmd
```

- [ ] **Step 2: Implementar**

`IntegrateRecursiveConfig`: adicionar `reference_scale: float = 1.0` (docstring: fator que converte os valores do reference para as unidades da altura integrada — ex.: hAvg.png uint8 0-255 → z físico).

Em `_run_integration`, nos dois ramos de `-reference`, após o extend:

```python
        if config.reference_scale != 1.0:
            cmd.extend(["scale", str(config.reference_scale)])
```

Em `build_integration_config` (hybrid/main.py): `reference_scale=float(integration_params.get("reference_scale", 1.0))`, e aviso quando `use_reference` sem `reference_scale`:

```python
    if integration_params.get("use_reference", False) and "reference_scale" not in integration_params:
        logging.warning(
            "use_reference=True but hybrid.integration.reference_scale is not set: "
            "hAvg.png is uint8 (0-255) while the integrated heights are in physical "
            "units — the C error report (devE) compares incommensurable quantities "
            "(INT-06). Set reference_scale to convert gray levels to height units."
        )
```

Em `configs/hb_experiment.yaml`, seção `hybrid.integration`:

```yaml
    # Factor converting reference (hAvg.png, 0-255 gray) values into the SAME
    # height units as the integrated Z (z_foc units when pixel_size is set).
    # Required for the C error report (devE) to be interpretable (INT-06).
    # E.g. if 255 gray spans 110 z_foc units of height: 110/255 ≈ 0.431.
    # reference_scale: 1.0
```

- [ ] **Step 3: Rodar → PASS. Step 4: Commit + pin** (`fix(INT-06): reference_scale converts uint8 reference into height units`).

---

### Task 18: INT-08 — `-slopes` com 2 canais: promover a 3 no wrapper

**Files:**
- Modify: `src/hybrid_stereo_method/hybrid/integrate.py`
- Modify: `tests/test_convention_integration.py` (remover xfail)

- [ ] **Step 1: Remover o xfail (vira o teste-evidência em verde)**

Em `tests/test_convention_integration.py`, remover o decorator
`@pytest.mark.xfail(...)` de `test_constant_slopes_recover_ramp_and_decide_convention`
(atualizar o comentário do módulo: INT-08 corrigido — o wrapper promove 2→3 canais).

Run: `PYTHONPATH=src pytest tests/test_convention_integration.py -v -s`
Expected: FAIL com o crash `slope map {G} must have 3 channels` (evidência atual).

- [ ] **Step 2: Implementar**

Em `integrate_slopes_to_height`, antes de delegar:

```python
    slope_map = np.asarray(slope_map)
    if slope_map.ndim == 3 and slope_map.shape[2] == 2:
        # INT-08: o topo do C aceita 2 ou 3 canais mas o solver iterativo exige
        # exatamente 3 (canal 2 = peso). Promove 2->3 com peso 1.
        weights = np.ones_like(slope_map[..., :1])
        slope_map = np.concatenate([slope_map, weights], axis=-1)
```
Atualizar a docstring: o mapa é gravado sempre com 3 canais (peso default 1).

- [ ] **Step 3: Rodar**

Run: `PYTHONPATH=src pytest tests/test_convention_integration.py -v -s`
Expected: 2 passed (mais o xfail INT-08 removido) — e a tabela de candidatos impressa
deve eleger `z = +ax*x + ay*y` (consistente com a refutação CONV-1/2). Se o assert de
convenção falhar, PARAR e reportar (não ajustar o teste).
Depois: `PYTHONPATH=src pytest -q` (suíte completa).

- [ ] **Step 4: Commit + pin** (`fix(INT-08): promote 2-channel slope maps to 3 channels (weight=1)`).

---

### Task 19: IO-01 — precisão do writer FNI

**Files:**
- Modify: `src/hybrid_stereo_method/infrastructure/io/image_io.py`
- Test: `tests/test_fni_roundtrip.py`

- [ ] **Step 1: Teste que falha**

```python
def test_roundtrip_preserves_float64_precision(tmp_path):
    """IO-01: %.7e truncava float64 a ~8 dígitos; com %.16e o round-trip
    Python->Python é exato em float64 (o reader aloca float32 — comparar
    contra float32(arr) é o contrato; o ARQUIVO deve carregar float64)."""
    arr = np.array([[0.123456789012345, -9.87654321098765e10]])
    path = tmp_path / "p.fni"
    convert_image_array_to_fni(arr, path)
    text = path.read_text()
    assert "+1.2345678901234500e-01" in text, "writer ainda trunca a 7 casas (IO-01)"
```

- [ ] **Step 2: Implementar** — nas duas f-strings do writer, trocar `:+.7e` por `:+.16e` (e comentário citando IO-01: float64 dos zMos preservado no arquivo; consumidores float32 leem o prefixo).

- [ ] **Step 3: Rodar** `tests/test_fni_roundtrip.py` inteiro → PASS. **Step 4: Commit + pin** (`fix(IO-01): FNI writer keeps full float64 precision (%.16e)`).

---

### Task 20: IO-02 + IO-03 — reader FNI valida completude e linhas malformadas

**Files:**
- Modify: `src/hybrid_stereo_method/infrastructure/io/image_io.py`
- Test: `tests/test_fni_roundtrip.py`

- [ ] **Step 1: Testes que falham**

```python
def test_truncated_fni_raises(tmp_path):
    """IO-02: pixels ausentes ficavam silenciosamente 0."""
    arr = _asymmetric((4, 4))
    path = tmp_path / "t.fni"
    convert_image_array_to_fni(arr, path)
    lines = path.read_text().splitlines(keepends=True)
    # remove duas linhas de dados do meio
    data_idx = [i for i, ln in enumerate(lines) if ln.strip() and ln[0].isspace() is False and ln[0].isdigit() or ln.lstrip()[:1].isdigit()]
    del lines[data_idx[5]]
    del lines[data_idx[4]]
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match="incomplete"):
        read_fni_to_image_array(path)


def test_malformed_short_line_raises(tmp_path):
    """IO-03: linha com menos campos era pulada em silêncio (pixel ficava 0)."""
    path = tmp_path / "m.fni"
    path.write_text(
        "begin float_image_t (format of 2006-03-25)\n"
        "NC = 1\nNX = 2\nNY = 1\n"
        "    0     0 +1.0000000e+00\n"
        "    1\n"  # truncada
        "\nend float_image_t\n"
    )
    with pytest.raises(ValueError, match="[Mm]alformed"):
        read_fni_to_image_array(path)
```

- [ ] **Step 2: Implementar**

No reader:

```python
    filled = np.zeros((ny, nx), dtype=bool)

    for line in lines:
        if line.strip() == "" or line.startswith("begin") or line.startswith("end") or "=" in line:
            continue
        parts = line.split()
        if len(parts) < 2 + nc:
            # IO-03: linha de dados truncada não pode ser silenciada (o pixel
            # ficaria 0, indistinguível de valor legítimo — IO-02).
            raise ValueError(
                f"Malformed FNI data line (expected {2 + nc} fields, got {len(parts)}): {line!r}"
            )
        ...
        filled[y, x] = True

    if not filled.all():
        missing = int((~filled).sum())
        raise ValueError(
            f"FNI file incomplete: {missing} of {ny * nx} pixels missing (IO-02)"
        )
    return image_array
```

- [ ] **Step 3: Rodar** `tests/test_fni_roundtrip.py` + suíte rápida → PASS. **Step 4: Dois commits** (`fix(IO-02): FNI reader verifies all pixels were filled`; `fix(IO-03): malformed short FNI data lines raise instead of silent skip`) — ou um commit `fix(IO-02,IO-03)` se a separação ficar artificial (mesmas linhas) + pins.

---

### Task 21: IO-06 — chamadas de `save_image` com 5 posicionais

**Files:**
- Modify: `src/hybrid_stereo_method/multifocus/image_alignment.py`

(Código morto sem callers — correção no lugar, sem TDD; verificação por compilação/lint.)

- [ ] **Step 1: Corrigir as 3 chamadas**

```python
    save_image(save_path, ref_save_as, reference_img, normalize=False)
    ...
        save_image(match_path, match_save_as, imMatches, normalize=False)
        save_image(save_path, align_save_as, aligned_img, normalize=False)
```
(o `0, 255` antigo significava "preservar faixa 0-255" ⇒ `normalize=False`.)

- [ ] **Step 2: Verificar** `python -m py_compile src/hybrid_stereo_method/multifocus/image_alignment.py && ruff check src/` → ok.

- [ ] **Step 3: Commit + pin** (`fix(IO-06): correct save_image call signature in image_alignment`).

---

### Task 22: PS-09 — `convert_to_grayscale` aceita entrada mono

**Files:**
- Modify: `src/hybrid_stereo_method/infrastructure/utils.py`
- Test: `tests/test_photometric_synthetic.py`

- [ ] **Step 1: Teste que falha**

```python
def test_convert_to_grayscale_passthrough_for_mono():
    """PS-09: imagem já monocromática (2-D ou HxWx1) não pode crashar o cvtColor."""
    from hybrid_stereo_method.infrastructure.utils import convert_to_grayscale

    mono = np.full((8, 8), 7, dtype=np.uint8)
    out = convert_to_grayscale(mono)
    assert out.shape == (8, 8)
    np.testing.assert_array_equal(out, mono)

    mono1 = mono[..., None]
    out1 = convert_to_grayscale(mono1)
    assert out1.shape == (8, 8)
```

- [ ] **Step 2: Implementar**

```python
def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR image to grayscale (Rec.601 weights, BGR order).

    Already-monochrome inputs (2-D or HxWx1) pass through unchanged (PS-09 —
    cvtColor raises on single-channel input). Note: the legacy ``rps`` path
    (``ps_utils.converter_npy_para_cinza``) uses a divergent RGB-vs-BGR
    heuristic; this function is the canonical policy for the hybrid pipeline.
    """
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- [ ] **Step 3: Rodar → PASS. Step 4: Commit + pin** (`fix(PS-09): grayscale conversion passes mono inputs through; document policy`).

---

### Task 23: PS-10 — solver argmax (código morto) sem `inv` nu

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/wps.py` (`estimate_normals_argmax`)
- Test: `tests/test_photometric_synthetic.py`

- [ ] **Step 1: Teste que falha**

```python
def test_argmax_solver_handles_coplanar_lights():
    """PS-10: 3 luzes quase coplanares — inv() explode/instável; lstsq com rank
    check devolve NaN em vez de lixo."""
    from hybrid_stereo_method.photometric.wps import estimate_normals_argmax

    lights = np.array([[0.0, 0.5, 0.866], [0.0, 0.5, 0.866], [0.0, 0.5, 0.8660001]])
    images = [np.full((4, 4), v) for v in (100.0, 100.0, 100.0)]
    normals, _ = estimate_normals_argmax(images, lights)
    assert np.isnan(normals[0, 0]).all() or np.isfinite(normals[0, 0]).all()
    # nunca valores absurdos:
    finite = normals[np.isfinite(normals)]
    if finite.size:
        assert np.abs(finite).max() <= 1.0 + 1e-6
```

- [ ] **Step 2: Implementar** — substituir o solve:

```python
            # PS-10: lstsq com verificação de posto no lugar de inv() nu — as 3
            # luzes mais brilhantes podem ser quase coplanares (sistema singular).
            normal, _, rank, _ = np.linalg.lstsq(selected_lights, selected_values, rcond=None)
            norm = np.linalg.norm(normal)
            if rank < 3 or norm == 0 or not np.isfinite(norm):
                normals[i, j, :] = np.nan
                continue
            normal /= norm
            normals[i, j, :] = normal
```

- [ ] **Step 3: Rodar → PASS. Step 4: Commit + pin** (`fix(PS-10): argmax solver uses rank-checked lstsq instead of bare inv`).

---

### Task 24: PS-03 — outliers por mediana+MAD (remove o xfail de saturação)

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/wps.py`
- Modify: `tests/test_photometric_synthetic.py` (remover xfail PS-03)

- [ ] **Step 1: Remover o xfail**

Remover `@pytest.mark.xfail(...)` de `test_wps_robust_to_saturation`.
Run: `PYTHONPATH=src pytest tests/test_photometric_synthetic.py::test_wps_robust_to_saturation -q` → FAIL (erro 15.13° > 5°).

- [ ] **Step 2: Implementar**

No laço robusto, substituir:

```python
                r_avg = np.mean(residuals)  # Step (4): Compute average residual

                # Step (5): Discard outliers
                mask = residuals <= outlier_threshold_multiplier * r_avg
```
por:

```python
                # PS-03: limiar robusto — mediana + k*1.4826*MAD. A média era
                # inflada pelo próprio outlier (mascaramento), deixando
                # saturações passarem.
                r_med = np.median(residuals)
                mad = np.median(np.abs(residuals - r_med))
                if mad == 0.0:
                    break  # resíduos (quase) idênticos: nada a rejeitar
                mask = residuals <= r_med + outlier_threshold_multiplier * 1.4826 * mad
```

- [ ] **Step 3: Rodar** `tests/test_photometric_synthetic.py` inteiro `-v -s` → tudo PASS (registrar o novo erro angular impresso para o relatório). **Step 4: Commit + pin** (`fix(PS-03): median+MAD outlier rejection replaces mean-based threshold`).

---

### Task 25: PS-01 + PS-04 + PS-05 — albedo ρ=||m||, resíduos finais, confiança invariante a ganho

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/wps.py`
- Modify: `tests/test_photometric_synthetic.py` (remover xfail PS-01; novo teste de ganho)

- [ ] **Step 1 (PS-01): remover o xfail e ver falhar**

Remover `@pytest.mark.xfail(...)` de `test_wps_albedo_recovers_true_albedo`.
Run → FAIL (albedo ~2 vs 200).

- [ ] **Step 2 (PS-01/PS-04/PS-05): implementar**

Substituir o bloco final (após o `while`) por:

```python
            if len(selected_values) >= 3:
                # PS-01: no modelo I = rho*(L.n̂), o lstsq devolve m = rho*n̂ —
                # o albedo é ||m|| e a normal é m/||m||. (O antigo ||L@n̂|| era a
                # norma das intensidades preditas, crescendo com sqrt(n_luzes).)
                m = normal
                rho = np.linalg.norm(m)
                if rho == 0 or not np.isfinite(rho):
                    normals[i, j, :] = np.nan
                    confidence[i, j] = 0
                    continue
                normal = m / rho
                normals[i, j, :] = normal
                albedo[i, j] = rho

                # PS-04: resíduos RECOMPUTADOS do modelo final sobre o conjunto
                # final (antes, eram os do ajuste anterior à última remoção).
                residuals = np.abs(np.dot(selected_lights, m) - selected_values)

                # PS-05: resíduo normalizado pela escala do sinal (albedo) torna
                # a confiança invariante a ganho radiométrico (0-255 vs 0-1).
                N = len(selected_values)
                M = num_images
                residual_std = np.std(residuals) / (rho + epsilon)
                confidence[i, j] = (N / M) * (1 / (1 + residual_std))

                selected_areas[i, j, original_indices] = 255
```

- [ ] **Step 3 (PS-05): teste de invariância a ganho**

```python
def test_confidence_invariant_to_radiometric_gain():
    """PS-05: a confiança não pode depender de a entrada estar em 0-255 ou 0-1."""
    size = 24
    n_gt = normals_from_height(gaussian_bump(size, amplitude=3.0))
    lights = ring_lights(6, tilt_deg=30.0)
    rng = np.random.default_rng(0)
    images255 = [
        render_lambertian(n_gt, light, albedo=200.0) + rng.normal(0, 1.0, (size, size))
        for light in lights
    ]
    images01 = [img / 255.0 for img in images255]

    _, _, conf255, _ = estimate_normals_argmax_lstsq_robust(images255, lights, {})
    _, _, conf01, _ = estimate_normals_argmax_lstsq_robust(images01, lights, {})
    np.testing.assert_allclose(conf255, conf01, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 4: Rodar** `tests/test_photometric_synthetic.py -v -s` completo → PASS. **Step 5: três commits + pins** (`fix(PS-01): albedo = ||m|| from the unnormalized lstsq solution`; `fix(PS-04): confidence uses residuals of the final fit and final set`; `fix(PS-05): residuals normalized by albedo make confidence gain-invariant`). Se a separação em 3 commits ficar artificial (mesmo bloco), usar `fix(PS-01,PS-04,PS-05): ...` e pinar os três IDs no mesmo docs commit.

---

### Task 26: PS-11 — visualização sem mutação in-place e segura em headless

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/visualization.py`
- Modify: `src/hybrid_stereo_method/photometric/main_wps.py` (não monkeypatch mais necessário nos testes, mas mantê-los é inofensivo)
- Test: `tests/test_photometric_synthetic.py`

- [ ] **Step 1: Teste que falha**

```python
def test_disp_functions_do_not_mutate_input_and_are_headless_safe(tmp_path):
    """PS-11: disp_* trocavam canais IN-PLACE numa view do array do chamador e
    bloqueavam com cv2.imshow+waitKey(0). Default: só salvar, sem display."""
    from hybrid_stereo_method.photometric.visualization import (
        disp_channels,
        disp_normalmap,
    )

    normals = np.random.default_rng(0).uniform(-1, 1, (8, 8, 3))
    normals[0, 0] = np.nan  # sombra: não pode quebrar a visualização
    before = normals.copy()

    disp_normalmap(normal=normals, height=8, width=8, save_path=str(tmp_path))
    disp_channels(normal_in=normals, height=8, width=8, save_path=str(tmp_path))

    np.testing.assert_array_equal(
        np.nan_to_num(normals), np.nan_to_num(before)
    ), "disp_* mutou o array do chamador (PS-11)"
    assert (tmp_path / "normal_map.png").exists()
    assert (tmp_path / "Channels.png").exists()
```

Run → FAIL (trava no `waitKey(0)` — rodar com timeout) — na prática o teste já
falha por bloqueio; se o runner não suportar, validar a mutação com
`display=False` após implementar e tratar o bloqueio como verificado por inspeção.

- [ ] **Step 2: Implementar**

Nas três funções `disp_*`:
1. Parâmetro novo `display: bool = False`; todo o bloco `cv2.imshow/waitKey/destroyWindow` entra em `if display:`.
2. Operar sobre cópia: `N = np.reshape(normal, (height, width, 3)).copy()` (idem `disp_channels`/`disp_channels_3d`).
3. NaN nas normais: `N = np.nan_to_num(N, nan=0.0)` antes do rescale para imagem.
4. Docstrings: registrar PS-11 (default save-only; `display=True` para interativo).

- [ ] **Step 3: Rodar** o teste novo + `tests/test_e2e_hybrid.py` (os monkeypatches dos testes e2e continuam válidos) → PASS. **Step 4: Commit + pin** (`fix(PS-11): visualization works on copies and is save-only by default`).

---

### Task 27: CONV-1 + CONV-2 — convenção explícita e chave `flip_lights_y`

As metades "físicas" de CONV-1/CONV-2 não são decidíveis sem dados reais (conclusão da auditoria). A correção possível é (a) tornar a reconciliação do eixo-y das luzes EXPLÍCITA e testável via config, e (b) documentar a convenção adotada ponta a ponta.

**Files:**
- Modify: `src/hybrid_stereo_method/photometric/main_wps.py`
- Modify: `configs/hb_experiment.yaml`, `configs/wps_experiment.yaml`
- Test: `tests/test_photometric_synthetic.py`

- [ ] **Step 1: Teste que falha**

```python
def test_reconcile_lights_flip_y():
    """CONV-2: lights.npy reais são POV-Ray (y-up); a imagem numpy é y-down.
    A reconciliação deve ser uma chave explícita, não implícita."""
    from hybrid_stereo_method.photometric.main_wps import reconcile_lights

    lights = np.array([[0.1, 0.2, 0.97], [-0.3, -0.4, 0.86]])
    out = reconcile_lights(lights, flip_y=True)
    np.testing.assert_allclose(out[:, 0], lights[:, 0])
    np.testing.assert_allclose(out[:, 1], -lights[:, 1])
    np.testing.assert_allclose(out[:, 2], lights[:, 2])
    # default: identidade, e NÃO muta a entrada
    out2 = reconcile_lights(lights, flip_y=False)
    np.testing.assert_allclose(out2, lights)
```

- [ ] **Step 2: Implementar**

Em `main_wps.py`:

```python
def reconcile_lights(light_sources: np.ndarray, flip_y: bool) -> np.ndarray:
    """Reconcile the light-direction frame with the numpy image frame (CONV-2).

    The pipeline convention (tests/synthetic_utils.py, decided by the ramp
    convention test) is: image rows = y growing DOWNWARD, normals
    n = (-dz/dx, -dz/dy, 1)/|.|, z growing toward the camera (CONV-1). Real
    ``lights.npy`` generated in y-up frames (e.g. POV-Ray) must have their y
    component negated — set ``photometric.flip_lights_y: true``.
    """
    if not flip_y:
        return light_sources
    out = np.array(light_sources, copy=True)
    out[:, 1] = -out[:, 1]
    logging.info("flip_lights_y: negated lights y-axis (y-up -> y-down, CONV-2)")
    return out
```

Após `light_sources = np.load(light_path)`:

```python
    light_sources = reconcile_lights(
        light_sources,
        flip_y=bool(parameters.get("photometric", {}).get("flip_lights_y", False)),
    )
```

Em `configs/hb_experiment.yaml`, seção `photometric`:

```yaml
  # Set true when lights.npy was generated in a y-up frame (e.g. POV-Ray):
  # the image frame is y-down (numpy rows), so the lights' y must be negated
  # to keep lights and normals in the same frame (CONV-2). Decide with a real
  # ground-truth comparison (integrate with both settings, keep the one whose
  # height map matches the physical surface orientation).
  flip_lights_y: false
```

- [ ] **Step 3: Rodar → PASS. Step 4: Commit + pin** — no docs commit, atualizar CONV-1 e CONV-2 para "mitigado: reconciliação explícita via `flip_lights_y`; decisão física continua exigindo dado real (procedimento documentado no config)" (`fix(CONV-2): explicit flip_lights_y reconciles y-up light frames; document CONV-1 convention`).

---

### Task 28: Build do C + INT-07 — `-maxLevel` default

**Files:**
- Modify: `csrc/integrate_recursive/gus_integrate_recursive.c`
- Modify (se necessário): `csrc/integrate_recursive/Makefile` / headers
- Binário: `csrc/integrate_recursive/gus_integrate_recursive` (recompilado)

- [ ] **Step 1: Diagnosticar o build**

```bash
cd csrc/integrate_recursive && make 2>&1 | tee /tmp/make.log
```
Estado conhecido: `make` falha hoje com `No rule to make target 'include/affirm.h'` (no worktree) e, segundo a auditoria, com `-Werror=comment` (comentário aninhado nas linhas 8-9) e include `<bool.h>`. Investigar na ordem: (1) regra `%.ho: %.h` — os `.h` existem em `include/`; provável problema de timestamps/ordem no worktree (`touch include/*.ho` resolve se os `.ho` forem pré-compilados válidos); (2) comentário aninhado nas linhas 8-9 do `gus_integrate_recursive.c` — converter para `//`:

```c
/* Last edited on 2025-04-03 16:59:20 by stolfi */
// Option 1: Use "zero" as initialization method
// ./gus_integrate_recursive -initial zero 0 -outPrefix teste_blabla -normals .../normal_map_with_residuals.fni
```

- [ ] **Step 2: INT-07 — corrigir o default**

Na linha ~783:

```c
    if (argparser_keyword_present(pp, "-maxLevel"))
      { o->maxLevel = (uint32_t)argparser_get_next_int(pp, 0, INT64_MAX); }
    else
      { o->maxLevel = DEFAULT_MAX_LEVEL; }
```
(era `DEFAULT_MAX_ITER` — copy-paste; `DEFAULT_MAX_LEVEL=30` definido e nunca usado).

- [ ] **Step 3: Rebuild e validar o binário**

```bash
cd csrc/integrate_recursive && make build-prog && cd ../..
PYTHONPATH=src pytest tests/test_convention_integration.py tests/test_integration_units.py tests/test_e2e_hybrid.py -q
```
Expected: build ok e TODOS os testes binário-dependentes verdes com o binário novo.
**Se o build não fechar** (dependências do sistema ausentes): commitar só a correção
de fonte, manter o binário antigo (INT-07 é inativo no fluxo Python — `-maxLevel`
sempre fornecido) e registrar no relatório que o binário não foi recompilado.

- [ ] **Step 4: Commit + pin** (`fix(INT-07): -maxLevel default uses DEFAULT_MAX_LEVEL; unbreak C build` — incluir o binário recompilado no commit se o rebuild passou).

---

### Task 29: Consolidação final — relatório, baseline e suíte

**Files:**
- Modify: `docs/superpowers/reports/2026-06-04-method-audit.md`
- Modify: `docs/superpowers/reports/audit-notes/06-test-results.md`

- [ ] **Step 1: Suíte completa e lint**

```bash
PYTHONPATH=src pytest -q          # tudo verde, ZERO xfail restante dos 4 evidência
ruff check src/ tests/ && ruff format --check src/ tests/
mypy src/ || true                 # registrar estado (não bloquear por erros pré-existentes)
```

- [ ] **Step 2: Re-medir a baseline E2E**

```bash
PYTHONPATH=src pytest tests/test_e2e_hybrid.py -v -s 2>&1 | grep "BASELINE"
```
Registrar RMSE/a/b/r novos em `06-test-results.md` (seção nova `## Baseline pós-correções`)
e na seção 4 do relatório, comparando com 0.0742→0.0691 da auditoria.

- [ ] **Step 3: Atualizar o sumário executivo do relatório**

- Tabela 1.1: recontar — todos os achados ativos com status `corrigido (<sha>)`; manter os 2 refutados; CONV-1/CONV-2 como `mitigado` (metade física aberta, procedimento documentado); REG-01 adicionado e corrigido.
- Seção 1.2: anotar em cada um dos 5 destaques o SHA da correção.
- Tabela de convenções (§3): vereditos finais (4 → consistente/corrigido; 5 → corrigido; 6 → corrigido).
- §4 Bloqueios: atualizar o estado do build do C conforme a Task 28.
- §5.3: acrescentar os novos arquivos de teste.

- [ ] **Step 4: Commit final**

```bash
git add docs/superpowers/reports/
git commit -m "docs(audit): consolidate corrections — all active findings fixed, new E2E baseline"
```

- [ ] **Step 5: Apresentar ao usuário** o resumo (achados corrigidos × SHAs, baseline antes/depois, o que permanece aberto por exigir dados reais: metades físicas de CONV-1/CONV-2 e calibração de `pixel_size`/`reference_scale`/`flip_lights_y` nos datasets reais).

---

## Self-review (executado na escrita)

- **Cobertura:** 28 achados remanescentes mapeados: MF-03 (T10), MF-04 (T10), MF-05/06/11 (T11), MF-08 (T12), MF-10 (T14), MF-13 (T13), MF-14 (T2), PS-01/04/05 (T25), PS-02 (T6), PS-03 (T24), PS-06 (T7), PS-07 (T8), PS-08 (T9), PS-09 (T22), PS-10 (T23), PS-11 (T26), INT-01 (T5), INT-02 (T15), INT-03 (T7), INT-05 (T16), INT-06 (T17), INT-07 (T28), INT-08 (T18), IO-01 (T19), IO-02/03 (T20), IO-04/05 (fechados via T8 — pin no docs da T8), IO-06 (T21), CONV-1/2 (T27), CONV-5 (T8+T9), CONV-6 (T4) + REG-01 (T3) + build C (T28).
- **xfails:** os 4 viram verdes: PS-01 (T25), PS-03 (T24), MF-14 (T2), INT-08 (T18).
- **Tipos/assinaturas:** `mosaic(..., wSel=None, min_confidence=0.0)` usado por T10 e consumidores; `collect_light_dirs(files, data_path)`; `pair_mosaics_to_lights(paths, n_lights)`; `build_integration_config` ganha validação INT-01 + `reference_scale`; `IntegrateRecursiveConfig.reference_scale`.
- **Riscos declarados:** thresholds de testes sintéticos existentes podem mudar com MF-06/MF-08 (passos preveem atualização justificada, nunca afrouxamento cego); o assert de convenção da rampa (T18) é um STOP se falhar; rebuild do C tem fallback documentado.
