# Method Correctness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Executar a auditoria de corretude definida em `docs/superpowers/specs/2026-06-04-method-audit-design.md`, produzindo o relatório de achados e a suíte de testes sintéticos permanente.

**Architecture:** Três fases — (1) leitura crítica por estágio com checklists e registro de achados em notas de trabalho; (2) rastreamento transversal de 6 convenções produtor→consumidor; (3) testes sintéticos com ground truth analítico em `tests/` que confirmam/refutam os achados. Consolidação final em um relatório único.

**Tech Stack:** Python 3 (numpy, opencv, pytest), binário C `gus_integrate_recursive`.

**Natureza das tarefas:** Tarefas 2–6 são de *análise* — o "código" delas é o registro estruturado de achados em arquivos de notas. Tarefas 7–12 são de *código de teste* (TDD invertido: o teste é escrito para passar se o método estiver correto; a falha dele é a evidência do achado — nunca "conserte" o teste para ele passar, registre o achado). Tarefa 13 consolida.

**Regra permanente:** nesta auditoria NÃO se corrige código de produção. Se encontrar um bug, registre o achado; a correção é trabalho futuro, fora deste plano.

---

## Pré-requisitos (verificar antes da Tarefa 1)

```bash
cd /home/lelis/Documents/Projetos/hybrid-stereo-method
pip install -e ".[dev]"          # se ainda não instalado
python -c "import hybrid_stereo_method, cv2, numpy; print('ok')"
```
Expected: `ok`

---

## Estrutura de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `docs/superpowers/reports/audit-notes/TEMPLATE.md` | Template de achado (criado na Tarefa 1, usado por todas) |
| `docs/superpowers/reports/audit-notes/01-multifocus.md` | Achados Fase 1.1 |
| `docs/superpowers/reports/audit-notes/02-photometric.md` | Achados Fase 1.2 |
| `docs/superpowers/reports/audit-notes/03-integration.md` | Achados Fase 1.3 |
| `docs/superpowers/reports/audit-notes/04-io.md` | Achados Fase 1.4 |
| `docs/superpowers/reports/audit-notes/05-conventions.md` | Tabela de convenções + achados Fase 2 |
| `docs/superpowers/reports/audit-notes/06-test-results.md` | Resultados dos testes sintéticos e baseline E2E |
| `tests/synthetic_utils.py` | Geradores sintéticos compartilhados (superfícies, normais, render, stack de foco, fit afim) |
| `tests/test_synthetic_utils.py` | Sanidade dos geradores |
| `tests/test_fni_roundtrip.py` | Fase 3.2 — round-trip FNI Python↔Python |
| `tests/test_convention_integration.py` | Fase 3.2 — convenções Python↔C (rampa, bump) |
| `tests/test_photometric_synthetic.py` | Fase 3.1 — fotométrico |
| `tests/test_multifocus_synthetic.py` | Fase 3.1 — multifocus |
| `tests/test_e2e_hybrid.py` | Fase 3.3 — pipeline completo |
| `docs/superpowers/reports/2026-06-04-method-audit.md` | Relatório final consolidado |

---

### Task 1: Scaffolding das notas de auditoria

**Files:**
- Create: `docs/superpowers/reports/audit-notes/TEMPLATE.md`

- [ ] **Step 1: Criar o template de achado**

```markdown
# Template de achado de auditoria

Copie o bloco abaixo para registrar cada achado. IDs por estágio: MF-xx (multifocus),
PS-xx (fotométrico), INT-xx (integração), IO-xx (infra/IO), CONV-xx (convenções).
Numere sequencialmente dentro de cada prefixo.

---

## <ID>: <título curto>

- **Localização:** `caminho/arquivo.py:linha`
- **Tipo:** conceitual | implementação
- **Severidade:** crítico | alto | médio | baixo
- **Status:** suspeita | confirmado (ref. ao teste) | refutado (ref. ao teste)

**Descrição:** o que está errado e por quê (1 parágrafo; cite a teoria quando o tipo
for conceitual).

**Evidência:** trecho de código, raciocínio matemático, ou nome do teste sintético.

**Sugestão de correção:** o que mudar (NÃO aplicar).

---

Critérios de severidade (do spec):
- crítico: corrompe o resultado científico
- alto: erro mensurável no resultado
- médio: degrada robustez/precisão em casos comuns
- baixo: caso de borda

Critério de inclusão: só corretude, precisão ou robustez. Estilo/performance ficam fora.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/reports/audit-notes/TEMPLATE.md
git commit -m "audit: add findings template and notes scaffolding"
```

---

### Task 2: Auditoria Fase 1.1 — Multifocus

**Files:**
- Read: `src/hybrid_stereo_method/multifocus/indicators/laplacian.py`, `fourier.py`, `wavelet.py`, `non_linear_res.py`, `applicator.py`
- Read: `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py`, `depth_refinement.py`, `mosaic.py`, `image_alignment.py`, `math_utils.py`, `main.py`, `utils.py`
- Read: `src/hybrid_stereo_method/hybrid/main.py:86-163` (preparação dos stacks e mosaicos por luz)
- Create: `docs/superpowers/reports/audit-notes/01-multifocus.md`

- [ ] **Step 1: Ler os indicadores de foco e responder o checklist**

Para cada indicador (laplacian, fourier, wavelet, non_linear_res) e para `applicator.focus_indicator`:
1. A medida é *local* (janela/kernel) ou global? Medida global de nitidez não serve para seleção por pixel.
2. Como bordas da imagem são tratadas (padding do filtro introduz falsa alta frequência)?
3. A normalização do stack em `applicator.py:80-93` (clip no percentil 1 global + divisão pelo max global) preserva a comparabilidade *entre frames*? Subtrair `min_val` só quando `min_val < 0` e dividir por `max_val` só quando `> 0` tem casos degenerados?
4. Regiões sem textura: qual o valor do indicador e o que `argmax_fuzzy` faz com ele?

- [ ] **Step 2: Ler `argmax_fuzzy.py` e responder o checklist**

1. `compute_argmax_fuzzy_1d` (linhas 121-208): o ajuste parabólico assume espaçamento uniforme dos frames (x = índices inteiros). O `z_foc` real é uniforme? Onde a não-uniformidade entraria (a conversão índice→z é feita depois, no `mosaic`)? A composição parabola-em-índice + interpolação-em-z é equivalente a ajustar em z?
2. Linha 181: `k_fuzzy = max(0, min(n, k_fuzzy))` — o índice válido máximo é `n-1`, não `n`. Consequência no `mosaic` (que faz clamp próprio)?
3. Linha 128-129: `if focus_values[k_max] == 0: return n / 2, 0` — devolve o índice do meio com confiança 0; mas `mosaic` usa `iSel` SEM olhar confiança. Pixels sem textura viram profundidade `z_foc[n/2]`. Conceitual?
4. `calculate_weights` (104-118): usar os próprios valores de foco como pesos da regressão parabólica enviesa o vértice? Justificativa teórica?
5. `find_index_of_max_sum` (81-101): a janela de soma-de-3 pode escolher um índice diferente do argmax verdadeiro? Comportamento com picos duplos?
6. Confiança `conf = abs(A) / fnoc` (linha 187): unidades/escala dependem da normalização do stack — comparável entre pixels?

- [ ] **Step 3: Ler `mosaic.py`, `math_utils.py` e `depth_refinement.py` e responder o checklist**

1. `mosaic.py:72`: `zMos[i,j] = interpolate(zFoc, k_fuzzy)` — verificar `linear_interpolation`/`quadratic_interpolation` em `math_utils.py`: indexam `zFoc` corretamente para `k_fuzzy` fracionário? Extrapolam?
2. `mosaic` ignora a confiança `wSel` — pixels com conf 0 entram no mosaico e no `zMos` como se válidos.
3. `depth_refinement.py` (graph-cut): termo de dados e suavidade em escalas compatíveis? Labels discretos destroem o sub-pixel do `iSel`? Em que ponto do pipeline é chamado (é chamado?) — verificar se `main.py` o usa.
4. `image_alignment.py`: é usado no pipeline? Se sim, o mesmo alinhamento é aplicado ao stack médio (que gera `iSel`) e aos stacks por luz (que geram os mosaicos)? Desalinhamento entre eles invalida o reuso do `iSel`.

- [ ] **Step 4: Verificar os leads específicos já identificados na preparação do plano**

Investigar e registrar veredito (confirma a suspeita pela leitura ou refuta) para cada um:

1. **`hybrid/main.py:101-103`** — filtro por substring `if f"{zf_dir}" in file`: com ≥10 planos focais, `"zf1"` casa com `".../zf10/..."` — a média do zf1 incluiria imagens do zf10. (O commit `505d262` corrigiu isso para `L*` mas aparentemente não para `zf*`.)
2. **`hybrid/main.py:90-96`** — `zf_directories = sorted(...)` é lexicográfico: com ≥10 planos, a ordem vira `zf1, zf10, zf2, ...`, desalinhando `average_images_paths` da lista `z_foc` do YAML (que está em ordem natural). Mesmo problema em `filtered_files = sorted(...)`.
3. **`hybrid/main.py:111`** — `save_image(..., f"average_{zf_dir}.png", average_image)` usa `normalize=True` (default): cada média por-zf é esticada min-max independentemente ANTES da medida de foco. Isso altera a relação de contraste entre frames — a curva de foco medida sobre PNGs re-esticados ainda tem o pico no lugar certo? (O pico de nitidez é por estrutura local, mas o stretch muda a amplitude relativa entre frames; analisar o efeito sobre o ajuste parabólico.)
4. **Quantização**: as médias passam por PNG uint8 (`save_image` → `read_images`) — perda de precisão antes da medida de foco.

- [ ] **Step 5: Registrar achados em `01-multifocus.md`**

Criar o arquivo com cabeçalho `# Fase 1.1 — Achados Multifocus` + data, e um bloco por achado no formato do `TEMPLATE.md`. Perguntas do checklist cuja resposta for "está correto" devem virar uma linha numa seção final `## Verificado sem achado` (para o apêndice do relatório).

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/audit-notes/01-multifocus.md
git commit -m "audit: phase 1.1 multifocus findings"
```

---

### Task 3: Auditoria Fase 1.2 — Fotométrico

**Files:**
- Read: `src/hybrid_stereo_method/photometric/wps.py`, `rps.py`, `ps_utils.py`, `solvers/numerics.py`, `main.py`, `main_wps.py`, `visualization.py`
- Read: `src/hybrid_stereo_method/infrastructure/utils.py` (`convert_to_grayscale`, `normalize`)
- Create: `docs/superpowers/reports/audit-notes/02-photometric.md`

- [ ] **Step 1: Ler `wps.py` e responder o checklist**

1. `estimate_normals_argmax_lstsq_robust` (linha 112+): montagem de `I = L·n` — `lstsq(selected_lights, selected_values)` devolve `n` NÃO normalizado cuja norma é o albedo (modelo `I = ρ·(L·n̂)`). Mas a linha 198 calcula `albedo[i,j] = np.linalg.norm(np.dot(selected_lights, normal))` com `normal` JÁ normalizado — isso é a norma do vetor de intensidades preditas, que cresce com `sqrt(nº de luzes)` e não é o albedo ρ. Conceitual?
2. Rejeição de sombra (linha 152): threshold *relativo ao máximo do pixel* (`pixel_values/v_max > shadow_threshold` com default `1e-3`) — com 8 bits isso rejeita quase nada (1/255 ≈ 4e-3 > 1e-3 sempre que houver qualquer sinal). O critério rejeita sombras reais? Highlights/saturação nunca são rejeitados — só o loop de outliers cobre isso?
3. Loop de outliers (163-185): threshold `3×média(|residual|)`; terminação garantida (conjunto só encolhe)? O `normal` usado após `break` interno por `<3` é descartado corretamente (verificar o guard da linha 187)?
4. Confiança `(N/M) * 1/(1+residual_std)` (linha 204): escala dos resíduos depende da faixa de intensidade (0-255 vs 0-1) — confiança não invariante a ganho. Consequência para uso posterior?
5. Pixels com NaN em `normals` (sombra/degenerado): quem consome `normal_map.npy` depois (integração!) e o que faz com NaN? (registrar como lead para INT/CONV).
6. `estimate_normals_argmax` (linha 8): `np.linalg.inv(selected_lights)` — sem guard de singularidade; as 3 luzes mais brilhantes podem ser quase coplanares. Esta função é usada em algum entry point?

- [ ] **Step 2: Ler `rps.py` e `solvers/numerics.py` e responder o checklist**

1. Formulações L2/L1/SBL/RPCA: a montagem da matriz de medidas (pixels × imagens) e a recuperação de `N` seguem a formulação padrão (Ikehata et al. para SBL/RPCA)? Máscara aplicada de forma consistente?
2. Convenções de saída do `rps` (orientação/normalização das normais, ordem dos canais) são as MESMAS do `wps`? `photometric/main.py` e `main_wps.py` produzem `normal_map.npy` intercambiáveis?
3. `solvers/numerics.py`: estabilidade numérica (divisões, normas zero, rcond).

- [ ] **Step 3: Ler `main_wps.py` + radiometria de entrada e responder o checklist**

1. Cadeia radiométrica: `hybrid/main.py:160` salva `sMos.png` com `normalize=False` (clip 0-255 uint8) e `main_wps` lê o PNG → a entrada do PS está quantizada a 8 bits (e o `.fni` float que existe não é usado). Perda de precisão relevante?
2. `convert_to_grayscale` em `infrastructure/utils.py`: coeficientes corretos para a ordem BGR do cv2? Funciona com float e uint8?
3. Linearidade: os `sVal.png` de entrada são assumidos lineares (sem gamma)? Há algum ponto onde gamma é aplicado/removido? Modelo Lambertiano exige intensidades lineares.
4. `main_wps.py:135` passa `wps_params = parameters["photometric"]["solver"]` — os nomes de chave do YAML (`configs/*.yaml`) batem com os `wps_params.get(...)` do `wps.py`? (chave errada = default silencioso).
5. Avaliação com ground truth (162-177): `evaluate_angular_error` em `ps_utils.py` — fórmula correta (arccos do dot com clip [-1,1])? NaN nas normais propagam para a média?
6. `visualization.py:137` e `:168`: `disp_normalmap`/`disp_channels` fazem swap de canais IN-PLACE sobre uma view do array de normais do chamador — hoje benigno porque o `.npy` é salvo antes, mas registrar como risco (e `cv2.imshow`+`waitKey(0)` bloqueia execução headless — impede automação do pipeline).

- [ ] **Step 4: Registrar achados em `02-photometric.md`**

Mesmo formato da Task 2 Step 5 (template + seção `## Verificado sem achado`).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/audit-notes/02-photometric.md
git commit -m "audit: phase 1.2 photometric findings"
```

---

### Task 4: Auditoria Fase 1.3 — Integração (Python + C)

**Files:**
- Read: `src/hybrid_stereo_method/hybrid/integrate.py`, `src/hybrid_stereo_method/hybrid/main.py:196-292`
- Read: `csrc/integrate_recursive/gus_integrate_recursive.c`
- Read (lib-src, só o caminho executado): `pst_normal_map.c` (conversão normal→slope), `pst_slope_map.c`, `pst_integrate.c`, `pst_integrate_recursive.c`, `pst_integrate_iterative.c`, `pst_imgsys.c`, `pst_imgsys_solve.c`, `pst_height_map.c`, `pst_interpolate.c`, `float_image_mscale.c`, `float_image.c` (semântica de eixos/indexação)
- Create: `docs/superpowers/reports/audit-notes/03-integration.md`

- [ ] **Step 1: Ler o lado Python (`integrate.py`, `hybrid/main.py`) e responder o checklist**

1. **`integrate.py:171-180`**: se `{prefix}-00-end-Z.fni` não existir, o código silenciosamente lê `{prefix}-ini-Z.fni` — o CHUTE INICIAL — e o devolve como resultado. Em que condições o end-Z não é escrito? Isso pode mascarar uma integração que falhou?
2. **`hybrid/main.py:215`**: `initial_method` default é `"hints"` mesmo quando `use_hints` é False (nenhum `-hints` passado) — o que o C faz com `-initial hints` sem mapa de hints?
3. O `normal_map` passado tem shape (H,W,3) com possíveis NaN (do wps) e SEM canal de peso — a confiança calculada pelo PS nunca chega ao integrador. O FNI escrito conterá `nan` textual — o parser C (`fget`/`float_image`) aceita?
4. Hints: docstring diz hints com shape (H+1,W+1) (grade de vértices), mas `zMos_with_confidence.fni` é (H,W,2) (grade de células). O C aceita/redimensiona ou corrompe? E as UNIDADES: `zMos` está em unidades físicas de `z_foc`, a altura integrada está em unidades de pixel (slope adimensional × passo de 1 pixel) — `hints_weight` mistura grandezas de escalas diferentes?
5. Reference: `hAvg.png` lido como uint8 (0-255) vs altura integrada em unidades de pixel — a análise de erro do C compara grandezas comensuráveis?
6. `slopes_scale`: passado como `scale sx sy` (sem hífen, `integrate.py:105`) — conferir na gramática de argumentos do C se é assim mesmo que se declara (argparser do Stolfi normalmente usa keywords com hífen).

- [ ] **Step 2: Ler `gus_integrate_recursive.c` (main) e responder o checklist**

1. Gramática real dos argumentos vs o que `integrate.py` monta (flags, ordem, `scale`, `-sortSys T`, `-initial <método> <noise>`): cada item confere?
2. Com `-normals`, qual função converte normal→slope e qual a convenção de eixos assumida (y cresce para cima ou para baixo? `nz` mínimo? sinal de `dZ/dY`)?
3. Que arquivos de saída são escritos e quando (`-00-end-Z.fni` é garantido em sucesso? códigos de erro retornados?).

- [ ] **Step 3: Ler a cadeia `pst_*` e responder o checklist**

1. `pst_normal_map.c`: fórmula slope = f(normal) — sinais; tratamento de `nz ≤ 0` e NaN; pesos derivados de quê?
2. `pst_imgsys.c`/`pst_imgsys_solve.c`: a equação montada por célula é o balanço de fluxo padrão (Poisson com pesos)? Fronteiras (Neumann natural?) e buracos (peso 0) excluem ou contaminam vizinhos? Critério de parada `convTol` compara o quê?
3. `pst_integrate_recursive.c` + `float_image_mscale.c`/`pst_*_shrink/expand`: a restrição de slopes para o nível grosseiro multiplica os slopes por 2 (ou divide?) para compensar o passo dobrado? Pesos restritos como (média? mínimo?)? A prolongação interpola alturas e corrige escala?
4. `pst_height_map.c`/`pst_map_compare.c`: a comparação com reference remove média/plano antes do erro? Em que unidades reporta?
5. `float_image.c`: a indexação `(col, row)` do C com row 0 = qual extremo da imagem? Comparar com o writer Python (`image_io.py:171-180` escreve y=0 primeiro = topo do array numpy). Registrar a conclusão preliminar para a Fase 2 (CONV-3).

- [ ] **Step 4: Registrar achados em `03-integration.md`**

Mesmo formato (template + `## Verificado sem achado`). Se a leitura do C ficar inconclusiva em algum ponto (ex.: semântica do eixo y), registrar como `suspeita` apontando o teste da Task 9 que decide.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/audit-notes/03-integration.md
git commit -m "audit: phase 1.3 integration findings"
```

---

### Task 5: Auditoria Fase 1.4 — Infra/IO

**Files:**
- Read: `src/hybrid_stereo_method/infrastructure/io/image_io.py`, `src/hybrid_stereo_method/infrastructure/utils.py`
- Create: `docs/superpowers/reports/audit-notes/04-io.md`

- [ ] **Step 1: Responder o checklist de IO**

1. `convert_image_array_to_fni` (145-181): escreve `%.7e` (~7-8 dígitos significativos) — suficiente para float32; mas o loop escreve a linha y na ordem 0..ny-1 — qual a semântica de y no formato float_image_t do C? (alimenta CONV-3).
2. `read_fni_to_image_array` (184-243): pixels ausentes ficam silenciosamente 0 (array inicializado com zeros, sem verificação de completude); linha com `=` no meio dos dados é pulada; valores `nan` no texto — `float("nan")` aceita, mas o que o resto do pipeline faz?
3. `read_image` com `IMREAD_UNCHANGED`: PNGs 16-bit são preservados na LEITURA, mas `save_image` SEMPRE grava uint8 (linha 138-140) — todo intermediário salvo como PNG é quantizado a 8 bits (médias do multifocus, sMos). Mapear todos os pontos do pipeline onde um array float passa por PNG e perde precisão.
4. `save_image` com `normalize=True` default: levantar TODOS os call sites (grep `save_image(`) e classificar cada um: visualização (ok) vs dado consumido depois (achado).
5. `infrastructure/utils.py`: `normalize` (caso constante), `calculate_avarage_of_images` (dtype da soma — uint8 overflow?), `convert_to_grayscale` (coeficientes BGR; comportamento com float).

- [ ] **Step 2: Registrar achados em `04-io.md`** (template + `## Verificado sem achado`)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/reports/audit-notes/04-io.md
git commit -m "audit: phase 1.4 io findings"
```

---

### Task 6: Fase 2 — Rastreamento transversal de convenções

**Files:**
- Read: trechos já mapeados nas Tasks 2-5 (reler conforme necessário)
- Create: `docs/superpowers/reports/audit-notes/05-conventions.md`

- [ ] **Step 1: Rastrear cada uma das 6 convenções produtor→arquivo→consumidor**

Para cada convenção, citar `arquivo:linha` de CADA elo da cadeia e dar veredito
(consistente / inconsistente / decidido-por-teste):

1. **Eixo z/profundidade:** direção de crescimento de `zf` (dataset) → `z_foc` (YAML) → `zMos` (`mosaic.py:72`) → hints (`hybrid/main.py:231`) → altura `Z` do C → `height_map.npy`. As direções e unidades batem ponta a ponta?
2. **Normais e luzes:** frame de `lights.npy` (x,y,z de quê?) → normais do `wps` (mesmo frame por construção) → conversão normal→slope no C (`pst_normal_map.c`) → eixos da grade de integração. O y das luzes é o y-para-baixo do numpy ou y-para-cima?
3. **Origem/orientação da imagem:** numpy `[linha, coluna]` origem topo-esquerda → writer FNI (`image_io.py:171`) → leitor C (`float_image.c`) → writer C → leitor Python (`image_io.py:184`). Há um flip vertical na ida e outro na volta (cancelando para round-trips Python→C→Python, mas NÃO para a interpretação de `dZ/dY` e das luzes)?
4. **Escala dos gradientes:** slope adimensional (Δz por Δpixel) no C vs `z_foc` físico do multifocus vs `hints_weight` — grandezas comensuráveis? `slopes_scale` compensa?
5. **Radiometria entre etapas:** `sVal.png` (uint8) → médias re-esticadas (`hybrid/main.py:111`) → indicador de foco; `sMos.png` uint8 clipado → grayscale → PS. Onde a cadeia quebra linearidade ou comparabilidade?
6. **Contratos de arquivo:** ordem `natsorted(sMos_path_list)` (`hybrid/main.py:177`) vs linhas de `lights.npy`; ordem `sorted(zf_directories)` vs `z_foc`; existência e shape de `zMos_with_confidence.fni` vs o que o C espera de `-hints`.

- [ ] **Step 2: Escrever `05-conventions.md`**

Conteúdo: tabela-resumo (linhas = 6 convenções; colunas = produtor, consumidor, veredito, evidência) + um bloco de achado `CONV-xx` (formato TEMPLATE.md) para cada inconsistência encontrada. Convenções cuja decisão depende de executar o C apontam para os testes da Task 9 com status `suspeita`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/reports/audit-notes/05-conventions.md
git commit -m "audit: phase 2 convention trace"
```

---

### Task 7: Geradores sintéticos compartilhados

**Files:**
- Create: `tests/synthetic_utils.py`
- Test: `tests/test_synthetic_utils.py`

- [ ] **Step 1: Escrever os testes de sanidade dos geradores**

```python
# tests/test_synthetic_utils.py
import numpy as np

from synthetic_utils import (
    affine_fit_rmse,
    defocus_stack,
    gaussian_bump,
    normals_from_height,
    ramp,
    render_lambertian,
    ring_lights,
    texture,
)


def test_normals_are_unit_and_point_to_camera():
    z = gaussian_bump(32, amplitude=4.0)
    n = normals_from_height(z)
    assert n.shape == (32, 32, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-12)
    assert (n[..., 2] > 0).all()  # nz aponta para a câmera


def test_ramp_normals_match_analytic():
    # z = ax*x + ay*y  ->  n ∝ (-ax, -ay, 1)
    ax, ay = 0.3, -0.2
    n = normals_from_height(ramp(16, ax=ax, ay=ay))
    expected = np.array([-ax, -ay, 1.0])
    expected /= np.linalg.norm(expected)
    interior = n[2:-2, 2:-2]  # np.gradient é unilateral nas bordas
    np.testing.assert_allclose(interior, np.broadcast_to(expected, interior.shape), atol=1e-10)


def test_ring_lights_unit_norm():
    lights = ring_lights(6, tilt_deg=30.0)
    assert lights.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(lights, axis=-1), 1.0, atol=1e-12)
    assert (lights[:, 2] > 0).all()


def test_render_lambertian_range_and_max():
    z = gaussian_bump(32, amplitude=4.0)
    n = normals_from_height(z)
    img = render_lambertian(n, np.array([0.0, 0.0, 1.0]), albedo=200.0)
    assert img.min() >= 0.0
    assert img.max() <= 200.0 + 1e-9


def test_defocus_stack_sharpest_frame_tracks_depth():
    size, n_frames = 48, 7
    z_foc = list(range(n_frames))
    depth = np.full((size, size), 4.0)  # plano em z=4 -> frame 4 é o mais nítido
    sharp = 255.0 * texture(size, seed=0)
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)
    assert stack.shape == (n_frames, size, size)
    # variância do laplaciano como proxy de nitidez por frame
    import cv2

    sharpness = [cv2.Laplacian(f, cv2.CV_64F).var() for f in stack]
    assert int(np.argmax(sharpness)) == 4


def test_affine_fit_rmse_exact_for_affine_pair():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, (10, 10))
    est = 3.0 * gt - 7.0
    rmse, _ = affine_fit_rmse(est, gt)
    assert rmse < 1e-12
```

- [ ] **Step 2: Rodar e verificar que falham por módulo ausente**

Run: `pytest tests/test_synthetic_utils.py -v`
Expected: FAIL/ERROR com `ModuleNotFoundError: No module named 'synthetic_utils'`

- [ ] **Step 3: Implementar `tests/synthetic_utils.py`**

```python
"""Geradores de dados sintéticos com ground truth analítico para a auditoria.

Convenções (as mesmas presumidas pelo pipeline Python, e testadas contra o C):
- arrays numpy [linha=y, coluna=x], origem no topo-esquerdo, y cresce para BAIXO;
- altura z cresce em direção à câmera; normais n = (-dz/dx, -dz/dy, 1)/|.|, nz > 0;
- luzes no mesmo frame das normais.
"""

import cv2
import numpy as np


def gaussian_bump(size, amplitude=6.0, sigma_frac=0.22):
    """Mapa de altura z[y, x] = A * exp(-((x-cx)^2 + (y-cy)^2) / (2 s^2))."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    s = sigma_frac * size
    return amplitude * np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2.0 * s * s)))


def ramp(size, ax=0.05, ay=0.02):
    """Mapa de altura z[y, x] = ax * x + ay * y (y = índice de linha)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    return ax * x + ay * y


def normals_from_height(z):
    """Normais unitárias por diferenças centrais: n = (-dz/dx, -dz/dy, 1)/|.|."""
    dz_dy, dz_dx = np.gradient(z)
    n = np.stack([-dz_dx, -dz_dy, np.ones_like(z)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def ring_lights(n_lights=6, tilt_deg=30.0):
    """Direções de luz unitárias num cone em torno de +z."""
    t = np.deg2rad(tilt_deg)
    az = np.linspace(0.0, 2.0 * np.pi, n_lights, endpoint=False)
    return np.stack(
        [np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.full(n_lights, np.cos(t))],
        axis=-1,
    )


def render_lambertian(normals, light, albedo=1.0):
    """I = albedo * max(0, n . l). `albedo` pode ser escalar ou mapa (h, w)."""
    return albedo * np.clip(normals @ np.asarray(light, dtype=np.float64), 0.0, None)


def texture(size, seed=0):
    """Textura aleatória de alta frequência em [0.2, 1.0] (foco precisa de textura)."""
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.uniform(0.0, 1.0, (size, size)), (0, 0), 1.0)
    t = (t - t.min()) / (t.max() - t.min())
    return 0.2 + 0.8 * t


def defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5):
    """Pilha de foco sintética: frame k = `sharp` desfocada por
    sigma(x, y) = blur_per_unit * |depth(x, y) - z_foc[k]|.

    Implementação: banco de cópias borradas em passos de 0.25 sigma,
    interpoladas linearmente por pixel.
    """
    sharp = np.asarray(sharp, dtype=np.float64)
    sigmas = blur_per_unit * np.abs(depth[None, :, :] - np.asarray(z_foc)[:, None, None])
    step = 0.25
    n_levels = int(np.ceil(sigmas.max() / step)) + 2
    bank = [sharp]
    for k in range(1, n_levels):
        bank.append(cv2.GaussianBlur(sharp, (0, 0), k * step))
    bank = np.stack(bank)  # (n_levels, h, w)

    h, w = sharp.shape
    rows, cols = np.mgrid[0:h, 0:w]
    frames = []
    for k in range(len(z_foc)):
        idx = sigmas[k] / step
        i0 = np.clip(np.floor(idx).astype(int), 0, n_levels - 2)
        frac = idx - i0
        frames.append((1.0 - frac) * bank[i0, rows, cols] + frac * bank[i0 + 1, rows, cols])
    return np.stack(frames)


def affine_fit_rmse(est, gt):
    """RMSE de gt vs (a*est + b) com a, b ótimos por mínimos quadrados.

    Atenção: o fit afim absorve escala global, offset E INVERSÃO DE SINAL —
    use-o para medir forma, e o teste de rampa para decidir convenções de sinal.
    Retorna (rmse, (a, b)).
    """
    est = np.asarray(est, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    a = np.stack([est, np.ones_like(est)], axis=1)
    coef, *_ = np.linalg.lstsq(a, gt, rcond=None)
    rmse = float(np.sqrt(np.mean((a @ coef - gt) ** 2)))
    return rmse, (float(coef[0]), float(coef[1]))
```

- [ ] **Step 4: Rodar e verificar que passam**

Run: `pytest tests/test_synthetic_utils.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic_utils.py tests/test_synthetic_utils.py
git commit -m "audit: add synthetic ground-truth generators for audit tests"
```

---

### Task 8: Fase 3.2 — Round-trip FNI

**Files:**
- Test: `tests/test_fni_roundtrip.py`
- Create: `docs/superpowers/reports/audit-notes/06-test-results.md`

- [ ] **Step 1: Escrever os testes de round-trip**

```python
# tests/test_fni_roundtrip.py
"""Fase 3.2 — round-trip FNI Python -> Python.

Padrões ASSIMÉTRICOS em x e y: um flip ou transposição silenciosa não passa.
Se um teste falhar, NÃO conserte o teste: registre o achado (IO-xx) com a saída.
"""
import numpy as np
import pytest

from hybrid_stereo_method.infrastructure.io.image_io import (
    convert_image_array_to_fni,
    read_fni_to_image_array,
)


def _asymmetric(shape):
    """Array sem nenhuma simetria de flip/transposição."""
    return (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) ** 1.5 + 0.125) / 7.0


def test_roundtrip_2d(tmp_path):
    arr = _asymmetric((5, 9))  # retangular: transposição também quebraria o shape
    path = tmp_path / "a.fni"
    convert_image_array_to_fni(arr, path)
    back = read_fni_to_image_array(path)
    assert back.shape == arr.shape
    np.testing.assert_allclose(back, arr, rtol=1e-6)


def test_roundtrip_3channel(tmp_path):
    arr = _asymmetric((4, 6, 3))
    path = tmp_path / "b.fni"
    convert_image_array_to_fni(arr, path)
    back = read_fni_to_image_array(path)
    assert back.shape == arr.shape
    np.testing.assert_allclose(back, arr, rtol=1e-6)


def test_roundtrip_negative_and_large_values(tmp_path):
    arr = np.array([[-1.5e6, 3.25e-7], [0.0, -0.0]], dtype=np.float64)
    path = tmp_path / "c.fni"
    convert_image_array_to_fni(arr, path)
    np.testing.assert_allclose(read_fni_to_image_array(path), arr, rtol=1e-6, atol=1e-30)


def test_nan_handling_documented(tmp_path):
    """Sonda de comportamento: normais com NaN (sombra) são escritas em FNI
    pelo pipeline. Este teste DOCUMENTA o que o round-trip Python faz com NaN
    (o comportamento do lado C é avaliado na auditoria INT)."""
    arr = np.array([[1.0, np.nan], [2.0, 3.0]])
    path = tmp_path / "d.fni"
    convert_image_array_to_fni(arr, path)
    back = read_fni_to_image_array(path)
    assert np.isnan(back[0, 1]), "NaN não sobreviveu ao round-trip — registrar comportamento real"
    np.testing.assert_allclose(back[~np.isnan(back)], arr[~np.isnan(arr)], rtol=1e-6)
```

- [ ] **Step 2: Rodar**

Run: `pytest tests/test_fni_roundtrip.py -v`
Expected: 4 passed — OU falhas, que são achados IO-xx.

- [ ] **Step 3: Registrar o resultado**

Criar `docs/superpowers/reports/audit-notes/06-test-results.md` com cabeçalho
`# Resultados dos testes sintéticos` e uma seção `## test_fni_roundtrip` com:
comando executado, resultado (passed/failed), e — se falhou — o achado correspondente
no formato TEMPLATE.md (ou atualização de status de um achado existente de 04-io.md
para `confirmado`/`refutado`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fni_roundtrip.py docs/superpowers/reports/audit-notes/06-test-results.md
git commit -m "audit: phase 3.2 FNI roundtrip tests"
```

---

### Task 9: Fase 3.2 — Convenções Python↔C (requer binário)

**Files:**
- Test: `tests/test_convention_integration.py`
- Modify: `docs/superpowers/reports/audit-notes/06-test-results.md`
- Modify: `pyproject.toml` (registrar marker `slow`)

- [ ] **Step 1: Compilar o binário C**

```bash
cd csrc/integrate_recursive && make
ls -la gus_integrate_recursive
```
Expected: binário existe. Se o make FALHAR: registrar o bloqueio em
`06-test-results.md` (seção `## Bloqueios`) com a saída do erro, marcar os testes
desta task e da Task 12 como bloqueados, e seguir para a Task 10.

- [ ] **Step 2: Registrar o marker `slow` no pytest**

Em `pyproject.toml`, dentro da seção `[tool.pytest.ini_options]` existente, adicionar:

```toml
markers = ["slow: testes longos (end-to-end ou que invocam o binário C)"]
```

- [ ] **Step 3: Escrever os testes de convenção**

```python
# tests/test_convention_integration.py
"""Fase 3.2 — convenções Python <-> C via o integrador.

O teste da rampa decide as convenções de sinal/orientação (CONV-1..4): uma rampa
monotônica não tem simetria, então flip de y, inversão de sinal ou troca de eixos
produzem um candidato diferente como melhor ajuste.
Se um assert de convenção falhar, NÃO conserte o teste: o candidato vencedor
impresso É o achado (CONV-xx).
"""
import numpy as np
import pytest

from synthetic_utils import affine_fit_rmse, gaussian_bump, normals_from_height

from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    integrate_normals_to_height,
    integrate_slopes_to_height,
)

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

pytestmark = [needs_binary, pytest.mark.slow]


def test_constant_slopes_recover_ramp_and_decide_convention(tmp_path):
    size, ax, ay = 32, 0.05, 0.02
    slopes = np.zeros((size, size, 2))
    slopes[..., 0] = ax  # canal 0: dZ/dX (conforme docstring de integrate.py)
    slopes[..., 1] = ay  # canal 1: dZ/dY
    z = integrate_slopes_to_height(slopes, tmp_path, "ramp")
    assert z.shape == (size + 1, size + 1)

    y, x = np.mgrid[0 : size + 1, 0 : size + 1].astype(np.float64)
    candidates = {
        "z = +ax*x + ay*y (y do numpy, para baixo)": ax * x + ay * y,
        "z = +ax*x - ay*y (y invertido: para cima)": ax * x - ay * y,
        "z = -ax*x - ay*y (tudo invertido)": -(ax * x + ay * y),
        "z = -ax*x + ay*y": -ax * x + ay * y,
        "z = +ay*x + ax*y (eixos trocados)": ay * x + ax * y,
    }
    z0 = z - z.mean()
    errs = {k: float(np.sqrt(np.mean((z0 - (v - v.mean())) ** 2))) for k, v in candidates.items()}
    best = min(errs, key=errs.get)
    print("\nRMSE por candidato de convenção:")
    for k, v in sorted(errs.items(), key=lambda kv: kv[1]):
        print(f"  {v:12.6f}  {k}")

    # 1) o integrador integra: o melhor candidato ajusta bem
    assert errs[best] < 0.05 * (abs(ax) + abs(ay)) * size, errs
    # 2) convenção presumida pelo lado Python (docstrings/numpy): falha = achado CONV
    assert best == "z = +ax*x + ay*y (y do numpy, para baixo)", (
        f"convencao real do C: '{best}' — registrar CONV-xx com a tabela impressa"
    )


def test_normals_path_recovers_bump_shape(tmp_path):
    """Caminho -normals: forma recuperada a menos de afim (sinais já decididos
    pelo teste da rampa; o bump é simétrico e não os detecta)."""
    size = 48
    z_gt = gaussian_bump(size, amplitude=4.0)
    normals = normals_from_height(z_gt)
    z = integrate_normals_to_height(normals, tmp_path, "bump")
    est = z[:size, :size]  # grade de vértices (H+1) -> recorte comparável
    rmse, (a, b) = affine_fit_rmse(est, z_gt)
    print(f"\nbump: affine-fit rmse={rmse:.4f}, a={a:.4f}, b={b:.4f}, std(gt)={z_gt.std():.4f}")
    assert np.isfinite(est).all()
    assert rmse < 0.15 * z_gt.std(), f"forma não recuperada: rmse={rmse:.4f}"
```

- [ ] **Step 4: Rodar**

Run: `pytest tests/test_convention_integration.py -v -s`
Expected: 2 passed — OU falha no assert de convenção da rampa, cuja tabela impressa
é a evidência do achado CONV.

- [ ] **Step 5: Registrar resultados e atualizar achados**

Em `06-test-results.md`, adicionar seção `## test_convention_integration` com comando,
resultado e a tabela de candidatos impressa. Atualizar em `05-conventions.md` e
`03-integration.md` o status (`confirmado`/`refutado`) dos achados CONV/INT que este
teste decide (orientação do y, sinais dos slopes, fallback `-ini-Z.fni` se observado).

- [ ] **Step 6: Commit**

```bash
git add tests/test_convention_integration.py pyproject.toml docs/superpowers/reports/audit-notes/
git commit -m "audit: phase 3.2 Python<->C convention tests via ramp/bump integration"
```

---

### Task 10: Fase 3.1 — Testes sintéticos do fotométrico

**Files:**
- Test: `tests/test_photometric_synthetic.py`
- Modify: `docs/superpowers/reports/audit-notes/06-test-results.md`

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_photometric_synthetic.py
"""Fase 3.1 — fotométrico com ground truth analítico.

Geração e estimação usam o MESMO frame de coordenadas, então estes testes validam
a matemática interna do solver (não as convenções entre estágios — Task 9 cuida disso).
Falha = achado PS-xx; não conserte o teste.
"""
import numpy as np

from synthetic_utils import gaussian_bump, normals_from_height, ring_lights, render_lambertian

from hybrid_stereo_method.photometric.wps import estimate_normals_argmax_lstsq_robust


def _angular_error_deg(n_est, n_gt, valid):
    cos = np.clip(np.sum(n_est[valid] * n_gt[valid], axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def test_wps_recovers_normals_clean_data():
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=5.0))
    lights = ring_lights(6, tilt_deg=30.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]

    normals, albedo, confidence, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})

    valid = np.isfinite(normals).all(axis=-1)
    assert valid.mean() > 0.99, f"só {valid.mean():.1%} de pixels válidos em dados limpos"
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\nlimpo: erro angular médio = {ang.mean():.3f}°, p95 = {np.percentile(ang, 95):.3f}°")
    assert ang.mean() < 1.0, f"erro angular médio {ang.mean():.2f}° (esperado < 1° sem ruído)"


def test_wps_albedo_recovers_true_albedo():
    """Modelo I = rho * (L.n): com rho = 200 constante, o albedo estimado deve
    ser ~200 e NÃO depender do número de luzes. Falha aqui confirma o lead do
    albedo = ||L_sel @ n_normalizado|| (wps.py:198)."""
    size = 24
    n_gt = normals_from_height(gaussian_bump(size, amplitude=3.0))
    rho = 200.0
    for n_lights in (4, 8):
        lights = ring_lights(n_lights, tilt_deg=30.0)
        images = [render_lambertian(n_gt, light, albedo=rho) for light in lights]
        _, albedo, _, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
        med = float(np.median(albedo[albedo > 0]))
        print(f"\nn_lights={n_lights}: albedo mediano = {med:.1f} (verdadeiro: {rho})")
        assert abs(med - rho) < 0.1 * rho, (
            f"albedo mediano {med:.1f} != {rho} com {n_lights} luzes — "
            "se cresce com sqrt(n_lights), confirma o achado do albedo"
        )


def test_wps_robust_to_saturation():
    """Satura (clip) as intensidades em 60% do máximo em DUAS imagens — o laço
    robusto deve descartar os outliers e manter o erro angular baixo."""
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=5.0))
    lights = ring_lights(8, tilt_deg=30.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]
    cap = 0.6 * max(img.max() for img in images)
    images[0] = np.minimum(images[0], cap)
    images[1] = np.minimum(images[1], cap)

    normals, _, _, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
    valid = np.isfinite(normals).all(axis=-1)
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\nsaturado: erro angular médio = {ang.mean():.3f}°")
    assert ang.mean() < 5.0, f"robustez insuficiente a saturação: {ang.mean():.2f}°"


def test_wps_shadowed_pixels_flagged_not_garbage():
    """Luz rasante (tilt 75°) numa superfície inclinada gera attached shadows
    (n.l < 0 -> I = 0). Pixels com < 3 medições válidas devem virar NaN+conf 0,
    e os demais devem continuar precisos."""
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=10.0, sigma_frac=0.15))
    lights = ring_lights(5, tilt_deg=75.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]

    normals, _, confidence, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
    valid = np.isfinite(normals).all(axis=-1)
    assert (confidence[~valid] == 0).all(), "pixel inválido com confiança > 0"
    if valid.any():
        ang = _angular_error_deg(normals, n_gt, valid)
        print(f"\nsombras: válidos = {valid.mean():.1%}, erro médio nos válidos = {ang.mean():.2f}°")
        assert ang.mean() < 10.0
```

- [ ] **Step 2: Rodar**

Run: `pytest tests/test_photometric_synthetic.py -v -s`
Expected: passes e/ou falhas-evidência (em particular `test_wps_albedo_recovers_true_albedo`
deve confirmar ou refutar o lead PS do albedo).

- [ ] **Step 3: Registrar resultados** em `06-test-results.md` (seção
`## test_photometric_synthetic`, mesma estrutura) e atualizar status dos achados
PS-xx em `02-photometric.md`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_photometric_synthetic.py docs/superpowers/reports/audit-notes/
git commit -m "audit: phase 3.1 photometric synthetic tests"
```

---

### Task 11: Fase 3.1 — Testes sintéticos do multifocus

**Files:**
- Test: `tests/test_multifocus_synthetic.py`
- Modify: `docs/superpowers/reports/audit-notes/06-test-results.md`

- [ ] **Step 1: Escrever os testes**

```python
# tests/test_multifocus_synthetic.py
"""Fase 3.1 — multifocus com pilha de foco sintética.

depth está em UNIDADES DE ÍNDICE de frame (z_foc = 0..n-1), então iSel é
diretamente comparável ao ground truth. Falha = achado MF-xx.
"""
import numpy as np

from synthetic_utils import defocus_stack, gaussian_bump, texture

from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy
from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator


def _run_multifocus(depth, size, n_frames, seed=3):
    z_foc = list(range(n_frames))
    sharp = 255.0 * texture(size, seed=seed)
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)
    fi = focus_indicator(
        stack,
        "laplacian",
        laplacian_kernel_size=5,
        radius=None,
        square=True,
        smooth=True,
        spatial_median_filter=False,
        zero_border=False,
    )
    iSel, wSel = compute_argmax_fuzzy(fi, False, "", {"r_max": 2})
    return iSel, wSel


def test_recovers_tilted_plane_depth():
    size, n_frames = 64, 9
    x = np.mgrid[0:size, 0:size][1].astype(np.float64)
    depth = 1.5 + 5.0 * x / (size - 1)  # rampa em x: 1.5 .. 6.5 (índices de frame)
    iSel, _ = _run_multifocus(depth, size, n_frames)

    interior = (slice(8, -8), slice(8, -8))  # evita efeitos de borda do filtro
    err = np.abs(iSel[interior] - depth[interior])
    print(f"\nplano: mediana|iSel-z| = {np.median(err):.3f} frames, p90 = {np.percentile(err, 90):.3f}")
    assert np.median(err) < 0.5, f"erro sub-frame não atingido: mediana {np.median(err):.2f}"


def test_recovers_bump_depth():
    size, n_frames = 64, 9
    depth = 2.0 + gaussian_bump(size, amplitude=4.0)  # 2 .. 6
    iSel, _ = _run_multifocus(depth, size, n_frames)

    interior = (slice(8, -8), slice(8, -8))
    err = np.abs(iSel[interior] - depth[interior])
    print(f"\nbump: mediana|iSel-z| = {np.median(err):.3f} frames")
    assert np.median(err) < 0.5


def test_textureless_region_gets_zero_confidence():
    """Região central SEM textura: a profundidade lá é indecidível; o método deve
    sinalizar confiança ~0 (e o que iSel devolve lá é documentado — lead MF do
    'return n/2')."""
    size, n_frames = 64, 9
    depth = np.full((size, size), 4.0)
    z_foc = list(range(n_frames))
    sharp = 255.0 * texture(size, seed=5)
    sharp[24:40, 24:40] = 128.0  # quadrado plano, sem textura
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)
    fi = focus_indicator(
        stack, "laplacian", laplacian_kernel_size=5, radius=None,
        square=True, smooth=True, spatial_median_filter=False, zero_border=False,
    )
    iSel, wSel = compute_argmax_fuzzy(fi, False, "", {"r_max": 2})

    flat = wSel[28:36, 28:36]       # miolo da região sem textura
    textured = wSel[4:16, 4:16]
    print(f"\nconf média: sem textura = {flat.mean():.4f}, com textura = {textured.mean():.4f}")
    print(f"iSel na região sem textura: mediana = {np.median(iSel[28:36, 28:36]):.2f} (gt = 4.0)")
    assert flat.mean() < 0.5 * textured.mean(), (
        "confiança em região sem textura não é distintamente menor"
    )
```

- [ ] **Step 2: Rodar**

Run: `pytest tests/test_multifocus_synthetic.py -v -s`
Expected: passes e/ou falhas-evidência. Os valores impressos (erro mediano em frames,
iSel na região sem textura) vão para as notas mesmo quando o teste passa.

- [ ] **Step 3: Registrar resultados** em `06-test-results.md` (seção
`## test_multifocus_synthetic`) e atualizar status dos achados MF-xx em `01-multifocus.md`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multifocus_synthetic.py docs/superpowers/reports/audit-notes/
git commit -m "audit: phase 3.1 multifocus synthetic tests"
```

---

### Task 12: Fase 3.3 — Teste end-to-end do pipeline híbrido

**Files:**
- Test: `tests/test_e2e_hybrid.py`
- Modify: `docs/superpowers/reports/audit-notes/06-test-results.md`

- [ ] **Step 1: Escrever o teste E2E**

Decisões embutidas (justificadas pela auditoria):
- 9 planos focais (`zf0..zf8`) e 6 luzes (`L0..L5`) — todos com 1 dígito, para o
  resultado NÃO ser contaminado pelos suspeitos de substring/sort lexicográfico
  (esses são cobertos pelos achados MF da Task 2, não por este teste).
- `disp_*` do `main_wps` são monkeypatchados (chamam `cv2.imshow`+`waitKey(0)` e
  travariam o teste headless).
- `initial_method="zero"`, sem hints/reference: mede a linha de base do caminho
  principal; o acoplamento dos hints é avaliado nos achados INT/CONV.

```python
# tests/test_e2e_hybrid.py
"""Fase 3.3 — pipeline híbrido completo sobre dataset sintético com ground truth.

Roda sempre (não é sob demanda): o RMSE com fit afim é a LINHA DE BASE do estado
atual do pipeline, registrada no relatório. O fit afim absorve escala/offset/sinal
globais — as convenções de sinal são decididas pela Task 9, não aqui.
"""
import cv2
import numpy as np
import pytest

from synthetic_utils import (
    affine_fit_rmse,
    defocus_stack,
    gaussian_bump,
    normals_from_height,
    render_lambertian,
    ring_lights,
    texture,
)

from hybrid_stereo_method.hybrid.integrate import DEFAULT_EXECUTABLE

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

pytestmark = [needs_binary, pytest.mark.slow]

SIZE = 64
N_FRAMES = 9
N_LIGHTS = 6


def _build_dataset(root):
    """Layout L<n>/zf<m>/sVal.png + lights.npy + sharp/hAvg.png. Retorna depth gt."""
    depth = 2.0 + gaussian_bump(SIZE, amplitude=4.0)  # 2..6, dentro de z_foc 0..8
    normals = normals_from_height(depth)
    lights = ring_lights(N_LIGHTS, tilt_deg=30.0)
    albedo_map = 80.0 + 150.0 * texture(SIZE, seed=1)  # textura p/ foco E sombreamento p/ PS
    z_foc = [float(k) for k in range(N_FRAMES)]

    data_dir = root / "synth"
    for li in range(N_LIGHTS):
        shaded = render_lambertian(normals, lights[li], albedo=albedo_map)
        stack = defocus_stack(shaded, depth, z_foc, blur_per_unit=1.5)
        for k in range(N_FRAMES):
            frame_dir = data_dir / f"L{li}" / f"zf{k}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            img = np.clip(stack[k], 0, 255).astype(np.uint8)
            cv2.imwrite(str(frame_dir / "sVal.png"), cv2.merge([img, img, img]))

    np.save(data_dir / "lights.npy", lights)

    sharp_dir = data_dir / "sharp"
    sharp_dir.mkdir(parents=True, exist_ok=True)
    h_vis = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    cv2.imwrite(str(sharp_dir / "hAvg.png"), h_vis)
    return depth, z_foc


def test_hybrid_pipeline_end_to_end(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import hybrid_stereo_method.photometric.main_wps as main_wps_mod

    # disp_* usam cv2.imshow + waitKey(0): travariam headless (achado PS já registrado)
    monkeypatch.setattr(main_wps_mod, "disp_normalmap", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels_3d", lambda **kw: None)

    from hybrid_stereo_method.hybrid.main import main as hybrid_main

    raw = tmp_path / "raw"
    depth_gt, z_foc = _build_dataset(raw)

    parameters = {
        "experiment": {
            "type": "hybrid",
            "paths": {
                "input": str(raw),
                "data_folder": "synth",
                "output": str(tmp_path / "results"),
            },
            "settings": {"debug": False, "gabaritos": False},
        },
        "multifocus": {
            "focus_measure": {
                "method": "laplacian",
                "parameters": {"kernel_size": 5, "radius": None},
                "preprocessing": {
                    "square": True,
                    "smooth": True,
                    "spatial_median_filter": False,
                    "zero_border": False,  # zero_borders(40px) apagaria quase tudo em 64px
                },
            },
            "optimization": {"r_max": 2},
            "parameters": {"z_foc": z_foc, "interpolation": "linear_interpolation"},
        },
        "photometric": {
            "solver": {
                "epsilon": 1e-6,
                "shadow_threshold": 1e-3,
                "outlier_threshold_multiplier": 3,
            }
        },
        "hybrid": {
            "integration": {
                "initial_method": "zero",
                "use_hints": False,
                "use_reference": False,
                "max_iter": 20000,
                "conv_tol": 5e-7,
            }
        },
    }

    hybrid_main(parameters)

    out_dirs = list((tmp_path / "results").glob("*_synth"))
    assert len(out_dirs) == 1, f"esperava 1 pasta de saída, achei {out_dirs}"
    height_path = out_dirs[0] / "integration" / "height_map.npy"
    assert height_path.exists(), "pipeline terminou sem height_map.npy"

    height = np.load(height_path)
    est = height[:SIZE, :SIZE]  # grade de vértices (H+1, W+1) -> recorte
    finite = np.isfinite(est)
    assert finite.mean() > 0.95, f"só {finite.mean():.1%} do mapa de altura é finito"

    interior = (slice(8, -8), slice(8, -8))
    rmse, (a, b) = affine_fit_rmse(est[interior], depth_gt[interior])
    corr = float(np.corrcoef(est[interior].ravel(), depth_gt[interior].ravel())[0, 1])
    print(
        f"\n=== BASELINE E2E === affine-fit RMSE = {rmse:.4f} "
        f"(std gt = {depth_gt[interior].std():.4f}), a = {a:.4f}, b = {b:.4f}, "
        f"pearson r = {corr:.4f}"
    )
    # Linha de base, não gate de qualidade: registre os números no relatório.
    assert np.isfinite(rmse)
```

- [ ] **Step 2: Rodar (demora alguns minutos — laços por pixel do wps/argmax)**

Run: `pytest tests/test_e2e_hybrid.py -v -s`
Expected: 1 passed com a linha `=== BASELINE E2E ===` impressa (ou falha em etapa
intermediária — que é um achado de pipeline, registrar com o traceback).

- [ ] **Step 3: Registrar a linha de base** em `06-test-results.md`
(seção `## test_e2e_hybrid — BASELINE`): RMSE com fit afim, coeficientes a/b,
correlação de Pearson, e interpretação de 2 linhas (ex.: "r ≈ 1 e a > 0: forma e
orientação corretas; a ≠ 1: escala absoluta não recuperada — coerente com CONV-4").

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_hybrid.py docs/superpowers/reports/audit-notes/06-test-results.md
git commit -m "audit: phase 3.3 end-to-end hybrid pipeline baseline test"
```

---

### Task 13: Consolidar o relatório final

**Files:**
- Create: `docs/superpowers/reports/2026-06-04-method-audit.md`
- Read: todos os `docs/superpowers/reports/audit-notes/*.md`

- [ ] **Step 1: Montar o relatório a partir das notas**

Estrutura obrigatória (do spec, seção "Relatório final"):

```markdown
# Auditoria de corretude dos métodos — Relatório de achados

**Data:** <data de conclusão>  **Spec:** docs/superpowers/specs/2026-06-04-method-audit-design.md

## 1. Sumário executivo
- Tabela: contagem de achados por severidade (crítico/alto/médio/baixo) × status
  (confirmado/suspeita/refutado). Achados refutados contam separado.
- Os 3-5 achados mais importantes, uma frase cada, com ID.

## 2. Achados por estágio
### 2.1 Multifocus (MF-xx)      <- copiar de 01-multifocus.md, já atualizados
### 2.2 Fotométrico (PS-xx)     <- de 02-photometric.md
### 2.3 Integração (INT-xx)     <- de 03-integration.md
### 2.4 Infra/IO (IO-xx)        <- de 04-io.md
### 2.5 Convenções (CONV-xx)    <- de 05-conventions.md

## 3. Tabela de convenções
<- a tabela-resumo de 05-conventions.md, com vereditos finais pós-testes

## 4. Linha de base end-to-end
<- números e interpretação de 06-test-results.md (## test_e2e_hybrid)
   + bloqueios, se houver

## 5. Apêndice — cobertura da varredura
- O que foi auditado (lista de arquivos por estágio, das seções
  "Verificado sem achado" das notas)
- O que ficou fora (do spec): lib-src/ vendorizada (exceto caminho pst_*),
  notebooks/, core/, estilo/performance
- Suíte de testes permanente criada (lista dos arquivos tests/*)
```

- [ ] **Step 2: Verificação de completude contra o spec (critério de pronto)**

Conferir e corrigir antes de commitar:
1. Todos os arquivos do escopo (tabela do spec) aparecem ou num achado ou em
   "Verificado sem achado"? Os que não foram lidos por bloqueio estão declarados?
2. As 6 convenções têm veredito final?
3. Todo achado tem ID único, localização `arquivo:linha`, tipo, severidade, status?
4. Nenhum achado `confirmado` sem referência a teste/evidência concreta?
5. O baseline E2E (ou o bloqueio do binário) está documentado?

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/reports/2026-06-04-method-audit.md
git commit -m "audit: consolidate method correctness audit report"
```

- [ ] **Step 4: Apresentar o sumário executivo ao usuário** e perguntar se quer
priorizar correções (trabalho futuro, fora deste plano).
