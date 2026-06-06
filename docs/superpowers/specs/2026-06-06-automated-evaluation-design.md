# Avaliação automatizada dos resultados do pipeline híbrido — Design

**Data:** 2026-06-06
**Status:** aprovado em brainstorming (aguardando revisão da spec escrita)

## Objetivo

Feature de avaliação automatizada que compara as saídas de cada etapa do pipeline
híbrido (multifocus, fotométrico, integração) e o resultado final contra o ground
truth presente nos dados de entrada. Roda **independente do experimento** (post-hoc,
sobre qualquer pasta de resultados existente) e também como **hook opcional** ao
final de `hybrid/main.py`.

Dataset de referência: o usado em `configs/hb_experiment.yaml`
(`data/raw/hybrid_stereo/2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/`),
imagens 422×512 (H×W), 12 luzes (`L000`–`L011`), 12 planos focais
(`zf015`–`zf125`, passo 10, em unidades z_foc).

## Decisões de design (aprovadas)

| Decisão | Escolha |
|---|---|
| Invocação | CLI standalone (`--results_dir`) **e** hook opcional no pipeline |
| Saídas | `metrics.json` + mapas de erro PNG + `report.md` consolidado |
| Escala de altura | Métricas **afim-invariantes** (`gt ≈ a·est + b` por mínimos quadrados); sem dependência de `reference_scale`/`pixel_size` |
| Escopo | 4 avaliações: zMos vs hAvg; sMos vs sVal por luz; normais vs sNrm; height_map final vs hAvg — mais seleção de foco vs pilha shrp (GT adicional encontrado) e síntese do ganho do híbrido |
| Arquitetura | Opção A — subpacote `evaluation/` com avaliadores por etapa |

## Arquitetura

Novo subpacote `src/hybrid_stereo_method/evaluation/`:

```
evaluation/
├── __init__.py
├── main.py          # CLI + run_evaluation() (função chamada pelo hook)
├── loaders.py       # descoberta e carregamento de artefatos + ground truth
├── metrics.py       # funções puras de métrica (sem I/O)
├── evaluators.py    # um avaliador por etapa (arrays → dict de métricas + mapas)
└── report.py        # metrics.json + mapas de erro PNG + report.md
```

### Responsabilidades

- **`metrics.py`** — funções puras (arrays → números):
  - `affine_fit_rmse(est, gt)` promovida de `tests/synthetic_utils.py:85`
    (o módulo de teste passa a reexportar daqui — sem duplicação);
  - `angular_error_deg(n_est, n_gt)` (adaptação de
    `photometric/ps_utils.py::evaluate_angular_error`);
  - `pearson_r`; wrappers de PSNR/SSIM de `skimage.metrics`
    (scikit-image já é dependência do projeto).
- **`loaders.py`** — resolve artefatos na pasta de resultados e GT por convenção
  de caminho; decodifica `sNrm.png` → vetores unitários (`(v/255)·2−1`,
  renormalizados); lê `hAvg.png`/`hDev.png` como uint16; FNI via
  `infrastructure/io/image_io.py`.
- **`evaluators.py`** — `evaluate_multifocus()`, `evaluate_photometric()`,
  `evaluate_integration()`: recebem arrays, devolvem dict de métricas + mapas de
  erro (arrays). Independentes entre si: falta de artefato/GT pula só aquela
  avaliação.
- **`report.py`** — consolida em `<results_dir>/evaluation/`: `metrics.json`,
  PNGs de mapas de erro e `report.md`.

### Invocação

```bash
# Standalone, sobre qualquer resultado já existente:
python -m hybrid_stereo_method.evaluation.main --results_dir <pasta_timestamped> [--data_dir <pasta_dados>]
```

- O pipeline já copia `sharp/` para a pasta de resultados (`hybrid/main.py:221`),
  então o GT de altura/normais geralmente está dentro do próprio resultado.
- `--data_dir` é necessário apenas para o GT por luz (`L*/sharp/sVal.png`) e a
  pilha `shrp` por zf (não copiados); ausente → essas avaliações são puladas com
  aviso.
- **Hook**: seção `evaluation: {enabled: true}` no YAML; `hybrid/main.py` chama
  `run_evaluation(output_path, data_path)` como passo final, em `try/except` —
  falha na avaliação nunca derruba um experimento que já produziu resultados.
- Melhoria pontual incluída: o pipeline passa a salvar o config resolvido
  (`parameters.yaml`) na pasta de resultados, permitindo ao modo standalone
  descobrir o `data_dir` sozinho quando o arquivo existir. Precedência do
  `data_dir`: flag `--data_dir` > `parameters.yaml` > apenas GT local (`sharp/`
  copiado).

## Inventário de ground truth

### GT canônico — `sharp/` no topo do dataset

| Arquivo | Conteúdo | Avalia |
|---|---|---|
| `hAvg.png` (uint16) | Altura GT | `zMos` (multifocus) e `height_map` final |
| `sNrm.png` (uint8 RGB) | Normais GT; decodificação `(v/255)·2−1` (norma≈1 verificada empiricamente) | `normal_map` (fotométrico) |
| `L*/sharp/sVal.png` | All-in-focus de referência por luz | mosaicos `sMos` |
| `hDev.png` (uint16) | Desvio/incerteza da altura GT por pixel | máscara opcional de confiabilidade do GT |

### GT adicional

- **`shrp.png` por plano focal `zf*`** (uint16): nitidez GT de cada plano.
  O argmax da pilha sobre os 12 zf dá o plano de foco GT por pixel — ground
  truth direto para a seleção de foco do multifocus (sem ajuste afim).
  Verificação empírica: pilha varia por zf; argmax mapeado a z_foc correlaciona
  com hAvg (r≈0.77; fundo plano domina o índice 0).
- **Descartados** (documentado): `selected-pixels.png` (1 por luz; semântica
  incerta) e um `hAvg.fni` avulso dentro de um único `zf*` (não confiável /
  não sistemático). `hAvg.png`/`sNrm.png` dentro dos `zf*` não são idênticos ao
  `sharp/` do topo (provável renormalização por pasta) — somente o `sharp/` do
  topo é canônico.

### Layout esperado da pasta de resultados

```
results_dir/
├── sharp/                      ← GT (copiado pelo pipeline) [fallback: data_dir/sharp/]
├── multifocus_stereo/average/  ← zMos.fni, iSel.fni, wSel.fni
├── multifocus_stereo/L*/       ← sMos.png (por luz)
├── photometric_stereo/         ← normal_map.npy, confidence.npy
├── integration/                ← height_map.npy
└── evaluation/                 ← NOVO: metrics.json, report.md, mapas de erro
```

## Métricas por etapa

Comparações de altura usam métricas **afim-invariantes**: ajuste
`gt ≈ a·est + b` por mínimos quadrados nos pixels válidos. Pixels inválidos
(NaN no estimado; opcionalmente `hDev` alto no GT) são excluídos; a **fração de
pixels válidos** é sempre reportada.

1. **Multifocus — profundidade** (`zMos.fni` vs `hAvg.png`):
   RMSE pós-ajuste afim, `a`, `b`, Pearson `r`, MAE; mapa de erro
   `|gt_norm − (a·zMos + b)_norm|` (PNG).
2. **Multifocus — seleção de foco** (`zMos.fni` vs `z_gt` derivado da pilha `shrp`):
   `z_gt = z_foc[argmax(pilha shrp)]`, com os valores de z_foc parseados dos
   nomes das pastas `zf*` do dataset (ex.: `zf045.0000-df020.0000` → 45.0) —
   sem dependência de config. Comparação direta em unidades z_foc (sem ajuste
   afim); erro também expresso em frames (÷ passo focal): mediana, média, p90,
   % de acerto exato e dentro de ±1 frame; mapa de erro (PNG).
   Nota: `iSel.fni` **não** é usado — é salvo normalizado
   (`multifocus/main.py:180`), o que o torna incomensurável com índices de
   frame; `zMos.fni` é o artefato cru equivalente.
3. **Multifocus — mosaicos** (`L*/sMos.png` vs `L*/sharp/sVal.png`):
   PSNR e SSIM por luz + média e mínimo entre luzes (o mínimo aponta a pior luz,
   que degrada o fotométrico); tabela por luz no relatório.
4. **Fotométrico — normais** (`normal_map.npy` vs `sNrm.png` decodificado):
   erro angular em graus — média, mediana, p95. Máscara: exclui NaN do estimado
   e fundo do GT (norma ≈ 0 antes de renormalizar).
   **Resolução do CONV-2**: erro calculado duas vezes — GT como está e GT com
   `ny` invertido; ambos reportados, o menor marcado como principal, e o
   relatório registra a orientação vencedora (decide empiricamente o
   `flip_lights_y` do dataset). Mapa de erro angular (PNG com colormap).
5. **Integração — altura final** (`height_map.npy` vs `hAvg.png`):
   mesmas métricas do item 1. `height_map` é vertex-grid `(H+1, W+1)` (INT-05):
   convertido para cell-grid `(H, W)` pela média dos 4 vértices de cada célula
   antes de comparar.
6. **Síntese — ganho do híbrido**: tabela multifocus sozinho (item 1) vs final
   (item 5) na **interseção** das máscaras de pixels válidos (comparação justa);
   métrica-resumo `ganho = RMSE_multifocus / RMSE_final` (>1 = combinação
   melhorou).

## Saídas

Em `<results_dir>/evaluation/`:

```
evaluation/
├── metrics.json
├── report.md
├── multifocus_depth_error.png
├── multifocus_focus_selection_error.png
├── photometric_angular_error.png
└── integration_height_error.png
```

- **`metrics.json`** — um bloco por etapa com `status` (`"ok"` |
  `"skipped: <motivo>"`), métricas, fração de pixels válidos e caminhos dos
  artefatos/GT usados (rastreabilidade). Bloco `meta`: timestamp, results_dir,
  data_dir, versão do pacote.
- **`report.md`** — tabelas de métricas por etapa, PSNR/SSIM por luz, síntese
  multifocus vs final, veredito do CONV-2, imagens de erro embutidas. Pensado
  para comparação entre experimentos e material de tese.

## Tratamento de erros

- Artefato ou GT ausente → etapa `skipped` com motivo; o run continua
  (avaliação parcial é melhor que nenhuma).
- Shapes incompatíveis (ex.: resultado de outro dataset) → etapa `skipped` com
  mensagem clara; nunca exceção não tratada.
- Pixels válidos < 1% → métricas reportadas com flag `low_validity: true`.
- Exit code ≠ 0 apenas se **nenhuma** etapa pôde ser avaliada.
- Logging no padrão do projeto (console + arquivo em `evaluation/`).

## Configuração (hook)

```yaml
evaluation:
  enabled: true          # default false — opt-in
  hdev_mask: false       # máscara opcional por incerteza do GT (hDev)
  hdev_threshold: 0.1    # fração do range do hDev acima da qual o pixel é excluído
```

Quando `hdev_mask: true`, as métricas de altura são reportadas com e sem a
máscara de hDev (dois conjuntos de números), e a fração mascarada é registrada.

## Testes

- **Unitários `metrics.py`**: casos analíticos — `affine_fit_rmse(2x+3, x)` →
  rmse≈0, a=2, b=3; erro angular de rotações conhecidas; PSNR de imagem
  idêntica → ∞ (tratado); SSIM de idêntica → 1.
- **Unitários `loaders.py`**: decodificação de `sNrm` sintético (ida e volta),
  leitura uint16, conversão vertex→cell.
- **Integração**: pasta de resultados sintética mínima (gerada com
  `tests/synthetic_utils.py`, que já cria superfície + normais + GT coerentes)
  → `run_evaluation()` fim a fim → valida `metrics.json` e `report.md` gerados
  e que etapa com artefato faltante vira `skipped`.
- `tests/synthetic_utils.py::affine_fit_rmse` passa a reexportar de
  `evaluation.metrics`.

## Fora do escopo

- Erro absoluto em unidades físicas (exigiria `reference_scale`/`pixel_size`
  calibrados; as métricas afim-invariantes cobrem a necessidade atual).
- Interpretação de `selected-pixels.png` e dos GT internos aos `zf*`.
- Comparação agregada entre múltiplos experimentos (o `metrics.json`
  estruturado já viabiliza isso externamente).
- Parsing do relatório de erro (`devE`) do solver C.
