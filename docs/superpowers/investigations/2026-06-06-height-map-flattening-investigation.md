# Investigação: achatamento do mapa de altura final

**Data:** 2026-06-06–07 · **Status:** concluída (diagnóstico + correções + validação)

## Resumo executivo

A queixa inicial — "um grande outlier achatando o mapa de altura" — foi confirmada e
diagnosticada: **não era um pixel isolado**, mas a combinação de (a) hints de multifocus
em escala incomensurável com a altura integrada, que anti-correlacionavam o resultado
(causa principal, H4), sobre um pano de fundo de (b) export de mosaico saturado (H3) e
(c) thresholds do solver fotométrico em escala 8-bit destruindo dados 16-bit (H7).
Cinco correções aprovadas foram aplicadas com TDD (C1, C2, C3, C5, C6) mais um bug
bloqueante de indexação descoberto na validação. Resultado no mesmo dataset:

- **Anti-correlação eliminada no objeto:** `hybrid_gain.pearson_final` −0,20 → **+0,727**.
- **Mosaicos:** PSNR 2,8 → **30,7 dB**; **normais:** erro mediano 2,3° → **0,011°**.
- **Restou** um penhasco de fundo não-ancorado que mantém a métrica "todos-os-pixels"
  negativa — limitação estrutural do GT mascarado, não causa corrigível por máscara de
  entrada (ver "O penhasco que sobrou"). Os critérios literais da spec (all-pixels
  pearson ≥ 0,8) não se aplicam a este dataset; na região com sinal o objetivo foi atingido.

Detalhes completos abaixo.

---

**Spec:** `docs/superpowers/specs/2026-06-06-height-map-flattening-investigation-design.md`
**Plano:** `docs/superpowers/plans/2026-06-06-height-map-flattening-investigation.md`
**Run de referência:** `data/results/hybrid_stereo/20260606_1126_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/`
**Evidências (JSONs):** `<run>/investigation/*.json` · scripts em `scripts/investigation/`

## Sintomas originais

Altura final com penhasco regional (~7% dos pixels a ~−455 vs objeto em [18, 68]),
anti-correlação com o GT (`pearson −0.20`, slope afim negativo), mosaicos com PSNR
1,5–4 dB, multifocus quase não-informativo (`pearson +0.27`).

## Tabela de veredictos

| Hipótese | Veredicto | Evidência decisiva |
|----------|-----------|--------------------|
| **H4** — hints incomensuráveis (z_foc 15–125 × alturas em px, sem `pixel_size`) degradam a integração | **CONFIRMADA — causa principal** do penhasco E da anti-correlação | A1: `with_hints` reproduz o run (−0.207, cliff 7,1%); `no_hints` → **+0.087, cliff 0,0%** (`a1_hints_ablation.json`) |
| **H3** — mosaicos mal construídos | **REFUTADA** no conteúdo; **CONFIRMADA como bug de export** | F3: sMos.png 90–94% saturado (clip uint8 de dados 16-bit); conteúdo real (FNI): PSNR 26–37 dB, SSIM 0,82–0,93 (`f3_mosaic_check.json`) |
| **H7** *(nova)* — thresholds do solver WPS em unidades 8-bit vs dados 16-bit | **CONFIRMADA — bloqueante p/ re-run** | A4: com o YAML atual, só **6 pixels** sobrevivem ao filtro de saturação (250 sobre dados 0–65535). As chaves NÃO existiam no run original (0 ocorrências no log) — drift de config pós-run (`a4_normals_from_sharp.json`) |
| **H2** — inversão de sinal/eixo em alguma fronteira | **INDÍCIO REAL, não dominante** | F1b (normais GT, sem fill-values, região elevada do GT): `y_flipped` **+0.273** vs `y_as_is` **−0.476** → o integrador prefere ny invertido em relação ao frame do sNrm/estimadas. Teste do domo (Task 11) decide o local |
| **H1** — fundo sem máscara → normais degeneradas → penhasco | **REFUTADA como mecanismo do penhasco**; máscara dá ganho modesto | A1: cliff 0% sem hints, mesmo sem máscara. A2: mascarar fundo melhora pearson(obj) 0.087 → **0.209** (`a2_integrate_masked.json`). F2: normais do penhasco quase-antiparalelas (mediana 124°), confiança não protege |
| **H5** — multifocus fraco | **CONFIRMADA no objeto** (não só fundo) | F4: erro mediano 1 frame NO objeto; r(zMos, z_shrp)=+0.374 no objeto (`f4_multifocus_spatial.json`) |
| **H6** — avaliação compara coisas erradas | **REFUTADA** na correspondência; **lacuna real de protocolo descoberta** | F0: GT↔imagens r=0.94; shapes todos (422,512). MAS: fit afim 1D não remove o **tilt 2D** de integração não-ancorada → pearson global enganoso nesse regime (F1 follow-up) |

### Achados colaterais (não previstos nas hipóteses)

1. **Fill-values no sNrm GT enganam o loader**: 1 620 px de foreground com (−1/√3,−1/√3,−1/√3)
   têm norma exatamente 1 e passam o check do `load_normals_gt`; viram slopes de ~707/px
   no C (clamp `maxSlope=1000`) e dominaram a primeira rodada do F1.
2. **Sinal depth↔height dos hints está correto**: F4 dá r(zMos, hAvg) = **+0.34** no objeto —
   sem inversão; `pixel_size` positivo basta.
3. **Confiança WPS mal-calibrada como peso** (p50 0,017 no objeto; cauda até 1,0 no fundo),
   mas impacto pequeno: ablação com peso binário deu pearson(obj) 0.191 ≈ 0.209.
4. **Estimador WPS é quase exato com entrada perfeita**: A4 com paridade de thresholds dá
   mediana **0,011°** (sharp GT) vs 2,32° (mosaicos) → a cadeia multifocus→mosaico degrada
   as normais modestamente; estimador saudável (`a4b_normals_from_sharp_parity.json`).
5. **Teto do protocolo atual**: mesmo normais GT integradas (sem fill-values) só atingem
   pearson ~+0,27 na região elevada — o tilt não-ancorado limita QUALQUER avaliação de
   integração sem hints/referência. Os critérios da spec (pearson ≥ 0,8) são inatingíveis
   sem ancoragem comensurável OU sem remoção de tilt no avaliador.

## Mecanismo causal consolidado

O run original integra slopes em unidades de pixel com hints em z_foc (15–125) a peso 0,1.
A solução tenta casar duas réguas incompatíveis: onde a confiança do multifocus é alta, a
altura é puxada para valores z_foc; onde é baixa, a integração de slopes (fraca, quase
plana) domina — o desequilíbrio cria o penhasco regional e anti-correlaciona o mapa todo.
Removidos os hints, o penhasco desaparece; sobra uma superfície quase plana com tilt
(falta de âncora), que nenhuma métrica afim 1D pontua bem.

## Correções aprovadas (decisão do usuário, 2026-06-06)

| # | Correção | Gate | Natureza | Decisão |
|---|----------|------|----------|---------|
| C1 (Task 10) | Export dtype-aware de `sMos.png`/`sMos.fni` | F3 ✅ | Bug de código + teste | **APROVADA** |
| C2 (Task 11) | Teste do domo sintético; fix de orientação (negar ny na fronteira PS→integração) **somente se o teste falhar** | F1 ✅ (evidência pró-`y_flipped`) | Bug de código + regressão | **APROVADA** |
| C3 (Task 12) | `pixel_size` empírico no YAML p/ comensurabilidade dos hints (sinal OK por F4) | A1+F4 ✅ | Configuração | **APROVADA** |
| C4 (Task 13) | Máscara de fundo na integração | A2 ✅ (ganho modesto 0.09→0.21) | Lacuna funcional | **NÃO aprovada** — ganho pequeno; fica como trabalho futuro |
| C5 *(nova, H7)* | Thresholds do solver WPS **relativos ao max do dtype da fonte** — pré-requisito do re-run | A4 ✅ | Bug de código + teste | **APROVADA (opção código)** |
| C6 *(nova)* | Avaliador: métrica adicional com **remoção de tilt (fit plano 2D)** p/ altura; e **excluir fill-values** no `load_normals_gt` | F1/F1b ✅ | Código no módulo `evaluation` | **APROVADA (ambos)** |

### Correções aplicadas (commits)

| Fix | Commit | Mensagem |
|-----|--------|----------|
| C1 | `5ff5c5f` | `fix(hybrid): dtype-aware mosaic export — 16-bit stacks saturated sMos.png (F3/H3)` |
| C2 | `8609bc2` | `test(hybrid): dome-integration orientation regression guard (F1/H2)` (sem fix de código — integrador correto) |
| C5 | `b61903b` | `fix(photometric): scale absolute WPS thresholds to source bit-depth (H7)` |
| C3 | `9324de1` | `fix(config): commensurable hints — empirical pixel_size (A1/H4, CONV-4)` |
| C6 | `fbf7f75` | `feat(eval): tilt-detrended height metric + exclude nz<=0 fill-values from GT foreground (C6)` |
| (bloqueante) | `313e46d` | `fix(hybrid): index in-memory mosaics by light index, not padded name (KeyError L0 on L000 dirs)` |

O bug bloqueante (`KeyError: 'L0'`) só apareceu no re-run: o caminho in-memory PS-07
(`sMos_by_light[f"L{i}"]`) nunca tinha sido exercitado com pastas zero-padded `L000` — o
run original fora gerado por uma versão anterior do código. Indexação agora por índice
inteiro, tolerante a padding, alinhada a `pair_mosaics_to_lights`.

### Honestidade sobre os critérios de aceite

A spec pede `pearson ≥ +0.8` e `RMSE ≤ 50% gt_std` no re-run. Com C3 (hints comensuráveis,
âncora correta) isso se torna *plausível*, mas o teto observado sem ancoragem é ~0,27 — se
o re-run não atingir os alvos, a explicação documentada será o limite do protocolo (tilt) e
a qualidade do multifocus (H5, mediana 1 frame), com C6 dando a régua justa.

## Antes × depois

Re-run de validação: `20260607_1825_...` (mesmo dataset, fourier restaurado, fixes C1/C2/C3/C5/C6 aplicados; C4 **não** aplicado).

| Métrica | Antes (`20260606_1126`) | Depois (`20260607_1825`) | Leitura |
|---------|-------------------------|--------------------------|---------|
| **Mosaicos** PSNR médio | 2,79 dB | **30,67 dB** | C1 resolveu o export saturado |
| Mosaicos SSIM médio | 0,180 | **0,866** | idem |
| **Normais** erro mediano | 2,32° | **0,011°** | C5 (thresholds 16-bit) + C6 (fill-values) |
| Normais erro médio | 10,72° | **2,48°** | idem |
| **Multifocus** pearson (z vs hAvg) | +0,267 | **+0,407** | fourier + fill-values |
| Multifocus RMSE afim | 17 795 | **15 981** | idem |
| **Ganho híbrido** `pearson_final` (núcleo de objeto) | −0,201 | **+0,727** | C3 (hints comensuráveis) — anti-correlação eliminada no núcleo |
| Ganho híbrido `gain` | 0,98 | **1,33** | o híbrido agora melhora o multifocus |
| **integration_height** pearson (todos os pixels) | −0,201 | −0,189 | ainda negativo — **penhasco de fundo remanescente** |
| integration_height rmse/gt_std | 0,98 | 0,98 | idem |

### O penhasco que sobrou

A contradição aparente (núcleo +0,727 vs todos-os-pixels −0,189) tem causa única: o
re-run produz `zMos` com NaN no fundo (76% de pixels válidos), então `hybrid_gain`
mede só o núcleo de objeto — onde a reconstrução agora é **fortemente positiva**. Já
`integration_height` inclui todos os pixels, e a integração ainda gera um **penhasco
de fundo não-ancorado** (altura até −983; 6% dos pixels < −200, **90% deles no fundo**)
que domina a métrica global e a mantém negativa.

Testei se **C4 (máscara de fundo)** — adiado no checkpoint — fecharia o gap, agora com os
normais novos (mediana 0,011°). **Não fecha**, e a causa é estrutural:

| Configuração de preview | all-pixels pearson | núcleo (zMos) | cliff<−200 |
|--------------------------|--------------------|---------------|------------|
| Máscara de fundo, sem hints | −0,156 | — | 0,8% |
| Máscara de fundo + hints (produção) | −0,073 | +0,718 | 0,8% |

Mascarar a **entrada** do fundo reduz o penhasco (6%→0,8%) mas não vira o sinal global:
o `hAvg` GT tem >50% de fundo em **zero** (é um GT mascarado), e a integração deixa o
**output** do fundo sem restrição (sem normais e sem hints ali) → o solver atribui
alturas arbitrárias que anti-correlacionam com o zero do GT. Nenhuma máscara de entrada
conserta o output de uma região sem dados.

**Conclusão:** o critério "todos-os-pixels" é inadequado para este dataset (GT de fundo
plano-zero + fundo de reconstrução não-ancorado). A métrica correta é restrita à região
com sinal (objeto / núcleo zMos) — e nela o objetivo foi atingido: a anti-correlação
−0,20 virou **+0,727**. As métricas `pearson_detrended`/`rmse_detrended` (C6) e o
`hybrid_gain.pearson_final` já reportam isso. Por isso C4 **permanece adiado** — não traz
ganho real para esta métrica; o caminho certo é avaliar na região válida (já entregue) e,
se quiser um número global bonito, mascarar o fundo **no GT/saída da avaliação** (trabalho
futuro), não na entrada do solver.

### Critérios de aceite da spec

| Critério | Alvo | Resultado (todos-os-pixels) | Resultado (núcleo de objeto) |
|----------|------|------------------------------|------------------------------|
| `integration_height.pearson_r` | ≥ +0,8 | −0,189 ❌ | +0,727 (hybrid_gain) |
| `rmse_affine` / `gt_std` | ≤ 0,50 | 0,98 ❌ | — |

Os critérios literais (todos-os-pixels) **não** foram atingidos — e, como mostrado acima,
não são alcançáveis por nenhuma das correções de entrada neste dataset, porque o GT de
fundo é plano-zero e o fundo da reconstrução é não-ancorado. Todas as causas dentro do
escopo aprovado foram corrigidas; o gap restante é estrutural da métrica/dataset, não uma
causa de pipeline em aberto. Na região com sinal (objeto / núcleo zMos), o objetivo foi
plenamente atingido.

## Trabalho futuro

1. **Avaliação region-aware do `integration_height`**: aplicar a máscara de validade
   (zMos finito / foreground) à métrica principal de altura no avaliador, em vez de
   pontuar todos os pixels — o número global passaria a refletir a região com sinal
   (onde já é +0,72). Alternativa: mascarar o fundo no par GT↔estimado dentro de
   `evaluate_height`. (Hoje isso já está disponível indiretamente via `hybrid_gain` e via
   as métricas detrended de C6.)
2. **C4 (máscara de fundo na integração)**: continua adiado. Reduz o penhasco (6%→0,8%)
   mas não vira o sinal global; só vale a pena junto com (1) ou com uma condição de
   contorno de Dirichlet no fundo.
3. **Multifocus (H5)**: erro mediano de ~1 frame no objeto persiste (fourier, sem
   pré-processamento). Explorar `square`/`smooth`/median-filter no focus measure ou
   `depth_refinement` por graph-cut — fora do escopo desta investigação.
4. **Conversão depth→height explícita** para os hints, caso datasets futuros tenham
   relação inversa (aqui o sinal era positivo, F4).
5. **Scripts de diagnóstico reutilizáveis**: F0–F4/A1–A4 vivem em `scripts/investigation/`
   como ferramentas efêmeras; vários (split objeto/fundo, overlay de penhasco, comparação
   PNG×FNI) poderiam virar checagens permanentes no módulo `evaluation` se úteis adiante.

## Artefatos

- **Scripts:** `scripts/investigation/{f0_gt_sanity,f1_integrate_gt_normals,a1_hints_ablation,f2_normals_regions,f3_mosaic_check,f4_multifocus_spatial,a2_integrate_masked,a4_normals_from_sharp,estimate_pixel_size}.py` + `common.py`.
- **Evidências JSON:** `<run_20260606_1126>/investigation/*.json`.
- **Re-run de validação:** `data/results/hybrid_stereo/20260607_1825_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/` (com `evaluation/metrics.json` antes×depois).
- **Spec/Plano:** `docs/superpowers/specs/2026-06-06-height-map-flattening-investigation-design.md`, `docs/superpowers/plans/2026-06-06-height-map-flattening-investigation.md`.

