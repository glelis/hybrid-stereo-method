# Investigação: achatamento do mapa de altura final — diagnóstico

**Data:** 2026-06-06 · **Status:** diagnóstico concluído; correções aguardando aprovação
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

## Correções propostas (aguardando sua aprovação)

| # | Correção | Gate | Natureza | Status |
|---|----------|------|----------|--------|
| C1 (Task 10) | Export dtype-aware de `sMos.png`/`sMos.fni` | F3 ✅ aberto | Bug de código + teste | Pronta p/ executar |
| C2 (Task 11) | Teste do domo sintético; fix de orientação (negar ny na fronteira PS→integração) **somente se o teste falhar** | F1 ✅ aberto (evidência pró-`y_flipped`) | Bug de código + regressão | Pronta p/ executar |
| C3 (Task 12) | `pixel_size` empírico no YAML p/ comensurabilidade dos hints (sinal OK por F4) | A1+F4 ✅ aberto | Configuração | Pronta p/ executar |
| C4 (Task 13) | Máscara de fundo na integração (mask.png > derivação por intensidade) | A2 ✅ (ganho modesto 0.09→0.21) | Lacuna funcional | **Decisão de produto: vale o ganho modesto? Estratégia default ok?** |
| C5 *(nova, H7)* | Thresholds do solver relativos à escala da fonte (ou valores 16-bit no YAML) — **pré-requisito do re-run de validação** | A4 ✅ | Configuração (mínимо) ou código (robusto) | **Precisa decisão: config-only vs fix no código** |
| C6 *(nova)* | Métrica adicional no avaliador: fit plano 2D (remove tilt) p/ altura integrada; e excluir fill-values no `load_normals_gt` | F1/F1b ✅ | Código no módulo `evaluation` | **Precisa aprovação (módulo recém-entregue)** |

### Honestidade sobre os critérios de aceite

A spec pede `pearson ≥ +0.8` e `RMSE ≤ 50% gt_std` no re-run. Com C3 (hints comensuráveis,
âncora correta) isso se torna *plausível*, mas o teto observado sem ancoragem é ~0,27 — se
o re-run não atingir os alvos, a explicação documentada será o limite do protocolo (tilt) e
a qualidade do multifocus (H5, mediana 1 frame), com C6 dando a régua justa.

## Antes × depois

Preenchido após o re-run de validação (Task 14).
