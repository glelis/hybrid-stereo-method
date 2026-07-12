# Relatório — Sweep de parâmetros do método híbrido

- **Data:** 2026-07-10
- **Dataset:** `2025-03-08-stQ-melon24-amb0.00-glo0.50`
- **Runs:** 15 configurações em `data/results/hybrid_sweep/<nome>/` (configs em `configs/sweep/`)
- **Reprodução:** `bash configs/sweep/run_all.sh` · tabela: `python configs/sweep/summarize.py`

## Contexto

Antes do sweep, dois bugs foram corrigidos (pré-requisitos dos resultados abaixo):

1. **Clipping uint16 → uint8** em `save_image(normalize=False)`: os `sVal.png` reais são
   PNGs de 16 bits e a média por luz do stack "average" era gravada com 96,9% dos pixels
   saturados em branco, destruindo a entrada do multifocus (commit `16e0d19`).
2. **Filtro de substring `"av"`** em `hybrid/main.py`: zerava a lista de mosaicos do
   fotométrico quando o caminho de saída continha "av" (ex.: `03_wavelet`) (commit `09afd35`).

Diagnóstico de partida: as normais fotométricas já eram ótimas (erro médio 3°); o gargalo
era a profundidade multifocus (Pearson r 0,29 no baseline). O sweep focou nesse elo.

## Desenho do experimento

**Round 1** — um fator por vez a partir do baseline (atribuição causal limpa):
método da medida de foco (fourier/laplaciano/wavelet), banda do fourier (radius 0,05/0,10/0,20),
denoise da medida de foco (smooth + mediana espacial), janela do fuzzy argmax (r_max 2→4),
interpolação do mosaico (linear→quadrática), `top_k` do fotométrico e peso dos hints (0,1→0,3).

**Round 2** — combinações fatoriais dos vencedores do round 1, mais o eixo
`zero_border: False` (identificado em um run avulso fora do sweep).

## Resultados (métricas vs ground truth)

| config | mf Pearson r | foco mediano (frames) | foco ±1fr (%) | altura Pearson r | altura RMSE |
|---|---|---|---|---|---|
| 01_baseline | 0,288 | 4,00 | 30,8 | 0,731 | 12598 |
| 02_laplacian | 0,514 | 1,17 | 47,1 | 0,784 | 11455 |
| 03_wavelet | 0,279 | 4,36 | 31,6 | 0,653 | 13981 |
| 04_fourier_low_freq | 0,462 | 1,73 | 39,8 | 0,808 | 10892 |
| 05_fourier_high_freq | 0,063 | 6,00 | 17,9 | 0,436 | 16617 |
| 06_denoise_focus | 0,476 | 1,00 | 50,4 | 0,753 | 12156 |
| 07_fuzzy_window4 | 0,288 | 4,00 | 30,3 | 0,739 | 12450 |
| 08_quadratic_interp | 0,288 | 4,00 | 30,8 | 0,732 | 12592 |
| 09_photometric_top5 | 0,288 | 4,00 | 30,8 | 0,731 | 12598 |
| 10_strong_hints | 0,288 | 4,00 | 30,8 | 0,698 | 13227 |
| 11_lowfreq_denoise | 0,575 | 0,87 | 53,4 | 0,831 | 10263 |
| 12_lowfreq_noborder | 0,592 | 0,82 | 56,9 | 0,829 | 10325 |
| **13_lowfreq_denoise_noborder** | 0,782 | **0,56** | 76,2 | **0,859** | **9458** |
| 14_laplacian_denoise | 0,568 | 0,82 | 54,3 | 0,825 | 10428 |
| 15_laplacian_denoise_noborder | **0,818** | 0,55 | **76,6** | 0,842 | 9956 |

## Configuração recomendada

**`13_lowfreq_denoise_noborder`** (`configs/sweep/13_lowfreq_denoise_noborder.yaml`):

```yaml
multifocus.focus_measure.parameters.radius: 0.05      # era 0.1
multifocus.focus_measure.preprocessing.smooth: true    # era false
multifocus.focus_measure.preprocessing.spatial_median_filter: true  # era false
multifocus.focus_measure.preprocessing.zero_border: false
```

Contra o baseline: altura final r 0,731 → **0,859**, RMSE 12598 → **9458**,
erro mediano de seleção de foco 4,0 → **0,56 frames**.

## O que foi aprendido

1. **Banda de frequência do fourier é o fator mais sensível.** Frequências baixas
   (radius 0,05) melhoram muito; altas (0,20) destroem o resultado (r 0,44 na altura).
   A textura útil de foco deste dataset está nas frequências médias-baixas.
2. **Os três fatores vencedores são quase perfeitamente complementares** — banda baixa,
   denoise e zero_border off contribuem de forma independente (visível na cadeia
   04 → 11/12 → 13), então combiná-los soma os ganhos.
3. **Laplaciano é o melhor para o multifocus isolado** (config 15, r 0,818), mas o
   fourier low-freq gera a melhor altura final. Se o multifocus for usado sozinho,
   preferir a config 15.
4. **Wavelet é o pior método de foco** para este dataset (pior que o baseline).
5. **Hints mais fortes pioram** (peso 0,3 < 0,1): o prior do multifocus ajuda como
   regularização fraca, não como restrição forte.
6. **`r_max` e a interpolação do mosaico têm efeito desprezível** neste dataset.
7. **`photometric.solver.top_k` é um no-op**: o solver robusto usado pelo pipeline
   (`estimate_normals_argmax_lstsq_robust`) só consome `shadow_threshold` e
   `outlier_threshold_multiplier`. Um próximo sweep do fotométrico deve variar esses dois.
8. **O "ganho do híbrido" cai nas configs boas** (1,40 → 1,22) — não é regressão:
   o multifocus melhora tanto que sobra menos para o fotométrico corrigir.
9. **Metodologia:** mudar um fator por vez foi o que permitiu detectar o no-op do
   `top_k` e uma edição manual de config no meio dos experimentos (runs com o mesmo
   YAML são bit-a-bit reprodutíveis; qualquer divergência indica mudança de entrada).
