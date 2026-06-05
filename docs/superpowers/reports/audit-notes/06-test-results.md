# Resultados dos testes sintéticos

Data: 2026-06-04

---

## test_fni_roundtrip (Task 8)

- **Comando:** `pytest tests/test_fni_roundtrip.py -v`
- **Resultado:** `4 passed in 0.36s`

```
tests/test_fni_roundtrip.py::test_roundtrip_2d PASSED                    [ 25%]
tests/test_fni_roundtrip.py::test_roundtrip_3channel PASSED              [ 50%]
tests/test_fni_roundtrip.py::test_roundtrip_negative_and_large_values PASSED [ 75%]
tests/test_fni_roundtrip.py::test_nan_handling_documented PASSED         [100%]
```

- **Interpretação:**
  - Round-trip Python↔Python sem flip/transposição confirmado: o teste 2D com array (5,9) assimétrico e o teste 3-channel com (4,6,3) reconstroem shapes e valores idênticos. Consistente com a entrada "Verificado" de `04-io.md` (indexação `[y,x]` confere em escrita e leitura — agora com evidência executável).
  - NaN sobrevive ao round-trip Python: `test_nan_handling_documented` passa, confirmando que `float("+nan")` no parser Python reconstrói NaN corretamente. Consistente com a nota "Verificado" em `04-io.md` (`float("+nan")` devolve `NaN` em CPython).
  - Precisão efetiva `rtol~6e-8` (max rel diff observado): `read_fni_to_image_array` aloca `float32`, então o round-trip float64→texto `%.7e`→float32 introduz erro de ~6e-8 relativo — dentro de `rtol=1e-6` e consistente com IO-01 (truncamento `%.7e` limita-se ao epsilon de float32 ~1.19e-7). O retorno `float32` do reader não estava explicitamente documentado nos achados; confirma que a fronteira de precisão é no reader, não apenas no writer.
  - Valores negativos, grandes e zero sobrevivem corretamente (incluindo `-0.0` que é armazenado como `+0.0000000e+00` e recuperado como `0.0` — comportamento float normal sem achado).

---

## test_convention_integration (Task 9)

- **Comando:** `pytest tests/test_convention_integration.py -v -s`
- **Build do binário C:** `cd csrc/integrate_recursive && make` **FALHA** (ver `## Bloqueios`), mas o binário `gus_integrate_recursive` **já está versionado e funcional** (executa e responde ao `--help`); `DEFAULT_EXECUTABLE.exists()` é `True`, então o guard `needs_binary` NÃO pulou os testes — eles rodaram contra o binário pré-compilado.
- **Resultado:** `1 failed, 1 passed in 0.26s`
  - `test_constant_slopes_recover_ramp_and_decide_convention` — **FAILED** (crash ao invocar o binário; NÃO foi um assert de convenção)
  - `test_normals_path_recovers_bump_shape` — **PASSED**

### (1) Caminho `-slopes` (rampa): CRASH do binário — a tabela de convenção NÃO chegou a ser impressa

O teste falhou **antes** de calcular/imprimir a tabela de candidatos: o `integrate_slopes_to_height` levantou `RuntimeError` dentro da chamada ao C, antes de retornar `z`. Logo **não há tabela de RMSE por candidato** para registrar — o assert de convenção (`best == "z = +ax*x + ay*y ..."`) nunca foi alcançado.

Saída de erro do binário (verbatim, do `RuntimeError`/stderr capturado):
```
RuntimeError: Integration failed: reading the slope map {G} ...
Reading .../ramp_slopes.fni ...
allocating the height map {Z} ...
zeroing the initial solution ...
writing out the initial guess ...
wrote .../ramp-ini-Z.fni
  writing .../ramp-00-beg-G.fni ...
    writing .../ramp-01-beg-G.fni ...
      writing .../ramp-02-beg-G.fni ...
        writing .../ramp-03-beg-G.fni ...
          writing .../ramp-04-beg-G.fni ...
pst_integrate_iterative.c:47: ** (pst_integrate_iterative) slope map {G} must have 3 channels
```

Causa-raiz (confirmada na fonte C):
- O entry-point de topo aceita 2 **ou** 3 canais: `demand((NC_G == 2) || (NC_G == 3), "gradient map {G} must have 2 or 3 channels")` (`gus_integrate_recursive.c:503`).
- Mas o solver iterativo interno exige **exatamente 3 canais**: `demand(NC_G == 3, "slope map {G} must have 3 channels")` (`lib-src/pst_integrate_iterative.c:47`).
- O `integrate_slopes_to_height` grava um mapa `(H,W,2)` (canais dZ/dX, dZ/dY) — exatamente o que a docstring documenta como aceitável (`integrate.py:261-263`: "shape (H, W, 2) or (H, W, 3)"). O topo aceita; a recursão **estoura** dentro de `pst_integrate_recursive`→`pst_integrate_iterative`, no nível 0 após escrever os `-NN-beg-G.fni`.
- **Conclusão:** o caminho de **slopes com 2 canais** está quebrado por construção (topo promete 2/3 canais, solver exige 3, e o topo não promove 2→3 antes de recursar). O caminho `-normals` não sofre disso (normais entram com 3 canais Nx,Ny,Nz). **Achado novo de implementação** (registrar como INT/CONV; ver abaixo).

Consequência para CONV-1/CONV-2: o teste da rampa **não pôde decidir** o sinal/orientação de dZ/dY **pelo caminho `-slopes`** com o wrapper Python atual — ele crasha antes de integrar. A decisão de sinal continua **pendente de execução** por essa via; um teste futuro precisaria (a) passar um mapa de 3 canais (dZ/dX, dZ/dY, peso) ao `-slopes`, ou (b) decidir o sinal pelo caminho `-normals` com uma rampa assimétrica. O teste, como escrito (2 canais), expõe o bug do wrapper em vez de medir a convenção.

### (2) Caminho `-normals` (bump): PASSED — integração end-to-end funciona

```
bump: affine-fit rmse=0.0732, a=0.9989, b=1.1207, std(gt)=1.0384
```
- `rmse=0.0732 < 0.15·std(gt)=0.1558` ⇒ forma recuperada a menos de afim. `a≈0.999` (ganho ~unitário) e `b≈1.12` (offset, esperado: constante de integração arbitrária). `np.isfinite(est).all()` ok.
- Confirma: o caminho `-normals` integra um bump simétrico e produz `bump-00-end-Z.fni` corretamente (callback `reportHeights` final ⇒ end-Z existe em sucesso). Como o bump é simétrico, este caminho **não** decide sinais de X/Y (conforme docstring do próprio teste).

### Interpretação / o que isto decide

- A convenção real do C **NÃO foi decidida pelo teste da rampa** porque o caminho `-slopes`/2-canais crasha antes de integrar. CONV-1 e CONV-2 permanecem indecidos-por-execução nesta via; a leitura estática (sinais `-nx/nz`, `-ny/nz` corretos internamente; y-up das luzes vs y-down do numpy não reconciliado) segue de pé como **suspeita não refutada**.
- A integração `-normals` é confirmada funcional end-to-end (INT verificações de "end-Z garantido em sucesso").
- O crash confirma empiricamente a análise de **INT-02**: um `demand` de canais falho aborta o binário com retorno != 0, `subprocess.run(check=True)` levanta `CalledProcessError` **antes** do fallback `-ini-Z.fni` — o fallback **não** é alcançado num crash (apesar de o `ramp-ini-Z.fni` ter sido escrito em disco, ele nunca é lido).
- **Achado novo de implementação:** `integrate_slopes_to_height` com 2 canais é inutilizável (topo aceita 2/3, solver iterativo exige 3) — documentação do wrapper (`integrate.py:261-263`) incorreta para o caso de 2 canais.

### Extensão: rampa via -normals

Como o caminho `-slopes`/2-canais crasha (INT-08), a rampa foi reenviada pelo caminho **`-normals`**, que funciona (bump passou). Um plano inclinado `z = ax·x + ay·y` tem normal constante `n ∝ (-ax, -ay, 1)`; integrar essa normal e comparar a altura recuperada contra os 5 candidatos de orientação decide a convenção real do eixo-y/sinal do integrador.

- **Comando:** `pytest tests/test_convention_integration.py::test_ramp_normals_decide_convention -v -s -o addopts=""`
- **Resultado:** `1 passed in 0.15s` (ambos os asserts passaram: o melhor candidato ajusta bem **e** coincide com a convenção presumida pelo lado Python).
- **Tabela (verbatim):**
```
RMSE por candidato de convenção (rampa via -normals):
      0.000052  z = +ax*x + ay*y (y do numpy, para baixo)
      0.380857  z = +ax*x - ay*y (y invertido: para cima)
      0.403960  z = +ay*x + ax*y (eixos trocados)
      0.952142  z = -ax*x + ay*y
      1.025489  z = -ax*x - ay*y (tudo invertido)
```
- **Interpretação — o que isto DECIDE:**
  - O candidato vencedor é `z = +ax*x + ay*y (y do numpy, para baixo)` com RMSE `0.000052`, ~3-4 ordens de grandeza abaixo do segundo colocado (`0.38`, o flip de y). A separação é nítida — a rampa assimétrica resolve o sinal/orientação sem ambiguidade.
  - **CONV-1 (sinal end-to-end):** REFUTADO como inconsistência para o caminho `-normals`. A cadeia completa numpy→FNI→C→FNI→numpy preserva o sinal de ponta a ponta (não há inversão global; `-ax*x-ay*y` é o pior candidato).
  - **CONV-2 (direção do eixo-y do integrador):** REFUTADO como inconsistência **apenas no que toca ao eixo-y do integrador**. O `y` do numpy (para baixo) é preservado no round-trip; qualquer convenção interna y-up do C **cancela** na ida-e-volta (o flip de y, `+ax*x-ay*y`, é fortemente rejeitado). Eixos NÃO estão trocados (`+ay*x+ax*y` rejeitado).
  - **Escopo / o que permanece em aberto:** este teste decide a convenção do eixo-y **do integrador** (caminho `-normals`). A outra metade de CONV-2 — o referencial-y do `lights.npy` (luzes vs. eixos da imagem durante o PS) — é um elo **separado** e NÃO é exercido aqui; permanece em aberto, a ser sondado pelo teste end-to-end da Task 12 (que cobre a cadeia completa incluindo as luzes).
  - **convention-3 (origem/offset):** não afetada — o ajuste é feito sobre `z - z.mean()` (a constante de integração arbitrária é removida), consistente com o que já era esperado.

---

## test_photometric_synthetic (Task 10)

- **Comando:** `pytest tests/test_photometric_synthetic.py -v -s`
- **Resultado (1ª execução, antes do xfail):** `2 failed, 2 passed in 0.92s`
- **Resultado final (com xfail em PS-01 e PS-03):** `2 passed, 2 xfailed in 1.01s`

### Saída verbatim (1ª execução)

```
tests/test_photometric_synthetic.py::test_wps_recovers_normals_clean_data
limpo: erro angular médio = 0.004°, p95 = 0.013°
PASSED

tests/test_photometric_synthetic.py::test_wps_albedo_recovers_true_albedo
n_lights=4: albedo mediano = 1.7 (verdadeiro: 200.0)
FAILED
AssertionError: albedo mediano 1.7 != 200.0 com 4 luzes — se cresce com sqrt(n_lights), confirma o achado do albedo

tests/test_photometric_synthetic.py::test_wps_robust_to_saturation
saturado: erro angular médio = 15.127°
FAILED
AssertionError: robustez insuficiente a saturação: 15.13°

tests/test_photometric_synthetic.py::test_wps_shadowed_pixels_flagged_not_garbage
sombras: válidos = 100.0%, erro médio nos válidos = 0.00°
PASSED
```

### Análise adicional: albedo por nº de luzes (sonda extra após captura)

```
n_lights=4: albedo mediano = 1.7051, ratio med/sqrt(n_lights) = 0.8526
n_lights=8: albedo mediano = 2.4114, ratio med/sqrt(n_lights) = 0.8526
```

O ratio `med / sqrt(n_lights)` é **constante** (0.8526) para 4 e 8 luzes, confirmando exatamente a lei de escala `albedo ≈ sqrt(N) × const(geometria)` prevista por PS-01. O valor verdadeiro ρ = 200 não é sequer aproximado (med/ρ ≈ 0.009 em ambos os casos).

### Interpretação — o que cada teste decide

#### test_wps_recovers_normals_clean_data — PASSED
- Erro angular médio = **0.004°**, p95 = **0.013°** com 6 luzes e dados sintéticos limpos (ρ=200, sem ruído).
- **Confirma:** a matemática Woodham (formulação `L·n̂=I`, `lstsq`, normalização posterior) está **correta** para dados sem perturbação. Valida o item "Verificado sem achado — `np.linalg.lstsq(..., rcond=None)` no solver robusto" de `02-photometric.md`. Os erros abaixo de 0.1° são devidos apenas a arredondamento float32 e `epsilon=1e-6`, sem erro sistemático.

#### test_wps_albedo_recovers_true_albedo — XFAIL (confirma PS-01)
- 4 luzes: albedo mediano = **1.7** (verdadeiro: 200); 8 luzes: **2.4**.
- O ratio `med / sqrt(n_lights)` é **constante = 0.8526**, confirmando a lei de escala `albedo ≈ sqrt(N) × f(geometria)`.
- **PS-01 CONFIRMADO:** `wps.py:198` computa `albedo = ||L_sel @ n̂||` (norma das intensidades preditas com normal unitária), que é `sqrt(Σ_k cos²θ_k)` — não o albedo ρ. A quantidade cresce com sqrt(N) conforme previsto. O xfail foi aplicado APÓS captura da falha bruta com as medições acima registradas.

#### test_wps_robust_to_saturation — XFAIL (confirma PS-03)
- Erro angular médio = **15.13°** com 2 das 8 luzes saturadas (cap a 60% do máximo), acima do limiar de 5° esperado para um solver robusto.
- **PS-03 CONFIRMADO:** o loop de outliers com `3 × mean(|residuals|)` falha quando a saturação afeta 25% das luzes. A média dos resíduos é inflada pelas luzes saturadas, elevando o limiar a ponto de não rejeitar nada (mascaramento clássico previsto em PS-03). O xfail foi aplicado APÓS captura da falha.

#### test_wps_shadowed_pixels_flagged_not_garbage — PASSED
- Luz rasante (tilt 75°), 5 luzes: **100% pixels válidos**, erro angular médio nos válidos = **0.00°**.
- **Interpretação PS-02:** com tilt 75° e bump de amplitude 10 σ=0.15, a superfície inclinada *tinha* regiões com n·l < 0 — o teste DE FATO exercita sombras attached (~10.9% das entradas (pixel,luz) com n·l<0 foram corretamente rejeitadas por entrada; 32.9% dos pixels tinham ao menos uma luz sombreada). O threshold `pixel_values / v_max > shadow_threshold` é avaliado **por entrada** (por luz): uma luz sombreada (I ≈ eps) é rejeitada sempre que o pixel tem ao menos uma luz iluminada (v_max grande), independentemente de quantas outras luzes estão em sombra. A rejeição funciona como projetada **neste regime** porque o render float produz zeros exatos nas sombras (I = eps → razão ≈ eps/v_max ≪ 1e-3). A fraqueza do threshold 1e-3 é específica para valores de sombra próximos-mas-não-zero, que o render float não produz. Verificação independente do revisor: substituindo os zeros exatos pelo piso 8-bit (1/255 ≈ 3.9e-3 > 1e-3), o erro angular médio sobe de 0.004° para 3.6° (máx 20°) — reforça manter PS-02 como suspeita para dados reais de 8 bits. **PS-02 não confirmado por este teste** para o caso sintético float (a fraqueza do threshold não é ativada); permanece suspeita para dados quantizados a 8 bits.

### Achados decididos por esta Task

| Achado | Status anterior | Status após Task 10 |
|--------|----------------|---------------------|
| PS-01 | suspeita | **confirmado** (teste xfail; med 1.7/2.4 vs ρ=200; ratio sqrt-N constante) |
| PS-02 | suspeita | **não confirmado** pelo teste sintético (permanece suspeita para 8-bit) |
| PS-03 | suspeita | **confirmado** (teste xfail; 15.13° com 2/8 luzes saturadas) |
| Woodham correto | verificado sem achado (estático) | **confirmado por execução** (0.004° em dados limpos) |

---

## test_multifocus_synthetic (Task 11)

- **Comando:** `pytest tests/test_multifocus_synthetic.py -v -s`
- **Resultado:** `3 passed in 1.34s`

```
tests/test_multifocus_synthetic.py::test_recovers_tilted_plane_depth
plano: mediana|iSel-z| = 0.181 frames, p90 = 0.403
PASSED
tests/test_multifocus_synthetic.py::test_recovers_bump_depth
bump: mediana|iSel-z| = 0.171 frames
PASSED
tests/test_multifocus_synthetic.py::test_textureless_region_gets_zero_confidence
conf média: sem textura = 0.3299, com textura = 0.8666
iSel na região sem textura: mediana = 1.90 (gt = 4.0)
PASSED
```

### Interpretação

#### test_recovers_tilted_plane_depth — PASSED

- Erro mediano `|iSel - z| = 0.181 frames`, p90 = `0.403 frames` no interior (borda 8px excluída), contra limiar de 0.5.
- **Sub-frame atingido:** a precisão de ~0.18 frames confirma que o conjunto `focus_indicator(laplacian)` + `compute_argmax_fuzzy` produz profundidade sub-pixel válida em uma superfície texturizada com gradiente suave de foco. A pipeline principal é matematicamente coerente neste cenário, **mesmo com os achados MF-05/MF-06 em vigor** — o viés do `find_index_of_max_sum` e dos pesos da regressão NÃO manifesta erro mensurável acima de 0.5 frames neste cenário de pilha uniforme e pico limpo.

#### test_recovers_bump_depth — PASSED

- Erro mediano `0.171 frames` para um bump gaussiano (profundidade variável, curvatura não-nula no centro).
- **Sub-frame atingido** também no caso não-linear: a interpolação parabólica acompanha a variação espacial de profundidade do bump com erro abaixo de 0.5 frames. Reforça que MF-05/MF-06 não introduzem erro sistemático detectável neste cenário.

#### test_textureless_region_gets_zero_confidence — PASSED — mas com evidência importante para MF-03

- **Separação de confiança:** conf textureless = `0.3299`, conf texturizada = `0.8666`; razão `0.3299 / 0.8666 = 0.381 < 0.5` — o threshold do teste é satisfeito, logo a confiança **é distintamente menor** na região sem textura.
- **iSel na região sem textura: mediana = `1.90` (gt = 4.0, n/2 = 4.5):** O resultado NÃO é n/2 = 4.5 como previsto pelo achado MF-03. O mecanismo responsável pelo valor `1.90` foi apurado e a hipótese anterior ("kernel 3×3 / border leak do indicador") está PROVADAMENTE ERRADA: a janela medida `[28:36,28:36]` fica ≥4 px da borda do patch, enquanto o alcance máximo do estágio indicador é Laplaciano k=5 (2 px) + suavização 3×3 (1 px) = 3 px — fisicamente não alcança; alimentando a imagem nítida sem desfoque ao indicador, a resposta na janela é exatamente 0.0.

  **Mecanismo correto — blur de desfoque do `defocus_stack`:** com `sigma = blur_per_unit × |k - k_focus|` (até sigma=6 px nos frames extremos para `blur_per_unit=1.5`), a textura vizinha é espalhada para dentro do patch plano. Sonda direta confirma: `stack[:,30,30] ≈ [140.7, 134.3, 129.2, 128, 128, 128, 129.2, 134.3, 140.7]` — interior é 128 apenas nos frames em foco (k=3,4,5) e sobe a ~140 no desfoque pesado. A curva de foco `fi[:,30,30]` fica invertida/bimodal com picos em k=1 e k=7; o fit fuzzy produz iSel≈1.24 por pixel → mediana 1.90.

  **Implicação para MF-03/MF-04 (mais forte que a hipótese anterior):** medidas de foco **falham abertas perto de fronteiras de textura** — pixels interiores planos ganham pico espúrio puxado para os frames de desfoque que maximizam o vazamento da textura vizinha (dependente dos dados; pode ficar longe tanto do plano verdadeiro quanto de n/2). O cenário de "textura zero absoluta" do MF-03 (`focus_values[k_max] == 0`) NÃO foi ativado porque o vazamento de desfoque garante `focus_values[k_max] > 0` sempre que houver textura vizinha; a condição requer contraste estritamente zero em todos os frames.

  - **Aliasing do cenário:** o gt `4.0` foi escolhido próximo ao n/2 `4.5`; o valor observado `1.90` está *longe* de ambos, revelando que a região é governada pelo leak de desfoque, não pela condição de "foco máximo no meio do stack".
  - **A separação de confiança funciona** pela path normal do polyfit: a curva de vazamento de desfoque é baixa, larga e tem pouca curvatura — a baixa curvatura `|A|` no interior plano produz conf baixa em relação às bordas texturizadas, que o `normalize()` global amplifica. Porém, a confiança baixa aqui não é uma condição limpa de "sem sinal"; o canal de confiança não está validado como detector geral de regiões sem textura.

### Achados decididos por esta Task

| Achado | Status anterior | Status após Task 11 |
|--------|----------------|---------------------|
| MF-05 (`find_index_of_max_sum` vs argmax) | suspeita | **não manifesta erro>0.5 frames** nos cenários de pico limpo/único (suspeita mantida; o cenário multi-pico do exemplo de evidência não foi testado) |
| MF-06 (pesos da regressão enviesam vértice) | suspeita | **não manifesta erro>0.5 frames** em pilha uniforme (suspeita mantida; em curvas com pico assimétrico ou baixo SNR pode diferir) |
| MF-03 (pixels sem textura → n/2) | suspeita | **o path `return n/2, 0` NÃO foi ativado** neste cenário; mecanismo real é o blur de desfoque que espalha textura vizinha para dentro do patch (hipótese do kernel 3×3 refutada por sonda direta). iSel mediano `1.90` é artefato de vazamento de desfoque, não o n/2 previsto. Status: suspeita com evidência de difícil ativação em condições realistas; o bug existe no código (`argmax_fuzzy.py:129`) mas requer contraste estritamente zero. Implicação extra: medidas de foco falham abertas perto de fronteiras de textura (pico espúrio dependente dos dados) |
| MF-07 (confiança não comparável) | suspeita | **separação observada** (0.33 vs 0.87, razão 0.38 < 0.5) funciona dentro de imagem; a confiança baixa decorre da curva de vazamento de desfoque (baixa, larga, pouca curvatura), não de condição limpa de "sem sinal" — canal não validado como detector geral de regiões sem textura; não-comparabilidade *entre* imagens distintas permanece suspeita |
| `argmax_fuzzy` + `focus_indicator` corretos em dados sintéticos limpos | — | **confirmado por execução** (median 0.18/0.17 frames sub-pixel em rampa e bump) |

---

## Bloqueios

### Build do C (`make`) falha — NÃO bloqueante para Task 9 (binário pré-compilado versionado)

`cd csrc/integrate_recursive && make` falha na recompilação do fonte. **Porém o binário `gus_integrate_recursive` está versionado (rastreado pelo git) e é funcional**, então os testes da Task 9 puderam rodar contra ele. Registro o erro de build aqui para visibilidade (NÃO consertado — sem mudanças em fonte de produção):

```
gus_integrate_recursive.c:9:1: error: "/*" within comment [-Werror=comment]
    9 | /*./gus_integrate_recursive -initial zero 0 -outPrefix teste_blabla -normals .../normal_map_with_residuals.fni */
gus_integrate_recursive.c:312:10: fatal error: bool.h: No such file or directory
  312 | #include <bool.h>
      |          ^~~~~~~~
cc1: all warnings being treated as errors
make: *** [Makefile:126: gus_integrate_recursive.o] Error 1
```

- 1º erro: `-Werror=comment` — um `/*` aninhado dentro de um comentário de bloco (`gus_integrate_recursive.c:9`).
- 2º erro: `#include <bool.h>` (`:312`) não encontrado no caminho de include de sistema (o `bool.h` do projeto vive em `include/`, alcançável só via `"bool.h"` ou se o `-Iinclude` cobrir — mas o build para no `-Werror` antes).
- **Impacto:** nenhum para Task 9 (o binário existente roda). Recompilar do zero exigiria editar o fonte C de produção, o que está fora do escopo desta task (sem mudanças de produção).
