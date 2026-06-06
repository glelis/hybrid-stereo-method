# Fase 1.1 — Achados Multifocus

Data: 2026-06-04

Auditoria de corretude (matemática/teórica) e de implementação (bugs/precisão/robustez)
do estágio de *multifocus stereo* (shape-from-focus). Teoria de referência: Nayar &
Nakagawa — a profundidade por pixel vem do índice do *focal stack* que maximiza uma
medida de nitidez **local**; o sub-pixel vem de um ajuste local em torno do pico.

Cada achado cita `arquivo:linha` reais (verificados no estado atual do branch
`nova_iteracao`). Na Fase 1 os achados são "suspeita", salvo quando o código por si só
prova o defeito ("confirmado (por inspeção)").

---

## MF-01: Filtro por substring agrupa planos focais errados na média por `zf`

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:101-103`
- **Tipo:** implementação
- **Severidade:** crítico
- **Status:** confirmado (por inspeção)

**Descrição:** A média por plano focal usa
`[file for file in input_files_path if f"{zf_dir}" in file and "sVal.png" in file]`.
O teste `f"{zf_dir}" in file` é substring, não componente de caminho exato. Com ≥10
planos (todos os configs reais usam 12: `zf1..zf12`), `zf_dir == "zf1"` casa também com
`.../zf10/...`, `.../zf11/...` e `.../zf12/...`. Logo a "média do zf1" mistura imagens de
quatro planos focais distintos, corrompendo a curva de foco que alimenta toda a seleção
de profundidade. O commit 505d262 corrigiu um bug estruturalmente idêntico de substring
em outro ponto do mesmo arquivo — a seleção de mosaicos por luz (`sMos.png`, hoje
`hybrid/main.py:177-182`) —, mas a mesma classe de defeito (substring em vez de componente
de caminho exato) permanece aqui para `zf*`.

**Evidência:** reprodução direta —
`[p for p in paths if "zf1" in p and "sVal.png" in p]` retornou
`['.../zf1/...', '.../zf10/...', '.../zf11/...', '.../zf12/...']`. O `iSel` (mapa de
seleção) gerado nessa média é depois reaproveitado para *todas* as luzes
(`hybrid/main.py:154`), propagando o erro ao mosaico fotométrico inteiro.

**Sugestão de correção:** filtrar por componente de caminho exato, como já feito para
`L*` (ex.: `os.path.basename(os.path.dirname(file)) == zf_dir`).

**Correção aplicada:** e9eabb2 (2026-06-05) — helper `select_files_by_parent_dir` com `Path(f).parent.name == dir_name` substitui o substring match em `hybrid/main.py`; 13 testes novos em `tests/test_hybrid_path_selection.py`.

---

## MF-02: Ordenação lexicográfica dos planos focais desalinha `average_images_paths` de `z_foc`

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:90-96`, `101`
- **Tipo:** implementação
- **Severidade:** crítico
- **Status:** confirmado (por inspeção)

**Descrição:** `zf_directories = sorted({...})` ordena lexicograficamente. Com 12 planos a
ordem vira `zf1, zf10, zf11, zf12, zf2, zf3, ...`, enquanto `z_foc` no YAML está em ordem
natural crescente (`[15, 25, ..., 125]`, `configs/hb_experiment.yaml:39`). O laço
constrói `average_images_paths` na ordem lexicográfica, mas `mosaic`/`compute_argmax_fuzzy`
indexam `z_foc[k]` pela posição `k` no stack. Resultado: o índice de foco `k` é mapeado ao
valor de profundidade errado (o frame na posição 1 do stack é o zf10, mas recebe
`z_foc[1] = 25`). A relação índice→profundidade fica permutada e não-monotônica. O mesmo
`sorted(...)` lexicográfico afeta `filtered_files` (linha 101); embora `sVal` por plano
seja agregado por média, a ordenação dos *diretórios* é o vetor que importa.

**Evidência:** `sorted({'zf1','zf2','zf10','zf11','zf12'})` =
`['zf1','zf10','zf11','zf12','zf2']`. `hybrid/main.py:131` lê `z_foc` em ordem natural do
YAML e o passa a `mosaic`, que faz `zMos[i,j] = zFoc[K]` por posição
(`mosaic.py:57,72`). `hybrid/main.py` não possui verificação de comprimento de `z_foc` — a
validação `len(zFoc) != image_stack.shape[0]` está em `multifocus/main.py:132-135`. Não
há reordenação intermediária que reconcilie as duas ordens.

**Sugestão de correção:** ordenar os diretórios `zf` por chave numérica (ex.: `natsorted`
ou `key=lambda s: int(s[2:])`), garantindo correspondência posicional com `z_foc`.

**Correção aplicada:** e9eabb2 (2026-06-05) — helper `collect_dirs_with_prefix` com `natsorted` substitui `sorted({...})` para `zf_directories` e `light_directories` em `hybrid/main.py`; garantia posicional com `z_foc` e `lights.npy`.

---

## MF-03: Pixels sem textura recebem profundidade do meio do stack (`z_foc[n/2]`)

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:128-129`; consumo em `mosaic.py:53-74`
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** corrigido (`80069c4`) — pico nulo retorna NaN + conf 0; mascarado pelo mosaic (MF-04)

**Descrição:** Quando o pico de foco é nulo (`focus_values[k_max] == 0`, região sem
textura/contraste), `compute_argmax_fuzzy_1d` retorna `(n/2, 0)`: índice do meio com
confiança 0. Teoricamente, regiões sem textura não têm profundidade observável por
shape-from-focus e deveriam ser marcadas como inválidas (e descartadas/interpoladas a
jusante). Em vez disso recebem um índice plausível (`n/2`) com confiança 0 — mas o
`mosaic` usa `iSel` **sem consultar a confiança** (ver MF-04), então esses pixels entram
no `zMos` como se válidos, com profundidade `z_foc[n/2]` (= 75 nos configs, pois n=12 e
z_foc[6]=75). Isso introduz
um plano falso "do meio" em todas as áreas lisas. Além disso, `n/2` é float (ex.: 6.0) e
será interpolado normalmente, parecendo um valor sub-pixel legítimo.

**Evidência:** `argmax_fuzzy.py:129` `return n / 2, 0`. `mosaic.py:53` lê
`k_fuzzy = iSel[i, j]` e nunca lê `wSel`. A confiança 0 só é usada como peso opcional na
integração (canal extra em `zMos_with_confidence.fni`, `main.py:158`), mas o `zMos.fni`
puro e o `sMos` não a respeitam.

**Evidência executável (Task 11):** `test_textureless_region_gets_zero_confidence` criou
um quadrado plano `sharp[24:40,24:40] = 128` dentro de uma imagem texturizada. O path
`return n/2, 0` **NÃO foi ativado** — e o mecanismo real NÃO é o suavizamento do
indicador (kernel 3×3 + Laplaciano k=5 alcança no máximo 3 px desde a borda; a janela
medida `[28:36,28:36]` fica ≥4 px da borda, fisicamente fora do alcance; alimentando a
imagem nítida sem desfoque ao indicador a resposta na janela é exatamente 0.0).

O mecanismo real é o **blur de desfoque** do `defocus_stack`: com `sigma = blur_per_unit ×
|k - k_focus|` (até sigma=6 px nos frames extremos), a textura vizinha é espalhada para
dentro do patch plano. Sonda confirmada: `stack[:,30,30] ≈ [140.7, 134.3, 129.2, 128,
128, 128, 129.2, 134.3, 140.7]` — o interior é 128 apenas nos frames em foco (k=3,4,5)
e sobe a ~140 no desfoque pesado. A curva de foco `fi[:,30,30]` fica invertida/bimodal
com picos em k=1 e k=7; o fit fuzzy produz iSel≈1.24 por pixel → mediana observada 1.90
(gt = 4.0; n/2 = 4.5).

**Implicação (mais forte que a hipótese anterior):** medidas de foco **falham abertas
perto de fronteiras de textura** — pixels interiores planos ganham pico espúrio puxado
para os frames de desfoque que maximizam o vazamento da textura vizinha (dependente dos
dados; pode ficar longe tanto do plano verdadeiro quanto de n/2). Esta é a instanciação
realista do defeito MF-03/MF-04: a confiança baixa (0.33 vs 0.87) é real e decorre da
mesma curva de vazamento (baixa, larga, pouca curvatura) — não de ausência de sinal.

A condição `focus_values[k_max] == 0` requer contraste rigorosamente zero em todos os
frames — difícil em dados reais (o vazamento de desfoque garante que `focus_values[k_max]
> 0` sempre que houver textura vizinha). O bug existe no código e pode manifestar-se em
regiões de foco fisicamente nulo (espelhos, saturação completa, área fora do campo).

**Sugestão de correção:** propagar invalidez (NaN ou sentinela) em vez de `n/2`, ou fazer
o `mosaic`/integração mascararem pixels com `wSel == 0`.

**Correção aplicada:** `80069c4` (2026-06-06) — pico nulo retorna `(np.nan, 0)` em vez de `(n/2, 0)`; mascarado pelo mosaic (MF-04). Teste: `test_zero_peak_returns_nan_not_middle_frame` em `tests/test_multifocus_synthetic.py`.

---

## MF-04: `mosaic` ignora a confiança `wSel` ao compor `sMos`/`zMos`

- **Localização:** `src/hybrid_stereo_method/multifocus/mosaic.py:51-74`
- **Tipo:** implementação
- **Severidade:** alto
- **Status:** corrigido (`bd7a76e`)

**Descrição:** `mosaic(iSel, image_stack, zFoc, ...)` recebe apenas o mapa de índices
`iSel`; a confiança `wSel` não é passada nem consultada. Todo pixel — inclusive os de
confiança 0 (MF-03), ajustes parabólicos rejeitados (`k_fuzzy = 0` quando `A>0`, ver
MF-08) ou fits degenerados — é tratado como observação válida e escrito em `zMos` e
`sMos`. Pixels com `k_fuzzy = 0` (parábola convexa rejeitada) viram `z_foc[0]`, criando um
"plano de fundo" artificial onde o ajuste falhou. O comentário em `main.py:154` diz não
normalizar o `zMos` para preservar Z físico, mas não há filtragem por confiança em nenhum
ponto do `zMos`/`sMos` que vão para a integração e o fotométrico.

**Evidência:** assinatura `def mosaic(iSel, image_stack, zFoc, interpolation_type)` —
sem `wSel`. `argmax_fuzzy.py:175` zera `k_fuzzy` (não a confiança apenas) em fits
convexos; `mosaic.py:64-65` mapeia `i0<0`→frame 0, e `k_fuzzy==0`→`z_foc[0]`.

**Sugestão de correção:** passar `wSel` ao `mosaic` e mascarar/interpolar pixels abaixo de
um limiar de confiança, ou ao menos propagar o canal de confiança para o `sMos` usado pelo
fotométrico.

**Correção aplicada:** `bd7a76e` (2026-06-06) — `mosaic()` aceita novos parâmetros `wSel` e
`min_confidence`; pixels com `wSel[i,j] <= min_confidence` ou `iSel` não-finito recebem
`zMos=NaN` (inválido explícito) e `sMos=frame mais próximo` (PS precisa de valor em todo
pixel). `multifocus/main.py`: viz NaN-safe com `nan_to_num(0)` antes de `save_image` e FNIs
normalizados; `wSel` exportado raw (R² já em [0,1]); `zMos_with_confidence` usa
`wSel_export=where(isfinite(zMos), wSel, 0)` — peso 0 nos pixels inválidos exclui o hint
no integrador C. `hybrid/main.py` laço de luzes: `mosaic(..., wSel=wSel_avg)`.
Teste: `test_mosaic_masks_zero_confidence_pixels` (em `tests/test_multifocus_synthetic.py`).
E2E baseline (após correção): RMSE=0.0690, a=0.9982, b=3.1317, pearson r=0.9975.

---

## MF-05: `find_index_of_max_sum` pode escolher janela que não contém o pico verdadeiro

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:81-101`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** corrigido (`5335ab8`)

**Descrição:** O índice inicial do ajuste é escolhido maximizando a soma de 3 frames
consecutivos, não o argmax verdadeiro. A soma-de-3 favorece "platôs" largos sobre picos
estreitos e altos. Em shape-from-focus o foco correto corresponde ao **máximo local** da
medida de nitidez; substituí-lo por um filtro passa-baixa de janela 3 desloca a seleção
para regiões de foco "morno mas largo". Em curvas com pico secundário largo (texturas com
duas frequências, ou ruído), o ponto de partida do ajuste parabólico — e portanto o
sub-pixel resultante — pode aterrissar no lobo errado.

**Evidência:** reprodução com `fv = [0,0,0,9,0,5,6,5,0,0]`: `argmax = 3` (pico estreito
alto), mas `find_index_of_max_sum = 6` (centro do bump largo). O ajuste parabólico será
então centrado em torno de 6, ignorando o foco real em 3.

**Evidência executável (Task 11):** `test_recovers_tilted_plane_depth` e
`test_recovers_bump_depth` produziram erros medianos de `0.181` e `0.171 frames`
respectivamente — bem abaixo do limiar de 0.5. Nesses cenários a curva de foco sintética
é unimodal e suave, de modo que o `find_index_of_max_sum` e o argmax verdadeiro escolhem
o mesmo lóbulo. O viés do MF-05 **não se manifesta em pilhas com pico único limpo**; para
decidir definitivamente é necessário um cenário de dois picos (pico estreito alto vs. platô
largo), que permanece como trabalho futuro.

**Sugestão de correção:** usar o argmax verdadeiro como centro do ajuste (com tratamento
de empates/picos múltiplos), ou justificar/documentar a suavização e limitá-la a casos
ruidosos. Decidir por teste sintético (curva de foco com pico estreito + bump largo).

**Correção aplicada:** `5335ab8` (2026-06-06) — `find_peak_index` (argmax verdadeiro com desempate pela maior soma dos vizinhos via `np.pad(fv, 1, mode="edge")`) substitui `find_index_of_max_sum`. Repro pinado: `fv=[0,0,0,9,0,5,6,5,0,0]` → max-sum escolhia índice 6, argmax verdadeiro é 3. Teste: `tests/test_argmax_fit.py::test_peak_index_is_true_argmax`. Status: corrigido.

---

## MF-06: Pesos da regressão = próprios valores de foco enviesam o vértice da parábola

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:104-118`, `158,164`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** corrigido (`a9a6549`)

**Descrição:** `calculate_weights` usa os próprios valores de foco (normalizados pela soma,
+1e-6) como pesos `w` da regressão parabólica ponderada. Ponderar a regressão pelo valor
da variável dependente quebra a premissa de mínimos quadrados (pesos deveriam refletir a
*incerteza*/variância das amostras, não a magnitude do sinal). Dar mais peso aos frames
de maior foco "puxa" o ajuste para o topo observado e enviesa o vértice `-B/2A` em direção
ao frame de maior valor bruto — justamente o que o sub-pixel deveria refinar de forma
imparcial. Não há justificativa teórica para esse esquema de pesos; é uma escolha *ad hoc*
que distorce a estimativa de pico.

**Evidência:** `w_list = calculate_weights(focus_values[k0:k1+1])` (linha 158) entra em
`np.polyfit(..., w=w_list)` (linha 164). `calculate_weights` retorna
`value/total_focus + 1e-6` (linha 118) — peso ∝ valor de foco.

**Evidência executável (Task 11):** em `test_recovers_tilted_plane_depth` e
`test_recovers_bump_depth`, os erros medianos foram `0.181` e `0.171 frames`. Com uma
pilha sintética uniforme (mesmo `blur_per_unit` por pixel, pico quase simétrico), os pesos
proporcionais ao valor de foco não introduzem viés acima de 0.5 frames — a curva é
aproximadamente simétrica em torno do frame de maior foco e a ponderação não desloca o
vértice significativamente. Para curvas assimétricas (gradiente de textura variável,
oclusão parcial, ruído não-uniforme) o viés pode ser maior: suspeita mantida.

**Sugestão de correção:** usar regressão não ponderada (pesos uniformes) ou pesos
baseados em incerteza real; comparar via teste sintético com pico parabólico conhecido
(o ajuste ponderado deve dar erro de vértice maior que o não ponderado).

**Correção aplicada:** `a9a6549` (2026-06-06) — `calculate_weights` e todo uso de `w_list` removidos; `np.polyfit` chamado sem pesos; fallback simplificado para um único `try/except LinAlgError`. O bloco R² (MF-07) atualizado para estatísticas não-ponderadas (`y_bar = mean(y)`, `ss_res/ss_tot` sem w). CSV de debug: coluna `w_list` removida do header e da linha. Vértice pré-correção k=4.284 (erro 0.016, gaussiana em 4.3), pós k=4.215 (erro 0.085) — ambos < 0.1; erro de recuperação de profundidade: plano 0.181→0.183, bump 0.171→0.173 frames (ambos < 0.5). Teste: `tests/test_argmax_fit.py::test_parabola_vertex_unbiased_by_value_weights`. Status: corrigido.

**Nota:** o caso de teste (gaussiana simétrica) não discrimina pré/pós correção — ambos passam com erro < 0,1 (0,016 pré vs 0,085 pós); a correção é teórica (premissas de mínimos quadrados) e o viés do método antigo manifesta-se em curvas assimétricas, não cobertas por teste sintético.

---

## MF-07: Confiança `conf = |A| / fnoc` não é comparável entre pixels

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:183-187`; normalização global em `applicator.py:83-93`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita (separação dentro de uma imagem funciona; comparabilidade entre imagens distintas permanece em aberto)

**Descrição:** A confiança é a curvatura do ajuste dividida pelo valor de foco no pico
(`|A|/fnoc`). A escala de `A` e de `fnoc` depende da normalização do stack (clip no
percentil 1 global + divisão pelo máximo global, `applicator.py:83-93`), que é uma
transformação *afim global*, não por-pixel. Como a amplitude da curva de foco varia
fortemente entre pixels texturizados e lisos, `|A|/fnoc` mistura unidades de curvatura e
de amplitude de modo que o número resultante não tem significado uniforme entre pixels.
Pior, esse `wSel` é depois passado por `normalize()` (min-max global, `argmax_fuzzy.py:76`
e `main.py:147,158`), de modo que a "confiança" final é relativa ao maior `|A|/fnoc` da
imagem inteira — sensível a outliers de borda.

**Evidência:** `conf = abs(A) / fnoc` (linha 187), com `A,B,C` de um polyfit sobre valores
normalizados globalmente; `wSel = normalize(wSel)` (linha 76). A integração usa esse canal
como peso (`zMos_with_confidence`), então o viés se propaga ao integrador.

**Evidência executável (Task 11):** `test_textureless_region_gets_zero_confidence`
confirmou que **dentro de uma mesma imagem** a confiança separa corretamente regiões
texturizadas (0.87) de regiões sem textura (0.33), com razão 0.38 — bem abaixo do
threshold 0.5. A `normalize()` global funciona como ranking relativo dentro da imagem.
Importante: a confiança baixa no patch decorre da mesma curva de vazamento de desfoque
(baixa, larga, pouca curvatura no interior plano), não de uma condição limpa de "sem
sinal" — o canal de confiança não está validado como detector geral de regiões sem textura
(apenas como ranking relativo de curvatura dentro de uma imagem com esta configuração
específica). O achado MF-07 (não-comparabilidade *entre* imagens distintas, dependência de
outliers de borda) permanece suspeita para uso multi-imagem — não exercitado por este
teste.

**Sugestão de correção:** definir confiança em escala invariante (ex.: razão pico/segundo-
pico, ou R² do ajuste local), independente da normalização global de amplitude.

**Correção aplicada:** `86299a6` (2026-06-05) — confiança = R² (ponderado) do ajuste local
(escala-invariante, [0,1]); `normalize()` global do wSel removido. O R² usa os mesmos pesos
do `polyfit` ponderado, medindo o quão bem a parábola explica os pontos priorizados pelo
ajuste (o pico), com guarda `ss_tot < polyfit_epsilon → conf 0` para janela plana. Toda a
lógica de rejeição existente (convexa/quase-plana, `fnoc < 0`, fallbacks degenerados,
`focus_values[k_max] == 0`) foi preservada. Consumidores de wSel: `main.py:154-155`
(FNIs iSel/wSel via `normalize()` próprio — mantidos, idempotentes para [0,1] com min 0) e
`main.py:166` (`zMos_with_confidence` via `normalize(wSel)` — mantido, quase no-op). Nenhum
consumidor quebra matematicamente.

**Mudança de semântica deliberada (registrada):** o teste
`test_textureless_region_gets_zero_confidence` foi substituído por
`test_confidence_is_scale_invariant_goodness_of_fit`. Com a métrica antiga `|A|/fnoc` a
região sem textura recebia conf distintamente menor (sem=0.33, com=0.87, separação 0.38),
pois media força de pico. O R² é qualidade de ajuste: uma curva suave de vazamento de
desfoque (região lisa) ajusta uma parábola tão bem quanto um pico real, então R² **não**
separa textura de ausência de textura (valores novos: sem=0.79, com=0.70). Isto confirma a
ressalva já registrada acima — o canal de confiança não é (nem era) um detector validado de
regiões sem textura. A detecção de "sem sinal" permanece no ramo `focus_values[k_max] == 0`
→ conf 0.

---

## MF-08: Normalização global do stack altera comparabilidade entre frames e tem caso degenerado

- **Localização:** `src/hybrid_stereo_method/multifocus/indicators/applicator.py:80-93`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** corrigido (`903472d`, 2026-06-06)

**Descrição:** Após calcular o indicador de foco, o stack inteiro é clipado no percentil 1
*global* e dividido pelo máximo *global*. (1) O clip `np.clip(stack, p1, +inf)` substitui
todos os valores abaixo do p1 global pelo p1 — isso **achata o fundo** de frames com pouca
energia de foco, comprimindo justamente a parte da curva onde se mediria a transição de
foco; não preserva fielmente a forma relativa entre frames. (2) As guardas
`if min_val < 0: subtrai min_val` e `if max_val > 0: divide` têm casos degenerados: se,
após o clip, `min_val == p1 > 0` (caso normal, pois indicadores são `|.|` ≥ 0), o ramo de
subtração nunca roda, então o stack normalizado **não** começa em 0 — o piso fica em
`p1/max_val`. Não é um bug fatal (afeta `C`, não o vértice `-B/2A`), mas torna o `fnoc` e a
confiança (MF-07) dependentes desse piso. Se `max_val == 0` (stack todo zero), nada é
dividido e o stack permanece zero — caso tratado a jusante por MF-03, porém sem aviso.

**Evidência:** `p1 = np.percentile(stack, 1); stack = np.clip(stack, p1, inf)`
(linhas 83-84); `if min_val < 0` / `if max_val > 0` (linhas 90-92). Indicadores retornam
magnitude ≥ 0: `np.abs(...)` em laplacian.py:36 e fourier.py:36; `np.sqrt(cH**2 + cV**2 +
cD**2)` em wavelet.py:19 — logo o ramo de subtração praticamente nunca executa.

**Sugestão de correção:** decidir explicitamente o piso (subtrair `min_val`
incondicionalmente após clip, ou não clipar o fundo) e tratar `max_val == 0` com aviso/
máscara. Validar por teste sintético que o vértice do ajuste é insensível à escolha.

**Correção aplicada:** `903472d` (2026-06-06) — substituído o par de guardas
(`if min_val < 0` / `if max_val > 0`) por deslocamento incondicional
`focus_indicator_stack -= np.min(focus_indicator_stack)` pós-clip, seguido de
`if max_val > 0: stack /= max_val` e `else: logging.warning("…no focus signal…")`.
O deslocamento afim e uniforme entre frames não altera o argmax por pixel nem o vértice
da parábola sub-pixel (MF-07 R² também é invariante a deslocamento de escala).
A guarda `if min_val < 0` nunca disparava porque todos os indicadores retornam
`|.|>=0`. Teste: `tests/test_multifocus_synthetic.py::test_focus_indicator_normalization_floor_and_degenerate_stack`.

---

## MF-09: Indicador de Fourier é global (FFT da imagem inteira), não local por pixel

- **Localização:** `src/hybrid_stereo_method/multifocus/indicators/fourier.py:5-38`
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** suspeita

**Descrição:** O indicador "fourier" aplica FFT 2D na **imagem inteira**, multiplica por
uma máscara passa-alta global e faz IFFT, devolvendo o módulo. Embora o resultado seja um
mapa 2D (um valor por pixel), cada pixel da reconstrução passa-alta depende de *todas* as
frequências da imagem inteira (a IFFT é uma convolução com kernel de suporte global). Em
shape-from-focus a medida de nitidez deve ser **local** (energia de alta frequência numa
janela em torno do pixel) para que a seleção por pixel seja válida; um filtro passa-alta
de suporte global espalha a resposta de bordas fortes por toda a imagem (ringing), de modo
que pixels lisos próximos a uma borda nítida herdam "foco" que não é deles. Isso viola a
premissa de localidade da seleção por pixel.

**Evidência:** `np.fft.fft2(image)` sobre a imagem completa (linha 20), máscara
construída sobre `height,width` inteiros (`create_gaussian_elliptical_mask`, linha 27),
`np.fft.ifft2` global (linha 34). Não há janelamento (STFT) nem filtragem em patches.

**Sugestão de correção:** substituir por filtragem passa-alta *local* (convolução com
kernel de suporte pequeno, ou energia de alta frequência em janela deslizante / DCT por
blocos), preservando localidade.

**Reavaliação (2026-06-05):** o mecanismo "ringing global" está parcialmente refutado: a
máscara é *gaussiana* na frequência, cujo equivalente espacial é `δ − gaussiana` com
`σ = 1/(2π·radius)` ≈ 1,6 px (radius=0.1) — suporte efetivo compacto, sem sidelobes.
Verificado numericamente: no interior, a implementação FFT é idêntica (erro rel. ~1e-6) a
um unsharp mask local, e a resposta ao impulso cai a ~1e-8 do pico em 10 px. Porém há
não-localidade **real**: o wraparound circular da FFT — uma banda brilhante na coluna 0
produz resposta espúria de 0.37 (vs. 5e-9 no interior) na borda *oposta* da imagem.

**Correção aplicada:** `510e042` (2026-06-05) — `calculate_fourier_focus_indicator`
reescrito como filtro passa-alta local explícito:
`|img/255 − GaussianBlur(img/255, σ=1/(2π·radius), BORDER_REFLECT)|`. Equivalente no
interior à formulação FFT legada (pinado por teste com erro rel. < 1e-3 para radius 0.05 e
0.1), elimina o wraparound nas bordas. 5 testes novos em `tests/test_fourier_locality.py`
(wraparound, equivalência ao legado, resposta ao impulso compacta, discriminação
nítido/borrado, shape/dtype/não-negatividade). Helpers de máscara
(`create_gaussian_elliptical_mask` etc.) mantidos — o teste de equivalência usa a
máscara legada como referência.

---

## MF-10: `non_linear_res` não está integrado e ignora a máscara de pesos no ajuste

- **Localização:** `src/hybrid_stereo_method/multifocus/indicators/non_linear_res.py:5-55`; despacho em `applicator.py:36-43`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** corrigido (2026-06-06)

**Descrição:** (1) `applicator.focus_indicator` só despacha para `fourier`, `laplacian` e
`wavelet`; `non_linear_res.calcular_indicador_foco` não é alcançável pelo pipeline (se
`focus_indicator_type` não casar nenhum ramo, `focus_indicator` fica indefinida e o código
quebra mais adiante — não há `else`/erro). (2) Conceitualmente, o ajuste linear local
`G = A + Bx + Cy` é resolvido por `lstsq` **sem** a máscara de pesos `W`, embora `W` seja
usada depois no resíduo `F = Σ (Q² · W)`. Para coerência teórica (resíduo ponderado de um
ajuste de plano), o ajuste deveria ser ponderado pela mesma `W`. Como o módulo está fora
do fluxo, o impacto atual é nulo, mas o achado fica registrado caso seja reativado.

**Evidência:** `applicator.py:36-43` tem ramos `fourier`/`laplacian`/`wavelet` e nenhum
`non_linear`/`else`. `non_linear_res.py:41` chama `lstsq(X, intensidades)` sem pesos;
`mascara_pesos` (linha 7) só aparece no resíduo (linha 50).

**Sugestão de correção:** se reativado, resolver o ajuste com a mesma ponderação `W`
(WLS) e adicionar o ramo de despacho com `else: raise ValueError`.

**Correção aplicada:** `1aadea3` (2026-06-06) — ramo `non_linear_res` adicionado ao
despacho em `applicator.py`; `else: raise ValueError(f"Unknown focus_indicator_type: {t!r}")`
elimina o UnboundLocalError silencioso. Em `non_linear_res.py`: `lstsq(X, intensidades)`
substituído por WLS `lstsq(X * sw[:, None], intensidades * sw)` com `sw = sqrt(W.flatten())`
(coerência entre ajuste e resíduo ponderado); `X`, `sw`, `X_sw` hoistados fora do loop de
pixels (constantes); residual `F < eps` clampado a 0.0 (evita amplificação de ruído
numérico ~1e-27 na normalização). 2 testes: `test_focus_indicator_unknown_type_raises`
e `test_non_linear_res_dispatch_and_plane_residual` em `tests/test_multifocus_synthetic.py`.

---

## MF-11: `k_fuzzy = max(0, min(n, k_fuzzy))` deveria usar `n-1` como teto

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:181`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** corrigido (`22f4470`)

**Descrição:** O índice válido máximo do stack é `n-1`, mas o clamp permite `k_fuzzy == n`.
Um vértice fora do intervalo amostrado já indica um ajuste mal condicionado (pico além do
último frame), e devolver `n` é semanticamente um índice inexistente. O `mosaic` se
protege: `i0 = floor(n)` cai no ramo `i0+1 >= n_frames` e usa o último frame/`z_foc[n-1]`
(`mosaic.py:67-69`), e o ramo `crop` faz `min(int(n), n-1)`; logo não há out-of-bounds.
Mas o clamp incorreto mascara a extrapolação em vez de sinalizá-la, e o valor `n` chega a
ser gravado no CSV de debug como se fosse índice legítimo.

**Evidência:** `k_fuzzy = max(0, min(n, k_fuzzy))` (linha 181) vs. clamp correto do mosaic
`min(max(int(k_fuzzy), 0), n_frames - 1)` (mosaic.py:56) e guarda `i0 + 1 >= n_frames`
(mosaic.py:67).

**Sugestão de correção:** usar `min(n - 1, ...)` e, idealmente, baixar a confiança quando
o vértice cai fora do intervalo `[0, n-1]`.

**Correção aplicada:** `22f4470` (2026-06-06) — ramo `else` (côncavo) reestruturado: se `k_raw < 0 or k_raw > n - 1`, clamp a `[0.0, float(n-1)]` e conf=0 (vértice extrapolado = pico não bracketado); o bloco R² fica apenas no ramo de vértice interno. Pré-correção: `fv=[0,0.05,0.1,0.3,0.7,1.0]` → k_raw=7.5, old clamp `min(n=6, 7.5)=6` > n-1=5, conf=1.0; pós k=5.0, conf=0. Teste: `tests/test_argmax_fit.py::test_vertex_outside_stack_clamps_to_n_minus_1_with_zero_conf`.

**Post-Task-11 E2E baseline (após MF-05/MF-06/MF-11):** RMSE 0.0697, a 0.9954, r 0.9975 (deslocamento desprezível vs baseline anterior 0.0690 após MF-12/PS-07).

---

## MF-12: Quantização uint8 (PNG) das médias por `zf` antes da medida de foco

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:107-112` (save) + `:104,149` (re-leitura); `infrastructure/io/image_io.py:138,140`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita

**Descrição:** Cada média por plano focal é gravada como PNG e relida antes da medida de
foco: `calculate_avarage_of_images` produz float, mas é convertido a `uint8`
(`infrastructure/utils.py:106-107`) e `save_image` grava `uint8`
(`image_io.py:138/140`). A média de muitas imagens reduz ruído e ganha bits efetivos de
precisão, que são descartados ao quantizar em 8 bits. A medida de foco (Laplaciano/FFT)
opera sobre derivadas de alta frequência, sensíveis a degraus de quantização: a curva de
foco em regiões de baixo contraste pode ficar "escadeada", piorando o ajuste sub-pixel.
O ideal seria manter as médias em float (ou ≥16 bits) até a medida de foco.

**Evidência:** `calculate_avarage_of_images` retorna `mean_image.astype(np.uint8)` para
entrada uint8 (utils.py:107); o pipeline grava PNG (`save_image`, main.py:111) e relê
(`read_images`, main.py:104) antes de `focus_indicator`. O FNI/float só é usado bem
depois.

**Sugestão de correção:** passar as médias em float diretamente à medida de foco (sem
round-trip PNG uint8), ou salvar em formato sem perda de precisão. Quantificar o impacto
por teste sintético comparando curva de foco float vs. uint8.

**Correção aplicada:** (2026-06-05) — médias float em memória (`filtered_images`) + `.npy` de consulta; PNG só visualização. `calculate_avarage_of_images` agora retorna `float32` sem cast de volta a `uint8`/`uint16`; `hybrid/main.py` popula `parameters["filtered_images"]` com a lista de arrays float e salva `.npy` junto ao PNG; `multifocus/main.py` usa `filtered_images` quando disponível, saltando `read_images`. Baseline RMSE afim: 0.0742 → 0.0691 (melhora de 6,9%). Testes: 3 unitários + 1 integração em `tests/test_average_float_path.py`; suite completa 46 passed, 4 xfailed.

---

## MF-13: Ajuste parabólico em espaço de índice + conversão índice→z só é exato se `z_foc` for uniforme

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:156,180`; conversão em `src/hybrid_stereo_method/multifocus/mosaic.py:72`
- **Tipo:** conceitual
- **Severidade:** baixo
- **Status:** suspeita

**Descrição:** `compute_argmax_fuzzy_1d` ajusta a parábola sobre as posições inteiras dos
frames (`x_list = list(range(k0, k1+1))`, linha 156) e devolve o vértice no **espaço de
índice** (`k_fuzzy = -B/(2A)`, linha 180). A profundidade física só é obtida depois, em
`mosaic.py:72`, via `zMos[i,j] = interpolate(zFoc, k_fuzzy)`, ou seja, mapeando o índice
sub-pixel para z por interpolação na lista `z_foc`. A composição "ajustar-em-índice +
interpolar-em-z" só é equivalente a "ajustar-em-z" quando o mapa índice→z é **afim**, isto
é, quando `z_foc` é uniformemente espaçado (`z_k = z_0 + Δ·k`): uma reparametrização afim
de uma quadrática preserva a posição do vértice, logo o vértice em índice mapeia
exatamente para o pico em z. Se `z_foc` for **não-uniforme**, o mapa índice→z é não-linear
e o vértice da parábola em índice **não** corresponde, em geral, ao pico de nitidez em
coordenadas z (a estimativa de profundidade fica enviesada na direção do lado de maior
espaçamento). O código não impõe uniformidade: `z_foc` é uma lista livre no YAML
(`hybrid/main.py:131`, sem verificação de comprimento — a validação `len(zFoc) !=
image_stack.shape[0]` está em `multifocus/main.py:132-135`; `hybrid/main.py` não possui
essa guarda) e o próprio `mosaic`
interpola sobre `zFoc` como se espaçamento arbitrário fosse esperado
(`linear_interpolation`/`quadratic_interpolation` recebem `zFoc`, `mosaic.py:72`).

**Evidência:** ajuste em índice — `x_list = list(range(k0, k1 + 1))` (linha 156),
vértice `k_fuzzy = -B / (2 * A)` (linha 180). Conversão tardia para z —
`zMos[i, j] = interpolate(zFoc, k_fuzzy)` (mosaic.py:72). Todos os configs reais usam
espaçamento **uniforme** (`z_foc: [15, 25, 35, ..., 125]`, passo 10:
`configs/hb_experiment.yaml:39`, `configs/ms_experiment.yaml:40`,
`configs/test_ms.yaml:18`), caso em que o mapa é afim e não há viés — por isso a
severidade é **baixa** (defeito latente, só se manifesta se algum experimento adotar
`z_foc` não-uniforme).

**Sugestão de correção:** se `z_foc` puder ser não-uniforme, ajustar a parábola
diretamente em z (`x_list = z_foc[k0:k1+1]` e vértice já em z), ou validar/assertir
espaçamento uniforme de `z_foc` na entrada (e documentar a premissa). NÃO aplicar.

**Correção aplicada:** `9ef5721` (2026-06-06) — mitigação/documentação da premissa. Adicionada `check_z_foc_uniformity()` em `multifocus/main.py` (nível módulo): calcula `np.diff(z_foc)` e emite `logging.WARNING` quando os passos não são todos iguais (tolerância `rtol=1e-6`, `atol=0`). Chamada imediatamente após a guarda `len(zFoc) != image_stack.shape[0]`. O viés para `z_foc` não-uniforme permanece por design (correção completa exigiria refatorar o ajuste para espaço-z). Teste: `tests/test_multifocus_synthetic.py::test_warn_nonuniform_z_foc` (`caplog` verifica mensagem "non-uniform" para passos [10,10,15]; sem aviso para passos [10,10,10]). Status: **corrigido** (mitigação — premissa documentada e avisada).

---

## MF-14: Detecção de diretórios de luz `L*` só olha o pai imediato — vazia no layout `L<n>/zf<m>/sVal.png` limpo

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:122-128` (e o laço dependente `134-163`)
- **Tipo:** implementação
- **Severidade:** crítico
- **Status:** **confirmado (por execução)** — `tests/test_e2e_hybrid.py::test_hybrid_pipeline_end_to_end`, Task 12, `xfail(strict=True, raises=ValueError)`, XFAILED; raw em `06-test-results.md`.

**Descrição:** `light_directories` é construído a partir de
`{os.path.basename(os.path.dirname(path)) ... if ...startswith("L")}` sobre **todos** os
arquivos de `find_all_files(data_folder)`. No layout documentado `L<n>/zf<m>/sVal.png` (o
mesmo de `CLAUDE.md` e dos dados reais), o pai **imediato** de cada `sVal.png` é sempre um
diretório `zf<m>` — nunca um `L<n>`. A detecção só inspeciona esse pai imediato, então num
dataset que contenha **apenas** os stacks `sVal.png` (como o sintético desta task), o
conjunto `light_directories` fica **vazio**: o laço `134-163` não executa, nenhum
`multifocus_stereo/L*/sMos.png` é escrito e o Passo 2 (fotométrico) aborta em
`photometric/main_wps.py:123` com `ValueError: Number of images (0) does not match number
of light directions (6)`. O modo de falha é **alto e claro** (ValueError explícito com
mensagem informativa) — não corrupção silenciosa.

Verificação independente do revisor sobre os **11 datasets reais** em `data/raw/hybrid_stereo/`
executando a mesma lógica de detecção: apenas **3/11 detectam alguma luz** — exatamente os três
com `selected-pixels.png` diretamente sob `L*/` (`...glo0 (2).50`, `...glo0.50`, `lamb-gls`).
Os outros **8/11 detectam ZERO luzes** e falhariam com o mesmo ValueError — incluindo os
datasets "melon" que seguem o layout `L*/zf*/sVal.png` sem arquivo avulso sob `L*/`. Portanto
a narrativa "os dados reais funcionam por acidente" é parcialmente incorreta: a **maioria dos
dados reais também falha** com este bug; apenas os 3 datasets com detritos de filesystem sob
`L*/` são poupados — e nesses, a detecção depende de arquivos acidentais, não dos stacks.

A correção do commit 505d262 (seleção de `sMos.png` por componente exato) cobriu o *consumo*
dos mosaicos, mas a *detecção* das luzes a montante continua dependendo de detritos no
diretório, não dos stacks.

**Justificativa da severidade `crítico`:** (a) o pipeline é inutilizável no layout documentado
limpo — a maioria dos datasets reais (8/11) falha com o mesmo ValueError que o sintético; (b)
nos 3 datasets onde "funciona", a seleção de luzes é dirigida por arquivos acidentais,
criando risco real de associação luz↔mosaico errada (ver CONV-6); o modo de falha observado
é ALTO e claro (ValueError em `main_wps.py:123`), mas os itens (a)+(b) sustentam o rótulo
`crítico` independentemente da ausência de corrupção silenciosa.

**Evidência:** reprodução direta da detecção sobre `L0/zf0/sVal.png ...`:
`light_directories = []`, `zf_directories = ['zf0','zf1','zf2']` (o ramo `zf*` funciona
porque ali o pai imediato realmente começa com `zf`). Execução E2E: o estágio de média por
`zf` completa (9 `average_zf*.png` gravados), e o pipeline morre exatamente no PS por 0
imagens (~1,2 s de parede). Inspeção do dataset real confirma `L000/selected-pixels.png`
(pai imediato = `L000`) como o detrito que salva a detecção na prática.

**Sugestão de correção:** derivar `light_directories` de um componente `L*` em
**qualquer** posição do caminho relativo a `data_folder` (ex.: varrer
`Path(path).relative_to(data_path).parts` por um componente que case `^L\d+`), em vez de
apenas `os.path.dirname` (pai imediato). Idealmente derivar luzes e planos focais do mesmo
varredura estruturada para garantir o pareamento luz↔mosaico (ver CONV-6).

**Correção aplicada:** `e827a17` (2026-06-05) — novo helper `collect_light_dirs` em `hybrid/main.py` varre todos os componentes do caminho relativo a `data_path` buscando `re.fullmatch(r"L\d+", part)`, em vez de inspecionar apenas o pai imediato. Chamada no `main()` substituída de `collect_dirs_with_prefix(..., prefix="L")` para `collect_light_dirs(input_files_path, os.path.join(input_path, data_foldername))`. xfail removido de `test_hybrid_pipeline_end_to_end`; variante `_with_workaround` e `marker.txt` removidos. Baseline E2E (dataset limpo, sem workaround): RMSE=0.3420 (std gt=0.9780), a=0.2791, b=3.2083, pearson r=0.9369. 2 novos testes unitários em `tests/test_hybrid_path_selection.py`.

---

## Verificado sem achado

- **Laplaciano é local e com normalização prévia coerente** — `cv2.Laplacian` é um kernel
  local de segunda derivada e a normalização `/255` é feita por imagem antes do `abs`
  (`indicators/laplacian.py:25,32,36`); medida local válida para seleção por pixel.
- **Wavelet é local** — `pywt.wavedec2` Haar produz coeficientes de detalhe locais; a
  magnitude de alta frequência é por sub-banda (`indicators/wavelet.py:9-19`). (A redução
  de resolução por nível 2 reduz a resolução do mapa, mas não é erro de corretude.)
- **`zero_border` trata bordas zerando uma faixa** — `utils.zero_borders(img, 40)`
  aplicado *antes* do filtro (`applicator.py:32-34`); evita propagar a alta frequência
  espúria do padding implícito do `cv2.Laplacian` para a seleção (opcional via config).
- **`assert np.max(stack) >= 0`** — indicadores retornam `np.abs(...)`, então a asserção
  de não-negatividade (`applicator.py:69`) é coerente com os indicadores ativos.
- **`linear_interpolation` indexa `zFoc`/stack corretamente e clampa, não extrapola** —
  `kint` clampado a `[0, nframes-1]`, `s ∈ [0,1]`, interpolação convexa entre vizinhos
  (`math_utils.py:57-82`); o `mosaic` só a chama quando `0 <= i0` e `i0+1 < n_frames`
  (`mosaic.py:61-74`), evitando extrapolação.
- **`quadratic_interpolation` mantém `s ∈ [-0.5,0.5]` e regulariza singularidade** —
  janela centrada em `kint`, assert `-0.5 <= s <= 0.5`, fallback com regularização
  `1e-8` em matriz singular (`math_utils.py:4-54`); não extrapola além das guardas do
  `mosaic`.
- **`mosaic` clampa o índice e protege contra out-of-bounds** — ramos `i0<0`,
  `i0+1>=n_frames` e `crop` com `min(max(int,0),n-1)` (`mosaic.py:55-69`); mesmo recebendo
  `k_fuzzy == n` de MF-11 não há acesso inválido.
- **`compute_argmax_fuzzy_1d` exige nº mínimo de frames** — `find_index_of_max_sum` levanta
  erro com `< 3` frames (`argmax_fuzzy.py:91-94`) e há `assert n >= 2*r+1`
  (`argmax_fuzzy.py:141`); consistente com o commit 604b4d8.
- **`focus_indicator(laplacian)` + `compute_argmax_fuzzy` produz profundidade sub-pixel válida em dados sintéticos** — `test_recovers_tilted_plane_depth` e `test_recovers_bump_depth` (Task 11, `tests/test_multifocus_synthetic.py`) confirmam por execução que a pipeline multifocus recupera profundidade com erro mediano `0.18` e `0.17 frames` respectivamente (limiar: 0.5 frames), em pilha sintética de 9 frames com `blur_per_unit=1.5`, textura aleatória suavizada e gradientes de profundidade plano (rampa) e curvo (bump gaussiano). Evidência executável de que a interpolação parabólica sub-pixel funciona corretamente para os cenários de pico único e limpo que dominam os experimentos reais.
- **Rejeição de ajuste convexo/quase-plano** — `if A > 0 or |A| < polyfit_epsilon` zera o
  resultado em vez de calcular `-B/2A` de uma parábola sem máximo (`argmax_fuzzy.py:174`);
  matematicamente correto (parábola côncava é a única com máximo interior). (O efeito
  colateral de gravar `k_fuzzy=0` sem máscara de confiança está coberto por MF-04.)
- **Fallback de `polyfit` em LinAlgError** — tentativa ponderada → não ponderada → retorno
  do `k_max` com confiança 0 (`argmax_fuzzy.py:160-171`); robustez adequada.
- **`depth_refinement.py` (graph-cut) NÃO é usado no pipeline** — não há import em nenhum
  módulo do pacote (grep vazio); o arquivo faz `from utils import *` (caminho inexistente
  no pacote) e referencia um `base_path` hard-coded de outro projeto
  (`depth_refinement.py:6,82-84`). É script morto; portanto não há risco de labels
  discretos destruírem o sub-pixel do `iSel` no fluxo atual. (Registrar como código morto
  fora do escopo de corretude do pipeline.)
- **`image_alignment.py` NÃO é usado no pipeline** — sem imports no pacote; também faz
  `from utils import *` (`image_alignment.py:6`). Nenhum alinhamento é aplicado nem ao
  stack médio nem aos stacks por luz no fluxo real; a pergunta de "mesmo alinhamento em
  ambos" é vacuamente satisfeita (nenhum alinhamento em nenhum). Eventual desalinhamento
  físico das aquisições não é tratado — fica como observação, não achado de código.
- **`save_image(normalize=True)` nas médias por `zf`** — lead **escalado para o achado
  IO-05 em 04-io.md**. O `cv2.normalize` min-max por imagem (`image_io.py:138`) estica
  cada `average_{zf}.png` independentemente para [0,255]: planos desfocados (baixo
  contraste) são ampliados ao mesmo intervalo dos planos nítidos, destruindo a relação de
  intensidade *entre planos* antes da medida de foco. Isso é distinto de MF-12 (que trata
  da *quantização* a uint8) e de MF-08 (normalização global do stack dentro do
  `applicator`): o esticamento min-max **por-frame** pode deslocar o argmax de foco para o
  plano errado — altera a escala relativa do |Laplaciano| entre frames e pode mover o
  argmax. Classificado como **alto, suspeita**; decisão pelos testes sintéticos das Tasks
  11/12. NÃO está "sem achado" — está rastreado como IO-05.
