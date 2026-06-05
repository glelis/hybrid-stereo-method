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

---

## MF-03: Pixels sem textura recebem profundidade do meio do stack (`z_foc[n/2]`)

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:128-129`; consumo em `mosaic.py:53-74`
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** suspeita

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

**Sugestão de correção:** propagar invalidez (NaN ou sentinela) em vez de `n/2`, ou fazer
o `mosaic`/integração mascararem pixels com `wSel == 0`.

---

## MF-04: `mosaic` ignora a confiança `wSel` ao compor `sMos`/`zMos`

- **Localização:** `src/hybrid_stereo_method/multifocus/mosaic.py:51-74`
- **Tipo:** implementação
- **Severidade:** alto
- **Status:** suspeita

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

---

## MF-05: `find_index_of_max_sum` pode escolher janela que não contém o pico verdadeiro

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:81-101`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita

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

**Sugestão de correção:** usar o argmax verdadeiro como centro do ajuste (com tratamento
de empates/picos múltiplos), ou justificar/documentar a suavização e limitá-la a casos
ruidosos. Decidir por teste sintético (curva de foco com pico estreito + bump largo).

---

## MF-06: Pesos da regressão = próprios valores de foco enviesam o vértice da parábola

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:104-118`, `158,164`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita

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

**Sugestão de correção:** usar regressão não ponderada (pesos uniformes) ou pesos
baseados em incerteza real; comparar via teste sintético com pico parabólico conhecido
(o ajuste ponderado deve dar erro de vértice maior que o não ponderado).

---

## MF-07: Confiança `conf = |A| / fnoc` não é comparável entre pixels

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:183-187`; normalização global em `applicator.py:83-93`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita

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

**Sugestão de correção:** definir confiança em escala invariante (ex.: razão pico/segundo-
pico, ou R² do ajuste local), independente da normalização global de amplitude.

---

## MF-08: Normalização global do stack altera comparabilidade entre frames e tem caso degenerado

- **Localização:** `src/hybrid_stereo_method/multifocus/indicators/applicator.py:80-93`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita

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

---

## MF-10: `non_linear_res` não está integrado e ignora a máscara de pesos no ajuste

- **Localização:** `src/hybrid_stereo_method/multifocus/indicators/non_linear_res.py:5-55`; despacho em `applicator.py:36-43`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** confirmado (por inspeção)

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

---

## MF-11: `k_fuzzy = max(0, min(n, k_fuzzy))` deveria usar `n-1` como teto

- **Localização:** `src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:181`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** confirmado (por inspeção)

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
