# Fase 1.2 — Achados Fotométrico

Data: 2026-06-04

Auditoria de corretude (matemática/teórica) e de implementação (bugs/precisão/robustez)
do estágio de *photometric stereo*. Teoria de referência: modelo Lambertiano de Woodham
(1980) — `I = ρ · (L · n̂)`, com `n̂` unitário, `ρ` o albedo escalar e `L` direções de luz
unitárias; intensidades **lineares** em radiância. Solvers robustos seguem Ikehata et al.
(L1/SBL, CVPR 2012) e Wu et al. (RPCA, ACCV 2010).

Fluxo no pipeline híbrido: `hybrid/main.py` constrói mosaicos all-in-focus por luz, grava
`sMos.png` com `save_image(..., normalize=False)` (clip 0-255, uint8), coleta-os em
`sMos_path_list` (natsorted) e chama `photometric/main_wps.py:main`, que os pareia
posicionalmente com as linhas de `lights.npy` e chama
`wps.estimate_normals_argmax_lstsq_robust`. O `normal_map.npy` resultante é integrado pelo
solver C (`-normals`).

Cada achado cita `arquivo:linha` reais (verificados no branch `nova_iteracao`). Na Fase 1
os achados são "suspeita", salvo quando o código por si só prova o defeito
("confirmado (por inspeção)").

---

## PS-01: Albedo calculado como norma das intensidades preditas, não como ρ

- **Localização:** `src/hybrid_stereo_method/photometric/wps.py:198`
- **Tipo:** conceitual
- **Severidade:** baixo
- **Status:** suspeita (decide: `tests/test_photometric_synthetic.py` — albedo-vs-número-de-luzes)

**Descrição:** No modelo `I = ρ·(L·n̂)`, a solução de mínimos quadrados de `L · m = I` dá um
vetor `m` **não-normalizado** cuja **norma é o albedo** `ρ` (e cuja direção é `n̂`). O código
faz o oposto: em `wps.py:189-194` ele **normaliza** `normal` (descartando a magnitude que
era o albedo) e depois, em `wps.py:198`, calcula
`albedo[i,j] = np.linalg.norm(np.dot(selected_lights, normal))` — isto é, a norma do vetor
de intensidades **preditas** `L·n̂` com `n̂` já unitário. Essa quantidade é
`sqrt(Σ_k (L_k·n̂)²)`, que (i) **não** é o albedo `ρ` e (ii) **cresce com o número de luzes
válidas** `N` (mais termos não-negativos na soma sob a raiz) e com a geometria das luzes
selecionadas. Logo o "albedo" é uma grandeza dependente de `N` e da configuração de luz, sem
significado fotométrico. O albedo correto seria a norma de `m` **antes** da normalização
(ou `ρ = ||m||` a partir do `lstsq`).

O defeito é **latente**: o `albedo` retornado por `estimate_normals_argmax_lstsq_robust` é
desempacotado em `main_wps.py:135` mas **nunca é salvo nem consumido** em nenhum ponto do
pacote (verificado por grep — não vai para FNI, nem para a integração, nem para o canal de
confiança). A severidade sobe para alto/crítico se o albedo passar a ser reportado ou
consumido (ex.: figuras da tese, mapa de reflectância), momento em que o viés dependente de
`N` se torna um erro mensurável nos resultados.

**Evidência:** `lstsq(selected_lights, selected_values)` em `wps.py:165` devolve `normal`
não-normalizado; `wps.py:194` faz `normal /= norm` (perde `||m||`); `wps.py:198` recomputa
a partir de `n̂` unitário. Como `N` varia por pixel (rejeição de sombra PS-02 + remoção de
outliers PS-03), pixels com mais luzes válidas recebem "albedo" sistematicamente maior. O
teste sintético de albedo-vs-número-de-luzes (dados limpos, mesmo ρ, variar nº de luzes)
deve mostrar o "albedo" crescendo com o nº de luzes — confirmando o defeito.

**Sugestão de correção:** definir `albedo[i,j] = np.linalg.norm(m)` onde `m` é a solução
`lstsq` **antes** de normalizar (`ρ = ||m||`, `n̂ = m/||m||`). NÃO aplicar.

---

## PS-02: Limiar de sombra relativo ao máximo do pixel quase nunca rejeita em 8 bits

- **Localização:** `src/hybrid_stereo_method/photometric/wps.py:151-152` (default em `:135`)
- **Tipo:** implementação
- **Severidade:** alto
- **Status:** suspeita (decide: `tests/test_photometric_synthetic.py` — shadow handling)

**Descrição:** A rejeição de sombra usa `pixel_values / v_max > shadow_threshold` com
`shadow_threshold` default `1e-3` (`wps.py:135`) e `v_max` o **máximo daquele pixel**. Com
entrada quantizada a 8 bits (PS-07), o menor valor não-nulo possível é `1/255 ≈ 3.9e-3`, já
maior que `1e-3`. Além disso, `images = images + epsilon` (`wps.py:139`) soma `1e-6` a tudo,
então mesmo pixels exatamente 0 viram `1e-6`, e a razão `1e-6 / v_max` só fica abaixo de
`1e-3` se `v_max > 1e-3`. Resultado prático: salvo pixels em sombra **literalmente total**
(0 em todas as luzes, caso em que `v_max ≈ 1e-6` e a razão de todos vira ~1, não rejeitando
nada), o critério **não rejeita sombras reais** — uma intensidade de `5/255` sob uma luz e
`250/255` sob outra produz razão `0.02 > 1e-3` e o ponto sombreado entra no `lstsq`.
Sombras (regiões onde `L·n̂ ≤ 0`, ou attached/cast shadows) violam o modelo Lambertiano e
enviesam a normal; deixá-las entrar contradiz o propósito declarado do solver "robusto".
Além disso o threshold ser **relativo** ao próprio pixel (e não absoluto/à dinâmica da cena)
torna-o praticamente um no-op. Highlights/saturação **nunca** são tratados por este passo —
ficam inteiramente a cargo do loop de outliers (PS-03), que pode falhar quando a saturação
afeta a maioria das luzes.

**Evidência:** `v_max = np.max(pixel_values)` (`wps.py:151`); `valid_indices =
pixel_values / v_max > shadow_threshold` (`wps.py:152`); default `1e-3` < `1/255`. Com
`epsilon=1e-6` somado em `wps.py:139`, nenhum valor é exatamente 0. O teste de
shadow-handling (render sintético com sombras attached conhecidas) deve mostrar que pixels
sombreados não são descartados.

**Sugestão de correção:** usar limiar **absoluto** em radiância linear (ou uma fração da
dinâmica global da cena), não relativo ao máximo do pixel; e adicionar um limiar superior
para saturação/highlight. NÃO aplicar.

---

## PS-03: Remoção de outliers por 3×média(|residual|) — limiar dependente de escala e estatística não robusta

- **Localização:** `src/hybrid_stereo_method/photometric/wps.py:163-185`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita (decide: `tests/test_photometric_synthetic.py` — saturation robustness)

**Descrição:** O critério de outlier é `residuals <= outlier_threshold_multiplier *
mean(|residuals|)` com multiplicador default 3 (`wps.py:136,174`). Dois problemas: (1) a
**média do valor absoluto** dos resíduos não é uma estatística robusta — um único outlier
grande (ex.: highlight saturado) **infla a própria média**, elevando o limiar e podendo
**mascarar** o outlier que deveria ser removido (mascaramento clássico; o correto seria
mediana/MAD). (2) O limiar escala com a faixa de intensidade (0-255 vs 0-1), mas isso só
afeta a comparação relativa, que é homogênea — o problema real é o uso da média. A
**terminação** do `while True` está garantida: a cada iteração ou `mask` mantém todos
(`break` em `wps.py:175-176`) ou o conjunto **encolhe estritamente** (`selected_values =
selected_values[mask]` com `sum(mask) < len`), e o piso `< 3` força saída (`wps.py:182-185`);
o conjunto é monotonicamente decrescente e finito, logo o laço termina.

**Evidência:** `mask = residuals <= outlier_threshold_multiplier * r_avg` com
`r_avg = np.mean(residuals)` sobre `np.abs(...)` (`wps.py:168-174`). Média não robusta a
outliers fortes. O teste de saturation-robustness (1-2 luzes saturadas) deve mostrar normais
enviesadas quando a saturação infla `r_avg`.

**Sugestão de correção:** usar mediana + MAD (ou IQR) para o limiar de outlier em vez de
3×média; opcionalmente limitar o número de iterações. NÃO aplicar.

---

## PS-04: `residual_std` usado na confiança é o do ajuste ANTES da última remoção de outliers

- **Localização:** `src/hybrid_stereo_method/photometric/wps.py:163-176`, `203-206`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** verificado por inspeção de fluxo — risco de manutenção

**Descrição:** A análise de fluxo mostra que **todos** os caminhos que alcançam o bloco de
confiança (guard `>= 3`) chegam com `residuals`, `normal` e `selected_*` mutuamente
alinhados: o único ponto de saída válido do laço é o `break` em `wps.py:175-176`, que só
dispara quando `mask` mantém todos os elementos (`sum(mask) == len`), garantindo que
`residuals` foi computado exatamente sobre o conjunto atual. O código está, portanto,
**correto por construção hoje**.

O achado é de **fragilidade a refatoração futura**: há acoplamento implícito entre o ponto
de `break` e a validade de `residual_std` — qualquer alteração na estrutura do laço (ex.:
adicionar um segundo critério de convergência, reorganizar a atribuição de `residuals`)
pode silenciosamente desalinhar os vetores sem que haja nenhuma asserção ou comentário que
proteja essa invariante. Combinado com PS-01/PS-02, a confiança ainda herda a dependência de
`N` e de escala, mas isso é contabilizado nesses achados, não aqui.

**Evidência:** `residuals` reatribuído a cada volta (`wps.py:168`); `break` só ocorre quando
`mask` mantém todos (`wps.py:175`), de modo que `residuals` e `selected_*` estão alinhados
no ponto de saída normal. O alinhamento é garantido pela topologia do laço, não por uma
invariante explícita.

**Sugestão de correção:** recomputar `residuals` explicitamente a partir do `normal` final
e do conjunto final antes de derivar confiança, removendo a dependência da ordem do
`break`. NÃO aplicar.

---

## PS-05: Confiança `(N/M)·1/(1+residual_std)` não é invariante a ganho radiométrico

- **Localização:** `src/hybrid_stereo_method/photometric/wps.py:200-206`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita (decide: teste sintético confiança 0-1 vs 0-255)

**Descrição:** A confiança é `(N/M) · 1/(1 + residual_std)` (`wps.py:204-205`), onde
`residual_std` é o desvio-padrão dos resíduos `|L·n̂ − I|`. Esses resíduos estão na **mesma
unidade das intensidades** `I`. Com `n̂` **unitário** (a magnitude/albedo foi descartada,
ver PS-01), `L·n̂` é `O(1)`, mas `I` está na faixa **0-255** (PS-07). Logo `residual_std`
é tipicamente enorme em escala 0-255 e o fator `1/(1+residual_std) → 0` para quase todos os
pixels; se a entrada fosse 0-1, o mesmo pixel teria confiança muito maior. A confiança,
portanto, **não é invariante a ganho** (multiplicar todas as imagens por uma constante muda
a confiança), o que é fisicamente indesejável e torna o número não comparável entre datasets
com exposições diferentes. Pior: como `L·n̂` (com `n̂` unitário) e `I` (≈albedo·L·n̂, faixa
0-255) estão em **escalas diferentes**, o resíduo mistura erro de modelo com a discrepância
de escala albedo≠1 — ver PS-01. Consequência a jusante: o canal de confiança vai para
`normal_map_with_residuals.fni` (`main_wps.py:147-151`) e, no pipeline híbrido, a confiança
do *multifocus* (não esta) é que alimenta os hints; mesmo assim, qualquer uso desta
confiança como peso herda o viés de escala. Assim como PS-01 (albedo), o canal de confiança
fotométrica **não tem consumidor ativo** — `normal_map_with_residuals.fni` não é lido pela
integração (hints vêm do multifocus via `hybrid/main.py:231`), o que limita o impacto atual
a relatório; a severidade do viés de escala sobe se a confiança passar a ser usada como peso
na integração ou em fusão de dados.

**Evidência:** `residual_std = np.std(residuals)` (`wps.py:203`) sobre resíduos em unidade de
`I`; `confidence = (N/M) * (1/(1+residual_std))` (`wps.py:204-205`). `n̂` unitário
(`wps.py:194`) ⇒ `L·n̂ = O(1)` ≠ `I ∈ [0,255]`.

**Sugestão de correção:** normalizar o resíduo pela escala do sinal (ex.: `residual_std /
(albedo + eps)` ou trabalhar com intensidades em 0-1 e albedo explícito) para tornar a
confiança invariante a ganho. NÃO aplicar.

---

## PS-06: NaN nas normais (sombra/degenerado) escritos diretamente no FNI consumido pelo solver C

- **Localização:** nascem em `src/hybrid_stereo_method/photometric/wps.py:155,183,191`; gravados em `main_wps.py:144`; consumidos em `hybrid/main.py:208,254` → `hybrid/integrate.py:231,94` → `infrastructure/io/image_io.py:173-177`
- **Tipo:** implementação
- **Severidade:** alto
- **Status:** suspeita (lead para INT/CONV; decide: teste end-to-end híbrido com pixels sombreados)

**Descrição:** Pixels com menos de 3 luzes válidas (sombra, PS-02) ou solução degenerada
recebem `normals[i,j,:] = np.nan` (`wps.py:155`, `:183`, `:191`). Esse `normals` é salvo
**sem tratamento** em `normal_map.npy` (`main_wps.py:144`). No pipeline híbrido,
`hybrid/main.py:208` carrega o `.npy` e o passa a `integrate_normals_to_height`
(`hybrid/main.py:254`), que chama `convert_image_array_to_fni(normal_map, ...)`
(`hybrid/integrate.py:94`). A conversão formata cada componente com `f"{v:+.7e}"`
(`image_io.py:173-177`) — para `NaN` isso gera literalmente a string `+nan` (ou
`+nan`/`nan` conforme a plataforma) **dentro do FNI**, que é então lido pelo binário C
`gus_integrate_recursive -normals`. O solver de integração não recebe nenhuma máscara de
validade pela via das normais (apenas os *hints* do multifocus têm canal de confiança), e
um `nan` propagado num solver multigrid de integração tende a **contaminar toda a malha**
(qualquer soma/relaxação que toque um nó NaN vira NaN). Registrar como lead para a auditoria
de integração (INT) e de convenções (CONV): verificar se o C trata `nan`/sentinela e se
existe máscara de foreground.

**Evidência:** `np.nan` atribuído em `wps.py:155,183,191`; `np.save(normal_map_path,
normals)` em `main_wps.py:144`; `convert_image_array_to_fni(normal_map, input_fni_path)` em
`hybrid/integrate.py:94`; formatação `f"{...:+.7e}"` sem checagem de finitude em
`image_io.py:173,176`. Não há `np.nan_to_num`/máscara em nenhum ponto do caminho.

**Sugestão de correção:** substituir NaN por um valor seguro (ex.: normal `(0,0,1)` com
confiança 0) e propagar uma máscara de validade ao integrador; ou garantir que o solver C
ignore nós marcados. NÃO aplicar — registrar para a fase de integração.

---

## PS-07: Entrada do PS quantizada a 8 bits (sMos.png) — sMos.fni float existe mas é ignorado

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:160-163`; releitura em `main_wps.py:109` via `read_images` (`image_io.py:90` `IMREAD_UNCHANGED`)
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita (decide: `tests/test_photometric_synthetic.py` — clean-data accuracy float vs uint8)

**Descrição:** O mosaico all-in-focus por luz é gravado **duas vezes**: como `sMos.png`
(`save_image(..., normalize=False)` → `np.clip(img,0,255).astype(np.uint8)`,
`image_io.py:140`) e como `sMos.fni` float (`sMos_light/255.0`, `hybrid/main.py:161-162`).
O fotométrico, porém, lê o **PNG** (`main_wps.py:109` coleta `sMos_path_list` de
`hybrid/main.py:177-184`, que filtra `sMos.png`), descartando o `.fni` float. Assim a
entrada do modelo Lambertiano fica quantizada a **8 bits** — apesar de o mosaico vir de
interpolação sub-pixel (`mosaic.py:74`) e de médias por plano focal, que produzem float com
mais bits efetivos. Como o PS resolve `L·n̂ = I` por mínimos quadrados, o ruído de
quantização (±0.5/255) entra direto nas normais; em regiões de baixo albedo/baixo contraste
entre luzes, a relação sinal-ruído cai e a normal fica mais ruidosa. Também há `clip(...,
0, 255)` (`image_io.py:140`): qualquer valor de mosaico >255 é **saturado silenciosamente**
antes do PS, violando a linearidade exigida pelo modelo.

**Evidência:** `save_image(... "sMos.png" ..., normalize=False)` (`hybrid/main.py:160`),
clip+uint8 em `image_io.py:140`; `sMos.fni` float gravado mas não referenciado em nenhum
consumo do PS (a seleção em `hybrid/main.py:181` é `os.path.basename(file) == "sMos.png"`).
O teste de clean-data-accuracy pode comparar normais a partir do `.fni` float vs do `.png`
uint8 sob dados sintéticos sem ruído.

**Sugestão de correção:** alimentar o PS com o `sMos.fni` float (ou um array em memória) em
vez do PNG uint8, e remover o clip a 255 (ou usar dtype de maior profundidade). NÃO aplicar.

---

## PS-08: Linearidade radiométrica (gamma) não é tratada em nenhum ponto

- **Localização:** cadeia inteira (`hybrid/main.py` médias/mosaico → `main_wps.py` → `wps.py`); nenhum decode de gamma encontrado
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita (decide: depende do protocolo de aquisição dos sVal.png — registrar premissa)

**Descrição:** O modelo Lambertiano `I = ρ·(L·n̂)` exige intensidades **lineares** em
radiância. Os `sVal.png` de entrada são lidos e usados diretamente (médias por `zf`,
mosaico, grayscale, `lstsq`) sem nenhum passo de **linearização/remoção de gamma** (grep por
`gamma`/`srgb`/`pow` no pacote fotométrico não retorna nenhum decode radiométrico; os hits
em `wps.py`/`numerics.py` são "linear system"/`GAMMA_THR`, não gamma de imagem). Se os PNGs
forem codificados em sRGB/gamma (≈2.2), o `lstsq` ajusta um modelo linear a dados
não-lineares e as normais ficam sistematicamente enviesadas (curvatura aparente da
superfície distorcida). Como o pipeline não documenta nem impõe linearidade, isto é uma
premissa não verificada. Se a aquisição já entrega imagens lineares (câmera científica em
RAW/linear), não há erro — daí "suspeita", a decidir pelo protocolo de dados.

**Evidência:** ausência de qualquer linearização no caminho; intensidades cruas entram em
`np.linalg.lstsq` (`wps.py:165`). O comentário em `hybrid/main.py:157-159` justifica
`normalize=False` para preservar a relação cross-light, o que **só** é fisicamente válido se
as imagens já forem lineares.

**Sugestão de correção:** documentar e/ou impor a premissa de linearidade (linearizar na
entrada se os PNGs forem gamma-encoded). NÃO aplicar.

---

## PS-09: `convert_to_grayscale` falha em entrada já monocromática e mistura convenções de coeficientes

- **Localização:** `src/hybrid_stereo_method/infrastructure/utils.py:74-85`; chamada em `main_wps.py:115`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** confirmado (por inspeção, item de robustez) / coeficientes: suspeita

**Descrição:** `convert_to_grayscale` chama `cv2.cvtColor(img, COLOR_BGR2GRAY)`
incondicionalmente. (1) **Robustez:** se a imagem de entrada já for **mono-canal**
(2D), `cvtColor(BGR2GRAY)` **lança exceção** ("Invalid number of channels", reproduzido
nesta auditoria). No fluxo híbrido, esse crash é **inalcançável** com os dados atuais:
`sMos.png` tem sempre 3 canais porque `mosaic` desempacota `n,h,w,chanels` (`mosaic.py:32`)
exigindo stack 4D — os `sVal.png` são sempre coloridos no dataset-alvo. O crash é um
**caso de borda para datasets futuros** com `sVal.png` em tons de cinza (entrada mono →
stack 3D → mosaico 2D → `sMos.png` mono-canal → crash em `convert_to_grayscale`). (2)
**Coeficientes/convenção (CONV):** `cv2.COLOR_BGR2GRAY` usa pesos Rec.601
`0.299R+0.587G+0.114B` sobre a ordem **BGR do cv2**, o que está correto para imagens lidas
por cv2. Porém o **outro** caminho (`rps.py` via `ps_utils.converter_npy_para_cinza`,
`ps_utils.py:146-156`) usa pesos `0.3R+0.59G+0.11B` e **detecta RGB vs BGR por média de
canais** (heurística `mean(canal0) > mean(canal2)`), que pode classificar errado dependendo
da cena — divergência de convenção (Rec.601 vs heurística de ps_utils) entre os dois entry
points, observada como achado cruzado CONV. `convert_to_grayscale` aceita float (converte
float64→float32, `utils.py:83-84`) e uint8, mas não trata entrada já 2D.

**Evidência:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` (`utils.py:85`) — reprodução com PNG
mono lançou `Bad number of channels ... scn is 1`. Heurística divergente em
`ps_utils.py:150-155` (`converter_npy_para_cinza`).

**Sugestão de correção:** checar `img.ndim`/nº de canais e retornar a imagem inalterada se
já for mono; unificar a política de grayscale entre os dois entry points. NÃO aplicar.

---

## PS-10: `estimate_normals_argmax` inverte 3 luzes sem guard de singularidade (mas é código morto)

- **Localização:** `src/hybrid_stereo_method/photometric/wps.py:8-52` (linha crítica `:44`)
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** confirmado (por inspeção, código morto)

**Descrição:** `estimate_normals_argmax` resolve o sistema com `np.linalg.inv(selected_lights)`
(`wps.py:44`) sobre as **3 luzes mais brilhantes** do pixel. Se essas três direções forem
quase **coplanares** (comum: as luzes mais brilhantes para uma dada normal tendem a estar
agrupadas), a matriz é quase singular e `inv` produz uma normal numericamente instável (sem
`try/except`, sem `rcond`, sem fallback). Conceitualmente, fotometric stereo exige luzes
**não-coplanares** para condicionar o sistema; escolher sempre as 3 mais brilhantes não
garante isso. Severidade **baixa** porque a função é **código morto**: grep por
`estimate_normals_argmax(` em `src/` não retorna **nenhuma** chamada; o único entry point
fotométrico do híbrido usa `estimate_normals_argmax_lstsq_robust` (`main_wps.py:135`) e o
`main.py` usa `RPS`. O mesmo vale para `estimate_normals_argmax_lstsq` (`wps.py:55`), também
sem chamadas.

**Evidência:** `normal = np.dot(np.linalg.inv(selected_lights), selected_values)`
(`wps.py:44`), sem guard. `grep -rn "estimate_normals_argmax(" src/` → vazio;
`estimate_normals_argmax_lstsq` só aparece em sua própria `def` (`wps.py:55`). `config`
`top_k: 3` (configs/*.yaml) só seria usado por essas funções mortas.

**Sugestão de correção:** se reativadas, usar `lstsq`/`pinv` com `rcond` e checar
condicionamento das 3 luzes (ou selecionar luzes que maximizem o volume/triângulo esférico).
NÃO aplicar.

---

## PS-11: `disp_normalmap`/`disp_channels` trocam canais IN-PLACE no array do chamador e bloqueiam em headless

- **Localização:** `src/hybrid_stereo_method/photometric/visualization.py:136-137`, `165-168`, `141-142`, `184-185`, `254-255`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** confirmado (por inspeção)

**Descrição:** (1) **Mutação in-place:** `disp_normalmap` faz
`N = np.reshape(normal, (h,w,3))` — `np.reshape` devolve uma **view** quando possível — e em
seguida `N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()` (`visualization.py:136-137`),
trocando canais **sobre o array do chamador** (`normals`). Em `main_wps.py` o `np.save` do
`normal_map.npy` ocorre **antes** (`main_wps.py:144`) das chamadas de visualização
(`main_wps.py:181-197`), então hoje é benigno; mas `disp_normalmap` é chamado **antes** de
`disp_channels` (`main_wps.py:181` vs `:182`), e `disp_channels` repete o swap
(`visualization.py:168`) sobre o **mesmo** array — dois swaps encadeados sobre a view do
chamador são um risco latente se a ordem/uso mudar. (2) **Headless:** `cv2.imshow` +
`cv2.waitKey(0)` (`visualization.py:141-142`, `184-185`, `254-255`) com `delay=0`
**bloqueiam** aguardando tecla e exigem display — impede automação/execução headless do
pipeline (o híbrido chama esses displays com `delay=0` via `main_wps.py`).

**Evidência:** `N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()`
(`visualization.py:137` e `:168`) sobre `np.reshape(normal, ...)` (view). `cv2.imshow(name,
...); cv2.waitKey(delay)` com `delay=0` (`visualization.py:141-142`). `main_wps.py:181-197`
chama as três funções de display incondicionalmente.

**Sugestão de correção:** operar sobre cópia (`N = normal.reshape(...).copy()`) e tornar a
visualização opcional/condicional a flag de debug (ou usar `cv2.imwrite` sem `imshow`). NÃO
aplicar.

---

## Verificado sem achado

- **`evaluate_angular_error` — fórmula correta** — `arccos` do produto interno por linha com
  clip explícito a `[-1, 1]` (`ps_utils.py:134-140`) e conversão para graus; matematicamente
  o erro angular padrão entre normais. *Caveat de propagação de NaN:* se `normal` contiver
  NaN (PS-06), `aesum` vira NaN, os clips `>1.0`/`<-1.0` **não** capturam NaN (comparações
  com NaN são False), `arccos(nan)=nan` e `np.mean` do erro vira NaN
  (`main_wps.py:176`); registrado como consequência de PS-06, não achado independente, pois
  a fórmula em si está correta e o `background` mascara apenas os pixels da máscara, não os
  NaN fora dela.
- **`np.linalg.lstsq(..., rcond=None)` no solver robusto** — uso de `lstsq` (não `inv`) com
  `rcond=None` em `wps.py:165` é adequado para o sistema sobre-determinado `L·n̂=I` com ≥3
  luzes; a normalização posterior (`wps.py:194`) trata o fator de escala. (O defeito está em
  *o que* se faz com a magnitude — PS-01 — não no solver.)
- **Guard de norma zero antes de normalizar a normal** — `if norm == 0: nan` em
  `wps.py:189-193` evita divisão por zero ao normalizar soluções degeneradas.
- **Guard de nº mínimo de luzes** — `if np.sum(valid_indices) < 3` (`wps.py:154`) e
  `if len(selected_values) < 3` (`wps.py:182`) impedem `lstsq` subdeterminado; coerente com o
  mínimo teórico de 3 luzes não-coplanares.
- **Chaves de config do solver batem com os `get` do `wps.py`** — `epsilon`,
  `shadow_threshold`, `outlier_threshold_multiplier` em `configs/{hb,ps,wps}_experiment.yaml`
  (seção `photometric.solver`) correspondem exatamente aos `wps_params.get(...)` de
  `wps.py:134-136`; `wps_params = parameters["photometric"]["solver"]` (`main_wps.py:133`)
  lê a seção certa. (`top_k` está nos YAML mas só é consumido por código morto — ver PS-10;
  não é "chave errada", é parâmetro órfão.)
- **Pareamento imagem↔luz tem checagem de contagem** — `if len(images) !=
  light_sources.shape[0]: raise ValueError` (`main_wps.py:122-127`) impede desalinhamento
  silencioso de contagem entre `sMos_path_list` e linhas de `lights.npy`. (A *ordem* do
  pareamento — natsort vs ordem física das luzes — é questão de convenção, fica para CONV.)
- **`_solve_l2` (RPS) segue Woodham** — `N = lstsq(L.T, M.T)[0].T` e `normalize(N, axis=1)`
  (`rps.py:144-145`) é a formulação padrão (`M = N·Lᵀ`, recupera `N` por mínimos quadrados,
  normaliza para absorver albedo). Máscara aplicada via `background_ind` zerando linhas
  (`rps.py:146-147`).
- **`_solve_l1`/`_solve_sbl` (RPS) seguem Ikehata et al.** — montam `A = L.T`,
  `b = M[index,:].T` por pixel e chamam `L1_residual_min`/`sparse_bayesian_learning`
  (`rps.py:159-232`); recuperação de `N` consistente com a regressão esparsa de resíduo do
  paper citado. Máscara via `foreground_ind`.
- **`_solve_rpca` (RPS) segue Wu et al.** — decompõe `M = A + E` por RPCA (`rpca_inexact_alm`)
  e resolve `N = lstsq(L.T, A)` (`rps.py:285-298`); formulação low-rank + sparse correta;
  máscara aplicada a `M` antes da decomposição e `N` re-expandido para o frame completo.
- **Convenção de saída do RPS e do WPS é a mesma** — ambos produzem `normal_map.npy` em
  `(H, W, 3)` com normais unitárias por linha (`rps`: `normalize(N, axis=1)` + reshape em
  `save_normalmap_as_npy`, `ps_utils.py:116`; `wps`: `normal/=norm` + array `(h,w,3)`,
  `wps.py:194`,`main_wps.py:144`); orientação/ordem de canais idênticas no ponto de gravação
  (o swap RGB↔BGR só ocorre na visualização, PS-11, depois do save). Logo os dois
  `normal_map.npy` são intercambiáveis para a integração — a única diferença é background
  zerado (`rps`) vs NaN (`wps`, PS-06).
- **`numerics.py` — estabilização numérica presente** — `L1_residual_min` usa
  `eps=1e-8` em `1.0/max(sqrt(|r|), eps)` (`numerics.py:42`) evitando divisão por zero;
  `sparse_bayesian_learning` usa `GAMMA_THR=1e-8` (`numerics.py:62,86`) e
  `lambda2·I` como regularizador de `C` antes de `np.linalg.solve` (`numerics.py:64,75,77`);
  `rpca_inexact_alm` usa `dual_norm` e `d_norm` com guard implícito (entradas não-nulas) e
  `1.0/mu` como threshold de SVD. Todos os `lstsq` usam `rcond=None`. Sem divisões por norma
  zero não guardadas no caminho ativo.
