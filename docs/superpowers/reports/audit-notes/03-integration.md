# Fase 1.3 — Achados Integração

Data: 2026-06-04

Auditoria de corretude (matemática/teórica) e de implementação (bugs/precisão/robustez)
do estágio de **integração de superfície**: `hybrid/integrate.py` grava o mapa de normais
num FNI textual e invoca o binário C `csrc/integrate_recursive/gus_integrate_recursive`
(integrador multigrid recursivo de campo de gradiente da família pst/Stolfi), depois relê o
FNI de altura. Teoria de referência: integração de campo de gradiente por mínimos quadrados
ponderados (Poisson com pesos / balanço de fluxo por célula), Horn & Brooks; multigrid
geométrico (restrição/prolongação). O auditor lê apenas o caminho executado e o **contrato**
Python↔C; a biblioteca pst vendida não é auditada além do que o checklist pede.

Cada achado cita `arquivo:linha` reais (verificados no branch `nova_iteracao`, incluindo os
`.c`). Na Fase 1 os achados são "suspeita", salvo quando o código por si só prova o defeito
("confirmado (por inspeção)"). Semânticas de C que dependem de execução remetem aos testes de
rampa/bump da Fase 3 (`tests/test_convention_integration.py`, Task 9).

Resumo do caminho executado (modo `-normals`): `main` lê o FNI de normais como imagem de 3
canais (`gus_integrate_recursive.c:441`), converte normal→slope com
`pst_normal_map_to_slope_map(N, maxSlope=1000)` (`:444`), aplica `scale` opcional aos
gradientes (`:449-450`), garante consistência de pixel (`:451`), aloca `Z` com `(NX+1)×(NY+1)`
2 canais (`:457-459`), lê hints/reference se dados, define o chute inicial conforme
`-initial` (`:534-549`), e chama `pst_integrate_recursive` (`:602`). Em sucesso escreve
`{PREFIX}-00-end-Z.fni` (via `reportHeights`→`pst_height_map_analyze_and_write`, com
`iter=-1`⇒sufixo `-end`).

---

## INT-01: `initial_method` default "hints" sem `-hints` aborta o binário (ou, com hints, mistura unidades como chute)

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:214`; consumido em `gus_integrate_recursive.c:543-546` (`demand(H != NULL, ...)`)
- **Tipo:** implementação
- **Severidade:** alto
- **Status:** confirmado (por inspeção) — o ramo de aborto; "suspeita" para o impacto numérico do chute

**Descrição:** `IntegrateRecursiveConfig(initial_method=integration_params.get("initial_method", "hints"), ...)`
usa default `"hints"` (`main.py:214`). Se o YAML omitir `initial_method` **e** `use_hints`
for falso/ausente, nenhum `-hints` é montado (`integrate.py:108-115`), mas
`config.initial_method == "hints"` é sempre passado em `-initial hints 0.0`
(`integrate.py:129`). No C, `tire_parse_options` exige `-initial` via
`argparser_get_keyword` (`gus_integrate_recursive.c:759`, que **encerra com erro** se ausente;
Python sempre o fornece, então isso é benigno), e em `tire_compute_and_write_height_map` o
ramo `strcmp(o->initial_opt,"hints")==0` executa `demand(H != NULL, "hints height map not
spacified")` (`:545`). `demand` é `affirm` (`affirm.h:29`), que **aborta** o processo →
`subprocess.run(check=True)` levanta `CalledProcessError` → `RuntimeError` (`integrate.py:163-167`).
Ou seja: uma config válida do ponto de vista do schema Python (sem `initial_method`, sem
hints) provoca **falha dura** da integração em vez de cair para o chute "zero" seguro. O
default deveria ser `"zero"` (como o próprio binário documenta, `gus_integrate_recursive.c:253`,
e como o `IntegrateRecursiveConfig` dataclass usa, `integrate.py:45`). Observação: a config
empacotada `configs/hb_experiment.yaml:112` define `initial_method: "zero"` e
`use_hints: True`, mascarando o problema no fluxo-padrão; o defeito é a divergência entre o
default do dataclass (`"zero"`) e o default do `.get(...)` (`"hints"`).

Mesmo quando hints existem e `initial=hints` é legítimo, o chute inicial copia
`H[0]` (altura em unidades físicas de `z_foc`, ver INT-04) para `Z[0]`
(`gus_integrate_recursive.c:546`, `float_image_assign_channel_rectangle`), enquanto o sistema
resolve alturas em unidades de pixel; se as escalas diferirem, o solver parte de um ponto
deslocado (apenas mais iterações, não erro final, pois é só chute) — registrado aqui como
contexto, o erro de unidade em si é INT-04.

**Evidência:** `.get("initial_method", "hints")` (`main.py:214`) vs dataclass default
`"zero"` (`integrate.py:45`); `demand(H != NULL, ...)` no ramo "hints"
(`gus_integrate_recursive.c:545`). `demand`→`affirm` aborta (`affirm.h:29`).

**Sugestão de correção:** default `"zero"` no `.get` (alinhar com o dataclass e o binário); ou
validar que `initial_method=="hints"` ⇒ `use_hints` verdadeiro antes de invocar. NÃO aplicar.

---

## INT-02: `{PREFIX}-00-end-Z.fni` ausente → lê silenciosamente o CHUTE INICIAL `-ini-Z.fni` e o devolve como resultado

- **Localização:** `src/hybrid_stereo_method/hybrid/integrate.py:171-180`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita — comportamento de crash **confirmado por teste** (Task 9); fallback obsoleto/silencioso segue suspeita (decide: Task 12)

**Atualização Task 9 (2026-06-04):** O teste da rampa fez o binário abortar por `demand` de
canais (`pst_integrate_iterative.c:47`), retorno != 0. Observado empiricamente: `subprocess.run(
check=True)` levanta `CalledProcessError` (→ `RuntimeError` em `integrate.py:167`) **antes** de
o fallback `-ini-Z.fni` (`integrate.py:175`) ser alcançado — **confirma** a "Nota sobre
severidade" abaixo: num crash o fallback não é exercido. Notar que o `ramp-ini-Z.fni` **foi
escrito** em disco pelo binário (antes do abort), mas nunca é lido. O caminho `-normals` (bump,
PASSED) produziu `bump-00-end-Z.fni` normalmente. Ver `06-test-results.md`.

**Descrição:** Após a execução, o Python procura `{PREFIX}-00-end-Z.fni` (`integrate.py:171`).
Se não existir, faz **fallback silencioso** para `{PREFIX}-ini-Z.fni` (`integrate.py:175`) e
o retorna como mapa de altura. Mas `{PREFIX}-ini-Z.fni` é, por construção, o **chute inicial**
gravado por `tire_compute_and_write_height_map` **antes** de qualquer iteração
(`gus_integrate_recursive.c:559-562`, `"%s-ini-Z.fni"`). Com `-initial zero` esse arquivo é
todo zero; com `-initial hints` é o mapa de hints cru. Logo o fallback pode **mascarar uma
integração que não produziu saída**, devolvendo o chute como se fosse o resultado integrado,
sem qualquer aviso ou exceção. O `-00-end-Z.fni` é gravado pelo callback `reportHeights` com
`final=TRUE` (`gus_integrate_recursive.c:683-687` → `pst_height_map_analyze_and_write` com
`iter=-1` ⇒ sufixo `-end`, `float_image_mscale_file_name:163`); ele só **deixa de existir** se
o binário abortar **antes** desse ponto (ex.: `demand` falho de tamanho/canais, INT-01,
INT-04) — mas nesse caso `subprocess.run(check=True)` já teria levantado `CalledProcessError`
e o fallback nunca seria alcançado. O risco real é: (a) binário que retorna 0 mas não chega a
escrever o end-Z (caminho de erro silencioso futuro), ou (b) `output_prefix`/`cwd`
divergentes fazendo o `-00-end-Z.fni` cair noutro diretório enquanto um `-ini-Z.fni` de
execução anterior persiste — devolvendo um resultado obsoleto. A combinação "fallback para o
chute" + "sem checagem de convergência" é frágil.

**Evidência:** `height_fni_path = .../"{prefix}-00-end-Z.fni"`; se `not exists`, tenta
`.../"{prefix}-ini-Z.fni"` e só então levanta (`integrate.py:171-180`). O `-ini-Z.fni` é o
chute pré-iteração (`gus_integrate_recursive.c:559-562`). Nenhum parse do status de
convergência (o C imprime "gave up"/"converged" só em stderr, `pst_imgsys_solve.c:117,121`).

**Nota sobre severidade:** A revisão técnica recalibra para **médio**. O caminho perigoso
(returncode 0 sem end-Z) não ocorre no código atual: `pst_imgsys_solve.c:124` chama o report
final (`reportHeights`) **incondicionalmente** ao concluir, e aborts disparam `exit != 0`
capturado por `check=True`. Os riscos vivos (erro silencioso futuro, `-ini-Z.fni` de execução
anterior persistindo se `output_prefix`/`cwd` divergirem) são especulativos — fragilidade
latente que devolveria resultado errado **se** disparada, mas inalcançável no fluxo atual.
Severidade **médio** é consistente com INT-07 (também inalcançável) = baixo, pois o impacto
potencial é maior que INT-07, mas o caminho de disparo está atualmente bloqueado.

**Sugestão de correção:** remover o fallback para `-ini-Z.fni` (é semanticamente o chute, não
o resultado); se o end-Z faltar, sempre erro. Opcionalmente, parsear o stderr para detectar
não-convergência e avisar. NÃO aplicar.

---

## INT-03: NaN nas normais (PS-06) viram peso 0 — não contaminam a malha, mas removem o pixel sem máscara de foreground

- **Localização:** parser FNI C (`float_image_read`, lib pré-compilada) + `pst_normal_map_to_slope_map` (`csrc/.../lib-src/pst_normal_map.c:236-243`); origem em `wps.py:155,183,191`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita (decide: teste end-to-end híbrido com pixels sombreados, Task 12)

**Descrição:** Fechamento do lead PS-06. O FNI gravado por Python formata cada componente
com `f"{v:+.7e}"` (`image_io.py:173,176`), produzindo a string textual `+nan` para
`NaN`. O parser C (`float_image_read`, via `fget`/`nget`) lê com `strtod`/`fget_double`, que
em libc POSIX **aceita** `"nan"`/`"+nan"` e devolve `NAN` (a leitura não falha). O NaN então
**não contamina a malha**, ao contrário do receio do PS-06. O mecanismo real de contenção é
duplo:

(a) **Guarda de linha 241** (`pst_normal_map.c:240-242`): `if (! isfinite(grd.c[0]) ||
! isfinite(mag) || mag==0)` → `w = 0`. Esta guarda inspeciona apenas `grd.c[0]` (= `−nx/nz`)
e `mag = r3_L_inf_norm(&nrm)`. Ela **não é suficiente por si só**: se somente `ny` fosse NaN
enquanto `nx` e `nz` fossem finitos, `grd.c[0] = −nx/nz` seria finito; além disso,
`r3_L_inf_norm` (`r3.c:116-125`) percorre os componentes com comparação `> d`, e como
`NaN > d` é sempre falsa em IEEE 754, o componente NaN é **silenciosamente pulado** — `mag`
seria calculado sem ele e a guarda da linha 241 **não** zeraria esse pixel.

(b) **Backstop real**: `pst_map_ensure_pixel_consistency(G, 2)` em
`gus_integrate_recursive.c:451` percorre todos os canais (dados + peso) via
`pst_ensure_pixel_consistency` (`pst_basic.c:59`): se **qualquer** canal for não-finito ou o
peso for zero, zera o peso e NaN-iza todos os canais de dados. Esse é o verdadeiro "NaN → peso
0" para todos os padrões de NaN parcial.

Na prática do pipeline real, o resultado não muda: `wps.py:155,183,191` sempre escreve NaN
nos **3 componentes** em simultâneo para pixels inválidos, logo `grd.c[0]` é NaN e a guarda
da linha 241 também captura o pixel. Portanto o NaN entra no mapa de slopes com **peso zero**,
e a garantia é sólida por esta combinação — mas o backstop `pst_map_ensure_pixel_consistency`
é o mecanismo correto a citar (ver entrada "Verificado" sobre sanitização pixel-consistency).

No construtor do sistema, `append_edge_term` só adiciona a aresta se `fabs(wD) >= FLUFF=1e-140`
(`pst_integrate.c:277`); arestas de peso zero são **excluídas** (`pst_slope_map_get_edge_data`
devolve `w=0, d=NAN` e o termo é descartado). Um vértice cercado só por arestas de peso zero
recebe um "fudge term" que o puxa a 0 com peso 1 (`pst_integrate.c:212-216,326-339`),
evitando diagonal nula no Gauss-Seidel (`pst_imgsys_solve.c:90`). Portanto **NaN não propaga
NaN** — o defeito remanescente é **conceitual/de robustez**: pixels sombreados são
**silenciosamente zerados/excluídos** sem que o PS comunique uma máscara de foreground;
regiões grandes de sombra viram buracos de peso 0 cuja altura é determinada apenas pelo
fudge-to-zero ou pela vizinhança via termos diagonais, podendo introduzir vieses de borda. A
premissa "o solver ignora NaN com graça" depende de a libc aceitar `+nan` no parse — a
confirmar no teste (round-trip FNI, Task 8) por ser dependente de plataforma/locale.

**Evidência:** Guarda parcial `!isfinite(grd.c[0]) || !isfinite(mag) || mag==0` →
`w = 0` (`pst_normal_map.c:240-242`); limitação: inspeciona só `grd.c[0]` e `r3_L_inf_norm`
silencia NaN parcial (`r3.c:116-125`). Backstop real: `pst_map_ensure_pixel_consistency(G, 2)`
(`gus_integrate_recursive.c:451`) via `pst_basic.c:59` — zera peso e NaN-iza todos os canais
se qualquer componente for não-finito (ver entrada "Verificado" sobre pixel-consistency).
Pipeline real: `wps.py:155,183,191` NaN-iza os 3 componentes simultaneamente → `grd.c[0]` é
NaN → guarda da linha 241 também atinge o pixel. Exclusão de aresta `fabs(wD) >= FLUFF`
(`pst_integrate.c:277`); fudge-to-zero (`pst_integrate.c:326-339`);
`demand(cf_k != 0.0, "...zero in the diagonal")` (`pst_imgsys_solve.c:90`) — só seria
violado se um vértice ficasse sem nenhum termo, o que o fudge impede.

**Sugestão de correção:** PS deve emitir um canal de peso explícito (normal_map (H,W,4)) com
0 nos sombreados em vez de NaN, e/ou propagar uma máscara; documentar que sombras viram
peso-0. NÃO aplicar (corrigir na origem, PS).

---

## INT-04: hints em unidades físicas de `z_foc` somados a alturas em unidades de pixel; `hints_weight` mistura grandezas de escalas diferentes

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:227,231`; `integrate.py:108-114`; consumo em `pst_integrate.c:294-324` (`append_hints_term`)
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** suspeita (decide: teste de rampa/bump com hints, Task 9/12)

**Descrição:** O termo de hints adiciona à equação de cada vértice
`qoo = woo*(H[0,x,y] - Z[x,y])²` com `woo = hintsWeight * H[1,x,y]`
(`pst_integrate.c:111-113,308-323`), competindo no mesmo sistema de mínimos quadrados com os
termos de aresta `(d - (Z[b]-Z[a]))²`. Os termos de aresta têm `d = ux*gx+uy*gy` = diferença
de altura **por célula** (unidade: altura-por-pixel); a solução `Z` fica, portanto, em
**unidades de altura tais que uma célula do grid vale 1 em X/Y** — efetivamente "altura em
unidades de pixel". Já `H[0]` vem de `zMos_with_confidence.fni` (multifocus), cujos valores
são alturas em **unidades físicas do plano focal `z_foc`** (passo entre `zf`), uma escala
totalmente diferente da altura-por-pixel. O `-hints` é montado **sem** `scale`
(`integrate.py:109,114` passa só `path` e `weight`; `tire_parse_scale` então fixa
`hints_scale=1.0`, `gus_integrate_recursive.c:746,825`). Assim o solver mistura, na mesma
soma ponderada, um alvo `H[0]` numa escala e diferenças de slope noutra: `hints_weight=0.1`
(`configs/hb_experiment.yaml:121`) não tem significado físico consistente porque pondera duas
grandezas incomensuráveis. O resultado tende a uma altura "entre" as duas escalas, enviesando
a superfície na direção do `z_foc` proporcionalmente a `hints_weight`. Some-se a isto que o
peso `H[1]` (confiança do multifocus) entra direto sem normalização de escala.

**Evidência:** termo `woo*(H[0]-Z)²` no mesmo sistema dos termos de slope
(`pst_integrate.c:111-113,318-323`); `-hints` sem `scale` (`integrate.py:108-114`) ⇒
`hints_scale=1.0`; hints em `z_foc` (multifocus) vs `Z` em altura-por-pixel (slopes
adimensionais integrados sobre células unitárias). A docstring de `integrate_normals_to_height`
chama hints de `(H+1,W+1)` "grade de vértices" (`integrate.py:217`), mas essa docstring
documenta o caminho in-memory `hints_map`, que o pipeline híbrido **não usa** (usa
`hints_fni_path`, `hybrid/main.py:231`); o contrato de shape `(H+1,W+1)` não é verificado em
nenhum dos dois caminhos. O arquivo `zMos_with_confidence.fni` é de células `(H,W,2)` — ver INT-05.

**Sugestão de correção:** converter os hints para a mesma unidade das alturas integradas
(passar `-hints path scale Hsz weight` com `Hsz` = fator z_foc→pixel) antes de integrar; ou
documentar e impor que `zMos` já esteja em unidades de pixel. NÃO aplicar.

**Evidência executável (2026-06-05):** rampa sintética física com pixel de tamanho 2.5 em
unidades de z_foc: integração com escala default dá fit afim `a = 2.5003` contra o gt
físico (alturas em unidade de pixel — viés exatamente igual ao tamanho do pixel); com
`slopes_scale=(2.5, 2.5)` dá `a = 1.0001`. Com hints na escala de z_foc e peso moderado
(`w=0.1`, o default do config), a amplitude de grande escala da solução é ditada pelos
hints (`a` salta de 2.5 para ~1.04), enquanto o detalhe fino segue os slopes em escala
de pixel — uma quimera de duas escalas, exatamente o mecanismo previsto. **Achado anexo:**
`szero = TRUE` hard-coded (`pst_integrate_iterative.c:75`) força a solução a soma zero a
cada nível — hints **nunca** ancoram o nível absoluto (verificado: `z.mean() = 0.0` exato
para qualquer peso), apenas a forma/níveis relativos.

**Correção aplicada:** PENDING_SHA (2026-06-05) — novo parâmetro
`hybrid.integration.pixel_size` (tamanho lateral de 1 pixel nas mesmas unidades de
`z_foc`); o helper `build_integration_config` (`hybrid/main.py`) deriva
`slopes_scale = (pixel_size, pixel_size)`, então o C multiplica os slopes pelo passo
físico do pixel e `Z` sai em unidades de `z_foc` — comensurável com os hints, que entram
sem escala própria. Warning logado quando `use_hints=True` sem `pixel_size` configurado.
O parâmetro está documentado (comentado) em `configs/hb_experiment.yaml` — o valor é
geométrico e precisa ser medido por dataset. 10 testes em
`tests/test_integration_units.py` (derivação do config, warning, unidades físicas via
binário: `a≈1` com pixel_size vs `a≈2.5` default, consistência hints+slopes escalados
com rms demeaned 0,14% do std a peso 1000). Ver CONV-4 (mesma correção).

---

## INT-05: hints `(H,W,2)` (células) entregues a um alvo `(H+1,W+1)` (vértices) — C expande com `expand_by_one`, deslocando os hints meia-célula

- **Localização:** `gus_integrate_recursive.c:464,691-711` (`tire_read_fni_file` + `float_image_expand_by_one`); docstring divergente em `integrate.py:217`
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** suspeita (decide: teste de bump com hints em grade de células, Task 9)

**Descrição:** O mapa de altura `Z` tem `(NX_G+1)×(NY_G+1)` (vértices da grade,
`gus_integrate_recursive.c:457-459`), e o C **exige** que `H` tenha exatamente esse tamanho de
vértices (`demand((NX_H==NX_Z)&&(NY_H==NY_Z), ...)`, `:519`). O `zMos_with_confidence.fni` do
multifocus é uma grade de **células** `(NY_G, NX_G, 2)` (mesmo tamanho do mapa de
slopes/normais). O `tire_read_fni_file` é chamado com `(NX_Z, NY_Z)` e, ao detectar que o mapa
lido tem uma linha/coluna a menos (`NX_I==NX-1 && NY_I==NY-1`), **expande** com
`float_image_expand_by_one(I, 1)` (`:701-707`), produzindo um mapa de vértices a partir da
grade de células. Essa expansão é uma **conversão célula→vértice** que desloca os valores em
meia-célula (o valor da célula `(x,y)` passa a ser tratado como altura do vértice `(x,y)`),
introduzindo um viés de meio-pixel/extrapolação de borda nos hints. Não há corrupção de
tamanho (o C aceita), mas há **deslocamento sistemático** dos hints relativo às alturas que
eles deveriam ancorar. A docstring Python afirma `hints_map` com shape `(H+1, W+1)`
(`integrate.py:217`), o que **não** corresponde ao arquivo de células de fato passado — a
divergência indica que o autor pode não estar ciente da expansão implícita.

**Evidência:** `Z` é `(NX_G+1)×(NY_G+1)` (`:457-459`); `demand` de tamanho de vértices para `H`
(`:519`); expansão célula→vértice em `tire_read_fni_file` (`:701-707`,
`float_image_expand_by_one`). Hints do multifocus são grade de células (mesmo NX,NY do mapa de
normais). Docstring `(H+1,W+1)` em `integrate.py:217` ≠ arquivo real.

**Sugestão de correção:** gerar `zMos` já como grade de vértices `(H+1,W+1)`, ou documentar e
aceitar o erro de meia-célula como tolerável; alinhar a docstring. NÃO aplicar.

---

## INT-06: reference `hAvg.png` em uint8 (0-255) comparado a `Z` em unidades de pixel — análise de erro entre grandezas incomensuráveis

- **Localização:** `src/hybrid_stereo_method/hybrid/main.py:243,247`; comparação em `pst_map_compare.c:96-101,151-165`
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** suspeita (decide: teste de rampa com reference conhecido, Task 9)

**Descrição:** Quando `use_reference` é verdadeiro, `hAvg.png` é lido com `read_image`
(`IMREAD_UNCHANGED` ⇒ uint8 para PNG de 8 bits; `IMREAD_UNCHANGED` preserva a profundidade do
arquivo — um PNG de 16 bits retornaria uint16; `image_io.py:90`), convertido a `float32`
(`main.py:247`) e gravado como FNI de reference `R`. O `-reference` é montado **sem** `scale`
(`integrate.py:118-119` passa só o path; `tire_parse_scale` fixa `reference_scale=1.0`,
`gus_integrate_recursive.c:754,825`). O C compara `E = Z - R` célula a célula
(`pst_map_compare.c:96-97`) e reporta `avgE/devE` (`:151-165`). Mas `Z` está em
**altura-por-pixel** (slopes adimensionais integrados), enquanto `R` está em **níveis de
cinza 0-255** — grandezas incomensuráveis salvo coincidência de escala. Além disso, `Z`
carrega uma **constante de integração arbitrária** por componente conexa
(documentado em `gus_integrate_recursive.c:138-143`). A comparação remove a **média** do erro
(o mapa `E` é deslocado a média zero antes de escrever, `pst_map_compare.c:225`, e `devE` é o
desvio **em torno da média**, `:153,163`), o que cancela a constante de integração — bom —
mas **não** remove diferença de escala nem inclinação (tilt/plano). Logo o `devE` reportado
(o resumo de erro de uma linha, `pst_map_compare.c:235-243`) mistura erro real de
reconstrução com a discrepância de unidade pixel-vs-cinza, tornando o número de erro **não
interpretável** a menos que `hAvg` esteja, por convenção do dataset, já em unidades de altura
comensuráveis com `Z`. A análise é apenas diagnóstica (não realimenta o solve), limitando a
severidade a "médio".

**Evidência:** `read_image(hAvg.png)` ⇒ uint8 para PNG de 8 bits (uint16 para 16 bits; `IMREAD_UNCHANGED`, `image_io.py:90`), `astype(np.float32)`
(`main.py:247`), `-reference` sem `scale` (`integrate.py:118-119`) ⇒ `reference_scale=1.0`.
`E=Z-R` cru (`pst_map_compare.c:96-97`); `E` deslocado a média zero (`:225`); `devE` em torno
da média (`:153,163`); resumo em `:235-243`. `Z` em altura-por-pixel com constante de
integração arbitrária (`gus_integrate_recursive.c:138-143`).

**Sugestão de correção:** passar `-reference path scale Rsz` com `Rsz` levando cinza→altura,
ou fornecer `hAvg` já em unidades de altura; opcionalmente remover plano/tilt antes de
reportar erro. NÃO aplicar.

---

## INT-07: default de `-maxLevel` no C é `DEFAULT_MAX_ITER` (100000), não `DEFAULT_MAX_LEVEL` (30)

- **Localização:** `csrc/integrate_recursive/gus_integrate_recursive.c:783`
- **Tipo:** implementação
- **Severidade:** baixo
- **Status:** confirmado (por inspeção) — inativo no fluxo Python atual

**Descrição:** No parser, o ramo que define o default de `maxLevel` quando `-maxLevel` é
omitido atribui `o->maxLevel = DEFAULT_MAX_ITER` (=100000, `gus_integrate_recursive.c:783`),
em vez de `DEFAULT_MAX_LEVEL` (=30, definido em `:84` mas **nunca usado** no parser). O help
de `-maxLevel` (linhas 265-269) documenta o comportamento "recursão até um único pixel", não
um default numérico fixo de 30 — o que reforça o diagnóstico de copy-paste: `DEFAULT_MAX_LEVEL`
foi definido mas jamais conectado ao parser.
É um **copy-paste bug**: a recursão usaria até 100000 níveis se `-maxLevel` faltasse. O
critério de parada real também é `trivial = (NX_G<=3 && NY_G<=3)`
(`pst_integrate_recursive.c:27,92`), então a recursão pararia ao chegar a ~3×3 muito antes de
100000 níveis — o efeito prático é nulo (a recursão termina pelo tamanho, não pelo nível). No
fluxo Python o bug é **inalcançável**: `integrate.py:130` sempre passa `-maxLevel`
(default 30 do dataclass/`main.py:216`), de modo que o ramo defeituoso nunca executa. Fica
registrado como erro latente do binário (corretude do default documentado vs implementado).

**Evidência:** `o->maxLevel = DEFAULT_MAX_ITER;` no `else` de `-maxLevel`
(`gus_integrate_recursive.c:783`); `DEFAULT_MAX_LEVEL` =30 (`:84`) definido mas nunca usado
no parser; help de `-maxLevel` (`:265-269`) descreve comportamento "até pixel único" sem
mencionar default 30. `integrate.py:130` sempre fornece `-maxLevel`, neutralizando o ramo.

**Sugestão de correção:** trocar `DEFAULT_MAX_ITER` por `DEFAULT_MAX_LEVEL` na linha 783. NÃO
aplicar.

---

## INT-08: `integrate_slopes_to_height` com 2 canais crasha — topo aceita 2/3 canais, solver iterativo exige 3

- **Localização:** `src/hybrid_stereo_method/hybrid/integrate.py:261-263` (docstring "shape (H, W, 2) or (H, W, 3)"); `csrc/integrate_recursive/gus_integrate_recursive.c:503` (aceita 2 ou 3); `csrc/integrate_recursive/lib-src/pst_integrate_iterative.c:47` (exige 3)
- **Tipo:** implementação
- **Severidade:** médio
- **Status:** confirmado (teste: `tests/test_convention_integration.py::test_constant_slopes_recover_ramp_and_decide_convention`, Task 9 — crash reproduzido)

**Descrição:** O caminho `-slopes` com um mapa de **2 canais** (dZ/dX, dZ/dY), exatamente o que
`integrate_slopes_to_height` grava e o que sua docstring documenta como aceitável
(`integrate.py:261-263`), **aborta o binário**. O entry-point de topo aceita 2 **ou** 3 canais
(`demand((NC_G==2)||(NC_G==3), "gradient map {G} must have 2 or 3 channels")`,
`gus_integrate_recursive.c:503`) mas o solver iterativo interno exige **exatamente 3**
(`demand(NC_G==3, "slope map {G} must have 3 channels")`, `lib-src/pst_integrate_iterative.c:47`),
e o topo **não promove** 2→3 (não acrescenta um canal de peso) antes de recursar. O abort ocorre
no nível 0, após escrever `-ini-Z.fni` e os `-NN-beg-G.fni`. O caminho `-normals` não sofre disso
(normais entram com 3 canais Nx,Ny,Nz e `pst_normal_map_to_slope_map` produz um mapa de slope com
peso). No fluxo híbrido real o integrador é invocado via `-normals` (`hybrid/main.py`), então o
caminho quebrado é inalcançável **no pipeline atual** — mas a API pública `integrate_slopes_to_height`
está quebrada para o caso de 2 canais documentado.

**Evidência (Task 9):** crash reproduzido com `slopes` `(32,32,2)`: `pst_integrate_iterative.c:47:
** (pst_integrate_iterative) slope map {G} must have 3 channels` (stderr verbatim em
`06-test-results.md`); a tabela de candidatos de convenção do teste nunca foi impressa (crash antes
do retorno). Consequência transversal: o teste da rampa não pôde decidir CONV-1/CONV-2 por esta via.

**Sugestão de correção:** no wrapper, gravar slopes como 3 canais (dZ/dX, dZ/dY, peso=1) para o
`-slopes`; ou no C, promover 2→3 no topo antes de recursar; e corrigir a docstring de
`integrate.py:261-263`. NÃO aplicar.

---

## Verificado sem achado

- **Gramática de argumentos Python↔C confere (exceto defaults INT-01/INT-07).** O comando
  montado por `integrate.py:98-146` segue a gramática do parser C: `-normals {path}`
  (`gus_integrate_recursive.c:734-735`); `scale {sx} {sy}` (SEM hífen) logo após o mapa
  (`integrate.py:104-105` casa com `tire_parse_scale` em `:742`, que espera o keyword `scale`
  via `argparser_keyword_present_next`, `:820`); `-hints {path} {weight}` — o parser lê
  path, depois `tire_parse_scale` (ausente ⇒ scale=1), depois `weight` como double em
  `[0,1e10]` (`:744-748`), e Python passa exatamente `path weight` (`integrate.py:109,114`);
  `-reference {path}` (`:752-755`); `-initial {opt} {noise}` (`:759-761`); `-maxLevel`,
  `-maxIter`, `-convTol` (`:780-793`); `-sortSys T` (`integrate.py:135-136` ↔
  `argparser_get_next_bool`, `:795-796`); `-verbose` (`:805`); `-outPrefix {prefix}`
  (mandatório, `:807-809`). Aridades e ordens conferem; o `scale` é mesmo keyword sem hífen
  (contraria a suposição do checklist Step-1.6, mas é a gramática real). `-reportStep` está
  comentado em Python (`main.py:220`) ⇒ C usa default 0 (sem arquivos por-iteração).
- **`-normals` converte normal→slope com a convenção correta de sinais.**
  `pst_slope_from_normal` faz `dZdX = -nx/nz`, `dZdY = -ny/nz`
  (`pst_basic.c:49-50`), o que é a relação padrão entre a normal `(−dZ/dX, −dZ/dY, 1)/‖·‖` e o
  gradiente: dada a normal unitária, o gradiente é `(−nx/nz, −ny/nz)`. `nz` é pisado em
  `nzmin = hypot(nx,ny)/maxSlope` com `maxSlope=1000` (`pst_basic.c:46-48`,
  `gus_integrate_recursive.c:443`), evitando divisão por zero/`nz≤0` (normais quase horizontais
  ficam limitadas a slope 1000). A convenção de **sinal de Y** (y cresce para cima ou para
  baixo) é deixada ao usuário via `scale {sx} {sy}` com `sy` negativo
  (`gus_integrate_recursive.c:213-216`); Python passa `sy=+1` por default ⇒ assume a mesma
  orientação de Y do writer FNI. Se a convenção de Y do PS divergir, isto vira erro de
  convenção — **lead para CONV-3**, decidido pelo teste de rampa (Task 9).
  **Confirmado empiricamente (Task 9-ext):** a conversão normal→slope é consistente no
  round-trip — uma rampa `z=ax·x+ay·y` enviada como normal constante e re-integrada recupera
  exatamente `+ax·x+ay·y` (RMSE `5.2e-5`; flip-de-y `+ax·x-ay·y` rejeitado a `0.381`),
  i.e. `dZdX=-nx/nz`/`dZdY=-ny/nz` + integração preservam sinal e orientação com `sy=+1`
  (`tests/test_convention_integration.py::test_ramp_normals_decide_convention`, PASSED).
- **Equação por célula é o balanço de fluxo ponderado padrão (Poisson com pesos).**
  `pst_integrate_build_system` minimiza Σ de termos `wrs·(d_rs − (Z[viz] − Z[xy]))²`
  (axiais + diagonais) por mínimos quadrados, derivando a equação de equilíbrio
  `dQ/dZ[xy]=0` (`pst_integrate.c:59-141`); coeficientes `cf[0]+=wD`, `cf[nt]=-wD`,
  `rhs+=-wD·vD` (`:282-285`). Fronteiras são **Neumann natural** (arestas fora do grid
  retornam via `append_edge_term` com `return FALSE` por limites, `:247-252`, não entram na
  equação). Buracos de peso 0 são **excluídos** (não contaminam vizinhos): a aresta só entra
  se `wD≥FLUFF` (`:277`); vértices órfãos recebem fudge-to-zero (`:212-216,326-339`). O Gauss-
  Seidel exige diagonal não-nula (`demand(cf_k!=0)`, `pst_imgsys_solve.c:90`), garantida pelo
  fudge. `convTol` compara a **máxima variação de altura** entre iterações
  (`pst_imgsys_sol_change`, `:106,129-138`), não um resíduo — critério de parada por
  estabilidade do campo, coerente.
- **Restrição/prolongação multigrid são internamente consistentes em escala.** Na descida,
  o slope `G` é reduzido com `pst_slope_map_shrink(G, 1.0)` (`pst_integrate_recursive.c:98`) ⇒
  `pst_cell_map_shrink` faz **média** dos 4 slopes finos com `scale=1.0`
  (`pst_cell_map_shrink.c:72`): slope é "rise por célula", e como a célula grossa abrange 2
  células finas, manter a média (não somar) mantém o slope como rise-por-célula-grossa —
  correto. A altura `Z` é reduzida com `scale=0.5` (`pst_height_map_shrink(Z,0.5)`, `:106`) e
  re-expandida com `scale=2.0` (`pst_height_map_expand(...,2.0)`, `:126`): as alturas dobram ao
  voltar ao nível fino, compensando o passo dobrado — o fator ×2/÷2 **é** aplicado e se
  cancela no round-trip. Pesos restritos via soma harmônica/efetiva
  (`pst_vertex_map_shrink.c`: `(Σw·wH)²/Σw·wH²`; `pst_cell_map_shrink.c`: soma dos pesos). A
  consistência absoluta de escala fim-a-fim (rampa de slope conhecido ⇒ altura esperada)
  **fica para o teste da rampa (Task 9)**.
- **`-00-end-Z.fni` é garantido em sucesso.** O callback `reportHeights` é chamado com
  `final=TRUE` ao fim de cada nível por `pst_imgsys_solve_iterative` (`:124`) →
  `pst_height_map_analyze_and_write` com `iter=-1` ⇒ `float_image_mscale_file_name` gera o
  sufixo `-end` e tag `Z` (`float_image_mscale.c:163`,
  `gus_integrate_recursive.c:683-687`). O binário retorna 0 só após `pst_integrate_recursive`
  completar o nível 0 (`gus_integrate_recursive.c:479-488`), então em retorno 0 o
  `-00-end-Z.fni` existe. Erros (tamanho/canais/`-initial`) abortam via `demand` ⇒ retorno
  não-zero ⇒ `CalledProcessError` no Python (`integrate.py:163-167`). (O fallback INT-02 só é
  perigoso fora desse caminho feliz.)
- **Comparação com reference remove a média (constante de integração) antes de escrever o
  mapa E.** `pst_map_shift_values(E, wch, -1, avgE)` (`pst_map_compare.c:225`) desloca o erro
  a média zero, e `devE` é calculado em torno da média (`:153,163`), cancelando a constante de
  integração arbitrária do `Z`. (O que **não** é removido — escala e tilt — é o conteúdo de
  INT-06.) Unidades reportadas: as de `Z` e `R` como dados (sem normalização).
- **Indexação `float_image` (col,row) e o writer/reader FNI são mutuamente consistentes.**
  `float_image_get_sample(A, c, x, y)` indexa `[c, x, y]` com `x∈[0,NX)`, `y∈[0,NY)`
  (`float_image.c:31-37`). O FNI escrito pelo binário lista `x y val...` em ordem **row-major
  com y=0 primeiro** (verificado em `teste_20260329-08-end-Z.fni`: linhas `0 0`, `1 0`, …,
  `0 1`, …). O writer Python (`convert_image_array_to_fni`, `image_io.py:171-178`) também grava
  `y=0` (linha de topo do numpy) primeiro, e o reader (`read_fni_to_image_array`, `:227-241`)
  mapeia `image_array[y,x]=val` — round-trip Python↔C **preserva o array** (FNI-y ↔ numpy-row).
  O que permanece **aberto é a semântica física**: se o "y=0" do float_image é topo ou base
  da cena, e se isso casa com a convenção de Y do PS e do `scale sy=+1`. **Conclusão
  preliminar para CONV-3:** Python e C concordam na *indexação* (sem flip vertical no
  round-trip FNI); resta verificar o *sinal físico* de dZ/dY com o teste de rampa/bump da Fase
  3 (Task 9, `tests/test_convention_integration.py`) — registrado como suspeita-de-convenção,
  não como bug provado.
- **`pst_map_ensure_pixel_consistency` saneia pixels inválidos coerentemente.**
  `pst_ensure_pixel_consistency` (`pst_basic.c:54-66`) força `v[c]=NAN` (canais de dados) e
  `v[wch]=0` quando o pixel é "bad" (peso 0 ou qualquer componente não-finito), aplicado a
  `G` (`gus_integrate_recursive.c:451`), `H` (`:466`) e `R` (`:476`) na entrada — garante que
  NaN/peso-0 entrem no solver de forma controlada (alinha com INT-03).
