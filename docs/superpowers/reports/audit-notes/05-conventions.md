# Fase 2 — Rastreamento transversal de convenções e contratos

Data: 2026-06-04

> Nota: "NÃO aplicar" nas sugestões de correção deste arquivo é regra da FASE DE DIAGNÓSTICO
> (2026-06-04/05); as correções foram aplicadas na campanha de 2026-06-06 — ver os blocos
> "Correção aplicada" no relatório `2026-06-04-method-audit.md`.

Esta fase audita **exclusivamente as costuras entre estágios** do pipeline híbrido
(multifocus → fotométrico → integração C). Para cada uma das 6 convenções rastreia-se a
cadeia completa **produtor → arquivo → consumidor**, citando `arquivo:linha` para cada elo,
com um veredito (consistente / inconsistente / decidido-por-teste). Achados de Fase 1
(MF-xx, PS-xx, INT-xx, IO-xx) são **cross-referenciados, não reescritos**; achados novos de
costura recebem IDs CONV-xx.

Convenção de veredito:
- **consistente** — a leitura estática prova que as pontas batem.
- **inconsistente** — a leitura estática prova divergência (achado CONV-xx).
- **decidido-por-teste** — a leitura estática não decide o sinal/escala física; o teste
  exato da Fase 3 que decide é nomeado.

---

## (a) Tabela-resumo das 6 convenções

| # | Convenção | Produtor (file:line) | Consumidor (file:line) | Veredito | Evidência (curta) |
|---|---|---|---|---|---|
| 1 | Eixo z / profundidade (direção + unidade) | dataset `zf*` → `z_foc` YAML (`configs/hb_experiment.yaml:39`) → `zMos` (`mosaic.py:57,72`) | hints C (`hybrid/main.py:231`→`pst_integrate.c:308-323`) → `height_map.npy` (`hybrid/main.py:266`) | unidade: **inconsistente** (INT-04); sinal-integrador: **REFUTADO** (preservado) | sinal de `Z` no round-trip numpy↔C é preservado: `test_ramp_normals_decide_convention` PASSED (vencedor `+ax*x+ay*y`, RMSE `5.2e-5` vs `1.025` do invertido). Unidade `z_foc` vs altura-por-pixel segue = INT-04. Sinal-físico `zMos`(hints) vs `Z`: **em aberto** — Task 12 (E2E) não completou (MF-14) e luzes sintéticas no frame numpy não decidiriam; requer dados reais. Task 12-ext (workaround MF-14): cadeia completa com `a=+1.003`/`r=0.997` → não inverte z **no frame sintético** (não fecha a metade física) |
| 2 | Normais e luzes (frame / y-up vs y-down) | `lights.npy` = tuplas POV-Ray `<x,y,z>` (`data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:44-58`) → normais wps no mesmo frame (`wps.py:165,194`) | normal→slope C (`pst_basic.c:49-50`) → grade de integração (`pst_integrate.c`) | eixo-y do **integrador**: **REFUTADO** (preservado); luzes vs imagem: **EM ABERTO** | `test_ramp_normals_decide_convention` PASSED: `y` do numpy preservado no round-trip (`+ax*x+ay*y` vence; flip-y `+ax*x-ay*y` rejeitado, RMSE `0.381` vs `5.2e-5`). A reconciliação y-up(`lights.npy`)↔y-down(imagem) no PS é elo separado, **ainda aberto**. Task 12 (E2E) NÃO o fechou: pipeline aborta antes do PS (MF-14) e, por construção, luzes sintéticas estão no mesmo frame numpy das normais → não reproduz a questão POV-Ray; requer dados reais. Task 12-ext (workaround MF-14): cadeia completa com `r=0.997` → orientação preservada **no frame sintético**; metade real (POV-Ray) segue aberta |
| 3 | Origem/orientação da imagem (round-trip FNI) | writer FNI Python (`image_io.py:171-178`) → C (`float_image`/`tire_read_fni_file`) | writer C `-end-Z.fni` → reader Python (`image_io.py:227-241`) | **consistente** (indexação Task 8; orientação Task 9-ext) | round-trip preserva o array sem flip (`test_fni_roundtrip` 4/4, Task 8); a rampa via `-normals` confirma que orientação **e** sinal são preservados ponta a ponta (`test_ramp_normals_decide_convention` PASSED, RMSE `5.2e-5`) — não afeta a constante de integração (ajuste sobre `z-z.mean()`) |
| 4 | Escala dos gradientes (slope adimensional vs z físico) | slope `dZdX=-nx/nz` adimensional (`pst_basic.c:49-50`) | sistema de altura-por-pixel + hints em `z_foc` (`pst_integrate.c:308-323`) | **inconsistente** | INT-04/INT-05; `slopes_scale` default `(1,1)` (`integrate.py:53`) **nunca** configurado pelo híbrido → **CONV-4** consolida |
| 5 | Radiometria entre etapas (linearidade) | `sVal.png` uint8 → médias por-zf re-esticadas (`hybrid/main.py:111`) e mosaicos uint8 clipados (`hybrid/main.py:160`) | foco multifocus (`multifocus/main.py:96`); grayscale→PS (`main_wps.py:115,136`) | **corrigido (CONV-5)** | 3/3 pontos fechados: MF-12 (bd23651), PS-07 (45cbf57), PS-08 (7a4c8fc); gamma opcional documentado |
| 6 | Contratos de arquivo (pareamento/ordem/shape) | `natsorted(sMos_path_list)` (`hybrid/main.py:177`); `sorted(zf_directories)` (`hybrid/main.py:90`); `zMos_with_confidence.fni` `(H,W,2)` | linhas de `lights.npy` (só contagem, `main_wps.py:122-127`); `z_foc` posicional (`mosaic.py:57`); `-hints` espera vértices `(H+1,W+1)` (`gus_integrate_recursive.c:519`) | **inconsistente** | pareamento luz↔mosaico só por contagem (não por L→linha); ordem zf vs z_foc é MF-02; shape células→vértices é INT-05 → **CONV-6** consolida + risco novo de pareamento. Task 12: a guarda de contagem (`len(images)!=lights.shape[0]`) **pegou** o caso degenerado 0 vs 6 (MF-14) e abortou com mensagem clara — defesa funciona p/ 0, mas N-trocados segue suspeito |

---

## (b) Achados de inconsistência (CONV-xx)

---

## CONV-1: Direção do eixo z não é fixada/verificada ponta a ponta (z_foc cresce ↔ sinal de Z integrado)

- **Localização:** dataset `zf*` (pastas `organized_by_focus/zf*/`, reorganização pós-geração; o gerador emite pastas `F00/F01/...` por luz, não `zf*` diretamente) → `z_foc` (`configs/hb_experiment.yaml:39`) → `mosaic.py:57,72` → hints (`hybrid/main.py:231`) → `Z` do C (`gus_integrate_recursive.c:457-459`) → `height_map.npy` (`hybrid/main.py:266`)
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** **mitigado (b2c1eeb, 2026-06-06)** — a convenção adotada é agora documentada end-to-end (CONV-1): image rows = y DOWN, normais n = (-dz/dx, -dz/dy, 1)/|.|, z cresce em direção à câmera. O sinal do integrador foi REFUTADO por teste (`test_ramp_normals_decide_convention`, PASSED; rampa via normais, vencedor `z = +ax*x + ay*y`, RMSE `0.000052`). A decisão física (sinal `zMos` hints real vs `Z`) continua exigindo dado real com ground-truth — procedimento documentado no config (`flip_lights_y`). A via `-slopes`/2-canais segue indecidida-por-essa-via porque crashou (INT-08; `test_constant_slopes_recover_ramp_and_decide_convention`). Ressalva: isto decide o sinal/orientação do **integrador** no round-trip numpy↔C; o sinal-físico `zMos` (hints) vs. `Z` permanece sondado pela Task 12.

**Atualização Task 12 (2026-06-04):** O E2E híbrido (`tests/test_e2e_hybrid.py`,
XFAIL strict) **não conseguiu sondar** o sinal-físico `zMos`(hints) vs. `Z`: o pipeline
**não completa** — aborta no Passo 2 antes da integração por causa de MF-14 (detecção de
luzes vazia em layout `L<n>/zf<m>/` limpo; ver `01-multifocus.md` e `06-test-results.md`).
Além disso, mesmo se completasse, **não decidiria a questão física**: as luzes/normais
sintéticas são todas construídas no mesmo referencial numpy (`synthetic_utils.py`), sem o
acoplamento POV-Ray `zFoc`↔geometria-de-câmera que origina o desacordo. CONV-1 **permanece
com o sinal-físico em aberto** (refutado apenas no integrador via `-normals`, Task 9-ext);
decisão requer E2E com dados reais (com hints) e ground-truth de altura real.

**Atualização Task 12-ext (workaround MF-14, 2026-06-05):** A variante
`test_hybrid_pipeline_end_to_end_with_workaround` (marcador sob cada `L*` contorna MF-14)
**completa a cadeia ponta a ponta** e produz baseline `affine-fit RMSE = 0.0742` (std gt
`0.9780`), `a = 1.0030` (escala **positiva** ≈ +1), `b = 3.1279`, `pearson r = 0.9971`. O
`a > 0` e `r ≈ +1` são **evidência** de que a cadeia interna mosaico→PS→integração **não
inverte z** — mas **só para luzes no mesmo frame numpy das normais** (luzes sintéticas). A
metade física (frame real `lights.npy` POV-Ray) **continua aberta**: este número não a
decide. Verbatim e interpretação em `06-test-results.md`.

**Atualização Task 9-ext (2026-06-04):** A rampa reenviada pelo caminho `-normals`
(`test_ramp_normals_decide_convention`, PASSED) **decide** o sinal: a cadeia
numpy→FNI→C→FNI→numpy preserva o sinal de ponta a ponta — `z = +ax*x + ay*y` vence com
RMSE `0.000052`, ~4 ordens de grandeza abaixo de `-ax*x-ay*y` (`1.025489`). Não há inversão
global de `Z` no integrador. Tabela completa em `06-test-results.md`. (A via `-slopes`
abaixo continua bloqueada por INT-08.)

**Atualização Task 9 (2026-06-04):** O teste da rampa **não conseguiu decidir** o sinal de `Z`.
O caminho `integrate_slopes_to_height` com mapa de 2 canais aborta o binário antes da
integração: o topo aceita 2/3 canais (`gus_integrate_recursive.c:503`) mas o solver iterativo
exige exatamente 3 (`lib-src/pst_integrate_iterative.c:47: ** slope map {G} must have 3 channels`).
A tabela de candidatos de convenção nunca chegou a ser impressa. CONV-1 segue **suspeita não
refutada** (a leitura estática — sinal `-nx/nz` correto internamente, constante de integração
arbitrária — permanece). Decisão de sinal fica pendente de uma via que não crashe (caminho
`-normals` com rampa assimétrica, ou `-slopes` com 3º canal de peso). Detalhes em
`06-test-results.md`.

**Descrição:** A direção de crescimento de `z` **não é fixada nem verificada** em nenhum elo
da cadeia. No dataset gerado por POV-Ray, `zFoc` cresce do fundo da cena (`zScene_min`) para
a frente (`data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:98-100`:
`zFoc = zFoc_min + kfoc/(nfoc-1)*(zFoc_max - zFoc_min)`; `zScene_min=0.0`, `zScene_max=2.0`
em `make_images.py:62-63`), de modo que **`z_foc` maior = plano focal mais distante da
câmera** (mais ao fundo). A inferência direcional ("z_foc maior = mais longe da câmera")
apoia-se em `cam_dir=<0,0,1>` com a câmera no lado −z (convenção bcamlight), cena z∈[0,2] —
não mostrado nas linhas citadas, mas verificado no gerador (`main.pov:292`). O multifocus
grava `zMos[i,j] = z_foc[k_sel]` (`mosaic.py:57,72`) preservando essa direção física (alturas
em unidades de `z_foc`, "mais fundo = maior"). Já a altura integrada `Z` do solver C resulta
da integração do campo de slope `dZdX=-nx/nz` (`pst_basic.c:49-50`), cuja constante de
integração é **arbitrária por componente conexa** (`gus_integrate_recursive.c:138-143`) e
cujo sinal de crescimento depende da convenção de normal do PS — i.e. `Z` cresce na direção
`+nz` (em direção à câmera, se `nz>0` aponta para fora da superfície em direção ao
observador). Assim, **`zMos` (mais-fundo = maior) e `Z` (mais-perto-da-câmera = maior, por
convenção de normal) podem ter sinais opostos**: quando o híbrido usa
`zMos_with_confidence.fni` como *hints* no mesmo sistema (INT-04), um eventual desacordo de
sinal puxa a superfície na direção errada (não só desloca a constante). A direção é decidida
apenas empiricamente pela geometria do dataset, sem nenhuma asserção, documentação ou
conversão de sinal no código. Como o erro de **unidade** (z_foc físico vs altura-por-pixel)
já é INT-04, este achado isola o eixo **sinal/direção**, que a leitura estática não resolve.

**Evidência:** `zFoc` crescente p/ frente da cena (`data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:98-100`; números de linha variam entre as cópias de make_images.py; o lights.npy do dataset híbrido reproduz byte-a-byte as tuplas `light_dirs` deste gerador — proveniência por igualdade de valores); `zMos[i,j] = zFoc[...]` (`mosaic.py:57,72`); slope `-nx/nz`,`-ny/nz` (`pst_basic.c:49-50`); constante de integração arbitrária (`gus_integrate_recursive.c:138-143`); hints somados a `Z` no mesmo sistema (`pst_integrate.c:308-323` via `hybrid/main.py:231`). Nenhuma asserção de sinal em todo o caminho. O teste de rampa de slope constante conhecido (Task 9) recupera o sinal de `Z` e decide se ele concorda com a direção de `zMos`.

**Sugestão de correção:** documentar e fixar a convenção de direção de `z` ponta a ponta;
inserir asserção/conversão de sinal entre `zMos` (hints) e a altura integrada antes de
combiná-los. NÃO aplicar.

---

## CONV-2: Convenção do eixo Y das luzes (lights.npy) vs. imagem numpy não é documentada nem reconciliada

- **Localização:** `lights.npy` (gerado fora do pacote — tuplas POV-Ray em `data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:44-58`) → `np.load` (`main_wps.py:119`) → `reconcile_lights` (`main_wps.py`) → `wps.py:165,194` → `normal_map.npy` → `pst_basic.c:49-50` (`dZdY=-ny/nz`) → grade de integração C
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** **mitigado (b2c1eeb, 2026-06-06)** — reconciliação explícita via `flip_lights_y`; decisão física continua exigindo dado real (procedimento documentado no config). A função `reconcile_lights(light_sources, flip_y)` em `main_wps.py` nega a coluna y quando `photometric.flip_lights_y: true` (POV-Ray y-up → numpy y-down). Default `false` = identidade, não muta a entrada. O eixo-y do integrador foi REFUTADO por teste (`test_ramp_normals_decide_convention`, PASSED; vencedor `z = +ax*x + ay*y`, o flip-de-y `+ax*x-ay*y` rejeitado com RMSE `0.380857` vs `0.000052`). O referencial-y do `lights.npy` real (y-up POV-Ray) vs. eixos da imagem permanece em aberto — só decidível com `lights.npy` real + ground-truth de altura real.

**Atualização Task 12 (2026-06-04):** A cadeia E2E (`tests/test_e2e_hybrid.py`) **NÃO**
fechou esta metade — e, por construção, **não poderia**. (1) O pipeline aborta antes do PS
(MF-14), então nenhuma altura foi produzida. (2) Decisivo, mesmo que rodasse: as luzes
sintéticas vêm de `ring_lights(...)` construídas **no MESMO referencial numpy das normais**
(documentado no cabeçalho de `synthetic_utils.py`: "luzes no mesmo frame das normais").
Logo `L·N` é consistente por construção e o E2E sintético **não reproduz** a questão real —
o `lights.npy` real é gerado por POV-Ray (y-up) enquanto a imagem é indexada `[linha=y p/
baixo]`. **Recomendação:** a metade real de CONV-2 só pode ser decidida com `lights.npy`
real + ground-truth de altura real (comparando orientação do mapa integrado com/sem flip do
eixo-y das luzes); não há atalho sintético neste frame. CONV-2 (metade real) **permanece em
aberto**.

**Atualização Task 12-ext (workaround MF-14, 2026-06-05):** A variante
`test_hybrid_pipeline_end_to_end_with_workaround` agora **completa** a cadeia (contornando
MF-14) e dá `pearson r = 0.9971`, `a = 1.0030` (positivo). Isso confirma que a cadeia
mosaico→PS→integração **preserva a orientação para luzes no mesmo frame numpy das normais** —
uma linha de evidência sobre o elo interno, **mas não fecha a metade real**: as luzes
sintéticas continuam no frame numpy, não no frame POV-Ray real. CONV-2 (metade real)
**continua em aberto** — só decidível com `lights.npy` real + gt de altura real. Verbatim em
`06-test-results.md`.

**Atualização Task 9-ext (2026-06-04):** A rampa via `-normals`
(`test_ramp_normals_decide_convention`, PASSED) decide o eixo-y **do integrador**: o `y` do
numpy (para baixo) é preservado na ida-e-volta numpy→FNI→C→FNI→numpy. O candidato com y
invertido (`+ax*x-ay*y`) é fortemente rejeitado (RMSE `0.380857` vs `0.000052` do vencedor);
eixos não estão trocados (`+ay*x+ax*y` rejeitado, `0.403960`). Qualquer convenção interna
y-up do C **cancela** no round-trip. **Mas isto NÃO toca no referencial das luzes:** a
reconciliação y-up(`lights.npy`)↔y-down(imagem) durante o PS é um elo distinto, não exercido
aqui — segue em aberto para a Task 12. Tabela em `06-test-results.md`.

**Atualização Task 9 (2026-06-04):** Idem CONV-1 — o teste da rampa (`-slopes`, 2 canais)
abortou o binário em `pst_integrate_iterative.c:47` (slope map exige 3 canais) antes de medir
o sinal de `dZ/dY`. A reconciliação y-up(luzes)↔y-down(numpy) **continua não verificada
empiricamente** por essa via; ver `06-test-results.md`.

**Descrição:** O modelo Lambertiano resolve `I = ρ·(L·n̂)`, logo as normais estimadas vivem
**no mesmo frame de coordenadas das luzes** `lights.npy` (por construção do `lstsq` em
`wps.py:165`; PS-02). As `lights.npy` reais são **exatamente** as tuplas da lista `light_dirs`
declarada no gerador POV-Ray (`data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:44-58`;
o escalar `light_dir` em `make_images.py:169` é a declaração POV da cena por frame, distinto
da lista `light_dirs`) — verificado: as 12 linhas de `lights.npy` do dataset híbrido
reproduzem byte-a-byte as tuplas `light_dirs` deste gerador — proveniência por igualdade de
valores, na mesma ordem L00..L11, com **z>0** (luz acima do plano, na direção da câmera).
POV-Ray usa sistema
**left-handed com +y para cima**. Já a imagem que alimenta o PS é um array numpy
`[linha, coluna]` com **linha 0 = topo** (y crescendo **para baixo**, cf. writer/reader FNI
`image_io.py:171,239`). O passo normal→slope no C faz `dZdY = -ny/nz` (`pst_basic.c:49-50`) e
integra `Z` sobre a **grade de pixels** cujo eixo de linha é o y-para-baixo do numpy. **Não
há, em ponto algum do código ou da documentação, uma reconciliação explícita entre o "y para
cima" das luzes e o "y para baixo" das linhas da imagem.** Se as duas convenções diferirem em
sinal (o que é o caso típico entre POV-Ray y-up e numpy row-down), as normais ficam
**espelhadas em Y** e a superfície integrada sai invertida na vertical (ou com o sinal de
`dZ/dY` trocado). O integrador C oferece o escape `scale {sx} {sy}` com `sy` negativo
(`gus_integrate_recursive.c:214-216`) justamente para essa reconciliação, mas o Python passa
`slopes_scale` default `(1,1)` e **nunca** o configura (`integrate.py:53,104-105`;
`hybrid/main.py` não seta `slopes_scale`), assumindo implicitamente que luzes e linhas têm a
mesma orientação de Y — premissa **não verificada**. Procurei documentação/geração que fixe a
convenção: **nenhuma** existe no pacote (`src/`), nos configs ou nos specs (grep por
`y-up`/`y-down`/`handedness`/`left-handed`/`frame de luz` retorna apenas o próprio plano de
auditoria). Registro a convenção como **NÃO DOCUMENTADA** e o risco resultante como sinal de
`dZ/dY` indeterminado estaticamente.

**Evidência:** `lights.npy` = tuplas da lista `light_dirs` do gerador POV-Ray
(`data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:44-58`;
números de linha variam entre as cópias de make_images.py; o lights.npy do dataset híbrido
reproduz byte-a-byte as tuplas `light_dirs` deste gerador — proveniência por igualdade de
valores; coluna y = `[-0.554, -0.794, ...]`); POV-Ray = left-handed +y-up. Array da imagem
com linha 0 = topo (`image_io.py:171` escreve `y=0` primeiro; reader mapeia
`image_array[y,x]`, `image_io.py:239`). `dZdY=-ny/nz` (`pst_basic.c:49-50`).
`slopes_scale=(1,1)` default (`integrate.py:53`), nunca alterado pelo híbrido. Nenhum decode
de convenção de Y em `src/` (grep vazio). O teste de rampa (slope `dZ/dY` constante
conhecido, Task 9) decide o sinal físico de Y e se o pipeline o trata corretamente.

**Sugestão de correção:** documentar a convenção de eixo das `lights.npy` (y-up POV-Ray) e,
no boundary numpy↔C, ou negar a coluna y das luzes, ou passar `scale 1 -1` ao solver, de modo
explícito e testado. NÃO aplicar.

---

## CONV-4: `slopes_scale` default `(1,1)` nunca é configurado — gradientes adimensionais e hints em z físico ficam incomensuráveis

- **Localização:** `IntegrateRecursiveConfig.slopes_scale=(1.0,1.0)` (`hybrid/integrate.py:53`); `hybrid/main.py:213-221` (construção do config, **sem** `slopes_scale`); consumo C `gus_integrate_recursive.c:449-450`
- **Tipo:** conceitual
- **Severidade:** alto
- **Status:** suspeita (decide: teste rampa/bump com hints, Task 9/12)

**Descrição:** Consolidação de costura sobre INT-04/INT-05. Os slopes que o C integra são
**adimensionais** (`dZdX=-nx/nz`, rise por célula unitária; `pst_basic.c:49-50`), produzindo
`Z` em "altura-por-pixel". Os *hints* injetados pelo híbrido vêm de `zMos_with_confidence.fni`
em **unidades físicas de `z_foc`** (passo de ~10 entre planos; `configs/hb_experiment.yaml:39`),
uma escala completamente diferente — INT-04. O único mecanismo que reconciliaria as duas
escalas no boundary é o fator `scale {sx}{sy}` aplicado aos gradientes
(`gus_integrate_recursive.c:449-450`, `float_image_rescale_samples`), exposto em Python como
`slopes_scale`. Porém: (1) o `IntegrateRecursiveConfig` construído em `hybrid/main.py:213-221`
**não passa** `slopes_scale`, ficando no default `(1.0,1.0)` (`integrate.py:53`); (2) o
comando só anexa `scale` se `slopes_scale != (1,1)` (`integrate.py:104-105`), logo no fluxo
híbrido o `scale` **nunca é emitido**; e (3) o `-hints` também é montado **sem** `scale`,
fixando `hints_scale=1.0` (`integrate.py:109,114`). Resultado: o solver soma, no mesmo sistema
de mínimos quadrados, slopes adimensionais (×1) com um alvo de hints em escala de `z_foc`
(×1), grandezas **incomensuráveis**, e o `hints_weight=0.1` pondera duas unidades distintas
sem significado físico (INT-04). O `slopes_scale` existe como ponto de conserto, mas o
pipeline híbrido nunca o usa para tornar as grandezas comensuráveis.

**Evidência:** `slopes_scale: tuple = (1.0,1.0)` (`integrate.py:53`); ausência de
`slopes_scale=` em `IntegrateRecursiveConfig(...)` (`hybrid/main.py:213-221`); `if
config.slopes_scale != (1.0,1.0)` (`integrate.py:104`) ⇒ `scale` não anexado; `-hints` sem
scale (`integrate.py:109,114`) ⇒ `hints_scale=1.0` (`gus_integrate_recursive.c:746,825`).
Cross-ref INT-04 (mistura de unidades) e INT-05 (deslocamento célula→vértice). O teste de
rampa com hints (Task 9/12) mede o viés de escala resultante.

**Sugestão de correção:** computar e passar `slopes_scale`/`hints scale` que levem
`z_foc`→altura-por-pixel (fator = passo de `z_foc` em px), tornando hints e gradientes
comensuráveis; ou converter `zMos` para unidade de pixel antes de gravar os hints. NÃO aplicar.

**Correção aplicada:** `aa772ed` (2026-06-05) — `hybrid.integration.pixel_size`
(tamanho lateral de 1 pixel em unidades de `z_foc`) é o ponto de costura escolhido: o
helper `build_integration_config` (`hybrid/main.py`) deriva `slopes_scale = (pixel_size,
pixel_size)` e o `scale` passa a ser emitido ao binário (o mecanismo `integrate.py:104-105`
já existia, só nunca era acionado). `Z` sai em unidades físicas de `z_foc`, comensurável
com os hints sem `hints scale`. Direção inversa (converter `zMos` para pixel) foi
descartada: o height map físico é mais interpretável e o `zMos` permanece em z_foc.
Warning quando `use_hints=True` sem `pixel_size`. Evidência executável e detalhes em
INT-04 (`03-integration.md`); 10 testes em `tests/test_integration_units.py`. Nota: o
viés medido com escala default é exatamente `a = pixel_size` (2.5003 para pixel 2.5),
confirmando a previsão; CONV-4 passa de suspeita a **confirmado por execução e corrigido**.

---

## CONV-5: Cadeia radiométrica quebra a linearidade exigida pelo PS em três pontos distintos

- **Localização:** `sVal.png` (uint8) → médias por-zf (`hybrid/main.py:107-111`) → foco (`multifocus/main.py:96`) ; mosaico (`hybrid/main.py:154-162`) → grayscale (`main_wps.py:115`) → `lstsq` (`wps.py:165`)
- **Tipo:** conceitual
- **Severidade:** médio
- **Status:** corrigido/fechado por composição (MF-12 + PS-07 + PS-08)
- **Resolução parcial (composição, 45cbf57):** 2 de 3 pontos fechados; gamma (PS-08) pendente. Médias float em memória (MF-12) fechou o ponto 3 (stretch per-plano IO-05 saiu do caminho de dados); mosaicos float em memória (PS-07, este commit) fechou os pontos 5-6 (clip+uint8 e leitura PNG saíram do caminho de dados). Nenhum dado científico do pipeline híbrido passa por `save_image` antes de ser consumido. Ponto 1 (gamma PS-08) permanece pendente — a entrada `sVal.png` pode ser sRGB e o código não lineariza.
- **Fechado por composição (`7a4c8fc`, 2026-06-06):** 3 de 3 pontos fechados. PS-08 (este commit) fecha o ponto remanescente: `linearize_intensities` com config `photometric.parameters.gamma` (default 1.0 = identidade). Sequência completa: MF-12 (bd23651) médias float sem quantização + IO-05 saiu do caminho; PS-07 (45cbf57) mosaicos float sem clip+uint8 + leitura PNG saiu do caminho; PS-08 (7a4c8fc) gamma documentado e opcionalmente decodificado. **Status: corrigido/fechado por composição.**

**Descrição:** Consolidação da cadeia radiométrica completa numa única sequência, com cada
transformação e seu efeito sobre a linearidade/comparabilidade exigida pelo modelo
Lambertiano (`I=ρ·(L·n̂)`, intensidade **linear** em radiância):

1. **`sVal.png` (uint8 0-255)** — entrada. **Premissa não verificada:** se for sRGB/gamma,
   já é não-linear na origem (PS-08); o código nunca lineariza.
2. **Média por plano focal** `calculate_avarage_of_images` em float32, **re-quantizada a
   uint8** no retorno (`utils.py:106-107`) — perde bits efetivos da média (MF-12). *Linear,
   mas com perda de precisão.*
3. **`save_image(..., "average_{zf}.png")` com `normalize=True`** (`hybrid/main.py:111`) —
   **min-max stretch por-plano** (`image_io.py:138`): cada plano focal é esticado
   independentemente a [0,255], **destruindo a comparabilidade de intensidade entre planos**
   antes da medida de foco (IO-05). *Quebra de comparabilidade inter-plano — pode mover o
   argmax de foco.*
4. **Medida de foco** (`multifocus/main.py:96` relê esses PNGs) — opera sobre intensidades já
   esticadas por-plano; o `iSel` herda o viés do passo 3.
5. **Mosaico all-in-focus** por luz: `save_image(..., "sMos.png", normalize=False)`
   (`hybrid/main.py:160`) — **preserva** a relação cross-light (comentário explícito,
   `:157-159`), mas ainda **`clip(0,255)` + uint8** (`image_io.py:140`): satura
   silenciosamente valores >255 e quantiza a 8 bits (PS-07). *Linear na faixa, mas saturada e
   quantizada.*
6. **`sMos.fni` float existe** (`hybrid/main.py:161-162`) mas **é ignorado** pelo PS, que lê
   o PNG uint8 (`main_wps.py:109`, seleção `sMos.png` em `hybrid/main.py:181`) — PS-07.
7. **Grayscale** `convert_to_grayscale` (`main_wps.py:115`) — Rec.601 BGR (linear nos canais),
   ok; divergência de coeficientes só no caminho `rps`/`ps_utils` (PS-09).
8. **`lstsq`** (`wps.py:165`) — ajusta modelo **linear** a dados que podem ser não-lineares
   (passo 1) e foram saturados/quantizados (passo 5).

**Onde a cadeia quebra:** comparabilidade **inter-plano** no passo 3 (IO-05, afeta a
profundidade); linearidade **absoluta** no passo 1 (PS-08, gamma não tratada) e no passo 5
(saturação por `clip`, PS-07); precisão nos passos 2/5 (quantização uint8, MF-12/PS-07).
Nenhum estágio reintroduz linearidade. Este CONV-5 não adiciona um defeito novo de código —
costura os achados de radiometria numa sequência única para evidenciar que a cadeia **não
preserva** a linearidade ponta a ponta que tanto o foco (comparação inter-plano) quanto o PS
(modelo Lambertiano) pressupõem.

**Evidência:** passos citados acima com `arquivo:linha`; cross-ref MF-12, IO-05, PS-07, PS-08,
PS-09. Decidido em magnitude pelos testes sintéticos (float vs uint8, com/sem gamma) das
Tasks 10-12.

**Sugestão de correção:** alimentar foco e PS com float in-memory (sem round-trip PNG),
desativar `normalize` nas médias por-zf, remover o `clip(0,255)`, e documentar/impor a
premissa de linearidade radiométrica da aquisição. NÃO aplicar.

---

## CONV-6: Pareamento luz↔mosaico garantido só por contagem; ordens e shape entre estágios não são verificados por construção

- **Localização:** `natsorted(sMos_path_list)` (`hybrid/main.py:177-184`) vs `lights.npy` (`main_wps.py:119-127`); `sorted(zf_directories)` (`hybrid/main.py:90`) vs `z_foc` (`mosaic.py:57`); `zMos_with_confidence.fni` `(H,W,2)` vs `-hints` `(H+1,W+1)` (`gus_integrate_recursive.c:519`)
- **Tipo:** implementação
- **Severidade:** alto
- **Status:** suspeita, **reforçada por Task 12** (a guarda de contagem pegou o caso degenerado 0 vs 6; N-trocados não exercitado)

**Atualização Task 12 (2026-06-04):** O E2E (`tests/test_e2e_hybrid.py`, XFAIL strict)
exercitou a guarda de contagem (1) por um caminho inesperado: MF-14 (detecção de luzes
vazia) fez `sMos_path_list` ficar **vazio**, então `len(images)=0 != 6` e o PS abortou com
mensagem clara (`main_wps.py:123`) — a defesa por contagem **funciona para o caso
degenerado (0 mosaicos)**. Mas o cenário central de CONV-6 — N mosaicos pareados **na ordem
errada** com as N linhas de `lights.npy` (contagem bate, identidade não) — **não foi
exercitado** (nunca houve N mosaicos). CONV-6 permanece **suspeita**; um teste futuro
precisa de N luzes rotuladas e mosaicos embaralhados para decidir o pareamento por
identidade.

**Descrição:** Consolidação dos contratos de arquivo entre estágios. Três pareamentos
posicionais, nenhum verificado por identidade:

(1) **Luz ↔ mosaico (pareamento PS):** o híbrido coleta os mosaicos por luz com
`natsorted([... "sMos.png" ... dir != "average"])` (`hybrid/main.py:177-184`) e o PS os pareia
**posicionalmente** com as **linhas** de `lights.npy` (`main_wps.py:135-136`). A **única**
garantia é a checagem de **contagem** `len(images) != light_sources.shape[0]`
(`main_wps.py:122-127`) — **não há** verificação de que a i-ésima `sMos.png` (da pasta `L00n`)
corresponda à **i-ésima linha** de `lights.npy`. O pareamento correto depende de que (a) a
ordem `natsorted` das pastas `L*` coincida com (b) a ordem das linhas de `lights.npy`. Nos
dados reais as pastas são `L000..L011` (zero-padded) e as linhas de `lights.npy` estão na
ordem de declaração L00..L11 do gerador (`make_images.py:44-58`) — coincidem **por
convenção do dataset**, não por código. Se um dataset usar rótulos de luz não-zero-padded
(`L0..L11`), `natsorted` ainda ordena naturalmente, mas qualquer luz **ausente/extra** ou
reordenação de `lights.npy` desloca **todas** as normais sem disparar erro (a contagem ainda
bate). É um pareamento frágil por posição, não por chave `L<n>`→linha `n`.

(2) **Ordem zf ↔ z_foc:** `zf_directories = sorted({...})` (`hybrid/main.py:90`) é
**lexicográfico**; `z_foc` é posicional no YAML; o mapa índice→profundidade depende de as duas
ordens coincidirem — é exatamente **MF-02** (com rótulos `zf1..zf12`, `sorted` quebra a ordem
natural). *Nuance verificada nesta fase:* os datasets reais nomeiam os planos como
`zf075.0000-df022.5000` (float zero-padded), caso em que `sorted` == ordem natural; o defeito
MF-02 fica **latente** para o esquema `zf1..zf12` assumido pelos configs (`z_foc` de 12
entradas). Sem guarda de comprimento em `hybrid/main.py` (MF-02/MF-13).

(3) **Shape hints célula vs vértice:** `zMos_with_confidence.fni` é grade de **células**
`(H,W,2)`, mas o C exige hints na grade de **vértices** `(H+1,W+1)` e os expande com
`float_image_expand_by_one`, deslocando-os meia-célula — é **INT-05**. O contrato de shape não
é verificado no lado Python (a docstring `(H+1,W+1)` em `integrate.py:217` sequer corresponde
ao arquivo de células realmente passado).

O achado **novo** desta fase é o pareamento (1): a ausência de verificação L→linha por chave,
contando só com a coincidência de ordenação. (2) e (3) são cross-refs (MF-02/MF-13, INT-05).

**Evidência:** `natsorted(...)` dos `sMos.png` (`hybrid/main.py:177-184`); pareamento
posicional + checagem só de contagem (`main_wps.py:122-127,135`); `sorted(zf_directories)`
lexicográfico (`hybrid/main.py:90`, MF-02); `z_foc` posicional (`mosaic.py:57`); `demand` de
tamanho de vértices para hints (`gus_integrate_recursive.c:519`) + expansão célula→vértice
(INT-05). Lights reais em ordem de declaração (`data/raw/photometric_stereo/ex19_povball-txF/2025-01-15-glelis-pov/make_images.py:44-58`), pastas `L000..L011`
zero-padded (verificado no dataset). O teste end-to-end com luzes/planos rotulados (Task 12)
detecta desalinhamentos que a contagem não pega.

**Sugestão de correção:** parear luz↔mosaico por **chave extraída do path** (`L<n>` → linha
`n` de `lights.npy`) em vez de posição; ordenar `zf_directories` por chave numérica (MF-02);
verificar shape dos hints contra `(H+1,W+1)` no lado Python (INT-05). NÃO aplicar.

**Status atualizado:** pareamento luz↔mosaico corrigido (`c226924`, 2026-06-05) — ver entrada CONV-6 no relatório principal. Sub-achados (2) zf↔z_foc e (3) shape hints permanecem abertos.

---

## (c) Verificado sem achado

- **Convenção 3 — Indexação do round-trip FNI Python↔C é consistente (sem flip vertical).**
  O writer Python escreve as linhas `y=0..ny-1` na ordem (`image_io.py:171`), `y=0` = linha de
  topo do numpy, cada uma com `x=0..nx-1` (`image_io.py:173-178`); o reader reconstrói
  `image_array[y,x]=val` (`image_io.py:239,241`). O FNI escrito **pelo C** lista `x y val` em
  row-major com `y=0` primeiro (verificado em 03-integration via `teste_*-end-Z.fni`), e o C
  indexa `float_image_get_sample(A,c,x,y)` consistentemente. Logo o round-trip
  Python→C→Python **preserva o array** sem espelhamento (consolida IO "Round-trip de
  indexação" e INT "Indexação `float_image`"). O que permanece aberto — o **sinal físico** de
  `dZ/dY` (se a linha 0 é topo ou base da cena e se isso casa com o y das luzes) — **não** é
  questão de indexação FNI, e sim de CONV-1/CONV-2, decidida pelo teste de rampa (Task 9). A
  **indexação** está consistente; o **sinal** é decidido-por-teste.

- **Convenção 6 — A checagem de contagem imagem↔luz existe e impede o desalinhamento mais
  grosseiro.** `if len(images) != light_sources.shape[0]: raise ValueError`
  (`main_wps.py:122-127`) barra contagens divergentes (consolida o "Pareamento imagem↔luz tem
  checagem de contagem" de 02-photometric). A fragilidade remanescente — pareamento por
  posição e não por chave `L<n>` — está registrada em CONV-6, não aqui.

- **Convenção 2 — As normais do wps vivem, por construção, no mesmo frame das luzes.** O
  `lstsq(selected_lights, selected_values)` (`wps.py:165`) resolve `L·m=I` no frame de `L`, e
  a normalização `m/‖m‖` (`wps.py:194`) não muda o frame; logo não há transformação de
  coordenadas espúria **dentro** do PS (consolida o "Convenção de saída do RPS e do WPS é a
  mesma" de 02-photometric). O risco de frame é **externo**: a relação entre o eixo Y das
  luzes e o eixo de linha da imagem (CONV-2), não uma inconsistência interna do solver.

- **Convenção 1/4 — Sinais da conversão normal→slope no C estão corretos.** `dZdX=-nx/nz`,
  `dZdY=-ny/nz` (`pst_basic.c:49-50`) é a relação padrão entre normal unitária e gradiente, com
  piso `nzmin=hypot(nx,ny)/maxSlope` evitando divisão por zero (consolida o "`-normals`
  converte normal→slope com a convenção correta de sinais" de 03-integration). A corretude do
  **sinal interno** está verificada; o que resta é o **acoplamento de escala** (CONV-4,
  adimensional vs z físico) e o **sinal de Y de entrada** (CONV-2), ambos externos a esta
  conversão.
