---
date: 2026-06-09
type: bug-audit
scope: "full-repo (foco: matemática + integração)"
method: "multi-agent workflow, 12 finders, verificação adversarial 2-de-3"
spec: docs/superpowers/specs/2026-06-09-auditoria-bugs-design.md
---

# Relatório de Auditoria de Bugs — hybrid-stereo-method

## Sumário executivo

A auditoria confirmou **43 bugs** (cada um validado por pelo menos 2 das 3 lentes de verificação adversarial — matemática, literal e impacto): **4 críticos**, **6 altos**, **15 médios** e **18 baixos**. Outros **6 achados foram refutados** na verificação (Apêndice A). Não houve achados com verificação incompleta.

Os achados mais importantes:

**1. Destruição da radiometria na fronteira multifocus→photometric (BUG-001, BUG-002, BUG-004 — críticos).** O defeito mais grave do repositório é um único problema de projeto que aparece em três pontos da cadeia: `save_image` (image_io.py:129) aplica `cv2.normalize(..., NORM_MINMAX)` **independentemente por imagem** antes de gravar qualquer PNG. No pipeline híbrido, os mosaicos all-in-focus `sMos.png` de cada direção de luz são salvos assim e depois **relidos como entrada do photometric stereo**. Como cada imagem de luz recebe ganho e offset distintos, as razões de intensidade entre luzes — exatamente o sinal do qual o solver `I = L·n` estima as normais — são corrompidas. Nenhum caminho alternativo preserva a radiometria (o `sMos.fni` também é normalizado). As normais saem sistematicamente enviesadas, sem nenhum erro visível, e o viés se propaga aos slopes, à integração e ao height map final. Todas as três lentes confirmaram o defeito nas três manifestações, inclusive com verificação do fluxo real de `configs/hb_experiment.yaml`.

**2. Projeção zero-mean incompatível com o termo de hints no integrador C (BUG-006 — alto).** `pst_integrate_iterative` fixa `szero=TRUE`, fazendo o Gauss-Seidel subtrair a média de todas as variáveis a cada iteração. Isso é correto para o problema puro de gradientes (gauge livre), mas o sistema do pipeline inclui termos de hints com alturas **absolutas** do zMos. O ponto fixo da iteração não é o minimizador do funcional: o nível absoluto que os hints deveriam ancorar é descartado e a forma da superfície ganha um viés espacial proporcional à média dos hints — confirmado por derivação independente e simulação numérica. O propósito declarado do recurso de hints (commit 410719d) é anulado na configuração padrão do experimento híbrido.

**3. Sinal de dZ/dY invertido na fronteira Python↔C (BUG-007, BUG-008 — altos).** O FNI de normais é gravado top-down (convenção OpenCV), mas a biblioteca C de integração assume origem no canto inferior esquerdo com Y para cima, e o parâmetro de correção `G_SY` negativo nunca é emitido (`slopes_scale` fica no default `(1,1)` e não é exposto no YAML). Testes empíricos com o dataset configurado (melon24, com ground truth) provaram que as normais/luzes seguem convenção Y-para-cima: o integrador recebe dZ/dX correto e dZ/dY com sinal trocado, produzindo silenciosamente uma superfície distorcida — não um mero espelhamento global, pois X permanece consistente.

**4. Ponteiro de hints não inicializado no binário C (BUG-005 — alto).** Em `gus_integrate_recursive.c:461`, `float_image_t *H;` só recebe valor quando `-hints` é passado. Sem a opção (caso `use_hints: False` ou uso standalone), o programa testa e libera um ponteiro com lixo de pilha — comportamento indefinido que pode crashar ou contaminar o sistema linear com um "mapa de hints" de memória aleatória.

**5. Opção `wavelet` do indicador de foco inutilizável e janela de Hann com frequência invertida (BUG-003 — crítico; BUG-009 — alto).** O indicador wavelet extrai a banda errada da decomposição e devolve um mapa com 1/4 da resolução, estourando o mosaic com IndexError; e a interpolação quadrática usa `a = π·0.5·(m+1)` em vez de `a = π/(0.5·(m+1))`, gerando pesos pseudo-aleatórios (o ponto mais próximo do índice fuzzy recebe peso ~0,02 e pontos distantes ~0,98 — verificado numericamente). Ambos são latentes nas configs atuais, mas são opções anunciadas nos YAMLs que produzem resultado quebrado ou silenciosamente errado quando ativadas.

Transversalmente, a auditoria também encontrou um padrão recorrente de **descarte de informação de confiança** nas fronteiras entre estágios (BUG-014, BUG-015), **configurações ignoradas ou incompatíveis** entre YAMLs e código (BUG-010, BUG-021, BUG-026, BUG-030) e **falhas silenciosas por validação ausente** (BUG-020, BUG-016).

## Bugs confirmados

### Críticos

#### BUG-001 — Entrada do photometric stereo usa sMos.png re-normalizado min-max por imagem (radiometria destruída)

**Local:** src/hybrid_stereo_method/hybrid/main.py:148 · **Categoria:** integração · **Encontrado por:** Hybrid; Fluxo de configuração; Contratos de dados

O pipeline hybrid coleta os mosaicos por direção de luz via PNG: `sMos_path_list` aponta para os sMos.png salvos por save_image (image_io.py:129), que aplica `cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)` INDEPENDENTEMENTE em cada imagem. O photometric stereo (main_wps.py:107-115) relê esses PNGs e resolve I = L·n assumindo que as intensidades das diferentes direções de luz são radiometricamente comparáveis. A re-normalização min-max por imagem multiplica cada imagem de luz por um ganho diferente (255/(max-min)) e soma um offset, distorcendo as razões de intensidade entre luzes — exatamente a informação da qual as normais são estimadas. Nenhum caminho preserva a radiometria: o sMos.fni também é normalizado (multifocus/main.py:153).

```python
parameters["sMos_path_list"] = natsorted(
        [file for file in output_files if "sMos.png" in file and "av" not in file]
    )  # ... e em image_io.py:129: img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
```

**Impacto:** As normais estimadas pelo photometric stereo ficam sistematicamente erradas (viés dependente do range de cada imagem de luz), sem nenhum erro visível; o erro se propaga para o slope map e para o height map final.

**Sugestão de correção:** Criar uma fronteira sem perda: salvar sMos como .npy (ou FNI sem normalize) com os valores float 0-255 do mosaico, e fazer main_wps consumir esses arquivos; reservar save_image apenas para visualização.

**Verificação:**
- Matemática: confirmado — transformação afim Î_i = a_i·I_i + b_i com ganho/offset distintos por luz muda a direção de n̂ = (LᵀL)⁻¹LᵀÎ; o modelo de Woodham exige comparabilidade radiométrica entre luzes.
- Literal: confirmado — cadeia hybrid/main.py:147-149 → image_io.py:129 → main_wps.py:107-108 → wps.py:145-203 verificada linha a linha, sem qualquer compensação de ganho.
- Impacto: confirmado — é o fluxo principal de configs/hb_experiment.yaml (type: hybrid); a rota in-memory está comentada e o sMos.fni também é normalizado; o viés propaga via normal_map.npy até o height map.

#### BUG-002 — sMos.png re-normalizado min-max por imagem destrói a radiometria entre direções de luz antes do photometric stereo

**Local:** src/hybrid_stereo_method/infrastructure/io/image_io.py:129 · **Categoria:** integração · **Encontrado por:** Infrastructure; Convenções geométricas

save_image SEMPRE aplica cv2.normalize(..., 0, 255, NORM_MINMAX) por imagem antes de gravar o PNG. No modo híbrido, os mosaicos sMos de cada direção de luz L* são salvos via save_image (multifocus/main.py:151) e depois relidos como ENTRADA do photometric stereo (hybrid/main.py:147-149 coleta os 'sMos.png'; main_wps.py:107-108 os relê). O solver fotométrico resolve I = L·n por pixel (wps.py:162) e assume que as intensidades das f imagens são radiometricamente comparáveis (mesmo ganho/offset). A normalização min-max independente por imagem aplica a cada direção de luz um ganho a = 255/(max-min) e um offset b = -min·a desconhecidos e diferentes, corrompendo a razão de intensidades entre luzes. Nenhum caminho preserva a radiometria: o sMos.fni também é gravado normalizado (multifocus/main.py:153).

```python
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
```

**Impacto:** Todas as normais estimadas pelo photometric stereo ficam sistematicamente enviesadas (o lstsq de wps.py ajusta n para compensar ganhos/offsets espúrios por luz), sem erro visível. O erro se propaga para os slopes, para a integração e para o height_map final.

**Sugestão de correção:** Criar um caminho de dados sem re-normalização para a fronteira multifocus→photometric: gravar os sMos como .npy/.fni com valores brutos (0–255 do stack interpolado) e fazer main_wps.py consumi-los, reservando save_image apenas para visualização.

**Verificação:**
- Matemática: confirmado — o sistema L_k·g = a_k·ρ(L_k·n) + b_k com a_k, b_k distintos por luz não admite solução paralela a n; ganho uniforme seria absorvido pelo albedo, ganhos/offsets por imagem não.
- Literal: confirmado — todos os elos (image_io.py:129, multifocus/main.py:151/153, hybrid/main.py:147-149, main_wps.py:107-108, wps.py:162) verificados; sem transformação inversa em nenhum ponto.
- Impacto: confirmado — fluxo padrão do hb_experiment.yaml; neutralidade exigiria min/max idênticos em todos os mosaicos, o que não é garantido; viés propaga à integração (hybrid/main.py:167-220).

#### BUG-003 — Indicador wavelet extrai a banda errada e devolve mapa com 1/4 da resolução da imagem

**Local:** src/hybrid_stereo_method/multifocus/indicators/wavelet.py:16 · **Categoria:** matemática · **Encontrado por:** Multifocus — indicadores

Com `pywt.wavedec2(image, wavelet='haar', level=2)`, o retorno é `[cA2, (cH2,cV2,cD2), (cH1,cV1,cD1)]` — os detalhes ordenam-se do nível MAIS GROSSEIRO para o mais fino. A linha `cA, (cH, cV, cD) = coeffs[0], coeffs[1]` pega os detalhes do nível 2 (banda de frequência MÉDIA-BAIXA, ~[fs/8, fs/4]), e não os de alta frequência (`coeffs[-1]`), contrariando o próprio comentário do código ('high frequency'). Pior: esses coeficientes têm shape (ceil(H/4), ceil(W/4)) — verificado empiricamente: imagem 100x80 → coeffs[1] = (25,20) e coeffs[2] = (50,40). O mapa de foco retornado não tem correspondência pixel-a-pixel com a imagem (cada coeficiente cobre um bloco 4x4) e tem 1/4 das dimensões esperadas pelo restante do pipeline ((n,H,W) em applicator/argmax_fuzzy).

```python
cA, (cH, cV, cD) = coeffs[0], coeffs[1]

    # Calcular a magnitude das regiões wavelet de alta frequência
    high_freq_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
```

**Impacto:** Selecionar `method: 'wavelet'` no YAML produz focus_indicator_stack (n, H/4, W/4); compute_argmax_fuzzy gera iSel (H/4, W/4) e mosaic (que itera sobre H,W da image_stack) estoura com IndexError em `iSel[i, j]` — a opção 'wavelet' anunciada nos configs ('Options: fourier, laplacian, wavelet') é inutilizável. Mesmo que o shape fosse conciliado (upsampling), a banda usada é a mais grosseira da decomposição, medindo desfoque em escala de bloco 4x4 e não a nitidez fina que caracteriza foco — argmax sistematicamente menos seletivo.

**Sugestão de correção:** Usar o nível mais fino e reconstituir a resolução: ex. `cH, cV, cD = coeffs[-1]` (detalhes do nível 1, H/2 x W/2) ou somar as magnitudes de todos os níveis, e redimensionar o mapa de volta para (H, W) com `cv2.resize(..., (W, H))` antes de retornar, garantindo correspondência pixel-a-pixel com a image_stack.

**Verificação:**
- Matemática: confirmado — convenção do PyWavelets verificada empiricamente (coeffs[1] = nível 2, (25,20) para 100x80); sem resize no applicator, IndexError garantido no mosaic.
- Literal: confirmado — código, configs e fluxo conferidos; opção 'wavelet' anunciada nos YAMLs é de fato inutilizável; sem tratamento posterior.
- Impacto: refutado — nenhuma config existente usa 'wavelet' (defeito latente); ao ser ativado, a falha seria crash imediato e ruidoso, não corrupção silenciosa.

#### BUG-004 — WPS consome sMos.png re-normalizados min-max por imagem, quebrando o modelo fotométrico

**Local:** src/hybrid_stereo_method/photometric/main_wps.py:108 · **Categoria:** integração · **Encontrado por:** Photometric — WPS e utils

No modo hybrid, as imagens de entrada do solver WPS são os sMos.png gerados pelo multifocus (multifocus/main.py:151), que foram salvos por save_image com cv2.normalize(... NORM_MINMAX, 0, 255) POR IMAGEM (image_io.py:129). A normalização min-max é uma transformação AFIM independente por direção de luz (subtrai o mínimo e re-escala), destruindo tanto a razão de intensidades entre luzes quanto a linearidade I = albedo*(L·n) que o photometric stereo assume. A versão FNI (sMos.fni) também é normalizada (multifocus/main.py:153 usa normalize()), então nenhum caminho preserva a radiometria do mosaico. O solver roda sem erro e produz normais enviesadas.

```python
images = read_images(parameters.get("sMos_path_list"))
```

**Impacto:** As normais de todo o pipeline hybrid são calculadas a partir de intensidades com ganho e offset arbitrários e diferentes por direção de luz; o sistema I = L·n fica inconsistente e o normal_map.npy (e, por consequência, o height_map integrado) sai sistematicamente errado, sem nenhum erro visível.

**Sugestão de correção:** No multifocus, gravar sMos com valores físicos (float, sem min-max) em .npy ou FNI não normalizado, e fazer o main_wps consumir esse arquivo no modo hybrid; se PNG for inevitável, salvar com um único fator de escala global comum a todas as direções de luz (sem subtração de mínimo).

**Verificação:**
- Matemática: confirmado — n_est ∝ L⁻¹ diag(a_k) L ρn + termo de offset ≠ ρ'n; entradas sVal.png originais eram consistentes, a distorção é introduzida exclusivamente pelo save_image.
- Literal: confirmado — main_wps.py:108 lê os PNGs; única normalização posterior é do vetor normal final, que absorve apenas escala global comum.
- Impacto: confirmado — fluxo principal de hb_experiment.yaml; offsets corrompem também o shadow_threshold e a rejeição de outliers; efeito só seria neutro com min/max idênticos entre luzes (falso em geral).

### Altos

#### BUG-005 — Ponteiro do mapa de hints {H} não inicializado quando "-hints" não é passado

**Local:** csrc/integrate_recursive/gus_integrate_recursive.c:461 · **Categoria:** outro · **Encontrado por:** C — solver

A variável local `H` é declarada sem inicialização e só recebe valor dentro de `if (o->hints_file != NULL)` (linhas 462-467). Quando o programa é executado SEM "-hints" (ex.: hybrid.integration.use_hints: False no YAML, ou uso standalone), `H` contém lixo de pilha. Esse ponteiro indefinido é passado a `tire_compute_and_write_height_map(o, G, H, Z, R)` (linha 479), que testa `if (H != NULL)` e, com lixo não-nulo, chama `float_image_get_size(H, ...)` sobre memória aleatória; depois `if (H != NULL) { float_image_free(H); }` (linha 483) faz free de ponteiro inválido. Compare com `R`, que é corretamente inicializado com NULL na linha 471. É comportamento indefinido: pode crashar, mas também pode interpretar lixo como mapa de hints e montar um sistema com termos espúrios sem nenhum erro visível.

```c
float_image_t *H; /* Input hints map. */
    if (o->hints_file != NULL)
      { fprintf(stderr, "reading the hints height map {H} ...\n");
        H = tire_read_fni_file(o->hints_file, NX_Z, NY_Z, 1, TRUE);
```

**Impacto:** Qualquer execução do binário sem a opção -hints tem comportamento indefinido: crash intermitente ou, pior, sistema linear contaminado por um "mapa de hints" de memória aleatória, produzindo altura errada silenciosamente. No pipeline hybrid só não dispara porque o YAML usa use_hints=True.

**Sugestão de correção:** Inicializar `float_image_t *H = NULL;` na declaração (linha 461), igual ao tratamento de `R` (linhas 469-471).

**Verificação:**
- Matemática: confirmado — comportamento indefinido per C11 §6.3.2.1; caminho alcançável via use_hints:False (integrate.py omite -hints); contraste com R evidencia o lapso.
- Literal: confirmado — sem ramo else; parser torna -hints opcional com hints_file=NULL; H lido nas linhas 479 e 483.
- Impacto: confirmado — use_hints tem default False em hybrid/main.py:194 e mesmo com True o código cai sem -hints se o FNI de hints faltar (apenas warning); uso standalone também é suportado.

#### BUG-006 — szero=TRUE hardcoded é matematicamente incompatível com o termo de hints (alturas absolutas)

**Local:** csrc/integrate_recursive/lib-src/pst_integrate_iterative.c:75 · **Categoria:** matemática · **Encontrado por:** C — solver

Em `pst_integrate_iterative`, `szero` é fixado em TRUE ("Should be parameter") e repassado a `pst_imgsys_solve_iterative`, que, a cada iteração de Gauss-Seidel, subtrai a média de TODAS as variáveis (pst_imgsys_solve.c:96-102). Isso é correto para o problema puro de gradientes (gauge livre), mas o sistema montado por `pst_integrate_build_system` inclui termos de hints `woo*(H[0,x,y] - Z[x,y])^2` com valores ABSOLUTOS (no pipeline, zMos físico ~15–125 com peso 0.1*confiança). Os hints fixam o nível absoluto da solução; a projeção zero-mean a cada iteração o remove. O ponto fixo da iteração composta P∘GS não é o minimizador dos mínimos quadrados: satisfaz A·z̃ = b − m·(D−L)·1, onde m ≈ média do puxão dos hints (~ (wH/cf0)·média(zMos), ordem de unidades), e (D−L)·1 varia espacialmente (linhas de hint têm soma de linha positiva; bordas têm vizinhança assimétrica). Resultado: além de perder o nível absoluto que os hints deveriam dar, a forma da superfície ganha um viés espacial proporcional à média dos hints.

```c
bool_t szero = TRUE; /* Should be parameter. 1 means adjust sum to zero, 0 let it float. */
```

**Impacto:** Com use_hints=True (configuração do hb_experiment.yaml), o solver converge para uma solução que não é o mínimo do funcional gradientes+hints: o nível absoluto de zMos é descartado e a superfície sofre distorção espacial silenciosa proporcional à média dos hints (~70 unidades físicas) ponderada pelo peso dos hints. O propósito declarado do recurso de hints (ancorar Z) é anulado.

**Sugestão de correção:** Tornar `szero` parâmetro e usar FALSE sempre que H != NULL (os hints já fixam o gauge); alternativamente, centralizar os hints (subtrair a média de H) antes de montar o sistema, documentando que apenas a forma relativa é usada.

**Verificação:**
- Matemática: confirmado — derivação independente do ponto fixo reproduz a equação do achado e simulação numérica confirma (minimizador com média ~65 vs ponto fixo com média 0; distorção de forma máx. ~12 unidades).
- Literal: confirmado — szero=TRUE repassado sem sobrescrita em todos os níveis; nenhuma re-ancoragem posterior em C ou Python; rhs dos hints usa alturas absolutas.
- Impacto: confirmado — cadeia viva da config padrão (use_hints=True, hints_weight=0.1) até o height_map.npy final; o gauge não é livre com hints, logo o efeito não é neutro nem deslocamento constante.

#### BUG-007 — Orientação do eixo Y: FNI gravado top-down alimenta integrador C que assume Y para cima, sem G_SY negativo

**Local:** src/hybrid_stereo_method/hybrid/integrate.py:59 · **Categoria:** integração · **Encontrado por:** Hybrid; Interface Python↔C; Convenções geométricas; Fluxo de configuração; Contratos de dados

convert_image_array_to_fni grava as linhas do numpy na ordem y=0..NY-1, ou seja, FNI y=0 = topo da imagem (convenção OpenCV, Y para baixo — image_io.py:160-169). A biblioteca C assume sistema de imagem com origem no canto INFERIOR esquerdo e Y para cima (pst_normal_map.h), e o próprio help do programa avisa: 'the {G_SY} parameter should be negative if the direction of the {Y} axis assumed in the map is opposite to that assumed by this program' (gus_integrate_recursive.c:213-216). As normais do fotométrico estão no sistema de lights.npy (cena, tipicamente Y para cima); o C calcula dZdY = -ny/nz (pst_basic.c:51) e acumula ao longo das linhas do FNI, que correm para BAIXO na imagem. Como slopes_scale fica no default (1.0, 1.0), nunca é emitido (integrate.py:126 só emite se != (1,1)) e não há chave no hb_experiment.yaml para configurá-lo, nenhuma correção de sinal de Y é aplicada em lugar algum do pipeline.

```python
slopes_scale: tuple[float, float] = field(default_factory=lambda: (1.0, 1.0))
```

**Impacto:** Se lights.npy segue a convenção Y-para-cima (datasets gerados pelas ferramentas Stolfi, como '2025-03-08-stQ-...'), o dZ/dY entra no sistema com sinal invertido em relação à varredura das linhas: o campo de gradiente fica inconsistente em Y e a superfície integrada sai distorcida/espelhada em relevo na direção vertical — silenciosamente, pois o solver de mínimos quadrados sempre produz uma resposta.

**Sugestão de correção:** Definir e documentar a convenção de lights.npy; se Y-para-cima, passar slopes_scale=(1.0, -1.0) (G_SY = -1) na chamada do C (e expor a chave no YAML), ou alternativamente inverter a ordem das linhas (flipud) ao gravar o FNI de normais e de hints de forma consistente.

**Verificação:**
- Matemática: confirmado — teste empírico com ground truth do melon24: corr(-nx/nz, dh/dcol)=+0.95 mas corr(-ny/nz, dh/drow)=−0.95; eixo Y das normais oposto à varredura de linhas; nenhuma correção existe no pipeline.
- Literal: confirmado — default (1,1) nunca emitido, YAML sem chave de escala, sem flip em src/; resíduo de curl no dataset Stolfi cai de 0.195 para 0.116 ao inverter ny.
- Impacto: confirmado — Step 3 executa incondicionalmente na config padrão; teste com as 12 imagens reais provou frame Y-up de lights.npy; negar só dZ/dY torna o campo não-integrável e o solver converge silenciosamente para superfície distorcida.

#### BUG-008 — convert_image_array_to_fni grava FNI top-down enquanto a biblioteca C de integração assume Y para cima — sinal de dZ/dY potencialmente invertido

**Local:** src/hybrid_stereo_method/infrastructure/io/image_io.py:160 · **Categoria:** integração · **Encontrado por:** Infrastructure

O escritor FNI itera `for y in range(ny)` mapeando a linha 0 do numpy (TOPO da imagem, convenção OpenCV) para y=0 do FNI, sem qualquer flip ou opção de orientação. A biblioteca C consumidora assume explicitamente sistema de imagem com 'origin at the bottom left corner and Y axis pointing up' (pst_normal_map.h) e o help do binário avisa que G_SY deve ser negativo se a convenção do mapa for oposta. O integrate.py usa slopes_scale=(1.0,1.0) por default e o YAML não expõe esse parâmetro, então nenhuma correção é aplicada em lugar algum. Se as normais do fotométrico seguem convenção Y-para-cima (usual em datasets de PS / lights.npy do gerador Stolfi), a componente Ny fica com sinal trocado em relação à varredura top-down do FNI, e o C computa dZ/dY = -Ny/Nz com sinal invertido.

```python
for y in range(ny):
            for x in range(nx):
                if nc == 1:
                    value = image_array[y, x]
                    f.write(f"{x:5d} {y:5d} {value:+.7e}\n")
```

**Impacto:** Superfície integrada com curvatura vertical espelhada/invertida (côncavo vira convexo ao longo do eixo Y), inconsistência sistemática entre as equações de gradiente em X e em Y e conflito direto com os hints zMos — tudo silencioso. A confirmação final depende da convenção (não documentada) de lights.npy do dataset.

**Sugestão de correção:** Definir e documentar a convenção na fronteira: ou gravar o FNI com flip vertical (e flipar de volta na leitura do Z), ou negar Ny antes de serializar as normais, ou expor slopes_scale no YAML e passar sy=-1 ao binário (-normals file scale 1 -1). Adicionar um teste com superfície sintética (plano inclinado conhecido) para fixar o sinal.

**Verificação:**
- Matemática: confirmado — incerteza sobre a convenção resolvida empiricamente (bunny_lambert_noshadow: Ny=+0.61 na borda superior, −0.33 na inferior → Y-para-cima); dZ/dY invertido com dZ/dX correto.
- Literal: confirmado — writer sem flip, pst_normal_map.h Y-up, dZdY=−Ny/Nz, nenhuma compensação; teste com esfera ex20_povball provou lights.npy Y-up.
- Impacto: confirmado — no dataset exato do YAML (melon24), corr(dZ/dcol, −Nx/Nz)=+0.50 valida X enquanto Y aparece com sinal trocado em relação ao ground truth; caminho default do pipeline híbrido.

#### BUG-009 — Janela de cosseno (Hann) com frequência invertida na interpolação quadrática

**Local:** src/hybrid_stereo_method/multifocus/math_utils.py:32 · **Categoria:** matemática · **Encontrado por:** Multifocus — núcleo; Convenções geométricas; Fluxo de configuração; Contratos de dados

Em quadratic_interpolation, o fator da janela raised-cosine é calculado como `a = pi * 0.5 * (m + 1)` (multiplicação), quando o correto para uma janela de Hann centrada em k_fuzzy com meia-largura 0.5*(m+1) é `a = pi / (0.5 * (m + 1))`. Para m=5 o código usa a = 3π ≈ 9.42 em vez de π/3 ≈ 1.047. Com a = 3π, o cosseno completa ~1.5 ciclos por unidade de distância e os pesos w_j = 0.5*(1+cos(a*d)) oscilam de forma pseudo-aleatória dentro da janela. Verificação numérica: para k_fuzzy=5.3 (janela 3..7) os pesos do código são [0.0245, 0.9755, 0.0245, 0.9755, 0.0245] — o ponto MAIS PRÓXIMO de k_fuzzy (distância 0.3) recebe peso ~0.02 e pontos distantes recebem ~0.98; para s=0 os pesos são [1, 0, 1, 0, 1], excluindo completamente os vizinhos imediatos do ajuste. A fórmula pretendida daria pesos decrescentes com a distância ([0.13, 0.60, 0.98, 0.87, 0.40]).

```python
a = pi * 0.5 * (m + 1)
    w = [0.5 * (1 + cos(a * (k0 + j - k_fuzzy))) for j in range(m)]
```

**Impacto:** Quando multifocus.parameters.interpolation = 'quadratic_interpolation' (opção documentada no YAML), a regressão quadrática ponderada usa pesos essencialmente invertidos/oscilantes: o ajuste é dominado por pontos distantes do índice fuzzy e ignora os vizinhos imediatos. zMos e sMos saem silenciosamente errados (overshoot e ruído na profundidade física e na imagem all-in-focus), contaminando os hints da integração e a entrada do fotométrico. O default atual dos YAMLs é 'linear_interpolation', então o bug está latente, mas qualquer experimento com a opção quadrática produz resultado errado sem nenhum aviso.

**Sugestão de correção:** Trocar para `a = pi / (0.5 * (m + 1))` (divisão), garantindo que o peso decaia de 1 no centro (d=0) a 0 na borda da janela (|d| = 0.5*(m+1)).

**Verificação:**
- Matemática: confirmado — derivação independente (a = π/L) e reprodução numérica de todos os pesos citados; teste fim-a-fim: erro máx. 0.117 (código) vs 0.047 (correção).
- Literal: confirmado — linha 32 confere; com a=3π o cosseno completa ~1.5 ciclos por unidade, impossível ser janela intencional; alcançável via opção documentada nos YAMLs.
- Impacto: confirmado — knob anunciada ao usuário (mudança de uma linha no YAML); pesos invertidos corrompem zMos/sMos silenciosamente; latente nos experimentos atuais (default linear).

#### BUG-010 — main.py (RPS) usa chaves planas que não existem no YAML aninhado

**Local:** src/hybrid_stereo_method/photometric/main.py:27 · **Categoria:** integração · **Encontrado por:** Photometric — WPS e utils

O main do caminho RPS lê parameters.get("input_path"), "data_foldername" (linha 27), "output_path" (36), "data_scale" (77,82), "image_type" (76,81), "method_name" (86-95) e "debug". Mas read_yaml_parameters retorna o dict aninhado do YAML, e ps_experiment.yaml define essas informações em experiment.paths.input, experiment.paths.data_folder, experiment.paths.output, photometric.parameters.method, photometric.parameters.data_scale e photometric.parameters.image_extension. Todos os .get() retornam None: os.path.join(None, None) lança TypeError na linha 27; mesmo que passasse, method_name=None cairia em ValueError("Unsupported method: None"). Há ainda mismatch de nome: "image_type" no código vs "image_extension" no YAML.

```python
data_path = os.path.join(parameters.get("input_path"), parameters.get("data_foldername"))
```

**Impacto:** O caminho photometric standalone via RPS (configs/ps_experiment.yaml) está inoperante: falha imediatamente com TypeError. Qualquer comparação L2/L1/SBL/RPCA com o WPS é impossível com a configuração atual.

**Sugestão de correção:** Alinhar main.py à estrutura aninhada do YAML, como faz main_wps.py: input_path = parameters["experiment"]["paths"]["input"], data_foldername = parameters["experiment"]["paths"]["data_folder"], method = parameters["photometric"]["parameters"]["method"], data_scale e image_extension idem.

**Verificação:**
- Matemática: confirmado — read_yaml_parameters é yaml.safe_load puro; os.path.join(None, None) lança TypeError (reproduzido); README documenta exatamente o comando quebrado; nuance: test_ps.yaml (esquema plano legado) ainda funcionaria.
- Literal: confirmado — todas as chaves planas e o mismatch image_type/image_extension verificados; entry point do pyproject.toml ainda aponta para função cli inexistente.
- Impacto: confirmado — crash reproduzido por simulação; fluxo documentado no README está inoperante; pipeline híbrido (main_wps) não é afetado, mas o caminho standalone RPS sim.

### Médios

#### BUG-011 — pst_normal_from_slope sem os sinais negativos: inconsistente com pst_slope_from_normal

**Local:** csrc/integrate_recursive/lib-src/pst_basic.c:38 · **Categoria:** matemática · **Encontrado por:** C — solver

Para uma superfície Z(X,Y), a normal é proporcional a (−dZ/dX, −dZ/dY, 1). `pst_slope_from_normal` (linhas 44-52 do mesmo arquivo) usa corretamente dZdX = −nx/nz. Mas `pst_normal_from_slope` retorna (dZdX/m, dZdY/m, 1/m) SEM negar as componentes tangenciais. O round-trip `pst_slope_from_normal(pst_normal_from_slope(g))` devolve −g (gradiente negado). A função é usada em pst_normal_map.c:191 (`pst_normal_map_pixel_avg`, que converte normais→slopes na linha 175 com a convenção correta e converte de volta na 191 com a errada, devolvendo normais com Nx,Ny de sinal trocado) e em pst_normal_map.c:259 (`pst_normal_map_from_slope_map`).

```c
r3_t pst_normal_from_slope(r2_t *grd)
  { double dZdX = grd->c[0]; 
    double dZdY = grd->c[1]; 
    double m = sqrt(1.0 + dZdX*dZdX + dZdY*dZdY);
    double nx = dZdX/m; 
    double ny = dZdY/m;
```

**Impacto:** O caminho principal do pipeline (normais→slopes via pst_normal_map_to_slope_map) NÃO usa esta função, então o resultado atual do hybrid não é afetado. Porém qualquer uso de pst_normal_map_from_slope_map ou pst_normal_map_from_proc (geração de dados de teste/validação) produz normais com componentes X,Y de sinal invertido silenciosamente — superfícies espelhadas em relevo.

**Sugestão de correção:** Corrigir para `nx = -dZdX/m; ny = -dZdY/m;`, tornando-a inversa exata de pst_slope_from_normal.

**Verificação:**
- Matemática: confirmado — round-trip devolve −g; header exige normal "outwards-pointing"; usos citados verificados.
- Literal: confirmado — inconsistência interna irrefutável em pst_normal_map_pixel_avg, sem compensação posterior; funções afetadas sem chamadores no csrc atual (bug latente).
- Impacto: refutado — nenhum executável/script do pipeline chama as funções afetadas (código morto herdado da lib pst); o caminho real usa exclusivamente pst_slope_from_normal, a versão correta.

#### BUG-012 — Ordenação lexicográfica (sorted) do stack multifocal pode quebrar correspondência imagem↔z_foc

**Local:** src/hybrid_stereo_method/hybrid/main.py:90 · **Categoria:** integração · **Encontrado por:** Multifocus — alinhamento e pipeline

zf_directories (linha 77) e filtered_files (linhas 90 e 125) usam sorted() lexicográfico, enquanto o restante do pipeline usa natsorted (hybrid/main.py:147; image_io.py:71). multifocus/main.py:133-136 valida apenas o COMPRIMENTO de z_foc contra o número de imagens, nunca a ordem. Se os diretórios de foco não forem zero-padded (ex.: 'zf15', 'zf25', ..., 'zf105', 'zf125'), sorted() produz zf105 < zf115 < zf125 < zf15 < zf25..., embaralhando o stack em relação à lista crescente z_foc=[15..125] do YAML. O dataset atualmente configurado usa nomes zero-padded ('zf015.0000-df020.0000'), então o bug está latente, mas nada no código garante isso.

```python
filtered_files = sorted(
            [file for file in input_files_path if f"{zf_dir}" in file and "sVal.png" in file]
        )
```

**Impacto:** Com nomes não zero-padded, cada pixel de zMos recebe o z_foc da imagem errada (mosaic interpola zFoc pelo índice fuzzy do frame), produzindo um mapa de profundidade fisicamente errado sem nenhum erro visível — a validação de tamanho passa e o pipeline conclui normalmente.

**Sugestão de correção:** Trocar sorted() por natsorted() nas linhas 77, 90 e 125 de hybrid/main.py (o import já existe na linha 7), e idealmente validar que a ordem dos diretórios zf* corresponde à ordem crescente de z_foc.

**Verificação:**
- Matemática: confirmado — a bijeção ordenada frame↔zFoc é exigida pelo mosaic; ordenação lexicográfica diverge da numérica em nomes não-padded; só o comprimento é validado.
- Literal: confirmado — sorted() nas linhas 77/90/125 vs natsorted no resto; dataset atual é zero-padded (bug latente).
- Impacto: refutado — a linha 90 alimenta apenas uma média (comutativa) e todos os datasets reais do repo usam largura fixa zfXXX.XXXX, em que sorted ≡ ordem numérica; fragilidade de robustez, não bug manifestável.

#### BUG-013 — Médias por plano focal re-normalizadas min-max por frame antes do multifocus 'average'

**Local:** src/hybrid_stereo_method/hybrid/main.py:100 · **Categoria:** integração · **Encontrado por:** Multifocus — alinhamento e pipeline; Hybrid; Convenções geométricas; Fluxo de configuração; Contratos de dados

As imagens médias por zf são salvas com save_image (que aplica min-max por imagem, image_io.py:129) e os PNGs re-normalizados são relidos como entrada do multifocus (parameters['filtered_dir'] = average_images_paths; multifocus/main.py:96). Cada frame do stack de foco recebe um esticamento de contraste independente (ganho/offset distintos por zf). O indicador de foco (energia de alta frequência da FFT, fourier.py) escala com o contraste local, e o argmax fuzzy compara as magnitudes do indicador ENTRE frames para escolher o plano focal — a normalização global do stack em applicator.py:86-97 não desfaz o re-escalonamento por frame. Note que as execuções por direção de luz L* leem os sVal.png originais (hybrid/main.py:125-127), então só o ramo 'average' — justamente o que gera o zMos usado como hints — é afetado.

```python
save_image(average_image_path, f"average_{zf_dir}.png", average_image)
```

**Impacto:** Frames com faixa dinâmica menor são amplificados mais, inflando artificialmente seu indicador de foco e enviesando iSel/zMos do diretório 'average' — exatamente o mapa de profundidade usado como hints na integração. Erro numérico silencioso.

**Sugestão de correção:** Passar as médias em memória (ou .npy) ao multifocus sem o round-trip por save_image, ou gravar o PNG sem re-normalização (a média de uint8 já está em 0–255).

**Verificação:**
- Matemática: confirmado — indicador Fourier é linear e o passa-alta remove o offset mas o ganho propaga (F'_k = g_k·F_k); normalização global não desfaz ganhos heterogêneos; desloca k_max/k_fuzzy/confiança.
- Literal: confirmado — todos os elos verificados; ramos L* leem os sVal.png originais, só o 'average' é afetado.
- Impacto: confirmado — branch ativo na config padrão; frames borrados (faixa menor) ganham amplificação maior — viés na direção errada; o zMos do 'average' é justamente o hint da integração (use_hints: True).

#### BUG-014 — Confiança das normais do photometric é descartada na integração (usa normal_map.npy de 3 canais em vez do FNI de 4 canais)

**Local:** src/hybrid_stereo_method/hybrid/main.py:168 · **Categoria:** integração · **Encontrado por:** Contratos de dados

O photometric produz normal_map_with_residuals.fni com 4 canais (Nx,Ny,Nz,confidence∈[0,1]; main_wps.py:147-151), formato que o C aceita diretamente como peso por pixel (pst_normal_map_get_weight: canal 3 = peso; com 3 canais o peso vira 1.0 para todos). O hybrid, porém, carrega normal_map.npy (3 canais) e o regrava como FNI de 3 canais (integrate.py:115-116). Resultado: todo pixel não-NaN entra no sistema linear com peso 1, inclusive pixels onde o solver robusto rejeitou muitas observações ou teve resíduos altos. Apenas NaN vira peso 0 (pst_normal_map.c:241-243).

```python
normal_map_path = os.path.join(
        parameters["output_path_photometric"], "normal_map.npy"
    )
```

**Impacto:** Pixels de normal pouco confiável pesam igual aos bons na integração, contaminando o height map em regiões de sombra/especular sem nenhum aviso — a infraestrutura de pesos existe nas duas pontas mas a ponte joga a informação fora.

**Sugestão de correção:** Carregar normals e confidence (ou ler o próprio normal_map_with_residuals.fni) e passar o array (H,W,4) para integrate_normals_to_height, que já aceita 4 canais.

**Verificação:**
- Matemática: confirmado — confidence ∈ (0,1] real e variável; mínimos quadrados ponderados por confiança é o uso pretendido pelo próprio autor do C (exemplo no header usa o FNI de 4 canais).
- Literal: confirmado — o FNI de 4 canais é escrito e nunca lido por nenhum código; pst_normal_map_get_weight retorna 1.0 com NC==3.
- Impacto: confirmado — fluxo padrão do hybrid; perda real nos pixels de confiança baixa mas finita (NaN já vira peso 0); altera o height map com hints ativos.

#### BUG-015 — Confiança das normais do fotométrico descartada na fronteira Python→C

**Local:** src/hybrid_stereo_method/hybrid/main.py:173 · **Categoria:** integração · **Encontrado por:** Hybrid; Interface Python↔C; Convenções geométricas; Fluxo de configuração

O pipeline integra normal_map.npy de 3 canais (Nx,Ny,Nz), embora o fotométrico já produza normal_map_with_residuals.fni de 4 canais com a confiança por pixel (main_wps.py:147-151, confidence = (N/M)/(1+std), wps.py:196-198) e o binário C aceite exatamente esse formato — 'The normal map must be a three- or four-channel ... and an optional reliability weight' (gus_integrate_recursive.c:206-209), lido com wch=3 em tire_read_fni_file (linha 441) e usado como peso das equações via pst_normal_map_to_slope_map (pst_normal_map.c:239-242). Com 3 canais, todo pixel não-NaN entra com peso 1: normais de baixa qualidade (poucas luzes válidas, resíduo alto) pesam igual às boas.

```python
normal_map = np.load(normal_map_path)
```

**Impacto:** O sistema de mínimos quadrados do integrador pondera errado: pixels com normais ruidosas distorcem a superfície ao redor com o mesmo peso de pixels confiáveis. Resultado numericamente diferente (pior) do projetado, sem nenhum erro visível.

**Sugestão de correção:** Empilhar a confiança como 4º canal antes de chamar integrate_normals_to_height (np.concatenate([normals, confidence[...,None]], axis=-1)) ou passar diretamente o normal_map_with_residuals.fni já gerado pelo fotométrico.

**Verificação:**
- Matemática: confirmado — peso flui do canal 4 do normal map ao canal 2 do slope map e ao funcional Q(Z)=Σ w_e(d_e−ΔZ_e)²; caminho de 4 canais suportado ponta a ponta — descartar é falha de fiação.
- Literal: confirmado — normal_map_with_residuals.fni não é consumido por ninguém; wrapper Python já documenta (H,W,4).
- Impacto: confirmado — Step 3 sempre executa no fluxo hybrid; confiança não-uniforme em dados reais torna o efeito não-neutro no height map final.

#### BUG-016 — read_fni_to_image_array quebrado para NC=1: linhas de cabeçalho são interpretadas como dados

**Local:** src/hybrid_stereo_method/infrastructure/io/image_io.py:220 · **Categoria:** integração · **Encontrado por:** Infrastructure

No loop de parsing dos dados, as linhas de cabeçalho ('NC = 1', 'NX = ...', 'NY = ...') não são puladas — só 'begin', 'end' e linhas vazias são. O filtro `len(parts) < 2 + nc` deixa passar o cabeçalho quando nc==1, porque 'NC = 1'.split() tem 3 tokens e 2+nc==3. Em seguida `x = int(parts[0])` tenta int('NC') e lança ValueError. Confirmado empiricamente: o round-trip convert_image_array_to_fni → read_fni_to_image_array de um array (H,W) falha com "invalid literal for int() with base 10: 'NC'". Ou seja, NENHUM FNI de 1 canal (iSel.fni, wSel.fni, zMos.fni, todos produzidos pelo multifocus) pode ser relido por esta função. Só funciona por acaso para NC>=2 (cabeçalho tem 3 tokens < 2+nc).

```python
if len(parts) < 2 + nc:
            continue
        x = int(parts[0])
```

**Impacto:** Qualquer consumidor que tente reler os FNIs de 1 canal do pipeline (zMos.fni com Z físico, iSel.fni, wSel.fni) falha imediatamente; o pipeline atual só sobrevive porque relê apenas FNIs de 2 canais (height-00-end-Z.fni). A interface FNI declarada como round-trip é falsa para NC=1.

**Sugestão de correção:** Pular explicitamente as linhas de cabeçalho/metadata no loop de dados (ex.: `if '=' in line: continue`) ou iniciar o parse de dados somente após a linha 'NY = ...'.

**Verificação:**
- Matemática: confirmado — lógica do filtro analisada e round-trip NC=1 reproduzido empiricamente (ValueError em int('NC')); NC=2 funciona por coincidência.
- Literal: confirmado — reprodução empírica idêntica; a exceção interrompe a leitura imediatamente.
- Impacto: refutado — os únicos chamadores leem apenas os FNIs NC=2 produzidos pelo solver C (que garante NC_Z == 2); os FNIs de 1 canal nunca são relidos por código do repositório — defeito latente.

#### BUG-017 — Etapa de alinhamento desconectada do pipeline e módulo quebrado (imports/assinaturas inexistentes)

**Local:** src/hybrid_stereo_method/multifocus/image_alignment.py:6 · **Categoria:** integração · **Encontrado por:** Multifocus — alinhamento e pipeline

Nenhum main (multifocus/main.py, hybrid/main.py) importa image_alignment — o pipeline multifocal SEMPRE roda sobre o stack bruto, sem nenhuma compensação do 'focus breathing' (mudança de escala/magnificação entre planos focais). Além disso o módulo está inexecutável: 'from utils import *' (linha 6) não resolve dentro do pacote (o módulo correto seria hybrid_stereo_method.multifocus.utils) e, mesmo se resolvesse, multifocus/utils.py não define read_image nem save_image, usados em main_align (linhas 191, 195, 213, 216); a chamada save_image(save_path, ref_save_as, reference_img, 0, 255) passa 5 argumentos enquanto a save_image da infraestrutura aceita 3. Adicionalmente, salvar frames alinhados via save_image (min-max por imagem) + JPEG lossy reintroduziria distorção radiométrica e artefatos de alta frequência no stack se fosse religado como está.

```python
from utils import *
```

**Impacto:** Se as imagens do dataset tiverem desalinhamento/mudança de magnificação entre planos focais (típico de stacks reais de foco), o indicador de foco compara pixels que não correspondem ao mesmo ponto da cena, corrompendo iSel/zMos silenciosamente — e a etapa que deveria corrigir isso nunca executa. Para datasets sintéticos pré-alinhados o efeito é nulo, mas não há nenhuma verificação ou aviso.

**Sugestão de correção:** Decidir o destino do módulo: (a) integrá-lo de fato ao pipeline (corrigir imports para o pacote, usar cv2.imwrite sem re-normalização e PNG sem perdas, e invocá-lo opcionalmente via config antes do focus_indicator), ou (b) removê-lo/documentá-lo como legado, registrando explicitamente a premissa de que o stack de entrada já está alinhado.

**Verificação:**
- Matemática: confirmado — nenhum main importa o módulo; NameError/TypeError garantidos se religado; o efeito do focus breathing no argmax por pixel está corretamente descrito.
- Literal: confirmado — nenhuma referência externa ao módulo; imports e assinaturas quebrados verificados; distorção radiométrica do save_image + JPEG real.
- Impacto: refutado — código 100% morto: os mains produzem o mesmo resultado com ou sem o arquivo; o cenário de focus breathing é limitação de design condicionada a propriedade hipotética do dataset.

#### BUG-018 — Contrato invertido de align_im1_to_im2: alinha img2 a img1, oposto do nome e da docstring

**Local:** src/hybrid_stereo_method/multifocus/image_alignment.py:160 · **Categoria:** matemática · **Encontrado por:** Multifocus — alinhamento e pipeline

O nome e a docstring prometem "Aligns 'img1' with 'img2'" e retornam "aligned_img: 'img1' aligned with 'img2'", mas a implementação chama apply_homography(img2, img1, points2, points1), ou seja, computa H que mapeia pontos de img2 para img1 e faz warp de img2 para o referencial de img1 — exatamente o inverso do contrato. O único chamador (main_align:210) compensa passando (reference_img, target_img) e tratando o retorno como 'target alinhado à referência', então o fluxo atual funciona por dupla inversão.

```python
aligned_img = apply_homography(img2, img1, points2, points1)
```

**Impacto:** Qualquer chamador que siga a assinatura/documentação obtém o alinhamento na direção contrária (referencial trocado) silenciosamente: o warp é geometricamente válido e visualmente plausível, mas o stack resultante fica alinhado à imagem errada — erro difícil de detectar a olho.

**Sugestão de correção:** Corrigir a implementação para casar com o contrato (apply_homography(img1, img2, points1, points2)) e ajustar o chamador em main_align, ou renomear a função e a docstring para refletir que ela alinha img2 a img1.

**Verificação:**
- Matemática: confirmado — semântica de findHomography/warpPerspective verificada; o comentário interno da linha 159 admite a inversão; dupla inversão no único chamador.
- Literal: confirmado — contrato documentado invertido, sem compensação além do chamador interno; bug latente para novos chamadores.
- Impacto: refutado — único chamador compensa e o módulo inteiro é código morto (nenhum import; main_align falharia com NameError antes do alinhamento); problema de documentação em código inalcançável.

#### BUG-019 — zero_border zera a IMAGEM antes do indicador, criando degrau artificial de alta frequência (efeito oposto ao pretendido)

**Local:** src/hybrid_stereo_method/multifocus/indicators/applicator.py:34 · **Categoria:** matemática · **Encontrado por:** Multifocus — indicadores

Quando `zero_border=True`, `img = zero_borders(img, 40)` zera um anel de 40 px na imagem de ENTRADA antes de calcular o indicador. Isso cria um degrau abrupto (intensidade→0) exatamente na fronteira do anel, que é a estrutura de mais alta frequência possível — Laplaciano e filtro passa-alta de Fourier respondem com magnitude máxima ali, em TODOS os frames. O código comentado nas linhas 52-54 mostra a intenção original correta: zerar as bordas do INDICADOR, depois de calculado. A flag, como está, injeta o artefato de borda que deveria remover.

```python
if zero_border:
            # Zero out borders (remove edge artifacts)
            img = zero_borders(img, 40)
```

**Impacto:** Silencioso e duplo: (1) os pixels na fronteira do anel ganham foco espúrio máximo em todos os frames, corrompendo iSel/wSel numa moldura ao redor da imagem; (2) como a normalização do stack divide pelo máximo GLOBAL (linha 96), e o máximo passa a ser o artefato do degrau, todos os valores de foco reais são comprimidos para perto de zero — degradando wSel e o teste de curva plana (polyfit_epsilon) do argmax_fuzzy na imagem inteira. Hoje `zero_border: False` nos dois YAMLs, então só se manifesta quando o usuário ativa a opção — e aí piora exatamente o que prometia corrigir.

**Sugestão de correção:** Mover o zero_borders para depois do cálculo do indicador (reativar as linhas 52-54 e remover as linhas 32-34), zerando as bordas de `focus_indicator` em vez de `img`. Alternativamente, recortar (crop) a borda do indicador em vez de zerar a imagem.

**Verificação:**
- Matemática: confirmado — simulação numérica: resposta do |Laplaciano| 37-144x maior que a textura em frames desfocados, 54x no passa-alta de Fourier; compressão dos valores reais confirmada (0.90→0.68 na simulação).
- Literal: confirmado — código e intenção original (linhas comentadas) verificados; sem corte de borda posterior; latente (zero_border: False nos YAMLs).
- Impacto: refutado — branch nunca ativado: todas as configs e testes usam zero_border False (default False); armadilha de configuração latente, não bug manifesto.

#### BUG-020 — interpolation_type não reconhecido produz sMos e zMos inteiramente zerados em silêncio

**Local:** src/hybrid_stereo_method/multifocus/mosaic.py:44 · **Categoria:** integração · **Encontrado por:** Multifocus — núcleo

A cadeia if/elif (linhas 44, 58, 76) compara interpolation_type com três strings exatas ('crop', 'quadratic_interpolation', 'linear_interpolation') e não possui ramo else. sMos e zMos são inicializados com np.zeros (linhas 38-39); se o YAML contiver qualquer variação (ex.: 'linear', 'quadratic', erro de digitação), o loop percorre todos os pixels sem executar nada e a função retorna arrays totalmente zerados, sem erro nem warning. multifocus/main.py:138-139 passa a string do YAML diretamente, sem validação.

```python
if interpolation_type == "crop":
```

**Impacto:** Com um typo na chave multifocus.parameters.interpolation, o pipeline inteiro segue rodando com zMos = 0 em todo pixel — que é gravado SEM normalização em zMos.fni e zMos_with_confidence.fni como 'valor físico' e alimenta os hints da integração C com Z=0 e confiança normalizada não nula; sMos.png todo preto (ou re-normalizado para lixo) alimenta o fotométrico. Resultado final completamente errado sem nenhuma mensagem de erro.

**Sugestão de correção:** Adicionar `else: raise ValueError(f"interpolation_type desconhecido: {interpolation_type}")` (ou validar a string em multifocus/main.py antes de chamar mosaic).

**Verificação:**
- Matemática: confirmado — ausência de else/raise/warning e cadeia de propagação (zMos físico zerado nos hints) verificadas factualmente.
- Literal: confirmado — leitura direta; worktree paralelo já contém correção com raise ValueError, evidenciando bug reconhecido e ausente na árvore auditada.
- Impacto: refutado — todas as configs usam valor válido documentado; o cenário exige typo hipotético futuro — ausência de validação defensiva (latente), não bug manifesto.

#### BUG-021 — Chave photometric.parameters.method do YAML é ignorada silenciosamente pelo main_wps

**Local:** src/hybrid_stereo_method/photometric/main_wps.py:133 · **Categoria:** integração · **Encontrado por:** Fluxo de configuração

main_wps.py nunca lê parameters['photometric']['parameters']['method']: chama sempre estimate_normals_argmax_lstsq_robust (linha 133), qualquer que seja o método configurado. O ps_experiment.yaml documenta 'Options: L2, L1, SBL, RPCA, wodham_implementation_argmax' (ps_experiment.yaml:27), mas esses solvers só existem em photometric/main.py — que por sua vez lê um esquema FLAT totalmente diferente, incompatível com o esquema aninhado de ps_experiment.yaml (apenas test_ps.yaml é compatível), apesar de o README mandar rodar `photometric.main --param_file configs/ps_experiment.yaml`. Também data_scale e image_extension do YAML aninhado nunca são lidos por main_wps.py.

```python
normals, albedo, confidence, selected_areas = estimate_normals_argmax_lstsq_robust(
        images, light_sources, wps_params
    )
```

**Impacto:** Usuário que configura method: 'L2' (ou qualquer outro) em ps_experiment/wps_experiment/hb_experiment e roda main_wps obtém silenciosamente o solver robust-argmax, acreditando estar comparando métodos; e a combinação README+ps_experiment.yaml com photometric/main.py quebra por chaves inexistentes.

**Sugestão de correção:** Em main_wps.py, ler photometric.parameters.method e despachar para o solver correspondente (ou falhar com erro claro se o método não for suportado); unificar photometric/main.py para o mesmo esquema aninhado dos YAMLs de configs/.

**Verificação:**
- Matemática: confirmado — main_wps lê apenas photometric.solver e chama o solver hardcoded; combinação README+ps_experiment.yaml quebra com TypeError.
- Literal: confirmado — grep confirma que 'method', 'data_scale' e 'image_extension' nunca são lidos por main_wps; opções documentadas só existem no RPS.
- Impacto: confirmado — ps_experiment.yaml está commitado com method: 'L2' e só é consumível por main_wps, que o ignora silenciosamente e executa solver matematicamente distinto.

#### BUG-022 — Avaliação contra ground truth quebrada em três níveis (np.load, mask indefinida, eixo errado)

**Local:** src/hybrid_stereo_method/photometric/main_wps.py:163 · **Categoria:** integração · **Encontrado por:** Photometric — WPS e utils

Se gt_normal.npy existir: (1) np.load(filename=gt_normal_path) usa kwarg inexistente — a assinatura é np.load(file=...) — e lança TypeError; (2) a variável mask usada na linha 164 nunca é definida, pois a leitura da máscara está comentada na linha 122 — NameError; (3) mesmo corrigindo os dois, evaluate_angular_error (ps_utils.py:141-142) faz np.sum(ae, axis=1) assumindo arrays achatados (p,3), mas aqui N_gt e normals são (H,W,3): a soma seria sobre a LARGURA da imagem, não sobre os canais XYZ, produzindo um "erro angular" numericamente sem sentido, sem nenhum aviso. O main.py do RPS faz o reshape correto (linha 110) antes de chamar a mesma função.

```python
N_gt = np.load(filename=gt_normal_path)  # Load ground truth normal map
        angular_error = evaluate_angular_error(N_gt, normals, mask)
```

**Impacto:** Com gt_normal.npy presente, o run WPS aborta (TypeError/NameError); se os dois primeiros erros forem corrigidos ingenuamente, a métrica de erro angular reportada no log é matematicamente inválida (soma sobre eixo espacial em vez do canal), levando a conclusões erradas sobre a qualidade das normais.

**Sugestão de correção:** Usar N_gt = np.load(gt_normal_path); recuperar a leitura da máscara (ou passar None); e achatar ambos os mapas para (H*W, 3) antes de chamar evaluate_angular_error, como faz main.py:110, ou trocar axis=1 por axis=-1 em evaluate_angular_error.

**Verificação:**
- Matemática: confirmado — np.load não aceita kwarg 'filename' (verificado empiricamente); mask nunca definida; axis=1 sobre (H,W,3) soma sobre a largura, métrica sem significado.
- Literal: confirmado — três níveis verificados literalmente; main.py do RPS confirma o uso correto com reshape.
- Impacto: confirmado — não é código morto: o repo contém quatro datasets com gt_normal.npy no layout esperado; apontar a config do WPS para eles é mudança rotineira que causa abort imediato.

#### BUG-023 — evaluate_angular_error soma no eixo errado quando recebe mapa (H,W,3) do caminho WPS

**Local:** src/hybrid_stereo_method/photometric/ps_utils.py:142 · **Categoria:** matemática · **Encontrado por:** Fluxo de configuração; Contratos de dados

A função foi escrita para o formato RPS de normais achatadas (p,3), onde `np.sum(ae, axis=1)` soma sobre os 3 componentes do vetor. O main_wps.py:162-166 a chama com mapas (H,W,3) sem reshape: axis=1 então soma sobre a LARGURA da imagem, produzindo um array (H,3) sem sentido geométrico, que é clipado e passado a arccos — o 'erro angular médio' logado é um número inválido, sem exceção. (Obs.: hoje esse bloco nem chega a executar corretamente porque main_wps.py:163 usa `np.load(filename=...)` — kwarg inexistente — e main_wps.py:164 referencia `mask`, cuja leitura está comentada na linha 122; mas após corrigir esses crashes, o eixo errado produziria métrica silenciosamente errada.)

```python
ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
```

**Impacto:** Métrica de validação (erro angular médio vs ground truth) reportada com valor numericamente errado quando gt_normal.npy existir no modo WPS/hybrid, mascarando regressões de qualidade das normais.

**Sugestão de correção:** Usar `np.sum(ae, axis=-1)` (funciona para (p,3) e (H,W,3)) ou fazer reshape para (p,3) em main_wps antes da chamada; corrigir também np.load(file=...) e a leitura da máscara.

**Verificação:**
- Matemática: confirmado — reproduzido numericamente: com gt == normals (erro verdadeiro 0°), axis=1 reporta ~8.4° e axis=-1 dá ~0°; defeito latente atrás dos crashes de BUG-022, como descrito.
- Literal: confirmado — contrato (p,3) provado pelo reshape do caminho RPS; sem compensação posterior.
- Impacto: refutado — o único call-site que passaria (H,W,3) crasha deterministicamente antes (TypeError/NameError); o eixo errado só se manifestaria após corrigir dois outros bugs na mesma linha de código.

#### BUG-024 — Detecção RGB/BGR por média de canais é matematicamente inválida e pode trocar os pesos R e B por imagem na montagem de M

**Local:** src/hybrid_stereo_method/photometric/ps_utils.py:175 · **Categoria:** matemática · **Encontrado por:** Photometric — solvers; Photometric — WPS e utils

converter_npy_para_cinza decide se a imagem é RGB ou BGR comparando np.mean(canal 0) > np.mean(canal 2). É impossível inferir a ordem de armazenamento a partir das médias dos canais — o resultado depende apenas das cores da cena. Como load_images (chamada por RPS.load_images, rps.py:66-72) lê com cv2.imread (sempre BGR), qualquer imagem com dominância de azul (mean(B) > mean(R)) é classificada como 'RGB' e recebe gray = 0.3·B + 0.59·G + 0.11·R, com os pesos de R e B trocados. Pior: a decisão é tomada POR IMAGEM, então dentro do mesmo stack fotométrico imagens sob luzes diferentes podem ser convertidas com fórmulas diferentes, quebrando a consistência radiométrica entre as linhas de L e as colunas de M que os solvers L2/L1/SBL/RPCA assumem. Também diverge da conversão usada no caminho WPS (cv2.COLOR_BGR2GRAY, infrastructure/utils.py), ou seja, os dois caminhos PS produzem M diferentes para o mesmo dataset.

```python
if np.mean(matriz[:, :, 0]) > np.mean(matriz[:, :, 2]):  # Formato RGB
        r, g, b = matriz[:, :, 0], matriz[:, :, 1], matriz[:, :, 2]
    else:  # Formato BGR
        b, g, r = matriz[:, :, 0], matriz[:, :, 1], matriz[:, :, 2]
```

**Impacto:** A matriz de medidas M entregue aos solvers do RPS pode misturar imagens convertidas com 0.3R+0.59G+0.11B e outras com 0.3B+0.59G+0.11R no mesmo stack, distorcendo as razões de intensidade entre direções de luz. As normais resultantes ficam silenciosamente enviesadas (sem erro, sem aviso), especialmente em cenas coloridas não neutras.

**Sugestão de correção:** Remover a heurística e usar uma conversão fixa coerente com o leitor: como cv2.imread devolve BGR, usar sempre gray = 0.114·B + 0.587·G + 0.299·R (ou cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)), a mesma usada no caminho WPS. Se houver fontes .npy em RGB, documentar/parametrizar a ordem de canais em vez de inferi-la por médias.

**Verificação:**
- Matemática: confirmado — ordem de canais é convenção de armazenamento, impossível de inferir por médias; decisão por imagem viola a consistência m_ij = ρ_i(n_i·l_j) dos solvers.
- Literal: confirmado — código, cadeia de chamadas e divergência com o caminho WPS (BGR2GRAY) verificados linha a linha.
- Impacto: refutado — nos datasets reais 0/50 imagens ativam o branch trocado; combinação fixa uniforme é absorvida pela normalização per-pixel das normais; divergência de branch dentro de um stack só ocorre num caso ~neutro abaixo do ruído de quantização.

#### BUG-025 — Fórmula do albedo errada: usa a normal já normalizada, ignorando as intensidades

**Local:** src/hybrid_stereo_method/photometric/wps.py:190 · **Categoria:** matemática · **Encontrado por:** Photometric — WPS e utils

Em estimate_normals_argmax_lstsq_robust, o albedo é calculado como np.linalg.norm(np.dot(selected_lights, normal)) DEPOIS de normal /= np.linalg.norm(normal) (linha 186). Ou seja, é a norma do vetor de produtos escalares L·n_unitário — uma quantidade puramente geométrica que depende só das direções de luz e da orientação da superfície, e não das intensidades observadas. No modelo I = albedo*(L·n), o albedo correto é a magnitude da solução do lstsq ANTES da normalização: albedo = ||normal_lstsq||.

```python
albedo[i, j] = np.linalg.norm(np.dot(selected_lights, normal))
```

**Impacto:** O mapa de albedo retornado é silenciosamente errado (não reflete refletância da superfície). Hoje main_wps.py recebe o albedo (linha 133) mas não o salva, então o dano é latente — qualquer uso futuro (visualização, máscara por refletância, re-pesagem) herdará valores sem significado físico.

**Sugestão de correção:** Calcular albedo[i, j] = np.linalg.norm(normal) imediatamente após o lstsq, antes da linha normal /= np.linalg.norm(normal).

**Verificação:**
- Matemática: confirmado — a solução do lstsq é g = ρ·n̂ e o albedo correto é ‖g‖; a fórmula do código é puramente geométrica (superfícies branca e preta dariam o mesmo "albedo").
- Literal: confirmado — ordem das operações (normalização antes do cálculo) verificada; sem compensação posterior.
- Impacto: refutado — o albedo é desempacotado numa variável jamais usada/salva; saída morta no pipeline ativo, sem manifestação em execução realista.

### Baixos

#### BUG-026 — Chaves de configuração nunca lidas pelo código: focal_step, depth_refinement.* e experiment.settings.mask

**Local:** configs/hb_experiment.yaml:35 · **Categoria:** integração · **Encontrado por:** Fluxo de configuração

multifocus.parameters.focal_step (hb_experiment.yaml:35, ms_experiment.yaml:37) não é lido em nenhum lugar de src/ (grep sem ocorrências fora dos YAMLs). O bloco multifocus.depth_refinement (enabled, gaussian_size, unary_scale, pair_scale, n_iter; hb_experiment.yaml:74-80) não é consumido: depth_refinement.py é código morto, não importado por nenhum main. experiment.settings.mask: True não tem efeito no multifocus — focus_indicator tem parâmetros mask/mask_img (applicator.py:21-22), mas multifocus/main.py:112-121 nunca os passa; e em main_wps.py a leitura da máscara está comentada (linha 122).

```yaml
focal_step: 1
```

**Impacto:** Usuário que altera focal_step, habilita depth_refinement.enabled: True ou mask: True acredita estar mudando o algoritmo, mas o resultado é idêntico — divergência silenciosa entre configuração declarada e experimento executado.

**Sugestão de correção:** Remover as chaves órfãs dos YAMLs (ou implementar seu consumo) e emitir warning para chaves desconhecidas/não consumidas ao carregar a configuração.

**Verificação:**
- Matemática: confirmado — todos os três grupos de chaves verificados como não-lidos por grep e leitura do código.
- Literal: confirmado — focus_indicator é chamado com 8 argumentos posicionais sem mask; leitura da máscara comentada em main_wps.py:122.
- Impacto: refutado — todas as configs têm mask: False e enabled: False; os valores não-lidos coincidem com o comportamento efetivo; problema de higiene de configuração.

#### BUG-027 — SYNOPSIS do C documenta keyword 'weight' que o parser não aceita

**Local:** csrc/integrate_recursive/gus_integrate_recursive.c:15 · **Categoria:** integração · **Encontrado por:** C — solver; Interface Python↔C

O PROG_HELP declara a sintaxe '-hints {H_FNI_NAME} [ scale {H_SZ} ] weight {H_WT}', mas tire_parse_options (linhas 744-748) lê o peso diretamente com argparser_get_next_double após o nome do arquivo/scale, SEM consumir nenhum token 'weight' (o PROG_INFO_OPTS na linha 218 mostra a forma correta, sem 'weight'). O integrate.py segue o parser (envia '-hints {file} {peso}'), então o pipeline funciona; mas quem montar a chamada seguindo o SYNOPSIS terá o token 'weight' interpretado como número, abortando o programa. É um mismatch documentação↔parser na interface CLI.

```c
"    [ -hints {H_FNI_NAME} [ scale {H_SZ} ] weight {H_WT} ] \\\n" \
```

**Impacto:** Sem efeito no pipeline atual (integrate.py:131 já omite 'weight'); risco de erro de uso para chamadas manuais/futuras seguindo o help.

**Sugestão de correção:** Remover a palavra 'weight' do PROG_HELP (linha 15) para alinhar com o parser e com o PROG_INFO_OPTS, ou fazer o parser aceitar o keyword opcional.

**Verificação:**
- Matemática: confirmado — argparser_get_next_double aborta com 'should be a number' ao receber o token 'weight'; inconsistência interna com o PROG_INFO_OPTS.
- Literal: confirmado — parser lê o peso diretamente; mismatch documentação↔parser real.
- Impacto: refutado — único chamador do binário segue o parser; cenário manual hipotético falharia ruidosamente; inconsistência cosmética entre strings de documentação.

#### BUG-028 — Parâmetro wch de tire_read_fni_file é ignorado; canal de peso hardcoded como 1 na expansão

**Local:** csrc/integrate_recursive/gus_integrate_recursive.c:705 · **Categoria:** integração · **Encontrado por:** C — solver

A documentação de `tire_read_fni_file` (linhas 363-374) promete que, ao expandir um mapa de células para vértices, o canal `wch` é tratado como peso. Na implementação, `wch` nunca é usado (confirmado pelo compilador: warning "unused parameter 'wch'"); a chamada `float_image_expand_by_one(I, 1)` fixa o canal de peso em 1. Hoje coincide com os usos efetivos (hints e reference são lidos com wch=1; normais/slopes são lidos com NX=NY=-1 e nunca expandem), mas a interface mente: se um mapa de normais (wch=3) ou slopes (wch=2) precisasse ser expandido, o canal 1 (Ny ou dZdY) seria usado como peso na média harmônica, corrompendo valores silenciosamente.

```c
float_image_t *II = float_image_expand_by_one(I, 1);
```

**Impacto:** Nenhum efeito numérico nas chamadas atuais (zMos_with_confidence com 2 canais, ch1=confiança, casa com o hardcode). Bug latente de interface: qualquer chamada futura com wch != 1 e expansão ativa produziria pesos errados sem erro.

**Sugestão de correção:** Usar o parâmetro: `float_image_expand_by_one(I, wch)`.

**Verificação:**
- Matemática: confirmado — wch ignorado e semântica de float_image_expand_by_one verificadas; expansão com wch=2/3 usaria dZ/dY ou Ny como peso.
- Literal: confirmado — contrato documentado violado; chamadas atuais não afetadas (bug latente de interface).
- Impacto: refutado — o branch de expansão é inalcançável para wch≠1 e as chamadas que expandem usam wch=1 (idêntico ao hardcode); resultado bit-a-bit igual.

#### BUG-029 — Ordenação lexicográfica (sorted) em vez de natsorted para diretórios zf: pilha pode desalinhar da lista z_foc

**Local:** src/hybrid_stereo_method/hybrid/main.py:77 · **Categoria:** integração · **Encontrado por:** Contratos de dados

zf_directories (linha 77) e os filtered_files por luz (linhas 90 e 125-127) usam sorted() lexicográfico, enquanto o contrato com multifocus exige que a ordem das imagens da pilha corresponda elemento a elemento à lista z_foc crescente do YAML (multifocus/main.py:132-139 só valida o COMPRIMENTO). Com nomes não zero-padded (ex.: zf15, zf105, zf25), a ordem lexicográfica difere da ordem numérica (zf105 < zf15 < zf25), embaralhando silenciosamente o mapeamento frame→z_foc. O dataset atual usa nomes padded (zf015.0000-df020.0000...), então o bug está dormente, mas o próprio arquivo importa natsorted (linha 7) e só o usa para sMos_path_list (linha 147).

```python
zf_directories = sorted(
        set(
            [
                os.path.basename(os.path.dirname(path))
```

**Impacto:** Com datasets de nomenclatura não padded, cada frame recebe a distância focal de outro frame: zMos fica fisicamente errado (e os hints da integração junto), sem nenhum erro — o check de tamanho passa.

**Sugestão de correção:** Usar natsorted() em zf_directories e em todos os filtered_files (linhas 77, 90, 125), garantindo ordem numérica consistente com z_foc.

**Verificação:**
- Matemática: confirmado — zMos[i,j]=zFoc[K] indexa a lista pela posição do frame; "zf105"<"zf15" lexicograficamente, divergindo da ordem numérica.
- Literal: confirmado — sorted() vs natsorted verificados; pareamento posicional sem re-ordenação; dataset atual padded (latente).
- Impacto: refutado — todos os datasets do repo usam largura fixa zero-padded em que sorted ≡ ordem numérica; cenário não-padded é hipotético sem exemplar.

#### BUG-030 — Defaults no código divergem do YAML: initial_method 'hints' vs 'zero' e hints_weight 0.0 vs 0.1

**Local:** src/hybrid_stereo_method/hybrid/main.py:179 · **Categoria:** integração · **Encontrado por:** Hybrid; Fluxo de configuração

hybrid/main.py usa integration_params.get('initial_method', 'hints') enquanto hb_experiment.yaml define 'zero' (hb_experiment.yaml:111); e integration_params.get('hints_weight', 0.0) na linha 192 enquanto o YAML define 0.1 (hb_experiment.yaml:120). Se o bloco hybrid.integration for omitido/renomeado no YAML, o comportamento muda silenciosamente: hints_weight=0.0 faz os hints serem passados ao C com peso zero (sem efeito algum, sem aviso), e initial_method='hints' sem arquivo de hints aborta o binário (demand(H != NULL), gus_integrate_recursive.c:545).

```python
initial_method=integration_params.get("initial_method", "hints"),
```

**Impacto:** Configurações parcialmente preenchidas produzem pipeline com hints inertes (peso 0) ou abortam na integração, comportamento diferente do documentado no YAML de referência.

**Sugestão de correção:** Alinhar os defaults do código aos do YAML empacotado ('zero' e 0.1), ou falhar explicitamente quando hybrid.integration estiver ausente.

**Verificação:**
- Matemática: confirmado — divergências e abort do demand verificados; worktree paralelo já corrige exatamente isso, corroborando o bug.
- Literal: confirmado — combinação default internamente inconsistente; dataclass IntegrateRecursiveConfig usa 'zero', confirmando a anomalia em main.py:179.
- Impacto: refutado — a única config híbrida define todas as chaves, então os defaults nunca são consultados; pior caso seria falha ruidosa, não corrupção silenciosa.

#### BUG-031 — print_img_statistics: overflow silencioso de uint8/uint16 no cálculo do RMS

**Local:** src/hybrid_stereo_method/infrastructure/utils.py:23 · **Categoria:** matemática · **Encontrado por:** Infrastructure

`img**2` é calculado no dtype original da imagem. Para uint8, a potência satura/enrola módulo 256 (confirmado empiricamente: img cheio de 200 → img**2 máximo 64 em vez de 40000), então o RMS impresso é lixo. O mesmo ocorre com uint16 para valores > 255. v_min/v_max/mean estão corretos (np.average promove a float), mas rms não.

```python
rms = np.sqrt(np.average(img**2))
```

**Impacto:** Estatísticas de diagnóstico (rms) silenciosamente erradas para todas as imagens uint8/uint16 lidas com info=True — pode mascarar problemas radiométricos reais durante depuração do pipeline (justamente o tipo de bug que existe no save_image).

**Sugestão de correção:** Converter antes: `imgf = img.astype(np.float64)` e calcular rms/v_dev sobre imgf.

**Verificação:**
- Matemática: confirmado — wraparound módulo 256/65536 reproduzido empiricamente (rms 8.0 em vez de 200.0); caminho de dados real via read_image com IMREAD_UNCHANGED.
- Literal: confirmado — único caller ativo passa o resultado cru do cv2.imread sem conversão.
- Impacto: confirmado — info=True hardcoded nos mains; o dataset da config existente contém PNGs uint16 que disparam o overflow em toda execução realista (apenas diagnóstico, severidade baixa adequada).

#### BUG-032 — calculate_avarage_of_images trunca em vez de arredondar ao voltar para uint8/uint16

**Local:** src/hybrid_stereo_method/infrastructure/utils.py:84 · **Categoria:** matemática · **Encontrado por:** Infrastructure

A média é corretamente calculada em float32, mas `mean_image.astype(np.uint8)` TRUNCA a parte fracionária (confirmado: média de 100 e 101 → 100, não 101 ou arredondamento). Isso introduz um viés sistemático de -0,5 LSB nas imagens médias por zf usadas no modo hybrid do multifocus.

```python
return mean_image.astype(np.uint8)
```

**Impacto:** Viés de meio nível de cinza nas imagens average_{zf}.png que alimentam o indicador de foco; pequeno, mas é erro numérico silencioso e sistemático (sempre para baixo).

**Sugestão de correção:** Usar `np.round(mean_image).astype(np.uint8)` (idem para uint16), ou retornar float e deixar a quantização para o ponto de gravação.

**Verificação:**
- Matemática: confirmado — cast C trunca (verificado experimentalmente); viés esperado −(N−1)/(2N) ≈ −0,5 LSB, sistematicamente negativo.
- Literal: confirmado — sem np.rint/round/+0.5 na função nem compensação posterior.
- Impacto: refutado — o componente sistemático é offset DC, aniquilado pelos indicadores passa-alta (máscara com 0 exato no DC; Laplaciano de soma zero) e idêntico em todos os níveis zf (argmax invariante); efeito neutro no resultado final com as configs atuais.

#### BUG-033 — Marcadores degenerados inconsistentes (n/2 vs 0) e mosaico ignora wSel

**Local:** src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:129 · **Categoria:** integração · **Encontrado por:** Multifocus — núcleo

Pixels sem informação de foco recebem dois marcadores diferentes: foco máximo == 0 retorna iSel = n/2 (frame do meio, linha 129) e ajuste convexo/plano retorna iSel = 0 (linha 185), ambos com conf = 0. O mosaic (mosaic.py) consome apenas iSel e nunca consulta wSel, então pixels inválidos produzem zMos = zFoc[n/2] (~70 nas configs) num caso e zMos = zFoc[0] (15) no outro — dois valores físicos arbitrários e diferentes para o mesmo conceito de 'sem dado', e sMos recebe intensidades de frames arbitrários (frame 0 ou frame central, possivelmente desfocados).

```python
if focus_values[k_max] == 0:
        return n / 2, 0
```

**Impacto:** Na integração os hints desses pixels têm peso 0 (protegidos), mas o sMos contaminado alimenta o fotométrico via sMos.png sem nenhuma máscara ou peso — o photometric stereo recebe intensidades de frames desfocados como se fossem válidas, e o zMos.png/zMos.fni (sem canal de confiança) exibe degraus artificiais entre 15 e ~70 em regiões sem textura.

**Sugestão de correção:** Unificar o marcador degenerado (ex.: sempre n/2 ou NaN com conf 0) e propagar wSel para os consumidores de sMos/zMos, ou documentar que esses mapas só são válidos onde wSel > 0.

**Verificação:**
- Matemática: confirmado — dois (na verdade três) marcadores distintos para a mesma condição degenerada; mosaic nunca consulta wSel; valores 75 vs 15 com as configs.
- Literal: confirmado — todos os pontos conferidos; sMos alimenta o fotométrico sem máscara (carregamento comentado).
- Impacto: confirmado — o branch convexo dispara rotineiramente em regiões de baixa textura e o branch foco==0 é alcançável com laplacian (test_ms.yaml); altera zMos.fni standalone e, via sMos, as normais do híbrido.

#### BUG-034 — Clamp do índice fuzzy para [0, n] em vez de [0, n-1]

**Local:** src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:192 · **Categoria:** matemática · **Encontrado por:** Multifocus — núcleo

O vértice da parábola é limitado com `k_fuzzy = max(0, min(n, k_fuzzy))`, mas os índices válidos de frame vão de 0 a n-1. O limite superior n corresponde a um frame inexistente (um passo focal além do último). A linha comentada acima (191) mostra que houve hesitação sobre o intervalo correto.

```python
k_fuzzy = max(0, min(n, k_fuzzy))
```

**Impacto:** iSel pode conter valores em (n-1, n]. No mosaic os ramos de clamp (i1 >= n_frames) absorvem o caso colapsando para o último frame, então o efeito direto em zMos é limitado a saturação; porém iSel.fni é gravado com normalize(iSel) (multifocus/main.py:147), e a presença de valores até n distorce a escala min-max do FNI para consumidores downstream, além de inconsistência com a semântica documentada de índice de frame.

**Sugestão de correção:** Usar `k_fuzzy = max(0.0, min(n - 1, k_fuzzy))` (ou o clamp comentado [-0.5, n-0.5] de forma consistente com o resto do pipeline).

**Verificação:**
- Matemática: confirmado — domínio contínuo válido é [0, n−1]; o código mistura convenções (inferior de uma, superior de nenhuma); caso alcançável com pico na borda superior da pilha.
- Literal: confirmado — clamp e linha comentada conferem; mosaic satura, mas normalize(iSel) é afetado.
- Impacto: refutado — em (n−1, n] o mosaic produz exatamente o mesmo resultado do clamp "correto" (bit a bit); iSel.fni não tem consumidores no repositório.

#### BUG-035 — cv2.Sobel(dx=1, dy=1) calcula derivada mista, não magnitude do gradiente

**Local:** src/hybrid_stereo_method/multifocus/depth_refinement.py:21 · **Categoria:** matemática · **Encontrado por:** Multifocus — núcleo

O custo unário do graph-cut usa `cv2.Sobel(gray_img, cv2.CV_32F, 1, 1)`, que computa a derivada cruzada de segunda ordem ∂²I/∂x∂y, e não o gradiente. Como medida de nitidez isso é matematicamente errado: bordas puramente horizontais ou verticais têm resposta zero, e a medida responde apenas a estruturas diagonais/quinas. O correto é calcular Sobel(1,0) e Sobel(0,1) separadamente e combinar (gx²+gy²). Observação: a multiplicação por unary_scale na linha 22 é anulada pelo normalize min-max global da linha 24 (no-op). Ressalva: este arquivo é código morto no estado atual (não é importado por nenhum main; `from utils import *` na linha 6 nem resolveria no layout de pacote atual), embora exista a chave multifocus.depth_refinement.enabled nos YAMLs.

```python
grad = np.exp(-(cv2.Sobel(gray_img, cv2.CV_32F, 1, 1) ** 2))
```

**Impacto:** Se o refinamento por graph-cut for reativado (a config depth_refinement.enabled já existe nos YAMLs), o custo unário ignorará bordas alinhadas aos eixos, e o depth map refinado selecionará frames errados nessas regiões — silenciosamente, pois a medida ainda produz números plausíveis.

**Sugestão de correção:** Substituir por `gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1); grad = np.exp(-(gx**2 + gy**2))` ao reativar o módulo (e corrigir o import).

**Verificação:**
- Matemática: confirmado — Sobel(1,1) = derivada mista (verificação numérica: resposta 0.0 exata em borda vertical pura); correção proposta (Tenengrad) correta.
- Literal: confirmado — código e no-op do unary_scale verificados; chave de config existe mas o módulo não é importado.
- Impacto: refutado — código morto: a flag enabled não é lida por nenhum código (flag órfã) e o entry point standalone tem paths hardcoded de outro projeto.

#### BUG-036 — Chamadas a save_image com 5 argumentos — assinatura da função aceita só 3

**Local:** src/hybrid_stereo_method/multifocus/image_alignment.py:195 · **Categoria:** integração · **Encontrado por:** Infrastructure

image_alignment.py chama `save_image(save_path, ref_save_as, reference_img, 0, 255)` (e idem nas linhas 213 e 216), mas a assinatura em infrastructure/io/image_io.py:114 é `save_image(save_path, save_as, img)` — sem parâmetros de faixa. Indica que a função mudou de interface (antes aceitava min/max explícitos, hoje força min-max automático) e este chamador não foi atualizado. O módulo não é importado por nenhum main atualmente (código morto), então o TypeError só aparecerá se o alinhamento for reativado.

```python
save_image(save_path, ref_save_as, reference_img, 0, 255)
```

**Impacto:** Se o alinhamento de imagens for reabilitado no pipeline, falha imediata com TypeError; além disso evidencia que a semântica antiga (gravar na faixa fixa 0-255 sem re-normalizar) foi perdida na refatoração de save_image — a mesma perda que causa o bug crítico de radiometria.

**Sugestão de correção:** Atualizar as 3 chamadas para a assinatura atual ou, melhor, restaurar em save_image um parâmetro opcional de faixa fixa (sem min-max) e usá-lo aqui e nos sMos/average.

**Verificação:**
- Matemática: confirmado — histórico git mostra a assinatura antiga com (v_min, v_max), em que (0,255) era identidade — semântica perdida na refatoração, como alegado.
- Literal: confirmado — call sites incompatíveis com a interface atual (nuance: a falha seria NameError antes do TypeError, pois utils não exporta save_image).
- Impacto: refutado — módulo inalcançável (nenhum import, nenhum entry point/config); nota de higiene, não bug manifestável.

#### BUG-037 — Clip no percentil 1 GLOBAL achata curvas de foco fracas e contradiz o comentário

**Local:** src/hybrid_stereo_method/multifocus/indicators/applicator.py:87 · **Categoria:** matemática · **Encontrado por:** Multifocus — indicadores

`focus_indicator_stack = np.clip(focus_indicator_stack, p1, np.inf)` usa o percentil 1 calculado sobre o stack inteiro como PISO. O comentário acima diz 'Only clip strictly below zero if needed', mas como todos os indicadores são magnitudes (>= 0), p1 > 0 e o clip eleva o piso de todo o stack. Para pixels de baixa textura cuja curva focal inteira fica abaixo de p1 (~1% dos valores do stack por construção), a curva vira uma constante: no argmax_fuzzy o polyfit dá A≈0, cai no ramo 'plano' e retorna k_fuzzy=0 com conf=0 — em vez do pico fraco porém real. Pixels cuja janela de regressão (±r_max em torno do pico) cruza p1 têm a cauda da parábola achatada, deslocando o vértice -B/(2A) (estimativa subpixel) sem nenhum aviso.

```python
p1 = np.percentile(focus_indicator_stack, 1)
    focus_indicator_stack = np.clip(focus_indicator_stack, p1, np.inf)
```

**Impacto:** Pixels escuros/pouco texturizados recebem silenciosamente iSel=0 → zMos=zFoc[0] (15.0, o plano focal mais próximo) em vez da profundidade do pico fraco; e o índice subpixel fica enviesado onde a janela de ajuste toca o piso p1. O efeito é limitado (~1% dos valores), mas é resultado numericamente errado sem erro visível, propagado a zMos.fni e aos hints da integração híbrida (ainda que com confiança 0 no canal 1).

**Sugestão de correção:** Remover o clip ou trocá-lo por `np.clip(focus_indicator_stack, 0, np.inf)` (a intenção declarada no comentário). Se a remoção de outliers baixos for desejada, aplicá-la apenas para visualização, nunca antes do argmax/polyfit.

**Verificação:**
- Matemática: confirmado — fluxo traçado: curva constante após o clip cai no ramo |A| < polyfit_epsilon e retorna k_fuzzy=0/conf=0; viés subpixel trivialmente demonstrado.
- Literal: confirmado — clip com piso p1 global contradiz o comentário; normalização posterior não compensa; indicadores são magnitudes não-negativas.
- Impacto: confirmado — caminho ativo nas configs reais (fourier, p1 > 0); verificado numericamente (A≈5.5e-19 < epsilon; vértice deslocado 2.124→2.029); zMos.fni standalone afetado e viés subpixel entra nos hints com peso.

#### BUG-038 — create_stl_from_heightmap troca os eixos X/Y (scale_x aplicado ao índice de linha)

**Local:** src/hybrid_stereo_method/multifocus/utils.py:204 · **Categoria:** matemática · **Encontrado por:** Multifocus — alinhamento e pipeline

No loop, i indexa LINHAS (eixo Y da imagem) e j indexa COLUNAS (eixo X), mas os vértices são montados como [i*scale_x, j*scale_y, ...] — a escala X é aplicada ao índice de linha e a escala Y ao de coluna. A malha STL resultante é a transposta da imagem (reflexão pela diagonal), o que além de trocar largura/altura inverte a quiralidade da superfície e a orientação efetiva dos triângulos/normais. A função não é chamada por nenhum main atualmente (código exportável/auxiliar), mas o erro é puramente matemático e silencioso para quem a usar.

```python
v1 = [i * scale_x, j * scale_y, height_map[i, j] * scale_z]
```

**Impacto:** Qualquer STL exportado de um height_map sai espelhado/transposto em relação à imagem e ao mapa de altura do pipeline: com scale_x != scale_y as proporções físicas ficam erradas, e mesmo com escalas iguais o objeto sai refletido (quiralidade invertida) — inutilizável para comparação metrológica com o gabarito.

**Sugestão de correção:** Usar [j * scale_x, i * scale_y, height_map[i, j] * scale_z] (coluna→X, linha→Y) nos quatro vértices, e conferir a ordem dos vértices dos triângulos para manter as normais consistentes.

**Verificação:**
- Matemática: confirmado — o mapa (x,y,z)→(y,x,z) tem determinante −1 (reflexão), nenhuma rotação rígida corrige; convenção do próprio arquivo (plot comentado) é coluna=x.
- Literal: confirmado — i sobre linhas recebe scale_x sistematicamente nos 4 vértices, sem transposição compensatória.
- Impacto: refutado — função sem chamadores em todo o repositório (código morto); a troca é, no limite, questão de convenção sem comparação existente no pipeline.

#### BUG-039 — Parâmetro scale (data_scale do YAML) só é aplicado quando a imagem tem 3 canais — .npy 2D são silenciosamente não escalados

**Local:** src/hybrid_stereo_method/photometric/ps_utils.py:104 · **Categoria:** integração · **Encontrado por:** Photometric — solvers

A interface RPS.load_images/load_npyimages (rps.py:66-79) repassa scale=parameters.get('data_scale') para ps_utils.load_images/load_npyimages, mas em ambas as funções a multiplicação `im = im * scale` está DENTRO do bloco `if im.ndim == 3:` (load_images linhas 71-75, load_npyimages linhas 101-104). Um .npy já em escala de cinza (2D) entra no stack sem escala alguma. Se a pasta contiver mistura de arquivos 2D e 3D, as colunas de M ficam escaladas inconsistentemente entre direções de luz — exatamente o tipo de distorção radiométrica que os solvers PS não toleram. A docstring promete 'scaling factor to be multiplied to the image pixel after grayscale conversion', mas o código não cumpre para entradas 2D.

```python
if im.ndim == 3:
            # im = np.mean(im, axis=2)
            im = converter_npy_para_cinza(im)  # importar essa funcao
            im = im * scale
```

**Impacto:** Com stack homogêneo 2D o scale do YAML é silenciosamente ignorado (inócuo para as normais, pois são normalizadas, mas o data_scale configurado não tem efeito). Com stack misto 2D/3D, imagens de luzes diferentes entram em M com ganhos diferentes, produzindo normais erradas sem nenhum erro visível.

**Sugestão de correção:** Mover `im = im * scale` para fora do bloco `if im.ndim == 3:`, aplicando o fator incondicionalmente após a conversão para cinza, em load_images e load_npyimages.

**Verificação:**
- Matemática: confirmado — com stack misto, m = diag(g)·Lᵀn·ρ com ganhos distintos por coluna enviesa as normais; ganho uniforme é absorvido pelo albedo.
- Literal: confirmado — `im * scale` dentro do if em ambas as funções, sem reaplicação posterior; docstring não cumprida para 2D.
- Impacto: refutado — cv2.imread sempre retorna 3 canais; o único dataset .npy é 3D; todas as configs usam data_scale: 1.0 (identidade) — cenário danoso requer config/dataset inexistentes.

#### BUG-040 — calculate_gradient_consistency não calcula o curl: derivadas tomadas nos eixos trocados

**Local:** src/hybrid_stereo_method/photometric/ps_utils.py:303 · **Categoria:** matemática · **Encontrado por:** Photometric — WPS e utils

A função nomeia axis0 como x e axis1 como y, mas o canal 0 do gradient_map é a componente x do gradiente (direção da LARGURA = axis1, ver convert_normal_map_to_gradient_map que usa nx/nz). Logo DGxDy = (gradient_map[x, y+1, 0] - gradient_map[x, y-1, 0])/2 diferencia Gx ao longo de axis1, que é a própria direção x — isto é ∂Gx/∂x, não ∂Gx/∂y. Analogamente DGyDx = ∂Gy/∂y. O resultado é Zxx − Zyy em vez do curl ∂Gx/∂y − ∂Gy/∂x (que deveria ser ~0 para campos integráveis).

```python
DGxDy = (gradient_map[x, y + 1, 0] - gradient_map[x, y - 1, 0]) / 2
            DGyDx = (gradient_map[x + 1, y, 1] - gradient_map[x - 1, y, 1]) / 2
```

**Impacto:** A medida de 'consistência rotacional' retornada é uma quantidade sem relação com integrabilidade (diferença de curvaturas, não curl). Hoje a função não é chamada em nenhum main, mas qualquer uso futuro para validar/ponderar normais antes da integração daria diagnóstico silenciosamente errado.

**Sugestão de correção:** Trocar os eixos das diferenças finitas: DGxDy = (gradient_map[x+1, y, 0] - gradient_map[x-1, y, 0])/2 e DGyDx = (gradient_map[x, y+1, 1] - gradient_map[x, y-1, 1])/2 (com axis0 = y de imagem), documentando a convenção de eixos.

**Verificação:**
- Matemática: confirmado — convenção de canais do repositório estabelecida (canal 0 = direção da largura); o retorno é Zxx − Zyy, não o curl.
- Literal: confirmado — índices e convenção verificados; sem transposição compensatória.
- Impacto: refutado — função sem nenhum chamador no repositório (código morto); impacto puramente hipotético.

#### BUG-041 — convert_normal_map_to_gradient_map sem o sinal negativo da conversão normal→gradiente

**Local:** src/hybrid_stereo_method/photometric/ps_utils.py:325 · **Categoria:** matemática · **Encontrado por:** Photometric — WPS e utils; Convenções geométricas

A conversão correta de normal unitária para gradiente de altura é p = dZ/dX = -nx/nz e q = dZ/dY = -ny/nz (como faz o C em pst_basic.c:50-51). A função calcula +nx/nz e +ny/nz, invertendo o sinal de ambos os componentes do gradiente. Também não há proteção para nz≈0. Atualmente nenhum main importa essa função (código não exercitado pelo pipeline híbrido), mas ela é a única conversão normal→gradiente do lado Python e tende a ser usada em análises/validações.

```python
gradient_map[x, y, 0] = normal_map[x, y, 0] / normal_map[x, y, 2]
```

**Impacto:** Qualquer consumidor dessa função (p.ex. validação de consistência rotacional com calculate_gradient_consistency, ou uma futura integração em Python) obterá um campo de gradiente com relevo invertido (côncavo↔convexo) silenciosamente.

**Sugestão de correção:** Trocar para gradient = -n[...,0]/n[...,2] e -n[...,1]/n[...,2], com clamp de nz (como o maxSlope do C), e documentar a convenção de eixos.

**Verificação:**
- Matemática: confirmado — derivação independente: normal ∝ (−p,−q,1); o sinal negativo é invariante sob rotação/reflexão do plano XY, então +nx/nz não é correto sob nenhuma convenção com nz>0.
- Literal: confirmado — código confere; a conversão de produção (C) usa −nx/nz com clamp; sem compensação.
- Impacto: refutado — função nunca chamada por nenhum módulo (código morto); sem manifestação possível no pipeline atual.

#### BUG-042 — Validação de b em L1_residual_min usa 'and' em vez de 'or' — b com múltiplas colunas passa e o IRLS pondera todas as colunas pelo resíduo da primeira

**Local:** src/hybrid_stereo_method/photometric/solvers/numerics.py:27 · **Categoria:** matemática · **Encontrado por:** Photometric — solvers

A checagem `if np.ndim(b) != 2 and b.shape[1] != 1:` nunca rejeita um b 2D com mais de uma coluna (a primeira condição já é False). Se b for (m,k) com k>1, o lstsq resolve k sistemas, mas a atualização de pesos na linha 42 usa apenas a coluna 0 dos resíduos para ponderar TODAS as colunas, e o critério de parada `np.linalg.norm(x - xold)` mistura as colunas via broadcast com xold (n,1). O resultado para as colunas 1..k-1 é um IRLS com pesos errados, devolvido sem qualquer erro. (Herdado do upstream; no pipeline atual rps.py sempre passa b (f,1), então o caso só ocorre se a função for reutilizada com múltiplos pixels em lote.)

```python
if np.ndim(b) != 2 and b.shape[1] != 1:
        raise ValueError("b needs to be a column vector m x 1")
```

**Impacto:** Uso da função com b multi-coluna (por exemplo, vetorizar vários pixels numa chamada) produz normais erradas silenciosamente para todas as colunas exceto a primeira, em vez de levantar o ValueError pretendido.

**Sugestão de correção:** Trocar para `if np.ndim(b) != 2 or b.shape[1] != 1:` (e fazer essa validação antes de usar b), garantindo que apenas vetores-coluna m x 1 sejam aceitos.

**Verificação:**
- Matemática: confirmado — short-circuit do 'and' analisado; IRLS exige pesos por elemento de cada coluna, o [:, 0] aplica a coluna 0 a todas; b 1D produziria IndexError em vez do ValueError pretendido.
- Literal: confirmado — código literal confere; chamadores atuais sempre passam (f,1), bug latente.
- Impacto: refutado — únicos chamadores constroem b sempre (f,1); cenário multi-coluna é hipotético — hardening de validação defensiva.

#### BUG-043 — estimate_normals_argmax_lstsq: resíduo do lstsq pode ser vazio e confidence = 1/residuals gera inf/NaN

**Local:** src/hybrid_stereo_method/photometric/wps.py:96 · **Categoria:** matemática · **Encontrado por:** Photometric — WPS e utils

np.linalg.lstsq retorna o resíduo como array de tamanho 0 quando o sistema tem 3 equações (num_images == 3) ou posto deficiente — a atribuição residuals[i, j] = residual então falha (ou, em versões do numpy que aceitam, grava lixo). Além disso, na linha 101, confidence = 1 / residuals divide por zero para ajustes perfeitos (residual = 0), produzindo inf; a normalização min-max subsequente (linhas 102-104) com inf no máximo transforma TODOS os outros valores de confiança em 0 (ou NaN via inf-inf), silenciosamente.

```python
residuals[i, j] = residual
...
    confidence = 1 / residuals
```

**Impacto:** Com qualquer pixel de ajuste exato, o mapa de confiança inteiro colapsa para 0/NaN sem aviso; com 3 luzes o código quebra. A função está comentada em main_wps.py:131, mas é uma das três variantes oferecidas pelo módulo e seria escolhida sem que o usuário perceba o defeito.

**Sugestão de correção:** Calcular o resíduo explicitamente (np.linalg.norm(selected_lights @ normal - selected_values)) em vez de confiar no retorno do lstsq, e usar confidence = 1/(1+residuals) (como na variante robusta) para evitar divisão por zero antes da normalização.

**Verificação:**
- Matemática: confirmado — reproduzido empiricamente com numpy 2.2.6: shape (0,) com m<=n/posto deficiente → ValueError; colapso [[nan, 0],[0, 0]] na normalização com inf.
- Literal: confirmado — reprodução idêntica; com exatamente 3 luzes (configuração clássica de PS) a função quebra.
- Impacto: refutado — variante código morto: main_wps chama hardcoded a versão robusta; não há dispatch por configuração; ativar exigiria editar o fonte.

## Cobertura

**Unidades auditadas (12 finders, todos concluíram):**

| # | Unidade |
|---|---------|
| 1 | Multifocus — núcleo |
| 2 | Multifocus — indicadores |
| 3 | Multifocus — alinhamento e pipeline |
| 4 | Photometric — solvers |
| 5 | Photometric — WPS e utils |
| 6 | Hybrid |
| 7 | Infrastructure |
| 8 | C — solver |
| 9 | Interface Python↔C |
| 10 | Convenções geométricas |
| 11 | Fluxo de configuração |
| 12 | Contratos de dados |

**Falhas de agentes:** nenhum finder falhou; 0 agentes de mapa falharam; 0 verificadores falharam.

**Volume de achados:** 81 achados brutos → 49 achados únicos após deduplicação (32 merges). Dos 49 únicos, 43 foram confirmados (≥2 de 3 lentes) e 6 refutados (Apêndice A); 0 ficaram com verificação incompleta.

**Notas dos finders:**

- **Multifocus — núcleo:** Todos os quatro arquivos da unidade foram lidos integralmente: argmax_fuzzy.py (225 linhas), depth_refinement.py (85 linhas), math_utils.py (83 linhas), mosaic.py (94 linhas). Também foram lidos como contexto multifocus/utils.py, multifocus/main.py e as seções relevantes de configs/hb_experiment.yaml e ms_experiment.yaml. O bug da janela de cosseno foi confirmado numericamente. Não auditados (fora da unidade): indicators/applicator.py (normalização do indicador de foco), infrastructure/io/image_io.py e o lado hybrid/photometric — cobertos por outros agentes conforme o mapa de convenções. WEIGHTS e weighted_median_filter de depth_refinement foram verificados (soma 81, medIdx 40 — corretos).

- **Multifocus — indicadores:** Todos os 5 arquivos da unidade foram lidos integralmente, além dos consumidores diretos (multifocus/main.py, argmax_fuzzy.py, mosaic.py, multifocus/utils.py) e dos YAMLs hb/ms_experiment. Verificações que NÃO geraram achados: (1) non_linear_res.py é código morto — não é importado por nenhum módulo nem selecionável no applicator; auditada sua matemática mesmo assim: a matriz X pareia as coordenadas transpostas em relação ao flatten da vizinhança, mas como o conjunto de pontos é simétrico sob troca x↔y, o resíduo Q do ajuste planar e o F final são invariantes — não há erro numérico efetivo; (2) fourier.py: máscara gaussiana passa-alta `1 - g_y*g_x` é matematicamente correta (separável = exp(-(dx²+dy²)/2)), centro em N//2 coincide com o DC pós-fftshift para dimensões pares, par fftshift/ifftshift consistente; create_binary_elliptical_mask e apply_weighted_filter são código morto; (3) laplacian.py: linha 26 contém um literal `0` morto (inócuo) e a divisão /255.0 assume uint8 (entradas uint16 sairiam em escala 0-257, mas a normalização global do stack absorve, sem efeito no argmax); (4) applicator.py: kernel de suavização /16 corretamente normalizado; medianBlur float32 com ksize=5 é suportado; ordem dos argumentos posicionais em multifocus/main.py:112-121 confere com a assinatura; método desconhecido no YAML causaria TypeError visível (não silencioso), então não reportado; (5) ordem dos coeficientes de pywt.wavedec2 confirmada empiricamente com script Python. Não auditados argmax_fuzzy.py/mosaic.py em profundidade (outra unidade), apenas o suficiente para rastrear o impacto downstream dos indicadores.

- **Multifocus — alinhamento e pipeline:** Lidos integralmente: image_alignment.py, multifocus/utils.py, multifocus/main.py (unidade designada), além de hybrid/main.py, infrastructure/io/image_io.py e infrastructure/utils.py para auditar a integração, e inspeção da estrutura real do dataset configurado (data/raw/hybrid_stereo/2025-03-08-stQ-...) para validar a hipótese de ordenação zf*/L* (nomes zero-padded mitigam o achado no dataset atual). Não auditados (unidades de outros agentes): argmax_fuzzy.py, mosaic.py, indicators/applicator.py, depth_refinement.py (declarado morto no mapa), photometric/* e csrc/*. Observação não reportada como achado por falhar ruidosamente (fora do escopo): em multifocus/main.py:29 o modo hybrid define data_path sem o data_folder, então reference_images_path aponta para local errado e, com settings.gabaritos=True, a linha 172 lança IndexError; e o filtro de substring '"av" not in file' em hybrid/main.py:148 zera a lista de sMos se qualquer componente do path de saída contiver 'av' (não ocorre nos paths atuais).

- **Photometric — solvers:** Arquivos da unidade lidos integralmente: numerics.py (178 linhas) e rps.py (309 linhas). Ambos foram comparados token a token com o upstream yasumat/RobustPhotometricStereo: são funcionalmente idênticos ao original, exceto pelo parâmetro `scale` adicionado em load_images/load_npyimages. Verificados explicitamente a montagem dos sistemas L2, L1/IRLS, SBL e RPCA/ALM inexato — nenhum erro matemático encontrado nesses núcleos. A ausência de recuperação de albedo no RPS é por projeto, igual ao upstream. Para contexto de integração lidos também photometric/main.py, ps_utils.py e wps.py; wps.py NÃO faz parte desta unidade (observação lateral sobre o albedo encaminhada à unidade responsável). A convenção de eixos de lights.npy permanece indeterminável a partir destes arquivos; nenhum dos caminhos normaliza os vetores de luz, o que só é correto se lights.npy contiver vetores unitários (os gerados por ps_utils.light_direction são unitários).

- **Photometric — WPS e utils:** Arquivos da unidade lidos integralmente: wps.py, ps_utils.py, main.py, main_wps.py. Verificados nas fronteiras (somente trechos relevantes): infrastructure/io/image_io.py, infrastructure/utils.py, hybrid/main.py e configs hb/ps/wps_experiment.yaml. Não auditados: rps.py e visualization.py (fora da unidade designada). Não foi possível verificar, por depender do dataset, a correspondência de ORDEM entre as linhas de lights.npy e a lista natsorted de sMos.png por direção L* (hybrid/main.py:147-151) — se a ordem dos diretórios L* não casar com a ordem das linhas de lights.npy, as normais saem erradas silenciosamente; recomenda-se verificação manual. A convenção de eixos de lights.npy também não é verificável apenas pelo código.

- **Hybrid:** Lidos integralmente: hybrid/integrate.py e hybrid/main.py (unidade de revisão), e ainda, para verificação cruzada: gus_integrate_recursive.c (completo), image_io.py (completo), main_wps.py e wps.py (completos), configs/hb_experiment.yaml, e trechos de multifocus/main.py. Verificações empíricas: formato FNI gerado pelo C é compatível com read_fni_to_image_array; faixas de valores hints [15,125] vs Z final [-1439,394]; lights.npy do dataset configurado tem shape (12,3) com Nz>0 e as duas cópias no dataset são idênticas; diretórios zf/L do dataset são zero-padded. Não auditados: internals da libpst além das interfaces; a convenção Y de lights.npy não está documentada no repo, então o achado do sinal de dZ/dY é inferido da origem Stolfi do dataset e do próprio aviso no help do C — recomenda-se confirmação com superfície sintética. A montagem do token 'scale' após '-normals' foi verificada contra o parser C e funciona; não reportada como bug.

- **Infrastructure:** Lidos integralmente: image_io.py, infrastructure/utils.py, infrastructure/visualization.py. src/hybrid_stereo_method/core/ contém apenas __init__.py vazios — nada a auditar. visualization.py é só exibição (BGR→RGB correto, cmap 'grey' válido); sem achados numéricos. Verificada compatibilidade do parser Python com um FNI real escrito pelo binário C (NC=2 parseia corretamente; o bug de NC=1 confirmado por teste executado). Serialização de NaN como '+nan' é aceita por float() do Python e por strtod do C, mas o parser C não foi testado de fato com arquivo contendo '+nan'. Não auditado o restante da lib C (apenas pst_normal_map.h para confirmar a convenção Y-up e o formato de float_image). Achados de produção nesta unidade têm consumidores em multifocus/photometric/hybrid — sobreposições com outros agentes esperadas (D1/D3 do mapa de convenções).

- **C — solver:** Lidos integralmente: gus_integrate_recursive.c e, em lib-src/: pst_integrate.c, pst_integrate_iterative.c, pst_integrate_recursive.c, pst_imgsys.c, pst_imgsys_solve.c, pst_slope_map.c, pst_height_map.c, além dos módulos de suporte no caminho do solver (pst_basic.c, pst_normal_map.c, pst_map.c, pst_interpolate.c, pst_cell_map_shrink.c, pst_vertex_map_shrink.c, pst_vertex_map_expand.c, float_image_expand_by_one.c e float_image_rescale_samples). Verificados e considerados corretos: montagem dos mínimos quadrados, coeficientes de interpolação/extrapolação, termos diagonais, consistência de escalas do multigrid, ordem dos tokens da CLI emitida por integrate.py vs parser C, e fórmula de perturbação inicial. Não auditados linha a linha: float_image.c completo, float_image_mscale.c, pst_map_compare.c, pst_cell_map_clear.c/pst_vertex_map_clear.c (opção -clear não usada), argparser*.c, e os tests/ do Makefile. Observação: os symlinks de csrc/integrate_recursive/include/ apontam para um caminho inexistente, então o binário commitado pode não ser reproduzível a partir do código-fonte atual — questão de build, fora do escopo.

- **Interface Python↔C:** Lidos integralmente: hybrid/integrate.py, gus_integrate_recursive.c, image_io.py, hybrid/main.py, main_wps.py, wps.py, setup.py, Makefile. Lidos parcialmente (trechos relevantes à interface): pst_basic.c, pst_normal_map.c/.h, pst_integrate.c, pst_height_map.c, fget.c, filefmt.c, float_image.c, float_image_expand_by_one.h, multifocus/main.py (130-170), hb_experiment.yaml (integration). Verificações que PASSARAM: formato FNI do Python byte-compatível com float_image_read do C; NaN '+nan' aceito por fget_double e vira peso 0; ordem dos argumentos CLI confere com tire_parse_options; expansão (H,W)→(H+1,W+1) dos hints trata o canal 1 como peso corretamente; nome '{prefix}-00-end-Z.fni' confere com os artefatos de teste. Não auditados linha a linha: pst_integrate_iterative.c, pst_integrate_recursive.c, pst_imgsys*.c, pst_slope_map.c, pst_map_compare.c e demais arquivos do solver interno; o binário pré-compilado e libgus.a podem divergir dos fontes commitados (não verificável por inspeção). Nota: read_fni_to_image_array falha com ValueError visível ao ler FNI de 1 canal — hoje só FNIs de 2 canais são lidos, então não reportado por este finder.

- **Convenções geométricas:** Lidos integralmente: multifocus/{main,argmax_fuzzy,mosaic,math_utils,utils}.py, indicators/{applicator,fourier,laplacian}.py, photometric/{rps,ps_utils,wps,main_wps}.py, hybrid/{main,integrate}.py, infrastructure/{utils,io/image_io}.py, gus_integrate_recursive.c, configs/hb_experiment.yaml; lidos parcialmente: pst_basic.c, pst_normal_map.{h,c}. Não auditados: lib-src restante do C (solver Gauss-Seidel e redução multiescala assumidos corretos), photometric/main.py e solvers/numerics.py (caminho RPS, não usado no pipeline híbrido), photometric/visualization.py, multifocus/{image_alignment,depth_refinement,indicators/wavelet,indicators/non_linear_res}.py, demais YAMLs além do hb_experiment. A confirmação definitiva do achado de eixo Y depende da convenção (não documentada no repo) de lights.npy do dataset.

- **Fluxo de configuração:** Lidos por completo: todos os 6 YAMLs de configs/; multifocus/{main,argmax_fuzzy,mosaic,math_utils,utils}.py; indicators/{applicator,fourier,laplacian}.py; photometric/{main,main_wps,wps,rps,ps_utils}.py; hybrid/{main,integrate}.py; infrastructure/{utils,io/image_io}.py; gus_integrate_recursive.c e trecho de pst_normal_map.h. Não auditados: indicators/wavelet.py e non_linear_res.py (método 'wavelet' não usado nas configs), multifocus/{image_alignment,depth_refinement}.py (não importados pelos mains), photometric/{visualization,solvers/numerics}.py, infrastructure/visualization.py, lib-src/*.c do C além do header citado, tests/ e notebooks/. Observação: existe um worktree em .claude/worktrees/ com versões corrigidas de vários desses bugs (relatórios de auditoria anteriores); essas correções NÃO estão presentes na árvore/branch atual (fable_5), que foi o que se auditou. O finding do eixo Y (D3) e o de unidade dos hints (D4) dependem da convenção de lights.npy/datasets, não verificável só pelo código.

- **Contratos de dados:** Lidos integralmente: image_io.py, infrastructure/utils.py, multifocus/{main, argmax_fuzzy, mosaic, math_utils, utils, indicators/*}.py, photometric/{main_wps, wps, main, rps, ps_utils, visualization}.py, hybrid/{main, integrate}.py, gus_integrate_recursive.c (completo) e trechos relevantes de lib-src (pst_basic.c, pst_normal_map.c/.h, pst_integrate.c, pst_integrate_recursive.c, fget.c), configs hb/ms/ps/wps. Verificações executadas: round-trip FNI em Python (FNI de 1 canal quebra com ValueError — crash visível), pesos da janela quadrática (numérico), shape/conteúdo de lights.npy (12,3, Lz>0). Lacunas: solvers/numerics.py não auditado linha a linha; image_alignment.py, depth_refinement.py e non_linear_res.py não auditados (código morto); lib-src do C auditada apenas nos arquivos do caminho usado. Não-verificável por código: correspondência entre a ordem das linhas de lights.npy e a ordem natsorted dos diretórios L000..L011 (contrato implícito do dataset), e a convenção Y de lights.npy. Observações adicionais não reportadas como achados por serem crashes visíveis: chaves planas de photometric/main.py; np.load(filename=...) e mask indefinida em main_wps.py:163-164; resolução H/4 do wavelet; inconsistência de sinal de pst_normal_from_slope (fora do caminho usado pelo gus_integrate_recursive).

**Escopo excluído:** lib-src vendorizada (exceto os arquivos do solver listados nas notas), estilo/qualidade geral de código, robustez genérica.

## Apêndice A — Achados refutados na verificação

#### Confiança conf = |A|/fnoc explode quando fnoc → 0 e colapsa o mapa wSel após min-max

**Local:** src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:201

**Alegação:** A confiança |A|/fnoc explodiria para fnoc→0+ em regiões escuras/planas, e um único pixel outlier comprimiria via min-max a confiança de todos os pixels legítimos para ~0, invertendo a semântica do canal de peso dos hints.

**Motivo da refutação:** Refutado pelas três lentes; decisiva: **matemática** — pela ortogonalidade dos resíduos do ajuste WLS sobre dados não-negativos, fnoc·Σw² ≥ |A|·Σw²(x−x*)², acoplando |A| proporcionalmente a fnoc: a explosão é impossível. Verificação numérica massiva (420k+ amostras + otimização adversarial por differential evolution) mostrou conf efetivamente limitado a ~1.0; além disso o guard `abs(A) < polyfit_epsilon` zera exatamente o caso que o achado alegava passar. Sem outlier, o min-max não comprime nem inverte nada.

#### Pesos de importância passados a np.polyfit são elevados ao quadrado pelo solver

**Local:** src/hybrid_stereo_method/multifocus/argmax_fuzzy.py:118

**Alegação:** Como o parâmetro `w` do np.polyfit multiplica o resíduo não-quadrado, a ênfase efetiva dos pesos de foco seria quadrática em vez da linear "pretendida", puxando o vértice da parábola agressivamente demais para o pico de foco.

**Motivo da refutação:** O fato técnico sobre o np.polyfit é verdadeiro, mas refutado 2-de-3; decisivas: **literal** e **impacto** — a única intenção documentada (docstring: "higher focus values will have higher weights") é monotonicidade, preservada sob w²; não há especificação de ponderação linear em lugar algum, e ênfase quadrática é escolha de calibração plausível. Simulações com r_max=2 mostram deslocamento do vértice de apenas 0,0003–0,013 frame entre as duas ponderações — desprezível frente ao erro da própria aproximação parabólica.

#### normalize divide por zero silenciosamente em arrays constantes e propaga NaN para os FNIs

**Local:** src/hybrid_stereo_method/multifocus/utils.py:108

**Alegação:** Com wSel constante, (x−min)/(max−min) produziria array de NaN que, gravado no canal de confiança de zMos_with_confidence.fni, faria o integrador C "zerar silenciosamente" todos os hints, produzindo superfície aparentemente válida sem aviso.

**Motivo da refutação:** Refutado 2-de-3; decisivas: **matemática** e **impacto** — a premissa do descarte silencioso é falsa: o C executa `demand(isfinite(wH) && wH >= 0, "invalid hint weight")` (pst_integrate.c:309 / pst_basic.c:57), que aborta com exit(1); o Python usa subprocess.run(check=True) e relança RuntimeError logado. O modo de falha real é crash explícito e nenhuma superfície é gerada — não a corrupção silenciosa alegada. O gatilho (conf idêntica em todos os pixels da imagem) é além disso um cenário degenerado irreal.

#### Hints de profundidade passados ao solver C sem conversão de unidade/sinal (sem 'scale')

**Local:** src/hybrid_stereo_method/hybrid/integrate.py:131

**Alegação:** O comando `-hints {arquivo} {peso}` nunca emitiria o token `scale`, misturando no sistema linear unidades físicas de z_foc (15–125) com "altura por pixel" do Z integrado, possivelmente com sinais opostos — híbrido distorcido classificado como crítico.

**Motivo da refutação:** Refutado 2-de-3; decisivas: **matemática** e **impacto** — verificação empírica com o ground truth do dataset configurado mostrou que zMos cresce na MESMA direção da altura verdadeira (corr +0.85) e que, na convenção do gerador Stolfi do dataset (hs=01), 1 unidade de zFoc ≈ 1 pixel: o fator de escala correto é exatamente 1.0, o default do binário quando `scale` é omitido. A evidência citada (Z final [-1439, 394]) tem outra causa — outliers extremos de slopes com Nz≈0 na silhueta — e os hints na verdade puxam a solução PARA a faixa correta. A ausência de um campo hints_scale é, no máximo, limitação de extensibilidade.

#### Fallback silencioso para {prefix}-ini-Z.fni pode devolver o chute inicial como se fosse o resultado da integração

**Local:** src/hybrid_stereo_method/hybrid/integrate.py:194

**Alegação:** Se `{prefix}-00-end-Z.fni` não existisse, o Python cairia silenciosamente para `{prefix}-ini-Z.fni` (o chute inicial do C), devolvendo um mapa nulo como height map sem erro.

**Motivo da refutação:** Refutado 2-de-3; decisivas: **matemática** e **impacto** — o cenário é inalcançável: o C com exit 0 sempre escreve o end-Z (pst_imgsys_solve.c:124 chama reportSol(final=TRUE) incondicionalmente, inclusive sem convergência), todas as falhas usam demand/assert com exit ≠ 0, e o Python usa check=True, lançando RuntimeError antes do fallback. Um end-Z obsoleto de execução anterior seria lido pelo caminho primário, não pelo fallback. Código defensivo morto (code smell), não bug manifestável.

#### Default de -maxLevel usa a constante errada (DEFAULT_MAX_ITER em vez de DEFAULT_MAX_LEVEL)

**Local:** csrc/integrate_recursive/gus_integrate_recursive.c:783

**Alegação:** Sem `-maxLevel` na linha de comando, o parser atribuiria 100000 (DEFAULT_MAX_ITER) em vez de 30 (DEFAULT_MAX_LEVEL), divergindo do "default documentado".

**Motivo da refutação:** Refutado pelas três lentes; decisivas: **literal** e **matemática** — não existe "default 30 documentado": o help do próprio programa (linhas 118-120 e 265-269) diz que, sem `-maxLevel`, a recursão continua até o mapa reduzir a 1 pixel — exatamente o que o valor 100000 produz. Além disso 30 e 100000 são funcionalmente idênticos (a profundidade é limitada por log2 da dimensão da imagem; o comentário da própria constante a define como "qualquer valor maior que log2 da dimensão máxima"), e o pipeline Python sempre passa `-maxLevel`. Inconsistência cosmética de constante não usada.

