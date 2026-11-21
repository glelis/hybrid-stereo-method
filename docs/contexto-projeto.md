# Contexto do projeto — estéreo híbrido (EM + EF): dados de teste, gerador e specs

21/11/2026 · Gustavo · Exportado do doc https://claude.ai/artifact/L2ExWKwB7Di7RcVjLdAY9p (rev 29)

## Resumo

O projeto combina estéreo multifocal (EM: altura a partir da pilha focal) e estéreo fotométrico (EF: normais a partir de várias luzes). Os dados vêm da integração dos dois mapas por um integrador recursivo que usa a altura do EM como hint. Para testar isso é preciso uma cena com **pilha focal e várias luzes ao mesmo tempo**, além de GT de altura e de normais.

Hoje existem **3 conjuntos híbridos completos** (12 luzes × 12 focos + GT), todos renderizados pela biblioteca libmultifok, de Jorge Stolfi:

- **stQ-melon14 glo 0,00**, lambertiano com textura (o lamb-gls é o mesmo conjunto);
- **stQ-white01 glo 0,00**, lambertiano sem textura;
- **stQ-melon14 glo 0,50**, de 03/2025, semilustroso, com outros planos de foco.

O lustroso extremo (spots12, glo 0,90) foi publicado e se perdeu. O lustroso sem textura nunca foi gerado. Os dois podem ser gerados localmente compilando a libmultifok.

O restante de data/raw/ serve só para EF (POV-Ray, bunny, stB-noise01) ou só para EM (stacks sintéticos 320×240 e reais do microscópio MUFF). O inventário completo está na seção 2.

## 1. Onde ficam os dados

**Pasta canônica: data/raw/datasets/** (desde 21/11/2026). Ela reúne uma cópia de cada base de entrada, com nome padronizado, separada em hybrid/, photometric/, multifocus/, integration/, tools/ e references/. Traz também um README.md com as specs e um MANIFEST.tsv. Os dados locais entraram por hardlink: não ocupam espaço extra e as pastas antigas abaixo seguem intactas, por isso a pasta deve ser tratada como somente leitura. Os tgz originais estão em _archives/, e os downloads do GitHub e do Wayback, em _sources/ e _downloads_wayback/.

- data/raw/hybrid_stereo/: conjuntos híbridos e cenas POV-Ray. Tem várias duplicatas (seção 2.5).
- data/raw/hybrid_stereo_new/: cópias "limpas" usadas pelo pipeline atual (lamb-gls e 2025-03-08 glo 0,50).
- data/raw/photometric_stereo/: experimentos de EF (exNN_*, bunny*).
- data/raw/multifocus_stereo/: experimentos de EM (synthetic/scene_1–6, obj_1–3, new_objects, internet_focus_stack).
- ~/Downloads/: tgz originais de 2026 (lamb-gls, melon14, white01) e zwtest.tgz, também ligados por hardlink em datasets/_archives/. **O hardlink fica no mesmo disco e não é backup**, e o servidor de origem responde 403 desde pelo menos 11/2026. Convém fazer backup em outro disco ou na nuvem.

## 2. Inventário das imagens de teste

Na pilha focal, zf é a cota Z do plano de foco e df a profundidade de foco; "sharp" é o quadro tudo em foco. "Raios/px" = (2·hs+1)² × kr. Os PNG do libmultifok medem **512 × 422**: são 384 linhas de imagem mais 38 de escala de cinza no topo, que precisam ser cortadas.

### 2.1 Conjuntos híbridos (pilha focal + várias luzes)

| Conjunto | Local | Cena | Luzes | Focos (zf, df) | Material | Raios/px | GT |
|---|---|---|---|---|---|---|---|
| stQ-melon14-amb0.00-glo0.00 / 0512x0384-hs01-kr07 (07/2026; = lamb-gls de 04/2026) | hybrid_stereo_new/lamb-gls; hybrid_stereo/lamb-gls; ~/Downloads (lamb-gls.tgz e stQ-melon14-…tgz) | Q: piso + bola, pirâmide, cone e disco; 512×384 | 12 | 12: 7,2 → 288,8, passo 25,6; df 51,2 | lambertiano, textura melon14, amb 0 | 63 | sharp/ hAvg, sNrm, hDev (PNG 16 bits); o tgz melon14 traz também hAvg.fni e sNrm.fni |
| stQ-white01-amb0.00-glo0.00 / 0512x0384-hs02-kr07 (07/2026) | datasets/hybrid/ (extraído, com lights.npy); tgz em ~/Downloads | Q, sem textura | 12 | iguais aos do melon14 | lambertiano liso, amb 0 | 175 | idem, com FNI; traz um .tit com os focos e o horário do render |
| stQ-melon14-amb0.00-glo0.50 / 0512x0384-hs01-kr10 (03/2025) | hybrid_stereo_new/2025-03-08-…; hybrid_stereo/2025-03-08-… (+ cópias "(2)" e "(3)") | Q, textura melon14 | 12 | 12: 15 → 125, passo 10; df 20 | semilustroso (glo 0,50), amb 0 | 90 | sharp/ + hAvg.fni; derivados locais images_ps, organized_by_focus, media_iluminacao |
| stF-melon24-amb0.00-glo0.00 / 0512x0384-hs01-kr10 (02/2025) | hybrid_stereo/2025-02-09-stF-melon24-amb0-glo0 | F: objetos misturados, textura melon24 | 12 | 11: 26,25 → 93,75, passo 6,75; df 15 | lambertiano, amb 0 | 90 | sharp/ hAvg, sNrm, hDev; derivados images_mf e media_iluminacao |
| brasil_00 / 2017-06-07-1046 | hybrid_stereo/brasil_00 | real (microscópio MUFF/UFF) | 6 (L_00–L_05) | 35 quadros por luz | real | — | sem GT; 1280×1024 RGBA |

Atenção: a pasta 2025-03-08 tem "melon24" no nome, mas o conteúdo interno é stQ-melon14-amb0.00-glo0.50.

### 2.2 Só EF (várias luzes, um único foco)

| Conjunto | Local | Specs |
|---|---|---|
| stQ-melon24-amb0.00-glo0.50 (02/2025) | hybrid_stereo/2025-02-11-… | 512×384; 12 luzes; um único foco zf 75, df 22,5, mais sharp; hs01-kr10; GT hAvg e sNrm |
| stQ-melon24-amb0.25-glo0.50 (02/2025) | hybrid_stereo/2025-02-10-… | igual, com amb 0,25; tem também variantes amb 0,05 / glo 0,20 em L000 (zf 126, df 48) e em L005–L007 (zf 60, df 15) |
| stB-noise01 (02/2025) | photometric_stereo/ex23 | objeto único B, textura noise01; 512×384; 12 luzes; só sharp; amb 0; hs01-kr10; lights.npy e mask |
| POV-Ray povball / povpyra, txF e txT (01/2025) | hybrid_stereo/pov*; photometric_stereo/ex17–ex22 | 256×256; 12 luzes; esfera ou pirâmide sobre piso; ex19–ex22 trazem o gerador (make_images.py, main.pov, bcamlight.inc) e árvores out/ com 13 focos de −0,2 a 2,2. **Gama 2,2 e sem BRDF lambertiana: não usar como GT fotométrico** |
| bunny, bunny_2, bunny_lambert, bunny_lambert_noshadow, bunny_specular | photometric_stereo/ | 256×256; 50 luzes (lights.npy 50×3); gt_normal.npy; origem externa (não é do Stolfi) |
| ex1_muff, ex12–ex18_muff | photometric_stereo/ | experimentos de EF em recortes MUFF e em sintéticos: 1280×1024 (ex1, ex12), 256×256 (ex13–ex18), 400×100 (ex15). 12 luzes quando há lights.npy |

### 2.3 Só EM (pilha focal, uma luz)

| Conjunto | Local | Specs |
|---|---|---|
| stF-0320x0240-wavys-09 (10/2024) | multifocus_stereo/synthetic/scene_2 | bolas e discos numa caixa de 160×120×30 mm; ortográfico; df 6; zf 0,5 → 30,5, passo 3 (11 focos). Saídas -cs (quadro), -sh (sharp), -az (altura média; 0 = 0,5 mm, 255 = 30,5 mm), -dz (desvio) |
| stF-noise01-0320x0240-hs01-kr03 (30/10/2024) | synthetic/scene_6; scene_4 e scene_5 têm a mesma série | 320×240; 12 focos zf 27 → 93, passo 6, df 12, mais sharp; 27 raios/px; scene_5 traz sharp hAvg |
| scene_1, scene_3 | synthetic/ | 320×240, com gabaritos de altura e de variação; origem não identificada pelo nome (provavelmente stacks de 2024) |
| obj_1–obj_3, new_objects (7 sessões, 10/2018–01/2019) | multifocus_stereo/ | reais do microscópio MUFF; 1280×960 JPG; 30–749 quadros por sessão |
| internet_focus_stack | multifocus_stereo/ | stacks públicos: PCB 2048×1536, 540×540, lâminas 1024×1024 (4,7 GB) |
| zwtest (16/03/2025) | datasets/tools/zwtest (tgz em ~/Downloads) | 36 casos Z00–Z11 × W00–W02 (texto) para testar argmax e confiança do EM |

### 2.4 Publicados sem cópia local

Todos ficavam em https://www.ic.unicamp.br/~stolfi/temp/, que responde 403 em 21/11/2026.

| Arquivo | Data | O que era |
|---|---|---|
| stQ-spots12-amb0.00-glo0.90-0512x0384-hs01-kr07.tgz | 13/07/2026 | Cena Q totalmente lustrosa, com textura (seria o cenário C). **Perdido** |
| 2025-03-08-stQ-melon24-amb0.00-glo0.00.tgz | 09/03/2025 | Par lambertiano do conjunto glo 0,50 de 03/2025 |
| stR-0320x0240-wavys-09-2024-10-13.tgz | 13/10/2024 | Rampa para investigar a textura falsa fora de foco |
| img-bullsqr-0320x0240-hp0001/2/3-hr000x.tgz | 18/10/2024 | Teste de amostragem: 189, 1.725 e 4.753 raios/px |
| 2024-10-28-stF/stR-noise01-0320x0240-hs01-kr03.tgz | 28/10/2024 | Primeira versão do noise01 (a de 30/10 está em scene_6) |
| 2024-for-gustavo.tgz e 2024-for-gustavo-pov.tgz | 05/2024 | Figuras dos artigos TPAMI 2010 / TIP 2011 e cenas POV antigas |
| 18-cbramp-N-0032x0024-deb-Z.gif | 02/2025 | Animação da integração de um mapa de gradientes |
| …/microscope-uff/00-DATA/scans/objs/ | — | Repositório dos scans MUFF (de onde vem brasil_00) |

### 2.5 Duplicatas e limpeza

- lamb-gls.tgz e stQ-melon14-…-hs01-kr07.tgz são o mesmo render (md5 idêntico nos PNG). O melon14 só acrescenta os FNI e os .txt de debug.
- hybrid_stereo/lamb-gls, "lamb-gls (2)" e hybrid_stereo_new/lamb-gls guardam o mesmo conteúdo, com ou sem lights.npy e sharp/.
- hybrid_stereo/2025-03-08-… , "…glo0 (2).50", "…glo0 (3).50" e hybrid_stereo_new/2025-03-08-… são cópias do mesmo conjunto glo 0,50.

### 2.6 Achados no GitHub e no Wayback (21/11/2026)

O site ~stolfi continua respondendo 403. Para não perder nada, varri o índice do Internet Archive para ic.unicamp.br/~stolfi/ (cerca de 422 mil URLs) e os repositórios públicos JorgeStolfi. Nenhum dos tgz de 2024–2026 em temp/ chegou a ser arquivado. Apareceram, porém, dados que não tínhamos:

| Fonte | Conteúdo | Em datasets/ |
|---|---|---|
| C-IMG/gus_integrate_recursive/tests/in | cbabel e cplat3 (256×192, gradiente + altura sintéticos); fourg1 (512×422, MF-Z/PM-N do nosso pipeline + RF-Z ideal); fours1 (512×384, sharp do libmultifok) | integration/gus_integrate_recursive-tests-in |
| JSLIBS/libpst/tests/00-DATA/slope_to_height | 27 funções de teste (rampas, domo, cone, cbabel, cbramp, cplat3…) em G e Z | integration/libpst-slope_to_height |
| C-PST/00-DATA/slope_maps | funções 00–27 em 256² e 512², com G, N, W e Z (1,7 GB) | integration/cpst-slope_maps |
| Wayback: EXPORT/projects/photo-stereo (2011) | datasets-2011 (cbramp, dtface, hwave, sbabel, spdome, com e sem ruído, 64–512 px), integradores dz-to-z / dz_to_ht, relatório IC-10-09 | integration/stolfi-2011-photo-stereo-datasets; tools/integrators-2010-2011; references/ |
| C-IMG/multifok_analyze/tests/data | pilha focal real flower_01 (1280×1024, 25 quadros) + versão mono | multifocus/flower_01-multifok_analyze |
| C-IMG/multifok/tests/100_basics/in | brasil_00 reduzido (640×512, 50 quadros) + listas de teste | multifocus/real-muff/brasil_00-0640x0512-github |
| C-PST/00-DATA/fit_light_to_photo | esferas-gabarito (foto e sintético) para calibrar luzes | photometric/light-gauge-balls |
| JSLIBS/libmultifok/tests/00-DATA/texture | texturas do gerador (noise01–06 etc.) | tools/libmultifok-textures |
| Wayback temp/ | StolfiJ-PhotoStereoBucketing.pdf, artigo CVIU 2013 | references/ |

O fours1 e o fourg1 são a mesma cena Q do melon, já no formato do integrador. Servem de teste de regressão direto para o gus_integrate_recursive.

## 3. Como as imagens são geradas (libmultifok)

Os conjuntos st*-* vêm do traçador de raios distribuído em C de Stolfi: JSLIBS/libmultifok (GitHub público), programa tests/0100_scene_make_frame/test_mfok_scene_make_frame.c, chamado por make_images.py. A versão de 19/04/2025 desse script tem exatamente sceneType='Q', texture='melon14', HS=1, KR=7, ambient=0 e gloss=0. Ou seja, reproduz o melon14.

### 3.1 Nome do conjunto

Exemplo: `stQ-melon14-amb0.00-glo0.00/0512x0384-hs01-kr07/L000/zf058.4000-df051.2000/`

| Campo | Significado |
|---|---|
| stQ, stF, stB, stR | Tipo de cena. Q = 4 objetos (bola, pirâmide, cone, disco) em grade 2×2; F = objetos misturados; R = rampa; B/C/D/P = objeto único |
| melon14, melon24, white01, spots12, noise01 | Textura. melon14 = noise01×noise04; melon24 = noise02×noise04; white = lisa |
| ambA.AA / gloG.GG | Peso da luz ambiente e do brilho (0 = lambertiano; 0,90 = totalmente lustroso) |
| 0512x0384 | Resolução da imagem (sem a faixa de escala) |
| hsNN | Grade de (2·HS+1)² subamostras por pixel, com pesos em tenda |
| krNN | Raios por subamostra |
| L000–L011 | Índice da luz |
| zfZZZ.ZZZZ-dfDDD.DDDD | Plano de foco e profundidade de foco; sharp/ = tudo em foco |

### 3.2 Arquivos por quadro

| Arquivo | Conteúdo |
|---|---|
| sVal.png | Imagem observada (RGB). É a entrada |
| shrp.png | Nitidez por pixel = 1/vBlr, sem reescala |
| hAvg.png / .fni | Altura média dos raios do pixel. Em sharp/ é o GT de altura. PNG: Z ∈ [20, 276] → [0, 1] em 16 bits; FNI com 2 canais (o 2º é o peso) |
| hDev.png | Desvio da altura dentro do pixel: [0, 128] → [0, 1]. Marca bordas e descontinuidades |
| sNrm.png / .fni | Normal média. Em sharp/ é o GT de normais. PNG: [−1, 1] → [0, 1] por canal; FNI com 4 canais (o 4º é o peso) |
| pixel-rays-XXXX-YYYY.txt, pixel-data-XXXX-YYYY.txt, selected-pixels.png | Dump de debug dos pixels (107,171) e (277,171); colunas descritas em multifok_raytrace.h |

### 3.3 Geometria, câmera e luz

- Unidades de cena = pixels (ppu = 1). Cena Q: x ∈ [0, 512], y ∈ [0, 384], piso em Z = 20, zMax = 20 + 2·384/3 = 276; cada objeto fica num intervalo de Z diferente.
- Projeção ortográfica (raio central em (0, 0, −1)) e lente fina; os raios se inclinam até ~2° a partir do plano de foco.
- Focos 2026: NF = 12, passo (276 − 20)/10 = 25,6, df = 2 × passo. O primeiro e o último ficam meio passo fora da cena.
- A nitidez segue s ≈ 1/(a + b·(Z − zf)²); com df 51,2, a ≈ 0,259 e b ≈ 0,000814.
- **Luz pontual de cor (0,90; 0,85; 0,80)**, não branca. A ambiente é (0,8; 0,9; 1,0) × amb.
- Brilho: peso gloss × (0,3; 1,0; 0,5) por canal (verde mais lustroso). Lóbulo GLO_C·(r·v)²·(n·v); fração 1 − GLO_A·(n·l).
- Sombras: só a própria (n·l ≤ 0). Não achei código de sombra projetada; falta confirmar.
- **PNG gravado com Y invertido**: o pixel [0,0] fica embaixo à esquerda, daí sy = −1 no pipeline. A escala de cinza é acrescentada no topo.
- Custo: ~35 min por luz com hs02-kr07 (≈ 7 h por conjunto). Tamanhos: lamb-gls 223 MB, melon14 252 MB, white01 131 MB.

### 3.4 As 12 luzes padrão

Os tgz de 2026 não trazem lights.npy. O lights.npy local é idêntico ao dos conjuntos de 2025 e às direções fixas no código.

| Anel | Luzes | Elevação | Azimutes |
|---|---|---|---|
| Alto | L000–L005 | 35–43° | −44, −104, −162, 138, 72, 16 |
| Baixo | L006–L011 | 9–16° | −61, −150, 174, 119, 25, −7 |

### 3.5 Gerador POV-Ray (legado, 01/2025)

Está em photometric_stereo/ex19–ex22_povball-txF/*-glelis-pov/ (make_images.py, main.pov, bcamlight.inc, Makefile).

- Câmera em perspectiva; cena com Z ∈ [0, 2]; 13 focos de −0,2 a 2,2; quadros SHARP, MASK e ALB.
- Mesmas 12 direções de luz.
- Foi abandonado porque o POV-Ray não tem BRDF lambertiana e grava os PNG com gama 2,2.

## 4. Convenções e decisões de método

- Gradiente = (−nx/nz, −ny/nz). Se a bola virar sela, inverta só o Y. Calibrado para lamb-gls: sy = −1, pixel_size = 1,78.
- EM: use a média das 12 luzes em cada plano de foco, para reduzir o efeito das sombras. A confiança do EM é A/y(x), normalizada pelo percentil 90.
- EF robusto: descarte as imagens em sombra, depois rode MMQ iterativo com corte K·rms (K = 2 a 2,5). Confiança = (N/M)/hypot(1, rms). RPCA foi reprovado; Woodham com as 3 luzes mais fortes foi melhor.
- Integração: peso 0 nas bordas (Δnormal > 0,1). Pipeline: média das luzes → EM → mosaicos por luz → EF → integração com zMos como hint.
- Ao ler um PNG, sempre confira no numpy que a escala de cinza sai com degraus iguais (detecta gama).

## 5. Código e repositórios de referência

| Repositório | Para quê |
|---|---|
| JSLIBS/libmultifok | Gerador das cenas (seção 3) |
| C-IMG/gus_integrate_recursive | Integrador recursivo com hint do EM. Compilação: `( cd lib-src && make ) && make`; -initial é obrigatório |
| C-IMG/gus_fni_view, C-IMG/fni_view | Visualização 3D de FNI |
| JSLIBS/libimg (float_image.h) | Formato FNI; o cabeçalho deve dizer float_image_t |
| JSLIBS/libpst, libpstzin2 | Mapas de inclinação e altura, integração |
| glelis/thesis (privado; clone em ~/Documents/Projetos/thesis) | Tese; cap. 7 tem as tabelas de cenários A–H e A–D |
| Tese de Saracchini, cap. 8 | Integrador multiescala com dados incompletos |

## 6. Cobertura dos cenários de teste e próximos passos

| Cenário | Conjunto | Status |
|---|---|---|
| A — lambertiana com textura | stQ-melon14 glo 0,00 | Pronto e calibrado |
| B — lambertiana sem textura | stQ-white01 | Extraído em datasets/hybrid/ com lights.npy; falta calibrar |
| C — lustrosa com textura | spots12 (perdido); hoje melon14 glo 0,50 (03/2025) | Parcial: os focos e o kr diferem do A |
| D — lustrosa sem textura | — | Não existe |

- [ ] Fazer backup dos tgz (~/Downloads = datasets/_archives/) em outro disco e, depois de migrar o pipeline para datasets/, apagar as duplicatas de data/raw.
- [ ] Rodar scripts/calibrate_dataset.py no white01 (datasets/hybrid/stQ-white01-amb0.00-glo0.00-0512x0384-hs02-kr07).
- [ ] Compilar libmultifok e gerar C (melon14, gloss 0,90) e D (white01, gloss 0,90) com os mesmos focos e amostragem do A.
- [ ] Usar os FNI do sharp/ como GT, em vez dos PNG 16 bits, e o hDev como máscara de descontinuidades.
- [ ] Cortar as 38 linhas de escala de cinza e inverter Y ao ler os PNG.
- [ ] Confirmar se há sombra projetada no traçador.

## 7. Histórico resumido

| Período | Marco |
|---|---|
| 05/2024 | EQM aprovado: "Combinação Sinergística de Estéreo Fotométrico e Estéreo Multifocal" |
| 09–10/2024 | Dados MUFF reais; primeiros stacks sintéticos de EM (wavys, bullsqr, noise01); surge a nomenclatura hs/kr |
| 12/2024–01/2025 | Datasets de EF em POV-Ray; descoberto o problema de gama e de BRDF |
| 02/2025 | Traçador próprio (libmultifok): stB-noise01 e os primeiros melon24 |
| 03/2025 | Conjunto híbrido completo glo 0,50; zwtest; confiança do EM corrigida |
| 04/2025 | gus_integrate_recursive; convenções de sinal; primeiro teste híbrido completo |
| 04/2026 | lamb-gls (= melon14) |
| 07/2026 | melon14, white01 e spots12; definida a tabela de cenários A–D |
| 11/2026 | Servidor de Stolfi fechado (403); tese em revisão |
