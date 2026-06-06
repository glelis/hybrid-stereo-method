# Auditoria de corretude dos métodos — Relatório de achados

**Data:** 2026-06-05  **Spec:** docs/superpowers/specs/2026-06-04-method-audit-design.md

Este relatório consolida a varredura de corretude (matemática/teórica) e de implementação
(bugs/precisão/robustez) do pipeline híbrido **multifocus → fotométrico → integração C**.
Cada estágio tem suas notas completas (evidência verbatim, reproduções, linhas) nos arquivos
de `docs/superpowers/reports/audit-notes/`, referenciados em cada seção. Aqui constam entradas
compactas com ID, tipo, severidade, status, localização, evidência decisiva e sugestão de
correção. **Nenhuma correção foi aplicada** — o entregável é diagnóstico.

Convenção de severidade: **crítico** (corrompe o resultado científico) · **alto** (erro
mensurável) · **médio** (degrada robustez/precisão em casos comuns) · **baixo** (caso de borda).
Convenção de status: **confirmado** (por inspeção do código que prova o defeito, ou por
execução de teste) · **suspeita** (justificativa teórica não decidida por execução) ·
**refutado** (teste mostrou que a inconsistência não ocorre na via testada).

---

## 1. Sumário executivo

### 1.1 Contagem de achados por severidade × status

Total de IDs: **44** (MF 14, PS 11, INT 8, IO 6, CONV 5; CONV-3 não gerou achado). Destes, **2 são
refutados** e contados à parte, deixando **42 achados ativos**. "Confirmado" inclui confirmação por
inspeção (o código prova o defeito) e por execução de teste.

| Severidade | Confirmado | Suspeita | Total ativo |
|---|---|---|---|
| Crítico | 3 | 0 | **3** |
| Alto | 1 | 9 | **10** |
| Médio | 4 | 13 | **17** |
| Baixo | 11 | 1 | **12** |
| **Total ativo** | **19** | **23** | **42** |
| Refutado (à parte) | — | — | **2** |

Detalhamento exato (cada ID contado uma vez pela sua severidade/status final pós-testes):

- **Crítico (3, todos confirmados):** MF-01, MF-02 (por inspeção), MF-14 (por execução).
- **Alto (10):** confirmado — INT-01 (por inspeção, ramo de aborto). Suspeita — MF-03, MF-04, MF-09, PS-02, PS-06, IO-05, INT-04, CONV-4, CONV-6.
- **Médio (17):** confirmado — PS-03 (teste), INT-02 (corrigido `6728745`), INT-03 (corrigido `bdcb126`), INT-05 (corrigido `1bcefd4`), INT-06 (mitigado `c577632`), INT-08 (corrigido `7214d43`), IO-02 (inspeção), IO-04 (inspeção). Suspeita — MF-05, MF-06, MF-07, MF-08, MF-12, PS-05, PS-07, PS-08, CONV-5.
- **Baixo (12):** confirmado — PS-01 (teste), PS-04 (inspeção de fluxo/risco), PS-09, PS-10, PS-11, MF-10, MF-11, INT-07, IO-01, IO-03, IO-06 (por inspeção). Suspeita — MF-13.
- **Refutado (2):** CONV-1 e CONV-2 — a inconsistência de **sinal/orientação do integrador** foi refutada por teste (`-normals`, rampa assimétrica); a **metade física** de ambos (sinal-z real / frame y-up POV-Ray das luzes) **permanece em aberto** — não decidível por dados sintéticos.

### 1.2 Os achados mais importantes

1. **MF-14 (crítico, confirmado por execução)** — A detecção de diretórios de luz `L*` só inspeciona o pai imediato, que no layout documentado `L<n>/zf<m>/sVal.png` é sempre um `zf*`; em dataset limpo nenhuma luz é detectada e o pipeline aborta — **8/11 datasets reais também falham**.
2. **MF-01 (crítico, confirmado por inspeção)** — A média por plano focal usa `in` (substring), de modo que `"zf1"` casa também `zf10/zf11/zf12` (≥10 planos, todos os configs usam 12), misturando quatro planos na "média do zf1".
3. **MF-02 (crítico, confirmado por inspeção)** — `sorted()` lexicográfico dos diretórios `zf` desalinha a ordem dos frames de `z_foc` (ordem natural no YAML), permutando o mapa índice→profundidade.
4. **INT-04 + CONV-4 (alto, suspeita)** — Hints em unidades físicas de `z_foc` são somados, no mesmo sistema de mínimos quadrados, a alturas em unidades de pixel, com `slopes_scale`/`hints scale` nunca configurados — grandezas incomensuráveis ponderadas por `hints_weight`.
5. **PS-02 (alto, suspeita) / IO-05 (alto, suspeita)** — O limiar de sombra relativo (`1e-3`) é inócuo abaixo do piso de 8 bits (`1/255≈3.9e-3`); e as médias por plano focal são min-max-esticadas individualmente (`normalize=True`), destruindo a comparabilidade de intensidade entre planos antes da medida de foco.

---

## 2. Achados por estágio

### 2.1 Multifocus (MF-xx)

Notas completas: [`audit-notes/01-multifocus.md`](audit-notes/01-multifocus.md).

**MF-01 — Filtro por substring agrupa planos focais errados na média por `zf`** (implementação, crítico, confirmado por inspeção)
A média por plano usa `f"{zf_dir}" in file` (substring), então com ≥10 planos `"zf1"` casa também `zf10/zf11/zf12`. A "média do zf1" mistura quatro planos focais, corrompendo a curva de foco que alimenta toda a seleção de profundidade; o `iSel` resultante é reaproveitado para todas as luzes.
- Localização: `hybrid/main.py:101-103`
- Evidência: reprodução direta — `[p for p in paths if "zf1" in p and "sVal.png" in p]` retornou `zf1, zf10, zf11, zf12`.
- Correção sugerida: filtrar por componente de caminho exato (`os.path.basename(os.path.dirname(file)) == zf_dir`), como já feito para `L*`.
- **Correção aplicada:** e9eabb2 (2026-06-05) — helper `select_files_by_parent_dir` com `Path(f).parent.name == dir_name` substitui o substring match.

**MF-02 — Ordenação lexicográfica dos planos desalinha `average_images_paths` de `z_foc`** (implementação, crítico, confirmado por inspeção)
`zf_directories = sorted({...})` ordena `zf1, zf10, zf11, zf12, zf2, ...`, enquanto `z_foc` no YAML está em ordem natural; o índice de foco `k` recebe o valor de profundidade errado, permutando e tornando não-monotônica a relação índice→z.
- Localização: `hybrid/main.py:90-96,101`
- Evidência: `sorted({'zf1','zf2','zf10','zf11','zf12'})` = `['zf1','zf10','zf11','zf12','zf2']`; `mosaic` indexa `z_foc[k]` por posição (`mosaic.py:57,72`), sem reordenação intermediária.
- Correção sugerida: ordenar por chave numérica (`natsorted` ou `key=int(s[2:])`).
- **Correção aplicada:** e9eabb2 (2026-06-05) — helper `collect_dirs_with_prefix` com `natsorted` substitui `sorted({...})` para `zf_directories` e `light_directories`.

**MF-03 — Pixels sem textura recebem profundidade do meio do stack (`z_foc[n/2]`)** (conceitual, alto, suspeita — difícil de ativar em condições realistas)
Quando o pico de foco é nulo, `compute_argmax_fuzzy_1d` retorna `(n/2, 0)`; o `mosaic` usa `iSel` sem consultar a confiança, então esses pixels entram no `zMos` como plano falso "do meio". O path `return n/2,0` exige contraste rigorosamente zero — não ativado pelos testes.
- Localização: `argmax_fuzzy.py:128-129`; consumo em `mosaic.py:53-74`
- Evidência: `test_textureless_region_gets_zero_confidence` (Task 11) — o path NÃO foi ativado; o mecanismo realista é o **vazamento de desfoco** (textura vizinha espalhada para dentro do patch), produzindo iSel mediano `1.90` (gt 4.0, n/2 4.5), não n/2. Medidas de foco falham abertas perto de fronteiras de textura.
- Correção sugerida: propagar invalidez (NaN/sentinela) em vez de `n/2`, ou mascarar pixels com `wSel==0`.
- **Correção aplicada:** `80069c4` (2026-06-06) — pico nulo retorna NaN + conf 0; mascarado pelo mosaic (MF-04). Teste: `test_zero_peak_returns_nan_not_middle_frame`.

**MF-04 — `mosaic` ignora a confiança `wSel` ao compor `sMos`/`zMos`** (implementação, alto, suspeita)
`mosaic(iSel, image_stack, zFoc, ...)` não recebe nem consulta `wSel`; todo pixel — inclusive confiança 0 e fits convexos rejeitados (`k_fuzzy=0` → `z_foc[0]`) — é tratado como válido.
- Localização: `mosaic.py:51-74`
- Evidência: assinatura sem `wSel`; `argmax_fuzzy.py:175` zera `k_fuzzy`, `mosaic.py:64-65` mapeia para frame 0/`z_foc[0]`.
- Correção sugerida: passar `wSel` e mascarar/interpolar abaixo de um limiar de confiança.
- **Correção aplicada:** `bd7a76e` (2026-06-06) — `mosaic()` aceita `wSel/min_confidence`; pixels inválidos recebem `zMos=NaN`, `sMos=frame mais próximo`; `zMos_with_confidence` usa `wSel_export=where(isfinite(zMos), wSel, 0)` (peso 0 exclui hints inválidos no integrador C); `hybrid/main.py` passa `wSel=wSel_avg` no laço de luzes. Teste: `test_mosaic_masks_zero_confidence_pixels`.

**MF-05 — `find_index_of_max_sum` pode escolher janela que não contém o pico verdadeiro** (conceitual, médio, suspeita)
O início do ajuste maximiza a soma de 3 frames consecutivos (passa-baixa), favorecendo platôs largos sobre picos estreitos e altos; em curvas multi-pico o sub-pixel pode aterrissar no lobo errado.
- Localização: `argmax_fuzzy.py:81-101`
- Evidência: reprodução `fv=[0,0,0,9,0,5,6,5,0,0]` → argmax=3 mas max-sum=6. Tasks 11 (`test_recovers_tilted_plane_depth`/`_bump`): erro mediano 0.181/0.171 frames — **não manifesta erro>0.5 frames em pilhas com pico único limpo**; cenário multi-pico não testado.
- Correção sugerida: usar argmax verdadeiro como centro, com tratamento de empates.
- **Correção aplicada:** `5335ab8` (2026-06-06) — `find_peak_index` (argmax verdadeiro com desempate por suporte de vizinhança) substitui `find_index_of_max_sum`. Repro pinado: `fv=[0,0,0,9,0,5,6,5,0,0]` → old max-sum=6, new argmax=3. Teste: `tests/test_argmax_fit.py::test_peak_index_is_true_argmax`.

**MF-06 — Pesos da regressão = próprios valores de foco enviesam o vértice da parábola** (conceitual, médio, suspeita)
`calculate_weights` usa os valores de foco normalizados como pesos da regressão parabólica; ponderar pela variável dependente quebra a premissa de mínimos quadrados e enviesa o vértice `-B/2A` para o frame de maior valor bruto.
- Localização: `argmax_fuzzy.py:104-118,158,164`
- Evidência: Tasks 11 — erros 0.181/0.171 frames; **não manifesta erro>0.5 frames em pilha uniforme com pico simétrico**; suspeita mantida para picos assimétricos/baixo SNR.
- Correção sugerida: regressão não ponderada ou pesos por incerteza real.
- **Correção aplicada:** `a9a6549` (2026-06-06) — `calculate_weights` e `w_list` removidos; `np.polyfit` chamado sem pesos; R² (MF-07) atualizado para estatísticas não-ponderadas. Vértice pré-correção k=4.284 (erro 0.016), pós k=4.215 (erro 0.085) — ambos dentro de 0.1 do gt 4.3. Profundidade mediana: plano 0.183 frames, bump 0.173 frames (ambos < 0.5). Teste: `tests/test_argmax_fit.py::test_parabola_vertex_unbiased_by_value_weights`.
- Nota: o caso de teste (gaussiana simétrica) não discrimina pré/pós correção — ambos passam com erro < 0,1 (0,016 pré vs 0,085 pós); a correção é teórica (premissas de mínimos quadrados) e o viés do método antigo manifesta-se em curvas assimétricas, não cobertas por teste sintético.

**MF-07 — Confiança `|A|/fnoc` não é comparável entre pixels** (conceitual, médio, suspeita)
A confiança mistura curvatura e amplitude sob normalização global afim; após `normalize()` min-max global fica relativa ao maior `|A|/fnoc` da imagem, sensível a outliers de borda.
- Localização: `argmax_fuzzy.py:183-187`; normalização em `applicator.py:83-93`
- Evidência: `test_textureless_region_gets_zero_confidence` (Task 11) — **dentro de uma imagem** separa textura (0.87) de sem-textura (0.33), razão 0.38; comparabilidade *entre imagens distintas* permanece suspeita (não exercitada).
- Correção sugerida: confiança em escala invariante (razão pico/segundo-pico, ou R²).
- **Correção aplicada:** `86299a6` (2026-06-05) — confiança = R² do ajuste local (escala-invariante, [0,1]); normalize global do wSel removido. Mudança de semântica registrada: `test_textureless_region_gets_zero_confidence` → `test_confidence_is_scale_invariant_goodness_of_fit`; R² não separa textura/sem-textura (sem=0.79, com=0.70) pois é qualidade de ajuste, não força de pico (antigo |A|/fnoc: sem=0.33, com=0.87).

**MF-08 — Normalização global do stack altera comparabilidade entre frames e tem caso degenerado** (implementação, médio, suspeita)
O clip no percentil 1 global achata o fundo dos frames de baixa energia; a guarda `if min_val<0` praticamente nunca roda (indicadores são `|.|≥0`), então o piso fica em `p1/max_val`; `max_val==0` não gera aviso.
- Localização: `applicator.py:80-93`
- Evidência: `p1=np.percentile(stack,1); np.clip(stack,p1,inf)`; indicadores retornam magnitude ≥0 (`laplacian.py:36`, `fourier.py:36`, `wavelet.py:19`).
- Correção sugerida: decidir o piso explicitamente; tratar `max_val==0` com aviso/máscara.
- **Correção aplicada:** `903472d` (2026-06-06) — substituído o par de guardas (`if min_val<0` / `if max_val>0`) por deslocamento incondicional `stack -= stack.min()` pós-clip seguido de `if max_val>0: stack/=max_val else: logging.warning("…no focus signal…")`. O deslocamento é afim e uniforme entre frames (não altera argmax nem vértice da parábola); a guarda antiga nunca disparava pois indicadores são `|.|>=0`. Teste: `tests/test_multifocus_synthetic.py::test_focus_indicator_normalization_floor_and_degenerate_stack` — verifica `fi.min()==0.0, fi.max()==1.0` para stack normal e `fi==0` + WARNING para stack constante.

**MF-09 — Indicador de Fourier é global (FFT da imagem inteira), não local por pixel** (conceitual, alto, suspeita)
O indicador "fourier" faz FFT/máscara passa-alta/IFFT sobre a imagem inteira; cada pixel da reconstrução depende de todas as frequências (kernel de suporte global), espalhando resposta de bordas (ringing) e violando a localidade da seleção por pixel.
- Localização: `fourier.py:5-38`
- Evidência: `np.fft.fft2(image)` global (linha 20), máscara sobre dimensões inteiras, `ifft2` global; sem janelamento.
- Correção sugerida: filtragem passa-alta local (kernel pequeno, energia HF em janela, ou DCT por blocos).
- **Correção aplicada:** `510e042` (2026-06-05) — mecanismo reavaliado: a máscara gaussiana não ringa (equivale a unsharp mask local com σ=1/(2π·radius)≈1,6 px; idêntico no interior, erro ~1e-6); a não-localidade real era o *wraparound circular* da FFT (banda na coluna 0 → resposta 0.37 na borda oposta). Indicador reescrito como `|img − GaussianBlur(img, σ=1/(2π·radius), BORDER_REFLECT)|`: equivalente no interior (pinado por teste), sem wraparound. 5 testes em `tests/test_fourier_locality.py`.

**MF-10 — `non_linear_res` não está integrado e ignora a máscara de pesos no ajuste** (implementação, baixo, confirmado por inspeção)
`applicator.focus_indicator` só despacha fourier/laplacian/wavelet (sem `else`/erro); `non_linear_res` é inalcançável e resolve o ajuste de plano por `lstsq` sem os pesos `W` usados no resíduo. Impacto atual nulo (fora do fluxo).
- Localização: `non_linear_res.py:5-55`; despacho em `applicator.py:36-43`
- Evidência: ramos sem `else`; `non_linear_res.py:41` chama `lstsq` sem pesos.
- Correção sugerida: se reativado, WLS com a mesma `W` e `else: raise ValueError` no despacho.
- **Correção aplicada:** `1aadea3` (2026-06-06) — ramo `non_linear_res` adicionado ao despacho; `else: raise ValueError` elimina UnboundLocalError silencioso; `lstsq` substituído por WLS (`sqrt(W)` reescala linhas); `X`, `sw`, `X_sw` hoistados fora do loop; `F < eps` clampado a 0. Plano perfeito tem resíduo ~0; tipo desconhecido levanta ValueError. 2 testes em `tests/test_multifocus_synthetic.py`.

**MF-11 — `k_fuzzy = max(0, min(n, k_fuzzy))` deveria usar `n-1` como teto** (implementação, baixo, confirmado por inspeção)
O índice válido máximo é `n-1`, mas o clamp permite `k_fuzzy==n`; o `mosaic` se protege contra out-of-bounds, mas o clamp mascara a extrapolação em vez de sinalizá-la e o valor `n` é gravado no CSV de debug.
- Localização: `argmax_fuzzy.py:181`
- Evidência: `max(0, min(n, k_fuzzy))` vs clamp correto do mosaic `min(max(int,0), n_frames-1)` e guarda `i0+1>=n_frames`.
- Correção sugerida: usar `min(n-1, ...)` e baixar confiança quando o vértice cai fora de `[0,n-1]`.
- **Correção aplicada:** `22f4470` (2026-06-06) — vértice fora de [0, n-1] agora clampado a [0.0, float(n-1)] e conf=0; R² block movido para ramo não-extrapolado. Pré-correção: fv=[0,0.05,0.1,0.3,0.7,1.0] dava k=6 (>n-1=5), conf=1.0; pós k=5.0, conf=0. Teste: `tests/test_argmax_fit.py::test_vertex_outside_stack_clamps_to_n_minus_1_with_zero_conf`.

**MF-12 — Quantização uint8 (PNG) das médias por `zf` antes da medida de foco** (implementação, médio, suspeita)
A média (float) é gravada/relida como PNG uint8 antes da medida de foco; a média de muitas imagens ganha bits efetivos que são descartados, e a medida de foco (derivadas HF) é sensível aos degraus de quantização.
- Localização: `hybrid/main.py:107-112,104,149`; `image_io.py:138,140`; `utils.py:106-107`
- Evidência: `calculate_avarage_of_images` retorna `.astype(np.uint8)`; PNG gravado e relido antes de `focus_indicator`.
- Correção sugerida: passar as médias em float (ou ≥16 bits) à medida de foco.
- **Correção aplicada:** (2026-06-05) — médias float em memória (`filtered_images`) + `.npy` de consulta; PNG só visualização. Baseline RMSE afim: 0.0742 → 0.0691 (melhora de 6,9%). Testes: `tests/test_average_float_path.py`. Cross-ref IO-05.

**MF-13 — Ajuste parabólico em índice + conversão índice→z só é exato se `z_foc` for uniforme** (conceitual, baixo, corrigido)
A parábola é ajustada em espaço de índice e o vértice convertido a z por interpolação em `z_foc`; isso só equivale a ajustar em z quando o mapa índice→z é afim (espaçamento uniforme). Todos os configs reais usam passo uniforme (10) — defeito latente.
- Localização: `argmax_fuzzy.py:156,180`; `mosaic.py:72`
- Evidência: `x_list=range(k0,k1+1)`, `k_fuzzy=-B/2A`; configs com `z_foc` uniforme (`hb_experiment.yaml:39`) → mapa afim, sem viés.
- Correção sugerida: ajustar a parábola diretamente em z, ou assertir/documentar espaçamento uniforme.
- **Correção aplicada:** `9ef5721` (2026-06-06) — mitigação/documentação: `check_z_foc_uniformity()` em `multifocus/main.py` emite `logging.WARNING` quando `z_foc` é não-uniforme. O viés para espaçamento não-uniforme permanece por design. Teste: `test_warn_nonuniform_z_foc` em `tests/test_multifocus_synthetic.py`.

**MF-14 — Detecção de diretórios `L*` só olha o pai imediato — vazia no layout `L<n>/zf<m>/sVal.png` limpo** (implementação, crítico, confirmado por execução)
`light_directories` é derivado de `os.path.basename(os.path.dirname(path))` com `startswith("L")`; o pai imediato de cada `sVal.png` é sempre `zf<m>`. Em dataset limpo o conjunto fica vazio, o laço de mosaicos não roda e o Passo 2 aborta em `main_wps.py:123` com `ValueError: Number of images (0) does not match number of light directions (6)`. Verificação sobre os 11 datasets reais: apenas **3/11 detectam alguma luz** (os com `selected-pixels.png` avulso sob `L*/`); os outros **8/11 falham** com o mesmo ValueError — a maioria dos dados reais também quebra, e os 3 que "funcionam" dependem de detritos de filesystem, criando risco de associação luz↔mosaico errada (CONV-6).
- Localização: `hybrid/main.py:122-128` (laço dependente 134-163)
- Evidência: `tests/test_e2e_hybrid.py::test_hybrid_pipeline_end_to_end`, Task 12, `xfail(strict=True, raises=ValueError)`, XFAILED; reprodução direta `light_directories=[]`.
- Correção sugerida: derivar luzes de um componente `L*` em **qualquer** posição do caminho relativo (`Path(path).relative_to(data_path).parts` casando `^L\d+`), idealmente do mesmo varredura estruturada que dá os planos focais (CONV-6).
- **Correção aplicada:** `e827a17` (2026-06-05) — novo helper `collect_light_dirs` varre todos os componentes do caminho relativo a `data_path` (não só o pai imediato) buscando `re.fullmatch(r"L\d+", part)`; substitui `collect_dirs_with_prefix(..., prefix="L")` no `main()`. xfail removido de `test_hybrid_pipeline_end_to_end`; variante `_with_workaround` removida; `marker.txt` removido de `_build_small_dataset`. Baseline E2E (limpo, sem workaround): RMSE=0.3420 (std gt=0.9780), a=0.2791, b=3.2083, pearson r=0.9369. 2 novos testes em `tests/test_hybrid_path_selection.py`.

### 2.2 Fotométrico (PS-xx)

Notas completas: [`audit-notes/02-photometric.md`](audit-notes/02-photometric.md).

**PS-01 — Albedo calculado como norma das intensidades preditas, não como ρ** (conceitual, baixo, confirmado)
O código normaliza a normal (descartando `||m||`=albedo) e recomputa `albedo=||L·n̂||` com `n̂` unitário — quantidade que cresce com o número de luzes válidas `N`, não o albedo. Latente: o albedo nunca é salvo/consumido (sobe a alto/crítico se passar a ser reportado).
- Localização: `wps.py:198` (origem `:189-194`)
- Evidência: `test_wps_albedo_recovers_true_albedo` (Task 10, xfail) — albedo mediano 1.7 (4 luzes) e 2.4 (8 luzes) vs ρ=200; ratio `med/sqrt(N)` constante = 0.8526, confirmando `albedo≈sqrt(N)·f(geometria)`.
- Correção sugerida: `albedo=||m||` antes de normalizar (`ρ=||m||`, `n̂=m/||m||`).

**PS-02 — Limiar de sombra relativo ao máximo do pixel quase nunca rejeita em 8 bits** (implementação, alto, suspeita — não confirmado pelo teste sintético float; permanece para 8 bits reais)
O critério `pixel_values/v_max > shadow_threshold` (default `1e-3`) é avaliado por entrada; em dados 8 bits o menor valor não-nulo é `1/255≈3.9e-3 > 1e-3`, então sombras parciais entram no `lstsq`. Highlights/saturação não são tratados aqui.
- Localização: `wps.py:151-152` (default `:135`)
- Evidência: `test_wps_shadowed_pixels_flagged_not_garbage` PASSED com render float (zeros exatos nas sombras → rejeição funciona). Verificação do revisor: substituindo zeros pelo piso 8 bits, erro angular sobe de 0.004° para 3.6° médio / 20° máx — confirma a fraqueza para dados reais de 8 bits.
- Correção sugerida: limiar **absoluto** em radiância linear + limiar superior para saturação.
- **Correção aplicada:** `cfab906` (2026-06-06) — `shadow_absolute_threshold` e `saturation_threshold` adicionados a `estimate_normals_argmax_lstsq_robust`; ambos default `None` (off, comportamento anterior preservado). Confirmado por execução: piso 8-bit erro 3.644° → 0.005° com `shadow_absolute_threshold=2.0`; saturação erro 5.213° → 0.006° com `saturation_threshold=250.0`. Novas chaves em `configs/hb_experiment.yaml` e `configs/wps_experiment.yaml`. 2 novos testes passando; PS-01/PS-03 xfail inalterados.

**PS-03 — Remoção de outliers por 3×média(|residual|): limiar não robusto** (implementação, médio, confirmado)
O critério `residuals <= 3·mean(|residuals|)` usa a média (não robusta); um outlier grande infla a própria média e mascara o outlier que deveria ser removido (mascaramento clássico).
- Localização: `wps.py:163-185`
- Evidência: `test_wps_robust_to_saturation` (Task 10, xfail) — 2 de 8 luzes saturadas a 60% do máximo produziram erro angular médio 15.13° (limiar do teste 5°).
- Correção sugerida: mediana + MAD (ou IQR); opcionalmente limitar iterações.

**PS-04 — `residual_std` da confiança é o do ajuste antes da última remoção de outliers** (implementação, baixo, verificado por inspeção de fluxo — risco de manutenção)
Hoje correto por construção: o único `break` ocorre quando `mask` mantém todos, alinhando `residuals`/`normal`/`selected_*`. O achado é a fragilidade a refatoração (acoplamento implícito sem asserção).
- Localização: `wps.py:163-176,203-206`
- Evidência: `break` só quando `sum(mask)==len` (`wps.py:175`); alinhamento garantido pela topologia do laço, não por invariante explícita.
- Correção sugerida: recomputar `residuals` do `normal`/conjunto finais antes de derivar confiança.

**PS-05 — Confiança `(N/M)·1/(1+residual_std)` não é invariante a ganho radiométrico** (conceitual, médio, suspeita)
Os resíduos estão na unidade de `I` (0-255), mas `L·n̂` é O(1) (`n̂` unitário); `1/(1+residual_std)→0` para quase tudo em 0-255, e a confiança muda se a entrada for 0-1 (não invariante a ganho). Sem consumidor ativo (limita o impacto a relatório).
- Localização: `wps.py:200-206`
- Evidência: `residual_std=np.std(residuals)` sobre resíduos em unidade de `I`; `n̂` unitário ⇒ escalas diferentes.
- Correção sugerida: normalizar o resíduo pela escala do sinal (`/(albedo+eps)`) ou trabalhar em 0-1 com albedo explícito.

**PS-06 — NaN nas normais (sombra/degenerado) escritos no FNI consumido pelo solver C** (implementação, alto, suspeita — lead fechado por INT-03)
Pixels com <3 luzes válidas ou solução degenerada recebem `np.nan`, salvos sem tratamento em `normal_map.npy` e formatados como `+nan` no FNI; sem máscara de validade pela via das normais.
- Localização: `wps.py:155,183,191`; gravados em `main_wps.py:144`; consumidos via `hybrid/integrate.py:94` → `image_io.py:173-177`
- Evidência: `np.nan` atribuído; `f"{...:+.7e}"` sem checagem de finitude. (Contenção no C confirmada em INT-03 — NaN vira peso 0, não contamina a malha.)
- Correção sugerida: emitir canal de peso explícito (normal_map (H,W,4)) com 0 nos sombreados; propagar máscara de foreground.
- **Correção aplicada:** `bdcb126` (2026-06-06) — `main_wps.py` salva `confidence.npy`; `hybrid/main.py` concatena-o como canal 3 do `normal_map (H,W,4)` antes da integração. **Status: corrigido.**

**PS-07 — Entrada do PS quantizada a 8 bits (sMos.png) — sMos.fni float é ignorado** (implementação, médio, suspeita)
O mosaico é gravado como `sMos.png` (uint8) e `sMos.fni` (float), mas o PS lê o PNG, descartando o float; ruído de quantização (±0.5/255) entra direto nas normais e o `clip(0,255)` satura silenciosamente valores >255.
- Localização: `hybrid/main.py:160-163`; releitura em `main_wps.py:109` (`image_io.py:140`)
- Evidência: seleção `os.path.basename(file)=="sMos.png"` (`hybrid/main.py:181`); `.fni` não referenciado em consumo do PS.
- Correção sugerida: alimentar o PS com o `.fni` float (ou array em memória); remover o clip a 255.
- **Correção aplicada:** `45cbf57` (2026-06-06) — `hybrid/main.py` acumula `sMos_by_light` durante o laço de luzes e injeta `parameters["sMos_images"]`; `main_wps.py` prefere `sMos_images` (float64) quando presente, bypassando o round-trip PNG. `sMos.png`/`.fni` gravados só para visualização. Teste: `test_photometric_receives_float_mosaics_in_memory` (spy confirma `dtype.kind=='f'`). E2e baseline: RMSE afim = 0.0690, Pearson r = 0.9975. Fecha também o caminho de dados de IO-04 (`sMos.png` call-site) e IO-05 (by composition with MF-12); CONV-5: 2 de 3 pontos fechados (gamma PS-08 pendente). **Status: corrigido.**

**PS-08 — Linearidade radiométrica (gamma) não é tratada em nenhum ponto** (conceitual, médio, suspeita — depende do protocolo de aquisição)
O modelo Lambertiano exige intensidades lineares; os `sVal.png` são usados diretamente sem linearização. Se forem sRGB/gamma, o `lstsq` ajusta modelo linear a dados não-lineares e as normais ficam enviesadas. Premissa não verificada.
- Localização: cadeia inteira; nenhum decode de gamma encontrado
- Evidência: ausência de linearização; intensidades cruas em `np.linalg.lstsq` (`wps.py:165`).
- Correção sugerida: documentar/impor a premissa de linearidade (linearizar na entrada se gamma-encoded).
- **Correção aplicada:** `7a4c8fc` (2026-06-06) — `linearize_intensities(img, gamma)` adicionada em `main_wps.py` (module-level); aplicada após `convert_to_grayscale` com `gamma = parameters["photometric"]["parameters"].get("gamma", 1.0)`. Default 1.0 é a identidade (assume `sVal.png` linear); set `gamma: 2.2` para decodificar sRGB. Config `photometric.parameters.gamma: 1.0` adicionada a `hb_experiment.yaml` e `wps_experiment.yaml` com comentário explicativo. Teste: `test_linearize_gamma_helper` (verifica a=0.0, a=255.0 e ponto intermediário 255·(0.5^2.2) com rtol=1e-12; verifica gamma=1.0 é identidade). Fecha também CONV-5 (3º ponto — todos os 3 pontos agora corrigidos). **Status: corrigido.**

**PS-09 — `convert_to_grayscale` falha em entrada já monocromática e mistura convenções de coeficientes** (implementação, baixo, confirmado por inspeção / coeficientes: suspeita)
`cv2.cvtColor(BGR2GRAY)` lança exceção em imagem mono-canal (inalcançável com os dados atuais, mas caso de borda para datasets cinza). Divergência de convenção: este caminho usa Rec.601 BGR; o caminho `rps`/`ps_utils` usa `0.3/0.59/0.11` com heurística RGB-vs-BGR por média de canais.
- Localização: `utils.py:74-85`; chamada em `main_wps.py:115`
- Evidência: reprodução com PNG mono lançou "Bad number of channels"; heurística divergente em `ps_utils.py:150-155`.
- Correção sugerida: checar `ndim`/canais e retornar inalterada se já mono; unificar a política de grayscale entre os entry points.

**PS-10 — `estimate_normals_argmax` inverte 3 luzes sem guard de singularidade (código morto)** (implementação, baixo, confirmado por inspeção)
Resolve com `np.linalg.inv(selected_lights)` sobre as 3 luzes mais brilhantes; se quase coplanares, normal instável (sem try/except/rcond). Código morto — sem chamadas; `top_k:3` dos configs só seria usado por essas funções.
- Localização: `wps.py:8-52` (`:44`)
- Evidência: `grep "estimate_normals_argmax("` vazio; o híbrido usa `estimate_normals_argmax_lstsq_robust`.
- Correção sugerida: se reativadas, `lstsq`/`pinv` com `rcond` e checar condicionamento das 3 luzes.

**PS-11 — `disp_normalmap`/`disp_channels` trocam canais in-place e bloqueiam em headless** (implementação, baixo, confirmado por inspeção)
`np.reshape` devolve uma view e o swap `N[:,:,0],N[:,:,2]=...` muta o array do chamador; `cv2.imshow`+`waitKey(0)` bloqueiam e exigem display. Hoje benigno (save antes do display), mas risco latente.
- Localização: `visualization.py:136-137,165-168,141-142,184-185,254-255`
- Evidência: swap sobre `np.reshape(normal,...)` (view); `waitKey(0)`; `main_wps.py:181-197` chama os displays incondicionalmente.
- Correção sugerida: operar sobre cópia; tornar visualização opcional (flag de debug ou `cv2.imwrite`).

### 2.3 Integração (INT-xx)

Notas completas: [`audit-notes/03-integration.md`](audit-notes/03-integration.md).

**INT-01 — `initial_method` default "hints" sem `-hints` aborta o binário** (implementação, alto, confirmado por inspeção do ramo de aborto; suspeita p/ impacto numérico do chute)
O `.get("initial_method","hints")` diverge do default `"zero"` do dataclass e do binário. Sem `initial_method` e sem hints, o C executa `demand(H!=NULL,...)` no ramo "hints" e **aborta** → `CalledProcessError` → `RuntimeError`. A config empacotada (`zero`+`use_hints:True`) mascara o problema.
- Localização: `hybrid/main.py:214`; `gus_integrate_recursive.c:543-546`
- Evidência: `.get("initial_method","hints")` vs dataclass `"zero"` (`integrate.py:45`); `demand`→`affirm` aborta (`affirm.h:29`).
- Correção sugerida: default `"zero"` no `.get` (alinhar dataclass/binário); ou validar `initial=="hints" ⇒ use_hints`.
- **Correção aplicada:** `80f49cc` (2026-06-05) — default alterado para `"zero"` no `.get`; validação antecipada `ValueError` quando `initial_method=="hints"` e `use_hints` não estiver ativo, com mensagem que cita INT-01. 1 novo teste em `tests/test_integration_units.py` (`test_initial_method_defaults_to_zero_and_hints_requires_use_hints`), passando junto com os 10 existentes.

**INT-02 — `-00-end-Z.fni` ausente → lê silenciosamente o chute inicial `-ini-Z.fni`** (implementação, médio, suspeita — crash confirmado por teste; fallback obsoleto segue suspeita)
Se o end-Z não existir, o Python faz fallback para `-ini-Z.fni`, que é o **chute inicial** (zero ou hints crus), devolvendo-o como resultado sem aviso. O caminho perigoso (returncode 0 sem end-Z) é inalcançável hoje (report final incondicional); os riscos vivos (end-Z em outro cwd, `-ini-Z` obsoleto) são especulativos.
- Localização: `hybrid/integrate.py:171-180`
- Evidência: Task 9 — o crash da rampa (INT-08) levanta `CalledProcessError` **antes** do fallback; `ramp-ini-Z.fni` foi escrito mas nunca lido. Recalibrado para **médio** (impacto potencial > INT-07, mas disparo bloqueado).
- Correção sugerida: remover o fallback para `-ini-Z.fni`; se end-Z faltar, sempre erro.
- **Correção aplicada:** `6728745` (2026-06-06) — bloco fallback removido; `_run_integration` levanta `RuntimeError` diretamente se `{prefix}-00-end-Z.fni` não existir, com mensagem que cita `end-Z`. 1 novo teste `test_missing_end_z_raises_instead_of_returning_initial_guess` (fake solver, sem binário C) confirma o comportamento — `DID NOT RAISE` com o código antigo (fallback silencioso devolveu o `-ini-Z.fni`); passa com a correção. **Status: corrigido.**

**INT-03 — NaN nas normais (PS-06) viram peso 0 — não contaminam a malha, mas removem o pixel sem máscara de foreground** (implementação, médio, suspeita)
Fechamento do lead PS-06. O parser C aceita `+nan` (`strtod`); o NaN não propaga porque `pst_map_ensure_pixel_consistency(G,2)` zera o peso e NaN-iza todos os canais de qualquer pixel não-finito (backstop real). Defeito remanescente: pixels sombreados são silenciosamente zerados/excluídos sem o PS comunicar máscara de foreground — buracos de peso 0 podem introduzir vieses de borda.
- Localização: `pst_normal_map.c:236-243` (guarda parcial) + `pst_basic.c:59` (backstop); origem `wps.py:155,183,191`
- Evidência: guarda da linha 241 inspeciona só `grd.c[0]` e `r3_L_inf_norm` silencia NaN parcial; backstop `pst_map_ensure_pixel_consistency` cobre todos os padrões. Pipeline real NaN-iza os 3 componentes ⇒ ambos capturam o pixel.
- Correção sugerida: PS emitir canal de peso explícito (normal_map (H,W,4)) com 0 nos sombreados; documentar que sombras viram peso-0.
- **Correção aplicada:** `bdcb126` (2026-06-06) — `main_wps.py` salva `confidence.npy`; `hybrid/main.py` carrega e concatena como canal 3 do `normal_map (H,W,4)`, tornando o peso-0 explícito; `test_integrator_accepts_confidence_weight_channel` valida o canal peso+NaN. **Status: corrigido.**

**INT-04 — Hints em unidades físicas de `z_foc` somados a alturas em unidades de pixel** (conceitual, alto, suspeita)
O termo de hints `woo·(H[0]-Z)²` compete no mesmo sistema de mínimos quadrados com os termos de aresta; `H[0]` vem em unidades de `z_foc` (multifocus) e `Z` em altura-por-pixel — escalas incomensuráveis. O `-hints` é montado sem `scale` (`hints_scale=1.0`), de modo que `hints_weight=0.1` pondera grandezas incomensuráveis e enviesa a superfície na direção do `z_foc`.
- Localização: `hybrid/main.py:227,231`; `integrate.py:108-114`; `pst_integrate.c:294-324`
- Evidência: termo no mesmo sistema (`pst_integrate.c:111-113,318-323`); `-hints` sem scale (`integrate.py:109,114`) ⇒ `hints_scale=1.0`.
- Correção sugerida: passar `-hints path scale Hsz weight` com `Hsz`=z_foc→pixel; ou converter `zMos` para unidade de pixel antes de gravar os hints.
- **Correção aplicada:** `aa772ed` (2026-06-05) — novo `hybrid.integration.pixel_size` (tamanho lateral de 1 px em unidades de z_foc); `build_integration_config` deriva `slopes_scale=(pixel_size, pixel_size)` ⇒ `Z` sai em unidades de z_foc, comensurável com os hints. Confirmado por execução: viés com escala default é exatamente `a = pixel_size` (2.5003 p/ pixel 2.5); com pixel_size, `a = 1.0001`. Warning quando `use_hints` sem `pixel_size`. Achado anexo: `szero=TRUE` hard-coded (`pst_integrate_iterative.c:75`) ⇒ hints nunca ancoram nível absoluto, só forma. 10 testes em `tests/test_integration_units.py`.

**INT-05 — hints `(H,W,2)` (células) entregues a um alvo `(H+1,W+1)` (vértices) — C expande, deslocando meia-célula** (implementação, médio, corrigido `1bcefd4`)
`Z` tem dimensão de vértices `(NX+1,NY+1)` e o C exige hints desse tamanho; o `zMos_with_confidence.fni` é grade de células `(H,W,2)`, então `tire_read_fni_file` chama `float_image_expand_by_one`, conversão célula→vértice que desloca os hints meia-célula. Docstring Python `(H+1,W+1)` não corresponde ao arquivo de células real.
- Localização: `gus_integrate_recursive.c:464,691-711`; docstring `integrate.py:217`
- Evidência: `demand((NX_H==NX_Z)&&(NY_H==NY_Z))` (`:519`); expansão célula→vértice (`:701-707`).
- Correção sugerida: gerar `zMos` já como grade de vértices `(H+1,W+1)`, ou aceitar/documentar o erro de meia-célula; alinhar a docstring.
- **Correção aplicada:** `1bcefd4` (2026-06-06) — novo `hybrid/hints.py`: `cell_to_vertex_grid(z, w)` constrói `(H+1, W+1, 2)` a partir das saídas em memória `(zMos_avg, wSel_eff)` via média ponderada pela confiança das até-4 células adjacentes a cada vértice; para campo linear, interpola exatamente na posição do vértice (sem shift). Bloco `use_hints` em `main.py` substituído para usar esse helper em memória e gravar `hints_vertex.fni`; lookup de `zMos_with_confidence.fni` removido. Docstring `integrate.py:217` já estava correta para o caminho in-memory. 2 novos testes unitários: rampa linear e exclusão de células NaN.

**INT-06 — reference `hAvg.png` em uint8 (0-255) comparado a `Z` em unidades de pixel** (conceitual, médio, mitigado)
Com `use_reference`, `hAvg.png` (uint8) é comparado a `Z` (altura-por-pixel) sem `scale` (`reference_scale=1.0`); `E=Z-R` mistura grandezas incomensuráveis. A comparação remove a média (cancela a constante de integração) mas não a escala nem o tilt, tornando o `devE` reportado não interpretável. Apenas diagnóstico (não realimenta o solve).
- Localização: `hybrid/main.py:243,247`; `pst_map_compare.c:96-101,151-165`
- Evidência: `read_image(hAvg.png)`→uint8, `-reference` sem scale; `E=Z-R` cru; `Z` com constante de integração arbitrária.
- Correção sugerida: passar `-reference path scale Rsz` (cinza→altura) ou `hAvg` já em unidades de altura; opcionalmente remover plano/tilt antes de reportar.
- **Correção aplicada:** `c577632` (2026-06-06) — `reference_scale: float = 1.0` em `IntegrateRecursiveConfig`; quando `!= 1.0` os ramos `-reference` emitem `scale <value>` após o path. `build_integration_config` lê o campo do YAML e avisa quando `use_reference=True` sem `reference_scale` (valor depende de calibração por dataset). Entrada comentada em `configs/hb_experiment.yaml`. 8 novos testes. Status: mitigado (infra disponível; valor a calibrar pelo usuário).

**INT-07 — default de `-maxLevel` no C é `DEFAULT_MAX_ITER` (100000), não `DEFAULT_MAX_LEVEL` (30)** (implementação, baixo, confirmado por inspeção — inativo no fluxo Python)
Copy-paste bug: o `else` de `-maxLevel` atribui `DEFAULT_MAX_ITER`; `DEFAULT_MAX_LEVEL=30` é definido mas nunca usado. Efeito prático nulo (recursão para por tamanho `≤3×3`) e inalcançável pelo Python (`-maxLevel` sempre fornecido).
- Localização: `gus_integrate_recursive.c:783`
- Evidência: `o->maxLevel = DEFAULT_MAX_ITER;`; `integrate.py:130` sempre passa `-maxLevel`.
- Correção sugerida: trocar `DEFAULT_MAX_ITER` por `DEFAULT_MAX_LEVEL` na linha 783.

**INT-08 — `integrate_slopes_to_height` com 2 canais crasha — topo aceita 2/3, solver iterativo exige 3** (implementação, médio, confirmado por teste)
O caminho `-slopes` com mapa de 2 canais (o que o wrapper grava e documenta) aborta: o topo aceita 2 ou 3 canais mas o solver iterativo exige exatamente 3, e o topo não promove 2→3. O fluxo híbrido usa `-normals` (inalcançável no pipeline), mas a API pública está quebrada.
- Localização: `integrate.py:261-263`; `gus_integrate_recursive.c:503`; `pst_integrate_iterative.c:47`
- Evidência: `test_constant_slopes_recover_ramp_and_decide_convention` (Task 9) — crash `pst_integrate_iterative.c:47: slope map {G} must have 3 channels` (verbatim em 06-test-results); a tabela de convenção nunca foi impressa.
- **Correção aplicada:** `7214d43` (2026-06-06) — `integrate_slopes_to_height` promove 2→3 canais com `np.concatenate([slope_map, np.ones_like(slope_map[..., :1])], axis=-1)` antes de delegar a `_run_integration`; docstring atualizada. Evidência pós-correção: tabela de candidatos `test_constant_slopes_recover_ramp_and_decide_convention` (xfail removido):
  ```
  RMSE por candidato de convenção:
        0.000052  z = +ax*x + ay*y (y do numpy, para baixo)   ← vencedor
        0.380857  z = +ax*x - ay*y (y invertido: para cima)
        0.403960  z = +ay*x + ax*y (eixos trocados)
        0.952142  z = -ax*x + ay*y
        1.025489  z = -ax*x - ay*y (tudo invertido)
  ```
  Consistente com o vencedor via `-normals`. Suíte completa: 102 passed, 2 xfailed (PS-01, PS-03).

### 2.4 Infra/IO (IO-xx)

Notas completas: [`audit-notes/04-io.md`](audit-notes/04-io.md) (inclui a tabela call-site × `save_image`).

**IO-01 — `convert_image_array_to_fni` grava `%.7e` — trunca float64 a ~7-8 dígitos** (implementação, baixo, confirmado por inspeção)
O writer FNI usa `f"{...:+.7e}"` (8 dígitos significativos): exato para float32, trunca float64 (`zMos`/`zMos_with_confidence` são float64). Severidade baixa porque **ambos os consumidores são float32** (reader e integrador C), abaixo da precisão da fonte.
- Localização: `image_io.py:173,176`
- Evidência: round-trip `0.123456789012345 → +1.2345679e-01 → 0.12345679`; reader aloca float32 (`:223`). `test_fni_roundtrip` (Task 8) max rel diff ~6e-8 (epsilon float32).
- Correção sugerida: `%.17g` (ou ≥`%.15e`) para FNIs com dados físicos; documentar precisão.

**IO-02 — `read_fni_to_image_array` deixa pixels ausentes silenciosamente em 0** (implementação, médio, confirmado por inspeção)
O reader aloca `np.zeros` e preenche pixel-a-pixel, sem verificar que todos os `ny*nx` foram escritos; um FNI truncado deixa regiões em 0 (indistinguível de altura/slope/confiança legítima 0), contaminando o resultado sem sinal.
- Localização: `image_io.py:222-243`
- Evidência: `np.zeros(...)` + loop sem contagem; `return` incondicional.
- Correção sugerida: contador/máscara de pixels preenchidos e `ValueError` se `count != ny*nx`; ou inicializar com NaN.

**IO-03 — Parser FNI pula silenciosamente linhas de dados com menos de `2+nc` campos** (implementação, baixo, confirmado por inspeção)
`if len(parts) < 2+nc: continue` descarta linhas truncadas sem aviso (pixel fica 0, ver IO-02); assimetria: **falta** de campos é silenciada, mas **excesso** levanta `ValueError`.
- Localização: `image_io.py:231-232`
- Evidência: `continue` para falta vs `raise` para excesso (`:236-237`).
- Correção sugerida: unificar — linha malformada deve levantar (ou logar aviso).

**IO-04 — `save_image(normalize=True)` (default) quantiza e min-max-estica arrays float salvos como PNG** (implementação, médio, confirmado por inspeção)
`save_image` sempre grava uint8; o ramo default ainda aplica min-max stretch destrutivo. `read_image` preserva profundidade mas o writer nunca produz 16-bit — round-trip float→PNG→float perde precisão e (no default) escala. A tabela de call-sites classifica cada uso; o achado específico é IO-05.
- Localização: `image_io.py:137-142`; call-sites na tabela das notas
- Evidência: `cv2.normalize(...).astype(uint8)` (`:138`) e `np.clip(...).astype(uint8)` (`:140`).
- Correção sugerida: para dados consumidos a jusante, usar FNI; PNG só para visualização (ou 16-bit + `normalize=False`).
- **Resolução (composição, 45cbf57):** os dois call-sites de dados consumidos foram fechados por composição — `average_{zf}.png` por MF-12 e `sMos.png` por PS-07. Nenhum dado científico passa por `save_image` antes de ser consumido no pipeline híbrido. PNGs restantes são todos visualização.

**IO-05 — Médias por plano focal salvas com `normalize=True` — cada `average_{zf}.png` é min-max-esticado independentemente** (conceitual, alto, suspeita)
Cada plano focal é esticado individualmente a [0,255] antes da medida de foco; planos desfocados (baixo contraste) são ampliados ao mesmo intervalo dos nítidos, artificialmente igualando a escala inter-plano e podendo deslocar o argmax de foco — corrompendo `iSel`. Distinto de MF-12 (quantização) e MF-08 (normalização do applicator). O mesmo arquivo sabe usar `normalize=False` nos mosaicos por luz, mas não aqui.
- Localização: `hybrid/main.py:111`; relido em `multifocus/main.py:96`
- Evidência: `save_image(...,f"average_{zf_dir}.png",...)` sem 4º argumento ⇒ `normalize=True` ⇒ `cv2.normalize(...,NORM_MINMAX)` per-arquivo; comparar com `hybrid/main.py:160` (`normalize=False` cross-luz).
- Correção sugerida: salvar com `normalize=False` (idealmente FNI/16-bit); ou alimentar o multifocus in-memory.
- **Correção parcial (cross-ref MF-12, 2026-06-05):** o stretch per-plano saiu do CAMINHO DE DADOS (multifocus usa `filtered_images` float em memória); PNG de visualização continua com `normalize=True` — IO-05 não está totalmente resolvido (visualizações ainda esticadas), mas deixou de afetar o resultado científico.
- **Resolução (composição, 45cbf57):** com PS-07 também fechado (mosaicos float em memória), o conjunto completo de dados científicos do pipeline híbrido não passa por `save_image`. IO-05 afeta apenas a visualização dos `average_{zf}.png`.

**IO-06 — `save_image` chamado com 5 argumentos posicionais em `image_alignment.py` — `TypeError`** (implementação, baixo, confirmado por inspeção — inativo no pipeline)
A assinatura tem 4 parâmetros; as chamadas passam 5 posicionais (`...,0,255`), o 5º sem parâmetro ⇒ `TypeError`. Inalcançável (`main_align` sem callers); indício de drift de assinatura.
- Localização: `image_alignment.py:195,213,216`
- Evidência: assinatura de 4 params (`image_io.py:114-116`); 3 chamadas com 5 posicionais; `main_align` sem caller.
- Correção sugerida: corrigir para `save_image(path,name,img,normalize=False)`; se `main_align` for código morto, removê-lo.

### 2.5 Convenções (CONV-xx)

Notas completas: [`audit-notes/05-conventions.md`](audit-notes/05-conventions.md). (CONV-3 não gerou achado — a indexação do round-trip FNI é consistente; ver "Verificado sem achado" das notas e a Tabela de convenções abaixo.)

**CONV-1 — Direção do eixo z não é fixada/verificada ponta a ponta** (conceitual, alto, REFUTADO no integrador; metade física em aberto)
A direção de crescimento de `z` não é fixada nem verificada em nenhum elo. No dataset POV-Ray `z_foc` maior = plano mais distante; `zMos` preserva essa direção física, enquanto `Z` integrado tem constante arbitrária e sinal dependente da convenção de normal. Um desacordo de sinal `zMos`(hints)↔`Z` puxa a superfície na direção errada.
- Localização: dataset `zf*` → `z_foc` → `mosaic.py:57,72` → hints (`hybrid/main.py:231`) → `Z` (`gus_integrate_recursive.c:457-459`) → `height_map.npy`
- Evidência: `test_ramp_normals_decide_convention` (Task 9-ext, PASSED) — a cadeia numpy→FNI→C→FNI→numpy preserva o sinal end-to-end via `-normals` (vencedor `+ax·x+ay·y`, RMSE 5.2e-5 vs 1.025 do totalmente invertido) ⇒ **inconsistência de sinal do integrador REFUTADA**. Task 12-ext (workaround MF-14): `a=+1.003`, `r=0.9971` ⇒ cadeia interna não inverte z **no frame sintético**. A **metade física** (`zMos` real vs `Z`) permanece aberta — luzes sintéticas no mesmo frame numpy não a reproduzem; requer dados reais.
- Correção sugerida: documentar/fixar a convenção de direção de z; asserção/conversão de sinal entre `zMos` (hints) e `Z` antes de combiná-los.

**CONV-2 — Convenção do eixo Y das luzes (lights.npy) vs imagem numpy não é documentada nem reconciliada** (conceitual, alto, REFUTADO só no eixo-y do integrador; metade real em aberto)
As normais vivem no frame das luzes; `lights.npy` real são tuplas POV-Ray (left-handed, +y-up), mas a imagem é numpy `[linha,coluna]` com linha 0 = topo (y para baixo). Não há reconciliação explícita; o escape `scale 1 -1` nunca é usado (`slopes_scale=(1,1)`).
- Localização: `lights.npy` (gerador POV-Ray) → `main_wps.py:119` → `wps.py:165,194` → `pst_basic.c:49-50` → grade C
- Evidência: `test_ramp_normals_decide_convention` (Task 9-ext, PASSED) — o `y` do numpy é preservado no round-trip; flip-y (`+ax·x-ay·y`) rejeitado (RMSE 0.381 vs 5.2e-5) ⇒ **eixo-y do integrador REFUTADO**. A reconciliação y-up(luzes)↔y-down(imagem) no PS é elo separado, **em aberto**: o E2E sintético constrói luzes no mesmo frame numpy das normais (`synthetic_utils.py`), não reproduz a questão POV-Ray; só decidível com `lights.npy` real + gt de altura real.
- Correção sugerida: documentar a convenção das `lights.npy` (y-up POV-Ray) e, no boundary numpy↔C, negar a coluna y das luzes ou passar `scale 1 -1`, de modo explícito e testado.

**CONV-4 — `slopes_scale` default `(1,1)` nunca é configurado — gradientes adimensionais e hints em z físico incomensuráveis** (conceitual, alto, suspeita)
Consolidação de INT-04/INT-05. Os slopes integrados são adimensionais (`Z` em altura-por-pixel); os hints vêm em unidades de `z_foc`. O único mecanismo reconciliador (`scale`) nunca é emitido pelo híbrido (`slopes_scale=(1,1)`, `-hints` sem scale), e `hints_weight=0.1` pondera unidades distintas.
- Localização: `integrate.py:53`; `hybrid/main.py:213-221` (sem `slopes_scale`); `gus_integrate_recursive.c:449-450`
- Evidência: `slopes_scale=(1.0,1.0)` default; `if config.slopes_scale != (1.0,1.0)` ⇒ `scale` não anexado; `-hints` sem scale ⇒ `hints_scale=1.0`. Task 12-ext: `|a|≈1` é coincidência do setup sintético (passo z_foc=1=passo pixel), **não** exercita a escala real — CONV-4 permanece aberto.
- **Correção aplicada:** `aa772ed` (2026-06-05) — mesma correção de INT-04: `pixel_size` no config deriva `slopes_scale=(s,s)`, o `scale` passa a ser emitido e `Z` sai em z_foc. Rampa física com pixel 2.5 confirma a previsão (`a=2.5003` default → `a=1.0001` corrigido): **confirmado por execução e corrigido**.
- Correção sugerida: computar/passar `slopes_scale`/`hints scale` que levem `z_foc`→altura-por-pixel; ou converter `zMos` para unidade de pixel antes de gravar os hints.

**CONV-5 — Cadeia radiométrica quebra a linearidade exigida pelo PS em três pontos** (conceitual, médio, suspeita)
Consolida a sequência radiométrica: `sVal.png` possivelmente gamma (PS-08); média re-quantizada a uint8 (MF-12); min-max stretch por-plano nas médias (IO-05, quebra comparabilidade inter-plano); mosaico `clip(0,255)`+uint8 (PS-07); `sMos.fni` float ignorado (PS-07); `lstsq` ajusta modelo linear a dados não-lineares/saturados/quantizados. Nenhum estágio reintroduz linearidade.
- Localização: `hybrid/main.py:107-111,154-162` → `multifocus/main.py:96`; `main_wps.py:115` → `wps.py:165`
- Evidência: cross-ref MF-12, IO-05, PS-07, PS-08, PS-09; passos detalhados nas notas.
- Correção sugerida: alimentar foco e PS com float in-memory; desativar `normalize` nas médias por-zf; remover o `clip(0,255)`; documentar/impor a premissa de linearidade.
- **Resolução parcial (composição, 45cbf57):** 2 de 3 pontos fechados. MF-12 fechou o ponto de quantização das médias e IO-05; PS-07 (este commit) fechou os pontos do mosaico uint8/clip e leitura PNG. Nenhum dado científico do pipeline híbrido passa por round-trip PNG. Ponto remanescente: gamma PS-08 (linearidade da entrada `sVal.png`) — pendente Task 9.
- **Fechado por composição (`7a4c8fc`, 2026-06-06):** 3 de 3 pontos fechados. PS-08 (este commit) fecha o ponto remanescente adicionando `linearize_intensities` com config `gamma`. MF-12 (bd23651) + PS-07 (45cbf57) + PS-08 (7a4c8fc) = cadeia radiométrica completa: médias float, mosaicos float, gamma documentado/opcional. **Status: corrigido/fechado por composição.**

**CONV-6 — Pareamento luz↔mosaico garantido só por contagem; ordens e shape entre estágios não verificados por construção** (implementação, alto, suspeita — reforçada por Task 12)
Três pareamentos posicionais sem verificação por identidade: (1) luz↔mosaico pareado por posição com a única garantia da checagem de contagem (`len(images)!=lights.shape[0]`), não por chave `L<n>`→linha `n`; (2) ordem zf↔z_foc (MF-02); (3) shape hints célula vs vértice (INT-05). O achado novo é (1).
- Localização: `hybrid/main.py:177-184` vs `main_wps.py:119-127`; `hybrid/main.py:90` vs `mosaic.py:57`; `(H,W,2)` vs `gus_integrate_recursive.c:519`
- Evidência: Task 12 — a guarda de contagem pegou o caso degenerado (0 vs 6, MF-14) e abortou com mensagem clara; o cenário central (N mosaicos em ordem errada com N linhas) **não foi exercitado** (nunca houve N mosaicos). Permanece suspeita.
- Correção sugerida: parear luz↔mosaico por chave extraída do path (`L<n>`→linha `n`); ordenar `zf_directories` por chave numérica; verificar shape dos hints contra `(H+1,W+1)`.
- **Correção aplicada (pareamento luz↔mosaico):** `c226924` (2026-06-05) — helper `pair_mosaics_to_lights` extrai o índice numérico do diretório-pai `L<n>` de cada mosaico e verifica que o conjunto de índices é exatamente `0..n_lights-1`; ValueError com mensagem clara para faltante/extra/duplicado. Substitui o `natsorted(sMos_path_list)` posicional no Passo 2 de `main()`. Sub-achados (2) e (3) permanecem abertos (MF-02 corrigido separadamente; INT-05 pendente).

### 2.6 Regressões (REG-xx)

**REG-01 — `normalize=True` acidental nos mosaicos `sMos.png` por luz — regressão de `aa772ed`** (implementação, alto, confirmado por execução)
O commit `aa772ed` (fix INT-04/CONV-4) flipou acidentalmente `normalize=False` → `normalize=True` na chamada `save_image(..., "sMos.png", sMos_light, ...)` em `hybrid/main.py`. O comentário imediatamente acima explicava por que `normalize=False` é essencial: os mosaicos são a entrada do PS e um stretch min-max por imagem destrói as relações de intensidade entre luzes que o modelo `I = albedo · (L · N)` requer. A regressão ficou mascarada porque a correção de MF-14 (Task 2) também entrou no mesmo ciclo de correções; o E2E com normalize=False-correto dá RMSE 0.0691 / a 0.9960 / r 0.9975, enquanto com normalize=True (regredido) os maxima de todos os `sMos.png` sobem a 255 e o fit afim degrada para RMSE 0.342 / a 0.279 / r 0.937.
- Localização: `hybrid/main.py:274` (linha exata após correção)
- Tipo: implementação
- Severidade: alto
- Status: confirmado por execução — `test_hybrid_pipeline_end_to_end` falhou no assert `all(m < 255 for m in maxima)` com `max=[255,255,255,254,255,255]` antes da correção; passou após restaurar `normalize=False`
- Nova baseline pós-correção: RMSE = 0.0691 (std gt = 0.9780), a = 0.9960, b = 3.1316, pearson r = 0.9975
- **Correção aplicada:** `a9f9f04` (2026-06-05) — restaura `normalize=False` e adiciona assert radiométrico em `test_e2e_hybrid.py`.

---

## 3. Tabela de convenções

Resumo das 6 convenções com vereditos finais pós-testes (Fases 2 e 3). Vereditos: **consistente**
(a leitura estática prova que as pontas batem) · **inconsistente** (achado CONV-xx) ·
**refutado** (teste mostrou que a inconsistência não ocorre na via testada).

| # | Convenção | Produtor → Consumidor | Veredito final | Evidência decisiva |
|---|---|---|---|---|
| 1 | Eixo z / profundidade (direção + unidade) | `z_foc` → `zMos` (`mosaic.py:57,72`) → hints C → `height_map.npy` | **unidade: inconsistente (INT-04/CONV-4)**; **sinal-integrador: REFUTADO**; **sinal-físico: em aberto** | `test_ramp_normals_decide_convention` PASSED (vencedor `+ax·x+ay·y`, RMSE 5.2e-5 vs 1.025). Task 12-ext: `a=+1.003`, `r=0.997` (frame sintético). Metade física requer dados reais. |
| 2 | Normais e luzes (frame / y-up vs y-down) | `lights.npy` POV-Ray → wps → normal→slope C (`pst_basic.c:49-50`) | **eixo-y integrador: REFUTADO**; **luzes vs imagem (frame real): em aberto** | flip-y `+ax·x-ay·y` rejeitado (RMSE 0.381 vs 5.2e-5). Reconciliação y-up(luzes)↔y-down(imagem) não exercida — luzes sintéticas no mesmo frame numpy; requer `lights.npy` real. |
| 3 | Origem/orientação da imagem (round-trip FNI) | writer FNI Python → C → reader Python | **consistente** | `test_fni_roundtrip` 4/4 (Task 8): round-trip preserva o array sem flip/transposição; `test_ramp_normals` confirma orientação+sinal ponta a ponta. |
| 4 | Escala dos gradientes (slope adimensional vs z físico) | slope `-nx/nz` adimensional → sistema altura-por-pixel + hints em `z_foc` | **inconsistente (CONV-4)** | `slopes_scale` default `(1,1)` nunca configurado; hints sem `scale`. Task 12-ext: `|a|≈1` é coincidência sintética, não exercita a escala real. |
| 5 | Radiometria entre etapas (linearidade) | `sVal.png` uint8 → médias re-esticadas + mosaicos clipados → foco/PS | **corrigido (CONV-5)** | 3/3 pontos fechados: MF-12 (bd23651) médias float, PS-07 (45cbf57) mosaicos float, PS-08 (7a4c8fc) gamma opcional. |
| 6 | Contratos de arquivo (pareamento/ordem/shape) | `natsorted(sMos)`, `sorted(zf)`, `(H,W,2)` | **inconsistente (CONV-6)** | Pareamento luz↔mosaico só por contagem; Task 12: guarda pegou 0 vs 6 (MF-14); N-trocados não exercitado. |

---

## 4. Linha de base end-to-end

Notas completas: [`audit-notes/06-test-results.md`](audit-notes/06-test-results.md).

A linha de base do estado atual do pipeline **só pôde ser medida via workaround do MF-14**. O teste
E2E limpo (`test_hybrid_pipeline_end_to_end`, layout `L<n>/zf<m>/sVal.png` sem detritos) **não
completa**: aborta no Passo 2 com `ValueError: Number of images (0) does not match number of light
directions (6)` (MF-14, detecção de luzes vazia). A variante
`test_hybrid_pipeline_end_to_end_with_workaround` grava um arquivo marcador sob cada `L<n>/` (igual
ao detrito que salva a detecção nos 3/11 datasets reais), permitindo medir o restante da cadeia
(mosaicos por luz → PS → integração C).

**Baseline (workaround MF-14, época da auditoria):**
```
affine-fit RMSE = 0.0742 (std gt = 0.9780), a = 1.0030, b = 3.1279, pearson r = 0.9971
```
*(Após correção MF-12 — 2026-06-05, commit bd23651: médias por-zf em float sem quantização uint8, o RMSE passou a 0.0691, melhora de ~6,9%; a = 0.9960, b = 3.1316, r = 0.9975. Valores de a/r inalterados em essência. Ver `audit-notes/06-test-results.md` §Task 12 para o bloco completo.)*

Interpretação por número:
- **RMSE = 0.0742 vs std gt = 0.9780:** o resíduo após o fit afim é ~**7,6 %** do desvio-padrão do
  sinal de gt — recuperação estrutural boa. **É linha de base, não gate de qualidade.**
- **a = 1.0030 (escala, ≈ +1):** o sinal positivo indica que a cadeia mosaico→PS→integração
  **preserva a orientação** (não inverte z) **para luzes no mesmo referencial das normais**. O
  **módulo ≈ 1 é coincidência por construção do setup sintético**: o passo de `z_foc` no sintético
  é 1.0 (cada frame avança 1 unidade) e o passo de pixel também é 1 amostra — logo
  profundidade-em-z_foc e altura-em-pixels coincidem numericamente. Em dados reais o passo de
  `z_foc` é ~10 por frame e `a≈1` **não seria esperado**; o desacordo de escala de CONV-4 **não é
  exercitado** aqui. Não ler `a≈1` como evidência positiva sobre o tratamento de escala.
- **b = 3.1279 (offset):** o integrador resolve altura a menos de constante; absorvido pelo fit
  afim. Esperado, sem significado físico.
- **pearson r = 0.9971 (≈ +1):** correlação quase perfeita ⇒ a estrutura é recuperada com a
  orientação correta atravessando as luzes **no frame sintético**. **Não decide o frame do
  `lights.npy` real** (metade real de CONV-2).

**Caveat sintético:** as luzes vêm de `ring_lights` construídas no **mesmo referencial numpy das
normais** (`synthetic_utils.py`); o `r≈+1` confirma apenas que a cadeia interna não inverte
orientação quando luzes e normais já estão no mesmo frame — **não** reproduz a questão real de
CONV-2 (y-up POV-Ray). Não há atalho sintético.

### Bloqueios

- **Build do C (`make`) falha — NÃO bloqueante.** `cd csrc/integrate_recursive && make` falha com
  dois erros de compilação: (1) `-Werror=comment` por um `/*` aninhado em comentário de bloco
  (`gus_integrate_recursive.c:9`); (2) `#include <bool.h>` não encontrado no include de sistema
  (`:312`; o `bool.h` do projeto vive em `include/`). **Porém o binário `gus_integrate_recursive`
  está versionado (rastreado pelo git) e é funcional** — `DEFAULT_EXECUTABLE.exists()` é `True`, o
  guard `needs_binary` não pulou os testes, e a integração rodou contra o binário pré-compilado.
  Recompilar do zero exigiria editar fonte C de produção (fora do escopo do diagnóstico).
- **O que permanece aberto (requer dados reais):**
  - **CONV-2 (metade real):** o referencial-y do `lights.npy` real (y-up POV-Ray) vs eixos da
    imagem durante o PS — só decidível com `lights.npy` real + ground-truth de altura real,
    comparando a orientação do mapa integrado com/sem flip do eixo-y das luzes.
  - **CONV-1 (sinal-z físico):** o desacordo de sinal `zMos`(hints) real vs `Z` integrado — idem,
    requer E2E com hints e ground-truth de altura real (refutado apenas no integrador via
    `-normals`).
  - **CONV-4 (unidades):** o desacordo de escala slopes-adimensionais vs hints em `z_foc` físico
    não foi exercitado (o sintético tem `a≈1` por construção). **Recomendação de teste:** rodar o
    E2E com `z_foc` de passo realista (ex.: 10) e hints ativos, medindo o viés de escala no fit
    afim com/sem `slopes_scale` configurado.

---

## 5. Apêndice — cobertura da varredura

### 5.1 O que foi auditado (por estágio, das seções "Verificado sem achado")

**Multifocus** — Laplaciano e wavelet são locais (medida válida por pixel); `zero_border` evita
propagar HF do padding; asserção de não-negatividade coerente com indicadores `|.|`;
`linear_interpolation`/`quadratic_interpolation` indexam `zFoc`/stack corretamente, clampam e não
extrapolam; `mosaic` protege contra out-of-bounds (mesmo recebendo `k_fuzzy==n` de MF-11);
`compute_argmax_fuzzy_1d` exige nº mínimo de frames; rejeição de ajuste convexo correta; fallback de
`polyfit` em `LinAlgError` adequado; `depth_refinement.py` e `image_alignment.py` são **código morto**
(sem imports no pacote). Confirmado por execução: `focus_indicator(laplacian)`+`compute_argmax_fuzzy`
recupera profundidade sub-pixel (mediana 0.18/0.17 frames em rampa e bump, Task 11).

**Fotométrico** — `evaluate_angular_error` correta (`arccos` com clip a `[-1,1]`); solver robusto
usa `lstsq(..., rcond=None)` adequado (confirmado por execução: 0.004° em dados limpos, Task 10);
guards de norma zero e de nº mínimo de luzes presentes; chaves de config batem com os `get`; checagem
de contagem imagem↔luz existe; `_solve_l2/_l1/_sbl/_rpca` (RPS) seguem Woodham/Ikehata/Wu; convenção
de saída RPS≡WPS (ambos `(H,W,3)` unitários); `numerics.py` com estabilização numérica presente.

**Integração** — gramática de argumentos Python↔C confere (exceto defaults INT-01/INT-07);
`-normals` converte normal→slope com sinais corretos (`dZdX=-nx/nz`, `dZdY=-ny/nz`, piso `nzmin`),
confirmado por round-trip (Task 9-ext); equação por célula é o balanço de fluxo ponderado padrão
(Poisson com pesos, fronteiras Neumann natural, buracos de peso 0 excluídos, fudge-to-zero);
restrição/prolongação multigrid consistentes em escala; `-00-end-Z.fni` garantido em sucesso;
comparação com reference remove a média (cancela a constante de integração); indexação `float_image`
(col,row) e writer/reader FNI mutuamente consistentes; `pst_map_ensure_pixel_consistency` saneia
pixels inválidos coerentemente.

**Infra/IO** — round-trip de indexação Python↔C consistente sem flip (confirmado por
`test_fni_roundtrip`, Task 8); header/metadados FNI parseados corretamente; `float("nan")` aceita
`+nan` no parser Python; `normalize` (utils) tem guarda para array constante; `convert_to_grayscale`
lida com float64 via cast a float32 (coeficientes Rec.601 corretos para BGR);
`calculate_avarage_of_images` soma em float32 (sem overflow de uint8); `read_image` levanta em arquivo
ausente.

**Convenções** — CONV-3 (origem/orientação da imagem) é **consistente** — não gerou achado (a
indexação do round-trip FNI confere; o sinal físico de dZ/dY é coberto por CONV-1/CONV-2).

### 5.2 O que ficou fora (do spec)

- `csrc/integrate_recursive/lib-src/` como um todo (biblioteca vendorizada de propósito geral, ~80
  arquivos): apenas o caminho de código percorrido pela integração foi auditado.
- `notebooks/` e `core/` (scaffolding vazio).
- Estilo, performance e refatoração — só entraram achados que afetam corretude, precisão ou robustez.
- `multifocus/depth_refinement.py` e `multifocus/image_alignment.py`: registrados como **código
  morto** (sem callers no pacote), fora do escopo de corretude do fluxo ativo (IO-06 anota o
  `TypeError` latente em `image_alignment`).

**Cobertura do escopo:** todos os arquivos da tabela de escopo do spec (multifocus, fotométrico,
integração híbrida Python+C, infra/IO) aparecem em ao menos um achado ou na seção "Verificado sem
achado" das notas, ou estão declarados como código morto/fora-de-escopo com a razão.

### 5.3 Suíte de testes permanente criada

A Fase 3 deixou uma suíte de regressão permanente em `tests/`:

- `tests/synthetic_utils.py` — geradores sintéticos (stacks com desfoque, render Lambertiano, luzes
  em anel, rampas/bumps) — todos no mesmo referencial numpy (documentado no cabeçalho).
- `tests/test_synthetic_utils.py` — sanidade dos geradores.
- `tests/test_fni_roundtrip.py` — round-trip FNI Python↔Python (2D/3-canais/negativos/NaN).
- `tests/test_convention_integration.py` — convenção Python↔C via rampa/bump (`-normals` e `-slopes`);
  `test_constant_slopes_recover_ramp_and_decide_convention` decide CONV-1/CONV-2 via `-slopes`
  (INT-08 corrigido — xfail removido; vencedor confirmado `z = +ax*x + ay*y`).
- `tests/test_photometric_synthetic.py` — clean-data, albedo (PS-01 xfail), saturação (PS-03 xfail),
  sombras.
- `tests/test_multifocus_synthetic.py` — recuperação de profundidade (rampa/bump), confiança em
  região sem textura.
- `tests/test_e2e_hybrid.py` — E2E híbrido: o teste limpo é `xfail(strict)` (MF-14); a variante com
  workaround mede a baseline.

**Estado final da suíte** (`pytest tests/ -q`, 2026-06-05): **18 passed, 4 xfailed**. Os 4 xfailed
eram evidência confirmatória de achados: `test_wps_albedo_recovers_true_albedo` (PS-01),
`test_wps_robust_to_saturation` (PS-03), `test_hybrid_pipeline_end_to_end` (MF-14) e
`test_constant_slopes_recover_ramp_and_decide_convention` (INT-08).

**Estado pós-correções (2026-06-06):** `pytest -q` → **102 passed, 2 xfailed**. Os 2 xfailed
restantes são PS-01 e PS-03 (pendentes). `test_hybrid_pipeline_end_to_end` (MF-14) e
`test_constant_slopes_recover_ramp_and_decide_convention` (INT-08) passam sem xfail.
