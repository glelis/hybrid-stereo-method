# Design: Auditoria de corretude dos métodos do pipeline híbrido

**Data:** 2026-06-04
**Branch:** `nova_iteracao`
**Status:** aprovado pelo usuário

## Objetivo

Varredura uniforme de todos os métodos do pipeline (multifocus, fotométrico, integração,
IO) em busca de **erros conceituais** (matemática/premissa teórica incorreta) e
**problemas de implementação** (bugs, perda de precisão, robustez numérica).

**Entregável:** relatório de achados em `docs/superpowers/reports/2026-06-04-method-audit.md`.
Nenhuma correção é aplicada nesta etapa — apenas diagnóstico com sugestão de correção.

**Motivação:** há resultados ruins observados nos experimentos, sem sintoma específico
isolado; a auditoria é uma varredura uniforme, não direcionada.

## Escopo

| Área | Arquivos auditados |
|---|---|
| Multifocus | `multifocus/`: `indicators/` (`laplacian`, `fourier`, `wavelet`, `non_linear_res`, `applicator`), `argmax_fuzzy`, `depth_refinement`, `mosaic`, `image_alignment`, `math_utils`, `main`, `utils` |
| Fotométrico | `photometric/`: `wps`, `rps`, `ps_utils`, `solvers/numerics`, `main`, `main_wps` |
| Integração híbrida | `hybrid/main.py`, `hybrid/integrate.py`; lado C: `gus_integrate_recursive.c` e a cadeia `pst_*` efetivamente executada (`pst_slope_map`, `pst_normal_map`, `pst_integrate_*`, `pst_imgsys*`, `pst_height_map`, `pst_interpolate`) mais `float_image_mscale` do multigrid |
| Infra/IO | `infrastructure/io/image_io.py` (FNI ↔ array, radiometria), `infrastructure/utils.py` |

**Fora do escopo (deliberado):**

- `csrc/integrate_recursive/lib-src/` como um todo — biblioteca vendorizada de propósito
  geral (~80 arquivos); apenas o caminho de código percorrido pela integração é auditado.
- `notebooks/` e `core/` (scaffolding vazio).
- Estilo, performance e refatoração — só entram achados que afetam corretude, precisão
  ou robustez.

## Estrutura: três fases

### Fase 1 — Auditoria profunda por estágio

Leitura crítica na ordem do fluxo de dados, comparando o código com a teoria de cada
método.

**1.1 Multifocus** (referência: shape-from-focus, Nayar & Nakagawa):

- Indicadores de foco: medem conteúdo de alta frequência *local*? Janelas, bordas,
  normalização entre escalas, viés em regiões sem textura.
- `argmax_fuzzy`: validade matemática da interpolação sub-pixel do pico (espaçamento
  uniforme dos `zf`, empates, picos na borda do stack).
- `depth_refinement` (graph-cut): escalas compatíveis entre termo de dados e suavidade;
  a discretização de labels preserva o sub-pixel?
- `mosaic`: validade do `iSel` (calculado nas imagens médias) para cada `L*`
  individualmente — premissa central do híbrido; costura sem artefatos radiométricos.
- `image_alignment`: consistência do alinhamento entre o stack do `iSel` e os stacks
  por luz.

**1.2 Fotométrico** (referência: Woodham; mínimos quadrados ponderado/robusto):

- `wps.estimate_normals_argmax_lstsq_robust`: montagem de `I = L·n`, ordem/orientação
  de `lights.npy`, pesos, critério de robustez (sombra/highlight), albedo, normalização.
- `rps` (L2/L1/SBL/RPCA): formulações corretas; consistência de convenções com `wps`.
- Radiometria: linearidade das entradas, gamma, faixas 0–1 vs 0–255.

**1.3 Integração** (referência: integração de campo de gradientes; multigrid):

- Python (`hybrid/integrate.py`): conversão normais → gradientes (`p = -nx/nz`, sinais,
  `nz≈0`), hints/pesos, unidades (pixel vs mundo).
- C: sistema de Poisson discreto (`pst_imgsys`), pesos/buracos, restrição/prolongamento
  do multigrid (`*_shrink`/`*_expand`), convergência, fronteiras.

**1.4 Infra/IO:**

- FNI ↔ array: ordem de eixos, origem da imagem, canais, precisão, round-trips;
  PNG 8 vs 16 bits.

### Fase 2 — Passada transversal de convenções e contratos

Foco exclusivo nas costuras entre estágios. Técnica: para cada convenção, rastrear a
cadeia completa produtor → arquivo → consumidor.

1. **Eixo z / profundidade:** direção de crescimento de `zf`, `zMos` e da altura `Z` do
   integrador; sinal e escala dos hints.
2. **Normais e luzes:** `nz` aponta para a câmera? `lights.npy` no mesmo sistema de
   coordenadas (orientação de y)? Conversão normal → gradiente consistente com o C.
3. **Origem/orientação da imagem:** numpy `[linha, coluna]` origem no topo vs FNI/C;
   flips silenciosos que se cancelam em pares — verificar a cadeia toda.
4. **Escala dos gradientes:** altura-por-pixel vs por unidade de mundo; hints em
   índice-`zf` vs altura física.
5. **Radiometria entre etapas:** faixa/gamma dos `sMos` escritos vs lidos; semântica do
   canal de confiança de `zMos_with_confidence` entre Python e C.
6. **Contratos de arquivo:** paths injetados por `hybrid/main.py` vs consumidos pelos
   `main`s; ordem dos `L*` vs linhas de `lights.npy`.

**Saída:** tabela de convenções (convenção × produtor × consumidor × veredito) +
achados no formato padrão.

### Fase 3 — Testes sintéticos com ground truth analítico

Confirmam ou refutam achados das fases 1–2. Vivem em `tests/` como suíte permanente de
regressão (pytest já configurado). Imagens pequenas (~64–128 px). Binário C requerido
para os testes de integração; se não compilar, o bloqueio é reportado explicitamente.

- **3.1 Unitários por método** (sob demanda, para confirmar suspeitas):
  - Multifocus: stack sintético com desfoque gaussiano sobre superfície conhecida →
    erro de profundidade ≪ espaçamento entre planos focais.
  - Fotométrico: superfície analítica + render Lambertiano → erro angular < ~1° sem
    ruído; robustez com sombra/saturação injetadas.
  - Integração: campo de gradientes analítico → binário C → altura vs verdade (a menos
    de constante); caminhos com pesos/buracos e hints.
- **3.2 Convenção:** round-trip FNI com padrão assimétrico (Python↔Python e Python↔C);
  rampa monotônica atravessando fotométrico+integração para detectar inversões/flips.
- **3.3 End-to-end híbrido** (roda sempre): dataset sintético completo no layout
  `L<n>/zf<m>/sVal.png` + `lights.npy` + `sharp/hAvg.png`, `hybrid/main.py` de ponta a
  ponta, RMSE da altura final vs ground truth.

Os testes seguem TDD invertido: escreve-se o teste que *deveria* passar; a falha é a
evidência do achado.

## Relatório final

`docs/superpowers/reports/2026-06-04-method-audit.md`, contendo:

1. **Sumário executivo** — contagem por severidade; 3–5 achados mais importantes.
2. **Achados por estágio**, cada um com:
   - `ID` (`MF-xx`, `PS-xx`, `INT-xx`, `IO-xx`, `CONV-xx`)
   - Localização `arquivo:linha`
   - Tipo: **conceitual** | **implementação**
   - Severidade: **crítico** (corrompe o resultado científico) | **alto** (erro
     mensurável) | **médio** (degrada robustez/precisão em casos comuns) | **baixo**
     (caso de borda)
   - Status: **confirmado** (com referência ao teste) | **suspeita** (justificativa
     teórica)
   - Descrição, evidência, sugestão de correção (não aplicada)
3. **Tabela de convenções** (Fase 2).
4. **Resultado end-to-end** (RMSE sintético) como linha de base do estado atual.
5. **Apêndice:** o que foi auditado e o que ficou fora.

## Execução

- Leitura em sessões por estágio; subagentes de exploração podem fazer varreduras
  paralelas, mas todo achado passa por verificação antes de entrar no relatório.
- Ordem: Fase 1 (1.1 → 1.4) → Fase 2 → Fase 3 (3.2 e unitários sob demanda → 3.3) →
  consolidação do relatório.
- Critério de pronto: todos os arquivos do escopo lidos, as 6 convenções rastreadas com
  veredito, teste end-to-end executado (ou bloqueio documentado), relatório consolidado.
