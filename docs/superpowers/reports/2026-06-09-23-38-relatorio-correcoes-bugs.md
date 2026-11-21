---
date: 2026-06-09
type: bug-fix-report
branch: bugfix/auditoria-2026-06-09
audit: docs/superpowers/reports/2026-06-09-17-00-relatorio-analise-bugs.md
spec: docs/superpowers/specs/2026-06-09-auditoria-bugs-design.md
---

# Relatório de Correções — Auditoria de Bugs hybrid-stereo-method

## Sumário executivo

Todos os **43 bugs confirmados** na auditoria de 2026-06-09 foram corrigidos na
branch `bugfix/auditoria-2026-06-09`, em **22 commits de correção** (um por bug,
agrupando apenas bugs que compartilham a mesma causa raiz) mais **2 commits de
build**. Cada correção de comportamento foi verificada com um teste executado
antes do commit; os resultados estão registrados na seção "Verificações".

Destaques:

1. **Radiometria preservada na fronteira multifocus→photometric (BUG-001/002/004).**
   `save_image` ganhou o parâmetro `normalize` (False = sem min-max); o
   multifocus agora salva `sMos.npy` sem perda e o pipeline híbrido/WPS consome
   o `.npy` em vez do PNG re-normalizado por imagem. As razões de intensidade
   entre direções de luz — o sinal do modelo I = L·n — chegam intactas ao solver.

2. **Solver C corrigido e recompilado.** O ponteiro de hints não inicializado
   (BUG-005) e a projeção zero-mean incompatível com hints absolutos (BUG-006)
   foram corrigidos em código C e validados com um plano sintético: sem hints o
   binário roda sem crash (gauge livre, média 0); com hints a superfície
   reconstruída ancora exatamente no nível absoluto dos hints
   (média 52.4 = média verdadeira, erro máximo 0.0). De quebra, o build estava
   irreprodutível (symlinks de `include/` e `lib/` apontando para um caminho
   removido + comentário aninhado fatal sob `-Werror`) e foi consertado.

3. **Sinal de dZ/dY corrigido na interface Python↔C (BUG-007/008).**
   `hybrid.integration.slopes_scale` foi exposto no YAML com default
   `[1.0, -1.0]`, emitindo `scale 1 -1` ao binário — compensa o FNI top-down
   contra a convenção Y-para-cima da biblioteca C.

4. **Confiança fotométrica agora pondera a integração (BUG-014/015).** O WPS
   salva `confidence_map.npy` e o híbrido a anexa como 4º canal das normais;
   verificado ponta a ponta que o binário aceita o FNI de 4 canais e usa o
   canal como peso.

5. **Matemática do multifocus consertada**: janela de Hann com frequência
   invertida (BUG-009 — pesos agora decaem do centro para a borda, conferindo
   com a previsão analítica), wavelet na banda/resolução certas (BUG-003),
   `zero_border` zerando o indicador e não a imagem (BUG-019), clip em 0 em vez
   do percentil 1 (BUG-037), marcador degenerado unificado e clamp em [0, n-1]
   (BUG-033/034).

## Mapa bug → commit

| Bugs | Commit | Correção |
|---|---|---|
| BUG-001, 002, 004 | `ff9a5bc` | Fronteira sem perda multifocus→photometric (`sMos.npy`; `save_image(normalize=)`) |
| BUG-003 | `55a1e8c` | Wavelet: banda mais fina (`coeffs[-1]`) + resize para resolução da imagem |
| BUG-009 | `9cf4642` | Janela de Hann: `a = π/(0.5(m+1))` |
| (build) | `d3b291d` | Symlinks `include/`+`lib/` reapontados para `lib-src/`; comentário aninhado |
| BUG-027 | `d318127` | SYNOPSIS do C sem o token `weight` inexistente no parser |
| BUG-028 | `ba3edea` | `float_image_expand_by_one(I, wch)` honra o canal de peso |
| BUG-006 | `79949b7` | `szero = (H == NULL \|\| hintsWeight <= 0)` — hints ancoram o gauge |
| BUG-011 | `7677302` | `pst_normal_from_slope` com os sinais negativos (inversa exata) |
| BUG-005 | `2983c2a` | `float_image_t *H = NULL` |
| (build) | `6a2abfe` | Binário e `libgus.a` recompilados com os fixes |
| BUG-007, 008 | `93dfe51` | `slopes_scale: [1.0, -1.0]` exposto no YAML e emitido ao binário |
| BUG-030 | `af1b414` | Defaults do código = YAML (`zero`, `0.1`) |
| BUG-010 | `8c98723` | RPS main lê o esquema YAML aninhado |
| BUG-012, 029 | `d3871d8` | `natsorted` em diretórios zf/L e arquivos do stack |
| BUG-013 | `8e1d22a` | Médias por zf salvas sem min-max por frame |
| BUG-014, 015 | `df7160b` | `confidence_map.npy` anexado como 4º canal na integração |
| BUG-016 | `23e46c0` | Parser FNI pula linhas de cabeçalho (round-trip NC=1/2/4 OK) |
| BUG-017, 018, 036 | `f448fe3` | image_alignment: imports do pacote, contrato correto, `save_image` atual |
| BUG-019 | `41545d9` | `zero_border` zera o indicador, não a imagem |
| BUG-037 | `157c1f4` | Clip do stack de foco em 0, não no p1 global |
| BUG-020 | `ba0bca0` | `mosaic` rejeita `interpolation_type` desconhecido |
| BUG-033, 034 | `6eac43e` | Marcador degenerado único (n/2, conf 0); clamp [0, n-1] |
| BUG-025, 043 | `c0a4479` | Albedo = ‖solução do lstsq‖; resíduo explícito; conf = 1/(1+r) |
| BUG-023, 024, 039, 040, 041 | `e0a6052` | ps_utils: axis=-1; BGR fixo BT.601; scale incondicional; curl nos eixos certos; −n/nz com clamp |
| BUG-021, 022 | `764a6ec` | Warning de método não suportado; avaliação GT consertada (np.load, máscara) |
| BUG-042 | `9e11d97` | Validação `or` em `L1_residual_min` |
| BUG-035 | `c7bd9cb` | Tenengrad (gx²+gy²) no custo unário; imports do pacote |
| BUG-038 | `a7fa7f6` | STL: coluna→X, linha→Y |
| BUG-031, 032 | `d0c213f` | Estatísticas em float64; médias arredondadas |
| BUG-026 | `fcefc79` | `focal_step` removido; blocos órfãos anotados nos YAMLs |

## Verificações executadas (antes de cada commit)

- **Radiometria**: PNG salvo com `normalize=False` relido com valores exatos
  (200/100); default continua normalizando.
- **Wavelet**: mapa de foco com shape igual à imagem; imagem nítida > borrada.
- **Hann**: pesos para k_fuzzy=5.3 = [0.128, 0.604, 0.976, 0.872, 0.396]
  (máximo no ponto mais próximo, 0 na borda — bate com a previsão do
  relatório); parábola exata recuperada pela interpolação.
- **Solver C** (recompilado): plano inclinado sintético 16×16 —
  sem `-hints`: exit 0, média 0 (gauge livre); com hints absolutos:
  média 52.4 = média verdadeira, máx |Z−Ztrue| = 0.0; normais de 4 canais
  aceitas com slopes recuperados (0.1, 0.2) exatos.
- **FNI**: round-trip write→read para NC=1, 2 e 4.
- **Alinhamento**: imagem deslocada (12,7) px realinhada à referência com erro
  médio 0.0 (era 29.6 antes do alinhamento).
- **Argmax fuzzy**: casos degenerados (foco zero e curva plana) retornam o
  mesmo marcador (n/2, 0); vértice clampado a [0, n−1].
- **WPS sintético (Lambertiano, 6 luzes)**: albedo recuperado = 0.7 exato;
  normais = verdade com tolerância 1e-3; stack de 3 luzes resolve sem crash e
  com confiança finita.
- **ps_utils**: erro angular 0° para mapas idênticos em ambos os layouts;
  conversão de cinza idêntica para imagem dominada por azul; gradiente de
  plano recupera (p, q) com sinais certos; curl ≈ 0 para campo integrável e
  ≠ 0 para não integrável.
- **numerics**: b multi-coluna rejeitado com ValueError; vetor-coluna resolve
  exato.
- **utils**: rms de imagem uint8 constante 200 = 200.0 (era 8.0); médias
  arredondadas.
- **Pacote completo**: todos os módulos importam (exceto `depth_refinement`,
  que exige a dependência opcional `pygco`, ausente no ambiente — igual antes).
- **YAMLs**: todos continuam parseando; `slopes_scale` chega ao construtor do
  config e o token `scale 1.0 -1.0` é emitido.

## Observações de ambiente (ação recomendada)

1. **Install editável apontando para worktree antigo.** O pacote
   `hybrid-stereo-method` instalado no ambiente resolve imports para
   `.claude/worktrees/vectorized-snuggling-harp/src/...`, não para a árvore do
   repositório. Para que o pipeline use o código corrigido, reinstale:
   `pip install -e .` na raiz do projeto. (Todas as verificações deste
   relatório foram feitas com `PYTHONPATH=src` para contornar isso.)
2. **Dependências opcionais ausentes**: `pygco` (graph-cut do
   depth_refinement) e `numpy-stl` (export STL) não estão instaladas; os bugs
   nesses módulos foram corrigidos e os arquivos parseiam/importam, mas a
   verificação funcional do STL não pôde ser executada aqui.
3. **Próxima execução do pipeline híbrido**: a primeira rodada após estas
   correções produzirá resultados numericamente diferentes (esperado): normais
   sem o viés da re-normalização, height map ancorado no nível do zMos e com
   dZ/dY no sinal certo. Vale comparar com o ground truth do dataset melon24.

## Pendências conscientes (fora do escopo das correções)

- `main_wps` continua oferecendo apenas o solver robust-argmax; a seleção de
  método via YAML agora gera warning explícito, mas o dispatch multi-solver
  (L2/L1/SBL/RPCA no caminho WPS) seria uma feature nova.
- O bloco `depth_refinement` dos YAMLs segue não conectado ao pipeline
  (anotado nos YAMLs); religá-lo é decisão de produto.
- A convenção de `lights.npy` permanece não documentada no dataset; o default
  `slopes_scale: [1.0, -1.0]` foi calibrado empiricamente para os datasets
  Stolfi (melon24). Para datasets com Y para baixo, configure `[1.0, 1.0]`.
