# Design: Auditoria Profunda de Bugs — Repositório Completo

**Data:** 2026-06-09
**Status:** Aprovado para execução
**Entregável:** Relatório apenas (nenhuma alteração de código)
**Saída:** `docs/superpowers/reports/2026-06-09-relatorio-analise-bugs.md`

## Objetivo

Realizar uma análise profunda e completa em busca de bugs em todo o repositório
`hybrid-stereo-method`, com foco em:

1. **Correção matemática/algorítmica** — erros em fórmulas, convoluções, sistemas
   lineares, integração de superfície, normalizações, sinais trocados, eixos
   invertidos; bugs que produzem resultado errado silenciosamente.
2. **Integração entre módulos** — inconsistências entre multifocus, photometric e
   hybrid: convenções de eixos/unidades divergentes, parâmetros passados errado,
   interface Python↔C.

Fora de escopo: correções de código (relatório apenas), auditoria interna da
biblioteca C vendorizada além do solver, estilo/qualidade geral de código.

## Escopo

### Unidades de revisão (9 finders por unidade)

| # | Unidade | Arquivos |
|---|---------|----------|
| 1 | Multifocus — núcleo | `argmax_fuzzy.py`, `depth_refinement.py`, `math_utils.py`, `mosaic.py` |
| 2 | Multifocus — indicadores | `indicators/laplacian.py`, `fourier.py`, `wavelet.py`, `non_linear_res.py`, `applicator.py` |
| 3 | Multifocus — alinhamento e pipeline | `image_alignment.py`, `utils.py`, `main.py` |
| 4 | Photometric — solvers | `solvers/numerics.py`, `rps.py` |
| 5 | Photometric — WPS e utils | `wps.py`, `ps_utils.py`, `main.py`, `main_wps.py` |
| 6 | Hybrid | `hybrid/integrate.py`, `hybrid/main.py` |
| 7 | Infrastructure | `infrastructure/io/image_io.py`, `infrastructure/utils.py`, `infrastructure/visualization.py`, `core/` |
| 8 | C — solver | `csrc/integrate_recursive/gus_integrate_recursive.c`, `lib-src/pst_integrate*.c`, `pst_imgsys*.c`, `pst_slope_map.c`, `pst_height_map.c` |
| 9 | Interface Python↔C | `hybrid/integrate.py` ↔ binário C: argumentos CLI, formatos de arquivo (FNI), shapes, convenções de eixo |

A biblioteca vendorizada `csrc/integrate_recursive/lib-src/` (~120 arquivos,
estilo Jorge Stolfi/Unicamp) só é lida quando o código customizado chama algo
nela; não é auditada internamente além dos arquivos do solver listados na
unidade 8.

### Finders transversais (3 finders de integração)

| # | Dimensão | Foco |
|---|----------|------|
| 10 | Convenções geométricas | Sistemas de coordenadas, orientação de eixos (Y para cima/baixo), sinal de Z, unidades de profundidade entre multifocus → photometric → hybrid |
| 11 | Fluxo de configuração | YAML (`configs/`) ↔ código: parâmetros lidos com nome errado, defaults divergentes, unidades trocadas |
| 12 | Contratos de dados | Shapes, dtypes, faixas de normalização (0–1 vs 0–255), NaN como máscara, entre saídas de um módulo e entradas do próximo |

## Arquitetura do workflow (multi-agente, 4 fases)

Execução via tool Workflow (orquestração determinística, opt-in explícito do
usuário em 2026-06-09).

### Fase 1 — Mapear
2 agentes leem o pipeline de ponta a ponta (um seguindo o fluxo de dados
multifocus → hybrid, outro photometric → hybrid) e produzem um **mapa de
convenções**: o que cada fronteira de módulo espera em eixos, unidades,
formatos de arquivo e faixas de valores. O mapa é injetado no prompt de todos
os finders para que detectem desvios de convenção, não apenas bugs locais.

### Fase 2 — Encontrar
Os 12 finders (9 por unidade + 3 transversais) rodam em paralelo. Cada um
retorna achados estruturados via schema JSON: arquivo, linha, título,
descrição, evidência (trecho de código), severidade proposta
(crítico/alto/médio/baixo) e categoria (matemática/integração).

### Fase 3 — Verificar
Barreira de deduplicação (por arquivo+linha+tema, em código, não em agente).
Cada achado único é submetido a **3 verificadores adversariais independentes**,
cada um com uma lente distinta:

1. **Correção matemática** — a matemática alegada como errada está mesmo errada?
2. **Leitura literal do código** — o código realmente faz o que o achado afirma?
3. **Cenário de impacto** — existe entrada realista em que o bug se manifesta?

Prompt explícito para *refutar*; em caso de dúvida, refutar. Entra no relatório
o que sobrevive a **≥2 de 3** votos. Refutados vão para apêndice de descartados
com a justificativa.

### Fase 4 — Relatar
Agente de síntese produz o relatório final em português.

## Formato do relatório

Salvo em `docs/superpowers/reports/2026-06-09-relatorio-analise-bugs.md`,
estruturado para leitura humana e reuso por agente em sessões futuras:

```markdown
---
date: 2026-06-09
type: bug-audit
scope: full-repo (foco: matemática + integração)
method: multi-agent workflow, 12 finders, verificação adversarial 2-de-3
---

# Relatório de Auditoria de Bugs

## Sumário executivo
(contagem por severidade, os 3-5 achados mais importantes em prosa)

## Bugs confirmados
### Críticos / Altos / Médios / Baixos
Cada bug: ID estável (BUG-001…), arquivo:linha, categoria, descrição,
evidência (trecho), impacto no resultado, sugestão de correção.

## Cobertura
Unidades auditadas, finders que falharam, arquivos não lidos, lacunas.

## Apêndice: achados descartados na verificação
ID, alegação, motivo da refutação.
```

IDs estáveis (BUG-NNN) permitem referenciar achados em sessões futuras de
correção.

## Tratamento de falhas

- Finder ou verificador que falhar (retorno null) é registrado na seção de
  cobertura do relatório — nenhum corte silencioso.
- Achado cujo conjunto de verificadores ficar incompleto (<2 votos válidos) é
  marcado como "não verificado" no apêndice, não promovido a confirmado.

## Critérios de sucesso

1. Todas as 12 unidades/dimensões auditadas (ou lacuna explícita na cobertura).
2. Todo bug no corpo principal sobreviveu à verificação adversarial 2-de-3.
3. Cada bug tem localização exata (arquivo:linha) e evidência citada.
4. Nenhum arquivo do repositório modificado além do relatório (e deste spec).

## Próximo passo após aprovação

Este é um projeto de análise, não de implementação de código: o "plano de
implementação" é o próprio script do workflow descrito acima. Após aprovação
deste spec, o workflow é executado diretamente nesta sessão.
