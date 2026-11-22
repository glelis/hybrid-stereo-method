# Sweep v2: B, solver fotométrico (d1, d2; 2026-11-22)

Tabela: `B.md` / `B.csv`. São 62 runs: a grade 4×5 do plano mais a extensão 3×4 depois do canto. Nenhum falhou.

## Promovido: `shadow_threshold` 0,05 + `outlier_threshold_multiplier` 1,5 (nos dois)

| dataset | r altura | RMSE | normais (° médio / p95) | centro A2 |
|---|---|---|---|---|
| d1 | **0,9836** | 1728 | 0,19 / 0,13 | 0,9784 · 0,42 / 0,36 |
| d2 | **0,9849** | 2960 | 1,32 / 7,87 | 0,9810 · 1,84 / 11,44 |

Pela regra, o vencedor do d1 seria sh 0,2 (0,9836, RMSE 1726), empatado com sh 0,05. Promovi sh 0,05
com `--pick` (ruling no ledger), porque sh 0,2 derruba o d2 (0,9766).

## Achados

- **A expectativa registrada antes estava errada:** no d1 o efeito não é nulo. A rejeição de outliers
  melhora as normais (0,42° → 0,19°) e a altura (+0,005).
- **A rejeição de outliers é o fator dominante.** O r melhora de forma monotônica de om ∞ até om 1,5 e
  colapsa em om 1,25 (d1 0,951, d2 0,972: rejeição demais deixa poucas luzes por pixel).
  A grade original parava em om 2, que venceu no canto, e foi estendida.
- `shadow_threshold` tem efeito pequeno até 0,05. Em 0,1–0,2 fica neutro no d1 e, no d2, ruim com om 1,5.
- O ganho em B é menor que em A2, mas soma: comparado ao base (S0), d1 vai de 0,9705 a 0,9836 e d2 de 0,9751 a 0,9849.
