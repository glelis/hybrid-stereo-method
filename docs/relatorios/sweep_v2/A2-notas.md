# Sweep v2: A2, refino do multifocus (d1, d2; 2026-11-22)

Tabela completa: `A2.md` / `A2.csv`. Foram 72 runs: 54 do OFAT, 12 da extensão de `smooth_sigma` e 6 da combinação. Nenhum falhou.

## Vencedores promovidos

| dataset | vencedor | r altura | RMSE | r multifocus | melhor A1 |
|---|---|---|---|---|---|
| d1 | fourier r0,05 + smooth σ 3,5 + mediana | **0,9784** | 1980 | 0,925 | 0,9705 / mf 0,611 |
| d2 | laplaciano k5 + smooth σ 3,5 + mediana | **0,9810** | 3320 | 0,962 | 0,9752 / mf 0,901 |

No d2, fourier r0,05 com σ 3,5 dá 0,9793. Pelo r é empate, mas o RMSE fica 4 % pior, então pela regra não empata.

## Achados

- **`smooth_sigma` é o parâmetro que mais pesa no multifocus.** O kernel binomial 3×3 do default
  equivale a σ ≈ 0,7. O r da altura sobe de forma monotônica até σ 3,5 e cai um pouco em 5,0, nos 6 centros. A grade
  original parava em 2,5, que venceu na borda; por isso foi estendida (ruling no ledger).
- O ganho no **multifocus isolado** é muito maior que na altura final. No d1, o r do zMos vai
  de 0,61 para 0,93 (σ 3,5) e 0,97 (σ 5). Os hints ficam bem mais confiáveis; vale reavaliar
  `hints_weight` no C1.
- Mexer em `r_max` (1, 3 ou 4 em vez de 2) sempre piora. A interpolação quadrática e `median_ksize` 3 dão efeito
  nulo ou negativo.
- `square` ajuda sozinho (+0,004 com σ padrão), mas não soma com σ 3,5. A combinação σ 3,5 + square
  empata ou piora nos 6 centros: os dois fazem o mesmo papel (suavizar o indicador).
