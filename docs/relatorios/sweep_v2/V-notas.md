# Sweep v2: V, validação multi-dataset (d1–d4; 2026-11-22)

Tabelas: `V.md` (agregado d1–d3), `Vhints.md` e `Vfinal.md` (com os `.csv`). São 40 runs de V, 30 de
Vhints e 15 de Vfinal. Nenhum falhou.

## 1. O vencedor de B não generaliza: artefato de normais inválidas

| config | d1 | d2 | **d3 (teste)** | normais válidas (d3) |
|---|---|---|---|---|
| legacy13 | 0,9705 | 0,9751 | **0,9440** | 100 % |
| legacy13 + solver B (om 1,5) | 0,9668 | 0,9786 | **0,8028** | 82 % |
| d1_final (σ 3,5 + B) | 0,9836 | 0,9844 | 0,8800 | 83 % |
| d2_final (lap k5 σ 3,5 + B) | 0,9854 | 0,9849 | 0,9246 | 83 % |

`outlier_threshold_multiplier` 1,5 faz `wps.py` gravar NaN quando sobram menos de 3 luzes após a
rejeição. Isso atinge 15 % dos pixels em d1/d2 e 18 % em d3. Lá, a integração cai nos hints. Em d1/d2,
que são o **mesmo objeto** (melon14), isso parecia ganho; em d3, não. O erro angular "melhor" de B era
seleção: a métrica de normais só conta os pixels válidos. Até om 2 as normais ficam 100 % válidas.

**Lições:**
- Os relatórios agora têm a coluna `photometric_normals.valid_fraction`. A escolha do solver exige que ela seja ≥ 0,99.
- Dois datasets dev do mesmo objeto não detectam overfit. Sem d3 como teste, o sweep teria
  recomendado uma config que piora 0,06 em outra cena.

## 2. Hints: o peso ótimo é 1,0, não 0,1 (com o solver original)

Laplaciano k9 + σ 3,5, solver original (sh 0,001, om 3):

| hints_weight | 0 | 0,03 | 0,1 | 0,2 | 0,3 | 0,5 | **1,0** | 2,0 | 5,0 |
|---|---|---|---|---|---|---|---|---|---|
| d1 | 0,799 | 0,968 | 0,979 | 0,982 | 0,983 | 0,984 | **0,9840** | 0,984 | 0,983 |
| d2 | 0,813 | 0,968 | 0,980 | 0,983 | 0,985 | 0,985 | **0,9856** | 0,985 | 0,984 |
| d3 | 0,815 | 0,921 | 0,941 | 0,950 | 0,953 | 0,956 | **0,9588** | 0,9596 | 0,959 |
| d4 (sem textura) | **0,808** | 0,448 | 0,093 | −0,04 | −0,11 | — | — | — | — |

- Em C1, que rodou sobre o solver B, o pico era 0,1. Lá os pixels NaN já dependiam só dos hints, e
  aumentar o peso não trazia nada. Com as normais completas, pesar mais o zMos de σ 3,5 (r mf 0,93–0,97)
  ajuda até ~1,0.
- O ganho do híbrido sobre o multifocus sozinho (`hybrid_gain`) tem o pico no mesmo ponto (1,28–1,39). Ou seja,
  a altura não está só copiando o zMos.
- **d4 (sem textura) precisa de `use_hints: false`**: com hints o resultado é ruído (o foco não tem informação), sem hints dá 0,81.

## 3. Reconfirmação com hints 1,0 (Vfinal)

| config (hints 1,0) | d1 | d2 | d3 | média | mín |
|---|---|---|---|---|---|
| **lap9 + σ 3,5** | **0,9840** | 0,9856 | **0,9588** | **0,9761** | **0,9588** |
| lap9 + σ 3,5 + sh 0,05/om 2 | 0,9839 | 0,9857 | 0,9552 | 0,9750 | 0,9552 |
| fourier 0,05 + σ 3,5 + om 2 | 0,9831 | 0,9851 | 0,9524 | 0,9735 | 0,9524 |
| fourier 0,05 + σ 3,5 | 0,9835 | 0,9851 | 0,9512 | 0,9733 | 0,9512 |
| legacy13 | 0,9598 | 0,9785 | 0,9508 | 0,9630 | 0,9508 |

O solver "seguro" (om 2) não acrescenta nada com hints 1,0. O solver original fica.

## Config recomendada

| | legacy13 (atual) | **recomendada** | Δ |
|---|---|---|---|
| d1 melon14 | 0,9705 | **0,9840** | +0,0135 |
| d2 melon14 glo0,5 | 0,9751 | **0,9856** | +0,0105 |
| d3 stF-melon24 (teste) | 0,9440 | **0,9588** | +0,0148 |
| d4 white01 (teste, sem textura) | 0,078* | **0,8082** (hints off) | +0,77 |

\* V (max_iter 5000). O S0 deu 0,040 no d4 porque rodou antes da troca para max_iter 5000 (com 10000).
O S0conv só mediu d1/d2; em d4, um problema mal condicionado, a convergência ainda mexe no r.

Recomendada: **laplaciano k9, smooth com σ 3,5, mediana 5, zero_border off, r_max 2, interpolação
linear, solver original (shadow 0,001, outlier 3), `hints_weight` 1,0**. Em cena **sem textura**, `use_hints: false`.
