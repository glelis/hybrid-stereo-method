# Sweep v2: C, integração (d1, d2; 2026-11-22)

Tabelas: `C1.md`, `C2.md` e `C3.md` (com os `.csv`). São 40 runs, nenhum falhou.

## C1: `hints_weight` × `initial_method`

| hints_weight | 0 (sem hints) | 0,01 | 0,03 | 0,05 | **0,1** | 0,2 | 0,3 |
|---|---|---|---|---|---|---|---|
| d1 r | 0,763 | 0,973 | 0,981 | 0,983 | **0,9836** | 0,9834 | 0,983 |
| d2 r | 0,602 | 0,975 | 0,982 | 0,984 | **0,9849** | 0,9849 | 0,984 |

- **O 0,1 atual é o pico** (interior), igual nos dois. Mesmo com o multifocus bem melhor depois de A2,
  pesar mais os hints não ajuda: entre 0,1 e 0,2 dá empate, e 0,3 já piora.
- Sem hints a altura colapsa (0,60–0,76): nos datasets com textura, os hints são indispensáveis.
- **`initial_method` não tem efeito nenhum.** `zero` e `hints` dão métricas idênticas em todos os pesos.
  O binário recebe `-initial hints` (está no parameters.yaml). A hipótese é que a integração
  multiescala só usa o chute inicial no nível mais grosso, que converge por completo. Não foi verificado no
  código C.

## C2: sensibilidade de `pixel_size` (só reportado)

O r varia menos de 0,0002 entre ×0,8 e ×1,25 do calibrado nos dois datasets. Não há sinal de erro de
calibração: com hints, a escala do zMos e a das normais se acomodam. Nada a escalar.

## C3: a medida de foco depois de B e C1 (redesenhado; ruling no ledger)

Os top-3 de A1 entram com o σ 3,5 de A2, o solver de B e `hints_weight` 0,1:

| | d1 | d2 |
|---|---|---|
| fourier r0,05 | **0,9836** | 0,9844 (empate: Δr 0,0005, RMSE +1,8 %) |
| fourier r0,02 / laplaciano k9 | 0,9746 / — | — / **0,9851** |
| fourier r0,03 / laplaciano k5 | 0,9800 / — | — / 0,9849 |

- Nada passa à frente do centro fora do empate, então o centro fica promovido.
- **Fourier 0,05 + σ 3,5 empata com o melhor nos dois datasets dev**, por isso é o candidato natural a
  config única. O laplaciano k9 vale como alternativa para cenas com brilho (d2). Os dois vão para V.
