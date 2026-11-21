# Sweep v2 — A1: medida de foco × pré-processamento (d1, d2; 2026-11-21)

Tabela completa: `A1.md` / `A1.csv` (89 configs × 2 datasets, 178 runs, 0 falhas).

## Vencedores

| dataset | vencedor | r altura | top-3 promovido |
|---|---|---|---|
| d1 melon14 glo0 | fourier r0,05 + smooth + mediana | 0,9705 | fourier r0,05 / r0,02 / r0,03 |
| d2 melon14 glo0,5 | laplaciano k9 + smooth + mediana | 0,9752 | lap k9 / lap k5 / fourier r0,05 |

- O base (legacy13 = fourier 0,05 sm1 med1 zb0) é o vencedor no d1 e empata no d2
  (0,9751 vs 0,9752). **A1 não encontrou nada melhor que o base**; o platô é largo:
  fourier r 0,02–0,07 com smooth+mediana fica todo a < 0,0015 do topo.
- Nenhum vencedor na borda da grade (r 0,05 e k9 são interiores) → sem pontos extras.

## Efeitos principais (média do r da altura sobre o resto da grade)

| fator | d1 | d2 |
|---|---|---|
| smooth on / off | 0,958 / 0,946 | 0,963 / 0,955 |
| mediana on / off | 0,954 / 0,950 | 0,963 / 0,955 |
| zero_border off / on | 0,955 / 0,949 | **0,967 / 0,951** |
| fourier r 0,10 (pior raio) | 0,939 | 0,955 |
| fourier r 0,02 | 0,957 | 0,949 (pior raio no d2) |
| laplaciano k5 (pior kernel d1) | 0,944 | 0,961 |
| wavelet | 0,892 | 0,953 |

- smooth e mediana ajudam sempre; zero_border sempre atrapalha (mais no d2, com brilho
  especular). Os 4 melhores de cada dataset têm `sm1_med1_zb0`.
- Interação: sem denoise o raio pequeno colapsa no d2 (r0,02 sm0 med0: 0,935) — o
  ruído especular do glo0,5 exige filtragem; com denoise r0,02 volta a 0,968.
- O r do multifocus isolado e o da altura final correlacionam pouco no topo (lap k9 tem
  mf 0,90 vs fourier 0,885 no d2, altura igual): a integração com normais absorve
  diferenças pequenas de zMos. Wavelet é o único método claramente inferior.
