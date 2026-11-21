# Sweep v2: S0, que abrange calibração, smoke, determinismo e convergência (2026-11-21)

Tabelas geradas: `S0.md` e `S0conv.md` (nesta pasta).

## Bugs encontrados e corrigidos antes de qualquer número valer

1. **Faixa de escala não excluída nas pastas canônicas.** `scale_band_rows` só reconhecia
   o `WxH` no *início* do nome da pasta (`0512x0384-hs01-kr07`, do layout aninhado). Em
   `data/raw/datasets/hybrid/stQ-melon14-…-0512x0384-hs01-kr07` o tamanho fica no meio do
   nome, a faixa de 38 linhas entrava na avaliação e o d1 base dava r 0,881 em vez de 0,970.
   Também enviesava a calibração (`pixel_size` 1,78 em vez de 1,80). Corrigido, com teste. A
   calibração e o primeiro S0 foram refeitos.
2. **Pastas de luz com sufixo** (`L000-amb0.00-glo0.00`, d3) não eram reconhecidas pelo
   pipeline nem pela avaliação. Corrigido, com testes.
3. **Plano focal extra numa luz só** (d3: `zf060` existe só na L000). O pipeline passou a usar
   os planos comuns a todas as luzes (10 de 11), e a avaliação restringe o GT de foco aos
   planos do run.

## Achado: a máscara "objeto" do GT é a imagem inteira

Sem a faixa de escala, a frente do GT (`sNrm`) cobre 100 % da imagem nos 4 datasets
(`object_fraction = 1.0`). O "objeto 92 %" do relatório fundo-vs-normais era, na prática,
"imagem menos a faixa de escala" (38/422 = 9 %). Lá, o ganho 0,88 → 0,97 de "zerar o
peso do fundo" corresponde a tirar a faixa da integração e da avaliação. Por isso a métrica
no objeto coincide com a da imagem toda neste sweep. Ela foi mantida porque é barata e passa
a valer para datasets com fundo real.

## Calibração (`scripts/calibrate_dataset.py`, faixa excluída)

| dataset | sy=−1 / sy=+1 (Pearson, normais do GT) | pixel_size direto / via hAvg | adotado |
|---|---|---|---|
| d1 melon14 glo0 | 0,818 / 0,133 | 1,766 / 1,831 | 1,80 |
| d2 melon14 glo0,5 | 0,821 / 0,127 | 1,104 / 1,154 | 1,13 |
| d3 stF-melon24 | 0,870 / −0,001 | 1,086 / 1,193 | 1,14 |
| d4 white01 | 0,818 / 0,133 | 1,776 / 1,838 | 1,81 |

## Smoke (config base = legacy13: fourier 0,05 + smooth + mediana + zero_border off)

| dataset | r altura | r multifocus | foco mediano (frames) | normais (° médio) |
|---|---|---|---|---|
| d1 | 0,9705 | 0,611 | 0,60 | 0,84 |
| d2 | 0,9751 | 0,885 | 0,55 | 2,05 |
| d3 | 0,9440 | 0,830 | 0,56 | 3,15 |
| d4 | **0,0404** | −0,024 | 6,0 | — |

- legacy15 (laplaciano) fica abaixo do base em d1 (0,9605) e d2 (0,9699), embora o multifocus
  isolado dele seja um pouco melhor. Isso repete o padrão do sweep de 2026-07.
- **d4 (sem textura) quebra o híbrido.** O multifocus não tem informação (r ≈ 0), e com
  `hints_weight` 0,1 os hints dominam a integração. É o caso de estresse previsto: os estágios
  C e Vhints medem quanto o peso dos hints precisa cair.
- d3: o erro de foco em "frames" usa o passo mediano (7,5). Perto de z = 60, onde o passo é
  3,75, ele é só aproximado.

## Determinismo

Dois runs completos do d1 base com a mesma config deram `metrics.json` idêntico em todos os
blocos. O `tie_r` = 0,005 fica mantido.

## Convergência da integração (rerun sobre o base)

r da altura (objeto) por `max_iter` × `conv_tol`. O tempo é o do rerun, com 5 runs em paralelo.

| max_iter | d1 r | d2 r | tempo d1 |
|---|---|---|---|
| 2000 | 0,970532 | 0,975061 | 40–53 s |
| 5000 | 0,970473 | 0,975057 | 74–88 s |
| 10000 | 0,970462 | 0,975057 | 123–131 s |
| 20000 | 0,970462 | 0,975057 | 189–195 s |

`conv_tol` (5e-7 ou 1e-7) não muda nada. Todos ficam a menos de 0,0001 da referência.
**Adotado `max_iter: 5000`**, com margem para configs sem hints, que convergem mais devagar
no nível fino. Pela regra do plano, 2000 bastariam.

## Recursos

Cada run usa um núcleo só (~100 %): o runner usa `-j 6 --threads 2`. Run completo:
cerca de 6–8 min com 6 em paralelo; rerun de integração: cerca de 1,5 min. A poda deixa cerca de 89 MB por run.
