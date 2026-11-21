# Fundo da imagem, e não as normais, limita a reconstrução (lamb-gls)

**Data:** 2026-11-21. **Status:** achado registrado; nada implementado (decisão do usuário).

## Contexto
A ideia inicial era melhorar as normais do fotométrico, que integradas sozinhas davam
Pearson 0,48 com a altura do GT. Antes de mexer em qualquer coisa, medimos onde está a
perda. Run base: `data/results/hybrid_stereo/20261121_1235_lamb-gls` (config calibrado:
`sy = -1`, `pixel_size = 1.78`, `hints_weight = 0.1`).

## Medições

As normais do pipeline contra as do GT (`sNrm`) dão erro angular mediano de 0,002°, média
de 1,3° e p95 de 4,1°. Na prática, já são as normais do GT.

Pearson da altura integrada contra o `hAvg`. "Objeto" é a máscara de primeiro plano do GT,
92 % da imagem.

| Integração | Objeto | Imagem toda |
|---|---|---|
| Só normais do pipeline (peso = confiança) | 0,711 | 0,479 |
| Só normais do pipeline, peso 0 no fundo | 0,722 | 0,536 |
| Só normais do GT, peso 0 no fundo | 0,724 | 0,537 |
| Híbrido atual (hints 0,1) | **0,881** | 0,633 |
| Híbrido, peso 0 no fundo | **0,969** | 0,645 |

## Conclusões
1. **Melhorar as normais traria ganho quase nulo:** com as normais perfeitas, o resultado
   passa de 0,71 para 0,72.
2. **O fundo (8 % da imagem) é o gargalo.** No fundo o `hAvg` vai de 0 a 65535 sem
   significado, e a borda do objeto tem um degrau que a integração de normais não consegue
   reproduzir. O fundo entra na integração com confiança alta (0,81 em média) e normais
   planas (nz ≈ 1), e arrasta a superfície do objeto.
3. **Zerar o peso do fundo na integração** leva o híbrido de 0,88 para **0,97** no objeto.
   O teste usou a máscara do GT: o pipeline precisaria detectar o fundo sozinho.
4. A métrica da "imagem toda" mistura o fundo sem significado com o objeto e subestima a
   qualidade da reconstrução.

## Objetivo definido pelo usuário
O que importa é **a reconstrução do objeto**. O fundo pode ser destacado ou separado.

## Caminho sugerido (para quando for retomado)
- **Detectar o fundo a partir das entradas, sem usar o GT.** Pistas: no `sMos.npy` o fundo é
  mais claro (mediana 32767 contra 12844 no objeto; p95 do fundo saturado em 65535) e suas
  normais são planas (nz ≈ 1). Validar a máscara obtida contra a de primeiro plano do GT
  (IoU).
- **Integração:** peso 0 nos pixels de fundo (canal 4 das normais e canal de confiança dos
  hints).
- **Avaliação:** reportar a métrica do objeto como principal e a da imagem toda como
  secundária; opcionalmente, salvar a máscara de fundo como artefato do run.
- **Referência de ganho esperado:** de 0,88 a 0,97 no objeto, se a máscara detectada for tão
  boa quanto a do GT.
