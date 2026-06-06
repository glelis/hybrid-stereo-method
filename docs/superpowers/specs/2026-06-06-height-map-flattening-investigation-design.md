# Investigação: achatamento do mapa de altura final do pipeline híbrido

**Data:** 2026-06-06
**Status:** aprovado
**Run de referência:** `data/results/hybrid_stereo/20260606_1126_2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/`
**Dataset:** `data/raw/hybrid_stereo/2025-03-08-stQ-melon24-amb0.00-glo0 (2).50/` — o GT em `sharp/` corresponde ao objeto **melon14** (o nome da pasta externa "melon24" está desatualizado; confirmado pelo usuário).

## Problema

O mapa de altura final do pipeline híbrido está visivelmente ruim. A hipótese inicial do
usuário: um grande outlier está achatando o resto do mapa. A exploração preliminar
confirmou parcialmente e revelou sintomas adicionais:

1. **Penhasco de fundo, não pixel isolado:** o objeto vive em ~[18, 68] de altura, mas
   ~7% dos pixels caem para ~−455 — todas as primeiras 15 linhas do topo estão nessa
   faixa, estendendo-se até a linha ~363 em algumas colunas. Amplitude total ~790, da
   qual o objeto usa ~50 → relevo real comprimido em ~6% da escala.
2. **Anti-correlação com o GT:** `integration_height.pearson_r = −0.20`, slope afim
   `a = −35`. O mapa final não é só achatado — é *anti-correlacionado* com o GT.
3. **Mosaicos catastróficos:** PSNR 1,5–4 dB vs `sharp/sVal` (imagens não relacionadas
   dão ~5–10 dB); SSIM ~0,18.
4. **Multifocus quase não-informativo:** `multifocus_depth.pearson_r = 0.27`,
   RMSE afim ≈ 17.795 com `gt_std` ≈ 18.468; `focus_selection.exact_match = 29,6%`.
5. **Sem máscara:** `mask: False` na config e não existe `mask.png` no dataset — o fundo
   participa da estimativa de normais e da integração.

## Objetivo e critério de sucesso

Identificar as causas raiz dos sintomas acima e **corrigir tudo que for necessário**
(bugs, configuração e mudanças algorítmicas moderadas — escopo aprovado pelo usuário).

**Sucesso:** após as correções, re-run completo neste dataset com:

- `integration_height.pearson_r` fortemente **positivo**;
- RMSE afim bem abaixo de `gt_std` (18.468);
- penhasco de fundo eliminado, ou confinado a região justificadamente mascarada.

Causas não corrigíveis (limitações intrínsecas do método) ficam documentadas com
evidência no relatório de investigação.

## Hipóteses iniciais (ordenadas por evidência atual)

| # | Hipótese | Evidência a favor |
|---|----------|-------------------|
| H1 | Fundo sem máscara → normais degeneradas no fundo → integração cria rampa/penhasco | Banda no topo a −455; `mask: False`; sem `mask.png` |
| H2 | Inversão de sinal/convenção de eixo (z↔altura, ny, ordem de linhas) em alguma fronteira do pipeline | Anti-correlação *global* (r = −0.20, a = −35) é assinatura de sinal trocado, não de ruído |
| H3 | Mosaicos mal construídos (bug de montagem ou iSel ruim propagado) | PSNR 2–4 dB vs `sVal` |
| H4 | Hints do multifocus puxando a integração para longe da solução | `use_hints: True` com profundidade quase não-informativa (r = 0.27) |
| H5 | Seleção de foco fraca na origem da cadeia | `exact_match = 29,6%`; fourier sem pré-processamento |
| H6 | A avaliação compara coisas desalinhadas (grade de cantos 423×513 vs GT, dtype, resize) | Altura final tem +1 pixel por eixo; barata de descartar |

## Abordagem: bisseção reversa do pipeline

Partir do mapa final e caminhar de trás para frente. Cada fase responde **"o erro
catastrófico já existe neste estágio?"** com evidência quantitativa registrada antes de
avançar. A ordem é adaptativa: se F1 mostrar integrador saudável, o foco desloca para
F2–F4; se mostrar inversão, H2 vira prioridade.

- **F0 — Sanidade da avaliação e do GT** (descarta H6; confirma melon14):
  conferir alinhamento/resize/dtype na comparação da avaliação; correlacionar
  `sharp/sVal` com a imagem média do stack para confirmar correspondência GT↔imagens.
- **F1 — Integrador isolado** (testa H2, H4): rodar o binário C com as **normais do GT**
  (`sNrm` decodificado). Altura boa → integrador e convenções OK; altura
  anti-correlacionada → fronteira do bug de sinal encontrada. Rodar também com
  `use_hints: False` para medir o efeito isolado dos hints.
- **F2 — Normais** (testa H1, H2): erro angular do normal map estimado vs GT, separado
  por região (objeto × fundo). Sobrepor o mapa de erro angular com a máscara do
  penhasco: coincidência confirma H1.
- **F3 — Mosaicos** (testa H3): comparar cada `sMos` por luz com o `sharp/` interno do
  respectivo `L*` (não apenas o da raiz); PSNR/SSIM por região; separar "iSel errou" de
  "bug de montagem".
- **F4 — Multifocus** (testa H5): distribuição espacial do erro de iSel/profundidade —
  erro concentrado em fundo/regiões sem textura é esperado; erro no objeto é problema
  real.

## Ablações causais

Após localizar o(s) estágio(s) culpado(s), cada ablação muda **um** fator e mede o
efeito nas métricas finais:

| Ablação | Fator isolado | O que confirma |
|---------|--------------|----------------|
| A1 | `use_hints: False` | Contribuição dos hints do multifocus (H4) |
| A2 | Integração com máscara de fundo (derivada do GT `hAvg` ou por limiar de intensidade) | Se o penhasco some sem o fundo (H1) |
| A3 | Integração com normais GT (`sNrm`) | Teto de qualidade do integrador (reusa F1) |
| A4 | Normais estimadas a partir dos `sVal` sharp do GT (em vez dos mosaicos) | Degradação atribuível aos mosaicos vs ao estimador WPS |

A3 (integrador puro) + A4 (estimador puro) permitem atribuir a degradação a cada elo da
cadeia de forma quantitativa.

## Estratégia de correção

Para cada causa confirmada:

- **Bug de código** (ex.: sinal/convenção em `hybrid/integrate.py`, montagem do
  mosaico): fix direto + teste de regressão que captura a convenção correta.
- **Lacuna funcional** (ex.: ausência de suporte a máscara na integração/normais):
  implementação mínima — derivar máscara quando disponível e zerar pesos do fundo — sem
  redesenhar o pipeline.
- **Configuração** (ex.: `hints_weight`, pré-processamento do focus measure): ajuste no
  YAML do experimento com justificativa registrada.

Cada correção é commitada separadamente, com a evidência da ablação correspondente na
mensagem do commit. Correções independentes não se bloqueiam.

## Validação final

Re-run completo do pipeline híbrido neste dataset + avaliação automática
(`python -m hybrid_stereo_method.evaluation.main`). O relatório final traz a tabela
**antes × depois** de todas as métricas (`integration_height`, `normals`, `mosaics`,
`multifocus_depth`, `focus_selection`), com os critérios de aceite do objetivo.

## Entregáveis

1. **Relatório de investigação** —
   `docs/superpowers/investigations/2026-06-06-height-map-flattening-investigation.md`:
   hipóteses, evidências por fase, veredictos (confirmada/refutada/inconclusiva),
   tabela antes×depois. Commitado.
2. **Correções no código/config** — commits individuais com testes onde aplicável.
3. **Scripts de diagnóstico** — efêmeros; não viram módulo do pacote. O que for
   genuinamente reutilizável fica anotado no relatório como trabalho futuro (escopo
   "diagnóstico + correções", sem ferramenta nova, por decisão do usuário).

## Fora de escopo

- Novas ferramentas de diagnóstico permanentes no módulo `evaluation`.
- Refatorações não relacionadas às causas confirmadas.
- Outros datasets além do run de referência (a validação usa apenas este; generalização
  fica como trabalho futuro).
