# Relatório de Análise: Módulo Estéreo Multifocal (Multifocus)

Este relatório apresenta uma análise técnica da implementação do módulo de Estéreo Multifocal (`src/hybrid_stereo_method/multifocus`), avaliando-o sob as perspectivas de Engenharia de Software e de Ciência da Computação (Física e Matemática), especialmente no contexto da sua integração com um processo híbrido de reconstrução 3D.

---

## 1. Pontos Positivos (O que está bom)

### 1.1. Integração com Processos Híbridos (Confiança)
A implementação tem uma visão clara de integração: o script `main.py` exporta o mapa de profundidade (`zMos`) junto com um mapa de confiabilidade (`wSel`). Em métodos híbridos (ex: Multifoco + Estéreo Fotométrico), ter uma medida de incerteza/confiança por pixel é essencial para a fusão de sensores e otimização global.

### 1.2. Precisão Sub-frame (Matemática do Argmax Fuzzy)
O cálculo em `argmax_fuzzy.py` não se limita a escolher a imagem de maior foco discreta. A utilização de uma regressão polinomial quadrática (parábola) aos pontos ao redor do pico máximo para encontrar a profundidade contínua inter-frames (`k_fuzzy = -B / (2 * A)`) é classicamente correta e confere exatidão sub-milimétrica (se bem calibrada). A validação de concavidade (`A < 0`) e a definição da confiança baseada na curvatura (`abs(A) / fnoc`) refletem um bom entendimento do modelo físico de desfoque.

### 1.3. Estrutura de Software e Modularidade
- O software possui boa separação de responsabilidades (`main.py`, `mosaic.py`, `argmax_fuzzy.py`, `indicators/`).
- O suporte a arquivos de configuração (`yaml`) através do `read_yaml_parameters` permite parametrizar experimentos sem alterar o código-fonte, uma ótima prática experimental.

---

## 2. Falhas Críticas e Erros (O que está errado)

### 2.1. Destruição do Pico de Foco (Matemática/Sinal)
**Arquivo:** `indicators/applicator.py`
**Problema:** Na normalização do indicador de foco, o código utiliza `np.clip(focus_indicator_stack, p1, p90)`.
**Consequência:** Cortar os valores no 90º percentil é **catastrófico** para algoritmos de foco. O método de "Depth from Focus" depende matematicamente de encontrar o pico absoluto da curva passa-alta (foco). Ao fazer o clip em 90%, os 10% de maiores valores de foco tornam-se um platô plano. Isso destrói a parábola exata que `argmax_fuzzy.py` tenta ajustar, causando erros grosseiros na estimativa de profundidade sempre que o pixel estiver muito bem focado, pois a derivada no pico torna-se zero.

### 2.2. Perda de Escala Física na Exportação (Física)
**Arquivo:** `main.py`
**Problema:** O mapa de profundidade físico é normalizado antes de ser salvo: `convert_image_array_to_fni(normalize(zMos), ...)`
**Consequência:** `zMos` contém o valor das posições focais (`zFoc`). Ao aplicar a função `normalize()` (que converte tudo linearmente para `[0, 1]`), todas as unidades físicas (milímetros, micrômetros) relacionadas ao eixo Z do microscópio ou lente são jogadas fora. Num sistema híbrido físico, a escala real é a única forma de unir os dados de multifoco (físico/métrico) com as normais do estéreo fotométrico. Essa normalização "achata" e invalida a escala para fusão.

### 2.3. Código "Fantasma" e Quebrado
- **Arquivos não utilizados/quebrados:** O arquivo `weighted_filter.py` possui um comentário claro `## não está funcioando corretamente` e não é invocado no pipeline. O arquivo `depth_refinement.py` possui caminhos *hardcoded* (`base_path = "/home/lelis/..."`), sendo praticamente inexecutável por outros computadores sem modificação, além de não estar conectado ao fluxo principal `main.py`.
- **Degradação na Transformação (Alinhamento):** O `image_alignment.py` aplica alinhamento de forma sequencial transitiva (i com i+1). A aplicação sucessiva de homografias degrada a imagem iterativamente por conta de re-interpolações, o que pode mascarar as altas frequências cruciais para a deteção do foco.

---

## 3. Sugestões de Melhoria (O que pode melhorar)

### 3.1. Correção Imediata da Extração e Foco e Profundidade (Matemática)
- **Remover Clipping:** Eliminar o `np.clip` no percentil 90 em `applicator.py`. Se houver ruído impulsivo (outliers de foco), aplique um filtro de Mediana Espacial (`cv2.medianBlur`) 2D por plano focal antes ou depois da medida de foco, mas nunca corte os picos de intensidade da curva Z.
- **Exportação Bruta (Física):** Em `main.py`, salve o `zMos` não-normalizado. Certifique-se de que a IO de `.fni` (ou altere para um formato bruto como `.npy` ou `TIFF` de 32/64 bits) consiga persistir os floats nativos de `zMos`. Só normalize matrizes para a geração de imagens de visualização `.png`.

### 3.2. Aprimoramento do Alinhamento de Imagens (*Focus Breathing*)
Para resolver a "respiração" da lente (pequenas mudanças de magnificação ao alterar o foco), **não alinhe as imagens sequencialmente**. Escolha uma imagem de referência global (preferencialmente a imagem correspondente ao foco médio da pilha) e alinhe todas as outras imagens do *stack* unicamente contra essa referência. Isso resulta em apenas 1 operação de *warp* por imagem, minimizando imensamente a perda de dados de alta-frequência.

### 3.3. Limpeza de Repositório (Engenharia)
- **Isolamento da Matemática:** Remover o código comentado (`plot_3d`, `exibir_imagem`) do `utils.py`. Mova operações matemáticas complexas, como as funções polinomiais `quadratic_interpolation`, para um arquivo dedicado (por exemplo, `math_utils.py`), deixando o `utils.py` apenas para file I/O ou utilitários gerais.
- **Integração Real do Refinamento:** Utilizar o mapa de incerteza (1 - `wSel`) do `argmax_fuzzy` ativamente como peso (*unary cost*) nas penalidades de Graph-Cut do módulo `depth_refinement`. E acoplar esse algoritmo diretamente como um passo opcional acionável via YAML no `main.py`, substituindo o arquétipo de script isolado.
