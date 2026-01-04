# Relatório Técnico: Método Híbrido de Reconstrução 3D (Estéreo Multifocal + Fotométrico)

Este relatório apresenta uma análise detalhada dos métodos computacionais e físicos aplicados no repositório `hybrid-stereo-method`. O sistema combina **Shape-from-Focus (SFF)** (ou Estéreo Multifocal) e **Shape-from-Shading/Photometric Stereo (PS)** para reconstrução tridimensional de alta precisão.

## 1. Visão Geral do Pipeline
O projeto implementa uma abordagem híbrida onde:
1.  **Estéreo Multifocal (MFS)**: Recupera uma estimativa inicial, grosseira mas absoluta, da profundidade ($Z$) analisando a nitidez em uma pilha de imagens com diferentes planos focais.
2.  **Estéreo Fotométrico (PS)**: Recupera as normais da superfície ($\mathbf{N}$) com alta resolução de detalhes, utilizando variação de iluminação.
3.  **Integração (Fusão)**: Combina a profundidade absoluta do MFS com os detalhes de alta frequência do PS para gerar uma superfície final otimizada.

---

## 2. Estéreo Multifocal (Shape-from-Focus)
Localização no código: `src/multifocus_stereo/`

### 2.1. Fundamentação Física
O método baseia-se na óptica geométrica de lentes finas/delgadas. A relação entre a distância do objeto ($d_o$), distância focal da lente ($f$) e distância da imagem ($d_i$) é dada pela equação de Gauss: $\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}$.
Pontos fora do plano de foco aparecem borrados (Círculo de Confusão). O algoritmo busca, para cada pixel $(x,y)$, qual imagem da pilha ($k$) possui a maior "energia de alta frequência" (foco), inferindo a distância $d_o(k)$.

### 2.2. Implementação Computacional e Métodos

#### A. Medidores de Foco (Focus Measure Operators)
Arquivos: `focus_indicator_*.py`
Para quantificar o "grau de foco", o código aplica filtros que exaltam altas frequências (bordas/texturas):
*   **Laplaciano (`focus_indicator_laplacian.py`)**:
    *   *Técnica*: Aplica o operador Laplaciano ($\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$), implementado via convolução com kernels discretos (como o kernel $3\times3$ ou maiores).
    *   *Interpretação*: Regiões de alta segunda derivada correspondem a transições rápidas de intensidade (foco).
*   **Wavelet (`focus_indicator_wavelet.py`)**:
    *   *Técnica*: Utiliza a Transformada Wavelet Discreta (DWT, `pywt.wavedec2`).
    *   *Detalhe*: Calcula a energia dos coeficientes de detalhe horizontal ($cH$), vertical ($cV$) e diagonal ($cD$). A magnitude $\sqrt{cH^2 + cV^2 + cD^2}$ serve como métrica de foco.

#### B. Estimativa de Profundidade Sub-pixel (Argmax Fuzzy)
Arquivo: `argmax_fuzzy.py`
Em vez de simplesmente escolher o índice discreto $k$ da imagem com maior foco (o que daria uma profundidade "em degraus"), o código implementa uma interpolação:
1.  **Seleção de Janela**: Identifica o índice $k_{max}$ de maior foco. Seleciona vizinhos $[k_{max}-r, k_{max}+r]$.
2.  **Regressão Polinomial (Fitting)**: Ajusta uma parábola ($y = Ax^2 + Bx + C$) aos valores de foco locais.
3.  **Maximização Analítica**: O pico real de foco é calculado analiticamente onde a derivada é zero: $k_{fuzzy} = -B / (2A)$.
4.  **Cálculo de Incerteza**: A "acuidade" do pico (curvatura $A$) e o valor máximo ($f_{noc}$) são usados para estimar a confiança (`wSel/conf`). Um pico estreito e alto indica alta confiança; um pico achatado indica incerteza (região sem textura).

#### C. Mosaico (All-in-Focus)
Arquivo: `mosaic.py`
Reconstrói uma imagem totalmente focada (`sMos`) selecionando ou interpolando os pixels das imagens originais baseando-se no mapa de profundidade estimado.

---

## 3. Estéreo Fotométrico (Photometric Stereo)
Localização no código: `src/photometric_stereo/`

### 3.1. Fundamentação Física
Assume-se predominantemente o **Modelo de Reflexão Lambertiano**: A intensidade observada $I$ é proporcional ao cosseno do ângulo entre a luz incidente ($\mathbf{L}$) e a normal da superfície ($\mathbf{N}$):
$$I = \rho (\mathbf{L} \cdot \mathbf{N})$$
onde $\rho$ é o albedo (refletividade).
Com $M \ge 3$ imagens sob iluminações conhecidas $\mathbf{L}$, podemos resolver para $\mathbf{N}$.

### 3.2. Implementação Computacional e Solvers

#### A. Mínimos Quadrados (L2 - Woodham)
Arquivo: `rps.py`, método `_solve_l2`
*   *Técnica*: Resolve o sistema linear superdeterminado $\mathbf{I} = \mathbf{L} \mathbf{g}$ (onde $\mathbf{g} = \rho \mathbf{N}$) usando Mínimos Quadrados: $\mathbf{g} = (\mathbf{L}^T \mathbf{L})^{-1} \mathbf{L}^T \mathbf{I}$.
*   *Vantagem*: Rápido.
*   *Desvantagem*: Sensível a outliers (sombras, brilhos especulares).

#### B. Robust Photometric Stereo (L1 / Sparse Regression)
Arquivo: `rps.py`, método `_solve_l1`
*   *Técnica*: Modela o erro como esparso. Em vez de minimizar a soma dos quadrados dos erros (Gaussiano), minimiza a norma L1 dos resíduos: $\min \| \mathbf{I} - \mathbf{L}\mathbf{g} \|_1$.
*   *Ciência da Computação*: Isso é resolvido como um problema de otimização convexa ou iterativamente (IRLS - Iteratively Reweighted Least Squares). O código parece usar uma implementação numérica específica (`rpsnumerics`) para essa minimização.
*   *Aplicação*: Permite ignorar pixels que não seguem o modelo Lambertiano (e.g., sombras duras).

#### C. Sparse Bayesian Learning (SBL)
Arquivo: `rps.py`, método `_solve_sbl`
*   *Técnica*: Abordagem probabilística onde o erro é modelado com prioris de esparsidade. O algoritmo aprende quais observações são "inliers" e quais são "outliers" automaticamente durante a inferência.

#### D. Robust PCA (RPCA)
Arquivo: `rps.py`, método `_solve_rpca`
*   *Técnica*: Decompõe a matriz de observações $\mathbf{M}$ em duas matrizes: $\mathbf{A}$ (Low-rank, representando a estrutura Lambertiana ideal) e $\mathbf{E}$ (Sparse, representando erros/sombras).
    $$ \mathbf{M} = \mathbf{A} + \mathbf{E} $$
*   *Algoritmo*: Inexact ALM (Augmented Lagrange Multiplier). Resolve o problema de otimização $\min \|\mathbf{A}\|_* + \lambda \|\mathbf{E}\|_1$, onde $\|\cdot\|_*$ é a norma nuclear.

---

## 4. Integração de Superfície (Fusão)
Localização no código: `src/hybrid_method/integrate_recursive/` (Código C)

### 4.1. O Problema da Integração
Temos um campo de normais $\mathbf{N} = (n_x, n_y, n_z)$ obtido pelo PS. Os gradientes da superfície ($p = \partial Z / \partial x$, $q = \partial Z / \partial y$) são derivados como $p = -n_x/n_z$ e $q = -n_y/n_z$.
Queremos encontrar a superfície $Z(x,y)$ cujos gradientes melhor se ajustam a $p$ e $q$.
Isso leva à **Equação de Poisson**:
$$ \nabla^2 Z = \frac{\partial p}{\partial x} + \frac{\partial q}{\partial y} $$

### 4.2. Método Variacional Híbrido
O código `gus_integrate_recursive.c` implementa uma abordagem variacional robusta. Ele minimiza um funcional de energia ($E$) que combina dois termos:
1.  **Termo de Gradiente (PS)**: Obriga $Z$ a ter os gradientes fornecidos pelo PS.
2.  **Termo de Fidelidade (Hints/MFS)**: Obriga $Z$ a estar próximo da profundidade absoluta grosseira calculada pelo MFS.

$$ E(Z) = \iint \left( \|\nabla Z - \mathbf{g}_{PS}\|^2 + \lambda \|Z - Z_{MFS}\|^2 \right) dx dy $$

### 4.3. Algoritmo Multigrid / Recursivo
A solução direta de sistemas lineares grandes para imagens de alta resolução é computacionalmente proibitiva ($O(N^3)$ ou $O(N^2)$).
O código utiliza uma abordagem **Multigrid (Multiresolução/Recursiva)**:
1.  **Downsampling**: Reduz os mapas de entrada (gradientes e hints) sucessivamente até um tamanho muito pequeno (nível grosseiro).
2.  **Solve**: Resolve o sistema linear simples no nível mais grosseiro (com poucas variáveis).
3.  **Upsampling & Refinement**: A solução grosseira é interpolada para o nível acima e usada como *guess* inicial. O algoritmo então relaxa (suaviza) o erro usando iterações (provavelmente Gauss-Seidel ou SOR) no nível mais fino.
4.  Isso se repete até a resolução original.
*Vantagem*: Convergência muito mais rápida ($O(N)$) para componentes de baixa frequência da superfície.

A implementação em C utiliza estruturas de dados eficientes para imagens (`float_image_t`) e resolve o sistema esparso gerado pela discretização da equação de Poisson.

---

## 5. Resumo da Arquitetura
O sistema é um exemplo clássico de arquitetura de **Visão Computacional Baseada em Física**:
1.  **Entrada**: Imagens Raw.
2.  **Pré-processamento**: Alinhamento, conversão Grayscale.
3.  **Análise Local (MFS)**: Processamento de sinal 1D no tempo (stack de foco) para estimativa inicial. Uso de Lógica Fuzzy para robustez.
4.  **Inversão Global (PS)**: Álgebra Linear para inversão do modelo de luz. Uso de estimadores estatísticos robustos (L1, SBL) para lidar com ruído real.
5.  **Otimização Global (Integração)**: Cálculo Variacional numérico para fusão de dados heterogêneos (gradientes locais + profundidade global).
