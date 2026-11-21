# Relatório de Análise Técnica: Implementação Photometric Stereo (`src/hybrid_stereo_method/photometric`)

## 1. Visão Geral
O módulo `photometric` compreende as implementações de **Estéreo Fotométrico (Photometric Stereo - PS)** para a recuperação de normais de superfície e topografia fina 3D com base na variação de sombreamento sob diferentes ângulos de iluminação. A biblioteca suporta a modelagem clássica linear e avançadas modelagens robustas para suprimir ruídos, sombras projetadas e brilhos especulares, sendo desenhada tanto de forma autônoma quanto atuando na fase híbrida com informações do processo Multifoco.

## 2. Ponto de Entrada, Fluxos e Orquestração (`main.py` e `main_wps.py`)

Diferente do módulo focado local, este módulo fotométrico hospeda dois fluxos centrais em sua raiz baseados no método operacional:

- **`main.py` (Robust PS Classico):**
  Orquestra a reconstrução topográfica autônoma por classe `RPS`.
  1. **Inicialização:** Lê configurações do arquivo YAML, monta diretórios locais/outputs.
  2. **Carregamento:** Inicia o objeto `RPS()`, abastecendo vetores diretores de luzes globais a partir de `lights.npy`, além de máscaras morfológicas (`mask.png`) que definem o fundo inativo e arrays 3D de imagens capturadas escaláveis (`rps.load_images` / `rps.load_npyimages`).
  3. **Invocação Linear Dinâmica do Solver:** Dependendo no *string* do método requisitado no parâmetro ("L2", "L1", "SBL", "RPCA"), o código direciona o objeto de resolução sob paralelizacionamento MultiCore para o solucionador ideal sob temporizador logado.
  4. **Output Normatizado:** Converte campos resolutos e os salva fisicamente usando formato numpy (`normal_map.npy`), gera prints de log de Erro Angular (`evaluate_angular_error`) quando gabaritos de teste *ground truth* estao fornecidos.

- **`main_wps.py` (Weighted Photometric Stereo & Híbrido):**
  Focado para integrar-se ao método híbrido alimentando as posições ótimas de sub-foco.
  1. Ao invés de uma imagem focada global, pode receber uma lista de caminhos do conjunto "all-in-focus" (`sMos_path_list`) gerada no módulo Multifocus.
  2. Aciona implementações exclusivas parametrizadas de estimadores Argmax usando Mínimos Quadrados Clássicos Robusto a Ponderação (Weighted PS - `estimate_normals_argmax_lstsq_robust`).
  3. Gera informações complementares únicas, exportando um canal 4D: Vetor Normal em concorrência ao Resíduo LSQ Invertido, interpretado estatisticamente como **"Mapa de Confiança" (.fni)**.

## 3. Implementações Matemáticas do Core (`rps.py` e `wps.py`)

A lei de espalhamento Lambertiano dita que a Intensidade (`I`) seja o produto interno do albedo (`rho`), vetor luz (`L`) e Normal (`N`). $I = \rho L^T N$.

### Robust Photometric Stereo Clássico (`rps.py`):
Módulo envolto por Orientação a Objetos (Classe RPS) que encapsula `M` (medição), `L` (Luzes) e `N` (Matriz final). Contém os Solvers Analíticos:
  - **L2 Solver (Woodham 1980):** Execução base ingênua usando simples pseudo-inversa matricial direta resolvendo via função `numpy.linalg.lstsq` perante iluminações conhecidas.
  - **L1 Solver (IRLS):** Otimiza a divergência pela penalização $L1$ da norma de resíduo iterando $x_{n} = \min_{x} ||L \cdot x_{n-1} - M||_0$. É altamente tolerante a "Sombras Projetadas Críticas".
  - **Sparse Bayesian Learning (SBL):** Formula o problema como probabilístico em inferências de Bayes.
  - **Robust PCA (RPCA):** Utiliza matriz de Rank Baixo e um Ruído Esparso para separar perfeitamente erros especulares do sinal original não adulterado. As computações matemáticas massivas estão isoladas em `solvers/numerics.py`.

### Weighted Photometric Stereo (`wps.py`):
Opera na seleção contínua de blocos da luz baseado nos perfis da captura:
  - **Argmax Top-K (`estimate_normals_argmax`):** Ignora a nuvem completa de luz e atua isolando os K frames sob a claridade mais intensa por pixel, fugindo instantaneamente das sombras pela simples inversão invertível da matriz k-3x3.
  - **LSTSQ Argmax (`estimate_normals_argmax_lstsq`):** Realiza mapeamento integral via $L_2$ global e produz uma Confiança com base inversa das falhas no residual linear LSQ normalizado (1 / Resíduo_LSQ).
  - **Robusto Integrado Avançado (`_lstsq_robust`):** Filtra ativamente pixels inativados (taxa < `shadow_threshold`), entra numa repetição *While True* expurgando os distúrbios estatísticos iterativamente (valores fora da média do multíplice de tolerância). Ao encontrar platô numérico devolve as Normais Estabilizadas, o Albedo extraído e a Certeza do Cálculo Final ponderado ao número original de fotogramas disponíveis M vs N expurgados validando o ponto de desvio.

## 4. Núcleo Matemático Independente (`solvers/numerics.py`)

Agrega a matemática formal linear e cálculos diferenciais aplicadas do sistema por rotinas do artigo seminal do Prof. Yasuyuki Matsushita:
- Implementações vetorialmente polidas das iterações Mínimos Quadrados Re-Ponderados para a formulação da **Norma L1** (`L1_residual_min`).
- Métodos numéricos como *Inexact Augmented Lagrangian Multiplier* com cortes Singulares (SVD) para o cálculo matricial pesado da **RPCA** (`rpca_inexact_alm`), embutindo parâmetros flexíveis `lambda`, Operações de *Shrinkage* numérico matricial absoluto penalizador de tensores negativos/positivos.

## 5. Instrumentos de Suporte e Calibração Física (`ps_utils.py`)

Concentra funções independentes de conversão de dados multidimensionalidade em binário:
- Carregadores eficientes customizados com glob iterador entre listas dinâmicas baseadas nativamente nas strings e multiplicadores literais unificados. 
- Capacidade nativa de interpretar formato experimental (.npy 16bit floats e RAW channels auto RGB/BGR balance format) à representatividade cinza real ponderada luminosa (`converter_npy_para_cinza`).
- Rotinas físicas baseadas de calibração laboratorial como `find_white_sphere` usando Transformada de Hough Circulares (`cv2.HoughCircles`), em par com equações analítico-trigonométricas de reflexão reflexa espelhar pontual do vetor (`light_direction`) localizando globalmente do ângulo real (`XYZ`) das lanternas da bancada usando apenas imagens fotográficas primárias da esfera cromo calibração.
- Validadores analíticos integrados como o cálculo de erros Angulares matriciais por Produto ponto cruzado e coerências de Integração do Gradiente cruzado numérico (`calculate_gradient_consistency`).

## 6. Módulo de Visualização de Interface (`visualization.py`)

Lida de modo desvinculada na projeção puramente gráfica humana:
- Visualização dos três recortes ortogonais nativos normais XYZ perante o arranjo de tela matplotlib (`plotar_canais`).
- Exibição de mapas esféricos padronizados simulando a coloração normal padrão do Windows/OpenGL convertendo as frentes $[-1, +1]$ perante espaço cromático $[0, 255]$ ordenados BGR.
- Geração autoral nativa das visualizações das curvas dimensionais plotadas integralmente em malha 3D (`disp_channels_3d` sobre arquitetura `matplotlib.pyplot.projection="3d"` da biblioteca auxiliar), que é salva fotograficamente em memória (`BytesIO`) convertida perfeitamente ao array estático fotográfico.

## 7. Sumário da Execução e Artefatos Salvos
Ao rodar-se as pipelines propostas os canais de saída uniram-se em:
- **`normal_map.npy`**: Matriz Flutuante pura garantindo não existirem re-compressões de perdas em etapas futuras (Como processamento pelo Integração de Poisson C++ Híbrida iterativa para alturas Z).
- **`normal_map_with_residuals.fni`**: Imagens binárias do tensor mesclando dados métricos diretos em 4 Canais acompanhados para o cálculo uníssono das tolerância confiáveis de integração em C++.
- **Images PNG (Logs visuais)**: Várias saídas como Visualizações Unificadas Normais, canais fatiados separados com gráficos coloridos e máscaras destacadas comprovando em "Debug" que a segmentação topográfica (L2 vs WPS) realizou calibrações de laboratório aceitáveis do algoritmo sem desvios estruturais graves.
