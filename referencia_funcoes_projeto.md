# Referência de API do Projeto

Este documento mapeia as principais funções reutilizáveis do projeto `hybrid-stereo-method`, organizadas por módulo.

## 1. Módulo Comum (`src/common`)

### 1.1 Entrada/Saída (`src/common/io.py`)
Funções para manipulação de arquivos, leitura de imagens e logs.

| Função | Descrição |
| :--- | :--- |
| `read_yaml_parameters(yaml_file_path)` | Lê parâmetros de configuração de um arquivo YAML. Retorna um dicionário. |
| `log_parameters(params, prefix='')` | Loga recursivamente um dicionário de parâmetros de forma hierárquica. |
| `find_all_files(path)` | Busca recursivamente todos os arquivos em um diretório (ordenados via natsort). |
| `read_image(image_path, info=False)` | Lê uma imagem usando OpenCV. Se `info=True`, imprime estatísticas. |
| `read_images(image_paths, info=False)` | Lê uma lista de imagens. Retorna uma lista de arrays numpy. |
| `save_image(save_path, save_as, img)` | Normaliza (0-255) e salva uma imagem no disco. Cria diretórios se necessário. |
| `convert_image_array_to_fni(image_array, output_file)` | Salva uma imagem no formato FNI (Float Image, formato específico do projeto). |
| `read_fni_to_image_array(fni_file)` | Lê um arquivo FNI e o converte para array numpy. |

### 1.2 Utilitários Gerais (`src/common/utils.py`)
Funções auxiliares para processamento de imagens e estatísticas.

| Função | Descrição |
| :--- | :--- |
| `print_img_statistics(nome, img)` | Imprime min, max, média, RMS e desvio padrão de uma imagem. |
| `normalize_normals(normal_map)` | Normaliza vetores de um mapa de normais para terem comprimento unitário. |
| `convert_to_grayscale(img)` | Converte imagem BGR ou Float para escala de cinza. |
| `calculate_avarage_of_images(imagens)` | Calcula a imagem média pixel a pixel de uma lista de imagens. |

---

## 2. Estéreo Fotométrico (`src/photometric_stereo`)

### 2.1 Utilitários PS (`src/photometric_stereo/psutil.py`)
Ferramentas específicas para manipulação de luzes e mapas normais.

| Função | Descrição |
| :--- | :--- |
| `load_lighttxt(filename)` | Carrega direções de luz de um arquivo `.txt`. |
| `load_lightnpy(filename)` | Carrega direções de luz de um arquivo `.npy`. |
| `load_images(foldername, ext, scale)` | Carrega todas as imagens de uma pasta para uma matriz de medição $M$ (pixels x imagens). |
| `save_normalmap_as_npy(filename, normal, ...)` | Salva o mapa de normais calculado como `.npy`. |
| `evaluate_angular_error(gtnormal, normal)` | Calcula o erro angular (em graus) entre normais estimadas e ground-truth. |
| `converter_npy_para_cinza(matriz)` | Converte imagem RGB/BGR carregada via npy para cinza (pesos: 0.3, 0.59, 0.11). |
| `light_direction(A, B, C, ...)` | Calcula a direção da luz a partir de reflexos em uma esfera (light probe). |
| `find_white_sphere(image)` | Detecta uma esfera de calibração na imagem usando Transformada de Hough. |
| `calculate_gradient_consistency(gradient_map)` | Calcula a integrabilidade (consistência rotacional) do campo de gradientes. |

### 2.2 Algoritmos Numéricos (`src/photometric_stereo/rpsnumerics.py`)
Solvers numéricos para otimização robusta.

| Função | Descrição |
| :--- | :--- |
| `L1_residual_min(A, b)` | Resolve $\min \|Ax - b\|_1$ usando IRLS (Mínimos Quadrados Reweighted). Usado para PS Robusto. |
| `sparse_bayesian_learning(A, b)` | Resolve $\min \|Ax - b\|_0$ usando SBL. Abordagem probabilística para outliers esparsos. |
| `rpca_inexact_alm(D)` | Decompõe matriz $D$ em $A$ (Low-Rank) + $E$ (Sparse) usando Robust PCA via ALM Inexato. |

### 2.3 Classe Principal (`src/photometric_stereo/rps.py`)
*   **Classe `RPS`**: Encapsula o pipeline de Photometric Stereo.
    *   `solve(method)`: Executa a reconstrução. Métodos suportados: `L2_SOLVER` (Woodham), `L1_SOLVER`, `SBL_SOLVER`, `RPCA_SOLVER`.

---

## 3. Estéreo Multifocal (`src/multifocus_stereo`)

### 3.1 Indicadores de Foco
Funções que calculam mapas de nitidez $F(x,y)$ a partir de uma imagem $I$.

*   **Laplaciano** (`focus_indicator_laplacian.py`): Usa a segunda derivada como medida de alta frequência.
*   **Wavelet** (`focus_indicator_wavelet.py`): Usa a energia dos coeficientes de detalhe da Transformada Wavelet.

### 3.2 Lógica Fuzzy e Mosaico
*   **`compute_argmax_fuzzy(focus_stack)`** (`argmax_fuzzy.py`):
    *   Recebe uma pilha de mapas de foco.
    *   Interpola a posição do pico de foco ($k_{fuzzy}$) com precisão sub-pixel.
    *   Retorna o mapa de profundidade indexada (`iSel`) e o mapa de confiança (`wSel`).

*   **`mosaic(iSel, image_stack, zFoc)`** (`mosaic.py`):
    *   Constrói a imagem *All-in-Focus* combinando pixels das imagens originais baseando-se no mapa de índices `iSel`.



