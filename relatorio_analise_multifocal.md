# Relatório de Análise Técnica: Implementação Multifocus Stereo (`src/hybrid_stereo_method/multifocus`)

## 1. Visão Geral
O módulo `multifocus` implementa a técnica de **Depth-from-Focus (DFF) / Shape-from-Focus (SFF)**. Ele é capaz de produzir imagens com foco estendido (All-in-Focus) e mapas topográficos de profundidade 3D com precisão sub-pixel, a partir de uma pilha de imagens (*Z-Stack*) focadas em diferentes distâncias focais.

## 2. Ponto de Entrada, Entrada de Dados e Fluxo Principal (`main.py`)

A coordenação principal ocorre em `main.py` na função `main(parameters)`. 

**Passo-a-passo:**
1. **Leitura e Configuração:** O script lê caminhos (input/output) baseados na configuração YAML fornecida. Logs e pastas de debug são criados.
2. **Carregamento de Imagens:** As imagens são encontradas através de `find_all_files` e convertidas num numpy array 4D, depois convertidas para imagem monocromática (`convert_to_grayscale`).
3. **Indicador de Foco:** O foco é calculado via `focus_indicator()` que percorre as imagens com algoritmos dedicados (como Laplaciano ou Fourier) estimando o quão focado o pixel está.
4. **Argmax Fuzzy (Nitidez Sub-pixel):** O stack de foco é submetido à função `compute_argmax_fuzzy`. Esse mapeamento computa onde estaria o ponto exato da imagem mais nítida, contornando a captura discreta da câmera. A "confiança" desta modelagem é exportada paralelamente.
5. **Combinação do Mosaico (Mosaic):** O algoritmo funde os pixels (`sMos`) e modela a topografia (`zMos`) convertendo dos índices ideais interpolados via `mosaic()`. 
6. **Armazenamento:** Salva todas as visões coloridas (.png) e resultados matriciais flutuantes exatos (.fni) protegidos contra compressão visual, juntamente à confiança da nuvem (`wSel`).

## 3. Avaliação da Nitidez Cênica e Indicadores (`indicators/`)

Em `applicator.py`, através da função `focus_indicator`, são englobados métodos independentes que calculam o nível de frequência-focal do pixel. Opções de máscara base, mitigação das bordas nulas, medianas e pós-suavização espacial estão disponíveis. As abordagens implementadas são:

- **Transformada de Fourier (`fourier.py`):** Utiliza FFT 2D (`np.fft.fft2`). Emprega-se uma máscara elíptica Gaussiana passa-alta (`create_gaussian_elliptical_mask`) filtrando os elementos sem arestas (baixa frequência). A magnitude remanescente forma o indicativo focal do pixel.
- **Espaço Laplaciano (`laplacian.py`):** Aplica a segunda derivada utilizando clássico `cv2.Laplacian` e calculando as variações absolutas locais de alta frequência.
- **Wavelet (`wavelet.py`):** Utiliza Transformada Discreta de Wavelet `pywt.wavedec2`. Separa as componentes horizontais, verticais e diagonais extraindo sua magnitude matemática (níveis altíssimos condizem à boa nitidez e precisão).
- **Resolução de Resíduos LSQ (`non_linear_res.py`):** Aplica estimações de blocos mínimos quadrados por Mínimos Quadrados. A falta local de aderência perfeita aos resíduos Q evidencia zonas complexas das fendas ferais de bordos finos.

## 4. Otimização de Profundidade Sub-pixel (`argmax_fuzzy.py`)

Para inferir medidas espaciais além dos cliques da máquina mecânica, a lógica de Argmax não-discreto é adotada:

- **Busca Macroscópica (`find_index_of_max_sum`):** Avalia e detecta o intervalo majoritário mais crível de máximo foco somando os envoltórios de três amostras locais contra a amostra isolada oscilante.
- **Curva Parabólica (`compute_argmax_fuzzy_1d`):** Ajusta os pixels vizinhos extraídos à volta do pico (raios de `r_max`), construindo uma equação através da biblioteca `numpy.polyfit`.
- **Foco Preciso e Confiabilidade (`wSel`):** A extração do topo da crista parabólica produz o valor decimal (sub-pixel index, via `-B/2A`). O alongamento paraboloide acusa em `conf` a convicção algorítmica - cristas pontiagudas possuem altíssima certeza, enquanto respostas alongadas ou convexas forçam o descarte. 

## 5. Mosaicamento Dimensional e Interpolação (`mosaic.py` e `math_utils.py`)

Possuindo a lista com a dimensão original (os perfis capturados reais na física, via array YAML `zFoc`), interpolamos os dados:

- **Fusão (Mos):** Constrói-se um plano bidimensional focado perfeitamente unindo recortes que espelham o pixel `sMos` em seu canal. Em paralelo a coordenada `z` espacial é fixada por `zMos`.
- Em **`math_utils.py`**, além da montagem direta (Crop), possuímos **`linear_interpolation`** (transição proporcional entre 2 frames) e a **`quadratic_interpolation`** que ajusta funções seno e co-senoides em Mínimos Quadrados Regularizados fornecendo passagem ininterruptamente lisa de cores. 

## 6. Módulos Adicionais Base Opcionais

- **Alinhamento (`image_alignment.py`):** Trata instabilidades pré-ensaio quando a plataforma vibrar (Shift). Opera SIFT (`cv2.SIFT`), computa cruzamentos KNN validados e soluciona via RANSAC matrizes homográficas (`cv2.warpPerspective`) reentortando a matriz no lugar referencial ideal global.
- **Refinamento Topográfico (`depth_refinement.py`):** Se requerido, emprega Graph Cut (PyGCO) injetando rigidez do vizinho com função não-linear penalizando gradientes (preservador e despigmentador) usando `weighted_median_filter` suavizador matricial.
- **Apoio Operacional (`utils.py`):** Lida com exportações densas: Salva STLs puros 3D a partir da geometria, retira planícies e backgrounds não mapeados implementando Inteligência Artificial Segmentadora nativa (`rembg`), e processa formatos restritos com salvamento FNI para integrá-lo transparentemente.

## 7. Sumário da Execução

Ao final da varredura, os mapeamentos são exportados:
- **`zMos.png` / `zMos.fni`**: Topografia altimétrica original calculada do objeto (Mapeamento Base Z) com subcanal de precisão inclusa em Numpy arrays.
- **`sMos.png`**: Representando visual unificada colorida "Tudo-em-Foco".
- **`iSel.fni` / `wSel.fni`**: Registros subjacentes de calibrações de indexamento polinomial e estatísticas locais de acurácias validadas em laboratório.
