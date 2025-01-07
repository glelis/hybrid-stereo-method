import os
import cv2
import numpy as np
from natsort import natsorted

import matplotlib.pyplot as plt
# General
WEIGHTS = np.array(
        [[0, 0, 1, 2, 1, 0, 0],
        [0, 1, 2, 3, 2, 1, 0],
        [1, 2, 3, 4, 3, 2, 1],
        [2, 3, 4, 5, 4, 3, 2],
        [1, 2, 3, 4, 3, 2, 1],
        [0, 1, 2, 3, 2, 1, 0],
        [0, 0, 1, 2, 1, 0, 0]])


def convert_img_to_grayscale(img):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imGray

def convert_stack_to_grayscale(image_stack: np.array) -> np.array:
    """
    Converte uma stack de imagens coloridas para escala de cinza.

    Args:
        image_stack: Um array [kf, kx, ky, 3] de imagens coloridas.

    Returns:
        Um array [kf, kx, ky] de imagens em escala de cinza.
    """
    # Utiliza a função cvtColor do OpenCV para converter todas as imagens de uma vez
    grayscale_stack = np.array([convert_img_to_grayscale(img) for img in image_stack])
    return grayscale_stack



def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img


def print_img_stats(name, img):
    print("\n %s Minimo e Maximo\n" %name, np.min(img), np.max(img))


def print_img_statistics(nome, img):
    shape = img.shape
    v_max = np.max(img)
    v_min = np.min(img)
    v_mean = np.average(img)
    rms = np.sqrt(np.average(img**2))
    v_dev = np.sqrt(np.average((img-v_mean)**2))
    print(f'nome:{nome}, shape:{shape}, min:{v_min:.6f}, max:{v_max:.6f}, mean:{v_mean:.6f}, rms:{rms:.6f}, v_dev:{v_dev:.6f}')



def save_image(save_path, save_as, img, v_min, v_max):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # print_img_stats('Não normalizada: '+ save_as, img)
    img_norm = ((img - v_min)/(v_max - v_min))*255
    # print_img_stats('Normalizada: '+ save_as, img_norm)
    cv2.imwrite(os.path.join(save_path, save_as), img_norm)
    
def zero_borders(img, border_size):
    """
    Zera as bordas de uma imagem.
    :param img: numpy.ndarray, imagem original
    :param border_size: int, tamanho da borda a ser zerada
    :return: numpy.ndarray, imagem com as bordas zeradas
    """
    img_copy = img.copy()
    img_copy[:border_size, :] = 0
    img_copy[-border_size:, :] = 0
    img_copy[:, :border_size] = 0
    img_copy[:, -border_size:] = 0
    return img_copy


def find_all_files(path):
    all_files = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(file)
    
    return all_files

def read_images_from_path(img_path):
    img_list = []
    
    for root, dirs, files in os.walk(img_path):
        for file in natsorted(files):
        #for file in sorted(files):
            img_list.append(read_image(root + "/" + file))
            print(root + "/" + file)
    #print(img_list)
    return img_list


def normalize(x: np.array) -> np.array:
    """
    Normalizes the input array `x` to a range between 0 and 1.

    Parameters:
    x (numpy.ndarray): The input array to be normalized.

    Returns:
    numpy.ndarray: The normalized array with values scaled to the range [0, 1].

    Example:
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> normalize(x)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    max_, min_ = np.max(x), np.min(x)
    normalized = (x - min_) / (max_ - min_)
    return normalized

def normalize_max(x: np.array) -> np.array:
    """
    Normalizes the input array `x`from range [0, max] to range [0, 1]

    Parameters:
    x (numpy.ndarray): The input array to be normalized.

    Returns:
    numpy.ndarray: The normalized array with values scaled to the range [0, 1].
    """
    max_ = np.max(x)
    normalized = x / max_
    return normalized





def exibir_imagem(imagem_np_array):
  """
  Exibe um np.array que representa uma imagem.

  Args:
      imagem_np_array: O np.array que contém os dados da imagem.
  """

  if len(imagem_np_array.shape) == 2:  # Verifica se a imagem é em escala de cinza
    plt.imshow(imagem_np_array, cmap='gray')
  else:
    imagem_np_array = cv2.cvtColor(imagem_np_array, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_np_array, cmap='viridis')

  #plt.axis('off')  # Opcional: remove os eixos da imagem
  plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(depth_map, z_scale=1):
    
    #height_map = (255 - (depth_map))/255
    height_map = depth_map

    # Crie uma grade de coordenadas para os eixos X e Y
    x = np.linspace(0, height_map.shape[1] - 1, height_map.shape[1])
    y = np.linspace(0, height_map.shape[0] - 1, height_map.shape[0])
    x, y = np.meshgrid(x, y)

    # Criar a figura 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plotar a superfície 3D
    ax.plot_surface(x, y, height_map, cmap='viridis')

    ax.set_zlim(np.min(height_map), np.max(height_map) * z_scale)

    # Adicionar rótulos aos eixos
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altura (mm)')

    # Exibir a imagem 3D
    plt.show()





from stl import mesh

def create_stl_from_heightmap(height_map, scale=(1, 1, 1), output_file="output.stl"):
    """
    Cria um arquivo STL baseado em um mapa de altura.
    
    :param height_map: Uma matriz numpy representando o mapa de altura
    :param scale: Um tuplo de 3 valores representando a escala em x, y e z
    :param output_file: O nome do arquivo STL de saída
    """
    rows, cols = height_map.shape
    vertices = []

    # Escalas para as dimensões do modelo
    scale_x, scale_y, scale_z = scale

    # Gerar vértices a partir do mapa de altura
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Vértices do quadrado atual (em 3D)
            v1 = [i * scale_x, j * scale_y, height_map[i, j] * scale_z]
            v2 = [(i + 1) * scale_x, j * scale_y, height_map[i + 1, j] * scale_z]
            v3 = [i * scale_x, (j + 1) * scale_y, height_map[i, j + 1] * scale_z]
            v4 = [(i + 1) * scale_x, (j + 1) * scale_y, height_map[i + 1, j + 1] * scale_z]

            # Criar dois triângulos para cada quadrado
            vertices.append([v1, v2, v3])  # Triângulo 1
            vertices.append([v2, v4, v3])  # Triângulo 2

    # Criar a malha com os vértices
    vertices = np.array(vertices)
    stl_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(vertices):
        stl_mesh.vectors[i] = f

    # Salvar o arquivo STL
    stl_mesh.save(output_file)
    print(f"STL gerado e salvo em {output_file}")


import numpy as np
from PIL import Image

def negativo_imagem(imagem):
    # Converte a imagem de um array NumPy para uma imagem PIL
    imagem_pil = Image.fromarray(imagem.astype('uint8'))

    # Faz o negativo da imagem
    negativo = Image.eval(imagem_pil, lambda x: 255 - x)

    # Converte a imagem negativa de volta para um array NumPy
    negativo_np = np.array(negativo)

    return negativo_np


import numpy as np
from PIL import Image
from rembg import remove

def aplicar_mascara(imagem, img_referencia):
    """
    Aplica uma máscara a uma imagem.

    :param imagem: numpy.ndarray, imagem original
    :param referencia: numpy.ndarray, da imagem referencia que ira fornecer a mascara (mesma forma que a imagem)
    :return: numpy.ndarray, imagem resultante após aplicação da máscara
    """
    #retira a mascara
    mascara = remove(img_referencia, only_mask=True)
    mascara = np.where(mascara <10, 0, 1)

    # Verifica se a máscara e a imagem têm a mesma forma
    if imagem.shape != mascara.shape:
        raise ValueError("A máscara deve ter a mesma forma que a imagem.")

    # Aplica a máscara: mantém os pixels onde a máscara é diferente de zero
    imagem_resultante = np.where(mascara != 0, imagem, 0)  # Substitua 0 por outra cor se necessário

    return imagem_resultante


def calculate_error_image(reference_image, depth_map):
    """
    Calculate the error image by comparing the depth map with a reference image.
    
    Args:
        reference_image: The reference image.
        depth_map: The calculated depth map.

    Returns:
        The error image.
    """
    reference_image_normalized = normalize(convert_img_to_grayscale(reference_image))
    depth_map_normalized = normalize(depth_map)
    error_image = reference_image_normalized - depth_map_normalized
    return error_image






import os
from collections import defaultdict
import shutil

def reorganize_repository(base_path, output_path):
    # Create a dictionary to hold the grouped files
    grouped_files = defaultdict(list)
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        # Sort directories alphabetically
        dirs.sort()
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for file in sorted(os.listdir(dir_path)):
                # Extract the base name without extension
                base_name = os.path.splitext(file)[0]
                # Create a new file name with the order number
                new_file_name = f"{len(grouped_files[base_name]) + 1}_{file}"
                # Create the directory for the base name if it doesn't exist
                output_dir_path = os.path.join(output_path, base_name)
                os.makedirs(output_dir_path, exist_ok=True)
                # Create the full path for the new file
                new_file_path = os.path.join(output_dir_path, new_file_name)
                # Move the file to the new location with the new name
                shutil.copy(os.path.join(dir_path, file), new_file_path)
                # Add the new file name to the grouped files dictionary
                grouped_files[base_name].append(new_file_path)
    
    return grouped_files




def compute_fuzzynes(img_fuzzy):
    height, width = img_fuzzy.shape
    s = 0
    for i in range(height):
        for j in range (width):
            p = img_fuzzy[i, j]
            d = p - floor(p+0.5)
            s = s + abs(d)

    return s/(height*width)



from math import floor, pi, cos

def quadratic_interpolation(val, k_fuzzy):
    nframes = len(val)
    
    # Passo 1: Calcular o índice inteiro mais próximo de k_fuzzy
    kint = int(floor(k_fuzzy + 0.5))
    
    # Passo 2: Definir a janela de interpolação
    if kint <= 1:
        k0 = 0
        k1 = 2
    elif kint >= nframes - 2:
        k0 = nframes - 3
        k1 = nframes - 1
    else:
        k0 = kint - 2
        k1 = kint + 2
    
    m = k1 - k0 + 1
    
    # Passo 3: Calcular a diferença fracionária s
    s = k_fuzzy - kint
    assert -0.5 <= s <= 0.5
    
    # Passo 4: Definir os vetores x e y
    x = [k0 + j for j in range(m)]
    y = [val[k0 + j] for j in range(m)]
    
    # Passo 5: Calcular os pesos w
    a = pi * 0.5 * (m + 1)
    w = [0.5 * (1 + cos(a * (k0 + j - k_fuzzy))) for j in range(m)]
    
    # Passo 6: Regressão quadrática ponderada usando np.polyfit
    # A, B, C = tuple(np.polyfit(x, y, 2, w=w))
    
    # Passo 6: Regressão quadrática ponderada
    X = np.vstack([np.ones(m), x, np.square(x)]).T
    W = np.diag(w)
    
    # Verificar se a matriz é singular
    try:
        A = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    except np.linalg.LinAlgError:
        # Adicionar regularização para evitar singularidade
        regularization = 1e-8
        A = np.linalg.inv(X.T @ W @ X + regularization * np.eye(X.shape[1])) @ (X.T @ W @ y)
    
    # Coeficientes da parábola
    C, B, A = A
    
    # Passo 7: Calcular o valor interpolado vsel
    vsel = A * (k_fuzzy ** 2) + B * k_fuzzy + C
    
    # Passo 8: Retornar o valor interpolado
    return vsel



def linear_interpolation(val, k_fuzzy):
    nframes = len(val)
    
    # Passo 1: Calcular o índice inteiro mais próximo de k_fuzzy
    kint = int(floor(k_fuzzy))
    
    # Passo 2: Definir os índices de interpolação
    if kint < 0:
        k0 = 0
        k1 = 1
    elif kint >= nframes - 1:
        k0 = nframes - 2
        k1 = nframes - 1
    else:
        k0 = kint
        k1 = kint + 1
    
    # Passo 3: Calcular a diferença fracionária s
    s = k_fuzzy - k0
    #print(f"kint: {kint}, k0: {k0}, k1: {k1}, s: {s}, nframes: {nframes}, k_fuzzy: {k_fuzzy}")
    assert 0 <= s <= 1
    
    # Passo 4: Calcular o valor interpolado
    vsel = (1 - s) * val[k0] + s * val[k1]
    
    # Passo 5: Retornar o valor interpolado
    return vsel




