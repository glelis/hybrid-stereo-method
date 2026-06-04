import os
import shutil
from collections import defaultdict

import numpy as np

from hybrid_stereo_method.infrastructure.utils import convert_to_grayscale, normalize

try:
    from stl import mesh

    HAS_STL = True
except ImportError:
    HAS_STL = False

try:
    from rembg import remove

    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# General
WEIGHTS = np.array(
    [
        [0, 0, 1, 2, 1, 0, 0],
        [0, 1, 2, 3, 2, 1, 0],
        [1, 2, 3, 4, 3, 2, 1],
        [2, 3, 4, 5, 4, 3, 2],
        [1, 2, 3, 4, 3, 2, 1],
        [0, 1, 2, 3, 2, 1, 0],
        [0, 0, 1, 2, 1, 0, 0],
    ]
)


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


def create_stl_from_heightmap(height_map, scale=(1, 1, 1), output_file="output.stl"):
    """
    Cria um arquivo STL baseado em um mapa de altura.

    :param height_map: Uma matriz numpy representando o mapa de altura
    :param scale: Um tuplo de 3 valores representando a escala em x, y e z
    :param output_file: O nome do arquivo STL de saída
    """
    if not HAS_STL:
        raise ImportError(
            "numpy-stl is required for STL export. Install via: pip install numpy-stl"
        )
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


def aplicar_mascara(imagem, img_referencia):
    """
    Aplica uma máscara a uma imagem.

    :param imagem: numpy.ndarray, imagem original
    :param referencia: numpy.ndarray, da imagem referencia que ira fornecer a mascara (mesma forma que a imagem)
    :return: numpy.ndarray, imagem resultante após aplicação da máscara
    """
    if not HAS_REMBG:
        raise ImportError("rembg is required for this function. Install via: pip install rembg")
    # retira a mascara
    mascara = remove(img_referencia, only_mask=True)
    mascara = np.where(mascara < 10, 0, 1)

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
    reference_image_normalized = normalize(convert_to_grayscale(reference_image))
    depth_map_normalized = normalize(depth_map)
    return reference_image_normalized - depth_map_normalized


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
