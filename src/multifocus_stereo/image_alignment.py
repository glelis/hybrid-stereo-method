from __future__ import print_function

import cv2.xfeatures2d
from utils import *
from natsort import natsorted

import cv2
import numpy as np


def compute_descriptors(imGray):
    """
    Calcula keypoints e descritores SIFT para uma imagem em tons de cinza.

    Args:
        imGray: Uma imagem em tons de cinza representada como um array NumPy.

    Returns:
        keypoints: Uma lista de keypoints detectados.
        descriptors: Um array NumPy contendo os descritores calculados.
    """

    # Cria um objeto SIFT para detecção de keypoints e cálculo de descritores
    #sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT.create()
    # Detecta keypoints e calcula descritores usando SIFT
    keypoints, descriptors = sift.detectAndCompute(imGray, None)

    # Imprime o número de keypoints detectados e a forma do array de descritores
    print("keypoints: {}, descriptors: {}".format(len(keypoints), descriptors.shape))

    return keypoints, descriptors


def create_matcher(trees, checks):
    """
    Cria um objeto cv2.FlannBasedMatcher para correspondência de características.

    Args:
        trees: Número de árvores na estrutura de dados KD-Tree usada para busca rápida de vizinhos mais próximos.
        checks: Número de verificações realizadas durante a busca de correspondências. Aumentar esse valor melhora a precisão, mas também aumenta o tempo de processamento.

    Returns:
        matcher: Um objeto cv2.FlannBasedMatcher configurado para correspondência de características usando o algoritmo FLANN.
    """

    # Define o tipo de índice como KD-Tree
    FLANN_INDEX_KDTREE = 0

    # Parâmetros para a construção do índice KD-Tree
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)

    # Parâmetros para a busca de correspondências
    search_params = dict(checks=checks)

    # Cria o objeto matcher usando os parâmetros definidos
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    return matcher


def find_good_matches_loc(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor):
    """
    Encontra correspondências de boa qualidade entre dois conjuntos de keypoints e descritores, e retorna suas localizações.

    Args:
        matcher: Um objeto cv2.FlannBasedMatcher configurado para correspondência de características.
        keypoints1: Uma lista de keypoints da primeira imagem.
        descriptors1: Um array NumPy contendo os descritores da primeira imagem.
        keypoints2: Uma lista de keypoints da segunda imagem.
        descriptors2: Um array NumPy contendo os descritores da segunda imagem.
        factor: Um fator de limiar usado para filtrar correspondências ambíguas. Quanto menor o fator, mais rigoroso é o filtro.

    Returns:
        good_matches: Uma lista de correspondências de boa qualidade entre as duas imagens.
        points1: Um array NumPy contendo as coordenadas dos keypoints correspondentes na primeira imagem.
        points2: Um array NumPy contendo as coordenadas dos keypoints correspondentes na segunda imagem.
    """

    # Encontra os dois vizinhos mais próximos para cada descritor na primeira imagem
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Inicializa uma lista para armazenar as boas correspondências
    good_matches = []

    # Aplica o teste de razão de Lowe para filtrar correspondências ambíguas
    for m, n in matches:
        if m.distance < factor * n.distance:  # Mantém apenas correspondências onde a distância para o vizinho mais próximo é significativamente menor que a distância para o segundo vizinho mais próximo
            good_matches.append(m)

    # Extrai as coordenadas dos keypoints correspondentes nas duas imagens
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    return good_matches, points1, points2


def apply_homography(img1, img2, points1, points2):
    """
    Alinha a imagem 'img1' com a imagem 'img2' usando uma transformação de homografia calculada a partir de pontos correspondentes.

    Args:
        img1: A imagem a ser alinhada.
        img2: A imagem de referência para o alinhamento.
        points1: Um array NumPy contendo as coordenadas dos pontos na imagem 'img1'.
        points2: Um array NumPy contendo as coordenadas dos pontos correspondentes na imagem 'img2'.

    Returns:
        aligned_img: A imagem 'img1' alinhada com a imagem 'img2' usando a transformação de homografia.
    """

    # Obtém as dimensões da imagem de referência
    height, width, channels = img2.shape

    # Calcula a matriz de homografia usando RANSAC para robustez contra outliers
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Aplica a transformação de perspectiva à imagem 'img1' usando a homografia calculada
    aligned_img = cv2.warpPerspective(img1, homography, (width, height))

    return aligned_img


def align_im1_to_im2(img1, img2):
    """
    Alinha a imagem 'img1' com a imagem 'img2' usando correspondência de características e homografia.

    Args:
        img1: A imagem a ser alinhada.
        img2: A imagem de referência para o alinhamento.

    Returns:
        imMatches: Uma imagem mostrando as correspondências encontradas entre as duas imagens.
        aligned_img: A imagem 'img1' alinhada com a imagem 'img2'.
    """

    # Converte as imagens para tons de cinza
    #img1Gray = convert_img_to_grayscale(img1)
    #img2Gray = convert_img_to_grayscale(img2)

    img1Gray = img1
    img2Gray = img2

    # Calcula keypoints e descritores para ambas as imagens
    keypoints1, descriptors1 = compute_descriptors(img1Gray)
    keypoints2, descriptors2 = compute_descriptors(img2Gray)

    # Cria um objeto matcher para correspondência de características
    matcher = create_matcher(trees=5, checks=50)

    # Encontra correspondências de boa qualidade e suas localizações
    good_matches, points1, points2 = find_good_matches_loc(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor=0.80)

    # Desenha as correspondências encontradas em uma imagem
    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

    # Aplica a homografia para alinhar a imagem 'img2' com a imagem 'img1'
    aligned_img = apply_homography(img2, img1, points2, points1)

    return imMatches, aligned_img




def main_align(base_path):
    """
    Função principal que realiza o alinhamento de imagens em sequência.

    Args:
        base_path: O diretório base contendo as imagens de entrada.

    """

    #img_path = base_path + 'input/'
    img_path = base_path +  'imagens/'
    save_path = base_path + 'output/align_images/aligned/'
    match_path = base_path + 'output/align_images/match_save/'

    # Encontra todos os arquivos na pasta de imagens e os ordena naturalmente
    all_files = find_all_files(img_path)
    all_files = natsorted(all_files)
    print(all_files)
    
    #aligned_img_list = []
    #aligned_img_list.append(read_image(img_path + all_files[0]))

    aligned_img = read_image(img_path + all_files[0])

    # Itera sobre os arquivos, alinhando cada imagem com a próxima na sequência
    for i in range(len(all_files)-1):
        # Define os caminhos das imagens de origem e destino
        source_img_path = img_path + all_files[i]
        target_img_path = img_path + all_files[i+1]
        #source_img_path = img_path + all_files[0]
        #target_img_path = img_path + all_files[i]

        # Define os nomes dos arquivos de saída para as correspondências e a imagem alinhada
        match_save_as = "matches_" + str(i) + ".jpg" 
        align_save_as = "align_" + str(i) + ".jpg"

        # Lê a imagem de origem
        print("Reading a source image : ", source_img_path)
        # source_img = read_image(source_img_path)

        # Lê a imagem de destino
        print("Reading a target image : ", target_img_path);
        target_img = read_image(target_img_path)

        # Alinha as imagens
        print("Aligning images ...")
        #imMatches, aligned_img = align_im1_to_im2(source_img, target_img)
        imMatches, aligned_img = align_im1_to_im2(aligned_img, target_img)
        
        #aligned_img_list.append(aligned_img)
        
        # Salva a imagem com as correspondências
        print("Saving an feature matching image : ", save_path);
        save_image(match_path, match_save_as, imMatches,0,255)

        # Salva a imagem alinhada
        print("Saving an aligned image : ", save_path);
        save_image(save_path, align_save_as, aligned_img, 0, 255)

        # Adiciona uma linha em branco para separar a saída de cada iteração
        print("\n")