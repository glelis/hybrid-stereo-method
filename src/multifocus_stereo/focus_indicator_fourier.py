import cv2
import numpy as np
from multifocus_stereo.utils import *
from scipy.ndimage import convolve
from math import floor, sqrt, comb, exp, log, sin, cos, pi


def focus_indicator_fourier(aligned_img_stack:np.array):
    """
    Guive a stack overlap images, calculate the focus indicator for each image, using the square of the hight order of the fourier coeficients.

    Args:
        aligned_img_stack: A stack of overlap images, as an array[kf,kx,ky], imagages maybe color or gray.
    

    Returns:
        A stack a focus indicators images,as an array[kf,kx,ky], values: [0,1]
    """
    quadrado = True
    
    fi_stack = []

    for i, aligned_img in enumerate(aligned_img_stack):

        # 1. Converte a imagem para escala de cinza
        img_gray =  convert_img_to_grayscale(aligned_img).astype(np.float64)
        img_gray = img_gray/255

        # 2. Aplica a Transformada de Fourier 2D
        f_transformada = np.fft.fft2(img_gray)

        # 3. Desloca o zero da frequência para o centro
        f_centralizado = np.fft.fftshift(f_transformada)

        # 4. Cria um filtro passa-alta para manter apenas frequências altas
        linhas, colunas = img_gray.shape
        # raio = min(linhas, colunas) // 10  # Ajusta o raio para capturar alta frequência
        
        #mascara = np.ones((linhas, colunas), dtype=np.uint8)
        #cv2.circle(mascara, (colunas // 2, linhas // 2), raio, 0, -1)
        raio = 0.1
        mascara = mascara_eliptica_gaussiana(linhas, colunas, raio)
        #mascara = mascara_eliptica_binaria(linhas, colunas, raio)

        # 5. Aplica a máscara no domínio da frequência
        f_centralizado_filtrado = f_centralizado * mascara

        # 6. Desfaz a centralização e aplica a Transformada Inversa
        f_inversa = np.fft.ifftshift(f_centralizado_filtrado)
        imagem_reconstruida = np.fft.ifft2(f_inversa)

        imagem_reconstruida = np.real(imagem_reconstruida)

        print_img_statistics('fourier: img_reconstruida', imagem_reconstruida)
        #save_image('/home/lelis/Documents/Projetos/Stereo_Multifocus/depth_from_focus/data/synthetic/scene_3/organized_files/output/fourier/', f'fourier_img_reconstruida_{i}.png', imagem_reconstruida, np.min(imagem_reconstruida), np.max(imagem_reconstruida))

        
        # 7. Calcula o módulo ao quadrado da imagem reconstruída
        if quadrado:
            imagem_final = imagem_reconstruida ** 2
            
        else:    
            imagem_final = imagem_reconstruida

        print_img_statistics('fourier: img_final', imagem_final)
        #save_image('/home/lelis/Documents/Projetos/Stereo_Multifocus/depth_from_focus/data/synthetic/scene_3/organized_files/output/fourier/', f'fourier_img_final_{i}.png', imagem_reconstruida, 0, np.max(imagem_reconstruida))

        # 8. Aplica um borramento de 3x3                
        kernel = [[1,2,1], [2,4,2], [1,2,1]]
        imagem_final = convolve(imagem_final, kernel)
        # imagem_final = mediana_ponderada()

        # Zera as bordas da imagem
        imagem_final = zero_borders(imagem_final, 10)
        print_img_statistics('fourier: img convolucionada', imagem_final)
        #save_image('/home/lelis/Documents/Projetos/Stereo_Multifocus/depth_from_focus/data/synthetic/scene_3/organized_files/output/fourier/', f'fourier_img_final_convolucionada_{i}.png', imagem_reconstruida, 0, np.max(imagem_reconstruida))
        fi_stack.append(imagem_final)

    fi_stack = np.array(fi_stack)
    
    #fi_stack = abs(fi_stack)

    min_val = np.min(fi_stack)
    max_val = np.max(fi_stack)
    print(f'focus indicator before normalization(fourier) max_val: {max_val}, min_val: {min_val}')

    # Remove outliers by clipping the values to the 1st and 99th percentiles
    p1, p99 = np.percentile(fi_stack, [1, 99])
    fi_stack = np.clip(fi_stack, p1, p99)

    # Normalize the focus indicator to the range [0, 1]
    fi_stack = fi_stack/ max_val
    
    min_val = np.min(fi_stack)
    max_val = np.max(fi_stack)
    print(f'focus indicator after normalization(fourier) max_val: {max_val}, min_val: {min_val}')

    return fi_stack#, mascara, f_centralizado_filtrado, f_centralizado, f_transformada




def mascara_eliptica_gaussiana(linhas, colunas, raio):
    """
    Cria uma máscara gaussiana elíptica de tamanho e raio especificado, ajustando para uma elipse de acordo com o número de linhas e colunas da imagem.

    Args:
        linhas: Número de linhas da máscara.
        colunas: Número de colunas da máscara.
        raio: Raio do círculo base.

    Returns:
        Máscara gaussiana elíptica.
    """
    raio_j = raio * colunas
    raio_i = raio * linhas

    mascara = np.zeros((linhas, colunas))
    centro_i = linhas // 2
    centro_j = colunas // 2

    for i in range(linhas):
        di = (i - centro_i)/raio_i
        gi =  exp(-(di ** 2) / 2)
        for j in range(colunas):
            dj = (j - centro_j)/raio_j
            gj = exp(-(dj ** 2) / 2)
            mascara[i, j] = 1 - gi * gj

    return mascara

def mascara_eliptica_binaria(linhas, colunas, raio):
    """
    Cria uma máscara elíptica binária de tamanho e raio especificado, ajustando para uma elipse de acordo com o número de linhas e colunas da imagem.

    Args:
        linhas: Número de linhas da máscara.
        colunas: Número de colunas da máscara.
        raio: Raio do círculo base.

    Returns:
        Máscara elíptica binária.
    """
    raio_j = raio * colunas
    raio_i = raio * linhas

    mascara = np.ones((linhas, colunas), dtype=np.uint8)
    centro_i = linhas // 2
    centro_j = colunas // 2

    for i in range(linhas):
        di = (i - centro_i) / raio_i
        for j in range(colunas):
            dj = (j - centro_j) / raio_j
            if di**2 + dj**2 <= 1:
                mascara[i, j] = 0

    return mascara


def filtro_de_media(imagem:np.array,kernel:np.array)->np.array:
    """
    Aplica um filtro de média na imagem.

    Args:
        imagem: Imagem a ser filtrada.
    Returns:
        Imagem filtrada.
    """
    #kernel = [[1,2,1], [2,4,2], [1,2,1]]
    #kernel = np.ones((tamanho, tamanho)) / (tamanho ** 2)
    imagem_filtrada = cv2.filter2D(imagem, -1, kernel)

    return imagem_filtrada