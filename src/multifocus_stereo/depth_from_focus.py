import numpy as np
from multifocus_stereo.utils import save_image, linear_interpolation, quadratic_interpolation
from math import floor, sqrt, comb, exp, log, sin, cos, pi
from datetime import datetime
import logging


from multifocus_stereo.weighted_filter import *
from multifocus_stereo.argmax_fuzzy import *




def lap_sqr_range(focus_indicator_stack):
    """
    Calculate the square root of the average of the squared values of each pixel 
    across a stack of focus images.

    Args:
        focus_indicator_stack: Stack of focus measure images.

    Returns:
        The square root of the average of the squared values.
    """
    ni = len(focus_indicator_stack)
    nx, ny = focus_indicator_stack[0].shape

    # Use numpy operations for optimization
    soma_q = np.sum(focus_indicator_stack)

    return sqrt(soma_q / (ni * nx * ny))



def all_in_focus( aligned_img_stack:np.array, focus_indicator_stack:np.array, focal_step, interpolation_type:str, debug_data_path:str, debug:bool):
    
    height, width, _ = aligned_img_stack[0].shape
    n_frames = len(aligned_img_stack)

    
    all_in_focus_img = np.zeros_like(aligned_img_stack[0])
    focus_measure_img = np.zeros_like(all_in_focus_img)

    #cria uma lista de imagens com as partes selecionadas
    select_img_stack = [np.zeros_like(aligned_img_stack[0]) for _ in aligned_img_stack]


    logging.debug(f"height: {height}, width: {width}, n_frames: {n_frames}")


    # Calcula o argmax fuzzy para a pilha de indicadores de foco
    img_arg, img_conf = argmax_fuzzy(focus_indicator_stack, debug, debug_data_path)

    # Calcula a profundidade a partir do argmax fuzzy
    depth_map = img_arg * focal_step
    
    img_arg_filtered = filter_img_arg_mediana_2(depth_map, img_conf)
    depth_map_filtered = img_arg_filtered * focal_step

    save_image(debug_data_path, 'depth_map_filtered.png', depth_map_filtered, np.min(depth_map_filtered),np.max(depth_map_filtered))

    #Calculo da imagem all_in_focus
    for i in range(height): #linha
        for j in range(width): #coluna
            K_indice  = int(img_arg[i, j])
            k_fuzzy = img_arg[i, j]

            if interpolation_type == 'crop':
                all_in_focus_img[i, j, :] = aligned_img_stack[K_indice , i, j, :]
                select_img_stack[K_indice ][i, j] = 1

            elif interpolation_type == 'quadratic_interpolation':
                for c in range(3):
                    all_in_focus_img[i, j, c] = quadratic_interpolation(aligned_img_stack[:, i, j, c], k_fuzzy) #pode ser otimizado pois não precisa ser calculado para cada canal

            elif interpolation_type == 'linear_interpolation':
                i0 = int(np.floor(k_fuzzy))
                i1 = i0 + 1

                if i0 < 0:
                    all_in_focus_img[i, j, :] = aligned_img_stack[0, i, j, :]
                elif i1 >= n_frames:
                    all_in_focus_img[i, j, :] = aligned_img_stack[n_frames - 1, i, j, :]
                else:
                    for c in range(3):
                        all_in_focus_img[i, j, c] = linear_interpolation(aligned_img_stack[:, i, j, c], k_fuzzy)
            #lixo  
            focus_measure_img[i, j] = focus_indicator_stack[K_indice , i, j]


    return depth_map, all_in_focus_img, select_img_stack, focus_measure_img, img_conf



def mosaic(iSel, aligned_img_stack:np.array, zFoc:list, interpolation_type:str):
    
    n_frames, height, width, chanels = aligned_img_stack.shape

    
    sMos = np.zeros_like(aligned_img_stack[0])
    zMos = np.zeros((height, width))
    #focus_measure_img = np.zeros_like(sMos)

    #cria uma lista de imagens com as partes selecionadas
    #select_img_stack = [np.zeros_like(aligned_img_stack[0]) for _ in aligned_img_stack]


    logging.debug(f"Mosaic      height: {height}, width: {width}, n_frames: {n_frames}, chanels: {chanels}")


    # Calcula o argmax fuzzy para a pilha de indicadores de foco
    #img_arg, img_conf = argmax_fuzzy(focus_indicator_stack, debug, debug_data_path)

    # Calcula a profundidade a partir do argmax fuzzy
    #depth_map = img_arg * focal_step
    
    #verificar codigo para retirar ou otimizar
    #img_arg_filtered = filter_img_arg_mediana_2(depth_map, img_conf)
    #depth_map_filtered = img_arg_filtered * focal_step
    #save_image(debug_data_path, 'depth_map_filtered.png', depth_map_filtered, np.min(depth_map_filtered),np.max(depth_map_filtered))

    #Calculo da imagem all_in_focus
    for i in range(height): #linha
        for j in range(width): #coluna
            K_indice  = int(iSel[i, j])
            k_fuzzy = iSel[i, j]

            if interpolation_type == 'crop':
                 
                zMos[i, j] = zFoc[K_indice]

                sMos[i, j, :] = aligned_img_stack[K_indice , i, j, :]
                #select_img_stack[K_indice ][i, j] = 1

            elif interpolation_type == 'quadratic_interpolation':

                zMos[i, j] = quadratic_interpolation(zFoc, k_fuzzy)

                for c in range(3):
                    sMos[i, j, c] = quadratic_interpolation(aligned_img_stack[:, i, j, c], k_fuzzy) #pode ser otimizado pois não precisa ser calculado para cada canal


            elif interpolation_type == 'linear_interpolation':
                i0 = int(np.floor(k_fuzzy))
                i1 = i0 + 1

                if i0 < 0:
                    zMos[i, j] = zFoc[0]
                    sMos[i, j, :] = aligned_img_stack[0, i, j, :] 
 
                elif i1 >= n_frames:
                    zMos[i, j] = zFoc[n_frames - 1]
                    sMos[i, j, :] = aligned_img_stack[n_frames - 1, i, j, :]

                else:
                    zMos[i, j] = linear_interpolation(zFoc, k_fuzzy)
                    for c in range(3):
                        sMos[i, j, c] = linear_interpolation(aligned_img_stack[:, i, j, c], k_fuzzy)
        



    return sMos, zMos



