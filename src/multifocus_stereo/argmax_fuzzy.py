import numpy as np
import os
import csv
from multifocus_stereo.utils import normalize
import logging



def argmax_fuzzy(focus_indicator_stack:np.ndarray, debug:bool, debug_data_path:str)-> tuple:
    """
    Calcula o argmax difuso (fuzzy) para uma pilha de indicadores de foco e confiança.

    Args:
        focus_indicator_stack (list[np.ndarray]): Pilha de imagens com medidas de foco.

    Returns:
        tuple: Duas imagens, img_arg e img_conf(normalizada) contendo, respectivamente, 
               os valores de argmax fuzzy e as confiabilidades associadas.
    """


    if debug:
        # Cria um arquivo CSV para salvar informações de depuração
        if not os.path.exists(debug_data_path):
            os.makedirs(debug_data_path)
        with open(os.path.join(debug_data_path, 'debug.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['pixel_i', 'pixel_j', 'focus_values','x_list', 'y_list', 'w_list', 'k_fuzzy', 'conf', 'fnoc', 'A', 'B', 'C', ])

    logging.debug(f"Calculating argmax fuzzy for array {focus_indicator_stack.shape}, min_all: {np.min(focus_indicator_stack)}, max_all: {np.max(focus_indicator_stack)}")


    # Dimensões da pilha de foco
    #n_frames = len(focus_indicator_stack)
    _, height, width = focus_indicator_stack.shape

    # Inicializa as imagens de resultado e confiança com zeros (tipo float64)
    img_arg = np.zeros((height, width), dtype=np.float64)
    img_conf = np.zeros((height, width), dtype=np.float64)

    # Itera sobre cada pixel das imagens
    for i in range(height):
        for j in range(width):

            
            # Extrai as medidas de foco para o pixel atual ao longo dos frames
            focus_values = np.array([frame[i, j] for frame in focus_indicator_stack])
            # Calcula o argmax fuzzy e a confiança para o pixel atual
            img_arg[i, j], img_conf[i, j] = argmax_fuzzy_1d_v2(focus_values,[i,j], debug, debug_data_path)


            #normaliza o indicador de foco pixel por pixel e salvando na pilha de indicadores de foco
#            if np.max(focus_values) == 0:
#                focus_values_normalized = focus_values
#            else:
#                focus_values_normalized = [val /np.max(focus_values) for val in focus_values]
#
#            for k in range(n_frames):
#                focus_indicator_stack[k, i, j] = focus_values_normalized[k]

    img_conf = normalize(img_conf)

    return img_arg, img_conf





def find_index_of_max_sum(focus_values:np.array)->int:
        """
        Encontra o índice do valor máximo da soma de três elementos consecutivos em uma lista de valores de foco.

        Args:
            focus_values (np.array): Lista ou array de valores de foco.

        Returns:
            int: O índice do valor máximo da soma de três elementos consecutivos.
        """
        max_sum = -np.inf
        max_index = -1
        for i in range(1, len(focus_values) - 1):
            current_sum = focus_values[i - 1] + focus_values[i] + focus_values[i + 1]
            if current_sum > max_sum:
                max_sum = current_sum
                max_index = np.argmax([focus_values[i - 1], focus_values[i], focus_values[i + 1]])
                index = i + max_index - 1
        return index

def calculate_weights(focus_values:np.array)->np.array:
        """
        Calculate weights for the focus values. Higher focus values will have higher weights.

        Args:
            focus_values (list): List of focus values.

        Returns:
            list: List of weights corresponding to the focus values.
        """
        total_focus = sum(focus_values)
        if total_focus == 0:
            return [1] * len(focus_values)  # Avoid division by zero, return equal weights
        return [value / total_focus for value in focus_values]



def argmax_fuzzy_1d_v2(focus_values, pixel_location, debug, debug_data_path):
    

    n = len(focus_values)
    k_max = find_index_of_max_sum(focus_values)
    
    
    if focus_values[k_max] == 0:
        return n/2, 0
    
    # Calcula o raio r da regressão
    r_max = 2
    r = r_max
    if k_max - r < 0:
        r = k_max
    elif k_max + r >= n:
        r = n - k_max - 1
    if r <= 0:
        r=1

    # m = 2*r+1 #numero de pontos da regressao

    assert n >= 2*r+1, "insuficient images"

    # Escolha k0 e k1, de modo que k1-k0=2r e k0..k1, esta contigo em 0..n-1
    k0 = k_max- r #ponto inicial da regressao
    k1 = k_max + r #ponto final da regressao
    #ajusta k0 e k1 para que estejam dentro do intervalo
    if k0 < 0:
        r = r-1
        k1 = k1 - k0
        k0 = 0
    if k1 >= n:
        k0 = k0 - (k1 -n+1)
        k1 =n-1

    # aproxima uma funcao de segundo grau nos valores focus_values[k0..k1]
    x_list = [i for i in range(k0,k1+1)] #posicao dos pontos
    y_list = [(focus_values[i]) for i in range(k0,k1+1)]
    w_list = calculate_weights(focus_values[k0:k1+1])

    A, B, C = tuple(np.polyfit(x_list, y_list, 2, w=w_list)) #coeficientes da funcao de segundo grau

    if A > 0 or abs(A) < 1.0e-6: #se a funcao for convexa ou muito proxima de zero
        k_fuzzy = k_max
        conf = 0
        fnoc = 0

    else: #calcula o ponto de maximo da funcao
        k_fuzzy = -B/(2*A) #ponto de maximo da funcao x
        k_fuzzy = min(n-0.5, max(-0.5, k_fuzzy)) #garante que o ponto esta dentro do intervalo
        fnoc = -(B**2)/(4*A) + C # valor do foco funcao no ponto maximo y(x) 
        #conf = conf/(3*(rlap**2)) #normaliza a confianca
        if fnoc <0:
            conf = 0
        else:
            #conf = 1/(fnoc) #confianca do ponto de minimo
            conf = abs(A) 

    if debug:
        # Save debug information to a CSV file
        with open(os.path.join(debug_data_path, 'debug.csv'), 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([pixel_location[0], pixel_location[1], [focus_values[i] for i in range(len(focus_values))], x_list, y_list, w_list, k_fuzzy, conf, fnoc, A, B, C,])

    return k_fuzzy, conf 

