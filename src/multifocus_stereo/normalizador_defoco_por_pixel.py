import numpy as np

def normaliza_indicador_de_foco_por_pixel(focus_indicator_stack):
    
    n_frames, height, width = focus_indicator_stack.shape
    
    # Itera sobre cada pixel das imagens
    for i in range(height):
        for j in range(width):

            
            # Extrai as medidas de foco para o pixel atual ao longo dos frames
            focus_values = np.array([frame[i, j] for frame in focus_indicator_stack])
            
            #normaliza o indicador de foco pixel por pixel e salvando na pilha de indicadores de foco
            if np.max(focus_values) == 0:
               focus_values_normalized = focus_values
            else:
                focus_values_normalized = [val /np.max(focus_values) for val in focus_values]

            for k in range(n_frames):
                focus_indicator_stack[k, i, j] = focus_values_normalized[k]

    return focus_indicator_stack