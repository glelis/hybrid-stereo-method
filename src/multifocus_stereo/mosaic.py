import numpy as np
from multifocus_stereo.utils import linear_interpolation, quadratic_interpolation
import logging


def mosaic(iSel, image_stack:np.array, zFoc:list, interpolation_type:str):
    """
    Generates an all-in-focus image and a depth map from a stack of multi-focus images.

    Parameters:
        iSel (np.array): A 2D array (height x width) containing the fuzzy indices for each pixel.
        image_stack (np.array): A 4D array (n_frames x height x width x channels) representing the stack of multi-focus images.
        zFoc (list): A list of focus distances corresponding to each frame in the image stack.
        interpolation_type (str): The type of interpolation to use. Options are:
            - 'crop': Uses the nearest frame without interpolation.
            - 'quadratic_interpolation': Uses quadratic interpolation for smoother transitions.
            - 'linear_interpolation': Uses linear interpolation for smoother transitions.

    Returns:
        tuple:
            - sMos (np.array): The all-in-focus image (height x width x channels).
            - zMos (np.array): The depth map (height x width x channels).

    Notes:
        - The function processes each pixel independently, selecting or interpolating the appropriate focus value
          and corresponding pixel intensity from the image stack.
        - The interpolation type determines the method used to compute intermediate values when the focus index is not an integer.
    """
    
    n_frames, height, width, chanels = image_stack.shape

    logging.debug(f"Mosaic      height: {height}, width: {width}, n_frames: {n_frames}, chanels: {chanels}")
    
    sMos = np.zeros((height, width, chanels))
    zMos = np.zeros((height, width,chanels))

    #Calculo da imagem all_in_focus
    for i in range(height): #linha
        for j in range(width): #coluna

            
            if interpolation_type == 'crop':
                K_indice  = int(iSel[i, j])
                if K_indice < 0:
                    zMos[i, j] = zFoc[0]
                    sMos[i, j, :] = image_stack[0, i, j, :] 
 
                elif K_indice >= n_frames:
                    zMos[i, j] = zFoc[n_frames - 1]
                    sMos[i, j, :] = image_stack[n_frames - 1, i, j, :]

                else: 
                    zMos[i, j] = zFoc[K_indice]
                    sMos[i, j, :] = image_stack[K_indice , i, j, :]


            elif interpolation_type == 'quadratic_interpolation':
                k_fuzzy = iSel[i, j]
                i0 = int(np.floor(k_fuzzy))
                i1 = i0 + 1

                if i0 < 0:
                    zMos[i, j] = zFoc[0]
                    sMos[i, j, :] = image_stack[0, i, j, :] 
 
                elif i1 >= n_frames:
                    zMos[i, j] = zFoc[n_frames - 1]
                    sMos[i, j, :] = image_stack[n_frames - 1, i, j, :]
                
                else:
                    zMos[i, j] = quadratic_interpolation(zFoc, k_fuzzy)
                    for c in range(3):
                        sMos[i, j, c] = quadratic_interpolation(image_stack[:, i, j, c], k_fuzzy)


            elif interpolation_type == 'linear_interpolation':
                k_fuzzy = iSel[i, j]
                i0 = int(np.floor(k_fuzzy))
                i1 = i0 + 1

                if i0 < 0:
                    zMos[i, j] = zFoc[0]
                    sMos[i, j, :] = image_stack[0, i, j, :] 
 
                elif i1 >= n_frames:
                    zMos[i, j] = zFoc[n_frames - 1]
                    sMos[i, j, :] = image_stack[n_frames - 1, i, j, :]

                else:
                    zMos[i, j] = linear_interpolation(zFoc, k_fuzzy)
                    for c in range(3):
                        sMos[i, j, c] = linear_interpolation(image_stack[:, i, j, c], k_fuzzy)
        


    return sMos, zMos



