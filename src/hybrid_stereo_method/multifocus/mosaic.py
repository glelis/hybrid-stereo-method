import logging

import numpy as np

from hybrid_stereo_method.multifocus.math_utils import linear_interpolation, quadratic_interpolation


def mosaic(iSel, image_stack: np.array, zFoc: list, interpolation_type: str):
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

    logging.debug(
        f"Mosaic      height: {height}, width: {width}, n_frames: {n_frames}, chanels: {chanels}"
    )

    sMos = np.zeros((height, width, chanels))
    zMos = np.zeros((height, width))

    if interpolation_type == "crop":
        interpolate = None
    elif interpolation_type == "quadratic_interpolation":
        interpolate = quadratic_interpolation
    elif interpolation_type == "linear_interpolation":
        interpolate = linear_interpolation
    else:
        raise ValueError(f"Unknown interpolation_type: {interpolation_type}")

    # Calculo da imagem all_in_focus
    for i in range(height):  # linha
        for j in range(width):  # coluna
            k_fuzzy = iSel[i, j]

            if interpolate is None:  # crop: usa o frame mais próximo, sem interpolação
                K_indice = min(max(int(k_fuzzy), 0), n_frames - 1)
                zMos[i, j] = zFoc[K_indice]
                sMos[i, j, :] = image_stack[K_indice, i, j, :]
                continue

            i0 = int(np.floor(k_fuzzy))

            if i0 < 0:
                zMos[i, j] = zFoc[0]
                sMos[i, j, :] = image_stack[0, i, j, :]

            elif i0 + 1 >= n_frames:
                zMos[i, j] = zFoc[n_frames - 1]
                sMos[i, j, :] = image_stack[n_frames - 1, i, j, :]

            else:
                zMos[i, j] = interpolate(zFoc, k_fuzzy)
                for c in range(chanels):
                    sMos[i, j, c] = interpolate(image_stack[:, i, j, c], k_fuzzy)

    return sMos, zMos
