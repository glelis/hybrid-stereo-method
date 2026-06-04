import csv
import logging
import os
import warnings

import numpy as np

from hybrid_stereo_method.infrastructure.utils import normalize


def compute_argmax_fuzzy(
    focus_indicator_stack: np.ndarray, debug: bool, debug_data_path: str, fuzzy_params: dict = None
) -> tuple:
    """
    Calcula o argmax difuso (fuzzy) para uma pilha de indicadores de foco e confiança.

    Args:
        focus_indicator_stack (list[np.ndarray]): Pilha de imagens com medidas de foco.

    Returns:
        tuple: Duas imagens, iSel e wSel(normalizada) contendo, respectivamente,
               os valores de argmax fuzzy e as confiabilidades associadas.
    """

    csvfile = None
    csvwriter = None
    if debug:
        # Cria um arquivo CSV para salvar informações de depuração
        if not os.path.exists(debug_data_path):
            os.makedirs(debug_data_path)
        csvfile = open(os.path.join(debug_data_path, "debug.csv"), "w", newline="")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "pixel_i",
                "pixel_j",
                "focus_values",
                "x_list",
                "y_list",
                "w_list",
                "k_fuzzy",
                "conf",
                "fnoc",
                "A",
                "B",
                "C",
            ]
        )

    logging.debug(
        f"Calculating argmax fuzzy for array {focus_indicator_stack.shape}, min_all: {np.min(focus_indicator_stack)}, max_all: {np.max(focus_indicator_stack)}"
    )

    # Dimensões da pilha de foco
    _, height, width = focus_indicator_stack.shape

    # Inicializa as imagens de resultado e confiança com zeros (tipo float64)
    iSel = np.zeros((height, width), dtype=np.float64)
    wSel = np.zeros((height, width), dtype=np.float64)

    try:
        # Itera sobre cada pixel das imagens
        for i in range(height):
            for j in range(width):
                # Extrai as medidas de foco para o pixel atual ao longo dos frames
                focus_values = focus_indicator_stack[:, i, j]

                # Calcula o argmax fuzzy e a confiança para o pixel atual
                iSel[i, j], wSel[i, j] = compute_argmax_fuzzy_1d(
                    focus_values, [i, j], fuzzy_params, csv_writer=csvwriter
                )
    finally:
        if csvfile is not None:
            csvfile.close()

    wSel = normalize(wSel)

    return iSel, wSel


def find_index_of_max_sum(focus_values: np.array) -> int:
    """
    Encontra o índice do valor máximo da soma de três elementos consecutivos em uma lista de valores de foco.

    Args:
        focus_values (np.array): Lista ou array de valores de foco.

    Returns:
        int: O índice do valor máximo da soma de três elementos consecutivos.
    """
    max_sum = -np.inf
    for i in range(1, len(focus_values) - 1):
        current_sum = focus_values[i - 1] + focus_values[i] + focus_values[i + 1]
        if current_sum > max_sum:
            max_sum = current_sum
            index = i - 1 + int(np.argmax(focus_values[i - 1 : i + 2]))
    return index


def calculate_weights(focus_values: np.array) -> np.array:
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
    # Add small regularization to avoid zero weights which can cause SVD to not converge
    return [(value / total_focus) + 1e-6 for value in focus_values]


def compute_argmax_fuzzy_1d(focus_values, pixel_location, fuzzy_params=None, csv_writer=None):
    if fuzzy_params is None:
        fuzzy_params = {}

    n = len(focus_values)
    k_max = find_index_of_max_sum(focus_values)

    if focus_values[k_max] == 0:
        return n / 2, 0

    # Calcula o raio r da regressão
    r_max = fuzzy_params.get("r_max", 2)
    r = r_max
    if k_max - r < 0:
        r = k_max
    elif k_max + r >= n:
        r = n - k_max - 1
    if r <= 0:
        r = 1

    assert n >= 2 * r + 1, "insuficient images"

    # Escolha k0 e k1, de modo que k1-k0=2r e k0..k1, esta contigo em 0..n-1
    k0 = k_max - r  # ponto inicial da regressao
    k1 = k_max + r  # ponto final da regressao
    # ajusta k0 e k1 para que estejam dentro do intervalo
    if k0 < 0:
        r = r - 1
        k1 = k1 - k0
        k0 = 0
    if k1 >= n:
        k0 = k0 - (k1 - n + 1)
        k1 = n - 1

    # aproxima uma funcao de segundo grau nos valores focus_values[k0..k1]
    x_list = list(range(k0, k1 + 1))  # posicao dos pontos
    y_list = list(focus_values[k0 : k1 + 1])
    w_list = calculate_weights(focus_values[k0 : k1 + 1])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C = tuple(
                np.polyfit(x_list, y_list, 2, w=w_list)
            )  # coeficientes da funcao de segundo grau
    except np.linalg.LinAlgError:
        try:
            A, B, C = tuple(np.polyfit(x_list, y_list, 2))
        except np.linalg.LinAlgError:
            # If all fits fail, fallback to returning the max index
            return k_max, 0

    polyfit_epsilon = fuzzy_params.get("polyfit_epsilon", 1.0e-9)
    if A > 0 or abs(A) < polyfit_epsilon:  # se a funcao for convexa ou muito proxima de zero
        k_fuzzy = 0
        conf = 0
        fnoc = 0

    else:  # calcula o ponto de maximo da funcao
        k_fuzzy = -B / (2 * A)  # ponto de maximo da funcao x
        k_fuzzy = max(0, min(n, k_fuzzy))  # garante que o ponto esta dentro do intervalo

        fnoc = -(B**2) / (4 * A) + C  # valor do foco funcao no ponto maximo y(x)
        if fnoc < 0:
            conf = 0
        else:
            conf = abs(A) / fnoc  # confianca do ponto de maximo

    if csv_writer is not None:
        # Save debug information to the shared CSV file
        csv_writer.writerow(
            [
                pixel_location[0],
                pixel_location[1],
                [float(v) for v in focus_values],
                [int(x) for x in x_list],
                [float(y) for y in y_list],
                [float(w) for w in w_list],
                float(k_fuzzy),
                float(conf),
                float(fnoc),
                float(A),
                float(B),
                float(C),
            ]
        )

    return k_fuzzy, conf
