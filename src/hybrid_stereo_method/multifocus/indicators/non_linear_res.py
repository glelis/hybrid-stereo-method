import numpy as np
from scipy.linalg import lstsq

def calcular_indicador_foco(imagem):
    # Definindo a máscara de pesos W(x, y)
    mascara_pesos = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    # Pegando as dimensões da imagem
    altura, largura = imagem.shape

    # Inicializando uma matriz para armazenar os valores de foco
    foco = np.zeros((altura, largura))

    # Iterando sobre cada pixel da imagem (exceto as bordas)
    for x in range(1, altura - 1):
        for y in range(1, largura - 1):
            # Extraindo a vizinhança 3x3 ao redor do pixel (x, y)
            vizinhanca = imagem[x - 1:x + 2, y - 1:y + 2]

            # Definindo as coordenadas para o ajuste linear
            X = np.array([
                [-1, -1, 1],
                [0, -1, 1],
                [1, -1, 1],
                [-1, 0, 1],
                [0, 0, 1],
                [1, 0, 1],
                [-1, 1, 1],
                [0, 1, 1],
                [1, 1, 1]
            ])

            # Flatten da vizinhança para os valores de intensidade
            intensidades = vizinhanca.flatten()

            # Calculando o ajuste linear: G(x, y) = A + B*x + C*y
            # Usando least squares para obter os coeficientes A, B e C
            coeficientes, _, _, _ = lstsq(X, intensidades)

            # Calculando G(x, y) para cada ponto da vizinhança
            G = (X @ coeficientes).reshape((3, 3))

            # Calculando Q(x, y) = I(x, y) - G(x, y)
            Q = vizinhanca - G

            # Calculando F(pixel central) usando a fórmula fornecida
            F = np.sum((Q ** 2) * mascara_pesos)

            # Armazenando o valor de foco para o pixel central
            foco[x, y] = F

    return foco
