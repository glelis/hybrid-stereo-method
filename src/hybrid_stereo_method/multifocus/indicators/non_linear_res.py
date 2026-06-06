import numpy as np
from scipy.linalg import lstsq


def calcular_indicador_foco(imagem):
    # Definindo a máscara de pesos W(x, y)
    mascara_pesos = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Pegando as dimensões da imagem
    altura, largura = imagem.shape

    # Inicializando uma matriz para armazenar os valores de foco
    foco = np.zeros((altura, largura))

    # Coordenadas para o ajuste linear — constantes, hoistadas fora do loop
    X = np.array(
        [
            [-1, -1, 1],
            [0, -1, 1],
            [1, -1, 1],
            [-1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
            [-1, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    # MF-10: o ajuste do plano usa os MESMOS pesos W do resíduo final
    # (WLS via reescala por sqrt(w)); antes o lstsq era não ponderado.
    # sw é constante (mascara_pesos não muda), hoistado fora do loop.
    w = mascara_pesos.flatten().astype(np.float64)
    sw = np.sqrt(w)
    X_sw = X * sw[:, None]

    # Iterando sobre cada pixel da imagem (exceto as bordas)
    for x in range(1, altura - 1):
        for y in range(1, largura - 1):
            # Extraindo a vizinhança 3x3 ao redor do pixel (x, y)
            vizinhanca = imagem[x - 1 : x + 2, y - 1 : y + 2]

            # Flatten da vizinhança para os valores de intensidade
            intensidades = vizinhanca.flatten()

            # Calculando o ajuste linear: G(x, y) = A + B*x + C*y
            # Usando WLS: mesmos pesos do resíduo final para coerência teórica
            coeficientes, _, _, _ = lstsq(X_sw, intensidades * sw)

            # Calculando G(x, y) para cada ponto da vizinhança
            G = (X @ coeficientes).reshape((3, 3))

            # Calculando Q(x, y) = I(x, y) - G(x, y)
            Q = vizinhanca - G

            # Calculando F(pixel central) usando a fórmula fornecida
            F = np.sum((Q**2) * mascara_pesos)

            # Clamp residuals below machine precision to exactly 0 (WLS
            # solve of a perfect plane leaves F ~1e-27 due to float64 noise).
            foco[x, y] = F if np.finfo(np.float64).eps < F else 0.0

    return foco
