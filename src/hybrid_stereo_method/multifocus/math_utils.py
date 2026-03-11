from math import cos, floor, pi
import numpy as np

def quadratic_interpolation(val, k_fuzzy):
    nframes = len(val)

    # Passo 1: Calcular o índice inteiro mais próximo de k_fuzzy
    kint = int(floor(k_fuzzy + 0.5))

    # Passo 2: Definir a janela de interpolação
    if kint <= 1:
        k0 = 0
        k1 = 2
    elif kint >= nframes - 2:
        k0 = nframes - 3
        k1 = nframes - 1
    else:
        k0 = kint - 2
        k1 = kint + 2

    m = k1 - k0 + 1

    # Passo 3: Calcular a diferença fracionária s
    s = k_fuzzy - kint
    assert -0.5 <= s <= 0.5

    # Passo 4: Definir os vetores x e y
    x = [k0 + j for j in range(m)]
    y = [val[k0 + j] for j in range(m)]

    # Passo 5: Calcular os pesos w
    a = pi * 0.5 * (m + 1)
    w = [0.5 * (1 + cos(a * (k0 + j - k_fuzzy))) for j in range(m)]

    # Passo 6: Regressão quadrática ponderada
    X = np.vstack([np.ones(m), x, np.square(x)]).T
    W = np.diag(w)

    # Verificar se a matriz é singular
    try:
        A = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    except np.linalg.LinAlgError:
        # Adicionar regularização para evitar singularidade
        regularization = 1e-8
        A = np.linalg.inv(X.T @ W @ X + regularization * np.eye(X.shape[1])) @ (X.T @ W @ y)

    # Coeficientes da parábola
    C, B, A = A

    # Passo 7: Calcular o valor interpolado vsel
    vsel = A * (k_fuzzy**2) + B * k_fuzzy + C

    # Passo 8: Retornar o valor interpolado
    return vsel


def linear_interpolation(val, k_fuzzy):
    nframes = len(val)

    # Passo 1: Calcular o índice inteiro mais próximo de k_fuzzy
    kint = int(floor(k_fuzzy))

    # Passo 2: Definir os índices de interpolação
    if kint < 0:
        k0 = 0
        k1 = 1
    elif kint >= nframes - 1:
        k0 = nframes - 2
        k1 = nframes - 1
    else:
        k0 = kint
        k1 = kint + 1

    # Passo 3: Calcular a diferença fracionária s
    s = k_fuzzy - k0
    assert 0 <= s <= 1

    # Passo 4: Calcular o valor interpolado
    vsel = (1 - s) * val[k0] + s * val[k1]

    # Passo 5: Retornar o valor interpolado
    return vsel
