import numpy as np
## não está funcioando corretamente

def filter_img_arg_mediana_2(img_arg, img_conf):
    height, width = img_arg.shape
    
    img_filtered = np.zeros((height, width), dtype=np.uint8)

    for i in range(height): #linha
        for j in range(width): #coluna
            janela_arg = extrai_janela_3x3(img_arg, i, j)
            janela_conf = extrai_janela_3x3(img_conf, i, j)
            mediana_arg = median_ponderada_2(janela_arg, janela_conf)
            img_filtered[i, j] = mediana_arg

    return img_filtered

def extrai_janela_3x3(img, i, j):
    height, width = img.shape
    janela = []
    for di in -1,0,1:
        for dj in -1,0,1:
            i1 = i + di
            j1 = j + dj
            if i1 >=0 and i1<height and j1>=0 and j1<width:
                janela.append(img[i1, j1])
    
    return janela

def median_ponderada_2(janela_arg:np.asarray, janela_conf:np.asarray):
    """
    Calculate the weighted median of a list of values.
    This function takes two lists: one with values and another with corresponding weights.
    It calculates the weighted median, which is the value where the cumulative weight 
    is equal to half of the total weight.
    Parameters:
    janela_arg (list): List of values.
    janela_conf (list): List of weights corresponding to the values in janela_arg.
    Returns:
    float: The weighted median of the input values.
    """
    conf_total = sum(janela_conf)
    janela = zip(janela_arg, janela_conf)
    janela = sorted(janela, key=lambda x: x[1])

    soma_conf = 0
    k = 0

    while k<len(janela) and soma_conf <= conf_total/2:
        soma_conf += janela[k][1]
        k += 1

    k = k - 1
    assert k >= 0
    assert k<len(janela)
    
    # A soma dos confs de zero ate k deu maior que conf_total/2
    if k > 0:
        s0 = soma_conf - janela[k-1][1]
        s1 = soma_conf 
        v0 =janela[k-1][0]
        v1 = janela[k][0]
        r = (conf_total/2 - s0)/(s1 - s0)
        return v0 + r*(v1 - v0)
    else:
        return janela[k][0]

  