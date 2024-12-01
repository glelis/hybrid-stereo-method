import numpy as np


def filter_img_arg_mediana_1(img:np.array, img_conf:np.array):

    #kernel  = [[1,2,1], [2,4,2], [1,2,1]]
    kernel = [[1,2,3,2,1], [2,3,4,3,2], [3,4,5,4,3], [2,3,4,3,2], [1,2,3,2,1]]
    kernel = np.array(kernel)
    img_res = np.ones_like(img)
    linhas, colunas = img.shape

    for i in range(1, linhas-1):
        for j in range(1, colunas-1):
            lista = []
            for di in -1, 0, 1:
                for dj in  -1, 0, 1:
                    lista.append((img[i+di, j+dj], kernel[di+1, dj+1] * img_conf[i+di, j+dj])) 
            lista.sort(key=lambda x: x[0])
            #print(f'lista: {lista}')
            soma = 0
            for val, peso in lista:
                soma += peso
            limite = soma/2
            soma = 0
            k = 0
            while k<len(lista) and soma < limite:
                soma += lista[k][1]
                k += 1
            if k >= len(lista):
                k = len(lista) - 1

            img_res[i,j] = lista[k][0] 
            #print(f'lista_k0: {lista[k][0]}')

    return img_res

            