import matplotlib.pyplot as plt
import numpy as np
import cv2



def display_img(caminho_imagem):
    # Ler a imagem usando OpenCV
    imagem = cv2.imread(caminho_imagem)
    # Converter a imagem de BGR para RGB
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    # Exibir a imagem usando matplotlib
    plt.imshow(imagem_rgb)
    plt.axis('off')  # Ocultar os eixos
    plt.show()



from mpl_toolkits.mplot3d import Axes3D


def plotar_canais(imagem):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(imagem[:,:,0])
    axs[0].set_title('Canal 0')
    axs[0].axis('off')

    axs[1].imshow(imagem[:,:,1])
    axs[1].set_title('Canal 1')
    axs[1].axis('off')

    axs[2].imshow(imagem[:,:,2])
    axs[2].set_title('Canal 2')
    axs[2].axis('off')
    # Informações estatísticas
    print("Informações estatísticas dos canais:")
    for i in range(3):
        canal = imagem[:, :, i]
        print(f"Canal {i}:")
        print(f"  Mínimo: {canal.min()}")
        print(f"  Máximo: {canal.max()}")
        print(f"  Média: {canal.mean()}")
        print(f"  Desvio Padrão: {canal.std()}")
    plt.show()


def plotar_mapa_altura_3d(mapa_altura):
    """
    Esta função recebe um mapa de altura (numpy.ndarray) e plota em 3D.
    
    Parâmetros:
    mapa_altura (numpy.ndarray): O mapa de altura a ser plotado.
    
    Retorna:
    None
    """
    # Criar uma grade de coordenadas X e Y
    x = np.arange(mapa_altura.shape[1])
    y = np.arange(mapa_altura.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Criar a figura e o eixo 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar a superfície
    ax.plot_surface(x, y, mapa_altura, cmap='viridis')
    
    # Adicionar rótulos aos eixos
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altura')
    
    # Mostrar o gráfico
    plt.show()


def plotar_canais_3d(imagem):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    x = np.arange(imagem.shape[1])
    y = np.arange(imagem.shape[0])
    x, y = np.meshgrid(x, y)

    axs[0].plot_surface(x, y, imagem[:, :, 0], cmap='viridis')
    axs[0].set_title('Canal 0')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_zlabel('Altura')

    axs[1].plot_surface(x, y, imagem[:, :, 1], cmap='viridis')
    axs[1].set_title('Canal 1')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_zlabel('Altura')

    axs[2].plot_surface(x, y, imagem[:, :, 2], cmap='viridis')
    axs[2].set_title('Canal 2')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].set_zlabel('Altura')

    plt.show()
