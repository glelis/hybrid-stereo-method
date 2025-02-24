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




def plot_circle(image, center, radius):
    # Create a copy of the image to draw the circle on
    output_image = image.copy()
    
    # Draw the circle
    cv2.circle(output_image, center, radius, (255, 0, 0), 2)
    
    # Draw the center of the circle
    cv2.circle(output_image, center, 2, (0, 255, 0), 3)
    
    # Display the image with the circle
    plt.imshow(output_image, cmap='gray')
    plt.title('Detected Circle')
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
        channel = imagem[:, :, i]
        print(f"Canal {i}:")
        print(f"  Mínimo: {channel.min()}")
        print(f"  Máximo: {channel.max()}")
        print(f"  Média: {channel.mean()}")
        print(f"  Desvio Padrão: {channel.std()}")
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






def disp_channels(normal_in=None, height=None, width=None, delay=0, name=None, save_path=None):
    """
    Visualize normal as a normal map in a single window with all channels and save the image if a path is provided.
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :param save_path: path to save the final image
    :return: None
    """
    if normal_in is None:
        raise ValueError("Surface normal `normal` is None")
    
    # Reshape para coordenadas de imagem
    normal = np.reshape(normal_in, (height, width, 3))
    
    # Trocar canais RGB para BGR
    normal[:, :, 0], normal[:, :, 2] = normal[:, :, 2], normal[:, :, 0].copy()  # Swap RGB <-> BGR
    
    # Redimensionar valores para o intervalo [0, 255] (formato de imagem)
    normal = ((normal + 1.0) / 2.0 * 255).astype(np.uint8)

    # Separar os canais
    channel_0 = normal[:, :, 0]
    channel_1 = normal[:, :, 1]
    channel_2 = normal[:, :, 2]

    # Combinar os canais horizontalmente
    combined = cv2.hconcat([channel_0, channel_1, channel_2])

    # Exibir a imagem em uma única janela
    if name is None:
        name = 'Channel Visualization'
    cv2.imshow(name, combined)
    cv2.waitKey(delay)
    cv2.destroyWindow(name)
    cv2.waitKey(1)    # to deal with frozen window...

    # Salvar a imagem se um caminho for fornecido
    if save_path is not None:
        cv2.imwrite(save_path+"Channels.jpg", combined)


def disp_channels_3d(normal_in=None, height=None, width=None, delay=0, name=None, save_path=None):
    """
    Visualize normal as a normal map in 3D with all channels and save the image if a path is provided.
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :param save_path: path to save the final image
    :return: None
    """
    if normal_in is None:
        raise ValueError("Surface normal `normal` is None")
    
    # Reshape para coordenadas de imagem
    normal = np.reshape(normal_in, (height, width, 3))
    #normal = np.reshape(normal_in, (width, height, 3))
    
    # Trocar canais RGB para BGR
    #normal[:, :, 0], normal[:, :, 2] = normal[:, :, 2], normal[:, :, 0].copy()  # Swap RGB <-> BGR
    
    # Redimensionar valores para o intervalo [0, 255] (formato de imagem)
    #normal = ((normal + 1.0) / 2.0 * 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    x = np.arange(normal.shape[1])
    y = np.arange(normal.shape[0])
    x, y = np.meshgrid(x, y)

    axs[0].plot_surface(x, y, normal[:, :, 0], cmap='viridis')
    axs[0].set_title('X Axis')
    #axs[0].set_xlabel('X')
    #axs[0].set_ylabel('Y')
    axs[0].set_zlabel('intensity')

    axs[1].plot_surface(x, y, normal[:, :, 1], cmap='viridis')
    axs[1].set_title('Y Axis')
    #axs[1].set_xlabel('X')
    #axs[1].set_ylabel('Y')
    axs[1].set_zlabel('intensity')

    axs[2].plot_surface(x, y, normal[:, :, 2], cmap='viridis')
    axs[2].set_title('Z Axis')
    #axs[2].set_xlabel('X')
    #axs[2].set_ylabel('Y')
    axs[2].set_zlabel('intensity')

    # Salvar o gráfico como uma imagem em memória
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Fecha a figura para liberar memória

    # Converter a imagem para um array NumPy
    img_array = np.array(Image.open(buf))
    
    # Exibir a imagem em uma única janela
    if name is None:
        name = 'Channel Visualization 3D'
    cv2.imshow(name, img_array)
    cv2.waitKey(delay)
    cv2.destroyWindow(name)
    cv2.waitKey(1)    # to deal with frozen window...

    # Salvar a imagem se um caminho for fornecido
    if save_path is not None:
        cv2.imwrite(save_path+"Channels_3D.jpg", img_array)
    return img_array



#import numpy as np
#import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

#def disp_channels_3d(normal_in=None, height=None, width=None, delay=0, name=None):
#    """
#    Plota os canais de uma imagem em 3D e retorna o gráfico como um array NumPy.
#    :param imagem: Imagem 3D (altura x largura x 3).
#    :return: Array NumPy representando o gráfico gerado.
#    """
#
#    if normal_in is None:
#        raise ValueError("Surface normal `normal` is None")
#    
#    # Reshape para coordenadas de imagem
#    normal = np.reshape(normal_in, (height, width, 3))
#    
#    # Trocar canais RGB para BGR
#    normal[:, :, 0], normal[:, :, 2] = normal[:, :, 2], normal[:, :, 0].copy()  # Swap RGB <-> BGR
#    
#    # Redimensionar valores para o intervalo [0, 255] (formato de imagem)
#    normal = ((normal + 1.0) / 2.0 * 255).astype(np.uint8)
#
#    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
#
#    x = np.arange(normal.shape[1])
#    y = np.arange(normal.shape[0])
#    x, y = np.meshgrid(x, y)
#
#    axs[0].plot_surface(x, y, normal[:, :, 0], cmap='viridis')
#    axs[0].set_title('Canal 0')
#    axs[0].set_xlabel('X')
#    axs[0].set_ylabel('Y')
#    axs[0].set_zlabel('Altura')
#
#    axs[1].plot_surface(x, y, normal[:, :, 1], cmap='viridis')
#    axs[1].set_title('Canal 1')
#    axs[1].set_xlabel('X')
#    axs[1].set_ylabel('Y')
#    axs[1].set_zlabel('Altura')
#
#    axs[2].plot_surface(x, y, normal[:, :, 2], cmap='viridis')
#    axs[2].set_title('Canal 2')
#    axs[2].set_xlabel('X')
#    axs[2].set_ylabel('Y')
#    axs[2].set_zlabel('Altura')
#
#    # Salvar o gráfico como uma imagem em memória
#    buf = BytesIO()
#    plt.savefig(buf, format='png', bbox_inches='tight')
#    buf.seek(0)
#    plt.close(fig)  # Fecha a figura para liberar memória
#
#    # Converter a imagem para um array NumPy
#    img_array = np.array(Image.open(buf))
#
#    # Exibir a imagem em uma única janela
#    if name is None:
#        name = 'Channel Visualization'
#    cv2.imshow(name, img_array)
#    cv2.waitKey(delay)
#    cv2.destroyWindow(name)
#    cv2.waitKey(1)    # to deal with frozen window...
#
#
#    return img_array
