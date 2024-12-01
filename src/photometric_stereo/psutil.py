import cv2
import glob
import numpy as np

from math import hypot, atan2, cos, sin

def load_lighttxt(filename=None):
    """
    Load light file specified by filename.
    The format of lights.txt should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.txt
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.loadtxt(filename)
    return Lt.T


def load_lightnpy(filename=None):
    """
    Load light numpy array file specified by filename.
    The format of lights.npy should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.npy
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.load(filename)
    return Lt.T


def load_image(filename=None):
    """
    Load image specified by filename (read as a gray-scale)
    :param filename: filename of the image to be loaded
    :return img: loaded image
    """
    if filename is None:
        raise ValueError("filename is None")
    return cv2.imread(filename, 0)


def load_images(foldername=None, ext=None, scale=1.0):
    """
    Load images in the folder specified by the "foldername" that have extension "ext"
    :param foldername: foldername
    :param ext: file extension
    :param scale: scaling factor to be multiplied to the image pixel after grayscale conversion
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None or ext is None:
        raise ValueError("filename/ext is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*." + ext)):
        im = cv2.imread(fname).astype(np.float64)
        if im.ndim == 3:
            # Assuming that RGBA will not be an input
            #im = np.mean(im, axis=2)   # RGB -> Gray
            im = converter_npy_para_cinza(im) #importar essa funcao
            im = im * scale
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))

        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def load_npyimages(foldername=None, scale=1.0):
    """
    Load images in the folder specified by the "foldername" in the numpy format
    :param foldername: foldername
    :param scale: scaling factor to be multiplied to the image pixel after grayscale conversion
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None:
        raise ValueError("filename is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*.npy")):
        im = np.load(fname)
        #print(im.shape,im.min(),im.max())
        if im.ndim == 3:
            #im = np.mean(im, axis=2) 
            im = converter_npy_para_cinza(im) #importar essa funcao
            im = im * scale
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))

        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def disp_normalmap(normal=None, height=None, width=None, delay=0, name=None):
    """
    Visualize normal as a normal map
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :return: None
    """
    if normal is None:
        raise ValueError("Surface normal `normal` is None")
    N = np.reshape(normal, (height, width, 3))  # Reshape to image coordinates
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'
    cv2.imshow(name, N)
    cv2.waitKey(delay)
    cv2.destroyWindow(name)
    cv2.waitKey(1)    # to deal with frozen window...


def save_normalmap_as_npy(filename=None, normal=None, height=None, width=None):
    """
    Save surface normal array as a numpy array
    :param filename: filename of the normal array
    :param normal: surface normal array (height \times width \times 3)
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    N = np.reshape(normal, (height, width, 3))
    np.save(filename, N)


def load_normalmap_from_npy(filename=None):
    """
    Load surface normal array (which is a numpy array)
    :param filename: filename of the normal array
    :return: surface normal (numpy array) in formatted in (height, width, 3).
    """
    if filename is None:
        raise ValueError("filename is None")
    return np.load(filename)


def evaluate_angular_error(gtnormal=None, normal=None, background=None):
    if gtnormal is None or normal is None:
        raise ValueError("surface normal is not given")
    ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if background is not None:
        ae[background] = 0
    return ae




###################################################################################################
###################################################################################################



#def converter_npy_para_cinza(matriz):
#    """
#    Esta função recebe uma matriz em formato .npy e a converte para escala de cinza.
#    
#    Parâmetros:
#    matriz (numpy.ndarray): A matriz a ser convertida.
#    
#    Retorna:
#    numpy.ndarray: A matriz em escala de cinza.
#    """
#    return 0.3 * matriz[:, :, 0] + 0.59 * matriz[:, :, 1] + 0.11 * matriz[:, :, 2]
#



def converter_npy_para_cinza(matriz):
    """
    Converte uma matriz de imagem RGB ou BGR para escala de cinza, detectando automaticamente o formato.
    """
    if np.mean(matriz[:, :, 0]) > np.mean(matriz[:, :, 2]):  # Formato RGB
        r, g, b = matriz[:, :, 0], matriz[:, :, 1], matriz[:, :, 2]
    else:  # Formato BGR
        b, g, r = matriz[:, :, 0], matriz[:, :, 1], matriz[:, :, 2]

    resultado = 0.3 * r + 0.59 * g + 0.11 * b
    return resultado


def light_direction(A,B,C,A1,B1,D):

    """
    Calculate the direction of light based on given points.
    Parameters:
    A (tuple): Coordinates of point A (x, y).
    B (tuple): Coordinates of point B (x, y).
    C (tuple): Coordinates of point C (x, y).
    A1 (tuple): Coordinates of point A1 (x, y).
    B1 (tuple): Coordinates of point B1 (x, y).
    D (list): Coordinates of point D (x, y) or [-1, -1] if not defined.
    Returns:
    tuple: A tuple containing:
        - lx (float): x-component of the light direction.
        - ly (float): y-component of the light direction.
        - lz (float): z-component of the light direction.
        - R (float): Radius of the circle passing through points A, B, and C.
        - M (list): Midpoint coordinates of A and B.
    """

    M = [int((A[0]+ B[0])/2), int((A[1]+B[1])/2)]
    R = hypot(C[0] - M[0], C[1] - M[1])
    M1 = [(A1[0] + B1[0])/2, (A1[1] + B1[1])/2]

    D1 = [2*M1[0] - M[0], 2*M1[1] - M[1]]

    if D[0] == -1:
        D = D1
    else:
        E = hypot(D1[0] - D[0], D1[1] - D[1])
        print(E)

    H = hypot(D[0] - M[0], D[1] - M[1])
    tetha_zero = atan2(2*R, H)
    tetha = atan2(R+R/cos(tetha_zero), H)
    
    alpha = atan2(M[1] - D[1], M[0] - D[0])

    lx = cos(tetha) * cos(alpha)
    ly = cos(tetha) * sin(alpha)
    lz = sin(tetha)


    return lx, ly, lz, R, M



def criar_mascara_circulo_com_parametros(centro, raio, tamanho_imagem):
    """
    Esta função cria uma máscara binária com um círculo branco baseado no centro e raio fornecidos.
    
    Parâmetros:
    centro (tuple): As coordenadas (x, y) do centro do círculo.
    raio (int): O raio do círculo.
    tamanho_imagem (tuple): O tamanho da imagem (altura, largura).
    
    Retorna:
    numpy.ndarray: A máscara binária com o círculo branco.
    """
    # Criar uma máscara vazia
    mascara = np.zeros(tamanho_imagem, dtype=np.uint8)
    
    # Desenhar o círculo na máscara
    cv2.circle(mascara, centro, raio, 255 , thickness=cv2.FILLED)
    
    return mascara




def save_image_as_npy(image_array, file_path):
    """
    Save an image array to a .npy file.

    Parameters:
    image_array (numpy.ndarray): The image array to be saved.
    file_path (str): The path where the .npy file will be saved.
    """
    np.save(file_path, image_array)
    print(f"Image saved to {file_path}")


def calculate_gradient_consistency(gradient_map: np.ndarray) -> np.ndarray:
    """
    Calculate the rotational consistency R of the gradient map.

    Parameters:
    gradient_map (numpy.ndarray): An image with two channels representing the x and y components of the estimated gradient.

    Returns:
    numpy.ndarray: An image R where R[x, y] is the rotational consistency of the estimated gradient around (x, y).
    """
    nx, ny = gradient_map.shape[:2]
    consistency_map = np.zeros((nx, ny))

    for x in range(1, nx-1):
        for y in range(1, ny-1):
            DGxDy = (gradient_map[x, y+1, 0] - gradient_map[x, y-1, 0]) / 2
            DGyDx = (gradient_map[x+1, y, 1] - gradient_map[x-1, y, 1]) / 2
            consistency_map[x, y] = DGxDy - DGyDx

    return consistency_map

def convert_normal_map_to_gradient_map(normal_map):
    """
    Convert a normal map to a gradient map.

    Parameters:
    normal_map (numpy.ndarray): An image with three channels representing the x, y, and z components of the estimated normal.

    Returns:
    numpy.ndarray: An image with two channels representing the x and y components of the estimated gradient.
    """
    nx, ny = normal_map.shape[:2]
    gradient_map = np.zeros((nx, ny, 2))

    for x in range(nx):
        for y in range(ny):
            gradient_map[x, y, 0] = normal_map[x, y, 0] / normal_map[x, y, 2]
            gradient_map[x, y, 1] = normal_map[x, y, 1] / normal_map[x, y, 2]

    return gradient_map
