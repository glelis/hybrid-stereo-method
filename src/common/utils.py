import numpy as np
import cv2

def print_img_statistics(nome, img):
    """
    Print statistical information about an image.

    This function calculates and prints various statistical metrics for an input image,
    including shape, minimum value, maximum value, mean, root mean square (RMS), and
    standard deviation.

    Parameters
    ----------
    nome : str
        Name identifier for the image to be printed in the output
    img : numpy.ndarray
        The image array for which statistics will be calculated

    Returns
    -------
    None
        This function prints the statistics to standard output but does not return a value

    Examples
    --------
    >>> import numpy as np
    >>> test_img = np.array([[1, 2], [3, 4]])
    >>> print_img_statistics("test", test_img)
    nome:test, shape:(2, 2), min:1.000000, max:4.000000, mean:2.500000, rms:2.738613, v_dev:1.118034
    """
    shape = img.shape
    v_max = np.max(img)
    v_min = np.min(img)
    v_mean = np.average(img)
    rms = np.sqrt(np.average(img**2))
    v_dev = np.sqrt(np.average((img-v_mean)**2))
    print(f'nome:{nome}, shape:{shape}, min:{v_min:.6f}, max:{v_max:.6f}, mean:{v_mean:.6f}, rms:{rms:.6f}, v_dev:{v_dev:.6f}')



def normalize_normals(normal_map:np.ndarray) -> np.ndarray:
    """
    Normalize a normal map so that each normal vector has unit length.
    
    Parameters
    ----------
    normal_map : numpy.ndarray
        The normal map to be normalized. Expected shape is (H, W, 3) where
        the last dimension represents the XYZ components of the normal vectors.
        
    Returns
    -------
    numpy.ndarray
        Normalized normal map with the same shape as the input.
        
    Examples
    --------
    >>> import numpy as np
    >>> normals = np.array([[[1, 1, 1], [2, 0, 0]], [[0, 2, 0], [0, 0, 2]]])
    >>> normalized = normalize_normals(normals)
    >>> print(np.allclose(np.linalg.norm(normalized, axis=2), 1.0))
    True
    """
    # Check if the input is a valid normal map
    if normal_map.shape[2] != 3:
        raise ValueError("Normal map must have shape (H, W, 3)")
    
    # Calculate the magnitude of each normal vector
    norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    
    # Avoid division by zero
    norm = np.maximum(norm, 1e-10)
    
    # Normalize the normal vectors
    normalized_map = normal_map / norm
    
    return normalized_map



def convert_to_grayscale(img:np.ndarray) -> np.ndarray:
    """
    Convert an RGB/BGR image to grayscale.
    
    Parameters
    ----------
    img : numpy.ndarray
        The input color image in BGR format (as used by OpenCV)
        
    Returns
    -------
    numpy.ndarray
        Grayscale version of the input image
        
    Examples
    --------
    >>> import cv2
    >>> import numpy as np
    >>> color_img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    >>> gray_img = convert_to_grayscale(color_img)
    >>> gray_img.shape == (10, 10)
    True
    """
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    # Convert the BGR image to grayscale
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imGray





def calculate_avarage_of_images(imagens):
    """
    Calculate the average of a list of images, ensuring all images have the same dimensions and data type.

    Parameters:
    imagens (list of numpy.ndarray): List of images to be averaged. All images must have the same dimensions.

    Returns:
    numpy.ndarray: The averaged image with the same data type as the input images.

    Raises:
    ValueError: If the images do not have the same dimensions or if the data type is not supported.
    """
    # Ensure all images have the same dimensions
    if not all(imagem.shape == imagens[0].shape for imagem in imagens):
        raise ValueError("All images must have the same dimensions.")
    
    # Convert all images to float32 to avoid overflow issues during averaging
    imagens_float = [imagem.astype(np.float32) for imagem in imagens]
    
    # Calculate the average of the images
    media = np.mean(imagens_float, axis=0)
    
    # Convert the average back to the original data type of the images
    if imagens[0].dtype == np.uint8:
        media = media.astype(np.uint8)
    elif imagens[0].dtype == np.uint16:
        media = media.astype(np.uint16)
    else:
        raise ValueError("Unsupported data type.")
    
    return media
