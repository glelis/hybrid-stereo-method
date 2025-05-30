import cv2
import numpy as np

from common.utils import print_img_statistics
import os
from natsort import natsorted
import yaml
import logging


def read_yaml_parameters(yaml_file_path):
    """Read parameters from a YAML file.
    
    Args:
        yaml_file_path (str): Path to the YAML parameter file
        
    Returns:
        dict: Dictionary containing parameters
    """

    with open(yaml_file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    
    return parameters


def log_parameters(params:dict, prefix=''):
    """
    Recursively logs parameters from a nested dictionary or list structure.
    
    This function iterates through the provided parameter dictionary or list and logs each key-value pair.
    For nested dictionaries or lists, it recursively calls itself with an updated path prefix.
    
    Args:
        params (dict): The parameter dictionary to be logged
        prefix (str, optional): The prefix to prepend to parameter keys for hierarchical logging.
                               Defaults to an empty string.
    
    Example:
        >>> config = {'model': {'layers': 3, 'activation': 'relu'}, 'batch_size': 64}
        >>> log_parameters(config)
        # This will log:
        # Parameter - model.layers: 3
        # Parameter - model.activation: relu
        # Parameter - batch_size: 64
    """
    for key, value in params.items():
        current_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (dict, list)):
            log_parameters(value, current_path)
        else:
            logging.info(f"Parameter - {current_path}: {value}")


def find_all_files(path:str) -> list:
    """
    Recursively find all files in a directory path.
    
    Args:
        path (str): Directory path to search for files
        
    Returns:
        list: List of filenames found in the directory and its subdirectories
    
    Example:
        >>> files = find_all_files('/path/to/directory')
    """
    all_files = []
    
    # Walk through directory tree recursively
    for root, _, files in os.walk(path):
        # Add each file to the results list
        for file in natsorted(files):
            # Join root and filename to get full path
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    return all_files



def read_image(image_path: str, info=False) -> np.ndarray:
    """
    Read an image from the given path using OpenCV.
    Args:
        image_path (str): Path to the image file.
        info (bool, optional): If True, print statistics about the loaded image. Defaults to False.
    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    Raises:
        FileNotFoundError: If the image cannot be found at the specified path.
    Example:
        >>> img = read_image('path/to/image.png')
        >>> img_with_stats = read_image('path/to/image.png', info=True)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if info:
        print_img_statistics(os.path.basename(image_path), img)

    return img


def read_images(image_paths: list, info=False) -> list:
    """
    Read multiple images from the given paths using OpenCV.
    Args:
        image_paths (list): List of paths to the image files.
        info (bool, optional): If True, print statistics about the loaded images. Defaults to False.
    Returns:
        list: List of loaded images as NumPy arrays.
    Raises:
        FileNotFoundError: If any image cannot be found at the specified path.
    Example:
        >>> imgs = read_images(['path/to/image1.png', 'path/to/image2.png'])
        >>> imgs_with_stats = read_images(['path/to/image1.png', 'path/to/image2.png'], info=True)
    """
    images = [read_image(path, info) for path in image_paths]
    return images


def save_image(save_path: str, save_as: str, img: np.ndarray):
    """
    Save an image to the specified path after normalizing its pixel values.

    This function normalizes the image pixel values to the range [0, 255] using OpenCV's
    normalization method and saves the resulting image to the specified directory.

    Args:
        save_path (str): Directory where the image will be saved.
        save_as (str): Name of the saved image file.
        img (np.ndarray): Input image as a NumPy array.

    Raises:
        ValueError: If the input image is empty or invalid.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid.")

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Normalize the image to the range [0, 255]
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save the normalized image
    save_full_path = os.path.join(save_path, save_as)
    cv2.imwrite(save_full_path, img_norm)











def convert_image_array_to_fni(image_array: np.ndarray, output_file: str):
    """
    Convert a numpy array representing an image to FNI format.

    The FNI format is represented as a float_image_array_t file where each pixel's
    value is written with its coordinates.

    Parameters
    ----------
    image_array : np.array
        Input image array. Can be a 2D array (height, width) or 
        3D array (height, width, channels) for multi-channel images.
    output_file : str
        Path to the output FNI file to be created.

    Raises
    ------
    ValueError
        If the input array doesn't have the expected shape.
    """
    if len(image_array.shape) == 2:
        ny, nx = image_array.shape
        nc = 1
    elif len(image_array.shape) == 3:
        ny, nx, nc = image_array.shape
    else:
        raise ValueError("image_array must have shape (height, width) or (height, width, channels)")

    # Write output in float_image_array_t format
    with open(output_file, 'w') as f:
        # Write header metadata
        f.write("begin float_image_t (format of 2006-03-25)\n")
        f.write(f"NC = {nc}\n")
        f.write(f"NX = {nx}\n")
        f.write(f"NY = {ny}\n")
        
        # Write vector data for each pixel
        for y in range(ny):
            for x in range(nx):
                if nc == 1:
                    # For nc = 1, write single component per pixel
                    value = image_array[y, x]
                    f.write(f"{x:5d} {y:5d} {value:+.7e}\n")
                else:
                    # For nc > 1, write all components per pixel
                    values = image_array[y, x]
                    values_str = " ".join(f"{v:+.7e}" for v in values)
                    f.write(f"{x:5d} {y:5d} {values_str}\n")
            
            # Empty line after each row (part of format specification)
            f.write("\n")
        
        # Write footer
        f.write("end float_image_t\n")




def read_fni_to_image_array(fni_file: str) -> np.ndarray:
    """
    Read an FNI file and convert it to a NumPy image array.

    The FNI format is represented as a float_image_array_t file where each pixel's
    value is written with its coordinates.

    Parameters
    ----------
    fni_file : str
        Path to the FNI file to be read.

    Returns
    -------
    np.ndarray
        The reconstructed image as a NumPy array.

    Raises
    ------
    ValueError
        If the FNI file format is invalid.
    """
    with open(fni_file, 'r') as f:
        lines = f.readlines()

    # Validate header
    if not lines[0].startswith("begin float_image_t"):
        raise ValueError("Invalid FNI file format: Missing header")

    # Read metadata
    metadata = {}
    for line in lines[1:]:
        if line.strip() == "":
            continue
        if line.startswith("end float_image_t"):
            break
        if "=" in line:
            key, value = line.split("=")
            metadata[key.strip()] = int(value.strip())

    # Extract dimensions and number of channels
    ny = metadata.get("NY")
    nx = metadata.get("NX")
    nc = metadata.get("NC")

    if ny is None or nx is None or nc is None:
        raise ValueError("Invalid FNI file format: Missing metadata")

    # Initialize the image array
    if nc < 1:
        raise ValueError("Invalid number of channels (NC). NC must be >= 1.")
    image_array = np.zeros((ny, nx, nc), dtype=np.float32) if nc > 1 else np.zeros((ny, nx), dtype=np.float32)

    # Read pixel data
    for line in lines:
        if line.strip() == "" or line.startswith("begin") or line.startswith("end"):
            continue
        parts = line.split()
        if len(parts) < 2 + nc:
            continue
        x = int(parts[0])
        y = int(parts[1])
        values = list(map(float, parts[2:]))
        if len(values) != nc:
            raise ValueError(f"Invalid data for NC={nc} at pixel ({x}, {y}): {values}")
        if nc == 1:
            image_array[y, x] = values[0]
        else:
            image_array[y, x, :] = values

    return image_array
