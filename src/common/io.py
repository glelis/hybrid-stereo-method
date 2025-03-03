import cv2
import numpy as np

from common.utils import print_img_statistics


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
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if info:
        print_img_statistics("Loaded Image", img)
        
    return img





def convert_image_array_to_fni(image_array: np.ndarray, output_file: str):
    """
    Convert a numpy array representing an image to FNI format.

    The FNI format is represented as a float_image_array_t file where each pixel's
    value is written with its coordinates.

    Parameters
    ----------
    image_array : np.array
        Input image array. Can be a 2D array (height, width) or 
        3D array (height, width, 3) for color images.
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
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        ny, nx, nc = image_array.shape
    else:
        raise ValueError("image_array must have shape (height, width) or (height, width, channels=3)")

    # Write output in float_image_array_t format
    with open(output_file, 'w') as f:
        # Write header metadata
        f.write("begin float_image_array_t (format of 2006-03-25)\n")
        f.write(f"NC = {nc}\n")
        f.write(f"NX = {nx}\n")
        f.write(f"NY = {ny}\n")
        
        # Write vector data for each pixel
        for y in range(ny):
            for x in range(nx):
                if len(image_array.shape) == 2:
                    # For nc =1, write single component per pixel
                    value = image_array[y, x]
                    f.write(f"{x:5d} {y:5d} {value:+.7e}\n")
                else:
                    # For nc=3, write three components per pixel
                    v = image_array[y, x]
                    f.write(f"{x:5d} {y:5d} {v[0]:+.7e} {v[1]:+.7e} {v[2]:+.7e}\n")
            
            # Empty line after each row (part of format specification)
            f.write("\n")
        
        # Write footer
        f.write("end float_image_array_t\n")