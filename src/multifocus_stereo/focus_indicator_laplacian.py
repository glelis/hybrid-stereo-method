import cv2
import numpy as np
from multifocus_stereo.utils import *


def focus_indicator_laplacian(image_stack: np.ndarray, laplacian_kernel_size: int):
    """
    Given a stack of overlap images, calculate the focus indicator for each image 
    using the square of the laplacian.

    Args:
        image_stack: A stack of overlap images, as an array[kf,kx,ky], gray images.
        laplacian_kernel_size: Size of the kernel used for Laplacian operator (should be odd).

    Returns:
        A stack of focus indicators images, as an array[kf,kx,ky], values: [0,1]
    """
    
    # Validate kernel size
    if laplacian_kernel_size % 2 == 0:
        raise ValueError("Laplacian kernel size must be odd")
    

    num_images, h, w = image_stack.shape
    
    fi_stacked = np.zeros((num_images, h, w), dtype=np.float64)

    for i, aligned_img in enumerate(image_stack):
        
        aligned_img = aligned_img / 255
        
        laplacian_img = cv2.Laplacian(aligned_img, cv2.CV_64F, ksize=laplacian_kernel_size)
        
        laplacian_img = zero_borders(laplacian_img, 2 * laplacian_kernel_size + 1)
        laplacian_img = laplacian_img ** 2
        #laplacian_img = abs(laplacian_img)
        fi_stacked[i] = laplacian_img

    min_val = np.min(fi_stacked)
    max_val = np.max(fi_stacked)
    print(f'focus indicator before normalization (laplacian) max_val: {max_val}, min_val: {min_val}')
    
    # Avoid division by zero
    if max_val > 0:
        fi_stacked = fi_stacked / max_val
    else:
        print("Warning: Maximum value is zero, skipping normalization")

    return fi_stacked


