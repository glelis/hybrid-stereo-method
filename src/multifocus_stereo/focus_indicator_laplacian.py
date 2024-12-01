import cv2
import numpy as np
from multifocus_stereo.utils import *


def focus_indicator_laplacian(aligned_img_stacked:np.array, laplacian_kernel_size:int):
    """
    Guive a stack overlap images, and a kernel_size calculate the focus indicator for each image, using the square of the laplacian.

    Args:
        aligned_img_stacked: A stack of overlap images, as an array[kf,kx,ky], imagages maybe color or gray.
    

    Returns:
        A stack a focus indicators images,as an array[kf,kx,ky], values: [0,1]
    """
    
    fi_stacked = []

    for i, aligned_img in enumerate(aligned_img_stacked):
        img_gray = convert_img_to_grayscale(aligned_img)
        img_gray = img_gray/255

        laplacian_img = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=laplacian_kernel_size)

        
        #laplacian_img = zero_borders(laplacian_img, 2*laplacian_kernel_size+1)
        laplacian_img = laplacian_img **2
        laplacian_img = abs(laplacian_img)
        fi_stacked.append(laplacian_img)

    fi_stacked = np.array(fi_stacked)
    min_val = np.min(fi_stacked)
    max_val = np.max(fi_stacked)
    print(f'focus indicator before normalization (laplacian) max_val: {max_val}, min_val: {min_val}')
    fi_stacked = fi_stacked/ max_val

    return fi_stacked


