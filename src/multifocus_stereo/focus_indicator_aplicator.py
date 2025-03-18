from multifocus_stereo.focus_indicator_laplacian_2 import calculate_laplacian_focus_indicator
from multifocus_stereo.focus_indicator_fourier_2 import calculate_fourier_focus_indicator
from multifocus_stereo.focus_indicator_wavelet import calculate_wavelet_focus_indicator
from multifocus_stereo.utils import zero_borders
import cv2
import numpy as np
import logging



def focus_indicator(aligned_img_stack: np.ndarray, focus_indicator: str, laplacian_kernel_size=None, radius=None, square=False, smooth=False, zero_border=False) -> np.ndarray:

    logging.debug(f'Calculating focus indicator ({focus_indicator}) shape: {aligned_img_stack.shape}, min_all: {np.min(aligned_img_stack)}, max_all: {np.max(aligned_img_stack)}')
    
    fi_stack = []
    
    # Process each image individually
    for i, img in enumerate(aligned_img_stack):

        if focus_indicator == 'fourier':
            focus_map = calculate_fourier_focus_indicator(img, radius)

        elif focus_indicator =='laplacian':
            focus_map = calculate_laplacian_focus_indicator(img, laplacian_kernel_size)

        elif focus_indicator == 'wavelet':
            focus_map = calculate_wavelet_focus_indicator(img)

        if square:
            # Square the reconstructed image (enhances differences)
            focus_map = focus_map ** 2
        if smooth:
            # Apply smoothing kernel
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            focus_map = cv2.filter2D(focus_map, -1, kernel)
        if zero_border:
            # Zero out borders (remove edge artifacts)
            focus_map = zero_borders(focus_map, 10)


        #print_img_statistics(f'{focus_indicator}: img_final {i}', focus_map)
        fi_stack.append(focus_map)
    
    # Convert to numpy array for vectorized operations
    fi_stack = np.array(fi_stack)
    
    # Statistics before normalization
    min_val = np.min(fi_stack)
    max_val = np.max(fi_stack)
    
    logging.debug(f'Focus indicator before normalization ({focus_indicator}) min_val: {min_val}, max_val: {max_val}')

    
    # Remove outliers by clipping values
    #p1, p99 = np.percentile(fi_stack, [1, 99])
    #fi_stack = np.clip(fi_stack, p1, p99)
    
    # Normalize to [0,1] range
    if min_val < 0:
        fi_stack = fi_stack - min_val
    max_val = np.max(fi_stack)
    if max_val > 0:
        fi_stack = fi_stack / max_val
    
    
    logging.debug(f'Focus indicator after normalization ({focus_indicator}) min_val: {np.min(fi_stack)}, max_val: {np.max(fi_stack)}')
    
    
    return fi_stack