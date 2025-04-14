from multifocus_stereo.focus_indicator_laplacian import calculate_laplacian_focus_indicator
from multifocus_stereo.focus_indicator_fourier import calculate_fourier_focus_indicator
from multifocus_stereo.focus_indicator_wavelet import calculate_wavelet_focus_indicator
from multifocus_stereo.utils import zero_borders
import cv2
import numpy as np
import logging



def focus_indicator(image_stack: np.ndarray, focus_indicator_type: str, laplacian_kernel_size=None, radius=None, square=False, smooth=False, zero_border=False, mask=False, mask_img=None) -> np.ndarray:

    logging.debug(f'Calculating focus indicator ({focus_indicator_type}) shape: {image_stack.shape}, min_all: {np.min(image_stack)}, max_all: {np.max(image_stack)}')
    
    focus_indicator_stack = []
    
    # Process each image individually
    for i, img in enumerate(image_stack):

        if zero_border:
            # Zero out borders (remove edge artifacts)
            img = zero_borders(img, 40)

        if focus_indicator_type == 'fourier':
            focus_indicator = calculate_fourier_focus_indicator(img, radius)

        elif focus_indicator_type =='laplacian':
            focus_indicator = calculate_laplacian_focus_indicator(img, laplacian_kernel_size)

        elif focus_indicator_type == 'wavelet':
            focus_indicator = calculate_wavelet_focus_indicator(img)

        if square:
            # Square the reconstructed image (enhances differences)
            focus_indicator = focus_indicator ** 2
        if smooth:
            # Apply smoothing kernel
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            focus_indicator = cv2.filter2D(focus_indicator, -1, kernel)
        #if zero_border:
        #    # Zero out borders (remove edge artifacts)
        #    focus_indicator = zero_borders(focus_indicator, 40)
        if mask:
            focus_indicator = focus_indicator * mask_img


        #print_img_statistics(f'{focus_indicator}: img_final {i}', focus_indicator)
        focus_indicator_stack.append(focus_indicator)
    
    # Convert to numpy array for vectorized operations
    focus_indicator_stack = np.array(focus_indicator_stack)
    
    assert np.max(focus_indicator_stack) >= 0, "Focus indicator max values should be non-negative."

    # Statistics before normalization
    min_val = np.min(focus_indicator_stack)
    max_val = np.max(focus_indicator_stack)
    percentile = np.percentile(focus_indicator_stack, 90)
    
    logging.debug(f'Focus indicator before normalization ({focus_indicator_type}) min_val: {min_val}, max_val: {max_val}, percentil_90: {percentile}.')

    
    # Remove outliers by clipping values
    p1, p90 = np.percentile(focus_indicator_stack, [1, 90])
    focus_indicator_stack = np.clip(focus_indicator_stack, p1, p90)

    min_val = np.min(focus_indicator_stack)
    max_val = np.max(focus_indicator_stack)
    
    # Normalize to [0,1] range
    if min_val < 0:
        focus_indicator_stack = focus_indicator_stack - min_val
    if max_val > 0:
        focus_indicator_stack = focus_indicator_stack / max_val


        
    
    logging.debug(f'Focus indicator after normalization ({focus_indicator_type}) min_val: {np.min(focus_indicator_stack)}, max_val: {np.max(focus_indicator_stack)}')
    
    
    return focus_indicator_stack