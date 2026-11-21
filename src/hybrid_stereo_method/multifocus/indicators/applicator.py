import logging

import cv2
import numpy as np

from hybrid_stereo_method.multifocus.indicators.fourier import calculate_fourier_focus_indicator
from hybrid_stereo_method.multifocus.indicators.laplacian import calculate_laplacian_focus_indicator
from hybrid_stereo_method.multifocus.indicators.wavelet import calculate_wavelet_focus_indicator
from hybrid_stereo_method.multifocus.utils import zero_borders


def focus_indicator(
    image_stack: np.ndarray,
    focus_indicator_type: str,
    laplacian_kernel_size=None,
    radius=None,
    square=False,
    smooth=False,
    spatial_median_filter=False,
    zero_border=False,
    mask=False,
    mask_img=None,
) -> np.ndarray:
    logging.debug(
        f"Calculating focus indicator ({focus_indicator_type}) shape: {image_stack.shape}, min_all: {np.min(image_stack)}, max_all: {np.max(image_stack)}"
    )

    focus_indicator_stack = []

    # Process each image individually
    for i, img in enumerate(image_stack):
        if focus_indicator_type == "fourier":
            focus_indicator = calculate_fourier_focus_indicator(img, radius)

        elif focus_indicator_type == "laplacian":
            focus_indicator = calculate_laplacian_focus_indicator(img, laplacian_kernel_size)

        elif focus_indicator_type == "wavelet":
            focus_indicator = calculate_wavelet_focus_indicator(img)

        if square:
            # Square the reconstructed image (enhances differences)
            focus_indicator = focus_indicator**2
        if smooth:
            # Apply smoothing kernel
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            focus_indicator = cv2.filter2D(focus_indicator, -1, kernel)
        if zero_border:
            # Zero out borders of the computed INDICATOR (removes edge
            # artifacts). Zeroing the input image instead would create a sharp
            # intensity step at the ring boundary - the strongest possible
            # high-frequency structure - giving those pixels spurious maximal
            # focus in every frame.
            focus_indicator = zero_borders(focus_indicator, 40)


        if spatial_median_filter:
            # apply spatial median filter using cv2
            focus_indicator = np.float32(focus_indicator)
            # median blur requires a bit specific formats or we can use generic scipy median filter
            # Since cv2 medianBlur natively supports float32
            focus_indicator = cv2.medianBlur(focus_indicator, 5)

        if mask:
            focus_indicator = focus_indicator * mask_img

        # print_img_statistics(f'{focus_indicator}: img_final {i}', focus_indicator)
        focus_indicator_stack.append(focus_indicator)

    # Convert to numpy array for vectorized operations
    focus_indicator_stack = np.array(focus_indicator_stack)

    assert np.max(focus_indicator_stack) >= 0, "Focus indicator max values should be non-negative."

    # Statistics before normalization
    min_val = np.min(focus_indicator_stack)
    max_val = np.max(focus_indicator_stack)
    percentile = np.percentile(focus_indicator_stack, 90)

    logging.debug(
        f"Focus indicator before normalization ({focus_indicator_type}) min_val: {min_val}, max_val: {max_val}, percentil_90: {percentile}."
    )

    # Remove outliers by clipping values. Note: Do NOT clip focus peaks since that destroys the focal curve
    # Removed p90 clipping to keep strict physical values on focus peak intact
    # Only clip strictly below zero: a positive floor (e.g. the 1st percentile)
    # flattens weak-but-real focal curves into constants and biases the
    # subpixel vertex of curves whose regression window touches the floor.
    focus_indicator_stack = np.clip(focus_indicator_stack, 0, np.inf)

    min_val = np.min(focus_indicator_stack)
    max_val = np.max(focus_indicator_stack)

    # Normalize to [0,1] range
    if min_val < 0:
        focus_indicator_stack = focus_indicator_stack - min_val
    if max_val > 0:
        focus_indicator_stack = focus_indicator_stack / max_val

    logging.debug(
        f"Focus indicator after normalization ({focus_indicator_type}) min_val: {np.min(focus_indicator_stack)}, max_val: {np.max(focus_indicator_stack)}"
    )

    return focus_indicator_stack
