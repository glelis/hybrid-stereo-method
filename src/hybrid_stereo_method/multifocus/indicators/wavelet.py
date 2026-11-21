import cv2
import numpy as np
import pywt


def calculate_wavelet_focus_indicator(image, wavelet="haar", level=2):
    image = image / 255
    # Apply 2D Discrete Wavelet Transform (DWT)
    # This decomposes the image into approximation and detail coefficients
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # pywt.wavedec2 orders the detail bands from the COARSEST to the FINEST
    # level: coeffs = [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)].
    # The finest level (coeffs[-1]) carries the high-frequency content that
    # characterizes focus.
    # cH: Horizontal details (high frequency in horizontal direction)
    # cV: Vertical details (high frequency in vertical direction)
    # cD: Diagonal details (high frequency in diagonal direction)
    cH, cV, cD = coeffs[-1]

    # Calcular a magnitude das regiões wavelet de alta frequência
    high_freq_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)

    # The level-1 coefficients have roughly half the image resolution; resize
    # back to the input grid so the focus map is pixel-aligned with the stack
    # consumed by argmax_fuzzy/mosaic.
    height, width = image.shape[:2]
    return cv2.resize(high_freq_magnitude, (width, height), interpolation=cv2.INTER_LINEAR)
