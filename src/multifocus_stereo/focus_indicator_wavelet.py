import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

def calculate_wavelet_focus_indicator(image, wavelet='haar', level=2):

    image = image/255
    # Apply 2D Discrete Wavelet Transform (DWT)
    # This decomposes the image into approximation and detail coefficients
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    
    # Extract the approximation coefficients (cA) and detail coefficients (cH, cV, cD)
    # cA: Approximation (low-frequency component)
    # cH: Horizontal details (high frequency in horizontal direction)
    # cV: Vertical details (high frequency in vertical direction)
    # cD: Diagonal details (high frequency in diagonal direction)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]

    # Calcular a magnitude das regiões wavelet de alta frequência
    high_freq_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
    
    
    return high_freq_magnitude


