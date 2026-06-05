import cv2
import numpy as np


def calculate_fourier_focus_indicator(image: np.ndarray, radius: float) -> np.ndarray:
    """
    Compute the focus indicator for a single image as the magnitude of a
    local Gaussian high-pass filter (unsharp mask).

    Spatially equivalent, in the image interior, to the previous global-FFT
    formulation (FFT -> Gaussian elliptical high-pass mask -> IFFT) with the
    same `radius`: multiplying the spectrum by ``1 - exp(-f^2/(2*(radius*N)^2))``
    equals subtracting a Gaussian blur with ``sigma = 1/(2*pi*radius)`` pixels.
    The explicit local convolution removes the FFT circular wraparound, which
    leaked strong border responses to the opposite border (MF-09).

    Args:
        image: Input grayscale image
        radius: High-pass cutoff as a fraction of the spectrum
            (sigma = 1/(2*pi*radius) pixels; radius=0.1 -> sigma ~ 1.6 px)

    Returns:
        Focus indicator image with unnormalized values
    """
    # Normalize to [0,1]
    image = image / 255.0

    # Local Gaussian low-pass; reflect border avoids wraparound artifacts
    sigma = 1.0 / (2.0 * np.pi * radius)
    low_pass = cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )

    # High-pass magnitude = |image - low-pass| (unsharp mask)
    focus_map = np.abs(image - low_pass)

    return focus_map


def create_gaussian_elliptical_mask(height: int, width: int, radius: float) -> np.ndarray:
    """
    Create a Gaussian high-pass elliptical mask.

    Args:
        height: Number of rows in the mask
        width: Number of columns in the mask
        radius: Base radius as a fraction of dimensions

    Returns:
        Mask where center is 0 (low frequencies filtered) and edges approach 1 (high frequencies preserved)
    """
    radius_y = radius * height
    radius_x = radius * width

    center_y, center_x = height // 2, width // 2

    # Create coordinate grids - vectorized approach
    y, x = np.ogrid[:height, :width]

    # Calculate normalized distances
    dy = (y - center_y) / radius_y
    dx = (x - center_x) / radius_x

    # Create Gaussian mask components
    g_y = np.exp(-(dy**2) / 2)
    g_x = np.exp(-(dx**2) / 2)

    # Combine components and invert (1 - mask)
    mask = 1 - np.outer(g_y, g_x)

    return mask


def create_binary_elliptical_mask(height: int, width: int, radius: float) -> np.ndarray:
    """
    Create a binary elliptical mask of specified size and radius.

    Args:
        height: Number of rows in the mask
        width: Number of columns in the mask
        radius: Base radius as a fraction of dimensions

    Returns:
        Binary mask where inside ellipse is 0 and outside is 1
    """
    radius_y = radius * height
    radius_x = radius * width

    center_y, center_x = height // 2, width // 2

    # Create coordinate grids - vectorized approach
    y, x = np.ogrid[:height, :width]

    # Calculate normalized squared distances
    dy2 = ((y - center_y) / radius_y) ** 2
    dx2 = ((x - center_x) / radius_x) ** 2

    # Create mask: 1 outside ellipse, 0 inside
    mask = (dy2 + dx2 > 1).astype(np.uint8)

    return mask


def apply_weighted_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a weighted filter to an image.

    Args:
        image: Image to be filtered
        kernel: Filter kernel

    Returns:
        Filtered image
    """
    return cv2.filter2D(image, -1, kernel)
