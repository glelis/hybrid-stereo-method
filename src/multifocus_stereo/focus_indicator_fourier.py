import cv2
import numpy as np



def calculate_fourier_focus_indicator(image: np.ndarray, radius: float) -> np.ndarray:
    """
    Compute the focus indicator for a single image using high frequency
    Fourier coefficients.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Focus indicator image with unnormalized values
    """
    # Normalize to [0,1]
    image = image / 255.0
    
    # Apply 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    
    # Shift zero frequency to center
    f_centered = np.fft.fftshift(f_transform)
    
    # Create high-pass filter
    height, width = image.shape
    mask = create_gaussian_elliptical_mask(height, width, radius) #radius = 0.1
    
    
    # Apply mask in frequency domain
    f_filtered = f_centered * mask
    
    # Inverse shift and transform
    f_inverse = np.fft.ifftshift(f_filtered)
    focus_map = np.real(np.fft.ifft2(f_inverse))

    focus_map = np.abs(focus_map)
    

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
    g_y = np.exp(-(dy ** 2) / 2)
    g_x = np.exp(-(dx ** 2) / 2)
    
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