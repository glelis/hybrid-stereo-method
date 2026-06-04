import cv2
import numpy as np

# normalização deve ser feita no stack de imagens


def calculate_laplacian_focus_indicator(
    image: np.ndarray, laplacian_kernel_size: int
) -> np.ndarray:
    """
    Calculate focus indicator using Laplacian operator.

    The Laplacian operator is used to measure the second derivative of an image,
    which is sensitive to edges and textures. A high Laplacian response
    typically indicates areas that are in focus.

    Args:
        image: Input grayscale image
        laplacian_kernel_size: Size of the Laplacian kernel

    Returns:
        Normalized focus indicator map (8-bit)
    """
    # Normalize to [0,1]
    image = image / 255.0

    # Validate kernel size
    if laplacian_kernel_size % 2 == 0:
        raise ValueError("Laplacian kernel size must be odd")

    # Compute Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=laplacian_kernel_size)

    # Take absolute value to measure magnitude of edges
    # (normalização deve ser feita no stack de imagens)
    return np.abs(laplacian)
