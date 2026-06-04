"""Infrastructure utilities for the hybrid stereo method.

This module provides common utility functions used across the project.
"""

from __future__ import annotations

import cv2
import numpy as np


def print_img_statistics(name: str, img: np.ndarray) -> None:
    """Print statistical information about an image.

    Args:
        name: Name identifier for the image.
        img: The image array for which statistics will be calculated.
    """
    shape = img.shape
    v_max = np.max(img)
    v_min = np.min(img)
    v_mean = np.average(img)
    rms = np.sqrt(np.average(img**2))
    v_dev = np.sqrt(np.average((img - v_mean) ** 2))
    print(
        f"nome:{name}, shape:{shape}, min:{v_min:.6f}, max:{v_max:.6f}, "
        f"mean:{v_mean:.6f}, rms:{rms:.6f}, v_dev:{v_dev:.6f}"
    )


def normalize_normals(normal_map: np.ndarray) -> np.ndarray:
    """Normalize a normal map so that each normal vector has unit length.

    Args:
        normal_map: Normal map with shape (H, W, 3).

    Returns:
        Normalized normal map with the same shape as the input.

    Raises:
        ValueError: If the normal map doesn't have 3 channels.
    """
    if normal_map.shape[2] != 3:
        raise ValueError("Normal map must have shape (H, W, 3)")

    norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    norm = np.maximum(norm, 1e-10)  # Avoid division by zero
    return normal_map / norm


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalizes the input array `x` to a range between 0 and 1.

    Parameters:
    x (numpy.ndarray): The input array to be normalized.

    Returns:
    numpy.ndarray: The normalized array with values scaled to the range [0, 1].

    Example:
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> normalize(x)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    max_, min_ = np.max(x), np.min(x)
    return (x - min_) / (max_ - min_)


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR image to grayscale.

    Args:
        img: Input color image in BGR format.

    Returns:
        Grayscale version of the input image.
    """
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calculate_avarage_of_images(images: list[np.ndarray]) -> np.ndarray:
    """Calculate the pixel-wise average of a list of images.

    Args:
        images: List of images to average. All must have the same dimensions.

    Returns:
        The averaged image with the same dtype as input images.

    Raises:
        ValueError: If images have different dimensions or unsupported dtype.
    """
    if not all(image.shape == images[0].shape for image in images):
        raise ValueError("All images must have the same dimensions.")

    images_float = [image.astype(np.float32) for image in images]
    mean_image = np.mean(images_float, axis=0)

    if images[0].dtype == np.uint8:
        return mean_image.astype(np.uint8)
    elif images[0].dtype == np.uint16:
        return mean_image.astype(np.uint16)
    else:
        raise ValueError("Unsupported data type.")
