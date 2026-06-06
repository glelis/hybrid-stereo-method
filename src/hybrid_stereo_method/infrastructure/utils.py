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
    if max_ == min_:
        # Constant array: no range to normalize — return zeros instead of 0/0 nan
        return np.zeros_like(x, dtype=np.float64)
    return (x - min_) / (max_ - min_)


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR image to grayscale (Rec.601 weights, BGR order).

    Already-monochrome inputs (2-D or HxWx1) pass through without color
    conversion (PS-09 — cvtColor raises on single-channel input); the
    float64→float32 cast applies uniformly to ALL paths, including the mono
    passthrough. Note: the legacy ``rps`` path
    (``ps_utils.converter_npy_para_cinza``) uses a divergent RGB-vs-BGR
    heuristic; this function is the canonical policy for the hybrid pipeline.

    Args:
        img: Input image. BGR 3-channel, BGR 4-channel, or already mono (2-D or
            HxWx1). float64 inputs are cast to float32 before conversion.

    Returns:
        Grayscale image with shape (H, W).
    """
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calculate_avarage_of_images(images: list[np.ndarray]) -> np.ndarray:
    """Calculate the pixel-wise average of a list of images.

    Returns float32 regardless of input dtype, preserving sub-integer precision
    that would be lost by quantizing back to uint8/uint16.  Callers that need a
    uint8 image for display should quantize explicitly (e.g. `arr.astype(np.uint8)`
    or via ``save_image``).

    Args:
        images: List of images to average. All must have the same dimensions.
            Supported input dtypes: uint8, uint16, or any floating-point type.

    Returns:
        The averaged image as float32.

    Raises:
        ValueError: If images have different dimensions.
    """
    if not all(image.shape == images[0].shape for image in images):
        raise ValueError("All images must have the same dimensions.")

    images_float = [image.astype(np.float32) for image in images]
    return np.mean(images_float, axis=0)
