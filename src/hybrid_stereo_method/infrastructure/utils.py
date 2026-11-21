"""Infrastructure utilities for the hybrid stereo method.

This module provides common utility functions used across the project.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

# Focal-plane folder names carry the focal distance: "zf045.0000-df020.0000" → 45.0.
# Also matches the averaged frames written by the hybrid pipeline
# ("average_zf045.0000-df020.0000.png").
ZF_DIR_RE = re.compile(r"(?:^|_)zf(\d+(?:\.\d+)?)")


def print_img_statistics(name: str, img: np.ndarray) -> None:
    """Print statistical information about an image.

    Args:
        name: Name identifier for the image.
        img: The image array for which statistics will be calculated.
    """
    shape = img.shape
    # Compute in float64: squaring uint8/uint16 wraps around modulo 2**bits
    # and silently corrupts the rms/deviation statistics.
    imgf = img.astype(np.float64)
    v_max = np.max(img)
    v_min = np.min(img)
    v_mean = np.average(imgf)
    rms = np.sqrt(np.average(imgf**2))
    v_dev = np.sqrt(np.average((imgf - v_mean) ** 2))
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

    # Round before casting: a bare astype truncates and introduces a
    # systematic -0.5 LSB bias in the averaged images.
    if images[0].dtype == np.uint8:
        return np.round(mean_image).astype(np.uint8)
    elif images[0].dtype == np.uint16:
        return np.round(mean_image).astype(np.uint16)
    else:
        raise ValueError("Unsupported data type.")


def parse_zf_value(path_or_name: str) -> float | None:
    """Focal distance encoded in a ``zf*`` directory name, or None.

    Accepts a bare directory name (``"zf045.0000-df020.0000"``) or any file
    path; for a path, the nearest component (file or ancestor directory)
    carrying a ``zf<value>`` token is used.
    """
    parts = os.path.normpath(path_or_name).split(os.sep)
    for part in reversed(parts):
        m = ZF_DIR_RE.search(part)
        if m:
            return float(m.group(1))
    return None


def resolve_z_foc(
    configured: Sequence[float] | str | None,
    paths: Iterable[str],
    rtol: float = 1e-3,
) -> list[float]:
    """Resolve the ``multifocus.parameters.z_foc`` list against the dataset.

    ``paths`` are the stack frames (or their ``zf*`` folders) in stack order.
    - ``configured`` is None or ``"auto"`` → the values parsed from the folder
      names are returned.
    - ``configured`` is a list and the folder names carry z values → both must
      agree (same length, values within ``rtol``); otherwise ``ValueError``,
      because a z_foc list from another dataset silently puts zMos — and the
      absolute integration hints — in the wrong units.
    - folder names carry no z (e.g. flat ``images/`` dir) → the configured list
      is returned unchanged; ``"auto"`` then raises ``ValueError``.
    """
    paths = list(paths)
    parsed = [parse_zf_value(p) for p in paths]
    have_parsed = bool(parsed) and all(z is not None for z in parsed)

    if configured is None or (isinstance(configured, str) and configured.lower() == "auto"):
        if not have_parsed:
            raise ValueError(
                "z_foc: 'auto' requires the frames to live in zf<value>* folders; "
                "set an explicit z_foc list for this dataset."
            )
        return [float(z) for z in parsed]

    if isinstance(configured, str):
        raise ValueError(f"z_foc must be a list of numbers or 'auto', got {configured!r}")

    configured = [float(z) for z in configured]
    if have_parsed:
        if len(configured) != len(parsed) or not np.allclose(configured, parsed, rtol=rtol):
            raise ValueError(
                f"z_foc from the config {configured} does not match the zf* folders of "
                f"the dataset {parsed}. Fix the list or use z_foc: auto."
            )
    return configured
