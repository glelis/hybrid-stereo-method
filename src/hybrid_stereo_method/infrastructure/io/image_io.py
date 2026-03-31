"""Image I/O utilities for the hybrid stereo method.

This module provides functions for reading and writing images,
configuration files, and FNI format data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from natsort import natsorted

from hybrid_stereo_method.infrastructure.utils import print_img_statistics


def read_yaml_parameters(yaml_file_path: str | Path) -> dict[str, Any]:
    """Read parameters from a YAML file.

    Args:
        yaml_file_path: Path to the YAML parameter file.

    Returns:
        Dictionary containing parameters.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
    """
    with open(yaml_file_path) as file:
        parameters: dict[str, Any] = yaml.safe_load(file)
    return parameters


def log_parameters(params: dict[str, Any], prefix: str = "") -> None:
    """Recursively log parameters from a nested dictionary.

    Args:
        params: The parameter dictionary to be logged.
        prefix: Prefix to prepend to parameter keys for hierarchical logging.

    Example:
        >>> config = {'model': {'layers': 3}, 'batch_size': 64}
        >>> log_parameters(config)
        # Logs: Parameter - model.layers: 3
        # Logs: Parameter - batch_size: 64
    """
    for key, value in params.items():
        current_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_parameters(value, current_path)
        else:
            logging.info(f"Parameter - {current_path}: {value}")


def find_all_files(path: str | Path) -> list[str]:
    """Recursively find all files in a directory path.

    Args:
        path: Directory path to search for files.

    Returns:
        List of full file paths found in the directory and subdirectories.
    """
    all_files: list[str] = []
    for root, _, files in os.walk(path):
        for file in natsorted(files):
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def read_image(image_path: str | Path, info: bool = False) -> np.ndarray:
    """Read an image from the given path using OpenCV.

    Args:
        image_path: Path to the image file.
        info: If True, print statistics about the loaded image.

    Returns:
        The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be found at the specified path.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if info:
        print_img_statistics(os.path.basename(str(image_path)), img)
    return img


def read_images(image_paths: list[str], info: bool = False) -> list[np.ndarray]:
    """Read multiple images from the given paths.

    Args:
        image_paths: List of paths to the image files.
        info: If True, print statistics about the loaded images.

    Returns:
        List of loaded images as NumPy arrays.

    Raises:
        FileNotFoundError: If any image cannot be found.
    """
    return [read_image(path, info) for path in image_paths]


def save_image(save_path: str | Path, save_as: str, img: np.ndarray) -> None:
    """Save an image to the specified path after normalizing pixel values.

    Args:
        save_path: Directory where the image will be saved.
        save_as: Name of the saved image file.
        img: Input image as a NumPy array.

    Raises:
        ValueError: If the input image is empty or invalid.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or invalid.")

    os.makedirs(save_path, exist_ok=True)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_full_path = os.path.join(str(save_path), save_as)
    cv2.imwrite(save_full_path, img_norm)


def convert_image_array_to_fni(image_array: np.ndarray, output_file: str | Path) -> None:
    """Convert a numpy array to FNI (Float Image) format.

    The FNI format stores floating-point image data with coordinates.

    Args:
        image_array: Input image array (2D or 3D with channels).
        output_file: Path to the output FNI file.

    Raises:
        ValueError: If the input array has invalid shape.
    """
    if len(image_array.shape) == 2:
        ny, nx = image_array.shape
        nc = 1
    elif len(image_array.shape) == 3:
        ny, nx, nc = image_array.shape
    else:
        raise ValueError("image_array must have shape (height, width) or (height, width, channels)")

    with open(output_file, "w") as f:
        f.write("begin float_image_t (format of 2006-03-25)\n")
        f.write(f"NC = {nc}\n")
        f.write(f"NX = {nx}\n")
        f.write(f"NY = {ny}\n")

        for y in range(ny):
            for x in range(nx):
                if nc == 1:
                    value = image_array[y, x]
                    f.write(f"{x:5d} {y:5d} {value:+.7e}\n")
                else:
                    values = image_array[y, x]
                    values_str = " ".join(f"{v:+.7e}" for v in values)
                    f.write(f"{x:5d} {y:5d} {values_str}\n")
            f.write("\n")
        f.write("end float_image_t\n")


def read_fni_to_image_array(fni_file: str | Path) -> np.ndarray:
    """Read an FNI file and convert it to a NumPy image array.

    Args:
        fni_file: Path to the FNI file.

    Returns:
        The reconstructed image as a NumPy array.

    Raises:
        ValueError: If the FNI file format is invalid.
    """
    with open(fni_file) as f:
        lines = f.readlines()

    if not lines[0].startswith("begin float_image_t"):
        raise ValueError("Invalid FNI file format: Missing header")

    metadata: dict[str, int] = {}
    for line in lines[1:]:
        if line.strip() == "":
            continue
        if line.startswith("end float_image_t"):
            break
        if "=" in line:
            key, value = line.split("=")
            metadata[key.strip()] = int(value.strip())

    ny = metadata.get("NY")
    nx = metadata.get("NX")
    nc = metadata.get("NC")

    if ny is None or nx is None or nc is None:
        raise ValueError("Invalid FNI file format: Missing metadata")

    if nc < 1:
        raise ValueError("Invalid number of channels (NC). NC must be >= 1.")

    if nc > 1:
        image_array = np.zeros((ny, nx, nc), dtype=np.float32)
    else:
        image_array = np.zeros((ny, nx), dtype=np.float32)

    for line in lines:
        if line.strip() == "" or line.startswith("begin") or line.startswith("end") or "=" in line:
            continue
        parts = line.split()
        if len(parts) < 2 + nc:
            continue
        x = int(parts[0])
        y = int(parts[1])
        values = list(map(float, parts[2:]))
        if len(values) != nc:
            raise ValueError(f"Invalid data for NC={nc} at pixel ({x}, {y}): {values}")
        if nc == 1:
            image_array[y, x] = values[0]
        else:
            image_array[y, x, :] = values

    return image_array
