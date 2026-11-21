"""Photometric Stereo main module."""

import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np

from hybrid_stereo_method.infrastructure.io.image_io import read_yaml_parameters
from hybrid_stereo_method.photometric.ps_utils import (
    evaluate_angular_error,
    load_normalmap_from_npy,
)
from hybrid_stereo_method.photometric.rps import RPS
from hybrid_stereo_method.photometric.visualization import (
    disp_channels,
    disp_channels_3d,
    disp_normalmap,
)


def main(parameters):
    # Define paths
    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    # Extract parameters from the nested YAML structure (see ps_experiment.yaml)
    paths = parameters["experiment"]["paths"]
    ps_params = parameters["photometric"]["parameters"]
    input_path = paths["input"]
    data_foldername = paths["data_folder"]
    base_output_path = paths["output"]
    data_scale = ps_params.get("data_scale", 1.0)
    image_extension = ps_params.get("image_extension", "png")
    method_name = ps_params["method"]
    debug = parameters["experiment"]["settings"].get("debug", False)

    data_path = os.path.join(input_path, data_foldername)

    # Input files
    images_path = os.path.join(data_path, "images/")
    light_path = os.path.join(data_path, "lights.npy")
    mask_path = os.path.join(data_path, "mask.png")
    gt_normal_path = os.path.join(data_path, "gt_normal.npy")

    # Output files
    output_path = os.path.join(base_output_path, f"{current_time}_{data_foldername}/")
    normal_map_path = os.path.join(output_path, "normal_map.npy")

    # Ensure Output Directory Exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Logging Configuration
    logging.basicConfig(
        level=logging.INFO,  # Minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(
                os.path.join(output_path, "depth_from_focus.log")
            ),  # Output to file
        ],
    )

    logging.info(
        f"Starting photometric stereo experiment with parameters:\n"
        f"INPUT_PATH: '{input_path}'\n"
        f"DATA_FOLDERNAME: '{data_foldername}'\n"
        f"OUTPUT_PATH: '{output_path}'\n"
        f"DATA_SCALE: {data_scale}\n"
        f"IMAGE_EXTENSION: {image_extension}\n"
        f"METHOD: {method_name}\n"
        f"DEBUG: {debug}\n"
    )

    # **Initialize the RPS Model**
    rps = RPS()

    # **Load Input Files**
    rps.load_mask(filename=mask_path)  # Load the mask
    rps.load_lightnpy(filename=light_path)  # Load the light source coordinates

    # Load images based on the specified type
    if image_extension == "npy":
        rps.load_npyimages(foldername=images_path, scale=data_scale)
    else:
        rps.load_images(
            foldername=images_path,
            ext=image_extension,
            scale=data_scale,
        )

    # **Select Solver Method**
    if method_name == "L2":
        method = RPS.L2_SOLVER
    elif method_name == "L1":
        method = RPS.L1_SOLVER_MULTICORE
    elif method_name == "SBL":
        method = RPS.SBL_SOLVER_MULTICORE
    elif method_name == "RPCA":
        method = RPS.RPCA_SOLVER
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    # **Run the Solver**
    start_time = time.time()
    rps.solve(method)  # Solve the photometric stereo problem
    elapsed_time = time.time() - start_time
    logging.info(f"Photometric stereo solved in {elapsed_time:.2f} seconds")

    # **Save the Normal Map**
    rps.save_normalmap(filename=normal_map_path)
    logging.info(f"Normal map saved at {normal_map_path}")

    # **Evaluate the Result**
    if os.path.exists(gt_normal_path):  # Check if ground truth normal map exists
        N_gt = load_normalmap_from_npy(filename=gt_normal_path)
        N_gt = np.reshape(N_gt, (rps.height * rps.width, 3))  # Reshape for evaluation
        angular_error = evaluate_angular_error(
            N_gt, rps.N, rps.background_ind
        )  # Calculate angular error
        mean_error = np.mean(angular_error[:])
        logging.info(f"Mean angular error [degrees]: {mean_error:.2f}")
        print(f"Mean angular error [degrees]: {mean_error:.2f}")

    # **Display the Normal Map**
    disp_normalmap(normal=rps.N, height=rps.height, width=rps.width, save_path=output_path)

    disp_channels(
        normal_in=rps.N,
        height=rps.height,
        width=rps.width,
        delay=0,
        name="channels",
        save_path=output_path,
    )

    disp_channels_3d(
        normal_in=rps.N,
        height=rps.height,
        width=rps.width,
        delay=0,
        name="channels_3D",
        save_path=output_path,
    )

    # **Finish Execution**
    logging.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run photometric stereo method.")
    parser.add_argument(
        "--param_file", type=str, required=True, help="Path to the YAML parameter file."
    )

    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)

    main(parameters)
