import argparse
import logging
import os
from datetime import datetime

import numpy as np
from natsort import natsorted

from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from hybrid_stereo_method.infrastructure.io.image_io import (
    find_all_files,
    log_parameters,
    read_images,
    read_yaml_parameters,
    save_image,
)
from hybrid_stereo_method.infrastructure.utils import calculate_avarage_of_images
from hybrid_stereo_method.multifocus.main import main as multifocus_stereo_main
from hybrid_stereo_method.photometric.main_wps import main as photometric_stereo_main


def main(parameters):
    """
    Main function to execute the hybrid stereo method.
    
    This runs the complete pipeline:
    1. Multifocus stereo - depth from focus variation
    2. Photometric stereo - surface normals from lighting variation  
    3. Surface integration - height map from normals
    """

    # Define the current timestamp to name output folders and files
    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    # Define the output path
    base_output_path = parameters["experiment"]["paths"]["output"]
    data_foldername = parameters["experiment"]["paths"]["data_folder"]
    output_path = os.path.join(base_output_path, f"{current_time}_{data_foldername}")

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Configure logging to record information in the console and a file
    logging.basicConfig(
        level=logging.DEBUG,  # Minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(
                os.path.join(output_path, f"hybrid_stereo_{current_time}.log")
            ),  # File output
        ],
    )

    # Initial log for the experiment
    logging.info("Starting Hybrid stereo experiment")

    # Log the provided parameters
    log_parameters(parameters)

    # =========================================================================
    # Step 1: Multifocus Stereo
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 1: Multifocus Stereo")
    logging.info("=" * 60)
    
    # Find all files in the input directory
    input_path = parameters["experiment"]["paths"]["input"]
    input_files_path = find_all_files(os.path.join(input_path, data_foldername))

    # Process images to calculate the average for each 'zf' directory
    zf_directories = sorted(
        set(
            [
                os.path.basename(os.path.dirname(path))
                for path in input_files_path
                if os.path.basename(os.path.dirname(path)).startswith("zf")
            ]
        )
    )

    average_images_paths = []
    for zf_dir in zf_directories:
        # Filter relevant files in the directory
        filtered_files = sorted(
            [file for file in input_files_path if f"{zf_dir}" in file and "sVal.png" in file]
        )
        image_list = read_images(filtered_files, info=False)

        # Calculate the average of the images
        average_image = calculate_avarage_of_images(image_list)

        # Save the average image in the output directory
        average_image_path = os.path.join(output_path, "multifocus_stereo", "average", "images")
        save_image(average_image_path, f"average_{zf_dir}.png", average_image)
        average_images_paths.append(os.path.join(average_image_path, f"average_{zf_dir}.png"))

    # Update parameters with the paths of the average images and the output directory
    parameters["filtered_dir"] = average_images_paths
    parameters["output_path_multifocus"] = os.path.join(output_path, "multifocus_stereo", "average")

    # Execute the multifocus stereo method
    multifocus_stereo_main(parameters)

    # Process images for each light directory 'L'
    light_directories = sorted(
        set(
            [
                os.path.basename(os.path.dirname(path))
                for path in input_files_path
                if os.path.basename(os.path.dirname(path)).startswith("L")
            ]
        )
    )

    for light_dir in light_directories:
        logging.info(f"... Processing light directory: {light_dir} ...")

        # Filter relevant files in the directory
        filtered_files = sorted(
            [file for file in input_files_path if f"{light_dir}/zf" in file and "sVal.png" in file]
        )

        # Update parameters for the current directory
        parameters["filtered_dir"] = filtered_files
        parameters["output_path_multifocus"] = os.path.join(
            output_path, "multifocus_stereo", light_dir
        )

        # Execute the multifocus stereo method for the current directory
        multifocus_stereo_main(parameters)

    # =========================================================================
    # Step 2: Photometric Stereo
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 2: Photometric Stereo")
    logging.info("=" * 60)
    
    # Configure parameters for the photometric stereo method
    output_files = find_all_files(output_path)
    # Use the lossless sMos.npy files: the PNGs are min-max normalized per
    # image, which destroys the inter-light intensity ratios that the
    # photometric solver depends on.
    parameters["sMos_path_list"] = natsorted(
        [file for file in output_files if "sMos.npy" in file and "av" not in file]
    )
    parameters["output_path_photometric"] = os.path.join(output_path, "photometric_stereo")
    parameters["lights_path"] = [file for file in input_files_path if "lights.npy" in file][0]

    # Log the path of the lights file
    logging.info(f"Path to lights file: {parameters['lights_path']}")

    # Execute the photometric stereo method
    photometric_stereo_main(parameters)

    # =========================================================================
    # Step 3: Surface Integration
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 3: Surface Integration (Normal to Height)")
    logging.info("=" * 60)
    
    # Load the normal map from photometric stereo output
    normal_map_path = os.path.join(
        parameters["output_path_photometric"], "normal_map.npy"
    )
    
    if os.path.exists(normal_map_path):
        logging.info(f"Loading normal map from: {normal_map_path}")
        normal_map = np.load(normal_map_path)
        
        # Configure integration parameters
        integration_params = parameters.get("hybrid", {}).get("integration", {})
        
        integration_config = IntegrateRecursiveConfig(
            initial_method=integration_params.get("initial_method", "hints"),
            initial_noise=integration_params.get("initial_noise", 0.0),
            max_level=integration_params.get("max_level", 30),
            max_iter=integration_params.get("max_iter", 100000),
            conv_tol=integration_params.get("conv_tol", 0.0000005),
            verbose=parameters["experiment"]["settings"]["debug"],
            #report_step=integration_params.get("report_step", 1),
        )
        
        # Output directory for integration
        integration_output = os.path.join(output_path, "integration")
        
        hints_fni_path = None
        hints_weight = integration_params.get("hints_weight", 0.0)
        
        if integration_params.get("use_hints", False):
            # Locate zMos_with_confidence.fni inside multifocus_stereo/average
            hints_file = os.path.join(output_path, "multifocus_stereo", "average", "zMos_with_confidence.fni")
            if os.path.exists(hints_file):
                hints_fni_path = hints_file
                logging.info(f"Found hints map: {hints_fni_path}")
            else:
                logging.warning(f"Hints map requested but not found at: {hints_file}")
        
        logging.info("Running surface integration...")
        try:
            height_map = integrate_normals_to_height(
                normal_map=normal_map,
                output_dir=integration_output,
                output_prefix="height",
                config=integration_config,
                hints_fni_path=hints_fni_path,
                hints_weight=hints_weight,
            )
            
            # Save the height map as numpy array
            height_npy_path = os.path.join(integration_output, "height_map.npy")
            np.save(height_npy_path, height_map)
            logging.info(f"Height map saved to: {height_npy_path}")
            
            # Save as image for visualization
            save_image(integration_output, "height_map.png", height_map)
            logging.info(f"Height map visualization saved")
            
            logging.info(f"Height map shape: {height_map.shape}")
            logging.info(f"Height map range: [{height_map.min():.4f}, {height_map.max():.4f}]")
            
        except FileNotFoundError as e:
            logging.error(f"Integration executable not found: {e}")
            logging.error("Please build the C code: cd csrc/integrate_recursive && make")
        except Exception as e:
            logging.error(f"Integration failed: {e}")
    else:
        logging.warning(f"Normal map not found at: {normal_map_path}")
        logging.warning("Skipping surface integration step")

    logging.info("=" * 60)
    logging.info("Hybrid stereo pipeline complete!")
    logging.info(f"Results saved to: {output_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Executes the hybrid stereo method.")
    parser.add_argument(
        "--param_file", type=str, required=True, help="Path to the YAML parameter file."
    )

    # Read arguments provided by the user
    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)

    # Execute the main function
    main(parameters)

