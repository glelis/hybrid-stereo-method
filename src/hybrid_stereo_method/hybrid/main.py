import argparse
import logging
import os
from datetime import datetime

from natsort import natsorted

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
    """

    # Define the current timestamp to name output folders and files
    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    # Define the output path
    output_path = os.path.join(
        parameters.get("output_path"), f"{current_time}_{parameters.get('data_foldername')}"
    )

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

    logging.info("... Multifocus stereo ...")
    # Find all files in the input directory
    input_files_path = find_all_files(
        os.path.join(parameters.get("input_path"), parameters.get("data_foldername"))
    )

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

    logging.info("... Photometric Stereo ...")
    # Configure parameters for the photometric stereo method
    output_files = find_all_files(output_path)
    parameters["sMos_path_list"] = natsorted(
        [file for file in output_files if "sMos.png" in file and "av" not in file]
    )
    parameters["output_path_photometric"] = os.path.join(output_path, "photometric_stereo")
    parameters["lights_path"] = [file for file in input_files_path if "lights.npy" in file][0]

    # Log the path of the lights file
    logging.info(f"Path to lights file: {parameters['lights_path']}")

    # Execute the photometric stereo method
    photometric_stereo_main(parameters)


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
