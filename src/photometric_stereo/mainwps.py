from __future__ import print_function  # Compatibility with Python 2 (redundant in modern Python versions)
import os
import time
import logging
from datetime import datetime

import numpy as np
from photometric_stereo.rps import RPS
from photometric_stereo.psutil import load_normalmap_from_npy, evaluate_angular_error
from photometric_stereo.psvizualization import disp_normalmap, disp_channels, disp_channels_3d
import argparse

from common.io import read_yaml_parameters, find_all_files, read_images, convert_image_array_to_fni, log_parameters, save_image
from photometric_stereo.wps import estimate_normals_argmax, estimate_normals_argmax_lstsq, estimate_normals_argmax_lstsq_robust
from common.utils import convert_to_grayscale


def main(parameters):

    if parameters.get('hybrid_method') == True:
        # Define paths
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        data_path = os.path.join(parameters.get('input_path'), parameters.get('data_foldername'))
        
        # Input files
        images_path = parameters.get('sMos_path_list')
        light_path = parameters.get('lights_path')
        mask_path = os.path.join(data_path, "mask.png")
        gt_normal_path = os.path.join(data_path, "gt_normal.npy")
        
        # Output files
        output_path = parameters.get('output_path_photometric')
        normal_map_path = os.path.join(output_path, "normal_map.npy")
        selected_areas_path = os.path.join(output_path, "selected_areas")

                # Ensure Output Directory Exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)


    
    else:
        # Define paths
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        data_path = os.path.join(parameters.get('input_path'), parameters.get('data_foldername'))
        
        # Input files
        images_path = os.path.join(data_path, "images")
        light_path = os.path.join(data_path, "lights.npy")
        mask_path = os.path.join(data_path, "mask.png")
        gt_normal_path = os.path.join(data_path, "gt_normal.npy")
        
        # Output files
        output_path = os.path.join(parameters.get('output_path'), f'{current_time}_{parameters.get("data_foldername")}')
        normal_map_path = os.path.join(output_path, "normal_map.npy")
        selected_areas_path = os.path.join(output_path, "selected_areas")
        


        # Ensure Output Directory Exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Logging Configuration
        logging.basicConfig(
            level=logging.INFO,  # Minimum log level
            format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler(os.path.join(output_path, f'depth_from_focus.log')),  # Output to file
            ],
        )


        # Log parameter information
        logging.info(f"Starting photometric stereo experiment")
        
        # Log all parameters recursively
        log_parameters(parameters)




    # Load images
    logging.info("... Loading images ...")

    if parameters.get('hybrid_method') == True:
        images = read_images(parameters.get('sMos_path_list'))
        #images = parameters.get('sMos_list')
    else:
        images_paths = find_all_files(images_path)
        images = read_images(images_paths, info=True)

    # Convert to grayscale
    images = [convert_to_grayscale(img) for img in images]

    # Load light sources
    logging.info("... Loading light sources ...")
    light_sources = np.load(light_path)

    # Load mask
    #mask = read_image(mask_path, info=True)


    # Estimate normals
    logging.info("... Calculating  normals ...")
    start_time = time.time()
    #normals, selected_areas = estimate_normals_argmax(images, light_sources)
    #normals, residuals, confidence, selected_areas = estimate_normals_argmax_lstsq(images, light_sources)
    normals, albedo, confidence, selected_areas = estimate_normals_argmax_lstsq_robust(images, light_sources)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Photometric stereo solved in {elapsed_time:.2f} seconds")

    # Save normal map
    logging.info("... Saving normal map ...")
    np.save(normal_map_path, normals)


    #convert_image_array_to_fni(normals, os.path.join(output_path, "normal_map.fni"))

    # Add residuals as a new channel to normals
    confidence_extra_exp = np.expand_dims(confidence, axis=-1)
    normals_with_confidence = np.concatenate((normals, confidence_extra_exp), axis=-1)
    convert_image_array_to_fni(normals_with_confidence, os.path.join(output_path, "normal_map_with_residuals.fni"))
    logging.info(f"Normal map saved at: {normal_map_path}")


    logging.info("... Saving images ...")
    
    logging.info("Saving Selected indicator images")
    for i in range(selected_areas.shape[-1]):
        focus_indicator_img = selected_areas[:, :, i]
        save_image(selected_areas_path, f"{i:03d}_selected_indicator.png", focus_indicator_img)


    # Evaluate the Result
    if os.path.exists(gt_normal_path):  # Check if ground truth normal map exists
        N_gt = np.load(filename=gt_normal_path)  # Load ground truth normal map
        angular_error = evaluate_angular_error(N_gt, normals, mask)  # Calculate angular error
        mean_error = np.mean(angular_error[:])
        logging.info(f"Mean angular error [degrees]: {mean_error:.2f}")

    
    # Results visualization
    height = normals.shape[0]
    width = normals.shape[1]
    disp_normalmap(normal=normals, height=height, width=width, save_path=output_path)
    disp_channels(normal_in=normals, height=height, width=width, delay=0, name='channels', save_path=output_path)
    disp_channels_3d(normal_in=normals, height=height, width=width, delay=0, name='channels_3D', save_path=output_path)

    # Finish Execution
    logging.info("Process completed successfully.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run photometric stereo method.')
    parser.add_argument('--param_file', type=str, required=True, 
                       help='Path to the YAML parameter file.')

    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)
    
    main(parameters)



