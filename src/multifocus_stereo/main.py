import os
import numpy as np
from multifocus_stereo.utils import read_images_from_path, save_image, calculate_error_image, normalize

from common.io import read_yaml_parameters, find_all_files, read_images, convert_image_array_to_fni
from common.utils import print_img_statistics, convert_to_grayscale

from multifocus_stereo.depth_from_focus import all_in_focus
from multifocus_stereo.focus_indicator_laplacian import focus_indicator_laplacian
from multifocus_stereo.focus_indicator_fourier import focus_indicator_fourier
from multifocus_stereo.focus_indicator_aplicator import focus_indicator
from datetime import datetime
import logging
import argparse



def main(parameters):


    # Define paths
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    data_path = os.path.join(parameters.get('input_path'), parameters.get('data_foldername'))

    # Input files
    images_path = os.path.join(data_path, 'images')
    reference_images_path = os.path.join(parameters.get('input_path'), parameters.get('data_foldername'), 'references')

    # Output files
    output_path = os.path.join(parameters.get('output_path'), f'{current_time}_{parameters.get("data_foldername")}')

    focus_save_path = os.path.join(output_path, 'focus_indicator')
    select_save_path = os.path.join(output_path, 'select')
    depth_save_path = os.path.join(output_path, "depth_map")
    all_focus_save_path = os.path.join(output_path, "all_in_focus")
    error_image_path = os.path.join(output_path, 'error_image')
    focus_measure_img_path = os.path.join(output_path, 'focus_measure')
    conf_img_path = os.path.join(output_path, 'confidence')
    debug_data_path = os.path.join(output_path, 'debug_data')
    output_filename = "output.png"


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # **Logging Configuration**
    logging.basicConfig(
        level=logging.DEBUG,  # Minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(os.path.join(output_path, f'multifocus_stereo_{current_time}.log')),  # Output to file
        ],
    )

    # Log each parameter individually
    logging.info(f"Starting multifocus stereo experiment")
    logging.info(f"Parameter - input_path: {parameters.get('input_path')}")
    logging.info(f"Parameter - output_path: {output_path}")
    logging.info(f"Parameter - data_foldername: {parameters.get('data_foldername')}")
    logging.info(f"Parameter - focal_descriptor: {parameters.get('focal_descriptor')}")
    logging.info(f"Parameter - laplacian_kernel_size: {parameters.get('laplacian_kernel_size')}")
    #logging.info(f"Parameter - focus_measure_kernel_size: {parameters.get('focus_measure_kernel_size')}")
    #logging.info(f"Parameter - gaussian_size: {parameters.get('gaussian_size')}")
    logging.info(f"Parameter - focal_step: {parameters.get('focal_step')}")
    logging.info(f"Parameter - gabaritos: {parameters.get('gabaritos')}")
    logging.info(f"Parameter - interpolation_type: {parameters.get('interpolation_type')}")
    logging.info(f"Parameter - debug: {parameters.get('debug')}")


    # Load images
    logging.info("Reading images ...")
    images_paths = find_all_files(images_path)
    image_list = read_images(images_paths, info=True)

    image_stack = np.asarray(image_list)
    gray_image_stack = np.asarray([convert_to_grayscale(img) for img in image_stack])


    # Calcula o indicador de foco para cada imagem
    logging.info("Calculating focus indicator ...")
    focus_indicator_stack = focus_indicator(gray_image_stack, parameters.get('focal_descriptor'), parameters.get('laplacian_kernel_size'), True, True, True)


    logging.info("Extracting focus from each image ...")
    depth_map, all_in_focus_img, select_img_stack, focus_measure_img, confidence_map = all_in_focus(
        image_stack, focus_indicator_stack, parameters.get('focal_step'), parameters.get('interpolation_type'), debug_data_path,
        parameters.get('debug'),
    )




    logging.info("... Saving images ...")
    
    logging.info("Saving focus indicator images")
    for i, focus_indicator_img in enumerate(focus_indicator_stack):
        save_image(focus_save_path, f"{i:03d}_focus_indicator.png", focus_indicator_img, 0, 1)

    logging.info("Saving selected images")
    for i, select_img in enumerate(select_img_stack):
        save_image(select_save_path, f"{i:03d}_select.png", select_img, 0, 1)

    logging.info("Saving depth-from-focus image")
    save_image(depth_save_path, output_filename, normalize(depth_map), 0, 1)
    convert_image_array_to_fni(normalize(depth_map), os.path.join(depth_save_path, "depth_map.fni"))

    logging.info("Saving all-in-focus image")
    save_image(all_focus_save_path, output_filename, all_in_focus_img, 0, 255)

    logging.info("Saving focus measure image")
    #print(f'focus image: min={focus_measure_img.min()}, max={focus_measure_img.max()}')
    save_image(focus_measure_img_path, 'focus_measure.png', focus_measure_img, v_min=0, v_max=1)

    logging.info('Saving confidence image')
    logging.info(f'confidence image: min={confidence_map.min()}, max={confidence_map.max()}')
    save_image(conf_img_path, 'confidence.png', normalize(confidence_map), v_min=0, v_max=1)

    if parameters.get('gabaritos'):
        reference_image = read_images_from_path(reference_images_path)[0]  
        logging.info("Saving error image")
        error_image = calculate_error_image(reference_image, depth_map)
        save_image(error_image_path, 'error_image.png', error_image, -1, 1)
        logging.info("Done!")

    
    logging.info("All operations complete and exiting main function.")

    return depth_map, all_in_focus_img, confidence_map




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multifocus stereo method.')
    parser.add_argument('--param_file', type=str, required=True, 
                       help='Path to the YAML parameter file.')

    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)
    
    main(parameters)
