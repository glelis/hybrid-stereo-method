import os
import numpy as np
from multifocus_stereo.utils import (
    read_images_from_path,
    save_image,
    print_img_statistics,
    calculate_error_image,
    normalize
)

from multifocus_stereo.depth_from_focus import all_in_focus
from multifocus_stereo.focus_indicator_laplacian import focus_indicator_laplacian
from multifocus_stereo.focus_indicator_fourier import focus_indicator_fourier

from datetime import datetime
import logging
import argparse



def main(parameters):
    """
    Main function to execute the focus stacking and depth map extraction process.

    Args:
        base_path: Base directory containing the input images.
    """


    input_path = parameters['input_path']
    output_path = parameters['output_path']
    data_foldername = parameters['data_foldername']
    focal_descriptor = parameters['focal_descriptor']
    laplacian_kernel_size = parameters['laplacian_kernel_size']
    focus_measure_kernel_size = parameters['focus_measure_kernel_size']
    gaussian_size = parameters['gaussian_size']
    focal_step = parameters['focal_step']
    gabaritos = parameters['gabaritos']
    interpolation_type = parameters['interpolation_type']
    debug = parameters['debug']



    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    img_path = os.path.join(input_path, data_foldername, 'images')
    gabaritos_path = os.path.join(input_path, data_foldername, 'gabaritos')
    output_path = os.path.join(output_path, f'{data_foldername}_{current_time}')

    focus_save_path = os.path.join(output_path, 'focus_indicator')
    select_save_path = os.path.join(output_path, 'select')
    depth_save_path = os.path.join(output_path, "depth_map")
    all_focus_save_path = os.path.join(output_path, "all_in_focus")
    error_image_path = os.path.join(output_path, 'error_image')
    focus_measure_img_path = os.path.join(output_path, 'focus_measure')
    conf_img_path = os.path.join(output_path, 'confidence')
    debug_data_path = os.path.join(output_path, 'debug_data')
    save_as = "output.png"


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

    logging.info(
        f"Starting multifocus stereo experiment with parameters: "
        f"input_path={input_path}, output_path={output_path}"
        f"data_foldername={data_foldername}, focal_descriptor={focal_descriptor}, "
        f"laplacian_kernel_size={laplacian_kernel_size}, focus_measure_kernel_size={focus_measure_kernel_size}, "
        f"gaussian_size={gaussian_size}, focal_step={focal_step}, gabaritos={gabaritos}, "
        f"interpolation_type={interpolation_type}, debug={debug}"
    )

    # Leitura das imagens
    aligned_img_list = read_images_from_path(img_path)
    aligned_img_stack = np.asarray(aligned_img_list)


    # Calcula o indicador de foco para cada imagem
    logging.info("Calculating focus indicator ...")

    if focal_descriptor == 'laplacian':
        focus_indicator_stack = focus_indicator_laplacian(aligned_img_stack, laplacian_kernel_size)

    elif focal_descriptor == 'fourier':
        focus_indicator_stack = focus_indicator_fourier(aligned_img_stack)

    print_img_statistics('focus_indicator_stack', focus_indicator_stack)

    
    logging.info("Extracting focus from each image ...")
    depth_map, all_in_focus_img, select_img_stack, focus_measure_img, img_conf = all_in_focus(
        aligned_img_stack, focus_indicator_stack, focal_step, interpolation_type, debug_data_path,
        debug,
    )


    logging.info("... Saving images ...")
    
    logging.info("Saving focus indicator images")
    for i, focus_indicator_img in enumerate(focus_indicator_stack):
        save_image(focus_save_path, f"{i:03d}_focus_indicator.png", focus_indicator_img, 0, 1)

    logging.info("Saving selected images")
    for i, select_img in enumerate(select_img_stack):
        save_image(select_save_path, f"{i:03d}_select.png", select_img, 0, 1)

    logging.info("Saving depth-from-focus image")
    save_image(depth_save_path, save_as, normalize(depth_map), 0, 1)

    logging.info("Saving all-in-focus image")
    save_image(all_focus_save_path, save_as, all_in_focus_img, 0, 255)

    logging.info("Saving focus measure image")
    #print(f'focus image: min={focus_measure_img.min()}, max={focus_measure_img.max()}')
    save_image(focus_measure_img_path, 'focus_measure.png', focus_measure_img, v_min=0, v_max=1)

    logging.info('Saving confidence image')
    logging.info(f'confidence image: min={img_conf.min()}, max={img_conf.max()}')
    save_image(conf_img_path, 'confidence.png', normalize(img_conf), v_min=0, v_max=1)

    if gabaritos:
        reference_image = read_images_from_path(gabaritos_path)[0]  
        logging.info("Saving error image")
        error_image = calculate_error_image(reference_image, depth_map)
        save_image(error_image_path, 'error_image.png', error_image, -1, 1)
        logging.info("Done!")

    
    logging.info("Done!")

    return depth_map, all_in_focus_img, img_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multifocus stereo method.')
    parser.add_argument('--param_file', type=str, required=True, help='Path to the parameter file.')

    args = parser.parse_args()

    # Read parameters from the file
    parameters = {}
    with open(args.param_file, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            parameters[key] = value

    depth_map, all_in_focus_img, img_conf = main(parameters)
