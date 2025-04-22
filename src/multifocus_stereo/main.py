import os
import numpy as np
from multifocus_stereo.utils import calculate_error_image, normalize

from common.io import read_yaml_parameters, find_all_files, read_images, convert_image_array_to_fni, log_parameters, save_image, read_image
from common.utils import convert_to_grayscale

from multifocus_stereo.mosaic import mosaic
from multifocus_stereo.focus_indicator_aplicator import focus_indicator
from multifocus_stereo.argmax_fuzzy import compute_argmax_fuzzy
from datetime import datetime
import logging
import argparse




def main(parameters):


    if parameters.get('hybrid_method') == True:
        # Define paths
        data_path = os.path.join(parameters.get('input_path'))

        # Input files
        images_path = os.path.join(data_path, 'images')
        reference_images_path = os.path.join(data_path, 'references')

        # Output files
        output_path = parameters.get('output_path_multifocus')
        focus_save_path = os.path.join(output_path, 'focus_indicator')
        error_image_path = os.path.join(output_path, 'error_image')
        debug_data_path = os.path.join(output_path, 'debug_data')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        

    else:
        # Define paths
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        data_path = os.path.join(parameters.get('input_path'), parameters.get('data_foldername'))

        # Input files
        images_path = os.path.join(data_path, 'images')
        reference_images_path = os.path.join(data_path, 'references')

        # Output files
        output_path = os.path.join(parameters.get('output_path'), f'{current_time}_{parameters.get("data_foldername")}')
        focus_save_path = os.path.join(output_path, 'focus_indicator')
        error_image_path = os.path.join(output_path, 'error_image')
        debug_data_path = os.path.join(output_path, 'debug_data')



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

        # Log parameter information
        logging.info("Starting multifocus stereo experiment")
        
        # Log all parameters recursively
        log_parameters(parameters)




    # Load images
    logging.info("... Reading images ...")

    if parameters.get('hybrid_method') == True:
        image_list = read_images(parameters.get('filtered_dir'), info=True)

    else:
        images_paths = find_all_files(images_path)
        image_list = read_images(images_paths, info=True)

    image_stack = np.asarray(image_list)
    gray_image_stack = np.asarray([convert_to_grayscale(img) for img in image_stack])


    # Calcula o indicador de foco para cada imagem
    logging.info("... Calculating focus indicator ...")
    focus_indicator_stack = focus_indicator(gray_image_stack, parameters['focal_descriptor_paramiters']['focal_descriptor'], parameters['focal_descriptor_paramiters']['laplacian_kernel_size'], parameters['focal_descriptor_paramiters']['fourier_radius'], parameters['focal_descriptor_paramiters']['square'], parameters['focal_descriptor_paramiters']['smooth'], parameters['focal_descriptor_paramiters']['zero_border'])

    logging.info("... Calculating argmax fuzzy ...")
    # Calcula argmax fuzzy e confiança
    iSel, wSel = compute_argmax_fuzzy(focus_indicator_stack, parameters['debug'], debug_data_path)

    logging.info("... Calculating mosaic ...")
    # Calcula o mosaico
    zFoc = [15.000, 25.000, 35.000, 45.000, 55.000, 65.000, 75.000, 85.000, 95.000, 105.000, 115.000, 125.000]
    #zFoc = [i for i in range(image_stack.shape[0])]
    print(zFoc)
    sMos, zMos = mosaic(iSel, image_stack, zFoc, parameters['interpolation_type'])




    # Salvando imagens
    logging.info("... Saving Data ...")

    logging.info("saving iSel and wSel")
    save_image(output_path, 'iSel.png', iSel)
    save_image(output_path, 'wSel.png', wSel)
    convert_image_array_to_fni(normalize(iSel), os.path.join(output_path, "iSel.fni"))
    convert_image_array_to_fni(normalize(wSel), os.path.join(output_path, "wSel.fni"))
    
    logging.info("saving sMos and zMos")
    save_image(output_path, 'sMos.png', sMos)
    save_image(output_path, 'zMos.png', zMos)
    convert_image_array_to_fni(normalize(sMos), os.path.join(output_path, "sMos.fni"))
    convert_image_array_to_fni(normalize(zMos), os.path.join(output_path, "zMos.fni"))


    logging.info("... Saving images ...")
    
    logging.info("Saving focus indicator images")
    for i, focus_indicator_img in enumerate(focus_indicator_stack):
        save_image(focus_save_path, f"{i:03d}_focus_indicator.png", focus_indicator_img)


    if parameters.get('gabaritos'):
        reference_image = read_image(find_all_files(reference_images_path)[0], info=True)
        logging.info("Saving error image")
        error_image = calculate_error_image(reference_image, zMos)
        save_image(error_image_path, 'error_image.png', error_image)
        logging.info("Done!")

    logging.info("... All operations complete and exiting main function ...")

    return iSel, wSel, sMos, zMos



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multifocus stereo method.')
    parser.add_argument('--param_file', type=str, required=True, 
                       help='Path to the YAML parameter file.')

    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)
    
    main(parameters)
