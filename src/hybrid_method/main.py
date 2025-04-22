import os
from multifocus_stereo.main import main as multifocus_stereo_main
from photometric_stereo.mainwps import main as photometric_stereo_main
from common.utils import calculate_avarage_of_images
from common.io import read_yaml_parameters, log_parameters, find_all_files, read_images, save_image

from datetime import datetime
import logging
import argparse
from natsort import natsorted





def main(parameters):



    # Define paths
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    
    # Input files
    #data_path = os.path.join(parameters.get('input_path'))

    # Output files
    output_path = os.path.join(parameters.get('output_path'), f'{current_time}_{parameters.get("data_foldername")}')


    if not os.path.exists(output_path):
        os.makedirs(output_path)



    # Logging Configuration
    logging.basicConfig(
        level=logging.DEBUG,  # Minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(os.path.join(output_path, f'hybrid_stereo_{current_time}.log')),  # Output to file
        ],
    )

    # Log parameter information
    logging.info("Starting multifocus stereo experiment")
    
    # Log all parameters recursively
    log_parameters(parameters)


    
    all_files_path = find_all_files(os.path.join(parameters.get('input_path'), parameters.get('data_foldername')))

    # Avarage of images

    zf_dir_name = sorted(set([os.path.basename(os.path.dirname(path)) for path in all_files_path if os.path.basename(os.path.dirname(path)).startswith('zf')]))

    list_of_avarage_images = []
    for zf in zf_dir_name:

        filtered_dir = sorted([i for i in all_files_path if f'{zf}' in i and 'sVal.png' in i])
        image_list = read_images(filtered_dir, info=False)
        #image_list = [(image / image.max() * 255).astype(np.uint8) for image in image_list]
        mean = calculate_avarage_of_images(image_list)
        save_image(os.path.join(output_path, 'multifocus_stereo', 'average', 'images'),f'mean_{zf}.png', mean)
        list_of_avarage_images.append(os.path.join(output_path,'multifocus_stereo','average','images',f'mean_{zf}.png'))

    
    parameters['filtered_dir'] = list_of_avarage_images
    parameters['output_path_multifocus'] = os.path.join(output_path, 'multifocus_stereo','average')


    iSel, wSel, sMos, zMos = multifocus_stereo_main(parameters)



    #Multifocus stereo

    lights_dir_name = sorted(set([os.path.basename(os.path.dirname(path)) for path in all_files_path if os.path.basename(os.path.dirname(path)).startswith('L')]))

    #sMos_list = []

    for light in lights_dir_name:

        logging.info(f"\n ... Processing folder: {light} ...")
        filtered_dir = sorted([i for i in all_files_path if f'{light}/zf' in i and 'sVal.png' in i])

        parameters['filtered_dir'] = filtered_dir
        parameters['output_path_multifocus'] = os.path.join(output_path, 'multifocus_stereo',light)


        iSel, wSel, sMos, zMos = multifocus_stereo_main(parameters)
        #sMos_list.append(sMos)




    # Photometric stereo
    #output_path = '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo/20250422_1326_2025-03-08-stQ-melon24-amb0.00-glo0.50'
    output_pahts = find_all_files(output_path)
    parameters['sMos_path_list'] = natsorted([i for i in output_pahts if 'sMos.png' in i and 'av' not in  i])

    #parameters['sMos_list'] = sMos_list
    parameters['output_path_photometric'] = os.path.join(output_path, 'photometric_stereo')
    parameters['lights_path'] = [i for i in all_files_path if 'lights.npy' in i][0]
    print(parameters['lights_path'])


    photometric_stereo_main(parameters)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Hybrid stereo method.')
    parser.add_argument('--param_file', type=str, required=True, 
                       help='Path to the YAML parameter file.')

    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)
    
    main(parameters)
