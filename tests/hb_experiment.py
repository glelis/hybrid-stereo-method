from multifocus_stereo.main import main as ms_main
from photometric_stereo.main import main as ps_main
import os
import shutil




#for i in range(1,5):
#    parameters= {'input_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/brasil_00/2017-06-07-1046/',
#                'output_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/processed/hybrid_stereo/multifocus_stereo/brasil/',
#                'data_foldername': f'L_0{i}/V_00/align/',
#                'interpolation_type': 'crop', 
#                'focal_step': 1,
#                'focal_descriptor': 'fourier',
#                'laplacian_kernel_size': 1,
#                'gaussian_size': 9,
#                'focus_measure_kernel_size': 0,
#                'gabaritos': False,
#                'debug': False,
#                }
#    print(f"Starting multifocus stereo experiment with parameters: {parameters}")
#    ms_main(parameters)
#


def copy_file(src, dst):
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    shutil.copy2(src, dst)



for i in range(0,5):
    # Example usage
    src_file = f'/home/lelis/Documents/Projetos/hybrid-stereo-method/data/processed/hybrid_stereo/multifocus_stereo/brasil/L_0{i}/V_00/align/*/all_in_focus/output.png'
    dst_file = f'/home/lelis/Documents/Projetos/hybrid-stereo-method/data/processed/hybrid_stereo/photometric_stereo/brasil/images/output_{i}.png'
    copy_file(src_file, dst_file)




parameters= {'input_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/photometric_stereo/',
             'output_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/photometric_stereo/',
             'data_foldername': 'brasil',
             'data_scale': 1,
             'image_type': 'png',
             'method_name': 'L2',
             'debug': False,
             }

ps_main(parameters)