from multifocus_stereo.main import main as ms_main

parameters= {'input_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/multifocus_stereo/synthetic/',
             'output_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/multifocus_stereo/',
             'data_foldername': 'scene_5',
             'interpolation_type': 'crop',
             'focal_step': 1,
             'focal_descriptor': 'fourier',
             'laplacian_kernel_size': 1,
             'gaussian_size': 9,
             'focus_measure_kernel_size': 0,
             'gabaritos': False,
             'debug': False,
             }


ms_main(parameters)