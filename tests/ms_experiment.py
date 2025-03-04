from multifocus_stereo.main import main as ms_main


parameters = {
            'input_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/2025-02-09-stF-melon24-amb0-glo0/',
            'output_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo/',
            'data_foldername': 'media_iluminacao',
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