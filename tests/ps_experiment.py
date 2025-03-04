from photometric_stereo.main import main as ps_main

# methods: L2, L1, SBL, RPCA
# image_type: png, npy

parameters= {'input_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/',
             'output_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_stereo/',
             'data_foldername': '2025-02-09-stF-melon24-amb0-glo0',
             'data_scale': 1,
             'image_type': 'png',  
             'method_name': 'L2',
             'debug': False,
             }

ps_main(parameters)  