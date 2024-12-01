from photometric_stereo.main import main as ps_main


parameters= {'input_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/photometric_stereo/',
             'output_path': '/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/photometric_stereo/',
             'data_foldername': 'bunny_lambert',
             'data_scale': 1,
             'image_type': 'jpg',
             'method_name': 'L2',
             'debug': False,
             }

ps_main(parameters)