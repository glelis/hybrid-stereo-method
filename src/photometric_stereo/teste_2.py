import numpy as np
import cv2
import matplotlib.pyplot as plt

from photometric_stereo.psvizualization import disp_channels_3d, disp_channels

# Função para carregar imagens e fontes de luz
def load_images_and_light_sources(image_paths, light_sources):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Erro ao carregar a imagem: {path}")
        images.append(img.astype(np.float32) / 255.0)
    return np.stack(images, axis=-1), np.array(light_sources)

# Função para estimar as normais da superfície
def estimate_normals(images, light_sources):
    h, w, num_images = images.shape
    images = images.reshape(-1, num_images)

    # Resolver o sistema linear I = L * N
    normals = np.linalg.lstsq(light_sources, images.T, rcond=None)[0]

    # Normalizar as normais
    normals /= np.linalg.norm(normals, axis=0)
    normals = normals.T.reshape(h, w, 3)
    return normals

# Função para visualizar as normais
def visualize_normals(normals):
    normals_visual = (normals + 1) / 2  # Mapeia valores de [-1, 1] para [0, 1]
    plt.imshow(normals_visual)
    plt.axis('off')
    plt.title("Normais da Superfície")
    plt.show()

# Exemplo de uso
if __name__ == "__main__":


    image_paths = []

    light_sources = np.load('/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/photometric_stereo/ex_15_muff/lights.npy')

    for i in range(0,12):
        if i <10:
            image_paths.append(f"/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/photometric_stereo/ex_15_muff/images/light_0{i}.png")
        else:
            image_paths.append(f"/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/photometric_stereo/ex_15_muff/images/light_{i}.png")


    # Carregar as imagens e fontes de luz
    images, light_sources = load_images_and_light_sources(image_paths, light_sources)

    # Estimar as normais da superfície
    normals = estimate_normals(images, light_sources)

    # Visualizar as normais

    disp_channels(normals)
    disp_channels_3d(normals)










