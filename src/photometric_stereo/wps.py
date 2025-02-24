#Woodham implementation

import numpy as np



def estimate_normals_argmax(images, light_sources):
    images = images + 1e-6  # Adiciona um pequeno valor para evitar divisão por zero
    
    h, w, num_images = images.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    selected_areas = np.zeros((h, w, num_images), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel_values = images[i, j, :]
            #top_indices = np.argsort(pixel_values)[[11, 9, 7]]  
            top_indices = np.argsort(pixel_values)[-3:]
            #print(top_indices)  
            selected_values = pixel_values[top_indices]
            selected_lights = light_sources[top_indices, :]

            # Resolver o sistema linear I = L * N de forma direta
            normal = np.dot(np.linalg.inv(selected_lights), selected_values)
            #normal = np.matmul(np.linalg.inv(selected_lights), selected_values)

            # Normalizar a normal
            normal /= np.linalg.norm(normal)
            normals[i, j, :] = normal

            selected_areas[i, j, top_indices] = 255

    return normals, selected_areas