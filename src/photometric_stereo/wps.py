#Woodham implementation

import numpy as np



def estimate_normals_argmax(images, light_sources):
    images = np.stack(images, axis=-1) # Convert list of images to a 3D array
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



def estimate_normals_argmax_lstsq(images, light_sources):
    images = np.stack(images, axis=-1) # Convert list of images to a 3D array
    images = images + 1e-6  # Adiciona um pequeno valor para evitar divisão por zero
    
    h, w, num_images = images.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    selected_areas = np.zeros((h, w, num_images), dtype=np.float32)
    residuals =np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel_values = images[i, j, :]
            top_indices = np.argsort(pixel_values)#[-4:]
            #print(top_indices)  
            selected_values = pixel_values[top_indices]
            selected_lights = light_sources[top_indices, :]

            # Resolver o sistema linear I = L * N de forma direta
            normal, residual, _, _ = np.linalg.lstsq(selected_lights, selected_values.T, rcond=None)


            # Normalizar a normal
            normal /= np.linalg.norm(normal)
            normals[i, j, :] = normal
            residuals[i,j] = residual

            selected_areas[i, j, top_indices] = 255

    #residuals /= np.max(residuals)  # Normalize residuals to the range [0, 1]
    confidence = 1/residuals
    confidence = (confidence - np.min(confidence)) / (np.max(confidence) - np.min(confidence))  # Normalize confidence to the range [0, 1]

    return normals, residuals, confidence, selected_areas



def estimate_normals_argmax_lstsq_robust(images, light_sources):
    images = np.stack(images, axis=-1)  # Convert list of images to a 3D array
    images = images + 1e-6  # Add a small value to avoid division by zero
    
    h, w, num_images = images.shape

    normals = np.zeros((h, w, 3), dtype=np.float32)
    albedo = np.zeros((h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    selected_areas = np.zeros((h, w, num_images), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel_values = images[i, j, :]
            v_max = np.max(pixel_values)
            valid_indices = pixel_values / v_max > 1e-3  # Step (1): Reject shadowed pixels
            
            if np.sum(valid_indices) < 3:  # Step (2): Not enough valid images
                normals[i, j, :] = np.nan
                confidence[i, j] = 0
                continue
            
            selected_values = pixel_values[valid_indices]
            selected_lights = light_sources[valid_indices, :]
            original_indices = np.where(valid_indices)[0]  # Map to original indices
            
            while True:
                # Solve the linear system using least squares
                normal, _, _, _ = np.linalg.lstsq(selected_lights, selected_values.T, rcond=None)
                
                # Adjust residuals to match the size of selected_values
                residuals = np.abs(np.dot(selected_lights, normal) - selected_values)  # Compute residuals for each equation
                r_avg = np.mean(residuals)  # Step (4): Compute average residual
                
                # Step (5): Discard outliers
                mask = residuals <= 3 * r_avg
                if np.sum(mask) == len(selected_values):  # Step (6): Stabilization
                    break
                
                selected_values = selected_values[mask]
                selected_lights = selected_lights[mask, :]
                original_indices = original_indices[mask]  # Update original indices
                
                if len(selected_values) < 3:  # Not enough valid images
                    normals[i, j, :] = np.nan
                    confidence[i, j] = 0
                    break
            
            if len(selected_values) >= 3:
                # Normalize the normal vector
                normal /= np.linalg.norm(normal)
                normals[i, j, :] = normal
                
                # Compute albedo
                albedo[i, j] = np.linalg.norm(np.dot(selected_lights, normal))
                
                # Adjust confidence based on N, M, and residuals
                N = len(selected_values)
                M = num_images
                residual_std = np.std(residuals)
                confidence[i, j] = (N / M) * (1 / (1 + residual_std))  # Confidence decreases with fewer data and higher residuals

                # Update selected_areas using original indices
                selected_areas[i, j, original_indices] = 255


    return normals, albedo, confidence, selected_areas