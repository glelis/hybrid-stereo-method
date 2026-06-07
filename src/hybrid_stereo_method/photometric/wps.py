# Woodham implementation

import numpy as np

from hybrid_stereo_method.infrastructure.utils import normalize


def estimate_normals_argmax(images, light_sources, wps_params=None):
    """
    Estimates surface normals using the brightest pixels from multiple images.

    This function selects the 3 brightest images for each pixel and solves
    a linear system to estimate the surface normal.

    Args:
        images: List of grayscale images captured under different lighting.
        light_sources: Array of light source directions for each image.

    Returns:
        normals: Array of estimated surface normals for each pixel.
        selected_areas: Binary mask indicating which images were used for each pixel.
    """
    if wps_params is None:
        wps_params = {}
    
    epsilon = wps_params.get("epsilon", 1e-6)
    top_k = wps_params.get("top_k", 3)

    images = np.stack(images, axis=-1)  # Convert list of images to a 3D array
    images = images + epsilon  # Add a small value to avoid division by zero

    h, w, num_images = images.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    selected_areas = np.zeros((h, w, num_images), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel_values = images[i, j, :]
            top_indices = np.argsort(pixel_values)[-top_k:]
            selected_values = pixel_values[top_indices]
            selected_lights = light_sources[top_indices, :]

            # PS-10: lstsq com verificação de posto no lugar de inv() nu — as 3
            # luzes mais brilhantes podem ser quase coplanares (sistema singular).
            normal, _, rank, _ = np.linalg.lstsq(selected_lights, selected_values, rcond=None)
            norm = np.linalg.norm(normal)
            if rank < 3 or norm == 0 or not np.isfinite(norm):
                normals[i, j, :] = np.nan
                continue
            normal /= norm
            normals[i, j, :] = normal

            selected_areas[i, j, top_indices] = 255

    return normals, selected_areas


def estimate_normals_argmax_lstsq(images, light_sources, wps_params=None):
    """
    Estimates surface normals using least squares fitting on all pixels.

    This function uses the least squares method to solve the linear system
    for estimating surface normals, and also computes residuals for confidence.

    Args:
        images: List of grayscale images captured under different lighting.
        light_sources: Array of light source directions for each image.

    Returns:
        normals: Array of estimated surface normals for each pixel.
        residuals: Array of least squares residuals for each pixel.
        confidence: Confidence map based on inverse residuals.
        selected_areas: Binary mask indicating which images were used for each pixel.
    """
    if wps_params is None:
        wps_params = {}

    epsilon = wps_params.get("epsilon", 1e-6)

    images = np.stack(images, axis=-1)  # Convert list of images to a 3D array
    images = images + epsilon  # Add a small value to avoid division by zero

    h, w, num_images = images.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    selected_areas = np.zeros((h, w, num_images), dtype=np.float32)
    residuals = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel_values = images[i, j, :]
            top_indices = np.argsort(pixel_values)  # Use all indices
            selected_values = pixel_values[top_indices]
            selected_lights = light_sources[top_indices, :]

            # Solve the linear system I = L * N using least squares
            normal, residual, _, _ = np.linalg.lstsq(selected_lights, selected_values.T, rcond=None)

            # Normalize the normal
            normal /= np.linalg.norm(normal)
            normals[i, j, :] = normal
            # lstsq returns an empty residual array when the system is square
            # (exactly 3 images) or rank-deficient — treat as zero residual
            residuals[i, j] = residual[0] if residual.size else 0.0

            selected_areas[i, j, top_indices] = 255

    # Convert residuals to confidence values (epsilon avoids division by zero
    # for exact fits, which would otherwise produce inf/nan), then normalize
    # to [0, 1] (normalize() guards the constant-confidence case)
    confidence = normalize(1 / (residuals + epsilon))

    return normals, residuals, confidence, selected_areas


def estimate_normals_argmax_lstsq_robust(images, light_sources, wps_params=None):
    """
    Estimates surface normals using a robust least squares approach.

    This function implements a robust estimation by:
    1. Rejecting shadowed pixels
    2. Iteratively removing outliers based on residuals
    3. Computing confidence metrics based on available data and residual quality

    Args:
        images: List of grayscale images captured under different lighting.
        light_sources: Array of light source directions for each image.

    Returns:
        normals: Array of estimated surface normals for each pixel.
        albedo: Estimated albedo for each pixel.
        confidence: Confidence map based on available data and residual quality.
        selected_areas: Binary mask indicating which images were used for each pixel.
    """
    if wps_params is None:
        wps_params = {}

    epsilon = wps_params.get("epsilon", 1e-6)
    shadow_threshold = wps_params.get("shadow_threshold", 1e-3)
    outlier_threshold_multiplier = wps_params.get("outlier_threshold_multiplier", 3)
    # PS-03: multiplicador independente para a detecção unilateral de saturação
    # (critério b do laço robusto). Default 1.0: mais apertado que o critério
    # simétrico (k=3) pois saturação é sempre unilateral. Revisão de code-review
    # propôs elevar para 2.0 (argumento: 1-sided 1.0×1.4826×MAD exclui ~16%
    # da cauda nominal), mas 1.5 e 2.0 falham no teste sintético de saturação
    # (8 luzes, 2 saturadas a 60%: 5.79° e 7.56° > limiar 5°). Default mantido
    # em 1.0; exposto via config para ajuste por dataset.
    saturation_outlier_multiplier = wps_params.get("saturation_outlier_multiplier", 1.0)
    # PS-02: o limiar relativo é inócuo abaixo do piso de 8 bits (1/255 ≈ 3.9e-3
    # > 1e-3 sempre que houver sinal). Limiar ABSOLUTO em radiância linear
    # rejeita sombras reais; limiar superior rejeita medições saturadas.
    # H7 (investigação 2026-06-06): os limiares ABSOLUTOS são calibrados em
    # unidades 8-bit (0-255); intensity_max os reescala para a profundidade de
    # bits da fonte (default 255 => fator 1, retrocompatível). Sem isso,
    # saturation_threshold=250 descarta ~90% de dados 16-bit como saturação.
    intensity_max = float(wps_params.get("intensity_max", 255.0))
    intensity_scale = intensity_max / 255.0
    _shadow_abs_raw = wps_params.get("shadow_absolute_threshold")
    _saturation_raw = wps_params.get("saturation_threshold")
    shadow_absolute = None if _shadow_abs_raw is None else _shadow_abs_raw * intensity_scale
    saturation_threshold = (
        None if _saturation_raw is None else _saturation_raw * intensity_scale
    )

    images = np.stack(images, axis=-1)  # Convert list of images to a 3D array
    images = images + epsilon  # Add a small value to avoid division by zero

    h, w, num_images = images.shape

    normals = np.zeros((h, w, 3), dtype=np.float32)
    albedo = np.zeros((h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    selected_areas = np.zeros((h, w, num_images), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel_values = images[i, j, :]
            v_max = np.max(pixel_values)
            valid_indices = pixel_values / v_max > shadow_threshold  # Step (1): Reject shadowed pixels
            if shadow_absolute is not None:
                valid_indices &= pixel_values >= shadow_absolute
            if saturation_threshold is not None:
                valid_indices &= pixel_values <= saturation_threshold

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
                residuals = np.abs(
                    np.dot(selected_lights, normal) - selected_values
                )  # Compute residuals for each equation
                # PS-03: limiar robusto — dois critérios combinados:
                # (a) Simétrico: mediana + k*1.4826*MAD sobre |resíduos|.
                #     Robusto à inflação da média por outliers grandes (mascaramento),
                #     capturando highlights e erros normais de ambos os lados.
                # (b) Unilateral (saturação): quando I_pred >> I_obs, o sinal foi
                #     clipado. Detectado pelo MAD das sobreprevisões (I_pred - I_obs > 0):
                #     se algum valor estiver acima de med_overpred + k*1.4826*mad_overpred,
                #     é candidato a saturação e é descartado.
                r_med = np.median(residuals)
                mad = np.median(np.abs(residuals - r_med))
                if mad == 0.0:
                    break  # resíduos (quase) idênticos: nada a rejeitar
                mask = residuals <= r_med + outlier_threshold_multiplier * 1.4826 * mad
                # One-sided saturation check: I_pred - I_obs > 0 (overprediction)
                overpred = np.dot(selected_lights, normal) - selected_values
                sat_vals = overpred[overpred > 0]
                if len(sat_vals) >= 3:
                    sat_med = np.median(sat_vals)
                    sat_mad = np.median(np.abs(sat_vals - sat_med))
                    if sat_mad > 0:
                        sat_threshold = sat_med + saturation_outlier_multiplier * 1.4826 * sat_mad
                        mask &= ~(overpred > sat_threshold)
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
                # PS-01: no modelo I = rho*(L.n̂), o lstsq devolve m = rho*n̂ —
                # o albedo é ||m|| e a normal é m/||m||. (O antigo ||L@n̂|| era a
                # norma das intensidades preditas, crescendo com sqrt(n_luzes).)
                m = normal
                rho = np.linalg.norm(m)
                if rho == 0 or not np.isfinite(rho):
                    normals[i, j, :] = np.nan
                    confidence[i, j] = 0
                    continue
                normal = m / rho
                normals[i, j, :] = normal
                albedo[i, j] = rho

                # PS-04: resíduos RECOMPUTADOS do modelo final sobre o conjunto
                # final (antes, eram os do ajuste anterior à última remoção).
                residuals = np.abs(np.dot(selected_lights, m) - selected_values)

                # PS-05: resíduo normalizado pela escala do sinal (albedo) torna
                # a confiança invariante a ganho radiométrico (0-255 vs 0-1).
                N = len(selected_values)
                M = num_images
                residual_std = np.std(residuals) / (rho + epsilon)
                confidence[i, j] = (N / M) * (1 / (1 + residual_std))

                selected_areas[i, j, original_indices] = 255

    return normals, albedo, confidence, selected_areas
