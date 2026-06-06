import argparse
import logging
import os
from datetime import datetime

import numpy as np

from hybrid_stereo_method.infrastructure.io.image_io import (
    convert_image_array_to_fni,
    find_all_files,
    log_parameters,
    read_image,
    read_images,
    read_yaml_parameters,
    save_image,
)
from hybrid_stereo_method.infrastructure.utils import convert_to_grayscale, normalize
from hybrid_stereo_method.multifocus.argmax_fuzzy import compute_argmax_fuzzy
from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator
from hybrid_stereo_method.multifocus.mosaic import mosaic
from hybrid_stereo_method.multifocus.utils import calculate_error_image


def check_z_foc_uniformity(z_foc: list[float]) -> None:
    """Warn when z_foc spacing is non-uniform (MF-13): the sub-frame parabola is
    fitted in INDEX space and converted to z afterwards, which is only exact when
    the index->z map is affine (uniform spacing).

    Note: if all steps are 0 (degenerate repeated z_foc), np.allclose with atol=0
    passes silently — that pathological config is caught by the len/shape check
    elsewhere.
    """
    steps = np.diff(np.asarray(z_foc, dtype=np.float64))
    if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        logging.warning(
            "z_foc spacing is non-uniform (steps %s): the parabolic sub-frame fit "
            "is performed in index space and is only exact for uniform spacing "
            "(MF-13) — sub-frame depths may be biased between planes.",
            np.round(steps, 6).tolist(),
        )


def main(parameters):
    experiment_type = parameters.get("experiment", {}).get("type")
    
    if experiment_type == "hybrid":
        # Define paths
        data_path = os.path.join(parameters["experiment"]["paths"]["input"])

        # Input files
        images_path = os.path.join(data_path, "images")
        reference_images_path = os.path.join(data_path, "references")

        # Output files
        # For hybrid mode, output_path_multifocus is passed dynamically or derived, 
        # but here we rely on what passed in parameters or structure. 
        # In the original code, 'output_path_multifocus' was a top-level key set by hybrid/main.py
        # We need to ensure hybrid/main.py sets this key, or we check if it exists.
        output_path = parameters.get("output_path_multifocus") 
        # Note: hybrid/main.py sets 'output_path_multifocus' directly in the dict before calling this.
        
        focus_save_path = os.path.join(output_path, "focus_indicator")
        error_image_path = os.path.join(output_path, "error_image")
        debug_data_path = os.path.join(output_path, "debug_data")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    else:
        # Define paths
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        
        input_path = parameters["experiment"]["paths"]["input"]
        data_foldername = parameters["experiment"]["paths"]["data_folder"]
        
        data_path = os.path.join(input_path, data_foldername)

        # Input files
        images_path = os.path.join(data_path, "images")
        reference_images_path = os.path.join(data_path, "references")

        # Output files
        base_output_path = parameters["experiment"]["paths"]["output"]
        output_path = os.path.join(base_output_path, f"{current_time}_{data_foldername}")

        focus_save_path = os.path.join(output_path, "focus_indicator")
        error_image_path = os.path.join(output_path, "error_image")
        debug_data_path = os.path.join(output_path, "debug_data")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # **Logging Configuration**
        logging.basicConfig(
            level=logging.DEBUG,  # Minimum log level
            format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler(
                    os.path.join(output_path, f"multifocus_stereo_{current_time}.log")
                ),  # Output to file
            ],
        )

        # Log parameter information
        logging.info("Starting multifocus stereo experiment")

        # Log all parameters recursively
        log_parameters(parameters)

    # Load images
    logging.info("... Reading images ...")

    if experiment_type == "hybrid":
        # MF-12 fix: prefer in-memory float averages (filtered_images) over the
        # PNG round-trip (filtered_dir) to avoid uint8 quantization before the
        # focus measure.  Fall back to reading PNGs when filtered_images is absent
        # (e.g. legacy callers that only set filtered_dir).
        if parameters.get("filtered_images") is not None:
            logging.info("... Using in-memory float averages (filtered_images) ...")
            image_list = list(parameters["filtered_images"])
        else:
            image_list = read_images(parameters.get("filtered_dir"), info=True)

    else:
        images_paths = find_all_files(images_path)
        image_list = read_images(images_paths, info=True)

    image_stack = np.asarray(image_list)
    gray_image_stack = np.asarray([convert_to_grayscale(img) for img in image_stack])

    # Calcula o indicador de foco para cada imagem
    logging.info("... Calculating focus indicator ...")
    
    mf_params = parameters["multifocus"]
    focus_measure_params = mf_params["focus_measure"]
    preprocess_params = focus_measure_params["preprocessing"]
    
    focus_indicator_stack = focus_indicator(
        gray_image_stack,
        focus_measure_params["method"],
        focus_measure_params["parameters"]["kernel_size"],
        focus_measure_params["parameters"]["radius"],
        preprocess_params["square"],
        preprocess_params["smooth"],
        preprocess_params.get("spatial_median_filter", False),
        preprocess_params["zero_border"],
    )

    logging.info("... Calculating argmax fuzzy ...")
    # Calcula argmax fuzzy e confiança
    fuzzy_params = mf_params["optimization"]
    debug_mode = parameters["experiment"]["settings"]["debug"]
    iSel, wSel = compute_argmax_fuzzy(focus_indicator_stack, debug_mode, debug_data_path, fuzzy_params)

    logging.info("... Calculating mosaic ...")
    # Calcula o mosaico
    zFoc = mf_params["parameters"]["z_foc"]
    if len(zFoc) != image_stack.shape[0]:
        error_msg = f"Erro Crítico: A lista z_foc no arquivo YAML possui {len(zFoc)} elementos, mas a pasta contém {image_stack.shape[0]} imagens. Cada imagem deve possuir seu valor z_foc."
        logging.error(error_msg)
        raise ValueError(error_msg)
    check_z_foc_uniformity(zFoc)

    interpolation_type = mf_params["parameters"]["interpolation"]
    # MF-04: pass wSel so confidence-0 pixels are masked (zMos→NaN, sMos→nearest frame)
    sMos, zMos = mosaic(iSel, image_stack, zFoc, interpolation_type, wSel=wSel)

    # Salvando imagens
    logging.info("... Saving Data ...")

    logging.info("saving iSel and wSel")
    # MF-04: iSel/zMos may contain NaN (undecidable pixels from MF-03 or low-confidence
    # pixels masked here).  PNGs and normalized FNIs use nan_to_num(0) so save_image
    # (which calls cv2.normalize) does not see NaN.  The raw FNIs keep NaN intact so
    # downstream consumers (C integrator, wSel channel) can treat them as invalid.
    save_image(output_path, "iSel.png", np.nan_to_num(iSel, nan=0.0))
    save_image(output_path, "wSel.png", wSel)
    convert_image_array_to_fni(
        normalize(np.nan_to_num(iSel, nan=0.0)), os.path.join(output_path, "iSel.fni")
    )
    # MF-04/MF-07: export wSel raw (already in [0,1] as R²; no normalize stretch needed)
    convert_image_array_to_fni(wSel, os.path.join(output_path, "wSel.fni"))

    logging.info("saving sMos and zMos")
    save_image(output_path, "sMos.png", sMos)
    save_image(output_path, "zMos.png", np.nan_to_num(zMos, nan=0.0))
    convert_image_array_to_fni(normalize(sMos), os.path.join(output_path, "sMos.fni"))

    # Do NOT normalize zMos — keep physical Z distance values; NaN marks invalid pixels
    convert_image_array_to_fni(zMos, os.path.join(output_path, "zMos.fni"))

    # MF-04: weight channel is 0 where zMos is NaN so the C integrator ignores those
    # pixels as hints (weight=0 means "no hint").  wSel is exported raw (R² in [0,1]).
    wSel_export = np.where(np.isfinite(zMos), wSel, 0.0)
    zMos_with_confidence = np.stack((zMos, wSel_export), axis=-1)
    convert_image_array_to_fni(
        zMos_with_confidence, os.path.join(output_path, "zMos_with_confidence.fni")
    )

    logging.info("... Saving images ...")

    logging.info("Saving focus indicator images")
    for i, focus_indicator_img in enumerate(focus_indicator_stack):
        save_image(focus_save_path, f"{i:03d}_focus_indicator.png", focus_indicator_img)

    if parameters["experiment"]["settings"]["gabaritos"]:
        reference_image = read_image(find_all_files(reference_images_path)[0], info=True)
        logging.info("Saving error image")
        # MF-04: zMos may contain NaN (undecidable pixels). normalize() uses np.max/np.min
        # which propagate NaN → silently corrupt error image. Zero NaN for error visualization.
        error_image = calculate_error_image(reference_image, np.nan_to_num(zMos, nan=0.0))
        save_image(error_image_path, "error_image.png", error_image)
        logging.info("Done!")

    logging.info("... All operations complete and exiting main function ...")

    return iSel, wSel, sMos, zMos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multifocus stereo method.")
    parser.add_argument(
        "--param_file", type=str, required=True, help="Path to the YAML parameter file."
    )

    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)

    main(parameters)
