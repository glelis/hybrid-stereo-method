import argparse
import dataclasses
import logging
import os
import re
from datetime import datetime

import yaml
from natsort import natsorted

from hybrid_stereo_method.hybrid.integrate import IntegrateRecursiveConfig
from hybrid_stereo_method.hybrid.stages import (
    run_evaluation_stage,
    run_integration_stage,
    run_photometric_stage,
)
from hybrid_stereo_method.infrastructure.io.image_io import (
    find_all_files,
    log_parameters,
    read_images,
    read_yaml_parameters,
    save_image,
)
from hybrid_stereo_method.infrastructure.utils import (
    calculate_avarage_of_images,
    resolve_z_foc,
)
from hybrid_stereo_method.multifocus.main import main as multifocus_stereo_main


def select_per_light_smos(files):
    """Per-light sMos.npy paths for the photometric solver, in natural order.

    Excludes only the mosaic of the averaged stack (parent directory named
    "average") — matching on the directory component, not on a substring of
    the whole path, so output folders like "03_wavelet" are not filtered out.
    """
    return natsorted(
        file
        for file in files
        if os.path.basename(file) == "sMos.npy"
        and os.path.basename(os.path.dirname(file)) != "average"
    )


def collect_light_dirs(files: list[str], data_path: str) -> list[str]:
    """Unique ``L<n>`` directory components at ANY depth under ``data_path``.

    Fixes MF-14 (ported from worktree-vectorized-snuggling-harp): looking only
    at the immediate parent of each file finds nothing in the documented layout
    ``L<n>/zf<m>/sVal.png`` (the parent is always ``zf*``). Real datasets only
    worked by accident, because of stray files such as ``L000/selected-pixels.png``.
    """
    lights: set[str] = set()
    for f in files:
        rel = os.path.relpath(f, data_path)
        if rel.startswith(os.pardir):
            continue
        for part in rel.split(os.sep)[:-1]:  # exclude the filename itself
            if re.fullmatch(r"L\d+(?:-.+)?", part):  # L000 ou L000-amb0.00-glo0.00
                lights.add(part)
    return natsorted(lights)


def common_zf_dirs(files: list[str], data_path: str, light_dirs: list[str]) -> list[str]:
    """Planos focais (pastas zf*) presentes em TODAS as luzes, em ordem natural.

    Sem pastas de luz, todos os zf* encontrados. O stF-melon24 (2025-02-09)
    tem um plano extra (zf060) só na L000: usá-lo daria pilhas de tamanhos
    diferentes por luz e uma média "average" desse plano com uma luz só.
    """
    by_light: dict[str, set[str]] = {}
    all_zf: set[str] = set()
    for f in files:
        parent = os.path.basename(os.path.dirname(f))
        if not parent.startswith("zf"):
            continue
        all_zf.add(parent)
        rel = os.path.relpath(f, data_path).split(os.sep)
        for light in light_dirs:
            if light in rel[:-2]:
                by_light.setdefault(light, set()).add(parent)
    if not light_dirs:
        return natsorted(all_zf)
    common = set.intersection(*(by_light.get(light, set()) for light in light_dirs))
    dropped = natsorted(all_zf - common)
    if dropped:
        logging.warning("Planos focais ausentes em alguma luz, ignorados: %s", dropped)
    return natsorted(common)


def build_integration_config(integration_params: dict, debug: bool) -> IntegrateRecursiveConfig:
    """Build the solver config from the ``hybrid.integration`` YAML section.

    Units (INT-04/CONV-4): the slopes the C solver integrates (dZ/dX = -nx/nz)
    are dimensionless and summed over 1-px cells, so the raw heights come out
    in "height per pixel", while the multifocus hints (zMos, absolute) are in
    ``z_foc`` units. ``pixel_size`` — the lateral size of one pixel in the same
    units as ``z_foc`` — reconciles them: the slopes are multiplied by it, so the
    integrated heights come out in ``z_foc`` units, commensurable with the hints.

    ``slopes_scale`` keeps carrying the axis signs (sy = -1 for Y-up datasets:
    the FNI is written top-down while the C library assumes Y up); the final
    factors are ``(sx * pixel_size, sy * pixel_size)``.
    """
    pixel_size = integration_params.get("pixel_size")
    if pixel_size is None:
        if integration_params.get("use_hints", False):
            logging.warning(
                "use_hints=True but hybrid.integration.pixel_size is not set: the "
                "integrated heights stay in pixel units while the hints are in z_foc "
                "units — incommensurable scales (INT-04/CONV-4). Set pixel_size to "
                "the lateral size of one pixel in the same units as z_foc."
            )
        pixel_size = 1.0
    pixel_size = float(pixel_size)
    if pixel_size <= 0:
        raise ValueError(f"hybrid.integration.pixel_size must be > 0, got {pixel_size}")

    initial_method = integration_params.get("initial_method", "zero")
    if initial_method == "hints" and not integration_params.get("use_hints", False):
        raise ValueError(
            "hybrid.integration.initial_method='hints' requires use_hints: true — "
            "without a hints map the C solver aborts (INT-01)."
        )

    report_step = int(integration_params.get("report_step", 0))
    if report_step < 0:
        raise ValueError(f"hybrid.integration.report_step must be >= 0, got {report_step}")

    sx, sy = (float(v) for v in integration_params.get("slopes_scale", [1.0, -1.0]))
    return IntegrateRecursiveConfig(
        initial_method=initial_method,
        initial_noise=integration_params.get("initial_noise", 0.0),
        max_level=integration_params.get("max_level", 30),
        max_iter=integration_params.get("max_iter", 100000),
        conv_tol=integration_params.get("conv_tol", 0.0000005),
        slopes_scale=(sx * pixel_size, sy * pixel_size),
        verbose=debug,
        report_step=report_step,
    )


def fallback_without_hints(
    config: IntegrateRecursiveConfig, hints_fni_path: str | None
) -> IntegrateRecursiveConfig:
    """INT-01: '-initial hints' without '-hints' aborts the C solver.

    build_integration_config only sees the use_hints flag; when the hints file
    itself is missing (the multifocus did not write it), fall back to a zero
    start with a warning instead of letting the integration crash.
    """
    if config.initial_method == "hints" and hints_fni_path is None:
        logging.warning(
            "initial_method='hints' but no hints map is available; "
            "falling back to initial_method='zero'"
        )
        return dataclasses.replace(config, initial_method="zero")
    return config


def main(parameters):
    """
    Main function to execute the hybrid stereo method.
    
    This runs the complete pipeline:
    1. Multifocus stereo - depth from focus variation
    2. Photometric stereo - surface normals from lighting variation  
    3. Surface integration - height map from normals
    """

    # Define the current timestamp to name output folders and files
    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    # Define the output path
    base_output_path = parameters["experiment"]["paths"]["output"]
    data_foldername = parameters["experiment"]["paths"]["data_folder"]
    output_path = os.path.join(base_output_path, f"{current_time}_{data_foldername}")

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Configure logging to record information in the console and a file
    logging.basicConfig(
        level=logging.DEBUG,  # Minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(
                os.path.join(output_path, f"hybrid_stereo_{current_time}.log")
            ),  # File output
        ],
    )
    # O root em DEBUG também captura o DEBUG interno de bibliotecas
    # (findfont do matplotlib, chunks PNG do PIL) — silencia-as.
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Initial log for the experiment
    logging.info("Starting Hybrid stereo experiment")

    # Log the provided parameters
    log_parameters(parameters)

    # Find all files in the input directory and the focal planes (zf*) of the
    # stack. natsorted keeps numeric order even for non-zero-padded names
    # (zf15 < zf105), preserving the positional frame<->z_foc correspondence
    # required by mosaic.
    input_path = parameters["experiment"]["paths"]["input"]
    input_files_path = find_all_files(os.path.join(input_path, data_foldername))
    light_directories = collect_light_dirs(
        input_files_path, os.path.join(input_path, data_foldername)
    )
    zf_directories = common_zf_dirs(
        input_files_path, os.path.join(input_path, data_foldername), light_directories
    )

    # Resolve z_foc against the zf* folder names BEFORE saving parameters.yaml,
    # so the saved config (read by the evaluation) holds the values actually
    # used. A list from another dataset is rejected instead of silently putting
    # zMos and the absolute hints in the wrong units.
    mf_parameters = parameters["multifocus"]["parameters"]
    mf_parameters["z_foc"] = resolve_z_foc(mf_parameters.get("z_foc"), zf_directories)
    logging.info("z_foc resolved: %s", mf_parameters["z_foc"])

    # Salva o config resolvido para reuso (ex.: avaliação standalone descobre o
    # data_dir sozinha). Salvo ANTES de o dict ser mutado com arrays numpy.
    with open(os.path.join(output_path, "parameters.yaml"), "w") as f:
        yaml.safe_dump(parameters, f, sort_keys=False, allow_unicode=True)
    logging.info("Resolved parameters saved to: %s", os.path.join(output_path, "parameters.yaml"))

    # Validate the integration config up front: an invalid hybrid.integration
    # (pixel_size <= 0, initial_method='hints' without use_hints) must fail now,
    # not after the multifocus and photometric steps have run.
    integration_params = parameters.get("hybrid", {}).get("integration", {})
    integration_config = build_integration_config(
        integration_params, debug=parameters["experiment"]["settings"]["debug"]
    )

    # =========================================================================
    # Step 1: Multifocus Stereo
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 1: Multifocus Stereo")
    logging.info("=" * 60)
    
    # Process images to calculate the average for each 'zf' directory
    average_images_paths = []
    for zf_dir in zf_directories:
        # Filter relevant files in the directory
        filtered_files = natsorted(
            [file for file in input_files_path if f"{zf_dir}" in file and "sVal.png" in file]
        )
        image_list = read_images(filtered_files, info=False)

        # Calculate the average of the images
        average_image = calculate_avarage_of_images(image_list)

        # Save the average image in the output directory. The mean keeps the
        # input dtype (real sVal.png are 16-bit); normalize=False preserves
        # that radiometry instead of a per-frame min-max stretch that would
        # bias the focus indicator across frames.
        average_image_path = os.path.join(output_path, "multifocus_stereo", "average", "images")
        save_image(average_image_path, f"average_{zf_dir}.png", average_image, normalize=False)
        average_images_paths.append(os.path.join(average_image_path, f"average_{zf_dir}.png"))

    # Update parameters with the paths of the average images and the output directory
    parameters["filtered_dir"] = average_images_paths
    parameters["output_path_multifocus"] = os.path.join(output_path, "multifocus_stereo", "average")

    # Execute the multifocus stereo method
    multifocus_stereo_main(parameters)

    # Process images for each light directory 'L'

    for light_dir in light_directories:
        logging.info(f"... Processing light directory: {light_dir} ...")

        # Filter relevant files in the directory
        filtered_files = natsorted(
            [
                file
                for file in input_files_path
                if f"{light_dir}/zf" in file
                and "sVal.png" in file
                and os.path.basename(os.path.dirname(file)) in zf_directories
            ]
        )

        # Update parameters for the current directory
        parameters["filtered_dir"] = filtered_files
        parameters["output_path_multifocus"] = os.path.join(
            output_path, "multifocus_stereo", light_dir
        )

        # Execute the multifocus stereo method for the current directory
        multifocus_stereo_main(parameters)

    # =========================================================================
    # Step 2: Photometric Stereo
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 2: Photometric Stereo")
    logging.info("=" * 60)
    
    run_photometric_stage(parameters, output_path, input_files_path)

    # =========================================================================
    # Step 3: Surface Integration
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 3: Surface Integration (Normal to Height)")
    logging.info("=" * 60)
    run_integration_stage(output_path, parameters)

    # =========================================================================
    # Step 4 (opcional): Avaliação automática contra ground truth
    # =========================================================================
    run_evaluation_stage(output_path, parameters, os.path.join(input_path, data_foldername))

    logging.info("=" * 60)
    logging.info("Hybrid stereo pipeline complete!")
    logging.info(f"Results saved to: {output_path}")
    logging.info("=" * 60)


def cli() -> None:
    """Console entry point: ``--param_file <yaml>``."""
    parser = argparse.ArgumentParser(description="Executes the hybrid stereo method.")
    parser.add_argument(
        "--param_file", type=str, required=True, help="Path to the YAML parameter file."
    )
    args = parser.parse_args()
    main(read_yaml_parameters(args.param_file))


if __name__ == "__main__":
    cli()
