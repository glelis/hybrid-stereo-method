import argparse
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from natsort import natsorted

from hybrid_stereo_method.hybrid.hints import cell_to_vertex_grid
from hybrid_stereo_method.hybrid.integrate import (
    IntegrateRecursiveConfig,
    integrate_normals_to_height,
)
from hybrid_stereo_method.infrastructure.io.image_io import (
    convert_image_array_to_fni,
    find_all_files,
    log_parameters,
    read_image,
    read_images,
    read_yaml_parameters,
    save_image,
)
from hybrid_stereo_method.infrastructure.utils import calculate_avarage_of_images
from hybrid_stereo_method.multifocus.main import main as multifocus_stereo_main
from hybrid_stereo_method.multifocus.mosaic import mosaic
from hybrid_stereo_method.photometric.main_wps import main as photometric_stereo_main


def select_files_by_parent_dir(files: list[str], dir_name: str, filename_part: str) -> list[str]:
    """Return files whose immediate parent directory is exactly ``dir_name`` and whose
    basename contains ``filename_part``, in natural order.

    Fixes MF-01: the former ``f"{zf_dir}" in file`` substring test caused "zf1" to also
    match paths under "zf10", "zf11", "zf12", etc.
    """
    return natsorted(
        [f for f in files if Path(f).parent.name == dir_name and filename_part in Path(f).name]
    )


def select_light_stack_files(files: list[str], light_dir: str) -> list[str]:
    """Return sVal.png files under <light_dir>/zf*/ in natural order.

    Exact path-component checks:
    - immediate parent name starts with "zf" (not a substring match),
    - grandparent name is exactly ``light_dir`` (no L1/zf* bleeding into L10),
    - basename contains "sVal.png".
    Deeper nesting (e.g. L0/zf1/sub/sVal.png) is excluded because the grandparent
    of the file would be zf1, not light_dir.
    """
    return natsorted(
        [
            f
            for f in files
            if Path(f).parent.name.startswith("zf")
            and Path(f).parent.parent.name == light_dir
            and "sVal.png" in Path(f).name
        ]
    )


def collect_dirs_with_prefix(files: list[str], prefix: str) -> list[str]:
    """Return unique immediate-parent directory names starting with ``prefix``,
    sorted in natural order (numeric suffix, not lexicographic).

    Fixes MF-02: ``sorted({...})`` is lexicographic, so with ≥10 directories the
    numeric order is wrong (e.g. zf1, zf10, zf11, zf12, zf2, …).
    """
    dirs = {Path(path).parent.name for path in files if Path(path).parent.name.startswith(prefix)}
    return natsorted(dirs)


def collect_light_dirs(files: list[str], data_path: str | Path) -> list[str]:
    """Return unique ``L<n>`` directory components found at ANY depth under
    ``data_path``, in natural order.

    Fixes MF-14: the previous detection looked only at the immediate parent of
    each file, which in the documented layout ``L<n>/zf<m>/sVal.png`` is always
    a ``zf*`` directory — so no lights were detected on clean datasets and the
    pipeline aborted in the photometric step.
    """
    lights: set[str] = set()
    for f in files:
        try:
            parts = Path(f).relative_to(data_path).parts
        except ValueError:
            logging.warning(
                "collect_light_dirs: %r is not under data_path %r; skipping", f, str(data_path)
            )
            continue
        for part in parts[:-1]:  # exclude the filename itself
            if re.fullmatch(r"L\d+", part):
                lights.add(part)
    return natsorted(lights)


def light_index(light_dir_name: str) -> int:
    """Índice inteiro de uma pasta de luz ``L<n>`` (``L0`` e ``L000`` -> 0).

    Mesma convenção de ``pair_mosaics_to_lights``: o pareamento luz↔linha de
    lights.npy é por ÍNDICE, não por nome literal — então a indexação dos
    mosaicos in-memory precisa ser tolerante a zero-padding (fix da
    investigação 2026-06-06: ``sMos_by_light[f"L{i}"]`` quebrava em ``L000``).
    """
    m = re.fullmatch(r"L(\d+)", light_dir_name)
    if m is None:
        raise ValueError(f"nome de pasta de luz inválido (esperado L<n>): {light_dir_name!r}")
    return int(m.group(1))


def pair_mosaics_to_lights(mosaic_paths: list[str], n_lights: int) -> list[str]:
    """Order per-light mosaics by their ``L<n>`` index and verify the indices
    are exactly ``0..n_lights-1`` (one mosaic per lights.npy row).

    Fixes CONV-6: pairing was positional (natsorted paths vs lights.npy rows)
    with only a count check; a missing/extra light dir could silently shift
    every light↔mosaic association.
    """
    indexed: dict[int, str] = {}
    for p in mosaic_paths:
        m = re.fullmatch(r"L(\d+)", Path(p).parent.name)
        if m is None:
            raise ValueError(f"Mosaic path has no L<n> parent directory: {p}")
        idx = int(m.group(1))
        if idx in indexed:
            raise ValueError(f"Duplicate mosaic for light L{idx}: {p} and {indexed[idx]}")
        indexed[idx] = p
    if set(indexed) != set(range(n_lights)):
        raise ValueError(
            f"Mosaic light indices {sorted(indexed)} do not match lights.npy "
            f"rows 0..{n_lights - 1}: every row must have exactly one L<n> mosaic"
        )
    return [indexed[i] for i in range(n_lights)]


def mosaic_export_scale(source_dtype) -> float:
    """Valor máximo da escala radiométrica da fonte, para exportar mosaicos.

    Os mosaicos float herdam as unidades do stack de entrada (ex.: 0-65535
    para sVal.png de 16 bits). PNG (uint8) e FNI (0-1) precisam ser
    reescalados por ESTE máximo — usar 255 fixo satura dados 16-bit
    (investigação 2026-06-06, F3/H3).

    Fontes inteiras (uint8/uint16, todos os datasets atuais) usam
    ``np.iinfo(dtype).max``. Fontes float caem no padrão 0-255 (legado);
    se algum dataset futuro trouxer stacks float em outra escala (ex.: TIFF
    32-bit normalizado em [0, 1]), este padrão precisará ser revisto.
    """
    dt = np.dtype(source_dtype)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 255.0


def build_integration_config(integration_params: dict, debug: bool) -> IntegrateRecursiveConfig:
    """Build the solver config from the ``hybrid.integration`` YAML section.

    Fixes INT-04/CONV-4: the slopes the C solver integrates are dimensionless
    (dZdX = -nx/nz, physical rise per physical run) summed over 1-px cells, so
    the raw heights come out in "height-per-pixel" units, while the multifocus
    hints (zMos_with_confidence.fni) are in physical ``z_foc`` units.
    ``pixel_size`` — the lateral size of one pixel in the same units as
    ``z_foc`` — reconciles the two: passing ``slopes_scale = (pixel_size,
    pixel_size)`` makes the solver multiply each slope by the physical pixel
    step, so the integrated heights come out in ``z_foc`` units, commensurable
    with the hints (which then need no scale of their own).
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

    initial_method = integration_params.get("initial_method", "zero")
    if initial_method == "hints" and not integration_params.get("use_hints", False):
        raise ValueError(
            "hybrid.integration.initial_method='hints' requires use_hints: true — "
            "without a hints map the C solver aborts (INT-01)."
        )

    if (
        integration_params.get("use_reference", False)
        and "reference_scale" not in integration_params
    ):
        logging.warning(
            "use_reference=True but hybrid.integration.reference_scale is not set: "
            "hAvg.png is uint8 (0-255) while the integrated heights are in physical "
            "units — the C error report (devE) compares incommensurable quantities "
            "(INT-06). Set reference_scale to convert gray levels to height units."
        )

    return IntegrateRecursiveConfig(
        initial_method=initial_method,
        initial_noise=integration_params.get("initial_noise", 0.0),
        max_level=integration_params.get("max_level", 30),
        max_iter=integration_params.get("max_iter", 100000),
        conv_tol=integration_params.get("conv_tol", 0.0000005),
        verbose=debug,
        slopes_scale=(pixel_size, pixel_size),
        reference_scale=float(integration_params.get("reference_scale", 1.0)),
    )


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

    # Initial log for the experiment
    logging.info("Starting Hybrid stereo experiment")

    # Log the provided parameters
    log_parameters(parameters)

    # Salva o config resolvido para reuso (ex.: avaliação standalone descobre o
    # data_dir sozinha). Salvo ANTES de o dict ser mutado com arrays numpy.
    with open(os.path.join(output_path, "parameters.yaml"), "w") as f:
        yaml.safe_dump(parameters, f, sort_keys=False, allow_unicode=True)
    logging.info("Resolved parameters saved to: %s", os.path.join(output_path, "parameters.yaml"))

    # Copy the 'sharp' folder to the output directory
    input_path = parameters["experiment"]["paths"]["input"]
    sharp_input_path = os.path.join(input_path, data_foldername, "sharp")
    if os.path.exists(sharp_input_path):
        sharp_output_path = os.path.join(output_path, "sharp")
        shutil.copytree(sharp_input_path, sharp_output_path, dirs_exist_ok=True)
        logging.info(f"Copied 'sharp' directory to: {sharp_output_path}")
    else:
        logging.warning(f"'sharp' directory not found at: {sharp_input_path}")

    # =========================================================================
    # Step 1: Multifocus Stereo
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 1: Multifocus Stereo")
    logging.info("=" * 60)

    # Find all files in the input directory
    input_files_path = find_all_files(os.path.join(input_path, data_foldername))

    # Process images to calculate the average for each 'zf' directory.
    # collect_dirs_with_prefix uses natsorted so zf1..zf12 come in numeric order
    # (fixes MF-02: sorted() would give zf1,zf10,zf11,zf12,zf2,...).
    zf_directories = collect_dirs_with_prefix(input_files_path, prefix="zf")

    average_images_paths = []
    average_float_list = []
    for zf_dir in zf_directories:
        # select_files_by_parent_dir matches on the exact parent dir name
        # (fixes MF-01: the old substring "zf1" in file also matched zf10/zf11/zf12).
        filtered_files = select_files_by_parent_dir(
            input_files_path, dir_name=zf_dir, filename_part="sVal.png"
        )
        image_list = read_images(filtered_files, info=False)

        # Calculate the average of the images in float (MF-12 fix: no uint8 quantization)
        average_float = calculate_avarage_of_images(image_list)

        # Save the average image in the output directory (visualization only — uint8 PNG)
        average_image_path = Path(output_path) / "multifocus_stereo" / "average" / "images"
        save_image(str(average_image_path), f"average_{zf_dir}.png", average_float)
        average_images_paths.append(str(average_image_path / f"average_{zf_dir}.png"))

        # Save float average as .npy for offline inspection (not consumed by the pipeline)
        np.save(
            average_image_path / f"average_{zf_dir}.npy",
            average_float,
        )

        # Collect float arrays for in-memory pass-through (MF-12 fix)
        average_float_list.append(average_float)

    # Update parameters with the paths of the average images and the output directory.
    # filtered_dir is kept for backward compatibility / visualization paths.
    # filtered_images carries the float arrays directly, bypassing the PNG round-trip.
    parameters["filtered_dir"] = average_images_paths
    parameters["filtered_images"] = average_float_list
    parameters["output_path_multifocus"] = os.path.join(output_path, "multifocus_stereo", "average")

    # Execute the multifocus stereo method and capture the output for average configuration
    iSel_avg, wSel_avg, sMos_avg, zMos_avg = multifocus_stereo_main(parameters)

    # Process images for each light directory 'L'.
    # collect_light_dirs scans L<n> components at ANY depth in the path tree, so it
    # works on the documented layout L<n>/zf<m>/sVal.png where the immediate parent is
    # always zf* (fixes MF-14: collect_dirs_with_prefix only looked at the immediate
    # parent, finding nothing on clean datasets).
    light_directories = collect_light_dirs(
        input_files_path, os.path.join(input_path, data_foldername)
    )

    # Extract configuration for the mosaic
    zFoc = parameters["multifocus"]["parameters"]["z_foc"]
    interpolation_type = parameters["multifocus"]["parameters"]["interpolation"]

    # PS-07: collect float mosaics in memory, keyed by light index (int), to bypass
    # the uint8 PNG round-trip when feeding the photometric step.
    # Indexed by integer to be tolerant of zero-padding (L000 -> 0), aligning with
    # pair_mosaics_to_lights / lights.npy rows (fix investigação 2026-06-06).
    sMos_by_light: dict[int, np.ndarray] = {}

    for light_dir in light_directories:
        logging.info(f"... Processing light directory: {light_dir} ...")

        # Filter relevant files in the directory.
        # select_light_stack_files uses exact path-component checks so "L1" cannot
        # bleed into "L10" (fixes substring match introduced before MF-01/MF-02).
        filtered_files = select_light_stack_files(input_files_path, light_dir)

        output_path_multifocus = os.path.join(output_path, "multifocus_stereo", light_dir)
        if not os.path.exists(output_path_multifocus):
            os.makedirs(output_path_multifocus)

        # Read images
        image_list = read_images(filtered_files, info=True)
        image_stack = np.asarray(image_list)

        # Generate the mosaic for this light, given the mapping `iSel_avg` and configuration `zFoc`.
        # MF-04: pass wSel=wSel_avg so confidence-0 pixels are masked (zMos→NaN, sMos→nearest frame).
        # Only sMos_light is used here (the per-light all-in-focus image for the PS step); the
        # depth map _ is discarded (the shared average zMos is already exported from the avg run).
        logging.info("...... Generating mosaic from average iSel ...")
        sMos_light, _ = mosaic(iSel_avg, image_stack, zFoc, interpolation_type, wSel=wSel_avg)

        # PS-07: keep the float mosaic in memory (H,W,3) — PNG stays visualization only.
        sMos_by_light[light_index(light_dir)] = sMos_light

        # Save the mosaic images to the output directory.
        # normalize=False is essential: these mosaics are the photometric stereo
        # input, and a per-image min-max stretch would destroy the cross-light
        # intensity relationships that the I = albedo * (L . N) model requires.
        # F3/H3 (investigação 2026-06-06): escalar pelo max do dtype da FONTE —
        # sMos_light está nas unidades do stack (0-65535 p/ 16-bit); 255 fixo
        # saturava o PNG (avaliação de mosaico virava artefato) e deixava o
        # FNI em 0-257 em vez de 0-1.
        export_scale = mosaic_export_scale(image_stack.dtype)
        save_image(
            output_path_multifocus,
            "sMos.png",
            sMos_light * (255.0 / export_scale),
            normalize=False,
        )
        convert_image_array_to_fni(
            sMos_light / export_scale, str(Path(output_path_multifocus) / "sMos.fni")
        )

    # =========================================================================
    # Step 2: Photometric Stereo
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 2: Photometric Stereo")
    logging.info("=" * 60)

    # Configure parameters for the photometric stereo method.
    # Select the per-light mosaics (parent dir L*), excluding the 'average' one;
    # match exact path components, not substrings, so dataset/user paths that
    # happen to contain "av" cannot break the selection.
    output_files = find_all_files(output_path)
    mosaic_paths = [
        file
        for file in output_files
        if os.path.basename(file) == "sMos.png"
        and os.path.basename(os.path.dirname(file)) != "average"
    ]
    parameters["lights_path"] = [file for file in input_files_path if "lights.npy" in file][0]

    # Log the path of the lights file
    logging.info(f"Path to lights file: {parameters['lights_path']}")

    n_lights = int(np.load(parameters["lights_path"]).shape[0])
    # CONV-6: pair light<n> -> lights.npy row n by KEY, not by sort position
    parameters["sMos_path_list"] = pair_mosaics_to_lights(mosaic_paths, n_lights)

    # PS-07: hand the float mosaics to the PS in lights.npy row order, bypassing
    # the uint8 PNG round-trip (sMos.png stays as visualization only).
    # indexado por ÍNDICE de luz (tolerante a zero-padding: L000 -> 0), alinhado
    # a pair_mosaics_to_lights / linhas de lights.npy (investigação 2026-06-06).
    parameters["sMos_images"] = [sMos_by_light[i] for i in range(n_lights)]

    # H7 (investigação 2026-06-06): informa ao solver fotométrico a escala
    # radiométrica da fonte para que os limiares absolutos (calibrados em 8-bit)
    # sejam reescalados à profundidade de bits real (ex.: 65535 p/ stacks 16-bit).
    # image_stack é o último stack do laço per-light acima; todos os lights
    # compartilham o mesmo dtype da fonte.
    parameters.setdefault("photometric", {}).setdefault("solver", {})["intensity_max"] = (
        mosaic_export_scale(image_stack.dtype)
    )

    parameters["output_path_photometric"] = os.path.join(output_path, "photometric_stereo")

    # Execute the photometric stereo method
    photometric_stereo_main(parameters)

    # =========================================================================
    # Step 3: Surface Integration
    # =========================================================================
    logging.info("=" * 60)
    logging.info("STEP 3: Surface Integration (Normal to Height)")
    logging.info("=" * 60)

    # Load the normal map from photometric stereo output
    normal_map_path = os.path.join(parameters["output_path_photometric"], "normal_map.npy")

    if os.path.exists(normal_map_path):
        logging.info(f"Loading normal map from: {normal_map_path}")
        normal_map = np.load(normal_map_path)

        # PS-06/INT-03: attach the photometric confidence as the weight channel
        # (H,W,4) so shadowed/degenerate pixels (NaN normals, confidence 0) are
        # excluded by weight instead of relying only on the C NaN backstop.
        confidence_path = os.path.join(parameters["output_path_photometric"], "confidence.npy")
        if os.path.exists(confidence_path):
            confidence = np.load(confidence_path)
            normal_map = np.concatenate(
                [normal_map, confidence[..., None].astype(normal_map.dtype)], axis=-1
            )
            logging.info(
                "Attached confidence weight channel to normal map: shape %s",
                normal_map.shape,
            )
        else:
            logging.warning("confidence.npy not found — integrating normals without weight channel")

        # Configure integration parameters.
        # build_integration_config derives slopes_scale from pixel_size so the
        # integrated heights come out in z_foc units, commensurable with the
        # multifocus hints (fixes INT-04/CONV-4).
        integration_params = parameters.get("hybrid", {}).get("integration", {})

        integration_config = build_integration_config(
            integration_params, debug=parameters["experiment"]["settings"]["debug"]
        )

        # Output directory for integration
        integration_output = os.path.join(output_path, "integration")

        hints_fni_path = None
        hints_weight = integration_params.get("hints_weight", 0.0)

        if integration_params.get("use_hints", False):
            # INT-05: build VERTEX-grid hints (H+1, W+1, 2) from the cell-grid
            # multifocus outputs, instead of letting the C side expand the cell
            # grid with a half-cell shift.
            wSel_eff = np.where(np.isfinite(zMos_avg), wSel_avg, 0.0)
            hints_vertex = cell_to_vertex_grid(zMos_avg, wSel_eff)
            os.makedirs(integration_output, exist_ok=True)
            hints_fni_path = os.path.join(integration_output, "hints_vertex.fni")
            convert_image_array_to_fni(hints_vertex, hints_fni_path)
            logging.info(f"Wrote vertex-grid hints map: {hints_fni_path}")

        reference_fni_path = None
        if integration_params.get("use_reference", False):
            # Locate hAvg.png inside input/sharp
            h_avg_path = os.path.join(input_path, data_foldername, "sharp", "hAvg.png")
            if os.path.exists(h_avg_path):
                reference_img = read_image(h_avg_path)
                reference_fni_path = os.path.join(integration_output, "hAvg.fni")
                if not os.path.exists(integration_output):
                    os.makedirs(integration_output)
                convert_image_array_to_fni(reference_img.astype(np.float32), reference_fni_path)
                logging.info(f"Generated reference map: {reference_fni_path}")
            else:
                logging.warning(f"Reference map requested but not found at: {h_avg_path}")

        logging.info("Running surface integration...")
        try:
            height_map = integrate_normals_to_height(
                normal_map=normal_map,
                output_dir=integration_output,
                output_prefix="height",
                config=integration_config,
                hints_fni_path=hints_fni_path,
                hints_weight=hints_weight,
                reference_fni_path=reference_fni_path,
            )

            # Save the height map as numpy array
            height_npy_path = os.path.join(integration_output, "height_map.npy")
            np.save(height_npy_path, height_map)
            logging.info(f"Height map saved to: {height_npy_path}")

            # Save as image for visualization
            save_image(integration_output, "height_map.png", height_map)
            logging.info("Height map visualization saved")

            logging.info(f"Height map shape: {height_map.shape}")
            logging.info(f"Height map range: [{height_map.min():.4f}, {height_map.max():.4f}]")

        except FileNotFoundError:
            logging.error("Integration executable not found.")
            logging.error("Please build the C code: cd csrc/integrate_recursive && make")
            raise
        except Exception as e:
            logging.error(f"Integration failed: {e}")
            raise
    else:
        raise FileNotFoundError(
            f"Normal map not found at: {normal_map_path} — "
            "photometric stereo did not produce its output, cannot integrate."
        )

    # =========================================================================
    # Step 4 (opcional): Avaliação automática contra ground truth
    # =========================================================================
    eval_config = parameters.get("evaluation") or {}
    if eval_config.get("enabled", False):
        logging.info("=" * 60)
        logging.info("STEP 4: Automated Evaluation")
        logging.info("=" * 60)
        try:
            from hybrid_stereo_method.evaluation.main import run_evaluation

            run_evaluation(
                output_path,
                data_dir=os.path.join(input_path, data_foldername),
                config=eval_config,
            )
        except Exception:
            # Falha na avaliação NUNCA derruba um experimento que já produziu
            # resultados (spec 2026-06-06): registre e siga.
            logging.exception("Avaliação automática falhou — resultados preservados")

    logging.info("=" * 60)
    logging.info("Hybrid stereo pipeline complete!")
    logging.info(f"Results saved to: {output_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Executes the hybrid stereo method.")
    parser.add_argument(
        "--param_file", type=str, required=True, help="Path to the YAML parameter file."
    )

    # Read arguments provided by the user
    args = parser.parse_args()

    # Read parameters from the YAML file
    parameters = read_yaml_parameters(args.param_file)

    # Execute the main function
    main(parameters)
