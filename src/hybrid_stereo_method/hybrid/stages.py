"""Estágios 2–4 do pipeline híbrido (fotométrico, integração, avaliação).

Extraídos de hybrid/main.py para que hybrid/rerun.py possa reexecutar o
pipeline a partir de um estágio, reaproveitando as saídas de um run anterior.
O comportamento é o mesmo do main: cada função lê e grava em ``output_path``.
"""
import logging
import os

import numpy as np

from hybrid_stereo_method.hybrid.integrate import integrate_normals_to_height
from hybrid_stereo_method.infrastructure.io.image_io import find_all_files, save_image
from hybrid_stereo_method.photometric.main_wps import main as photometric_stereo_main


def run_photometric_stage(parameters: dict, output_path: str, input_files_path: list[str]) -> None:
    """Normais por fotométrico a partir dos sMos.npy por luz já em output_path."""
    # Lazy import: main importa este módulo.
    from hybrid_stereo_method.hybrid.main import select_per_light_smos

    # Use the lossless sMos.npy files: the PNGs are min-max normalized per
    # image, which destroys the inter-light intensity ratios that the
    # photometric solver depends on. Busca dentro de multifocus_stereo (que
    # pode ser um symlink num rerun: os.walk segue o diretório de topo).
    output_files = find_all_files(os.path.join(output_path, "multifocus_stereo"))
    parameters["sMos_path_list"] = select_per_light_smos(output_files)
    parameters["output_path_photometric"] = os.path.join(output_path, "photometric_stereo")
    parameters["lights_path"] = [file for file in input_files_path if "lights.npy" in file][0]

    logging.info(f"Path to lights file: {parameters['lights_path']}")
    photometric_stereo_main(parameters)


def run_integration_stage(output_path: str, parameters: dict) -> np.ndarray | None:
    """Integra normal_map (+ confiança) e, se pedido, os hints do zMos.

    Devolve o mapa de altura (vertex-grid) ou None se a integração não rodou.
    """
    from hybrid_stereo_method.hybrid.main import build_integration_config, fallback_without_hints

    integration_params = parameters.get("hybrid", {}).get("integration", {})
    integration_config = build_integration_config(
        integration_params, debug=parameters["experiment"]["settings"].get("debug", False)
    )
    photometric_dir = os.path.join(output_path, "photometric_stereo")
    normal_map_path = os.path.join(photometric_dir, "normal_map.npy")
    if not os.path.exists(normal_map_path):
        logging.warning(f"Normal map not found at: {normal_map_path}")
        logging.warning("Skipping surface integration step")
        return None

    logging.info(f"Loading normal map from: {normal_map_path}")
    normal_map = np.load(normal_map_path)

    # Attach the photometric confidence as the 4th channel: the C solver
    # uses it as the per-pixel reliability weight of each equation.
    # Without it every non-NaN normal enters the system with weight 1.
    confidence_map_path = os.path.join(photometric_dir, "confidence_map.npy")
    if os.path.exists(confidence_map_path):
        confidence_map = np.load(confidence_map_path)
        normal_map = np.concatenate((normal_map, confidence_map[..., np.newaxis]), axis=-1)
        logging.info("Using photometric confidence as integration weights")
    else:
        logging.warning(
            f"Confidence map not found at: {confidence_map_path}; "
            "all normals will be weighted equally"
        )

    integration_output = os.path.join(output_path, "integration")
    hints_fni_path = None
    hints_weight = integration_params.get("hints_weight", 0.1)
    if integration_params.get("use_hints", False):
        hints_file = os.path.join(
            output_path, "multifocus_stereo", "average", "zMos_with_confidence.fni"
        )
        if os.path.exists(hints_file):
            hints_fni_path = hints_file
            logging.info(f"Found hints map: {hints_fni_path}")
        else:
            logging.warning(f"Hints map requested but not found at: {hints_file}")

    integration_config = fallback_without_hints(integration_config, hints_fni_path)

    logging.info("Running surface integration...")
    try:
        height_map = integrate_normals_to_height(
            normal_map=normal_map,
            output_dir=integration_output,
            output_prefix="height",
            config=integration_config,
            hints_fni_path=hints_fni_path,
            hints_weight=hints_weight,
        )
    except FileNotFoundError as e:
        logging.error(f"Integration executable not found: {e}")
        logging.error("Please build the C code: cd csrc/integrate_recursive && make")
        return None
    except Exception as e:
        logging.error(f"Integration failed: {e}")
        return None

    height_npy_path = os.path.join(integration_output, "height_map.npy")
    np.save(height_npy_path, height_map)
    logging.info(f"Height map saved to: {height_npy_path}")
    save_image(integration_output, "height_map.png", height_map)
    logging.info("Height map visualization saved")
    logging.info(f"Height map shape: {height_map.shape}")
    logging.info(f"Height map range: [{height_map.min():.4f}, {height_map.max():.4f}]")
    return height_map


def run_evaluation_stage(output_path: str, parameters: dict, data_dir: str) -> None:
    """Avaliação automática contra o GT, se ``evaluation.enabled``."""
    eval_config = parameters.get("evaluation") or {}
    if not eval_config.get("enabled", False):
        return
    logging.info("=" * 60)
    logging.info("STEP 4: Automated Evaluation")
    logging.info("=" * 60)
    try:
        from hybrid_stereo_method.evaluation.main import run_evaluation

        run_evaluation(output_path, data_dir=data_dir, config=eval_config)
    except Exception:
        # Falha na avaliação NUNCA derruba um experimento que já produziu
        # resultados (spec 2026-06-06): registre e siga.
        logging.exception("Avaliação automática falhou — resultados preservados")
