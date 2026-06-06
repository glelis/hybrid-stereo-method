"""A4 — Estimador WPS com entrada perfeita (sharp GT por luz):
quanto do erro das normais vem dos mosaicos vs do estimador?
"""

from __future__ import annotations

import numpy as np

from hybrid_stereo_method.evaluation.loaders import find_sharp_dir, load_normals_gt
from hybrid_stereo_method.evaluation.metrics import angular_error_deg
from hybrid_stereo_method.infrastructure.io.image_io import read_image, read_yaml_parameters
from hybrid_stereo_method.infrastructure.utils import convert_to_grayscale
from hybrid_stereo_method.photometric.main_wps import reconcile_lights
from hybrid_stereo_method.photometric.wps import estimate_normals_argmax_lstsq_robust
from scripts.investigation.common import RAW, RESULTS, STACK_ROOT, save_json, summarize


def main() -> None:
    params = read_yaml_parameters("configs/hb_experiment.yaml")
    wps_params = params["photometric"]["solver"]
    lights = np.load(RAW / "lights.npy")
    lights = reconcile_lights(
        lights, flip_y=bool(params["photometric"].get("flip_lights_y", False))
    )

    images = []
    for i in range(lights.shape[0]):
        img = read_image(STACK_ROOT / f"L{i:03d}" / "sharp" / "sVal.png")
        images.append(convert_to_grayscale(np.asarray(img, dtype=np.float64)))

    normals_sharp, _albedo, conf, _sel = estimate_normals_argmax_lstsq_robust(
        images, lights, wps_params
    )

    n_gt, fg = load_normals_gt(find_sharp_dir(RESULTS, RAW))
    n_mosaic = np.load(RESULTS / "photometric_stereo" / "normal_map.npy")[..., :3]
    ae_sharp = angular_error_deg(normals_sharp[..., :3], n_gt)
    ae_mosaic = angular_error_deg(n_mosaic, n_gt)

    report = {
        "input_intensity_range": [float(min(i.min() for i in images)),
                                  float(max(i.max() for i in images))],
        "wps_thresholds": {k: wps_params[k] for k in
                           ("shadow_absolute_threshold", "saturation_threshold")
                           if k in wps_params},
        "angular_error_from_sharp_object": summarize("erro (sharp GT, objeto)", ae_sharp[fg]),
        "angular_error_from_mosaic_object": summarize("erro (mosaico, objeto)", ae_mosaic[fg]),
    }
    save_json("a4_normals_from_sharp.json", report)


if __name__ == "__main__":
    main()
