"""Thresholds absolutos do WPS escalam com a profundidade de bits da fonte (H7).

Investigação 2026-06-06: shadow_absolute_threshold/saturation_threshold são
calibrados em unidades 8-bit (0-255). Sem reescala, saturation_threshold=250
descarta ~90% de dados 16-bit como 'saturados'. intensity_max reescala os
limiares pela razão intensity_max/255 (default 255 => fator 1, retrocompatível).
"""

import numpy as np

from hybrid_stereo_method.photometric.wps import estimate_normals_argmax_lstsq_robust


def _three_light_setup():
    # 3 luzes ortonormais: sistema bem-posto para um único pixel.
    lights = np.eye(3)
    return lights


def _solve_single_pixel(intensities, lights, wps_params):
    # imagem 1x1 por luz
    images = [np.array([[v]], dtype=np.float64) for v in intensities]
    normals, _albedo, confidence, _sel = estimate_normals_argmax_lstsq_robust(
        images, lights, wps_params
    )
    return normals[0, 0], float(confidence[0, 0])


def test_16bit_bright_pixel_survives_with_intensity_max():
    """A 60000 (16-bit) NÃO é saturação quando intensity_max=65535 (limiar efetivo 64250)."""
    lights = _three_light_setup()
    params = {"saturation_threshold": 250.0, "intensity_max": 65535.0}
    normal, conf = _solve_single_pixel([60000.0, 60000.0, 60000.0], lights, params)
    assert not np.isnan(normal).any()
    assert conf > 0.0


def test_16bit_saturated_pixel_rejected_with_intensity_max():
    """A 65000 (16-bit) É saturação quando intensity_max=65535 (limiar efetivo 64250) ->
    <3 medições válidas -> normal NaN."""
    lights = _three_light_setup()
    params = {"saturation_threshold": 250.0, "intensity_max": 65535.0}
    normal, conf = _solve_single_pixel([65000.0, 65000.0, 65000.0], lights, params)
    assert np.isnan(normal).all()
    assert conf == 0.0


def test_default_intensity_max_is_8bit_backwards_compatible():
    """Sem intensity_max (default 255): saturation_threshold=250 rejeita 251 (8-bit)."""
    lights = _three_light_setup()
    params = {"saturation_threshold": 250.0}
    normal, _ = _solve_single_pixel([251.0, 251.0, 251.0], lights, params)
    assert np.isnan(normal).all()


def test_shadow_absolute_scales_with_intensity_max():
    """shadow_absolute_threshold=2.0 com intensity_max=65535 (efetivo ~514):
    pixel a 300 (16-bit) vira sombra -> rejeitado."""
    lights = _three_light_setup()
    params = {"shadow_absolute_threshold": 2.0, "intensity_max": 65535.0}
    normal, _ = _solve_single_pixel([300.0, 300.0, 300.0], lights, params)
    assert np.isnan(normal).all()
