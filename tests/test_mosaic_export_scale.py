"""Export de mosaico dtype-aware: PNG/FNI escalados pelo max do dtype da fonte.

Regressão da investigação 2026-06-06 (F3): sMos.png era np.clip(float16bit, 0, 255)
-> imagem ~toda branca; sMos.fni era /255 -> range 0-257 para dados 16-bit.
"""

import cv2
import numpy as np

from hybrid_stereo_method.hybrid.main import mosaic_export_scale


def test_export_scale_uint16_source():
    assert mosaic_export_scale(np.uint16) == 65535.0


def test_export_scale_uint8_source():
    assert mosaic_export_scale(np.uint8) == 255.0


def test_export_scale_float_source_defaults_to_255():
    # fonte float: assume já em escala 0-255 (comportamento legado)
    assert mosaic_export_scale(np.float64) == 255.0


def test_16bit_mosaic_png_is_not_saturated(tmp_path):
    from hybrid_stereo_method.infrastructure.io.image_io import save_image

    smos = np.full((8, 8, 3), 32768.0)  # cinza médio em unidades 16-bit
    scale = mosaic_export_scale(np.uint16)
    save_image(str(tmp_path), "sMos.png", smos * (255.0 / scale), normalize=False)
    png = cv2.imread(str(tmp_path / "sMos.png"), cv2.IMREAD_UNCHANGED)
    assert 120 <= png.mean() <= 135  # cinza médio, não branco saturado
