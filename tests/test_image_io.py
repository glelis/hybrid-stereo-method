"""Testes de round-trip de save_image (radiometria preservada com normalize=False)."""
import cv2
import numpy as np

from hybrid_stereo_method.infrastructure.io.image_io import save_image


def test_save_image_no_normalize_preserves_uint16(tmp_path):
    """sVal reais são PNGs de 16 bits: a média por luz deve sobreviver ao round-trip.

    Regressão do bug em que normalize=False clipava tudo a [0, 255] e saturava
    96% dos pixels do stack "average" em branco puro.
    """
    rng = np.random.default_rng(42)
    img = rng.integers(0, 65536, size=(8, 8, 3), dtype=np.uint16)

    save_image(tmp_path, "avg.png", img, normalize=False)

    out = cv2.imread(str(tmp_path / "avg.png"), cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, img)


def test_save_image_no_normalize_preserves_uint8(tmp_path):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

    save_image(tmp_path, "img.png", img, normalize=False)

    out = cv2.imread(str(tmp_path / "img.png"), cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_save_image_no_normalize_float_clips_to_uint8(tmp_path):
    """Floats sem dtype inteiro declarado seguem o comportamento antigo (0-255 uint8)."""
    img = np.array([[-3.0, 0.4, 128.6, 300.0]], dtype=np.float32)

    save_image(tmp_path, "f.png", img, normalize=False)

    out = cv2.imread(str(tmp_path / "f.png"), cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.array([[0, 0, 129, 255]], dtype=np.uint8))
