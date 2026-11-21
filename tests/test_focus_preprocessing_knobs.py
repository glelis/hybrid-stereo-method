"""Knobs de força do denoise da medida de foco (median_ksize, smooth_sigma)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from hybrid_stereo_method.multifocus.indicators.applicator import focus_indicator

DATA = Path(__file__).parent / "fixtures"


@pytest.fixture
def stack():
    return np.load(DATA / "focus_stack_seed42.npy")


def _run(stack, **kw):
    return focus_indicator(stack, "fourier", 31, 0.05, False, True, True, False, **kw)


def test_defaults_bit_identical(stack):
    # fixture gravado com o código ANTES dos knobs (smooth + median 5)
    expected = np.load(DATA / "focus_indicator_default_seed42.npy")
    assert np.array_equal(_run(stack), expected)
    assert np.array_equal(_run(stack, median_ksize=5, smooth_sigma=None), expected)


def test_median_ksize_3_differs(stack):
    out3 = _run(stack, median_ksize=3)
    out5 = _run(stack, median_ksize=5)
    assert out3.shape == out5.shape
    assert not np.array_equal(out3, out5)


def test_invalid_median_ksize_raises(stack):
    with pytest.raises(ValueError, match="3 or 5"):
        _run(stack, median_ksize=7)


def test_smooth_sigma_larger_is_smoother(stack):
    def roughness(out):
        return float(np.mean([cv2.Laplacian(f.astype(np.float64), cv2.CV_64F).var() for f in out]))

    assert roughness(_run(stack, smooth_sigma=3.0)) < roughness(_run(stack, smooth_sigma=1.0))
