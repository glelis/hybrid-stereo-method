"""MF-09: the Fourier focus indicator must be local (no global-support filtering).

The legacy implementation filtered the whole image in the frequency domain via
FFT -> Gaussian high-pass mask -> IFFT. The Gaussian mask itself does not ring
(its spatial kernel is a compact ``delta - Gaussian`` with
``sigma = 1 / (2*pi*radius)`` pixels), but the FFT's *circular* convolution
wrapped strong border structures around to the opposite border, so border
pixels inherited "focus" from the other side of the image.

These tests pin the corrected behavior:
- no wraparound leakage from one border to the opposite one;
- interior equivalence with the legacy FFT formulation (behavior preserved);
- compact impulse response (locality);
- basic sharp-vs-blurred discrimination.
"""

import cv2
import numpy as np
import pytest

from hybrid_stereo_method.multifocus.indicators.fourier import (
    calculate_fourier_focus_indicator,
    create_gaussian_elliptical_mask,
)

RADIUS = 0.1  # default used in configs (hb_experiment.yaml)


def legacy_fft_indicator(image: np.ndarray, radius: float) -> np.ndarray:
    """Reference copy of the pre-MF-09 global-FFT implementation."""
    image = image / 255.0
    f_transform = np.fft.fft2(image)
    f_centered = np.fft.fftshift(f_transform)
    height, width = image.shape
    mask = create_gaussian_elliptical_mask(height, width, radius)
    f_filtered = f_centered * mask
    f_inverse = np.fft.ifftshift(f_filtered)
    return np.abs(np.real(np.fft.ifft2(f_inverse)))


def test_no_wraparound_leakage_to_opposite_border():
    """A bright band at the left border must not produce focus response at the
    right border (the legacy FFT version leaked ~0.37 there via circular
    convolution, vs ~1e-9 in the interior)."""
    image = np.zeros((128, 128))
    image[:, :5] = 255.0

    focus_map = calculate_fourier_focus_indicator(image, RADIUS)

    edge_response = focus_map[:, :8].max()
    opposite_border_response = focus_map[:, 120:].max()

    assert edge_response > 0.1, "sanity: the edge itself must respond"
    assert opposite_border_response < 1e-3 * edge_response, (
        f"wraparound leakage at opposite border: {opposite_border_response:.3e} "
        f"(edge response {edge_response:.3e})"
    )


def test_interior_matches_legacy_fft_formulation():
    """In the image interior (away from borders) the local filter must match
    the legacy frequency-domain Gaussian high-pass to numerical precision."""
    rng = np.random.default_rng(42)
    image = rng.random((200, 300)) * 255.0

    for radius in (0.05, 0.1):  # values used in configs/
        new = calculate_fourier_focus_indicator(image, radius)
        old = legacy_fft_indicator(image, radius)

        margin = 30
        interior_new = new[margin:-margin, margin:-margin]
        interior_old = old[margin:-margin, margin:-margin]

        rel_err = np.linalg.norm(interior_new - interior_old) / np.linalg.norm(interior_old)
        assert rel_err < 1e-3, f"radius={radius}: interior rel err {rel_err:.3e}"


def test_impulse_response_is_compact():
    """Response to a single bright pixel must vanish a few sigmas away
    (sigma = 1/(2*pi*radius) ~ 1.6 px for radius=0.1)."""
    image = np.zeros((128, 128))
    image[64, 64] = 255.0

    focus_map = calculate_fourier_focus_indicator(image, RADIUS)
    peak = focus_map[64, 64]

    assert peak > 0.5, "sanity: impulse must respond at its own pixel"
    far_response = focus_map[64, 64 + 15]
    assert far_response < 1e-6 * peak, (
        f"non-local impulse response: {far_response:.3e} at 15 px (peak {peak:.3e})"
    )


def test_sharp_texture_scores_higher_than_blurred():
    """The indicator must still discriminate focus: a sharp texture patch must
    score higher than the same patch defocus-blurred."""
    rng = np.random.default_rng(7)
    sharp = (rng.random((100, 100)) * 255.0).astype(np.float64)
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3.0)

    score_sharp = calculate_fourier_focus_indicator(sharp, RADIUS).mean()
    score_blurred = calculate_fourier_focus_indicator(blurred, RADIUS).mean()

    assert score_sharp > 2.0 * score_blurred, (
        f"sharp {score_sharp:.4f} should clearly exceed blurred {score_blurred:.4f}"
    )


def test_output_shape_dtype_and_nonnegativity():
    rng = np.random.default_rng(0)
    image = rng.random((64, 80)) * 255.0

    focus_map = calculate_fourier_focus_indicator(image, RADIUS)

    assert focus_map.shape == image.shape
    assert np.issubdtype(focus_map.dtype, np.floating)
    assert np.all(focus_map >= 0), "applicator asserts non-negative indicators"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
