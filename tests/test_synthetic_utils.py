# tests/test_synthetic_utils.py
import numpy as np

from synthetic_utils import (
    affine_fit_rmse,
    defocus_stack,
    gaussian_bump,
    normals_from_height,
    ramp,
    render_lambertian,
    ring_lights,
    texture,
)


def test_normals_are_unit_and_point_to_camera():
    z = gaussian_bump(32, amplitude=4.0)
    n = normals_from_height(z)
    assert n.shape == (32, 32, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-12)
    assert (n[..., 2] > 0).all()  # nz aponta para a câmera


def test_ramp_normals_match_analytic():
    # z = ax*x + ay*y  ->  n ∝ (-ax, -ay, 1)
    ax, ay = 0.3, -0.2
    n = normals_from_height(ramp(16, ax=ax, ay=ay))
    expected = np.array([-ax, -ay, 1.0])
    expected /= np.linalg.norm(expected)
    interior = n[2:-2, 2:-2]  # np.gradient é unilateral nas bordas
    np.testing.assert_allclose(interior, np.broadcast_to(expected, interior.shape), atol=1e-10)


def test_ring_lights_unit_norm():
    lights = ring_lights(6, tilt_deg=30.0)
    assert lights.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(lights, axis=-1), 1.0, atol=1e-12)
    assert (lights[:, 2] > 0).all()


def test_render_lambertian_range_and_max():
    z = gaussian_bump(32, amplitude=4.0)
    n = normals_from_height(z)
    img = render_lambertian(n, np.array([0.0, 0.0, 1.0]), albedo=200.0)
    assert img.min() >= 0.0
    assert img.max() <= 200.0 + 1e-9


def test_defocus_stack_sharpest_frame_tracks_depth():
    size, n_frames = 48, 7
    z_foc = list(range(n_frames))
    depth = np.full((size, size), 4.0)  # plano em z=4 -> frame 4 é o mais nítido
    sharp = 255.0 * texture(size, seed=0)
    stack = defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5)
    assert stack.shape == (n_frames, size, size)
    # variância do laplaciano como proxy de nitidez por frame
    import cv2

    sharpness = [cv2.Laplacian(f, cv2.CV_64F).var() for f in stack]
    assert int(np.argmax(sharpness)) == 4


def test_affine_fit_rmse_exact_for_affine_pair():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, (10, 10))
    est = 3.0 * gt - 7.0
    rmse, _ = affine_fit_rmse(est, gt)
    assert rmse < 1e-12
