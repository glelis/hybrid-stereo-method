# tests/test_average_float_path.py
"""TDD for fix MF-12: float zf-averages passed in memory, .npy saved for inspection.

Unit test: calculate_avarage_of_images returns float (no quantization).
Integration-light: synthetic hybrid dataset confirms .npy files are written
and contain float arrays, and that the in-memory path runs end-to-end.
"""
import cv2
import numpy as np
import pytest

from hybrid_stereo_method.infrastructure.utils import calculate_avarage_of_images

# ---------------------------------------------------------------------------
# Unit tests — calculate_avarage_of_images dtype and precision
# ---------------------------------------------------------------------------


def test_average_of_uint8_images_returns_float_without_quantization():
    """Mean of [[100]] and [[101]] must be 100.5, not 100 (uint8 truncation)."""
    img_a = np.array([[[100, 100, 100]]], dtype=np.uint8)
    img_b = np.array([[[101, 101, 101]]], dtype=np.uint8)
    result = calculate_avarage_of_images([img_a, img_b])
    # After fix: float average, not uint8-truncated
    assert result.dtype in (np.float32, np.float64), (
        f"Expected float dtype, got {result.dtype}"
    )
    assert float(result[0, 0, 0]) == pytest.approx(100.5), (
        f"Expected 100.5, got {result[0, 0, 0]}"
    )


def test_average_of_uint8_preserves_sub_integer_precision():
    """Three images [[0]], [[1]], [[2]] -> mean=1.0 (no precision issue), but
    [[0]] and [[1]] -> mean=0.5 which uint8 would round to 0."""
    img_0 = np.array([[[0]]], dtype=np.uint8)
    img_1 = np.array([[[1]]], dtype=np.uint8)
    result = calculate_avarage_of_images([img_0, img_1])
    assert result.dtype in (np.float32, np.float64)
    assert float(result[0, 0, 0]) == pytest.approx(0.5)


def test_average_of_uint16_images_returns_float():
    """uint16 input should also return float after fix."""
    img_a = np.array([[[1000]]], dtype=np.uint16)
    img_b = np.array([[[1001]]], dtype=np.uint16)
    result = calculate_avarage_of_images([img_a, img_b])
    assert result.dtype in (np.float32, np.float64)
    assert float(result[0, 0, 0]) == pytest.approx(1000.5)


# ---------------------------------------------------------------------------
# Integration-light test — hybrid pipeline in-memory path + .npy files
# ---------------------------------------------------------------------------

needs_binary = pytest.mark.skipif(
    False,  # evaluated below after import
    reason="placeholder",
)

# Re-use slow marker from e2e test
pytestmark = pytest.mark.slow

# Check binary availability at collection time
from hybrid_stereo_method.hybrid.integrate import DEFAULT_EXECUTABLE  # noqa: E402

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

SIZE = 32
N_FRAMES = 5  # ≥3 required by argmax_fuzzy
N_LIGHTS = 3


def _build_small_dataset(root):
    """Build a minimal synthetic dataset: L<n>/zf<m>/sVal.png + lights.npy."""
    from synthetic_utils import (
        defocus_stack,
        gaussian_bump,
        normals_from_height,
        render_lambertian,
        ring_lights,
        texture,
    )

    depth = 2.0 + gaussian_bump(SIZE, amplitude=3.0)
    normals = normals_from_height(depth)
    lights = ring_lights(N_LIGHTS, tilt_deg=30.0)
    albedo_map = 80.0 + 150.0 * texture(SIZE, seed=42)
    z_foc = [float(k) for k in range(N_FRAMES)]

    data_dir = root / "synth_mf12"
    for li in range(N_LIGHTS):
        shaded = render_lambertian(normals, lights[li], albedo=albedo_map)
        stack = defocus_stack(shaded, depth, z_foc, blur_per_unit=1.5)
        for k in range(N_FRAMES):
            frame_dir = data_dir / f"L{li}" / f"zf{k}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            img = np.clip(stack[k], 0, 255).astype(np.uint8)
            cv2.imwrite(str(frame_dir / "sVal.png"), cv2.merge([img, img, img]))

    np.save(data_dir / "lights.npy", lights)

    # sharp/hAvg.png (optional, not used — just keep gabaritos=False)
    sharp_dir = data_dir / "sharp"
    sharp_dir.mkdir(parents=True, exist_ok=True)

    return depth, z_foc


@needs_binary
def test_float_averages_npy_files_written_and_pipeline_completes(tmp_path, monkeypatch):
    """Integration-light: verify .npy files exist, are float, and pipeline ends."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import hybrid_stereo_method.photometric.main_wps as main_wps_mod

    monkeypatch.setattr(main_wps_mod, "disp_normalmap", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels_3d", lambda **kw: None)

    # Spy on read_images in multifocus.main to assert the in-memory float path is taken.
    # The fallback (re-reading averaged PNGs from disk) would call read_images with a
    # list of paths ending in "average_zf<n>.png".  When filtered_images is wired
    # correctly, multifocus.main should use the in-memory list and never call
    # read_images with those paths.
    import hybrid_stereo_method.multifocus.main as multifocus_main_mod
    from hybrid_stereo_method.infrastructure.io.image_io import read_images as _real_read_images

    read_images_calls: list[list[str]] = []

    def _spy_read_images(paths, **kwargs):
        read_images_calls.append(list(paths) if paths is not None else [])
        return _real_read_images(paths, **kwargs)

    monkeypatch.setattr(multifocus_main_mod, "read_images", _spy_read_images)

    from hybrid_stereo_method.hybrid.main import main as hybrid_main

    raw = tmp_path / "raw"
    depth_gt, z_foc = _build_small_dataset(raw)

    parameters = {
        "experiment": {
            "type": "hybrid",
            "paths": {
                "input": str(raw),
                "data_folder": "synth_mf12",
                "output": str(tmp_path / "results"),
            },
            "settings": {"debug": False, "gabaritos": False},
        },
        "multifocus": {
            "focus_measure": {
                "method": "laplacian",
                "parameters": {"kernel_size": 5, "radius": None},
                "preprocessing": {
                    "square": True,
                    "smooth": True,
                    "spatial_median_filter": False,
                    "zero_border": False,
                },
            },
            "optimization": {"r_max": 2},
            "parameters": {"z_foc": z_foc, "interpolation": "linear_interpolation"},
        },
        "photometric": {
            "solver": {
                "epsilon": 1e-6,
                "shadow_threshold": 1e-3,
                "outlier_threshold_multiplier": 3,
            }
        },
        "hybrid": {
            "integration": {
                "initial_method": "zero",
                "use_hints": False,
                "use_reference": False,
                "max_iter": 5000,
                "conv_tol": 5e-6,
            }
        },
    }

    hybrid_main(parameters)

    # (wiring assertion 1) filtered_images must be a non-empty list of float arrays after
    # hybrid_main returns — hybrid/main.py mutates the dict in place.
    filtered_images = parameters.get("filtered_images")
    assert filtered_images is not None, (
        "parameters['filtered_images'] is None after hybrid_main — "
        "the in-memory float path was not wired"
    )
    assert len(filtered_images) == N_FRAMES, (
        f"Expected {N_FRAMES} float arrays in filtered_images, got {len(filtered_images)}"
    )
    for i, arr in enumerate(filtered_images):
        assert np.issubdtype(arr.dtype, np.floating), (
            f"filtered_images[{i}] has dtype {arr.dtype}, expected float"
        )

    # (wiring assertion 2) multifocus.main must NOT have re-read the average PNGs from
    # disk.  If filtered_images was ignored (regression), the fallback branch calls
    # read_images with the filtered_dir list whose entries end in "average_zf<n>.png".
    fallback_png_calls = [
        call
        for call in read_images_calls
        if any(str(p).endswith(".png") and "average_zf" in str(p) for p in call)
    ]
    assert fallback_png_calls == [], (
        f"multifocus.main called read_images with average PNG paths — "
        f"the in-memory float path was NOT taken (fallback triggered):\n{fallback_png_calls}"
    )

    out_dirs = list((tmp_path / "results").glob("*_synth_mf12"))
    assert len(out_dirs) == 1, f"Expected 1 output dir, got {out_dirs}"
    out_dir = out_dirs[0]

    # (a) .npy files exist alongside PNGs and contain float arrays
    avg_dir = out_dir / "multifocus_stereo" / "average" / "images"
    npy_files = sorted(avg_dir.glob("average_zf*.npy"))
    assert len(npy_files) == N_FRAMES, (
        f"Expected {N_FRAMES} .npy files, got {len(npy_files)}: {npy_files}"
    )
    for npy_path in npy_files:
        arr = np.load(npy_path)
        assert np.issubdtype(arr.dtype, np.floating), (
            f"{npy_path.name} has dtype {arr.dtype}, expected float"
        )

    # (b) pipeline completes: height_map.npy must exist
    height_path = out_dir / "integration" / "height_map.npy"
    assert height_path.exists(), "Pipeline did not produce height_map.npy"

    height = np.load(height_path)
    finite_frac = np.isfinite(height).mean()
    assert finite_frac > 0.90, (
        f"Only {finite_frac:.1%} of height map is finite — pipeline may have failed"
    )
