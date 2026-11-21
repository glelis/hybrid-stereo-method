# tests/test_photometric_synthetic.py
"""Fase 3.1 — fotométrico com ground truth analítico.

Geração e estimação usam o MESMO frame de coordenadas, então estes testes validam
a matemática interna do solver (não as convenções entre estágios — Task 9 cuida disso).
Falha = achado PS-xx; não conserte o teste.
"""
import numpy as np
from synthetic_utils import gaussian_bump, normals_from_height, render_lambertian, ring_lights

from hybrid_stereo_method.photometric.wps import estimate_normals_argmax_lstsq_robust


def _angular_error_deg(n_est, n_gt, valid):
    cos = np.clip(np.sum(n_est[valid] * n_gt[valid], axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def test_wps_recovers_normals_clean_data():
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=5.0))
    lights = ring_lights(6, tilt_deg=30.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]

    normals, albedo, confidence, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})

    valid = np.isfinite(normals).all(axis=-1)
    assert valid.mean() > 0.99, f"só {valid.mean():.1%} de pixels válidos em dados limpos"
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\nlimpo: erro angular médio = {ang.mean():.3f}°, p95 = {np.percentile(ang, 95):.3f}°")
    assert ang.mean() < 1.0, f"erro angular médio {ang.mean():.2f}° (esperado < 1° sem ruído)"


def test_wps_albedo_recovers_true_albedo():
    """Modelo I = rho * (L.n): com rho = 200 constante, o albedo estimado deve
    ser ~200 e NÃO depender do número de luzes. Falha aqui confirma o lead do
    albedo = ||L_sel @ n_normalizado|| (wps.py:198)."""
    size = 24
    n_gt = normals_from_height(gaussian_bump(size, amplitude=3.0))
    rho = 200.0
    for n_lights in (4, 8):
        lights = ring_lights(n_lights, tilt_deg=30.0)
        images = [render_lambertian(n_gt, light, albedo=rho) for light in lights]
        _, albedo, _, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
        med = float(np.median(albedo[albedo > 0]))
        print(f"\nn_lights={n_lights}: albedo mediano = {med:.1f} (verdadeiro: {rho})")
        assert abs(med - rho) < 0.1 * rho, (
            f"albedo mediano {med:.1f} != {rho} com {n_lights} luzes — "
            "se cresce com sqrt(n_lights), confirma o achado do albedo"
        )














def test_confidence_invariant_to_radiometric_gain():
    """PS-05: a confiança não pode depender de a entrada estar em 0-255 ou 0-1."""
    size = 24
    n_gt = normals_from_height(gaussian_bump(size, amplitude=3.0))
    lights = ring_lights(6, tilt_deg=30.0)
    rng = np.random.default_rng(0)
    images255 = [
        render_lambertian(n_gt, light, albedo=200.0) + rng.normal(0, 1.0, (size, size))
        for light in lights
    ]
    images01 = [img / 255.0 for img in images255]

    _, _, conf255, _ = estimate_normals_argmax_lstsq_robust(images255, lights, {})
    _, _, conf01, _ = estimate_normals_argmax_lstsq_robust(images01, lights, {})
    np.testing.assert_allclose(conf255, conf01, rtol=1e-3, atol=1e-4)


def test_wps_shadowed_pixels_flagged_not_garbage():
    """Luz rasante (tilt 75°) numa superfície inclinada gera attached shadows
    (n.l < 0 -> I = 0). Pixels com < 3 medições válidas devem virar NaN+conf 0,
    e os demais devem continuar precisos."""
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=10.0, sigma_frac=0.15))
    lights = ring_lights(5, tilt_deg=75.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]

    normals, _, confidence, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
    valid = np.isfinite(normals).all(axis=-1)
    assert (confidence[~valid] == 0).all(), "pixel inválido com confiança > 0"
    if valid.any():
        ang = _angular_error_deg(normals, n_gt, valid)
        print(f"\nsombras: válidos = {valid.mean():.1%}, erro médio nos válidos = {ang.mean():.2f}°")
        assert ang.mean() < 10.0


def test_disp_functions_do_not_mutate_input_and_are_headless_safe(tmp_path):
    """PS-11: disp_* trocavam canais IN-PLACE numa view do array do chamador e
    bloqueavam com cv2.imshow+waitKey(0). Default: só salvar, sem display."""
    from hybrid_stereo_method.photometric.visualization import (
        disp_channels,
        disp_normalmap,
    )

    normals = np.random.default_rng(0).uniform(-1, 1, (8, 8, 3))
    normals[0, 0] = np.nan  # sombra: não pode quebrar a visualização
    before = normals.copy()

    disp_normalmap(normal=normals, height=8, width=8, save_path=str(tmp_path))
    disp_channels(normal_in=normals, height=8, width=8, save_path=str(tmp_path))

    np.testing.assert_array_equal(np.nan_to_num(normals), np.nan_to_num(before)), (
        "disp_* mutou o array do chamador (PS-11)"
    )
    assert (tmp_path / "normal_map.png").exists()
    assert (tmp_path / "Channels.png").exists()


