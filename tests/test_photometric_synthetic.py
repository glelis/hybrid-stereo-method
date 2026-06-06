# tests/test_photometric_synthetic.py
"""Fase 3.1 — fotométrico com ground truth analítico.

Geração e estimação usam o MESMO frame de coordenadas, então estes testes validam
a matemática interna do solver (não as convenções entre estágios — Task 9 cuida disso).
Falha = achado PS-xx; não conserte o teste.
"""
import numpy as np
import pytest

from synthetic_utils import gaussian_bump, normals_from_height, ring_lights, render_lambertian

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


@pytest.mark.xfail(strict=True, reason="PS-01: albedo = ||L_sel @ n̂|| cresce com nº de luzes (wps.py:198) — XPASS = corrigido")
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


@pytest.mark.xfail(strict=True, reason="PS-03: loop de outliers usa 3×média (não robusta); saturação em 2/8 luzes infla r_avg e NÃO é descartada — XPASS = corrigido")
def test_wps_robust_to_saturation():
    """Satura (clip) as intensidades em 60% do máximo em DUAS imagens — o laço
    robusto deve descartar os outliers e manter o erro angular baixo."""
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=5.0))
    lights = ring_lights(8, tilt_deg=30.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]
    cap = 0.6 * max(img.max() for img in images)
    images[0] = np.minimum(images[0], cap)
    images[1] = np.minimum(images[1], cap)

    normals, _, _, _ = estimate_normals_argmax_lstsq_robust(images, lights, {})
    valid = np.isfinite(normals).all(axis=-1)
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\nsaturado: erro angular médio = {ang.mean():.3f}°")
    assert ang.mean() < 5.0, f"robustez insuficiente a saturação: {ang.mean():.2f}°"


def test_wps_rejects_8bit_floor_shadows():
    """PS-02: com o piso de 8 bits (sombras viram 1/255 em vez de 0), o limiar
    relativo 1e-3 não rejeita nada e o erro angular sobe para ~3.6° médio.
    Com limiar ABSOLUTO de sombra, os pixels sombreados são rejeitados e o
    erro volta a < 0.5°."""
    size = 48
    n_gt = normals_from_height(gaussian_bump(size, amplitude=10.0, sigma_frac=0.15))
    lights = ring_lights(5, tilt_deg=75.0)
    images = [render_lambertian(n_gt, light, albedo=200.0) for light in lights]
    floor = 255.0 / 255.0  # menor valor não-nulo de um sensor 8 bits, escala 0-255
    images = [np.maximum(img, floor) for img in images]

    normals, _, _, _ = estimate_normals_argmax_lstsq_robust(
        images, lights, {"shadow_absolute_threshold": 2.0}
    )
    valid = np.isfinite(normals).all(axis=-1)
    assert valid.any()
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\npiso 8-bit + limiar absoluto: erro médio = {ang.mean():.3f}°")
    assert ang.mean() < 0.5, f"sombras de piso 8-bit não rejeitadas: {ang.mean():.2f}°"


def test_wps_rejects_saturated_measurements():
    """PS-02: medições saturadas (>= saturation_threshold) devem sair do lstsq."""
    size = 32
    n_gt = normals_from_height(gaussian_bump(size, amplitude=5.0))
    lights = ring_lights(8, tilt_deg=30.0)
    images = [render_lambertian(n_gt, light, albedo=300.0) for light in lights]
    images_sat = [np.minimum(img, 255.0) for img in images]  # clipe do sensor

    normals, _, _, _ = estimate_normals_argmax_lstsq_robust(
        images_sat, lights, {"saturation_threshold": 250.0}
    )
    valid = np.isfinite(normals).all(axis=-1)
    ang = _angular_error_deg(normals, n_gt, valid)
    print(f"\nsaturação tratada: erro médio = {ang.mean():.3f}°")
    assert ang.mean() < 1.0


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
