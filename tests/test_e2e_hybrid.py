# tests/test_e2e_hybrid.py
"""Fase 3.3 — pipeline híbrido completo sobre dataset sintético com ground truth.

Roda sempre (não é sob demanda): o RMSE com fit afim é a LINHA DE BASE do estado
atual do pipeline, registrada no relatório. O fit afim absorve escala/offset/sinal
globais — as convenções de sinal são decididas pela Task 9, não aqui.
"""
import cv2
import numpy as np
import pytest

from synthetic_utils import (
    affine_fit_rmse,
    defocus_stack,
    gaussian_bump,
    normals_from_height,
    render_lambertian,
    ring_lights,
    texture,
)

from hybrid_stereo_method.hybrid.integrate import DEFAULT_EXECUTABLE

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

pytestmark = [
    needs_binary,
    pytest.mark.slow,
    # MF-14 (achado desta Task 12, raw capturado em 06-test-results.md): a detecção de
    # diretórios de luz em hybrid/main.py:122-128 só inspeciona o pai IMEDIATO de cada
    # arquivo (sempre um `zf*` no layout L<n>/zf<m>/sVal.png), nunca um `L*`. Num dataset
    # limpo (só os stacks sVal.png) light_directories fica VAZIO, nenhum sMos.png por luz é
    # escrito, e o PS aborta com "Number of images (0) does not match ... (6)" em
    # photometric/main_wps.py:123. Nos datasets reais a detecção só funciona por acidente,
    # porque há arquivos avulsos (ex.: L000/selected-pixels.png) cujo pai é `L*`.
    # xfail(strict) adicionado APÓS registrar a saída crua (regra de auditoria). A LINHA DE
    # BASE de RMSE não pôde ser medida: o pipeline não completa ponta a ponta.
    pytest.mark.xfail(
        reason="MF-14: light_directories vazio em layout L<n>/zf<m>/ limpo -> PS recebe 0 imagens",
        strict=True,
        raises=ValueError,
    ),
]

SIZE = 64
N_FRAMES = 9
N_LIGHTS = 6


def _build_dataset(root):
    """Layout L<n>/zf<m>/sVal.png + lights.npy + sharp/hAvg.png. Retorna depth gt."""
    depth = 2.0 + gaussian_bump(SIZE, amplitude=4.0)  # 2..6, dentro de z_foc 0..8
    normals = normals_from_height(depth)
    lights = ring_lights(N_LIGHTS, tilt_deg=30.0)
    albedo_map = 80.0 + 150.0 * texture(SIZE, seed=1)  # textura p/ foco E sombreamento p/ PS
    z_foc = [float(k) for k in range(N_FRAMES)]

    data_dir = root / "synth"
    for li in range(N_LIGHTS):
        shaded = render_lambertian(normals, lights[li], albedo=albedo_map)
        stack = defocus_stack(shaded, depth, z_foc, blur_per_unit=1.5)
        for k in range(N_FRAMES):
            frame_dir = data_dir / f"L{li}" / f"zf{k}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            img = np.clip(stack[k], 0, 255).astype(np.uint8)
            cv2.imwrite(str(frame_dir / "sVal.png"), cv2.merge([img, img, img]))

    np.save(data_dir / "lights.npy", lights)

    sharp_dir = data_dir / "sharp"
    sharp_dir.mkdir(parents=True, exist_ok=True)
    h_vis = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    cv2.imwrite(str(sharp_dir / "hAvg.png"), h_vis)
    return depth, z_foc


def test_hybrid_pipeline_end_to_end(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import hybrid_stereo_method.photometric.main_wps as main_wps_mod

    # disp_* usam cv2.imshow + waitKey(0): travariam headless (achado PS já registrado)
    monkeypatch.setattr(main_wps_mod, "disp_normalmap", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels", lambda **kw: None)
    monkeypatch.setattr(main_wps_mod, "disp_channels_3d", lambda **kw: None)

    from hybrid_stereo_method.hybrid.main import main as hybrid_main

    raw = tmp_path / "raw"
    depth_gt, z_foc = _build_dataset(raw)

    parameters = {
        "experiment": {
            "type": "hybrid",
            "paths": {
                "input": str(raw),
                "data_folder": "synth",
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
                    "zero_border": False,  # zero_borders(40px) apagaria quase tudo em 64px
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
                "max_iter": 20000,
                "conv_tol": 5e-7,
            }
        },
    }

    hybrid_main(parameters)

    out_dirs = list((tmp_path / "results").glob("*_synth"))
    assert len(out_dirs) == 1, f"esperava 1 pasta de saída, achei {out_dirs}"
    height_path = out_dirs[0] / "integration" / "height_map.npy"
    assert height_path.exists(), "pipeline terminou sem height_map.npy"

    height = np.load(height_path)
    est = height[:SIZE, :SIZE]  # grade de vértices (H+1, W+1) -> recorte
    finite = np.isfinite(est)
    assert finite.mean() > 0.95, f"só {finite.mean():.1%} do mapa de altura é finito"

    interior = (slice(8, -8), slice(8, -8))
    rmse, (a, b) = affine_fit_rmse(est[interior], depth_gt[interior])
    corr = float(np.corrcoef(est[interior].ravel(), depth_gt[interior].ravel())[0, 1])
    print(
        f"\n=== BASELINE E2E === affine-fit RMSE = {rmse:.4f} "
        f"(std gt = {depth_gt[interior].std():.4f}), a = {a:.4f}, b = {b:.4f}, "
        f"pearson r = {corr:.4f}"
    )
    # Linha de base, não gate de qualidade: registre os números no relatório.
    assert np.isfinite(rmse)
