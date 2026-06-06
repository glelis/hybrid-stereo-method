# tests/test_convention_integration.py
"""Fase 3.2 — convenções Python <-> C via o integrador.

O teste da rampa decide as convenções de sinal/orientação (CONV-1..4): uma rampa
monotônica não tem simetria, então flip de y, inversão de sinal ou troca de eixos
produzem um candidato diferente como melhor ajuste.
Se um assert de convenção falhar, NÃO conserte o teste: o candidato vencedor
impresso É o achado (CONV-xx).

INT-08 corrigido: o wrapper agora promove mapas de slope 2→3 canais (peso=1)
antes de delegar ao binário C; xfail removido de test_constant_slopes_recover_ramp_and_decide_convention.
"""
import numpy as np
import pytest

from synthetic_utils import affine_fit_rmse, gaussian_bump, normals_from_height, ramp

from hybrid_stereo_method.hybrid.integrate import (
    DEFAULT_EXECUTABLE,
    integrate_normals_to_height,
    integrate_slopes_to_height,
)

needs_binary = pytest.mark.skipif(
    not DEFAULT_EXECUTABLE.exists(),
    reason="binário C não compilado (cd csrc/integrate_recursive && make)",
)

pytestmark = [needs_binary, pytest.mark.slow]


def test_constant_slopes_recover_ramp_and_decide_convention(tmp_path):
    """Decide CONV-1/CONV-2 via -slopes: o wrapper promove 2→3 canais (INT-08 corrigido).
    Se o assert de convenção falhar, NÃO conserte o teste: o candidato vencedor
    impresso É o achado.
    """
    size, ax, ay = 32, 0.05, 0.02
    slopes = np.zeros((size, size, 2))
    slopes[..., 0] = ax  # canal 0: dZ/dX (conforme docstring de integrate.py)
    slopes[..., 1] = ay  # canal 1: dZ/dY
    z = integrate_slopes_to_height(slopes, tmp_path, "ramp")
    assert z.shape == (size + 1, size + 1)

    y, x = np.mgrid[0 : size + 1, 0 : size + 1].astype(np.float64)
    candidates = {
        "z = +ax*x + ay*y (y do numpy, para baixo)": ax * x + ay * y,
        "z = +ax*x - ay*y (y invertido: para cima)": ax * x - ay * y,
        "z = -ax*x - ay*y (tudo invertido)": -(ax * x + ay * y),
        "z = -ax*x + ay*y": -ax * x + ay * y,
        "z = +ay*x + ax*y (eixos trocados)": ay * x + ax * y,
    }
    z0 = z - z.mean()
    errs = {k: float(np.sqrt(np.mean((z0 - (v - v.mean())) ** 2))) for k, v in candidates.items()}
    best = min(errs, key=errs.get)
    print("\nRMSE por candidato de convenção:")
    for k, v in sorted(errs.items(), key=lambda kv: kv[1]):
        print(f"  {v:12.6f}  {k}")

    # 1) o integrador integra: o melhor candidato ajusta bem
    # 5% do slope máximo × tamanho da grade (≈22% do std da rampa); integrador quebrado (saída nula) daria ~46%
    assert errs[best] < 0.05 * (abs(ax) + abs(ay)) * size, errs
    # 2) convenção presumida pelo lado Python (docstrings/numpy): falha = achado CONV
    assert best == "z = +ax*x + ay*y (y do numpy, para baixo)", (
        f"convencao real do C: '{best}' — registrar CONV-xx com a tabela impressa"
    )


def test_normals_path_recovers_bump_shape(tmp_path):
    """Caminho -normals: forma recuperada a menos de afim (sinais já decididos
    pelo teste da rampa; o bump é simétrico e não os detecta)."""
    size = 48
    z_gt = gaussian_bump(size, amplitude=4.0)
    normals = normals_from_height(z_gt)
    z = integrate_normals_to_height(normals, tmp_path, "bump")
    est = z[:size, :size]  # o integrador devolve grade de vértices (H+1,W+1); recorta para comparar com o gt em células (meio-pixel absorvido pelo fit afim)
    rmse, (a, b) = affine_fit_rmse(est, z_gt)
    print(f"\nbump: affine-fit rmse={rmse:.4f}, a={a:.4f}, b={b:.4f}, std(gt)={z_gt.std():.4f}")
    assert np.isfinite(est).all()
    assert rmse < 0.15 * z_gt.std(), f"forma não recuperada: rmse={rmse:.4f}"


def test_ramp_normals_decide_convention(tmp_path):
    """Decide CONV-1/CONV-2 pela rampa via -normals (o caminho -slopes de 2
    canais crasha no C — INT-08). Plano inclinado: n constante ∝ (-ax, -ay, 1).
    Se o assert de convenção falhar, NÃO conserte o teste: o candidato vencedor
    impresso É o achado."""
    size, ax, ay = 32, 0.05, 0.02

    z_gt = ramp(size, ax=ax, ay=ay)
    normals = normals_from_height(z_gt)
    z = integrate_normals_to_height(normals, tmp_path, "ramp_n")
    assert z.shape == (size + 1, size + 1)

    y, x = np.mgrid[0 : size + 1, 0 : size + 1].astype(np.float64)
    candidates = {
        "z = +ax*x + ay*y (y do numpy, para baixo)": ax * x + ay * y,
        "z = +ax*x - ay*y (y invertido: para cima)": ax * x - ay * y,
        "z = -ax*x - ay*y (tudo invertido)": -(ax * x + ay * y),
        "z = -ax*x + ay*y": -ax * x + ay * y,
        "z = +ay*x + ax*y (eixos trocados)": ay * x + ax * y,
    }
    z0 = z - z.mean()
    errs = {k: float(np.sqrt(np.mean((z0 - (v - v.mean())) ** 2))) for k, v in candidates.items()}
    best = min(errs, key=errs.get)
    print("\nRMSE por candidato de convenção (rampa via -normals):")
    for k, v in sorted(errs.items(), key=lambda kv: kv[1]):
        print(f"  {v:12.6f}  {k}")

    # 5% do slope máximo × tamanho da grade (≈22% do std da rampa); integrador quebrado (saída nula) daria ~46%
    assert errs[best] < 0.05 * (abs(ax) + abs(ay)) * size, errs
    assert best == "z = +ax*x + ay*y (y do numpy, para baixo)", (
        f"convencao real do C: '{best}' — registrar/atualizar CONV-xx com a tabela impressa"
    )
