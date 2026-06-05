"""Geradores de dados sintéticos com ground truth analítico para a auditoria.

Convenções (as mesmas presumidas pelo pipeline Python, e testadas contra o C):
- arrays numpy [linha=y, coluna=x], origem no topo-esquerdo, y cresce para BAIXO;
- altura z cresce em direção à câmera; normais n = (-dz/dx, -dz/dy, 1)/|.|, nz > 0;
- luzes no mesmo frame das normais.
"""

import cv2
import numpy as np


def gaussian_bump(size, amplitude=6.0, sigma_frac=0.22):
    """Mapa de altura z[y, x] = A * exp(-((x-cx)^2 + (y-cy)^2) / (2 s^2))."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    s = sigma_frac * size
    return amplitude * np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2.0 * s * s)))


def ramp(size, ax=0.05, ay=0.02):
    """Mapa de altura z[y, x] = ax * x + ay * y (y = índice de linha)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    return ax * x + ay * y


def normals_from_height(z):
    """Normais unitárias por diferenças centrais: n = (-dz/dx, -dz/dy, 1)/|.|."""
    dz_dy, dz_dx = np.gradient(z)
    n = np.stack([-dz_dx, -dz_dy, np.ones_like(z)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def ring_lights(n_lights=6, tilt_deg=30.0):
    """Direções de luz unitárias num cone em torno de +z."""
    t = np.deg2rad(tilt_deg)
    az = np.linspace(0.0, 2.0 * np.pi, n_lights, endpoint=False)
    return np.stack(
        [np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.full(n_lights, np.cos(t))],
        axis=-1,
    )


def render_lambertian(normals, light, albedo=1.0):
    """I = albedo * max(0, n . l). `albedo` pode ser escalar ou mapa (h, w)."""
    return albedo * np.clip(normals @ np.asarray(light, dtype=np.float64), 0.0, None)


def texture(size, seed=0):
    """Textura aleatória de alta frequência em [0.2, 1.0] (foco precisa de textura)."""
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.uniform(0.0, 1.0, (size, size)), (0, 0), 1.0)
    t = (t - t.min()) / (t.max() - t.min())
    return 0.2 + 0.8 * t


def defocus_stack(sharp, depth, z_foc, blur_per_unit=1.5):
    """Pilha de foco sintética: frame k = `sharp` desfocada por
    sigma(x, y) = blur_per_unit * |depth(x, y) - z_foc[k]|.

    Implementação: banco de cópias borradas em passos de 0.25 sigma,
    interpoladas linearmente por pixel.
    """
    sharp = np.asarray(sharp, dtype=np.float64)
    sigmas = blur_per_unit * np.abs(depth[None, :, :] - np.asarray(z_foc)[:, None, None])
    step = 0.25
    n_levels = int(np.ceil(sigmas.max() / step)) + 2
    bank = [sharp]
    for k in range(1, n_levels):
        bank.append(cv2.GaussianBlur(sharp, (0, 0), k * step))
    bank = np.stack(bank)  # (n_levels, h, w)

    h, w = sharp.shape
    rows, cols = np.mgrid[0:h, 0:w]
    frames = []
    for k in range(len(z_foc)):
        idx = sigmas[k] / step
        i0 = np.clip(np.floor(idx).astype(int), 0, n_levels - 2)
        frac = idx - i0
        frames.append((1.0 - frac) * bank[i0, rows, cols] + frac * bank[i0 + 1, rows, cols])
    return np.stack(frames)


def affine_fit_rmse(est, gt):
    """RMSE de gt vs (a*est + b) com a, b ótimos por mínimos quadrados.

    Atenção: o fit afim absorve escala global, offset E INVERSÃO DE SINAL —
    use-o para medir forma, e o teste de rampa para decidir convenções de sinal.
    Retorna (rmse, (a, b)).
    """
    est = np.asarray(est, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    a = np.stack([est, np.ones_like(est)], axis=1)
    coef, *_ = np.linalg.lstsq(a, gt, rcond=None)
    rmse = float(np.sqrt(np.mean((a @ coef - gt) ** 2)))
    return rmse, (float(coef[0]), float(coef[1]))
