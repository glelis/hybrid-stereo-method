"""Gera os 10 configs do sweep de parâmetros a partir do hb_experiment.yaml.

Estratégia: baseline + um fator por vez, focando nos elos fracos do run de
2026-07-10 (profundidade multifocus r=0.35; normais já com 3° de erro médio):

- eixo "medida de foco": método (fourier/laplacian/wavelet) e banda do fourier;
- eixo "ruído do zMos": suavização/mediana espacial e janela do fuzzy argmax;
- eixo "mosaico": interpolação quadrática;
- eixo "fotométrico": top_k do solver robusto;
- eixo "acoplamento": peso dos hints do multifocus na integração.

Uso:  python configs/sweep/generate_configs.py
"""
import copy
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1] / "hb_experiment.yaml"
OUT_DIR = Path(__file__).resolve().parent
SWEEP_OUTPUT = "/home/lelis/Documents/Projetos/hybrid-stereo-method/data/results/hybrid_sweep"

# nome da config -> (descrição, patches {caminho.pontuado: valor})
CONFIGS = {
    "01_baseline": (
        "Config atual, ponto de referência",
        {},
    ),
    "02_laplacian": (
        "Medida de foco laplaciana (kernel 31) no lugar de fourier",
        {"multifocus.focus_measure.method": "laplacian"},
    ),
    "03_wavelet": (
        "Medida de foco wavelet no lugar de fourier",
        {"multifocus.focus_measure.method": "wavelet"},
    ),
    "04_fourier_low_freq": (
        "Fourier com radius 0.05 (frequências mais baixas, mais robusto a ruído)",
        {"multifocus.focus_measure.parameters.radius": 0.05},
    ),
    "05_fourier_high_freq": (
        "Fourier com radius 0.20 (frequências mais altas, mais detalhe)",
        {"multifocus.focus_measure.parameters.radius": 0.20},
    ),
    "06_denoise_focus": (
        "Suavização + mediana espacial na medida de foco (ataca o ruído do zMos)",
        {
            "multifocus.focus_measure.preprocessing.smooth": True,
            "multifocus.focus_measure.preprocessing.spatial_median_filter": True,
        },
    ),
    "07_fuzzy_window4": (
        "Janela do fuzzy argmax r_max=4 (curva de foco mais estável)",
        {"multifocus.optimization.r_max": 4},
    ),
    "08_quadratic_interp": (
        "Interpolação quadrática no mosaico (zMos subframe mais suave)",
        {"multifocus.parameters.interpolation": "quadratic_interpolation"},
    ),
    "09_photometric_top5": (
        "Solver fotométrico com top_k=5 imagens por pixel",
        {"photometric.solver.top_k": 5},
    ),
    "10_strong_hints": (
        "Peso dos hints do multifocus na integração 0.1 -> 0.3",
        {"hybrid.integration.hints_weight": 0.3},
    ),
    # ---- Round 2: combinações dos vencedores do round 1 ----
    # 04_fourier_low_freq venceu em altura final; 02_laplacian em profundidade
    # multifocus; 06_denoise_focus em seleção de foco. zero_border=False vem do
    # run avulso de 2026-07-10 15:34, que superou o baseline em multifocus.
    "11_lowfreq_denoise": (
        "Fourier radius 0.05 + suavização/mediana na medida de foco",
        {
            "multifocus.focus_measure.parameters.radius": 0.05,
            "multifocus.focus_measure.preprocessing.smooth": True,
            "multifocus.focus_measure.preprocessing.spatial_median_filter": True,
        },
    ),
    "12_lowfreq_noborder": (
        "Fourier radius 0.05 + zero_border desligado",
        {
            "multifocus.focus_measure.parameters.radius": 0.05,
            "multifocus.focus_measure.preprocessing.zero_border": False,
        },
    ),
    "13_lowfreq_denoise_noborder": (
        "Fourier radius 0.05 + denoise + zero_border desligado",
        {
            "multifocus.focus_measure.parameters.radius": 0.05,
            "multifocus.focus_measure.preprocessing.smooth": True,
            "multifocus.focus_measure.preprocessing.spatial_median_filter": True,
            "multifocus.focus_measure.preprocessing.zero_border": False,
        },
    ),
    "14_laplacian_denoise": (
        "Laplaciano + suavização/mediana na medida de foco",
        {
            "multifocus.focus_measure.method": "laplacian",
            "multifocus.focus_measure.preprocessing.smooth": True,
            "multifocus.focus_measure.preprocessing.spatial_median_filter": True,
        },
    ),
    "15_laplacian_denoise_noborder": (
        "Laplaciano + denoise + zero_border desligado",
        {
            "multifocus.focus_measure.method": "laplacian",
            "multifocus.focus_measure.preprocessing.smooth": True,
            "multifocus.focus_measure.preprocessing.spatial_median_filter": True,
            "multifocus.focus_measure.preprocessing.zero_border": False,
        },
    ),
}


def apply_patch(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"{dotted_key}: chave inexistente no config base")
    node[keys[-1]] = value


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    for name, (description, patches) in CONFIGS.items():
        cfg = copy.deepcopy(base)
        # Saída com nome fácil: data/results/hybrid_sweep/<nome-da-config>/
        cfg["experiment"]["paths"]["output"] = f"{SWEEP_OUTPUT}/{name}/"
        for dotted_key, value in patches.items():
            apply_patch(cfg, dotted_key, value)
        out = OUT_DIR / f"{name}.yaml"
        header = f"# Sweep {name}: {description}\n# Gerado por generate_configs.py a partir de hb_experiment.yaml\n"
        out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
