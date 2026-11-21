"""Definição dos estágios do sweep v2 e materialização dos configs.

Cada estágio é uma função ``(dataset_id, StageContext) -> list[RunSpec]``.
``RunSpec.overrides`` são ABSOLUTOS em relação a base.yaml (o acumulado dos
estágios anteriores + o delta deste); ``delta`` é só o que o estágio muda em
relação ao seu centro. O centro de cada estágio vem de
``centers/<dataset>/<estágio-anterior>.yaml``, gravado por
``summarize.py --promote`` (ver README.md).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results" / "sweep_v2"
GENERATED = SWEEP_DIR / "generated"
CENTERS = SWEEP_DIR / "centers"

FM = "multifocus.focus_measure"
PRE = f"{FM}.preprocessing"
INT = "hybrid.integration"
SOLVER = "photometric.solver"

Mode = Literal["full", "photometric", "integration"]


@dataclass
class RunSpec:
    stage: str
    dataset: str
    name: str
    overrides: dict[str, Any]
    mode: Mode
    source_run: str | None = None
    delta: dict[str, Any] = field(default_factory=dict)

    def out_dir(self, root: Path | None = None) -> Path:
        return (root or RESULTS) / self.stage / self.dataset / self.name


@dataclass
class StageContext:
    datasets: dict[str, dict]
    # centers[stage] = conteúdo de centers/<ds>/<stage>.yaml
    # ({winner: {...}, top: [{...}, ...]}), cada item com name/overrides/run_dir
    centers: dict[str, dict]

    def winner(self, stage: str) -> dict:
        if stage not in self.centers:
            raise SystemExit(f"centro ausente: promova o estágio {stage} antes (summarize --promote)")
        return self.centers[stage]["winner"]

    def top(self, stage: str) -> list[dict]:
        if stage not in self.centers or not self.centers[stage].get("top"):
            raise SystemExit(f"top-N ausente: rode summarize --stage {stage} --promote --top 3")
        return self.centers[stage]["top"]


# --- helpers de desenho ------------------------------------------------------


def grid(axes: dict[str, list]) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def ofat(axes: dict[str, list], center: dict[str, Any]) -> list[dict]:
    """Um eixo por vez; valores iguais ao do centro não geram spec."""
    out = []
    for key, values in axes.items():
        out.extend({key: v} for v in values if not (key in center and center[key] == v))
    return out


_ABBR = {
    "method": "",
    "radius": "r",
    "kernel_size": "k",
    "smooth": "sm",
    "spatial_median_filter": "med",
    "zero_border": "zb",
    "square": "sq",
    "median_ksize": "mk",
    "smooth_sigma": "ss",
    "r_max": "rmax",
    "interpolation": "",
    "shadow_threshold": "sh",
    "outlier_threshold_multiplier": "om",
    "hints_weight": "hw",
    "use_hints": "uh",
    "initial_method": "init",
    "pixel_size": "px",
    "max_iter": "it",
    "conv_tol": "tol",
}
_VALUE_ABBR = {"linear_interpolation": "lin", "quadratic_interpolation": "quad"}


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return "none"
    if isinstance(v, float):
        return f"{v:g}"
    return _VALUE_ABBR.get(str(v), str(v))


def config_name(delta: dict[str, Any]) -> str:
    """Nome curto, legível e determinístico para um delta de overrides."""
    if not delta:
        return "center"
    parts = []
    for key, value in delta.items():
        leaf = key.rsplit(".", 1)[-1]
        parts.append(_ABBR.get(leaf, leaf) + _fmt_value(value))
    name = "_".join(parts)
    if len(name) > 80:
        digest = hashlib.sha1(json.dumps(delta, sort_keys=True, default=str).encode()).hexdigest()
        name = f"h{digest[:8]}"
    return name


def _spec(stage, ds, delta, mode, base_overrides=None, source_run=None, prefix="") -> RunSpec:
    overrides = {**(base_overrides or {}), **delta}
    return RunSpec(stage, ds, prefix + config_name(delta), overrides, mode, source_run, dict(delta))


def _base_value(dotted: str) -> Any:
    node: Any = load_base()
    for k in dotted.split("."):
        node = node[k]
    return node


def _center_values(overrides: dict[str, Any], keys) -> dict[str, Any]:
    return {k: overrides.get(k, _base_value(k)) for k in keys}


# --- estágios ----------------------------------------------------------------


def stage_s0(ds: str, ctx: StageContext) -> list[RunSpec]:
    # A base É a config "legacy13" do sweep 2026-07 (fourier 0.05 + smooth +
    # median + zero_border off); legacy15 = laplaciano com o mesmo denoise.
    specs = [RunSpec("S0", ds, "base", {}, "full")]
    if ctx.datasets[ds].get("role") == "dev":
        specs.append(
            RunSpec("S0", ds, "legacy15", {f"{FM}.method": "laplacian"}, "full",
                    delta={f"{FM}.method": "laplacian"})
        )
    return specs


def stage_s0conv(ds: str, ctx: StageContext) -> list[RunSpec]:
    """Convergência da integração: rerun sobre o base de S0."""
    center = ctx.winner("S0")
    deltas = grid({f"{INT}.max_iter": [2000, 5000, 10000, 20000], f"{INT}.conv_tol": [5e-7, 1e-7]})
    return [_spec("S0conv", ds, d, "integration", center["overrides"], center["run_dir"]) for d in deltas]


A1_RADII = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
A1_KERNELS = [5, 9, 15, 21, 31]
A1_PRE = {f"{PRE}.smooth": [False, True], f"{PRE}.spatial_median_filter": [False, True],
          f"{PRE}.zero_border": [False, True]}


def stage_a1(ds: str, ctx: StageContext) -> list[RunSpec]:
    """Grade fatorial da medida de foco × denoise × zero_border (89 specs)."""
    specs = []
    pre = grid(A1_PRE)
    for r in A1_RADII:
        for p in pre:
            specs.append(_spec("A1", ds, {f"{FM}.method": "fourier", f"{FM}.parameters.radius": r, **p}, "full"))
    for k in A1_KERNELS:
        for p in pre:
            specs.append(_spec("A1", ds, {f"{FM}.method": "laplacian", f"{FM}.parameters.kernel_size": k, **p}, "full"))
    specs.append(_spec("A1", ds, {f"{FM}.method": "wavelet", f"{PRE}.smooth": True,
                                  f"{PRE}.spatial_median_filter": True, f"{PRE}.zero_border": False}, "full"))
    return specs


def stage_a2(ds: str, ctx: StageContext) -> list[RunSpec]:
    """OFAT do multifocus em torno dos top-3 de A1."""
    specs = []
    for i, center in enumerate(ctx.top("A1")):
        ov = center["overrides"]
        cv = _center_values(ov, [f"{PRE}.smooth", f"{PRE}.spatial_median_filter",
                                 "multifocus.optimization.r_max", "multifocus.parameters.interpolation",
                                 f"{PRE}.square"])
        axes: dict[str, list] = {
            "multifocus.optimization.r_max": [1, 2, 3, 4],
            "multifocus.parameters.interpolation": ["linear_interpolation", "quadratic_interpolation"],
            f"{PRE}.square": [False, True],
        }
        if cv[f"{PRE}.spatial_median_filter"]:
            axes[f"{PRE}.median_ksize"] = [3]
        if cv[f"{PRE}.smooth"]:
            axes[f"{PRE}.smooth_sigma"] = [0.75, 1.5, 2.5]
        for d in ofat(axes, cv):
            specs.append(_spec("A2", ds, d, "full", ov, prefix=f"c{i}_"))
    return specs


def stage_b(ds: str, ctx: StageContext) -> list[RunSpec]:
    center = ctx.winner("A2")
    deltas = grid({f"{SOLVER}.shadow_threshold": [1e-4, 1e-3, 1e-2, 5e-2],
                   f"{SOLVER}.outlier_threshold_multiplier": [2, 2.5, 3, 4, 1e9]})
    return [_spec("B", ds, d, "photometric", center["overrides"], center["run_dir"]) for d in deltas]


C_HINTS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]


def _c1_deltas() -> list[dict]:
    deltas = [{f"{INT}.use_hints": False, f"{INT}.initial_method": "zero"}]  # hints_weight 0
    for hw in C_HINTS:
        for init in ("zero", "hints"):
            deltas.append({f"{INT}.use_hints": True, f"{INT}.hints_weight": hw, f"{INT}.initial_method": init})
    return deltas


def stage_c1(ds: str, ctx: StageContext) -> list[RunSpec]:
    center = ctx.winner("B")
    return [_spec("C1", ds, d, "integration", center["overrides"], center["run_dir"]) for d in _c1_deltas()]


def stage_c2(ds: str, ctx: StageContext) -> list[RunSpec]:
    """Sensibilidade do pixel_size (só reportada, não promovida)."""
    center = ctx.winner("C1")
    px = ctx.datasets[ds]["pixel_size"]
    deltas = [{f"{INT}.pixel_size": round(px * f, 4)} for f in (0.8, 0.9, 1.1, 1.25)]
    return [_spec("C2", ds, d, "integration", center["overrides"], center["run_dir"]) for d in deltas]


def stage_c3(ds: str, ctx: StageContext) -> list[RunSpec]:
    """Interação: top-3 de A1 com o fotométrico de B e {hints vencedor, 0.1}."""
    c1 = ctx.winner("C1")["overrides"]
    b = ctx.winner("B")["overrides"]
    solver = {k: v for k, v in b.items() if k.startswith(SOLVER)}
    integ_win = {k: v for k, v in c1.items() if k.startswith(INT)}
    integ_ref = {f"{INT}.use_hints": True, f"{INT}.hints_weight": 0.1, f"{INT}.initial_method": "zero"}
    variants = [("win", integ_win)] + ([("ref", integ_ref)] if integ_win != integ_ref else [])
    specs = []
    for i, center in enumerate(ctx.top("A1")):
        for tag, integ in variants:
            delta = {**solver, **integ}
            ov = {**center["overrides"], **delta}
            specs.append(RunSpec("C3", ds, f"a1top{i}_{tag}", ov, "photometric", center["run_dir"], delta))
    return specs


def stage_v(ds: str, ctx: StageContext) -> list[RunSpec]:
    """Validação: cada candidato de candidates.yaml, completo, em cada dataset."""
    path = SWEEP_DIR / "candidates.yaml"
    if not path.exists():
        raise SystemExit("crie configs/sweep_v2/candidates.yaml ({nome: {chave.pontuada: valor}})")
    cands = yaml.safe_load(path.read_text()) or {}
    return [RunSpec("V", ds, name, dict(ov or {}), "full", delta=dict(ov or {})) for name, ov in cands.items()]


def stage_vhints(ds: str, ctx: StageContext) -> list[RunSpec]:
    """Mini-C nos datasets de teste: hints da recomendada (centro = V promovido)."""
    center = ctx.winner("V")
    deltas = [{f"{INT}.use_hints": False, f"{INT}.initial_method": "zero"},
              {f"{INT}.use_hints": True, f"{INT}.hints_weight": 0.03},
              {f"{INT}.use_hints": True, f"{INT}.hints_weight": 0.1}]
    return [_spec("Vhints", ds, d, "integration", center["overrides"], center["run_dir"]) for d in deltas]


STAGES: dict[str, Callable[[str, StageContext], list[RunSpec]]] = {
    "S0": stage_s0,
    "S0conv": stage_s0conv,
    "A1": stage_a1,
    "A2": stage_a2,
    "B": stage_b,
    "C1": stage_c1,
    "C2": stage_c2,
    "C3": stage_c3,
    "V": stage_v,
    "Vhints": stage_vhints,
}


# --- I/O ---------------------------------------------------------------------


def load_base() -> dict:
    return yaml.safe_load((SWEEP_DIR / "base.yaml").read_text())


def load_datasets() -> dict[str, dict]:
    return yaml.safe_load((SWEEP_DIR / "datasets.yaml").read_text())


def load_centers(ds: str) -> dict[str, dict]:
    d = CENTERS / ds
    if not d.is_dir():
        return {}
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(d.glob("*.yaml"))}


def apply_override(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"{dotted}: chave inexistente no config base")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"{dotted}: chave inexistente no config base")
    node[keys[-1]] = value


def materialize(spec: RunSpec, datasets: dict, base: dict, out_root: Path | None = None,
                extra: dict[str, Any] | None = None, gen_root: Path | None = None) -> Path:
    """Grava o YAML do spec e devolve o caminho."""
    ds = datasets[spec.dataset]
    if ds.get("pixel_size") is None:
        raise ValueError(f"{spec.dataset}: pixel_size nulo em datasets.yaml (calibre antes)")
    cfg = copy.deepcopy(base)
    input_dir = Path(ds["input"])
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    cfg["experiment"]["paths"]["input"] = str(input_dir)
    cfg["experiment"]["paths"]["data_folder"] = ds["data_folder"]
    cfg["experiment"]["paths"]["output"] = str(spec.out_dir(out_root))
    cfg["hybrid"]["integration"]["pixel_size"] = float(ds["pixel_size"])
    for key, value in {**spec.overrides, **(extra or {})}.items():
        apply_override(cfg, key, value)
    gen = (gen_root or (out_root / "_generated" if out_root else GENERATED)) / spec.stage / spec.dataset
    gen.mkdir(parents=True, exist_ok=True)
    path = gen / f"{spec.name}.yaml"
    header = f"# sweep v2 {spec.stage}/{spec.dataset}/{spec.name} (gerado por stages.py)\n"
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return path
