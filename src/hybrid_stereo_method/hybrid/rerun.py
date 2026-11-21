"""Reexecuta o pipeline híbrido a partir de um estágio, reaproveitando um run.

O multifocus custa minutos por run e não muda quando só os parâmetros do
fotométrico ou da integração mudam. Este CLI cria um run novo cujas pastas a
montante são symlinks para as do run de origem e roda só o resto:

    python -m hybrid_stereo_method.hybrid.rerun --source_run <run> \\
        --from_stage {photometric,integration} --param_file <yaml>

A config nova precisa ser idêntica à de origem em tudo que está a montante do
estágio (dataset, multifocus e, para ``integration``, fotométrico). Se não
for, o run aborta (código 2) com as diferenças: senão as métricas seriam
atribuídas a uma config que não as produziu.
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any

import yaml

from hybrid_stereo_method.hybrid.main import build_integration_config
from hybrid_stereo_method.hybrid.stages import (
    run_evaluation_stage,
    run_integration_stage,
    run_photometric_stage,
)
from hybrid_stereo_method.infrastructure.io.image_io import (
    find_all_files,
    log_parameters,
    read_yaml_parameters,
)

STAGES = ("photometric", "integration")


def _diff(new: Any, old: Any, prefix: str) -> list[str]:
    if isinstance(new, dict) and isinstance(old, dict):
        out: list[str] = []
        for key in sorted(set(new) | set(old), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in new:
                out.append(f"{path}: <ausente> != {old[key]!r}")
            elif key not in old:
                out.append(f"{path}: {new[key]!r} != <ausente>")
            else:
                out.extend(_diff(new[key], old[key], path))
        return out
    if prefix.endswith("parameters.z_foc") and new == "auto":
        return []  # 'auto' casa com a lista resolvida gravada na origem
    return [] if new == old else [f"{prefix}: {new!r} != {old!r}"]


def check_upstream_compatible(new: dict, source: dict, from_stage: str) -> list[str]:
    """Diferenças entre a config nova e a de origem a montante de ``from_stage``."""
    if from_stage not in STAGES:
        raise ValueError(f"from_stage must be one of {STAGES}, got {from_stage!r}")
    diffs: list[str] = []
    for key in ("input", "data_folder"):
        diffs += _diff(
            new.get("experiment", {}).get("paths", {}).get(key),
            source.get("experiment", {}).get("paths", {}).get(key),
            f"experiment.paths.{key}",
        )
    sections = ["multifocus"] + (["photometric"] if from_stage == "integration" else [])
    for section in sections:
        diffs += _diff(new.get(section, {}), source.get(section, {}), section)
    return diffs


def _setup_run_dir(parameters: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder = parameters["experiment"]["paths"]["data_folder"]
    output_path = os.path.join(parameters["experiment"]["paths"]["output"], f"{stamp}_{data_folder}")
    os.makedirs(output_path, exist_ok=False)
    handler = logging.FileHandler(os.path.join(output_path, f"hybrid_rerun_{stamp}.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return output_path


def rerun(source_run: str, from_stage: str, parameters: dict) -> str:
    """Executa o rerun e devolve o diretório do run novo."""
    source_run = os.path.abspath(source_run)
    with open(os.path.join(source_run, "parameters.yaml")) as f:
        source_params = yaml.safe_load(f)
    diffs = check_upstream_compatible(parameters, source_params, from_stage)
    if diffs:
        print(
            f"rerun: config incompatível com {source_run} a montante de '{from_stage}':",
            file=sys.stderr,
        )
        for d in diffs:
            print(f"  {d}", file=sys.stderr)
        raise SystemExit(2)

    # Valida a integração antes de criar qualquer coisa.
    build_integration_config(
        parameters.get("hybrid", {}).get("integration", {}),
        debug=parameters["experiment"]["settings"].get("debug", False),
    )
    # Mesmos valores resolvidos da origem (ex.: z_foc 'auto' -> lista).
    parameters["multifocus"] = source_params["multifocus"]
    parameters["rerun"] = {"source_run": source_run, "from_stage": from_stage}

    output_path = _setup_run_dir(parameters)
    logging.info("Rerun from stage '%s' of %s", from_stage, source_run)
    log_parameters(parameters)
    with open(os.path.join(output_path, "parameters.yaml"), "w") as f:
        yaml.safe_dump(parameters, f, sort_keys=False, allow_unicode=True)

    linked = ["multifocus_stereo"] + (["photometric_stereo"] if from_stage == "integration" else [])
    for name in linked:
        os.symlink(os.path.join(source_run, name), os.path.join(output_path, name))

    paths = parameters["experiment"]["paths"]
    data_dir = os.path.join(paths["input"], paths["data_folder"])
    if from_stage == "photometric":
        run_photometric_stage(parameters, output_path, find_all_files(data_dir))
    run_integration_stage(output_path, parameters)
    run_evaluation_stage(output_path, parameters, data_dir)
    logging.info("Rerun complete: %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source_run", required=True, help="Run de origem (com parameters.yaml).")
    parser.add_argument("--from_stage", required=True, choices=STAGES)
    parser.add_argument("--param_file", required=True, help="YAML da config nova.")
    args = parser.parse_args(argv)
    return rerun(args.source_run, args.from_stage, read_yaml_parameters(args.param_file))


def cli() -> None:
    """Console entry point (não devolve o caminho: sys.exit(str) sairia com 1)."""
    main()


if __name__ == "__main__":
    cli()
