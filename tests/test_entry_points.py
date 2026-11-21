"""Os console scripts do pyproject apontam para funções `cli` existentes."""

import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
SCRIPTS = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_console_script_target_is_callable(name):
    module_name, func_name = SCRIPTS[name].split(":")
    func = getattr(importlib.import_module(module_name), func_name, None)
    assert callable(func), f"{name} → {SCRIPTS[name]} não existe"


def test_makefile_run_targets_reference_existing_configs():
    root = PYPROJECT.parent
    makefile = (root / "Makefile").read_text()
    configs = [tok for tok in makefile.split() if tok.startswith("configs/") and tok.endswith(".yaml")]
    assert configs, "nenhum config referenciado no Makefile"
    missing = [c for c in configs if not (root / c).exists()]
    assert not missing, f"Makefile referencia configs inexistentes: {missing}"
