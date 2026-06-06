# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework for 3D reconstruction combining **multifocus stereo** (depth from focus variation) and **photometric stereo** (surface normals from lighting variation), integrated into a hybrid pipeline that produces height maps via a C-based recursive multigrid surface integrator.

## Setup and Commands

```bash
# Install in development mode (package lives in src/hybrid_stereo_method)
pip install -e ".[dev]"

# Build the C integration solver (required for the hybrid pipeline Step 3)
cd csrc/integrate_recursive && make

# Lint / format / type check (configured in pyproject.toml; line-length 100)
ruff check src/
ruff format src/
mypy src/

# Tests (pytest configured with --cov=hybrid_stereo_method; tests/ is currently empty)
pytest
pytest tests/test_foo.py::test_bar   # single test
```

Note: the README mentions `make format` / `make test`, but there is no root Makefile — use the ruff/mypy/pytest commands directly. The only Makefile is in `csrc/integrate_recursive/`.

## Running Experiments

All entry points take a YAML config via `--param_file` (examples in `configs/`):

```bash
python -m hybrid_stereo_method.multifocus.main --param_file configs/ms_experiment.yaml
python -m hybrid_stereo_method.photometric.main --param_file configs/ps_experiment.yaml
python -m hybrid_stereo_method.photometric.main_wps --param_file configs/wps_experiment.yaml
python -m hybrid_stereo_method.hybrid.main --param_file configs/hb_experiment.yaml
```

Configs are organized in sections: `experiment` (paths, settings), `multifocus`, `photometric`, `hybrid.integration`. Paths in configs are absolute and point into `data/raw/` (input) and `data/results/` (output). Outputs go to timestamped folders (`YYYYMMDD_HHMM_<data_folder>`).

Avaliação automatizada contra ground truth (standalone, sobre qualquer resultado já gerado;
também roda como hook ao final do pipeline híbrido se `evaluation.enabled: true` no YAML):

```bash
python -m hybrid_stereo_method.evaluation.main --results_dir <pasta_de_resultados> [--data_dir <pasta_do_dataset>]
```

Saídas em `<results_dir>/evaluation/`: `metrics.json`, `report.md` e mapas de erro PNG.

## Architecture

### Hybrid pipeline (`hybrid/main.py`) — the core flow

Three sequential steps, each consuming the previous step's outputs:

1. **Multifocus stereo** (`multifocus/`): runs on per-`zf` (focal plane) averaged images to compute the focus-selection map `iSel`, then reuses that single `iSel` to build an all-in-focus mosaic (`sMos.png`/`.fni`) for *each* light directory `L*`. Focus measures live in `multifocus/indicators/` (laplacian, fourier, wavelet); sub-pixel depth comes from `argmax_fuzzy.py`; optional graph-cut refinement in `depth_refinement.py`.
2. **Photometric stereo** (`photometric/main_wps.py` → `wps.py`): takes the per-light mosaics + `lights.npy` and estimates a normal map using `estimate_normals_argmax_lstsq_robust` (weighted/robust least-squares). `rps.py` contains an alternative `RPS` class with L2/L1/SBL/RPCA solvers used by `photometric/main.py`.
3. **Surface integration** (`hybrid/integrate.py`): wraps the C binary `csrc/integrate_recursive/gus_integrate_recursive` via subprocess. Optionally uses the multifocus depth map (`zMos_with_confidence.fni`) as *hints* and `sharp/hAvg.png` as a *reference* for error analysis (`use_hints` / `use_reference` in config).

The pipeline steps communicate via files written to the output directory, not in-memory — `main_wps.main()` and the integrator read paths that `hybrid/main.py` injects into the `parameters` dict (`sMos_path_list`, `lights_path`, `output_path_multifocus`, etc.).

### Python ↔ C boundary

Data is exchanged with the C solver through **FNI files** (float image format). Conversion helpers are `convert_image_array_to_fni` / `read_fni_to_image_array` in `infrastructure/io/image_io.py`. The C executable path is resolved relative to the repo root in `hybrid/integrate.py` (`DEFAULT_EXECUTABLE`).

### Expected dataset layout

Input data folders (under `data/raw/.../<data_folder>/`) follow the convention:
- `L<n>/zf<m>/...sVal.png` — image stacks per light source per focal plane
- `lights.npy` — light direction matrix
- `sharp/hAvg.png` — ground-truth/reference height map (optional)
- `mask.png`, `gt_normal.npy` — optional mask and ground-truth normals

### Other notes

- `notebooks/` holds exploratory analyses and thesis figures (`imagens_tese`); they are not part of the package.
- `core/` (entities/interfaces) and `tests/` exist but are currently empty scaffolding.
- Logging is configured per-run to both console and a log file in the output folder.
