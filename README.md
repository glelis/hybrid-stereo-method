# Hybrid Stereo Method

A comprehensive framework for 3D reconstruction combining multifocus stereo and photometric stereo techniques.

## Overview

This project implements hybrid stereo reconstruction methods:

- **Multifocus Stereo**: Depth from focal variation using Laplacian, Fourier, or Wavelet focus measures
- **Photometric Stereo**: Surface normals from varying illumination (L2, L1, SBL, RPCA solvers)
- **Hybrid Integration**: Combines both methods using recursive multigrid surface reconstruction

## Installation

```bash
# Clone the repository
git clone https://github.com/glelis/hybrid-stereo-method.git
cd hybrid-stereo-method

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Usage

### Multifocus Stereo

```bash
python -m hybrid_stereo_method.multifocus.main --param_file configs/ms_experiment.yaml
```

### Photometric Stereo

```bash
python -m hybrid_stereo_method.photometric.main --param_file configs/ps_experiment.yaml
```

### Hybrid Method

```bash
python -m hybrid_stereo_method.hybrid.main --param_file configs/hb_experiment.yaml
```

## Project Structure

```
hybrid-stereo-method/
├── src/hybrid_stereo_method/    # Main package
│   ├── core/                    # Domain entities and interfaces
│   ├── multifocus/              # Multifocus stereo module
│   │   ├── indicators/          # Focus measure operators
│   │   ├── argmax_fuzzy.py      # Fuzzy depth estimation
│   │   ├── mosaic.py            # All-in-focus image generation
│   │   └── main.py              # Entry point
│   ├── photometric/             # Photometric stereo module
│   │   ├── solvers/             # PS solver algorithms
│   │   ├── rps.py               # Robust Photometric Stereo class
│   │   └── main.py              # Entry point
│   ├── hybrid/                  # Hybrid integration module
│   └── infrastructure/          # I/O, utilities, visualization
│       └── io/                  # Image and config I/O
├── csrc/                        # C code for integration
│   └── integrate_recursive/     # Multigrid surface integration
├── configs/                     # Experiment configuration files
├── data/                        # Sample datasets
│   ├── raw/                     # Input data
│   └── results/                 # Output results
└── tests/                       # Test suite
```

## Configuration

Configuration is done via YAML files in `configs/`. Example:

```yaml
# Multifocus Stereo Configuration
input_path: '/path/to/data'
output_path: '/path/to/results'
data_foldername: 'dataset_name'

focal_descriptor_paramiters:
  focal_descriptor: 'laplacian'  # laplacian, fourier, wavelet
  laplacian_kernel_size: 5
  smooth: True
```

## Development

```bash
# Format code
make format

# Run linter
make lint

# Run tests
make test

# Type check
make typecheck
```

## Docker

```bash
# Build image
make docker-build

# Run tests in container
make docker-run
```

## License

MIT License

## Author

Gustavo Lelis - g.lelis.silva@gmail.com