.PHONY: install dev lint format typecheck test clean docker-build docker-run help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install package in production mode"
	@echo "  make dev          - Install package in development mode with dev dependencies"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code with ruff"
	@echo "  make typecheck    - Run mypy type checker"
	@echo "  make test         - Run pytest test suite"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run tests in Docker container"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=hybrid_stereo_method --cov-report=html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t hybrid-stereo-method .

docker-run:
	docker run --rm hybrid-stereo-method

# C code build (if using CMake)
build-c:
	cd csrc/integrate_recursive && cmake -B build && cmake --build build

# Run multifocus stereo experiment
run-multifocus:
	python -m hybrid_stereo_method.multifocus.main --param_file configs/multifocus_experiment.yaml

# Run photometric stereo experiment
run-photometric:
	python -m hybrid_stereo_method.photometric.main --param_file configs/photometric_experiment.yaml

# Run hybrid method experiment
run-hybrid:
	python -m hybrid_stereo_method.hybrid.main --param_file configs/hybrid_experiment.yaml
