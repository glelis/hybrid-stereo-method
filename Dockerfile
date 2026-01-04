FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-dev \
    libxext-dev \
    libxmu-dev \
    freeglut3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python package in development mode
RUN pip install --no-cache-dir -e ".[dev]"

# Build C components (if present)
COPY csrc/ ./csrc/
RUN if [ -d "csrc/integrate_recursive" ]; then \
    cd csrc/integrate_recursive && \
    if [ -f "CMakeLists.txt" ]; then \
        cmake -B build && \
        cmake --build build && \
        cmake --install build --prefix /usr/local; \
    elif [ -f "Makefile" ]; then \
        make; \
    fi; \
    fi

# Copy remaining files
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY data/ ./data/
COPY notebooks/ ./notebooks/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["pytest", "tests/", "-v"]
