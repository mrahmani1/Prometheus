# Prometheus Model Training Orchestrator Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Base image with system dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libopenmpi-dev \
    openmpi-bin \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 2: Python dependencies
FROM base as python-deps

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional scientific computing libraries
RUN pip install --no-cache-dir \
    isce3 \
    mintpy \
    pyproj \
    cartopy \
    obspy \
    pyrocko \
    pyfftw \
    numba \
    cupy-cuda11x \
    rapids-cudf \
    rapids-cuml

# Install quantum computing libraries (optional)
RUN pip install --no-cache-dir \
    qiskit \
    qiskit-optimization \
    qiskit-machine-learning \
    cirq \
    pennylane \
    || echo "Quantum libraries installation failed, continuing without them"

# Stage 3: Application
FROM python-deps as app

# Create non-root user
RUN useradd --create-home --shell /bin/bash prometheus
RUN chown -R prometheus:prometheus /app
USER prometheus

# Copy application code
COPY --chown=prometheus:prometheus . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/cache

# Set environment variables for the application
ENV PYTHONPATH=/app
ENV MODEL_CACHE_DIR=/app/cache
ENV DATA_DIR=/app/data
ENV MODELS_DIR=/app/models
ENV LOGS_DIR=/app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]