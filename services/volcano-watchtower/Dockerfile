# Multi-stage build for Volcano Watchtower Service
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgdal32 \
    libproj25 \
    libgeos-c1v5 \
    libfftw3-3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r prometheus && useradd -r -g prometheus prometheus

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/prometheus/.local

# Copy application code
COPY --chown=prometheus:prometheus . .

# Create necessary directories
RUN mkdir -p logs models cache alerts && \
    chown -R prometheus:prometheus logs models cache alerts

# Switch to non-root user
USER prometheus

# Add local packages to PATH
ENV PATH=/home/prometheus/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/backend:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8003/health', timeout=5)"

EXPOSE 8003

CMD ["python", "main.py"]