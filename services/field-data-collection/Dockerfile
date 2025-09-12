# Multi-stage build for Prometheus Field Data Collection Service
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL maintainer="Prometheus AI Platform Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="prometheus-field-data-collection" \
      org.label-schema.description="Mobile and field data collection system" \
      org.label-schema.url="https://prometheus-ai.com" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/prometheus-ai/platform" \
      org.label-schema.vendor="Prometheus AI" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libzbar0 \
    libzbar-dev \
    libopencv-dev \
    python3-opencv \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libzbar0 \
    libopencv-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r prometheus && useradd -r -g prometheus prometheus

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data/field_data/photos /app/data/field_data/audio && \
    chown -R prometheus:prometheus /app

# Switch to non-root user
USER prometheus

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "2"]