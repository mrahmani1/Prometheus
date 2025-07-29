# Multi-stage build for System Monitoring Service
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r monitoring && useradd -r -g monitoring monitoring

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/monitoring/.local

# Copy application code
COPY main.py .
COPY healthcheck.py .

# Set ownership
RUN chown -R monitoring:monitoring /app

# Switch to non-root user
USER monitoring

# Set PATH to include user-installed packages
ENV PATH=/home/monitoring/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Run the application
CMD ["python", "main.py"]