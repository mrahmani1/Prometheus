# Multi-stage build for Edge Device Manager
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

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r edgemanager && useradd -r -g edgemanager edgemanager

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/edgemanager/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R edgemanager:edgemanager /app

# Switch to non-root user
USER edgemanager

# Set PATH to include user-installed packages
ENV PATH=/home/edgemanager/.local/bin:$PATH

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["python", "main.py"]