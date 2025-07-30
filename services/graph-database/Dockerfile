# Multi-stage build for Neo4j Graph Database Service
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
RUN groupadd -r graphdb && useradd -r -g graphdb graphdb

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/graphdb/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R graphdb:graphdb /app

# Switch to non-root user
USER graphdb

# Set environment variables
ENV PATH=/home/graphdb/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]