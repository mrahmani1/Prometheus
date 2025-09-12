# Advanced Threat Detection and Response Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for security tools
RUN apt-get update && apt-get install -y \
    nmap \
    curl \
    wget \
    git \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy for container scanning
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash threat_detection
RUN chown -R threat_detection:threat_detection /app
USER threat_detection

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]