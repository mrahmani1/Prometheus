# Multi-stage build for production optimization
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Production stage
FROM node:18-alpine AS production

# Install security updates and required packages
RUN apk update && apk upgrade && \
    apk add --no-cache \
    dumb-init \
    curl \
    ca-certificates && \
    rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S prometheus -u 1001 -G nodejs

# Set working directory
WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder --chown=prometheus:nodejs /app/node_modules ./node_modules

# Copy application code
COPY --chown=prometheus:nodejs . .

# Create necessary directories
RUN mkdir -p logs keys keys/backup && \
    chown -R prometheus:nodejs logs keys

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8080
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Switch to non-root user
USER prometheus

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the application
CMD ["node", "main.js"]

# Labels for metadata
LABEL maintainer="Prometheus AI Team <ops@prometheus-ai.com>"
LABEL version="1.0.0"
LABEL description="External API Gateway for Prometheus AI Geodynamic Platform"
LABEL org.opencontainers.image.source="https://github.com/prometheus-ai/geodynamic-platform"
LABEL org.opencontainers.image.documentation="https://prometheus-ai.com/docs/api"
LABEL org.opencontainers.image.licenses="MIT"