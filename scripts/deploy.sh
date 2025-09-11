#!/bin/bash

# Prometheus AI Platform Deployment Script
# Usage: ./scripts/deploy.sh [development|production]

set -e

ENVIRONMENT=${1:-development}
PROJECT_NAME="prometheus-ai-platform"

echo "🚀 Deploying Prometheus AI Platform - Environment: $ENVIRONMENT"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/postgres
mkdir -p data/influxdb
mkdir -p data/neo4j
mkdir -p data/redis

if [ "$ENVIRONMENT" = "development" ]; then
    echo "🔧 Starting development environment..."
    
    # Stop any existing containers
    echo "🛑 Stopping existing containers..."
    docker-compose down --remove-orphans || true
    
    # Build and start services
    echo "🏗️ Building and starting services..."
    docker-compose up --build -d
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    echo "🏥 Checking service health..."
    
    # Check Consul
    if curl -f http://localhost:8500/v1/status/leader >/dev/null 2>&1; then
        echo "✅ Consul is ready"
    else
        echo "❌ Consul is not ready"
    fi
    
    # Check Redis
    if docker exec prometheus-redis redis-cli ping >/dev/null 2>&1; then
        echo "✅ Redis is ready"
    else
        echo "❌ Redis is not ready"
    fi
    
    # Check API Gateway
    if curl -f http://localhost:3000/health >/dev/null 2>&1; then
        echo "✅ API Gateway is ready"
    else
        echo "❌ API Gateway is not ready"
    fi
    
    # Check Helios Core
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        echo "✅ Helios Core is ready"
    else
        echo "❌ Helios Core is not ready"
    fi
    
    # Check Data Ingestion
    if curl -f http://localhost:8002/health >/dev/null 2>&1; then
        echo "✅ Data Ingestion is ready"
    else
        echo "❌ Data Ingestion is not ready"
    fi
    
    # Check Volcano Watchtower
    if curl -f http://localhost:8003/health >/dev/null 2>&1; then
        echo "✅ Volcano Watchtower is ready"
    else
        echo "❌ Volcano Watchtower is not ready"
    fi
    
    echo ""
    echo "🎉 Development environment is ready!"
    echo ""
    echo "📊 Service URLs:"
    echo "  • API Gateway:        http://localhost:3000"
    echo "  • Helios Core:        http://localhost:8001"
    echo "  • Data Ingestion:     http://localhost:8002"
    echo "  • Volcano Watchtower: http://localhost:8003"
    echo "  • Consul UI:          http://localhost:8500"
    echo "  • PostgreSQL:         localhost:5432"
    echo "  • InfluxDB:           http://localhost:8086"
    echo "  • Neo4j:              http://localhost:7474"
    echo "  • Redis:              localhost:6379"
    echo ""
    echo "🔑 Default credentials:"
    echo "  • PostgreSQL: prometheus/prometheus123"
    echo "  • InfluxDB:   admin/prometheus123"
    echo "  • Neo4j:      neo4j/prometheus123"
    echo "  • API Login:  admin/prometheus123"
    echo ""
    echo "📝 View logs with: docker-compose logs -f [service-name]"
    echo "🛑 Stop with: docker-compose down"

elif [ "$ENVIRONMENT" = "production" ]; then
    echo "🏭 Production deployment not yet implemented"
    echo "This would deploy to Kubernetes cluster"
    exit 1
else
    echo "❌ Invalid environment: $ENVIRONMENT"
    echo "Usage: $0 [development|production]"
    exit 1
fi