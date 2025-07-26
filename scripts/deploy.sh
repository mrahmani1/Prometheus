#!/bin/bash

# Prometheus AI Platform Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="prometheus-ai"
ENVIRONMENT=${1:-"development"}
DOCKER_COMPOSE_FILE="docker-compose.yml"

echo -e "${GREEN}🚀 Deploying Prometheus AI Platform - Environment: $ENVIRONMENT${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command_exists kubectl && [ "$ENVIRONMENT" != "development" ]; then
    echo -e "${RED}❌ kubectl is not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose && [ "$ENVIRONMENT" == "development" ]; then
    echo -e "${RED}❌ docker-compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Development deployment using Docker Compose
if [ "$ENVIRONMENT" == "development" ]; then
    echo -e "${YELLOW}🔧 Starting development environment with Docker Compose...${NC}"
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data/{influxdb,postgres,neo4j,redis,kafka}
    
    # Stop existing containers
    docker-compose -f $DOCKER_COMPOSE_FILE down
    
    # Build and start services
    docker-compose -f $DOCKER_COMPOSE_FILE up --build -d
    
    echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
    sleep 30
    
    # Check service health
    echo -e "${YELLOW}🔍 Checking service health...${NC}"
    
    # Wait for API Gateway to be ready
    for i in {1..30}; do
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            echo -e "${GREEN}✅ API Gateway is healthy${NC}"
            break
        fi
        echo "Waiting for API Gateway... ($i/30)"
        sleep 10
    done
    
    # Display service status
    echo -e "${GREEN}📊 Service Status:${NC}"
    docker-compose -f $DOCKER_COMPOSE_FILE ps
    
    echo -e "${GREEN}🎉 Development environment deployed successfully!${NC}"
    echo -e "${YELLOW}📝 Access points:${NC}"
    echo "  - API Gateway: http://localhost:8080"
    echo "  - API Documentation: http://localhost:8080/api/docs"
    echo "  - InfluxDB UI: http://localhost:8086"
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - Consul UI: http://localhost:8500"
    
    exit 0
fi

# Production/Staging deployment using Kubernetes
echo -e "${YELLOW}☸️  Deploying to Kubernetes environment: $ENVIRONMENT${NC}"

# Check if kubectl is configured
if ! kubectl cluster-info >/dev/null 2>&1; then
    echo -e "${RED}❌ kubectl is not configured or cluster is not accessible${NC}"
    exit 1
fi

# Create namespace if it doesn't exist
echo -e "${YELLOW}📦 Creating namespace: $NAMESPACE${NC}"
kubectl apply -f kubernetes/namespace.yaml

# Apply configurations
echo -e "${YELLOW}⚙️  Applying configurations...${NC}"
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml

# Apply persistent volumes
echo -e "${YELLOW}💾 Setting up persistent storage...${NC}"
kubectl apply -f kubernetes/persistent-volumes.yaml

# Wait for PVCs to be bound
echo -e "${YELLOW}⏳ Waiting for persistent volumes to be ready...${NC}"
kubectl wait --for=condition=Bound pvc --all -n $NAMESPACE --timeout=300s

# Deploy databases
echo -e "${YELLOW}🗄️  Deploying databases...${NC}"
kubectl apply -f kubernetes/database-deployments.yaml

# Wait for database deployments to be ready
echo -e "${YELLOW}⏳ Waiting for databases to be ready...${NC}"
kubectl wait --for=condition=available --timeout=600s deployment/influxdb -n $NAMESPACE
kubectl wait --for=condition=available --timeout=600s deployment/postgres -n $NAMESPACE
kubectl wait --for=condition=available --timeout=600s deployment/neo4j -n $NAMESPACE
kubectl wait --for=condition=available --timeout=600s deployment/redis -n $NAMESPACE

# Deploy API Gateway
echo -e "${YELLOW}🌐 Deploying API Gateway...${NC}"
kubectl apply -f kubernetes/api-gateway-deployment.yaml

# Wait for API Gateway to be ready
echo -e "${YELLOW}⏳ Waiting for API Gateway to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/api-gateway -n $NAMESPACE

# Get service status
echo -e "${GREEN}📊 Deployment Status:${NC}"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Get external IP for API Gateway
echo -e "${YELLOW}🔍 Getting external access information...${NC}"
EXTERNAL_IP=$(kubectl get service api-gateway-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

if [ "$EXTERNAL_IP" != "pending" ] && [ "$EXTERNAL_IP" != "" ]; then
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${YELLOW}📝 Access points:${NC}"
    echo "  - API Gateway: http://$EXTERNAL_IP"
    echo "  - API Documentation: http://$EXTERNAL_IP/api/docs"
    echo "  - Health Check: http://$EXTERNAL_IP/health"
else
    echo -e "${YELLOW}⏳ External IP is still being assigned. Check with:${NC}"
    echo "  kubectl get service api-gateway-service -n $NAMESPACE"
fi

# Run basic health check
echo -e "${YELLOW}🔍 Running health checks...${NC}"
if [ "$EXTERNAL_IP" != "pending" ] && [ "$EXTERNAL_IP" != "" ]; then
    if curl -f http://$EXTERNAL_IP/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Health check failed - services may still be starting${NC}"
    fi
fi

echo -e "${GREEN}🎉 Kubernetes deployment completed!${NC}"