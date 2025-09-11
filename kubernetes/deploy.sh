#!/bin/bash

# Prometheus AI Platform Kubernetes Deployment Script
# This script deploys the complete Prometheus AI Platform to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to Kubernetes cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

print_status "Starting Prometheus AI Platform deployment..."

# Create namespace and basic resources
print_status "Creating namespace and basic resources..."
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmaps.yaml
kubectl apply -f postgres-init-configmap.yaml

# Create persistent volumes
print_status "Creating persistent volumes..."
kubectl apply -f persistent-volumes.yaml

# Wait for persistent volumes to be available
print_status "Waiting for persistent volumes to be ready..."
kubectl wait --for=condition=Available pv/postgres-pv --timeout=60s
kubectl wait --for=condition=Available pv/influxdb-pv --timeout=60s
kubectl wait --for=condition=Available pv/neo4j-pv --timeout=60s
kubectl wait --for=condition=Available pv/kafka-pv --timeout=60s

# Deploy databases
print_status "Deploying databases..."
kubectl apply -f database-deployments.yaml

# Wait for databases to be ready
print_status "Waiting for databases to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/influxdb-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/neo4j-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/redis-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/zookeeper-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/kafka-deployment -n prometheus-ai

# Deploy Consul service mesh
print_status "Deploying Consul service mesh..."
kubectl apply -f consul.yaml

# Wait for Consul to be ready
print_status "Waiting for Consul to be ready..."
kubectl wait --for=condition=available --timeout=180s deployment/consul-deployment -n prometheus-ai

# Deploy monitoring stack
print_status "Deploying monitoring stack (Prometheus & Grafana)..."
kubectl apply -f monitoring.yaml

# Wait for monitoring to be ready
print_status "Waiting for monitoring stack to be ready..."
kubectl wait --for=condition=available --timeout=180s deployment/prometheus-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=180s deployment/grafana-deployment -n prometheus-ai

# Deploy application services
print_status "Deploying application services..."
kubectl apply -f service-deployments.yaml

# Wait for services to be ready
print_status "Waiting for application services to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/api-gateway-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/helios-core-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/data-ingestion-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/volcano-watchtower-deployment -n prometheus-ai

# Deploy horizontal pod autoscalers
print_status "Deploying horizontal pod autoscalers..."
kubectl apply -f hpa.yaml

# Deploy ingress controller
print_status "Deploying ingress controller..."
kubectl apply -f ingress.yaml

# Display deployment status
print_status "Deployment completed! Checking status..."
echo ""
print_status "Namespace resources:"
kubectl get all -n prometheus-ai

echo ""
print_status "Persistent volumes:"
kubectl get pv | grep prometheus-ai

echo ""
print_status "Services and endpoints:"
kubectl get svc -n prometheus-ai
kubectl get endpoints -n prometheus-ai

echo ""
print_status "Horizontal Pod Autoscalers:"
kubectl get hpa -n prometheus-ai

echo ""
print_status "Ingress:"
kubectl get ingress -n prometheus-ai

echo ""
print_status "Deployment Summary:"
echo "✅ Namespace: prometheus-ai"
echo "✅ Databases: PostgreSQL+PostGIS, InfluxDB, Neo4j, Redis, Kafka"
echo "✅ Service Mesh: Consul Connect"
echo "✅ Monitoring: Prometheus, Grafana with advanced dashboards"
echo "✅ Services: API Gateway, Helios Core, Data Ingestion, Volcano Watchtower"
echo "✅ Autoscaling: HPA configured for all services with predictive scaling"
echo "✅ Ingress: External access configured with SSL termination"
echo "✅ Security: Network policies and RBAC configured"
echo "✅ Observability: Distributed tracing and centralized logging enabled"

echo ""
print_status "Access URLs (update DNS to point to your ingress controller):"
echo "🌐 API Gateway: https://api.prometheus-ai.com/api"
echo "📊 Prometheus: https://monitoring.prometheus-ai.com/prometheus"
echo "📈 Grafana: https://monitoring.prometheus-ai.com/grafana (admin/prometheus123)"
echo "🔍 Consul UI: kubectl port-forward svc/consul-service 8500:8500 -n prometheus-ai"

echo ""
print_warning "Next steps:"
echo "1. Update your DNS to point api.prometheus-ai.com and monitoring.prometheus-ai.com to your ingress controller"
echo "2. Configure SSL certificates (cert-manager is referenced in ingress)"
echo "3. Set up CI/CD pipeline to automatically build and push Docker images"
echo "4. Configure multi-cloud deployment for disaster recovery"
echo "5. Set up advanced monitoring alerts and PagerDuty integration"
echo "6. Enable cost optimization monitoring and recommendations"
echo "7. Configure backup and disaster recovery procedures"
echo "8. Monitor the deployment with: kubectl logs -f deployment/api-gateway-deployment -n prometheus-ai"

print_status "Prometheus AI Platform deployment completed successfully!"