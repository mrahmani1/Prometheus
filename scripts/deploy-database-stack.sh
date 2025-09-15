#!/bin/bash

# Prometheus AI Platform Database Stack Deployment Script
# This script deploys the complete database infrastructure with monitoring and backup

set -euo pipefail

# Configuration
NAMESPACE="prometheus-ai"
ENVIRONMENT="${ENVIRONMENT:-development}"
HELM_RELEASE_NAME="prometheus-database-stack"
KUBECTL_TIMEOUT="300s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed (for local development)
    if [[ "$ENVIRONMENT" == "development" ]] && ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" monitoring=enabled environment="$ENVIRONMENT"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Deploy storage classes and persistent volumes
deploy_storage() {
    log_info "Deploying storage classes and persistent volumes..."
    
    kubectl apply -f kubernetes/database-storage.yaml --timeout="$KUBECTL_TIMEOUT"
    
    # Wait for storage classes to be ready
    kubectl wait --for=condition=Ready storageclass/fast-ssd --timeout="$KUBECTL_TIMEOUT" || true
    kubectl wait --for=condition=Ready storageclass/standard-hdd --timeout="$KUBECTL_TIMEOUT" || true
    
    log_success "Storage configuration deployed"
}

# Generate and apply secrets
deploy_secrets() {
    log_info "Generating and deploying database secrets..."
    
    # Check if secrets already exist
    if kubectl get secret database-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Database secrets already exist, skipping generation"
        return
    fi
    
    # Generate random passwords if not provided
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(openssl rand -base64 32)}"
    INFLUXDB_PASSWORD="${INFLUXDB_PASSWORD:-$(openssl rand -base64 32)}"
    TIMESCALEDB_PASSWORD="${TIMESCALEDB_PASSWORD:-$(openssl rand -base64 32)}"
    INFLUXDB_TOKEN="${INFLUXDB_TOKEN:-$(openssl rand -base64 64)}"
    
    # Create secrets
    kubectl create secret generic database-secrets -n "$NAMESPACE" \
        --from-literal=postgres-db="prometheus" \
        --from-literal=postgres-user="prometheus" \
        --from-literal=postgres-password="$POSTGRES_PASSWORD" \
        --from-literal=postgres-replication-password="$(openssl rand -base64 32)" \
        --from-literal=postgres-exporter-dsn="postgresql://prometheus:$POSTGRES_PASSWORD@postgres:5432/prometheus?sslmode=disable" \
        --from-literal=redis-password="$REDIS_PASSWORD" \
        --from-literal=neo4j-password="$NEO4J_PASSWORD" \
        --from-literal=neo4j-auth="neo4j/$NEO4J_PASSWORD" \
        --from-literal=influxdb-admin-user="admin" \
        --from-literal=influxdb-admin-password="$INFLUXDB_PASSWORD" \
        --from-literal=influxdb-admin-token="$INFLUXDB_TOKEN" \
        --from-literal=timescaledb-db="timescale" \
        --from-literal=timescaledb-user="timescale" \
        --from-literal=timescaledb-password="$TIMESCALEDB_PASSWORD" \
        --from-literal=backup-encryption-key="$(openssl rand -base64 64)" \
        --from-literal=grafana-admin-password="$(openssl rand -base64 32)"
    
    log_success "Database secrets created"
    
    # Save passwords to file for reference (in production, use proper secret management)
    if [[ "$ENVIRONMENT" == "development" ]]; then
        cat > .database-passwords <<EOF
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
NEO4J_PASSWORD=$NEO4J_PASSWORD
INFLUXDB_PASSWORD=$INFLUXDB_PASSWORD
TIMESCALEDB_PASSWORD=$TIMESCALEDB_PASSWORD
INFLUXDB_TOKEN=$INFLUXDB_TOKEN
EOF
        chmod 600 .database-passwords
        log_info "Database passwords saved to .database-passwords (development only)"
    fi
}

# Deploy database StatefulSets
deploy_databases() {
    log_info "Deploying database StatefulSets..."
    
    kubectl apply -f kubernetes/database-statefulsets.yaml --timeout="$KUBECTL_TIMEOUT"
    kubectl apply -f kubernetes/database-services.yaml --timeout="$KUBECTL_TIMEOUT"
    
    log_info "Waiting for databases to be ready..."
    
    # Wait for StatefulSets to be ready
    kubectl wait --for=condition=Ready statefulset/postgres -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    kubectl wait --for=condition=Ready statefulset/redis -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    kubectl wait --for=condition=Ready statefulset/influxdb -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    kubectl wait --for=condition=Ready statefulset/neo4j -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    kubectl wait --for=condition=Ready statefulset/timescaledb -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Database StatefulSets deployed and ready"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying database monitoring stack..."
    
    kubectl apply -f kubernetes/database-monitoring.yaml --timeout="$KUBECTL_TIMEOUT"
    
    # Wait for monitoring deployments to be ready
    kubectl wait --for=condition=Available deployment/prometheus-database-monitoring -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    kubectl wait --for=condition=Available deployment/grafana-database-monitoring -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Database monitoring deployed"
}

# Deploy backup CronJobs
deploy_backup() {
    log_info "Deploying database backup automation..."
    
    kubectl apply -f kubernetes/database-backup-cronjobs.yaml --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Database backup CronJobs deployed"
}

# Deploy service mesh configuration
deploy_service_mesh() {
    log_info "Deploying service mesh configuration..."
    
    # Check if Istio is installed
    if kubectl get crd gateways.networking.istio.io &> /dev/null; then
        kubectl apply -f kubernetes/database-service-mesh.yaml --timeout="$KUBECTL_TIMEOUT"
        log_success "Service mesh configuration deployed"
    else
        log_warning "Istio not detected, skipping service mesh configuration"
    fi
}

# Deploy scaling policies
deploy_scaling() {
    log_info "Deploying database scaling policies..."
    
    kubectl apply -f kubernetes/database-scaling-policies.yaml --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Database scaling policies deployed"
}

# Deploy using Helm (alternative to kubectl)
deploy_with_helm() {
    log_info "Deploying database stack using Helm..."
    
    # Add required Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    # Install or upgrade the Helm release
    helm upgrade --install "$HELM_RELEASE_NAME" ./helm/database-stack \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --set environment="$ENVIRONMENT" \
        --set global.storageClass="fast-ssd" \
        --timeout="$KUBECTL_TIMEOUT" \
        --wait
    
    log_success "Database stack deployed using Helm"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying database deployment..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check service status
    log_info "Checking service status..."
    kubectl get services -n "$NAMESPACE"
    
    # Check persistent volumes
    log_info "Checking persistent volumes..."
    kubectl get pvc -n "$NAMESPACE"
    
    # Test database connectivity
    log_info "Testing database connectivity..."
    
    # Test PostgreSQL
    if kubectl exec -n "$NAMESPACE" statefulset/postgres -- pg_isready -U prometheus -d prometheus &> /dev/null; then
        log_success "PostgreSQL is ready"
    else
        log_error "PostgreSQL is not ready"
    fi
    
    # Test Redis
    if kubectl exec -n "$NAMESPACE" statefulset/redis -- redis-cli ping | grep -q PONG; then
        log_success "Redis is ready"
    else
        log_error "Redis is not ready"
    fi
    
    # Test InfluxDB
    if kubectl exec -n "$NAMESPACE" statefulset/influxdb -- influx ping &> /dev/null; then
        log_success "InfluxDB is ready"
    else
        log_error "InfluxDB is not ready"
    fi
    
    log_success "Database deployment verification completed"
}

# Display connection information
display_connection_info() {
    log_info "Database connection information:"
    
    echo ""
    echo "PostgreSQL:"
    echo "  Host: postgres.prometheus-ai.svc.cluster.local"
    echo "  Port: 5432"
    echo "  Database: prometheus"
    echo "  Username: prometheus"
    echo ""
    echo "Redis:"
    echo "  Host: redis.prometheus-ai.svc.cluster.local"
    echo "  Port: 6379"
    echo ""
    echo "InfluxDB:"
    echo "  Host: influxdb.prometheus-ai.svc.cluster.local"
    echo "  Port: 8086"
    echo "  Organization: prometheus"
    echo "  Bucket: geological_data"
    echo ""
    echo "Neo4j:"
    echo "  HTTP: http://neo4j.prometheus-ai.svc.cluster.local:7474"
    echo "  Bolt: bolt://neo4j.prometheus-ai.svc.cluster.local:7687"
    echo "  Username: neo4j"
    echo ""
    echo "TimescaleDB:"
    echo "  Host: timescaledb.prometheus-ai.svc.cluster.local"
    echo "  Port: 5432"
    echo "  Database: timescale"
    echo "  Username: timescale"
    echo ""
    echo "Monitoring:"
    echo "  Prometheus: http://prometheus-database-monitoring.prometheus-ai.svc.cluster.local:9090"
    echo "  Grafana: http://grafana-database-monitoring.prometheus-ai.svc.cluster.local:3000"
    echo ""
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        echo "For local access, use port-forwarding:"
        echo "  kubectl port-forward -n prometheus-ai svc/postgres 5432:5432"
        echo "  kubectl port-forward -n prometheus-ai svc/redis 6379:6379"
        echo "  kubectl port-forward -n prometheus-ai svc/influxdb 8086:8086"
        echo "  kubectl port-forward -n prometheus-ai svc/neo4j 7474:7474"
        echo "  kubectl port-forward -n prometheus-ai svc/timescaledb 5433:5432"
        echo "  kubectl port-forward -n prometheus-ai svc/prometheus-database-monitoring 9090:9090"
        echo "  kubectl port-forward -n prometheus-ai svc/grafana-database-monitoring 3000:3000"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up database stack..."
    
    # Delete Helm release if it exists
    if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE_NAME"; then
        helm uninstall "$HELM_RELEASE_NAME" -n "$NAMESPACE"
    fi
    
    # Delete Kubernetes resources
    kubectl delete -f kubernetes/database-scaling-policies.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-service-mesh.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-backup-cronjobs.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-monitoring.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-services.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-statefulsets.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-storage.yaml --ignore-not-found=true
    
    # Delete namespace
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    log_success "Database stack cleanup completed"
}

# Main deployment function
main() {
    local action="${1:-deploy}"
    
    case "$action" in
        "deploy")
            log_info "Starting database stack deployment..."
            check_prerequisites
            create_namespace
            deploy_storage
            deploy_secrets
            deploy_databases
            deploy_monitoring
            deploy_backup
            deploy_service_mesh
            deploy_scaling
            verify_deployment
            display_connection_info
            log_success "Database stack deployment completed successfully!"
            ;;
        "helm")
            log_info "Starting database stack deployment with Helm..."
            check_prerequisites
            create_namespace
            deploy_storage
            deploy_secrets
            deploy_with_helm
            verify_deployment
            display_connection_info
            log_success "Database stack deployment with Helm completed successfully!"
            ;;
        "cleanup")
            cleanup
            ;;
        "verify")
            verify_deployment
            ;;
        "info")
            display_connection_info
            ;;
        *)
            echo "Usage: $0 {deploy|helm|cleanup|verify|info}"
            echo "  deploy  - Deploy database stack using kubectl"
            echo "  helm    - Deploy database stack using Helm"
            echo "  cleanup - Remove database stack"
            echo "  verify  - Verify deployment status"
            echo "  info    - Display connection information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"