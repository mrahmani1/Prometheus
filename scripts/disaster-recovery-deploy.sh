#!/bin/bash

# Prometheus AI Platform - Disaster Recovery Deployment Script
# Comprehensive deployment script for disaster recovery infrastructure

set -euo pipefail

# Configuration
NAMESPACE="prometheus-ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${1:-development}"

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
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
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
        log_success "Namespace $NAMESPACE created"
    fi
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building disaster recovery Docker image..."
    
    cd "$PROJECT_ROOT/services/disaster-recovery"
    
    # Build image
    docker build -t prometheus-ai/disaster-recovery:latest .
    
    # Tag for different environments
    case "$DEPLOYMENT_ENV" in
        "production")
            docker tag prometheus-ai/disaster-recovery:latest prometheus-ai/disaster-recovery:prod
            ;;
        "staging")
            docker tag prometheus-ai/disaster-recovery:latest prometheus-ai/disaster-recovery:staging
            ;;
        *)
            docker tag prometheus-ai/disaster-recovery:latest prometheus-ai/disaster-recovery:dev
            ;;
    esac
    
    log_success "Docker image built successfully"
    
    # Push to registry (if configured)
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        log_info "Pushing image to registry: $DOCKER_REGISTRY"
        docker tag prometheus-ai/disaster-recovery:latest "$DOCKER_REGISTRY/prometheus-ai/disaster-recovery:latest"
        docker push "$DOCKER_REGISTRY/prometheus-ai/disaster-recovery:latest"
        log_success "Image pushed to registry"
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets exist
    if kubectl get secret database-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Database secrets already exist"
    else
        log_error "Database secrets not found. Please create them first:"
        echo "kubectl create secret generic database-secrets -n $NAMESPACE \\"
        echo "  --from-literal=postgres-username=prometheus \\"
        echo "  --from-literal=postgres-password=\$POSTGRES_PASSWORD \\"
        echo "  --from-literal=postgres-database=prometheus \\"
        echo "  --from-literal=influxdb-token=\$INFLUXDB_TOKEN \\"
        echo "  --from-literal=neo4j-username=neo4j \\"
        echo "  --from-literal=neo4j-password=\$NEO4J_PASSWORD \\"
        echo "  --from-literal=redis-password=\$REDIS_PASSWORD"
        exit 1
    fi
    
    if kubectl get secret cloud-credentials -n "$NAMESPACE" &> /dev/null; then
        log_warning "Cloud credentials already exist"
    else
        log_warning "Cloud credentials not found. Creating placeholder..."
        kubectl create secret generic cloud-credentials -n "$NAMESPACE" \
            --from-literal=aws-access-key-id="${AWS_ACCESS_KEY_ID:-placeholder}" \
            --from-literal=aws-secret-access-key="${AWS_SECRET_ACCESS_KEY:-placeholder}"
    fi
    
    if kubectl get secret gcp-service-account -n "$NAMESPACE" &> /dev/null; then
        log_warning "GCP service account already exists"
    else
        log_warning "GCP service account not found. Creating placeholder..."
        echo '{}' | kubectl create secret generic gcp-service-account -n "$NAMESPACE" \
            --from-file=service-account.json=/dev/stdin
    fi
    
    log_success "Secrets deployment completed"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    
    kubectl apply -f "$PROJECT_ROOT/kubernetes/disaster-recovery-configmap.yaml"
    
    log_success "ConfigMaps deployed successfully"
}

# Deploy storage resources
deploy_storage() {
    log_info "Deploying storage resources..."
    
    # Create storage class if it doesn't exist
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        log_warning "Storage class 'fast-ssd' not found. Creating default..."
        cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
allowVolumeExpansion: true
EOF
    fi
    
    log_success "Storage resources deployed"
}

# Deploy disaster recovery service
deploy_service() {
    log_info "Deploying disaster recovery service..."
    
    kubectl apply -f "$PROJECT_ROOT/kubernetes/disaster-recovery-deployment.yaml"
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/disaster-recovery -n "$NAMESPACE"
    
    log_success "Disaster recovery service deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE" -l app=disaster-recovery
    
    # Check service status
    log_info "Checking service status..."
    kubectl get service disaster-recovery -n "$NAMESPACE"
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if kubectl exec -n "$NAMESPACE" deployment/disaster-recovery -- curl -f http://localhost:8080/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test readiness endpoint
    log_info "Testing readiness endpoint..."
    if kubectl exec -n "$NAMESPACE" deployment/disaster-recovery -- curl -f http://localhost:8080/ready; then
        log_success "Readiness check passed"
    else
        log_warning "Readiness check failed - service may still be initializing"
    fi
    
    log_success "Deployment verification completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create ServiceMonitor for Prometheus (if Prometheus operator is installed)
    if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
        cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: disaster-recovery
  namespace: $NAMESPACE
  labels:
    app: disaster-recovery
spec:
  selector:
    matchLabels:
      app: disaster-recovery
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF
        log_success "ServiceMonitor created"
    else
        log_warning "Prometheus operator not found, skipping ServiceMonitor creation"
    fi
}

# Create backup test
create_backup_test() {
    log_info "Creating initial backup test..."
    
    # Wait for service to be ready
    sleep 30
    
    # Port forward to test the service
    kubectl port-forward -n "$NAMESPACE" service/disaster-recovery 8080:8080 &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    # Create test backup
    if curl -X POST http://localhost:8080/api/v1/backup/create \
        -H "Content-Type: application/json" \
        -d '{
            "database_type": "postgresql",
            "database_name": "prometheus",
            "backup_type": "full"
        }' &> /dev/null; then
        log_success "Test backup created successfully"
    else
        log_warning "Test backup creation failed - this is normal if databases are not yet available"
    fi
    
    # Kill port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
}

# Setup disaster recovery testing schedule
setup_testing_schedule() {
    log_info "Setting up disaster recovery testing schedule..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dr-test-monthly
  namespace: $NAMESPACE
  labels:
    app: disaster-recovery
    component: testing
spec:
  schedule: "0 2 1 * *"  # Monthly on the 1st at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dr-test
            image: curlimages/curl:latest
            command:
            - /bin/sh
            - -c
            - |
              curl -X POST http://disaster-recovery:8080/api/v1/test/disaster-recovery \
                -H "Content-Type: application/json" \
                -d '{
                  "test_name": "Monthly DR Test",
                  "test_type": "backup_restore",
                  "scope": ["postgresql", "influxdb"]
                }'
          restartPolicy: OnFailure
EOF
    
    log_success "DR testing schedule created"
}

# Main deployment function
main() {
    log_info "Starting disaster recovery deployment for environment: $DEPLOYMENT_ENV"
    
    check_prerequisites
    create_namespace
    build_and_push_image
    deploy_secrets
    deploy_configmaps
    deploy_storage
    deploy_service
    verify_deployment
    setup_monitoring
    create_backup_test
    setup_testing_schedule
    
    log_success "Disaster recovery deployment completed successfully!"
    
    # Display useful information
    echo ""
    log_info "Useful commands:"
    echo "  View logs: kubectl logs -f deployment/disaster-recovery -n $NAMESPACE"
    echo "  Port forward: kubectl port-forward -n $NAMESPACE service/disaster-recovery 8080:8080"
    echo "  Check status: kubectl get pods -n $NAMESPACE -l app=disaster-recovery"
    echo "  Access API docs: http://localhost:8080/docs (after port forwarding)"
    echo ""
    log_info "API endpoints:"
    echo "  Health: GET /health"
    echo "  Readiness: GET /ready"
    echo "  Create backup: POST /api/v1/backup/create"
    echo "  List backups: GET /api/v1/backup/list"
    echo "  Replication status: GET /api/v1/replication/status"
    echo "  RTO/RPO metrics: GET /api/v1/metrics/rto-rpo"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"