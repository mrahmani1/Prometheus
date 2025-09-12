#!/bin/bash

# Multi-Cloud Disaster Recovery Deployment Script
# Deploys the disaster recovery service to Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="prometheus-ai"
SERVICE_NAME="disaster-recovery"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

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
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Docker (if building image)
    if [[ "${BUILD_IMAGE:-false}" == "true" ]] && ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."
    
    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_warning "Namespace ${NAMESPACE} already exists"
    else
        kubectl create namespace "${NAMESPACE}"
        log_success "Namespace ${NAMESPACE} created"
    fi
}

# Build Docker image
build_image() {
    if [[ "${BUILD_IMAGE:-false}" == "true" ]]; then
        log_info "Building Docker image..."
        
        cd services/disaster-recovery
        docker build -t "prometheus-ai/${SERVICE_NAME}:${IMAGE_TAG}" .
        cd ../..
        
        log_success "Docker image built successfully"
    else
        log_info "Skipping image build (BUILD_IMAGE not set to true)"
    fi
}

# Validate configuration
validate_configuration() {
    log_info "Validating configuration..."
    
    # Check if secrets file exists
    if [[ ! -f "kubernetes/disaster-recovery-secrets.yaml" ]]; then
        log_error "Secrets file not found: kubernetes/disaster-recovery-secrets.yaml"
        exit 1
    fi
    
    # Validate required environment variables in secrets
    local required_vars=(
        "gcp-project-id"
        "azure-subscription-id"
        "aws-account-id"
        "database-url"
        "jwt-secret-key"
    )
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "${var}:" kubernetes/disaster-recovery-secrets.yaml; then
            log_warning "Required variable ${var} not found in secrets file"
        fi
    done
    
    log_success "Configuration validation completed"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Apply secrets
    kubectl apply -f kubernetes/disaster-recovery-secrets.yaml
    
    log_success "Secrets deployed successfully"
}

# Deploy service
deploy_service() {
    log_info "Deploying ${SERVICE_NAME} service..."
    
    # Update image tag in deployment
    if [[ "${IMAGE_TAG}" != "latest" ]]; then
        sed -i.bak "s|image: prometheus-ai/${SERVICE_NAME}:latest|image: prometheus-ai/${SERVICE_NAME}:${IMAGE_TAG}|g" \
            kubernetes/disaster-recovery-deployment.yaml
    fi
    
    # Apply deployment
    kubectl apply -f kubernetes/disaster-recovery-deployment.yaml
    
    # Restore original file if modified
    if [[ -f "kubernetes/disaster-recovery-deployment.yaml.bak" ]]; then
        mv kubernetes/disaster-recovery-deployment.yaml.bak kubernetes/disaster-recovery-deployment.yaml
    fi
    
    log_success "Service deployment applied"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring configuration..."
    
    # Check if Prometheus operator is installed
    if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
        kubectl apply -f kubernetes/disaster-recovery-monitoring.yaml
        log_success "Monitoring configuration deployed"
    else
        log_warning "Prometheus operator not found, skipping monitoring deployment"
    fi
}

# Wait for deployment
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for deployment to be available
    kubectl wait --for=condition=available --timeout=300s \
        deployment/${SERVICE_NAME} -n ${NAMESPACE}
    
    # Check pod status
    kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME}
    
    log_success "Deployment is ready"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check service endpoints
    local service_ip
    service_ip=$(kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "${service_ip}" ]]; then
        log_success "Service is accessible at ${service_ip}:8000"
    else
        log_error "Service IP not found"
        return 1
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if kubectl exec -n ${NAMESPACE} deployment/${SERVICE_NAME} -- \
        curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Health endpoint is responding"
    else
        log_error "Health endpoint is not responding"
        return 1
    fi
    
    # Test readiness endpoint
    log_info "Testing readiness endpoint..."
    if kubectl exec -n ${NAMESPACE} deployment/${SERVICE_NAME} -- \
        curl -f http://localhost:8000/ready &> /dev/null; then
        log_success "Readiness endpoint is responding"
    else
        log_warning "Readiness endpoint is not responding (may be initializing)"
    fi
    
    log_success "Deployment verification completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    
    # Show pods
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME}
    echo
    
    # Show services
    echo "Services:"
    kubectl get svc -n ${NAMESPACE} -l app=${SERVICE_NAME}
    echo
    
    # Show ingress (if exists)
    if kubectl get ingress -n ${NAMESPACE} -l app=${SERVICE_NAME} &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n ${NAMESPACE} -l app=${SERVICE_NAME}
        echo
    fi
    
    # Show recent events
    echo "Recent Events:"
    kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10
}

# Show logs
show_logs() {
    if [[ "${SHOW_LOGS:-false}" == "true" ]]; then
        log_info "Showing recent logs..."
        kubectl logs -n ${NAMESPACE} deployment/${SERVICE_NAME} --tail=50
    fi
}

# Cleanup function
cleanup() {
    if [[ "${CLEANUP:-false}" == "true" ]]; then
        log_info "Cleaning up deployment..."
        
        kubectl delete -f kubernetes/disaster-recovery-deployment.yaml --ignore-not-found=true
        kubectl delete -f kubernetes/disaster-recovery-monitoring.yaml --ignore-not-found=true
        kubectl delete -f kubernetes/disaster-recovery-secrets.yaml --ignore-not-found=true
        
        log_success "Cleanup completed"
    fi
}

# Main deployment function
main() {
    log_info "Starting Multi-Cloud Disaster Recovery deployment..."
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Image Tag: ${IMAGE_TAG}"
    echo
    
    # Handle cleanup if requested
    if [[ "${CLEANUP:-false}" == "true" ]]; then
        cleanup
        exit 0
    fi
    
    # Run deployment steps
    check_prerequisites
    create_namespace
    build_image
    validate_configuration
    deploy_secrets
    deploy_service
    deploy_monitoring
    wait_for_deployment
    verify_deployment
    show_status
    show_logs
    
    echo
    log_success "Multi-Cloud Disaster Recovery deployment completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Configure cloud provider credentials in secrets"
    echo "  2. Test the service endpoints"
    echo "  3. Set up monitoring dashboards"
    echo "  4. Configure alerting rules"
    echo
    log_info "Useful commands:"
    echo "  kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME}"
    echo "  kubectl logs -f deployment/${SERVICE_NAME} -n ${NAMESPACE}"
    echo "  kubectl port-forward svc/${SERVICE_NAME} 8000:8000 -n ${NAMESPACE}"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        CLEANUP=true
        cleanup
        ;;
    "status")
        show_status
        ;;
    "logs")
        kubectl logs -f deployment/${SERVICE_NAME} -n ${NAMESPACE}
        ;;
    "help")
        echo "Usage: $0 [deploy|cleanup|status|logs|help]"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy the disaster recovery service (default)"
        echo "  cleanup  - Remove the deployment"
        echo "  status   - Show deployment status"
        echo "  logs     - Show service logs"
        echo "  help     - Show this help message"
        echo
        echo "Environment Variables:"
        echo "  ENVIRONMENT    - Deployment environment (default: production)"
        echo "  NAMESPACE      - Kubernetes namespace (default: prometheus-ai)"
        echo "  IMAGE_TAG      - Docker image tag (default: latest)"
        echo "  BUILD_IMAGE    - Build Docker image (default: false)"
        echo "  SHOW_LOGS      - Show logs after deployment (default: false)"
        echo "  CLEANUP        - Cleanup deployment (default: false)"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac