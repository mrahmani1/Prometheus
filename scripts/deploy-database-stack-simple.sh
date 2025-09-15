#!/bin/bash

# Prometheus AI Platform Database Stack Deployment Script
# This script deploys the complete database infrastructure with monitoring and backup

set -euo pipefail

NAMESPACE="prometheus-ai"
ENVIRONMENT="${ENVIRONMENT:-development}"

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_error() {
    echo "[ERROR] $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
}

deploy_storage() {
    log_info "Deploying storage configuration..."
    kubectl apply -f kubernetes/database-storage.yaml
    log_success "Storage configuration deployed"
}

deploy_secrets() {
    log_info "Deploying database secrets..."
    
    if kubectl get secret database-secrets -n "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Database secrets already exist"
        return
    fi
    
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    
    kubectl create secret generic database-secrets -n "$NAMESPACE" \
        --from-literal=postgres-db="prometheus" \
        --from-literal=postgres-user="prometheus" \
        --from-literal=postgres-password="$POSTGRES_PASSWORD"
    
    log_success "Database secrets created"
}

deploy_databases() {
    log_info "Deploying database StatefulSets..."
    
    kubectl apply -f kubernetes/database-statefulsets.yaml
    kubectl apply -f kubernetes/database-services.yaml
    
    log_success "Database StatefulSets deployed"
}

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
            log_success "Database stack deployment completed!"
            ;;
        *)
            echo "Usage: $0 {deploy}"
            exit 1
            ;;
    esac
}

main "$@"