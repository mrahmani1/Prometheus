#!/bin/bash

# Prometheus AI Platform Deployment Validation Script
# This script validates the deployment and performs comprehensive health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[VALIDATION]${NC} $1"
}

# Configuration
NAMESPACE="prometheus-ai"
TIMEOUT="300s"

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

print_header "Starting Prometheus AI Platform deployment validation..."

# Function to check deployment status
check_deployment() {
    local deployment=$1
    print_status "Checking deployment: $deployment"
    
    if kubectl get deployment $deployment -n $NAMESPACE &> /dev/null; then
        local ready=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
        local desired=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}')
        
        if [ "$ready" = "$desired" ] && [ "$ready" != "" ]; then
            echo "  ✅ $deployment: $ready/$desired replicas ready"
            return 0
        else
            echo "  ❌ $deployment: $ready/$desired replicas ready"
            return 1
        fi
    else
        echo "  ❌ $deployment: Not found"
        return 1
    fi
}

# Function to check service status
check_service() {
    local service=$1
    print_status "Checking service: $service"
    
    if kubectl get service $service -n $NAMESPACE &> /dev/null; then
        local endpoints=$(kubectl get endpoints $service -n $NAMESPACE -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
        if [ "$endpoints" -gt 0 ]; then
            echo "  ✅ $service: $endpoints endpoints available"
            return 0
        else
            echo "  ❌ $service: No endpoints available"
            return 1
        fi
    else
        echo "  ❌ $service: Not found"
        return 1
    fi
}

# Function to check pod health
check_pod_health() {
    local app=$1
    print_status "Checking pod health for app: $app"
    
    local pods=$(kubectl get pods -n $NAMESPACE -l app=$app -o jsonpath='{.items[*].metadata.name}')
    if [ -z "$pods" ]; then
        echo "  ❌ No pods found for app: $app"
        return 1
    fi
    
    local healthy=0
    local total=0
    
    for pod in $pods; do
        total=$((total + 1))
        local status=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.phase}')
        local ready=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        
        if [ "$status" = "Running" ] && [ "$ready" = "True" ]; then
            healthy=$((healthy + 1))
            echo "  ✅ $pod: Running and Ready"
        else
            echo "  ❌ $pod: $status, Ready: $ready"
        fi
    done
    
    if [ "$healthy" -eq "$total" ]; then
        echo "  ✅ All $total pods healthy for app: $app"
        return 0
    else
        echo "  ❌ Only $healthy/$total pods healthy for app: $app"
        return 1
    fi
}

# Function to check persistent volumes
check_persistent_volumes() {
    print_status "Checking persistent volumes"
    
    local pvcs=$(kubectl get pvc -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    if [ -z "$pvcs" ]; then
        echo "  ❌ No PVCs found"
        return 1
    fi
    
    local bound=0
    local total=0
    
    for pvc in $pvcs; do
        total=$((total + 1))
        local status=$(kubectl get pvc $pvc -n $NAMESPACE -o jsonpath='{.status.phase}')
        
        if [ "$status" = "Bound" ]; then
            bound=$((bound + 1))
            echo "  ✅ $pvc: Bound"
        else
            echo "  ❌ $pvc: $status"
        fi
    done
    
    if [ "$bound" -eq "$total" ]; then
        echo "  ✅ All $total PVCs bound"
        return 0
    else
        echo "  ❌ Only $bound/$total PVCs bound"
        return 1
    fi
}

# Function to check horizontal pod autoscalers
check_hpa() {
    print_status "Checking Horizontal Pod Autoscalers"
    
    local hpas=$(kubectl get hpa -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    if [ -z "$hpas" ]; then
        echo "  ❌ No HPAs found"
        return 1
    fi
    
    local active=0
    local total=0
    
    for hpa in $hpas; do
        total=$((total + 1))
        local current=$(kubectl get hpa $hpa -n $NAMESPACE -o jsonpath='{.status.currentReplicas}')
        local desired=$(kubectl get hpa $hpa -n $NAMESPACE -o jsonpath='{.status.desiredReplicas}')
        
        if [ "$current" = "$desired" ] && [ "$current" != "" ]; then
            active=$((active + 1))
            echo "  ✅ $hpa: $current/$desired replicas"
        else
            echo "  ❌ $hpa: $current/$desired replicas"
        fi
    done
    
    if [ "$active" -eq "$total" ]; then
        echo "  ✅ All $total HPAs active"
        return 0
    else
        echo "  ❌ Only $active/$total HPAs active"
        return 1
    fi
}

# Function to check ingress
check_ingress() {
    print_status "Checking Ingress configuration"
    
    local ingresses=$(kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    if [ -z "$ingresses" ]; then
        echo "  ❌ No Ingress found"
        return 1
    fi
    
    for ingress in $ingresses; do
        local hosts=$(kubectl get ingress $ingress -n $NAMESPACE -o jsonpath='{.spec.rules[*].host}')
        local ip=$(kubectl get ingress $ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        if [ -n "$ip" ]; then
            echo "  ✅ $ingress: $hosts -> $ip"
        else
            echo "  ❌ $ingress: No external IP assigned"
        fi
    done
}

# Function to perform health checks
perform_health_checks() {
    print_status "Performing application health checks"
    
    # Check API Gateway health
    if kubectl get service api-gateway-service -n $NAMESPACE &> /dev/null; then
        local api_ip=$(kubectl get service api-gateway-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$api_ip" ]; then
            if curl -f -s "http://$api_ip/health" > /dev/null; then
                echo "  ✅ API Gateway health check passed"
            else
                echo "  ❌ API Gateway health check failed"
            fi
        else
            # Try port-forward for health check
            kubectl port-forward service/api-gateway-service 8080:3000 -n $NAMESPACE &
            local pf_pid=$!
            sleep 5
            
            if curl -f -s "http://localhost:8080/health" > /dev/null; then
                echo "  ✅ API Gateway health check passed (via port-forward)"
            else
                echo "  ❌ API Gateway health check failed"
            fi
            
            kill $pf_pid 2>/dev/null || true
        fi
    fi
    
    # Check database connectivity
    if kubectl get pod -l app=postgres -n $NAMESPACE &> /dev/null; then
        local postgres_pod=$(kubectl get pod -l app=postgres -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
        if kubectl exec $postgres_pod -n $NAMESPACE -- pg_isready -U prometheus > /dev/null 2>&1; then
            echo "  ✅ PostgreSQL connectivity check passed"
        else
            echo "  ❌ PostgreSQL connectivity check failed"
        fi
    fi
    
    # Check Redis connectivity
    if kubectl get pod -l app=redis -n $NAMESPACE &> /dev/null; then
        local redis_pod=$(kubectl get pod -l app=redis -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
        if kubectl exec $redis_pod -n $NAMESPACE -- redis-cli ping | grep -q PONG; then
            echo "  ✅ Redis connectivity check passed"
        else
            echo "  ❌ Redis connectivity check failed"
        fi
    fi
}

# Function to check monitoring stack
check_monitoring() {
    print_status "Checking monitoring stack"
    
    # Check Prometheus
    if kubectl get service prometheus-service -n $NAMESPACE &> /dev/null; then
        kubectl port-forward service/prometheus-service 9090:9090 -n $NAMESPACE &
        local pf_pid=$!
        sleep 5
        
        if curl -f -s "http://localhost:9090/-/healthy" > /dev/null; then
            echo "  ✅ Prometheus health check passed"
        else
            echo "  ❌ Prometheus health check failed"
        fi
        
        kill $pf_pid 2>/dev/null || true
    fi
    
    # Check Grafana
    if kubectl get service grafana-service -n $NAMESPACE &> /dev/null; then
        kubectl port-forward service/grafana-service 3000:3000 -n $NAMESPACE &
        local pf_pid=$!
        sleep 5
        
        if curl -f -s "http://localhost:3000/api/health" > /dev/null; then
            echo "  ✅ Grafana health check passed"
        else
            echo "  ❌ Grafana health check failed"
        fi
        
        kill $pf_pid 2>/dev/null || true
    fi
}

# Function to generate validation report
generate_report() {
    print_header "Validation Report Summary"
    
    echo ""
    echo "Cluster Information:"
    kubectl cluster-info | head -1
    echo "Kubernetes Version: $(kubectl version --short --client | cut -d' ' -f3)"
    echo "Namespace: $NAMESPACE"
    echo "Validation Time: $(date)"
    
    echo ""
    echo "Resource Summary:"
    echo "Deployments: $(kubectl get deployments -n $NAMESPACE --no-headers | wc -l)"
    echo "Services: $(kubectl get services -n $NAMESPACE --no-headers | wc -l)"
    echo "Pods: $(kubectl get pods -n $NAMESPACE --no-headers | wc -l)"
    echo "PVCs: $(kubectl get pvc -n $NAMESPACE --no-headers | wc -l)"
    echo "HPAs: $(kubectl get hpa -n $NAMESPACE --no-headers | wc -l)"
    
    echo ""
    echo "Resource Usage:"
    kubectl top nodes 2>/dev/null || echo "Metrics server not available"
    kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Pod metrics not available"
}

# Main validation sequence
main() {
    local failed_checks=0
    
    # Check namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        print_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    print_header "Phase 1: Infrastructure Validation"
    
    # Check persistent volumes
    if ! check_persistent_volumes; then
        failed_checks=$((failed_checks + 1))
    fi
    
    print_header "Phase 2: Database Validation"
    
    # Check database deployments
    for db in postgres influxdb neo4j redis kafka; do
        if ! check_deployment "${db}-deployment"; then
            failed_checks=$((failed_checks + 1))
        fi
        if ! check_service "${db}-service"; then
            failed_checks=$((failed_checks + 1))
        fi
        if ! check_pod_health "$db"; then
            failed_checks=$((failed_checks + 1))
        fi
    done
    
    print_header "Phase 3: Service Mesh Validation"
    
    # Check Consul
    if ! check_deployment "consul-deployment"; then
        failed_checks=$((failed_checks + 1))
    fi
    if ! check_service "consul-service"; then
        failed_checks=$((failed_checks + 1))
    fi
    
    print_header "Phase 4: Application Services Validation"
    
    # Check application services
    for service in api-gateway helios-core data-ingestion volcano-watchtower; do
        if ! check_deployment "${service}-deployment"; then
            failed_checks=$((failed_checks + 1))
        fi
        if ! check_service "${service}-service"; then
            failed_checks=$((failed_checks + 1))
        fi
        if ! check_pod_health "$service"; then
            failed_checks=$((failed_checks + 1))
        fi
    done
    
    print_header "Phase 5: Monitoring Stack Validation"
    
    # Check monitoring services
    for monitor in prometheus grafana; do
        if ! check_deployment "${monitor}-deployment"; then
            failed_checks=$((failed_checks + 1))
        fi
        if ! check_service "${monitor}-service"; then
            failed_checks=$((failed_checks + 1))
        fi
    done
    
    print_header "Phase 6: Autoscaling Validation"
    
    # Check HPAs
    if ! check_hpa; then
        failed_checks=$((failed_checks + 1))
    fi
    
    print_header "Phase 7: Ingress Validation"
    
    # Check ingress
    if ! check_ingress; then
        failed_checks=$((failed_checks + 1))
    fi
    
    print_header "Phase 8: Health Checks"
    
    # Perform health checks
    perform_health_checks
    
    print_header "Phase 9: Monitoring Validation"
    
    # Check monitoring stack
    check_monitoring
    
    # Generate report
    generate_report
    
    # Final result
    echo ""
    if [ $failed_checks -eq 0 ]; then
        print_status "🎉 All validation checks passed! Deployment is healthy."
        echo ""
        print_status "Access URLs:"
        echo "🌐 API Gateway: kubectl port-forward svc/api-gateway-service 8080:3000 -n $NAMESPACE"
        echo "📊 Prometheus: kubectl port-forward svc/prometheus-service 9090:9090 -n $NAMESPACE"
        echo "📈 Grafana: kubectl port-forward svc/grafana-service 3000:3000 -n $NAMESPACE"
        echo "🔍 Consul UI: kubectl port-forward svc/consul-service 8500:8500 -n $NAMESPACE"
        exit 0
    else
        print_error "❌ $failed_checks validation checks failed. Please review the output above."
        echo ""
        print_warning "Troubleshooting commands:"
        echo "kubectl get all -n $NAMESPACE"
        echo "kubectl describe pods -n $NAMESPACE"
        echo "kubectl logs -l app=api-gateway -n $NAMESPACE"
        exit 1
    fi
}

# Run main function
main "$@"