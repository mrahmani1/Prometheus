#!/bin/bash

# Deploy Service Mesh Infrastructure for Prometheus AI Platform
# Advanced service mesh deployment with Consul Connect, circuit breakers, and observability

set -euo pipefail

# Configuration
NAMESPACE="prometheus-ai-mesh"
CONSUL_VERSION="1.16.1"
ENVOY_VERSION="1.27.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace "$NAMESPACE" istio-injection=enabled --overwrite
    
    log_success "Namespace created and labeled"
}

# Generate Consul gossip encryption key
generate_consul_key() {
    log_info "Generating Consul gossip encryption key..."
    
    GOSSIP_KEY=$(openssl rand -base64 32)
    
    kubectl create secret generic consul-gossip-key \
        --namespace="$NAMESPACE" \
        --from-literal=key="$GOSSIP_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Consul gossip key generated and stored"
}

# Create ConfigMaps for service mesh components
create_configmaps() {
    log_info "Creating ConfigMaps for service mesh components..."
    
    # Circuit Breaker ConfigMap
    kubectl create configmap circuit-breaker-code \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/circuit-breaker.py" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Rate Limiter ConfigMap
    kubectl create configmap rate-limiter-code \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/rate-limiter.py" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Traffic Manager ConfigMap
    kubectl create configmap traffic-manager-code \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/traffic-management.py" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Service Discovery ConfigMap
    kubectl create configmap service-discovery-code \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/service-discovery.py" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Chaos Monkey ConfigMap
    kubectl create configmap chaos-monkey-code \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/chaos-engineering.py" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Observability ConfigMap
    kubectl create configmap observability-code \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/observability.py" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Consul Configuration
    kubectl create configmap consul-config \
        --namespace="$NAMESPACE" \
        --from-file="$PROJECT_ROOT/services/service-mesh/consul-connect-config.yaml" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "ConfigMaps created"
}

# Deploy Consul using Helm
deploy_consul_helm() {
    log_info "Deploying Consul using Helm..."
    
    # Add HashiCorp Helm repository
    helm repo add hashicorp https://helm.releases.hashicorp.com
    helm repo update
    
    # Create Consul values file
    cat > /tmp/consul-values.yaml << EOF
global:
  name: consul
  datacenter: prometheus-ai
  image: "consul:${CONSUL_VERSION}"
  imageK8S: "hashicorp/consul-k8s-control-plane:1.2.1"
  
server:
  replicas: 3
  bootstrapExpect: 3
  storage: 10Gi
  storageClass: ""
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"
  
connectInject:
  enabled: true
  default: false
  
controller:
  enabled: true
  
meshGateway:
  enabled: true
  replicas: 2
  
ui:
  enabled: true
  service:
    type: LoadBalancer
  
client:
  enabled: true
  
dns:
  enabled: true
  
syncCatalog:
  enabled: true
  default: true
EOF
    
    # Deploy Consul
    helm upgrade --install consul hashicorp/consul \
        --namespace="$NAMESPACE" \
        --values=/tmp/consul-values.yaml \
        --wait \
        --timeout=10m
    
    # Clean up values file
    rm -f /tmp/consul-values.yaml
    
    log_success "Consul deployed via Helm"
}

# Deploy service mesh components
deploy_service_mesh() {
    log_info "Deploying service mesh components..."
    
    kubectl apply -f "$PROJECT_ROOT/kubernetes/service-mesh-deployment.yaml"
    
    log_success "Service mesh components deployed"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    local deployments=(
        "consul-server"
        "consul-connect-injector"
        "mesh-gateway"
        "circuit-breaker-service"
        "rate-limiter-service"
        "traffic-manager-service"
        "service-discovery"
        "mesh-observability"
    )
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for deployment: $deployment"
        kubectl wait --for=condition=available \
            --timeout=300s \
            deployment/"$deployment" \
            -n "$NAMESPACE" || {
            log_warning "Deployment $deployment did not become ready within timeout"
        }
    done
    
    # Wait for StatefulSet (Consul server)
    kubectl wait --for=condition=ready \
        --timeout=300s \
        pod -l app=consul-server \
        -n "$NAMESPACE" || {
        log_warning "Consul server pods did not become ready within timeout"
    }
    
    log_success "All deployments are ready"
}

# Configure Consul Connect policies
configure_consul_policies() {
    log_info "Configuring Consul Connect policies..."
    
    # Wait for Consul to be ready
    kubectl wait --for=condition=ready pod -l app=consul-server -n "$NAMESPACE" --timeout=300s
    
    # Get Consul pod name
    CONSUL_POD=$(kubectl get pods -n "$NAMESPACE" -l app=consul-server -o jsonpath='{.items[0].metadata.name}')
    
    # Configure service intentions
    kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul config write - << 'EOF'
Kind = "service-intentions"
Name = "api-gateway"
Sources = [
  {
    Name = "*"
    Action = "allow"
  }
]
EOF
    
    kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul config write - << 'EOF'
Kind = "service-intentions"
Name = "helios-core"
Sources = [
  {
    Name = "api-gateway"
    Action = "allow"
  },
  {
    Name = "data-ingestion"
    Action = "allow"
  },
  {
    Name = "volcano-watchtower"
    Action = "allow"
  }
]
EOF
    
    kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul config write - << 'EOF'
Kind = "proxy-defaults"
Name = "global"
Config = {
  protocol = "http"
  local_rate_limit = {
    token_bucket = {
      max_tokens = 1000
      tokens_per_fill = 100
      fill_interval = "1s"
    }
  }
  passive_health_check = {
    interval = "10s"
    max_failures = 3
    enforcing_consecutive_5xx = 100
  }
}
EOF
    
    log_success "Consul Connect policies configured"
}

# Setup monitoring and observability
setup_monitoring() {
    log_info "Setting up monitoring and observability..."
    
    # Create ServiceMonitor for Prometheus
    kubectl apply -f - << 'EOF'
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: service-mesh-monitoring
  namespace: prometheus-ai-mesh
  labels:
    app: service-mesh
spec:
  selector:
    matchLabels:
      app: consul-server
  endpoints:
  - port: http
    path: /v1/agent/metrics
    params:
      format: ['prometheus']
    interval: 30s
    scrapeTimeout: 10s
EOF
    
    # Create Grafana dashboard ConfigMap
    kubectl create configmap service-mesh-dashboard \
        --namespace="$NAMESPACE" \
        --from-literal=dashboard.json='{
          "dashboard": {
            "title": "Service Mesh Overview",
            "panels": [
              {
                "title": "Request Rate",
                "type": "graph",
                "targets": [
                  {
                    "expr": "sum(rate(service_mesh_requests_total[5m])) by (source_service, destination_service)"
                  }
                ]
              },
              {
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                  {
                    "expr": "sum(rate(service_mesh_requests_total{status=~\"5..\"}[5m])) / sum(rate(service_mesh_requests_total[5m]))"
                  }
                ]
              },
              {
                "title": "Response Time",
                "type": "graph",
                "targets": [
                  {
                    "expr": "histogram_quantile(0.95, sum(rate(service_mesh_request_duration_seconds_bucket[5m])) by (le, source_service, destination_service))"
                  }
                ]
              }
            ]
          }
        }' \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Monitoring and observability configured"
}

# Validate deployment
validate_deployment() {
    log_info "Validating service mesh deployment..."
    
    # Check Consul cluster status
    CONSUL_POD=$(kubectl get pods -n "$NAMESPACE" -l app=consul-server -o jsonpath='{.items[0].metadata.name}')
    
    log_info "Checking Consul cluster status..."
    kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul members
    
    # Check service registrations
    log_info "Checking service registrations..."
    kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul catalog services
    
    # Check Connect status
    log_info "Checking Connect status..."
    kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul connect ca get-config
    
    # Test service connectivity
    log_info "Testing service connectivity..."
    
    # Get service endpoints
    kubectl get endpoints -n "$NAMESPACE"
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -o wide
    
    log_success "Service mesh deployment validation completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    REPORT_FILE="/tmp/service-mesh-deployment-report.txt"
    
    cat > "$REPORT_FILE" << EOF
Service Mesh Deployment Report
==============================
Date: $(date)
Namespace: $NAMESPACE
Consul Version: $CONSUL_VERSION
Envoy Version: $ENVOY_VERSION

Deployed Components:
-------------------
EOF
    
    kubectl get all -n "$NAMESPACE" >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
    echo "Service Mesh Configuration:" >> "$REPORT_FILE"
    echo "-------------------------" >> "$REPORT_FILE"
    
    # Get Consul configuration
    CONSUL_POD=$(kubectl get pods -n "$NAMESPACE" -l app=consul-server -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$CONSUL_POD" ]]; then
        echo "Consul Services:" >> "$REPORT_FILE"
        kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul catalog services >> "$REPORT_FILE" 2>/dev/null || echo "Failed to get services" >> "$REPORT_FILE"
        
        echo "" >> "$REPORT_FILE"
        echo "Consul Members:" >> "$REPORT_FILE"
        kubectl exec -n "$NAMESPACE" "$CONSUL_POD" -- consul members >> "$REPORT_FILE" 2>/dev/null || echo "Failed to get members" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "Access Information:" >> "$REPORT_FILE"
    echo "-----------------" >> "$REPORT_FILE"
    echo "Consul UI: kubectl port-forward -n $NAMESPACE svc/consul-ui 8500:8500" >> "$REPORT_FILE"
    echo "Then access: http://localhost:8500" >> "$REPORT_FILE"
    
    log_success "Deployment report generated: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/consul-values.yaml
    rm -f /tmp/service-mesh-deployment-report.txt
}

# Main deployment function
main() {
    log_info "Starting service mesh deployment for Prometheus AI Platform"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    create_namespace
    generate_consul_key
    create_configmaps
    deploy_consul_helm
    deploy_service_mesh
    wait_for_deployments
    configure_consul_policies
    setup_monitoring
    validate_deployment
    generate_report
    
    log_success "Service mesh deployment completed successfully!"
    log_info "Access Consul UI with: kubectl port-forward -n $NAMESPACE svc/consul-ui 8500:8500"
    log_info "Then open: http://localhost:8500"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "validate")
        validate_deployment
        ;;
    "cleanup")
        log_info "Cleaning up service mesh deployment..."
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        log_success "Service mesh cleanup completed"
        ;;
    "status")
        log_info "Service mesh status:"
        kubectl get all -n "$NAMESPACE"
        ;;
    *)
        echo "Usage: $0 [deploy|validate|cleanup|status]"
        echo "  deploy   - Deploy the service mesh (default)"
        echo "  validate - Validate existing deployment"
        echo "  cleanup  - Remove the service mesh"
        echo "  status   - Show deployment status"
        exit 1
        ;;
esac