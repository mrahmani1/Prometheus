#!/bin/bash

# Real-Time Process Monitor Deployment Script
# Deploys the geological process monitoring system with all dependencies

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${1:-development}"
NAMESPACE="prometheus-ai"
SERVICE_NAME="real-time-process-monitor"

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
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
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

# Deploy database dependencies
deploy_databases() {
    log_info "Deploying database dependencies..."
    
    # Deploy Redis
    log_info "Deploying Redis..."
    helm repo add bitnami https://charts.bitnami.com/bitnami || true
    helm repo update
    
    helm upgrade --install redis bitnami/redis \
        --namespace "$NAMESPACE" \
        --set auth.enabled=true \
        --set auth.password="$(openssl rand -base64 32)" \
        --set master.persistence.enabled=true \
        --set master.persistence.size=10Gi \
        --set replica.replicaCount=2 \
        --set replica.persistence.enabled=true \
        --set replica.persistence.size=10Gi \
        --set metrics.enabled=true \
        --wait
    
    # Deploy InfluxDB
    log_info "Deploying InfluxDB..."
    helm upgrade --install influxdb bitnami/influxdb \
        --namespace "$NAMESPACE" \
        --set auth.admin.username=admin \
        --set auth.admin.password="$(openssl rand -base64 32)" \
        --set auth.admin.token="$(openssl rand -base64 64)" \
        --set persistence.enabled=true \
        --set persistence.size=50Gi \
        --set metrics.enabled=true \
        --wait
    
    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL..."
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace "$NAMESPACE" \
        --set auth.postgresPassword="$(openssl rand -base64 32)" \
        --set auth.database=prometheus \
        --set primary.persistence.enabled=true \
        --set primary.persistence.size=20Gi \
        --set readReplicas.replicaCount=1 \
        --set metrics.enabled=true \
        --wait
    
    log_success "Database dependencies deployed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    local image_tag="${SERVICE_NAME}:${DEPLOYMENT_ENV}-$(date +%Y%m%d-%H%M%S)"
    local registry="${DOCKER_REGISTRY:-localhost:5000}"
    local full_image="${registry}/${image_tag}"
    
    # Build Docker image
    docker build -t "$full_image" -f "$PROJECT_ROOT/backend/Dockerfile.real_time_monitor" "$PROJECT_ROOT"
    
    # Push to registry
    docker push "$full_image"
    
    log_success "Docker image built and pushed: $full_image"
    echo "$full_image" > "$PROJECT_ROOT/.last_image"
}

# Create Kubernetes manifests
create_k8s_manifests() {
    log_info "Creating Kubernetes manifests..."
    
    local manifests_dir="$PROJECT_ROOT/kubernetes/real-time-monitor"
    mkdir -p "$manifests_dir"
    
    # ConfigMap for configuration
    cat > "$manifests_dir/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${SERVICE_NAME}-config
  namespace: ${NAMESPACE}
data:
  config.yaml: |
$(sed 's/^/    /' "$PROJECT_ROOT/backend/config/real_time_monitor_config.yaml")
EOF

    # Secret for sensitive configuration
    cat > "$manifests_dir/secret.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: ${SERVICE_NAME}-secrets
  namespace: ${NAMESPACE}
type: Opaque
data:
  redis-password: $(kubectl get secret redis -n "$NAMESPACE" -o jsonpath="{.data.redis-password}" 2>/dev/null || echo "")
  influxdb-token: $(kubectl get secret influxdb -n "$NAMESPACE" -o jsonpath="{.data.admin-token}" 2>/dev/null || echo "")
  postgres-password: $(kubectl get secret postgresql -n "$NAMESPACE" -o jsonpath="{.data.postgres-password}" 2>/dev/null || echo "")
  jwt-secret: $(echo -n "$(openssl rand -base64 32)" | base64 -w 0)
  encryption-key: $(echo -n "$(openssl rand -base64 32)" | base64 -w 0)
EOF

    # Deployment
    local image_name
    if [[ -f "$PROJECT_ROOT/.last_image" ]]; then
        image_name=$(cat "$PROJECT_ROOT/.last_image")
    else
        image_name="${SERVICE_NAME}:latest"
    fi
    
    cat > "$manifests_dir/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
    version: ${DEPLOYMENT_ENV}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
  template:
    metadata:
      labels:
        app: ${SERVICE_NAME}
        version: ${DEPLOYMENT_ENV}
    spec:
      containers:
      - name: ${SERVICE_NAME}
        image: ${image_name}
        ports:
        - containerPort: 8765
          name: websocket
        - containerPort: 8080
          name: health
        - containerPort: 9090
          name: metrics
        env:
        - name: DEPLOYMENT_ENV
          value: "${DEPLOYMENT_ENV}"
        - name: REDIS_HOST
          value: "redis-master.${NAMESPACE}.svc.cluster.local"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ${SERVICE_NAME}-secrets
              key: redis-password
        - name: INFLUXDB_URL
          value: "http://influxdb.${NAMESPACE}.svc.cluster.local:8086"
        - name: INFLUXDB_TOKEN
          valueFrom:
            secretKeyRef:
              name: ${SERVICE_NAME}-secrets
              key: influxdb-token
        - name: POSTGRES_HOST
          value: "postgresql.${NAMESPACE}.svc.cluster.local"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ${SERVICE_NAME}-secrets
              key: postgres-password
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ${SERVICE_NAME}-secrets
              key: jwt-secret
        - name: ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: ${SERVICE_NAME}-secrets
              key: encryption-key
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: ${SERVICE_NAME}-config
EOF

    # Service
    cat > "$manifests_dir/service.yaml" << EOF
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
spec:
  selector:
    app: ${SERVICE_NAME}
  ports:
  - name: websocket
    port: 8765
    targetPort: 8765
    protocol: TCP
  - name: health
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
EOF

    # Ingress
    cat > "$manifests_dir/ingress.yaml" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: "${SERVICE_NAME}"
spec:
  rules:
  - host: monitor.prometheus-ai.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ${SERVICE_NAME}
            port:
              number: 8765
  - host: monitor-health.prometheus-ai.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ${SERVICE_NAME}
            port:
              number: 8080
EOF

    # HorizontalPodAutoscaler
    cat > "$manifests_dir/hpa.yaml" << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${SERVICE_NAME}
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

    # ServiceMonitor for Prometheus
    cat > "$manifests_dir/servicemonitor.yaml" << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
spec:
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF

    log_success "Kubernetes manifests created in $manifests_dir"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    local manifests_dir="$PROJECT_ROOT/kubernetes/real-time-monitor"
    
    # Apply manifests in order
    kubectl apply -f "$manifests_dir/configmap.yaml"
    kubectl apply -f "$manifests_dir/secret.yaml"
    kubectl apply -f "$manifests_dir/deployment.yaml"
    kubectl apply -f "$manifests_dir/service.yaml"
    kubectl apply -f "$manifests_dir/ingress.yaml"
    kubectl apply -f "$manifests_dir/hpa.yaml"
    kubectl apply -f "$manifests_dir/servicemonitor.yaml"
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/"$SERVICE_NAME" -n "$NAMESPACE" --timeout=300s
    
    log_success "Deployment completed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" --no-headers | wc -l)
    if [[ $pods -gt 0 ]]; then
        log_success "Found $pods running pods"
    else
        log_error "No pods found for $SERVICE_NAME"
        return 1
    fi
    
    # Check service endpoints
    local endpoints
    endpoints=$(kubectl get endpoints -n "$NAMESPACE" "$SERVICE_NAME" -o jsonpath='{.subsets[0].addresses}' | jq length 2>/dev/null || echo "0")
    if [[ $endpoints -gt 0 ]]; then
        log_success "Service has $endpoints ready endpoints"
    else
        log_warning "Service has no ready endpoints"
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if kubectl port-forward -n "$NAMESPACE" service/"$SERVICE_NAME" 8080:8080 &
    then
        local port_forward_pid=$!
        sleep 5
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed"
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    # Show deployment status
    kubectl get all -n "$NAMESPACE" -l app="$SERVICE_NAME"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Deploy Prometheus if not exists
    if ! kubectl get deployment prometheus -n "$NAMESPACE" &> /dev/null; then
        log_info "Deploying Prometheus..."
        helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace "$NAMESPACE" \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --wait
    fi
    
    # Deploy Grafana dashboard
    local dashboard_dir="$PROJECT_ROOT/monitoring/grafana/dashboards"
    mkdir -p "$dashboard_dir"
    
    cat > "$dashboard_dir/real-time-monitor-dashboard.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Real-Time Process Monitor",
    "tags": ["prometheus", "geological", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Data Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(geological_data_points_total[5m])",
            "legendFormat": "Data Points/sec"
          }
        ]
      },
      {
        "title": "Processing Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(processing_latency_seconds_bucket[5m]))",
            "legendFormat": "95th Percentile"
          }
        ]
      },
      {
        "title": "Active Processes",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_geological_processes",
            "legendFormat": "Processes"
          }
        ]
      },
      {
        "title": "Alert Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(geological_alerts_total[5m])",
            "legendFormat": "Alerts/sec"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    log_success "Monitoring setup completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f "$PROJECT_ROOT/.last_image"
}

# Main deployment function
main() {
    log_info "Starting Real-Time Process Monitor deployment for environment: $DEPLOYMENT_ENV"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    create_namespace
    
    if [[ "$DEPLOYMENT_ENV" != "development" ]]; then
        deploy_databases
        build_and_push_image
    fi
    
    create_k8s_manifests
    deploy_to_kubernetes
    verify_deployment
    setup_monitoring
    
    log_success "Real-Time Process Monitor deployment completed successfully!"
    
    # Show access information
    echo ""
    log_info "Access Information:"
    echo "  WebSocket Endpoint: ws://monitor.prometheus-ai.local"
    echo "  Health Check: http://monitor-health.prometheus-ai.local/health"
    echo "  Metrics: http://monitor-health.prometheus-ai.local/metrics"
    echo ""
    log_info "To access the dashboard locally:"
    echo "  kubectl port-forward -n $NAMESPACE service/$SERVICE_NAME 8765:8765"
    echo "  kubectl port-forward -n $NAMESPACE service/$SERVICE_NAME 8080:8080"
    echo ""
    log_info "To view logs:"
    echo "  kubectl logs -n $NAMESPACE -l app=$SERVICE_NAME -f"
}

# Run main function
main "$@"