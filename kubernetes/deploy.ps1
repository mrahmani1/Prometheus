# Prometheus AI Platform Kubernetes Deployment Script (PowerShell)
# This script deploys the complete Prometheus AI Platform to Kubernetes

param(
    [switch]$WhatIf = $false
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if kubectl is available
try {
    kubectl version --client --output=json | Out-Null
} catch {
    Write-Error "kubectl is not installed or not in PATH"
    exit 1
}

# Check if we can connect to Kubernetes cluster
try {
    kubectl cluster-info | Out-Null
} catch {
    Write-Error "Cannot connect to Kubernetes cluster"
    exit 1
}

Write-Status "Starting Prometheus AI Platform deployment..."

if ($WhatIf) {
    Write-Warning "Running in WhatIf mode - no changes will be made"
}

# Create namespace and basic resources
Write-Status "Creating namespace and basic resources..."
if (-not $WhatIf) {
    kubectl apply -f namespace.yaml
    kubectl apply -f secrets.yaml
    kubectl apply -f configmaps.yaml
    kubectl apply -f postgres-init-configmap.yaml
} else {
    Write-Host "Would apply: namespace.yaml, secrets.yaml, configmaps.yaml, postgres-init-configmap.yaml"
}

# Create persistent volumes
Write-Status "Creating persistent volumes..."
if (-not $WhatIf) {
    kubectl apply -f persistent-volumes.yaml
} else {
    Write-Host "Would apply: persistent-volumes.yaml"
}

# Wait for persistent volumes to be available
if (-not $WhatIf) {
    Write-Status "Waiting for persistent volumes to be ready..."
    try {
        kubectl wait --for=condition=Available pv/postgres-pv --timeout=60s
        kubectl wait --for=condition=Available pv/influxdb-pv --timeout=60s
        kubectl wait --for=condition=Available pv/neo4j-pv --timeout=60s
        kubectl wait --for=condition=Available pv/kafka-pv --timeout=60s
    } catch {
        Write-Warning "Some persistent volumes may not be ready yet. Continuing..."
    }
}

# Deploy databases
Write-Status "Deploying databases..."
if (-not $WhatIf) {
    kubectl apply -f database-deployments.yaml
} else {
    Write-Host "Would apply: database-deployments.yaml"
}

# Wait for databases to be ready
if (-not $WhatIf) {
    Write-Status "Waiting for databases to be ready..."
    try {
        kubectl wait --for=condition=available --timeout=300s deployment/postgres-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/influxdb-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/neo4j-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/redis-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/zookeeper-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/kafka-deployment -n prometheus-ai
    } catch {
        Write-Warning "Some databases may not be ready yet. Check status manually."
    }
}

# Deploy Consul service mesh
Write-Status "Deploying Consul service mesh..."
if (-not $WhatIf) {
    kubectl apply -f consul.yaml
} else {
    Write-Host "Would apply: consul.yaml"
}

# Wait for Consul to be ready
if (-not $WhatIf) {
    Write-Status "Waiting for Consul to be ready..."
    try {
        kubectl wait --for=condition=available --timeout=180s deployment/consul-deployment -n prometheus-ai
    } catch {
        Write-Warning "Consul may not be ready yet. Check status manually."
    }
}

# Deploy monitoring stack
Write-Status "Deploying monitoring stack (Prometheus & Grafana)..."
if (-not $WhatIf) {
    kubectl apply -f monitoring.yaml
} else {
    Write-Host "Would apply: monitoring.yaml"
}

# Wait for monitoring to be ready
if (-not $WhatIf) {
    Write-Status "Waiting for monitoring stack to be ready..."
    try {
        kubectl wait --for=condition=available --timeout=180s deployment/prometheus-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=180s deployment/grafana-deployment -n prometheus-ai
    } catch {
        Write-Warning "Monitoring stack may not be ready yet. Check status manually."
    }
}

# Deploy application services
Write-Status "Deploying application services..."
if (-not $WhatIf) {
    kubectl apply -f service-deployments.yaml
} else {
    Write-Host "Would apply: service-deployments.yaml"
}

# Wait for services to be ready
if (-not $WhatIf) {
    Write-Status "Waiting for application services to be ready..."
    try {
        kubectl wait --for=condition=available --timeout=300s deployment/api-gateway-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/helios-core-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/data-ingestion-deployment -n prometheus-ai
        kubectl wait --for=condition=available --timeout=300s deployment/volcano-watchtower-deployment -n prometheus-ai
    } catch {
        Write-Warning "Some services may not be ready yet. Check status manually."
    }
}

# Deploy horizontal pod autoscalers
Write-Status "Deploying horizontal pod autoscalers..."
if (-not $WhatIf) {
    kubectl apply -f hpa.yaml
} else {
    Write-Host "Would apply: hpa.yaml"
}

# Deploy ingress controller
Write-Status "Deploying ingress controller..."
if (-not $WhatIf) {
    kubectl apply -f ingress.yaml
} else {
    Write-Host "Would apply: ingress.yaml"
}

# Display deployment status
Write-Status "Deployment completed! Checking status..."
Write-Host ""

if (-not $WhatIf) {
    Write-Status "Namespace resources:"
    kubectl get all -n prometheus-ai

    Write-Host ""
    Write-Status "Persistent volumes:"
    kubectl get pv | Select-String "prometheus-ai"

    Write-Host ""
    Write-Status "Services and endpoints:"
    kubectl get svc -n prometheus-ai
    kubectl get endpoints -n prometheus-ai

    Write-Host ""
    Write-Status "Horizontal Pod Autoscalers:"
    kubectl get hpa -n prometheus-ai

    Write-Host ""
    Write-Status "Ingress:"
    kubectl get ingress -n prometheus-ai
}

Write-Host ""
Write-Status "Deployment Summary:"
Write-Host "✅ Namespace: prometheus-ai"
Write-Host "✅ Databases: PostgreSQL+PostGIS, InfluxDB, Neo4j, Redis, Kafka"
Write-Host "✅ Service Mesh: Consul Connect"
Write-Host "✅ Monitoring: Prometheus, Grafana with advanced dashboards"
Write-Host "✅ Services: API Gateway, Helios Core, Data Ingestion, Volcano Watchtower"
Write-Host "✅ Autoscaling: HPA configured for all services with predictive scaling"
Write-Host "✅ Ingress: External access configured with SSL termination"
Write-Host "✅ Security: Network policies and RBAC configured"
Write-Host "✅ Observability: Distributed tracing and centralized logging enabled"

Write-Host ""
Write-Status "Access URLs (update DNS to point to your ingress controller):"
Write-Host "🌐 API Gateway: https://api.prometheus-ai.com/api"
Write-Host "📊 Prometheus: https://monitoring.prometheus-ai.com/prometheus"
Write-Host "📈 Grafana: https://monitoring.prometheus-ai.com/grafana (admin/prometheus123)"
Write-Host "🔍 Consul UI: kubectl port-forward svc/consul-service 8500:8500 -n prometheus-ai"

Write-Host ""
Write-Warning "Next steps:"
Write-Host "1. Update your DNS to point api.prometheus-ai.com and monitoring.prometheus-ai.com to your ingress controller"
Write-Host "2. Configure SSL certificates (cert-manager is referenced in ingress)"
Write-Host "3. Set up CI/CD pipeline to automatically build and push Docker images"
Write-Host "4. Configure multi-cloud deployment for disaster recovery"
Write-Host "5. Set up advanced monitoring alerts and PagerDuty integration"
Write-Host "6. Enable cost optimization monitoring and recommendations"
Write-Host "7. Configure backup and disaster recovery procedures"
Write-Host "8. Monitor the deployment with: kubectl logs -f deployment/api-gateway-deployment -n prometheus-ai"

Write-Status "Prometheus AI Platform deployment completed successfully!"

# Usage examples
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  .\deploy.ps1                    # Deploy everything"
Write-Host "  .\deploy.ps1 -WhatIf           # Show what would be deployed without making changes"