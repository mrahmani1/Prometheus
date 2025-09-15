# Prometheus AI Platform Database Stack Deployment Script (PowerShell)
# This script deploys the complete database infrastructure with monitoring and backup

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "helm", "cleanup", "verify", "info")]
    [string]$Action = "deploy",
    
    [Parameter()]
    [string]$Environment = "development",
    
    [Parameter()]
    [string]$Namespace = "prometheus-ai",
    
    [Parameter()]
    [string]$HelmReleaseName = "prometheus-database-stack",
    
    [Parameter()]
    [string]$KubectlTimeout = "300s"
)

# Error handling
$ErrorActionPreference = "Stop"

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if kubectl is installed
    try {
        kubectl version --client | Out-Null
    }
    catch {
        Write-Error "kubectl is not installed or not in PATH"
        exit 1
    }
    
    # Check if helm is installed
    try {
        helm version | Out-Null
    }
    catch {
        Write-Error "helm is not installed or not in PATH"
        exit 1
    }
    
    # Check if docker is installed (for local development)
    if ($Environment -eq "development") {
        try {
            docker version | Out-Null
        }
        catch {
            Write-Error "docker is not installed or not in PATH"
            exit 1
        }
    }
    
    # Check cluster connectivity
    try {
        kubectl cluster-info | Out-Null
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
function New-DatabaseNamespace {
    Write-Info "Creating namespace: $Namespace"
    
    try {
        kubectl get namespace $Namespace | Out-Null
        Write-Warning "Namespace $Namespace already exists"
    }
    catch {
        kubectl create namespace $Namespace
        kubectl label namespace $Namespace monitoring=enabled environment=$Environment
        Write-Success "Namespace $Namespace created"
    }
}

# Deploy storage classes and persistent volumes
function Deploy-Storage {
    Write-Info "Deploying storage classes and persistent volumes..."
    
    kubectl apply -f kubernetes/database-storage.yaml --timeout=$KubectlTimeout
    
    # Wait for storage classes to be ready (best effort)
    try {
        kubectl wait --for=condition=Ready storageclass/fast-ssd --timeout=$KubectlTimeout
        kubectl wait --for=condition=Ready storageclass/standard-hdd --timeout=$KubectlTimeout
    }
    catch {
        Write-Warning "Storage classes may not be immediately ready"
    }
    
    Write-Success "Storage configuration deployed"
}

# Generate and apply secrets
function Deploy-Secrets {
    Write-Info "Generating and deploying database secrets..."
    
    # Check if secrets already exist
    try {
        kubectl get secret database-secrets -n $Namespace | Out-Null
        Write-Warning "Database secrets already exist, skipping generation"
        return
    }
    catch {
        # Secrets don't exist, continue with generation
    }
    
    # Generate random passwords if not provided
    $PostgresPassword = if ($env:POSTGRES_PASSWORD) { $env:POSTGRES_PASSWORD } else { [System.Web.Security.Membership]::GeneratePassword(32, 8) }
    $RedisPassword = if ($env:REDIS_PASSWORD) { $env:REDIS_PASSWORD } else { [System.Web.Security.Membership]::GeneratePassword(32, 8) }
    $Neo4jPassword = if ($env:NEO4J_PASSWORD) { $env:NEO4J_PASSWORD } else { [System.Web.Security.Membership]::GeneratePassword(32, 8) }
    $InfluxdbPassword = if ($env:INFLUXDB_PASSWORD) { $env:INFLUXDB_PASSWORD } else { [System.Web.Security.Membership]::GeneratePassword(32, 8) }
    $TimescaledbPassword = if ($env:TIMESCALEDB_PASSWORD) { $env:TIMESCALEDB_PASSWORD } else { [System.Web.Security.Membership]::GeneratePassword(32, 8) }
    $InfluxdbToken = if ($env:INFLUXDB_TOKEN) { $env:INFLUXDB_TOKEN } else { [System.Web.Security.Membership]::GeneratePassword(64, 16) }
    $ReplicationPassword = [System.Web.Security.Membership]::GeneratePassword(32, 8)
    $BackupKey = [System.Web.Security.Membership]::GeneratePassword(64, 16)
    $GrafanaPassword = [System.Web.Security.Membership]::GeneratePassword(32, 8)
    
    # Create secrets
    kubectl create secret generic database-secrets -n $Namespace `
        --from-literal=postgres-db="prometheus" `
        --from-literal=postgres-user="prometheus" `
        --from-literal=postgres-password="$PostgresPassword" `
        --from-literal=postgres-replication-password="$ReplicationPassword" `
        --from-literal=postgres-exporter-dsn="postgresql://prometheus:$PostgresPassword@postgres:5432/prometheus?sslmode=disable" `
        --from-literal=redis-password="$RedisPassword" `
        --from-literal=neo4j-password="$Neo4jPassword" `
        --from-literal=neo4j-auth="neo4j/$Neo4jPassword" `
        --from-literal=influxdb-admin-user="admin" `
        --from-literal=influxdb-admin-password="$InfluxdbPassword" `
        --from-literal=influxdb-admin-token="$InfluxdbToken" `
        --from-literal=timescaledb-db="timescale" `
        --from-literal=timescaledb-user="timescale" `
        --from-literal=timescaledb-password="$TimescaledbPassword" `
        --from-literal=backup-encryption-key="$BackupKey" `
        --from-literal=grafana-admin-password="$GrafanaPassword"
    
    Write-Success "Database secrets created"
    
    # Save passwords to file for reference (in development only)
    if ($Environment -eq "development") {
        $passwordFile = @"
POSTGRES_PASSWORD=$PostgresPassword
REDIS_PASSWORD=$RedisPassword
NEO4J_PASSWORD=$Neo4jPassword
INFLUXDB_PASSWORD=$InfluxdbPassword
TIMESCALEDB_PASSWORD=$TimescaledbPassword
INFLUXDB_TOKEN=$InfluxdbToken
GRAFANA_PASSWORD=$GrafanaPassword
"@
        $passwordFile | Out-File -FilePath ".database-passwords" -Encoding UTF8
        Write-Info "Database passwords saved to .database-passwords (development only)"
    }
}

# Deploy database StatefulSets
function Deploy-Databases {
    Write-Info "Deploying database StatefulSets..."
    
    kubectl apply -f kubernetes/database-statefulsets.yaml --timeout=$KubectlTimeout
    kubectl apply -f kubernetes/database-services.yaml --timeout=$KubectlTimeout
    
    Write-Info "Waiting for databases to be ready..."
    
    # Wait for StatefulSets to be ready
    kubectl wait --for=condition=Ready statefulset/postgres -n $Namespace --timeout=$KubectlTimeout
    kubectl wait --for=condition=Ready statefulset/redis -n $Namespace --timeout=$KubectlTimeout
    kubectl wait --for=condition=Ready statefulset/influxdb -n $Namespace --timeout=$KubectlTimeout
    kubectl wait --for=condition=Ready statefulset/neo4j -n $Namespace --timeout=$KubectlTimeout
    kubectl wait --for=condition=Ready statefulset/timescaledb -n $Namespace --timeout=$KubectlTimeout
    
    Write-Success "Database StatefulSets deployed and ready"
}

# Deploy monitoring stack
function Deploy-Monitoring {
    Write-Info "Deploying database monitoring stack..."
    
    kubectl apply -f kubernetes/database-monitoring.yaml --timeout=$KubectlTimeout
    
    # Wait for monitoring deployments to be ready
    kubectl wait --for=condition=Available deployment/prometheus-database-monitoring -n $Namespace --timeout=$KubectlTimeout
    kubectl wait --for=condition=Available deployment/grafana-database-monitoring -n $Namespace --timeout=$KubectlTimeout
    
    Write-Success "Database monitoring deployed"
}

# Deploy backup CronJobs
function Deploy-Backup {
    Write-Info "Deploying database backup automation..."
    
    kubectl apply -f kubernetes/database-backup-cronjobs.yaml --timeout=$KubectlTimeout
    
    Write-Success "Database backup CronJobs deployed"
}

# Deploy service mesh configuration
function Deploy-ServiceMesh {
    Write-Info "Deploying service mesh configuration..."
    
    # Check if Istio is installed
    try {
        kubectl get crd gateways.networking.istio.io | Out-Null
        kubectl apply -f kubernetes/database-service-mesh.yaml --timeout=$KubectlTimeout
        Write-Success "Service mesh configuration deployed"
    }
    catch {
        Write-Warning "Istio not detected, skipping service mesh configuration"
    }
}

# Deploy scaling policies
function Deploy-Scaling {
    Write-Info "Deploying database scaling policies..."
    
    kubectl apply -f kubernetes/database-scaling-policies.yaml --timeout=$KubectlTimeout
    
    Write-Success "Database scaling policies deployed"
}

# Deploy using Helm
function Deploy-WithHelm {
    Write-Info "Deploying database stack using Helm..."
    
    # Add required Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    # Install or upgrade the Helm release
    helm upgrade --install $HelmReleaseName ./helm/database-stack `
        --namespace $Namespace `
        --create-namespace `
        --set environment=$Environment `
        --set global.storageClass="fast-ssd" `
        --timeout=$KubectlTimeout `
        --wait
    
    Write-Success "Database stack deployed using Helm"
}

# Verify deployment
function Test-Deployment {
    Write-Info "Verifying database deployment..."
    
    # Check pod status
    Write-Info "Checking pod status..."
    kubectl get pods -n $Namespace -o wide
    
    # Check service status
    Write-Info "Checking service status..."
    kubectl get services -n $Namespace
    
    # Check persistent volumes
    Write-Info "Checking persistent volumes..."
    kubectl get pvc -n $Namespace
    
    # Test database connectivity
    Write-Info "Testing database connectivity..."
    
    # Test PostgreSQL
    try {
        kubectl exec -n $Namespace statefulset/postgres -- pg_isready -U prometheus -d prometheus | Out-Null
        Write-Success "PostgreSQL is ready"
    }
    catch {
        Write-Error "PostgreSQL is not ready"
    }
    
    # Test Redis
    try {
        $redisResult = kubectl exec -n $Namespace statefulset/redis -- redis-cli ping
        if ($redisResult -match "PONG") {
            Write-Success "Redis is ready"
        }
        else {
            Write-Error "Redis is not ready"
        }
    }
    catch {
        Write-Error "Redis is not ready"
    }
    
    # Test InfluxDB
    try {
        kubectl exec -n $Namespace statefulset/influxdb -- influx ping | Out-Null
        Write-Success "InfluxDB is ready"
    }
    catch {
        Write-Error "InfluxDB is not ready"
    }
    
    Write-Success "Database deployment verification completed"
}

# Display connection information
function Show-ConnectionInfo {
    Write-Info "Database connection information:"
    
    Write-Host ""
    Write-Host "PostgreSQL:"
    Write-Host "  Host: postgres.prometheus-ai.svc.cluster.local"
    Write-Host "  Port: 5432"
    Write-Host "  Database: prometheus"
    Write-Host "  Username: prometheus"
    Write-Host ""
    Write-Host "Redis:"
    Write-Host "  Host: redis.prometheus-ai.svc.cluster.local"
    Write-Host "  Port: 6379"
    Write-Host ""
    Write-Host "InfluxDB:"
    Write-Host "  Host: influxdb.prometheus-ai.svc.cluster.local"
    Write-Host "  Port: 8086"
    Write-Host "  Organization: prometheus"
    Write-Host "  Bucket: geological_data"
    Write-Host ""
    Write-Host "Neo4j:"
    Write-Host "  HTTP: http://neo4j.prometheus-ai.svc.cluster.local:7474"
    Write-Host "  Bolt: bolt://neo4j.prometheus-ai.svc.cluster.local:7687"
    Write-Host "  Username: neo4j"
    Write-Host ""
    Write-Host "TimescaleDB:"
    Write-Host "  Host: timescaledb.prometheus-ai.svc.cluster.local"
    Write-Host "  Port: 5432"
    Write-Host "  Database: timescale"
    Write-Host "  Username: timescale"
    Write-Host ""
    Write-Host "Monitoring:"
    Write-Host "  Prometheus: http://prometheus-database-monitoring.prometheus-ai.svc.cluster.local:9090"
    Write-Host "  Grafana: http://grafana-database-monitoring.prometheus-ai.svc.cluster.local:3000"
    Write-Host ""
    
    if ($Environment -eq "development") {
        Write-Host "For local access, use port-forwarding:"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/postgres 5432:5432"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/redis 6379:6379"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/influxdb 8086:8086"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/neo4j 7474:7474"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/timescaledb 5433:5432"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/prometheus-database-monitoring 9090:9090"
        Write-Host "  kubectl port-forward -n prometheus-ai svc/grafana-database-monitoring 3000:3000"
    }
}

# Cleanup function
function Remove-DatabaseStack {
    Write-Info "Cleaning up database stack..."
    
    # Delete Helm release if it exists
    try {
        $helmList = helm list -n $Namespace
        if ($helmList -match $HelmReleaseName) {
            helm uninstall $HelmReleaseName -n $Namespace
        }
    }
    catch {
        Write-Warning "No Helm release found to cleanup"
    }
    
    # Delete Kubernetes resources
    kubectl delete -f kubernetes/database-scaling-policies.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-service-mesh.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-backup-cronjobs.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-monitoring.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-services.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-statefulsets.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/database-storage.yaml --ignore-not-found=true
    
    # Delete namespace
    kubectl delete namespace $Namespace --ignore-not-found=true
    
    Write-Success "Database stack cleanup completed"
}

# Main execution
try {
    # Load System.Web assembly for password generation
    Add-Type -AssemblyName System.Web
    
    switch ($Action) {
        "deploy" {
            Write-Info "Starting database stack deployment..."
            Test-Prerequisites
            New-DatabaseNamespace
            Deploy-Storage
            Deploy-Secrets
            Deploy-Databases
            Deploy-Monitoring
            Deploy-Backup
            Deploy-ServiceMesh
            Deploy-Scaling
            Test-Deployment
            Show-ConnectionInfo
            Write-Success "Database stack deployment completed successfully!"
        }
        "helm" {
            Write-Info "Starting database stack deployment with Helm..."
            Test-Prerequisites
            New-DatabaseNamespace
            Deploy-Storage
            Deploy-Secrets
            Deploy-WithHelm
            Test-Deployment
            Show-ConnectionInfo
            Write-Success "Database stack deployment with Helm completed successfully!"
        }
        "cleanup" {
            Remove-DatabaseStack
        }
        "verify" {
            Test-Deployment
        }
        "info" {
            Show-ConnectionInfo
        }
        default {
            Write-Host "Usage: .\deploy-database-stack.ps1 [-Action] {deploy|helm|cleanup|verify|info}"
            Write-Host "  deploy  - Deploy database stack using kubectl"
            Write-Host "  helm    - Deploy database stack using Helm"
            Write-Host "  cleanup - Remove database stack"
            Write-Host "  verify  - Verify deployment status"
            Write-Host "  info    - Display connection information"
            exit 1
        }
    }
}
catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    exit 1
}