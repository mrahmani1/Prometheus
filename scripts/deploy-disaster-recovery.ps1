# Multi-Cloud Disaster Recovery Deployment Script (PowerShell)
# Deploys the disaster recovery service to Kubernetes

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "cleanup", "status", "logs", "help")]
    [string]$Command = "deploy",
    
    [string]$Namespace = "prometheus-ai",
    [string]$ServiceName = "disaster-recovery",
    [string]$ImageTag = "latest",
    [string]$Environment = "production",
    [switch]$BuildImage,
    [switch]$ShowLogs,
    [switch]$Cleanup
)

# Error handling
$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check kubectl
    try {
        $null = Get-Command kubectl -ErrorAction Stop
    }
    catch {
        Write-Error "kubectl is not installed or not in PATH"
        exit 1
    }
    
    # Check cluster connectivity
    try {
        $null = kubectl cluster-info 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "kubectl cluster-info failed"
        }
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    # Check Docker (if building image)
    if ($BuildImage) {
        try {
            $null = Get-Command docker -ErrorAction Stop
        }
        catch {
            Write-Error "Docker is not installed or not in PATH"
            exit 1
        }
    }
    
    Write-Success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
function New-Namespace {
    Write-Info "Creating namespace $Namespace..."
    
    $existingNamespace = kubectl get namespace $Namespace 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Warning "Namespace $Namespace already exists"
    }
    else {
        kubectl create namespace $Namespace
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Namespace $Namespace created"
        }
        else {
            Write-Error "Failed to create namespace $Namespace"
            exit 1
        }
    }
}

# Build Docker image
function Build-Image {
    if ($BuildImage) {
        Write-Info "Building Docker image..."
        
        Push-Location "services\disaster-recovery"
        try {
            docker build -t "prometheus-ai/$ServiceName`:$ImageTag" .
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Docker image built successfully"
            }
            else {
                Write-Error "Failed to build Docker image"
                exit 1
            }
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Info "Skipping image build (BuildImage not specified)"
    }
}

# Validate configuration
function Test-Configuration {
    Write-Info "Validating configuration..."
    
    # Check if secrets file exists
    if (-not (Test-Path "kubernetes\disaster-recovery-secrets.yaml")) {
        Write-Error "Secrets file not found: kubernetes\disaster-recovery-secrets.yaml"
        exit 1
    }
    
    # Validate required environment variables in secrets
    $requiredVars = @(
        "gcp-project-id",
        "azure-subscription-id",
        "aws-account-id",
        "database-url",
        "jwt-secret-key"
    )
    
    $secretsContent = Get-Content "kubernetes\disaster-recovery-secrets.yaml" -Raw
    
    foreach ($var in $requiredVars) {
        if ($secretsContent -notmatch "$var`:") {
            Write-Warning "Required variable $var not found in secrets file"
        }
    }
    
    Write-Success "Configuration validation completed"
}

# Deploy secrets
function Deploy-Secrets {
    Write-Info "Deploying secrets..."
    
    kubectl apply -f kubernetes\disaster-recovery-secrets.yaml
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Secrets deployed successfully"
    }
    else {
        Write-Error "Failed to deploy secrets"
        exit 1
    }
}

# Deploy service
function Deploy-Service {
    Write-Info "Deploying $ServiceName service..."
    
    # Create a temporary deployment file with updated image tag
    $deploymentContent = Get-Content "kubernetes\disaster-recovery-deployment.yaml" -Raw
    
    if ($ImageTag -ne "latest") {
        $deploymentContent = $deploymentContent -replace "image: prometheus-ai/$ServiceName`:latest", "image: prometheus-ai/$ServiceName`:$ImageTag"
    }
    
    # Apply deployment
    $deploymentContent | kubectl apply -f -
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Service deployment applied"
    }
    else {
        Write-Error "Failed to deploy service"
        exit 1
    }
}

# Deploy monitoring
function Deploy-Monitoring {
    Write-Info "Deploying monitoring configuration..."
    
    # Check if Prometheus operator is installed
    $crd = kubectl get crd servicemonitors.monitoring.coreos.com 2>$null
    if ($LASTEXITCODE -eq 0) {
        kubectl apply -f kubernetes\disaster-recovery-monitoring.yaml
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Monitoring configuration deployed"
        }
        else {
            Write-Warning "Failed to deploy monitoring configuration"
        }
    }
    else {
        Write-Warning "Prometheus operator not found, skipping monitoring deployment"
    }
}

# Wait for deployment
function Wait-ForDeployment {
    Write-Info "Waiting for deployment to be ready..."
    
    # Wait for deployment to be available
    kubectl wait --for=condition=available --timeout=300s deployment/$ServiceName -n $Namespace
    if ($LASTEXITCODE -eq 0) {
        # Check pod status
        kubectl get pods -n $Namespace -l app=$ServiceName
        Write-Success "Deployment is ready"
    }
    else {
        Write-Error "Deployment failed to become ready within timeout"
        exit 1
    }
}

# Verify deployment
function Test-Deployment {
    Write-Info "Verifying deployment..."
    
    # Check service endpoints
    $serviceIp = kubectl get svc $ServiceName -n $Namespace -o jsonpath='{.spec.clusterIP}' 2>$null
    
    if ($serviceIp) {
        Write-Success "Service is accessible at $serviceIp`:8000"
    }
    else {
        Write-Error "Service IP not found"
        return $false
    }
    
    # Test health endpoint
    Write-Info "Testing health endpoint..."
    $healthTest = kubectl exec -n $Namespace deployment/$ServiceName -- curl -f http://localhost:8000/health 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Health endpoint is responding"
    }
    else {
        Write-Error "Health endpoint is not responding"
        return $false
    }
    
    # Test readiness endpoint
    Write-Info "Testing readiness endpoint..."
    $readinessTest = kubectl exec -n $Namespace deployment/$ServiceName -- curl -f http://localhost:8000/ready 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Readiness endpoint is responding"
    }
    else {
        Write-Warning "Readiness endpoint is not responding (may be initializing)"
    }
    
    Write-Success "Deployment verification completed"
    return $true
}

# Show deployment status
function Show-Status {
    Write-Info "Deployment Status:"
    Write-Host ""
    
    # Show pods
    Write-Host "Pods:" -ForegroundColor $Colors.White
    kubectl get pods -n $Namespace -l app=$ServiceName
    Write-Host ""
    
    # Show services
    Write-Host "Services:" -ForegroundColor $Colors.White
    kubectl get svc -n $Namespace -l app=$ServiceName
    Write-Host ""
    
    # Show ingress (if exists)
    $ingress = kubectl get ingress -n $Namespace -l app=$ServiceName 2>$null
    if ($LASTEXITCODE -eq 0 -and $ingress) {
        Write-Host "Ingress:" -ForegroundColor $Colors.White
        kubectl get ingress -n $Namespace -l app=$ServiceName
        Write-Host ""
    }
    
    # Show recent events
    Write-Host "Recent Events:" -ForegroundColor $Colors.White
    kubectl get events -n $Namespace --sort-by='.lastTimestamp' | Select-Object -Last 10
}

# Show logs
function Show-Logs {
    if ($ShowLogs) {
        Write-Info "Showing recent logs..."
        kubectl logs -n $Namespace deployment/$ServiceName --tail=50
    }
}

# Cleanup function
function Remove-Deployment {
    Write-Info "Cleaning up deployment..."
    
    kubectl delete -f kubernetes\disaster-recovery-deployment.yaml --ignore-not-found=true
    kubectl delete -f kubernetes\disaster-recovery-monitoring.yaml --ignore-not-found=true
    kubectl delete -f kubernetes\disaster-recovery-secrets.yaml --ignore-not-found=true
    
    Write-Success "Cleanup completed"
}

# Main deployment function
function Start-Deployment {
    Write-Info "Starting Multi-Cloud Disaster Recovery deployment..."
    Write-Info "Environment: $Environment"
    Write-Info "Namespace: $Namespace"
    Write-Info "Image Tag: $ImageTag"
    Write-Host ""
    
    # Run deployment steps
    Test-Prerequisites
    New-Namespace
    Build-Image
    Test-Configuration
    Deploy-Secrets
    Deploy-Service
    Deploy-Monitoring
    Wait-ForDeployment
    $verificationResult = Test-Deployment
    Show-Status
    Show-Logs
    
    Write-Host ""
    if ($verificationResult) {
        Write-Success "Multi-Cloud Disaster Recovery deployment completed successfully!"
    }
    else {
        Write-Warning "Deployment completed with some issues. Check the logs for details."
    }
    
    Write-Host ""
    Write-Info "Next steps:"
    Write-Host "  1. Configure cloud provider credentials in secrets"
    Write-Host "  2. Test the service endpoints"
    Write-Host "  3. Set up monitoring dashboards"
    Write-Host "  4. Configure alerting rules"
    Write-Host ""
    Write-Info "Useful commands:"
    Write-Host "  kubectl get pods -n $Namespace -l app=$ServiceName"
    Write-Host "  kubectl logs -f deployment/$ServiceName -n $Namespace"
    Write-Host "  kubectl port-forward svc/$ServiceName 8000:8000 -n $Namespace"
}

# Handle script commands
switch ($Command) {
    "deploy" {
        Start-Deployment
    }
    "cleanup" {
        Remove-Deployment
    }
    "status" {
        Show-Status
    }
    "logs" {
        kubectl logs -f deployment/$ServiceName -n $Namespace
    }
    "help" {
        Write-Host "Usage: .\deploy-disaster-recovery.ps1 [deploy|cleanup|status|logs|help]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  deploy   - Deploy the disaster recovery service (default)"
        Write-Host "  cleanup  - Remove the deployment"
        Write-Host "  status   - Show deployment status"
        Write-Host "  logs     - Show service logs"
        Write-Host "  help     - Show this help message"
        Write-Host ""
        Write-Host "Parameters:"
        Write-Host "  -Environment    - Deployment environment (default: production)"
        Write-Host "  -Namespace      - Kubernetes namespace (default: prometheus-ai)"
        Write-Host "  -ImageTag       - Docker image tag (default: latest)"
        Write-Host "  -BuildImage     - Build Docker image"
        Write-Host "  -ShowLogs       - Show logs after deployment"
        Write-Host "  -Cleanup        - Cleanup deployment"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\deploy-disaster-recovery.ps1 deploy -BuildImage -ShowLogs"
        Write-Host "  .\deploy-disaster-recovery.ps1 cleanup"
        Write-Host "  .\deploy-disaster-recovery.ps1 status"
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host "Use '.\deploy-disaster-recovery.ps1 help' for usage information"
        exit 1
    }
}