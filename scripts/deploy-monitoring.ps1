# Multi-Cloud Monitoring Deployment Script for Windows
# Deploys comprehensive monitoring and observability system

param(
    [string]$Namespace = "prometheus-ai",
    [string]$Environment = "production",
    [switch]$SkipValidation,
    [switch]$DryRun
)

# Color functions for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput $Message "Green"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $Message "Yellow"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput $Message "Red"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput $Message "Cyan"
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $requiredCommands = @("kubectl", "helm")
    $missingCommands = @()
    
    foreach ($command in $requiredCommands) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            $missingCommands += $command
        }
    }
    
    if ($missingCommands.Count -gt 0) {
        Write-Error "Missing required commands: $($missingCommands -join ', ')"
        exit 1
    }
    
    # Check kubectl connectivity
    try {
        kubectl cluster-info | Out-Null
        Write-Success "kubectl connectivity verified"
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    Write-Success "All prerequisites met"
}

# Create namespace if it doesn't exist
function New-MonitoringNamespace {
    Write-Info "Creating namespace '$Namespace'..."
    
    $namespaceExists = kubectl get namespace $Namespace 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($DryRun) {
            Write-Info "[DRY RUN] Would create namespace: $Namespace"
        } else {
            kubectl create namespace $Namespace
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Namespace '$Namespace' created"
            } else {
                Write-Error "Failed to create namespace '$Namespace'"
                exit 1
            }
        }
    } else {
        Write-Info "Namespace '$Namespace' already exists"
    }
}

# Deploy monitoring components
function Deploy-MonitoringStack {
    Write-Info "Deploying monitoring stack..."
    
    $manifests = @(
        "kubernetes/advanced-monitoring.yaml",
        "kubernetes/monitoring-configs.yaml",
        "kubernetes/grafana-dashboards.yaml"
    )
    
    foreach ($manifest in $manifests) {
        if (Test-Path $manifest) {
            Write-Info "Applying manifest: $manifest"
            if ($DryRun) {
                Write-Info "[DRY RUN] Would apply: $manifest"
                kubectl apply -f $manifest --dry-run=client --namespace=$Namespace
            } else {
                kubectl apply -f $manifest --namespace=$Namespace
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Applied: $manifest"
                } else {
                    Write-Error "Failed to apply: $manifest"
                    exit 1
                }
            }
        } else {
            Write-Warning "Manifest not found: $manifest"
        }
    }
}

# Wait for deployments to be ready
function Wait-ForDeployments {
    if ($DryRun) {
        Write-Info "[DRY RUN] Would wait for deployments to be ready"
        return
    }
    
    Write-Info "Waiting for deployments to be ready..."
    
    $deployments = @(
        "prometheus-deployment",
        "grafana-deployment",
        "otel-collector",
        "jaeger-all-in-one",
        "alertmanager"
    )
    
    foreach ($deployment in $deployments) {
        Write-Info "Waiting for deployment: $deployment"
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment --namespace=$Namespace
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Deployment ready: $deployment"
        } else {
            Write-Error "Deployment failed or timed out: $deployment"
            exit 1
        }
    }
    
    # Wait for StatefulSets
    Write-Info "Waiting for StatefulSet: elasticsearch"
    kubectl wait --for=condition=ready --timeout=600s statefulset/elasticsearch --namespace=$Namespace
    if ($LASTEXITCODE -eq 0) {
        Write-Success "StatefulSet ready: elasticsearch"
    } else {
        Write-Error "StatefulSet failed or timed out: elasticsearch"
        exit 1
    }
}

# Validate monitoring setup
function Test-MonitoringSetup {
    if ($DryRun) {
        Write-Info "[DRY RUN] Would validate monitoring setup"
        return
    }
    
    Write-Info "Validating monitoring setup..."
    
    $services = @(
        "prometheus-service",
        "grafana-service",
        "otel-collector-service",
        "jaeger-service",
        "elasticsearch-service",
        "alertmanager-service"
    )
    
    foreach ($service in $services) {
        $serviceExists = kubectl get service $service --namespace=$Namespace 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Service running: $service"
        } else {
            Write-Error "Service not found: $service"
            return $false
        }
    }
    
    # Test Prometheus health
    Write-Info "Testing Prometheus health..."
    $prometheusJob = Start-Job -ScriptBlock {
        kubectl port-forward --namespace=$using:Namespace service/prometheus-service 9090:9090
    }
    
    Start-Sleep -Seconds 5
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Prometheus health check passed"
        } else {
            Write-Warning "Prometheus health check returned: $($response.StatusCode)"
        }
    }
    catch {
        Write-Warning "Prometheus health check failed: $($_.Exception.Message)"
    }
    finally {
        Stop-Job $prometheusJob -ErrorAction SilentlyContinue
        Remove-Job $prometheusJob -ErrorAction SilentlyContinue
    }
    
    # Test Grafana health
    Write-Info "Testing Grafana health..."
    $grafanaJob = Start-Job -ScriptBlock {
        kubectl port-forward --namespace=$using:Namespace service/grafana-service 3001:3001
    }
    
    Start-Sleep -Seconds 5
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/health" -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Grafana health check passed"
        } else {
            Write-Warning "Grafana health check returned: $($response.StatusCode)"
        }
    }
    catch {
        Write-Warning "Grafana health check failed: $($_.Exception.Message)"
    }
    finally {
        Stop-Job $grafanaJob -ErrorAction SilentlyContinue
        Remove-Job $grafanaJob -ErrorAction SilentlyContinue
    }
    
    Write-Success "Monitoring validation completed"
    return $true
}

# Setup monitoring secrets
function New-MonitoringSecrets {
    Write-Info "Setting up monitoring secrets..."
    
    # Check for environment variables and create secrets
    $secrets = @{
        "google-cloud-credentials" = @("GOOGLE_APPLICATION_CREDENTIALS")
        "azure-monitoring-credentials" = @("AZURE_WORKSPACE_ID", "AZURE_WORKSPACE_KEY")
        "aws-monitoring-credentials" = @("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        "alerting-credentials" = @("SLACK_WEBHOOK_URL", "PAGERDUTY_ROUTING_KEY", "SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD")
    }
    
    foreach ($secretName in $secrets.Keys) {
        $envVars = $secrets[$secretName]
        $secretData = @{}
        $hasData = $false
        
        foreach ($envVar in $envVars) {
            $value = [Environment]::GetEnvironmentVariable($envVar)
            if ($value) {
                $secretData[$envVar.ToLower().Replace("_", "-")] = $value
                $hasData = $true
            }
        }
        
        if ($hasData) {
            Write-Info "Creating secret: $secretName"
            if ($DryRun) {
                Write-Info "[DRY RUN] Would create secret: $secretName"
            } else {
                $secretArgs = @("create", "secret", "generic", $secretName, "--namespace=$Namespace")
                foreach ($key in $secretData.Keys) {
                    $secretArgs += "--from-literal=$key=$($secretData[$key])"
                }
                $secretArgs += "--dry-run=client", "-o", "yaml"
                
                $secretYaml = & kubectl $secretArgs
                $secretYaml | kubectl apply -f -
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Secret created: $secretName"
                } else {
                    Write-Warning "Failed to create secret: $secretName"
                }
            }
        } else {
            Write-Warning "No environment variables found for secret: $secretName"
        }
    }
}

# Print access information
function Show-AccessInformation {
    Write-Success "Monitoring deployment completed successfully!"
    Write-Host ""
    Write-Host "Access Information:" -ForegroundColor Yellow
    Write-Host "==================" -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "Prometheus:" -ForegroundColor Cyan
    Write-Host "  kubectl port-forward -n $Namespace service/prometheus-service 9090:9090" -ForegroundColor White
    Write-Host "  Then access: http://localhost:9090" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Grafana:" -ForegroundColor Cyan
    Write-Host "  kubectl port-forward -n $Namespace service/grafana-service 3001:3001" -ForegroundColor White
    Write-Host "  Then access: http://localhost:3001" -ForegroundColor White
    Write-Host "  Default credentials: admin/prometheus123" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Jaeger:" -ForegroundColor Cyan
    Write-Host "  kubectl port-forward -n $Namespace service/jaeger-service 16686:16686" -ForegroundColor White
    Write-Host "  Then access: http://localhost:16686" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Kibana:" -ForegroundColor Cyan
    Write-Host "  kubectl port-forward -n $Namespace service/kibana-service 5601:5601" -ForegroundColor White
    Write-Host "  Then access: http://localhost:5601/kibana" -ForegroundColor White
    Write-Host ""
    
    Write-Host "AlertManager:" -ForegroundColor Cyan
    Write-Host "  kubectl port-forward -n $Namespace service/alertmanager-service 9093:9093" -ForegroundColor White
    Write-Host "  Then access: http://localhost:9093" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Useful Commands:" -ForegroundColor Yellow
    Write-Host "================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Check pod status:" -ForegroundColor Cyan
    Write-Host "  kubectl get pods -n $Namespace" -ForegroundColor White
    Write-Host ""
    Write-Host "View logs:" -ForegroundColor Cyan
    Write-Host "  kubectl logs -f deployment/prometheus-deployment -n $Namespace" -ForegroundColor White
    Write-Host "  kubectl logs -f deployment/grafana-deployment -n $Namespace" -ForegroundColor White
    Write-Host ""
    Write-Host "Scale deployments:" -ForegroundColor Cyan
    Write-Host "  kubectl scale deployment prometheus-deployment --replicas=2 -n $Namespace" -ForegroundColor White
    Write-Host ""
}

# Main execution
function Main {
    Write-Info "Starting monitoring deployment..."
    Write-Info "Namespace: $Namespace"
    Write-Info "Environment: $Environment"
    if ($DryRun) {
        Write-Warning "DRY RUN MODE - No changes will be applied"
    }
    
    Test-Prerequisites
    New-MonitoringNamespace
    New-MonitoringSecrets
    Deploy-MonitoringStack
    
    if (-not $SkipValidation) {
        Wait-ForDeployments
        $validationResult = Test-MonitoringSetup
        
        if ($validationResult) {
            Show-AccessInformation
        } else {
            Write-Error "Monitoring validation failed"
            exit 1
        }
    } else {
        Write-Warning "Skipping validation as requested"
        Show-AccessInformation
    }
    
    Write-Success "Monitoring deployment completed successfully!"
}

# Execute main function
try {
    Main
}
catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    exit 1
}