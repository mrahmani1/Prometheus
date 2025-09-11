# Prometheus AI Platform Deployment Validation Script (PowerShell)
# This script validates the deployment and performs comprehensive health checks

param(
    [string]$Namespace = "prometheus-ai",
    [int]$TimeoutSeconds = 300,
    [switch]$Detailed = $false
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

function Write-Header {
    param([string]$Message)
    Write-Host "[VALIDATION] $Message" -ForegroundColor Blue
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

Write-Header "Starting Prometheus AI Platform deployment validation..."

# Function to check deployment status
function Test-Deployment {
    param([string]$DeploymentName)
    
    Write-Status "Checking deployment: $DeploymentName"
    
    try {
        $deployment = kubectl get deployment $DeploymentName -n $Namespace -o json | ConvertFrom-Json
        $ready = $deployment.status.readyReplicas
        $desired = $deployment.spec.replicas
        
        if ($ready -eq $desired -and $ready -gt 0) {
            Write-Host "  ✅ $DeploymentName`: $ready/$desired replicas ready" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ❌ $DeploymentName`: $ready/$desired replicas ready" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "  ❌ $DeploymentName`: Not found" -ForegroundColor Red
        return $false
    }
}

# Function to check service status
function Test-Service {
    param([string]$ServiceName)
    
    Write-Status "Checking service: $ServiceName"
    
    try {
        $endpoints = kubectl get endpoints $ServiceName -n $Namespace -o json | ConvertFrom-Json
        $addressCount = 0
        
        if ($endpoints.subsets) {
            foreach ($subset in $endpoints.subsets) {
                if ($subset.addresses) {
                    $addressCount += $subset.addresses.Count
                }
            }
        }
        
        if ($addressCount -gt 0) {
            Write-Host "  ✅ $ServiceName`: $addressCount endpoints available" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ❌ $ServiceName`: No endpoints available" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "  ❌ $ServiceName`: Not found" -ForegroundColor Red
        return $false
    }
}

# Function to check pod health
function Test-PodHealth {
    param([string]$AppLabel)
    
    Write-Status "Checking pod health for app: $AppLabel"
    
    try {
        $pods = kubectl get pods -n $Namespace -l "app=$AppLabel" -o json | ConvertFrom-Json
        
        if (-not $pods.items -or $pods.items.Count -eq 0) {
            Write-Host "  ❌ No pods found for app: $AppLabel" -ForegroundColor Red
            return $false
        }
        
        $healthy = 0
        $total = $pods.items.Count
        
        foreach ($pod in $pods.items) {
            $podName = $pod.metadata.name
            $status = $pod.status.phase
            $ready = $false
            
            if ($pod.status.conditions) {
                foreach ($condition in $pod.status.conditions) {
                    if ($condition.type -eq "Ready" -and $condition.status -eq "True") {
                        $ready = $true
                        break
                    }
                }
            }
            
            if ($status -eq "Running" -and $ready) {
                $healthy++
                Write-Host "  ✅ $podName`: Running and Ready" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $podName`: $status, Ready: $ready" -ForegroundColor Red
            }
        }
        
        if ($healthy -eq $total) {
            Write-Host "  ✅ All $total pods healthy for app: $AppLabel" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ❌ Only $healthy/$total pods healthy for app: $AppLabel" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "  ❌ Error checking pods for app: $AppLabel" -ForegroundColor Red
        return $false
    }
}

# Function to check persistent volumes
function Test-PersistentVolumes {
    Write-Status "Checking persistent volumes"
    
    try {
        $pvcs = kubectl get pvc -n $Namespace -o json | ConvertFrom-Json
        
        if (-not $pvcs.items -or $pvcs.items.Count -eq 0) {
            Write-Host "  ❌ No PVCs found" -ForegroundColor Red
            return $false
        }
        
        $bound = 0
        $total = $pvcs.items.Count
        
        foreach ($pvc in $pvcs.items) {
            $pvcName = $pvc.metadata.name
            $status = $pvc.status.phase
            
            if ($status -eq "Bound") {
                $bound++
                Write-Host "  ✅ $pvcName`: Bound" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $pvcName`: $status" -ForegroundColor Red
            }
        }
        
        if ($bound -eq $total) {
            Write-Host "  ✅ All $total PVCs bound" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ❌ Only $bound/$total PVCs bound" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "  ❌ Error checking PVCs" -ForegroundColor Red
        return $false
    }
}

# Function to check horizontal pod autoscalers
function Test-HPA {
    Write-Status "Checking Horizontal Pod Autoscalers"
    
    try {
        $hpas = kubectl get hpa -n $Namespace -o json | ConvertFrom-Json
        
        if (-not $hpas.items -or $hpas.items.Count -eq 0) {
            Write-Host "  ❌ No HPAs found" -ForegroundColor Red
            return $false
        }
        
        $active = 0
        $total = $hpas.items.Count
        
        foreach ($hpa in $hpas.items) {
            $hpaName = $hpa.metadata.name
            $current = $hpa.status.currentReplicas
            $desired = $hpa.status.desiredReplicas
            
            if ($current -eq $desired -and $current -gt 0) {
                $active++
                Write-Host "  ✅ $hpaName`: $current/$desired replicas" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $hpaName`: $current/$desired replicas" -ForegroundColor Red
            }
        }
        
        if ($active -eq $total) {
            Write-Host "  ✅ All $total HPAs active" -ForegroundColor Green
            return $true
        } else {
            Write-Host "  ❌ Only $active/$total HPAs active" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "  ❌ Error checking HPAs" -ForegroundColor Red
        return $false
    }
}

# Function to check ingress
function Test-Ingress {
    Write-Status "Checking Ingress configuration"
    
    try {
        $ingresses = kubectl get ingress -n $Namespace -o json | ConvertFrom-Json
        
        if (-not $ingresses.items -or $ingresses.items.Count -eq 0) {
            Write-Host "  ❌ No Ingress found" -ForegroundColor Red
            return $false
        }
        
        foreach ($ingress in $ingresses.items) {
            $ingressName = $ingress.metadata.name
            $hosts = @()
            
            if ($ingress.spec.rules) {
                foreach ($rule in $ingress.spec.rules) {
                    if ($rule.host) {
                        $hosts += $rule.host
                    }
                }
            }
            
            $ip = $null
            if ($ingress.status.loadBalancer.ingress) {
                $ip = $ingress.status.loadBalancer.ingress[0].ip
            }
            
            if ($ip) {
                Write-Host "  ✅ $ingressName`: $($hosts -join ', ') -> $ip" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $ingressName`: No external IP assigned" -ForegroundColor Red
            }
        }
        
        return $true
    } catch {
        Write-Host "  ❌ Error checking Ingress" -ForegroundColor Red
        return $false
    }
}

# Function to perform health checks
function Test-HealthChecks {
    Write-Status "Performing application health checks"
    
    # Check API Gateway health
    try {
        $service = kubectl get service api-gateway-service -n $Namespace -o json | ConvertFrom-Json
        $ip = $null
        
        if ($service.status.loadBalancer.ingress) {
            $ip = $service.status.loadBalancer.ingress[0].ip
        }
        
        if ($ip) {
            try {
                $response = Invoke-WebRequest -Uri "http://$ip/health" -TimeoutSec 10 -UseBasicParsing
                if ($response.StatusCode -eq 200) {
                    Write-Host "  ✅ API Gateway health check passed" -ForegroundColor Green
                } else {
                    Write-Host "  ❌ API Gateway health check failed" -ForegroundColor Red
                }
            } catch {
                Write-Host "  ❌ API Gateway health check failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            Write-Host "  ⚠️ API Gateway external IP not available, skipping health check" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ❌ Error checking API Gateway service" -ForegroundColor Red
    }
    
    # Check database connectivity
    try {
        $postgresPod = kubectl get pod -l app=postgres -n $Namespace -o jsonpath='{.items[0].metadata.name}' 2>$null
        if ($postgresPod) {
            $result = kubectl exec $postgresPod -n $Namespace -- pg_isready -U prometheus 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ PostgreSQL connectivity check passed" -ForegroundColor Green
            } else {
                Write-Host "  ❌ PostgreSQL connectivity check failed" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "  ❌ Error checking PostgreSQL connectivity" -ForegroundColor Red
    }
    
    # Check Redis connectivity
    try {
        $redisPod = kubectl get pod -l app=redis -n $Namespace -o jsonpath='{.items[0].metadata.name}' 2>$null
        if ($redisPod) {
            $result = kubectl exec $redisPod -n $Namespace -- redis-cli ping 2>$null
            if ($result -match "PONG") {
                Write-Host "  ✅ Redis connectivity check passed" -ForegroundColor Green
            } else {
                Write-Host "  ❌ Redis connectivity check failed" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "  ❌ Error checking Redis connectivity" -ForegroundColor Red
    }
}

# Function to generate validation report
function Write-ValidationReport {
    Write-Header "Validation Report Summary"
    
    Write-Host ""
    Write-Host "Cluster Information:"
    $clusterInfo = kubectl cluster-info 2>$null | Select-Object -First 1
    Write-Host $clusterInfo
    
    $clientVersion = kubectl version --client --short 2>$null
    Write-Host "Kubernetes Version: $clientVersion"
    Write-Host "Namespace: $Namespace"
    Write-Host "Validation Time: $(Get-Date)"
    
    Write-Host ""
    Write-Host "Resource Summary:"
    
    try {
        $deployments = (kubectl get deployments -n $Namespace --no-headers 2>$null | Measure-Object).Count
        $services = (kubectl get services -n $Namespace --no-headers 2>$null | Measure-Object).Count
        $pods = (kubectl get pods -n $Namespace --no-headers 2>$null | Measure-Object).Count
        $pvcs = (kubectl get pvc -n $Namespace --no-headers 2>$null | Measure-Object).Count
        $hpas = (kubectl get hpa -n $Namespace --no-headers 2>$null | Measure-Object).Count
        
        Write-Host "Deployments: $deployments"
        Write-Host "Services: $services"
        Write-Host "Pods: $pods"
        Write-Host "PVCs: $pvcs"
        Write-Host "HPAs: $hpas"
    } catch {
        Write-Host "Error retrieving resource counts"
    }
    
    Write-Host ""
    Write-Host "Resource Usage:"
    try {
        kubectl top nodes 2>$null
        kubectl top pods -n $Namespace 2>$null
    } catch {
        Write-Host "Metrics server not available"
    }
}

# Main validation function
function Start-Validation {
    $failedChecks = 0
    
    # Check namespace exists
    try {
        kubectl get namespace $Namespace | Out-Null
    } catch {
        Write-Error "Namespace $Namespace does not exist"
        exit 1
    }
    
    Write-Header "Phase 1: Infrastructure Validation"
    
    # Check persistent volumes
    if (-not (Test-PersistentVolumes)) {
        $failedChecks++
    }
    
    Write-Header "Phase 2: Database Validation"
    
    # Check database deployments
    $databases = @("postgres", "influxdb", "neo4j", "redis", "kafka")
    foreach ($db in $databases) {
        if (-not (Test-Deployment "$db-deployment")) {
            $failedChecks++
        }
        if (-not (Test-Service "$db-service")) {
            $failedChecks++
        }
        if (-not (Test-PodHealth $db)) {
            $failedChecks++
        }
    }
    
    Write-Header "Phase 3: Service Mesh Validation"
    
    # Check Consul
    if (-not (Test-Deployment "consul-deployment")) {
        $failedChecks++
    }
    if (-not (Test-Service "consul-service")) {
        $failedChecks++
    }
    
    Write-Header "Phase 4: Application Services Validation"
    
    # Check application services
    $services = @("api-gateway", "helios-core", "data-ingestion", "volcano-watchtower")
    foreach ($service in $services) {
        if (-not (Test-Deployment "$service-deployment")) {
            $failedChecks++
        }
        if (-not (Test-Service "$service-service")) {
            $failedChecks++
        }
        if (-not (Test-PodHealth $service)) {
            $failedChecks++
        }
    }
    
    Write-Header "Phase 5: Monitoring Stack Validation"
    
    # Check monitoring services
    $monitors = @("prometheus", "grafana")
    foreach ($monitor in $monitors) {
        if (-not (Test-Deployment "$monitor-deployment")) {
            $failedChecks++
        }
        if (-not (Test-Service "$monitor-service")) {
            $failedChecks++
        }
    }
    
    Write-Header "Phase 6: Autoscaling Validation"
    
    # Check HPAs
    if (-not (Test-HPA)) {
        $failedChecks++
    }
    
    Write-Header "Phase 7: Ingress Validation"
    
    # Check ingress
    Test-Ingress | Out-Null
    
    Write-Header "Phase 8: Health Checks"
    
    # Perform health checks
    Test-HealthChecks
    
    # Generate report
    Write-ValidationReport
    
    # Final result
    Write-Host ""
    if ($failedChecks -eq 0) {
        Write-Status "🎉 All validation checks passed! Deployment is healthy."
        Write-Host ""
        Write-Status "Access URLs:"
        Write-Host "🌐 API Gateway: kubectl port-forward svc/api-gateway-service 8080:3000 -n $Namespace"
        Write-Host "📊 Prometheus: kubectl port-forward svc/prometheus-service 9090:9090 -n $Namespace"
        Write-Host "📈 Grafana: kubectl port-forward svc/grafana-service 3000:3000 -n $Namespace"
        Write-Host "🔍 Consul UI: kubectl port-forward svc/consul-service 8500:8500 -n $Namespace"
        exit 0
    } else {
        Write-Error "❌ $failedChecks validation checks failed. Please review the output above."
        Write-Host ""
        Write-Warning "Troubleshooting commands:"
        Write-Host "kubectl get all -n $Namespace"
        Write-Host "kubectl describe pods -n $Namespace"
        Write-Host "kubectl logs -l app=api-gateway -n $Namespace"
        exit 1
    }
}

# Run validation
Start-Validation

# Usage examples
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  .\validate.ps1                           # Validate with default settings"
Write-Host "  .\validate.ps1 -Namespace custom-ns      # Validate specific namespace"
Write-Host "  .\validate.ps1 -Detailed                 # Show detailed output"
Write-Host "  .\validate.ps1 -TimeoutSeconds 600       # Custom timeout"