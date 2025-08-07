# Azure VM connection details
$VM_USER = "azureuser"
$VM_IP = "20.185.141.227"
$KEY_PATH = "prometheus-training_key.pem"
$REMOTE_PATH = "/home/azureuser/prometheus/"

# HTML files to deploy
$HTML_FILES = @(
    "advanced_insar_analysis.html",
    "advanced_tectonic_simulation.html",
    "earthquake_stress_visualization.html",
    "volcanic_system_simulator.html",
    "mantle_dynamics_simulator.html",
    "professional_tectonic_simulation.html",
    "scientific_tectonic_simulation.html",
    "volcano_magma_simulation.html",
    "test_integration.html",
    "ml_dashboard.html",
    "debug_dashboard.html",
    "error_viewer.html",
    "gplates_advanced.html",
    "prometheus_navigation.html"
)

# Python scripts to deploy
$PYTHON_FILES = @(
    "check_html_errors.py"
)

Write-Host "Deploying files to Azure VM..." -ForegroundColor Green

# Create remote directory if it doesn't exist
Write-Host "Creating remote directory..."
ssh -i $KEY_PATH -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "mkdir -p $REMOTE_PATH"

# Copy HTML files
foreach ($file in $HTML_FILES) {
    if (Test-Path $file) {
        Write-Host "Copying $file..." -ForegroundColor Yellow
        scp -i $KEY_PATH -o StrictHostKeyChecking=no $file "${VM_USER}@${VM_IP}:${REMOTE_PATH}"
    } else {
        Write-Host "Warning: $file not found" -ForegroundColor Red
    }
}

# Copy Python scripts
foreach ($file in $PYTHON_FILES) {
    if (Test-Path $file) {
        Write-Host "Copying $file..." -ForegroundColor Yellow
        scp -i $KEY_PATH -o StrictHostKeyChecking=no $file "${VM_USER}@${VM_IP}:${REMOTE_PATH}"
    } else {
        Write-Host "Warning: $file not found" -ForegroundColor Red
    }
}

Write-Host "`nRunning HTML validation on Azure VM..." -ForegroundColor Green
ssh -i $KEY_PATH -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "cd $REMOTE_PATH && python3 check_html_errors.py"

Write-Host "`nSetting up HTTP server on Azure VM..." -ForegroundColor Green
ssh -i $KEY_PATH -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "cd $REMOTE_PATH && pkill -f 'python3 -m http.server' 2>/dev/null; nohup python3 -m http.server 8080 > server.log 2>&1 &"

Write-Host "`nDeployment complete!" -ForegroundColor Green
Write-Host "You can access the files at: http://${VM_IP}:8080/" -ForegroundColor Cyan
Write-Host "Available pages:" -ForegroundColor Cyan
foreach ($file in $HTML_FILES) {
    Write-Host "  http://${VM_IP}:8080/$file" -ForegroundColor White
}