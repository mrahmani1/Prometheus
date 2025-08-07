#!/bin/bash

# Azure VM connection details
VM_USER="azureuser"
VM_IP="20.185.141.227"
KEY_PATH="prometheus-training_key.pem"
REMOTE_PATH="/home/azureuser/prometheus/"

# HTML files to deploy
HTML_FILES=(
    "advanced_insar_analysis.html"
    "advanced_tectonic_simulation.html"
    "earthquake_stress_visualization.html"
    "volcanic_system_simulator.html"
    "mantle_dynamics_simulator.html"
    "professional_tectonic_simulation.html"
    "scientific_tectonic_simulation.html"
    "volcano_magma_simulation.html"
    "test_integration.html"
    "ml_dashboard.html"
    "debug_dashboard.html"
    "error_viewer.html"
    "gplates_advanced.html"
    "prometheus_navigation.html"
)

# Python scripts to deploy
PYTHON_FILES=(
    "check_html_errors.py"
)

echo "Deploying files to Azure VM..."

# Create remote directory if it doesn't exist
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "mkdir -p $REMOTE_PATH"

# Copy HTML files
for file in "${HTML_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Copying $file..."
        scp -i "$KEY_PATH" -o StrictHostKeyChecking=no "$file" "$VM_USER@$VM_IP:$REMOTE_PATH"
    else
        echo "Warning: $file not found"
    fi
done

# Copy Python scripts
for file in "${PYTHON_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Copying $file..."
        scp -i "$KEY_PATH" -o StrictHostKeyChecking=no "$file" "$VM_USER@$VM_IP:$REMOTE_PATH"
    else
        echo "Warning: $file not found"
    fi
done

echo "Running HTML validation on Azure VM..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "cd $REMOTE_PATH && python3 check_html_errors.py"

echo "Setting up simple HTTP server on Azure VM..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "cd $REMOTE_PATH && nohup python3 -m http.server 8080 > server.log 2>&1 &"

echo "Deployment complete!"
echo "You can access the files at: http://$VM_IP:8080/"
echo "Available pages:"
for file in "${HTML_FILES[@]}"; do
    echo "  http://$VM_IP:8080/$file"
done