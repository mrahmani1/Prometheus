#!/bin/bash

# SSH Deployment Script for Prometheus Fixes
echo "========================================"
echo "Deploying Prometheus Fixes to Azure VM"
echo "========================================"
echo ""

# Configuration
REMOTE_HOST="20.80.89.29"
REMOTE_USER="azureuser"
REMOTE_PATH="/home/azureuser/volcano_app/"
LOCAL_PATH="C:/Users/Mario/Desktop/Prometheus"

# Files to deploy
FILES=(
    "critical_ui_fixes.js"
    "ml_dashboard_fix.js"
    "mantle_ui_components.js"
    "prometheus_enhanced_loader.js"
    "tasks.md"
)

echo "Files to deploy:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done
echo ""

# Check SSH key
echo "Checking SSH configuration..."
if [ -f ~/.ssh/id_rsa ]; then
    echo "  ✓ SSH private key found"
else
    echo "  ⚠ No SSH key found at ~/.ssh/id_rsa"
    echo "  You may need to specify your SSH key location"
fi
echo ""

# Deploy files
echo "Starting deployment..."
echo ""

for file in "${FILES[@]}"; do
    echo "Deploying $file..."
    scp -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=10 \
        "${LOCAL_PATH}/${file}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ $file deployed successfully"
    else
        echo "  ✗ Failed to deploy $file"
    fi
    echo ""
done

echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Test the fixes at:"
echo "  http://${REMOTE_HOST}/"
echo ""
echo "Specific pages to test:"
echo "  - /ml_dashboard.html"
echo "  - /mantle_dynamics_simulator.html"
echo "  - /advanced_insar_analysis.html"
echo "  - /volcanic_system_simulator.html"
echo ""