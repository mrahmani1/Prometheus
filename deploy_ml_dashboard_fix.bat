@echo off
echo ============================================
echo ML Dashboard Fix Deployment Script
echo ============================================
echo.
echo This script will deploy the ML Dashboard fixes to Azure VM
echo.
echo Files to deploy:
echo - ml_dashboard_fix.js
echo - prometheus_enhanced_loader.js
echo.
echo Target: azureuser@20.80.89.29:/home/azureuser/volcano_app/
echo.
echo Please ensure you have:
echo 1. SSH key configured for the Azure VM
echo 2. Network connectivity to 20.80.89.29
echo.
pause

echo.
echo Deploying ml_dashboard_fix.js...
scp "C:\Users\Mario\Desktop\Prometheus\ml_dashboard_fix.js" azureuser@20.80.89.29:/home/azureuser/volcano_app/

echo.
echo Deploying prometheus_enhanced_loader.js...
scp "C:\Users\Mario\Desktop\Prometheus\prometheus_enhanced_loader.js" azureuser@20.80.89.29:/home/azureuser/volcano_app/

echo.
echo ============================================
echo Deployment Complete!
echo ============================================
echo.
echo The ML Dashboard fixes have been deployed.
echo.
echo To verify:
echo 1. Navigate to http://20.80.89.29/ml_dashboard.html
echo 2. Check that charts are rendering properly
echo 3. Verify panel layout is using flexbox/grid
echo 4. Test data layer toggles
echo 5. Confirm error boundaries are working
echo.
pause