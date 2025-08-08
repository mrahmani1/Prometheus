@echo off
echo ============================================
echo Mantle Dynamics UI Fix Deployment Script
echo ============================================
echo.
echo This script will deploy the Mantle Dynamics UI fixes to Azure VM
echo.
echo Files to deploy:
echo - mantle_ui_components.js
echo - prometheus_enhanced_loader.js (updated)
echo.
echo Target: azureuser@20.80.89.29:/home/azureuser/volcano_app/
echo.
echo Please ensure you have:
echo 1. SSH key configured for the Azure VM
echo 2. Network connectivity to 20.80.89.29
echo.
pause

echo.
echo Deploying mantle_ui_components.js...
scp "C:\Users\Mario\Desktop\Prometheus\mantle_ui_components.js" azureuser@20.80.89.29:/home/azureuser/volcano_app/

echo.
echo Deploying updated prometheus_enhanced_loader.js...
scp "C:\Users\Mario\Desktop\Prometheus\prometheus_enhanced_loader.js" azureuser@20.80.89.29:/home/azureuser/volcano_app/

echo.
echo ============================================
echo Deployment Complete!
echo ============================================
echo.
echo The Mantle Dynamics UI fixes have been deployed.
echo.
echo To verify the fixes:
echo 1. Navigate to http://20.80.89.29/mantle_dynamics_simulator.html
echo 2. Check that sliders are responsive with touch support
echo 3. Verify control panels are properly aligned
echo 4. Test data layer checkboxes functionality
echo 5. Confirm visualization mode selector works
echo 6. Test draggable panels
echo 7. Verify mobile responsiveness
echo.
echo Key improvements:
echo - React-like component architecture
echo - Touch gesture support for mobile devices
echo - Proper event delegation for all controls
echo - Modern slider components with visual feedback
echo - Draggable and collapsible panels
echo - Responsive design for all screen sizes
echo - Data layer toggle system
echo.
pause