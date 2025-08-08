@echo off
echo ============================================
echo DEPLOYING ALL FIXES WITH SSH KEY
echo ============================================
echo.

REM Configuration
set KEY_FILE=C:\Users\Mario\Desktop\Prometheus\prometheus-training_key.pem
set REMOTE_HOST=20.80.89.29
set REMOTE_USER=azureuser
set REMOTE_PATH=/home/azureuser/volcano_app/

echo Using SSH Key: %KEY_FILE%
echo Target: %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%
echo.

echo Deploying critical_ui_fixes.js...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no "C:\Users\Mario\Desktop\Prometheus\critical_ui_fixes.js" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%

echo Deploying ml_dashboard_fix.js...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no "C:\Users\Mario\Desktop\Prometheus\ml_dashboard_fix.js" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%

echo Deploying mantle_ui_components.js...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no "C:\Users\Mario\Desktop\Prometheus\mantle_ui_components.js" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%

echo Deploying prometheus_enhanced_loader.js...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no "C:\Users\Mario\Desktop\Prometheus\prometheus_enhanced_loader.js" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%

echo Deploying tasks.md...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no "C:\Users\Mario\Desktop\Prometheus\tasks.md" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%

echo.
echo ============================================
echo DEPLOYMENT COMPLETE!
echo ============================================
echo.
echo All fixes have been deployed to the Azure VM.
echo.
echo Test the fixes at:
echo   http://%REMOTE_HOST%/
echo.
echo Specific pages to verify:
echo   - http://%REMOTE_HOST%/ml_dashboard.html
echo   - http://%REMOTE_HOST%/mantle_dynamics_simulator.html
echo   - http://%REMOTE_HOST%/advanced_insar_analysis.html
echo   - http://%REMOTE_HOST%/volcanic_system_simulator.html
echo.
pause