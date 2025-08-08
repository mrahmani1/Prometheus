@echo off
echo ============================================
echo DEPLOYING ALL PROMETHEUS FIXES TO AZURE VM
echo ============================================
echo.

REM Set variables
set REMOTE_HOST=20.80.89.29
set REMOTE_USER=azureuser
set REMOTE_PATH=/home/azureuser/volcano_app/
set LOCAL_PATH=C:\Users\Mario\Desktop\Prometheus

echo Deploying the following files:
echo   - critical_ui_fixes.js
echo   - ml_dashboard_fix.js
echo   - mantle_ui_components.js
echo   - prometheus_enhanced_loader.js
echo   - tasks.md
echo.
echo To: %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%
echo.

REM Deploy all files in one command
echo Executing deployment...
scp "%LOCAL_PATH%\critical_ui_fixes.js" "%LOCAL_PATH%\ml_dashboard_fix.js" "%LOCAL_PATH%\mantle_ui_components.js" "%LOCAL_PATH%\prometheus_enhanced_loader.js" "%LOCAL_PATH%\tasks.md" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%

echo.
echo ============================================
echo DEPLOYMENT ATTEMPT COMPLETE
echo ============================================
echo.
echo If the deployment succeeded, the fixes are now live at:
echo   http://%REMOTE_HOST%/
echo.
echo Test these pages specifically:
echo   - ml_dashboard.html
echo   - mantle_dynamics_simulator.html
echo   - advanced_insar_analysis.html
echo   - volcanic_system_simulator.html
echo.
pause