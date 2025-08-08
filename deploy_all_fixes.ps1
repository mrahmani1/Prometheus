# PowerShell Deployment Script for Prometheus Fixes
# Run this script in PowerShell as Administrator if needed

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Prometheus Platform - Deploy All Recent Fixes" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$sourceDir = "C:\Users\Mario\Desktop\Prometheus"
$remoteHost = "20.80.89.29"
$remoteUser = "azureuser"
$remotePath = "/home/azureuser/volcano_app/"

# Files to deploy
$files = @(
    "critical_ui_fixes.js",
    "ml_dashboard_fix.js", 
    "mantle_ui_components.js",
    "prometheus_enhanced_loader.js",
    "tasks.md"
)

Write-Host "Files to deploy:" -ForegroundColor Yellow
foreach ($file in $files) {
    Write-Host "  - $file" -ForegroundColor White
}
Write-Host ""
Write-Host "Target: ${remoteUser}@${remoteHost}:${remotePath}" -ForegroundColor Yellow
Write-Host ""

# Check if files exist
Write-Host "Checking files..." -ForegroundColor Green
$missingFiles = @()
foreach ($file in $files) {
    $filePath = Join-Path $sourceDir $file
    if (Test-Path $filePath) {
        Write-Host "  ✓ $file exists" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file missing" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Some files are missing. Please check the source directory." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting deployment..." -ForegroundColor Cyan
Write-Host ""

# Deploy each file
foreach ($file in $files) {
    $localPath = Join-Path $sourceDir $file
    Write-Host "Deploying $file..." -ForegroundColor Yellow
    
    # Using SCP command
    $scpCommand = "scp `"$localPath`" ${remoteUser}@${remoteHost}:${remotePath}"
    
    try {
        Invoke-Expression $scpCommand
        Write-Host "  ✓ $file deployed successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to deploy $file" -ForegroundColor Red
        Write-Host "    Error: $_" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Visit http://${remoteHost}/ to test the fixes" -ForegroundColor White
Write-Host "2. Check the following pages:" -ForegroundColor White
Write-Host "   - /ml_dashboard.html (ML Dashboard fixes)" -ForegroundColor White
Write-Host "   - /mantle_dynamics_simulator.html (Mantle UI fixes)" -ForegroundColor White
Write-Host "   - /advanced_insar_analysis.html (InSAR fixes)" -ForegroundColor White
Write-Host "   - /volcanic_system_simulator.html (Volcanic system fixes)" -ForegroundColor White
Write-Host ""
Write-Host "Key improvements deployed:" -ForegroundColor Yellow
Write-Host "  • Fixed panel overlaps and z-index issues" -ForegroundColor White
Write-Host "  • Chart.js v4.4.1 with error boundaries" -ForegroundColor White
Write-Host "  • React-like component system for UI" -ForegroundColor White
Write-Host "  • Touch support for mobile devices" -ForegroundColor White
Write-Host "  • Data layer toggle functionality" -ForegroundColor White
Write-Host "  • Responsive grid layouts" -ForegroundColor White
Write-Host "  • Draggable and collapsible panels" -ForegroundColor White
Write-Host ""

# Alternative: Create a combined deployment command
Write-Host "Alternative: Manual SCP command (copy and paste):" -ForegroundColor Yellow
$allFiles = ($files | ForEach-Object { "`"$sourceDir\$_`"" }) -join " "
Write-Host "scp $allFiles ${remoteUser}@${remoteHost}:${remotePath}" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"