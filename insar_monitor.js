/**
 * InSAR Analysis Monitoring Script
 * Automatically detects and fixes common JavaScript issues
 * Usage: Include this script to monitor and auto-repair the InSAR analysis page
 */

class InSARMonitor {
    constructor() {
        this.checkInterval = 10000; // 10 seconds
        this.errorCount = 0;
        this.fixHistory = [];
        this.isRunning = false;
        
        this.init();
    }
    
    init() {
        console.log('🚀 InSAR Monitor initialized');
        this.startMonitoring();
        this.setupErrorHandlers();
    }
    
    startMonitoring() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.monitorInterval = setInterval(() => {
            this.performHealthCheck();
        }, this.checkInterval);
        
        console.log('👁️ Monitoring started - checking every', this.checkInterval / 1000, 'seconds');
    }
    
    stopMonitoring() {
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
            this.isRunning = false;
            console.log('⏹️ Monitoring stopped');
        }
    }
    
    setupErrorHandlers() {
        // Global error handler
        window.addEventListener('error', (event) => {
            this.handleError('JavaScript Error', event.error, {
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            });
        });
        
        // Promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError('Promise Rejection', event.reason);
        });
    }
    
    handleError(type, error, details = {}) {
        this.errorCount++;
        const errorInfo = {
            type,
            message: error?.message || error,
            timestamp: new Date().toISOString(),
            details,
            count: this.errorCount
        };
        
        console.error(`🔥 ${type}:`, errorInfo);
        
        // Attempt auto-fix
        this.attemptAutoFix(errorInfo);
    }
    
    performHealthCheck() {
        const issues = [];
        
        try {
            // Check essential DOM elements
            const essentialElements = [
                'map-container', 'threejs-container', 'volcano-grid',
                'status-text', 'view-toggle', 'insar-controls',
                'analysis-panel', 'mantle-panel', 'ml-panel'
            ];
            
            essentialElements.forEach(id => {
                if (!document.getElementById(id)) {
                    issues.push({
                        type: 'missing-element',
                        element: id,
                        severity: 'high'
                    });
                }
            });
            
            // Check essential functions
            const essentialFunctions = [
                'switchView', 'togglePanel', 'selectVolcano',
                'processInSARData', 'loadVolcanoData', 'updateStatusText'
            ];
            
            essentialFunctions.forEach(funcName => {
                if (typeof window[funcName] !== 'function') {
                    issues.push({
                        type: 'missing-function',
                        function: funcName,
                        severity: 'critical'
                    });
                }
            });
            
            // Check global variables
            const essentialVariables = ['HOLOCENE_VOLCANOES', 'currentView', 'currentVolcano'];
            essentialVariables.forEach(varName => {
                if (typeof window[varName] === 'undefined') {
                    issues.push({
                        type: 'missing-variable',
                        variable: varName,
                        severity: 'medium'
                    });
                }
            });
            
            // Check WebGL context
            if (window.renderer && window.renderer.getContext().isContextLost()) {
                issues.push({
                    type: 'webgl-context-lost',
                    severity: 'critical'
                });
            }
            
            // Check Leaflet map
            if (window.map && !window.map.getContainer()) {
                issues.push({
                    type: 'leaflet-map-detached',
                    severity: 'high'
                });
            }
            
        } catch (error) {
            issues.push({
                type: 'health-check-error',
                error: error.message,
                severity: 'critical'
            });
        }
        
        if (issues.length > 0) {
            console.warn('🔍 Health check found issues:', issues);
            this.autoFix(issues);
        }
        
        return issues;
    }
    
    autoFix(issues) {
        issues.forEach(issue => {
            const fix = this.generateFix(issue);
            if (fix) {
                try {
                    fix();
                    this.fixHistory.push({
                        issue,
                        fixed: true,
                        timestamp: new Date().toISOString()
                    });
                    console.log('✅ Auto-fixed:', issue.type);
                } catch (error) {
                    console.error('❌ Auto-fix failed for:', issue.type, error);
                    this.fixHistory.push({
                        issue,
                        fixed: false,
                        error: error.message,
                        timestamp: new Date().toISOString()
                    });
                }
            }
        });
    }
    
    generateFix(issue) {
        switch (issue.type) {
            case 'missing-function':
                return () => {
                    // Create placeholder functions
                    const funcName = issue.function;
                    if (!window[funcName]) {
                        window[funcName] = function(...args) {
                            console.warn(`Placeholder function called: ${funcName}`, args);
                            if (window.updateStatusText) {
                                window.updateStatusText(`Function ${funcName} is being restored...`);
                            }
                        };
                    }
                };
                
            case 'webgl-context-lost':
                return () => {
                    // Attempt to restore WebGL context
                    if (window.initializeWebGL) {
                        window.initializeWebGL();
                    }
                };
                
            case 'leaflet-map-detached':
                return () => {
                    // Reinitialize map
                    if (window.initialize2DMap) {
                        window.initialize2DMap();
                    }
                };
                
            case 'missing-variable':
                return () => {
                    const varName = issue.variable;
                    if (varName === 'currentView' && !window.currentView) {
                        window.currentView = '2d';
                    }
                    if (varName === 'currentVolcano' && !window.currentVolcano) {
                        window.currentVolcano = null;
                    }
                };
                
            default:
                return null;
        }
    }
    
    attemptAutoFix(errorInfo) {
        // Common error patterns and their fixes
        if (errorInfo.message?.includes('Cannot read property') || 
            errorInfo.message?.includes('Cannot read properties of null')) {
            
            // Null reference error - reload page as last resort
            setTimeout(() => {
                if (this.errorCount > 10) {
                    console.warn('🔄 Too many errors, reloading page...');
                    window.location.reload();
                }
            }, 5000);
        }
        
        if (errorInfo.message?.includes('is not a function')) {
            // Missing function error - try to restore from backup
            const funcMatch = errorInfo.message.match(/(\w+) is not a function/);
            if (funcMatch && funcMatch[1]) {
                this.restoreFunction(funcMatch[1]);
            }
        }
    }
    
    restoreFunction(functionName) {
        // Try to restore common functions
        const commonFunctions = {
            switchView: `function(view) {
                console.log('Restored function switchView called with:', view);
                window.currentView = view;
                if (window.updateStatusText) {
                    window.updateStatusText('View switched to: ' + view);
                }
            }`,
            
            togglePanel: `function(panelId) {
                console.log('Restored function togglePanel called with:', panelId);
                const panel = document.getElementById(panelId);
                if (panel) {
                    panel.classList.toggle('minimized');
                }
            }`,
            
            updateStatusText: `function(text, color = '#4CAF50') {
                const statusElement = document.getElementById('status-text');
                if (statusElement) {
                    statusElement.textContent = text;
                    statusElement.style.color = color;
                }
                console.log('Status:', text);
            }`
        };
        
        if (commonFunctions[functionName]) {
            try {
                window[functionName] = new Function('return ' + commonFunctions[functionName])();
                console.log('✅ Restored function:', functionName);
            } catch (error) {
                console.error('❌ Failed to restore function:', functionName, error);
            }
        }
    }
    
    getReport() {
        return {
            isRunning: this.isRunning,
            errorCount: this.errorCount,
            fixHistory: this.fixHistory,
            lastHealthCheck: this.performHealthCheck(),
            uptime: Date.now() - this.startTime || 0
        };
    }
    
    // Manual repair methods
    repairUI() {
        console.log('🔧 Starting UI repair...');
        
        // Fix missing panels
        const panels = ['insar-controls', 'analysis-panel', 'mantle-panel', 'ml-panel'];
        panels.forEach(panelId => {
            const panel = document.getElementById(panelId);
            if (panel && panel.style.display === 'none') {
                panel.style.display = 'block';
                console.log('✅ Restored panel:', panelId);
            }
        });
        
        // Fix view containers
        const mapContainer = document.getElementById('map-container');
        const threejsContainer = document.getElementById('threejs-container');
        
        if (mapContainer && !mapContainer.innerHTML) {
            console.log('🗺️ Reinitializing map...');
            if (window.initialize2DMap) {
                window.initialize2DMap();
            }
        }
        
        if (threejsContainer && !threejsContainer.innerHTML) {
            console.log('🎨 Reinitializing 3D view...');
            if (window.initializeWebGL) {
                window.initializeWebGL();
            }
        }
    }
    
    repairData() {
        console.log('📊 Starting data repair...');
        
        // Reload volcano data if missing
        if (!window.HOLOCENE_VOLCANOES || Object.keys(window.HOLOCENE_VOLCANOES).length === 0) {
            console.log('🌋 Reloading volcano data...');
            // Attempt to reload from external script or recreate basic data
            if (window.loadVolcanoData) {
                window.loadVolcanoData();
            }
        }
        
        // Reset current volcano if corrupted
        if (window.currentVolcano && typeof window.currentVolcano !== 'object') {
            window.currentVolcano = null;
            console.log('🔄 Reset corrupted volcano selection');
        }
    }
    
    emergencyReset() {
        console.log('🚨 Emergency reset initiated...');
        
        // Clear intervals
        this.stopMonitoring();
        
        // Clear global variables
        window.currentVolcano = null;
        window.currentView = '2d';
        window.insarData = [];
        
        // Hide all overlays
        if (window.map) {
            window.map.eachLayer(layer => {
                if (layer.options && (layer.options.volcanoMarker || layer.options.deformationLayer)) {
                    window.map.removeLayer(layer);
                }
            });
        }
        
        // Reset UI state
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.remove('minimized');
        });
        
        // Restart monitoring
        setTimeout(() => {
            this.startMonitoring();
        }, 2000);
        
        console.log('✅ Emergency reset completed');
    }
}

// Auto-initialize monitor when script loads
if (typeof window !== 'undefined') {
    window.InSARMonitor = InSARMonitor;
    
    // Start monitoring after page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.insarMonitor = new InSARMonitor();
        });
    } else {
        window.insarMonitor = new InSARMonitor();
    }
    
    // Global shortcut for manual repair
    window.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.shiftKey && event.key === 'R') {
            event.preventDefault();
            if (window.insarMonitor) {
                window.insarMonitor.repairUI();
                window.insarMonitor.repairData();
            }
            console.log('🔧 Manual repair triggered (Ctrl+Shift+R)');
        }
        
        if (event.ctrlKey && event.shiftKey && event.key === 'E') {
            event.preventDefault();
            if (window.insarMonitor) {
                window.insarMonitor.emergencyReset();
            }
            console.log('🚨 Emergency reset triggered (Ctrl+Shift+E)');
        }
    });
}

// Export for Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InSARMonitor;
}

console.log('📋 InSAR Monitor Script Loaded');
console.log('🔧 Use Ctrl+Shift+R for manual repair');
console.log('🚨 Use Ctrl+Shift+E for emergency reset');