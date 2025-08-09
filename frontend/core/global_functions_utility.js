/**
 * Prometheus Global Functions Utility
 * Ensures critical functions are available globally across all pages
 */

(function() {
    'use strict';
    
    // Global function safety checker and provider
    class GlobalFunctionUtility {
        constructor() {
            this.functions = new Map();
            this.initializeCore();
        }

        initializeCore() {
            // Ensure switchView function is always available
            this.ensureFunction('switchView', this.createFallbackSwitchView());
            
            // Ensure other critical functions
            this.ensureFunction('togglePanel', this.createFallbackTogglePanel());
            this.ensureFunction('updateView', this.createFallbackUpdateView());
            
            // Log successful initialization
            console.log('[GlobalFunctionUtility] Core functions initialized');
        }

        ensureFunction(name, fallback) {
            if (typeof window[name] === 'undefined') {
                window[name] = fallback;
                this.functions.set(name, 'fallback');
                console.warn(`[GlobalFunctionUtility] Added fallback for missing function: ${name}`);
            } else {
                this.functions.set(name, 'original');
                console.log(`[GlobalFunctionUtility] Function already exists: ${name}`);
            }
        }

        createFallbackSwitchView() {
            return function(view) {
                console.log(`[Fallback] switchView called with: ${view}`);
                
                // Try common view switching patterns
                const viewMappings = {
                    '2d': ['map-container', 'leaflet-map', 'map-view'],
                    '3d': ['canvas-container', 'three-canvas', '3d-view', 'webgl-container'],
                    'data': ['data-container', 'chart-container', 'data-view'],
                    'analysis': ['analysis-container', 'analysis-view']
                };

                // Hide all potential containers first
                Object.values(viewMappings).flat().forEach(containerId => {
                    const element = document.getElementById(containerId);
                    if (element) {
                        element.style.display = 'none';
                    }
                });

                // Show the requested view containers
                if (viewMappings[view]) {
                    viewMappings[view].forEach(containerId => {
                        const element = document.getElementById(containerId);
                        if (element) {
                            element.style.display = 'block';
                            console.log(`[Fallback] Showed container: ${containerId}`);
                        }
                    });
                }

                // Update button states if they exist
                document.querySelectorAll('.view-toggle-btn, .view-btn, [data-view-btn]').forEach(btn => {
                    btn.classList.remove('active');
                    if (btn.id === `btn-${view}` || btn.dataset.view === view) {
                        btn.classList.add('active');
                    }
                });

                // Trigger custom event for other components to listen to
                window.dispatchEvent(new CustomEvent('viewChanged', { 
                    detail: { view, source: 'fallback' } 
                }));

                return true;
            };
        }

        createFallbackTogglePanel() {
            return function(panelId) {
                console.log(`[Fallback] togglePanel called with: ${panelId}`);
                
                const panel = document.getElementById(panelId);
                if (panel) {
                    const isVisible = panel.style.display !== 'none';
                    panel.style.display = isVisible ? 'none' : 'block';
                    
                    // Add/remove minimized class
                    if (isVisible) {
                        panel.classList.add('minimized');
                    } else {
                        panel.classList.remove('minimized');
                    }
                    
                    console.log(`[Fallback] Panel ${panelId} ${isVisible ? 'hidden' : 'shown'}`);
                    return !isVisible;
                } else {
                    console.warn(`[Fallback] Panel not found: ${panelId}`);
                    return false;
                }
            };
        }

        createFallbackUpdateView() {
            return function(options = {}) {
                console.log(`[Fallback] updateView called with:`, options);
                
                // Generic view update logic
                if (options.view) {
                    window.switchView(options.view);
                }
                
                if (options.data) {
                    // Update any data displays
                    const dataElements = document.querySelectorAll('[data-update-target]');
                    dataElements.forEach(el => {
                        const target = el.dataset.updateTarget;
                        if (options.data[target]) {
                            el.textContent = options.data[target];
                        }
                    });
                }
                
                return true;
            };
        }

        // Public API
        registerFunction(name, func) {
            if (typeof func === 'function') {
                window[name] = func;
                this.functions.set(name, 'registered');
                console.log(`[GlobalFunctionUtility] Registered function: ${name}`);
            }
        }

        getFunctionStatus(name) {
            return this.functions.get(name) || 'not found';
        }

        getAllFunctions() {
            const status = {};
            this.functions.forEach((type, name) => {
                status[name] = type;
            });
            return status;
        }

        // Error recovery mechanisms
        recoverMissingFunction(name) {
            if (typeof window[name] === 'undefined') {
                switch (name) {
                    case 'switchView':
                        this.ensureFunction(name, this.createFallbackSwitchView());
                        break;
                    case 'togglePanel':
                        this.ensureFunction(name, this.createFallbackTogglePanel());
                        break;
                    case 'updateView':
                        this.ensureFunction(name, this.createFallbackUpdateView());
                        break;
                    default:
                        console.warn(`[GlobalFunctionUtility] No recovery available for: ${name}`);
                        return false;
                }
                return true;
            }
            return false;
        }

        // Diagnostic tools
        testFunctions() {
            const results = {};
            this.functions.forEach((type, name) => {
                try {
                    const func = window[name];
                    results[name] = {
                        exists: typeof func === 'function',
                        type: type,
                        callable: true
                    };
                } catch (error) {
                    results[name] = {
                        exists: false,
                        type: type,
                        callable: false,
                        error: error.message
                    };
                }
            });
            return results;
        }
    }

    // Initialize the utility when DOM is ready
    function initializeGlobalFunctions() {
        if (typeof window.globalFunctionUtility === 'undefined') {
            window.globalFunctionUtility = new GlobalFunctionUtility();
            
            // Set up periodic health checks
            setInterval(() => {
                if (window.prometheusMonitor) {
                    const testResults = window.globalFunctionUtility.testFunctions();
                    const failedFunctions = Object.entries(testResults)
                        .filter(([name, result]) => !result.exists || !result.callable);
                    
                    if (failedFunctions.length > 0) {
                        window.prometheusMonitor.addCustomError('missing_functions', 
                            'Critical functions missing or non-callable', 
                            { failedFunctions: failedFunctions.map(([name]) => name) });
                        
                        // Attempt recovery
                        failedFunctions.forEach(([name]) => {
                            window.globalFunctionUtility.recoverMissingFunction(name);
                        });
                    }
                }
            }, 30000); // Check every 30 seconds
        }
    }

    // Initialize based on document state
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeGlobalFunctions);
    } else {
        initializeGlobalFunctions();
    }

    // Handle cases where functions might be overridden after initialization
    const originalSwitchView = window.switchView;
    Object.defineProperty(window, 'switchView', {
        set: function(value) {
            if (typeof value === 'function') {
                window.globalFunctionUtility?.functions.set('switchView', 'overridden');
                console.log('[GlobalFunctionUtility] switchView function was overridden');
            }
            return value;
        },
        get: function() {
            return originalSwitchView || window.globalFunctionUtility?.createFallbackSwitchView();
        },
        configurable: true
    });

})();

// Expose utility functions globally for debugging
window.debugGlobalFunctions = function() {
    if (window.globalFunctionUtility) {
        console.table(window.globalFunctionUtility.getAllFunctions());
        console.log('Test results:', window.globalFunctionUtility.testFunctions());
    } else {
        console.error('Global Function Utility not initialized');
    }
};