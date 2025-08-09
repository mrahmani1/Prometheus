/**
 * Prometheus Universal Fix Script
 * Comprehensive solution for all JavaScript errors across Prometheus platform
 * Version: 2.0
 * Date: 2025-08-07
 */

// Prevent redeclaration errors and create global namespace
if (typeof window.PrometheusUniversalFix === 'undefined') {
    window.PrometheusUniversalFix = {
        initialized: false,
        libraries: {},
        fallbacks: {},
        errors: [],
        status: {
            webgl: false,
            three: false,
            chart: false,
            plotly: false,
            leaflet: false
        }
    };
}

(function() {
    'use strict';
    
    const PUF = window.PrometheusUniversalFix;
    
    // Error Collection System
    const ErrorCollector = {
        errors: [],
        
        log: function(error, context = 'Unknown') {
            const errorInfo = {
                timestamp: new Date().toISOString(),
                context: context,
                message: error.message || error,
                stack: error.stack,
                url: window.location.href,
                userAgent: navigator.userAgent
            };
            
            this.errors.push(errorInfo);
            PUF.errors.push(errorInfo);
            
            // Send to error collection endpoint if available
            this.sendError(errorInfo);
            
            console.warn('[Prometheus Error]', errorInfo);
        },
        
        sendError: function(errorInfo) {
            try {
                if (typeof fetch !== 'undefined') {
                    fetch('/api/errors', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(errorInfo)
                    }).catch(() => {});
                }
            } catch (e) {}
        }
    };
    
    // Library Detection and Fallbacks
    const LibraryManager = {
        checkLibrary: function(name, testFn) {
            try {
                const available = testFn();
                PUF.status[name] = available;
                if (available) {
                    console.log(`✓ ${name} library detected`);
                } else {
                    console.warn(`✗ ${name} library not available, using fallback`);
                    this.loadFallback(name);
                }
                return available;
            } catch (e) {
                ErrorCollector.log(e, `Library check: ${name}`);
                PUF.status[name] = false;
                this.loadFallback(name);
                return false;
            }
        },
        
        loadFallback: function(name) {
            const fallbacks = {
                three: this.createThreeFallback,
                chart: this.createChartFallback,
                plotly: this.createPlotlyFallback,
                leaflet: this.createLeafletFallback
            };
            
            if (fallbacks[name]) {
                try {
                    fallbacks[name]();
                } catch (e) {
                    ErrorCollector.log(e, `Loading fallback: ${name}`);
                }
            }
        },
        
        createThreeFallback: function() {
            if (typeof THREE === 'undefined') {
                window.THREE = {
                    WebGLRenderer: function(options) {
                        return {
                            domElement: document.createElement('div'),
                            setSize: function() {},
                            render: function() {},
                            getContext: function() { return null; },
                            capabilities: {},
                            dispose: function() {}
                        };
                    },
                    Scene: function() { return { add: function() {} }; },
                    PerspectiveCamera: function() { return {}; },
                    Vector3: function(x, y, z) { 
                        return { x: x || 0, y: y || 0, z: z || 0, set: function() {} }; 
                    },
                    Color: function() { return {}; },
                    Mesh: function() { return {}; },
                    BoxGeometry: function() { return {}; },
                    MeshBasicMaterial: function() { return {}; },
                    DirectionalLight: function() { return {}; },
                    AmbientLight: function() { return {}; }
                };
            }
        },
        
        createChartFallback: function() {
            if (typeof Chart === 'undefined') {
                window.Chart = function(ctx, config) {
                    return {
                        update: function() {},
                        destroy: function() {},
                        data: config.data || {},
                        options: config.options || {}
                    };
                };
                Chart.register = function() {};
            }
        },
        
        createPlotlyFallback: function() {
            if (typeof Plotly === 'undefined') {
                window.Plotly = {
                    newPlot: function(div, data, layout) {
                        if (div && typeof div === 'string') div = document.getElementById(div);
                        if (div) div.innerHTML = '<div style="padding: 20px; text-align: center;">Plotly not available</div>';
                    },
                    update: function() {},
                    react: function() {},
                    redraw: function() {}
                };
            }
        },
        
        createLeafletFallback: function() {
            if (typeof L === 'undefined') {
                window.L = {
                    map: function(id) {
                        return {
                            setView: function() { return this; },
                            addLayer: function() { return this; },
                            removeLayer: function() { return this; },
                            on: function() { return this; }
                        };
                    },
                    tileLayer: function() {
                        return { addTo: function() {} };
                    },
                    marker: function() {
                        return { addTo: function() {} };
                    }
                };
            }
        }
    };
    
    // WebGL Detection and Fixes
    const WebGLManager = {
        init: function() {
            try {
                const canvas = document.createElement('canvas');
                const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                
                if (gl) {
                    PUF.status.webgl = true;
                    this.fixWebGLExtensions(gl);
                    console.log('✓ WebGL available');
                } else {
                    PUF.status.webgl = false;
                    this.createWebGLFallback();
                    console.warn('✗ WebGL not available, using fallback');
                }
            } catch (e) {
                ErrorCollector.log(e, 'WebGL initialization');
                this.createWebGLFallback();
            }
        },
        
        fixWebGLExtensions: function(gl) {
            const extensions = [
                'EXT_color_buffer_float',
                'WEBGL_color_buffer_float',
                'OES_texture_float',
                'OES_texture_float_linear',
                'EXT_float_blend',
                'WEBGL_depth_texture'
            ];
            
            extensions.forEach(ext => {
                try {
                    gl.getExtension(ext);
                } catch (e) {
                    // Silently fail for unsupported extensions
                }
            });
        },
        
        createWebGLFallback: function() {
            // Create fallback for WebGL-dependent functions
            window.fixWebGLExtensions = function() {};
        }
    };
    
    // HOLOCENE_VOLCANOES fix to prevent redeclaration
    const VolcanoManager = {
        init: function() {
            if (typeof window.HOLOCENE_VOLCANOES === 'undefined') {
                window.HOLOCENE_VOLCANOES = {};
            }
        }
    };
    
    // Universal Functions Library
    const UniversalFunctions = {
        // View switching functionality
        switchView: function(view) {
            try {
                // Hide all views
                const allViews = document.querySelectorAll('[class*="view"], [id*="view"]');
                allViews.forEach(v => {
                    if (v.style) v.style.display = 'none';
                });
                
                // Show target view
                const targetView = document.getElementById(view) || 
                                 document.querySelector(`.${view}`) ||
                                 document.querySelector(`[data-view="${view}"]`);
                
                if (targetView && targetView.style) {
                    targetView.style.display = 'block';
                }
                
                // Update navigation
                const navItems = document.querySelectorAll('.nav-item, .tab, [data-tab]');
                navItems.forEach(item => {
                    if (item.classList) {
                        item.classList.remove('active');
                        if (item.textContent.toLowerCase().includes(view.toLowerCase()) ||
                            item.getAttribute('data-view') === view) {
                            item.classList.add('active');
                        }
                    }
                });
                
                // Fire custom event
                window.dispatchEvent(new CustomEvent('viewChanged', { detail: { view } }));
                
            } catch (e) {
                ErrorCollector.log(e, 'switchView function');
            }
        },
        
        // Toggle panel functionality
        togglePanel: function(panelId) {
            try {
                const panel = document.getElementById(panelId);
                if (!panel) return;
                
                const isHidden = panel.style.display === 'none' || 
                               !panel.style.display && window.getComputedStyle(panel).display === 'none';
                
                panel.style.display = isHidden ? 'block' : 'none';
            } catch (e) {
                ErrorCollector.log(e, 'togglePanel function');
            }
        },
        
        // Initialize visualization
        initVisualization: function(containerId, type = '3d') {
            try {
                const container = document.getElementById(containerId);
                if (!container) return null;
                
                if (type === '3d' && window.THREE) {
                    return this.init3DScene(container);
                } else if (type === 'chart' && window.Chart) {
                    return this.initChart(container);
                } else {
                    return this.initFallbackVisualization(container, type);
                }
            } catch (e) {
                ErrorCollector.log(e, 'initVisualization function');
                return null;
            }
        },
        
        init3DScene: function(container) {
            try {
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                
                renderer.setSize(container.clientWidth, container.clientHeight);
                container.appendChild(renderer.domElement);
                
                return { scene, camera, renderer };
            } catch (e) {
                ErrorCollector.log(e, 'init3DScene');
                return this.initFallbackVisualization(container, '3d');
            }
        },
        
        initChart: function(container) {
            try {
                const canvas = document.createElement('canvas');
                container.appendChild(canvas);
                
                return new Chart(canvas, {
                    type: 'line',
                    data: { labels: [], datasets: [] },
                    options: { responsive: true }
                });
            } catch (e) {
                ErrorCollector.log(e, 'initChart');
                return this.initFallbackVisualization(container, 'chart');
            }
        },
        
        initFallbackVisualization: function(container, type) {
            container.innerHTML = `
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                    background: linear-gradient(45deg, #1a1a2e, #16213e);
                    color: white;
                    text-align: center;
                    font-family: monospace;
                ">
                    <div>
                        <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                        <div>${type.toUpperCase()} Visualization Loading...</div>
                        <div style="font-size: 12px; opacity: 0.7; margin-top: 5px;">
                            Fallback mode active
                        </div>
                    </div>
                </div>
            `;
            return { fallback: true, type };
        },
        
        // Animation frame management
        animationFrameManager: {
            activeAnimations: new Set(),
            
            start: function(callback, id) {
                if (this.activeAnimations.has(id)) return;
                
                this.activeAnimations.add(id);
                
                const animate = () => {
                    if (!this.activeAnimations.has(id)) return;
                    
                    try {
                        callback();
                    } catch (e) {
                        ErrorCollector.log(e, `Animation: ${id}`);
                    }
                    
                    requestAnimationFrame(animate);
                };
                
                animate();
            },
            
            stop: function(id) {
                this.activeAnimations.delete(id);
            },
            
            stopAll: function() {
                this.activeAnimations.clear();
            }
        },
        
        // Safe event handling
        addSafeEventListener: function(element, event, handler, options) {
            try {
                if (typeof element === 'string') {
                    element = document.getElementById(element) || document.querySelector(element);
                }
                
                if (!element) return;
                
                const safeHandler = function(e) {
                    try {
                        handler.call(this, e);
                    } catch (error) {
                        ErrorCollector.log(error, `Event handler: ${event}`);
                    }
                };
                
                element.addEventListener(event, safeHandler, options);
                return { element, event, handler: safeHandler };
            } catch (e) {
                ErrorCollector.log(e, 'addSafeEventListener');
            }
        },
        
        // Data loading with error handling
        loadData: function(url, callback) {
            fetch(url)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    try {
                        callback(data);
                    } catch (e) {
                        ErrorCollector.log(e, 'Data callback');
                    }
                })
                .catch(error => {
                    ErrorCollector.log(error, 'Data loading');
                    // Provide fallback data
                    callback({ error: true, message: error.message });
                });
        }
    };
    
    // Auto-recovery system
    const AutoRecovery = {
        init: function() {
            // Listen for uncaught errors
            window.addEventListener('error', (e) => {
                ErrorCollector.log(e.error || e, 'Uncaught error');
                this.attemptRecovery(e.error || e);
            });
            
            // Listen for unhandled promise rejections
            window.addEventListener('unhandledrejection', (e) => {
                ErrorCollector.log(e.reason, 'Unhandled promise rejection');
                e.preventDefault(); // Prevent console spam
            });
        },
        
        attemptRecovery: function(error) {
            const message = error.message || error.toString();
            
            // Common recovery strategies
            if (message.includes('THREE is not defined')) {
                LibraryManager.loadFallback('three');
            } else if (message.includes('Chart is not defined')) {
                LibraryManager.loadFallback('chart');
            } else if (message.includes('Plotly is not defined')) {
                LibraryManager.loadFallback('plotly');
            } else if (message.includes('HOLOCENE_VOLCANOES')) {
                VolcanoManager.init();
            } else if (message.includes('switchView is not defined')) {
                window.switchView = UniversalFunctions.switchView;
            }
        }
    };
    
    // Status monitoring
    const StatusMonitor = {
        getStatus: function() {
            return {
                libraries: PUF.status,
                errors: PUF.errors.length,
                timestamp: new Date().toISOString(),
                page: window.location.pathname,
                userAgent: navigator.userAgent.substring(0, 50)
            };
        },
        
        sendStatus: function() {
            try {
                if (typeof fetch !== 'undefined') {
                    fetch('/api/status', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.getStatus())
                    }).catch(() => {});
                }
            } catch (e) {}
        }
    };
    
    // Main initialization
    function initialize() {
        if (PUF.initialized) return;
        
        console.log('🚀 Initializing Prometheus Universal Fix System...');
        
        // Initialize components
        WebGLManager.init();
        VolcanoManager.init();
        AutoRecovery.init();
        
        // Check libraries
        LibraryManager.checkLibrary('three', () => typeof THREE !== 'undefined');
        LibraryManager.checkLibrary('chart', () => typeof Chart !== 'undefined');
        LibraryManager.checkLibrary('plotly', () => typeof Plotly !== 'undefined');
        LibraryManager.checkLibrary('leaflet', () => typeof L !== 'undefined');
        
        // Expose universal functions
        Object.assign(window, {
            switchView: UniversalFunctions.switchView,
            togglePanel: UniversalFunctions.togglePanel,
            initVisualization: UniversalFunctions.initVisualization,
            addSafeEventListener: UniversalFunctions.addSafeEventListener,
            loadData: UniversalFunctions.loadData,
            animationFrameManager: UniversalFunctions.animationFrameManager,
            PrometheusStatus: StatusMonitor.getStatus
        });
        
        // Set up status monitoring
        setInterval(() => StatusMonitor.sendStatus(), 30000);
        
        PUF.initialized = true;
        console.log('✅ Prometheus Universal Fix System initialized successfully');
        
        // Dispatch ready event
        window.dispatchEvent(new CustomEvent('PrometheusUniversalFixReady'));
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
    
})();

// Additional try-catch wrappers for common operations
window.safeExecute = function(fn, context = 'Unknown operation') {
    try {
        return fn();
    } catch (e) {
        if (window.PrometheusUniversalFix) {
            window.PrometheusUniversalFix.errors.push({
                timestamp: new Date().toISOString(),
                context: context,
                message: e.message,
                stack: e.stack
            });
        }
        console.warn(`[Safe Execute] Error in ${context}:`, e);
        return null;
    }
};

// Expose error collector globally
window.PrometheusErrorCollector = {
    log: function(error, context) {
        if (window.PrometheusUniversalFix) {
            window.PrometheusUniversalFix.errors.push({
                timestamp: new Date().toISOString(),
                context: context || 'Manual log',
                message: error.message || error,
                stack: error.stack
            });
        }
        console.warn('[Prometheus Error]', error);
    }
};

console.log('📦 Prometheus Universal Fix Script loaded');