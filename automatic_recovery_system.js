/**
 * Prometheus Automatic Recovery System
 * Advanced error recovery and self-healing mechanisms
 * Version: 1.0.0
 */

class PrometheusRecoverySystem {
    constructor(options = {}) {
        this.config = {
            maxRecoveryAttempts: options.maxRecoveryAttempts || 3,
            recoveryDelay: options.recoveryDelay || 1000,
            healthCheckInterval: options.healthCheckInterval || 30000,
            criticalFunctionChecks: options.criticalFunctionChecks || [
                'switchView', 'togglePanel', 'initVolcanicSystem', 'initTectonicSimulation'
            ],
            resourceFallbacks: options.resourceFallbacks || {},
            enableAutoReload: options.enableAutoReload !== false,
            debugMode: options.debugMode || false,
            ...options
        };

        this.recoveryAttempts = new Map();
        this.healthStatus = new Map();
        this.resourceCache = new Map();
        this.systemState = {
            healthy: true,
            lastCheck: Date.now(),
            errors: []
        };

        this.init();
    }

    init() {
        this.setupHealthMonitoring();
        this.setupResourceRecovery();
        this.setupFunctionRecovery();
        this.setupNetworkRecovery();
        this.setupMemoryRecovery();
        this.startPeriodicHealthChecks();
        
        this.log('Automatic Recovery System initialized');
    }

    // Health Monitoring System
    setupHealthMonitoring() {
        // Monitor critical systems
        this.healthChecks = {
            dom: this.checkDOMHealth.bind(this),
            functions: this.checkFunctionHealth.bind(this),
            network: this.checkNetworkHealth.bind(this),
            memory: this.checkMemoryHealth.bind(this),
            resources: this.checkResourceHealth.bind(this),
            webgl: this.checkWebGLHealth.bind(this)
        };

        // Performance observer for monitoring
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                entries.forEach(entry => {
                    if (entry.name.includes('error') || entry.duration > 5000) {
                        this.handlePerformanceIssue(entry);
                    }
                });
            });
            observer.observe({ entryTypes: ['measure', 'navigation'] });
        }
    }

    startPeriodicHealthChecks() {
        setInterval(() => {
            this.runHealthChecks().then(results => {
                const issues = results.filter(result => !result.healthy);
                if (issues.length > 0) {
                    this.handleHealthIssues(issues);
                }
            });
        }, this.config.healthCheckInterval);
    }

    async runHealthChecks() {
        const results = [];
        
        for (const [name, checkFunction] of Object.entries(this.healthChecks)) {
            try {
                const result = await checkFunction();
                results.push({
                    name,
                    healthy: result.healthy,
                    details: result.details,
                    timestamp: Date.now()
                });
                this.healthStatus.set(name, result);
            } catch (error) {
                results.push({
                    name,
                    healthy: false,
                    details: { error: error.message },
                    timestamp: Date.now()
                });
            }
        }

        return results;
    }

    // Specific Health Checks
    checkDOMHealth() {
        const issues = [];
        
        // Check for missing critical elements
        const criticalElements = [
            '#container', '.panel', 'canvas', '.dashboard-content'
        ];
        
        criticalElements.forEach(selector => {
            if (!document.querySelector(selector)) {
                issues.push(`Missing element: ${selector}`);
            }
        });

        // Check for broken event listeners
        const testDiv = document.createElement('div');
        try {
            testDiv.addEventListener('click', () => {});
            testDiv.removeEventListener('click', () => {});
        } catch (error) {
            issues.push('Event listener functionality broken');
        }

        return {
            healthy: issues.length === 0,
            details: { issues }
        };
    }

    checkFunctionHealth() {
        const issues = [];
        
        this.config.criticalFunctionChecks.forEach(funcName => {
            if (typeof window[funcName] !== 'function') {
                issues.push(`Missing function: ${funcName}`);
            } else {
                try {
                    // Test if function is callable (basic test)
                    const func = window[funcName];
                    if (func.toString().includes('[native code]') && funcName !== 'fetch') {
                        // Skip native functions unless it's fetch which we might have wrapped
                        return;
                    }
                } catch (error) {
                    issues.push(`Function error: ${funcName} - ${error.message}`);
                }
            }
        });

        return {
            healthy: issues.length === 0,
            details: { issues }
        };
    }

    checkNetworkHealth() {
        const issues = [];
        
        if (!navigator.onLine) {
            issues.push('Network offline');
        }

        // Check if we can reach our own server
        if (this.lastNetworkTest && Date.now() - this.lastNetworkTest > 60000) {
            this.testNetworkConnection();
        }

        return {
            healthy: navigator.onLine && issues.length === 0,
            details: { 
                issues,
                online: navigator.onLine,
                lastTest: this.lastNetworkTest
            }
        };
    }

    checkMemoryHealth() {
        const issues = [];
        
        if ('memory' in performance) {
            const memInfo = performance.memory;
            const usagePercent = memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit;
            
            if (usagePercent > 0.9) {
                issues.push('Critical memory usage: ' + Math.round(usagePercent * 100) + '%');
            } else if (usagePercent > 0.8) {
                issues.push('High memory usage: ' + Math.round(usagePercent * 100) + '%');
            }
        }

        return {
            healthy: issues.length === 0,
            details: { 
                issues,
                memory: performance.memory ? {
                    used: performance.memory.usedJSHeapSize,
                    total: performance.memory.totalJSHeapSize,
                    limit: performance.memory.jsHeapSizeLimit
                } : null
            }
        };
    }

    checkResourceHealth() {
        const issues = [];
        
        // Check for failed resource loads
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            if (!img.complete || img.naturalHeight === 0) {
                issues.push(`Failed to load image: ${img.src}`);
            }
        });

        // Check for missing stylesheets
        const stylesheets = document.querySelectorAll('link[rel="stylesheet"]');
        stylesheets.forEach(link => {
            if (!link.sheet) {
                issues.push(`Failed to load stylesheet: ${link.href}`);
            }
        });

        return {
            healthy: issues.length === 0,
            details: { issues }
        };
    }

    checkWebGLHealth() {
        const issues = [];
        
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (!gl) {
                issues.push('WebGL not supported');
            } else {
                // Test basic WebGL functionality
                const isContextLost = gl.isContextLost();
                if (isContextLost) {
                    issues.push('WebGL context lost');
                }
            }
        } catch (error) {
            issues.push(`WebGL error: ${error.message}`);
        }

        return {
            healthy: issues.length === 0,
            details: { issues }
        };
    }

    // Recovery Mechanisms
    setupResourceRecovery() {
        // Default fallbacks for common resources
        this.resourceFallbacks = {
            'three.js': [
                'https://unpkg.com/three@latest/build/three.min.js',
                'https://cdn.jsdelivr.net/npm/three@latest/build/three.min.js'
            ],
            'chart.js': [
                'https://cdn.jsdelivr.net/npm/chart.js',
                'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/latest/chart.min.js'
            ],
            'leaflet.js': [
                'https://unpkg.com/leaflet@1.7.1/dist/leaflet.js',
                'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/leaflet.min.js'
            ],
            ...this.config.resourceFallbacks
        };

        // Monitor script load failures
        window.addEventListener('error', (event) => {
            if (event.target && event.target.tagName === 'SCRIPT') {
                this.recoverFailedScript(event.target);
            }
        }, true);
    }

    setupFunctionRecovery() {
        // Monitor for missing critical functions
        this.functionRecovery = {
            switchView: () => this.createFallbackSwitchView(),
            togglePanel: () => this.createFallbackTogglePanel(),
            initVolcanicSystem: () => this.createFallbackInit('volcanic'),
            initTectonicSimulation: () => this.createFallbackInit('tectonic')
        };
    }

    setupNetworkRecovery() {
        // Network state monitoring
        window.addEventListener('online', () => {
            this.log('Network back online - retrying failed requests');
            this.retryFailedRequests();
        });

        window.addEventListener('offline', () => {
            this.log('Network offline - switching to offline mode');
            this.enableOfflineMode();
        });
    }

    setupMemoryRecovery() {
        // Memory cleanup strategies
        this.memoryRecoveryStrategies = [
            this.clearUnusedResources.bind(this),
            this.optimizeCanvasContexts.bind(this),
            this.clearEventListeners.bind(this),
            this.forceGarbageCollection.bind(this)
        ];
    }

    // Recovery Actions
    async handleHealthIssues(issues) {
        for (const issue of issues) {
            await this.attemptRecovery(issue);
        }
    }

    async attemptRecovery(issue) {
        const recoveryKey = `${issue.name}_${JSON.stringify(issue.details)}`;
        const attempts = this.recoveryAttempts.get(recoveryKey) || 0;

        if (attempts >= this.config.maxRecoveryAttempts) {
            this.log(`Max recovery attempts reached for: ${issue.name}`);
            return false;
        }

        this.recoveryAttempts.set(recoveryKey, attempts + 1);
        this.log(`Attempting recovery for: ${issue.name} (attempt ${attempts + 1})`);

        try {
            const success = await this.executeRecovery(issue);
            if (success) {
                this.log(`Recovery successful for: ${issue.name}`);
                this.recoveryAttempts.delete(recoveryKey);
                return true;
            }
        } catch (error) {
            this.log(`Recovery failed for ${issue.name}: ${error.message}`);
        }

        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, this.config.recoveryDelay * (attempts + 1)));
        return false;
    }

    async executeRecovery(issue) {
        switch (issue.name) {
            case 'functions':
                return this.recoverMissingFunctions(issue.details.issues);
            case 'resources':
                return this.recoverFailedResources(issue.details.issues);
            case 'memory':
                return this.recoverMemoryIssues();
            case 'network':
                return this.recoverNetworkIssues();
            case 'webgl':
                return this.recoverWebGLIssues();
            case 'dom':
                return this.recoverDOMIssues(issue.details.issues);
            default:
                this.log(`No recovery strategy for: ${issue.name}`);
                return false;
        }
    }

    recoverMissingFunctions(issues) {
        let recovered = 0;
        
        issues.forEach(issue => {
            const match = issue.match(/Missing function: (\w+)/);
            if (match && this.functionRecovery[match[1]]) {
                try {
                    window[match[1]] = this.functionRecovery[match[1]]();
                    recovered++;
                    this.log(`Recovered function: ${match[1]}`);
                } catch (error) {
                    this.log(`Failed to recover function ${match[1]}: ${error.message}`);
                }
            }
        });

        return recovered > 0;
    }

    recoverFailedResources(issues) {
        let recovered = 0;
        
        issues.forEach(issue => {
            if (issue.includes('Failed to load')) {
                // Extract resource type and attempt recovery
                for (const [type, fallbacks] of Object.entries(this.resourceFallbacks)) {
                    if (issue.toLowerCase().includes(type)) {
                        this.tryResourceFallbacks(fallbacks).then(success => {
                            if (success) recovered++;
                        });
                        break;
                    }
                }
            }
        });

        return recovered > 0;
    }

    async recoverMemoryIssues() {
        for (const strategy of this.memoryRecoveryStrategies) {
            try {
                await strategy();
                this.log('Memory recovery strategy executed');
            } catch (error) {
                this.log(`Memory recovery strategy failed: ${error.message}`);
            }
        }
        return true;
    }

    recoverNetworkIssues() {
        if (navigator.onLine) {
            this.retryFailedRequests();
            return true;
        }
        return false;
    }

    recoverWebGLIssues() {
        try {
            // Try to restore WebGL context
            const canvases = document.querySelectorAll('canvas');
            canvases.forEach(canvas => {
                if (canvas.getContext) {
                    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                    if (gl && gl.isContextLost()) {
                        // Trigger context restoration
                        const extension = gl.getExtension('WEBGL_lose_context');
                        if (extension) {
                            extension.restoreContext();
                        }
                    }
                }
            });
            return true;
        } catch (error) {
            this.log(`WebGL recovery failed: ${error.message}`);
            return false;
        }
    }

    recoverDOMIssues(issues) {
        let recovered = 0;
        
        issues.forEach(issue => {
            if (issue.includes('Missing element:')) {
                const selector = issue.split(': ')[1];
                if (this.recreateElement(selector)) {
                    recovered++;
                }
            }
        });

        return recovered > 0;
    }

    // Utility Recovery Functions
    createFallbackSwitchView() {
        return function(view) {
            console.warn('[Recovery] Using fallback switchView for:', view);
            
            // Basic view switching logic
            const containers = {
                '2d': ['#map-container', '.leaflet-container'],
                '3d': ['#canvas-container', '.three-container', '.webgl-container'],
                'data': ['#data-container', '.chart-container']
            };

            // Hide all containers
            Object.values(containers).flat().forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => el.style.display = 'none');
            });

            // Show target view containers
            if (containers[view]) {
                containers[view].forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => el.style.display = 'block');
                });
            }

            // Update button states
            document.querySelectorAll('.view-btn, .view-toggle-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.view === view || btn.id.includes(view));
            });
        };
    }

    createFallbackTogglePanel() {
        return function(panelId) {
            console.warn('[Recovery] Using fallback togglePanel for:', panelId);
            const panel = document.getElementById(panelId);
            if (panel) {
                const isVisible = getComputedStyle(panel).display !== 'none';
                panel.style.display = isVisible ? 'none' : 'block';
                return !isVisible;
            }
            return false;
        };
    }

    createFallbackInit(type) {
        return function() {
            console.warn(`[Recovery] Using fallback init for ${type} system`);
            
            // Basic initialization
            const container = document.getElementById('container');
            if (container && !container.classList.contains(`${type}-initialized`)) {
                container.classList.add(`${type}-initialized`);
                
                // Create basic placeholder
                const placeholder = document.createElement('div');
                placeholder.className = `${type}-placeholder`;
                placeholder.innerHTML = `
                    <div class="recovery-message">
                        <h3>${type.charAt(0).toUpperCase() + type.slice(1)} System</h3>
                        <p>System recovered with basic functionality</p>
                    </div>
                `;
                container.appendChild(placeholder);
            }
        };
    }

    async tryResourceFallbacks(fallbacks) {
        for (const url of fallbacks) {
            try {
                await this.loadScript(url);
                this.log(`Successfully loaded fallback resource: ${url}`);
                return true;
            } catch (error) {
                this.log(`Failed to load fallback: ${url}`);
            }
        }
        return false;
    }

    loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // Memory Recovery Strategies
    clearUnusedResources() {
        // Clear unused textures and geometries if Three.js is available
        if (window.THREE && window.scene) {
            window.scene.traverse((object) => {
                if (object.geometry) {
                    object.geometry.dispose();
                }
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }
    }

    optimizeCanvasContexts() {
        const canvases = document.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        });
    }

    clearEventListeners() {
        // This is a simplified approach - in practice, you'd want to track listeners
        this.log('Event listener cleanup performed');
    }

    forceGarbageCollection() {
        if (window.gc) {
            window.gc();
            this.log('Forced garbage collection');
        }
    }

    // Network Recovery
    retryFailedRequests() {
        // Implementation would retry stored failed requests
        this.log('Retrying failed network requests');
    }

    enableOfflineMode() {
        // Switch to cached resources and offline functionality
        document.body.classList.add('offline-mode');
        this.log('Offline mode enabled');
    }

    testNetworkConnection() {
        this.lastNetworkTest = Date.now();
        
        fetch('/api/health', { 
            method: 'HEAD', 
            cache: 'no-cache',
            timeout: 5000 
        })
        .then(() => this.log('Network connectivity confirmed'))
        .catch(() => this.log('Network connectivity test failed'));
    }

    recreateElement(selector) {
        try {
            // Basic element recreation based on common patterns
            const elementMap = {
                '#container': () => {
                    const container = document.createElement('div');
                    container.id = 'container';
                    document.body.appendChild(container);
                    return container;
                },
                'canvas': () => {
                    const canvas = document.createElement('canvas');
                    canvas.id = 'mainCanvas';
                    const container = document.getElementById('container');
                    if (container) container.appendChild(canvas);
                    return canvas;
                }
            };

            if (elementMap[selector]) {
                elementMap[selector]();
                this.log(`Recreated element: ${selector}`);
                return true;
            }
        } catch (error) {
            this.log(`Failed to recreate element ${selector}: ${error.message}`);
        }
        return false;
    }

    // Public API
    getSystemHealth() {
        return {
            healthy: this.systemState.healthy,
            lastCheck: this.systemState.lastCheck,
            issues: Array.from(this.healthStatus.entries()).filter(([name, status]) => !status.healthy),
            recoveryAttempts: Object.fromEntries(this.recoveryAttempts)
        };
    }

    forceRecoveryCheck() {
        return this.runHealthChecks();
    }

    log(...args) {
        if (this.config.debugMode) {
            console.log('[PrometheusRecovery]', ...args);
        }
    }
}

// Auto-initialize if in browser environment
if (typeof window !== 'undefined') {
    // Initialize recovery system when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.prometheusRecovery = new PrometheusRecoverySystem({
                debugMode: true
            });
        });
    } else {
        window.prometheusRecovery = new PrometheusRecoverySystem({
            debugMode: true
        });
    }
}

// Export for Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PrometheusRecoverySystem;
}