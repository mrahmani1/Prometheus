/**
 * Prometheus Remote Monitoring & Debugging Agent
 * Comprehensive error capture and reporting system
 * Version: 2.0.0
 * Target: http://20.80.89.29:8080/
 */

class PrometheusRemoteMonitor {
    constructor(options = {}) {
        this.config = {
            endpoint: options.endpoint || 'http://20.80.89.29:8080/api/errors',
            maxErrors: options.maxErrors || 1000,
            batchSize: options.batchSize || 10,
            flushInterval: options.flushInterval || 30000, // 30 seconds
            retryAttempts: options.retryAttempts || 3,
            enableRecovery: options.enableRecovery || true,
            debugMode: options.debugMode || false,
            sessionId: this.generateSessionId(),
            userId: options.userId || 'anonymous',
            ...options
        };

        this.errorBuffer = [];
        this.performanceMetrics = {};
        this.networkFailures = [];
        this.consoleCapture = [];
        this.resourceErrors = [];
        this.isOnline = navigator.onLine;
        this.lastHeartbeat = Date.now();
        
        this.init();
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    init() {
        this.setupErrorCapture();
        this.setupConsoleCapture();
        this.setupNetworkMonitoring();
        this.setupPerformanceMonitoring();
        this.setupResourceErrorCapture();
        this.setupUnhandledRejectionCapture();
        this.setupVisibilityTracking();
        this.setupHeartbeat();
        this.startFlushTimer();
        
        if (this.config.enableRecovery) {
            this.setupRecoveryMechanisms();
        }

        // Send initialization event
        this.captureEvent('monitor_initialized', {
            userAgent: navigator.userAgent,
            url: window.location.href,
            timestamp: new Date().toISOString(),
            config: this.config
        });

        this.log('Prometheus Remote Monitor initialized', this.config);
    }

    setupErrorCapture() {
        // Global error handler
        window.addEventListener('error', (event) => {
            const errorData = {
                type: 'javascript_error',
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno,
                stack: event.error ? event.error.stack : null,
                url: window.location.href,
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString(),
                sessionId: this.config.sessionId,
                userId: this.config.userId
            };

            this.captureError(errorData);
            
            // Attempt recovery if enabled
            if (this.config.enableRecovery) {
                this.attemptRecovery('javascript_error', errorData);
            }
        });

        // Capture specific function errors
        this.wrapFunction('fetch', this.monitorFetch.bind(this));
        this.wrapFunction('XMLHttpRequest', this.monitorXHR.bind(this));
    }

    setupConsoleCapture() {
        const originalConsole = {
            log: console.log,
            warn: console.warn,
            error: console.error,
            info: console.info,
            debug: console.debug
        };

        ['log', 'warn', 'error', 'info', 'debug'].forEach(method => {
            console[method] = (...args) => {
                // Call original method
                originalConsole[method].apply(console, args);

                // Capture for monitoring
                const logData = {
                    type: 'console_' + method,
                    message: args.map(arg => 
                        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
                    ).join(' '),
                    args: args,
                    url: window.location.href,
                    timestamp: new Date().toISOString(),
                    sessionId: this.config.sessionId,
                    stack: new Error().stack
                };

                if (method === 'error' || method === 'warn') {
                    this.captureError(logData);
                } else {
                    this.consoleCapture.push(logData);
                    if (this.consoleCapture.length > 100) {
                        this.consoleCapture.shift();
                    }
                }
            };
        });
    }

    setupNetworkMonitoring() {
        // Monitor network status
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.captureEvent('network_online', { timestamp: new Date().toISOString() });
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.captureEvent('network_offline', { timestamp: new Date().toISOString() });
        });

        // Monitor failed requests
        this.monitorFailedRequests();
    }

    setupPerformanceMonitoring() {
        // Performance metrics collection
        if ('performance' in window) {
            // Page load performance
            window.addEventListener('load', () => {
                setTimeout(() => {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    if (perfData) {
                        this.performanceMetrics.pageLoad = {
                            domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                            loadComplete: perfData.loadEventEnd - perfData.loadEventStart,
                            firstPaint: this.getFirstPaint(),
                            timestamp: new Date().toISOString()
                        };
                        this.captureEvent('performance_metrics', this.performanceMetrics.pageLoad);
                    }
                }, 1000);
            });

            // Memory usage monitoring
            if ('memory' in performance) {
                setInterval(() => {
                    const memInfo = performance.memory;
                    const memoryData = {
                        usedJSHeapSize: memInfo.usedJSHeapSize,
                        totalJSHeapSize: memInfo.totalJSHeapSize,
                        jsHeapSizeLimit: memInfo.jsHeapSizeLimit,
                        timestamp: new Date().toISOString()
                    };
                    
                    // Alert on high memory usage
                    if (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit > 0.8) {
                        this.captureError({
                            type: 'high_memory_usage',
                            message: 'Memory usage exceeds 80%',
                            memoryData: memoryData,
                            url: window.location.href,
                            timestamp: new Date().toISOString()
                        });
                    }
                }, 30000); // Check every 30 seconds
            }
        }
    }

    setupResourceErrorCapture() {
        // Monitor failed resource loads (images, scripts, stylesheets)
        window.addEventListener('error', (event) => {
            if (event.target !== window) {
                const resourceError = {
                    type: 'resource_error',
                    element: event.target.tagName,
                    source: event.target.src || event.target.href,
                    message: 'Failed to load resource',
                    url: window.location.href,
                    timestamp: new Date().toISOString(),
                    sessionId: this.config.sessionId
                };
                
                this.resourceErrors.push(resourceError);
                this.captureError(resourceError);
                
                // Attempt resource recovery
                if (this.config.enableRecovery) {
                    this.attemptResourceRecovery(event.target);
                }
            }
        }, true);
    }

    setupUnhandledRejectionCapture() {
        window.addEventListener('unhandledrejection', (event) => {
            const rejectionData = {
                type: 'unhandled_promise_rejection',
                reason: event.reason,
                message: event.reason ? event.reason.toString() : 'Unhandled Promise Rejection',
                stack: event.reason && event.reason.stack ? event.reason.stack : null,
                url: window.location.href,
                timestamp: new Date().toISOString(),
                sessionId: this.config.sessionId
            };

            this.captureError(rejectionData);
        });
    }

    setupVisibilityTracking() {
        document.addEventListener('visibilitychange', () => {
            const visibilityData = {
                type: 'visibility_change',
                hidden: document.hidden,
                visibilityState: document.visibilityState,
                timestamp: new Date().toISOString()
            };
            this.captureEvent('visibility_change', visibilityData);
        });
    }

    setupHeartbeat() {
        setInterval(() => {
            this.lastHeartbeat = Date.now();
            this.captureEvent('heartbeat', {
                timestamp: new Date().toISOString(),
                uptime: Date.now() - this.config.sessionStart || Date.now(),
                errorCount: this.errorBuffer.length,
                isOnline: this.isOnline
            });
        }, 60000); // Every minute
    }

    setupRecoveryMechanisms() {
        // Global recovery strategies
        this.recoveryStrategies = {
            'javascript_error': this.recoverFromJSError.bind(this),
            'network_error': this.recoverFromNetworkError.bind(this),
            'resource_error': this.recoverFromResourceError.bind(this),
            'high_memory_usage': this.recoverFromMemoryIssue.bind(this)
        };
    }

    wrapFunction(name, wrapper) {
        const original = window[name];
        if (original) {
            window[name] = wrapper(original);
        }
    }

    monitorFetch(originalFetch) {
        return function(...args) {
            const startTime = performance.now();
            return originalFetch.apply(this, args)
                .then(response => {
                    const endTime = performance.now();
                    window.prometheusMonitor.captureEvent('fetch_success', {
                        url: args[0],
                        status: response.status,
                        duration: endTime - startTime,
                        timestamp: new Date().toISOString()
                    });
                    return response;
                })
                .catch(error => {
                    const endTime = performance.now();
                    window.prometheusMonitor.captureError({
                        type: 'fetch_error',
                        url: args[0],
                        message: error.message,
                        duration: endTime - startTime,
                        timestamp: new Date().toISOString()
                    });
                    throw error;
                });
        };
    }

    monitorXHR(originalXHR) {
        return function(...args) {
            const xhr = new originalXHR(...args);
            const originalOpen = xhr.open;
            const originalSend = xhr.send;

            let requestUrl = '';
            let startTime = 0;

            xhr.open = function(method, url, ...rest) {
                requestUrl = url;
                return originalOpen.call(this, method, url, ...rest);
            };

            xhr.send = function(...sendArgs) {
                startTime = performance.now();
                
                xhr.addEventListener('loadend', () => {
                    const endTime = performance.now();
                    const eventData = {
                        url: requestUrl,
                        status: xhr.status,
                        duration: endTime - startTime,
                        timestamp: new Date().toISOString()
                    };

                    if (xhr.status >= 400) {
                        window.prometheusMonitor.captureError({
                            type: 'xhr_error',
                            ...eventData,
                            statusText: xhr.statusText
                        });
                    } else {
                        window.prometheusMonitor.captureEvent('xhr_success', eventData);
                    }
                });

                return originalSend.call(this, ...sendArgs);
            };

            return xhr;
        };
    }

    attemptRecovery(errorType, errorData) {
        if (this.recoveryStrategies[errorType]) {
            try {
                this.recoveryStrategies[errorType](errorData);
                this.captureEvent('recovery_attempted', {
                    errorType,
                    timestamp: new Date().toISOString(),
                    success: true
                });
            } catch (recoveryError) {
                this.captureError({
                    type: 'recovery_failed',
                    originalError: errorType,
                    recoveryError: recoveryError.message,
                    timestamp: new Date().toISOString()
                });
            }
        }
    }

    recoverFromJSError(errorData) {
        // Attempt to reload specific components or reinitialize functions
        if (errorData.message.includes('HOLOCENE_VOLCANOES')) {
            this.fixHoloceneVolcanoesError();
        }
        if (errorData.message.includes('switchView')) {
            this.fixSwitchViewError();
        }
    }

    recoverFromNetworkError(errorData) {
        // Retry failed requests with exponential backoff
        if (this.isOnline) {
            setTimeout(() => {
                // Retry logic here
                this.log('Retrying failed network request:', errorData.url);
            }, Math.pow(2, this.config.retryAttempts) * 1000);
        }
    }

    recoverFromResourceError(element) {
        // Try alternative CDN or local fallback
        if (element.tagName === 'SCRIPT') {
            this.tryFallbackScript(element);
        } else if (element.tagName === 'IMG') {
            this.tryFallbackImage(element);
        }
    }

    fixHoloceneVolcanoesError() {
        if (typeof window.HOLOCENE_VOLCANOES === 'undefined' && typeof HOLOCENE_VOLCANOES !== 'undefined') {
            window.HOLOCENE_VOLCANOES = HOLOCENE_VOLCANOES;
            this.log('Fixed HOLOCENE_VOLCANOES global reference');
        }
    }

    fixSwitchViewError() {
        if (typeof window.switchView === 'undefined') {
            window.switchView = function(view) {
                console.warn('switchView called with view:', view, '- implementing fallback');
                // Basic fallback implementation
                const elements = document.querySelectorAll('[data-view]');
                elements.forEach(el => {
                    if (el.dataset.view === view) {
                        el.style.display = 'block';
                    } else {
                        el.style.display = 'none';
                    }
                });
            };
            this.log('Added fallback switchView function');
        }
    }

    tryFallbackScript(scriptElement) {
        const fallbacks = {
            'three.js': ['https://unpkg.com/three@latest/build/three.min.js'],
            'chart.js': ['https://cdn.jsdelivr.net/npm/chart.js'],
            'plotly.js': ['https://cdn.plot.ly/plotly-latest.min.js'],
            'leaflet.js': ['https://unpkg.com/leaflet@1.7.1/dist/leaflet.js']
        };

        const src = scriptElement.src;
        for (const [key, urls] of Object.entries(fallbacks)) {
            if (src.includes(key)) {
                this.loadFallbackScript(urls[0]);
                break;
            }
        }
    }

    loadFallbackScript(url) {
        const script = document.createElement('script');
        script.src = url;
        script.onload = () => this.log('Fallback script loaded:', url);
        script.onerror = () => this.captureError({
            type: 'fallback_script_failed',
            url: url,
            timestamp: new Date().toISOString()
        });
        document.head.appendChild(script);
    }

    getFirstPaint() {
        if ('getEntriesByType' in performance) {
            const paints = performance.getEntriesByType('paint');
            const firstPaint = paints.find(paint => paint.name === 'first-paint');
            return firstPaint ? firstPaint.startTime : null;
        }
        return null;
    }

    captureError(errorData) {
        // Add additional context
        const enrichedError = {
            ...errorData,
            page: {
                url: window.location.href,
                referrer: document.referrer,
                title: document.title
            },
            browser: {
                userAgent: navigator.userAgent,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine,
                language: navigator.language
            },
            screen: {
                width: screen.width,
                height: screen.height,
                availWidth: screen.availWidth,
                availHeight: screen.availHeight
            },
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            severity: this.determineSeverity(errorData),
            sessionId: this.config.sessionId,
            userId: this.config.userId
        };

        this.errorBuffer.push(enrichedError);

        // Immediate flush for critical errors
        if (enrichedError.severity === 'critical' || this.errorBuffer.length >= this.config.batchSize) {
            this.flush();
        }

        this.log('Error captured:', enrichedError);
    }

    captureEvent(eventType, eventData) {
        const event = {
            type: eventType,
            data: eventData,
            timestamp: new Date().toISOString(),
            sessionId: this.config.sessionId,
            url: window.location.href
        };

        // Store in appropriate buffer based on event type
        if (eventType.includes('error')) {
            this.captureError(event);
        } else {
            // Could be stored separately or sent immediately for real-time monitoring
            if (this.config.debugMode) {
                this.log('Event captured:', event);
            }
        }
    }

    determineSeverity(errorData) {
        const criticalKeywords = ['CORS', 'CSP', 'Security', 'Auth', 'Permission'];
        const highKeywords = ['TypeError', 'ReferenceError', 'Network'];
        const mediumKeywords = ['Warning', '404', 'Resource'];

        const message = errorData.message || '';
        
        if (criticalKeywords.some(keyword => message.includes(keyword))) {
            return 'critical';
        } else if (highKeywords.some(keyword => message.includes(keyword))) {
            return 'high';
        } else if (mediumKeywords.some(keyword => message.includes(keyword))) {
            return 'medium';
        }
        return 'low';
    }

    flush() {
        if (this.errorBuffer.length === 0) return;

        const payload = {
            errors: this.errorBuffer.splice(0, this.config.batchSize),
            metadata: {
                sessionId: this.config.sessionId,
                userId: this.config.userId,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent,
                url: window.location.href
            }
        };

        this.sendToServer(payload);
    }

    async sendToServer(payload) {
        if (!this.isOnline) {
            this.log('Offline - storing payload for later transmission');
            this.storeOffline(payload);
            return;
        }

        try {
            const response = await fetch(this.config.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            this.log('Errors sent to server successfully');
        } catch (error) {
            this.log('Failed to send errors to server:', error);
            this.storeOffline(payload);
        }
    }

    storeOffline(payload) {
        try {
            const stored = JSON.parse(localStorage.getItem('prometheus_errors') || '[]');
            stored.push(payload);
            localStorage.setItem('prometheus_errors', JSON.stringify(stored.slice(-100))); // Keep last 100
        } catch (error) {
            this.log('Failed to store errors offline:', error);
        }
    }

    retrieveOfflineErrors() {
        try {
            const stored = JSON.parse(localStorage.getItem('prometheus_errors') || '[]');
            localStorage.removeItem('prometheus_errors');
            return stored;
        } catch (error) {
            this.log('Failed to retrieve offline errors:', error);
            return [];
        }
    }

    startFlushTimer() {
        setInterval(() => {
            this.flush();
            
            // Send offline errors when back online
            if (this.isOnline) {
                const offlineErrors = this.retrieveOfflineErrors();
                offlineErrors.forEach(payload => this.sendToServer(payload));
            }
        }, this.config.flushInterval);
    }

    log(...args) {
        if (this.config.debugMode) {
            console.log('[PrometheusMonitor]', ...args);
        }
    }

    // Public API methods
    setUserId(userId) {
        this.config.userId = userId;
    }

    addCustomError(errorType, message, context = {}) {
        this.captureError({
            type: 'custom_' + errorType,
            message: message,
            context: context,
            timestamp: new Date().toISOString(),
            manual: true
        });
    }

    getSessionId() {
        return this.config.sessionId;
    }

    getErrorStats() {
        return {
            bufferedErrors: this.errorBuffer.length,
            totalCaptured: this.errorBuffer.length + (this.sentErrors || 0),
            isOnline: this.isOnline,
            lastHeartbeat: this.lastHeartbeat,
            sessionId: this.config.sessionId
        };
    }

    // Destroy method for cleanup
    destroy() {
        // Clear timers and remove event listeners
        this.flush(); // Send any remaining errors
        this.log('Prometheus Remote Monitor destroyed');
    }
}

// Auto-initialize if in browser environment
if (typeof window !== 'undefined') {
    // Initialize monitor when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.prometheusMonitor = new PrometheusRemoteMonitor({
                debugMode: true,
                enableRecovery: true
            });
        });
    } else {
        window.prometheusMonitor = new PrometheusRemoteMonitor({
            debugMode: true,
            enableRecovery: true
        });
    }
}

// Export for Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PrometheusRemoteMonitor;
}