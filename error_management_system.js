/**
 * Error Management System - Task 1.5
 * Comprehensive error boundary system with stack trace visualization and categorization
 * @author Prometheus Platform
 * @version 2.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🚨 Error Management System - Initializing comprehensive error handling...');
    
    // Error categories and their properties
    const ERROR_CATEGORIES = {
        CRITICAL: { color: '#ff0000', icon: '🔴', priority: 1 },
        WARNING: { color: '#ff9800', icon: '🟠', priority: 2 },
        INFO: { color: '#2196f3', icon: '🔵', priority: 3 },
        DEBUG: { color: '#9e9e9e', icon: '⚪', priority: 4 },
        NETWORK: { color: '#e91e63', icon: '🌐', priority: 2 },
        WEBGL: { color: '#9c27b0', icon: '🎮', priority: 1 },
        SCRIPT: { color: '#ff5722', icon: '📜', priority: 1 },
        PERFORMANCE: { color: '#00bcd4', icon: '⚡', priority: 3 }
    };
    
    // Main Error Management System
    class ErrorManagementSystem {
        constructor() {
            this.errors = [];
            this.maxErrors = 1000;
            this.listeners = new Set();
            this.categories = new Map();
            this.filters = new Set();
            this.isPaused = false;
            this.exportFormats = ['json', 'csv', 'txt', 'html'];
            
            // Initialize categories
            Object.keys(ERROR_CATEGORIES).forEach(cat => {
                this.categories.set(cat, []);
            });
            
            // Setup global error handlers
            this.setupGlobalHandlers();
            
            // Create UI if on error viewer page
            if (window.location.pathname.includes('error_viewer')) {
                this.createErrorViewerUI();
            }
            
            // Create floating error indicator
            this.createFloatingIndicator();
        }
        
        setupGlobalHandlers() {
            // Window error handler
            const originalError = window.onerror;
            window.onerror = (message, source, lineno, colno, error) => {
                this.captureError({
                    type: 'SCRIPT',
                    message,
                    source,
                    lineno,
                    colno,
                    stack: error?.stack,
                    timestamp: Date.now()
                });
                
                if (originalError) {
                    return originalError(message, source, lineno, colno, error);
                }
                return true;
            };
            
            // Promise rejection handler
            window.addEventListener('unhandledrejection', (event) => {
                this.captureError({
                    type: 'CRITICAL',
                    message: 'Unhandled Promise Rejection',
                    reason: event.reason,
                    promise: event.promise,
                    stack: event.reason?.stack,
                    timestamp: Date.now()
                });
            });
            
            // Console error interceptor
            const originalConsoleError = console.error;
            console.error = (...args) => {
                this.captureError({
                    type: this.categorizeError(args[0]),
                    message: args.join(' '),
                    source: 'console.error',
                    timestamp: Date.now(),
                    stack: new Error().stack
                });
                originalConsoleError.apply(console, args);
            };
            
            // Network error handler
            this.setupNetworkErrorHandler();
            
            // Performance observer
            this.setupPerformanceObserver();
        }
        
        setupNetworkErrorHandler() {
            // Intercept fetch
            const originalFetch = window.fetch;
            window.fetch = async (...args) => {
                try {
                    const response = await originalFetch(...args);
                    if (!response.ok) {
                        this.captureError({
                            type: 'NETWORK',
                            message: `HTTP ${response.status}: ${response.statusText}`,
                            url: args[0],
                            status: response.status,
                            timestamp: Date.now()
                        });
                    }
                    return response;
                } catch (error) {
                    this.captureError({
                        type: 'NETWORK',
                        message: error.message,
                        url: args[0],
                        timestamp: Date.now()
                    });
                    throw error;
                }
            };
            
            // XMLHttpRequest interceptor
            const originalOpen = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(...args) {
                this.addEventListener('error', () => {
                    ErrorManager.captureError({
                        type: 'NETWORK',
                        message: 'XMLHttpRequest failed',
                        url: args[1],
                        method: args[0],
                        timestamp: Date.now()
                    });
                });
                return originalOpen.apply(this, args);
            };
        }
        
        setupPerformanceObserver() {
            if (!window.PerformanceObserver) return;
            
            try {
                // Long task observer
                const longTaskObserver = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (entry.duration > 50) {
                            this.captureError({
                                type: 'PERFORMANCE',
                                message: `Long task detected: ${entry.duration.toFixed(2)}ms`,
                                duration: entry.duration,
                                timestamp: Date.now()
                            });
                        }
                    }
                });
                
                longTaskObserver.observe({ entryTypes: ['longtask'] });
            } catch (e) {
                console.warn('Performance observer setup failed:', e);
            }
        }
        
        categorizeError(message) {
            if (typeof message !== 'string') return 'INFO';
            
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('webgl') || lowerMessage.includes('three')) {
                return 'WEBGL';
            }
            if (lowerMessage.includes('network') || lowerMessage.includes('fetch')) {
                return 'NETWORK';
            }
            if (lowerMessage.includes('performance') || lowerMessage.includes('memory')) {
                return 'PERFORMANCE';
            }
            if (lowerMessage.includes('error') || lowerMessage.includes('exception')) {
                return 'CRITICAL';
            }
            if (lowerMessage.includes('warning') || lowerMessage.includes('warn')) {
                return 'WARNING';
            }
            
            return 'INFO';
        }
        
        captureError(errorData) {
            if (this.isPaused) return;
            
            // Add unique ID
            errorData.id = Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            
            // Add browser info
            errorData.userAgent = navigator.userAgent;
            errorData.url = window.location.href;
            
            // Parse stack trace if available
            if (errorData.stack) {
                errorData.parsedStack = this.parseStackTrace(errorData.stack);
            }
            
            // Add to main list
            this.errors.unshift(errorData);
            if (this.errors.length > this.maxErrors) {
                this.errors.pop();
            }
            
            // Add to category
            const category = errorData.type || 'INFO';
            if (this.categories.has(category)) {
                this.categories.get(category).unshift(errorData);
            }
            
            // Notify listeners
            this.notifyListeners(errorData);
            
            // Update UI
            this.updateUI(errorData);
        }
        
        parseStackTrace(stack) {
            if (!stack) return [];
            
            const lines = stack.split('\n');
            const parsed = [];
            
            for (const line of lines) {
                const match = line.match(/at\s+(.+?)\s+\((.+?):(\d+):(\d+)\)/);
                if (match) {
                    parsed.push({
                        function: match[1],
                        file: match[2],
                        line: parseInt(match[3]),
                        column: parseInt(match[4])
                    });
                } else {
                    const simpleMatch = line.match(/at\s+(.+?):(\d+):(\d+)/);
                    if (simpleMatch) {
                        parsed.push({
                            function: 'anonymous',
                            file: simpleMatch[1],
                            line: parseInt(simpleMatch[2]),
                            column: parseInt(simpleMatch[3])
                        });
                    }
                }
            }
            
            return parsed;
        }
        
        createErrorViewerUI() {
            console.log('Creating Error Viewer UI...');
            
            // Find or create container
            let container = document.getElementById('error-viewer-container');
            if (!container) {
                container = document.createElement('div');
                container.id = 'error-viewer-container';
                document.body.appendChild(container);
            }
            
            container.innerHTML = `
                <div class="error-viewer">
                    <div class="viewer-header">
                        <h2>🚨 Error Management System</h2>
                        <div class="viewer-stats">
                            <span class="stat-item">
                                <span class="stat-label">Total:</span>
                                <span class="stat-value" id="total-errors">0</span>
                            </span>
                            <span class="stat-item">
                                <span class="stat-label">Critical:</span>
                                <span class="stat-value critical" id="critical-errors">0</span>
                            </span>
                            <span class="stat-item">
                                <span class="stat-label">Warnings:</span>
                                <span class="stat-value warning" id="warning-errors">0</span>
                            </span>
                        </div>
                    </div>
                    
                    <div class="viewer-controls">
                        <div class="control-group">
                            <button id="clear-errors" class="btn">🗑️ Clear All</button>
                            <button id="pause-capture" class="btn">⏸️ Pause</button>
                            <button id="export-errors" class="btn">💾 Export</button>
                        </div>
                        
                        <div class="control-group">
                            <input type="text" id="error-search" placeholder="Search errors..." class="search-input">
                            <select id="category-filter" class="filter-select">
                                <option value="">All Categories</option>
                                ${Object.keys(ERROR_CATEGORIES).map(cat => 
                                    `<option value="${cat}">${ERROR_CATEGORIES[cat].icon} ${cat}</option>`
                                ).join('')}
                            </select>
                        </div>
                    </div>
                    
                    <div class="error-list" id="error-list">
                        <!-- Errors will be populated here -->
                    </div>
                    
                    <div class="error-details" id="error-details" style="display: none;">
                        <div class="details-header">
                            <h3>Error Details</h3>
                            <button class="close-btn" onclick="ErrorManager.closeDetails()">✕</button>
                        </div>
                        <div class="details-content" id="details-content">
                            <!-- Details will be shown here -->
                        </div>
                    </div>
                </div>
            `;
            
            // Add styles
            this.addErrorViewerStyles();
            
            // Setup event listeners
            this.setupUIEventListeners();
            
            // Initial update
            this.updateErrorList();
        }
        
        addErrorViewerStyles() {
            if (document.getElementById('error-viewer-styles')) return;
            
            const styles = document.createElement('style');
            styles.id = 'error-viewer-styles';
            styles.textContent = `
                .error-viewer {
                    background: rgba(0, 0, 0, 0.95);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    font-family: 'Monaco', 'Menlo', monospace;
                    min-height: 600px;
                }
                
                .viewer-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #333;
                }
                
                .viewer-stats {
                    display: flex;
                    gap: 20px;
                }
                
                .stat-item {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }
                
                .stat-label {
                    color: #888;
                    font-size: 12px;
                }
                
                .stat-value {
                    font-size: 18px;
                    font-weight: bold;
                }
                
                .stat-value.critical {
                    color: #ff0000;
                }
                
                .stat-value.warning {
                    color: #ff9800;
                }
                
                .viewer-controls {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                    gap: 15px;
                    flex-wrap: wrap;
                }
                
                .control-group {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                
                .btn {
                    padding: 8px 15px;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    color: white;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .btn:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
                
                .search-input {
                    padding: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    color: white;
                    border-radius: 5px;
                    width: 200px;
                }
                
                .filter-select {
                    padding: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    color: white;
                    border-radius: 5px;
                }
                
                .filter-select option {
                    background: #1a1a1a;
                }
                
                .error-list {
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #333;
                    border-radius: 5px;
                    padding: 10px;
                }
                
                .error-item {
                    padding: 10px;
                    margin: 5px 0;
                    background: rgba(255, 255, 255, 0.05);
                    border-left: 3px solid #666;
                    border-radius: 3px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .error-item:hover {
                    background: rgba(255, 255, 255, 0.1);
                    transform: translateX(5px);
                }
                
                .error-item.critical {
                    border-left-color: #ff0000;
                }
                
                .error-item.warning {
                    border-left-color: #ff9800;
                }
                
                .error-item.network {
                    border-left-color: #e91e63;
                }
                
                .error-item.webgl {
                    border-left-color: #9c27b0;
                }
                
                .error-header-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 5px;
                }
                
                .error-category {
                    display: inline-block;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
                
                .error-time {
                    font-size: 11px;
                    color: #888;
                }
                
                .error-message {
                    color: #ffc107;
                    margin-bottom: 5px;
                    word-break: break-word;
                }
                
                .error-source {
                    font-size: 11px;
                    color: #888;
                }
                
                .error-details {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(0, 0, 0, 0.98);
                    border: 2px solid #333;
                    border-radius: 10px;
                    padding: 20px;
                    max-width: 800px;
                    max-height: 600px;
                    overflow-y: auto;
                    z-index: 10000;
                }
                
                .details-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #333;
                }
                
                .close-btn {
                    background: transparent;
                    border: none;
                    color: white;
                    font-size: 20px;
                    cursor: pointer;
                }
                
                .stack-trace {
                    background: rgba(0, 0, 0, 0.5);
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 12px;
                    overflow-x: auto;
                }
                
                .stack-frame {
                    margin: 5px 0;
                    padding: 5px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 3px;
                }
                
                .floating-indicator {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: rgba(0, 0, 0, 0.9);
                    border: 1px solid #333;
                    border-radius: 50px;
                    padding: 10px 15px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    cursor: pointer;
                    z-index: 9999;
                    transition: all 0.3s;
                }
                
                .floating-indicator:hover {
                    transform: scale(1.05);
                    box-shadow: 0 5px 15px rgba(255, 0, 0, 0.3);
                }
                
                .error-count-badge {
                    background: #ff0000;
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    font-weight: bold;
                }
                
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }
                
                .pulse {
                    animation: pulse 0.5s;
                }
            `;
            document.head.appendChild(styles);
        }
        
        setupUIEventListeners() {
            // Clear button
            document.getElementById('clear-errors')?.addEventListener('click', () => {
                this.clearErrors();
            });
            
            // Pause button
            document.getElementById('pause-capture')?.addEventListener('click', (e) => {
                this.togglePause();
                e.target.textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
            });
            
            // Export button
            document.getElementById('export-errors')?.addEventListener('click', () => {
                this.showExportDialog();
            });
            
            // Search
            document.getElementById('error-search')?.addEventListener('input', (e) => {
                this.filterErrors(e.target.value);
            });
            
            // Category filter
            document.getElementById('category-filter')?.addEventListener('change', (e) => {
                this.filterByCategory(e.target.value);
            });
        }
        
        updateUI(error) {
            // Update stats
            this.updateStats();
            
            // Add to list if visible
            if (document.getElementById('error-list')) {
                this.addErrorToList(error);
            }
            
            // Update floating indicator
            this.updateFloatingIndicator();
        }
        
        updateStats() {
            const total = this.errors.length;
            const critical = this.errors.filter(e => e.type === 'CRITICAL').length;
            const warnings = this.errors.filter(e => e.type === 'WARNING').length;
            
            const totalEl = document.getElementById('total-errors');
            const criticalEl = document.getElementById('critical-errors');
            const warningEl = document.getElementById('warning-errors');
            
            if (totalEl) totalEl.textContent = total;
            if (criticalEl) criticalEl.textContent = critical;
            if (warningEl) warningEl.textContent = warnings;
        }
        
        addErrorToList(error) {
            const list = document.getElementById('error-list');
            if (!list) return;
            
            const item = document.createElement('div');
            item.className = `error-item ${error.type.toLowerCase()}`;
            item.dataset.errorId = error.id;
            
            const category = ERROR_CATEGORIES[error.type] || ERROR_CATEGORIES.INFO;
            
            item.innerHTML = `
                <div class="error-header-row">
                    <span class="error-category" style="background: ${category.color}20; color: ${category.color}">
                        ${category.icon} ${error.type}
                    </span>
                    <span class="error-time">${new Date(error.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="error-message">${this.escapeHtml(error.message || 'No message')}</div>
                <div class="error-source">${this.escapeHtml(error.source || error.url || 'Unknown source')}</div>
            `;
            
            item.addEventListener('click', () => {
                this.showErrorDetails(error);
            });
            
            // Add to top of list
            list.insertBefore(item, list.firstChild);
            
            // Limit displayed items
            while (list.children.length > 100) {
                list.removeChild(list.lastChild);
            }
        }
        
        showErrorDetails(error) {
            const details = document.getElementById('error-details');
            const content = document.getElementById('details-content');
            
            if (!details || !content) return;
            
            const category = ERROR_CATEGORIES[error.type] || ERROR_CATEGORIES.INFO;
            
            content.innerHTML = `
                <div class="detail-section">
                    <h4>General Information</h4>
                    <table class="detail-table">
                        <tr><td>Type:</td><td><span style="color: ${category.color}">${category.icon} ${error.type}</span></td></tr>
                        <tr><td>Time:</td><td>${new Date(error.timestamp).toLocaleString()}</td></tr>
                        <tr><td>Message:</td><td>${this.escapeHtml(error.message || 'N/A')}</td></tr>
                        <tr><td>Source:</td><td>${this.escapeHtml(error.source || 'N/A')}</td></tr>
                        <tr><td>URL:</td><td>${this.escapeHtml(error.url || 'N/A')}</td></tr>
                    </table>
                </div>
                
                ${error.parsedStack ? `
                    <div class="detail-section">
                        <h4>Stack Trace</h4>
                        <div class="stack-trace">
                            ${error.parsedStack.map(frame => `
                                <div class="stack-frame">
                                    ${this.escapeHtml(frame.function)} at ${this.escapeHtml(frame.file)}:${frame.line}:${frame.column}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${error.stack && !error.parsedStack ? `
                    <div class="detail-section">
                        <h4>Raw Stack Trace</h4>
                        <pre class="stack-trace">${this.escapeHtml(error.stack)}</pre>
                    </div>
                ` : ''}
                
                <div class="detail-section">
                    <h4>Browser Information</h4>
                    <table class="detail-table">
                        <tr><td>User Agent:</td><td>${this.escapeHtml(error.userAgent || navigator.userAgent)}</td></tr>
                        <tr><td>Platform:</td><td>${navigator.platform}</td></tr>
                        <tr><td>Language:</td><td>${navigator.language}</td></tr>
                    </table>
                </div>
            `;
            
            details.style.display = 'block';
        }
        
        closeDetails() {
            const details = document.getElementById('error-details');
            if (details) {
                details.style.display = 'none';
            }
        }
        
        createFloatingIndicator() {
            // Don't create on error viewer page
            if (window.location.pathname.includes('error_viewer')) return;
            
            const indicator = document.createElement('div');
            indicator.id = 'error-indicator';
            indicator.className = 'floating-indicator';
            indicator.innerHTML = `
                <span>🚨</span>
                <span class="error-count-badge">0</span>
                <span>Errors</span>
            `;
            
            indicator.addEventListener('click', () => {
                // Open error viewer in new window or navigate
                if (window.location.pathname.includes('error_viewer')) {
                    this.createErrorViewerUI();
                } else {
                    window.open('/error_viewer.html', '_blank');
                }
            });
            
            document.body.appendChild(indicator);
        }
        
        updateFloatingIndicator() {
            const badge = document.querySelector('.error-count-badge');
            if (badge) {
                const count = this.errors.length;
                badge.textContent = count > 99 ? '99+' : count;
                badge.classList.add('pulse');
                setTimeout(() => badge.classList.remove('pulse'), 500);
            }
        }
        
        updateErrorList() {
            const list = document.getElementById('error-list');
            if (!list) return;
            
            list.innerHTML = '';
            
            const displayErrors = this.filters.size > 0 ? 
                this.errors.filter(e => !this.filters.has(e.type)) : 
                this.errors;
            
            displayErrors.slice(0, 100).forEach(error => {
                this.addErrorToList(error);
            });
        }
        
        filterErrors(searchTerm) {
            const items = document.querySelectorAll('.error-item');
            const lower = searchTerm.toLowerCase();
            
            items.forEach(item => {
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(lower) ? '' : 'none';
            });
        }
        
        filterByCategory(category) {
            const items = document.querySelectorAll('.error-item');
            
            items.forEach(item => {
                if (category === '' || item.classList.contains(category.toLowerCase())) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        }
        
        clearErrors() {
            this.errors = [];
            this.categories.forEach(cat => cat.length = 0);
            this.updateErrorList();
            this.updateStats();
            this.updateFloatingIndicator();
        }
        
        togglePause() {
            this.isPaused = !this.isPaused;
            console.log(`Error capture ${this.isPaused ? 'paused' : 'resumed'}`);
        }
        
        showExportDialog() {
            const dialog = document.createElement('div');
            dialog.className = 'export-dialog';
            dialog.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.95);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #333;
                z-index: 10001;
            `;
            
            dialog.innerHTML = `
                <h3>Export Errors</h3>
                <div style="margin: 15px 0;">
                    <label>Format:</label>
                    <select id="export-format" style="margin-left: 10px;">
                        <option value="json">JSON</option>
                        <option value="csv">CSV</option>
                        <option value="txt">Text</option>
                        <option value="html">HTML Report</option>
                    </select>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button onclick="ErrorManager.doExport()" class="btn">Export</button>
                    <button onclick="this.parentElement.parentElement.remove()" class="btn">Cancel</button>
                </div>
            `;
            
            document.body.appendChild(dialog);
        }
        
        doExport() {
            const format = document.getElementById('export-format')?.value || 'json';
            const data = this.exportData(format);
            const blob = new Blob([data], { type: this.getMimeType(format) });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `errors_${Date.now()}.${format}`;
            a.click();
            URL.revokeObjectURL(url);
            
            // Close dialog
            document.querySelector('.export-dialog')?.remove();
        }
        
        exportData(format) {
            switch (format) {
                case 'json':
                    return JSON.stringify(this.errors, null, 2);
                    
                case 'csv':
                    const headers = ['Time', 'Type', 'Message', 'Source', 'URL'];
                    const rows = this.errors.map(e => [
                        new Date(e.timestamp).toISOString(),
                        e.type,
                        (e.message || '').replace(/"/g, '""'),
                        (e.source || '').replace(/"/g, '""'),
                        e.url || ''
                    ]);
                    return [headers, ...rows].map(row => 
                        row.map(cell => `"${cell}"`).join(',')
                    ).join('\n');
                    
                case 'txt':
                    return this.errors.map(e => 
                        `[${new Date(e.timestamp).toLocaleString()}] ${e.type}: ${e.message}\n  Source: ${e.source || 'N/A'}\n`
                    ).join('\n');
                    
                case 'html':
                    return this.generateHTMLReport();
                    
                default:
                    return JSON.stringify(this.errors);
            }
        }
        
        generateHTMLReport() {
            const stats = {
                total: this.errors.length,
                critical: this.errors.filter(e => e.type === 'CRITICAL').length,
                warnings: this.errors.filter(e => e.type === 'WARNING').length
            };
            
            return `
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report - ${new Date().toLocaleString()}</title>
                    <style>
                        body { font-family: Arial, sans-serif; background: #1a1a1a; color: white; padding: 20px; }
                        h1 { color: #ff6b35; }
                        .stats { display: flex; gap: 20px; margin: 20px 0; }
                        .stat { padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; }
                        .error { margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.05); border-left: 3px solid #666; }
                        .error.critical { border-left-color: #ff0000; }
                        .error.warning { border-left-color: #ff9800; }
                    </style>
                </head>
                <body>
                    <h1>Error Report</h1>
                    <div class="stats">
                        <div class="stat">Total Errors: ${stats.total}</div>
                        <div class="stat">Critical: ${stats.critical}</div>
                        <div class="stat">Warnings: ${stats.warnings}</div>
                    </div>
                    <div class="errors">
                        ${this.errors.map(e => `
                            <div class="error ${e.type.toLowerCase()}">
                                <strong>[${new Date(e.timestamp).toLocaleString()}] ${e.type}</strong><br>
                                ${this.escapeHtml(e.message || 'No message')}<br>
                                <small>Source: ${this.escapeHtml(e.source || 'Unknown')}</small>
                            </div>
                        `).join('')}
                    </div>
                </body>
                </html>
            `;
        }
        
        getMimeType(format) {
            const types = {
                json: 'application/json',
                csv: 'text/csv',
                txt: 'text/plain',
                html: 'text/html'
            };
            return types[format] || 'text/plain';
        }
        
        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        subscribe(callback) {
            this.listeners.add(callback);
        }
        
        unsubscribe(callback) {
            this.listeners.delete(callback);
        }
        
        notifyListeners(error) {
            this.listeners.forEach(callback => {
                try {
                    callback(error);
                } catch (e) {
                    console.warn('Error in listener:', e);
                }
            });
        }
        
        getErrors(category = null) {
            if (category && this.categories.has(category)) {
                return this.categories.get(category);
            }
            return this.errors;
        }
        
        getStats() {
            const stats = {};
            this.categories.forEach((errors, category) => {
                stats[category] = errors.length;
            });
            stats.total = this.errors.length;
            return stats;
        }
    }
    
    // Initialize and expose globally
    const ErrorManager = new ErrorManagementSystem();
    
    // Public API
    global.ErrorManager = ErrorManager;
    global.captureError = ErrorManager.captureError.bind(ErrorManager);
    global.getErrorStats = ErrorManager.getStats.bind(ErrorManager);
    
    console.log('✅ Error Management System initialized. Access via window.ErrorManager');
    
})(window);