/**
 * Suppress Source Map Warnings
 * Handles known source map issues with WebAssembly modules
 */

(function() {
    'use strict';
    
    // Store original console methods
    const originalWarn = console.warn;
    const originalError = console.error;
    
    // Filter for known harmless warnings
    const suppressedPatterns = [
        'Source map error',
        'Error: URL constructor',
        'WebAssembly.Module',
        'Source Map URL: null',
        'plotly-latest.min.js and plotly-latest.js are NO LONGER',
        'HOLOCENE_VOLCANOES not loaded, using fallback'
    ];
    
    // Override console.warn
    console.warn = function(...args) {
        const message = args.join(' ');
        for (const pattern of suppressedPatterns) {
            if (message.includes(pattern)) {
                // Log to debug instead of warn
                console.debug('[Suppressed Warning]', ...args);
                return;
            }
        }
        originalWarn.apply(console, args);
    };
    
    // Override console.error for source map errors
    console.error = function(...args) {
        const message = args.join(' ');
        if (message.includes('Source map error') || message.includes('URL constructor')) {
            console.debug('[Suppressed Error]', ...args);
            return;
        }
        originalError.apply(console, args);
    };
    
    // Also suppress the Firefox-specific source map warnings
    if (typeof window !== 'undefined') {
        const originalAddEventListener = window.addEventListener;
        window.addEventListener = function(type, listener, options) {
            if (type === 'error') {
                const wrappedListener = function(event) {
                    if (event.message && (
                        event.message.includes('Source map error') ||
                        event.message.includes('WebAssembly.Module')
                    )) {
                        console.debug('[Suppressed Error Event]', event.message);
                        event.preventDefault();
                        event.stopPropagation();
                        return false;
                    }
                    return listener.call(this, event);
                };
                return originalAddEventListener.call(this, type, wrappedListener, options);
            }
            return originalAddEventListener.call(this, type, listener, options);
        };
    }
    
    console.log('🔇 Source map warning suppression active');
})();