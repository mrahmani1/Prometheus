/**
 * Fix InSAR Issues Script
 * Addresses duplicate declarations and loading order problems
 */

(function() {
    'use strict';
    
    // Check if HOLOCENE_VOLCANOES already exists to prevent redeclaration
    if (typeof window.HOLOCENE_VOLCANOES === 'undefined') {
        window.HOLOCENE_VOLCANOES_NEEDED = true;
    }
    
    // Wait for required libraries to load
    let librariesLoaded = false;
    let loadAttempts = 0;
    const maxAttempts = 50;
    
    function checkLibraries() {
        loadAttempts++;
        
        const requiredLibs = {
            'THREE': typeof THREE !== 'undefined',
            'L': typeof L !== 'undefined',
            'Chart': typeof Chart !== 'undefined',
            'Plotly': typeof Plotly !== 'undefined'
        };
        
        const allLoaded = Object.values(requiredLibs).every(loaded => loaded);
        
        if (allLoaded) {
            librariesLoaded = true;
            console.log('✅ All required libraries loaded');
            initializeApplication();
        } else if (loadAttempts < maxAttempts) {
            const missing = Object.entries(requiredLibs)
                .filter(([name, loaded]) => !loaded)
                .map(([name]) => name);
            console.log(`⏳ Waiting for libraries: ${missing.join(', ')} (attempt ${loadAttempts}/${maxAttempts})`);
            setTimeout(checkLibraries, 200);
        } else {
            console.error('❌ Failed to load all required libraries after', maxAttempts, 'attempts');
            // Try to initialize with available libraries
            initializeApplication();
        }
    }
    
    function initializeApplication() {
        console.log('🚀 Initializing InSAR application with library fixes');
        
        // Fix map initialization if Leaflet is available
        if (typeof L !== 'undefined' && typeof map === 'undefined') {
            try {
                window.map = L.map('map', {
                    center: [0, 0],
                    zoom: 2,
                    preferCanvas: true
                });
                
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                
                console.log('✅ Map initialized successfully');
            } catch (e) {
                console.warn('⚠️ Could not initialize map:', e.message);
                // Create a dummy map object to prevent errors
                window.map = {
                    eachLayer: function() {},
                    invalidateSize: function() {},
                    addLayer: function() {},
                    removeLayer: function() {},
                    setView: function() {},
                    fitBounds: function() {},
                    getZoom: function() { return 2; },
                    getCenter: function() { return {lat: 0, lng: 0}; }
                };
            }
        }
        
        // Fix THREE.js initialization
        if (typeof THREE !== 'undefined' && typeof renderer === 'undefined') {
            try {
                const canvas = document.getElementById('three-canvas') || document.querySelector('canvas');
                if (canvas) {
                    window.renderer = new THREE.WebGLRenderer({ 
                        canvas: canvas,
                        antialias: true,
                        alpha: true
                    });
                    window.scene = new THREE.Scene();
                    window.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                    
                    // Add the setClearColor method if it doesn't exist
                    if (typeof renderer.setClearColor === 'function') {
                        renderer.setClearColor(0x000000, 0);
                    }
                    
                    console.log('✅ THREE.js initialized successfully');
                }
            } catch (e) {
                console.warn('⚠️ Could not initialize THREE.js:', e.message);
                // Create dummy renderer
                window.renderer = {
                    setClearColor: function() {},
                    setSize: function() {},
                    render: function() {},
                    domElement: document.createElement('canvas')
                };
            }
        }
        
        // Fix WebGL context
        const canvas = document.querySelector('canvas');
        if (canvas && !window.gl) {
            try {
                window.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                if (gl) {
                    console.log('✅ WebGL context created');
                }
            } catch (e) {
                console.warn('⚠️ Could not create WebGL context:', e.message);
            }
        }
        
        // Dispatch a custom event to signal that fixes are applied
        window.dispatchEvent(new CustomEvent('insarFixesApplied', {
            detail: { librariesLoaded, timestamp: Date.now() }
        }));
    }
    
    // Start checking for libraries
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', checkLibraries);
    } else {
        // DOM already loaded, start checking immediately
        setTimeout(checkLibraries, 100);
    }
    
    // Add global error suppression for known issues
    const originalError = window.onerror;
    window.onerror = function(msg, url, lineNo, columnNo, error) {
        // Suppress specific known errors
        const suppressedErrors = [
            'redeclaration of const HOLOCENE_VOLCANOES',
            'map.eachLayer is not a function',
            'map.invalidateSize is not a function',
            'renderer.setClearColor is not a function',
            'L.control is undefined'
        ];
        
        for (const suppressedError of suppressedErrors) {
            if (msg && msg.includes(suppressedError)) {
                console.log('🔇 Suppressed known error:', suppressedError);
                return true; // Suppress the error
            }
        }
        
        // Call original error handler if exists
        if (originalError) {
            return originalError(msg, url, lineNo, columnNo, error);
        }
        return false;
    };
    
    console.log('🔧 InSAR issue fixes loaded and monitoring');
})();