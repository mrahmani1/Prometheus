// Emergency CDN Fix for Blocked Resources
// Addresses MIME type mismatch and CloudFlare access issues

(function() {
    'use strict';
    
    console.log('🚨 Emergency CDN Fix loading...');
    
    // Track failed CDN loads
    const failedCDNs = new Set();
    const loadedLibraries = new Set();
    
    // Alternative CDN sources for critical libraries
    const CDN_ALTERNATIVES = {
        'three.js': [
            'https://unpkg.com/three@0.160.1/build/three.min.js',
            'https://cdn.skypack.dev/three@0.160.1',
            'https://esm.run/three@0.160.1'
        ],
        'tensorflow': [
            'https://unpkg.com/@tensorflow/tfjs@4.15.1/dist/tf.min.js',
            'https://cdn.skypack.dev/@tensorflow/tfjs@4.15.1',
            'https://esm.run/@tensorflow/tfjs@4.15.1'
        ],
        'chart.js': [
            'https://unpkg.com/chart.js@4.4.1/dist/chart.umd.js',
            'https://cdn.skypack.dev/chart.js@4.4.1',
            'https://esm.run/chart.js@4.4.1'
        ],
        'leaflet': [
            'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
            'https://cdn.skypack.dev/leaflet@1.9.4'
        ]
    };
    
    // Function to load script with multiple fallbacks
    function loadScriptWithFallbacks(libraryName, urls, callback) {
        if (loadedLibraries.has(libraryName)) {
            callback(null);
            return;
        }
        
        let urlIndex = 0;
        
        function tryNextUrl() {
            if (urlIndex >= urls.length) {
                console.error(`❌ All CDN sources failed for ${libraryName}`);
                callback(new Error(`Failed to load ${libraryName}`));
                return;
            }
            
            const script = document.createElement('script');
            const url = urls[urlIndex];
            
            script.onload = function() {
                console.log(`✅ Successfully loaded ${libraryName} from: ${url}`);
                loadedLibraries.add(libraryName);
                callback(null);
            };
            
            script.onerror = function() {
                console.warn(`⚠️ Failed to load ${libraryName} from: ${url}`);
                failedCDNs.add(url);
                urlIndex++;
                setTimeout(tryNextUrl, 100); // Brief delay before trying next URL
            };
            
            script.src = url;
            document.head.appendChild(script);
        }
        
        tryNextUrl();
    }
    
    // Emergency library loader
    function emergencyLoadLibraries() {
        const loadPromises = [];
        
        // Check if THREE.js is missing
        if (typeof THREE === 'undefined') {
            loadPromises.push(new Promise(resolve => {
                loadScriptWithFallbacks('three.js', CDN_ALTERNATIVES['three.js'], resolve);
            }));
        }
        
        // Check if TensorFlow is missing
        if (typeof tf === 'undefined') {
            loadPromises.push(new Promise(resolve => {
                loadScriptWithFallbacks('tensorflow', CDN_ALTERNATIVES['tensorflow'], resolve);
            }));
        }
        
        // Check if Chart.js is missing
        if (typeof Chart === 'undefined') {
            loadPromises.push(new Promise(resolve => {
                loadScriptWithFallbacks('chart.js', CDN_ALTERNATIVES['chart.js'], resolve);
            }));
        }
        
        // Check if Leaflet is missing
        if (typeof L === 'undefined') {
            loadPromises.push(new Promise(resolve => {
                loadScriptWithFallbacks('leaflet', CDN_ALTERNATIVES['leaflet'], resolve);
            }));
        }
        
        Promise.allSettled(loadPromises).then(results => {
            console.log('📊 Emergency CDN loading results:', results);
            
            // Trigger reinitialization after libraries load
            setTimeout(() => {
                if (typeof reinitializeFailedSystems === 'function') {
                    reinitializeFailedSystems();
                }
            }, 1000);
        });
    }
    
    // Global reinitialization function
    window.reinitializeFailedSystems = function() {
        console.log('🔄 Reinitializing failed systems...');
        
        // Reinitialize THREE.js scenes if available
        if (typeof THREE !== 'undefined') {
            try {
                if (typeof initializeWebGL === 'function') {
                    initializeWebGL();
                }
            } catch (e) {
                console.warn('WebGL reinitialization failed:', e);
            }
        }
        
        // Reinitialize Leaflet maps if available
        if (typeof L !== 'undefined') {
            try {
                if (typeof initialize2DMap === 'function') {
                    initialize2DMap();
                }
            } catch (e) {
                console.warn('Leaflet reinitialization failed:', e);
            }
        }
        
        // Reinitialize charts if available
        if (typeof Chart !== 'undefined') {
            try {
                if (typeof initializeCharts === 'function') {
                    initializeCharts();
                }
            } catch (e) {
                console.warn('Charts reinitialization failed:', e);
            }
        }
    };
    
    // Override console.error to catch CDN loading failures
    const originalError = console.error;
    console.error = function(...args) {
        const message = args.join(' ');
        if (message.includes('MIME type') || message.includes('blocked')) {
            console.log('🚨 CDN loading issue detected, triggering emergency loader...');
            setTimeout(emergencyLoadLibraries, 100);
        }
        originalError.apply(console, args);
    };
    
    // Start emergency loading immediately
    emergencyLoadLibraries();
    
    // Add global selectVolcano function if missing
    if (typeof window.selectVolcano === 'undefined') {
        window.selectVolcano = function(volcanoId) {
            console.log('🌋 Emergency selectVolcano called for:', volcanoId);
            
            // Try to use comprehensive volcano selector if available
            if (window.initializeVolcanoSelector) {
                try {
                    // Find the volcano in our comprehensive database
                    const volcano = findVolcanoById(volcanoId);
                    if (volcano) {
                        console.log('Found volcano:', volcano);
                        // Trigger volcano selection
                        if (typeof updateVolcanoDisplay === 'function') {
                            updateVolcanoDisplay(volcano);
                        }
                    }
                } catch (e) {
                    console.warn('Volcano selection failed:', e);
                }
            }
        };
    }
    
    // Helper function to find volcano by ID
    function findVolcanoById(volcanoId) {
        if (typeof VOLCANIC_DATABASE !== 'undefined') {
            for (const continent of Object.values(VOLCANIC_DATABASE)) {
                for (const country of Object.values(continent)) {
                    for (const region of Object.values(country)) {
                        if (Array.isArray(region)) {
                            const found = region.find(v => 
                                v.toLowerCase().includes(volcanoId.toLowerCase()) ||
                                volcanoId.toLowerCase().includes(v.toLowerCase())
                            );
                            if (found) return found;
                        }
                    }
                }
            }
        }
        return null;
    }
    
    console.log('✅ Emergency CDN Fix initialized');
})();