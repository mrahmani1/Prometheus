/**
 * CDN Fallback System 2025
 * Provides offline fallbacks and redundant loading for critical libraries
 */
(function() {
    'use strict';
    console.log('🔄 CDN Fallback System initializing...');

    // CDN fallback configurations
    const CDN_FALLBACKS = {
        'three': [
            'https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js',
            'https://unpkg.com/three@0.160.0/build/three.min.js',
            'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js'
        ],
        'leaflet': [
            'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
            'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js',
            'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js'
        ],
        'd3': [
            'https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js',
            'https://unpkg.com/d3@7/dist/d3.min.js',
            'https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js'
        ],
        'chart': [
            'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js',
            'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js',
            'https://unpkg.com/chart.js@4.4.1/dist/chart.umd.js'
        ],
        'plotly': [
            'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.29.1/plotly.min.js',
            'https://cdn.plot.ly/plotly-2.29.1.min.js',
            'https://unpkg.com/plotly.js@2.29.1/dist/plotly.min.js'
        ],
        'tensorflow': [
            'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.1/dist/tf.min.js',
            'https://unpkg.com/@tensorflow/tfjs@4.15.1/dist/tf.min.js',
            'https://cdnjs.cloudflare.com/ajax/libs/tensorflow/4.15.1/tf.min.js'
        ]
    };

    // Library detection functions
    const LIBRARY_CHECKS = {
        'three': () => typeof THREE !== 'undefined',
        'leaflet': () => typeof L !== 'undefined',
        'd3': () => typeof d3 !== 'undefined',
        'chart': () => typeof Chart !== 'undefined',
        'plotly': () => typeof Plotly !== 'undefined',
        'tensorflow': () => typeof tf !== 'undefined'
    };

    // Simple offline fallbacks (basic functionality)
    const OFFLINE_FALLBACKS = {
        'three': function() {
            window.THREE = {
                Scene: function() { return { add: function() {}, remove: function() {} }; },
                PerspectiveCamera: function() { return { position: { set: function() {} } }; },
                WebGLRenderer: function() { 
                    return { 
                        setSize: function() {},
                        render: function() {},
                        domElement: document.createElement('canvas')
                    }; 
                },
                BoxGeometry: function() { return {}; },
                MeshBasicMaterial: function() { return {}; },
                Mesh: function() { return { position: { set: function() {} } }; }
            };
            console.log('🔧 THREE.js offline fallback activated');
        },
        'leaflet': function() {
            window.L = {
                map: function(id) {
                    return {
                        setView: function() { return this; },
                        addLayer: function() { return this; },
                        removeLayer: function() { return this; },
                        invalidateSize: function() { return this; },
                        getContainer: function() { return document.getElementById(id) || document.createElement('div'); }
                    };
                },
                tileLayer: function() {
                    return {
                        addTo: function() { return this; }
                    };
                },
                marker: function() {
                    return {
                        addTo: function() { return this; },
                        bindPopup: function() { return this; }
                    };
                }
            };
            console.log('🗺️ Leaflet offline fallback activated');
        },
        'd3': function() {
            window.d3 = {
                select: function() { 
                    return { 
                        selectAll: function() { return this; },
                        data: function() { return this; },
                        enter: function() { return this; },
                        append: function() { return this; },
                        attr: function() { return this; },
                        style: function() { return this; }
                    }; 
                },
                selectAll: function() { return this.select(); },
                scaleLinear: function() {
                    return {
                        domain: function() { return this; },
                        range: function() { return this; }
                    };
                }
            };
            console.log('📊 D3.js offline fallback activated');
        },
        'chart': function() {
            window.Chart = function() {
                return {
                    update: function() {},
                    destroy: function() {},
                    resize: function() {}
                };
            };
            window.Chart.register = function() {};
            console.log('📈 Chart.js offline fallback activated');
        },
        'plotly': function() {
            window.Plotly = {
                newPlot: function() { return Promise.resolve(); },
                restyle: function() { return Promise.resolve(); },
                relayout: function() { return Promise.resolve(); }
            };
            console.log('📉 Plotly offline fallback activated');
        },
        'tensorflow': function() {
            window.tf = {
                loadLayersModel: function() { return Promise.resolve({ predict: function() { return { dataSync: function() { return [0]; } }; } }); },
                tensor: function() { return { dataSync: function() { return [0]; } }; }
            };
            console.log('🤖 TensorFlow.js offline fallback activated');
        }
    };

    /**
     * Load script with fallback mechanism
     */
    function loadScriptWithFallback(library, fallbacks, callback) {
        let currentIndex = 0;

        function tryNextFallback() {
            if (currentIndex >= fallbacks.length) {
                console.warn(`❌ All CDN sources failed for ${library}, using offline fallback`);
                if (OFFLINE_FALLBACKS[library]) {
                    OFFLINE_FALLBACKS[library]();
                }
                if (callback) callback(false);
                return;
            }

            const script = document.createElement('script');
            script.src = fallbacks[currentIndex];
            script.timeout = 10000; // 10 second timeout

            script.onload = function() {
                console.log(`✅ Loaded ${library} from: ${fallbacks[currentIndex]}`);
                if (callback) callback(true);
            };

            script.onerror = function() {
                console.warn(`❌ Failed to load ${library} from: ${fallbacks[currentIndex]}`);
                currentIndex++;
                tryNextFallback();
            };

            // Add timeout handling
            const timeoutId = setTimeout(() => {
                console.warn(`⏰ Timeout loading ${library} from: ${fallbacks[currentIndex]}`);
                script.remove();
                currentIndex++;
                tryNextFallback();
            }, 10000);

            script.onload = function() {
                clearTimeout(timeoutId);
                console.log(`✅ Loaded ${library} from: ${fallbacks[currentIndex]}`);
                if (callback) callback(true);
            };

            document.head.appendChild(script);
        }

        tryNextFallback();
    }

    /**
     * Check and load missing libraries
     */
    function initializeFallbackSystem() {
        const librariesToLoad = [];

        // Check which libraries are missing
        for (const [library, checkFunction] of Object.entries(LIBRARY_CHECKS)) {
            if (!checkFunction()) {
                librariesToLoad.push(library);
            }
        }

        if (librariesToLoad.length === 0) {
            console.log('✅ All required libraries already loaded');
            return;
        }

        console.log(`🔄 Loading ${librariesToLoad.length} missing libraries:`, librariesToLoad);

        // Load missing libraries with fallbacks
        librariesToLoad.forEach(library => {
            if (CDN_FALLBACKS[library]) {
                loadScriptWithFallback(library, CDN_FALLBACKS[library]);
            }
        });
    }

    /**
     * Periodic health check for libraries
     */
    function startHealthCheck() {
        setInterval(() => {
            for (const [library, checkFunction] of Object.entries(LIBRARY_CHECKS)) {
                if (!checkFunction() && OFFLINE_FALLBACKS[library]) {
                    console.warn(`🔧 ${library} became unavailable, activating offline fallback`);
                    OFFLINE_FALLBACKS[library]();
                }
            }
        }, 30000); // Check every 30 seconds
    }

    // Initialize the system
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(initializeFallbackSystem, 1000);
            startHealthCheck();
        });
    } else {
        setTimeout(initializeFallbackSystem, 1000);
        startHealthCheck();
    }

    // Expose global function for manual activation
    window.activateCDNFallbacks = initializeFallbackSystem;

    console.log('✅ CDN Fallback System ready');
})();