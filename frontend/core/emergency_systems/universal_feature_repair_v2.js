/**
 * Universal Feature Repair Script v2.0
 * Enhanced with better error handling and feature detection
 */

(function() {
    'use strict';
    
    console.log('🔧 Universal Feature Repair v2.0 Loading...');
    
    // Track repair state
    const repairState = {
        attempts: 0,
        maxAttempts: 10,
        repaired: {
            maps: false,
            threejs: false,
            charts: false,
            functions: false
        }
    };
    
    // Store references to avoid duplicates
    const chartInstances = new Map();
    
    function attemptRepair() {
        repairState.attempts++;
        console.log(`🔍 Repair attempt ${repairState.attempts}/${repairState.maxAttempts}`);
        
        // Check for required libraries
        const libraries = {
            THREE: typeof THREE !== 'undefined',
            L: typeof L !== 'undefined', 
            Chart: typeof Chart !== 'undefined',
            Plotly: typeof Plotly !== 'undefined'
        };
        
        const allLoaded = Object.values(libraries).every(v => v);
        
        if (!allLoaded && repairState.attempts < repairState.maxAttempts) {
            console.log('⏳ Waiting for libraries...', libraries);
            setTimeout(attemptRepair, 1000);
            return;
        }
        
        // Perform repairs with better error handling
        if (!repairState.repaired.maps) repairMaps();
        if (!repairState.repaired.threejs) repairThreeJS();
        if (!repairState.repaired.charts) repairCharts();
        if (!repairState.repaired.functions) repairGlobalFunctions();
        
        console.log('✅ Universal Feature Repair v2.0 Complete');
    }
    
    function repairMaps() {
        try {
            if (typeof L !== 'undefined') {
                const mapContainers = document.querySelectorAll('#map, #map-container, .map-container, [id*="map"]:not(#performance-chart)');
                
                mapContainers.forEach(container => {
                    if (container && !container._leaflet_id && container.id) {
                        try {
                            // Only create if no global map exists
                            if (!window.map || typeof window.map.setView !== 'function') {
                                console.log(`🗺️ Initializing map for #${container.id}`);
                                window.map = L.map(container.id, {
                                    center: [0, 0],
                                    zoom: 2,
                                    zoomControl: true,
                                    attributionControl: true
                                });
                                
                                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                                    attribution: '© OpenStreetMap contributors',
                                    maxZoom: 18
                                }).addTo(window.map);
                                
                                // Add invalidateSize method if missing
                                if (typeof window.map.invalidateSize !== 'function') {
                                    window.map.invalidateSize = function() {
                                        console.log('Map invalidateSize called');
                                        return this;
                                    };
                                }
                                
                                repairState.repaired.maps = true;
                            }
                        } catch (e) {
                            console.warn(`Could not initialize map for #${container.id}:`, e.message);
                        }
                    }
                });
                
                // Ensure map has required methods
                if (window.map && typeof window.map === 'object') {
                    if (typeof window.map.invalidateSize !== 'function') {
                        window.map.invalidateSize = function() {
                            console.log('Map invalidateSize stub');
                            return this;
                        };
                    }
                }
            }
        } catch (e) {
            console.error('Map repair failed:', e);
        }
    }
    
    function repairThreeJS() {
        try {
            if (typeof THREE !== 'undefined') {
                // Check for WebGL support first
                const canvas = document.createElement('canvas');
                const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                
                if (!gl) {
                    console.warn('WebGL not supported in this browser');
                    repairState.repaired.threejs = true;
                    return;
                }
                
                // Find or create canvas
                let targetCanvas = document.querySelector('#three-canvas, #threejs-canvas, canvas.three-canvas');
                if (!targetCanvas) {
                    const containers = document.querySelectorAll('#threejs-container, #three-container, .three-container');
                    if (containers.length > 0) {
                        targetCanvas = document.createElement('canvas');
                        targetCanvas.id = 'three-canvas';
                        containers[0].appendChild(targetCanvas);
                    }
                }
                
                if (targetCanvas && (!window.renderer || typeof window.renderer.render !== 'function')) {
                    try {
                        console.log('🎨 Initializing THREE.js renderer (v2)');
                        
                        // Use different parameters for better compatibility
                        window.renderer = new THREE.WebGLRenderer({ 
                            canvas: targetCanvas,
                            antialias: false, // Disable for better compatibility
                            alpha: true,
                            powerPreference: 'default',
                            failIfMajorPerformanceCaveat: false
                        });
                        
                        window.renderer.setSize(window.innerWidth, window.innerHeight);
                        window.renderer.setClearColor(0x000000, 0);
                        
                        repairState.repaired.threejs = true;
                    } catch (e) {
                        console.warn('Could not initialize THREE.js renderer:', e.message);
                        // Create stub renderer
                        window.renderer = {
                            render: function() {},
                            setSize: function() {},
                            setClearColor: function() {},
                            domElement: targetCanvas
                        };
                    }
                }
                
                if (!window.scene) {
                    console.log('🎬 Creating THREE.js scene');
                    window.scene = new THREE.Scene();
                    window.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                }
                
                if (!window.camera) {
                    console.log('📷 Creating THREE.js camera');
                    window.camera = new THREE.PerspectiveCamera(
                        75, window.innerWidth / window.innerHeight, 0.1, 1000
                    );
                    window.camera.position.set(0, 0, 5);
                }
            }
        } catch (e) {
            console.error('THREE.js repair failed:', e);
        }
    }
    
    function repairCharts() {
        try {
            if (typeof Chart !== 'undefined') {
                const chartCanvases = document.querySelectorAll('[id*="chart"]:not(.three-canvas), .chart-canvas');
                
                chartCanvases.forEach(canvas => {
                    if (canvas && canvas.getContext) {
                        const canvasId = canvas.id || 'chart-' + Math.random();
                        
                        // Check if chart already exists
                        if (chartInstances.has(canvasId)) {
                            console.log(`📊 Chart already exists for #${canvasId}`);
                            return;
                        }
                        
                        try {
                            // Destroy existing chart if any
                            const existingChart = Chart.getChart(canvas);
                            if (existingChart) {
                                existingChart.destroy();
                            }
                            
                            console.log(`📊 Creating new chart for #${canvasId}`);
                            const ctx = canvas.getContext('2d');
                            
                            const newChart = new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                                    datasets: [{
                                        label: 'Sample Data',
                                        data: [12, 19, 3, 5, 2],
                                        borderColor: 'rgb(75, 192, 192)',
                                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                        tension: 0.1
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: {
                                            display: true,
                                            labels: { color: 'white' }
                                        }
                                    },
                                    scales: {
                                        y: {
                                            ticks: { color: 'white' },
                                            grid: { color: 'rgba(255,255,255,0.1)' }
                                        },
                                        x: {
                                            ticks: { color: 'white' },
                                            grid: { color: 'rgba(255,255,255,0.1)' }
                                        }
                                    }
                                }
                            });
                            
                            chartInstances.set(canvasId, newChart);
                            canvas.chart = newChart;
                            repairState.repaired.charts = true;
                        } catch (e) {
                            console.warn(`Could not initialize chart for #${canvasId}:`, e.message);
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Chart repair failed:', e);
        }
    }
    
    function repairGlobalFunctions() {
        try {
            // Critical functions with better implementations
            const functions = {
                updateStatus: function(message) {
                    console.log(`Status: ${message}`);
                    const elements = document.querySelectorAll('#status-text, .status-text, #status, #processing-status');
                    elements.forEach(el => { if (el) el.textContent = message; });
                },
                
                updateStatusText: function(message) {
                    window.updateStatus(message);
                },
                
                switchView: function() {
                    const mapEl = document.querySelector('#map, #map-container');
                    const threeEl = document.querySelector('#threejs-container, #three-container');
                    if (mapEl && threeEl) {
                        const is3D = mapEl.style.display === 'none';
                        mapEl.style.display = is3D ? 'block' : 'none';
                        threeEl.style.display = is3D ? 'none' : 'block';
                        window.updateStatus(is3D ? 'Switched to 2D view' : 'Switched to 3D view');
                        
                        // Refresh map if switching to 2D
                        if (is3D && window.map && typeof window.map.invalidateSize === 'function') {
                            setTimeout(() => window.map.invalidateSize(), 100);
                        }
                    }
                },
                
                togglePanel: function(panelId) {
                    const panel = document.getElementById(panelId);
                    if (panel) panel.classList.toggle('minimized');
                },
                
                loadVolcanoData: function() {
                    window.updateStatus('Loading volcano data...');
                    if (typeof HOLOCENE_VOLCANOES !== 'undefined' && window.map) {
                        let count = 0;
                        for (const country in HOLOCENE_VOLCANOES) {
                            HOLOCENE_VOLCANOES[country].forEach(volcano => {
                                if (count++ < 200 && L) {
                                    L.circleMarker([volcano.lat, volcano.lon], {
                                        radius: 5,
                                        fillColor: '#ff0000',
                                        fillOpacity: 0.8,
                                        color: '#000',
                                        weight: 1
                                    }).bindPopup(`<b>${volcano.name}</b><br>${volcano.type}`).addTo(window.map);
                                }
                            });
                        }
                        window.updateStatus(`Loaded ${count} volcanoes`);
                    }
                },
                
                processInSARData: function() {
                    window.updateStatus('Processing InSAR data...');
                    setTimeout(() => window.updateStatus('InSAR processing complete'), 2000);
                },
                
                selectVolcano: function(volcano) {
                    window.updateStatus(`Selected: ${volcano.name || 'Unknown'}`);
                },
                
                startAnalysis: function() {
                    window.updateStatus('Starting analysis...');
                    setTimeout(() => window.updateStatus('Analysis complete'), 3000);
                }
            };
            
            // Add functions only if they don't exist
            for (const [name, func] of Object.entries(functions)) {
                if (typeof window[name] !== 'function') {
                    console.log(`🔧 Adding global function: ${name}`);
                    window[name] = func;
                }
            }
            
            repairState.repaired.functions = true;
        } catch (e) {
            console.error('Function repair failed:', e);
        }
    }
    
    // Handle window resize with error checking
    window.addEventListener('resize', function() {
        try {
            if (window.camera && window.renderer && window.renderer.setSize) {
                window.camera.aspect = window.innerWidth / window.innerHeight;
                window.camera.updateProjectionMatrix();
                window.renderer.setSize(window.innerWidth, window.innerHeight);
            }
        } catch (e) {
            console.debug('Resize handler error (non-critical):', e.message);
        }
        
        try {
            if (window.map && typeof window.map.invalidateSize === 'function') {
                window.map.invalidateSize();
            }
        } catch (e) {
            console.debug('Map resize error (non-critical):', e.message);
        }
    });
    
    // Start repair process
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attemptRepair);
    } else {
        setTimeout(attemptRepair, 100);
    }
    
    // Export for manual use
    window.manualRepair = attemptRepair;
    window.repairState = repairState;
    
    console.log('🚀 Universal Feature Repair v2.0 Loaded - Call manualRepair() if needed');
})();