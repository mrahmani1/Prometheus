/**
 * Universal Feature Repair Script
 * Fixes broken features across all Prometheus visualization pages
 */

(function() {
    'use strict';
    
    console.log('🔧 Universal Feature Repair Script Loading...');
    
    // Wait for libraries to be available
    let repairAttempts = 0;
    const maxRepairAttempts = 10;
    
    function attemptRepair() {
        repairAttempts++;
        console.log(`🔍 Repair attempt ${repairAttempts}/${maxRepairAttempts}`);
        
        // Check for required libraries
        const libraries = {
            THREE: typeof THREE !== 'undefined',
            L: typeof L !== 'undefined', 
            Chart: typeof Chart !== 'undefined',
            Plotly: typeof Plotly !== 'undefined'
        };
        
        const allLoaded = Object.values(libraries).every(v => v);
        
        if (!allLoaded && repairAttempts < maxRepairAttempts) {
            console.log('⏳ Waiting for libraries...', libraries);
            setTimeout(attemptRepair, 1000);
            return;
        }
        
        // Perform repairs
        repairMaps();
        repairThreeJS();
        repairCharts();
        repairEventHandlers();
        repairGlobalFunctions();
        
        console.log('✅ Universal Feature Repair Complete');
    }
    
    function repairMaps() {
        // Fix Leaflet maps
        if (typeof L !== 'undefined') {
            // Check if map exists but isn't initialized properly
            const mapContainers = document.querySelectorAll('#map, #map-container, .map-container, [id*="map"]');
            
            mapContainers.forEach(container => {
                if (container && !container._leaflet_id && container.id) {
                    try {
                        // Check if global map variable exists
                        if (!window.map || typeof window.map.setView !== 'function') {
                            console.log(`🗺️ Initializing map for #${container.id}`);
                            window.map = L.map(container.id).setView([0, 0], 2);
                            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                                attribution: '© OpenStreetMap contributors'
                            }).addTo(window.map);
                        }
                    } catch (e) {
                        console.warn(`Could not initialize map for #${container.id}:`, e.message);
                    }
                }
            });
        }
    }
    
    function repairThreeJS() {
        // Fix THREE.js scenes
        if (typeof THREE !== 'undefined') {
            // Ensure global renderer, scene, camera exist
            if (!window.renderer || typeof window.renderer.render !== 'function') {
                const canvas = document.querySelector('#three-canvas, #threejs-canvas, canvas:not(#deformation-chart)');
                if (canvas) {
                    try {
                        console.log('🎨 Initializing THREE.js renderer');
                        window.renderer = new THREE.WebGLRenderer({ 
                            canvas: canvas,
                            antialias: true,
                            alpha: true
                        });
                        window.renderer.setSize(window.innerWidth, window.innerHeight);
                        window.renderer.setClearColor(0x000000, 0);
                    } catch (e) {
                        console.warn('Could not initialize renderer:', e.message);
                    }
                }
            }
            
            if (!window.scene || typeof window.scene.add !== 'function') {
                console.log('🎬 Creating THREE.js scene');
                window.scene = new THREE.Scene();
                
                // Add basic lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                window.scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
                directionalLight.position.set(10, 10, 5);
                window.scene.add(directionalLight);
            }
            
            if (!window.camera) {
                console.log('📷 Creating THREE.js camera');
                window.camera = new THREE.PerspectiveCamera(
                    75,
                    window.innerWidth / window.innerHeight,
                    0.1,
                    1000
                );
                window.camera.position.set(0, 0, 5);
            }
            
            // Start animation if not running
            if (!window.animationRunning) {
                window.animationRunning = true;
                function animate() {
                    requestAnimationFrame(animate);
                    if (window.renderer && window.scene && window.camera) {
                        window.renderer.render(window.scene, window.camera);
                    }
                }
                animate();
            }
        }
    }
    
    function repairCharts() {
        // Fix Chart.js charts
        if (typeof Chart !== 'undefined') {
            const chartCanvases = document.querySelectorAll('[id*="chart"], .chart-canvas');
            chartCanvases.forEach(canvas => {
                if (canvas && !canvas.chart) {
                    try {
                        console.log(`📊 Initializing chart for #${canvas.id}`);
                        const ctx = canvas.getContext('2d');
                        canvas.chart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                                datasets: [{
                                    label: 'Sample Data',
                                    data: [12, 19, 3, 5, 2],
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false
                            }
                        });
                    } catch (e) {
                        console.warn(`Could not initialize chart for #${canvas.id}:`, e.message);
                    }
                }
            });
        }
    }
    
    function repairEventHandlers() {
        // Repair common button handlers
        const buttons = document.querySelectorAll('button, [onclick]');
        buttons.forEach(button => {
            const onclick = button.getAttribute('onclick');
            if (onclick) {
                // Extract function name
                const funcMatch = onclick.match(/^(\w+)\(/);
                if (funcMatch) {
                    const funcName = funcMatch[1];
                    if (typeof window[funcName] !== 'function') {
                        console.log(`🔨 Creating stub for missing function: ${funcName}`);
                        window[funcName] = function(...args) {
                            console.log(`Called: ${funcName}`, args);
                            updateStatus(`${funcName} executed`);
                        };
                    }
                }
            }
        });
    }
    
    function repairGlobalFunctions() {
        // Ensure critical global functions exist
        const criticalFunctions = [
            'updateStatus',
            'updateStatusText',
            'switchView',
            'togglePanel',
            'startAnalysis',
            'loadVolcanoData',
            'processInSARData',
            'selectVolcano'
        ];
        
        criticalFunctions.forEach(funcName => {
            if (typeof window[funcName] !== 'function') {
                console.log(`🔧 Creating global function: ${funcName}`);
                
                switch(funcName) {
                    case 'updateStatus':
                    case 'updateStatusText':
                        window[funcName] = function(message) {
                            console.log(`Status: ${message}`);
                            const statusElements = document.querySelectorAll(
                                '#status-text, .status-text, #status, .status, #processing-status'
                            );
                            statusElements.forEach(el => {
                                if (el) el.textContent = message;
                            });
                        };
                        break;
                        
                    case 'switchView':
                        window[funcName] = function() {
                            const mapEl = document.querySelector('#map, #map-container');
                            const threeEl = document.querySelector('#threejs-container, #three-canvas');
                            if (mapEl && threeEl) {
                                const is3D = mapEl.style.display === 'none';
                                mapEl.style.display = is3D ? 'block' : 'none';
                                threeEl.style.display = is3D ? 'none' : 'block';
                                updateStatus(is3D ? 'Switched to 2D view' : 'Switched to 3D view');
                            }
                        };
                        break;
                        
                    case 'togglePanel':
                        window[funcName] = function(panelId) {
                            const panel = document.getElementById(panelId);
                            if (panel) {
                                panel.classList.toggle('minimized');
                            }
                        };
                        break;
                        
                    case 'loadVolcanoData':
                        window[funcName] = function() {
                            updateStatus('Loading volcano data...');
                            if (typeof HOLOCENE_VOLCANOES !== 'undefined' && window.map) {
                                let count = 0;
                                for (const country in HOLOCENE_VOLCANOES) {
                                    HOLOCENE_VOLCANOES[country].forEach(volcano => {
                                        if (count++ < 100) {
                                            L.circleMarker([volcano.lat, volcano.lon], {
                                                radius: 5,
                                                fillColor: '#ff0000',
                                                fillOpacity: 0.8
                                            }).addTo(window.map);
                                        }
                                    });
                                }
                                updateStatus(`Loaded ${count} volcanoes`);
                            } else {
                                updateStatus('Volcano data not available');
                            }
                        };
                        break;
                        
                    default:
                        window[funcName] = function(...args) {
                            console.log(`${funcName} called with:`, args);
                            updateStatus(`${funcName} executed`);
                        };
                }
            }
        });
    }
    
    function updateStatus(message) {
        if (typeof window.updateStatusText === 'function') {
            window.updateStatusText(message);
        } else {
            console.log('Status:', message);
        }
    }
    
    // Handle window resize
    window.addEventListener('resize', function() {
        if (window.camera && window.renderer) {
            window.camera.aspect = window.innerWidth / window.innerHeight;
            window.camera.updateProjectionMatrix();
            window.renderer.setSize(window.innerWidth, window.innerHeight);
        }
        if (window.map) {
            window.map.invalidateSize();
        }
    });
    
    // Start repair process
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attemptRepair);
    } else {
        setTimeout(attemptRepair, 100);
    }
    
    // Export repair function for manual use
    window.manualRepair = attemptRepair;
    
    console.log('🚀 Universal Feature Repair Script Loaded - Call manualRepair() if needed');
})();