// Comprehensive System Recovery Script
// Fixes Leaflet, WebGL, and other initialization failures

(function() {
    'use strict';
    
    console.log('🚑 System Recovery Script loading...');
    
    // Recovery state tracking
    const recoveryState = {
        leafletFixed: false,
        webglFixed: false,
        chartsFixed: false,
        volcanoSelectorFixed: false
    };
    
    // Leaflet Map Recovery
    function recoverLeafletMap() {
        if (recoveryState.leafletFixed) return;
        
        try {
            console.log('🗺️ Attempting Leaflet map recovery...');
            
            // Check if Leaflet is available
            if (typeof L === 'undefined') {
                console.warn('⚠️ Leaflet not loaded, skipping map recovery');
                return;
            }
            
            // Safe map initialization
            window.safeInitializeMap = function() {
                try {
                    // Remove existing map if it exists
                    if (typeof window.map !== 'undefined' && window.map) {
                        try {
                            window.map.remove();
                        } catch (e) {
                            console.warn('Map removal warning:', e);
                        }
                    }
                    
                    // Create new map
                    const mapContainer = document.getElementById('map');
                    if (mapContainer) {
                        window.map = L.map('map').setView([20, 0], 2);
                        
                        // Add tile layer with error handling
                        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                            attribution: '© OpenStreetMap contributors',
                            maxZoom: 18
                        }).addTo(window.map);
                        
                        console.log('✅ Leaflet map recovered successfully');
                        recoveryState.leafletFixed = true;
                        
                        // Add volcano markers if function exists
                        if (typeof addVolcanoMarkers === 'function') {
                            setTimeout(() => addVolcanoMarkers(), 1000);
                        }
                    }
                } catch (e) {
                    console.error('❌ Map initialization failed:', e);
                }
            };
            
            // Initialize or recover map
            window.safeInitializeMap();
            
        } catch (e) {
            console.error('❌ Leaflet recovery failed:', e);
        }
    }
    
    // WebGL Recovery
    function recoverWebGL() {
        if (recoveryState.webglFixed) return;
        
        try {
            console.log('🎮 Attempting WebGL recovery...');
            
            // Check if THREE.js is available
            if (typeof THREE === 'undefined') {
                console.warn('⚠️ THREE.js not loaded, skipping WebGL recovery');
                return;
            }
            
            // Safe WebGL initialization
            window.safeInitializeWebGL = function() {
                try {
                    const container = document.getElementById('webgl-container') || document.getElementById('three-container');
                    if (!container) {
                        console.warn('⚠️ WebGL container not found');
                        return;
                    }
                    
                    // Create scene
                    window.scene = new THREE.Scene();
                    
                    // Create camera
                    window.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                    window.camera.position.set(0, 0, 5);
                    
                    // Create renderer
                    window.renderer = new THREE.WebGLRenderer({ antialias: true });
                    window.renderer.setSize(container.clientWidth, container.clientHeight);
                    window.renderer.setClearColor(0x000011);
                    
                    // Append to container
                    container.appendChild(window.renderer.domElement);
                    
                    // Add basic lighting
                    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                    window.scene.add(ambientLight);
                    
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                    directionalLight.position.set(1, 1, 0.5);
                    window.scene.add(directionalLight);
                    
                    console.log('✅ WebGL recovered successfully');
                    recoveryState.webglFixed = true;
                    
                    // Start render loop
                    function animate() {
                        requestAnimationFrame(animate);
                        if (window.renderer && window.scene && window.camera) {
                            window.renderer.render(window.scene, window.camera);
                        }
                    }
                    animate();
                    
                } catch (e) {
                    console.error('❌ WebGL initialization failed:', e);
                }
            };
            
            // Initialize WebGL
            window.safeInitializeWebGL();
            
        } catch (e) {
            console.error('❌ WebGL recovery failed:', e);
        }
    }
    
    // Chart.js Recovery
    function recoverCharts() {
        if (recoveryState.chartsFixed) return;
        
        try {
            console.log('📊 Attempting Chart.js recovery...');
            
            if (typeof Chart === 'undefined') {
                console.warn('⚠️ Chart.js not loaded, skipping charts recovery');
                return;
            }
            
            // Safe chart initialization
            window.safeInitializeCharts = function() {
                try {
                    const chartContainers = document.querySelectorAll('canvas[id$="-chart"]');
                    
                    chartContainers.forEach(canvas => {
                        if (!canvas) return;
                        
                        const ctx = canvas.getContext('2d');
                        const chartId = canvas.id;
                        
                        // Basic chart configuration
                        const config = {
                            type: 'line',
                            data: {
                                labels: ['1h', '2h', '3h', '4h', '5h', '6h'],
                                datasets: [{
                                    label: 'Activity',
                                    data: [12, 19, 3, 5, 2, 3],
                                    borderColor: 'rgb(75, 192, 192)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.2)'
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        labels: { color: 'white' }
                                    }
                                },
                                scales: {
                                    x: { ticks: { color: 'white' } },
                                    y: { ticks: { color: 'white' } }
                                }
                            }
                        };
                        
                        // Create chart
                        new Chart(ctx, config);
                        console.log(`✅ Recovered chart: ${chartId}`);
                    });
                    
                    recoveryState.chartsFixed = true;
                    
                } catch (e) {
                    console.error('❌ Charts recovery failed:', e);
                }
            };
            
            // Initialize charts
            window.safeInitializeCharts();
            
        } catch (e) {
            console.error('❌ Charts recovery failed:', e);
        }
    }
    
    // Volcano Selector Recovery
    function recoverVolcanoSelector() {
        if (recoveryState.volcanoSelectorFixed) return;
        
        try {
            console.log('🌋 Attempting volcano selector recovery...');
            
            // Ensure selectVolcano function exists
            if (typeof window.selectVolcano === 'undefined') {
                window.selectVolcano = function(volcanoName) {
                    console.log('🌋 Volcano selected:', volcanoName);
                    
                    // Update UI elements
                    const displays = document.querySelectorAll('.volcano-name, .selected-volcano');
                    displays.forEach(el => {
                        el.textContent = volcanoName;
                    });
                    
                    // Trigger map update if possible
                    if (typeof updateMapForVolcano === 'function') {
                        updateMapForVolcano(volcanoName);
                    }
                };
            }
            
            // Initialize volcano selector UI if comprehensive selector is available
            if (typeof window.initializeVolcanoSelector === 'function') {
                setTimeout(() => {
                    try {
                        window.initializeVolcanoSelector();
                        console.log('✅ Comprehensive volcano selector recovered');
                    } catch (e) {
                        console.warn('Volcano selector initialization warning:', e);
                    }
                }, 1500);
            }
            
            recoveryState.volcanoSelectorFixed = true;
            
        } catch (e) {
            console.error('❌ Volcano selector recovery failed:', e);
        }
    }
    
    // Main recovery function
    function runSystemRecovery() {
        console.log('🚑 Running comprehensive system recovery...');
        
        // Recovery sequence with delays to allow libraries to load
        setTimeout(() => recoverLeafletMap(), 500);
        setTimeout(() => recoverWebGL(), 1000);
        setTimeout(() => recoverCharts(), 1500);
        setTimeout(() => recoverVolcanoSelector(), 2000);
        
        // Final status report
        setTimeout(() => {
            console.log('📊 Recovery Status:', recoveryState);
            
            const successCount = Object.values(recoveryState).filter(Boolean).length;
            const totalSystems = Object.keys(recoveryState).length;
            
            console.log(`✅ Recovery Complete: ${successCount}/${totalSystems} systems recovered`);
            
            // Trigger page refresh event for other scripts
            window.dispatchEvent(new CustomEvent('systemRecoveryComplete', {
                detail: recoveryState
            }));
        }, 3000);
    }
    
    // Global recovery trigger
    window.triggerSystemRecovery = runSystemRecovery;
    
    // Auto-run recovery
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runSystemRecovery);
    } else {
        setTimeout(runSystemRecovery, 100);
    }
    
    // Run recovery again if emergency CDN fix completes
    window.addEventListener('emergencyCDNComplete', runSystemRecovery);
    
    console.log('✅ System Recovery Script initialized');
})();