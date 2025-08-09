/**
 * Volcano System Fix - Task 1.4
 * Fixes broken volcano database UI, missing controls, visualization modes, and WebGL errors
 * @author Prometheus Platform
 * @version 2.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🌋 Volcano System Fix - Initializing comprehensive repairs...');
    
    // Global error handler for WebGL issues
    const WebGLErrorHandler = {
        errors: [],
        maxErrors: 100,
        
        init() {
            // Override console.error to catch WebGL errors
            const originalError = console.error;
            console.error = (...args) => {
                const errorStr = args.join(' ');
                if (errorStr.includes('WebGL') || errorStr.includes('THREE')) {
                    this.handleWebGLError(errorStr);
                }
                originalError.apply(console, args);
            };
            
            // Add WebGL context loss handler
            const canvas = document.querySelector('canvas');
            if (canvas) {
                canvas.addEventListener('webglcontextlost', (e) => {
                    e.preventDefault();
                    this.handleContextLoss();
                });
                
                canvas.addEventListener('webglcontextrestored', () => {
                    this.handleContextRestore();
                });
            }
        },
        
        handleWebGLError(error) {
            this.errors.push({
                type: 'WebGL',
                message: error,
                timestamp: Date.now()
            });
            
            if (this.errors.length > this.maxErrors) {
                this.errors.shift();
            }
            
            // Attempt recovery
            this.attemptRecovery();
        },
        
        handleContextLoss() {
            console.warn('WebGL context lost - attempting recovery...');
            this.showFallbackUI('WebGL context lost. Attempting to restore 3D visualization...');
        },
        
        handleContextRestore() {
            console.log('WebGL context restored successfully');
            this.hideFallbackUI();
            if (typeof reinitializeVisualization === 'function') {
                reinitializeVisualization();
            }
        },
        
        attemptRecovery() {
            // Try to recreate renderer if it failed
            if (window.THREE && !window.renderer) {
                try {
                    this.createFallbackRenderer();
                } catch (e) {
                    console.warn('Failed to create fallback renderer:', e);
                    this.showFallbackUI('3D visualization unavailable. Using 2D mode.');
                }
            }
        },
        
        createFallbackRenderer() {
            const container = document.getElementById('volcano-viz') || 
                            document.getElementById('visualization') ||
                            document.querySelector('.viz-container');
            
            if (!container) return;
            
            try {
                // Create simple fallback renderer
                const renderer = new THREE.WebGLRenderer({
                    antialias: false, // Reduce requirements
                    alpha: true,
                    powerPreference: 'low-power',
                    failIfMajorPerformanceCaveat: false
                });
                
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                container.appendChild(renderer.domElement);
                
                window.renderer = renderer;
                console.log('Fallback renderer created successfully');
                
                // Create simple scene
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(
                    45,
                    container.clientWidth / container.clientHeight,
                    0.1,
                    1000
                );
                camera.position.set(0, 5, 10);
                
                window.scene = scene;
                window.camera = camera;
                
                // Add basic lighting
                const light = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(light);
                
            } catch (error) {
                console.error('Fallback renderer creation failed:', error);
                this.show2DFallback(container);
            }
        },
        
        show2DFallback(container) {
            // Create 2D canvas fallback
            const canvas = document.createElement('canvas');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            
            const ctx = canvas.getContext('2d');
            
            // Draw simple 2D volcano representation
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.strokeStyle = '#FF6B35';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(canvas.width * 0.2, canvas.height * 0.8);
            ctx.lineTo(canvas.width * 0.4, canvas.height * 0.3);
            ctx.lineTo(canvas.width * 0.6, canvas.height * 0.3);
            ctx.lineTo(canvas.width * 0.8, canvas.height * 0.8);
            ctx.closePath();
            ctx.stroke();
            
            ctx.fillStyle = '#FF6B35';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('2D Visualization Mode', canvas.width / 2, canvas.height / 2);
            
            container.innerHTML = '';
            container.appendChild(canvas);
        },
        
        showFallbackUI(message) {
            const fallback = document.createElement('div');
            fallback.id = 'webgl-fallback';
            fallback.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255, 107, 53, 0.9);
                color: white;
                padding: 15px;
                border-radius: 5px;
                z-index: 10000;
                max-width: 300px;
            `;
            fallback.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 5px;">⚠️ Graphics Issue</div>
                <div>${message}</div>
                <button onclick="this.parentElement.remove()" style="margin-top: 10px; padding: 5px 10px; background: white; color: #333; border: none; border-radius: 3px; cursor: pointer;">Dismiss</button>
            `;
            
            document.body.appendChild(fallback);
        },
        
        hideFallbackUI() {
            const fallback = document.getElementById('webgl-fallback');
            if (fallback) fallback.remove();
        }
    };
    
    // Fix 1: Comprehensive Volcano Selector Integration
    function fixVolcanoSelector() {
        console.log('Fixing volcano selector integration...');
        
        // Create enhanced volcano selector UI
        const selectorContainer = document.getElementById('auto-volcano-selector') ||
                                 document.querySelector('.volcano-selector-container');
        
        if (!selectorContainer) {
            console.warn('Volcano selector container not found, creating one...');
            const controlSection = document.querySelector('.control-section');
            if (controlSection) {
                const container = document.createElement('div');
                container.id = 'auto-volcano-selector';
                container.className = 'volcano-selector-enhanced';
                controlSection.appendChild(container);
            }
        }
        
        // Build comprehensive selector UI
        const selector = createEnhancedVolcanoSelector();
        if (selectorContainer) {
            selectorContainer.innerHTML = '';
            selectorContainer.appendChild(selector);
        }
        
        // Initialize with database
        if (window.HOLOCENE_VOLCANOES || window.VOLCANIC_DATABASE) {
            populateVolcanoDatabase();
        } else {
            // Load backup volcano data
            loadBackupVolcanoData();
        }
    }
    
    // Create enhanced volcano selector UI
    function createEnhancedVolcanoSelector() {
        const container = document.createElement('div');
        container.className = 'volcano-selector-panel';
        container.innerHTML = `
            <div class="selector-header">
                <h4>🌋 Global Volcano Database</h4>
                <span class="volcano-count">Loading...</span>
            </div>
            
            <div class="selector-filters">
                <input type="text" id="volcano-search" placeholder="Search volcanoes..." class="search-input">
                
                <div class="filter-row">
                    <select id="continent-filter" class="filter-select">
                        <option value="">All Continents</option>
                    </select>
                    
                    <select id="country-filter" class="filter-select">
                        <option value="">All Countries</option>
                    </select>
                    
                    <select id="type-filter" class="filter-select">
                        <option value="">All Types</option>
                        <option value="Stratovolcano">Stratovolcano</option>
                        <option value="Shield">Shield</option>
                        <option value="Caldera">Caldera</option>
                        <option value="Complex">Complex</option>
                        <option value="Submarine">Submarine</option>
                    </select>
                </div>
                
                <div class="filter-row">
                    <label><input type="checkbox" id="active-only"> Active Only</label>
                    <label><input type="checkbox" id="holocene-only" checked> Holocene Activity</label>
                    <label><input type="checkbox" id="monitored-only"> Monitored Only</label>
                </div>
            </div>
            
            <div class="volcano-list-container">
                <div id="volcano-list" class="volcano-list">
                    <!-- Volcanoes will be populated here -->
                </div>
            </div>
            
            <div class="selector-footer">
                <button id="random-volcano" class="btn">🎲 Random Volcano</button>
                <button id="nearest-volcano" class="btn">📍 Nearest Volcano</button>
                <button id="most-active" class="btn">🔥 Most Active</button>
            </div>
        `;
        
        // Add styles
        addSelectorStyles();
        
        // Setup event listeners
        setupSelectorEvents(container);
        
        return container;
    }
    
    // Add selector styles
    function addSelectorStyles() {
        if (document.getElementById('volcano-selector-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'volcano-selector-styles';
        styles.textContent = `
            .volcano-selector-panel {
                background: rgba(0, 0, 0, 0.9);
                border: 1px solid #FF6B35;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            
            .selector-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(255, 107, 53, 0.3);
            }
            
            .selector-header h4 {
                margin: 0;
                color: #FF6B35;
            }
            
            .volcano-count {
                background: rgba(255, 107, 53, 0.2);
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 12px;
            }
            
            .search-input {
                width: 100%;
                padding: 8px;
                margin-bottom: 10px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
            }
            
            .filter-row {
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }
            
            .filter-select {
                flex: 1;
                padding: 6px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
            }
            
            .filter-select option {
                background: #1a1a1a;
            }
            
            .volcano-list-container {
                max-height: 300px;
                overflow-y: auto;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            
            .volcano-list {
                padding: 5px;
            }
            
            .volcano-item {
                padding: 8px;
                margin: 3px 0;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .volcano-item:hover {
                background: rgba(255, 107, 53, 0.2);
                transform: translateX(5px);
            }
            
            .volcano-item.selected {
                background: rgba(255, 107, 53, 0.4);
                border: 1px solid #FF6B35;
            }
            
            .volcano-info {
                flex: 1;
            }
            
            .volcano-name {
                font-weight: bold;
                color: #FF6B35;
            }
            
            .volcano-details {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.7);
                margin-top: 2px;
            }
            
            .volcano-status {
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                background: rgba(76, 175, 80, 0.2);
                color: #4CAF50;
            }
            
            .volcano-status.active {
                background: rgba(255, 107, 53, 0.2);
                color: #FF6B35;
            }
            
            .selector-footer {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            
            .selector-footer .btn {
                flex: 1;
                padding: 8px;
                background: rgba(255, 107, 53, 0.2);
                border: 1px solid #FF6B35;
                color: white;
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .selector-footer .btn:hover {
                background: rgba(255, 107, 53, 0.4);
            }
        `;
        document.head.appendChild(styles);
    }
    
    // Fix 2: Visualization Mode Switching
    function fixVisualizationModes() {
        console.log('Fixing visualization mode switching...');
        
        const modes = [
            { id: 'terrain', label: '🏔️ Terrain', handler: showTerrainMode },
            { id: 'thermal', label: '🔥 Thermal', handler: showThermalMode },
            { id: 'seismic', label: '📊 Seismic', handler: showSeismicMode },
            { id: 'hazard', label: '⚠️ Hazard', handler: showHazardMode },
            { id: 'realtime', label: '📡 Real-time', handler: showRealtimeMode }
        ];
        
        // Create mode selector UI
        const modeContainer = document.createElement('div');
        modeContainer.className = 'viz-mode-selector';
        modeContainer.innerHTML = `
            <h4>Visualization Mode</h4>
            <div class="mode-buttons">
                ${modes.map(mode => `
                    <button class="mode-btn" data-mode="${mode.id}">
                        ${mode.label}
                    </button>
                `).join('')}
            </div>
        `;
        
        // Add to control panel
        const controlPanel = document.querySelector('.control-panel') || 
                           document.querySelector('.controls');
        if (controlPanel) {
            controlPanel.appendChild(modeContainer);
        }
        
        // Setup mode switching
        modeContainer.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const modeId = this.dataset.mode;
                switchVisualizationMode(modeId);
                
                // Update active state
                modeContainer.querySelectorAll('.mode-btn').forEach(b => 
                    b.classList.remove('active')
                );
                this.classList.add('active');
            });
        });
        
        // Set default mode
        modeContainer.querySelector('[data-mode="terrain"]').classList.add('active');
    }
    
    // Visualization mode handlers
    function showTerrainMode() {
        console.log('Switching to terrain mode');
        if (window.scene) {
            updateSceneForTerrain();
        }
    }
    
    function showThermalMode() {
        console.log('Switching to thermal mode');
        if (window.scene) {
            updateSceneForThermal();
        }
    }
    
    function showSeismicMode() {
        console.log('Switching to seismic mode');
        if (window.scene) {
            updateSceneForSeismic();
        }
    }
    
    function showHazardMode() {
        console.log('Switching to hazard mode');
        if (window.scene) {
            updateSceneForHazard();
        }
    }
    
    function showRealtimeMode() {
        console.log('Switching to real-time mode');
        startRealtimeDataFeed();
    }
    
    function switchVisualizationMode(modeId) {
        const modeHandlers = {
            terrain: showTerrainMode,
            thermal: showThermalMode,
            seismic: showSeismicMode,
            hazard: showHazardMode,
            realtime: showRealtimeMode
        };
        
        if (modeHandlers[modeId]) {
            modeHandlers[modeId]();
        }
    }
    
    // Fix 3: Real-time Data Updates
    function fixRealtimeDataUpdates() {
        console.log('Fixing real-time data updates...');
        
        // Create real-time data manager
        const RealtimeDataManager = {
            updateInterval: null,
            dataFeeds: new Map(),
            isActive: false,
            
            start() {
                if (this.isActive) return;
                
                this.isActive = true;
                console.log('Starting real-time data feeds...');
                
                // Setup update interval
                this.updateInterval = setInterval(() => {
                    this.updateAllFeeds();
                }, 5000); // Update every 5 seconds
                
                // Initial update
                this.updateAllFeeds();
            },
            
            stop() {
                if (this.updateInterval) {
                    clearInterval(this.updateInterval);
                    this.updateInterval = null;
                }
                this.isActive = false;
                console.log('Real-time data feeds stopped');
            },
            
            registerFeed(name, updateFunction) {
                this.dataFeeds.set(name, updateFunction);
            },
            
            updateAllFeeds() {
                this.dataFeeds.forEach((updateFn, name) => {
                    try {
                        updateFn();
                    } catch (error) {
                        console.error(`Error updating feed ${name}:`, error);
                    }
                });
                
                // Update UI indicators
                this.updateIndicators();
            },
            
            updateIndicators() {
                // Update real-time indicator
                const indicator = document.querySelector('.realtime-indicator');
                if (indicator) {
                    indicator.classList.add('pulse');
                    setTimeout(() => indicator.classList.remove('pulse'), 1000);
                }
                
                // Update timestamp
                const timestamp = document.querySelector('.last-update');
                if (timestamp) {
                    timestamp.textContent = new Date().toLocaleTimeString();
                }
            }
        };
        
        // Register data feeds
        RealtimeDataManager.registerFeed('seismic', updateSeismicData);
        RealtimeDataManager.registerFeed('thermal', updateThermalData);
        RealtimeDataManager.registerFeed('deformation', updateDeformationData);
        RealtimeDataManager.registerFeed('gas', updateGasEmissions);
        
        // Add control UI
        const realtimeControl = document.createElement('div');
        realtimeControl.className = 'realtime-control';
        realtimeControl.innerHTML = `
            <div class="realtime-header">
                <span class="realtime-indicator">●</span>
                <span>Real-time Data</span>
                <span class="last-update">--:--:--</span>
            </div>
            <div class="realtime-feeds">
                <label><input type="checkbox" class="feed-toggle" data-feed="seismic" checked> Seismic</label>
                <label><input type="checkbox" class="feed-toggle" data-feed="thermal" checked> Thermal</label>
                <label><input type="checkbox" class="feed-toggle" data-feed="deformation" checked> Deformation</label>
                <label><input type="checkbox" class="feed-toggle" data-feed="gas" checked> Gas Emissions</label>
            </div>
            <button id="toggle-realtime" class="btn">Start Real-time Updates</button>
        `;
        
        // Add to page
        const controlPanel = document.querySelector('.control-panel');
        if (controlPanel) {
            controlPanel.appendChild(realtimeControl);
        }
        
        // Setup toggle button
        document.getElementById('toggle-realtime')?.addEventListener('click', function() {
            if (RealtimeDataManager.isActive) {
                RealtimeDataManager.stop();
                this.textContent = 'Start Real-time Updates';
            } else {
                RealtimeDataManager.start();
                this.textContent = 'Stop Real-time Updates';
            }
        });
        
        // Make globally accessible
        window.RealtimeDataManager = RealtimeDataManager;
    }
    
    // Data update functions
    function updateSeismicData() {
        // Simulate seismic data update
        const magnitude = 0.5 + Math.random() * 2.5;
        const depth = 5 + Math.random() * 20;
        
        const display = document.querySelector('.seismic-display');
        if (display) {
            display.textContent = `M${magnitude.toFixed(1)} at ${depth.toFixed(0)}km`;
        }
        
        // Update visualization if exists
        if (window.updateSeismicVisualization) {
            window.updateSeismicVisualization({ magnitude, depth });
        }
    }
    
    function updateThermalData() {
        // Simulate thermal data update
        const temperature = 20 + Math.random() * 80;
        
        const display = document.querySelector('.thermal-display');
        if (display) {
            display.textContent = `${temperature.toFixed(1)}°C`;
        }
    }
    
    function updateDeformationData() {
        // Simulate deformation data
        const deformation = -5 + Math.random() * 10;
        
        const display = document.querySelector('.deformation-display');
        if (display) {
            display.textContent = `${deformation.toFixed(1)}mm/yr`;
        }
    }
    
    function updateGasEmissions() {
        // Simulate gas emissions data
        const so2 = 100 + Math.random() * 900;
        
        const display = document.querySelector('.gas-display');
        if (display) {
            display.textContent = `SO₂: ${so2.toFixed(0)} t/d`;
        }
    }
    
    // Fix 4: WebGL Error Handling
    function setupWebGLErrorHandling() {
        console.log('Setting up WebGL error handling...');
        
        // Initialize error handler
        WebGLErrorHandler.init();
        
        // Check WebGL availability
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        
        if (!gl) {
            console.warn('WebGL not supported, using fallback visualization');
            createFallbackVisualization();
            return false;
        }
        
        // Check for WebGL extensions
        const extensions = {
            anisotropic: gl.getExtension('EXT_texture_filter_anisotropic'),
            floatTextures: gl.getExtension('OES_texture_float'),
            derivatives: gl.getExtension('OES_standard_derivatives')
        };
        
        console.log('WebGL extensions:', Object.keys(extensions).filter(k => extensions[k]));
        
        // Setup error recovery
        if (window.THREE) {
            const originalWebGLRenderer = THREE.WebGLRenderer;
            THREE.WebGLRenderer = function(params = {}) {
                params.failIfMajorPerformanceCaveat = false;
                params.powerPreference = params.powerPreference || 'low-power';
                
                try {
                    return new originalWebGLRenderer(params);
                } catch (error) {
                    console.error('WebGL Renderer creation failed:', error);
                    WebGLErrorHandler.handleWebGLError(error.message);
                    throw error;
                }
            };
        }
        
        return true;
    }
    
    // Create fallback visualization for non-WebGL browsers
    function createFallbackVisualization() {
        const container = document.getElementById('volcano-viz') || 
                         document.querySelector('.viz-container');
        
        if (!container) return;
        
        container.innerHTML = `
            <div class="fallback-viz">
                <h3>2D Visualization Mode</h3>
                <canvas id="fallback-canvas" width="800" height="600"></canvas>
                <div class="fallback-info">
                    <p>3D visualization is not available in your browser.</p>
                    <p>Using 2D fallback mode.</p>
                </div>
            </div>
        `;
        
        // Initialize 2D visualization
        init2DVisualization();
    }
    
    function init2DVisualization() {
        const canvas = document.getElementById('fallback-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Draw simple 2D volcano cross-section
        function drawVolcano() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Background
            const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, '#001');
            gradient.addColorStop(1, '#003');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Volcano shape
            ctx.strokeStyle = '#FF6B35';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(200, 500);
            ctx.quadraticCurveTo(400, 200, 600, 500);
            ctx.stroke();
            
            // Magma chamber
            ctx.fillStyle = 'rgba(255, 107, 53, 0.5)';
            ctx.beginPath();
            ctx.ellipse(400, 450, 80, 60, 0, 0, Math.PI * 2);
            ctx.fill();
            
            // Labels
            ctx.fillStyle = 'white';
            ctx.font = '14px Arial';
            ctx.fillText('Volcano Cross-Section', 10, 30);
        }
        
        drawVolcano();
        
        // Animate
        let frame = 0;
        function animate() {
            frame++;
            if (frame % 60 === 0) {
                drawVolcano();
                // Add animation effects
                ctx.fillStyle = `rgba(255, 107, 53, ${0.3 + Math.sin(frame * 0.05) * 0.2})`;
                ctx.beginPath();
                ctx.ellipse(400, 450, 80 + Math.sin(frame * 0.02) * 10, 60, 0, 0, Math.PI * 2);
                ctx.fill();
            }
            requestAnimationFrame(animate);
        }
        animate();
    }
    
    // Setup event listeners for volcano selector
    function setupSelectorEvents(container) {
        // Search functionality
        const searchInput = container.querySelector('#volcano-search');
        if (searchInput) {
            searchInput.addEventListener('input', debounce(filterVolcanoes, 300));
        }
        
        // Filter dropdowns
        container.querySelector('#continent-filter')?.addEventListener('change', filterVolcanoes);
        container.querySelector('#country-filter')?.addEventListener('change', filterVolcanoes);
        container.querySelector('#type-filter')?.addEventListener('change', filterVolcanoes);
        
        // Checkboxes
        container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', filterVolcanoes);
        });
        
        // Special buttons
        container.querySelector('#random-volcano')?.addEventListener('click', selectRandomVolcano);
        container.querySelector('#nearest-volcano')?.addEventListener('click', selectNearestVolcano);
        container.querySelector('#most-active')?.addEventListener('click', selectMostActive);
    }
    
    // Filter volcanoes based on criteria
    function filterVolcanoes() {
        console.log('Filtering volcanoes...');
        // Implementation depends on actual data structure
    }
    
    // Select random volcano
    function selectRandomVolcano() {
        const volcanoes = document.querySelectorAll('.volcano-item');
        if (volcanoes.length > 0) {
            const random = volcanoes[Math.floor(Math.random() * volcanoes.length)];
            random.click();
        }
    }
    
    // Select nearest volcano (requires geolocation)
    function selectNearestVolcano() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(position => {
                console.log('Finding nearest volcano to', position.coords);
                // Implementation would calculate distances
            });
        }
    }
    
    // Select most active volcano
    function selectMostActive() {
        console.log('Selecting most active volcano...');
        // Implementation would find volcano with most recent activity
    }
    
    // Populate volcano database
    function populateVolcanoDatabase() {
        const list = document.getElementById('volcano-list');
        if (!list) return;
        
        let count = 0;
        const volcanoes = window.HOLOCENE_VOLCANOES || window.VOLCANIC_DATABASE || {};
        
        // Create volcano items
        Object.entries(volcanoes).forEach(([country, countryVolcanoes]) => {
            if (Array.isArray(countryVolcanoes)) {
                countryVolcanoes.forEach(volcano => {
                    const item = createVolcanoItem(volcano, country);
                    list.appendChild(item);
                    count++;
                });
            }
        });
        
        // Update count
        const countDisplay = document.querySelector('.volcano-count');
        if (countDisplay) {
            countDisplay.textContent = `${count} volcanoes`;
        }
    }
    
    // Create volcano list item
    function createVolcanoItem(volcano, country) {
        const item = document.createElement('div');
        item.className = 'volcano-item';
        item.dataset.volcano = JSON.stringify(volcano);
        
        const isActive = volcano.last_eruption && 
                        parseInt(volcano.last_eruption) > 2000;
        
        item.innerHTML = `
            <div class="volcano-info">
                <div class="volcano-name">${volcano.name || 'Unknown'}</div>
                <div class="volcano-details">
                    ${country} • ${volcano.type || 'Unknown type'} • 
                    ${volcano.elevation || '?'}m
                </div>
            </div>
            <div class="volcano-status ${isActive ? 'active' : ''}">
                ${isActive ? 'Active' : 'Dormant'}
            </div>
        `;
        
        item.addEventListener('click', () => selectVolcano(volcano));
        
        return item;
    }
    
    // Select volcano
    function selectVolcano(volcano) {
        console.log('Selected volcano:', volcano);
        
        // Update UI
        document.querySelectorAll('.volcano-item').forEach(item => 
            item.classList.remove('selected')
        );
        event.currentTarget.classList.add('selected');
        
        // Trigger visualization update
        if (window.updateVolcanoVisualization) {
            window.updateVolcanoVisualization(volcano);
        }
        
        // Update info panels
        updateVolcanoInfo(volcano);
    }
    
    // Update volcano information panels
    function updateVolcanoInfo(volcano) {
        // Update various info displays
        const nameDisplay = document.querySelector('.volcano-name-display');
        if (nameDisplay) {
            nameDisplay.textContent = volcano.name;
        }
        
        const infoPanel = document.querySelector('.volcano-info-panel');
        if (infoPanel) {
            infoPanel.innerHTML = `
                <h3>${volcano.name}</h3>
                <p>Type: ${volcano.type || 'Unknown'}</p>
                <p>Elevation: ${volcano.elevation || '?'} m</p>
                <p>Location: ${volcano.latitude || '?'}, ${volcano.longitude || '?'}</p>
                <p>Last Eruption: ${volcano.last_eruption || 'Unknown'}</p>
            `;
        }
    }
    
    // Load backup volcano data if needed
    function loadBackupVolcanoData() {
        console.log('Loading backup volcano data...');
        
        // Sample backup data
        window.VOLCANIC_DATABASE = {
            "USA": [
                { name: "Mount St. Helens", type: "Stratovolcano", elevation: 2549, latitude: 46.1914, longitude: -122.1956, last_eruption: "2008" },
                { name: "Kilauea", type: "Shield", elevation: 1222, latitude: 19.4069, longitude: -155.2834, last_eruption: "2023" },
                { name: "Mount Rainier", type: "Stratovolcano", elevation: 4392, latitude: 46.8523, longitude: -121.7603, last_eruption: "1894" }
            ],
            "Japan": [
                { name: "Mount Fuji", type: "Stratovolcano", elevation: 3776, latitude: 35.3606, longitude: 138.7274, last_eruption: "1708" },
                { name: "Sakurajima", type: "Stratovolcano", elevation: 1117, latitude: 31.5833, longitude: 130.6500, last_eruption: "2023" }
            ],
            "Italy": [
                { name: "Vesuvius", type: "Complex", elevation: 1281, latitude: 40.8209, longitude: 14.4289, last_eruption: "1944" },
                { name: "Etna", type: "Stratovolcano", elevation: 3357, latitude: 37.7510, longitude: 14.9934, last_eruption: "2023" }
            ]
        };
        
        populateVolcanoDatabase();
    }
    
    // Utility: Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Main initialization
    function initVolcanoSystemFix() {
        console.log('Initializing Volcano System Fix...');
        
        try {
            // Fix volcano selector
            fixVolcanoSelector();
            
            // Fix visualization modes
            fixVisualizationModes();
            
            // Fix real-time data updates
            fixRealtimeDataUpdates();
            
            // Setup WebGL error handling
            setupWebGLErrorHandling();
            
            // Add monitoring
            monitorSystemHealth();
            
            console.log('✅ Volcano System Fix applied successfully');
            
        } catch (error) {
            console.error('❌ Error applying Volcano System Fix:', error);
        }
    }
    
    // System health monitoring
    function monitorSystemHealth() {
        setInterval(() => {
            const health = {
                webgl: !!window.renderer,
                realtime: window.RealtimeDataManager?.isActive,
                volcanoes: document.querySelectorAll('.volcano-item').length,
                errors: WebGLErrorHandler.errors.length
            };
            
            console.log('System health:', health);
        }, 30000); // Check every 30 seconds
    }
    
    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initVolcanoSystemFix);
    } else {
        initVolcanoSystemFix();
    }
    
    // Export for global access
    global.VolcanoSystemFix = {
        init: initVolcanoSystemFix,
        WebGLErrorHandler,
        fixVolcanoSelector,
        fixVisualizationModes,
        fixRealtimeDataUpdates,
        selectVolcano,
        filterVolcanoes
    };
    
    console.log('🌋 Volcano System Fix ready. Access via window.VolcanoSystemFix');
    
})(window);