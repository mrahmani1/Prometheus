/**
 * Advanced Map System - Task 2.4
 * Interactive map system with Mapbox GL JS and Leaflet fallback for geological data visualization
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🗺️ Advanced Map System - Initializing interactive geological mapping...');
    
    class AdvancedMapSystem {
        constructor(container, options = {}) {
            this.container = typeof container === 'string' ? 
                document.getElementById(container) || document.querySelector(container) : 
                container;
            
            if (!this.container) {
                throw new Error('Container element not found');
            }
            
            this.options = {
                // Map provider preferences
                preferMapbox: true,
                mapboxToken: null, // Set this for Mapbox GL JS
                
                // Default map settings
                center: [0, 0],
                zoom: 2,
                minZoom: 1,
                maxZoom: 18,
                
                // 3D terrain
                enable3D: true,
                terrainExaggeration: 1.5,
                
                // Clustering
                enableClustering: true,
                clusterRadius: 50,
                maxClusterZoom: 14,
                
                // Drawing tools
                drawingEnabled: true,
                drawingTools: ['point', 'polygon', 'rectangle', 'circle'],
                
                // Time animation
                timeAnimationEnabled: false,
                timeField: 'timestamp',
                
                // Styling
                mapStyle: 'geological',
                customStyles: {},
                
                // Data layers
                layers: [],
                
                // Events
                onMapReady: null,
                onFeatureClick: null,
                onFeatureHover: null,
                onDrawComplete: null,
                onTimeChange: null,
                
                ...options
            };
            
            this.state = {
                mapReady: false,
                currentProvider: null,
                map: null,
                markers: new Map(),
                layers: new Map(),
                clusters: null,
                drawingMode: null,
                drawnFeatures: [],
                selectedFeatures: new Set(),
                timeRange: null,
                currentTime: null,
                animationPlaying: false
            };
            
            this.eventBus = new EventTarget();
            this.init();
        }
        
        async init() {
            try {
                this.createMapContainer();
                await this.initializeMap();
                this.setupControls();
                this.setupEventListeners();
                
                console.log('✅ Advanced Map System initialized with', this.state.currentProvider);
                
                if (this.options.onMapReady) {
                    this.options.onMapReady(this);
                }
                
                this.emit('map:ready', { provider: this.state.currentProvider });
                
            } catch (error) {
                console.error('❌ Failed to initialize map system:', error);
                this.showError('Failed to initialize map system: ' + error.message);
            }
        }
        
        createMapContainer() {
            this.container.className = 'advanced-map-system';
            this.container.innerHTML = `
                <div class="map-loading" id="map-loading">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Initializing map...</div>
                </div>
                
                <div class="map-container" id="map-container">
                    <!-- Map will be initialized here -->
                </div>
                
                <div class="map-controls" id="map-controls">
                    <div class="control-group map-layers">
                        <button class="control-btn" id="layers-btn" title="Layers">🗂️</button>
                        <div class="layers-panel" id="layers-panel">
                            <div class="panel-header">
                                <h4>Map Layers</h4>
                                <button class="close-btn">×</button>
                            </div>
                            <div class="layers-list" id="layers-list">
                                <!-- Layers will be populated here -->
                            </div>
                        </div>
                    </div>
                    
                    <div class="control-group map-styles">
                        <button class="control-btn" id="styles-btn" title="Map Style">🎨</button>
                        <div class="styles-panel" id="styles-panel">
                            <div class="panel-header">
                                <h4>Map Styles</h4>
                                <button class="close-btn">×</button>
                            </div>
                            <div class="styles-grid">
                                <button class="style-option active" data-style="geological">Geological</button>
                                <button class="style-option" data-style="satellite">Satellite</button>
                                <button class="style-option" data-style="terrain">Terrain</button>
                                <button class="style-option" data-style="dark">Dark</button>
                                <button class="style-option" data-style="light">Light</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="control-group map-drawing">
                        <button class="control-btn" id="drawing-btn" title="Drawing Tools">✏️</button>
                        <div class="drawing-panel" id="drawing-panel">
                            <div class="panel-header">
                                <h4>Drawing Tools</h4>
                                <button class="close-btn">×</button>
                            </div>
                            <div class="drawing-tools">
                                <button class="tool-btn" data-tool="point">📍 Point</button>
                                <button class="tool-btn" data-tool="polygon">⬟ Polygon</button>
                                <button class="tool-btn" data-tool="rectangle">⬛ Rectangle</button>
                                <button class="tool-btn" data-tool="circle">⭕ Circle</button>
                                <button class="tool-btn clear-btn">🗑️ Clear All</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="control-group map-3d">
                        <button class="control-btn" id="toggle-3d" title="Toggle 3D Terrain">🏔️</button>
                    </div>
                    
                    <div class="control-group map-fullscreen">
                        <button class="control-btn" id="fullscreen-btn" title="Fullscreen">⛶</button>
                    </div>
                </div>
                
                <div class="map-info-panel" id="info-panel" style="display: none;">
                    <div class="info-content" id="info-content"></div>
                    <button class="close-info">×</button>
                </div>
                
                <div class="time-controls" id="time-controls" style="display: none;">
                    <div class="time-slider-container">
                        <button class="time-btn" id="play-pause">▶️</button>
                        <input type="range" class="time-slider" id="time-slider" min="0" max="100" value="0">
                        <button class="time-btn" id="time-reset">⏹️</button>
                    </div>
                    <div class="time-display" id="time-display">2024-01-01</div>
                    <div class="time-speed">
                        <label>Speed:</label>
                        <select id="time-speed-select">
                            <option value="1">1x</option>
                            <option value="2">2x</option>
                            <option value="5">5x</option>
                            <option value="10" selected>10x</option>
                        </select>
                    </div>
                </div>
                
                <div class="map-legend" id="map-legend">
                    <div class="legend-header">
                        <h4>Legend</h4>
                        <button class="toggle-legend">−</button>
                    </div>
                    <div class="legend-content" id="legend-content">
                        <!-- Legend items will be populated here -->
                    </div>
                </div>
                
                <div class="map-error" id="map-error" style="display: none;">
                    <div class="error-content">
                        <div class="error-icon">⚠️</div>
                        <div class="error-message" id="error-message"></div>
                        <button class="retry-btn" id="retry-btn">Retry</button>
                    </div>
                </div>
            `;
            
            this.elements = {
                loading: this.container.querySelector('#map-loading'),
                mapContainer: this.container.querySelector('#map-container'),
                controls: this.container.querySelector('#map-controls'),
                infoPanel: this.container.querySelector('#info-panel'),
                timeControls: this.container.querySelector('#time-controls'),
                legend: this.container.querySelector('#map-legend'),
                error: this.container.querySelector('#map-error')
            };
            
            this.injectStyles();
        }
        
        async initializeMap() {
            this.elements.loading.style.display = 'flex';
            
            try {
                // Try Mapbox first if preferred and token available
                if (this.options.preferMapbox && this.options.mapboxToken && window.mapboxgl) {
                    await this.initMapboxGL();
                } else if (window.L) {
                    await this.initLeaflet();
                } else {
                    throw new Error('No map library available (Mapbox GL JS or Leaflet)');
                }
                
                this.state.mapReady = true;
                this.elements.loading.style.display = 'none';
                
            } catch (error) {
                console.warn('Map initialization failed:', error);
                // Try fallback
                if (this.state.currentProvider !== 'leaflet' && window.L) {
                    console.log('Falling back to Leaflet...');
                    await this.initLeaflet();
                } else {
                    throw error;
                }
            }
        }
        
        async initMapboxGL() {
            if (!window.mapboxgl) {
                throw new Error('Mapbox GL JS not loaded');
            }
            
            window.mapboxgl.accessToken = this.options.mapboxToken;
            
            this.state.map = new window.mapboxgl.Map({
                container: this.elements.mapContainer,
                style: this.getMapboxStyle(),
                center: this.options.center,
                zoom: this.options.zoom,
                minZoom: this.options.minZoom,
                maxZoom: this.options.maxZoom,
                antialias: true
            });
            
            this.state.currentProvider = 'mapbox';
            
            // Wait for map to load
            await new Promise((resolve, reject) => {
                this.state.map.on('load', resolve);
                this.state.map.on('error', reject);
                setTimeout(() => reject(new Error('Map load timeout')), 10000);
            });
            
            // Add 3D terrain if enabled
            if (this.options.enable3D) {
                this.setup3DTerrain();
            }
            
            this.setupMapboxEventListeners();
        }
        
        async initLeaflet() {
            if (!window.L) {
                throw new Error('Leaflet not loaded');
            }
            
            this.state.map = window.L.map(this.elements.mapContainer, {
                center: this.options.center,
                zoom: this.options.zoom,
                minZoom: this.options.minZoom,
                maxZoom: this.options.maxZoom,
                zoomControl: false
            });
            
            this.state.currentProvider = 'leaflet';
            
            // Add tile layer
            const tileLayer = this.getLeafletTileLayer();
            tileLayer.addTo(this.state.map);
            
            // Setup clustering if enabled
            if (this.options.enableClustering && window.L.markerClusterGroup) {
                this.state.clusters = window.L.markerClusterGroup({
                    maxClusterRadius: this.options.clusterRadius,
                    disableClusteringAtZoom: this.options.maxClusterZoom
                });
                this.state.map.addLayer(this.state.clusters);
            }
            
            this.setupLeafletEventListeners();
        }
        
        getMapboxStyle() {
            const styles = {
                geological: 'mapbox://styles/mapbox/outdoors-v11',
                satellite: 'mapbox://styles/mapbox/satellite-v9',
                terrain: 'mapbox://styles/mapbox/outdoors-v11',
                dark: 'mapbox://styles/mapbox/dark-v10',
                light: 'mapbox://styles/mapbox/light-v10'
            };
            
            return styles[this.options.mapStyle] || styles.geological;
        }
        
        getLeafletTileLayer() {
            const tileLayers = {
                geological: window.L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenTopoMap contributors'
                }),
                satellite: window.L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                    attribution: '© Esri'
                }),
                terrain: window.L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenTopoMap contributors'
                }),
                dark: window.L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                    attribution: '© CARTO'
                }),
                light: window.L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                    attribution: '© CARTO'
                })
            };
            
            return tileLayers[this.options.mapStyle] || tileLayers.geological;
        }
        
        setup3DTerrain() {
            if (this.state.currentProvider !== 'mapbox') return;
            
            try {
                this.state.map.addSource('mapbox-dem', {
                    type: 'raster-dem',
                    url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
                    tileSize: 512,
                    maxzoom: 14
                });
                
                this.state.map.setTerrain({ 
                    source: 'mapbox-dem', 
                    exaggeration: this.options.terrainExaggeration 
                });
                
                // Add sky layer
                this.state.map.addLayer({
                    id: 'sky',
                    type: 'sky',
                    paint: {
                        'sky-type': 'atmosphere',
                        'sky-atmosphere-sun': [0.0, 0.0],
                        'sky-atmosphere-sun-intensity': 15
                    }
                });
                
            } catch (error) {
                console.warn('Failed to setup 3D terrain:', error);
            }
        }
        
        setupMapboxEventListeners() {
            const map = this.state.map;
            
            map.on('click', (e) => this.handleMapClick(e));
            map.on('mousemove', (e) => this.handleMapHover(e));
            map.on('moveend', () => this.emit('map:moveend'));
            map.on('zoomend', () => this.emit('map:zoomend'));
        }
        
        setupLeafletEventListeners() {
            const map = this.state.map;
            
            map.on('click', (e) => this.handleMapClick(e));
            map.on('mousemove', (e) => this.handleMapHover(e));
            map.on('moveend', () => this.emit('map:moveend'));
            map.on('zoomend', () => this.emit('map:zoomend'));
        }
        
        setupControls() {
            // Layer control
            const layersBtn = this.container.querySelector('#layers-btn');
            const layersPanel = this.container.querySelector('#layers-panel');
            
            layersBtn?.addEventListener('click', () => {
                this.togglePanel(layersPanel);
            });
            
            // Style control
            const stylesBtn = this.container.querySelector('#styles-btn');
            const stylesPanel = this.container.querySelector('#styles-panel');
            
            stylesBtn?.addEventListener('click', () => {
                this.togglePanel(stylesPanel);
            });
            
            // Drawing control
            const drawingBtn = this.container.querySelector('#drawing-btn');
            const drawingPanel = this.container.querySelector('#drawing-panel');
            
            drawingBtn?.addEventListener('click', () => {
                this.togglePanel(drawingPanel);
            });
            
            // 3D toggle
            const toggle3D = this.container.querySelector('#toggle-3d');
            toggle3D?.addEventListener('click', () => {
                this.toggle3D();
            });
            
            // Fullscreen
            const fullscreenBtn = this.container.querySelector('#fullscreen-btn');
            fullscreenBtn?.addEventListener('click', () => {
                this.toggleFullscreen();
            });
            
            // Style options
            this.container.querySelectorAll('.style-option').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.setMapStyle(btn.dataset.style);
                });
            });
            
            // Drawing tools
            this.container.querySelectorAll('.tool-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (btn.classList.contains('clear-btn')) {
                        this.clearDrawnFeatures();
                    } else {
                        this.setDrawingMode(btn.dataset.tool);
                    }
                });
            });
            
            // Close buttons
            this.container.querySelectorAll('.close-btn, .close-info').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const panel = e.target.closest('.layers-panel, .styles-panel, .drawing-panel, .map-info-panel');
                    if (panel) {
                        panel.style.display = 'none';
                    }
                });
            });
            
            // Time controls
            this.setupTimeControls();
        }
        
        setupTimeControls() {
            if (!this.options.timeAnimationEnabled) return;
            
            this.elements.timeControls.style.display = 'block';
            
            const playPauseBtn = this.container.querySelector('#play-pause');
            const timeSlider = this.container.querySelector('#time-slider');
            const resetBtn = this.container.querySelector('#time-reset');
            const speedSelect = this.container.querySelector('#time-speed-select');
            
            playPauseBtn?.addEventListener('click', () => {
                this.toggleTimeAnimation();
            });
            
            timeSlider?.addEventListener('input', (e) => {
                this.setTimePosition(e.target.value);
            });
            
            resetBtn?.addEventListener('click', () => {
                this.resetTimeAnimation();
            });
            
            speedSelect?.addEventListener('change', (e) => {
                this.setAnimationSpeed(e.target.value);
            });
        }
        
        setupEventListeners() {
            // Handle window resize
            window.addEventListener('resize', () => {
                this.resize();
            });
            
            // Handle fullscreen changes
            document.addEventListener('fullscreenchange', () => {
                this.handleFullscreenChange();
            });
            
            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (!this.container.contains(document.activeElement)) return;
                
                switch (e.key) {
                    case 'Escape':
                        this.setDrawingMode(null);
                        this.hideInfoPanel();
                        break;
                    case 'f':
                    case 'F':
                        if (e.ctrlKey || e.metaKey) {
                            e.preventDefault();
                            this.toggleFullscreen();
                        }
                        break;
                    case ' ':
                        if (this.options.timeAnimationEnabled) {
                            e.preventDefault();
                            this.toggleTimeAnimation();
                        }
                        break;
                }
            });
        }
        
        // Public API Methods
        
        addLayer(layer) {
            if (!this.state.mapReady) {
                console.warn('Map not ready, queuing layer addition');
                setTimeout(() => this.addLayer(layer), 100);
                return;
            }
            
            const layerId = layer.id || this.generateId();
            
            try {
                if (this.state.currentProvider === 'mapbox') {
                    this.addMapboxLayer(layer, layerId);
                } else {
                    this.addLeafletLayer(layer, layerId);
                }
                
                this.state.layers.set(layerId, layer);
                this.updateLayersPanel();
                this.updateLegend();
                
                this.emit('layer:added', { layer, layerId });
                
            } catch (error) {
                console.error('Failed to add layer:', error);
            }
        }
        
        removeLayer(layerId) {
            if (!this.state.layers.has(layerId)) return;
            
            try {
                if (this.state.currentProvider === 'mapbox') {
                    this.removeMapboxLayer(layerId);
                } else {
                    this.removeLeafletLayer(layerId);
                }
                
                this.state.layers.delete(layerId);
                this.updateLayersPanel();
                this.updateLegend();
                
                this.emit('layer:removed', { layerId });
                
            } catch (error) {
                console.error('Failed to remove layer:', error);
            }
        }
        
        addMarkers(features, options = {}) {
            if (!Array.isArray(features)) {
                features = [features];
            }
            
            features.forEach(feature => {
                const markerId = feature.id || this.generateId();
                
                if (this.state.currentProvider === 'mapbox') {
                    this.addMapboxMarker(feature, markerId, options);
                } else {
                    this.addLeafletMarker(feature, markerId, options);
                }
                
                this.state.markers.set(markerId, feature);
            });
            
            this.emit('markers:added', { features, options });
        }
        
        addMapboxMarker(feature, markerId, options) {
            const marker = new window.mapboxgl.Marker(options)
                .setLngLat(feature.geometry.coordinates)
                .addTo(this.state.map);
            
            if (feature.properties && feature.properties.popup) {
                const popup = new window.mapboxgl.Popup({ offset: 25 })
                    .setHTML(feature.properties.popup);
                marker.setPopup(popup);
            }
            
            marker._markerId = markerId;
        }
        
        addLeafletMarker(feature, markerId, options) {
            const coords = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
            const marker = window.L.marker(coords, options);
            
            if (feature.properties && feature.properties.popup) {
                marker.bindPopup(feature.properties.popup);
            }
            
            if (this.state.clusters) {
                this.state.clusters.addLayer(marker);
            } else {
                marker.addTo(this.state.map);
            }
            
            marker._markerId = markerId;
        }
        
        setMapStyle(styleName) {
            this.options.mapStyle = styleName;
            
            // Update active style button
            this.container.querySelectorAll('.style-option').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.style === styleName);
            });
            
            if (this.state.currentProvider === 'mapbox') {
                this.state.map.setStyle(this.getMapboxStyle());
            } else {
                // For Leaflet, we need to remove old tile layer and add new one
                this.state.map.eachLayer(layer => {
                    if (layer instanceof window.L.TileLayer) {
                        this.state.map.removeLayer(layer);
                    }
                });
                
                const newTileLayer = this.getLeafletTileLayer();
                newTileLayer.addTo(this.state.map);
            }
            
            this.emit('style:changed', { style: styleName });
        }
        
        setDrawingMode(mode) {
            this.state.drawingMode = mode;
            
            // Update active drawing tool
            this.container.querySelectorAll('.tool-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tool === mode);
            });
            
            if (mode) {
                this.container.classList.add('drawing-mode');
                this.startDrawing(mode);
            } else {
                this.container.classList.remove('drawing-mode');
                this.stopDrawing();
            }
            
            this.emit('drawing:mode-changed', { mode });
        }
        
        startDrawing(mode) {
            // Implementation depends on the drawing library used
            console.log('Starting drawing mode:', mode);
        }
        
        stopDrawing() {
            console.log('Stopping drawing mode');
        }
        
        clearDrawnFeatures() {
            this.state.drawnFeatures = [];
            this.emit('drawing:cleared');
        }
        
        toggle3D() {
            if (this.state.currentProvider !== 'mapbox') {
                console.warn('3D terrain only available with Mapbox GL JS');
                return;
            }
            
            const terrain = this.state.map.getTerrain();
            
            if (terrain) {
                this.state.map.setTerrain(null);
                this.container.querySelector('#toggle-3d').classList.remove('active');
            } else {
                this.setup3DTerrain();
                this.container.querySelector('#toggle-3d').classList.add('active');
            }
            
            this.emit('3d:toggled', { enabled: !terrain });
        }
        
        toggleFullscreen() {
            if (!document.fullscreenElement) {
                this.container.requestFullscreen().catch(err => {
                    console.error('Failed to enter fullscreen:', err);
                });
            } else {
                document.exitFullscreen();
            }
        }
        
        handleFullscreenChange() {
            const isFullscreen = !!document.fullscreenElement;
            this.container.classList.toggle('fullscreen', isFullscreen);
            
            // Resize map after fullscreen change
            setTimeout(() => this.resize(), 100);
            
            this.emit('fullscreen:changed', { fullscreen: isFullscreen });
        }
        
        resize() {
            if (!this.state.mapReady) return;
            
            if (this.state.currentProvider === 'mapbox') {
                this.state.map.resize();
            } else {
                this.state.map.invalidateSize();
            }
        }
        
        flyTo(coordinates, zoom = null) {
            if (!this.state.mapReady) return;
            
            const options = {
                center: coordinates,
                ...(zoom !== null && { zoom })
            };
            
            if (this.state.currentProvider === 'mapbox') {
                this.state.map.flyTo(options);
            } else {
                this.state.map.flyTo(coordinates, zoom || this.state.map.getZoom());
            }
        }
        
        fitBounds(bounds) {
            if (!this.state.mapReady) return;
            
            if (this.state.currentProvider === 'mapbox') {
                this.state.map.fitBounds(bounds, { padding: 50 });
            } else {
                this.state.map.fitBounds(bounds, { padding: [50, 50] });
            }
        }
        
        // Time Animation Methods
        
        toggleTimeAnimation() {
            this.state.animationPlaying = !this.state.animationPlaying;
            
            const playPauseBtn = this.container.querySelector('#play-pause');
            if (playPauseBtn) {
                playPauseBtn.textContent = this.state.animationPlaying ? '⏸️' : '▶️';
            }
            
            if (this.state.animationPlaying) {
                this.startTimeAnimation();
            } else {
                this.stopTimeAnimation();
            }
            
            this.emit('time:toggle', { playing: this.state.animationPlaying });
        }
        
        startTimeAnimation() {
            if (this.timeAnimationInterval) {
                clearInterval(this.timeAnimationInterval);
            }
            
            const speed = parseInt(this.container.querySelector('#time-speed-select')?.value || 10);
            
            this.timeAnimationInterval = setInterval(() => {
                this.advanceTime();
            }, 1000 / speed);
        }
        
        stopTimeAnimation() {
            if (this.timeAnimationInterval) {
                clearInterval(this.timeAnimationInterval);
                this.timeAnimationInterval = null;
            }
        }
        
        advanceTime() {
            const slider = this.container.querySelector('#time-slider');
            if (!slider) return;
            
            const currentValue = parseInt(slider.value);
            const maxValue = parseInt(slider.max);
            
            if (currentValue >= maxValue) {
                this.resetTimeAnimation();
            } else {
                slider.value = currentValue + 1;
                this.setTimePosition(slider.value);
            }
        }
        
        setTimePosition(position) {
            this.state.currentTime = position;
            
            // Update time display
            const timeDisplay = this.container.querySelector('#time-display');
            if (timeDisplay) {
                timeDisplay.textContent = this.formatTimePosition(position);
            }
            
            // Filter features based on time
            this.filterFeaturesByTime(position);
            
            this.emit('time:changed', { position, time: this.state.currentTime });
            
            if (this.options.onTimeChange) {
                this.options.onTimeChange(position);
            }
        }
        
        resetTimeAnimation() {
            this.stopTimeAnimation();
            
            const slider = this.container.querySelector('#time-slider');
            const playPauseBtn = this.container.querySelector('#play-pause');
            
            if (slider) slider.value = 0;
            if (playPauseBtn) playPauseBtn.textContent = '▶️';
            
            this.state.animationPlaying = false;
            this.setTimePosition(0);
        }
        
        setAnimationSpeed(speed) {
            if (this.state.animationPlaying) {
                this.stopTimeAnimation();
                this.startTimeAnimation();
            }
        }
        
        formatTimePosition(position) {
            // Convert position to actual date based on time range
            if (this.state.timeRange) {
                const { start, end } = this.state.timeRange;
                const totalTime = end - start;
                const currentTime = start + (totalTime * position / 100);
                return new Date(currentTime).toLocaleDateString();
            }
            
            return new Date().toLocaleDateString();
        }
        
        filterFeaturesByTime(position) {
            // Implementation depends on data structure and requirements
            console.log('Filtering features by time position:', position);
        }
        
        // Event handling
        
        handleMapClick(e) {
            const features = this.getFeatures(e);
            
            if (features.length > 0) {
                this.showFeatureInfo(features[0], e);
                
                if (this.options.onFeatureClick) {
                    this.options.onFeatureClick(features[0], e);
                }
                
                this.emit('feature:click', { feature: features[0], event: e });
            } else {
                this.hideInfoPanel();
            }
        }
        
        handleMapHover(e) {
            const features = this.getFeatures(e);
            
            if (features.length > 0) {
                this.container.style.cursor = 'pointer';
                
                if (this.options.onFeatureHover) {
                    this.options.onFeatureHover(features[0], e);
                }
                
                this.emit('feature:hover', { feature: features[0], event: e });
            } else {
                this.container.style.cursor = '';
            }
        }
        
        getFeatures(e) {
            // Get features at the clicked/hovered point
            if (this.state.currentProvider === 'mapbox') {
                return this.state.map.queryRenderedFeatures(e.point);
            } else {
                // For Leaflet, this would require more complex implementation
                return [];
            }
        }
        
        showFeatureInfo(feature, event) {
            const infoPanel = this.elements.infoPanel;
            const infoContent = this.container.querySelector('#info-content');
            
            if (!infoPanel || !infoContent) return;
            
            // Generate feature information HTML
            let html = '<div class="feature-info">';
            
            if (feature.properties) {
                Object.entries(feature.properties).forEach(([key, value]) => {
                    html += `<div class="info-row">
                        <span class="info-label">${key}:</span>
                        <span class="info-value">${value}</span>
                    </div>`;
                });
            }
            
            html += '</div>';
            
            infoContent.innerHTML = html;
            infoPanel.style.display = 'block';
            
            // Position the panel near the click point
            if (event && event.point) {
                infoPanel.style.left = event.point.x + 10 + 'px';
                infoPanel.style.top = event.point.y + 10 + 'px';
            }
        }
        
        hideInfoPanel() {
            if (this.elements.infoPanel) {
                this.elements.infoPanel.style.display = 'none';
            }
        }
        
        // UI helper methods
        
        togglePanel(panel) {
            const isVisible = panel.style.display === 'block';
            
            // Hide all other panels
            this.container.querySelectorAll('.layers-panel, .styles-panel, .drawing-panel').forEach(p => {
                if (p !== panel) p.style.display = 'none';
            });
            
            panel.style.display = isVisible ? 'none' : 'block';
        }
        
        updateLayersPanel() {
            const layersList = this.container.querySelector('#layers-list');
            if (!layersList) return;
            
            layersList.innerHTML = '';
            
            this.state.layers.forEach((layer, layerId) => {
                const layerItem = document.createElement('div');
                layerItem.className = 'layer-item';
                layerItem.innerHTML = `
                    <label>
                        <input type="checkbox" class="layer-toggle" data-layer-id="${layerId}" checked>
                        <span class="layer-name">${layer.name || layerId}</span>
                    </label>
                    <div class="layer-controls">
                        <input type="range" class="opacity-slider" min="0" max="100" value="100" 
                               data-layer-id="${layerId}" title="Opacity">
                        <button class="remove-layer" data-layer-id="${layerId}" title="Remove">🗑️</button>
                    </div>
                `;
                
                layersList.appendChild(layerItem);
            });
            
            // Setup layer control event listeners
            layersList.querySelectorAll('.layer-toggle').forEach(toggle => {
                toggle.addEventListener('change', (e) => {
                    this.toggleLayerVisibility(e.target.dataset.layerId, e.target.checked);
                });
            });
            
            layersList.querySelectorAll('.opacity-slider').forEach(slider => {
                slider.addEventListener('input', (e) => {
                    this.setLayerOpacity(e.target.dataset.layerId, e.target.value / 100);
                });
            });
            
            layersList.querySelectorAll('.remove-layer').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    this.removeLayer(e.target.dataset.layerId);
                });
            });
        }
        
        updateLegend() {
            const legendContent = this.container.querySelector('#legend-content');
            if (!legendContent) return;
            
            legendContent.innerHTML = '';
            
            // Generate legend items based on active layers
            this.state.layers.forEach((layer, layerId) => {
                if (layer.legend) {
                    const legendItem = document.createElement('div');
                    legendItem.className = 'legend-item';
                    legendItem.innerHTML = layer.legend;
                    legendContent.appendChild(legendItem);
                }
            });
        }
        
        toggleLayerVisibility(layerId, visible) {
            // Implementation depends on map provider
            console.log('Toggling layer visibility:', layerId, visible);
        }
        
        setLayerOpacity(layerId, opacity) {
            // Implementation depends on map provider
            console.log('Setting layer opacity:', layerId, opacity);
        }
        
        showError(message) {
            const errorElement = this.elements.error;
            const errorMessage = this.container.querySelector('#error-message');
            
            if (errorElement && errorMessage) {
                errorMessage.textContent = message;
                errorElement.style.display = 'flex';
                
                // Setup retry button
                const retryBtn = this.container.querySelector('#retry-btn');
                retryBtn?.addEventListener('click', () => {
                    errorElement.style.display = 'none';
                    this.init();
                });
            }
        }
        
        // Utility methods
        
        generateId() {
            return 'map_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        emit(event, data) {
            this.eventBus.dispatchEvent(new CustomEvent(event, { detail: data }));
        }
        
        on(event, callback) {
            this.eventBus.addEventListener(event, callback);
        }
        
        off(event, callback) {
            this.eventBus.removeEventListener(event, callback);
        }
        
        injectStyles() {
            if (document.getElementById('advanced-map-styles')) return;
            
            const styles = document.createElement('style');
            styles.id = 'advanced-map-styles';
            styles.textContent = `
                .advanced-map-system {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    background: #f0f0f0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                }
                
                .map-loading {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: var(--bg-secondary, #ffffff);
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                }
                
                .loading-spinner {
                    width: 48px;
                    height: 48px;
                    border: 4px solid var(--border, #e0e0e0);
                    border-top: 4px solid var(--accent, #007bff);
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin-bottom: 16px;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .loading-text {
                    color: var(--text-secondary, #666);
                    font-size: 14px;
                }
                
                .map-container {
                    width: 100%;
                    height: 100%;
                }
                
                .map-controls {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    z-index: 500;
                }
                
                .control-group {
                    position: relative;
                }
                
                .control-btn {
                    width: 40px;
                    height: 40px;
                    background: var(--bg-secondary, #ffffff);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 18px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: all 0.2s;
                }
                
                .control-btn:hover {
                    background: var(--bg-tertiary, #f8f9fa);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }
                
                .control-btn.active {
                    background: var(--accent, #007bff);
                    color: white;
                    border-color: var(--accent, #007bff);
                }
                
                .layers-panel, .styles-panel, .drawing-panel {
                    position: absolute;
                    top: 0;
                    right: 50px;
                    background: var(--bg-secondary, #ffffff);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    min-width: 200px;
                    display: none;
                    z-index: 600;
                }
                
                .panel-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px 16px;
                    border-bottom: 1px solid var(--border, #e0e0e0);
                    background: var(--bg-tertiary, #f8f9fa);
                }
                
                .panel-header h4 {
                    margin: 0;
                    font-size: 14px;
                    font-weight: 600;
                }
                
                .close-btn {
                    background: none;
                    border: none;
                    font-size: 18px;
                    cursor: pointer;
                    color: var(--text-secondary, #666);
                }
                
                .layers-list {
                    padding: 8px;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .layer-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px;
                    border-radius: 4px;
                    margin: 4px 0;
                }
                
                .layer-item:hover {
                    background: var(--bg-tertiary, #f8f9fa);
                }
                
                .layer-controls {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .opacity-slider {
                    width: 60px;
                }
                
                .remove-layer {
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 14px;
                    color: var(--error, #dc3545);
                }
                
                .styles-grid {
                    padding: 16px;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 8px;
                }
                
                .style-option {
                    padding: 8px 12px;
                    background: var(--bg-tertiary, #f8f9fa);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    text-align: center;
                    transition: all 0.2s;
                }
                
                .style-option:hover {
                    background: var(--accent, #007bff);
                    color: white;
                }
                
                .style-option.active {
                    background: var(--accent, #007bff);
                    color: white;
                    border-color: var(--accent, #007bff);
                }
                
                .drawing-tools {
                    padding: 16px;
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                
                .tool-btn {
                    padding: 8px 12px;
                    background: var(--bg-tertiary, #f8f9fa);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    text-align: left;
                    transition: all 0.2s;
                }
                
                .tool-btn:hover {
                    background: var(--accent, #007bff);
                    color: white;
                }
                
                .tool-btn.active {
                    background: var(--accent, #007bff);
                    color: white;
                }
                
                .tool-btn.clear-btn {
                    background: var(--error, #dc3545);
                    color: white;
                    border-color: var(--error, #dc3545);
                }
                
                .map-info-panel {
                    position: absolute;
                    background: var(--bg-secondary, #ffffff);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    padding: 16px;
                    min-width: 200px;
                    max-width: 300px;
                    z-index: 700;
                }
                
                .close-info {
                    position: absolute;
                    top: 8px;
                    right: 8px;
                    background: none;
                    border: none;
                    font-size: 16px;
                    cursor: pointer;
                    color: var(--text-secondary, #666);
                }
                
                .feature-info {
                    font-size: 13px;
                }
                
                .info-row {
                    margin: 8px 0;
                    display: flex;
                    justify-content: space-between;
                }
                
                .info-label {
                    font-weight: 600;
                    color: var(--text-secondary, #666);
                }
                
                .info-value {
                    color: var(--text-primary, #333);
                }
                
                .time-controls {
                    position: absolute;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: var(--bg-secondary, #ffffff);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 8px;
                    padding: 16px;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 500;
                }
                
                .time-slider-container {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .time-btn {
                    background: var(--bg-tertiary, #f8f9fa);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 4px;
                    padding: 8px;
                    cursor: pointer;
                    font-size: 14px;
                }
                
                .time-btn:hover {
                    background: var(--accent, #007bff);
                    color: white;
                }
                
                .time-slider {
                    width: 200px;
                }
                
                .time-display {
                    font-weight: 600;
                    min-width: 80px;
                    text-align: center;
                }
                
                .time-speed label {
                    font-size: 12px;
                    margin-right: 4px;
                }
                
                .map-legend {
                    position: absolute;
                    bottom: 20px;
                    left: 20px;
                    background: var(--bg-secondary, #ffffff);
                    border: 1px solid var(--border, #ccc);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    min-width: 150px;
                    z-index: 500;
                }
                
                .legend-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px 16px;
                    border-bottom: 1px solid var(--border, #e0e0e0);
                    background: var(--bg-tertiary, #f8f9fa);
                }
                
                .legend-header h4 {
                    margin: 0;
                    font-size: 14px;
                    font-weight: 600;
                }
                
                .toggle-legend {
                    background: none;
                    border: none;
                    font-size: 16px;
                    cursor: pointer;
                }
                
                .legend-content {
                    padding: 12px 16px;
                    max-height: 200px;
                    overflow-y: auto;
                }
                
                .legend-item {
                    margin: 8px 0;
                    font-size: 12px;
                }
                
                .map-error {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: var(--bg-secondary, #ffffff);
                    border: 1px solid var(--error, #dc3545);
                    border-radius: 8px;
                    padding: 24px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 1000;
                }
                
                .error-icon {
                    font-size: 48px;
                    margin-bottom: 16px;
                }
                
                .error-message {
                    color: var(--error, #dc3545);
                    margin-bottom: 16px;
                    font-weight: 600;
                }
                
                .retry-btn {
                    background: var(--accent, #007bff);
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-weight: 600;
                }
                
                .advanced-map-system.fullscreen {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    z-index: 9999;
                }
                
                .advanced-map-system.drawing-mode {
                    cursor: crosshair;
                }
                
                /* Responsive design */
                @media (max-width: 768px) {
                    .map-controls {
                        top: 10px;
                        right: 10px;
                        gap: 8px;
                    }
                    
                    .control-btn {
                        width: 36px;
                        height: 36px;
                        font-size: 16px;
                    }
                    
                    .layers-panel, .styles-panel, .drawing-panel {
                        right: 46px;
                        min-width: 180px;
                    }
                    
                    .time-controls {
                        bottom: 10px;
                        left: 10px;
                        right: 10px;
                        transform: none;
                        flex-wrap: wrap;
                        padding: 12px;
                    }
                    
                    .time-slider {
                        width: 150px;
                    }
                    
                    .map-legend {
                        bottom: 10px;
                        left: 10px;
                        right: 10px;
                        max-width: none;
                    }
                }
            `;
            
            document.head.appendChild(styles);
        }
        
        destroy() {
            // Clean up resources
            if (this.timeAnimationInterval) {
                clearInterval(this.timeAnimationInterval);
            }
            
            if (this.state.map) {
                if (this.state.currentProvider === 'mapbox') {
                    this.state.map.remove();
                } else {
                    this.state.map.remove();
                }
            }
            
            this.container.innerHTML = '';
        }
    }
    
    // Export to global scope
    global.AdvancedMapSystem = AdvancedMapSystem;
    
    console.log('✨ Advanced Map System ready. Access via window.AdvancedMapSystem');
    
})(window);