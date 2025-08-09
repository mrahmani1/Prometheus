/**
 * Map Controls - Task 2.4
 * Advanced control widgets and tools for the map system
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🕹️ Map Controls - Initializing advanced control widgets...');
    
    // Base Control Class
    class MapControl {
        constructor(map, options = {}) {
            this.map = map;
            this.options = {
                position: 'top-right',
                className: '',
                ...options
            };
            
            this.element = null;
            this.visible = true;
            this.enabled = true;
            
            this.create();
        }
        
        create() {
            this.element = document.createElement('div');
            this.element.className = `map-control ${this.options.className}`;
            this.render();
            this.bindEvents();
            this.addToMap();
        }
        
        render() {
            // Override in subclasses
        }
        
        bindEvents() {
            // Override in subclasses
        }
        
        addToMap() {
            if (!this.map || !this.map.container) return;
            
            const container = this.map.container;
            let controlContainer = container.querySelector(`.controls-${this.options.position}`);
            
            if (!controlContainer) {
                controlContainer = document.createElement('div');
                controlContainer.className = `map-controls-container controls-${this.options.position}`;
                container.appendChild(controlContainer);
            }
            
            controlContainer.appendChild(this.element);
        }
        
        show() {
            this.visible = true;
            this.element.style.display = '';
        }
        
        hide() {
            this.visible = false;
            this.element.style.display = 'none';
        }
        
        enable() {
            this.enabled = true;
            this.element.classList.remove('disabled');
        }
        
        disable() {
            this.enabled = false;
            this.element.classList.add('disabled');
        }
        
        destroy() {
            if (this.element) {
                this.element.remove();
            }
        }
    }
    
    // Navigation Control
    class NavigationControl extends MapControl {
        constructor(map, options = {}) {
            super(map, {
                showZoom: true,
                showCompass: true,
                showPitch: true,
                ...options
            });
        }
        
        render() {
            let html = '<div class="navigation-control">';
            
            if (this.options.showZoom) {
                html += `
                    <div class="zoom-controls">
                        <button class="zoom-in" title="Zoom In">+</button>
                        <button class="zoom-out" title="Zoom Out">−</button>
                    </div>
                `;
            }
            
            if (this.options.showCompass) {
                html += `
                    <div class="compass-control" title="Reset Bearing">
                        <div class="compass-needle">⇧</div>
                    </div>
                `;
            }
            
            if (this.options.showPitch) {
                html += `
                    <div class="pitch-controls">
                        <button class="pitch-up" title="Increase Pitch">⬆</button>
                        <button class="pitch-down" title="Decrease Pitch">⬇</button>
                    </div>
                `;
            }
            
            html += '</div>';
            this.element.innerHTML = html;
        }
        
        bindEvents() {
            const zoomIn = this.element.querySelector('.zoom-in');
            const zoomOut = this.element.querySelector('.zoom-out');
            const compass = this.element.querySelector('.compass-control');
            const pitchUp = this.element.querySelector('.pitch-up');
            const pitchDown = this.element.querySelector('.pitch-down');
            
            zoomIn?.addEventListener('click', () => {
                if (this.map.state.map) {
                    if (this.map.state.currentProvider === 'mapbox') {
                        this.map.state.map.zoomIn();
                    } else {
                        this.map.state.map.zoomIn();
                    }
                }
            });
            
            zoomOut?.addEventListener('click', () => {
                if (this.map.state.map) {
                    if (this.map.state.currentProvider === 'mapbox') {
                        this.map.state.map.zoomOut();
                    } else {
                        this.map.state.map.zoomOut();
                    }
                }
            });
            
            compass?.addEventListener('click', () => {
                if (this.map.state.map && this.map.state.currentProvider === 'mapbox') {
                    this.map.state.map.easeTo({ bearing: 0, pitch: 0 });
                }
            });
            
            pitchUp?.addEventListener('click', () => {
                if (this.map.state.map && this.map.state.currentProvider === 'mapbox') {
                    const currentPitch = this.map.state.map.getPitch();
                    this.map.state.map.easeTo({ pitch: Math.min(60, currentPitch + 15) });
                }
            });
            
            pitchDown?.addEventListener('click', () => {
                if (this.map.state.map && this.map.state.currentProvider === 'mapbox') {
                    const currentPitch = this.map.state.map.getPitch();
                    this.map.state.map.easeTo({ pitch: Math.max(0, currentPitch - 15) });
                }
            });
        }
    }
    
    // Scale Control
    class ScaleControl extends MapControl {
        constructor(map, options = {}) {
            super(map, {
                position: 'bottom-left',
                unit: 'metric',
                maxWidth: 100,
                ...options
            });
            
            this.updateScale = this.updateScale.bind(this);
        }
        
        render() {
            this.element.innerHTML = `
                <div class="scale-control">
                    <div class="scale-line"></div>
                    <div class="scale-text">0 km</div>
                </div>
            `;
            
            this.scaleLine = this.element.querySelector('.scale-line');
            this.scaleText = this.element.querySelector('.scale-text');
            
            // Update scale when map moves
            if (this.map.state.map) {
                if (this.map.state.currentProvider === 'mapbox') {
                    this.map.state.map.on('zoom', this.updateScale);
                    this.map.state.map.on('move', this.updateScale);
                } else {
                    this.map.state.map.on('zoom', this.updateScale);
                    this.map.state.map.on('move', this.updateScale);
                }
            }
            
            this.updateScale();
        }
        
        updateScale() {
            if (!this.map.state.map) return;
            
            try {
                let distance, unit;
                
                if (this.map.state.currentProvider === 'mapbox') {
                    const zoom = this.map.state.map.getZoom();
                    const lat = this.map.state.map.getCenter().lat;
                    
                    // Calculate scale based on zoom and latitude
                    const metersPerPixel = this.getMetersPerPixel(zoom, lat);
                    distance = metersPerPixel * this.options.maxWidth;
                    
                } else {
                    // For Leaflet
                    const center = this.map.state.map.getCenter();
                    const zoom = this.map.state.map.getZoom();
                    const metersPerPixel = this.getMetersPerPixel(zoom, center.lat);
                    distance = metersPerPixel * this.options.maxWidth;
                }
                
                // Convert to appropriate unit
                if (this.options.unit === 'metric') {
                    if (distance >= 1000) {
                        distance = distance / 1000;
                        unit = 'km';
                    } else {
                        unit = 'm';
                    }
                } else {
                    // Imperial
                    distance = distance * 3.28084; // Convert to feet
                    if (distance >= 5280) {
                        distance = distance / 5280;
                        unit = 'mi';
                    } else {
                        unit = 'ft';
                    }
                }
                
                // Round to nice numbers
                const rounded = this.getRoundedDistance(distance);
                const width = (rounded / distance) * this.options.maxWidth;
                
                this.scaleLine.style.width = width + 'px';
                this.scaleText.textContent = `${rounded} ${unit}`;
                
            } catch (error) {
                console.warn('Failed to update scale:', error);
            }
        }
        
        getMetersPerPixel(zoom, latitude) {
            const earthCircumference = 40075017;
            const latitudeRadians = latitude * (Math.PI / 180);
            return earthCircumference * Math.cos(latitudeRadians) / Math.pow(2, zoom + 8);
        }
        
        getRoundedDistance(distance) {
            const pow10 = Math.pow(10, Math.floor(Math.log(distance) / Math.LN10));
            let normalized = distance / pow10;
            
            if (normalized >= 5) normalized = 10;
            else if (normalized >= 2) normalized = 5;
            else if (normalized >= 1) normalized = 2;
            else normalized = 1;
            
            return normalized * pow10;
        }
    }
    
    // Layer Control
    class LayerControl extends MapControl {
        constructor(map, options = {}) {
            super(map, {
                position: 'top-right',
                baseLayers: {},
                overlayLayers: {},
                collapsed: true,
                ...options
            });
            
            this.baseLayers = new Map();
            this.overlayLayers = new Map();
        }
        
        render() {
            const isCollapsed = this.options.collapsed;
            
            this.element.innerHTML = `
                <div class="layer-control ${isCollapsed ? 'collapsed' : ''}">
                    <div class="layer-control-toggle">
                        <span class="toggle-icon">🗂️</span>
                    </div>
                    <div class="layer-control-content">
                        <div class="base-layers-section">
                            <h4>Base Layers</h4>
                            <div class="base-layers-list"></div>
                        </div>
                        <div class="overlay-layers-section">
                            <h4>Overlays</h4>
                            <div class="overlay-layers-list"></div>
                        </div>
                    </div>
                </div>
            `;
            
            this.toggleButton = this.element.querySelector('.layer-control-toggle');
            this.content = this.element.querySelector('.layer-control-content');
            this.baseLayersList = this.element.querySelector('.base-layers-list');
            this.overlayLayersList = this.element.querySelector('.overlay-layers-list');
            
            this.populateLayers();
        }
        
        bindEvents() {
            this.toggleButton?.addEventListener('click', () => {
                this.toggle();
            });
            
            // Close when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.element.contains(e.target)) {
                    this.collapse();
                }
            });
        }
        
        toggle() {
            if (this.element.classList.contains('collapsed')) {
                this.expand();
            } else {
                this.collapse();
            }
        }
        
        expand() {
            this.element.classList.remove('collapsed');
        }
        
        collapse() {
            this.element.classList.add('collapsed');
        }
        
        addBaseLayer(layer, name) {
            this.baseLayers.set(name, layer);
            this.populateBaseLayers();
        }
        
        addOverlayLayer(layer, name) {
            this.overlayLayers.set(name, layer);
            this.populateOverlayLayers();
        }
        
        removeLayer(name) {
            this.baseLayers.delete(name);
            this.overlayLayers.delete(name);
            this.populateLayers();
        }
        
        populateLayers() {
            this.populateBaseLayers();
            this.populateOverlayLayers();
        }
        
        populateBaseLayers() {
            if (!this.baseLayersList) return;
            
            this.baseLayersList.innerHTML = '';
            
            this.baseLayers.forEach((layer, name) => {
                const item = document.createElement('div');
                item.className = 'layer-item';
                item.innerHTML = `
                    <label>
                        <input type="radio" name="base-layer" value="${name}">
                        <span class="layer-name">${name}</span>
                    </label>
                `;
                
                const radio = item.querySelector('input');
                radio.addEventListener('change', () => {
                    if (radio.checked) {
                        this.setBaseLayer(name, layer);
                    }
                });
                
                this.baseLayersList.appendChild(item);
            });
        }
        
        populateOverlayLayers() {
            if (!this.overlayLayersList) return;
            
            this.overlayLayersList.innerHTML = '';
            
            this.overlayLayers.forEach((layer, name) => {
                const item = document.createElement('div');
                item.className = 'layer-item';
                item.innerHTML = `
                    <label>
                        <input type="checkbox" value="${name}">
                        <span class="layer-name">${name}</span>
                    </label>
                    <div class="layer-controls">
                        <input type="range" class="opacity-slider" min="0" max="100" value="100" title="Opacity">
                    </div>
                `;
                
                const checkbox = item.querySelector('input[type="checkbox"]');
                const opacitySlider = item.querySelector('.opacity-slider');
                
                checkbox.addEventListener('change', () => {
                    if (checkbox.checked) {
                        this.addOverlayToMap(name, layer);
                    } else {
                        this.removeOverlayFromMap(name, layer);
                    }
                });
                
                opacitySlider.addEventListener('input', () => {
                    this.setLayerOpacity(name, layer, opacitySlider.value / 100);
                });
                
                this.overlayLayersList.appendChild(item);
            });
        }
        
        setBaseLayer(name, layer) {
            // Implementation depends on map provider
            console.log('Setting base layer:', name);
        }
        
        addOverlayToMap(name, layer) {
            // Implementation depends on map provider
            console.log('Adding overlay layer:', name);
        }
        
        removeOverlayFromMap(name, layer) {
            // Implementation depends on map provider
            console.log('Removing overlay layer:', name);
        }
        
        setLayerOpacity(name, layer, opacity) {
            // Implementation depends on map provider
            console.log('Setting layer opacity:', name, opacity);
        }
    }
    
    // Draw Control
    class DrawControl extends MapControl {
        constructor(map, options = {}) {
            super(map, {
                position: 'top-left',
                tools: ['point', 'line', 'polygon', 'rectangle', 'circle'],
                showMeasurements: true,
                ...options
            });
            
            this.activetool = null;
            this.isDrawing = false;
            this.currentFeature = null;
            this.drawnFeatures = [];
            this.measurements = new Map();
        }
        
        render() {
            let html = '<div class="draw-control">';
            
            this.options.tools.forEach(tool => {
                const icons = {
                    point: '📍',
                    line: '📏',
                    polygon: '⬟',
                    rectangle: '⬛',
                    circle: '⭕',
                    edit: '✏️',
                    delete: '🗑️'
                };
                
                html += `
                    <button class="draw-tool" data-tool="${tool}" title="${tool.charAt(0).toUpperCase() + tool.slice(1)}">
                        ${icons[tool] || '🔧'}
                    </button>
                `;
            });
            
            html += `
                <button class="draw-tool" data-tool="edit" title="Edit">✏️</button>
                <button class="draw-tool" data-tool="delete" title="Delete All">🗑️</button>
            `;
            
            html += '</div>';
            this.element.innerHTML = html;
        }
        
        bindEvents() {
            this.element.querySelectorAll('.draw-tool').forEach(button => {
                button.addEventListener('click', () => {
                    const tool = button.dataset.tool;
                    
                    if (tool === 'delete') {
                        this.deleteAllFeatures();
                    } else if (tool === 'edit') {
                        this.toggleEditMode();
                    } else {
                        this.setActiveTool(tool);
                    }
                });
            });
        }
        
        setActiveTool(tool) {
            // Deactivate previous tool
            if (this.activetool) {
                this.deactivateTool(this.activeTool);
            }
            
            this.activeTree = tool;
            
            // Update UI
            this.element.querySelectorAll('.draw-tool').forEach(button => {
                button.classList.toggle('active', button.dataset.tool === tool);
            });
            
            // Activate new tool
            this.activateTool(tool);
            
            // Change cursor
            if (this.map.container) {
                this.map.container.style.cursor = 'crosshair';
            }
        }
        
        activateTool(tool) {
            // Setup event listeners based on tool
            if (this.map.state.map) {
                if (this.map.state.currentProvider === 'mapbox') {
                    this.activateMapboxTool(tool);
                } else {
                    this.activateLeafletTool(tool);
                }
            }
        }
        
        deactivateTool(tool) {
            this.isDrawing = false;
            this.currentFeature = null;
            
            // Reset cursor
            if (this.map.container) {
                this.map.container.style.cursor = '';
            }
            
            // Remove event listeners
            if (this.map.state.currentProvider === 'mapbox') {
                this.deactivateMapboxTool(tool);
            } else {
                this.deactivateLeafletTool(tool);
            }
        }
        
        activateMapboxTool(tool) {
            const map = this.map.state.map;
            
            switch (tool) {
                case 'point':
                    map.on('click', this.onMapClick.bind(this));
                    break;
                case 'line':
                case 'polygon':
                    map.on('click', this.onMapClick.bind(this));
                    map.on('dblclick', this.onMapDoubleClick.bind(this));
                    break;
                case 'rectangle':
                case 'circle':
                    map.on('mousedown', this.onMapMouseDown.bind(this));
                    break;
            }
        }
        
        deactivateMapboxTool(tool) {
            const map = this.map.state.map;
            
            map.off('click', this.onMapClick.bind(this));
            map.off('dblclick', this.onMapDoubleClick.bind(this));
            map.off('mousedown', this.onMapMouseDown.bind(this));
            map.off('mousemove', this.onMapMouseMove.bind(this));
            map.off('mouseup', this.onMapMouseUp.bind(this));
        }
        
        activateLeafletTool(tool) {
            // Implement Leaflet drawing tools
            console.log('Activating Leaflet tool:', tool);
        }
        
        deactivateLeafletTool(tool) {
            // Implement Leaflet drawing tools cleanup
            console.log('Deactivating Leaflet tool:', tool);
        }
        
        onMapClick(e) {
            if (!this.activeTree) return;
            
            switch (this.activeTree) {
                case 'point':
                    this.createPoint(e.lngLat || [e.latlng.lng, e.latlng.lat]);
                    break;
                case 'line':
                case 'polygon':
                    this.addPointToFeature(e.lngLat || [e.latlng.lng, e.latlng.lat]);
                    break;
            }
        }
        
        onMapDoubleClick(e) {
            if (this.activeTree === 'line' || this.activeTree === 'polygon') {
                this.finishDrawing();
            }
        }
        
        onMapMouseDown(e) {
            if (this.activeTree === 'rectangle' || this.activeTree === 'circle') {
                this.startAreaDrawing(e.lngLat || [e.latlng.lng, e.latlng.lat]);
            }
        }
        
        onMapMouseMove(e) {
            if (this.isDrawing && (this.activeTree === 'rectangle' || this.activeTree === 'circle')) {
                this.updateAreaDrawing(e.lngLat || [e.latlng.lng, e.latlng.lat]);
            }
        }
        
        onMapMouseUp(e) {
            if (this.isDrawing) {
                this.finishAreaDrawing(e.lngLat || [e.latlng.lng, e.latlng.lat]);
            }
        }
        
        createPoint(coordinates) {
            const feature = {
                type: 'Feature',
                geometry: {
                    type: 'Point',
                    coordinates: coordinates
                },
                properties: {
                    id: this.generateId(),
                    type: 'point',
                    created: Date.now()
                }
            };
            
            this.addFeature(feature);
        }
        
        addPointToFeature(coordinates) {
            if (!this.currentFeature) {
                // Start new feature
                const geometryType = this.activeTree === 'line' ? 'LineString' : 'Polygon';
                this.currentFeature = {
                    type: 'Feature',
                    geometry: {
                        type: geometryType,
                        coordinates: geometryType === 'Polygon' ? [[coordinates]] : [coordinates]
                    },
                    properties: {
                        id: this.generateId(),
                        type: this.activeTree,
                        created: Date.now()
                    }
                };
            } else {
                // Add point to existing feature
                if (this.activeTree === 'polygon') {
                    this.currentFeature.geometry.coordinates[0].push(coordinates);
                } else {
                    this.currentFeature.geometry.coordinates.push(coordinates);
                }
            }
            
            this.updateFeatureVisualization();
        }
        
        startAreaDrawing(startCoords) {
            this.isDrawing = true;
            this.currentFeature = {
                type: 'Feature',
                geometry: {
                    type: 'Polygon',
                    coordinates: [[]]
                },
                properties: {
                    id: this.generateId(),
                    type: this.activeTree,
                    created: Date.now(),
                    startCoords: startCoords
                }
            };
        }
        
        updateAreaDrawing(currentCoords) {
            if (!this.currentFeature) return;
            
            const startCoords = this.currentFeature.properties.startCoords;
            
            if (this.activeTree === 'rectangle') {
                this.currentFeature.geometry.coordinates = [[
                    startCoords,
                    [currentCoords[0], startCoords[1]],
                    currentCoords,
                    [startCoords[0], currentCoords[1]],
                    startCoords
                ]];
            } else if (this.activeTree === 'circle') {
                const radius = this.calculateDistance(startCoords, currentCoords);
                this.currentFeature.geometry.coordinates = [this.createCircleCoordinates(startCoords, radius)];
            }
            
            this.updateFeatureVisualization();
        }
        
        finishAreaDrawing() {
            this.isDrawing = false;
            if (this.currentFeature) {
                this.addFeature(this.currentFeature);
                this.currentFeature = null;
            }
        }
        
        finishDrawing() {
            if (this.currentFeature) {
                // Close polygon if needed
                if (this.activeTree === 'polygon') {
                    const coords = this.currentFeature.geometry.coordinates[0];
                    if (coords.length > 2) {
                        coords.push(coords[0]); // Close the polygon
                    }
                }
                
                this.addFeature(this.currentFeature);
                this.currentFeature = null;
            }
        }
        
        addFeature(feature) {
            this.drawnFeatures.push(feature);
            
            // Calculate measurements if enabled
            if (this.options.showMeasurements) {
                this.calculateMeasurements(feature);
            }
            
            // Add to map visualization
            this.addFeatureToMap(feature);
            
            // Emit event
            this.map.emit('draw:feature-created', { feature });
        }
        
        addFeatureToMap(feature) {
            // Implementation depends on map provider
            console.log('Adding feature to map:', feature);
        }
        
        updateFeatureVisualization() {
            // Update the visual representation of the current feature being drawn
            console.log('Updating feature visualization');
        }
        
        calculateMeasurements(feature) {
            const geometry = feature.geometry;
            let measurement = '';
            
            switch (geometry.type) {
                case 'Point':
                    const coords = geometry.coordinates;
                    measurement = `${coords[1].toFixed(6)}, ${coords[0].toFixed(6)}`;
                    break;
                    
                case 'LineString':
                    const length = this.calculateLineLength(geometry.coordinates);
                    measurement = this.formatDistance(length);
                    break;
                    
                case 'Polygon':
                    const area = this.calculatePolygonArea(geometry.coordinates[0]);
                    const perimeter = this.calculateLineLength(geometry.coordinates[0]);
                    measurement = `Area: ${this.formatArea(area)}, Perimeter: ${this.formatDistance(perimeter)}`;
                    break;
            }
            
            this.measurements.set(feature.properties.id, measurement);
            feature.properties.measurement = measurement;
        }
        
        calculateLineLength(coordinates) {
            let length = 0;
            for (let i = 1; i < coordinates.length; i++) {
                length += this.calculateDistance(coordinates[i-1], coordinates[i]);
            }
            return length;
        }
        
        calculatePolygonArea(coordinates) {
            // Using the shoelace formula
            let area = 0;
            const n = coordinates.length - 1; // Exclude closing point
            
            for (let i = 0; i < n; i++) {
                const j = (i + 1) % n;
                area += coordinates[i][0] * coordinates[j][1];
                area -= coordinates[j][0] * coordinates[i][1];
            }
            
            area = Math.abs(area) / 2;
            
            // Convert to square meters (approximate)
            const lat = coordinates[0][1];
            const metersPerDegree = 111320 * Math.cos(lat * Math.PI / 180);
            return area * metersPerDegree * metersPerDegree;
        }
        
        calculateDistance(coord1, coord2) {
            // Haversine formula
            const R = 6371000; // Earth's radius in meters
            const lat1 = coord1[1] * Math.PI / 180;
            const lat2 = coord2[1] * Math.PI / 180;
            const deltaLat = (coord2[1] - coord1[1]) * Math.PI / 180;
            const deltaLng = (coord2[0] - coord1[0]) * Math.PI / 180;
            
            const a = Math.sin(deltaLat/2) * Math.sin(deltaLat/2) +
                      Math.cos(lat1) * Math.cos(lat2) *
                      Math.sin(deltaLng/2) * Math.sin(deltaLng/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            
            return R * c;
        }
        
        createCircleCoordinates(center, radius, points = 64) {
            const coordinates = [];
            for (let i = 0; i < points; i++) {
                const angle = (i * 360 / points) * Math.PI / 180;
                const lat = center[1] + (radius / 111320) * Math.cos(angle);
                const lng = center[0] + (radius / (111320 * Math.cos(center[1] * Math.PI / 180))) * Math.sin(angle);
                coordinates.push([lng, lat]);
            }
            coordinates.push(coordinates[0]); // Close the circle
            return coordinates;
        }
        
        formatDistance(meters) {
            if (meters >= 1000) {
                return (meters / 1000).toFixed(2) + ' km';
            } else {
                return meters.toFixed(0) + ' m';
            }
        }
        
        formatArea(sqMeters) {
            if (sqMeters >= 1000000) {
                return (sqMeters / 1000000).toFixed(2) + ' km²';
            } else if (sqMeters >= 10000) {
                return (sqMeters / 10000).toFixed(2) + ' ha';
            } else {
                return sqMeters.toFixed(0) + ' m²';
            }
        }
        
        toggleEditMode() {
            console.log('Toggle edit mode');
        }
        
        deleteAllFeatures() {
            this.drawnFeatures = [];
            this.measurements.clear();
            this.currentFeature = null;
            
            // Remove from map
            console.log('Deleting all drawn features');
            
            this.map.emit('draw:features-cleared');
        }
        
        generateId() {
            return 'draw_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
    }
    
    // Geocoder Control
    class GeocoderControl extends MapControl {
        constructor(map, options = {}) {
            super(map, {
                position: 'top-left',
                placeholder: 'Search for places...',
                limit: 5,
                ...options
            });
            
            this.results = [];
            this.selectedIndex = -1;
        }
        
        render() {
            this.element.innerHTML = `
                <div class="geocoder-control">
                    <div class="geocoder-input-container">
                        <input type="text" class="geocoder-input" placeholder="${this.options.placeholder}">
                        <button class="geocoder-clear" style="display: none;">×</button>
                    </div>
                    <div class="geocoder-results" style="display: none;"></div>
                </div>
            `;
            
            this.input = this.element.querySelector('.geocoder-input');
            this.clearButton = this.element.querySelector('.geocoder-clear');
            this.resultsContainer = this.element.querySelector('.geocoder-results');
        }
        
        bindEvents() {
            this.input?.addEventListener('input', (e) => {
                this.handleInput(e.target.value);
            });
            
            this.input?.addEventListener('keydown', (e) => {
                this.handleKeyDown(e);
            });
            
            this.clearButton?.addEventListener('click', () => {
                this.clear();
            });
            
            // Close results when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.element.contains(e.target)) {
                    this.hideResults();
                }
            });
        }
        
        handleInput(value) {
            if (value.length > 0) {
                this.clearButton.style.display = 'block';
                this.search(value);
            } else {
                this.clearButton.style.display = 'none';
                this.hideResults();
            }
        }
        
        handleKeyDown(e) {
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    this.selectNext();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.selectPrevious();
                    break;
                case 'Enter':
                    e.preventDefault();
                    this.selectCurrent();
                    break;
                case 'Escape':
                    this.hideResults();
                    break;
            }
        }
        
        async search(query) {
            try {
                // Use different geocoding services
                let results = [];
                
                // Try Mapbox Geocoding API if available
                if (this.options.mapboxToken) {
                    results = await this.searchMapbox(query);
                } else {
                    // Fallback to Nominatim
                    results = await this.searchNominatim(query);
                }
                
                this.results = results.slice(0, this.options.limit);
                this.selectedIndex = -1;
                this.showResults();
                
            } catch (error) {
                console.error('Geocoding search failed:', error);
                this.results = [];
                this.hideResults();
            }
        }
        
        async searchMapbox(query) {
            const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json?access_token=${this.options.mapboxToken}&limit=${this.options.limit}`;
            const response = await fetch(url);
            const data = await response.json();
            
            return data.features.map(feature => ({
                name: feature.place_name,
                coordinates: feature.center,
                bbox: feature.bbox
            }));
        }
        
        async searchNominatim(query) {
            const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=${this.options.limit}`;
            const response = await fetch(url);
            const data = await response.json();
            
            return data.map(item => ({
                name: item.display_name,
                coordinates: [parseFloat(item.lon), parseFloat(item.lat)],
                bbox: item.boundingbox ? [
                    parseFloat(item.boundingbox[2]),
                    parseFloat(item.boundingbox[0]),
                    parseFloat(item.boundingbox[3]),
                    parseFloat(item.boundingbox[1])
                ] : null
            }));
        }
        
        showResults() {
            if (this.results.length === 0) {
                this.hideResults();
                return;
            }
            
            this.resultsContainer.innerHTML = '';
            
            this.results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'geocoder-result-item';
                item.textContent = result.name;
                
                item.addEventListener('click', () => {
                    this.selectResult(result);
                });
                
                this.resultsContainer.appendChild(item);
            });
            
            this.resultsContainer.style.display = 'block';
        }
        
        hideResults() {
            this.resultsContainer.style.display = 'none';
            this.selectedIndex = -1;
        }
        
        selectNext() {
            if (this.results.length === 0) return;
            
            this.selectedIndex = Math.min(this.selectedIndex + 1, this.results.length - 1);
            this.updateSelectedResult();
        }
        
        selectPrevious() {
            if (this.results.length === 0) return;
            
            this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
            this.updateSelectedResult();
        }
        
        updateSelectedResult() {
            const items = this.resultsContainer.querySelectorAll('.geocoder-result-item');
            items.forEach((item, index) => {
                item.classList.toggle('selected', index === this.selectedIndex);
            });
        }
        
        selectCurrent() {
            if (this.selectedIndex >= 0 && this.selectedIndex < this.results.length) {
                this.selectResult(this.results[this.selectedIndex]);
            }
        }
        
        selectResult(result) {
            this.input.value = result.name;
            this.hideResults();
            
            // Fly to location
            if (result.bbox) {
                this.map.fitBounds(result.bbox);
            } else {
                this.map.flyTo(result.coordinates, 14);
            }
            
            this.map.emit('geocoder:result-selected', { result });
        }
        
        clear() {
            this.input.value = '';
            this.clearButton.style.display = 'none';
            this.hideResults();
            this.results = [];
        }
    }
    
    // Add control styles
    const controlStyles = document.createElement('style');
    controlStyles.id = 'map-control-styles';
    controlStyles.textContent = `
        .map-controls-container {
            position: absolute;
            pointer-events: none;
            z-index: 1000;
        }
        
        .map-controls-container > * {
            pointer-events: auto;
        }
        
        .controls-top-left {
            top: 10px;
            left: 10px;
        }
        
        .controls-top-right {
            top: 10px;
            right: 10px;
        }
        
        .controls-bottom-left {
            bottom: 10px;
            left: 10px;
        }
        
        .controls-bottom-right {
            bottom: 10px;
            right: 10px;
        }
        
        .map-control {
            margin: 5px;
        }
        
        /* Navigation Control */
        .navigation-control {
            background: var(--bg-secondary, #ffffff);
            border: 1px solid var(--border, #ccc);
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .zoom-controls, .pitch-controls {
            display: flex;
            flex-direction: column;
        }
        
        .zoom-controls button, .pitch-controls button {
            width: 30px;
            height: 30px;
            background: var(--bg-secondary, #ffffff);
            border: none;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s;
        }
        
        .zoom-controls button:hover, .pitch-controls button:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        .zoom-controls button:not(:last-child), .pitch-controls button:not(:last-child) {
            border-bottom: 1px solid var(--border, #e0e0e0);
        }
        
        .compass-control {
            width: 30px;
            height: 30px;
            background: var(--bg-secondary, #ffffff);
            border-top: 1px solid var(--border, #e0e0e0);
            border-bottom: 1px solid var(--border, #e0e0e0);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: background-color 0.2s;
        }
        
        .compass-control:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        /* Scale Control */
        .scale-control {
            background: var(--bg-secondary, rgba(255, 255, 255, 0.9));
            border: 1px solid var(--border, #ccc);
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }
        
        .scale-line {
            height: 2px;
            background: var(--text-primary, #333);
            margin-bottom: 2px;
            border-right: 2px solid var(--text-primary, #333);
            border-left: 2px solid var(--text-primary, #333);
        }
        
        .scale-text {
            text-align: center;
            color: var(--text-primary, #333);
            font-weight: 600;
        }
        
        /* Layer Control */
        .layer-control {
            position: relative;
        }
        
        .layer-control-toggle {
            width: 40px;
            height: 40px;
            background: var(--bg-secondary, #ffffff);
            border: 1px solid var(--border, #ccc);
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: background-color 0.2s;
        }
        
        .layer-control-toggle:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        .layer-control-content {
            position: absolute;
            top: 0;
            left: 50px;
            background: var(--bg-secondary, #ffffff);
            border: 1px solid var(--border, #ccc);
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            min-width: 200px;
            max-height: 400px;
            overflow-y: auto;
            display: block;
        }
        
        .layer-control.collapsed .layer-control-content {
            display: none;
        }
        
        .base-layers-section, .overlay-layers-section {
            padding: 8px;
        }
        
        .base-layers-section:not(:last-child) {
            border-bottom: 1px solid var(--border, #e0e0e0);
        }
        
        .base-layers-section h4, .overlay-layers-section h4 {
            margin: 0 0 8px 0;
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-secondary, #666);
        }
        
        .layer-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px;
            border-radius: 3px;
            font-size: 13px;
        }
        
        .layer-item:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        .layer-item label {
            display: flex;
            align-items: center;
            cursor: pointer;
        }
        
        .layer-item input {
            margin-right: 6px;
        }
        
        .layer-controls {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .opacity-slider {
            width: 60px;
        }
        
        /* Draw Control */
        .draw-control {
            background: var(--bg-secondary, #ffffff);
            border: 1px solid var(--border, #ccc);
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .draw-tool {
            width: 40px;
            height: 40px;
            background: var(--bg-secondary, #ffffff);
            border: none;
            border-bottom: 1px solid var(--border, #e0e0e0);
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s;
        }
        
        .draw-tool:last-child {
            border-bottom: none;
        }
        
        .draw-tool:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        .draw-tool.active {
            background: var(--accent, #007bff);
            color: white;
        }
        
        /* Geocoder Control */
        .geocoder-control {
            position: relative;
        }
        
        .geocoder-input-container {
            position: relative;
            display: flex;
            align-items: center;
        }
        
        .geocoder-input {
            width: 240px;
            padding: 8px 32px 8px 12px;
            border: 1px solid var(--border, #ccc);
            border-radius: 6px;
            background: var(--bg-secondary, #ffffff);
            font-size: 14px;
            outline: none;
        }
        
        .geocoder-input:focus {
            border-color: var(--accent, #007bff);
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }
        
        .geocoder-clear {
            position: absolute;
            right: 8px;
            background: none;
            border: none;
            font-size: 18px;
            cursor: pointer;
            color: var(--text-secondary, #666);
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .geocoder-clear:hover {
            color: var(--text-primary, #333);
        }
        
        .geocoder-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--bg-secondary, #ffffff);
            border: 1px solid var(--border, #ccc);
            border-top: none;
            border-radius: 0 0 6px 6px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            max-height: 200px;
            overflow-y: auto;
            z-index: 1100;
        }
        
        .geocoder-result-item {
            padding: 10px 12px;
            cursor: pointer;
            font-size: 13px;
            border-bottom: 1px solid var(--border, #e0e0e0);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .geocoder-result-item:last-child {
            border-bottom: none;
        }
        
        .geocoder-result-item:hover,
        .geocoder-result-item.selected {
            background: var(--accent, #007bff);
            color: white;
        }
        
        /* Disabled state */
        .map-control.disabled {
            opacity: 0.6;
            pointer-events: none;
        }
    `;
    
    if (!document.getElementById('map-control-styles')) {
        document.head.appendChild(controlStyles);
    }
    
    // Export controls to global scope
    global.MapControls = {
        MapControl,
        NavigationControl,
        ScaleControl,
        LayerControl,
        DrawControl,
        GeocoderControl
    };
    
    console.log('✨ Map Controls ready. Access via window.MapControls');
    
})(window);