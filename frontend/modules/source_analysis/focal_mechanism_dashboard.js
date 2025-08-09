/**
 * Advanced Focal Mechanism Analysis Dashboard
 * Real-time source characterization interface with moment tensor visualization,
 * stress field mapping, and comprehensive earthquake triggering analysis
 */

class FocalMechanismDashboard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        this.config = {
            updateInterval: 5000, // 5 seconds
            maxEvents: 1000,
            stressGridResolution: 50,
            animationDuration: 1000,
            colorSchemes: {
                stress: ['#2E86C1', '#FFFFFF', '#E74C3C'],
                magnitude: ['#FFEB3B', '#FF9800', '#F44336'],
                depth: ['#E8F5E8', '#4CAF50', '#1B5E20']
            },
            ...config
        };
        
        this.data = {
            events: [],
            momentTensors: new Map(),
            stressField: null,
            triggering: {
                links: [],
                cascades: []
            }
        };
        
        this.visualization = {
            beachballs: new Map(),
            stressContours: null,
            triggeringNetwork: null
        };
        
        this.websocket = null;
        this.isConnected = false;
        
        this.init();
    }
    
    init() {
        this.createUI();
        this.setupEventListeners();
        this.initializeVisualization();
        this.connectWebSocket();
        this.startDataUpdates();
    }
    
    createUI() {
        this.container.innerHTML = `
            <div class="focal-mechanism-dashboard">
                <!-- Header Controls -->
                <div class="dashboard-header">
                    <div class="control-panel">
                        <div class="control-group">
                            <label for="analysis-mode">Analysis Mode:</label>
                            <select id="analysis-mode" class="control-select">
                                <option value="realtime">Real-time</option>
                                <option value="historical">Historical</option>
                                <option value="catalog">Catalog Analysis</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label for="visualization-type">Visualization:</label>
                            <select id="visualization-type" class="control-select">
                                <option value="beachballs">Focal Mechanisms</option>
                                <option value="stress-field">Stress Field</option>
                                <option value="triggering">Triggering Network</option>
                                <option value="temporal">Temporal Evolution</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label for="magnitude-filter">Min Magnitude:</label>
                            <input type="range" id="magnitude-filter" min="2.0" max="8.0" step="0.1" value="4.0" class="control-slider">
                            <span id="magnitude-value">4.0</span>
                        </div>
                        
                        <button id="calculate-stress" class="btn btn-primary">Calculate Stress Field</button>
                        <button id="analyze-triggering" class="btn btn-secondary">Analyze Triggering</button>
                        <button id="export-data" class="btn btn-outline">Export Results</button>
                    </div>
                    
                    <div class="status-panel">
                        <div class="status-item">
                            <span class="status-label">Connection:</span>
                            <span id="connection-status" class="status-value disconnected">Disconnected</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Events:</span>
                            <span id="event-count" class="status-value">0</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Processing:</span>
                            <span id="processing-status" class="status-value">Idle</span>
                        </div>
                    </div>
                </div>
                
                <!-- Main Visualization Area -->
                <div class="visualization-container">
                    <div class="main-map-container">
                        <canvas id="main-map-canvas" class="main-canvas"></canvas>
                        <div class="map-overlay">
                            <div class="legend-panel" id="legend-panel">
                                <!-- Dynamic legend content -->
                            </div>
                            <div class="scale-panel" id="scale-panel">
                                <div class="scale-bar"></div>
                                <span class="scale-text">50 km</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="side-panels">
                        <!-- Moment Tensor Details -->
                        <div class="panel" id="mt-details-panel">
                            <h3>Moment Tensor Analysis</h3>
                            <div class="mt-summary" id="mt-summary">
                                <div class="no-selection">Select an event to view details</div>
                            </div>
                            <div class="beachball-viewer" id="beachball-viewer">
                                <canvas id="beachball-canvas"></canvas>
                            </div>
                            <div class="mt-parameters" id="mt-parameters">
                                <!-- Dynamic MT parameters -->
                            </div>
                        </div>
                        
                        <!-- Stress Analysis -->
                        <div class="panel" id="stress-analysis-panel">
                            <h3>Stress Field Analysis</h3>
                            <div class="stress-controls">
                                <div class="control-row">
                                    <label>Target Mechanism:</label>
                                    <div class="mechanism-inputs">
                                        <input type="number" id="target-strike" placeholder="Strike" min="0" max="360" value="0">
                                        <input type="number" id="target-dip" placeholder="Dip" min="0" max="90" value="90">
                                        <input type="number" id="target-rake" placeholder="Rake" min="-180" max="180" value="0">
                                    </div>
                                </div>
                                <div class="control-row">
                                    <label for="friction-coefficient">Friction (μ):</label>
                                    <input type="number" id="friction-coefficient" min="0.1" max="1.0" step="0.1" value="0.6">
                                </div>
                                <div class="control-row">
                                    <label for="stress-threshold">Trigger Threshold:</label>
                                    <input type="number" id="stress-threshold" min="0.001" max="0.1" step="0.001" value="0.01">
                                    <span class="unit">MPa</span>
                                </div>
                            </div>
                            <div class="stress-results" id="stress-results">
                                <!-- Dynamic stress results -->
                            </div>
                        </div>
                        
                        <!-- Triggering Analysis -->
                        <div class="panel" id="triggering-panel">
                            <h3>Earthquake Triggering</h3>
                            <div class="triggering-stats" id="triggering-stats">
                                <!-- Dynamic triggering statistics -->
                            </div>
                            <div class="cascade-tree" id="cascade-tree">
                                <!-- Dynamic cascade visualization -->
                            </div>
                            <div class="temporal-plot" id="temporal-plot">
                                <canvas id="temporal-canvas"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Event List -->
                <div class="event-list-container">
                    <div class="event-list-header">
                        <h3>Earthquake Catalog</h3>
                        <div class="list-controls">
                            <input type="text" id="event-search" placeholder="Search events..." class="search-input">
                            <select id="sort-by" class="sort-select">
                                <option value="time">Time</option>
                                <option value="magnitude">Magnitude</option>
                                <option value="depth">Depth</option>
                                <option value="distance">Distance</option>
                            </select>
                        </div>
                    </div>
                    <div class="event-list-body" id="event-list-body">
                        <!-- Dynamic event list -->
                    </div>
                </div>
            </div>
        `;
        
        this.setupUIStyles();
    }
    
    setupUIStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .focal-mechanism-dashboard {
                display: flex;
                flex-direction: column;
                height: 100vh;
                background: #f8f9fa;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px 20px;
                background: #ffffff;
                border-bottom: 1px solid #e9ecef;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .control-panel {
                display: flex;
                gap: 20px;
                align-items: center;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            
            .control-group label {
                font-size: 12px;
                color: #6c757d;
                font-weight: 500;
            }
            
            .control-select, .control-slider {
                padding: 6px 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
            
            .control-slider {
                width: 120px;
            }
            
            .btn {
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
                font-size: 14px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .btn-primary {
                background: #007bff;
                color: white;
            }
            
            .btn-primary:hover {
                background: #0056b3;
            }
            
            .btn-secondary {
                background: #6c757d;
                color: white;
            }
            
            .btn-outline {
                background: transparent;
                border: 1px solid #007bff;
                color: #007bff;
            }
            
            .status-panel {
                display: flex;
                gap: 20px;
            }
            
            .status-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 3px;
            }
            
            .status-label {
                font-size: 11px;
                color: #6c757d;
                text-transform: uppercase;
                font-weight: 600;
            }
            
            .status-value {
                font-size: 14px;
                font-weight: 500;
            }
            
            .status-value.connected {
                color: #28a745;
            }
            
            .status-value.disconnected {
                color: #dc3545;
            }
            
            .visualization-container {
                display: flex;
                flex: 1;
                min-height: 0;
            }
            
            .main-map-container {
                flex: 2;
                position: relative;
                background: #ffffff;
                border-right: 1px solid #e9ecef;
            }
            
            .main-canvas {
                width: 100%;
                height: 100%;
                display: block;
            }
            
            .map-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                pointer-events: none;
            }
            
            .legend-panel {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                pointer-events: auto;
                max-width: 250px;
            }
            
            .scale-panel {
                position: absolute;
                bottom: 20px;
                left: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .scale-bar {
                width: 50px;
                height: 3px;
                background: #333;
            }
            
            .scale-text {
                font-size: 12px;
                color: #333;
                font-weight: 500;
            }
            
            .side-panels {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: #ffffff;
                min-width: 350px;
                max-width: 450px;
            }
            
            .panel {
                flex: 1;
                padding: 20px;
                border-bottom: 1px solid #e9ecef;
                overflow-y: auto;
            }
            
            .panel h3 {
                margin: 0 0 15px 0;
                font-size: 16px;
                color: #343a40;
                border-bottom: 2px solid #007bff;
                padding-bottom: 8px;
            }
            
            .mt-summary {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
            }
            
            .no-selection {
                text-align: center;
                color: #6c757d;
                font-style: italic;
            }
            
            .beachball-viewer {
                display: flex;
                justify-content: center;
                margin: 15px 0;
            }
            
            #beachball-canvas {
                border: 1px solid #e9ecef;
                border-radius: 50%;
            }
            
            .mt-parameters {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                font-size: 14px;
            }
            
            .parameter-item {
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #f1f3f4;
            }
            
            .parameter-label {
                color: #6c757d;
                font-weight: 500;
            }
            
            .parameter-value {
                color: #343a40;
                font-weight: 600;
            }
            
            .stress-controls {
                margin-bottom: 20px;
            }
            
            .control-row {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }
            
            .control-row label {
                min-width: 80px;
                font-size: 12px;
                color: #6c757d;
                font-weight: 500;
            }
            
            .mechanism-inputs {
                display: flex;
                gap: 5px;
            }
            
            .mechanism-inputs input {
                width: 60px;
                padding: 4px 6px;
                border: 1px solid #ced4da;
                border-radius: 3px;
                font-size: 12px;
            }
            
            .unit {
                font-size: 11px;
                color: #6c757d;
            }
            
            .stress-results {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                font-size: 14px;
            }
            
            .triggering-stats {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .stat-item {
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 6px;
            }
            
            .stat-value {
                font-size: 24px;
                font-weight: 600;
                color: #007bff;
            }
            
            .stat-label {
                font-size: 12px;
                color: #6c757d;
                text-transform: uppercase;
            }
            
            .cascade-tree {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                min-height: 120px;
            }
            
            .temporal-plot canvas {
                width: 100%;
                height: 150px;
                border: 1px solid #e9ecef;
                border-radius: 4px;
            }
            
            .event-list-container {
                height: 200px;
                background: #ffffff;
                border-top: 1px solid #e9ecef;
            }
            
            .event-list-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px 20px;
                background: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
            }
            
            .event-list-header h3 {
                margin: 0;
                font-size: 16px;
                color: #343a40;
            }
            
            .list-controls {
                display: flex;
                gap: 10px;
            }
            
            .search-input, .sort-select {
                padding: 6px 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
            
            .event-list-body {
                overflow-y: auto;
                height: calc(100% - 60px);
            }
            
            .event-item {
                display: flex;
                align-items: center;
                padding: 10px 20px;
                border-bottom: 1px solid #f1f3f4;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .event-item:hover {
                background: #f8f9fa;
            }
            
            .event-item.selected {
                background: #e3f2fd;
                border-left: 4px solid #007bff;
            }
            
            .event-magnitude {
                width: 60px;
                text-align: center;
                font-weight: 600;
                color: #007bff;
            }
            
            .event-details {
                flex: 1;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .event-location {
                font-weight: 500;
                color: #343a40;
            }
            
            .event-time {
                font-size: 12px;
                color: #6c757d;
            }
            
            .event-depth {
                font-size: 12px;
                color: #6c757d;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    setupEventListeners() {
        // Visualization type change
        document.getElementById('visualization-type').addEventListener('change', (e) => {
            this.switchVisualizationType(e.target.value);
        });
        
        // Magnitude filter
        const magFilter = document.getElementById('magnitude-filter');
        const magValue = document.getElementById('magnitude-value');
        
        magFilter.addEventListener('input', (e) => {
            magValue.textContent = e.target.value;
            this.filterEventsByMagnitude(parseFloat(e.target.value));
        });
        
        // Control buttons
        document.getElementById('calculate-stress').addEventListener('click', () => {
            this.calculateStressField();
        });
        
        document.getElementById('analyze-triggering').addEventListener('click', () => {
            this.analyzeTriggering();
        });
        
        document.getElementById('export-data').addEventListener('click', () => {
            this.exportData();
        });
        
        // Search and sorting
        document.getElementById('event-search').addEventListener('input', (e) => {
            this.filterEvents(e.target.value);
        });
        
        document.getElementById('sort-by').addEventListener('change', (e) => {
            this.sortEvents(e.target.value);
        });
        
        // Canvas events
        const canvas = document.getElementById('main-map-canvas');
        canvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e);
        });
        
        canvas.addEventListener('mousemove', (e) => {
            this.handleCanvasMouseMove(e);
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.resizeVisualization();
        });
    }
    
    initializeVisualization() {
        const canvas = document.getElementById('main-map-canvas');
        const ctx = canvas.getContext('2d');
        
        this.canvas = canvas;
        this.ctx = ctx;
        
        this.resizeVisualization();
        this.drawBaseMap();
    }
    
    resizeVisualization() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        
        this.updateVisualization();
    }
    
    connectWebSocket() {
        try {
            this.websocket = new WebSocket('ws://localhost:8000/ws/focal-mechanisms');
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus('Connected', 'connected');
                console.log('WebSocket connected to focal mechanism service');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealtimeData(data);
            };
            
            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('Disconnected', 'disconnected');
                console.log('WebSocket disconnected');
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.connectWebSocket();
                    }
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error', 'disconnected');
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus('Failed', 'disconnected');
        }
    }
    
    handleRealtimeData(data) {
        const timestamp = new Date();
        
        switch (data.type) {
            case 'moment_tensor':
                this.processMomentTensorData(data, timestamp);
                break;
                
            case 'stress_field':
                this.processStressFieldData(data, timestamp);
                break;
                
            case 'triggering_analysis':
                this.processTriggeringData(data, timestamp);
                break;
                
            case 'earthquake_event':
                this.processEarthquakeEvent(data, timestamp);
                break;
                
            default:
                console.warn('Unknown data type:', data.type);
        }
        
        this.updateVisualization();
        this.updateEventList();
    }
    
    processMomentTensorData(data, timestamp) {
        const eventId = data.event_id;
        
        this.data.momentTensors.set(eventId, {
            tensor: data.moment_tensor,
            nodal_planes: data.nodal_planes,
            magnitude: data.magnitude_info,
            source_analysis: data.source_analysis,
            quality_metrics: data.quality_metrics,
            uncertainties: data.uncertainties,
            timestamp: timestamp
        });
        
        // Update event if it exists
        const event = this.data.events.find(e => e.event_id === eventId);
        if (event) {
            event.has_moment_tensor = true;
            event.moment_tensor_data = this.data.momentTensors.get(eventId);
        }
        
        this.updateProcessingStatus('Processed MT Solution');
    }
    
    processStressFieldData(data, timestamp) {
        this.data.stressField = {
            grid_points: data.grid_points,
            stress_tensors: data.stress_tensors,
            method: data.interpolation_method,
            quality: data.quality_metrics,
            timestamp: timestamp
        };
        
        this.updateProcessingStatus('Updated Stress Field');
    }
    
    processTriggeringData(data, timestamp) {
        if (data.triggering_links) {
            this.data.triggering.links = data.triggering_links;
        }
        
        if (data.cascade_analysis) {
            this.data.triggering.cascades = data.cascade_analysis;
        }
        
        this.updateTriggeringStats(data.statistics);
        this.updateProcessingStatus('Analyzed Triggering');
    }
    
    processEarthquakeEvent(data, timestamp) {
        const event = {
            event_id: data.event_id,
            latitude: data.latitude,
            longitude: data.longitude,
            depth: data.depth,
            magnitude: data.magnitude,
            timestamp: new Date(data.timestamp),
            has_moment_tensor: false,
            moment_tensor_data: null
        };
        
        // Add to events array (maintain chronological order)
        const insertIndex = this.data.events.findIndex(e => e.timestamp > event.timestamp);
        if (insertIndex === -1) {
            this.data.events.push(event);
        } else {
            this.data.events.splice(insertIndex, 0, event);
        }
        
        // Limit event count
        if (this.data.events.length > this.config.maxEvents) {
            this.data.events = this.data.events.slice(-this.config.maxEvents);
        }
        
        this.updateEventCount();
    }
    
    drawBaseMap() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw coordinate grid
        ctx.strokeStyle = '#e9ecef';
        ctx.lineWidth = 1;
        
        const gridSpacing = 50;
        for (let x = 0; x < canvas.width; x += gridSpacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        
        for (let y = 0; y < canvas.height; y += gridSpacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    }
    
    updateVisualization() {
        const visualizationType = document.getElementById('visualization-type').value;
        
        this.drawBaseMap();
        
        switch (visualizationType) {
            case 'beachballs':
                this.drawFocalMechanisms();
                break;
                
            case 'stress-field':
                this.drawStressField();
                break;
                
            case 'triggering':
                this.drawTriggeringNetwork();
                break;
                
            case 'temporal':
                this.drawTemporalEvolution();
                break;
        }
        
        this.updateLegend(visualizationType);
    }
    
    drawFocalMechanisms() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        
        // Draw events with focal mechanisms
        this.data.events.forEach(event => {
            if (!event.has_moment_tensor) return;
            
            const coords = this.projectCoordinates(event.latitude, event.longitude);
            if (!coords) return;
            
            const mtData = event.moment_tensor_data;
            const radius = this.getMagnitudeRadius(event.magnitude);
            
            // Draw beachball
            this.drawBeachball(ctx, coords.x, coords.y, radius, mtData.nodal_planes);
            
            // Draw magnitude label
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`M${event.magnitude.toFixed(1)}`, coords.x, coords.y - radius - 5);
        });
        
        // Draw events without focal mechanisms as circles
        this.data.events.forEach(event => {
            if (event.has_moment_tensor) return;
            
            const coords = this.projectCoordinates(event.latitude, event.longitude);
            if (!coords) return;
            
            const radius = this.getMagnitudeRadius(event.magnitude);
            
            ctx.fillStyle = this.getMagnitudeColor(event.magnitude);
            ctx.beginPath();
            ctx.arc(coords.x, coords.y, radius, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }
    
    drawBeachball(ctx, x, y, radius, nodalPlanes) {
        if (!nodalPlanes || !nodalPlanes.plane1) return;
        
        const plane1 = nodalPlanes.plane1;
        
        // Simplified beachball representation
        // In a full implementation, this would properly calculate
        // the projection of fault planes onto a stereonet
        
        ctx.save();
        ctx.translate(x, y);
        
        // Draw outer circle
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(0, 0, radius, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Simplified representation of fault plane
        const strike = (plane1.strike * Math.PI) / 180;
        const dip = (plane1.dip * Math.PI) / 180;
        
        // Draw fault plane as a diameter
        ctx.beginPath();
        ctx.moveTo(-radius * Math.cos(strike), -radius * Math.sin(strike));
        ctx.lineTo(radius * Math.cos(strike), radius * Math.sin(strike));
        ctx.stroke();
        
        // Fill compressional quadrants
        const rake = (plane1.rake * Math.PI) / 180;
        
        if (Math.abs(rake) < Math.PI / 2) { // Reverse faulting component
            ctx.fillStyle = '#000000';
        } else { // Normal faulting component
            ctx.fillStyle = '#ffffff';
        }
        
        // Simplified quadrant filling
        ctx.beginPath();
        ctx.arc(0, 0, radius, strike, strike + Math.PI);
        ctx.closePath();
        ctx.fill();
        
        ctx.restore();
    }
    
    drawStressField() {
        if (!this.data.stressField || !this.data.stressField.stress_tensors) return;
        
        const ctx = this.ctx;
        const stressData = this.data.stressField.stress_tensors;
        
        // Draw stress field as contours or color-coded grid
        stressData.forEach(tensor => {
            if (!tensor.location) return;
            
            const coords = this.projectCoordinates(tensor.location[0], tensor.location[1]);
            if (!coords) return;
            
            // Color-code by principal stress orientation
            const hue = (tensor.azimuth1 / 360) * 360;
            const saturation = tensor.confidence * 100;
            const lightness = 70;
            
            ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            
            const size = 8;
            ctx.fillRect(coords.x - size/2, coords.y - size/2, size, size);
            
            // Draw stress orientation as a line
            if (tensor.confidence > 0.5) {
                const angle = (tensor.azimuth1 * Math.PI) / 180;
                const length = 15;
                
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(coords.x - length * Math.cos(angle), coords.y - length * Math.sin(angle));
                ctx.lineTo(coords.x + length * Math.cos(angle), coords.y + length * Math.sin(angle));
                ctx.stroke();
            }
        });
    }
    
    drawTriggeringNetwork() {
        const ctx = this.ctx;
        
        // Draw triggering links as arrows
        this.data.triggering.links.forEach(link => {
            const sourceCoords = this.projectCoordinates(
                link.source_event.latitude, link.source_event.longitude
            );
            const targetCoords = this.projectCoordinates(
                link.triggered_event.latitude, link.triggered_event.longitude
            );
            
            if (!sourceCoords || !targetCoords) return;
            
            // Color by stress change magnitude
            const intensity = Math.min(Math.abs(link.stress_change) / 0.1, 1);
            const color = link.stress_change > 0 ? 
                `rgba(231, 76, 60, ${intensity})` : 
                `rgba(52, 152, 219, ${intensity})`;
            
            this.drawArrow(ctx, sourceCoords.x, sourceCoords.y, 
                         targetCoords.x, targetCoords.y, color);
        });
        
        // Draw events
        this.drawFocalMechanisms();
    }
    
    drawArrow(ctx, fromX, fromY, toX, toY, color) {
        const headLength = 10;
        const angle = Math.atan2(toY - fromY, toX - fromX);
        
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        
        // Draw line
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        
        // Draw arrowhead
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI/6), 
                  toY - headLength * Math.sin(angle - Math.PI/6));
        ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI/6), 
                  toY - headLength * Math.sin(angle + Math.PI/6));
        ctx.closePath();
        ctx.fill();
    }
    
    drawTemporalEvolution() {
        const canvas = document.getElementById('temporal-canvas');
        const ctx = canvas.getContext('2d');
        
        if (!this.data.events.length) return;
        
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 150;
        
        const margin = { left: 50, right: 20, top: 20, bottom: 30 };
        const plotWidth = canvas.width - margin.left - margin.right;
        const plotHeight = canvas.height - margin.top - margin.bottom;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Get time range
        const timeRange = this.getTimeRange();
        if (!timeRange) return;
        
        // Draw time series
        const events = this.data.events.slice(-100); // Last 100 events
        
        ctx.fillStyle = '#007bff';
        events.forEach(event => {
            const x = margin.left + (event.timestamp - timeRange.start) / 
                     (timeRange.end - timeRange.start) * plotWidth;
            const y = margin.top + (1 - event.magnitude / 8) * plotHeight;
            
            const radius = Math.max(2, event.magnitude / 2);
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, canvas.height - margin.bottom);
        ctx.stroke();
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, canvas.height - margin.bottom);
        ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Time', canvas.width / 2, canvas.height - 5);
        
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Magnitude', 0, 0);
        ctx.restore();
    }
    
    projectCoordinates(lat, lon) {
        // Simple projection for demonstration
        // In a real implementation, would use proper map projection
        const canvas = this.canvas;
        
        // Assume viewport covers a small region
        const latRange = { min: 44, max: 46 };
        const lonRange = { min: -122, max: -120 };
        
        if (lat < latRange.min || lat > latRange.max || 
            lon < lonRange.min || lon > lonRange.max) {
            return null;
        }
        
        const x = ((lon - lonRange.min) / (lonRange.max - lonRange.min)) * canvas.width;
        const y = ((latRange.max - lat) / (latRange.max - latRange.min)) * canvas.height;
        
        return { x, y };
    }
    
    getMagnitudeRadius(magnitude) {
        return Math.max(3, magnitude * 2);
    }
    
    getMagnitudeColor(magnitude) {
        if (magnitude < 4) return '#4CAF50';
        if (magnitude < 5) return '#FF9800';
        if (magnitude < 6) return '#FF5722';
        return '#F44336';
    }
    
    getTimeRange() {
        if (!this.data.events.length) return null;
        
        const times = this.data.events.map(e => e.timestamp);
        return {
            start: new Date(Math.min(...times)),
            end: new Date(Math.max(...times))
        };
    }
    
    switchVisualizationType(type) {
        this.updateVisualization();
        
        // Show/hide relevant panels
        const panels = {
            'beachballs': ['mt-details-panel'],
            'stress-field': ['stress-analysis-panel'],
            'triggering': ['triggering-panel'],
            'temporal': ['triggering-panel']
        };
        
        // Reset panel visibility
        document.querySelectorAll('.panel').forEach(panel => {
            panel.style.display = 'block';
        });
        
        // Highlight relevant panels
        if (panels[type]) {
            document.querySelectorAll('.panel').forEach(panel => {
                if (!panels[type].includes(panel.id)) {
                    panel.style.opacity = '0.7';
                } else {
                    panel.style.opacity = '1';
                }
            });
        }
    }
    
    filterEventsByMagnitude(minMagnitude) {
        // Re-render with filtered events
        this.updateVisualization();
    }
    
    calculateStressField() {
        this.updateProcessingStatus('Calculating stress field...');
        
        const targetStrike = document.getElementById('target-strike').value;
        const targetDip = document.getElementById('target-dip').value;
        const targetRake = document.getElementById('target-rake').value;
        const frictionCoeff = document.getElementById('friction-coefficient').value;
        
        const request = {
            type: 'calculate_stress_field',
            parameters: {
                target_mechanism: [parseFloat(targetStrike), parseFloat(targetDip), parseFloat(targetRake)],
                friction_coefficient: parseFloat(frictionCoeff),
                grid_resolution: this.config.stressGridResolution
            }
        };
        
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify(request));
        }
    }
    
    analyzeTriggering() {
        this.updateProcessingStatus('Analyzing triggering relationships...');
        
        const threshold = document.getElementById('stress-threshold').value;
        
        const request = {
            type: 'analyze_triggering',
            parameters: {
                triggering_threshold: parseFloat(threshold),
                time_window_hours: 24
            }
        };
        
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify(request));
        }
    }
    
    handleCanvasClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Find clicked event
        const clickedEvent = this.findEventAtPosition(x, y);
        
        if (clickedEvent) {
            this.selectEvent(clickedEvent);
        }
    }
    
    handleCanvasMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Update cursor based on hover state
        const hoveredEvent = this.findEventAtPosition(x, y);
        this.canvas.style.cursor = hoveredEvent ? 'pointer' : 'default';
    }
    
    findEventAtPosition(x, y) {
        const tolerance = 10;
        
        for (const event of this.data.events) {
            const coords = this.projectCoordinates(event.latitude, event.longitude);
            if (!coords) continue;
            
            const distance = Math.sqrt(
                Math.pow(x - coords.x, 2) + Math.pow(y - coords.y, 2)
            );
            
            if (distance <= tolerance) {
                return event;
            }
        }
        
        return null;
    }
    
    selectEvent(event) {
        // Update MT details panel
        this.updateMomentTensorPanel(event);
        
        // Highlight in event list
        this.highlightEventInList(event);
        
        // Update visualization to show selection
        this.updateVisualization();
    }
    
    updateMomentTensorPanel(event) {
        const summaryDiv = document.getElementById('mt-summary');
        const parametersDiv = document.getElementById('mt-parameters');
        
        if (!event.has_moment_tensor) {
            summaryDiv.innerHTML = `
                <div class="event-basic-info">
                    <h4>Event ${event.event_id}</h4>
                    <p>M${event.magnitude} - ${event.depth} km depth</p>
                    <p>${event.timestamp.toLocaleString()}</p>
                    <p><em>No moment tensor solution available</em></p>
                </div>
            `;
            parametersDiv.innerHTML = '';
            return;
        }
        
        const mtData = event.moment_tensor_data;
        
        summaryDiv.innerHTML = `
            <div class="event-basic-info">
                <h4>Event ${event.event_id}</h4>
                <p>M${mtData.magnitude.moment_magnitude.toFixed(2)} - ${event.depth} km depth</p>
                <p>${event.timestamp.toLocaleString()}</p>
                <p>Source: ${mtData.source_analysis.source_type.toUpperCase()}</p>
            </div>
        `;
        
        const plane1 = mtData.nodal_planes.plane1;
        const plane2 = mtData.nodal_planes.plane2;
        
        parametersDiv.innerHTML = `
            <div class="parameter-item">
                <span class="parameter-label">Strike 1:</span>
                <span class="parameter-value">${plane1.strike.toFixed(1)}°</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Dip 1:</span>
                <span class="parameter-value">${plane1.dip.toFixed(1)}°</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Rake 1:</span>
                <span class="parameter-value">${plane1.rake.toFixed(1)}°</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Strike 2:</span>
                <span class="parameter-value">${plane2.strike.toFixed(1)}°</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Dip 2:</span>
                <span class="parameter-value">${plane2.dip.toFixed(1)}°</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Rake 2:</span>
                <span class="parameter-value">${plane2.rake.toFixed(1)}°</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">DC %:</span>
                <span class="parameter-value">${mtData.source_analysis.dc_percentage.toFixed(1)}%</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">CLVD %:</span>
                <span class="parameter-value">${mtData.source_analysis.clvd_percentage.toFixed(1)}%</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">ISO %:</span>
                <span class="parameter-value">${mtData.source_analysis.iso_percentage.toFixed(1)}%</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Var. Red.:</span>
                <span class="parameter-value">${mtData.quality_metrics.variance_reduction.toFixed(3)}</span>
            </div>
        `;
        
        // Draw beachball
        this.drawSelectedBeachball(mtData.nodal_planes);
    }
    
    drawSelectedBeachball(nodalPlanes) {
        const canvas = document.getElementById('beachball-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = 120;
        canvas.height = 120;
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 50;
        
        this.drawBeachball(ctx, centerX, centerY, radius, nodalPlanes);
    }
    
    updateEventList() {
        const listBody = document.getElementById('event-list-body');
        const minMagnitude = parseFloat(document.getElementById('magnitude-filter').value);
        
        const filteredEvents = this.data.events.filter(event => event.magnitude >= minMagnitude);
        
        listBody.innerHTML = '';
        
        filteredEvents.slice(-50).reverse().forEach(event => {
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-item';
            eventDiv.dataset.eventId = event.event_id;
            
            const timeStr = event.timestamp.toLocaleString();
            const locationStr = `${event.latitude.toFixed(2)}, ${event.longitude.toFixed(2)}`;
            
            eventDiv.innerHTML = `
                <div class="event-magnitude">M${event.magnitude.toFixed(1)}</div>
                <div class="event-details">
                    <div>
                        <div class="event-location">${locationStr}</div>
                        <div class="event-time">${timeStr}</div>
                    </div>
                    <div class="event-depth">${event.depth} km</div>
                </div>
            `;
            
            eventDiv.addEventListener('click', () => {
                this.selectEvent(event);
            });
            
            listBody.appendChild(eventDiv);
        });
        
        this.updateEventCount();
    }
    
    updateLegend(visualizationType) {
        const legendPanel = document.getElementById('legend-panel');
        
        let legendContent = '';
        
        switch (visualizationType) {
            case 'beachballs':
                legendContent = `
                    <h4>Focal Mechanisms</h4>
                    <div class="legend-item">
                        <div class="legend-symbol" style="background: black; width: 15px; height: 15px; border-radius: 50%;"></div>
                        <span>Compressional</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-symbol" style="background: white; width: 15px; height: 15px; border-radius: 50%; border: 1px solid black;"></div>
                        <span>Tensional</span>
                    </div>
                `;
                break;
                
            case 'stress-field':
                legendContent = `
                    <h4>Stress Field</h4>
                    <div class="stress-scale">
                        <div class="stress-bar" style="background: linear-gradient(to right, blue, white, red); height: 20px; width: 100px;"></div>
                        <div class="stress-labels" style="display: flex; justify-content: space-between; font-size: 11px;">
                            <span>-0.1</span><span>0</span><span>+0.1 MPa</span>
                        </div>
                    </div>
                `;
                break;
                
            case 'triggering':
                legendContent = `
                    <h4>Triggering Network</h4>
                    <div class="legend-item">
                        <div style="width: 20px; height: 2px; background: rgba(231, 76, 60, 0.8);"></div>
                        <span>Triggering</span>
                    </div>
                    <div class="legend-item">
                        <div style="width: 20px; height: 2px; background: rgba(52, 152, 219, 0.8);"></div>
                        <span>Inhibiting</span>
                    </div>
                `;
                break;
        }
        
        legendPanel.innerHTML = legendContent;
    }
    
    updateConnectionStatus(status, className) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status;
        statusElement.className = `status-value ${className}`;
    }
    
    updateEventCount() {
        const countElement = document.getElementById('event-count');
        countElement.textContent = this.data.events.length.toString();
    }
    
    updateProcessingStatus(status) {
        const statusElement = document.getElementById('processing-status');
        statusElement.textContent = status;
        
        // Reset to idle after 3 seconds
        setTimeout(() => {
            statusElement.textContent = 'Idle';
        }, 3000);
    }
    
    updateTriggeringStats(stats) {
        const statsDiv = document.getElementById('triggering-stats');
        
        if (!stats) return;
        
        statsDiv.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${stats.triggered_count || 0}</div>
                <div class="stat-label">Triggered</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.inhibited_count || 0}</div>
                <div class="stat-label">Inhibited</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${((stats.triggering_rate || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">Trigger Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${(stats.mean_triggering_stress || 0).toFixed(3)}</div>
                <div class="stat-label">Mean Stress (MPa)</div>
            </div>
        `;
    }
    
    highlightEventInList(event) {
        // Remove previous selections
        document.querySelectorAll('.event-item.selected').forEach(item => {
            item.classList.remove('selected');
        });
        
        // Add selection to current event
        const eventItem = document.querySelector(`[data-event-id="${event.event_id}"]`);
        if (eventItem) {
            eventItem.classList.add('selected');
        }
    }
    
    filterEvents(searchTerm) {
        // Implementation for event filtering
        this.updateEventList();
    }
    
    sortEvents(sortBy) {
        // Implementation for event sorting
        switch (sortBy) {
            case 'time':
                this.data.events.sort((a, b) => b.timestamp - a.timestamp);
                break;
            case 'magnitude':
                this.data.events.sort((a, b) => b.magnitude - a.magnitude);
                break;
            case 'depth':
                this.data.events.sort((a, b) => a.depth - b.depth);
                break;
        }
        
        this.updateEventList();
    }
    
    exportData() {
        const exportData = {
            timestamp: new Date().toISOString(),
            events: this.data.events,
            moment_tensors: Array.from(this.data.momentTensors.entries()),
            stress_field: this.data.stressField,
            triggering_analysis: this.data.triggering
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `focal_mechanism_analysis_${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    startDataUpdates() {
        // Periodic data updates
        setInterval(() => {
            if (this.isConnected && this.websocket) {
                // Request status update
                this.websocket.send(JSON.stringify({
                    type: 'status_request'
                }));
            }
        }, this.config.updateInterval);
    }
    
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        this.container.innerHTML = '';
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('focal-mechanism-dashboard')) {
        window.focalMechanismDashboard = new FocalMechanismDashboard('focal-mechanism-dashboard');
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FocalMechanismDashboard;
}