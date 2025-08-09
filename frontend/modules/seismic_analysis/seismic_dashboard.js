/**
 * Real-time Seismic Analysis Dashboard
 * Interactive interface for deep learning seismic processing,
 * array analysis, and ambient noise monitoring
 */

class SeismicDashboard {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Dashboard state
        this.isRealTime = false;
        this.processingMode = 'transformer'; // transformer, array, ambient_noise
        this.selectedStations = [];
        this.analysisResults = {};
        
        // Data streams
        this.waveformData = new Map();
        this.correlationData = new Map();
        this.velocityChangeData = new Map();
        
        // Visualization components
        this.charts = {
            waveforms: null,
            beamForming: null,
            phasePicks: null,
            velocityChanges: null,
            correlations: null,
            arrayResponse: null
        };
        
        // WebSocket connection for real-time data
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // Processing parameters
        this.processingParams = {
            transformer: {
                confidence_threshold: 0.7,
                frequency_bands: [[1, 5], [5, 15], [15, 30]],
                window_size: 60
            },
            array: {
                frequency_range: [1.0, 8.0],
                slowness_range: [0.0, 0.3],
                beamforming_method: 'ml_enhanced'
            },
            ambient_noise: {
                frequency_bands: [[0.1, 0.2], [0.2, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 5.0]],
                correlation_window: 3600,
                velocity_threshold: 0.01
            }
        };
        
        this.initialize();
    }
    
    initialize() {
        this.createDashboardLayout();
        this.setupEventListeners();
        this.initializeCharts();
        this.loadStationConfiguration();
    }
    
    createDashboardLayout() {
        this.container.innerHTML = `
            <div class="seismic-dashboard">
                <!-- Header Controls -->
                <div class="dashboard-header">
                    <div class="header-left">
                        <h2>Seismic Analysis Dashboard</h2>
                        <div class="status-indicator">
                            <span id="connection-status" class="status-disconnected">Disconnected</span>
                            <span id="processing-status">Idle</span>
                        </div>
                    </div>
                    <div class="header-right">
                        <div class="control-group">
                            <label>Processing Mode:</label>
                            <select id="processing-mode">
                                <option value="transformer">Transformer Analysis</option>
                                <option value="array">Array Processing</option>
                                <option value="ambient_noise">Ambient Noise</option>
                            </select>
                        </div>
                        <button id="real-time-toggle" class="btn btn-primary">Start Real-time</button>
                        <button id="export-results" class="btn btn-secondary">Export Results</button>
                    </div>
                </div>
                
                <!-- Configuration Panel -->
                <div class="config-panel" id="config-panel">
                    <div class="config-tabs">
                        <button class="tab-button active" data-tab="stations">Stations</button>
                        <button class="tab-button" data-tab="processing">Processing</button>
                        <button class="tab-button" data-tab="visualization">Visualization</button>
                    </div>
                    
                    <!-- Station Configuration -->
                    <div class="tab-content active" id="stations-tab">
                        <div class="station-selector">
                            <h4>Station Selection</h4>
                            <div id="station-list" class="station-grid"></div>
                            <div class="array-geometry">
                                <h5>Array Geometry</h5>
                                <canvas id="array-geometry-canvas" width="300" height="300"></canvas>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Processing Parameters -->
                    <div class="tab-content" id="processing-tab">
                        <div id="transformer-params" class="param-group">
                            <h4>Transformer Parameters</h4>
                            <div class="param-row">
                                <label>Confidence Threshold:</label>
                                <input type="range" id="confidence-threshold" min="0.1" max="1.0" step="0.1" value="0.7">
                                <span class="param-value">0.7</span>
                            </div>
                            <div class="param-row">
                                <label>Window Size (s):</label>
                                <input type="number" id="window-size" min="10" max="300" value="60">
                            </div>
                        </div>
                        
                        <div id="array-params" class="param-group" style="display: none;">
                            <h4>Array Processing Parameters</h4>
                            <div class="param-row">
                                <label>Frequency Range (Hz):</label>
                                <input type="number" id="freq-low" min="0.1" max="50" step="0.1" value="1.0" placeholder="Low">
                                <input type="number" id="freq-high" min="0.1" max="50" step="0.1" value="8.0" placeholder="High">
                            </div>
                            <div class="param-row">
                                <label>Beamforming Method:</label>
                                <select id="beamforming-method">
                                    <option value="conventional">Conventional</option>
                                    <option value="adaptive">Adaptive (MVDR)</option>
                                    <option value="ml_enhanced" selected>ML Enhanced</option>
                                </select>
                            </div>
                        </div>
                        
                        <div id="noise-params" class="param-group" style="display: none;">
                            <h4>Ambient Noise Parameters</h4>
                            <div class="param-row">
                                <label>Correlation Window (s):</label>
                                <input type="number" id="correlation-window" min="900" max="7200" value="3600">
                            </div>
                            <div class="param-row">
                                <label>Velocity Change Threshold (%):</label>
                                <input type="number" id="velocity-threshold" min="0.1" max="10" step="0.1" value="1.0">
                            </div>
                        </div>
                    </div>
                    
                    <!-- Visualization Settings -->
                    <div class="tab-content" id="visualization-tab">
                        <div class="viz-controls">
                            <h4>Display Options</h4>
                            <div class="checkbox-group">
                                <label><input type="checkbox" id="show-waveforms" checked> Waveforms</label>
                                <label><input type="checkbox" id="show-phase-picks" checked> Phase Picks</label>
                                <label><input type="checkbox" id="show-beam-power"> Beam Power</label>
                                <label><input type="checkbox" id="show-correlations"> Correlations</label>
                                <label><input type="checkbox" id="show-velocity-changes"> Velocity Changes</label>
                            </div>
                            <div class="param-row">
                                <label>Time Window (hours):</label>
                                <input type="number" id="display-window" min="1" max="48" value="6">
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Main Analysis Area -->
                <div class="analysis-area">
                    <div class="analysis-grid">
                        <!-- Waveform Display -->
                        <div class="chart-container" id="waveforms-container">
                            <div class="chart-header">
                                <h4>Real-time Waveforms</h4>
                                <div class="chart-controls">
                                    <button class="btn-small" id="auto-scale">Auto Scale</button>
                                    <button class="btn-small" id="filter-toggle">Filter: OFF</button>
                                </div>
                            </div>
                            <canvas id="waveforms-chart" width="800" height="300"></canvas>
                        </div>
                        
                        <!-- Phase Picking Results -->
                        <div class="chart-container" id="phase-picks-container">
                            <div class="chart-header">
                                <h4>Phase Picks & Analysis</h4>
                                <div class="stats-display">
                                    <span id="p-pick-count">P: 0</span>
                                    <span id="s-pick-count">S: 0</span>
                                    <span id="magnitude-estimate">M: -</span>
                                </div>
                            </div>
                            <canvas id="phase-picks-chart" width="800" height="200"></canvas>
                        </div>
                        
                        <!-- Array Processing Results -->
                        <div class="chart-container" id="array-container" style="display: none;">
                            <div class="chart-header">
                                <h4>Array Processing</h4>
                                <div class="array-stats">
                                    <span id="best-slowness">Slowness: -</span>
                                    <span id="best-azimuth">Azimuth: -</span>
                                    <span id="coherence">Coherence: -</span>
                                </div>
                            </div>
                            <div class="array-charts">
                                <canvas id="beam-power-chart" width="400" height="300"></canvas>
                                <canvas id="array-response-chart" width="400" height="300"></canvas>
                            </div>
                        </div>
                        
                        <!-- Ambient Noise Results -->
                        <div class="chart-container" id="ambient-noise-container" style="display: none;">
                            <div class="chart-header">
                                <h4>Ambient Noise Monitoring</h4>
                                <div class="noise-stats">
                                    <span id="correlation-quality">Quality: -</span>
                                    <span id="velocity-change">dV/V: -</span>
                                    <span id="health-status">Health: -</span>
                                </div>
                            </div>
                            <div class="noise-charts">
                                <canvas id="correlations-chart" width="400" height="250"></canvas>
                                <canvas id="velocity-changes-chart" width="400" height="250"></canvas>
                            </div>
                        </div>
                        
                        <!-- Event Log -->
                        <div class="event-log-container">
                            <div class="chart-header">
                                <h4>Event Log</h4>
                                <button class="btn-small" id="clear-log">Clear</button>
                            </div>
                            <div id="event-log" class="event-log"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Alerts Panel -->
                <div class="alerts-panel" id="alerts-panel" style="display: none;">
                    <div class="alert-header">
                        <h4>Real-time Alerts</h4>
                        <button id="close-alerts" class="close-btn">&times;</button>
                    </div>
                    <div id="alerts-content" class="alerts-content"></div>
                </div>
            </div>
        `;
        
        this.addDashboardStyles();
    }
    
    addDashboardStyles() {
        const styles = `
            <style>
                .seismic-dashboard {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #f5f7fa;
                    min-height: 100vh;
                    padding: 20px;
                }
                
                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                
                .header-left h2 {
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }
                
                .status-indicator {
                    display: flex;
                    gap: 15px;
                }
                
                .status-connected { color: #27ae60; font-weight: bold; }
                .status-disconnected { color: #e74c3c; font-weight: bold; }
                
                .header-right {
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }
                
                .control-group {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .config-panel {
                    background: white;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                
                .config-tabs {
                    display: flex;
                    border-bottom: 1px solid #ecf0f1;
                }
                
                .tab-button {
                    padding: 15px 25px;
                    border: none;
                    background: transparent;
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.3s;
                }
                
                .tab-button.active {
                    background: #3498db;
                    color: white;
                }
                
                .tab-content {
                    display: none;
                    padding: 20px;
                }
                
                .tab-content.active {
                    display: block;
                }
                
                .station-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                    gap: 10px;
                    margin: 15px 0;
                }
                
                .station-item {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 10px;
                    border: 1px solid #ecf0f1;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                
                .station-item:hover {
                    background: #f8f9fa;
                    border-color: #3498db;
                }
                
                .station-item.selected {
                    background: #e3f2fd;
                    border-color: #2196f3;
                }
                
                .param-group {
                    margin-bottom: 25px;
                }
                
                .param-row {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 10px 0;
                }
                
                .param-row label {
                    min-width: 150px;
                    font-weight: 500;
                }
                
                .param-value {
                    font-weight: bold;
                    color: #2c3e50;
                }
                
                .analysis-area {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                
                .analysis-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                
                .chart-container {
                    border: 1px solid #ecf0f1;
                    border-radius: 8px;
                    overflow: hidden;
                }
                
                .chart-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 15px 20px;
                    background: #f8f9fa;
                    border-bottom: 1px solid #ecf0f1;
                }
                
                .chart-header h4 {
                    margin: 0;
                    color: #2c3e50;
                }
                
                .chart-controls, .stats-display, .array-stats, .noise-stats {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                
                .array-charts, .noise-charts {
                    display: flex;
                    padding: 10px;
                }
                
                .event-log-container {
                    grid-column: 1 / -1;
                    max-height: 300px;
                }
                
                .event-log {
                    max-height: 200px;
                    overflow-y: auto;
                    padding: 15px;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                    line-height: 1.4;
                    background: #2c3e50;
                    color: #ecf0f1;
                }
                
                .event-entry {
                    margin: 2px 0;
                    padding: 2px 0;
                }
                
                .event-info { color: #3498db; }
                .event-warning { color: #f39c12; }
                .event-error { color: #e74c3c; }
                .event-success { color: #27ae60; }
                
                .alerts-panel {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    width: 350px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    z-index: 1000;
                    border-left: 4px solid #e74c3c;
                }
                
                .alert-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 15px 20px;
                    background: #e74c3c;
                    color: white;
                }
                
                .alerts-content {
                    padding: 15px 20px;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .alert-item {
                    padding: 10px;
                    margin: 5px 0;
                    border-left: 3px solid #f39c12;
                    background: #fdf2e9;
                    border-radius: 4px;
                }
                
                .alert-critical {
                    border-left-color: #e74c3c;
                    background: #fdedec;
                }
                
                .btn {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.3s;
                }
                
                .btn-primary {
                    background: #3498db;
                    color: white;
                }
                
                .btn-primary:hover {
                    background: #2980b9;
                }
                
                .btn-secondary {
                    background: #95a5a6;
                    color: white;
                }
                
                .btn-secondary:hover {
                    background: #7f8c8d;
                }
                
                .btn-small {
                    padding: 5px 10px;
                    font-size: 12px;
                    border: 1px solid #bdc3c7;
                    background: white;
                    border-radius: 3px;
                    cursor: pointer;
                }
                
                .btn-small:hover {
                    background: #ecf0f1;
                }
                
                .close-btn {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 20px;
                    cursor: pointer;
                    padding: 0;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                canvas {
                    display: block;
                    max-width: 100%;
                }
                
                select, input[type="number"], input[type="range"] {
                    padding: 8px;
                    border: 1px solid #bdc3c7;
                    border-radius: 4px;
                    font-size: 14px;
                }
                
                input[type="range"] {
                    width: 100px;
                }
                
                .checkbox-group {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin: 15px 0;
                }
                
                .checkbox-group label {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    cursor: pointer;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Processing mode change
        document.getElementById('processing-mode').addEventListener('change', (e) => {
            this.switchProcessingMode(e.target.value);
        });
        
        // Real-time toggle
        document.getElementById('real-time-toggle').addEventListener('click', () => {
            this.toggleRealTime();
        });
        
        // Parameter updates
        document.getElementById('confidence-threshold').addEventListener('input', (e) => {
            document.querySelector('#confidence-threshold + .param-value').textContent = e.target.value;
            this.updateProcessingParams();
        });
        
        document.getElementById('beamforming-method').addEventListener('change', () => {
            this.updateProcessingParams();
        });
        
        // Chart controls
        document.getElementById('auto-scale').addEventListener('click', () => {
            this.autoScaleWaveforms();
        });
        
        document.getElementById('filter-toggle').addEventListener('click', () => {
            this.toggleWaveformFilter();
        });
        
        document.getElementById('clear-log').addEventListener('click', () => {
            this.clearEventLog();
        });
        
        // Export results
        document.getElementById('export-results').addEventListener('click', () => {
            this.exportResults();
        });
        
        // Close alerts
        document.getElementById('close-alerts').addEventListener('click', () => {
            this.hideAlerts();
        });
        
        // Display options
        document.querySelectorAll('#visualization-tab input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateDisplayOptions());
        });
    }
    
    initializeCharts() {
        // Initialize all chart canvases
        const chartIds = ['waveforms-chart', 'phase-picks-chart', 'beam-power-chart', 
                         'array-response-chart', 'correlations-chart', 'velocity-changes-chart'];
        
        chartIds.forEach(id => {
            const canvas = document.getElementById(id);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                this.charts[id.replace('-chart', '').replace('-', '_')] = {
                    canvas: canvas,
                    context: ctx,
                    data: [],
                    options: {}
                };
            }
        });
        
        // Start animation loop
        this.startAnimationLoop();
    }
    
    loadStationConfiguration() {
        // Simulate loading station configuration
        const stations = [
            { id: 'STA01', name: 'Station 01', lat: 40.7128, lon: -74.0060, network: 'XX' },
            { id: 'STA02', name: 'Station 02', lat: 40.7589, lon: -73.9851, network: 'XX' },
            { id: 'STA03', name: 'Station 03', lat: 40.6892, lon: -74.0445, network: 'XX' },
            { id: 'STA04', name: 'Station 04', lat: 40.7282, lon: -73.7949, network: 'XX' },
            { id: 'STA05', name: 'Station 05', lat: 40.6782, lon: -73.9442, network: 'XX' }
        ];
        
        const stationList = document.getElementById('station-list');
        stationList.innerHTML = '';
        
        stations.forEach(station => {
            const stationItem = document.createElement('div');
            stationItem.className = 'station-item';
            stationItem.innerHTML = `
                <input type="checkbox" id="${station.id}" value="${station.id}">
                <label for="${station.id}">${station.name}</label>
            `;
            
            stationItem.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    const checkbox = stationItem.querySelector('input[type="checkbox"]');
                    checkbox.checked = !checkbox.checked;
                }
                this.updateStationSelection();
            });
            
            stationList.appendChild(stationItem);
        });
        
        // Draw array geometry
        this.drawArrayGeometry(stations);
    }
    
    drawArrayGeometry(stations) {
        const canvas = document.getElementById('array-geometry-canvas');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Simple visualization of station positions
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = 100;
        
        // Reference station at center
        const refLat = stations[0].lat;
        const refLon = stations[0].lon;
        
        stations.forEach((station, index) => {
            const x = centerX + (station.lon - refLon) * scale * 100;
            const y = centerY - (station.lat - refLat) * scale * 100;
            
            // Draw station
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = this.selectedStations.includes(station.id) ? '#3498db' : '#95a5a6';
            ctx.fill();
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(station.id, x, y - 15);
        });
        
        // Draw connections between selected stations
        if (this.selectedStations.length > 1) {
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            
            for (let i = 0; i < this.selectedStations.length - 1; i++) {
                const station1 = stations.find(s => s.id === this.selectedStations[i]);
                const station2 = stations.find(s => s.id === this.selectedStations[i + 1]);
                
                if (station1 && station2) {
                    const x1 = centerX + (station1.lon - refLon) * scale * 100;
                    const y1 = centerY - (station1.lat - refLat) * scale * 100;
                    const x2 = centerX + (station2.lon - refLon) * scale * 100;
                    const y2 = centerY - (station2.lat - refLat) * scale * 100;
                    
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            }
            ctx.setLineDash([]);
        }
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }
    
    switchProcessingMode(mode) {
        this.processingMode = mode;
        
        // Update parameter visibility
        document.querySelectorAll('.param-group').forEach(group => {
            group.style.display = 'none';
        });
        
        // Update chart visibility
        document.getElementById('array-container').style.display = 'none';
        document.getElementById('ambient-noise-container').style.display = 'none';
        
        switch (mode) {
            case 'transformer':
                document.getElementById('transformer-params').style.display = 'block';
                break;
            case 'array':
                document.getElementById('array-params').style.display = 'block';
                document.getElementById('array-container').style.display = 'block';
                break;
            case 'ambient_noise':
                document.getElementById('noise-params').style.display = 'block';
                document.getElementById('ambient-noise-container').style.display = 'block';
                break;
        }
        
        this.logEvent('info', `Switched to ${mode} processing mode`);
    }
    
    toggleRealTime() {
        const button = document.getElementById('real-time-toggle');
        
        if (this.isRealTime) {
            this.stopRealTime();
            button.textContent = 'Start Real-time';
            button.className = 'btn btn-primary';
        } else {
            this.startRealTime();
            button.textContent = 'Stop Real-time';
            button.className = 'btn btn-warning';
        }
    }
    
    startRealTime() {
        if (this.selectedStations.length === 0) {
            this.showAlert('Please select at least one station', 'warning');
            return;
        }
        
        this.isRealTime = true;
        this.connectWebSocket();
        this.updateStatus('processing', 'Processing');
        this.logEvent('success', 'Real-time processing started');
    }
    
    stopRealTime() {
        this.isRealTime = false;
        if (this.websocket) {
            this.websocket.close();
        }
        this.updateStatus('connection', 'Disconnected');
        this.updateStatus('processing', 'Idle');
        this.logEvent('info', 'Real-time processing stopped');
    }
    
    connectWebSocket() {
        try {
            // In a real implementation, this would connect to your backend WebSocket
            const wsUrl = `ws://localhost:8080/seismic-stream?stations=${this.selectedStations.join(',')}`;
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                this.updateStatus('connection', 'Connected');
                this.reconnectAttempts = 0;
                this.logEvent('success', 'WebSocket connected');
            };
            
            this.websocket.onmessage = (event) => {
                this.handleRealtimeData(JSON.parse(event.data));
            };
            
            this.websocket.onclose = () => {
                this.updateStatus('connection', 'Disconnected');
                if (this.isRealTime && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.logEvent('warning', `Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.connectWebSocket(), 2000);
                }
            };
            
            this.websocket.onerror = (error) => {
                this.logEvent('error', `WebSocket error: ${error.message}`);
            };
        } catch (error) {
            // Simulate real-time data for demo
            this.simulateRealTimeData();
        }
    }
    
    simulateRealTimeData() {
        if (!this.isRealTime) return;
        
        this.updateStatus('connection', 'Connected');
        
        // Generate synthetic data
        const timestamp = new Date();
        const sampleRate = 50; // Hz
        const duration = 1; // seconds
        const nSamples = sampleRate * duration;
        
        const data = {
            timestamp: timestamp.toISOString(),
            type: this.processingMode,
            stations: {}
        };
        
        // Generate waveform data
        this.selectedStations.forEach(stationId => {
            const waveform = [];
            for (let i = 0; i < nSamples; i++) {
                const t = i / sampleRate;
                let signal = 0;
                
                // Add some random noise
                signal += (Math.random() - 0.5) * 0.1;
                
                // Occasionally add a simulated earthquake signal
                if (Math.random() < 0.001) {
                    const freq = 2 + Math.random() * 8; // 2-10 Hz
                    signal += Math.sin(2 * Math.PI * freq * t) * Math.exp(-t * 2) * 2;
                }
                
                waveform.push(signal);
            }
            
            data.stations[stationId] = {
                waveform: waveform,
                sample_rate: sampleRate
            };
        });
        
        // Add processing results based on mode
        if (this.processingMode === 'transformer') {
            data.phase_picks = this.simulatePhasePickResults();
        } else if (this.processingMode === 'array') {
            data.array_results = this.simulateArrayResults();
        } else if (this.processingMode === 'ambient_noise') {
            data.noise_results = this.simulateNoiseResults();
        }
        
        this.handleRealtimeData(data);
        
        // Continue simulation
        setTimeout(() => this.simulateRealTimeData(), 1000);
    }
    
    simulatePhasePickResults() {
        const results = {
            p_picks: [],
            s_picks: [],
            magnitude: null,
            first_motion: null
        };
        
        // Occasionally simulate phase picks
        if (Math.random() < 0.1) {
            const pTime = Math.random() * 60; // within 60 seconds
            results.p_picks.push({
                time: pTime,
                confidence: 0.7 + Math.random() * 0.3,
                station: this.selectedStations[0]
            });
            
            if (Math.random() < 0.8) {
                results.s_picks.push({
                    time: pTime + 2 + Math.random() * 8,
                    confidence: 0.6 + Math.random() * 0.3,
                    station: this.selectedStations[0]
                });
            }
            
            if (Math.random() < 0.3) {
                results.magnitude = {
                    value: 2.0 + Math.random() * 4.0,
                    uncertainty: 0.2 + Math.random() * 0.3
                };
                
                results.first_motion = {
                    polarity: Math.random() < 0.5 ? 'up' : 'down',
                    confidence: 0.6 + Math.random() * 0.4
                };
            }
        }
        
        return results;
    }
    
    simulateArrayResults() {
        return {
            best_slowness: 0.05 + Math.random() * 0.2,
            best_azimuth: Math.random() * 360,
            coherence: 0.3 + Math.random() * 0.6,
            beam_power: Array.from({length: 100}, () => Math.random()),
            source_type: ['local', 'regional', 'teleseismic', 'noise'][Math.floor(Math.random() * 4)]
        };
    }
    
    simulateNoiseResults() {
        return {
            correlation_quality: 0.4 + Math.random() * 0.5,
            velocity_change: (Math.random() - 0.5) * 0.02, // ±1%
            health_status: ['excellent', 'good', 'fair', 'poor'][Math.floor(Math.random() * 4)]
        };
    }
    
    handleRealtimeData(data) {
        const timestamp = new Date(data.timestamp);
        
        // Update waveform displays
        if (data.stations) {
            this.updateWaveformDisplay(data.stations, timestamp);
        }
        
        // Update analysis results based on processing mode
        switch (data.type) {
            case 'transformer':
                if (data.phase_picks) {
                    this.updatePhasePickDisplay(data.phase_picks, timestamp);
                }
                break;
            case 'array':
                if (data.array_results) {
                    this.updateArrayDisplay(data.array_results, timestamp);
                }
                break;
            case 'ambient_noise':
                if (data.noise_results) {
                    this.updateNoiseDisplay(data.noise_results, timestamp);
                }
                break;
        }
        
        this.checkForAlerts(data);
    }
    
    updateWaveformDisplay(stationsData, timestamp) {
        const canvas = document.getElementById('waveforms-chart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'];
        const stationIds = Object.keys(stationsData);
        const nStations = stationIds.length;
        
        if (nStations === 0) return;
        
        const traceHeight = canvas.height / nStations;
        const padding = 20;
        
        stationIds.forEach((stationId, index) => {
            const waveform = stationsData[stationId].waveform;
            const y0 = (index + 0.5) * traceHeight;
            
            ctx.strokeStyle = colors[index % colors.length];
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            
            // Normalize and plot waveform
            const maxAmp = Math.max(...waveform.map(Math.abs));
            const scale = (traceHeight - padding) / 2 / Math.max(maxAmp, 0.1);
            
            waveform.forEach((sample, i) => {
                const x = (i / waveform.length) * (canvas.width - 2 * padding) + padding;
                const y = y0 - sample * scale;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Station label
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.fillText(stationId, 5, y0 - traceHeight/4);
        });
        
        // Time axis
        ctx.strokeStyle = '#bdc3c7';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, canvas.height - 20);
        ctx.lineTo(canvas.width - padding, canvas.height - 20);
        ctx.stroke();
        
        // Store latest data for animation
        this.waveformData.set(timestamp, stationsData);
    }
    
    updatePhasePickDisplay(phaseData, timestamp) {
        // Update phase pick statistics
        document.getElementById('p-pick-count').textContent = `P: ${phaseData.p_picks.length}`;
        document.getElementById('s-pick-count').textContent = `S: ${phaseData.s_picks.length}`;
        
        if (phaseData.magnitude) {
            document.getElementById('magnitude-estimate').textContent = 
                `M: ${phaseData.magnitude.value.toFixed(1)}`;
        }
        
        // Draw phase picks on canvas
        const canvas = document.getElementById('phase-picks-chart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw P picks
        ctx.fillStyle = '#3498db';
        phaseData.p_picks.forEach(pick => {
            const x = (pick.time / 60) * canvas.width; // Assuming 60s window
            const confidence = pick.confidence;
            
            ctx.globalAlpha = confidence;
            ctx.fillRect(x - 2, 10, 4, canvas.height - 40);
            ctx.fillText('P', x - 5, 25);
        });
        
        // Draw S picks
        ctx.fillStyle = '#e74c3c';
        phaseData.s_picks.forEach(pick => {
            const x = (pick.time / 60) * canvas.width;
            const confidence = pick.confidence;
            
            ctx.globalAlpha = confidence;
            ctx.fillRect(x - 2, 10, 4, canvas.height - 40);
            ctx.fillText('S', x - 5, 45);
        });
        
        ctx.globalAlpha = 1.0;
        
        // Log significant events
        if (phaseData.p_picks.length > 0 || phaseData.s_picks.length > 0) {
            this.logEvent('success', `Phase picks: ${phaseData.p_picks.length} P, ${phaseData.s_picks.length} S`);
            
            if (phaseData.magnitude) {
                this.logEvent('info', `Magnitude estimate: ${phaseData.magnitude.value.toFixed(1)} ± ${phaseData.magnitude.uncertainty.toFixed(1)}`);
            }
        }
    }
    
    updateArrayDisplay(arrayData, timestamp) {
        // Update array statistics
        document.getElementById('best-slowness').textContent = 
            `Slowness: ${arrayData.best_slowness.toFixed(3)} s/km`;
        document.getElementById('best-azimuth').textContent = 
            `Azimuth: ${arrayData.best_azimuth.toFixed(1)}°`;
        document.getElementById('coherence').textContent = 
            `Coherence: ${arrayData.coherence.toFixed(2)}`;
        
        // Draw beam power plot
        const canvas = document.getElementById('beam-power-chart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (arrayData.beam_power && arrayData.beam_power.length > 0) {
            const maxPower = Math.max(...arrayData.beam_power);
            const gridSize = Math.sqrt(arrayData.beam_power.length);
            
            arrayData.beam_power.forEach((power, index) => {
                const x = (index % gridSize) / gridSize * canvas.width;
                const y = Math.floor(index / gridSize) / gridSize * canvas.height;
                const intensity = power / maxPower;
                
                ctx.fillStyle = `rgba(255, ${Math.floor(255 * (1 - intensity))}, 0, ${intensity})`;
                ctx.fillRect(x, y, canvas.width / gridSize, canvas.height / gridSize);
            });
        }
        
        // Log significant detections
        if (arrayData.coherence > 0.7) {
            this.logEvent('success', `High coherence detection: ${arrayData.source_type} source`);
        }
    }
    
    updateNoiseDisplay(noiseData, timestamp) {
        // Update noise monitoring statistics
        document.getElementById('correlation-quality').textContent = 
            `Quality: ${noiseData.correlation_quality.toFixed(2)}`;
        document.getElementById('velocity-change').textContent = 
            `dV/V: ${(noiseData.velocity_change * 100).toFixed(2)}%`;
        document.getElementById('health-status').textContent = 
            `Health: ${noiseData.health_status}`;
        
        // Store velocity change data for trending
        this.velocityChangeData.set(timestamp, noiseData.velocity_change);
        
        // Draw velocity changes over time
        const canvas = document.getElementById('velocity-changes-chart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const data = Array.from(this.velocityChangeData.entries()).slice(-100); // Last 100 points
        if (data.length > 1) {
            const maxChange = Math.max(...data.map(d => Math.abs(d[1])));
            const scale = canvas.height / 2 / Math.max(maxChange, 0.01);
            
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            data.forEach((point, index) => {
                const x = (index / (data.length - 1)) * canvas.width;
                const y = canvas.height / 2 - point[1] * scale;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Zero line
            ctx.strokeStyle = '#bdc3c7';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        }
        
        // Check for significant velocity changes
        if (Math.abs(noiseData.velocity_change) > this.processingParams.ambient_noise.velocity_threshold / 100) {
            this.logEvent('warning', `Significant velocity change detected: ${(noiseData.velocity_change * 100).toFixed(2)}%`);
        }
    }
    
    checkForAlerts(data) {
        const alerts = [];
        
        // Check for significant events based on processing mode
        if (data.type === 'transformer' && data.phase_picks) {
            if (data.phase_picks.magnitude && data.phase_picks.magnitude.value > 4.0) {
                alerts.push({
                    type: 'critical',
                    message: `Large magnitude event detected: M${data.phase_picks.magnitude.value.toFixed(1)}`,
                    timestamp: new Date(data.timestamp)
                });
            }
        }
        
        if (data.type === 'array' && data.array_results) {
            if (data.array_results.coherence > 0.8 && data.array_results.best_slowness < 0.1) {
                alerts.push({
                    type: 'warning',
                    message: `High coherence local detection: ${data.array_results.source_type}`,
                    timestamp: new Date(data.timestamp)
                });
            }
        }
        
        if (data.type === 'ambient_noise' && data.noise_results) {
            if (data.noise_results.health_status === 'critical') {
                alerts.push({
                    type: 'critical',
                    message: `Critical structural health status detected`,
                    timestamp: new Date(data.timestamp)
                });
            }
        }
        
        // Show alerts if any
        if (alerts.length > 0) {
            this.showAlerts(alerts);
        }
    }
    
    showAlert(message, type = 'info') {
        this.showAlerts([{ type, message, timestamp: new Date() }]);
    }
    
    showAlerts(alerts) {
        const alertsPanel = document.getElementById('alerts-panel');
        const alertsContent = document.getElementById('alerts-content');
        
        alerts.forEach(alert => {
            const alertItem = document.createElement('div');
            alertItem.className = `alert-item ${alert.type === 'critical' ? 'alert-critical' : ''}`;
            alertItem.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 5px;">
                    ${alert.type.toUpperCase()}: ${alert.message}
                </div>
                <div style="font-size: 12px; color: #7f8c8d;">
                    ${alert.timestamp.toLocaleTimeString()}
                </div>
            `;
            
            alertsContent.insertBefore(alertItem, alertsContent.firstChild);
        });
        
        // Remove old alerts (keep last 10)
        while (alertsContent.children.length > 10) {
            alertsContent.removeChild(alertsContent.lastChild);
        }
        
        alertsPanel.style.display = 'block';
        
        // Auto-hide info alerts after 5 seconds
        alerts.forEach(alert => {
            if (alert.type === 'info') {
                setTimeout(() => {
                    if (alertsContent.children.length <= 1) {
                        this.hideAlerts();
                    }
                }, 5000);
            }
        });
    }
    
    hideAlerts() {
        document.getElementById('alerts-panel').style.display = 'none';
    }
    
    updateStationSelection() {
        this.selectedStations = Array.from(
            document.querySelectorAll('#station-list input:checked')
        ).map(input => input.value);
        
        // Update visual selection
        document.querySelectorAll('.station-item').forEach(item => {
            const checkbox = item.querySelector('input');
            item.classList.toggle('selected', checkbox.checked);
        });
        
        // Redraw array geometry
        const stations = [
            { id: 'STA01', name: 'Station 01', lat: 40.7128, lon: -74.0060, network: 'XX' },
            { id: 'STA02', name: 'Station 02', lat: 40.7589, lon: -73.9851, network: 'XX' },
            { id: 'STA03', name: 'Station 03', lat: 40.6892, lon: -74.0445, network: 'XX' },
            { id: 'STA04', name: 'Station 04', lat: 40.7282, lon: -73.7949, network: 'XX' },
            { id: 'STA05', name: 'Station 05', lat: 40.6782, lon: -73.9442, network: 'XX' }
        ];
        this.drawArrayGeometry(stations);
        
        this.logEvent('info', `Selected stations: ${this.selectedStations.join(', ')}`);
    }
    
    updateProcessingParams() {
        // Update processing parameters based on current values
        const mode = this.processingMode;
        
        if (mode === 'transformer') {
            this.processingParams.transformer.confidence_threshold = 
                parseFloat(document.getElementById('confidence-threshold').value);
            this.processingParams.transformer.window_size = 
                parseInt(document.getElementById('window-size').value);
        } else if (mode === 'array') {
            this.processingParams.array.frequency_range = [
                parseFloat(document.getElementById('freq-low').value),
                parseFloat(document.getElementById('freq-high').value)
            ];
            this.processingParams.array.beamforming_method = 
                document.getElementById('beamforming-method').value;
        } else if (mode === 'ambient_noise') {
            this.processingParams.ambient_noise.correlation_window = 
                parseInt(document.getElementById('correlation-window').value);
            this.processingParams.ambient_noise.velocity_threshold = 
                parseFloat(document.getElementById('velocity-threshold').value) / 100;
        }
        
        this.logEvent('info', `Updated ${mode} processing parameters`);
    }
    
    updateDisplayOptions() {
        // Update chart visibility based on checkboxes
        const options = {
            waveforms: document.getElementById('show-waveforms').checked,
            phase_picks: document.getElementById('show-phase-picks').checked,
            beam_power: document.getElementById('show-beam-power').checked,
            correlations: document.getElementById('show-correlations').checked,
            velocity_changes: document.getElementById('show-velocity-changes').checked
        };
        
        // Apply visibility settings
        document.getElementById('waveforms-container').style.display = 
            options.waveforms ? 'block' : 'none';
        document.getElementById('phase-picks-container').style.display = 
            options.phase_picks ? 'block' : 'none';
    }
    
    updateStatus(type, status) {
        const element = document.getElementById(`${type}-status`);
        if (element) {
            element.textContent = status;
            element.className = `status-${status.toLowerCase().replace(' ', '-')}`;
        }
    }
    
    logEvent(type, message) {
        const eventLog = document.getElementById('event-log');
        const timestamp = new Date().toLocaleTimeString();
        
        const eventEntry = document.createElement('div');
        eventEntry.className = `event-entry event-${type}`;
        eventEntry.textContent = `[${timestamp}] ${message}`;
        
        eventLog.insertBefore(eventEntry, eventLog.firstChild);
        
        // Keep only last 100 events
        while (eventLog.children.length > 100) {
            eventLog.removeChild(eventLog.lastChild);
        }
        
        // Auto-scroll to top
        eventLog.scrollTop = 0;
    }
    
    clearEventLog() {
        document.getElementById('event-log').innerHTML = '';
        this.logEvent('info', 'Event log cleared');
    }
    
    autoScaleWaveforms() {
        this.logEvent('info', 'Auto-scaling waveforms');
        // Implementation would rescale the waveform display
    }
    
    toggleWaveformFilter() {
        const button = document.getElementById('filter-toggle');
        const isFilterOn = button.textContent.includes('ON');
        
        button.textContent = `Filter: ${isFilterOn ? 'OFF' : 'ON'}`;
        this.logEvent('info', `Waveform filter ${isFilterOn ? 'disabled' : 'enabled'}`);
    }
    
    startAnimationLoop() {
        const animate = () => {
            // Smooth animations for charts would go here
            requestAnimationFrame(animate);
        };
        animate();
    }
    
    exportResults() {
        const results = {
            timestamp: new Date().toISOString(),
            processing_mode: this.processingMode,
            selected_stations: this.selectedStations,
            parameters: this.processingParams[this.processingMode],
            waveform_data: Array.from(this.waveformData.entries()).slice(-10),
            analysis_results: this.analysisResults
        };
        
        // Create and download JSON file
        const blob = new Blob([JSON.stringify(results, null, 2)], 
                            { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `seismic_analysis_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.logEvent('success', 'Results exported to file');
    }
    
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        this.isRealTime = false;
        this.container.innerHTML = '';
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('seismic-dashboard-container')) {
        const dashboard = new SeismicDashboard('seismic-dashboard-container');
        window.seismicDashboard = dashboard; // Make globally accessible
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SeismicDashboard;
}