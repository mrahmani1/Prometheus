/**
 * Advanced Geological Anomaly Detection Dashboard
 * Real-time monitoring and analysis with comprehensive causal discovery
 */

class GeologicalAnomalyDashboard {
    constructor(containerId, config = {}) {
        this.container = document.getElementById(containerId);
        this.config = {
            apiEndpoint: config.apiEndpoint || '/api/v1/anomaly-detection',
            updateInterval: config.updateInterval || 5000,
            maxDataPoints: config.maxDataPoints || 1000,
            enableRealTime: config.enableRealTime !== false,
            enableWebSocket: config.enableWebSocket !== false,
            ...config
        };
        
        this.charts = {};
        this.data = {
            timeSeries: [],
            anomalies: [],
            patterns: [],
            causalRelationships: []
        };
        
        this.websocket = null;
        this.isConnected = false;
        this.updateTimer = null;
        
        this.initialize();
    }
    
    async initialize() {
        try {
            this.createDashboardLayout();
            this.initializeCharts();
            
            if (this.config.enableWebSocket) {
                await this.setupWebSocket();
            }
            
            if (this.config.enableRealTime) {
                this.startRealTimeUpdates();
            }
            
            await this.loadInitialData();
            
            console.log('Geological Anomaly Dashboard initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize Geological Anomaly Dashboard:', error);
            this.showError('Failed to initialize dashboard: ' + error.message);
        }
    }
    
    createDashboardLayout() {
        this.container.innerHTML = `
            <div class="anomaly-dashboard">
                <!-- Header with controls -->
                <div class="dashboard-header">
                    <h2>Geological Anomaly Detection</h2>
                    <div class="dashboard-controls">
                        <button id="realTimeToggle" class="btn btn-primary">
                            <i class="fas fa-pause"></i> Real-Time
                        </button>
                        <button id="detectAnomalies" class="btn btn-success">
                            <i class="fas fa-search"></i> Detect Anomalies
                        </button>
                        <button id="causalAnalysis" class="btn btn-info">
                            <i class="fas fa-project-diagram"></i> Causal Analysis
                        </button>
                        <select id="timeRange" class="form-control">
                            <option value="1h">Last Hour</option>
                            <option value="24h" selected>Last 24 Hours</option>
                            <option value="7d">Last 7 Days</option>
                            <option value="30d">Last 30 Days</option>
                        </select>
                    </div>
                </div>
                
                <!-- Status indicators -->
                <div class="status-row">
                    <div class="status-card">
                        <div class="status-icon">
                            <i class="fas fa-wifi"></i>
                        </div>
                        <div class="status-content">
                            <div class="status-title">Connection</div>
                            <div id="connectionStatus" class="status-value">Connecting...</div>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <div class="status-icon alert">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <div class="status-content">
                            <div class="status-title">Active Anomalies</div>
                            <div id="activeAnomalies" class="status-value">0</div>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <div class="status-icon success">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="status-content">
                            <div class="status-title">Detection Rate</div>
                            <div id="detectionRate" class="status-value">0/min</div>
                        </div>
                    </div>
                    
                    <div class="status-card">
                        <div class="status-icon info">
                            <i class="fas fa-brain"></i>
                        </div>
                        <div class="status-content">
                            <div class="status-title">Causal Links</div>
                            <div id="causalLinks" class="status-value">0</div>
                        </div>
                    </div>
                </div>
                
                <!-- Main content area -->
                <div class="dashboard-grid">
                    <!-- Real-time anomaly chart -->
                    <div class="chart-container">
                        <div class="chart-header">
                            <h3>Real-Time Anomaly Scores</h3>
                            <div class="chart-controls">
                                <button id="toggleAnomalyThreshold" class="btn btn-sm btn-outline-primary">
                                    Threshold
                                </button>
                            </div>
                        </div>
                        <div id="anomalyTimeSeriesChart" class="chart"></div>
                    </div>
                    
                    <!-- Pattern analysis -->
                    <div class="chart-container">
                        <div class="chart-header">
                            <h3>Detected Patterns</h3>
                            <div class="chart-controls">
                                <select id="patternType" class="form-control form-control-sm">
                                    <option value="">All Types</option>
                                    <option value="seismic">Seismic</option>
                                    <option value="volcanic">Volcanic</option>
                                    <option value="thermal">Thermal</option>
                                    <option value="geochemical">Geochemical</option>
                                </select>
                            </div>
                        </div>
                        <div id="patternChart" class="chart"></div>
                    </div>
                    
                    <!-- Causal network visualization -->
                    <div class="chart-container full-width">
                        <div class="chart-header">
                            <h3>Causal Relationship Network</h3>
                            <div class="chart-controls">
                                <input type="range" id="strengthFilter" min="0" max="1" step="0.1" value="0.3">
                                <label for="strengthFilter">Min Strength: 0.3</label>
                            </div>
                        </div>
                        <div id="causalNetworkChart" class="chart network-chart"></div>
                    </div>
                    
                    <!-- Anomaly details table -->
                    <div class="chart-container full-width">
                        <div class="chart-header">
                            <h3>Anomaly Details</h3>
                            <div class="chart-controls">
                                <button id="exportAnomalies" class="btn btn-sm btn-outline-success">
                                    <i class="fas fa-download"></i> Export
                                </button>
                            </div>
                        </div>
                        <div id="anomalyTable" class="table-container"></div>
                    </div>
                </div>
                
                <!-- Alert panel -->
                <div id="alertPanel" class="alert-panel"></div>
            </div>
        `;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Real-time toggle
        document.getElementById('realTimeToggle').addEventListener('click', () => {
            this.toggleRealTime();
        });
        
        // Detect anomalies button
        document.getElementById('detectAnomalies').addEventListener('click', () => {
            this.performAnomalyDetection();
        });
        
        // Causal analysis button
        document.getElementById('causalAnalysis').addEventListener('click', () => {
            this.performCausalAnalysis();
        });
        
        // Time range selector
        document.getElementById('timeRange').addEventListener('change', (e) => {
            this.updateTimeRange(e.target.value);
        });
        
        // Pattern type filter
        document.getElementById('patternType').addEventListener('change', (e) => {
            this.filterPatterns(e.target.value);
        });
        
        // Strength filter for causal network
        document.getElementById('strengthFilter').addEventListener('input', (e) => {
            document.querySelector('label[for="strengthFilter"]').textContent = 
                `Min Strength: ${e.target.value}`;
            this.updateCausalNetwork(parseFloat(e.target.value));
        });
        
        // Export anomalies
        document.getElementById('exportAnomalies').addEventListener('click', () => {
            this.exportAnomalies();
        });
    }
    
    initializeCharts() {
        // Initialize anomaly time series chart
        this.charts.anomalyTimeSeries = new Chart(
            document.getElementById('anomalyTimeSeriesChart'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Anomaly Score',
                            data: [],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            borderWidth: 2,
                            fill: true
                        },
                        {
                            label: 'Threshold',
                            data: [],
                            borderColor: '#f39c12',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                displayFormats: {
                                    minute: 'HH:mm'
                                }
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    }
                }
            }
        );
        
        // Initialize pattern chart
        this.charts.patterns = new Chart(
            document.getElementById('patternChart'),
            {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            }
        );
        
        // Initialize causal network using D3.js
        this.initializeCausalNetwork();
        
        // Initialize anomaly table
        this.initializeAnomalyTable();
    }
    
    initializeCausalNetwork() {
        const container = d3.select('#causalNetworkChart');
        const width = container.node().offsetWidth;
        const height = 400;
        
        this.networkSvg = container
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));
    }
    
    initializeAnomalyTable() {
        const container = document.getElementById('anomalyTable');
        container.innerHTML = `
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Confidence</th>
                        <th>Strength</th>
                        <th>Location</th>
                        <th>Causal Factors</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="anomalyTableBody">
                </tbody>
            </table>
        `;
    }
    
    async setupWebSocket() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws/earth-state`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus('Connected', 'success');
                console.log('WebSocket connected');
                
                // Subscribe to anomaly updates
                this.websocket.send(JSON.stringify({
                    type: 'subscribe',
                    stream: 'anomaly_detection'
                }));
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('Disconnected', 'error');
                console.log('WebSocket disconnected');
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.setupWebSocket();
                    }
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error', 'error');
            };
            
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.updateConnectionStatus('Failed', 'error');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'anomaly_detected':
                this.handleAnomalyUpdate(data.data);
                break;
            case 'pattern_discovered':
                this.handlePatternUpdate(data.data);
                break;
            case 'causal_relationship_updated':
                this.handleCausalUpdate(data.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }
    
    handleAnomalyUpdate(anomalyData) {
        // Add to time series data
        const timestamp = new Date(anomalyData.timestamp);
        this.data.timeSeries.push({
            timestamp,
            anomalyScore: anomalyData.anomaly_score,
            isAnomaly: anomalyData.is_anomaly
        });
        
        // Keep only recent data points
        if (this.data.timeSeries.length > this.config.maxDataPoints) {
            this.data.timeSeries = this.data.timeSeries.slice(-this.config.maxDataPoints);
        }
        
        // Update charts
        this.updateAnomalyTimeSeriesChart();
        
        // Show alert if high-confidence anomaly
        if (anomalyData.is_anomaly && anomalyData.confidence > 0.8) {
            this.showAnomalyAlert(anomalyData);
        }
        
        // Update status indicators
        this.updateStatusIndicators();
    }
    
    async loadInitialData() {
        try {
            // Load historical patterns
            const patternsResponse = await fetch(`${this.config.apiEndpoint}/patterns`, {
                headers: {
                    'Authorization': `Bearer ${this.getAuthToken()}`
                }
            });
            
            if (patternsResponse.ok) {
                const patternsData = await patternsResponse.json();
                this.data.patterns = patternsData.patterns;
                this.updatePatternChart();
            }
            
            // Load initial causal relationships
            await this.loadCausalRelationships();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load initial data: ' + error.message);
        }
    }
    
    async loadCausalRelationships() {
        try {
            // Get causal relationships for key geological variables
            const variables = ['seismic_activity', 'thermal_anomaly', 'gas_emissions', 'deformation_rate'];
            const allRelationships = [];
            
            for (const variable of variables) {
                const response = await fetch(`${this.config.apiEndpoint}/causality/${variable}`, {
                    headers: {
                        'Authorization': `Bearer ${this.getAuthToken()}`
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    allRelationships.push(...data.causal_relationships);
                }
            }
            
            this.data.causalRelationships = allRelationships;
            this.updateCausalNetwork(0.3);
            
        } catch (error) {
            console.error('Failed to load causal relationships:', error);
        }
    }
    
    async performAnomalyDetection() {
        try {
            const button = document.getElementById('detectAnomalies');
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            // Get sample data for analysis (in real implementation, this would be actual geological data)
            const sampleData = this.generateSampleData();
            
            const response = await fetch(`${this.config.apiEndpoint}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                },
                body: JSON.stringify({
                    data: sampleData
                })
            });
            
            if (response.ok) {
                const results = await response.json();
                this.processAnomalyResults(results);
                this.showSuccess('Anomaly detection completed successfully');
            } else {
                throw new Error('Anomaly detection failed');
            }
            
        } catch (error) {
            console.error('Anomaly detection error:', error);
            this.showError('Anomaly detection failed: ' + error.message);
        } finally {
            const button = document.getElementById('detectAnomalies');
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-search"></i> Detect Anomalies';
        }
    }
    
    async performCausalAnalysis() {
        try {
            const button = document.getElementById('causalAnalysis');
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            await this.loadCausalRelationships();
            this.showSuccess('Causal analysis completed');
            
        } catch (error) {
            console.error('Causal analysis error:', error);
            this.showError('Causal analysis failed: ' + error.message);
        } finally {
            const button = document.getElementById('causalAnalysis');
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-project-diagram"></i> Causal Analysis';
        }
    }
    
    updateAnomalyTimeSeriesChart() {
        const chart = this.charts.anomalyTimeSeries;
        const recentData = this.data.timeSeries.slice(-100); // Last 100 points
        
        chart.data.labels = recentData.map(d => d.timestamp);
        chart.data.datasets[0].data = recentData.map(d => d.anomalyScore);
        chart.data.datasets[1].data = recentData.map(d => 0.7); // Threshold line
        
        chart.update('none'); // No animation for real-time updates
    }
    
    updatePatternChart() {
        const chart = this.charts.patterns;
        const patternTypes = {};
        
        this.data.patterns.forEach(pattern => {
            patternTypes[pattern.pattern_type] = (patternTypes[pattern.pattern_type] || 0) + 1;
        });
        
        chart.data.labels = Object.keys(patternTypes);
        chart.data.datasets[0].data = Object.values(patternTypes);
        chart.update();
    }
    
    updateCausalNetwork(minStrength = 0.3) {
        const filteredRelationships = this.data.causalRelationships
            .filter(rel => rel.strength >= minStrength);
        
        const nodes = new Set();
        const links = [];
        
        filteredRelationships.forEach(rel => {
            nodes.add(rel.cause_variable);
            nodes.add(rel.effect_variable);
            links.push({
                source: rel.cause_variable,
                target: rel.effect_variable,
                strength: rel.strength,
                confidence: rel.confidence
            });
        });
        
        const nodeArray = Array.from(nodes).map(id => ({ id }));
        
        this.networkSvg.selectAll('*').remove();
        
        const link = this.networkSvg
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-width', d => Math.sqrt(d.strength * 10));
        
        const node = this.networkSvg
            .selectAll('circle')
            .data(nodeArray)
            .enter().append('circle')
            .attr('r', 10)
            .attr('fill', '#3498db')
            .call(d3.drag()
                .on('start', this.dragStarted.bind(this))
                .on('drag', this.dragged.bind(this))
                .on('end', this.dragEnded.bind(this)));
        
        const text = this.networkSvg
            .selectAll('text')
            .data(nodeArray)
            .enter().append('text')
            .text(d => d.id.replace('_', ' '))
            .attr('font-size', '12px')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em');
        
        this.simulation
            .nodes(nodeArray)
            .on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                text
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });
        
        this.simulation.force('link').links(links);
        this.simulation.restart();
    }
    
    // D3 drag handlers
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    toggleRealTime() {
        const button = document.getElementById('realTimeToggle');
        const icon = button.querySelector('i');
        
        if (this.config.enableRealTime) {
            this.stopRealTimeUpdates();
            this.config.enableRealTime = false;
            icon.className = 'fas fa-play';
            button.classList.remove('btn-primary');
            button.classList.add('btn-secondary');
        } else {
            this.startRealTimeUpdates();
            this.config.enableRealTime = true;
            icon.className = 'fas fa-pause';
            button.classList.remove('btn-secondary');
            button.classList.add('btn-primary');
        }
    }
    
    startRealTimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }
        
        this.updateTimer = setInterval(() => {
            if (!this.isConnected) {
                // Simulate real-time data if WebSocket is not connected
                this.simulateRealTimeData();
            }
        }, this.config.updateInterval);
    }
    
    stopRealTimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }
    
    simulateRealTimeData() {
        // Generate simulated anomaly data for demonstration
        const now = new Date();
        const anomalyScore = Math.random() * 0.3 + (Math.random() > 0.9 ? 0.6 : 0);
        const isAnomaly = anomalyScore > 0.7;
        
        this.handleAnomalyUpdate({
            timestamp: now.toISOString(),
            anomaly_score: anomalyScore,
            is_anomaly: isAnomaly,
            confidence: Math.random() * 0.4 + 0.6
        });
    }
    
    generateSampleData() {
        // Generate sample multivariate geological data
        const numSamples = 1000;
        const numFeatures = 8;
        const data = [];
        
        for (let i = 0; i < numSamples; i++) {
            const sample = [];
            for (let j = 0; j < numFeatures; j++) {
                // Add some anomalies
                let value = Math.random();
                if (Math.random() > 0.95) {
                    value += 2 + Math.random(); // Create anomaly
                }
                sample.push(value);
            }
            data.push(sample);
        }
        
        return data;
    }
    
    showAnomalyAlert(anomalyData) {
        const alertPanel = document.getElementById('alertPanel');
        const alertId = `alert_${Date.now()}`;
        
        const alert = document.createElement('div');
        alert.id = alertId;
        alert.className = 'alert alert-danger alert-dismissible fade show';
        alert.innerHTML = `
            <strong>High-Confidence Anomaly Detected!</strong>
            <br>Score: ${anomalyData.anomaly_score.toFixed(3)}
            <br>Confidence: ${(anomalyData.confidence * 100).toFixed(1)}%
            <br>Time: ${new Date(anomalyData.timestamp).toLocaleString()}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;
        
        alertPanel.appendChild(alert);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                alertElement.remove();
            }
        }, 10000);
    }
    
    updateStatusIndicators() {
        // Update active anomalies count
        const activeAnomalies = this.data.timeSeries
            .filter(d => d.isAnomaly && (Date.now() - d.timestamp) < 3600000).length;
        document.getElementById('activeAnomalies').textContent = activeAnomalies;
        
        // Update detection rate
        const recentDetections = this.data.timeSeries
            .filter(d => (Date.now() - d.timestamp) < 60000).length;
        document.getElementById('detectionRate').textContent = `${recentDetections}/min`;
        
        // Update causal links count
        document.getElementById('causalLinks').textContent = this.data.causalRelationships.length;
    }
    
    updateConnectionStatus(status, type) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.textContent = status;
        statusElement.className = `status-value ${type}`;
    }
    
    showError(message) {
        console.error(message);
        // Implement error display logic
    }
    
    showSuccess(message) {
        console.log(message);
        // Implement success display logic
    }
    
    getAuthToken() {
        // Return authentication token
        return localStorage.getItem('auth_token') || 'test_key_123';
    }
    
    exportAnomalies() {
        const csvContent = "data:text/csv;charset=utf-8," + 
            "Timestamp,Anomaly Score,Is Anomaly,Confidence\n" +
            this.data.timeSeries.map(d => 
                `${d.timestamp.toISOString()},${d.anomalyScore},${d.isAnomaly},${d.confidence || 'N/A'}`
            ).join("\n");
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "geological_anomalies.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }
        
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GeologicalAnomalyDashboard;
} else if (typeof window !== 'undefined') {
    window.GeologicalAnomalyDashboard = GeologicalAnomalyDashboard;
}