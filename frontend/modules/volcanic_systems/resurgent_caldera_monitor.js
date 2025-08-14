/**
 * Resurgent Caldera Monitoring System - Frontend Component
 * Advanced 3D visualization and real-time monitoring interface
 * for supereruption-capable resurgent calderas
 */

class ResurgentCalderaMonitor {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.config = {
            apiBaseUrl: config.apiBaseUrl || '/api/caldera-monitoring',
            updateInterval: config.updateInterval || 60000, // 1 minute
            alertThresholds: config.alertThresholds || {
                deformation: 10, // mm/day
                seismic: 5.0,    // energy increase factor
                gas: 3.0,        // emission increase factor
                thermal: 50      // temperature anomaly °C
            },
            visualization: {
                enable3D: true,
                showStressField: true,
                showMagmaChambersIn3D: true,
                showDeformation: true,
                animateEvolution: true
            },
            ...config
        };
        
        // State management
        this.calderaData = null;
        this.monitoringSession = null;
        this.realTimeData = null;
        this.alertLevel = 'normal';
        
        // Visualization components
        this.scene3D = null;
        this.plotlyDiv = null;
        this.leafletMap = null;
        
        // Update timers
        this.updateTimer = null;
        this.animationTimer = null;
        
        // Event handlers
        this.eventHandlers = new Map();
        
        this.init();
    }
    
    async init() {
        try {
            await this.createUI();
            await this.initializeVisualization();
            await this.startMonitoring();
            
            console.log('Resurgent Caldera Monitor initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Resurgent Caldera Monitor:', error);
            this.showError('Initialization failed: ' + error.message);
        }
    }
    
    async createUI() {
        this.container.innerHTML = `
            <div class="resurgent-caldera-monitor">
                <!-- Header and Controls -->
                <div class="monitor-header">
                    <div class="header-left">
                        <h2><i class="fas fa-mountain"></i> Resurgent Caldera Monitor</h2>
                        <div class="caldera-selector">
                            <label>Caldera:</label>
                            <select id="calderaSelect">
                                <option value="">Select Caldera...</option>
                                <option value="yellowstone">Yellowstone</option>
                                <option value="long_valley">Long Valley</option>
                                <option value="valles">Valles</option>
                                <option value="campi_flegrei">Campi Flegrei</option>
                                <option value="taupo">Taupo</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="header-center">
                        <div class="alert-status" id="alertStatus">
                            <div class="alert-indicator normal" id="alertIndicator"></div>
                            <span class="alert-text" id="alertText">Normal</span>
                        </div>
                        <div class="monitoring-status">
                            <span class="status-label">Monitoring:</span>
                            <span class="status-value" id="monitoringStatus">Inactive</span>
                        </div>
                    </div>
                    
                    <div class="header-right">
                        <div class="control-buttons">
                            <button id="startMonitoring" class="btn btn-primary" title="Start Monitoring">
                                <i class="fas fa-play"></i> Start
                            </button>
                            <button id="stopMonitoring" class="btn btn-secondary" title="Stop Monitoring">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                            <button id="exportData" class="btn btn-info" title="Export Data">
                                <i class="fas fa-download"></i> Export
                            </button>
                            <button id="settingsBtn" class="btn btn-outline-secondary" title="Settings">
                                <i class="fas fa-cog"></i>
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Main Dashboard -->
                <div class="monitor-dashboard">
                    <!-- Left Panel - Real-time Data -->
                    <div class="dashboard-panel left-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-tachometer-alt"></i> Real-time Data</h3>
                            <div class="last-update" id="lastUpdate">Updated: Never</div>
                        </div>
                        
                        <div class="data-metrics">
                            <!-- Deformation Metrics -->
                            <div class="metric-group">
                                <h4><i class="fas fa-arrows-alt"></i> Ground Deformation</h4>
                                <div class="metric-cards">
                                    <div class="metric-card">
                                        <div class="metric-label">Max Uplift</div>
                                        <div class="metric-value" id="maxUplift">-- mm</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Deformation Rate</div>
                                        <div class="metric-value" id="deformationRate">-- mm/day</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Volume Change</div>
                                        <div class="metric-value" id="volumeChange">-- m³</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Seismic Metrics -->
                            <div class="metric-group">
                                <h4><i class="fas fa-wave-square"></i> Seismic Activity</h4>
                                <div class="metric-cards">
                                    <div class="metric-card">
                                        <div class="metric-label">Events (24h)</div>
                                        <div class="metric-value" id="seismicEvents">--</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Max Magnitude</div>
                                        <div class="metric-value" id="maxMagnitude">--</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Energy Release</div>
                                        <div class="metric-value" id="energyRelease">-- J</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Gas Emission Metrics -->
                            <div class="metric-group">
                                <h4><i class="fas fa-cloud"></i> Gas Emissions</h4>
                                <div class="metric-cards">
                                    <div class="metric-card">
                                        <div class="metric-label">CO₂ Flux</div>
                                        <div class="metric-value" id="co2Flux">-- t/day</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">SO₂ Flux</div>
                                        <div class="metric-value" id="so2Flux">-- t/day</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">³He/⁴He Ratio</div>
                                        <div class="metric-value" id="heRatio">--</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Thermal Metrics -->
                            <div class="metric-group">
                                <h4><i class="fas fa-thermometer-half"></i> Thermal Activity</h4>
                                <div class="metric-cards">
                                    <div class="metric-card">
                                        <div class="metric-label">Max Temperature</div>
                                        <div class="metric-value" id="maxTemperature">-- °C</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Thermal Anomaly</div>
                                        <div class="metric-value" id="thermalAnomaly">-- °C</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Heat Flux</div>
                                        <div class="metric-value" id="heatFlux">-- MW</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- ML Predictions -->
                        <div class="prediction-panel">
                            <h4><i class="fas fa-brain"></i> ML Predictions</h4>
                            <div class="prediction-cards">
                                <div class="prediction-card">
                                    <div class="prediction-label">Melt Fraction</div>
                                    <div class="prediction-value" id="meltFraction">--%</div>
                                    <div class="prediction-confidence" id="meltConfidence">±--%</div>
                                </div>
                                <div class="prediction-card">
                                    <div class="prediction-label">Roof Stability</div>
                                    <div class="prediction-value" id="roofStability">--</div>
                                    <div class="prediction-confidence" id="stabilityConfidence">±--%</div>
                                </div>
                                <div class="prediction-card">
                                    <div class="prediction-label">Eruption Risk</div>
                                    <div class="prediction-value" id="eruptionRisk">--%</div>
                                    <div class="prediction-confidence" id="riskConfidence">±--%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Center Panel - 3D Visualization -->
                    <div class="dashboard-panel center-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-cube"></i> 3D Caldera System</h3>
                            <div class="viz-controls">
                                <button class="viz-btn" id="showStressField" title="Toggle Stress Field">
                                    <i class="fas fa-compress-arrows-alt"></i>
                                </button>
                                <button class="viz-btn" id="showMagmaChamber" title="Toggle Magma Chamber">
                                    <i class="fas fa-fire"></i>
                                </button>
                                <button class="viz-btn" id="showDeformation" title="Toggle Deformation">
                                    <i class="fas fa-arrows-alt"></i>
                                </button>
                                <button class="viz-btn" id="animateEvolution" title="Animate Evolution">
                                    <i class="fas fa-play-circle"></i>
                                </button>
                                <button class="viz-btn" id="resetView" title="Reset View">
                                    <i class="fas fa-home"></i>
                                </button>
                            </div>
                        </div>
                        
                        <div class="viz-container" id="viz3D">
                            <div class="loading-spinner" id="vizLoading">
                                <i class="fas fa-spinner fa-spin"></i>
                                <p>Loading 3D visualization...</p>
                            </div>
                        </div>
                        
                        <!-- Time Series Panel -->
                        <div class="timeseries-panel">
                            <div class="timeseries-header">
                                <h4>Time Series Analysis</h4>
                                <div class="timeseries-controls">
                                    <select id="timeseriesParameter">
                                        <option value="deformation">Ground Deformation</option>
                                        <option value="seismic">Seismic Energy</option>
                                        <option value="gas">Gas Emissions</option>
                                        <option value="thermal">Thermal Output</option>
                                    </select>
                                    <select id="timeRange">
                                        <option value="24h">Last 24 Hours</option>
                                        <option value="7d">Last 7 Days</option>
                                        <option value="30d">Last 30 Days</option>
                                        <option value="1y">Last Year</option>
                                    </select>
                                </div>
                            </div>
                            <div class="timeseries-viz" id="timeseriesChart"></div>
                        </div>
                    </div>
                    
                    <!-- Right Panel - Analysis & Alerts -->
                    <div class="dashboard-panel right-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-exclamation-triangle"></i> Analysis & Alerts</h3>
                        </div>
                        
                        <!-- Alert Panel -->
                        <div class="alert-panel">
                            <h4>Active Alerts</h4>
                            <div class="alert-list" id="alertList">
                                <div class="no-alerts">No active alerts</div>
                            </div>
                        </div>
                        
                        <!-- Analysis Results -->
                        <div class="analysis-panel">
                            <h4>Latest Analysis</h4>
                            <div class="analysis-content" id="analysisContent">
                                <div class="analysis-item">
                                    <div class="analysis-label">Overall Threat Level:</div>
                                    <div class="analysis-value" id="threatLevel">Normal</div>
                                </div>
                                <div class="analysis-item">
                                    <div class="analysis-label">Confidence:</div>
                                    <div class="analysis-value" id="analysisConfidence">--%</div>
                                </div>
                                <div class="analysis-item">
                                    <div class="analysis-label">Next Analysis:</div>
                                    <div class="analysis-value" id="nextAnalysis">--</div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Key Findings -->
                        <div class="findings-panel">
                            <h4>Key Findings</h4>
                            <div class="findings-list" id="findingsList">
                                <div class="no-findings">No findings available</div>
                            </div>
                        </div>
                        
                        <!-- Recommendations -->
                        <div class="recommendations-panel">
                            <h4>Recommendations</h4>
                            <div class="recommendations-list" id="recommendationsList">
                                <div class="no-recommendations">No recommendations available</div>
                            </div>
                        </div>
                        
                        <!-- Data Sources -->
                        <div class="data-sources-panel">
                            <h4>Data Sources Status</h4>
                            <div class="source-list" id="sourceList">
                                <div class="source-item">
                                    <div class="source-icon"><i class="fas fa-satellite"></i></div>
                                    <div class="source-name">Satellite InSAR</div>
                                    <div class="source-status offline" id="satelliteStatus">Offline</div>
                                </div>
                                <div class="source-item">
                                    <div class="source-icon"><i class="fas fa-wave-square"></i></div>
                                    <div class="source-name">Seismic Network</div>
                                    <div class="source-status offline" id="seismicStatus">Offline</div>
                                </div>
                                <div class="source-item">
                                    <div class="source-icon"><i class="fas fa-map-pin"></i></div>
                                    <div class="source-name">GPS Network</div>
                                    <div class="source-status offline" id="gpsStatus">Offline</div>
                                </div>
                                <div class="source-item">
                                    <div class="source-icon"><i class="fas fa-cloud"></i></div>
                                    <div class="source-name">Gas Monitoring</div>
                                    <div class="source-status offline" id="gasStatus">Offline</div>
                                </div>
                                <div class="source-item">
                                    <div class="source-icon"><i class="fas fa-thermometer-half"></i></div>
                                    <div class="source-name">Thermal Cameras</div>
                                    <div class="source-status offline" id="thermalStatus">Offline</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Settings Modal -->
                <div class="modal" id="settingsModal">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Monitor Settings</h3>
                            <span class="modal-close" id="closeSettings">&times;</span>
                        </div>
                        <div class="modal-body">
                            <div class="settings-section">
                                <h4>Update Frequency</h4>
                                <select id="updateFrequency">
                                    <option value="30000">30 seconds</option>
                                    <option value="60000" selected>1 minute</option>
                                    <option value="300000">5 minutes</option>
                                    <option value="600000">10 minutes</option>
                                </select>
                            </div>
                            
                            <div class="settings-section">
                                <h4>Alert Thresholds</h4>
                                <div class="threshold-inputs">
                                    <label>Deformation (mm/day):</label>
                                    <input type="number" id="deformationThreshold" value="10" step="0.1">
                                    
                                    <label>Seismic Energy Factor:</label>
                                    <input type="number" id="seismicThreshold" value="5" step="0.1">
                                    
                                    <label>Gas Emission Factor:</label>
                                    <input type="number" id="gasThreshold" value="3" step="0.1">
                                    
                                    <label>Thermal Anomaly (°C):</label>
                                    <input type="number" id="thermalThreshold" value="50" step="1">
                                </div>
                            </div>
                            
                            <div class="settings-section">
                                <h4>Visualization Options</h4>
                                <div class="checkbox-group">
                                    <label><input type="checkbox" id="enable3D" checked> Enable 3D Visualization</label>
                                    <label><input type="checkbox" id="showStressFieldOption" checked> Show Stress Field</label>
                                    <label><input type="checkbox" id="showMagmaOption" checked> Show Magma Chambers</label>
                                    <label><input type="checkbox" id="showDeformationOption" checked> Show Deformation</label>
                                    <label><input type="checkbox" id="animateOption" checked> Animate Evolution</label>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-secondary" id="cancelSettings">Cancel</button>
                            <button class="btn btn-primary" id="saveSettings">Save Settings</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add event listeners
        this.attachEventListeners();
    }
    
    attachEventListeners() {
        // Caldera selection
        document.getElementById('calderaSelect').addEventListener('change', (e) => {
            this.selectCaldera(e.target.value);
        });
        
        // Control buttons
        document.getElementById('startMonitoring').addEventListener('click', () => {
            this.startMonitoring();
        });
        
        document.getElementById('stopMonitoring').addEventListener('click', () => {
            this.stopMonitoring();
        });
        
        document.getElementById('exportData').addEventListener('click', () => {
            this.exportData();
        });
        
        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => {
            document.getElementById('settingsModal').style.display = 'block';
        });
        
        document.getElementById('closeSettings').addEventListener('click', () => {
            document.getElementById('settingsModal').style.display = 'none';
        });
        
        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveSettings();
        });
        
        // Visualization controls
        document.getElementById('showStressField').addEventListener('click', () => {
            this.toggleStressField();
        });
        
        document.getElementById('showMagmaChamber').addEventListener('click', () => {
            this.toggleMagmaChamber();
        });
        
        document.getElementById('showDeformation').addEventListener('click', () => {
            this.toggleDeformation();
        });
        
        document.getElementById('animateEvolution').addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        document.getElementById('resetView').addEventListener('click', () => {
            this.resetView();
        });
        
        // Time series controls
        document.getElementById('timeseriesParameter').addEventListener('change', (e) => {
            this.updateTimeSeries(e.target.value);
        });
        
        document.getElementById('timeRange').addEventListener('change', (e) => {
            this.updateTimeRange(e.target.value);
        });
    }
    
    async selectCaldera(calderaId) {
        if (!calderaId) return;
        
        try {
            this.showLoading('Loading caldera data...');
            
            // Fetch caldera information
            const response = await fetch(\`\${this.config.apiBaseUrl}/calderas/\${calderaId}\`);
            if (!response.ok) throw new Error('Failed to fetch caldera data');
            
            this.calderaData = await response.json();
            
            // Update UI with caldera info
            await this.updateCalderaInfo();
            
            // Initialize 3D visualization
            await this.initialize3DVisualization();
            
            this.hideLoading();
            
        } catch (error) {
            console.error('Error selecting caldera:', error);
            this.showError('Failed to load caldera: ' + error.message);
        }
    }
    
    async startMonitoring() {
        if (!this.calderaData) {
            this.showError('Please select a caldera first');
            return;
        }
        
        try {
            // Start monitoring session
            const response = await fetch(\`\${this.config.apiBaseUrl}/monitoring/start\`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    calderaId: this.calderaData.id,
                    monitoringType: 'continuous',
                    dataSources: ['satellite_insar', 'seismic', 'gps', 'gas_emissions', 'thermal']
                })
            });
            
            if (!response.ok) throw new Error('Failed to start monitoring');
            
            const result = await response.json();
            this.monitoringSession = result.sessionId;
            
            // Update UI
            document.getElementById('monitoringStatus').textContent = 'Active';
            document.getElementById('monitoringStatus').className = 'status-value active';
            
            // Start real-time updates
            this.startRealTimeUpdates();
            
            this.showNotification('Monitoring started successfully', 'success');
            
        } catch (error) {
            console.error('Error starting monitoring:', error);
            this.showError('Failed to start monitoring: ' + error.message);
        }
    }
    
    async stopMonitoring() {
        if (!this.monitoringSession) return;
        
        try {
            // Stop monitoring session
            const response = await fetch(\`\${this.config.apiBaseUrl}/monitoring/\${this.monitoringSession}/stop\`, {
                method: 'POST'
            });
            
            if (!response.ok) throw new Error('Failed to stop monitoring');
            
            // Stop real-time updates
            this.stopRealTimeUpdates();
            
            // Update UI
            document.getElementById('monitoringStatus').textContent = 'Inactive';
            document.getElementById('monitoringStatus').className = 'status-value inactive';
            
            this.monitoringSession = null;
            this.showNotification('Monitoring stopped', 'info');
            
        } catch (error) {
            console.error('Error stopping monitoring:', error);
            this.showError('Failed to stop monitoring: ' + error.message);
        }
    }
    
    startRealTimeUpdates() {
        if (this.updateTimer) return;
        
        this.updateTimer = setInterval(async () => {
            await this.fetchRealTimeData();
        }, this.config.updateInterval);
        
        // Initial fetch
        this.fetchRealTimeData();
    }
    
    stopRealTimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }
    
    async fetchRealTimeData() {
        if (!this.monitoringSession) return;
        
        try {
            const response = await fetch(\`\${this.config.apiBaseUrl}/monitoring/\${this.monitoringSession}/status\`);
            if (!response.ok) throw new Error('Failed to fetch real-time data');
            
            const data = await response.json();
            this.realTimeData = data;
            
            // Update UI
            this.updateMetrics(data);
            this.updateAlerts(data);
            this.updateAnalysis(data);
            this.updateDataSources(data);
            this.update3DVisualization(data);
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = 
                \`Updated: \${new Date().toLocaleTimeString()}\`;
                
        } catch (error) {
            console.error('Error fetching real-time data:', error);
            // Don't show error for every failed fetch, just log it
        }
    }
    
    updateMetrics(data) {
        const analysis = data.lastAnalysis;
        if (!analysis) return;
        
        // Deformation metrics
        const deformation = analysis.analysisComponents.deformation;
        if (deformation) {
            document.getElementById('maxUplift').textContent = \`\${deformation.maxUpliftMm?.toFixed(1) || 0} mm\`;
            document.getElementById('deformationRate').textContent = \`\${deformation.deformationRateMmDay?.toFixed(2) || 0} mm/day\`;
            document.getElementById('volumeChange').textContent = \`\${(deformation.volumeChangeM3 || 0).toLocaleString()} m³\`;
        }
        
        // Seismic metrics
        const seismic = analysis.analysisComponents.seismic;
        if (seismic) {
            document.getElementById('seismicEvents').textContent = seismic.totalEvents || 0;
            document.getElementById('maxMagnitude').textContent = seismic.maxMagnitude?.toFixed(1) || 0;
            document.getElementById('energyRelease').textContent = \`\${(seismic.energyRelease || 0).toExponential(2)} J\`;
        }
        
        // Gas metrics
        const gas = analysis.analysisComponents.gas;
        if (gas) {
            document.getElementById('co2Flux').textContent = \`\${gas.co2FluxAvg?.toFixed(0) || 0} t/day\`;
            document.getElementById('so2Flux').textContent = \`\${gas.so2FluxAvg?.toFixed(1) || 0} t/day\`;
            document.getElementById('heRatio').textContent = gas.he3He4RatioAvg?.toFixed(2) || 0;
        }
        
        // Thermal metrics
        const thermal = analysis.analysisComponents.thermal;
        if (thermal) {
            document.getElementById('maxTemperature').textContent = \`\${thermal.maxTemperatureC?.toFixed(1) || 0} °C\`;
            document.getElementById('thermalAnomaly').textContent = \`\${thermal.temperatureAnomalyC?.toFixed(1) || 0} °C\`;
            document.getElementById('heatFlux').textContent = \`\${thermal.heatFluxMw?.toFixed(0) || 0} MW\`;
        }
    }
    
    updateAlerts(data) {
        const alertIndicator = document.getElementById('alertIndicator');
        const alertText = document.getElementById('alertText');
        const alertList = document.getElementById('alertList');
        
        if (data.alertLevel) {
            this.alertLevel = data.alertLevel;
            
            // Update alert status indicator
            alertIndicator.className = \`alert-indicator \${data.alertLevel}\`;
            alertText.textContent = data.alertLevel.charAt(0).toUpperCase() + data.alertLevel.slice(1);
            
            // Update alert list
            if (data.alertsGenerated && data.alertsGenerated.length > 0) {
                alertList.innerHTML = data.alertsGenerated.map(alert => \`
                    <div class="alert-item \${alert.severity}">
                        <div class="alert-header">
                            <i class="fas fa-exclamation-triangle"></i>
                            <span class="alert-type">\${alert.alertType}</span>
                            <span class="alert-time">\${new Date(alert.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <div class="alert-message">\${alert.message}</div>
                    </div>
                \`).join('');
            } else {
                alertList.innerHTML = '<div class="no-alerts">No active alerts</div>';
            }
        }
    }
    
    updateAnalysis(data) {
        const analysis = data.lastAnalysis;
        if (!analysis) return;
        
        // Update threat level
        document.getElementById('threatLevel').textContent = 
            analysis.overallThreatLevel.charAt(0).toUpperCase() + analysis.overallThreatLevel.slice(1);
        document.getElementById('threatLevel').className = \`analysis-value \${analysis.overallThreatLevel}\`;
        
        // Update confidence
        document.getElementById('analysisConfidence').textContent = 
            \`\${(analysis.confidenceScore * 100).toFixed(1)}%\`;
        
        // Update next analysis time
        document.getElementById('nextAnalysis').textContent = 
            new Date(analysis.nextScheduledAnalysis).toLocaleString();
        
        // Update findings
        const findingsList = document.getElementById('findingsList');
        if (analysis.keyFindings && analysis.keyFindings.length > 0) {
            findingsList.innerHTML = analysis.keyFindings.map(finding => \`
                <div class="finding-item">
                    <i class="fas fa-info-circle"></i>
                    <span>\${finding}</span>
                </div>
            \`).join('');
        } else {
            findingsList.innerHTML = '<div class="no-findings">No findings available</div>';
        }
        
        // Update recommendations
        const recommendationsList = document.getElementById('recommendationsList');
        if (analysis.recommendations && analysis.recommendations.length > 0) {
            recommendationsList.innerHTML = analysis.recommendations.map(rec => \`
                <div class="recommendation-item">
                    <i class="fas fa-lightbulb"></i>
                    <span>\${rec}</span>
                </div>
            \`).join('');
        } else {
            recommendationsList.innerHTML = '<div class="no-recommendations">No recommendations available</div>';
        }
    }
    
    updateDataSources(data) {
        const dataInventory = data.dataInventory;
        if (!dataInventory) return;
        
        const sources = ['satellite', 'seismic', 'gps', 'gas', 'thermal'];
        
        sources.forEach(source => {
            const statusElement = document.getElementById(\`\${source}Status\`);
            const hasData = dataInventory.byType && dataInventory.byType[source] > 0;
            
            if (hasData) {
                statusElement.textContent = 'Online';
                statusElement.className = 'source-status online';
            } else {
                statusElement.textContent = 'Offline';
                statusElement.className = 'source-status offline';
            }
        });
    }
    
    async initialize3DVisualization() {
        if (!this.calderaData) return;
        
        const vizContainer = document.getElementById('viz3D');
        
        // Initialize Plotly 3D visualization
        const traces = await this.create3DTraces();
        
        const layout = {
            scene: {
                xaxis: { title: 'Longitude' },
                yaxis: { title: 'Latitude' },
                zaxis: { title: 'Elevation (km)' },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true
        };
        
        const config = {
            displayModeBar: true,
            responsive: true
        };
        
        await Plotly.newPlot('viz3D', traces, layout, config);
        
        // Hide loading spinner
        document.getElementById('vizLoading').style.display = 'none';
    }
    
    async create3DTraces() {
        const traces = [];
        
        if (!this.calderaData) return traces;
        
        // Create caldera rim
        const rim = this.createCalderaRim();
        traces.push(rim);
        
        // Create topography surface
        const topo = await this.createTopographySurface();
        if (topo) traces.push(topo);
        
        // Add magma chamber if available
        if (this.config.visualization.showMagmaChambersIn3D) {
            const chamber = this.createMagmaChamber();
            if (chamber) traces.push(chamber);
        }
        
        // Add stress field if available
        if (this.config.visualization.showStressField && this.realTimeData) {
            const stressField = this.createStressFieldVisualization();
            if (stressField) traces.push(stressField);
        }
        
        return traces;
    }
    
    createCalderaRim() {
        // Create circular rim based on caldera dimensions
        const center = this.calderaData.location || [0, 0];
        const radius = (this.calderaData.diameterKm || 10) / 2;
        
        const theta = [];
        const x = [];
        const y = [];
        const z = [];
        
        for (let i = 0; i <= 100; i++) {
            const angle = (i / 100) * 2 * Math.PI;
            theta.push(angle);
            x.push(center[0] + radius * Math.cos(angle) / 111); // Convert to degrees
            y.push(center[1] + radius * Math.sin(angle) / 111);
            z.push(this.calderaData.rimHeightM / 1000 || 2); // Convert to km
        }
        
        return {
            type: 'scatter3d',
            mode: 'lines',
            x: x,
            y: y,
            z: z,
            line: {
                color: 'red',
                width: 5
            },
            name: 'Caldera Rim'
        };
    }
    
    async createTopographySurface() {
        // In a real implementation, would fetch DEM data
        // For now, create synthetic topography
        
        if (!this.calderaData.location) return null;
        
        const center = this.calderaData.location;
        const size = (this.calderaData.diameterKm || 10) * 1.5;
        
        const x = [];
        const y = [];
        const z = [];
        
        const resolution = 20;
        for (let i = 0; i < resolution; i++) {
            const row_x = [];
            const row_y = [];
            const row_z = [];
            
            for (let j = 0; j < resolution; j++) {
                const lon = center[0] + (i - resolution/2) * size / 111 / resolution;
                const lat = center[1] + (j - resolution/2) * size / 111 / resolution;
                
                // Synthetic topography with caldera depression
                const dist = Math.sqrt((i - resolution/2)**2 + (j - resolution/2)**2);
                const elevation = 2 - 0.5 * Math.exp(-dist/3); // km
                
                row_x.push(lon);
                row_y.push(lat);
                row_z.push(elevation);
            }
            
            x.push(row_x);
            y.push(row_y);
            z.push(row_z);
        }
        
        return {
            type: 'surface',
            x: x,
            y: y,
            z: z,
            colorscale: 'Earth',
            opacity: 0.7,
            name: 'Topography'
        };
    }
    
    createMagmaChamber() {
        if (!this.calderaData.magmaChamberCharacteristics) return null;
        
        const center = this.calderaData.location || [0, 0];
        const depth = this.calderaData.magmaChamberCharacteristics.depthKm || 8;
        const volume = this.calderaData.magmaChamberCharacteristics.volumeKm3 || 1000;
        
        // Approximate as ellipsoid
        const radius = Math.pow(volume * 3 / (4 * Math.PI), 1/3);
        
        // Create sphere points
        const x = [];
        const y = [];
        const z = [];
        
        const phi_points = 20;
        const theta_points = 20;
        
        for (let i = 0; i < phi_points; i++) {
            for (let j = 0; j < theta_points; j++) {
                const phi = (i / phi_points) * Math.PI;
                const theta = (j / theta_points) * 2 * Math.PI;
                
                x.push(center[0] + radius * Math.sin(phi) * Math.cos(theta) / 111);
                y.push(center[1] + radius * Math.sin(phi) * Math.sin(theta) / 111);
                z.push(-depth + radius * Math.cos(phi));
            }
        }
        
        return {
            type: 'scatter3d',
            mode: 'markers',
            x: x,
            y: y,
            z: z,
            marker: {
                size: 2,
                color: 'orange',
                opacity: 0.6
            },
            name: 'Magma Chamber'
        };
    }
    
    update3DVisualization(data) {
        if (!this.plotlyDiv) return;
        
        // Update visualization with new data
        // This would include real-time deformation, stress fields, etc.
    }
    
    async exportData() {
        if (!this.realTimeData) {
            this.showError('No data available to export');
            return;
        }
        
        try {
            const exportData = {
                caldera: this.calderaData,
                monitoringSession: this.monitoringSession,
                realTimeData: this.realTimeData,
                exportTime: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = \`caldera_monitoring_\${this.calderaData?.id || 'data'}_\${new Date().toISOString().split('T')[0]}.json\`;
            a.click();
            
            URL.revokeObjectURL(url);
            this.showNotification('Data exported successfully', 'success');
            
        } catch (error) {
            console.error('Error exporting data:', error);
            this.showError('Failed to export data: ' + error.message);
        }
    }
    
    showLoading(message = 'Loading...') {
        // Implementation for loading indicator
        console.log('Loading:', message);
    }
    
    hideLoading() {
        // Implementation for hiding loading indicator
        console.log('Loading complete');
    }
    
    showError(message) {
        // Implementation for error display
        console.error('Error:', message);
        alert('Error: ' + message);
    }
    
    showNotification(message, type = 'info') {
        // Implementation for notification display
        console.log(\`\${type.toUpperCase()}: \${message}\`);
    }
    
    // Visualization control methods
    toggleStressField() {
        this.config.visualization.showStressField = !this.config.visualization.showStressField;
        this.update3DVisualization();
    }
    
    toggleMagmaChamber() {
        this.config.visualization.showMagmaChambersIn3D = !this.config.visualization.showMagmaChambersIn3D;
        this.update3DVisualization();
    }
    
    toggleDeformation() {
        this.config.visualization.showDeformation = !this.config.visualization.showDeformation;
        this.update3DVisualization();
    }
    
    toggleAnimation() {
        this.config.visualization.animateEvolution = !this.config.visualization.animateEvolution;
        // Implementation for animation control
    }
    
    resetView() {
        if (this.plotlyDiv) {
            Plotly.relayout('viz3D', {
                'scene.camera': {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            });
        }
    }
    
    updateTimeSeries(parameter) {
        // Implementation for time series updates
        console.log('Updating time series for parameter:', parameter);
    }
    
    updateTimeRange(range) {
        // Implementation for time range updates
        console.log('Updating time range:', range);
    }
    
    saveSettings() {
        // Get settings from modal
        const updateFrequency = document.getElementById('updateFrequency').value;
        const deformationThreshold = document.getElementById('deformationThreshold').value;
        const seismicThreshold = document.getElementById('seismicThreshold').value;
        const gasThreshold = document.getElementById('gasThreshold').value;
        const thermalThreshold = document.getElementById('thermalThreshold').value;
        
        // Update config
        this.config.updateInterval = parseInt(updateFrequency);
        this.config.alertThresholds = {
            deformation: parseFloat(deformationThreshold),
            seismic: parseFloat(seismicThreshold),
            gas: parseFloat(gasThreshold),
            thermal: parseFloat(thermalThreshold)
        };
        
        // Update visualization options
        this.config.visualization.enable3D = document.getElementById('enable3D').checked;
        this.config.visualization.showStressField = document.getElementById('showStressFieldOption').checked;
        this.config.visualization.showMagmaChambersIn3D = document.getElementById('showMagmaOption').checked;
        this.config.visualization.showDeformation = document.getElementById('showDeformationOption').checked;
        this.config.visualization.animateEvolution = document.getElementById('animateOption').checked;
        
        // Restart updates with new frequency
        if (this.updateTimer) {
            this.stopRealTimeUpdates();
            this.startRealTimeUpdates();
        }
        
        // Close modal
        document.getElementById('settingsModal').style.display = 'none';
        
        this.showNotification('Settings saved successfully', 'success');
    }
    
    destroy() {
        // Clean up resources
        this.stopRealTimeUpdates();
        
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
        }
        
        // Remove event listeners
        this.eventHandlers.clear();
        
        // Clean up Plotly
        if (this.plotlyDiv) {
            Plotly.purge('viz3D');
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResurgentCalderaMonitor;
}

// Global registration for direct HTML usage
if (typeof window !== 'undefined') {
    window.ResurgentCalderaMonitor = ResurgentCalderaMonitor;
}