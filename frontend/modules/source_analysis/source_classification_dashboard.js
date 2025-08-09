/**
 * Source Classification Dashboard
 * Interactive visualization interface for ML-guided source characterization
 * Advanced real-time parameter optimization and model comparison
 */

class SourceClassificationDashboard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            width: 1400,
            height: 900,
            backgroundColor: '#1a1a1a',
            primaryColor: '#00ff88',
            secondaryColor: '#ff6b35',
            accentColor: '#4ecdc4',
            warningColor: '#ffd23f',
            errorColor: '#ff4757',
            textColor: '#ffffff',
            gridColor: '#333333',
            api_base: '/api/source-characterization',
            update_interval: 2000,
            max_history_points: 1000,
            visualization_quality: 'high',
            enable_3d: true,
            enable_realtime: true,
            ...config
        };

        // Core components
        this.container = null;
        this.scenes = {};
        this.renderers = {};
        this.cameras = {};
        this.controls = {};
        
        // Data management
        this.currentData = null;
        this.classificationHistory = [];
        this.optimizationHistory = [];
        this.modelPerformance = {};
        this.uncertaintyData = {};
        
        // UI components
        this.panels = {};
        this.charts = {};
        this.controls_ui = {};
        this.activeTab = 'overview';
        
        // Analysis state
        this.currentModel = null;
        this.isOptimizing = false;
        this.selectedParameters = new Set();
        this.parameterBounds = {};
        
        // WebGL and visualization
        this.materials = {};
        this.geometries = {};
        this.meshes = {};
        
        // Real-time updates
        this.updateTimer = null;
        this.websocket = null;
        
        this.initialize();
    }

    async initialize() {
        console.log('Initializing Source Classification Dashboard...');
        
        try {
            await this.createInterface();
            await this.setupWebGL();
            await this.initializeCharts();
            await this.loadInitialData();
            await this.setupEventHandlers();
            await this.startRealTimeUpdates();
            
            console.log('Source Classification Dashboard initialized successfully');
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Initialization failed: ' + error.message);
        }
    }

    async createInterface() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            throw new Error(`Container ${this.containerId} not found`);
        }

        this.container.innerHTML = `
            <div class="source-dashboard-container">
                <!-- Header -->
                <div class="dashboard-header">
                    <div class="header-left">
                        <h1 class="dashboard-title">
                            <i class="fas fa-brain"></i>
                            ML-Guided Source Characterization
                        </h1>
                        <div class="status-indicators">
                            <div class="status-indicator" id="connection-status">
                                <i class="fas fa-circle"></i>
                                <span>Connecting...</span>
                            </div>
                            <div class="status-indicator" id="optimization-status">
                                <i class="fas fa-cog"></i>
                                <span>Ready</span>
                            </div>
                        </div>
                    </div>
                    <div class="header-right">
                        <div class="control-group">
                            <button class="btn btn-primary" id="start-optimization">
                                <i class="fas fa-play"></i>
                                Start Optimization
                            </button>
                            <button class="btn btn-secondary" id="export-results">
                                <i class="fas fa-download"></i>
                                Export Results
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Navigation Tabs -->
                <div class="dashboard-tabs">
                    <div class="tab-list">
                        <button class="tab-button active" data-tab="overview">
                            <i class="fas fa-chart-line"></i>
                            Overview
                        </button>
                        <button class="tab-button" data-tab="classification">
                            <i class="fas fa-tags"></i>
                            Classification
                        </button>
                        <button class="tab-button" data-tab="optimization">
                            <i class="fas fa-sliders-h"></i>
                            Optimization
                        </button>
                        <button class="tab-button" data-tab="models">
                            <i class="fas fa-project-diagram"></i>
                            Model Comparison
                        </button>
                        <button class="tab-button" data-tab="uncertainty">
                            <i class="fas fa-question-circle"></i>
                            Uncertainty
                        </button>
                        <button class="tab-button" data-tab="visualization">
                            <i class="fas fa-cube"></i>
                            3D Visualization
                        </button>
                    </div>
                </div>

                <!-- Main Content Area -->
                <div class="dashboard-content">
                    <!-- Overview Tab -->
                    <div class="tab-content active" id="overview-tab">
                        <div class="overview-grid">
                            <div class="overview-card">
                                <h3>Classification Results</h3>
                                <div class="classification-summary" id="classification-summary">
                                    <div class="predicted-type">
                                        <span class="label">Predicted Type:</span>
                                        <span class="value" id="predicted-type">Loading...</span>
                                        <span class="confidence" id="prediction-confidence">0%</span>
                                    </div>
                                    <div class="confidence-chart" id="confidence-chart"></div>
                                </div>
                            </div>

                            <div class="overview-card">
                                <h3>Optimization Progress</h3>
                                <div class="optimization-progress" id="optimization-progress">
                                    <div class="progress-metrics">
                                        <div class="metric">
                                            <span class="label">Best Objective:</span>
                                            <span class="value" id="best-objective">--</span>
                                        </div>
                                        <div class="metric">
                                            <span class="label">Trials:</span>
                                            <span class="value" id="trial-count">0</span>
                                        </div>
                                        <div class="metric">
                                            <span class="label">Convergence:</span>
                                            <span class="value" id="convergence-status">--</span>
                                        </div>
                                    </div>
                                    <div class="convergence-chart" id="convergence-chart"></div>
                                </div>
                            </div>

                            <div class="overview-card full-width">
                                <h3>Real-time Parameter Evolution</h3>
                                <div class="parameter-evolution" id="parameter-evolution"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Classification Tab -->
                    <div class="tab-content" id="classification-tab">
                        <div class="classification-grid">
                            <div class="classification-panel">
                                <h3>Source Type Probabilities</h3>
                                <div class="probability-chart" id="probability-chart"></div>
                            </div>
                            <div class="classification-panel">
                                <h3>Feature Importance</h3>
                                <div class="feature-importance" id="feature-importance"></div>
                            </div>
                            <div class="classification-panel">
                                <h3>Classification History</h3>
                                <div class="classification-timeline" id="classification-timeline"></div>
                            </div>
                            <div class="classification-panel">
                                <h3>Confusion Matrix</h3>
                                <div class="confusion-matrix" id="confusion-matrix"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Optimization Tab -->
                    <div class="tab-content" id="optimization-tab">
                        <div class="optimization-layout">
                            <div class="optimization-controls">
                                <h3>Parameter Configuration</h3>
                                <div class="parameter-controls" id="parameter-controls">
                                    <!-- Dynamic parameter controls will be inserted here -->
                                </div>
                                <div class="optimization-settings">
                                    <h4>Optimization Settings</h4>
                                    <div class="setting-group">
                                        <label for="algorithm-select">Algorithm:</label>
                                        <select id="algorithm-select">
                                            <option value="gaussian_process">Gaussian Process</option>
                                            <option value="differential_evolution">Differential Evolution</option>
                                            <option value="optuna_tpe">Optuna TPE</option>
                                            <option value="random_search">Random Search</option>
                                        </select>
                                    </div>
                                    <div class="setting-group">
                                        <label for="max-trials">Max Trials:</label>
                                        <input type="number" id="max-trials" value="100" min="10" max="1000">
                                    </div>
                                    <div class="setting-group">
                                        <label for="parallel-workers">Parallel Workers:</label>
                                        <input type="number" id="parallel-workers" value="4" min="1" max="16">
                                    </div>
                                </div>
                            </div>
                            <div class="optimization-visualization">
                                <h3>Parameter Space Exploration</h3>
                                <div class="parameter-space" id="parameter-space-3d"></div>
                                <div class="acquisition-function" id="acquisition-function"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Model Comparison Tab -->
                    <div class="tab-content" id="models-tab">
                        <div class="models-layout">
                            <div class="model-list">
                                <h3>Available Models</h3>
                                <div class="model-cards" id="model-cards">
                                    <!-- Model cards will be dynamically generated -->
                                </div>
                            </div>
                            <div class="model-comparison">
                                <h3>Performance Comparison</h3>
                                <div class="comparison-chart" id="model-comparison-chart"></div>
                                <div class="model-metrics" id="model-metrics-table"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Uncertainty Tab -->
                    <div class="tab-content" id="uncertainty-tab">
                        <div class="uncertainty-layout">
                            <div class="uncertainty-analysis">
                                <h3>Parameter Uncertainty</h3>
                                <div class="uncertainty-heatmap" id="uncertainty-heatmap"></div>
                            </div>
                            <div class="sensitivity-analysis">
                                <h3>Sensitivity Analysis</h3>
                                <div class="sensitivity-chart" id="sensitivity-chart"></div>
                            </div>
                            <div class="bootstrap-results">
                                <h3>Bootstrap Confidence Intervals</h3>
                                <div class="bootstrap-chart" id="bootstrap-chart"></div>
                            </div>
                        </div>
                    </div>

                    <!-- 3D Visualization Tab -->
                    <div class="tab-content" id="visualization-tab">
                        <div class="visualization-layout">
                            <div class="visualization-controls">
                                <h3>Visualization Controls</h3>
                                <div class="control-group">
                                    <label>View Mode:</label>
                                    <select id="view-mode">
                                        <option value="source_geometry">Source Geometry</option>
                                        <option value="deformation_field">Deformation Field</option>
                                        <option value="uncertainty_volume">Uncertainty Volume</option>
                                        <option value="parameter_space">Parameter Space</option>
                                    </select>
                                </div>
                                <div class="control-group">
                                    <label>Color Scheme:</label>
                                    <select id="color-scheme">
                                        <option value="viridis">Viridis</option>
                                        <option value="plasma">Plasma</option>
                                        <option value="coolwarm">Cool-Warm</option>
                                        <option value="seismic">Seismic</option>
                                    </select>
                                </div>
                                <div class="control-group">
                                    <label>Animation Speed:</label>
                                    <input type="range" id="animation-speed" min="0.1" max="2.0" step="0.1" value="1.0">
                                    <span id="speed-value">1.0x</span>
                                </div>
                            </div>
                            <div class="visualization-3d">
                                <div class="webgl-container" id="webgl-container"></div>
                                <div class="visualization-info" id="visualization-info">
                                    <div class="info-panel">
                                        <h4>Current Model</h4>
                                        <div id="model-info"></div>
                                    </div>
                                    <div class="legend-panel">
                                        <h4>Legend</h4>
                                        <div id="color-legend"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Loading Overlay -->
                <div class="loading-overlay" id="loading-overlay">
                    <div class="loading-spinner"></div>
                    <div class="loading-message">Processing...</div>
                </div>

                <!-- Error Modal -->
                <div class="modal" id="error-modal">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Error</h3>
                            <button class="modal-close">&times;</button>
                        </div>
                        <div class="modal-body" id="error-message"></div>
                    </div>
                </div>
            </div>
        `;

        // Apply styles
        this.applyStyles();
    }

    applyStyles() {
        const styles = `
            <style>
            .source-dashboard-container {
                width: 100%;
                height: 100vh;
                background: ${this.config.backgroundColor};
                color: ${this.config.textColor};
                font-family: 'Arial', sans-serif;
                overflow: hidden;
            }

            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                border-bottom: 2px solid ${this.config.primaryColor};
            }

            .dashboard-title {
                margin: 0;
                font-size: 24px;
                font-weight: bold;
                color: ${this.config.primaryColor};
            }

            .dashboard-title i {
                margin-right: 10px;
                animation: pulse 2s infinite;
            }

            .status-indicators {
                display: flex;
                gap: 20px;
                margin-top: 10px;
            }

            .status-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }

            .status-indicator i {
                font-size: 12px;
            }

            .status-indicator.connected i {
                color: ${this.config.primaryColor};
            }

            .status-indicator.optimizing i {
                color: ${this.config.warningColor};
                animation: spin 1s linear infinite;
            }

            .control-group {
                display: flex;
                gap: 10px;
            }

            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .btn-primary {
                background: ${this.config.primaryColor};
                color: ${this.config.backgroundColor};
            }

            .btn-secondary {
                background: ${this.config.secondaryColor};
                color: white;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }

            .dashboard-tabs {
                background: #2c3e50;
                padding: 0 20px;
                border-bottom: 1px solid ${this.config.gridColor};
            }

            .tab-list {
                display: flex;
                gap: 0;
            }

            .tab-button {
                padding: 15px 25px;
                background: none;
                border: none;
                color: #95a5a6;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
                font-weight: 500;
            }

            .tab-button:hover {
                color: ${this.config.textColor};
                background: rgba(255,255,255,0.1);
            }

            .tab-button.active {
                color: ${this.config.primaryColor};
                border-bottom-color: ${this.config.primaryColor};
            }

            .dashboard-content {
                height: calc(100vh - 140px);
                overflow: hidden;
            }

            .tab-content {
                display: none;
                height: 100%;
                padding: 20px;
                overflow-y: auto;
            }

            .tab-content.active {
                display: block;
            }

            /* Overview Tab Styles */
            .overview-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: auto 1fr;
                gap: 20px;
                height: 100%;
            }

            .overview-card {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
            }

            .overview-card.full-width {
                grid-column: 1 / -1;
            }

            .overview-card h3 {
                margin: 0 0 15px 0;
                color: ${this.config.primaryColor};
                font-size: 18px;
            }

            .classification-summary {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }

            .predicted-type {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 15px;
                background: rgba(0,255,136,0.1);
                border-radius: 8px;
                border-left: 4px solid ${this.config.primaryColor};
            }

            .predicted-type .label {
                font-weight: bold;
                color: #bdc3c7;
            }

            .predicted-type .value {
                font-size: 18px;
                font-weight: bold;
                color: ${this.config.primaryColor};
            }

            .predicted-type .confidence {
                margin-left: auto;
                padding: 5px 10px;
                background: ${this.config.primaryColor};
                color: ${this.config.backgroundColor};
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
            }

            .progress-metrics {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }

            .metric {
                text-align: center;
                padding: 15px;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }

            .metric .label {
                display: block;
                font-size: 12px;
                color: #95a5a6;
                margin-bottom: 5px;
            }

            .metric .value {
                font-size: 20px;
                font-weight: bold;
                color: ${this.config.accentColor};
            }

            /* Classification Tab Styles */
            .classification-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 20px;
                height: 100%;
            }

            .classification-panel {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
            }

            .classification-panel h3 {
                margin: 0 0 15px 0;
                color: ${this.config.primaryColor};
                font-size: 16px;
            }

            /* Optimization Tab Styles */
            .optimization-layout {
                display: grid;
                grid-template-columns: 350px 1fr;
                gap: 20px;
                height: 100%;
            }

            .optimization-controls {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
                overflow-y: auto;
            }

            .optimization-controls h3, .optimization-controls h4 {
                margin: 0 0 15px 0;
                color: ${this.config.primaryColor};
            }

            .parameter-controls {
                margin-bottom: 30px;
            }

            .parameter-control {
                margin-bottom: 20px;
                padding: 15px;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                border-left: 3px solid ${this.config.accentColor};
            }

            .parameter-control label {
                display: block;
                font-weight: bold;
                margin-bottom: 8px;
                color: #ecf0f1;
            }

            .parameter-control input, .parameter-control select {
                width: 100%;
                padding: 8px;
                background: #2c3e50;
                border: 1px solid ${this.config.gridColor};
                border-radius: 4px;
                color: ${this.config.textColor};
            }

            .setting-group {
                margin-bottom: 15px;
            }

            .setting-group label {
                display: block;
                font-size: 14px;
                margin-bottom: 5px;
                color: #bdc3c7;
            }

            .optimization-visualization {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
            }

            /* Models Tab Styles */
            .models-layout {
                display: grid;
                grid-template-columns: 350px 1fr;
                gap: 20px;
                height: 100%;
            }

            .model-list {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
                overflow-y: auto;
            }

            .model-card {
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border: 1px solid transparent;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .model-card:hover {
                border-color: ${this.config.accentColor};
                transform: translateX(5px);
            }

            .model-card.active {
                border-color: ${this.config.primaryColor};
                background: rgba(0,255,136,0.1);
            }

            .model-card h4 {
                margin: 0 0 8px 0;
                color: ${this.config.primaryColor};
            }

            .model-card .performance {
                font-size: 12px;
                color: #95a5a6;
            }

            .model-comparison {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
            }

            /* Uncertainty Tab Styles */
            .uncertainty-layout {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 20px;
                height: 100%;
            }

            .uncertainty-layout > div {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
            }

            /* Visualization Tab Styles */
            .visualization-layout {
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 20px;
                height: 100%;
            }

            .visualization-controls {
                background: #34495e;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid ${this.config.gridColor};
            }

            .visualization-3d {
                position: relative;
                background: #2c3e50;
                border-radius: 10px;
                border: 1px solid ${this.config.gridColor};
                overflow: hidden;
            }

            .webgl-container {
                width: 100%;
                height: 100%;
            }

            .visualization-info {
                position: absolute;
                top: 20px;
                right: 20px;
                width: 250px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }

            .info-panel, .legend-panel {
                background: rgba(52,73,94,0.9);
                border-radius: 8px;
                padding: 15px;
                border: 1px solid ${this.config.gridColor};
                backdrop-filter: blur(10px);
            }

            .info-panel h4, .legend-panel h4 {
                margin: 0 0 10px 0;
                color: ${this.config.primaryColor};
                font-size: 14px;
            }

            /* Loading and Modal Styles */
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                display: none;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                flex-direction: column;
                gap: 20px;
            }

            .loading-overlay.active {
                display: flex;
            }

            .loading-spinner {
                width: 50px;
                height: 50px;
                border: 4px solid ${this.config.gridColor};
                border-top: 4px solid ${this.config.primaryColor};
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            .loading-message {
                font-size: 18px;
                color: ${this.config.textColor};
            }

            .modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                display: none;
                justify-content: center;
                align-items: center;
                z-index: 1001;
            }

            .modal.active {
                display: flex;
            }

            .modal-content {
                background: #34495e;
                border-radius: 10px;
                border: 2px solid ${this.config.errorColor};
                max-width: 500px;
                width: 90%;
            }

            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                border-bottom: 1px solid ${this.config.gridColor};
            }

            .modal-header h3 {
                margin: 0;
                color: ${this.config.errorColor};
            }

            .modal-close {
                background: none;
                border: none;
                color: ${this.config.textColor};
                font-size: 24px;
                cursor: pointer;
            }

            .modal-body {
                padding: 20px;
                color: ${this.config.textColor};
            }

            /* Animations */
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }

            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            /* Chart containers */
            .chart-container {
                width: 100%;
                height: 300px;
                background: rgba(255,255,255,0.02);
                border-radius: 8px;
                border: 1px solid ${this.config.gridColor};
            }

            /* Responsive design */
            @media (max-width: 1200px) {
                .overview-grid {
                    grid-template-columns: 1fr;
                }
                
                .classification-grid {
                    grid-template-columns: 1fr;
                }
                
                .optimization-layout,
                .models-layout,
                .visualization-layout {
                    grid-template-columns: 1fr;
                }
                
                .visualization-info {
                    position: relative;
                    width: 100%;
                    top: 0;
                    right: 0;
                    flex-direction: row;
                }
            }
            </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
    }

    async setupWebGL() {
        if (!this.config.enable_3d) return;

        const container = document.getElementById('webgl-container');
        if (!container) return;

        // Initialize Three.js scene
        this.scenes.main = new THREE.Scene();
        this.scenes.main.background = new THREE.Color(0x2c3e50);

        // Camera
        this.cameras.main = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            100000
        );
        this.cameras.main.position.set(10000, 5000, 10000);

        // Renderer
        this.renderers.main = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true 
        });
        this.renderers.main.setSize(container.clientWidth, container.clientHeight);
        this.renderers.main.shadowMap.enabled = true;
        this.renderers.main.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderers.main.domElement);

        // Controls
        this.controls.main = new THREE.OrbitControls(
            this.cameras.main, 
            this.renderers.main.domElement
        );
        this.controls.main.enableDamping = true;
        this.controls.main.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scenes.main.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(20000, 20000, 10000);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scenes.main.add(directionalLight);

        // Initialize materials and geometries
        this.initializeMaterials();
        this.initializeGeometries();

        // Start render loop
        this.startRenderLoop();

        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    initializeMaterials() {
        // Source materials
        this.materials.fault = new THREE.MeshPhongMaterial({
            color: this.config.primaryColor,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });

        this.materials.dike = new THREE.MeshPhongMaterial({
            color: this.config.secondaryColor,
            transparent: true,
            opacity: 0.7
        });

        this.materials.point_source = new THREE.MeshPhongMaterial({
            color: this.config.accentColor,
            emissive: new THREE.Color(this.config.accentColor).multiplyScalar(0.2)
        });

        // Deformation field material
        this.materials.deformation = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                colorScale: { value: new THREE.Vector3(1, 0, 0) }
            },
            vertexShader: `
                attribute float displacement;
                varying vec3 vPosition;
                varying float vDisplacement;
                
                void main() {
                    vPosition = position;
                    vDisplacement = displacement;
                    
                    vec3 newPosition = position + normal * displacement * 100.0;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
                }
            `,
            fragmentShader: `
                varying vec3 vPosition;
                varying float vDisplacement;
                uniform float time;
                uniform vec3 colorScale;
                
                vec3 getColor(float value) {
                    vec3 color1 = vec3(0.0, 0.0, 1.0); // Blue
                    vec3 color2 = vec3(0.0, 1.0, 0.0); // Green
                    vec3 color3 = vec3(1.0, 0.0, 0.0); // Red
                    
                    float t = clamp(value * 0.5 + 0.5, 0.0, 1.0);
                    
                    if (t < 0.5) {
                        return mix(color1, color2, t * 2.0);
                    } else {
                        return mix(color2, color3, (t - 0.5) * 2.0);
                    }
                }
                
                void main() {
                    vec3 color = getColor(vDisplacement);
                    gl_FragColor = vec4(color, 0.8);
                }
            `,
            transparent: true
        });

        // Uncertainty volume material
        this.materials.uncertainty = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                opacity: { value: 0.3 }
            },
            vertexShader: `
                attribute float uncertainty;
                varying float vUncertainty;
                
                void main() {
                    vUncertainty = uncertainty;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                varying float vUncertainty;
                uniform float time;
                uniform float opacity;
                
                void main() {
                    float alpha = opacity * vUncertainty;
                    vec3 color = vec3(1.0, 1.0 - vUncertainty, 0.0);
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            depthWrite: false
        });
    }

    initializeGeometries() {
        // Basic geometries
        this.geometries.sphere = new THREE.SphereGeometry(1, 32, 16);
        this.geometries.box = new THREE.BoxGeometry(1, 1, 1);
        this.geometries.cylinder = new THREE.CylinderGeometry(1, 1, 1, 32);
        this.geometries.plane = new THREE.PlaneGeometry(1, 1, 32, 32);
    }

    startRenderLoop() {
        const animate = (time) => {
            requestAnimationFrame(animate);

            // Update controls
            if (this.controls.main) {
                this.controls.main.update();
            }

            // Update shader uniforms
            if (this.materials.deformation && this.materials.deformation.uniforms) {
                this.materials.deformation.uniforms.time.value = time * 0.001;
            }

            if (this.materials.uncertainty && this.materials.uncertainty.uniforms) {
                this.materials.uncertainty.uniforms.time.value = time * 0.001;
            }

            // Render scene
            if (this.renderers.main && this.scenes.main && this.cameras.main) {
                this.renderers.main.render(this.scenes.main, this.cameras.main);
            }
        };

        animate(0);
    }

    handleResize() {
        const container = document.getElementById('webgl-container');
        if (!container || !this.renderers.main || !this.cameras.main) return;

        const width = container.clientWidth;
        const height = container.clientHeight;

        this.cameras.main.aspect = width / height;
        this.cameras.main.updateProjectionMatrix();

        this.renderers.main.setSize(width, height);
    }

    async initializeCharts() {
        // Initialize all chart containers with Plotly
        await this.initializeConfidenceChart();
        await this.initializeConvergenceChart();
        await this.initializeParameterEvolutionChart();
        await this.initializeProbabilityChart();
        await this.initializeFeatureImportanceChart();
        await this.initializeModelComparisonChart();
        await this.initializeUncertaintyHeatmap();
        await this.initializeSensitivityChart();
    }

    async initializeConfidenceChart() {
        const container = document.getElementById('confidence-chart');
        if (!container) return;

        const data = [{
            type: 'bar',
            x: ['Fault', 'Dike', 'Sill', 'Point Source'],
            y: [0, 0, 0, 0],
            marker: {
                color: [this.config.primaryColor, this.config.secondaryColor, 
                       this.config.accentColor, this.config.warningColor]
            }
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor },
            yaxis: { color: this.config.textColor, title: 'Confidence' },
            margin: { t: 20, b: 40, l: 60, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.confidence = container;
    }

    async initializeConvergenceChart() {
        const container = document.getElementById('convergence-chart');
        if (!container) return;

        const data = [{
            type: 'scatter',
            mode: 'lines',
            x: [],
            y: [],
            line: { color: this.config.primaryColor, width: 2 },
            name: 'Best Objective'
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor, title: 'Trial' },
            yaxis: { color: this.config.textColor, title: 'Objective Value' },
            margin: { t: 20, b: 40, l: 60, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.convergence = container;
    }

    async initializeParameterEvolutionChart() {
        const container = document.getElementById('parameter-evolution');
        if (!container) return;

        const data = [];
        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor, title: 'Time' },
            yaxis: { color: this.config.textColor, title: 'Parameter Value' },
            margin: { t: 20, b: 40, l: 60, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.parameterEvolution = container;
    }

    async initializeProbabilityChart() {
        const container = document.getElementById('probability-chart');
        if (!container) return;

        const data = [{
            type: 'pie',
            labels: ['Fault', 'Dike', 'Sill', 'Point Source', 'Distributed', 'Other'],
            values: [0, 0, 0, 0, 0, 0],
            marker: {
                colors: [this.config.primaryColor, this.config.secondaryColor,
                        this.config.accentColor, this.config.warningColor,
                        '#9b59b6', '#e74c3c']
            }
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            margin: { t: 20, b: 20, l: 20, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.probability = container;
    }

    async initializeFeatureImportanceChart() {
        const container = document.getElementById('feature-importance');
        if (!container) return;

        const data = [{
            type: 'bar',
            orientation: 'h',
            x: [],
            y: [],
            marker: { color: this.config.accentColor }
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor, title: 'Importance' },
            yaxis: { color: this.config.textColor },
            margin: { t: 20, b: 40, l: 100, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.featureImportance = container;
    }

    async initializeModelComparisonChart() {
        const container = document.getElementById('model-comparison-chart');
        if (!container) return;

        const data = [{
            type: 'bar',
            x: [],
            y: [],
            marker: { color: this.config.secondaryColor },
            name: 'AIC Score'
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor },
            yaxis: { color: this.config.textColor, title: 'AIC Score' },
            margin: { t: 20, b: 60, l: 60, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.modelComparison = container;
    }

    async initializeUncertaintyHeatmap() {
        const container = document.getElementById('uncertainty-heatmap');
        if (!container) return;

        const data = [{
            type: 'heatmap',
            z: [],
            x: [],
            y: [],
            colorscale: 'Viridis'
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor },
            yaxis: { color: this.config.textColor },
            margin: { t: 20, b: 60, l: 60, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.uncertaintyHeatmap = container;
    }

    async initializeSensitivityChart() {
        const container = document.getElementById('sensitivity-chart');
        if (!container) return;

        const data = [{
            type: 'scatter',
            mode: 'markers+lines',
            x: [],
            y: [],
            marker: { color: this.config.warningColor, size: 8 },
            line: { color: this.config.warningColor }
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor, title: 'Parameter Change (%)' },
            yaxis: { color: this.config.textColor, title: 'Objective Change (%)' },
            margin: { t: 20, b: 60, l: 60, r: 20 }
        };

        Plotly.newPlot(container, data, layout, {responsive: true});
        this.charts.sensitivity = container;
    }

    async setupEventHandlers() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab || e.target.closest('.tab-button').dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Control buttons
        document.getElementById('start-optimization')?.addEventListener('click', () => {
            this.startOptimization();
        });

        document.getElementById('export-results')?.addEventListener('click', () => {
            this.exportResults();
        });

        // Parameter controls
        document.getElementById('algorithm-select')?.addEventListener('change', (e) => {
            this.config.algorithm = e.target.value;
        });

        document.getElementById('max-trials')?.addEventListener('change', (e) => {
            this.config.max_trials = parseInt(e.target.value);
        });

        document.getElementById('parallel-workers')?.addEventListener('change', (e) => {
            this.config.parallel_workers = parseInt(e.target.value);
        });

        // Visualization controls
        document.getElementById('view-mode')?.addEventListener('change', (e) => {
            this.updateVisualization(e.target.value);
        });

        document.getElementById('color-scheme')?.addEventListener('change', (e) => {
            this.updateColorScheme(e.target.value);
        });

        document.getElementById('animation-speed')?.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = `${speed}x`;
            this.updateAnimationSpeed(speed);
        });

        // Modal close
        document.querySelector('.modal-close')?.addEventListener('click', () => {
            this.hideModal();
        });

        // Close modal on outside click
        document.getElementById('error-modal')?.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.hideModal();
            }
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.toggle('active', button.dataset.tab === tabName);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        this.activeTab = tabName;

        // Trigger resize for charts
        setTimeout(() => this.resizeCharts(), 100);
    }

    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(chart);
            }
        });
    }

    async loadInitialData() {
        this.showLoading('Loading initial data...');

        try {
            // Load configuration and available models
            const response = await fetch(`${this.config.api_base}/status`);
            const data = await response.json();

            if (data.success) {
                this.updateStatus('connected');
                await this.updateModelList(data.available_models);
                await this.updateParameterControls(data.available_parameters);
            }

        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.updateStatus('disconnected');
        } finally {
            this.hideLoading();
        }
    }

    updateStatus(status) {
        const connectionStatus = document.getElementById('connection-status');
        const optimizationStatus = document.getElementById('optimization-status');

        if (connectionStatus) {
            if (status === 'connected') {
                connectionStatus.classList.add('connected');
                connectionStatus.querySelector('span').textContent = 'Connected';
            } else {
                connectionStatus.classList.remove('connected');
                connectionStatus.querySelector('span').textContent = 'Disconnected';
            }
        }

        if (optimizationStatus) {
            if (this.isOptimizing) {
                optimizationStatus.classList.add('optimizing');
                optimizationStatus.querySelector('span').textContent = 'Optimizing...';
            } else {
                optimizationStatus.classList.remove('optimizing');
                optimizationStatus.querySelector('span').textContent = 'Ready';
            }
        }
    }

    async updateModelList(models) {
        const container = document.getElementById('model-cards');
        if (!container || !models) return;

        container.innerHTML = models.map(model => `
            <div class="model-card" data-model="${model.name}">
                <h4>${model.display_name}</h4>
                <div class="performance">
                    <div>Type: ${model.type}</div>
                    <div>Parameters: ${model.parameter_count}</div>
                    <div>Status: ${model.status}</div>
                </div>
            </div>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', () => {
                container.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                this.selectModel(card.dataset.model);
            });
        });
    }

    async updateParameterControls(parameters) {
        const container = document.getElementById('parameter-controls');
        if (!container || !parameters) return;

        container.innerHTML = parameters.map(param => {
            let control;
            
            if (param.type === 'continuous') {
                control = `
                    <input type="range" 
                           id="${param.name}" 
                           min="${param.bounds[0]}" 
                           max="${param.bounds[1]}" 
                           step="${(param.bounds[1] - param.bounds[0]) / 100}"
                           value="${param.default || param.bounds[0]}">
                    <div class="range-values">
                        <span>${param.bounds[0]}</span>
                        <span id="${param.name}-value">${param.default || param.bounds[0]}</span>
                        <span>${param.bounds[1]}</span>
                    </div>
                `;
            } else if (param.type === 'categorical') {
                control = `
                    <select id="${param.name}">
                        ${param.choices.map(choice => 
                            `<option value="${choice}" ${choice === param.default ? 'selected' : ''}>${choice}</option>`
                        ).join('')}
                    </select>
                `;
            } else {
                control = `
                    <input type="number" 
                           id="${param.name}" 
                           min="${param.bounds[0]}" 
                           max="${param.bounds[1]}" 
                           value="${param.default || param.bounds[0]}">
                `;
            }

            return `
                <div class="parameter-control">
                    <label for="${param.name}">${param.display_name}:</label>
                    ${control}
                    <small>${param.description}</small>
                </div>
            `;
        }).join('');

        // Add change handlers
        container.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('input', (e) => {
                if (input.type === 'range') {
                    const valueSpan = document.getElementById(`${input.id}-value`);
                    if (valueSpan) {
                        valueSpan.textContent = parseFloat(e.target.value).toFixed(3);
                    }
                }
                this.updateParameter(input.id, e.target.value);
            });
        });
    }

    async startOptimization() {
        if (this.isOptimizing) {
            this.stopOptimization();
            return;
        }

        this.isOptimizing = true;
        this.updateStatus('optimizing');
        
        const button = document.getElementById('start-optimization');
        if (button) {
            button.innerHTML = '<i class="fas fa-stop"></i> Stop Optimization';
        }

        try {
            const optimizationConfig = this.getOptimizationConfig();
            
            const response = await fetch(`${this.config.api_base}/optimize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(optimizationConfig)
            });

            const result = await response.json();

            if (result.success) {
                this.optimizationId = result.optimization_id;
                this.startOptimizationMonitoring();
            } else {
                throw new Error(result.error || 'Optimization failed to start');
            }

        } catch (error) {
            console.error('Optimization failed:', error);
            this.showError('Optimization failed: ' + error.message);
            this.stopOptimization();
        }
    }

    stopOptimization() {
        this.isOptimizing = false;
        this.updateStatus('ready');

        const button = document.getElementById('start-optimization');
        if (button) {
            button.innerHTML = '<i class="fas fa-play"></i> Start Optimization';
        }

        if (this.optimizationId) {
            fetch(`${this.config.api_base}/optimize/${this.optimizationId}/stop`, {
                method: 'POST'
            });
        }

        if (this.optimizationMonitor) {
            clearInterval(this.optimizationMonitor);
            this.optimizationMonitor = null;
        }
    }

    getOptimizationConfig() {
        const config = {
            algorithm: document.getElementById('algorithm-select')?.value || 'gaussian_process',
            max_trials: parseInt(document.getElementById('max-trials')?.value || 100),
            parallel_workers: parseInt(document.getElementById('parallel-workers')?.value || 4),
            parameters: {}
        };

        // Collect parameter values
        document.querySelectorAll('#parameter-controls input, #parameter-controls select').forEach(input => {
            if (input.type === 'range' || input.type === 'number') {
                config.parameters[input.id] = parseFloat(input.value);
            } else {
                config.parameters[input.id] = input.value;
            }
        });

        return config;
    }

    startOptimizationMonitoring() {
        this.optimizationMonitor = setInterval(async () => {
            try {
                const response = await fetch(`${this.config.api_base}/optimize/${this.optimizationId}/status`);
                const data = await response.json();

                if (data.success) {
                    this.updateOptimizationProgress(data);

                    if (data.status === 'completed' || data.status === 'failed') {
                        this.stopOptimization();
                        
                        if (data.status === 'completed') {
                            this.showOptimizationResults(data.results);
                        } else {
                            this.showError('Optimization failed: ' + data.error);
                        }
                    }
                }
            } catch (error) {
                console.error('Failed to fetch optimization status:', error);
            }
        }, this.config.update_interval);
    }

    updateOptimizationProgress(data) {
        // Update progress metrics
        document.getElementById('best-objective').textContent = 
            data.best_objective ? data.best_objective.toFixed(6) : '--';
        document.getElementById('trial-count').textContent = data.trial_count || 0;
        document.getElementById('convergence-status').textContent = 
            data.converged ? 'Converged' : 'Running';

        // Update convergence chart
        if (data.optimization_history && this.charts.convergence) {
            const trials = data.optimization_history.map((_, i) => i + 1);
            const objectives = data.optimization_history.map(trial => trial.objective_value);

            const update = {
                x: [trials],
                y: [objectives]
            };

            Plotly.restyle(this.charts.convergence, update, [0]);
        }

        // Update parameter evolution
        if (data.parameter_history && this.charts.parameterEvolution) {
            this.updateParameterEvolution(data.parameter_history);
        }

        // Update classification if available
        if (data.classification_results) {
            this.updateClassificationResults(data.classification_results);
        }
    }

    updateParameterEvolution(parameterHistory) {
        if (!parameterHistory.length) return;

        const paramNames = Object.keys(parameterHistory[0].parameters);
        const data = paramNames.map((paramName, index) => ({
            type: 'scatter',
            mode: 'lines',
            name: paramName,
            x: parameterHistory.map((_, i) => i + 1),
            y: parameterHistory.map(trial => trial.parameters[paramName]),
            line: { 
                color: this.getParameterColor(index),
                width: 2
            }
        }));

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor, title: 'Trial' },
            yaxis: { color: this.config.textColor, title: 'Parameter Value' },
            margin: { t: 20, b: 40, l: 60, r: 20 }
        };

        Plotly.react(this.charts.parameterEvolution, data, layout);
    }

    getParameterColor(index) {
        const colors = [
            this.config.primaryColor,
            this.config.secondaryColor,
            this.config.accentColor,
            this.config.warningColor,
            '#9b59b6',
            '#e74c3c',
            '#f39c12',
            '#2ecc71'
        ];
        return colors[index % colors.length];
    }

    updateClassificationResults(results) {
        // Update predicted type
        document.getElementById('predicted-type').textContent = 
            results.predicted_type || 'Unknown';
        
        const confidence = Math.round((results.confidence || 0) * 100);
        document.getElementById('prediction-confidence').textContent = `${confidence}%`;

        // Update confidence chart
        if (results.class_probabilities && this.charts.confidence) {
            const sourceTypes = Object.keys(results.class_probabilities);
            const probabilities = Object.values(results.class_probabilities);

            const update = {
                x: [sourceTypes],
                y: [probabilities]
            };

            Plotly.restyle(this.charts.confidence, update, [0]);
        }

        // Update probability pie chart
        if (results.class_probabilities && this.charts.probability) {
            const labels = Object.keys(results.class_probabilities);
            const values = Object.values(results.class_probabilities);

            const update = {
                labels: [labels],
                values: [values]
            };

            Plotly.restyle(this.charts.probability, update, [0]);
        }
    }

    showOptimizationResults(results) {
        console.log('Optimization completed:', results);
        
        // Update all visualizations with final results
        this.updateFinalResults(results);
        
        // Show success message
        this.showMessage('Optimization completed successfully!', 'success');
    }

    updateFinalResults(results) {
        // Update model comparison
        if (results.model_comparison && this.charts.modelComparison) {
            const modelNames = results.model_comparison.map(m => m.model_name);
            const aicScores = results.model_comparison.map(m => m.aic_score);

            const update = {
                x: [modelNames],
                y: [aicScores]
            };

            Plotly.restyle(this.charts.modelComparison, update, [0]);
        }

        // Update uncertainty visualization
        if (results.uncertainty_analysis) {
            this.updateUncertaintyVisualization(results.uncertainty_analysis);
        }

        // Update 3D visualization
        if (results.best_model_parameters) {
            this.update3DVisualization(results.best_model_parameters);
        }
    }

    updateUncertaintyVisualization(uncertaintyData) {
        if (!uncertaintyData.parameter_correlation || !this.charts.uncertaintyHeatmap) return;

        const matrix = uncertaintyData.parameter_correlation;
        const paramNames = uncertaintyData.parameter_names;

        const data = [{
            type: 'heatmap',
            z: matrix,
            x: paramNames,
            y: paramNames,
            colorscale: 'RdYlBu',
            zmin: -1,
            zmax: 1
        }];

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: this.config.textColor },
            xaxis: { color: this.config.textColor },
            yaxis: { color: this.config.textColor },
            margin: { t: 20, b: 60, l: 60, r: 20 }
        };

        Plotly.react(this.charts.uncertaintyHeatmap, data, layout);
    }

    update3DVisualization(parameters) {
        if (!this.scenes.main || !parameters) return;

        // Clear existing meshes
        Object.values(this.meshes).forEach(mesh => {
            if (mesh.parent) {
                mesh.parent.remove(mesh);
            }
        });
        this.meshes = {};

        // Create source geometry based on parameters
        const sourceType = parameters.source_type || 'point_source';
        
        switch (sourceType) {
            case 'fault':
                this.create3DFault(parameters);
                break;
            case 'dike':
                this.create3DDike(parameters);
                break;
            case 'point_source':
                this.create3DPointSource(parameters);
                break;
            default:
                this.create3DPointSource(parameters);
        }

        // Create deformation field if available
        if (parameters.deformation_field) {
            this.create3DDeformationField(parameters.deformation_field);
        }
    }

    create3DFault(parameters) {
        const geometry = this.geometries.plane.clone();
        geometry.scale(
            parameters.length || 1000,
            parameters.width || 1000,
            1
        );

        const mesh = new THREE.Mesh(geometry, this.materials.fault);
        mesh.position.set(
            parameters.x0 || 0,
            parameters.y0 || 0,
            parameters.z0 || 1000
        );

        if (parameters.strike !== undefined) {
            mesh.rotation.z = THREE.MathUtils.degToRad(parameters.strike);
        }
        if (parameters.dip !== undefined) {
            mesh.rotation.x = THREE.MathUtils.degToRad(90 - parameters.dip);
        }

        this.scenes.main.add(mesh);
        this.meshes.fault = mesh;
    }

    create3DDike(parameters) {
        const geometry = this.geometries.box.clone();
        geometry.scale(
            parameters.length || 1000,
            parameters.thickness || 10,
            parameters.height || 500
        );

        const mesh = new THREE.Mesh(geometry, this.materials.dike);
        mesh.position.set(
            parameters.x0 || 0,
            parameters.y0 || 0,
            parameters.z0 || 1000
        );

        this.scenes.main.add(mesh);
        this.meshes.dike = mesh;
    }

    create3DPointSource(parameters) {
        const radius = Math.abs(parameters.volume_change || 1000) / 1000;
        const geometry = this.geometries.sphere.clone();
        geometry.scale(radius, radius, radius);

        const mesh = new THREE.Mesh(geometry, this.materials.point_source);
        mesh.position.set(
            parameters.x0 || 0,
            parameters.y0 || 0,
            parameters.z0 || 1000
        );

        this.scenes.main.add(mesh);
        this.meshes.point_source = mesh;
    }

    create3DDeformationField(deformationData) {
        if (!deformationData.coordinates || !deformationData.displacements) return;

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(deformationData.coordinates.flat());
        const displacements = new Float32Array(deformationData.displacements.map(d => 
            Math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
        ));

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('displacement', new THREE.BufferAttribute(displacements, 1));

        const mesh = new THREE.Points(geometry, this.materials.deformation);
        this.scenes.main.add(mesh);
        this.meshes.deformation = mesh;
    }

    selectModel(modelName) {
        this.currentModel = modelName;
        console.log(`Selected model: ${modelName}`);
        
        // Update model info panel
        const infoPanel = document.getElementById('model-info');
        if (infoPanel) {
            infoPanel.innerHTML = `
                <div><strong>Model:</strong> ${modelName}</div>
                <div><strong>Type:</strong> ${this.getModelType(modelName)}</div>
                <div><strong>Status:</strong> Active</div>
            `;
        }
    }

    getModelType(modelName) {
        const typeMap = {
            'elastic_dislocation': 'Elastic Dislocation',
            'viscoelastic': 'Viscoelastic',
            'poroelastic': 'Poroelastic',
            'distributed_slip': 'Distributed Slip',
            'composite': 'Composite Model'
        };
        return typeMap[modelName] || modelName;
    }

    updateParameter(paramName, value) {
        this.parameterBounds[paramName] = parseFloat(value) || value;
        console.log(`Parameter ${paramName} updated to ${value}`);
    }

    updateVisualization(viewMode) {
        console.log(`Switching to view mode: ${viewMode}`);
        
        // Hide all visualization meshes
        Object.values(this.meshes).forEach(mesh => {
            mesh.visible = false;
        });

        // Show relevant meshes for the selected view mode
        switch (viewMode) {
            case 'source_geometry':
                if (this.meshes.fault) this.meshes.fault.visible = true;
                if (this.meshes.dike) this.meshes.dike.visible = true;
                if (this.meshes.point_source) this.meshes.point_source.visible = true;
                break;
            case 'deformation_field':
                if (this.meshes.deformation) this.meshes.deformation.visible = true;
                break;
            case 'uncertainty_volume':
                if (this.meshes.uncertainty) this.meshes.uncertainty.visible = true;
                break;
        }
    }

    updateColorScheme(scheme) {
        console.log(`Switching to color scheme: ${scheme}`);
        
        // Update material colors based on scheme
        const schemeColors = this.getColorScheme(scheme);
        
        if (this.materials.deformation && this.materials.deformation.uniforms) {
            this.materials.deformation.uniforms.colorScale.value.set(
                schemeColors.primary.r, schemeColors.primary.g, schemeColors.primary.b
            );
        }
    }

    getColorScheme(schemeName) {
        const schemes = {
            viridis: { primary: new THREE.Color(0x440154), secondary: new THREE.Color(0xfde725) },
            plasma: { primary: new THREE.Color(0x0d0887), secondary: new THREE.Color(0xf0f921) },
            coolwarm: { primary: new THREE.Color(0x3b4cc0), secondary: new THREE.Color(0xb40426) },
            seismic: { primary: new THREE.Color(0x000080), secondary: new THREE.Color(0x800000) }
        };
        return schemes[schemeName] || schemes.viridis;
    }

    updateAnimationSpeed(speed) {
        this.config.animation_speed = speed;
    }

    async startRealTimeUpdates() {
        if (!this.config.enable_realtime) return;

        this.updateTimer = setInterval(async () => {
            if (!this.isOptimizing) {
                await this.fetchLatestData();
            }
        }, this.config.update_interval);
    }

    async fetchLatestData() {
        try {
            const response = await fetch(`${this.config.api_base}/latest`);
            const data = await response.json();

            if (data.success && data.results) {
                this.currentData = data.results;
                this.updateDisplays();
            }
        } catch (error) {
            console.warn('Failed to fetch latest data:', error);
        }
    }

    updateDisplays() {
        if (!this.currentData) return;

        // Update based on active tab
        switch (this.activeTab) {
            case 'overview':
                this.updateOverviewDisplays();
                break;
            case 'classification':
                this.updateClassificationDisplays();
                break;
            case 'optimization':
                this.updateOptimizationDisplays();
                break;
            case 'models':
                this.updateModelDisplays();
                break;
            case 'uncertainty':
                this.updateUncertaintyDisplays();
                break;
            case 'visualization':
                this.updateVisualizationDisplays();
                break;
        }
    }

    updateOverviewDisplays() {
        // Implementation for overview updates
    }

    updateClassificationDisplays() {
        // Implementation for classification updates
    }

    updateOptimizationDisplays() {
        // Implementation for optimization updates
    }

    updateModelDisplays() {
        // Implementation for model updates
    }

    updateUncertaintyDisplays() {
        // Implementation for uncertainty updates
    }

    updateVisualizationDisplays() {
        // Implementation for visualization updates
    }

    async exportResults() {
        if (!this.currentData) {
            this.showError('No data available to export');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            configuration: this.config,
            results: this.currentData,
            optimization_history: this.optimizationHistory,
            classification_history: this.classificationHistory
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `source_characterization_results_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        const messageElement = overlay?.querySelector('.loading-message');
        
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        if (overlay) {
            overlay.classList.add('active');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    showError(message) {
        const modal = document.getElementById('error-modal');
        const messageElement = document.getElementById('error-message');
        
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        if (modal) {
            modal.classList.add('active');
        }
    }

    hideModal() {
        const modal = document.getElementById('error-modal');
        if (modal) {
            modal.classList.remove('active');
        }
    }

    showMessage(message, type = 'info') {
        console.log(`${type.toUpperCase()}: ${message}`);
        
        // Create temporary notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${type === 'success' ? this.config.primaryColor : this.config.errorColor};
            color: white;
            border-radius: 5px;
            z-index: 1002;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        document.body.appendChild(notification);
        
        // Fade in
        setTimeout(() => notification.style.opacity = '1', 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    destroy() {
        // Clean up resources
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }
        
        if (this.optimizationMonitor) {
            clearInterval(this.optimizationMonitor);
        }
        
        if (this.websocket) {
            this.websocket.close();
        }

        // Dispose Three.js resources
        Object.values(this.scenes).forEach(scene => {
            scene.traverse(object => {
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        });

        Object.values(this.renderers).forEach(renderer => renderer.dispose());
        
        console.log('Source Classification Dashboard destroyed');
    }
}

// Example usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SourceClassificationDashboard;
} else if (typeof window !== 'undefined') {
    window.SourceClassificationDashboard = SourceClassificationDashboard;
}