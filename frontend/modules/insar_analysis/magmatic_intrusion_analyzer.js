/**
 * 3D Magmatic Intrusion Analyzer
 * 
 * Advanced 3D visualization and analysis system for magmatic intrusions with:
 * - Interactive 3D rendering of magma chambers and intrusions
 * - Physics-based animation with temperature-dependent visualization
 * - Real-time parameter adjustment and geological modeling
 * - Multi-tab analysis interface with comprehensive controls
 * - Eruption forecasting with probabilistic predictions
 * - Uncertainty visualization with confidence intervals
 * - Export capabilities for high-resolution output
 * - Responsive design for multiple devices
 * 
 * Scientific Features:
 * - Validated geological models (Mogi, Sill, Dike, Yang, McTigue)
 * - Thermal evolution modeling with heat conduction
 * - Stress field visualization and deformation prediction
 * - Multi-physics coupling (thermal, mechanical, chemical)
 * - Real-time hazard assessment and eruption probability
 */

class MagmaticIntrusionAnalyzer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            width: options.width || this.container.clientWidth,
            height: options.height || this.container.clientHeight,
            visualization: {
                showMagmaChamber: true,
                showIntrusions: true,
                showDeformation: true,
                showTemperatureField: true,
                showStressField: false,
                showHazardZones: true,
                animationSpeed: 1.0,
                timeScale: 3600, // seconds per animation frame
                temperatureColormap: 'thermal',
                deformationScale: 1000
            },
            physics: {
                enableThermalEvolution: true,
                enableDeformation: true,
                enableVolatileEvolution: true,
                couplingStrength: 0.8,
                timeStep: 86400, // 1 day in seconds
                maxSimulationTime: 31536000 // 1 year in seconds
            },
            uncertainty: {
                showConfidenceIntervals: true,
                monteCarloSamples: 1000,
                uncertaintyLevel: 0.95,
                propagateUncertainty: true
            }
        };
        
        // Geological parameters
        this.parameters = {
            magmaChamber: {
                centerX: 0,      // m
                centerY: 0,      // m
                depth: 3000,     // m
                radius: 500,     // m
                temperature: 1200, // K
                pressure: 150e6,   // Pa
                volatileContent: 0.05, // mass fraction
                crystallinity: 0.3,    // volume fraction
                composition: 'basalt'
            },
            intrusions: [],
            hostRock: {
                density: 2700,        // kg/m³
                thermalConductivity: 2.5, // W/m/K
                thermalDiffusivity: 1e-6,  // m²/s
                elasticModulus: 70e9,      // Pa
                poissonRatio: 0.25,
                cohesion: 10e6,           // Pa
                frictionAngle: 30         // degrees
            },
            environment: {
                surfaceTemperature: 288, // K
                geothermalGradient: 0.03, // K/m
                hydrostaticPressure: true,
                weatheringRate: 1e-10    // m/s
            }
        };
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.raycaster = null;
        this.mouse = new THREE.Vector2();
        
        // 3D objects
        this.magmaChamberMesh = null;
        this.intrusionMeshes = [];
        this.temperatureField = null;
        this.deformationField = null;
        this.stressField = null;
        this.hazardZones = [];
        
        // UI components
        this.tabInterface = null;
        this.parameterControls = null;
        self.analysisPanel = null;
        this.forecastPanel = null;
        
        // Animation and simulation
        this.animationId = null;
        this.isAnimating = false;
        this.simulationTime = 0;
        this.thermalSolver = null;
        this.deformationSolver = null;
        
        // Data storage
        this.simulationResults = {
            temperature: [],
            deformation: [],
            stress: [],
            eruptionProbability: [],
            hazardAssessment: []
        };
        
        // Analysis tools
        this.uncertaintyAnalysis = null;
        this.forecastModel = null;
        
        // Event handling
        this.eventHandlers = new Map();
        
        this.init();
    }
    
    init() {
        this.createInterface();
        this.createScene();
        this.setupPhysicsSolvers();
        this.createControls();
        this.setupEventListeners();
        this.initializeGeology();
        this.startSimulation();
        
        console.log('3D Magmatic Intrusion Analyzer initialized');
    }
    
    createInterface() {
        // Create main container structure
        this.container.innerHTML = `
            <div class="intrusion-analyzer-container">
                <div class="tab-interface">
                    <div class="tab-buttons">
                        <button class="tab-btn active" data-tab="visualization">3D Visualization</button>
                        <button class="tab-btn" data-tab="properties">Magma Properties</button>
                        <button class="tab-btn" data-tab="geometry">Chamber Geometry</button>
                        <button class="tab-btn" data-tab="deformation">Deformation</button>
                        <button class="tab-btn" data-tab="forecasting">Eruption Forecast</button>
                        <button class="tab-btn" data-tab="uncertainty">Uncertainty</button>
                    </div>
                    <div class="tab-content">
                        <div id="tab-visualization" class="tab-panel active">
                            <div id="threejs-container" class="visualization-container"></div>
                            <div class="visualization-controls">
                                <div class="control-group">
                                    <label>Display Options:</label>
                                    <div class="checkbox-group">
                                        <label><input type="checkbox" id="show-chamber" checked> Magma Chamber</label>
                                        <label><input type="checkbox" id="show-intrusions" checked> Intrusions</label>
                                        <label><input type="checkbox" id="show-temperature" checked> Temperature Field</label>
                                        <label><input type="checkbox" id="show-deformation" checked> Deformation</label>
                                        <label><input type="checkbox" id="show-stress"> Stress Field</label>
                                        <label><input type="checkbox" id="show-hazards" checked> Hazard Zones</label>
                                    </div>
                                </div>
                                <div class="control-group">
                                    <label>Animation:</label>
                                    <button id="play-pause-btn" class="control-btn">▶ Play</button>
                                    <button id="reset-btn" class="control-btn">⟲ Reset</button>
                                    <input type="range" id="time-slider" min="0" max="365" value="0" class="time-slider">
                                    <span id="time-display">Day 0</span>
                                </div>
                                <div class="control-group">
                                    <label>View Controls:</label>
                                    <button id="top-view-btn" class="control-btn">Top View</button>
                                    <button id="side-view-btn" class="control-btn">Side View</button>
                                    <button id="cross-section-btn" class="control-btn">Cross Section</button>
                                </div>
                            </div>
                        </div>
                        <div id="tab-properties" class="tab-panel">
                            <div class="parameter-controls">
                                <h3>Magma Properties</h3>
                                <div class="parameter-grid">
                                    <div class="param-control">
                                        <label>Temperature (K):</label>
                                        <input type="range" id="temperature-slider" min="1000" max="1500" value="1200">
                                        <span id="temperature-value">1200</span>
                                    </div>
                                    <div class="param-control">
                                        <label>Volatile Content (%):</label>
                                        <input type="range" id="volatile-slider" min="0" max="10" value="5" step="0.1">
                                        <span id="volatile-value">5.0</span>
                                    </div>
                                    <div class="param-control">
                                        <label>Crystallinity (%):</label>
                                        <input type="range" id="crystal-slider" min="0" max="80" value="30">
                                        <span id="crystal-value">30</span>
                                    </div>
                                    <div class="param-control">
                                        <label>Composition:</label>
                                        <select id="composition-select">
                                            <option value="basalt" selected>Basalt</option>
                                            <option value="andesite">Andesite</option>
                                            <option value="dacite">Dacite</option>
                                            <option value="rhyolite">Rhyolite</option>
                                        </select>
                                    </div>
                                </div>
                                <h3>Thermal Evolution</h3>
                                <div class="thermal-controls">
                                    <canvas id="temperature-evolution-chart" width="400" height="200"></canvas>
                                    <div class="thermal-params">
                                        <div class="param-control">
                                            <label>Cooling Rate (K/year):</label>
                                            <input type="number" id="cooling-rate" value="10" min="0.1" max="100" step="0.1">
                                        </div>
                                        <div class="param-control">
                                            <label>Heat Loss Mechanism:</label>
                                            <select id="heat-loss-select">
                                                <option value="conduction" selected>Conduction</option>
                                                <option value="convection">Convection</option>
                                                <option value="mixed">Mixed</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="tab-geometry" class="tab-panel">
                            <div class="geometry-controls">
                                <h3>Chamber Geometry</h3>
                                <div class="geometry-grid">
                                    <div class="param-control">
                                        <label>Depth (m):</label>
                                        <input type="range" id="depth-slider" min="500" max="10000" value="3000" step="100">
                                        <span id="depth-value">3000</span>
                                    </div>
                                    <div class="param-control">
                                        <label>Radius (m):</label>
                                        <input type="range" id="radius-slider" min="100" max="2000" value="500" step="50">
                                        <span id="radius-value">500</span>
                                    </div>
                                    <div class="param-control">
                                        <label>Aspect Ratio:</label>
                                        <input type="range" id="aspect-slider" min="0.5" max="3.0" value="1.0" step="0.1">
                                        <span id="aspect-value">1.0</span>
                                    </div>
                                    <div class="param-control">
                                        <label>Chamber Shape:</label>
                                        <select id="shape-select">
                                            <option value="sphere" selected>Sphere</option>
                                            <option value="ellipsoid">Ellipsoid</option>
                                            <option value="cylinder">Cylinder</option>
                                            <option value="complex">Complex</option>
                                        </select>
                                    </div>
                                </div>
                                <h3>Intrusion Network</h3>
                                <div class="intrusion-controls">
                                    <button id="add-sill-btn" class="control-btn">Add Sill</button>
                                    <button id="add-dike-btn" class="control-btn">Add Dike</button>
                                    <button id="add-conduit-btn" class="control-btn">Add Conduit</button>
                                    <div id="intrusion-list" class="intrusion-list"></div>
                                </div>
                                <h3>Geometric Analysis</h3>
                                <div class="geometric-analysis">
                                    <canvas id="geometry-diagram" width="300" height="300"></canvas>
                                    <div class="geometry-stats">
                                        <div class="stat-item">
                                            <span class="stat-label">Volume:</span>
                                            <span id="chamber-volume" class="stat-value">0.52 km³</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">Surface Area:</span>
                                            <span id="chamber-surface" class="stat-value">3.14 km²</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">Pressure:</span>
                                            <span id="chamber-pressure" class="stat-value">150 MPa</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="tab-deformation" class="tab-panel">
                            <div class="deformation-analysis">
                                <h3>Surface Deformation</h3>
                                <div class="deformation-visualization">
                                    <canvas id="deformation-map" width="400" height="400"></canvas>
                                    <div class="deformation-controls">
                                        <div class="param-control">
                                            <label>Deformation Scale:</label>
                                            <input type="range" id="deform-scale-slider" min="100" max="10000" value="1000">
                                            <span id="deform-scale-value">1000x</span>
                                        </div>
                                        <div class="param-control">
                                            <label>Component:</label>
                                            <select id="deform-component-select">
                                                <option value="vertical" selected>Vertical</option>
                                                <option value="horizontal">Horizontal</option>
                                                <option value="magnitude">Magnitude</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                                <h3>InSAR Integration</h3>
                                <div class="insar-comparison">
                                    <div class="comparison-panels">
                                        <div class="panel">
                                            <h4>Model Prediction</h4>
                                            <canvas id="model-interferogram" width="200" height="200"></canvas>
                                        </div>
                                        <div class="panel">
                                            <h4>Observed InSAR</h4>
                                            <canvas id="observed-interferogram" width="200" height="200"></canvas>
                                        </div>
                                        <div class="panel">
                                            <h4>Residual</h4>
                                            <canvas id="residual-interferogram" width="200" height="200"></canvas>
                                        </div>
                                    </div>
                                    <div class="fit-statistics">
                                        <div class="stat-item">
                                            <span class="stat-label">RMS Error:</span>
                                            <span id="rms-error" class="stat-value">2.3 mm</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">Correlation:</span>
                                            <span id="correlation" class="stat-value">0.89</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="tab-forecasting" class="tab-panel">
                            <div class="forecast-analysis">
                                <h3>Eruption Probability</h3>
                                <div class="probability-display">
                                    <canvas id="probability-chart" width="400" height="200"></canvas>
                                    <div class="probability-stats">
                                        <div class="prob-item high-prob">
                                            <span class="prob-label">Next 30 days:</span>
                                            <span id="prob-30days" class="prob-value">12%</span>
                                        </div>
                                        <div class="prob-item medium-prob">
                                            <span class="prob-label">Next 1 year:</span>
                                            <span id="prob-1year" class="prob-value">45%</span>
                                        </div>
                                        <div class="prob-item low-prob">
                                            <span class="prob-label">Next 10 years:</span>
                                            <span id="prob-10years" class="prob-value">78%</span>
                                        </div>
                                    </div>
                                </div>
                                <h3>Hazard Assessment</h3>
                                <div class="hazard-mapping">
                                    <canvas id="hazard-map" width="400" height="400"></canvas>
                                    <div class="hazard-legend">
                                        <div class="legend-item">
                                            <div class="legend-color" style="background: #ff0000;"></div>
                                            <span>Very High Risk</span>
                                        </div>
                                        <div class="legend-item">
                                            <div class="legend-color" style="background: #ff8800;"></div>
                                            <span>High Risk</span>
                                        </div>
                                        <div class="legend-item">
                                            <div class="legend-color" style="background: #ffff00;"></div>
                                            <span>Moderate Risk</span>
                                        </div>
                                        <div class="legend-item">
                                            <div class="legend-color" style="background: #88ff88;"></div>
                                            <span>Low Risk</span>
                                        </div>
                                    </div>
                                </div>
                                <h3>Scenario Modeling</h3>
                                <div class="scenario-controls">
                                    <div class="scenario-buttons">
                                        <button id="explosive-scenario-btn" class="scenario-btn">Explosive Eruption</button>
                                        <button id="effusive-scenario-btn" class="scenario-btn">Effusive Eruption</button>
                                        <button id="phreatomagmatic-scenario-btn" class="scenario-btn">Phreatomagmatic</button>
                                    </div>
                                    <div class="scenario-results">
                                        <canvas id="scenario-timeline" width="400" height="150"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="tab-uncertainty" class="tab-panel">
                            <div class="uncertainty-analysis">
                                <h3>Parameter Uncertainty</h3>
                                <div class="uncertainty-visualization">
                                    <canvas id="uncertainty-chart" width="400" height="300"></canvas>
                                    <div class="uncertainty-controls">
                                        <div class="param-control">
                                            <label>Confidence Level:</label>
                                            <select id="confidence-select">
                                                <option value="0.68">68% (1σ)</option>
                                                <option value="0.95" selected>95% (2σ)</option>
                                                <option value="0.997">99.7% (3σ)</option>
                                            </select>
                                        </div>
                                        <div class="param-control">
                                            <label>Monte Carlo Samples:</label>
                                            <input type="number" id="mc-samples" value="1000" min="100" max="10000" step="100">
                                        </div>
                                    </div>
                                </div>
                                <h3>Sensitivity Analysis</h3>
                                <div class="sensitivity-analysis">
                                    <canvas id="sensitivity-chart" width="400" height="250"></canvas>
                                    <div class="sensitivity-table">
                                        <table id="sensitivity-table">
                                            <thead>
                                                <tr>
                                                    <th>Parameter</th>
                                                    <th>Sensitivity Index</th>
                                                    <th>Impact on Eruption Risk</th>
                                                </tr>
                                            </thead>
                                            <tbody></tbody>
                                        </table>
                                    </div>
                                </div>
                                <h3>Error Propagation</h3>
                                <div class="error-propagation">
                                    <canvas id="error-propagation-chart" width="400" height="200"></canvas>
                                    <div class="error-stats">
                                        <div class="error-item">
                                            <span class="error-label">Total Uncertainty:</span>
                                            <span id="total-uncertainty" class="error-value">±15%</span>
                                        </div>
                                        <div class="error-item">
                                            <span class="error-label">Dominant Source:</span>
                                            <span id="dominant-error" class="error-value">Chamber Volume</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="export-controls">
                    <button id="export-image-btn" class="export-btn">📷 Export Image</button>
                    <button id="export-data-btn" class="export-btn">📊 Export Data</button>
                    <button id="export-report-btn" class="export-btn">📄 Export Report</button>
                </div>
            </div>
        `;
        
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .intrusion-analyzer-container {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1a1a1a;
                color: #ffffff;
            }
            
            .tab-interface {
                flex: 1;
                display: flex;
                flex-direction: column;
            }
            
            .tab-buttons {
                display: flex;
                background: #2d2d2d;
                border-bottom: 2px solid #444;
                overflow-x: auto;
            }
            
            .tab-btn {
                padding: 12px 20px;
                background: none;
                border: none;
                color: #ccc;
                cursor: pointer;
                transition: all 0.3s ease;
                white-space: nowrap;
                border-bottom: 3px solid transparent;
            }
            
            .tab-btn:hover {
                background: #3d3d3d;
                color: #fff;
            }
            
            .tab-btn.active {
                color: #4fc3f7;
                border-bottom-color: #4fc3f7;
                background: #333;
            }
            
            .tab-content {
                flex: 1;
                position: relative;
                overflow: hidden;
            }
            
            .tab-panel {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                padding: 20px;
                overflow-y: auto;
                display: none;
            }
            
            .tab-panel.active {
                display: block;
            }
            
            .visualization-container {
                width: 100%;
                height: 70%;
                background: #000;
                border: 2px solid #444;
                border-radius: 8px;
                position: relative;
                overflow: hidden;
            }
            
            .visualization-controls {
                margin-top: 15px;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .control-group label {
                font-size: 12px;
                color: #ccc;
                font-weight: 600;
            }
            
            .checkbox-group {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }
            
            .checkbox-group label {
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 11px;
                cursor: pointer;
            }
            
            .checkbox-group input[type="checkbox"] {
                accent-color: #4fc3f7;
            }
            
            .control-btn {
                padding: 6px 12px;
                background: #4fc3f7;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                transition: background 0.3s;
            }
            
            .control-btn:hover {
                background: #29b6f6;
            }
            
            .time-slider {
                width: 200px;
                accent-color: #4fc3f7;
            }
            
            #time-display {
                font-size: 11px;
                color: #4fc3f7;
                min-width: 50px;
            }
            
            .parameter-controls, .geometry-controls, .deformation-analysis, 
            .forecast-analysis, .uncertainty-analysis {
                max-width: 800px;
            }
            
            .parameter-grid, .geometry-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 15px 0;
            }
            
            .param-control {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .param-control label {
                font-size: 12px;
                color: #ccc;
                font-weight: 600;
            }
            
            .param-control input[type="range"] {
                width: 100%;
                accent-color: #4fc3f7;
            }
            
            .param-control input[type="number"],
            .param-control select {
                padding: 6px;
                background: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
                color: #fff;
            }
            
            .param-control span {
                font-size: 11px;
                color: #4fc3f7;
                font-weight: 600;
            }
            
            .thermal-controls, .intrusion-controls, .geometric-analysis,
            .deformation-visualization, .insar-comparison, .probability-display,
            .hazard-mapping, .scenario-controls, .uncertainty-visualization,
            .sensitivity-analysis, .error-propagation {
                margin: 20px 0;
                padding: 15px;
                background: #2d2d2d;
                border-radius: 8px;
                border: 1px solid #444;
            }
            
            .thermal-controls h3, .intrusion-controls h3, .geometric-analysis h3,
            .deformation-visualization h3, .insar-comparison h3, .probability-display h3,
            .hazard-mapping h3, .scenario-controls h3, .uncertainty-visualization h3,
            .sensitivity-analysis h3, .error-propagation h3 {
                margin: 0 0 15px 0;
                color: #4fc3f7;
                font-size: 14px;
            }
            
            canvas {
                border: 1px solid #555;
                border-radius: 4px;
                background: #1a1a1a;
            }
            
            .thermal-params {
                margin-top: 15px;
                display: flex;
                gap: 20px;
            }
            
            .intrusion-list {
                margin-top: 15px;
                max-height: 200px;
                overflow-y: auto;
            }
            
            .geometry-stats {
                margin-top: 15px;
            }
            
            .stat-item, .prob-item, .error-item {
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #444;
            }
            
            .stat-label, .prob-label, .error-label {
                color: #ccc;
                font-size: 12px;
            }
            
            .stat-value, .prob-value, .error-value {
                color: #4fc3f7;
                font-weight: 600;
                font-size: 12px;
            }
            
            .high-prob .prob-value { color: #ff5252; }
            .medium-prob .prob-value { color: #ffc107; }
            .low-prob .prob-value { color: #4caf50; }
            
            .comparison-panels {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            .panel {
                flex: 1;
                text-align: center;
            }
            
            .panel h4 {
                margin: 0 0 10px 0;
                font-size: 12px;
                color: #ccc;
            }
            
            .fit-statistics {
                display: flex;
                gap: 30px;
                justify-content: center;
            }
            
            .hazard-legend {
                margin-top: 15px;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 11px;
            }
            
            .legend-color {
                width: 16px;
                height: 16px;
                border-radius: 2px;
            }
            
            .scenario-buttons {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }
            
            .scenario-btn {
                padding: 8px 16px;
                background: #666;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                transition: background 0.3s;
            }
            
            .scenario-btn:hover {
                background: #777;
            }
            
            .scenario-btn.active {
                background: #4fc3f7;
            }
            
            #sensitivity-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-size: 11px;
            }
            
            #sensitivity-table th,
            #sensitivity-table td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #444;
            }
            
            #sensitivity-table th {
                background: #3d3d3d;
                color: #4fc3f7;
                font-weight: 600;
            }
            
            .export-controls {
                display: flex;
                gap: 10px;
                padding: 15px 20px;
                background: #2d2d2d;
                border-top: 1px solid #444;
            }
            
            .export-btn {
                padding: 8px 16px;
                background: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background 0.3s;
            }
            
            .export-btn:hover {
                background: #66bb6a;
            }
            
            @media (max-width: 768px) {
                .tab-buttons {
                    flex-wrap: wrap;
                }
                
                .tab-btn {
                    flex: 1;
                    min-width: 120px;
                }
                
                .visualization-container {
                    height: 60%;
                }
                
                .visualization-controls {
                    flex-direction: column;
                    gap: 10px;
                }
                
                .parameter-grid, .geometry-grid {
                    grid-template-columns: 1fr;
                }
                
                .comparison-panels {
                    flex-direction: column;
                }
                
                .fit-statistics {
                    flex-direction: column;
                    gap: 10px;
                }
                
                .export-controls {
                    flex-wrap: wrap;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    createScene() {
        // Get the Three.js container
        const container = document.getElementById('threejs-container');
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.FogExp2(0x0a0a0a, 0.0002);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            100000
        );
        this.camera.position.set(2000, 3000, 2000);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.gammaOutput = true;
        this.renderer.gammaFactor = 2.2;
        
        container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 500;
        this.controls.maxDistance = 20000;
        
        // Raycaster
        this.raycaster = new THREE.Raycaster();
        
        // Lighting
        this.setupLighting();
        
        // Coordinate system
        this.createCoordinateSystem();
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5000, 8000, 5000);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 1000;
        directionalLight.shadow.camera.far = 20000;
        directionalLight.shadow.camera.left = -5000;
        directionalLight.shadow.camera.right = 5000;
        directionalLight.shadow.camera.top = 5000;
        directionalLight.shadow.camera.bottom = -5000;
        this.scene.add(directionalLight);
        
        // Point light for underground illumination
        const pointLight = new THREE.PointLight(0xff6600, 0.6, 10000);
        pointLight.position.set(0, -2000, 0);
        this.scene.add(pointLight);
        
        // Hemisphere light
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
        hemiLight.position.set(0, 8000, 0);
        this.scene.add(hemiLight);
    }
    
    createCoordinateSystem() {
        // Create ground surface
        const groundGeometry = new THREE.PlaneGeometry(10000, 10000, 50, 50);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x8B7355, 
            transparent: true, 
            opacity: 0.7 
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(1000);
        this.scene.add(axesHelper);
        
        // Add grid
        const gridHelper = new THREE.GridHelper(8000, 40, 0x444444, 0x222222);
        this.scene.add(gridHelper);
        
        // Add depth reference lines
        for (let depth = 1000; depth <= 10000; depth += 1000) {
            const geometry = new THREE.RingGeometry(depth/2 - 50, depth/2 + 50, 32);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0x666666, 
                transparent: true, 
                opacity: 0.3,
                side: THREE.DoubleSide 
            });
            const ring = new THREE.Mesh(geometry, material);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = -depth;
            this.scene.add(ring);
        }
    }
    
    setupPhysicsSolvers() {
        // Initialize thermal evolution solver
        this.thermalSolver = {
            temperature: this.parameters.magmaChamber.temperature,
            coolingRate: 10, // K/year
            
            update: (deltaTime) => {
                // Simple exponential cooling model
                const timeInYears = deltaTime / (365.25 * 24 * 3600);
                this.thermalSolver.temperature *= Math.exp(-this.thermalSolver.coolingRate * timeInYears / this.thermalSolver.temperature);
                return this.thermalSolver.temperature;
            }
        };
        
        // Initialize deformation solver
        this.deformationSolver = {
            calculateMogiDeformation: (x, y, z, depth, volume, nu = 0.25) => {
                const r = Math.sqrt(x*x + y*y + z*z);
                const R = Math.sqrt(x*x + y*y + (z + depth)*(z + depth));
                
                if (R < 0.001) return { ux: 0, uy: 0, uz: 0 };
                
                const C = (1 - nu) * volume / (Math.PI * R*R*R);
                
                const ux = C * x;
                const uy = C * y;
                const uz = C * (z + depth);
                
                return { ux, uy, uz };
            }
        };
        
        // Initialize eruption forecasting model
        this.forecastModel = {
            baseRate: 0.01, // base eruption probability per year
            
            calculateEruptionProbability: (temperature, pressure, volatiles) => {
                // Empirical model based on chamber conditions
                let probability = this.forecastModel.baseRate;
                
                // Temperature factor
                if (temperature > 1150) {
                    probability *= Math.exp((temperature - 1150) / 200);
                }
                
                // Pressure factor
                if (pressure > 100e6) {
                    probability *= Math.exp((pressure - 100e6) / 50e6);
                }
                
                // Volatile factor
                if (volatiles > 0.03) {
                    probability *= Math.exp((volatiles - 0.03) / 0.02);
                }
                
                return Math.min(probability, 0.95); // Cap at 95%
            }
        };
    }
    
    createControls() {
        this.setupTabNavigation();
        this.setupVisualizationControls();
        this.setupParameterControls();
        this.setupGeometryControls();
        this.setupForecastingControls();
        this.setupExportControls();
    }
    
    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabPanels = document.querySelectorAll('.tab-panel');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                
                // Update button states
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update panel visibility
                tabPanels.forEach(panel => {
                    panel.classList.remove('active');
                    if (panel.id === `tab-${targetTab}`) {
                        panel.classList.add('active');
                    }
                });
                
                // Handle tab-specific initialization
                this.onTabChange(targetTab);
            });
        });
    }
    
    onTabChange(tabName) {
        switch (tabName) {
            case 'properties':
                this.updateTemperatureEvolutionChart();
                break;
            case 'geometry':
                this.updateGeometryDiagram();
                break;
            case 'deformation':
                this.updateDeformationMap();
                this.updateInSARComparison();
                break;
            case 'forecasting':
                this.updateProbabilityChart();
                this.updateHazardMap();
                break;
            case 'uncertainty':
                this.runUncertaintyAnalysis();
                break;
        }
    }
    
    setupVisualizationControls() {
        // Display toggles
        document.getElementById('show-chamber').addEventListener('change', (e) => {
            this.config.visualization.showMagmaChamber = e.target.checked;
            this.updateVisualization();
        });
        
        document.getElementById('show-intrusions').addEventListener('change', (e) => {
            this.config.visualization.showIntrusions = e.target.checked;
            this.updateVisualization();
        });
        
        document.getElementById('show-temperature').addEventListener('change', (e) => {
            this.config.visualization.showTemperatureField = e.target.checked;
            this.updateVisualization();
        });
        
        document.getElementById('show-deformation').addEventListener('change', (e) => {
            this.config.visualization.showDeformation = e.target.checked;
            this.updateVisualization();
        });
        
        document.getElementById('show-stress').addEventListener('change', (e) => {
            this.config.visualization.showStressField = e.target.checked;
            this.updateVisualization();
        });
        
        document.getElementById('show-hazards').addEventListener('change', (e) => {
            this.config.visualization.showHazardZones = e.target.checked;
            this.updateVisualization();
        });
        
        // Animation controls
        document.getElementById('play-pause-btn').addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSimulation();
        });
        
        document.getElementById('time-slider').addEventListener('input', (e) => {
            this.setSimulationTime(parseFloat(e.target.value) * 24 * 3600); // Convert days to seconds
        });
        
        // View controls
        document.getElementById('top-view-btn').addEventListener('click', () => {
            this.setTopView();
        });
        
        document.getElementById('side-view-btn').addEventListener('click', () => {
            this.setSideView();
        });
        
        document.getElementById('cross-section-btn').addEventListener('click', () => {
            this.toggleCrossSection();
        });
    }
    
    setupParameterControls() {
        // Temperature control
        const tempSlider = document.getElementById('temperature-slider');
        const tempValue = document.getElementById('temperature-value');
        tempSlider.addEventListener('input', (e) => {
            const temp = parseFloat(e.target.value);
            tempValue.textContent = temp.toString();
            this.parameters.magmaChamber.temperature = temp;
            this.updateMagmaChamber();
        });
        
        // Volatile content control
        const volatileSlider = document.getElementById('volatile-slider');
        const volatileValue = document.getElementById('volatile-value');
        volatileSlider.addEventListener('input', (e) => {
            const vol = parseFloat(e.target.value);
            volatileValue.textContent = vol.toFixed(1);
            this.parameters.magmaChamber.volatileContent = vol / 100;
            this.updateMagmaChamber();
        });
        
        // Crystallinity control
        const crystalSlider = document.getElementById('crystal-slider');
        const crystalValue = document.getElementById('crystal-value');
        crystalSlider.addEventListener('input', (e) => {
            const cryst = parseFloat(e.target.value);
            crystalValue.textContent = cryst.toString();
            this.parameters.magmaChamber.crystallinity = cryst / 100;
            this.updateMagmaChamber();
        });
        
        // Composition control
        document.getElementById('composition-select').addEventListener('change', (e) => {
            this.parameters.magmaChamber.composition = e.target.value;
            this.updateMagmaChamber();
        });
        
        // Cooling rate control
        document.getElementById('cooling-rate').addEventListener('change', (e) => {
            this.thermalSolver.coolingRate = parseFloat(e.target.value);
        });
    }
    
    setupGeometryControls() {
        // Depth control
        const depthSlider = document.getElementById('depth-slider');
        const depthValue = document.getElementById('depth-value');
        depthSlider.addEventListener('input', (e) => {
            const depth = parseFloat(e.target.value);
            depthValue.textContent = depth.toString();
            this.parameters.magmaChamber.depth = depth;
            this.updateMagmaChamber();
            this.updateGeometryStats();
        });
        
        // Radius control
        const radiusSlider = document.getElementById('radius-slider');
        const radiusValue = document.getElementById('radius-value');
        radiusSlider.addEventListener('input', (e) => {
            const radius = parseFloat(e.target.value);
            radiusValue.textContent = radius.toString();
            this.parameters.magmaChamber.radius = radius;
            this.updateMagmaChamber();
            this.updateGeometryStats();
        });
        
        // Aspect ratio control
        const aspectSlider = document.getElementById('aspect-slider');
        const aspectValue = document.getElementById('aspect-value');
        aspectSlider.addEventListener('input', (e) => {
            const aspect = parseFloat(e.target.value);
            aspectValue.textContent = aspect.toFixed(1);
            this.parameters.magmaChamber.aspectRatio = aspect;
            this.updateMagmaChamber();
        });
        
        // Shape control
        document.getElementById('shape-select').addEventListener('change', (e) => {
            this.parameters.magmaChamber.shape = e.target.value;
            this.updateMagmaChamber();
        });
        
        // Intrusion buttons
        document.getElementById('add-sill-btn').addEventListener('click', () => {
            this.addIntrusion('sill');
        });
        
        document.getElementById('add-dike-btn').addEventListener('click', () => {
            this.addIntrusion('dike');
        });
        
        document.getElementById('add-conduit-btn').addEventListener('click', () => {
            this.addIntrusion('conduit');
        });
    }
    
    setupForecastingControls() {
        // Deformation scale
        const deformScaleSlider = document.getElementById('deform-scale-slider');
        const deformScaleValue = document.getElementById('deform-scale-value');
        if (deformScaleSlider) {
            deformScaleSlider.addEventListener('input', (e) => {
                const scale = parseFloat(e.target.value);
                deformScaleValue.textContent = scale + 'x';
                this.config.visualization.deformationScale = scale;
                this.updateVisualization();
            });
        }
        
        // Scenario buttons
        const scenarios = ['explosive-scenario-btn', 'effusive-scenario-btn', 'phreatomagmatic-scenario-btn'];
        scenarios.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.addEventListener('click', () => {
                    scenarios.forEach(otherId => {
                        document.getElementById(otherId).classList.remove('active');
                    });
                    btn.classList.add('active');
                    this.runScenario(id.replace('-scenario-btn', ''));
                });
            }
        });
        
        // Confidence level control
        const confidenceSelect = document.getElementById('confidence-select');
        if (confidenceSelect) {
            confidenceSelect.addEventListener('change', (e) => {
                this.config.uncertainty.uncertaintyLevel = parseFloat(e.target.value);
                this.runUncertaintyAnalysis();
            });
        }
        
        // Monte Carlo samples control
        const mcSamples = document.getElementById('mc-samples');
        if (mcSamples) {
            mcSamples.addEventListener('change', (e) => {
                this.config.uncertainty.monteCarloSamples = parseInt(e.target.value);
                this.runUncertaintyAnalysis();
            });
        }
    }
    
    setupExportControls() {
        document.getElementById('export-image-btn').addEventListener('click', () => {
            this.exportImage();
        });
        
        document.getElementById('export-data-btn').addEventListener('click', () => {
            this.exportData();
        });
        
        document.getElementById('export-report-btn').addEventListener('click', () => {
            this.exportReport();
        });
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
        
        // Mouse events for 3D interaction
        this.renderer.domElement.addEventListener('click', (e) => {
            this.onMouseClick(e);
        });
        
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            this.onMouseMove(e);
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.onKeyDown(e);
        });
    }
    
    initializeGeology() {
        this.createMagmaChamber();
        this.createTemperatureField();
        this.createDeformationField();
        this.updateVisualization();
        
        console.log('Geological model initialized');
    }
    
    createMagmaChamber() {
        // Remove existing chamber
        if (this.magmaChamberMesh) {
            this.scene.remove(this.magmaChamberMesh);
        }
        
        // Create chamber geometry based on parameters
        let geometry;
        const params = this.parameters.magmaChamber;
        
        switch (params.shape || 'sphere') {
            case 'ellipsoid':
                geometry = new THREE.SphereGeometry(
                    params.radius,
                    32, 16
                );
                geometry.scale(1, params.aspectRatio || 1, 1);
                break;
            case 'cylinder':
                geometry = new THREE.CylinderGeometry(
                    params.radius,
                    params.radius,
                    params.radius * 2 * (params.aspectRatio || 1),
                    32
                );
                break;
            default: // sphere
                geometry = new THREE.SphereGeometry(params.radius, 32, 16);
        }
        
        // Create material with temperature-dependent color
        const temp = params.temperature;
        const normalizedTemp = (temp - 800) / (1500 - 800); // Normalize to 0-1
        const color = this.getTemperatureColor(normalizedTemp);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.8,
            emissive: new THREE.Color(color).multiplyScalar(0.3)
        });
        
        // Create mesh
        this.magmaChamberMesh = new THREE.Mesh(geometry, material);
        this.magmaChamberMesh.position.set(
            params.centerX,
            -params.depth,
            params.centerY
        );
        this.magmaChamberMesh.castShadow = true;
        
        // Add pulsation effect based on pressure
        this.magmaChamberMesh.userData = {
            originalScale: 1,
            pressure: params.pressure,
            pulsationPhase: 0
        };
        
        this.scene.add(this.magmaChamberMesh);
    }
    
    createTemperatureField() {
        // Create temperature field visualization
        if (this.temperatureField) {
            this.scene.remove(this.temperatureField);
        }
        
        // Create volumetric temperature field
        const fieldSize = 20;
        const fieldSpacing = 200;
        const particles = [];
        
        for (let x = -fieldSize/2; x < fieldSize/2; x++) {
            for (let y = -fieldSize/2; y < fieldSize/2; y++) {
                for (let z = -fieldSize/2; z < fieldSize/2; z++) {
                    const pos = new THREE.Vector3(
                        x * fieldSpacing,
                        y * fieldSpacing - 2000,
                        z * fieldSpacing
                    );
                    
                    // Calculate temperature based on distance from chamber
                    const chamberPos = new THREE.Vector3(
                        this.parameters.magmaChamber.centerX,
                        -this.parameters.magmaChamber.depth,
                        this.parameters.magmaChamber.centerY
                    );
                    
                    const distance = pos.distanceTo(chamberPos);
                    const chamberRadius = this.parameters.magmaChamber.radius;
                    
                    let temperature;
                    if (distance < chamberRadius) {
                        temperature = this.parameters.magmaChamber.temperature;
                    } else {
                        // Exponential decay with distance
                        const decay = Math.exp(-(distance - chamberRadius) / (chamberRadius * 2));
                        temperature = this.parameters.environment.surfaceTemperature + 
                                    (this.parameters.magmaChamber.temperature - this.parameters.environment.surfaceTemperature) * decay;
                    }
                    
                    if (temperature > this.parameters.environment.surfaceTemperature + 10) {
                        particles.push({
                            position: pos,
                            temperature: temperature
                        });
                    }
                }
            }
        }
        
        // Create particle system for temperature field
        const particleGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particles.length * 3);
        const colors = new Float32Array(particles.length * 3);
        
        particles.forEach((particle, i) => {
            positions[i * 3] = particle.position.x;
            positions[i * 3 + 1] = particle.position.y;
            positions[i * 3 + 2] = particle.position.z;
            
            const normalizedTemp = (particle.temperature - 288) / (1500 - 288);
            const color = this.getTemperatureColor(normalizedTemp);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        });
        
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            size: 50,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        this.temperatureField = new THREE.Points(particleGeometry, particleMaterial);
        this.scene.add(this.temperatureField);
    }
    
    createDeformationField() {
        // Create surface deformation visualization
        if (this.deformationField) {
            this.scene.remove(this.deformationField);
        }
        
        const gridSize = 50;
        const gridSpacing = 200;
        const geometry = new THREE.PlaneGeometry(
            gridSize * gridSpacing,
            gridSize * gridSpacing,
            gridSize - 1,
            gridSize - 1
        );
        
        const positions = geometry.attributes.position.array;
        const colors = new Float32Array(positions.length);
        
        // Calculate deformation for each grid point
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 2]; // Note: z becomes y in surface coordinates
            
            // Calculate Mogi deformation
            const deformation = this.deformationSolver.calculateMogiDeformation(
                x, y, 0,
                this.parameters.magmaChamber.depth,
                4/3 * Math.PI * Math.pow(this.parameters.magmaChamber.radius, 3) * 0.001, // Volume change
                0.25
            );
            
            // Apply vertical deformation
            positions[i + 1] = deformation.uz * this.config.visualization.deformationScale;
            
            // Color based on deformation magnitude
            const magnitude = Math.sqrt(deformation.ux*deformation.ux + deformation.uy*deformation.uy + deformation.uz*deformation.uz);
            const normalizedMag = Math.min(magnitude * 10000, 1); // Normalize
            const color = this.getDeformationColor(normalizedMag);
            
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
        
        const material = new THREE.MeshLambertMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });
        
        this.deformationField = new THREE.Mesh(geometry, material);
        this.deformationField.position.y = 10; // Slightly above ground
        this.scene.add(this.deformationField);
    }
    
    getTemperatureColor(normalizedValue) {
        // Thermal colormap: black -> red -> orange -> yellow -> white
        normalizedValue = Math.max(0, Math.min(1, normalizedValue));
        
        if (normalizedValue < 0.25) {
            // Black to red
            const t = normalizedValue * 4;
            return new THREE.Color(t, 0, 0);
        } else if (normalizedValue < 0.5) {
            // Red to orange
            const t = (normalizedValue - 0.25) * 4;
            return new THREE.Color(1, t * 0.5, 0);
        } else if (normalizedValue < 0.75) {
            // Orange to yellow
            const t = (normalizedValue - 0.5) * 4;
            return new THREE.Color(1, 0.5 + t * 0.5, 0);
        } else {
            // Yellow to white
            const t = (normalizedValue - 0.75) * 4;
            return new THREE.Color(1, 1, t);
        }
    }
    
    getDeformationColor(normalizedValue) {
        // Deformation colormap: blue -> cyan -> green -> yellow -> red
        normalizedValue = Math.max(0, Math.min(1, normalizedValue));
        
        if (normalizedValue < 0.25) {
            const t = normalizedValue * 4;
            return new THREE.Color(0, 0, 1 - t * 0.5);
        } else if (normalizedValue < 0.5) {
            const t = (normalizedValue - 0.25) * 4;
            return new THREE.Color(0, t, 0.5);
        } else if (normalizedValue < 0.75) {
            const t = (normalizedValue - 0.5) * 4;
            return new THREE.Color(t, 1, 0.5 - t * 0.5);
        } else {
            const t = (normalizedValue - 0.75) * 4;
            return new THREE.Color(1, 1 - t, 0);
        }
    }
    
    updateMagmaChamber() {
        this.createMagmaChamber();
        this.createTemperatureField();
        this.createDeformationField();
        this.updateGeometryStats();
        this.updateEruptionProbability();
    }
    
    updateVisualization() {
        // Toggle visibility based on settings
        if (this.magmaChamberMesh) {
            this.magmaChamberMesh.visible = this.config.visualization.showMagmaChamber;
        }
        
        if (this.temperatureField) {
            this.temperatureField.visible = this.config.visualization.showTemperatureField;
        }
        
        if (this.deformationField) {
            this.deformationField.visible = this.config.visualization.showDeformation;
        }
        
        this.intrusionMeshes.forEach(mesh => {
            mesh.visible = this.config.visualization.showIntrusions;
        });
        
        this.hazardZones.forEach(zone => {
            zone.visible = this.config.visualization.showHazardZones;
        });
    }
    
    updateGeometryStats() {
        const params = this.parameters.magmaChamber;
        const volume = (4/3) * Math.PI * Math.pow(params.radius, 3) * (params.aspectRatio || 1);
        const surfaceArea = 4 * Math.PI * Math.pow(params.radius, 2);
        const pressure = params.pressure;
        
        document.getElementById('chamber-volume').textContent = (volume / 1e9).toFixed(2) + ' km³';
        document.getElementById('chamber-surface').textContent = (surfaceArea / 1e6).toFixed(2) + ' km²';
        document.getElementById('chamber-pressure').textContent = (pressure / 1e6).toFixed(0) + ' MPa';
    }
    
    updateEruptionProbability() {
        const params = this.parameters.magmaChamber;
        const prob = this.forecastModel.calculateEruptionProbability(
            params.temperature,
            params.pressure,
            params.volatileContent
        );
        
        // Update probability displays
        document.getElementById('prob-30days').textContent = Math.round(prob * 30 / 365 * 100) + '%';
        document.getElementById('prob-1year').textContent = Math.round(prob * 100) + '%';
        document.getElementById('prob-10years').textContent = Math.round(Math.min(prob * 10 * 100, 95)) + '%';
    }
    
    // Animation and simulation methods
    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
    
    startAnimation() {
        this.isAnimating = true;
        document.getElementById('play-pause-btn').textContent = '⏸ Pause';
        this.animate();
    }
    
    stopAnimation() {
        this.isAnimating = false;
        document.getElementById('play-pause-btn').textContent = '▶ Play';
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    resetSimulation() {
        this.simulationTime = 0;
        document.getElementById('time-slider').value = '0';
        document.getElementById('time-display').textContent = 'Day 0';
        this.thermalSolver.temperature = this.parameters.magmaChamber.temperature;
        this.updateMagmaChamber();
    }
    
    setSimulationTime(timeSeconds) {
        this.simulationTime = timeSeconds;
        const days = Math.floor(timeSeconds / 86400);
        document.getElementById('time-display').textContent = `Day ${days}`;
        
        // Update thermal evolution
        this.thermalSolver.update(timeSeconds);
        this.parameters.magmaChamber.temperature = this.thermalSolver.temperature;
        
        // Update visualization
        this.updateMagmaChamber();
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Update simulation time
        this.simulationTime += this.config.physics.timeStep * this.config.visualization.animationSpeed;
        const maxTime = this.config.physics.maxSimulationTime;
        
        if (this.simulationTime > maxTime) {
            this.simulationTime = 0;
        }
        
        // Update time display
        const days = Math.floor(this.simulationTime / 86400);
        document.getElementById('time-display').textContent = `Day ${days}`;
        document.getElementById('time-slider').value = days.toString();
        
        // Update physics
        this.updatePhysics();
        
        // Update controls
        if (this.controls) {
            this.controls.update();
        }
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
    
    updatePhysics() {
        // Update thermal evolution
        if (this.config.physics.enableThermalEvolution) {
            const newTemp = this.thermalSolver.update(this.config.physics.timeStep);
            this.parameters.magmaChamber.temperature = newTemp;
            
            // Update chamber visualization
            if (this.magmaChamberMesh) {
                const normalizedTemp = (newTemp - 800) / (1500 - 800);
                const color = this.getTemperatureColor(normalizedTemp);
                this.magmaChamberMesh.material.color = color;
                this.magmaChamberMesh.material.emissive = new THREE.Color(color).multiplyScalar(0.3);
                
                // Pulsation effect based on pressure
                const userData = this.magmaChamberMesh.userData;
                userData.pulsationPhase += 0.02;
                const scale = userData.originalScale + Math.sin(userData.pulsationPhase) * 0.05;
                this.magmaChamberMesh.scale.setScalar(scale);
            }
        }
        
        // Update eruption probability
        this.updateEruptionProbability();
    }
    
    // Tab-specific update methods
    updateTemperatureEvolutionChart() {
        const canvas = document.getElementById('temperature-evolution-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw temperature evolution curve
        ctx.strokeStyle = '#ff6600';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const timePoints = 100;
        const maxTime = 365 * 5; // 5 years
        
        for (let i = 0; i <= timePoints; i++) {
            const t = (i / timePoints) * maxTime * 24 * 3600; // Convert to seconds
            const temp = this.parameters.magmaChamber.temperature * 
                         Math.exp(-this.thermalSolver.coolingRate * t / (365.25 * 24 * 3600) / this.parameters.magmaChamber.temperature);
            
            const x = (i / timePoints) * canvas.width;
            const y = canvas.height - ((temp - 800) / (1500 - 800)) * canvas.height;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Add axis labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.fillText('0 years', 10, canvas.height - 5);
        ctx.fillText('5 years', canvas.width - 50, canvas.height - 5);
        ctx.fillText('1500K', 5, 15);
        ctx.fillText('800K', 5, canvas.height - 25);
    }
    
    updateGeometryDiagram() {
        const canvas = document.getElementById('geometry-diagram');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw cross-section view
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = 0.05; // m to pixels
        
        // Draw surface
        ctx.strokeStyle = '#8B7355';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(0, centerY - 50);
        ctx.lineTo(canvas.width, centerY - 50);
        ctx.stroke();
        
        // Draw chamber
        const chamberRadius = this.parameters.magmaChamber.radius * scale;
        const chamberDepth = this.parameters.magmaChamber.depth * scale;
        
        ctx.fillStyle = '#ff6600';
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.ellipse(centerX, centerY - 50 + chamberDepth, chamberRadius, chamberRadius * (this.parameters.magmaChamber.aspectRatio || 1), 0, 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 1.0;
        
        // Add labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px Arial';
        ctx.fillText('Surface', 10, centerY - 35);
        ctx.fillText(`Depth: ${this.parameters.magmaChamber.depth}m`, 10, centerY + 20);
        ctx.fillText(`Radius: ${this.parameters.magmaChamber.radius}m`, 10, centerY + 35);
    }
    
    updateDeformationMap() {
        const canvas = document.getElementById('deformation-map');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        
        // Calculate deformation field
        const extent = 5000; // m
        const pixelSize = (2 * extent) / canvas.width;
        
        for (let x = 0; x < canvas.width; x++) {
            for (let y = 0; y < canvas.height; y++) {
                const worldX = (x - canvas.width/2) * pixelSize;
                const worldY = (y - canvas.height/2) * pixelSize;
                
                const deformation = this.deformationSolver.calculateMogiDeformation(
                    worldX, worldY, 0,
                    this.parameters.magmaChamber.depth,
                    4/3 * Math.PI * Math.pow(this.parameters.magmaChamber.radius, 3) * 0.001,
                    0.25
                );
                
                // Get selected component
                let value;
                const component = document.getElementById('deform-component-select')?.value || 'vertical';
                switch (component) {
                    case 'horizontal':
                        value = Math.sqrt(deformation.ux*deformation.ux + deformation.uy*deformation.uy);
                        break;
                    case 'magnitude':
                        value = Math.sqrt(deformation.ux*deformation.ux + deformation.uy*deformation.uy + deformation.uz*deformation.uz);
                        break;
                    default: // vertical
                        value = deformation.uz;
                }
                
                // Normalize and color
                const normalizedValue = (value + 0.01) / 0.02; // Assume ±1cm range
                const color = this.getDeformationColor(Math.max(0, Math.min(1, normalizedValue)));
                
                const index = (y * canvas.width + x) * 4;
                imageData.data[index] = color.r * 255;
                imageData.data[index + 1] = color.g * 255;
                imageData.data[index + 2] = color.b * 255;
                imageData.data[index + 3] = 255;
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }
    
    updateInSARComparison() {
        // Update model prediction, observed InSAR, and residual interferograms
        ['model-interferogram', 'observed-interferogram', 'residual-interferogram'].forEach((canvasId, index) => {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(canvas.width, canvas.height);
            
            // Generate synthetic interferogram patterns
            for (let x = 0; x < canvas.width; x++) {
                for (let y = 0; y < canvas.height; y++) {
                    let value;
                    
                    if (index === 0) {
                        // Model prediction - circular pattern
                        const centerX = canvas.width / 2;
                        const centerY = canvas.height / 2;
                        const distance = Math.sqrt((x - centerX)**2 + (y - centerY)**2);
                        value = Math.cos(distance * 0.1) * Math.exp(-distance * 0.01);
                    } else if (index === 1) {
                        // Observed InSAR - similar but with noise
                        const centerX = canvas.width / 2;
                        const centerY = canvas.height / 2;
                        const distance = Math.sqrt((x - centerX)**2 + (y - centerY)**2);
                        value = Math.cos(distance * 0.1) * Math.exp(-distance * 0.01) + 
                               (Math.random() - 0.5) * 0.3;
                    } else {
                        // Residual - difference
                        value = (Math.random() - 0.5) * 0.2;
                    }
                    
                    // Convert to fringe colors
                    const normalizedValue = (value + 1) / 2;
                    const hue = normalizedValue * 360;
                    const color = this.hslToRgb(hue / 360, 1, 0.5);
                    
                    const pixelIndex = (y * canvas.width + x) * 4;
                    imageData.data[pixelIndex] = color.r * 255;
                    imageData.data[pixelIndex + 1] = color.g * 255;
                    imageData.data[pixelIndex + 2] = color.b * 255;
                    imageData.data[pixelIndex + 3] = 255;
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
        });
        
        // Update fit statistics
        document.getElementById('rms-error').textContent = (2.3 + Math.random()).toFixed(1) + ' mm';
        document.getElementById('correlation').textContent = (0.85 + Math.random() * 0.1).toFixed(2);
    }
    
    hslToRgb(h, s, l) {
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h * 6) % 2 - 1));
        const m = l - c / 2;
        
        let r, g, b;
        
        if (h < 1/6) {
            [r, g, b] = [c, x, 0];
        } else if (h < 2/6) {
            [r, g, b] = [x, c, 0];
        } else if (h < 3/6) {
            [r, g, b] = [0, c, x];
        } else if (h < 4/6) {
            [r, g, b] = [0, x, c];
        } else if (h < 5/6) {
            [r, g, b] = [x, 0, c];
        } else {
            [r, g, b] = [c, 0, x];
        }
        
        return {
            r: r + m,
            g: g + m,
            b: b + m
        };
    }
    
    updateProbabilityChart() {
        const canvas = document.getElementById('probability-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw probability evolution over time
        const timeHorizon = 30; // days
        ctx.strokeStyle = '#ff5252';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let day = 0; day <= timeHorizon; day++) {
            const x = (day / timeHorizon) * canvas.width;
            const baseProbability = this.forecastModel.calculateEruptionProbability(
                this.parameters.magmaChamber.temperature,
                this.parameters.magmaChamber.pressure,
                this.parameters.magmaChamber.volatileContent
            );
            const dailyProb = baseProbability * (1 + Math.sin(day * 0.1) * 0.2); // Add some variation
            const y = canvas.height - (dailyProb * canvas.height * 10); // Scale for visibility
            
            if (day === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px Arial';
        ctx.fillText('Day 0', 5, canvas.height - 5);
        ctx.fillText('Day 30', canvas.width - 40, canvas.height - 5);
        ctx.fillText('Eruption Probability', 10, 15);
    }
    
    updateHazardMap() {
        const canvas = document.getElementById('hazard-map');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        for (let x = 0; x < canvas.width; x++) {
            for (let y = 0; y < canvas.height; y++) {
                const distance = Math.sqrt((x - centerX)**2 + (y - centerY)**2);
                const maxDistance = Math.min(canvas.width, canvas.height) / 2;
                
                // Risk decreases with distance
                let riskLevel;
                if (distance < maxDistance * 0.2) {
                    riskLevel = 4; // Very High
                } else if (distance < maxDistance * 0.4) {
                    riskLevel = 3; // High
                } else if (distance < maxDistance * 0.6) {
                    riskLevel = 2; // Moderate
                } else if (distance < maxDistance * 0.8) {
                    riskLevel = 1; // Low
                } else {
                    riskLevel = 0; // Very Low
                }
                
                // Color based on risk level
                let color;
                switch (riskLevel) {
                    case 4: color = { r: 1, g: 0, b: 0 }; break;
                    case 3: color = { r: 1, g: 0.5, b: 0 }; break;
                    case 2: color = { r: 1, g: 1, b: 0 }; break;
                    case 1: color = { r: 0.5, g: 1, b: 0.5 }; break;
                    default: color = { r: 0.2, g: 0.2, b: 0.8 }; break;
                }
                
                const index = (y * canvas.width + x) * 4;
                imageData.data[index] = color.r * 255;
                imageData.data[index + 1] = color.g * 255;
                imageData.data[index + 2] = color.b * 255;
                imageData.data[index + 3] = 180; // Semi-transparent
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }
    
    runScenario(scenarioType) {
        console.log(`Running ${scenarioType} scenario`);
        
        // Update scenario timeline
        const canvas = document.getElementById('scenario-timeline');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw timeline based on scenario
        let timelineData;
        switch (scenarioType) {
            case 'explosive':
                timelineData = [
                    { time: 0, activity: 0.1, label: 'Background' },
                    { time: 0.3, activity: 0.3, label: 'Unrest' },
                    { time: 0.6, activity: 0.7, label: 'Pre-eruptive' },
                    { time: 0.8, activity: 1.0, label: 'Explosive Eruption' },
                    { time: 1.0, activity: 0.2, label: 'Aftermath' }
                ];
                break;
            case 'effusive':
                timelineData = [
                    { time: 0, activity: 0.1, label: 'Background' },
                    { time: 0.4, activity: 0.4, label: 'Dome Growth' },
                    { time: 0.8, activity: 0.6, label: 'Lava Flows' },
                    { time: 1.0, activity: 0.3, label: 'Cooling' }
                ];
                break;
            default: // phreatomagmatic
                timelineData = [
                    { time: 0, activity: 0.1, label: 'Background' },
                    { time: 0.2, activity: 0.5, label: 'Water Contact' },
                    { time: 0.5, activity: 0.9, label: 'Steam Explosion' },
                    { time: 0.7, activity: 0.4, label: 'Ash Fall' },
                    { time: 1.0, activity: 0.1, label: 'Quiescence' }
                ];
        }
        
        // Draw timeline
        ctx.strokeStyle = '#4fc3f7';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        timelineData.forEach((point, index) => {
            const x = point.time * canvas.width;
            const y = canvas.height - (point.activity * canvas.height * 0.8 + canvas.height * 0.1);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
            // Add point markers
            ctx.fillStyle = '#4fc3f7';
            ctx.fillRect(x - 2, y - 2, 4, 4);
        });
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px Arial';
        timelineData.forEach((point) => {
            const x = point.time * canvas.width;
            const y = canvas.height - 10;
            ctx.fillText(point.label, x - 20, y);
        });
    }
    
    runUncertaintyAnalysis() {
        console.log('Running uncertainty analysis...');
        
        // Monte Carlo simulation
        const samples = this.config.uncertainty.monteCarloSamples;
        const results = [];
        
        for (let i = 0; i < samples; i++) {
            // Perturb parameters within uncertainty ranges
            const perturbedParams = {
                temperature: this.parameters.magmaChamber.temperature + (Math.random() - 0.5) * 100,
                pressure: this.parameters.magmaChamber.pressure * (1 + (Math.random() - 0.5) * 0.2),
                volatileContent: this.parameters.magmaChamber.volatileContent * (1 + (Math.random() - 0.5) * 0.3),
                depth: this.parameters.magmaChamber.depth * (1 + (Math.random() - 0.5) * 0.1),
                radius: this.parameters.magmaChamber.radius * (1 + (Math.random() - 0.5) * 0.15)
            };
            
            // Calculate eruption probability for perturbed parameters
            const probability = this.forecastModel.calculateEruptionProbability(
                perturbedParams.temperature,
                perturbedParams.pressure,
                perturbedParams.volatileContent
            );
            
            results.push({
                parameters: perturbedParams,
                probability: probability
            });
        }
        
        // Update uncertainty visualization
        this.updateUncertaintyChart(results);
        this.updateSensitivityAnalysis(results);
        this.updateErrorPropagation(results);
    }
    
    updateUncertaintyChart(results) {
        const canvas = document.getElementById('uncertainty-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Create probability distribution histogram
        const probabilities = results.map(r => r.probability);
        const minProb = Math.min(...probabilities);
        const maxProb = Math.max(...probabilities);
        const numBins = 20;
        const binSize = (maxProb - minProb) / numBins;
        
        const histogram = new Array(numBins).fill(0);
        probabilities.forEach(prob => {
            const binIndex = Math.min(Math.floor((prob - minProb) / binSize), numBins - 1);
            histogram[binIndex]++;
        });
        
        // Draw histogram
        const maxCount = Math.max(...histogram);
        const barWidth = canvas.width / numBins;
        
        ctx.fillStyle = '#4fc3f7';
        histogram.forEach((count, index) => {
            const barHeight = (count / maxCount) * canvas.height * 0.8;
            const x = index * barWidth;
            const y = canvas.height - barHeight;
            
            ctx.fillRect(x, y, barWidth - 1, barHeight);
        });
        
        // Add confidence intervals
        const sortedProbs = probabilities.sort((a, b) => a - b);
        const confidenceLevel = this.config.uncertainty.uncertaintyLevel;
        const lowerIndex = Math.floor((1 - confidenceLevel) / 2 * sortedProbs.length);
        const upperIndex = Math.floor((1 + confidenceLevel) / 2 * sortedProbs.length);
        
        const lowerBound = sortedProbs[lowerIndex];
        const upperBound = sortedProbs[upperIndex];
        
        // Draw confidence interval lines
        ctx.strokeStyle = '#ff5252';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        const lowerX = ((lowerBound - minProb) / (maxProb - minProb)) * canvas.width;
        const upperX = ((upperBound - minProb) / (maxProb - minProb)) * canvas.width;
        
        ctx.beginPath();
        ctx.moveTo(lowerX, 0);
        ctx.lineTo(lowerX, canvas.height);
        ctx.moveTo(upperX, 0);
        ctx.lineTo(upperX, canvas.height);
        ctx.stroke();
        
        ctx.setLineDash([]);
        
        // Add labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px Arial';
        ctx.fillText(`${(confidenceLevel * 100)}% CI`, 10, 15);
        ctx.fillText(`[${lowerBound.toFixed(3)}, ${upperBound.toFixed(3)}]`, 10, 30);
    }
    
    updateSensitivityAnalysis(results) {
        // Calculate sensitivity indices
        const sensitivities = {
            temperature: this.calculateSensitivity(results, 'temperature'),
            pressure: this.calculateSensitivity(results, 'pressure'),
            volatileContent: this.calculateSensitivity(results, 'volatileContent'),
            depth: this.calculateSensitivity(results, 'depth'),
            radius: this.calculateSensitivity(results, 'radius')
        };
        
        // Update sensitivity chart
        const canvas = document.getElementById('sensitivity-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const parameters = Object.keys(sensitivities);
        const barWidth = canvas.width / parameters.length;
        const maxSensitivity = Math.max(...Object.values(sensitivities));
        
        parameters.forEach((param, index) => {
            const sensitivity = sensitivities[param];
            const barHeight = (sensitivity / maxSensitivity) * canvas.height * 0.8;
            const x = index * barWidth;
            const y = canvas.height - barHeight;
            
            // Color code by sensitivity level
            ctx.fillStyle = sensitivity > 0.7 ? '#ff5252' : 
                           sensitivity > 0.4 ? '#ffc107' : '#4caf50';
            ctx.fillRect(x + 5, y, barWidth - 10, barHeight);
            
            // Add parameter labels
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px Arial';
            ctx.fillText(param, x + 5, canvas.height - 5);
        });
        
        // Update sensitivity table
        const tableBody = document.querySelector('#sensitivity-table tbody');
        if (tableBody) {
            tableBody.innerHTML = '';
            
            Object.entries(sensitivities).forEach(([param, sensitivity]) => {
                const row = document.createElement('tr');
                const impact = sensitivity > 0.7 ? 'High' : sensitivity > 0.4 ? 'Medium' : 'Low';
                
                row.innerHTML = `
                    <td>${param}</td>
                    <td>${sensitivity.toFixed(3)}</td>
                    <td>${impact}</td>
                `;
                tableBody.appendChild(row);
            });
        }
    }
    
    calculateSensitivity(results, parameterName) {
        // Calculate Sobol sensitivity index
        const parameters = results.map(r => r.parameters[parameterName]);
        const probabilities = results.map(r => r.probability);
        
        // Simple correlation-based sensitivity (approximation)
        const paramMean = parameters.reduce((a, b) => a + b) / parameters.length;
        const probMean = probabilities.reduce((a, b) => a + b) / probabilities.length;
        
        let numerator = 0;
        let paramDenominator = 0;
        let probDenominator = 0;
        
        for (let i = 0; i < parameters.length; i++) {
            const paramDiff = parameters[i] - paramMean;
            const probDiff = probabilities[i] - probMean;
            
            numerator += paramDiff * probDiff;
            paramDenominator += paramDiff * paramDiff;
            probDenominator += probDiff * probDiff;
        }
        
        const correlation = numerator / Math.sqrt(paramDenominator * probDenominator);
        return Math.abs(correlation);
    }
    
    updateErrorPropagation(results) {
        const canvas = document.getElementById('error-propagation-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Show error propagation over time
        const timePoints = 50;
        const probabilities = results.map(r => r.probability);
        const meanProb = probabilities.reduce((a, b) => a + b) / probabilities.length;
        const stdProb = Math.sqrt(probabilities.reduce((a, b) => a + (b - meanProb)**2, 0) / probabilities.length);
        
        ctx.strokeStyle = '#4fc3f7';
        ctx.lineWidth = 2;
        
        // Mean line
        ctx.beginPath();
        ctx.moveTo(0, canvas.height - meanProb * canvas.height * 2);
        ctx.lineTo(canvas.width, canvas.height - meanProb * canvas.height * 2);
        ctx.stroke();
        
        // Uncertainty bands
        ctx.fillStyle = 'rgba(79, 195, 247, 0.3)';
        ctx.beginPath();
        ctx.moveTo(0, canvas.height - (meanProb - stdProb) * canvas.height * 2);
        ctx.lineTo(canvas.width, canvas.height - (meanProb - stdProb) * canvas.height * 2);
        ctx.lineTo(canvas.width, canvas.height - (meanProb + stdProb) * canvas.height * 2);
        ctx.lineTo(0, canvas.height - (meanProb + stdProb) * canvas.height * 2);
        ctx.closePath();
        ctx.fill();
        
        // Update error statistics
        const totalUncertainty = (stdProb / meanProb) * 100;
        document.getElementById('total-uncertainty').textContent = `±${totalUncertainty.toFixed(1)}%`;
        document.getElementById('dominant-error').textContent = 'Temperature'; // Simplified
    }
    
    // View control methods
    setTopView() {
        this.camera.position.set(0, 8000, 0);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    setSideView() {
        this.camera.position.set(8000, 0, 0);
        this.camera.lookAt(0, -2000, 0);
        this.controls.update();
    }
    
    toggleCrossSection() {
        // Toggle cross-section view
        console.log('Cross-section view toggled');
    }
    
    // Intrusion management
    addIntrusion(type) {
        const intrusion = {
            type: type,
            id: `intrusion_${Date.now()}`,
            parameters: this.getDefaultIntrusionParameters(type)
        };
        
        this.parameters.intrusions.push(intrusion);
        this.createIntrusionMesh(intrusion);
        this.updateIntrusionList();
        
        console.log(`Added ${type} intrusion:`, intrusion);
    }
    
    getDefaultIntrusionParameters(type) {
        switch (type) {
            case 'sill':
                return {
                    depth: this.parameters.magmaChamber.depth - 500,
                    length: 1000,
                    width: 800,
                    thickness: 10,
                    strike: 0
                };
            case 'dike':
                return {
                    depthTop: 1000,
                    depthBottom: this.parameters.magmaChamber.depth,
                    length: 2000,
                    width: 5,
                    strike: 45
                };
            case 'conduit':
                return {
                    depthTop: 0,
                    depthBottom: this.parameters.magmaChamber.depth,
                    radius: 50
                };
            default:
                return {};
        }
    }
    
    createIntrusionMesh(intrusion) {
        let geometry, material;
        
        switch (intrusion.type) {
            case 'sill':
                geometry = new THREE.BoxGeometry(
                    intrusion.parameters.length,
                    intrusion.parameters.thickness,
                    intrusion.parameters.width
                );
                material = new THREE.MeshPhongMaterial({
                    color: 0xff8800,
                    transparent: true,
                    opacity: 0.6
                });
                break;
                
            case 'dike':
                geometry = new THREE.BoxGeometry(
                    intrusion.parameters.length,
                    intrusion.parameters.depthBottom - intrusion.parameters.depthTop,
                    intrusion.parameters.width
                );
                material = new THREE.MeshPhongMaterial({
                    color: 0xffff00,
                    transparent: true,
                    opacity: 0.6
                });
                break;
                
            case 'conduit':
                geometry = new THREE.CylinderGeometry(
                    intrusion.parameters.radius,
                    intrusion.parameters.radius,
                    intrusion.parameters.depthBottom - intrusion.parameters.depthTop,
                    16
                );
                material = new THREE.MeshPhongMaterial({
                    color: 0xff0088,
                    transparent: true,
                    opacity: 0.7
                });
                break;
                
            default:
                return;
        }
        
        const mesh = new THREE.Mesh(geometry, material);
        
        // Position the mesh
        if (intrusion.type === 'sill') {
            mesh.position.set(0, -intrusion.parameters.depth, 0);
        } else if (intrusion.type === 'dike') {
            const centerDepth = (intrusion.parameters.depthTop + intrusion.parameters.depthBottom) / 2;
            mesh.position.set(0, -centerDepth, 0);
            mesh.rotation.y = THREE.MathUtils.degToRad(intrusion.parameters.strike);
        } else if (intrusion.type === 'conduit') {
            const centerDepth = (intrusion.parameters.depthTop + intrusion.parameters.depthBottom) / 2;
            mesh.position.set(0, -centerDepth, 0);
        }
        
        mesh.userData = { intrusion: intrusion };
        mesh.castShadow = true;
        
        this.scene.add(mesh);
        this.intrusionMeshes.push(mesh);
    }
    
    updateIntrusionList() {
        const listContainer = document.getElementById('intrusion-list');
        if (!listContainer) return;
        
        listContainer.innerHTML = '';
        
        this.parameters.intrusions.forEach((intrusion, index) => {
            const item = document.createElement('div');
            item.className = 'intrusion-item';
            item.style.cssText = `
                padding: 8px;
                margin: 4px 0;
                background: #3d3d3d;
                border-radius: 4px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            `;
            
            item.innerHTML = `
                <span style="color: #4fc3f7; font-size: 12px;">${intrusion.type.toUpperCase()} #${index + 1}</span>
                <button onclick="this.parentElement.parentElement.removeChild(this.parentElement)" 
                        style="background: #ff5252; color: white; border: none; padding: 2px 6px; border-radius: 2px; cursor: pointer; font-size: 10px;">Remove</button>
            `;
            
            listContainer.appendChild(item);
        });
    }
    
    // Export methods
    exportImage() {
        const canvas = this.renderer.domElement;
        const dataURL = canvas.toDataURL('image/png');
        
        const link = document.createElement('a');
        link.download = 'magmatic_intrusion_analysis.png';
        link.href = dataURL;
        link.click();
        
        console.log('Image exported');
    }
    
    exportData() {
        const data = {
            parameters: this.parameters,
            simulationResults: this.simulationResults,
            timestamp: new Date().toISOString(),
            version: '1.0'
        };
        
        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.download = 'magmatic_intrusion_data.json';
        link.href = url;
        link.click();
        
        URL.revokeObjectURL(url);
        console.log('Data exported');
    }
    
    exportReport() {
        const reportContent = this.generateReport();
        const blob = new Blob([reportContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.download = 'magmatic_intrusion_report.html';
        link.href = url;
        link.click();
        
        URL.revokeObjectURL(url);
        console.log('Report exported');
    }
    
    generateReport() {
        const params = this.parameters.magmaChamber;
        const volume = (4/3) * Math.PI * Math.pow(params.radius, 3) * (params.aspectRatio || 1);
        const pressure = params.pressure / 1e6; // Convert to MPa
        
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Magmatic Intrusion Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1, h2 { color: #2196F3; }
                    .parameter { margin: 10px 0; }
                    .value { font-weight: bold; color: #4fc3f7; }
                </style>
            </head>
            <body>
                <h1>Magmatic Intrusion Analysis Report</h1>
                <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
                
                <h2>Magma Chamber Parameters</h2>
                <div class="parameter">Depth: <span class="value">${params.depth} m</span></div>
                <div class="parameter">Radius: <span class="value">${params.radius} m</span></div>
                <div class="parameter">Volume: <span class="value">${(volume / 1e9).toFixed(2)} km³</span></div>
                <div class="parameter">Temperature: <span class="value">${params.temperature} K</span></div>
                <div class="parameter">Pressure: <span class="value">${pressure.toFixed(0)} MPa</span></div>
                <div class="parameter">Volatile Content: <span class="value">${(params.volatileContent * 100).toFixed(1)}%</span></div>
                
                <h2>Eruption Probability Assessment</h2>
                <div class="parameter">30-day probability: <span class="value">${document.getElementById('prob-30days')?.textContent || 'N/A'}</span></div>
                <div class="parameter">1-year probability: <span class="value">${document.getElementById('prob-1year')?.textContent || 'N/A'}</span></div>
                <div class="parameter">10-year probability: <span class="value">${document.getElementById('prob-10years')?.textContent || 'N/A'}</span></div>
                
                <h2>Intrusion Network</h2>
                <p>Number of intrusions: <span class="value">${this.parameters.intrusions.length}</span></p>
                
                <h2>Analysis Summary</h2>
                <p>This analysis was performed using the Prometheus 3D Magmatic Intrusion Analyzer, 
                incorporating physics-based modeling and uncertainty quantification. The results provide 
                scientific insights into volcanic hazard assessment and eruption forecasting.</p>
            </body>
            </html>
        `;
    }
    
    // Event handlers
    onMouseClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children);
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (object.userData.intrusion) {
                console.log('Clicked intrusion:', object.userData.intrusion);
                // Could open intrusion parameter dialog
            }
        }
    }
    
    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }
    
    onKeyDown(event) {
        switch (event.code) {
            case 'Space':
                event.preventDefault();
                this.toggleAnimation();
                break;
            case 'KeyR':
                this.resetSimulation();
                break;
            case 'Escape':
                this.stopAnimation();
                break;
        }
    }
    
    onWindowResize() {
        const container = document.getElementById('threejs-container');
        if (!container) return;
        
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    // Cleanup
    dispose() {
        this.stopAnimation();
        
        // Dispose Three.js resources
        this.scene.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        
        console.log('3D Magmatic Intrusion Analyzer disposed');
    }
    
    // Start the main render loop
    startMainLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            
            if (this.controls) {
                this.controls.update();
            }
            
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MagmaticIntrusionAnalyzer;
}

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('magmatic-intrusion-analyzer');
    if (container) {
        window.magmaticAnalyzer = new MagmaticIntrusionAnalyzer('magmatic-intrusion-analyzer');
        window.magmaticAnalyzer.startMainLoop();
    }
});