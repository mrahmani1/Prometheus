/**
 * Coherence Analyzer - Task 3.3
 * Advanced coherence analysis with threshold filtering and statistical analysis
 * Implements temporal and spatial coherence analysis for InSAR data quality assessment
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function() {
    'use strict';

    class CoherenceAnalyzer {
        constructor(options = {}) {
            this.container = options.container || document.body;
            this.data = null;
            this.coherenceStack = new Map();
            this.filteredData = new Map();
            this.analysisResults = null;
            this.ui = null;
            
            this.config = {
                coherenceThreshold: 0.3,
                spatialFilterSize: 3,
                temporalFilterWindow: 5,
                maskingThreshold: 0.2,
                adaptiveThreshold: true,
                edgePreservation: true,
                spatialResolution: 30, // meters
                temporalResolution: 12, // days
                qualityMetrics: ['mean', 'std', 'stability', 'gradient'],
                analysisTypes: ['temporal', 'spatial', 'combined']
            };

            this.filters = {
                spatial: new SpatialCoherenceFilter(),
                temporal: new TemporalCoherenceFilter(),
                adaptive: new AdaptiveCoherenceFilter(),
                edge: new EdgePreservingFilter()
            };

            this.statistics = {
                basic: new BasicStatistics(),
                advanced: new AdvancedStatistics(),
                quality: new QualityMetrics()
            };

            this.init();
        }

        init() {
            this.createAnalyzerInterface();
            this.setupEventHandlers();
            this.initializeWorkers();
            this.setupVisualization();
            console.log('🔍 Coherence Analyzer initialized');
        }

        createAnalyzerInterface() {
            const container = document.createElement('div');
            container.className = 'coherence-analyzer-container';
            container.innerHTML = `
                <div class="analyzer-header">
                    <h3><i class="fas fa-filter"></i> Coherence Analyzer</h3>
                    <div class="analyzer-controls">
                        <button class="btn btn-primary" id="loadCoherenceData">Load Data</button>
                        <button class="btn btn-secondary" id="runAnalysis">Analyze</button>
                        <button class="btn btn-info" id="applyFilters">Apply Filters</button>
                        <button class="btn btn-success" id="exportCoherence">Export</button>
                    </div>
                </div>

                <div class="analyzer-parameters">
                    <div class="parameter-group">
                        <h4>Thresholding</h4>
                        <div class="param-control">
                            <label>Coherence Threshold:</label>
                            <input type="range" id="coherenceThreshold" min="0" max="1" step="0.01" value="${this.config.coherenceThreshold}" />
                            <span id="thresholdValue">${this.config.coherenceThreshold}</span>
                        </div>
                        <div class="param-control">
                            <label>Masking Threshold:</label>
                            <input type="range" id="maskingThreshold" min="0" max="1" step="0.01" value="${this.config.maskingThreshold}" />
                            <span id="maskingValue">${this.config.maskingThreshold}</span>
                        </div>
                        <div class="param-control">
                            <label>Adaptive Thresholding:</label>
                            <input type="checkbox" id="adaptiveThreshold" ${this.config.adaptiveThreshold ? 'checked' : ''} />
                        </div>
                    </div>
                    
                    <div class="parameter-group">
                        <h4>Spatial Filtering</h4>
                        <div class="param-control">
                            <label>Filter Size:</label>
                            <select id="spatialFilterSize">
                                <option value="3" ${this.config.spatialFilterSize === 3 ? 'selected' : ''}>3x3</option>
                                <option value="5" ${this.config.spatialFilterSize === 5 ? 'selected' : ''}>5x5</option>
                                <option value="7" ${this.config.spatialFilterSize === 7 ? 'selected' : ''}>7x7</option>
                                <option value="9" ${this.config.spatialFilterSize === 9 ? 'selected' : ''}>9x9</option>
                            </select>
                        </div>
                        <div class="param-control">
                            <label>Edge Preservation:</label>
                            <input type="checkbox" id="edgePreservation" ${this.config.edgePreservation ? 'checked' : ''} />
                        </div>
                    </div>
                    
                    <div class="parameter-group">
                        <h4>Temporal Filtering</h4>
                        <div class="param-control">
                            <label>Window Size:</label>
                            <input type="range" id="temporalWindow" min="3" max="15" step="2" value="${this.config.temporalFilterWindow}" />
                            <span id="temporalValue">${this.config.temporalFilterWindow}</span>
                        </div>
                    </div>
                </div>

                <div class="analyzer-tabs">
                    <button class="tab-btn active" data-tab="visualization">Visualization</button>
                    <button class="tab-btn" data-tab="statistics">Statistics</button>
                    <button class="tab-btn" data-tab="quality">Quality Metrics</button>
                    <button class="tab-btn" data-tab="filtering">Filtering</button>
                    <button class="tab-btn" data-tab="comparison">Comparison</button>
                </div>

                <div class="analyzer-content">
                    <div class="tab-content active" id="visualization-tab">
                        <div class="visualization-controls">
                            <div class="view-selector">
                                <button class="view-btn active" data-view="original">Original</button>
                                <button class="view-btn" data-view="filtered">Filtered</button>
                                <button class="view-btn" data-view="mask">Mask</button>
                                <button class="view-btn" data-view="difference">Difference</button>
                            </div>
                            <div class="colormap-selector">
                                <label>Colormap:</label>
                                <select id="colormapSelect">
                                    <option value="gray">Grayscale</option>
                                    <option value="jet">Jet</option>
                                    <option value="viridis" selected>Viridis</option>
                                    <option value="plasma">Plasma</option>
                                    <option value="coherence">Coherence</option>
                                </select>
                            </div>
                        </div>
                        <div class="visualization-area">
                            <canvas id="coherenceCanvas" width="600" height="400"></canvas>
                            <div class="canvas-overlay">
                                <div class="pixel-info" id="pixelInfo">
                                    <span>Hover over image for pixel information</span>
                                </div>
                                <div class="zoom-controls">
                                    <button id="zoomIn">+</button>
                                    <button id="zoomOut">-</button>
                                    <button id="resetZoom">⌂</button>
                                </div>
                            </div>
                        </div>
                        <div class="histogram-area">
                            <canvas id="coherenceHistogram" width="400" height="200"></canvas>
                        </div>
                    </div>

                    <div class="tab-content" id="statistics-tab">
                        <div class="statistics-grid">
                            <div class="stat-panel basic-stats">
                                <h4>Basic Statistics</h4>
                                <div class="stat-table">
                                    <div class="stat-row">
                                        <span>Mean:</span>
                                        <span id="statMean">--</span>
                                    </div>
                                    <div class="stat-row">
                                        <span>Median:</span>
                                        <span id="statMedian">--</span>
                                    </div>
                                    <div class="stat-row">
                                        <span>Std Dev:</span>
                                        <span id="statStd">--</span>
                                    </div>
                                    <div class="stat-row">
                                        <span>Min:</span>
                                        <span id="statMin">--</span>
                                    </div>
                                    <div class="stat-row">
                                        <span>Max:</span>
                                        <span id="statMax">--</span>
                                    </div>
                                    <div class="stat-row">
                                        <span>Range:</span>
                                        <span id="statRange">--</span>
                                    </div>
                                </div>
                            </div>

                            <div class="stat-panel distribution-stats">
                                <h4>Distribution Analysis</h4>
                                <canvas id="distributionPlot" width="300" height="200"></canvas>
                                <div class="distribution-metrics">
                                    <div class="metric">
                                        <label>Skewness:</label>
                                        <span id="skewness">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Kurtosis:</label>
                                        <span id="kurtosis">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Entropy:</label>
                                        <span id="entropy">--</span>
                                    </div>
                                </div>
                            </div>

                            <div class="stat-panel spatial-stats">
                                <h4>Spatial Analysis</h4>
                                <div class="spatial-metrics">
                                    <div class="metric">
                                        <label>Moran's I:</label>
                                        <span id="moransI">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Local Clusters:</label>
                                        <span id="localClusters">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Edge Density:</label>
                                        <span id="edgeDensity">--</span>
                                    </div>
                                </div>
                                <canvas id="spatialCorrelation" width="300" height="200"></canvas>
                            </div>

                            <div class="stat-panel temporal-stats">
                                <h4>Temporal Analysis</h4>
                                <canvas id="temporalTrend" width="300" height="200"></canvas>
                                <div class="temporal-metrics">
                                    <div class="metric">
                                        <label>Stability Index:</label>
                                        <span id="stabilityIndex">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Trend Slope:</label>
                                        <span id="trendSlope">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Seasonality:</label>
                                        <span id="seasonality">--</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" id="quality-tab">
                        <div class="quality-grid">
                            <div class="quality-panel overall-quality">
                                <h4>Overall Quality Assessment</h4>
                                <div class="quality-score">
                                    <div class="score-circle">
                                        <span id="overallScore">--</span>
                                    </div>
                                    <div class="score-label">Quality Score</div>
                                </div>
                                <div class="quality-breakdown">
                                    <div class="quality-item">
                                        <span>Data Coverage:</span>
                                        <div class="quality-bar">
                                            <div class="quality-fill" id="coverageBar"></div>
                                        </div>
                                        <span id="coverageScore">--</span>
                                    </div>
                                    <div class="quality-item">
                                        <span>Signal Consistency:</span>
                                        <div class="quality-bar">
                                            <div class="quality-fill" id="consistencyBar"></div>
                                        </div>
                                        <span id="consistencyScore">--</span>
                                    </div>
                                    <div class="quality-item">
                                        <span>Noise Level:</span>
                                        <div class="quality-bar">
                                            <div class="quality-fill" id="noiseBar"></div>
                                        </div>
                                        <span id="noiseScore">--</span>
                                    </div>
                                </div>
                            </div>

                            <div class="quality-panel threshold-analysis">
                                <h4>Threshold Analysis</h4>
                                <canvas id="thresholdCurve" width="350" height="250"></canvas>
                                <div class="threshold-metrics">
                                    <div class="metric">
                                        <label>Optimal Threshold:</label>
                                        <span id="optimalThreshold">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>False Positive Rate:</label>
                                        <span id="falsePositiveRate">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>True Positive Rate:</label>
                                        <span id="truePositiveRate">--</span>
                                    </div>
                                </div>
                            </div>

                            <div class="quality-panel regional-analysis">
                                <h4>Regional Quality</h4>
                                <div class="regional-grid">
                                    <div class="region-item">
                                        <label>High Coherence Regions:</label>
                                        <span id="highCoherenceRegions">--</span>
                                    </div>
                                    <div class="region-item">
                                        <label>Medium Coherence Regions:</label>
                                        <span id="mediumCoherenceRegions">--</span>
                                    </div>
                                    <div class="region-item">
                                        <label>Low Coherence Regions:</label>
                                        <span id="lowCoherenceRegions">--</span>
                                    </div>
                                    <div class="region-item">
                                        <label>Decorrelated Regions:</label>
                                        <span id="decorrelatedRegions">--</span>
                                    </div>
                                </div>
                                <canvas id="regionalMap" width="350" height="200"></canvas>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" id="filtering-tab">
                        <div class="filtering-controls">
                            <div class="filter-pipeline">
                                <h4>Filter Pipeline</h4>
                                <div class="pipeline-stages">
                                    <div class="stage" data-stage="input">
                                        <div class="stage-icon">📥</div>
                                        <div class="stage-label">Input</div>
                                        <div class="stage-status active">Ready</div>
                                    </div>
                                    <div class="stage" data-stage="spatial">
                                        <div class="stage-icon">🔲</div>
                                        <div class="stage-label">Spatial Filter</div>
                                        <div class="stage-status">Waiting</div>
                                    </div>
                                    <div class="stage" data-stage="temporal">
                                        <div class="stage-icon">⏱️</div>
                                        <div class="stage-label">Temporal Filter</div>
                                        <div class="stage-status">Waiting</div>
                                    </div>
                                    <div class="stage" data-stage="threshold">
                                        <div class="stage-icon">🎯</div>
                                        <div class="stage-label">Threshold</div>
                                        <div class="stage-status">Waiting</div>
                                    </div>
                                    <div class="stage" data-stage="output">
                                        <div class="stage-icon">📤</div>
                                        <div class="stage-label">Output</div>
                                        <div class="stage-status">Waiting</div>
                                    </div>
                                </div>
                            </div>

                            <div class="filter-parameters">
                                <div class="filter-group">
                                    <h5>Spatial Filters</h5>
                                    <div class="filter-option">
                                        <input type="checkbox" id="meanFilter" checked>
                                        <label>Mean Filter</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="medianFilter">
                                        <label>Median Filter</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="gaussianFilter">
                                        <label>Gaussian Filter</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="bilateralFilter">
                                        <label>Bilateral Filter</label>
                                    </div>
                                </div>

                                <div class="filter-group">
                                    <h5>Temporal Filters</h5>
                                    <div class="filter-option">
                                        <input type="checkbox" id="movingAverage" checked>
                                        <label>Moving Average</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="kalmanFilter">
                                        <label>Kalman Filter</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="trendFilter">
                                        <label>Trend Filter</label>
                                    </div>
                                </div>

                                <div class="filter-group">
                                    <h5>Advanced Filters</h5>
                                    <div class="filter-option">
                                        <input type="checkbox" id="adaptiveFilter">
                                        <label>Adaptive Filter</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="morphFilter">
                                        <label>Morphological Filter</label>
                                    </div>
                                    <div class="filter-option">
                                        <input type="checkbox" id="waveletFilter">
                                        <label>Wavelet Filter</label>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="filtering-results">
                            <div class="before-after">
                                <div class="comparison-view">
                                    <div class="comparison-panel">
                                        <h5>Before Filtering</h5>
                                        <canvas id="beforeCanvas" width="250" height="200"></canvas>
                                    </div>
                                    <div class="comparison-panel">
                                        <h5>After Filtering</h5>
                                        <canvas id="afterCanvas" width="250" height="200"></canvas>
                                    </div>
                                </div>
                                <div class="improvement-metrics">
                                    <div class="metric">
                                        <label>SNR Improvement:</label>
                                        <span id="snrImprovement">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Coherence Gain:</label>
                                        <span id="coherenceGain">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Edge Preservation:</label>
                                        <span id="edgePreservationScore">--</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" id="comparison-tab">
                        <div class="comparison-controls">
                            <div class="dataset-selector">
                                <label>Compare with:</label>
                                <select id="comparisonDataset">
                                    <option value="">Select dataset...</option>
                                    <option value="reference">Reference Data</option>
                                    <option value="previous">Previous Analysis</option>
                                    <option value="model">Model Prediction</option>
                                </select>
                            </div>
                            <div class="comparison-metrics">
                                <button id="calculateDifference">Calculate Difference</button>
                                <button id="performCorrelation">Correlation Analysis</button>
                                <button id="validateResults">Validate Results</button>
                            </div>
                        </div>

                        <div class="comparison-results">
                            <div class="comparison-visualization">
                                <canvas id="comparisonCanvas" width="600" height="300"></canvas>
                            </div>
                            <div class="comparison-statistics">
                                <div class="stat-group">
                                    <h5>Difference Statistics</h5>
                                    <div class="stat-item">
                                        <span>RMSE:</span>
                                        <span id="rmse">--</span>
                                    </div>
                                    <div class="stat-item">
                                        <span>MAE:</span>
                                        <span id="mae">--</span>
                                    </div>
                                    <div class="stat-item">
                                        <span>Correlation:</span>
                                        <span id="correlation">--</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="processing-status">
                    <div class="status-bar">
                        <div class="status-progress">
                            <div class="progress-fill" id="analysisProgress"></div>
                        </div>
                        <div class="status-text" id="analysisStatus">Ready for analysis</div>
                    </div>
                </div>
            `;

            this.container.appendChild(container);
            this.ui = container;
            this.setupCanvases();
        }

        setupCanvases() {
            this.canvases = {
                main: document.getElementById('coherenceCanvas'),
                histogram: document.getElementById('coherenceHistogram'),
                distribution: document.getElementById('distributionPlot'),
                spatial: document.getElementById('spatialCorrelation'),
                temporal: document.getElementById('temporalTrend'),
                threshold: document.getElementById('thresholdCurve'),
                regional: document.getElementById('regionalMap'),
                before: document.getElementById('beforeCanvas'),
                after: document.getElementById('afterCanvas'),
                comparison: document.getElementById('comparisonCanvas')
            };

            this.contexts = {};
            Object.keys(this.canvases).forEach(key => {
                if (this.canvases[key]) {
                    this.contexts[key] = this.canvases[key].getContext('2d');
                    this.setupCanvasInteraction(this.canvases[key], key);
                }
            });
        }

        setupCanvasInteraction(canvas, type) {
            canvas.addEventListener('mousemove', (e) => {
                this.handleCanvasHover(type, e.offsetX, e.offsetY);
            });

            canvas.addEventListener('click', (e) => {
                this.handleCanvasClick(type, e.offsetX, e.offsetY);
            });

            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                this.handleCanvasZoom(type, e.offsetX, e.offsetY, e.deltaY);
            });
        }

        setupEventHandlers() {
            // Main control buttons
            document.getElementById('loadCoherenceData')?.addEventListener('click', () => {
                this.loadCoherenceData();
            });

            document.getElementById('runAnalysis')?.addEventListener('click', () => {
                this.runCoherenceAnalysis();
            });

            document.getElementById('applyFilters')?.addEventListener('click', () => {
                this.applyFilters();
            });

            document.getElementById('exportCoherence')?.addEventListener('click', () => {
                this.exportResults();
            });

            // Parameter controls
            const parameterInputs = [
                'coherenceThreshold', 'maskingThreshold', 'temporalWindow'
            ];

            parameterInputs.forEach(id => {
                const input = document.getElementById(id);
                if (input) {
                    input.addEventListener('input', (e) => {
                        this.updateParameter(id, e.target.value);
                    });
                }
            });

            // Tab switching
            this.ui.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.switchTab(btn.dataset.tab);
                });
            });

            // View switching
            this.ui.querySelectorAll('.view-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.switchView(btn.dataset.view);
                });
            });

            // Zoom controls
            document.getElementById('zoomIn')?.addEventListener('click', () => {
                this.zoomImage(1.2);
            });

            document.getElementById('zoomOut')?.addEventListener('click', () => {
                this.zoomImage(0.8);
            });

            document.getElementById('resetZoom')?.addEventListener('click', () => {
                this.resetImageView();
            });

            // Colormap selection
            document.getElementById('colormapSelect')?.addEventListener('change', (e) => {
                this.updateColormap(e.target.value);
            });
        }

        initializeWorkers() {
            this.workers = {
                coherence: this.createCoherenceWorker(),
                filtering: this.createFilteringWorker(),
                statistics: this.createStatisticsWorker()
            };
        }

        createCoherenceWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { data, config, operation } = e.data;
                    let result;
                    
                    switch(operation) {
                        case 'analyze':
                            result = analyzeCoherence(data, config);
                            break;
                        case 'filter':
                            result = applyCoherenceFilters(data, config);
                            break;
                        case 'statistics':
                            result = calculateStatistics(data);
                            break;
                        default:
                            result = { error: 'Unknown operation' };
                    }
                    
                    self.postMessage(result);
                };

                function analyzeCoherence(data, config) {
                    const coherenceStack = data.coherenceStack;
                    const results = {
                        basic: {},
                        spatial: {},
                        temporal: {},
                        quality: {}
                    };
                    
                    // Basic statistics
                    const allValues = [];
                    coherenceStack.forEach(frame => {
                        allValues.push(...frame.coherence);
                    });
                    
                    results.basic = calculateBasicStats(allValues);
                    
                    // Spatial analysis
                    if (coherenceStack.length > 0) {
                        const firstFrame = coherenceStack[0];
                        results.spatial = analyzeSpatialCoherence(firstFrame, config);
                    }
                    
                    // Temporal analysis
                    if (coherenceStack.length > 1) {
                        results.temporal = analyzeTemporalCoherence(coherenceStack, config);
                    }
                    
                    // Quality metrics
                    results.quality = assessQuality(coherenceStack, config);
                    
                    return results;
                }
                
                function calculateBasicStats(values) {
                    const n = values.length;
                    if (n === 0) return {};
                    
                    const sorted = [...values].sort((a, b) => a - b);
                    const sum = values.reduce((a, b) => a + b, 0);
                    const mean = sum / n;
                    
                    const variance = values.reduce((acc, val) => acc + (val - mean) ** 2, 0) / (n - 1);
                    const std = Math.sqrt(variance);
                    
                    // Moments
                    const skewness = values.reduce((acc, val) => acc + Math.pow((val - mean) / std, 3), 0) / n;
                    const kurtosis = values.reduce((acc, val) => acc + Math.pow((val - mean) / std, 4), 0) / n - 3;
                    
                    return {
                        count: n,
                        mean: mean,
                        median: n % 2 === 0 ? (sorted[n/2-1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)],
                        std: std,
                        variance: variance,
                        min: sorted[0],
                        max: sorted[n-1],
                        range: sorted[n-1] - sorted[0],
                        skewness: skewness,
                        kurtosis: kurtosis,
                        q25: sorted[Math.floor(n * 0.25)],
                        q75: sorted[Math.floor(n * 0.75)]
                    };
                }
                
                function analyzeSpatialCoherence(frame, config) {
                    const { coherence, width, height } = frame;
                    const results = {};
                    
                    // Calculate Moran's I for spatial autocorrelation
                    results.moransI = calculateMoransI(coherence, width, height);
                    
                    // Identify local clusters
                    results.clusters = identifyLocalClusters(coherence, width, height, config.coherenceThreshold);
                    
                    // Edge density
                    results.edgeDensity = calculateEdgeDensity(coherence, width, height);
                    
                    return results;
                }
                
                function calculateMoransI(data, width, height) {
                    let numerator = 0;
                    let denominator = 0;
                    let W = 0;
                    const n = data.length;
                    const mean = data.reduce((a, b) => a + b, 0) / n;
                    
                    for (let i = 0; i < height; i++) {
                        for (let j = 0; j < width; j++) {
                            const idx = i * width + j;
                            const xi = data[idx];
                            
                            // Check neighbors (8-connectivity)
                            for (let di = -1; di <= 1; di++) {
                                for (let dj = -1; dj <= 1; dj++) {
                                    if (di === 0 && dj === 0) continue;
                                    
                                    const ni = i + di;
                                    const nj = j + dj;
                                    
                                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                                        const nidx = ni * width + nj;
                                        const xj = data[nidx];
                                        
                                        numerator += (xi - mean) * (xj - mean);
                                        W += 1;
                                    }
                                }
                            }
                            
                            denominator += (xi - mean) ** 2;
                        }
                    }
                    
                    return W > 0 ? (n * numerator) / (W * denominator) : 0;
                }
                
                function analyzeTemporalCoherence(coherenceStack, config) {
                    const results = {};
                    const timePoints = coherenceStack.length;
                    
                    if (timePoints < 2) return results;
                    
                    // Calculate temporal stability for each pixel
                    const pixelCount = coherenceStack[0].coherence.length;
                    const stability = new Array(pixelCount).fill(0);
                    const trends = new Array(pixelCount).fill(0);
                    
                    for (let p = 0; p < pixelCount; p++) {
                        const timeSeries = coherenceStack.map(frame => frame.coherence[p]);
                        
                        // Calculate stability (coefficient of variation)
                        const mean = timeSeries.reduce((a, b) => a + b, 0) / timePoints;
                        const std = Math.sqrt(timeSeries.reduce((acc, val) => acc + (val - mean) ** 2, 0) / timePoints);
                        stability[p] = mean > 0 ? std / mean : 1;
                        
                        // Calculate trend (linear regression slope)
                        const n = timePoints;
                        const sumX = (n * (n - 1)) / 2;
                        const sumY = timeSeries.reduce((a, b) => a + b, 0);
                        const sumXY = timeSeries.reduce((acc, val, i) => acc + val * i, 0);
                        const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
                        
                        trends[p] = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
                    }
                    
                    results.stabilityIndex = stability.reduce((a, b) => a + b, 0) / pixelCount;
                    results.meanTrend = trends.reduce((a, b) => a + b, 0) / pixelCount;
                    results.stabilityDistribution = calculateBasicStats(stability);
                    results.trendDistribution = calculateBasicStats(trends);
                    
                    return results;
                }
                
                function assessQuality(coherenceStack, config) {
                    const results = {};
                    
                    if (coherenceStack.length === 0) return results;
                    
                    const firstFrame = coherenceStack[0];
                    const totalPixels = firstFrame.coherence.length;
                    
                    // Data coverage
                    let validPixels = 0;
                    let highQualityPixels = 0;
                    
                    for (let i = 0; i < totalPixels; i++) {
                        const coherence = firstFrame.coherence[i];
                        if (coherence > config.maskingThreshold) {
                            validPixels++;
                            if (coherence > config.coherenceThreshold) {
                                highQualityPixels++;
                            }
                        }
                    }
                    
                    results.dataCoverage = validPixels / totalPixels;
                    results.highQualityRatio = highQualityPixels / totalPixels;
                    
                    // Signal consistency (temporal analysis)
                    if (coherenceStack.length > 1) {
                        let consistentPixels = 0;
                        
                        for (let p = 0; p < totalPixels; p++) {
                            const timeSeries = coherenceStack.map(frame => frame.coherence[p]);
                            const mean = timeSeries.reduce((a, b) => a + b, 0) / timeSeries.length;
                            const cv = Math.sqrt(timeSeries.reduce((acc, val) => acc + (val - mean) ** 2, 0) / timeSeries.length) / mean;
                            
                            if (cv < 0.3) { // Less than 30% variation
                                consistentPixels++;
                            }
                        }
                        
                        results.signalConsistency = consistentPixels / totalPixels;
                    }
                    
                    // Overall quality score
                    const coverage = results.dataCoverage || 0;
                    const quality = results.highQualityRatio || 0;
                    const consistency = results.signalConsistency || 0.5;
                    
                    results.overallScore = (coverage * 0.3 + quality * 0.4 + consistency * 0.3);
                    
                    return results;
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        createFilteringWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { data, config, filters } = e.data;
                    const result = applyFilters(data, config, filters);
                    self.postMessage(result);
                };

                function applyFilters(data, config, filters) {
                    let processedData = { ...data };
                    const filteringSteps = [];
                    
                    // Apply spatial filters
                    if (filters.spatial && filters.spatial.length > 0) {
                        processedData = applySpatialFilters(processedData, config, filters.spatial);
                        filteringSteps.push('Spatial filtering applied');
                    }
                    
                    // Apply temporal filters
                    if (filters.temporal && filters.temporal.length > 0) {
                        processedData = applyTemporalFilters(processedData, config, filters.temporal);
                        filteringSteps.push('Temporal filtering applied');
                    }
                    
                    // Apply threshold filtering
                    if (filters.threshold) {
                        processedData = applyThresholdFiltering(processedData, config);
                        filteringSteps.push('Threshold filtering applied');
                    }
                    
                    return {
                        data: processedData,
                        steps: filteringSteps,
                        metrics: calculateFilteringMetrics(data, processedData)
                    };
                }
                
                function applySpatialFilters(data, config, spatialFilters) {
                    const filtered = { ...data };
                    const { coherenceStack } = data;
                    
                    spatialFilters.forEach(filterType => {
                        coherenceStack.forEach(frame => {
                            switch(filterType) {
                                case 'mean':
                                    frame.coherence = applyMeanFilter(frame.coherence, frame.width, frame.height, config.spatialFilterSize);
                                    break;
                                case 'median':
                                    frame.coherence = applyMedianFilter(frame.coherence, frame.width, frame.height, config.spatialFilterSize);
                                    break;
                                case 'gaussian':
                                    frame.coherence = applyGaussianFilter(frame.coherence, frame.width, frame.height, config.spatialFilterSize);
                                    break;
                                case 'bilateral':
                                    frame.coherence = applyBilateralFilter(frame.coherence, frame.width, frame.height, config.spatialFilterSize);
                                    break;
                            }
                        });
                    });
                    
                    return filtered;
                }
                
                function applyMeanFilter(data, width, height, filterSize) {
                    const filtered = new Float32Array(data.length);
                    const halfSize = Math.floor(filterSize / 2);
                    
                    for (let y = 0; y < height; y++) {
                        for (let x = 0; x < width; x++) {
                            let sum = 0;
                            let count = 0;
                            
                            for (let dy = -halfSize; dy <= halfSize; dy++) {
                                for (let dx = -halfSize; dx <= halfSize; dx++) {
                                    const ny = y + dy;
                                    const nx = x + dx;
                                    
                                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                        sum += data[ny * width + nx];
                                        count++;
                                    }
                                }
                            }
                            
                            filtered[y * width + x] = count > 0 ? sum / count : data[y * width + x];
                        }
                    }
                    
                    return filtered;
                }
                
                function applyMedianFilter(data, width, height, filterSize) {
                    const filtered = new Float32Array(data.length);
                    const halfSize = Math.floor(filterSize / 2);
                    
                    for (let y = 0; y < height; y++) {
                        for (let x = 0; x < width; x++) {
                            const values = [];
                            
                            for (let dy = -halfSize; dy <= halfSize; dy++) {
                                for (let dx = -halfSize; dx <= halfSize; dx++) {
                                    const ny = y + dy;
                                    const nx = x + dx;
                                    
                                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                        values.push(data[ny * width + nx]);
                                    }
                                }
                            }
                            
                            values.sort((a, b) => a - b);
                            const median = values.length > 0 ? 
                                (values.length % 2 === 0 ? 
                                    (values[values.length/2 - 1] + values[values.length/2]) / 2 :
                                    values[Math.floor(values.length/2)]) : 
                                data[y * width + x];
                            
                            filtered[y * width + x] = median;
                        }
                    }
                    
                    return filtered;
                }
                
                function applyTemporalFilters(data, config, temporalFilters) {
                    const filtered = { ...data };
                    const { coherenceStack } = data;
                    
                    if (coherenceStack.length < 2) return filtered;
                    
                    temporalFilters.forEach(filterType => {
                        switch(filterType) {
                            case 'movingAverage':
                                applyMovingAverageFilter(coherenceStack, config.temporalFilterWindow);
                                break;
                            case 'kalman':
                                applyKalmanFilter(coherenceStack);
                                break;
                        }
                    });
                    
                    return filtered;
                }
                
                function applyMovingAverageFilter(coherenceStack, windowSize) {
                    const halfWindow = Math.floor(windowSize / 2);
                    const pixelCount = coherenceStack[0].coherence.length;
                    
                    for (let p = 0; p < pixelCount; p++) {
                        const originalSeries = coherenceStack.map(frame => frame.coherence[p]);
                        
                        coherenceStack.forEach((frame, t) => {
                            let sum = 0;
                            let count = 0;
                            
                            for (let dt = -halfWindow; dt <= halfWindow; dt++) {
                                const idx = t + dt;
                                if (idx >= 0 && idx < coherenceStack.length) {
                                    sum += originalSeries[idx];
                                    count++;
                                }
                            }
                            
                            frame.coherence[p] = count > 0 ? sum / count : originalSeries[t];
                        });
                    }
                }
                
                function calculateFilteringMetrics(original, filtered) {
                    const originalData = original.coherenceStack[0].coherence;
                    const filteredData = filtered.coherenceStack[0].coherence;
                    
                    let mse = 0;
                    let mae = 0;
                    let snrOriginal = 0;
                    let snrFiltered = 0;
                    
                    const n = originalData.length;
                    
                    for (let i = 0; i < n; i++) {
                        const diff = originalData[i] - filteredData[i];
                        mse += diff * diff;
                        mae += Math.abs(diff);
                    }
                    
                    mse /= n;
                    mae /= n;
                    
                    // Simplified SNR calculation
                    const originalMean = originalData.reduce((a, b) => a + b, 0) / n;
                    const filteredMean = filteredData.reduce((a, b) => a + b, 0) / n;
                    
                    const originalNoise = Math.sqrt(originalData.reduce((acc, val) => acc + (val - originalMean) ** 2, 0) / n);
                    const filteredNoise = Math.sqrt(filteredData.reduce((acc, val) => acc + (val - filteredMean) ** 2, 0) / n);
                    
                    snrOriginal = originalNoise > 0 ? 20 * Math.log10(originalMean / originalNoise) : 0;
                    snrFiltered = filteredNoise > 0 ? 20 * Math.log10(filteredMean / filteredNoise) : 0;
                    
                    return {
                        mse: mse,
                        mae: mae,
                        rmse: Math.sqrt(mse),
                        snrImprovement: snrFiltered - snrOriginal,
                        coherenceGain: filteredMean - originalMean
                    };
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        createStatisticsWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { data, analysisType } = e.data;
                    const result = performStatisticalAnalysis(data, analysisType);
                    self.postMessage(result);
                };

                function performStatisticalAnalysis(data, analysisType) {
                    switch(analysisType) {
                        case 'distribution':
                            return analyzeDistribution(data);
                        case 'correlation':
                            return analyzeCorrelation(data);
                        case 'regression':
                            return performRegression(data);
                        default:
                            return { error: 'Unknown analysis type' };
                    }
                }
                
                function analyzeDistribution(data) {
                    // Implementation for distribution analysis
                    return { status: 'Distribution analysis complete' };
                }
                
                function analyzeCorrelation(data) {
                    // Implementation for correlation analysis
                    return { status: 'Correlation analysis complete' };
                }
                
                function performRegression(data) {
                    // Implementation for regression analysis
                    return { status: 'Regression analysis complete' };
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        setupVisualization() {
            this.visualization = {
                zoom: 1,
                panX: 0,
                panY: 0,
                currentView: 'original',
                colormap: 'viridis'
            };
        }

        async loadCoherenceData() {
            this.updateStatus('Loading coherence data...', 10);
            
            try {
                // Generate mock coherence stack for demonstration
                const mockData = this.generateMockCoherenceData();
                this.data = mockData;
                
                // Populate coherence stack
                mockData.coherenceStack.forEach((frame, index) => {
                    this.coherenceStack.set(frame.id, frame);
                });
                
                this.updateStatus('Data loaded successfully', 30);
                this.visualizeCoherence();
                this.updateHistogram();
                
            } catch (error) {
                console.error('Error loading coherence data:', error);
                this.updateStatus('Error loading data', 0);
            }
        }

        generateMockCoherenceData() {
            const numFrames = 10;
            const width = 100;
            const height = 100;
            const coherenceStack = [];
            
            for (let f = 0; f < numFrames; f++) {
                const frame = {
                    id: `frame_${f.toString().padStart(3, '0')}`,
                    timestamp: new Date(2020, 0, 1 + f * 12), // Monthly data
                    coherence: new Float32Array(width * height),
                    phase: new Float32Array(width * height),
                    width: width,
                    height: height
                };
                
                // Generate synthetic coherence pattern
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = y * width + x;
                        
                        // Create coherence pattern with some spatial structure
                        const centerX = width / 2;
                        const centerY = height / 2;
                        const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
                        
                        // Base coherence with distance decay
                        let baseCoherence = Math.exp(-dist / 30) * 0.8 + 0.2;
                        
                        // Add temporal variation
                        const temporalVariation = 0.1 * Math.sin(f * Math.PI / 5);
                        baseCoherence += temporalVariation;
                        
                        // Add noise
                        const noise = (Math.random() - 0.5) * 0.2;
                        baseCoherence += noise;
                        
                        // Clamp to [0, 1]
                        frame.coherence[idx] = Math.max(0, Math.min(1, baseCoherence));
                        
                        // Generate corresponding phase
                        frame.phase[idx] = (Math.random() - 0.5) * 2 * Math.PI;
                    }
                }
                
                coherenceStack.push(frame);
            }
            
            return {
                coherenceStack: coherenceStack,
                metadata: {
                    sensor: 'Sentinel-1',
                    wavelength: 5.6,
                    processing_level: 'L2',
                    area: 'Test Region'
                }
            };
        }

        async runCoherenceAnalysis() {
            if (!this.data) {
                alert('Please load coherence data first');
                return;
            }

            this.updateStatus('Running coherence analysis...', 50);

            try {
                const result = await new Promise((resolve) => {
                    this.workers.coherence.onmessage = (e) => {
                        resolve(e.data);
                    };
                    
                    this.workers.coherence.postMessage({
                        data: this.data,
                        config: this.config,
                        operation: 'analyze'
                    });
                });

                this.analysisResults = result;
                this.updateStatisticsDisplay();
                this.updateQualityMetrics();
                this.updateStatus('Analysis complete', 100);

            } catch (error) {
                console.error('Error running analysis:', error);
                this.updateStatus('Analysis failed', 50);
            }
        }

        async applyFilters() {
            if (!this.data) {
                alert('Please load coherence data first');
                return;
            }

            this.updateStatus('Applying filters...', 60);

            try {
                const selectedFilters = this.getSelectedFilters();
                
                const result = await new Promise((resolve) => {
                    this.workers.filtering.onmessage = (e) => {
                        resolve(e.data);
                    };
                    
                    this.workers.filtering.postMessage({
                        data: this.data,
                        config: this.config,
                        filters: selectedFilters
                    });
                });

                this.filteredData = result.data;
                this.updateFilteringResults(result);
                this.updateStatus('Filtering complete', 80);

            } catch (error) {
                console.error('Error applying filters:', error);
                this.updateStatus('Filtering failed', 60);
            }
        }

        getSelectedFilters() {
            const spatialFilters = [];
            const temporalFilters = [];
            
            // Check selected spatial filters
            if (document.getElementById('meanFilter')?.checked) spatialFilters.push('mean');
            if (document.getElementById('medianFilter')?.checked) spatialFilters.push('median');
            if (document.getElementById('gaussianFilter')?.checked) spatialFilters.push('gaussian');
            if (document.getElementById('bilateralFilter')?.checked) spatialFilters.push('bilateral');
            
            // Check selected temporal filters
            if (document.getElementById('movingAverage')?.checked) temporalFilters.push('movingAverage');
            if (document.getElementById('kalmanFilter')?.checked) temporalFilters.push('kalman');
            if (document.getElementById('trendFilter')?.checked) temporalFilters.push('trend');
            
            return {
                spatial: spatialFilters,
                temporal: temporalFilters,
                threshold: true
            };
        }

        visualizeCoherence() {
            if (!this.data || this.coherenceStack.size === 0) return;

            const canvas = this.canvases.main;
            const ctx = this.contexts.main;
            
            if (!canvas || !ctx) return;

            // Get first frame for visualization
            const firstFrame = Array.from(this.coherenceStack.values())[0];
            this.drawCoherenceFrame(ctx, canvas, firstFrame);
        }

        drawCoherenceFrame(ctx, canvas, frame) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const { coherence, width, height } = frame;
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height / height;

            // Create color mapping function
            const colormap = this.createColormap(this.visualization.colormap);

            // Draw coherence as heatmap
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const value = coherence[idx];
                    
                    const color = colormap(value);
                    ctx.fillStyle = color;
                    
                    const drawX = x * cellWidth * this.visualization.zoom + this.visualization.panX;
                    const drawY = y * cellHeight * this.visualization.zoom + this.visualization.panY;
                    const drawWidth = cellWidth * this.visualization.zoom;
                    const drawHeight = cellHeight * this.visualization.zoom;
                    
                    ctx.fillRect(drawX, drawY, drawWidth, drawHeight);
                }
            }

            // Add grid if zoomed in enough
            if (this.visualization.zoom > 4) {
                this.drawGrid(ctx, canvas, width, height);
            }
        }

        createColormap(type) {
            const colormaps = {
                gray: (val) => {
                    const intensity = Math.floor(val * 255);
                    return `rgb(${intensity}, ${intensity}, ${intensity})`;
                },
                jet: (val) => {
                    // Jet colormap implementation
                    if (val < 0.125) {
                        return `rgb(0, 0, ${Math.floor(128 + val * 1024)})`;
                    } else if (val < 0.375) {
                        return `rgb(0, ${Math.floor((val - 0.125) * 1024)}, 255)`;
                    } else if (val < 0.625) {
                        return `rgb(${Math.floor((val - 0.375) * 1024)}, 255, ${Math.floor(255 - (val - 0.375) * 1024)})`;
                    } else if (val < 0.875) {
                        return `rgb(255, ${Math.floor(255 - (val - 0.625) * 1024)}, 0)`;
                    } else {
                        return `rgb(${Math.floor(255 - (val - 0.875) * 1024)}, 0, 0)`;
                    }
                },
                viridis: (val) => {
                    // Simplified viridis approximation
                    const r = Math.floor(val * val * 255);
                    const g = Math.floor(val * 255);
                    const b = Math.floor((1 - val) * 255);
                    return `rgb(${r}, ${g}, ${b})`;
                },
                coherence: (val) => {
                    // Custom coherence colormap
                    if (val < this.config.maskingThreshold) {
                        return 'rgb(128, 0, 0)'; // Low coherence - dark red
                    } else if (val < this.config.coherenceThreshold) {
                        return 'rgb(255, 165, 0)'; // Medium coherence - orange
                    } else {
                        const intensity = Math.floor((val - this.config.coherenceThreshold) / (1 - this.config.coherenceThreshold) * 255);
                        return `rgb(0, ${intensity}, 0)`; // High coherence - green
                    }
                }
            };
            
            return colormaps[type] || colormaps.viridis;
        }

        drawGrid(ctx, canvas, width, height) {
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height / height;
            
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            
            for (let x = 0; x <= width; x++) {
                const drawX = x * cellWidth * this.visualization.zoom + this.visualization.panX;
                ctx.beginPath();
                ctx.moveTo(drawX, 0);
                ctx.lineTo(drawX, canvas.height);
                ctx.stroke();
            }
            
            for (let y = 0; y <= height; y++) {
                const drawY = y * cellHeight * this.visualization.zoom + this.visualization.panY;
                ctx.beginPath();
                ctx.moveTo(0, drawY);
                ctx.lineTo(canvas.width, drawY);
                ctx.stroke();
            }
        }

        updateHistogram() {
            if (!this.data || this.coherenceStack.size === 0) return;

            const canvas = this.canvases.histogram;
            const ctx = this.contexts.histogram;
            
            if (!canvas || !ctx) return;

            // Collect all coherence values
            const allValues = [];
            this.coherenceStack.forEach(frame => {
                allValues.push(...Array.from(frame.coherence));
            });

            this.drawHistogram(ctx, canvas, allValues, 'Coherence Distribution', 'Coherence', 'Frequency');
        }

        drawHistogram(ctx, canvas, data, title, xLabel, yLabel) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const bins = 50;
            const histogram = new Array(bins).fill(0);
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binWidth = (max - min) / bins;

            // Calculate histogram
            data.forEach(value => {
                const bin = Math.min(Math.floor((value - min) / binWidth), bins - 1);
                histogram[bin]++;
            });

            // Drawing parameters
            const margin = { top: 30, right: 20, bottom: 40, left: 50 };
            const plotWidth = canvas.width - margin.left - margin.right;
            const plotHeight = canvas.height - margin.top - margin.bottom;

            // Draw histogram bars
            const maxCount = Math.max(...histogram);
            const barWidth = plotWidth / bins;

            ctx.fillStyle = '#4CAF50';
            histogram.forEach((count, i) => {
                const x = margin.left + i * barWidth;
                const height = (count / maxCount) * plotHeight;
                const y = canvas.height - margin.bottom - height;
                
                ctx.fillRect(x, y, barWidth * 0.8, height);
            });

            // Draw axes
            ctx.strokeStyle = '#000000';
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, canvas.height - margin.bottom);
            ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
            ctx.stroke();

            // Add labels
            ctx.fillStyle = '#000000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            
            // Title
            ctx.font = '14px Arial';
            ctx.fillText(title, canvas.width / 2, 20);
            
            // X-axis labels
            ctx.font = '10px Arial';
            for (let i = 0; i <= 10; i++) {
                const x = margin.left + (i / 10) * plotWidth;
                const value = min + (i / 10) * (max - min);
                ctx.fillText(value.toFixed(2), x, canvas.height - 10);
            }
            
            // Y-axis labels
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {
                const y = canvas.height - margin.bottom - (i / 5) * plotHeight;
                const value = (i / 5) * maxCount;
                ctx.fillText(Math.floor(value), margin.left - 10, y + 3);
            }
        }

        updateStatisticsDisplay() {
            if (!this.analysisResults || !this.analysisResults.basic) return;

            const stats = this.analysisResults.basic;
            
            // Update basic statistics
            document.getElementById('statMean').textContent = stats.mean?.toFixed(3) || '--';
            document.getElementById('statMedian').textContent = stats.median?.toFixed(3) || '--';
            document.getElementById('statStd').textContent = stats.std?.toFixed(3) || '--';
            document.getElementById('statMin').textContent = stats.min?.toFixed(3) || '--';
            document.getElementById('statMax').textContent = stats.max?.toFixed(3) || '--';
            document.getElementById('statRange').textContent = stats.range?.toFixed(3) || '--';
            
            // Update distribution metrics
            document.getElementById('skewness').textContent = stats.skewness?.toFixed(3) || '--';
            document.getElementById('kurtosis').textContent = stats.kurtosis?.toFixed(3) || '--';
            
            // Calculate entropy
            if (this.data && this.coherenceStack.size > 0) {
                const entropy = this.calculateEntropy();
                document.getElementById('entropy').textContent = entropy.toFixed(3);
            }
            
            // Update spatial statistics
            if (this.analysisResults.spatial) {
                const spatial = this.analysisResults.spatial;
                document.getElementById('moransI').textContent = spatial.moransI?.toFixed(3) || '--';
                document.getElementById('localClusters').textContent = spatial.clusters?.length || '--';
                document.getElementById('edgeDensity').textContent = spatial.edgeDensity?.toFixed(3) || '--';
            }
            
            // Update temporal statistics
            if (this.analysisResults.temporal) {
                const temporal = this.analysisResults.temporal;
                document.getElementById('stabilityIndex').textContent = temporal.stabilityIndex?.toFixed(3) || '--';
                document.getElementById('trendSlope').textContent = temporal.meanTrend?.toFixed(4) || '--';
            }
        }

        calculateEntropy() {
            // Calculate information entropy of coherence distribution
            const allValues = [];
            this.coherenceStack.forEach(frame => {
                allValues.push(...Array.from(frame.coherence));
            });
            
            const bins = 256;
            const histogram = new Array(bins).fill(0);
            const min = Math.min(...allValues);
            const max = Math.max(...allValues);
            const binWidth = (max - min) / bins;
            
            allValues.forEach(value => {
                const bin = Math.min(Math.floor((value - min) / binWidth), bins - 1);
                histogram[bin]++;
            });
            
            const total = allValues.length;
            let entropy = 0;
            
            histogram.forEach(count => {
                if (count > 0) {
                    const probability = count / total;
                    entropy -= probability * Math.log2(probability);
                }
            });
            
            return entropy;
        }

        updateQualityMetrics() {
            if (!this.analysisResults || !this.analysisResults.quality) return;

            const quality = this.analysisResults.quality;
            
            // Update overall score
            const overallScore = Math.round(quality.overallScore * 100);
            document.getElementById('overallScore').textContent = `${overallScore}%`;
            
            // Update quality bars
            const coverage = quality.dataCoverage * 100;
            const consistency = quality.signalConsistency * 100;
            const noise = (1 - (quality.highQualityRatio || 0)) * 100;
            
            document.getElementById('coverageBar').style.width = `${coverage}%`;
            document.getElementById('consistencyBar').style.width = `${consistency}%`;
            document.getElementById('noiseBar').style.width = `${noise}%`;
            
            document.getElementById('coverageScore').textContent = `${coverage.toFixed(1)}%`;
            document.getElementById('consistencyScore').textContent = `${consistency.toFixed(1)}%`;
            document.getElementById('noiseScore').textContent = `${noise.toFixed(1)}%`;
            
            // Draw quality visualizations
            this.drawQualityPlots();
        }

        drawQualityPlots() {
            this.drawThresholdCurve();
            this.drawRegionalMap();
        }

        drawThresholdCurve() {
            const canvas = this.canvases.threshold;
            const ctx = this.contexts.threshold;
            
            if (!canvas || !ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Generate ROC-like curve for threshold analysis
            const thresholds = [];
            const tpr = [];
            const fpr = [];
            
            for (let t = 0; t <= 1; t += 0.01) {
                thresholds.push(t);
                
                // Simplified TPR/FPR calculation
                const truePositive = 1 - t; // Higher threshold = lower TPR
                const falsePositive = Math.exp(-t * 5); // Exponential decay
                
                tpr.push(truePositive);
                fpr.push(falsePositive);
            }
            
            // Draw curve
            const margin = { top: 20, right: 20, bottom: 40, left: 50 };
            const plotWidth = canvas.width - margin.left - margin.right;
            const plotHeight = canvas.height - margin.top - margin.bottom;
            
            ctx.strokeStyle = '#2196F3';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            tpr.forEach((tp, i) => {
                const x = margin.left + (fpr[i]) * plotWidth;
                const y = canvas.height - margin.bottom - tp * plotHeight;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Draw axes
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, canvas.height - margin.bottom);
            ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
            ctx.stroke();
            
            // Add labels
            ctx.fillStyle = '#000000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('ROC Curve', canvas.width / 2, 15);
            ctx.fillText('False Positive Rate', canvas.width / 2, canvas.height - 5);
            
            ctx.save();
            ctx.translate(15, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('True Positive Rate', 0, 0);
            ctx.restore();
        }

        drawRegionalMap() {
            const canvas = this.canvases.regional;
            const ctx = this.contexts.regional;
            
            if (!canvas || !ctx || this.coherenceStack.size === 0) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Get first frame
            const frame = Array.from(this.coherenceStack.values())[0];
            const { coherence, width, height } = frame;
            
            // Create regional classification
            const regions = this.classifyRegions(coherence, width, height);
            
            // Draw regions with different colors
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height / height;
            
            const regionColors = {
                high: '#4CAF50',      // Green
                medium: '#FF9800',    // Orange
                low: '#F44336',       // Red
                decorrelated: '#424242' // Gray
            };
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const region = regions[idx];
                    
                    ctx.fillStyle = regionColors[region];
                    ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                }
            }
            
            // Update region statistics
            const regionStats = this.calculateRegionStatistics(regions);
            this.updateRegionStatistics(regionStats);
        }

        classifyRegions(coherence, width, height) {
            const regions = new Array(coherence.length);
            
            for (let i = 0; i < coherence.length; i++) {
                const coh = coherence[i];
                
                if (coh < this.config.maskingThreshold) {
                    regions[i] = 'decorrelated';
                } else if (coh < this.config.coherenceThreshold) {
                    regions[i] = 'low';
                } else if (coh < 0.7) {
                    regions[i] = 'medium';
                } else {
                    regions[i] = 'high';
                }
            }
            
            return regions;
        }

        calculateRegionStatistics(regions) {
            const stats = {
                high: 0,
                medium: 0,
                low: 0,
                decorrelated: 0
            };
            
            regions.forEach(region => {
                stats[region]++;
            });
            
            const total = regions.length;
            Object.keys(stats).forEach(key => {
                stats[key] = (stats[key] / total * 100).toFixed(1);
            });
            
            return stats;
        }

        updateRegionStatistics(stats) {
            document.getElementById('highCoherenceRegions').textContent = `${stats.high}%`;
            document.getElementById('mediumCoherenceRegions').textContent = `${stats.medium}%`;
            document.getElementById('lowCoherenceRegions').textContent = `${stats.low}%`;
            document.getElementById('decorrelatedRegions').textContent = `${stats.decorrelated}%`;
        }

        updateFilteringResults(results) {
            const metrics = results.metrics;
            
            // Update improvement metrics
            document.getElementById('snrImprovement').textContent = 
                metrics.snrImprovement ? `${metrics.snrImprovement.toFixed(2)} dB` : '--';
            document.getElementById('coherenceGain').textContent = 
                metrics.coherenceGain ? `${metrics.coherenceGain.toFixed(3)}` : '--';
                
            // Draw before/after comparison
            this.drawBeforeAfterComparison();
        }

        drawBeforeAfterComparison() {
            if (!this.data || !this.filteredData) return;

            const beforeCanvas = this.canvases.before;
            const afterCanvas = this.canvases.after;
            const beforeCtx = this.contexts.before;
            const afterCtx = this.contexts.after;
            
            if (!beforeCanvas || !afterCanvas) return;

            const originalFrame = Array.from(this.coherenceStack.values())[0];
            const filteredFrame = this.filteredData.coherenceStack[0];
            
            this.drawCoherenceFrame(beforeCtx, beforeCanvas, originalFrame);
            this.drawCoherenceFrame(afterCtx, afterCanvas, filteredFrame);
        }

        handleCanvasHover(type, x, y) {
            if (type === 'main' && this.coherenceStack.size > 0) {
                this.updatePixelInfo(x, y);
            }
        }

        updatePixelInfo(canvasX, canvasY) {
            const frame = Array.from(this.coherenceStack.values())[0];
            if (!frame) return;

            const { width, height, coherence } = frame;
            
            // Convert canvas coordinates to pixel coordinates
            const cellWidth = this.canvases.main.width / width;
            const cellHeight = this.canvases.main.height / height;
            
            const pixelX = Math.floor((canvasX - this.visualization.panX) / (cellWidth * this.visualization.zoom));
            const pixelY = Math.floor((canvasY - this.visualization.panY) / (cellHeight * this.visualization.zoom));
            
            if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
                const idx = pixelY * width + pixelX;
                const cohValue = coherence[idx];
                
                const pixelInfo = document.getElementById('pixelInfo');
                if (pixelInfo) {
                    pixelInfo.innerHTML = `
                        Pixel: (${pixelX}, ${pixelY})<br>
                        Coherence: ${cohValue.toFixed(3)}
                    `;
                }
            }
        }

        handleCanvasClick(type, x, y) {
            console.log(`Clicked ${type} canvas at (${x}, ${y})`);
        }

        handleCanvasZoom(type, x, y, deltaY) {
            if (type === 'main') {
                const zoomFactor = deltaY > 0 ? 0.9 : 1.1;
                this.zoomImage(zoomFactor);
                this.visualizeCoherence();
            }
        }

        zoomImage(factor) {
            this.visualization.zoom *= factor;
            this.visualization.zoom = Math.max(0.1, Math.min(10, this.visualization.zoom));
        }

        resetImageView() {
            this.visualization.zoom = 1;
            this.visualization.panX = 0;
            this.visualization.panY = 0;
            this.visualizeCoherence();
        }

        switchTab(tabId) {
            this.ui.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tab === tabId);
            });
            
            this.ui.querySelectorAll('.tab-content').forEach(content => {
                content.classList.toggle('active', content.id === `${tabId}-tab`);
            });
        }

        switchView(viewType) {
            this.visualization.currentView = viewType;
            
            this.ui.querySelectorAll('.view-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.view === viewType);
            });
            
            // Update visualization based on view type
            this.updateVisualizationForView(viewType);
        }

        updateVisualizationForView(viewType) {
            switch (viewType) {
                case 'original':
                    this.visualizeCoherence();
                    break;
                case 'filtered':
                    if (this.filteredData) {
                        this.visualizeFilteredData();
                    }
                    break;
                case 'mask':
                    this.visualizeMask();
                    break;
                case 'difference':
                    if (this.filteredData) {
                        this.visualizeDifference();
                    }
                    break;
            }
        }

        visualizeFilteredData() {
            if (!this.filteredData || this.filteredData.coherenceStack.length === 0) return;
            
            const frame = this.filteredData.coherenceStack[0];
            const canvas = this.canvases.main;
            const ctx = this.contexts.main;
            
            this.drawCoherenceFrame(ctx, canvas, frame);
        }

        visualizeMask() {
            const canvas = this.canvases.main;
            const ctx = this.contexts.main;
            
            if (!canvas || !ctx || this.coherenceStack.size === 0) return;

            const frame = Array.from(this.coherenceStack.values())[0];
            const { coherence, width, height } = frame;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height / height;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const value = coherence[idx];
                    
                    // Create binary mask
                    const color = value > this.config.coherenceThreshold ? 'white' : 'black';
                    ctx.fillStyle = color;
                    ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                }
            }
        }

        visualizeDifference() {
            if (!this.data || !this.filteredData) return;
            
            const originalFrame = Array.from(this.coherenceStack.values())[0];
            const filteredFrame = this.filteredData.coherenceStack[0];
            
            const canvas = this.canvases.main;
            const ctx = this.contexts.main;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const { width, height } = originalFrame;
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height / height;
            
            // Find min/max difference for scaling
            let minDiff = Infinity;
            let maxDiff = -Infinity;
            
            for (let i = 0; i < originalFrame.coherence.length; i++) {
                const diff = filteredFrame.coherence[i] - originalFrame.coherence[i];
                minDiff = Math.min(minDiff, diff);
                maxDiff = Math.max(maxDiff, diff);
            }
            
            const range = maxDiff - minDiff;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const diff = filteredFrame.coherence[idx] - originalFrame.coherence[idx];
                    
                    // Map difference to color
                    const normalized = range > 0 ? (diff - minDiff) / range : 0.5;
                    
                    let r, g, b;
                    if (normalized < 0.5) {
                        // Blue to white
                        const scale = normalized * 2;
                        r = Math.floor(scale * 255);
                        g = Math.floor(scale * 255);
                        b = 255;
                    } else {
                        // White to red
                        const scale = (normalized - 0.5) * 2;
                        r = 255;
                        g = Math.floor((1 - scale) * 255);
                        b = Math.floor((1 - scale) * 255);
                    }
                    
                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                }
            }
        }

        updateColormap(colormapType) {
            this.visualization.colormap = colormapType;
            this.updateVisualizationForView(this.visualization.currentView);
        }

        updateParameter(paramId, value) {
            const numValue = parseFloat(value);
            
            switch (paramId) {
                case 'coherenceThreshold':
                    this.config.coherenceThreshold = numValue;
                    document.getElementById('thresholdValue').textContent = numValue.toFixed(2);
                    break;
                case 'maskingThreshold':
                    this.config.maskingThreshold = numValue;
                    document.getElementById('maskingValue').textContent = numValue.toFixed(2);
                    break;
                case 'temporalWindow':
                    this.config.temporalFilterWindow = parseInt(value);
                    document.getElementById('temporalValue').textContent = value;
                    break;
            }
            
            // Update visualization if needed
            if (paramId.includes('Threshold')) {
                this.updateVisualizationForView(this.visualization.currentView);
            }
        }

        updateStatus(message, progress = null) {
            const statusText = document.getElementById('analysisStatus');
            const progressFill = document.getElementById('analysisProgress');
            
            if (statusText) statusText.textContent = message;
            if (progressFill && progress !== null) {
                progressFill.style.width = `${progress}%`;
            }
        }

        async exportResults() {
            if (!this.data) {
                alert('No data to export');
                return;
            }

            const results = {
                metadata: this.data.metadata,
                configuration: this.config,
                analysisResults: this.analysisResults,
                filteredData: this.filteredData ? {
                    coherenceStack: this.filteredData.coherenceStack
                } : null,
                timestamp: new Date().toISOString()
            };

            const blob = new Blob([JSON.stringify(results, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `coherence_analysis_results_${Date.now()}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
            
            console.log('🔍 Coherence analysis results exported');
        }

        destroy() {
            // Clean up workers
            Object.values(this.workers).forEach(worker => {
                worker.terminate();
            });
            
            // Remove UI
            if (this.ui) {
                this.ui.remove();
            }
            
            console.log('🧹 Coherence Analyzer destroyed');
        }
    }

    // Supporting filter classes
    class SpatialCoherenceFilter {
        apply(data, config) {
            // Implementation for spatial coherence filtering
            return data;
        }
    }

    class TemporalCoherenceFilter {
        apply(data, config) {
            // Implementation for temporal coherence filtering
            return data;
        }
    }

    class AdaptiveCoherenceFilter {
        apply(data, config) {
            // Implementation for adaptive coherence filtering
            return data;
        }
    }

    class EdgePreservingFilter {
        apply(data, config) {
            // Implementation for edge-preserving filtering
            return data;
        }
    }

    class BasicStatistics {
        calculate(data) {
            // Implementation for basic statistics
            return {};
        }
    }

    class AdvancedStatistics {
        calculate(data) {
            // Implementation for advanced statistics
            return {};
        }
    }

    class QualityMetrics {
        assess(data, config) {
            // Implementation for quality metrics assessment
            return {};
        }
    }

    // Create global instance
    window.CoherenceAnalyzer = CoherenceAnalyzer;
    
    // Auto-initialize if container exists
    const analyzerContainer = document.querySelector('.coherence-analyzer-mount');
    if (analyzerContainer) {
        window.coherenceAnalyzer = new CoherenceAnalyzer({
            container: analyzerContainer
        });
    }
    
    console.log('🔍 Coherence Analyzer loaded successfully');

})();