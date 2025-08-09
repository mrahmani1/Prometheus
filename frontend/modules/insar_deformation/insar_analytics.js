/**
 * InSAR Analytics Engine - Task 3.3
 * Advanced analytical tools for InSAR data processing and visualization
 * Implements velocity field extraction, PCA, and advanced statistical analysis
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function() {
    'use strict';

    class InSARAnalytics {
        constructor(options = {}) {
            this.container = options.container || document.body;
            this.data = null;
            this.interferograms = new Map();
            this.velocityField = null;
            this.pcaResults = null;
            this.networkGraph = null;
            this.analyticsUI = null;
            
            this.config = {
                coherenceThreshold: 0.3,
                temporalBaseline: 365, // days
                spatialBaseline: 300,  // meters
                wavelength: 5.6,       // cm (C-band)
                incidenceAngle: 23,    // degrees
                minObservations: 10,
                velocityUnit: 'mm/year',
                smoothingKernel: 3,
                outlierThreshold: 2.5,
                pcaComponents: 5
            };

            this.algorithms = {
                sbas: new SBASProcessor(),
                psInSAR: new PSInSARProcessor(),
                statistics: new StatisticalAnalyzer(),
                network: new NetworkAnalyzer()
            };

            this.init();
        }

        init() {
            this.createAnalyticsInterface();
            this.setupEventHandlers();
            this.initializeWorkers();
            console.log('🔬 InSAR Analytics Engine initialized');
        }

        createAnalyticsInterface() {
            const container = document.createElement('div');
            container.className = 'insar-analytics-container';
            container.innerHTML = `
                <div class="analytics-header">
                    <h3><i class="fas fa-chart-line"></i> InSAR Analytics</h3>
                    <div class="analytics-controls">
                        <button class="btn btn-primary" id="loadData">Load Data</button>
                        <button class="btn btn-secondary" id="processVelocity">Process Velocity</button>
                        <button class="btn btn-info" id="runPCA">Run PCA</button>
                        <button class="btn btn-success" id="exportResults">Export</button>
                    </div>
                </div>
                
                <div class="analytics-tabs">
                    <button class="tab-btn active" data-tab="velocity">Velocity Field</button>
                    <button class="tab-btn" data-tab="pca">PCA Analysis</button>
                    <button class="tab-btn" data-tab="network">Network</button>
                    <button class="tab-btn" data-tab="statistics">Statistics</button>
                </div>

                <div class="analytics-content">
                    <div class="tab-content active" id="velocity-tab">
                        <div class="velocity-controls">
                            <div class="control-group">
                                <label>Reference Point:</label>
                                <input type="text" id="refPoint" placeholder="lat,lon" />
                                <button id="selectRef">Select on Map</button>
                            </div>
                            <div class="control-group">
                                <label>Temporal Baseline (days):</label>
                                <input type="range" id="temporalRange" min="12" max="1095" value="${this.config.temporalBaseline}" />
                                <span id="temporalValue">${this.config.temporalBaseline}</span>
                            </div>
                            <div class="control-group">
                                <label>Spatial Baseline (m):</label>
                                <input type="range" id="spatialRange" min="50" max="1000" value="${this.config.spatialBaseline}" />
                                <span id="spatialValue">${this.config.spatialBaseline}</span>
                            </div>
                        </div>
                        <div class="velocity-visualization">
                            <canvas id="velocityCanvas" width="800" height="600"></canvas>
                            <div class="velocity-stats">
                                <div class="stat-item">
                                    <span class="stat-label">Mean Velocity:</span>
                                    <span class="stat-value" id="meanVel">--</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Std Deviation:</span>
                                    <span class="stat-value" id="stdVel">--</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Max Velocity:</span>
                                    <span class="stat-value" id="maxVel">--</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" id="pca-tab">
                        <div class="pca-controls">
                            <div class="control-group">
                                <label>Number of Components:</label>
                                <input type="number" id="pcaComponents" min="1" max="20" value="${this.config.pcaComponents}" />
                            </div>
                            <div class="control-group">
                                <label>Variance Threshold:</label>
                                <input type="range" id="varianceThreshold" min="0.1" max="1.0" step="0.05" value="0.95" />
                                <span id="varianceValue">0.95</span>
                            </div>
                        </div>
                        <div class="pca-visualization">
                            <div class="pca-plots">
                                <canvas id="screeCanvas" width="400" height="300"></canvas>
                                <canvas id="loadingsCanvas" width="400" height="300"></canvas>
                            </div>
                            <div class="pca-components">
                                <div id="componentsList"></div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" id="network-tab">
                        <div class="network-controls">
                            <div class="control-group">
                                <label>Layout Type:</label>
                                <select id="networkLayout">
                                    <option value="force">Force-directed</option>
                                    <option value="circular">Circular</option>
                                    <option value="hierarchical">Hierarchical</option>
                                    <option value="temporal">Temporal</option>
                                </select>
                            </div>
                            <div class="control-group">
                                <label>Edge Weight:</label>
                                <select id="edgeWeight">
                                    <option value="coherence">Coherence</option>
                                    <option value="baseline">Baseline</option>
                                    <option value="temporal">Temporal</option>
                                </select>
                            </div>
                        </div>
                        <div class="network-visualization">
                            <canvas id="networkCanvas" width="800" height="600"></canvas>
                            <div class="network-info">
                                <div class="info-item">
                                    <span>Nodes: <span id="nodeCount">0</span></span>
                                </div>
                                <div class="info-item">
                                    <span>Edges: <span id="edgeCount">0</span></span>
                                </div>
                                <div class="info-item">
                                    <span>Density: <span id="networkDensity">0</span></span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" id="statistics-tab">
                        <div class="statistics-grid">
                            <div class="stat-panel">
                                <h4>Temporal Coherence</h4>
                                <canvas id="coherenceHistogram" width="300" height="200"></canvas>
                            </div>
                            <div class="stat-panel">
                                <h4>Phase Distribution</h4>
                                <canvas id="phaseHistogram" width="300" height="200"></canvas>
                            </div>
                            <div class="stat-panel">
                                <h4>Velocity Distribution</h4>
                                <canvas id="velocityHistogram" width="300" height="200"></canvas>
                            </div>
                            <div class="stat-panel">
                                <h4>Quality Metrics</h4>
                                <div class="quality-metrics">
                                    <div class="metric">
                                        <label>Mean Coherence:</label>
                                        <span id="meanCoherence">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Phase Stability:</label>
                                        <span id="phaseStability">--</span>
                                    </div>
                                    <div class="metric">
                                        <label>Temporal Consistency:</label>
                                        <span id="temporalConsistency">--</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="processing-status">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div class="status-text" id="statusText">Ready</div>
                </div>
            `;

            this.container.appendChild(container);
            this.analyticsUI = container;
            this.setupCanvases();
        }

        setupCanvases() {
            this.canvases = {
                velocity: document.getElementById('velocityCanvas'),
                scree: document.getElementById('screeCanvas'),
                loadings: document.getElementById('loadingsCanvas'),
                network: document.getElementById('networkCanvas'),
                coherenceHist: document.getElementById('coherenceHistogram'),
                phaseHist: document.getElementById('phaseHistogram'),
                velocityHist: document.getElementById('velocityHistogram')
            };

            this.contexts = {};
            Object.keys(this.canvases).forEach(key => {
                if (this.canvases[key]) {
                    this.contexts[key] = this.canvases[key].getContext('2d');
                    this.setupCanvasInteractivity(this.canvases[key], key);
                }
            });
        }

        setupCanvasInteractivity(canvas, type) {
            let isDragging = false;
            let lastPoint = null;

            canvas.addEventListener('mousedown', (e) => {
                isDragging = true;
                lastPoint = { x: e.offsetX, y: e.offsetY };
                this.handleCanvasClick(type, e.offsetX, e.offsetY);
            });

            canvas.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const currentPoint = { x: e.offsetX, y: e.offsetY };
                    this.handleCanvasDrag(type, lastPoint, currentPoint);
                    lastPoint = currentPoint;
                } else {
                    this.handleCanvasHover(type, e.offsetX, e.offsetY);
                }
            });

            canvas.addEventListener('mouseup', () => {
                isDragging = false;
                lastPoint = null;
            });

            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                this.handleCanvasZoom(type, e.offsetX, e.offsetY, e.deltaY);
            });
        }

        setupEventHandlers() {
            // Tab switching
            const tabButtons = this.analyticsUI.querySelectorAll('.tab-btn');
            tabButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    const tabId = btn.dataset.tab;
                    this.switchTab(tabId);
                });
            });

            // Control handlers
            document.getElementById('loadData')?.addEventListener('click', () => {
                this.loadInSARData();
            });

            document.getElementById('processVelocity')?.addEventListener('click', () => {
                this.processVelocityField();
            });

            document.getElementById('runPCA')?.addEventListener('click', () => {
                this.runPCAAnalysis();
            });

            document.getElementById('exportResults')?.addEventListener('click', () => {
                this.exportResults();
            });

            // Slider handlers
            const sliders = ['temporalRange', 'spatialRange', 'varianceThreshold'];
            sliders.forEach(sliderId => {
                const slider = document.getElementById(sliderId);
                if (slider) {
                    slider.addEventListener('input', (e) => {
                        const valueSpan = document.getElementById(sliderId.replace('Range', 'Value'));
                        if (valueSpan) valueSpan.textContent = e.target.value;
                        this.updateConfiguration(sliderId, e.target.value);
                    });
                }
            });
        }

        initializeWorkers() {
            this.workers = {
                velocity: this.createVelocityWorker(),
                pca: this.createPCAWorker(),
                coherence: this.createCoherenceWorker()
            };
        }

        createVelocityWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { interferograms, config } = e.data;
                    const result = processVelocityField(interferograms, config);
                    self.postMessage(result);
                };

                function processVelocityField(interferograms, config) {
                    const velocities = new Map();
                    const observations = new Map();
                    
                    // SBAS velocity estimation
                    interferograms.forEach((ifg, key) => {
                        const timeSpan = ifg.temporal_baseline / 365.25; // years
                        if (timeSpan === 0) return;
                        
                        for (let i = 0; i < ifg.phase.length; i++) {
                            if (ifg.coherence[i] < config.coherenceThreshold) continue;
                            
                            const pixelKey = \`\${i}\`;
                            if (!velocities.has(pixelKey)) {
                                velocities.set(pixelKey, []);
                                observations.set(pixelKey, []);
                            }
                            
                            const unwrappedPhase = unwrapPhase(ifg.phase[i]);
                            const losDisplacement = (unwrappedPhase * config.wavelength) / (4 * Math.PI);
                            const velocity = losDisplacement / timeSpan;
                            
                            velocities.get(pixelKey).push(velocity);
                            observations.get(pixelKey).push({
                                time: ifg.acquisition_time,
                                phase: unwrappedPhase,
                                coherence: ifg.coherence[i]
                            });
                        }
                    });
                    
                    // Calculate mean velocities and statistics
                    const results = {};
                    velocities.forEach((vels, pixelKey) => {
                        if (vels.length < config.minObservations) return;
                        
                        const meanVel = vels.reduce((a, b) => a + b, 0) / vels.length;
                        const stdVel = Math.sqrt(vels.reduce((acc, v) => acc + Math.pow(v - meanVel, 2), 0) / vels.length);
                        
                        // Remove outliers
                        const filteredVels = vels.filter(v => Math.abs(v - meanVel) < config.outlierThreshold * stdVel);
                        const finalMeanVel = filteredVels.reduce((a, b) => a + b, 0) / filteredVels.length;
                        
                        results[pixelKey] = {
                            velocity: finalMeanVel,
                            std: stdVel,
                            observations: filteredVels.length,
                            quality: filteredVels.length / vels.length
                        };
                    });
                    
                    return results;
                }
                
                function unwrapPhase(wrappedPhase) {
                    // Simple 1D phase unwrapping
                    let unwrapped = wrappedPhase;
                    while (unwrapped > Math.PI) unwrapped -= 2 * Math.PI;
                    while (unwrapped < -Math.PI) unwrapped += 2 * Math.PI;
                    return unwrapped;
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        createPCAWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { data, components } = e.data;
                    const result = performPCA(data, components);
                    self.postMessage(result);
                };

                function performPCA(data, numComponents) {
                    const n = data.length;
                    const m = data[0].length;
                    
                    // Center the data
                    const means = new Array(m).fill(0);
                    for (let j = 0; j < m; j++) {
                        for (let i = 0; i < n; i++) {
                            means[j] += data[i][j];
                        }
                        means[j] /= n;
                    }
                    
                    const centeredData = data.map(row => 
                        row.map((val, j) => val - means[j])
                    );
                    
                    // Compute covariance matrix
                    const covariance = computeCovariance(centeredData);
                    
                    // Eigendecomposition
                    const { eigenValues, eigenVectors } = eigenDecomposition(covariance);
                    
                    // Sort by eigenvalue magnitude
                    const sorted = eigenValues.map((val, idx) => ({ val, vec: eigenVectors[idx], idx }))
                        .sort((a, b) => b.val - a.val)
                        .slice(0, numComponents);
                    
                    // Calculate explained variance
                    const totalVariance = eigenValues.reduce((a, b) => a + b, 0);
                    const explainedVariance = sorted.map(s => s.val / totalVariance);
                    const cumulativeVariance = explainedVariance.reduce((acc, curr, idx) => {
                        acc.push(idx === 0 ? curr : acc[idx - 1] + curr);
                        return acc;
                    }, []);
                    
                    // Transform data
                    const transformedData = centeredData.map(row => 
                        sorted.map(s => dotProduct(row, s.vec))
                    );
                    
                    return {
                        eigenValues: sorted.map(s => s.val),
                        eigenVectors: sorted.map(s => s.vec),
                        explainedVariance,
                        cumulativeVariance,
                        transformedData,
                        means
                    };
                }
                
                function computeCovariance(data) {
                    const n = data.length;
                    const m = data[0].length;
                    const cov = Array(m).fill().map(() => Array(m).fill(0));
                    
                    for (let i = 0; i < m; i++) {
                        for (let j = 0; j < m; j++) {
                            let sum = 0;
                            for (let k = 0; k < n; k++) {
                                sum += data[k][i] * data[k][j];
                            }
                            cov[i][j] = sum / (n - 1);
                        }
                    }
                    return cov;
                }
                
                function eigenDecomposition(matrix) {
                    // Simplified Jacobi method for small matrices
                    const n = matrix.length;
                    const eigenVectors = Array(n).fill().map((_, i) => 
                        Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
                    );
                    const eigenValues = matrix.map(row => [...row]);
                    
                    // Power iteration approximation for demo
                    const vals = [];
                    const vecs = [];
                    
                    for (let k = 0; k < n; k++) {
                        let vec = Array(n).fill().map(() => Math.random());
                        let val = 0;
                        
                        for (let iter = 0; iter < 100; iter++) {
                            const newVec = matrixVectorProduct(eigenValues, vec);
                            val = dotProduct(vec, newVec) / dotProduct(vec, vec);
                            vec = normalize(newVec);
                        }
                        
                        vals.push(Math.abs(val));
                        vecs.push(vec);
                    }
                    
                    return { eigenValues: vals, eigenVectors: vecs };
                }
                
                function matrixVectorProduct(matrix, vector) {
                    return matrix.map(row => dotProduct(row, vector));
                }
                
                function dotProduct(a, b) {
                    return a.reduce((sum, val, i) => sum + val * b[i], 0);
                }
                
                function normalize(vector) {
                    const norm = Math.sqrt(dotProduct(vector, vector));
                    return vector.map(val => val / norm);
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        createCoherenceWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { interferogram } = e.data;
                    const result = analyzeCoherence(interferogram);
                    self.postMessage(result);
                };

                function analyzeCoherence(ifg) {
                    const coherence = ifg.coherence;
                    const phase = ifg.phase;
                    
                    const stats = {
                        mean: coherence.reduce((a, b) => a + b, 0) / coherence.length,
                        std: 0,
                        min: Math.min(...coherence),
                        max: Math.max(...coherence),
                        histogram: new Array(20).fill(0),
                        phaseStability: 0
                    };
                    
                    // Standard deviation
                    stats.std = Math.sqrt(
                        coherence.reduce((acc, val) => acc + Math.pow(val - stats.mean, 2), 0) / coherence.length
                    );
                    
                    // Histogram
                    coherence.forEach(val => {
                        const bin = Math.min(Math.floor(val * 20), 19);
                        stats.histogram[bin]++;
                    });
                    
                    // Phase stability metric
                    let stablePixels = 0;
                    for (let i = 0; i < coherence.length; i++) {
                        if (coherence[i] > 0.5 && Math.abs(phase[i]) < Math.PI / 2) {
                            stablePixels++;
                        }
                    }
                    stats.phaseStability = stablePixels / coherence.length;
                    
                    return stats;
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        async loadInSARData() {
            this.updateStatus('Loading InSAR data...', 10);
            
            try {
                // Simulate loading InSAR data
                const mockData = this.generateMockInSARData();
                this.data = mockData;
                
                // Process each interferogram
                for (let i = 0; i < mockData.interferograms.length; i++) {
                    const ifg = mockData.interferograms[i];
                    this.interferograms.set(ifg.id, ifg);
                    
                    this.updateStatus(`Processing interferogram ${i + 1}/${mockData.interferograms.length}...`, 
                                     10 + (i / mockData.interferograms.length) * 40);
                    
                    // Analyze coherence for this interferogram
                    await this.analyzeInterferogramCoherence(ifg);
                }
                
                this.updateNetworkVisualization();
                this.updateStatistics();
                this.updateStatus('Data loaded successfully', 50);
                
            } catch (error) {
                console.error('Error loading InSAR data:', error);
                this.updateStatus('Error loading data', 0);
            }
        }

        generateMockInSARData() {
            const numInterferograms = 20;
            const width = 100;
            const height = 100;
            const baseDate = new Date('2020-01-01');
            
            const interferograms = [];
            
            for (let i = 0; i < numInterferograms; i++) {
                const masterDate = new Date(baseDate.getTime() + i * 12 * 24 * 60 * 60 * 1000); // 12-day intervals
                const slaveDate = new Date(masterDate.getTime() + (24 + Math.random() * 360) * 24 * 60 * 60 * 1000);
                
                const phase = new Float32Array(width * height);
                const coherence = new Float32Array(width * height);
                const amplitude = new Float32Array(width * height);
                
                // Generate synthetic volcanic deformation pattern
                const centerX = width / 2 + (Math.random() - 0.5) * 10;
                const centerY = height / 2 + (Math.random() - 0.5) * 10;
                const maxDeformation = 0.05 + Math.random() * 0.1; // radians
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = y * width + x;
                        const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
                        
                        // Mogi-like deformation pattern
                        const deformation = maxDeformation * Math.exp(-dist / 20) * 
                                          Math.sin((slaveDate - masterDate) / (1000 * 60 * 60 * 24 * 365) * 2 * Math.PI);
                        
                        phase[idx] = deformation + (Math.random() - 0.5) * 0.2;
                        coherence[idx] = Math.max(0, Math.min(1, 0.8 - dist / 100 + (Math.random() - 0.5) * 0.3));
                        amplitude[idx] = 1000 + Math.random() * 500;
                    }
                }
                
                interferograms.push({
                    id: `ifg_${i.toString().padStart(3, '0')}`,
                    master_date: masterDate.toISOString().split('T')[0],
                    slave_date: slaveDate.toISOString().split('T')[0],
                    temporal_baseline: (slaveDate - masterDate) / (1000 * 60 * 60 * 24),
                    spatial_baseline: 50 + Math.random() * 400,
                    acquisition_time: masterDate,
                    phase: phase,
                    coherence: coherence,
                    amplitude: amplitude,
                    width: width,
                    height: height
                });
            }
            
            return {
                interferograms: interferograms,
                metadata: {
                    sensor: 'Sentinel-1',
                    wavelength: 5.6,
                    incidence_angle: 23,
                    processing_level: 'L2',
                    area: 'Volcanic Test Site'
                }
            };
        }

        async analyzeInterferogramCoherence(interferogram) {
            return new Promise((resolve) => {
                this.workers.coherence.onmessage = (e) => {
                    interferogram.coherence_stats = e.data;
                    resolve(e.data);
                };
                
                this.workers.coherence.postMessage({ interferogram });
            });
        }

        async processVelocityField() {
            if (!this.data || this.interferograms.size === 0) {
                alert('Please load InSAR data first');
                return;
            }

            this.updateStatus('Processing velocity field...', 50);

            try {
                const interferogramArray = Array.from(this.interferograms.values());
                
                const result = await new Promise((resolve) => {
                    this.workers.velocity.onmessage = (e) => {
                        resolve(e.data);
                    };
                    
                    this.workers.velocity.postMessage({
                        interferograms: interferogramArray,
                        config: this.config
                    });
                });

                this.velocityField = result;
                this.visualizeVelocityField();
                this.updateVelocityStatistics();
                this.updateStatus('Velocity field processed', 75);

            } catch (error) {
                console.error('Error processing velocity field:', error);
                this.updateStatus('Error processing velocity', 50);
            }
        }

        visualizeVelocityField() {
            if (!this.velocityField) return;

            const canvas = this.canvases.velocity;
            const ctx = this.contexts.velocity;
            
            if (!canvas || !ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Get velocity values and statistics
            const velocities = Object.values(this.velocityField).map(v => v.velocity);
            const minVel = Math.min(...velocities);
            const maxVel = Math.max(...velocities);
            const velRange = maxVel - minVel;

            // Set up color map
            const colorMap = this.createColorMap('velocity', minVel, maxVel);

            // Draw velocity field as heatmap
            const gridSize = Math.sqrt(Object.keys(this.velocityField).length);
            const cellWidth = canvas.width / gridSize;
            const cellHeight = canvas.height / gridSize;

            Object.entries(this.velocityField).forEach(([pixelKey, data]) => {
                const idx = parseInt(pixelKey);
                const x = idx % gridSize;
                const y = Math.floor(idx / gridSize);
                
                const normalizedVel = (data.velocity - minVel) / velRange;
                const color = colorMap(normalizedVel);
                
                ctx.fillStyle = color;
                ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
            });

            // Add color bar
            this.drawColorBar(ctx, canvas, colorMap, minVel, maxVel, 'mm/year');

            // Add contour lines for major velocity zones
            this.drawVelocityContours(ctx, canvas, this.velocityField, minVel, maxVel);
        }

        createColorMap(type, min, max) {
            const colors = {
                velocity: ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'],
                coherence: ['#000000', '#404040', '#808080', '#C0C0C0', '#FFFFFF'],
                phase: ['#FF0000', '#FF8000', '#FFFF00', '#00FF00', '#0080FF', '#0000FF', '#8000FF', '#FF0080']
            };

            const colorArray = colors[type] || colors.velocity;
            
            return function(normalized) {
                const scaled = Math.max(0, Math.min(1, normalized));
                const index = scaled * (colorArray.length - 1);
                const lower = Math.floor(index);
                const upper = Math.ceil(index);
                const fraction = index - lower;

                if (lower === upper) {
                    return colorArray[lower];
                }

                const lowerColor = hexToRgb(colorArray[lower]);
                const upperColor = hexToRgb(colorArray[upper]);

                const r = Math.round(lowerColor.r + (upperColor.r - lowerColor.r) * fraction);
                const g = Math.round(lowerColor.g + (upperColor.g - lowerColor.g) * fraction);
                const b = Math.round(lowerColor.b + (upperColor.b - lowerColor.b) * fraction);

                return `rgb(${r}, ${g}, ${b})`;
            };

            function hexToRgb(hex) {
                const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                return result ? {
                    r: parseInt(result[1], 16),
                    g: parseInt(result[2], 16),
                    b: parseInt(result[3], 16)
                } : null;
            }
        }

        drawColorBar(ctx, canvas, colorMap, min, max, unit) {
            const barWidth = 20;
            const barHeight = 200;
            const barX = canvas.width - barWidth - 10;
            const barY = 50;

            // Draw color bar
            for (let i = 0; i < barHeight; i++) {
                const normalized = 1 - i / barHeight;
                ctx.fillStyle = colorMap(normalized);
                ctx.fillRect(barX, barY + i, barWidth, 1);
            }

            // Draw border
            ctx.strokeStyle = '#000000';
            ctx.strokeRect(barX, barY, barWidth, barHeight);

            // Draw labels
            ctx.fillStyle = '#000000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'left';
            
            const steps = 5;
            for (let i = 0; i <= steps; i++) {
                const y = barY + (i / steps) * barHeight;
                const value = max - (i / steps) * (max - min);
                ctx.fillText(value.toFixed(1), barX + barWidth + 5, y + 3);
            }

            // Unit label
            ctx.save();
            ctx.translate(barX + barWidth + 40, barY + barHeight / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillText(unit, 0, 0);
            ctx.restore();
        }

        drawVelocityContours(ctx, canvas, velocityField, min, max) {
            // Simplified contour drawing
            const contourLevels = 5;
            const levelStep = (max - min) / contourLevels;

            ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.lineWidth = 1;

            for (let level = 1; level < contourLevels; level++) {
                const contourValue = min + level * levelStep;
                this.drawContourLevel(ctx, canvas, velocityField, contourValue);
            }
        }

        drawContourLevel(ctx, canvas, velocityField, value) {
            // Marching squares algorithm (simplified)
            const gridSize = Math.sqrt(Object.keys(velocityField).length);
            const cellWidth = canvas.width / gridSize;
            const cellHeight = canvas.height / gridSize;

            ctx.beginPath();
            
            for (let y = 0; y < gridSize - 1; y++) {
                for (let x = 0; x < gridSize - 1; x++) {
                    const tl = velocityField[y * gridSize + x]?.velocity || 0;
                    const tr = velocityField[y * gridSize + x + 1]?.velocity || 0;
                    const bl = velocityField[(y + 1) * gridSize + x]?.velocity || 0;
                    const br = velocityField[(y + 1) * gridSize + x + 1]?.velocity || 0;

                    // Simplified contour detection
                    if ((tl > value && br < value) || (tl < value && br > value)) {
                        ctx.moveTo(x * cellWidth, y * cellHeight);
                        ctx.lineTo((x + 1) * cellWidth, (y + 1) * cellHeight);
                    }
                    if ((tr > value && bl < value) || (tr < value && bl > value)) {
                        ctx.moveTo((x + 1) * cellWidth, y * cellHeight);
                        ctx.lineTo(x * cellWidth, (y + 1) * cellHeight);
                    }
                }
            }
            
            ctx.stroke();
        }

        async runPCAAnalysis() {
            if (!this.data || this.interferograms.size === 0) {
                alert('Please load InSAR data first');
                return;
            }

            this.updateStatus('Running PCA analysis...', 75);

            try {
                // Prepare data matrix (observations x features)
                const interferogramArray = Array.from(this.interferograms.values());
                const dataMatrix = interferogramArray.map(ifg => Array.from(ifg.phase));
                
                const numComponents = parseInt(document.getElementById('pcaComponents')?.value || this.config.pcaComponents);

                const result = await new Promise((resolve) => {
                    this.workers.pca.onmessage = (e) => {
                        resolve(e.data);
                    };
                    
                    this.workers.pca.postMessage({
                        data: dataMatrix,
                        components: numComponents
                    });
                });

                this.pcaResults = result;
                this.visualizePCAResults();
                this.updateStatus('PCA analysis complete', 90);

            } catch (error) {
                console.error('Error running PCA analysis:', error);
                this.updateStatus('Error running PCA', 75);
            }
        }

        visualizePCAResults() {
            if (!this.pcaResults) return;

            this.drawScreePlot();
            this.drawLoadingsPlot();
            this.updatePCAComponents();
        }

        drawScreePlot() {
            const canvas = this.canvases.scree;
            const ctx = this.contexts.scree;
            
            if (!canvas || !ctx || !this.pcaResults) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const eigenValues = this.pcaResults.eigenValues;
            const explainedVar = this.pcaResults.explainedVariance;

            // Set up plot area
            const margin = { top: 20, right: 20, bottom: 40, left: 50 };
            const plotWidth = canvas.width - margin.left - margin.right;
            const plotHeight = canvas.height - margin.top - margin.bottom;

            // Draw axes
            ctx.strokeStyle = '#000000';
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, canvas.height - margin.bottom);
            ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
            ctx.stroke();

            // Draw bars
            const barWidth = plotWidth / eigenValues.length * 0.8;
            const maxVar = Math.max(...explainedVar);

            ctx.fillStyle = '#4CAF50';
            explainedVar.forEach((variance, i) => {
                const x = margin.left + (i + 0.1) * (plotWidth / eigenValues.length);
                const height = (variance / maxVar) * plotHeight;
                const y = canvas.height - margin.bottom - height;
                
                ctx.fillRect(x, y, barWidth, height);
            });

            // Add labels
            ctx.fillStyle = '#000000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';

            eigenValues.forEach((_, i) => {
                const x = margin.left + (i + 0.5) * (plotWidth / eigenValues.length);
                ctx.fillText(`PC${i + 1}`, x, canvas.height - 10);
            });

            // Y-axis labels
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {
                const y = canvas.height - margin.bottom - (i / 5) * plotHeight;
                const value = (i / 5) * maxVar;
                ctx.fillText(value.toFixed(2), margin.left - 10, y + 3);
            }

            // Title
            ctx.textAlign = 'center';
            ctx.font = '14px Arial';
            ctx.fillText('Explained Variance Ratio', canvas.width / 2, 15);
        }

        drawLoadingsPlot() {
            const canvas = this.canvases.loadings;
            const ctx = this.contexts.loadings;
            
            if (!canvas || !ctx || !this.pcaResults) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const loadings = this.pcaResults.eigenVectors;
            if (loadings.length < 2) return;

            // Plot first two components
            const pc1 = loadings[0];
            const pc2 = loadings[1];

            const margin = { top: 20, right: 20, bottom: 40, left: 50 };
            const plotWidth = canvas.width - margin.left - margin.right;
            const plotHeight = canvas.height - margin.top - margin.bottom;

            // Find ranges
            const pc1Range = [Math.min(...pc1), Math.max(...pc1)];
            const pc2Range = [Math.min(...pc2), Math.max(...pc2)];

            // Draw axes
            ctx.strokeStyle = '#000000';
            ctx.beginPath();
            
            // Zero lines
            const zeroX = margin.left + plotWidth / 2;
            const zeroY = margin.top + plotHeight / 2;
            
            ctx.moveTo(margin.left, zeroY);
            ctx.lineTo(canvas.width - margin.right, zeroY);
            ctx.moveTo(zeroX, margin.top);
            ctx.lineTo(zeroX, canvas.height - margin.bottom);
            ctx.stroke();

            // Plot loadings as points
            ctx.fillStyle = '#2196F3';
            pc1.forEach((val1, i) => {
                if (i >= pc2.length) return;
                
                const val2 = pc2[i];
                const x = margin.left + ((val1 - pc1Range[0]) / (pc1Range[1] - pc1Range[0])) * plotWidth;
                const y = canvas.height - margin.bottom - ((val2 - pc2Range[0]) / (pc2Range[1] - pc2Range[0])) * plotHeight;
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });

            // Labels
            ctx.fillStyle = '#000000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('PC1 Loadings', canvas.width / 2, canvas.height - 10);
            
            ctx.save();
            ctx.translate(15, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('PC2 Loadings', 0, 0);
            ctx.restore();
        }

        updatePCAComponents() {
            const componentsList = document.getElementById('componentsList');
            if (!componentsList || !this.pcaResults) return;

            const components = this.pcaResults.explainedVariance;
            const cumulative = this.pcaResults.cumulativeVariance;

            let html = '<div class="components-table">';
            html += '<div class="table-header">';
            html += '<span>Component</span><span>Variance</span><span>Cumulative</span>';
            html += '</div>';

            components.forEach((variance, i) => {
                html += '<div class="table-row">';
                html += `<span>PC${i + 1}</span>`;
                html += `<span>${(variance * 100).toFixed(1)}%</span>`;
                html += `<span>${(cumulative[i] * 100).toFixed(1)}%</span>`;
                html += '</div>';
            });

            html += '</div>';
            componentsList.innerHTML = html;
        }

        updateNetworkVisualization() {
            const canvas = this.canvases.network;
            const ctx = this.contexts.network;
            
            if (!canvas || !ctx || this.interferograms.size === 0) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Create network from interferograms
            const nodes = new Map();
            const edges = [];

            // Create nodes from acquisition dates
            this.interferograms.forEach((ifg) => {
                if (!nodes.has(ifg.master_date)) {
                    nodes.set(ifg.master_date, {
                        id: ifg.master_date,
                        date: new Date(ifg.master_date),
                        x: Math.random() * canvas.width,
                        y: Math.random() * canvas.height,
                        vx: 0,
                        vy: 0
                    });
                }
                if (!nodes.has(ifg.slave_date)) {
                    nodes.set(ifg.slave_date, {
                        id: ifg.slave_date,
                        date: new Date(ifg.slave_date),
                        x: Math.random() * canvas.width,
                        y: Math.random() * canvas.height,
                        vx: 0,
                        vy: 0
                    });
                }

                edges.push({
                    source: ifg.master_date,
                    target: ifg.slave_date,
                    weight: ifg.coherence_stats?.mean || 0.5,
                    temporal_baseline: ifg.temporal_baseline,
                    spatial_baseline: ifg.spatial_baseline
                });
            });

            // Force-directed layout simulation
            this.simulateNetworkLayout(Array.from(nodes.values()), edges);

            // Draw edges
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.6)';
            edges.forEach(edge => {
                const source = nodes.get(edge.source);
                const target = nodes.get(edge.target);
                
                if (source && target) {
                    ctx.lineWidth = edge.weight * 3;
                    ctx.beginPath();
                    ctx.moveTo(source.x, source.y);
                    ctx.lineTo(target.x, target.y);
                    ctx.stroke();
                }
            });

            // Draw nodes
            nodes.forEach(node => {
                ctx.fillStyle = '#2196F3';
                ctx.beginPath();
                ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
                ctx.fill();

                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Date label
                ctx.fillStyle = '#000000';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(node.date.getFullYear().toString(), node.x, node.y + 20);
            });

            // Update network info
            document.getElementById('nodeCount').textContent = nodes.size;
            document.getElementById('edgeCount').textContent = edges.length;
            document.getElementById('networkDensity').textContent = 
                (2 * edges.length / (nodes.size * (nodes.size - 1))).toFixed(3);
        }

        simulateNetworkLayout(nodes, edges) {
            const iterations = 50;
            const k = Math.sqrt((this.canvases.network.width * this.canvases.network.height) / nodes.length);
            
            for (let iter = 0; iter < iterations; iter++) {
                // Repulsive forces
                nodes.forEach(node1 => {
                    node1.vx = 0;
                    node1.vy = 0;
                    
                    nodes.forEach(node2 => {
                        if (node1 === node2) return;
                        
                        const dx = node1.x - node2.x;
                        const dy = node1.y - node2.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        
                        if (distance > 0) {
                            const force = k * k / distance;
                            node1.vx += (dx / distance) * force;
                            node1.vy += (dy / distance) * force;
                        }
                    });
                });

                // Attractive forces
                edges.forEach(edge => {
                    const source = nodes.find(n => n.id === edge.source);
                    const target = nodes.find(n => n.id === edge.target);
                    
                    if (source && target) {
                        const dx = target.x - source.x;
                        const dy = target.y - source.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        
                        if (distance > 0) {
                            const force = distance * distance / k;
                            const fx = (dx / distance) * force * edge.weight;
                            const fy = (dy / distance) * force * edge.weight;
                            
                            source.vx += fx;
                            source.vy += fy;
                            target.vx -= fx;
                            target.vy -= fy;
                        }
                    }
                });

                // Update positions
                nodes.forEach(node => {
                    const velocity = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
                    if (velocity > 0) {
                        const temp = Math.min(velocity, 10) / velocity;
                        node.x += node.vx * temp;
                        node.y += node.vy * temp;
                    }
                    
                    // Keep within bounds
                    node.x = Math.max(20, Math.min(this.canvases.network.width - 20, node.x));
                    node.y = Math.max(20, Math.min(this.canvases.network.height - 20, node.y));
                });
            }
        }

        updateVelocityStatistics() {
            if (!this.velocityField) return;

            const velocities = Object.values(this.velocityField).map(v => v.velocity);
            const mean = velocities.reduce((a, b) => a + b, 0) / velocities.length;
            const std = Math.sqrt(velocities.reduce((acc, v) => acc + (v - mean) ** 2, 0) / velocities.length);
            const max = Math.max(...velocities);

            document.getElementById('meanVel').textContent = `${mean.toFixed(2)} mm/year`;
            document.getElementById('stdVel').textContent = `${std.toFixed(2)} mm/year`;
            document.getElementById('maxVel').textContent = `${max.toFixed(2)} mm/year`;
        }

        updateStatistics() {
            if (this.interferograms.size === 0) return;

            // Compute overall statistics
            let totalCoherence = 0;
            let totalPixels = 0;
            let phaseStability = 0;
            let temporalConsistency = 0;

            this.interferograms.forEach(ifg => {
                if (ifg.coherence_stats) {
                    const pixels = ifg.coherence.length;
                    totalCoherence += ifg.coherence_stats.mean * pixels;
                    totalPixels += pixels;
                    phaseStability += ifg.coherence_stats.phaseStability;
                }
            });

            const meanCoherence = totalPixels > 0 ? totalCoherence / totalPixels : 0;
            phaseStability /= this.interferograms.size;
            
            // Compute temporal consistency
            const datesSorted = Array.from(this.interferograms.values())
                .map(ifg => new Date(ifg.master_date))
                .sort((a, b) => a - b);
            
            let consistentPairs = 0;
            let totalPairs = 0;
            
            for (let i = 0; i < datesSorted.length - 1; i++) {
                for (let j = i + 1; j < datesSorted.length; j++) {
                    const daysDiff = (datesSorted[j] - datesSorted[i]) / (1000 * 60 * 60 * 24);
                    if (daysDiff <= this.config.temporalBaseline) {
                        consistentPairs++;
                    }
                    totalPairs++;
                }
            }
            
            temporalConsistency = totalPairs > 0 ? consistentPairs / totalPairs : 0;

            // Update UI
            document.getElementById('meanCoherence').textContent = meanCoherence.toFixed(3);
            document.getElementById('phaseStability').textContent = phaseStability.toFixed(3);
            document.getElementById('temporalConsistency').textContent = temporalConsistency.toFixed(3);

            // Draw histograms
            this.drawHistograms();
        }

        drawHistograms() {
            this.drawCoherenceHistogram();
            this.drawPhaseHistogram();
            this.drawVelocityHistogram();
        }

        drawCoherenceHistogram() {
            const canvas = this.canvases.coherenceHist;
            const ctx = this.contexts.coherenceHist;
            
            if (!canvas || !ctx || this.interferograms.size === 0) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Collect coherence data
            const coherenceData = [];
            this.interferograms.forEach(ifg => {
                coherenceData.push(...Array.from(ifg.coherence));
            });

            // Create histogram
            const bins = 20;
            const histogram = new Array(bins).fill(0);
            const binWidth = 1.0 / bins;

            coherenceData.forEach(val => {
                const bin = Math.min(Math.floor(val / binWidth), bins - 1);
                histogram[bin]++;
            });

            // Draw histogram
            const maxCount = Math.max(...histogram);
            const barWidth = canvas.width / bins;
            const maxHeight = canvas.height - 30;

            ctx.fillStyle = '#2196F3';
            histogram.forEach((count, i) => {
                const height = (count / maxCount) * maxHeight;
                const x = i * barWidth;
                const y = canvas.height - height - 15;
                
                ctx.fillRect(x, y, barWidth * 0.8, height);
            });

            // Add labels
            ctx.fillStyle = '#000000';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            
            for (let i = 0; i <= 5; i++) {
                const x = (i / 5) * canvas.width;
                const value = i / 5;
                ctx.fillText(value.toFixed(1), x, canvas.height - 5);
            }
        }

        drawPhaseHistogram() {
            const canvas = this.canvases.phaseHist;
            const ctx = this.contexts.phaseHist;
            
            if (!canvas || !ctx || this.interferograms.size === 0) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Collect phase data
            const phaseData = [];
            this.interferograms.forEach(ifg => {
                phaseData.push(...Array.from(ifg.phase));
            });

            // Create histogram
            const bins = 20;
            const histogram = new Array(bins).fill(0);
            const minPhase = -Math.PI;
            const maxPhase = Math.PI;
            const binWidth = (maxPhase - minPhase) / bins;

            phaseData.forEach(val => {
                const bin = Math.min(Math.floor((val - minPhase) / binWidth), bins - 1);
                if (bin >= 0) histogram[bin]++;
            });

            // Draw histogram
            const maxCount = Math.max(...histogram);
            const barWidth = canvas.width / bins;
            const maxHeight = canvas.height - 30;

            ctx.fillStyle = '#FF9800';
            histogram.forEach((count, i) => {
                const height = (count / maxCount) * maxHeight;
                const x = i * barWidth;
                const y = canvas.height - height - 15;
                
                ctx.fillRect(x, y, barWidth * 0.8, height);
            });

            // Add labels
            ctx.fillStyle = '#000000';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            
            for (let i = 0; i <= 4; i++) {
                const x = (i / 4) * canvas.width;
                const value = minPhase + (i / 4) * (maxPhase - minPhase);
                ctx.fillText(value.toFixed(1), x, canvas.height - 5);
            }
        }

        drawVelocityHistogram() {
            const canvas = this.canvases.velocityHist;
            const ctx = this.contexts.velocityHist;
            
            if (!canvas || !ctx || !this.velocityField) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const velocities = Object.values(this.velocityField).map(v => v.velocity);
            const minVel = Math.min(...velocities);
            const maxVel = Math.max(...velocities);

            // Create histogram
            const bins = 20;
            const histogram = new Array(bins).fill(0);
            const binWidth = (maxVel - minVel) / bins;

            velocities.forEach(val => {
                const bin = Math.min(Math.floor((val - minVel) / binWidth), bins - 1);
                if (bin >= 0) histogram[bin]++;
            });

            // Draw histogram
            const maxCount = Math.max(...histogram);
            const barWidth = canvas.width / bins;
            const maxHeight = canvas.height - 30;

            ctx.fillStyle = '#4CAF50';
            histogram.forEach((count, i) => {
                const height = (count / maxCount) * maxHeight;
                const x = i * barWidth;
                const y = canvas.height - height - 15;
                
                ctx.fillRect(x, y, barWidth * 0.8, height);
            });

            // Add labels
            ctx.fillStyle = '#000000';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            
            for (let i = 0; i <= 4; i++) {
                const x = (i / 4) * canvas.width;
                const value = minVel + (i / 4) * (maxVel - minVel);
                ctx.fillText(value.toFixed(1), x, canvas.height - 5);
            }
        }

        handleCanvasClick(type, x, y) {
            console.log(`Clicked ${type} canvas at (${x}, ${y})`);
            
            if (type === 'velocity' && this.velocityField) {
                this.selectPixelForAnalysis(x, y);
            } else if (type === 'network') {
                this.selectNetworkNode(x, y);
            }
        }

        handleCanvasHover(type, x, y) {
            // Show tooltips on hover
            if (type === 'velocity' && this.velocityField) {
                this.showVelocityTooltip(x, y);
            }
        }

        handleCanvasDrag(type, start, end) {
            if (type === 'network') {
                // Pan network view
                this.panNetworkView(start, end);
            }
        }

        handleCanvasZoom(type, x, y, delta) {
            if (type === 'velocity') {
                this.zoomVelocityView(x, y, delta);
            } else if (type === 'network') {
                this.zoomNetworkView(x, y, delta);
            }
        }

        selectPixelForAnalysis(x, y) {
            // Convert canvas coordinates to pixel indices
            const canvas = this.canvases.velocity;
            const gridSize = Math.sqrt(Object.keys(this.velocityField).length);
            
            const pixelX = Math.floor((x / canvas.width) * gridSize);
            const pixelY = Math.floor((y / canvas.height) * gridSize);
            const pixelKey = (pixelY * gridSize + pixelX).toString();
            
            if (this.velocityField[pixelKey]) {
                const data = this.velocityField[pixelKey];
                this.showPixelAnalysis(pixelKey, data);
            }
        }

        showPixelAnalysis(pixelKey, data) {
            const analysis = `
                <div class="pixel-analysis">
                    <h4>Pixel Analysis</h4>
                    <p><strong>Pixel:</strong> ${pixelKey}</p>
                    <p><strong>Velocity:</strong> ${data.velocity.toFixed(2)} mm/year</p>
                    <p><strong>Std Dev:</strong> ${data.std.toFixed(2)} mm/year</p>
                    <p><strong>Observations:</strong> ${data.observations}</p>
                    <p><strong>Quality:</strong> ${(data.quality * 100).toFixed(1)}%</p>
                </div>
            `;
            
            // Show in a popup or update a panel
            this.showTooltip(analysis);
        }

        showTooltip(content) {
            let tooltip = document.getElementById('analyticsTooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'analyticsTooltip';
                tooltip.className = 'analytics-tooltip';
                document.body.appendChild(tooltip);
            }
            
            tooltip.innerHTML = content;
            tooltip.style.display = 'block';
            
            // Position tooltip
            tooltip.style.left = event.pageX + 10 + 'px';
            tooltip.style.top = event.pageY - 50 + 'px';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                tooltip.style.display = 'none';
            }, 5000);
        }

        switchTab(tabId) {
            // Update tab buttons
            this.analyticsUI.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tab === tabId);
            });
            
            // Update tab content
            this.analyticsUI.querySelectorAll('.tab-content').forEach(content => {
                content.classList.toggle('active', content.id === `${tabId}-tab`);
            });
        }

        updateConfiguration(param, value) {
            const numValue = parseFloat(value);
            
            switch (param) {
                case 'temporalRange':
                    this.config.temporalBaseline = numValue;
                    break;
                case 'spatialRange':
                    this.config.spatialBaseline = numValue;
                    break;
                case 'varianceThreshold':
                    this.config.varianceThreshold = numValue;
                    break;
            }
            
            // Re-process if data is available
            if (this.data && (param === 'temporalRange' || param === 'spatialRange')) {
                this.processVelocityField();
            }
        }

        updateStatus(message, progress = null) {
            const statusText = document.getElementById('statusText');
            const progressFill = document.getElementById('progressFill');
            
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
                velocityField: this.velocityField,
                pcaResults: this.pcaResults,
                interferograms: Array.from(this.interferograms.entries()),
                timestamp: new Date().toISOString()
            };

            // Create downloadable file
            const blob = new Blob([JSON.stringify(results, null, 2)], { 
                type: 'application/json' 
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `insar_analytics_results_${Date.now()}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
            
            console.log('📊 Results exported successfully');
        }

        destroy() {
            // Clean up workers
            Object.values(this.workers).forEach(worker => {
                worker.terminate();
            });
            
            // Remove UI
            if (this.analyticsUI) {
                this.analyticsUI.remove();
            }
            
            console.log('🧹 InSAR Analytics Engine destroyed');
        }
    }

    // Supporting classes for processing algorithms
    class SBASProcessor {
        process(interferograms, config) {
            // Small Baseline Subset processing
            const timeSeriesLength = interferograms.length;
            const results = new Map();
            
            interferograms.forEach((ifg, index) => {
                // Process each interferogram for SBAS
                const pixelCount = ifg.phase.length;
                
                for (let i = 0; i < pixelCount; i++) {
                    const pixelKey = `pixel_${i}`;
                    
                    if (!results.has(pixelKey)) {
                        results.set(pixelKey, {
                            timeSeries: new Array(timeSeriesLength).fill(0),
                            velocity: 0,
                            quality: 0
                        });
                    }
                    
                    const pixel = results.get(pixelKey);
                    pixel.timeSeries[index] = ifg.phase[i];
                    pixel.quality += ifg.coherence[i];
                }
            });
            
            // Calculate velocities from time series
            results.forEach((pixel, key) => {
                pixel.quality /= interferograms.length;
                
                // Simple linear regression for velocity
                const n = pixel.timeSeries.length;
                const sumX = (n * (n - 1)) / 2;
                const sumY = pixel.timeSeries.reduce((a, b) => a + b, 0);
                const sumXY = pixel.timeSeries.reduce((acc, val, i) => acc + val * i, 0);
                const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
                
                pixel.velocity = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            });
            
            return results;
        }
    }

    class PSInSARProcessor {
        process(interferograms, config) {
            // Persistent Scatterer processing
            const results = new Map();
            const pixelCount = interferograms[0]?.phase.length || 0;
            
            for (let i = 0; i < pixelCount; i++) {
                const pixelKey = `pixel_${i}`;
                const amplitudes = interferograms.map(ifg => ifg.amplitude[i]);
                const phases = interferograms.map(ifg => ifg.phase[i]);
                const coherences = interferograms.map(ifg => ifg.coherence[i]);
                
                // Calculate amplitude stability index
                const meanAmp = amplitudes.reduce((a, b) => a + b, 0) / amplitudes.length;
                const stdAmp = Math.sqrt(
                    amplitudes.reduce((acc, amp) => acc + (amp - meanAmp) ** 2, 0) / amplitudes.length
                );
                const stabilityIndex = stdAmp / meanAmp;
                
                // Only consider pixels with high stability
                if (stabilityIndex < config.psThreshold || 0.2) {
                    const meanCoherence = coherences.reduce((a, b) => a + b, 0) / coherences.length;
                    
                    results.set(pixelKey, {
                        stabilityIndex: stabilityIndex,
                        meanCoherence: meanCoherence,
                        phases: phases,
                        isPS: true
                    });
                }
            }
            
            return results;
        }
    }

    class StatisticalAnalyzer {
        calculateStatistics(data) {
            const values = Array.isArray(data) ? data : Object.values(data);
            const n = values.length;
            
            if (n === 0) return null;
            
            const mean = values.reduce((a, b) => a + b, 0) / n;
            const variance = values.reduce((acc, val) => acc + (val - mean) ** 2, 0) / (n - 1);
            const std = Math.sqrt(variance);
            
            const sorted = [...values].sort((a, b) => a - b);
            const median = n % 2 === 0 ? 
                (sorted[n/2 - 1] + sorted[n/2]) / 2 : 
                sorted[Math.floor(n/2)];
            
            const q1 = sorted[Math.floor(n * 0.25)];
            const q3 = sorted[Math.floor(n * 0.75)];
            
            return {
                count: n,
                mean: mean,
                median: median,
                std: std,
                variance: variance,
                min: sorted[0],
                max: sorted[n-1],
                q1: q1,
                q3: q3,
                iqr: q3 - q1
            };
        }
        
        detectOutliers(data, method = 'iqr') {
            const stats = this.calculateStatistics(data);
            if (!stats) return [];
            
            const outliers = [];
            
            if (method === 'iqr') {
                const lowerBound = stats.q1 - 1.5 * stats.iqr;
                const upperBound = stats.q3 + 1.5 * stats.iqr;
                
                data.forEach((value, index) => {
                    if (value < lowerBound || value > upperBound) {
                        outliers.push({ index, value, method: 'iqr' });
                    }
                });
            } else if (method === 'zscore') {
                const threshold = 2.5;
                
                data.forEach((value, index) => {
                    const zscore = Math.abs(value - stats.mean) / stats.std;
                    if (zscore > threshold) {
                        outliers.push({ index, value, zscore, method: 'zscore' });
                    }
                });
            }
            
            return outliers;
        }
    }

    class NetworkAnalyzer {
        analyzeNetwork(nodes, edges) {
            const nodeCount = nodes.length;
            const edgeCount = edges.length;
            const maxEdges = nodeCount * (nodeCount - 1) / 2;
            
            const density = maxEdges > 0 ? edgeCount / maxEdges : 0;
            
            // Calculate degree for each node
            const degrees = new Map();
            nodes.forEach(node => degrees.set(node.id, 0));
            
            edges.forEach(edge => {
                degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
                degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
            });
            
            const degreeValues = Array.from(degrees.values());
            const meanDegree = degreeValues.reduce((a, b) => a + b, 0) / nodeCount;
            
            // Find connected components
            const components = this.findConnectedComponents(nodes, edges);
            
            return {
                nodeCount,
                edgeCount,
                density,
                meanDegree,
                components: components.length,
                largestComponent: Math.max(...components.map(c => c.length))
            };
        }
        
        findConnectedComponents(nodes, edges) {
            const visited = new Set();
            const components = [];
            const adjacency = new Map();
            
            // Build adjacency list
            nodes.forEach(node => adjacency.set(node.id, []));
            edges.forEach(edge => {
                adjacency.get(edge.source)?.push(edge.target);
                adjacency.get(edge.target)?.push(edge.source);
            });
            
            // DFS to find components
            const dfs = (nodeId, component) => {
                if (visited.has(nodeId)) return;
                
                visited.add(nodeId);
                component.push(nodeId);
                
                const neighbors = adjacency.get(nodeId) || [];
                neighbors.forEach(neighbor => {
                    if (!visited.has(neighbor)) {
                        dfs(neighbor, component);
                    }
                });
            };
            
            nodes.forEach(node => {
                if (!visited.has(node.id)) {
                    const component = [];
                    dfs(node.id, component);
                    components.push(component);
                }
            });
            
            return components;
        }
    }

    // Create global instance
    window.InSARAnalytics = InSARAnalytics;
    
    // Auto-initialize if container exists
    const analyticsContainer = document.querySelector('.insar-analytics-mount');
    if (analyticsContainer) {
        window.insarAnalytics = new InSARAnalytics({
            container: analyticsContainer
        });
    }
    
    console.log('📊 InSAR Analytics Engine loaded successfully');

})();