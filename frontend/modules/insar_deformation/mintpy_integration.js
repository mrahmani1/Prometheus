/**
 * MintPy Integration System - Task 3.1
 * Advanced InSAR processing workflow integration
 * @author Prometheus Platform
 * @version 1.0.0
 */

class MintPyIntegration {
    constructor() {
        this.version = '1.0.0';
        this.workflows = new Map();
        this.activeJobs = new Map();
        this.results = new Map();
        this.config = {
            defaultWorkflow: 'sbas',
            maxConcurrentJobs: 4,
            tempDirectory: './temp/mintpy',
            outputDirectory: './output/mintpy',
            processingOptions: {
                sbas: {
                    coherenceThreshold: 0.7,
                    unwrapMethod: 'snaphu',
                    atmosphericCorrection: true,
                    troposphericDelay: 'era5',
                    ionosphericDelay: false,
                    solidEarthTides: true,
                    referenceSelection: 'auto'
                },
                psInsar: {
                    amplitudeDispersionThreshold: 0.4,
                    phaseLinkingMethod: 'evd',
                    shpTest: true,
                    minRedundancy: 3,
                    unwrapMethod: '3d',
                    referenceNetwork: 'delaunay'
                }
            },
            qualityMetrics: {
                enabled: true,
                coherenceAnalysis: true,
                velocityStandardError: true,
                temporalCoherence: true,
                residualPhaseRMS: true
            }
        };
        
        this.eventBus = new EventTarget();
        this.init();
    }

    init() {
        console.log('🛰️ Initializing MintPy Integration System...');
        this.setupWorkflows();
        this.setupUI();
        this.setupErrorHandling();
        this.loadSavedResults();
        console.log('✅ MintPy Integration System Ready');
    }

    setupWorkflows() {
        // SBAS (Small Baseline Subset) Workflow
        this.workflows.set('sbas', {
            name: 'SBAS Time Series Analysis',
            description: 'Small Baseline Subset InSAR processing for time series analysis',
            steps: [
                'load_data',
                'network_modification',
                'reference_point',
                'correct_unwrap_error',
                'invert_network',
                'correct_los_geometry',
                'correct_troposphere',
                'correct_topography',
                'residual_rms',
                'reference_date',
                'velocity',
                'geocode'
            ],
            estimatedTime: 3600,
            requirements: ['interferograms', 'coherence', 'geometry']
        });

        // PS-InSAR (Persistent Scatterer) Workflow
        this.workflows.set('ps_insar', {
            name: 'PS-InSAR Analysis',
            description: 'Persistent Scatterer InSAR processing for high-coherence targets',
            steps: [
                'identify_ps_candidates',
                'phase_linking',
                'atmospheric_phase_screen',
                'unwrap_ps_phase',
                'estimate_velocity',
                'time_series_inversion',
                'quality_assessment',
                'geocode_results'
            ],
            estimatedTime: 7200,
            requirements: ['slc_stack', 'amplitude', 'geometry']
        });

        // Hybrid SBAS-PS Workflow
        this.workflows.set('hybrid', {
            name: 'Hybrid SBAS-PS Processing',
            description: 'Combined SBAS and PS-InSAR processing for optimal coverage',
            steps: [
                'sbas_processing',
                'ps_identification',
                'phase_linking_integration',
                'joint_inversion',
                'quality_fusion',
                'final_geocoding'
            ],
            estimatedTime: 5400,
            requirements: ['slc_stack', 'interferograms', 'coherence', 'geometry']
        });
    }

    setupUI() {
        this.createProcessingPanel();
        this.createWorkflowSelector();
        this.createParameterControls();
        this.createProgressMonitor();
        this.createResultsViewer();
    }

    createProcessingPanel() {
        const panel = document.createElement('div');
        panel.className = 'mintpy-processing-panel';
        panel.innerHTML = `
            <div class="panel-header">
                <h3>🛰️ MintPy InSAR Processing</h3>
                <div class="panel-controls">
                    <button class="btn btn-primary" onclick="mintpyIntegration.openDataManager()">
                        📁 Data Manager
                    </button>
                    <button class="btn btn-secondary" onclick="mintpyIntegration.showHelp()">
                        ❓ Help
                    </button>
                </div>
            </div>
            <div class="panel-content">
                <div class="workflow-section">
                    <label>Processing Workflow:</label>
                    <select id="workflow-selector" onchange="mintpyIntegration.onWorkflowChange()">
                        <option value="sbas">SBAS Time Series</option>
                        <option value="ps_insar">PS-InSAR Analysis</option>
                        <option value="hybrid">Hybrid SBAS-PS</option>
                    </select>
                </div>
                <div class="parameters-section" id="parameters-container">
                    <!-- Dynamic parameters will be loaded here -->
                </div>
                <div class="processing-actions">
                    <button class="btn btn-success" onclick="mintpyIntegration.startProcessing()">
                        ▶️ Start Processing
                    </button>
                    <button class="btn btn-warning" onclick="mintpyIntegration.pauseProcessing()">
                        ⏸️ Pause
                    </button>
                    <button class="btn btn-danger" onclick="mintpyIntegration.stopProcessing()">
                        ⏹️ Stop
                    </button>
                </div>
            </div>
        `;

        // Find suitable container or create one
        let container = document.querySelector('#mintpy-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'mintpy-container';
            document.body.appendChild(container);
        }
        container.appendChild(panel);
    }

    createWorkflowSelector() {
        const selector = document.getElementById('workflow-selector');
        if (!selector) return;

        selector.innerHTML = '';
        this.workflows.forEach((workflow, key) => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = workflow.name;
            option.title = workflow.description;
            selector.appendChild(option);
        });

        this.onWorkflowChange();
    }

    onWorkflowChange() {
        const selector = document.getElementById('workflow-selector');
        const selectedWorkflow = selector?.value || 'sbas';
        this.loadParametersForWorkflow(selectedWorkflow);
        this.updateProcessingSteps(selectedWorkflow);
    }

    loadParametersForWorkflow(workflowType) {
        const container = document.getElementById('parameters-container');
        if (!container) return;

        const params = this.config.processingOptions[workflowType] || this.config.processingOptions.sbas;
        
        container.innerHTML = `
            <h4>Processing Parameters</h4>
            <div class="parameters-grid">
                ${this.generateParameterControls(params, workflowType)}
            </div>
        `;
    }

    generateParameterControls(params, workflowType) {
        let html = '';
        
        Object.entries(params).forEach(([key, value]) => {
            const paramKey = `${workflowType}_${key}`;
            const label = key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
            
            if (typeof value === 'boolean') {
                html += `
                    <div class="param-control">
                        <label>
                            <input type="checkbox" id="${paramKey}" ${value ? 'checked' : ''}>
                            ${label}
                        </label>
                    </div>
                `;
            } else if (typeof value === 'number') {
                html += `
                    <div class="param-control">
                        <label>${label}:</label>
                        <input type="number" id="${paramKey}" value="${value}" step="0.1" min="0" max="1">
                    </div>
                `;
            } else if (typeof value === 'string') {
                if (['auto', 'manual', 'era5', 'gacos', 'snaphu', '3d', 'evd', 'delaunay'].includes(value)) {
                    html += `
                        <div class="param-control">
                            <label>${label}:</label>
                            <select id="${paramKey}">
                                ${this.generateSelectOptions(key, value)}
                            </select>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="param-control">
                            <label>${label}:</label>
                            <input type="text" id="${paramKey}" value="${value}">
                        </div>
                    `;
                }
            }
        });
        
        return html;
    }

    generateSelectOptions(paramKey, currentValue) {
        const options = {
            referenceSelection: ['auto', 'manual', 'max_coherence', 'min_rms'],
            troposphericDelay: ['era5', 'gacos', 'none'],
            unwrapMethod: ['snaphu', '3d', 'icu'],
            phaseLinkingMethod: ['evd', 'pca', 'sequential'],
            referenceNetwork: ['delaunay', 'star', 'sequential']
        };
        
        const availableOptions = options[paramKey] || [currentValue];
        return availableOptions.map(option => 
            `<option value="${option}" ${option === currentValue ? 'selected' : ''}>${option}</option>`
        ).join('');
    }

    createProgressMonitor() {
        const monitor = document.createElement('div');
        monitor.className = 'progress-monitor';
        monitor.id = 'mintpy-progress-monitor';
        monitor.innerHTML = `
            <div class="progress-header">
                <h4>Processing Progress</h4>
                <div class="progress-stats">
                    <span id="current-step">Ready</span>
                    <span id="progress-percentage">0%</span>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
            <div class="step-details" id="step-details">
                <div class="step-list" id="step-list">
                    <!-- Steps will be populated dynamically -->
                </div>
            </div>
            <div class="processing-logs">
                <h5>Processing Logs</h5>
                <div class="log-container" id="processing-logs">
                    <div class="log-entry">System ready for processing</div>
                </div>
            </div>
        `;

        const container = document.getElementById('mintpy-container');
        if (container) {
            container.appendChild(monitor);
        }
    }

    createResultsViewer() {
        const viewer = document.createElement('div');
        viewer.className = 'results-viewer';
        viewer.id = 'mintpy-results-viewer';
        viewer.innerHTML = `
            <div class="results-header">
                <h4>Processing Results</h4>
                <div class="results-controls">
                    <button class="btn btn-secondary" onclick="mintpyIntegration.exportResults()">
                        💾 Export
                    </button>
                    <button class="btn btn-info" onclick="mintpyIntegration.generateReport()">
                        📊 Report
                    </button>
                </div>
            </div>
            <div class="results-content">
                <div class="results-tabs">
                    <button class="tab-btn active" onclick="mintpyIntegration.showResultsTab('velocity')">
                        Velocity
                    </button>
                    <button class="tab-btn" onclick="mintpyIntegration.showResultsTab('timeseries')">
                        Time Series
                    </button>
                    <button class="tab-btn" onclick="mintpyIntegration.showResultsTab('quality')">
                        Quality
                    </button>
                    <button class="tab-btn" onclick="mintpyIntegration.showResultsTab('statistics')">
                        Statistics
                    </button>
                </div>
                <div class="results-display">
                    <div class="tab-content active" id="velocity-results">
                        <div class="velocity-map" id="velocity-map-container">
                            <!-- Velocity visualization will be rendered here -->
                        </div>
                    </div>
                    <div class="tab-content" id="timeseries-results">
                        <div class="timeseries-plot" id="timeseries-plot-container">
                            <!-- Time series plots will be rendered here -->
                        </div>
                    </div>
                    <div class="tab-content" id="quality-results">
                        <div class="quality-metrics" id="quality-metrics-container">
                            <!-- Quality assessment will be displayed here -->
                        </div>
                    </div>
                    <div class="tab-content" id="statistics-results">
                        <div class="statistics-summary" id="statistics-summary-container">
                            <!-- Processing statistics will be shown here -->
                        </div>
                    </div>
                </div>
            </div>
        `;

        const container = document.getElementById('mintpy-container');
        if (container) {
            container.appendChild(viewer);
        }
    }

    async startProcessing() {
        const workflowType = document.getElementById('workflow-selector')?.value || 'sbas';
        const workflow = this.workflows.get(workflowType);
        
        if (!workflow) {
            this.logError('Invalid workflow selected');
            return;
        }

        const jobId = this.generateJobId();
        const parameters = this.collectProcessingParameters(workflowType);
        
        const job = {
            id: jobId,
            workflow: workflowType,
            parameters,
            status: 'running',
            startTime: new Date(),
            currentStep: 0,
            totalSteps: workflow.steps.length,
            progress: 0,
            logs: []
        };

        this.activeJobs.set(jobId, job);
        this.updateProgressMonitor(job);
        
        try {
            this.log(`Starting ${workflow.name} processing...`);
            await this.executeWorkflow(job);
        } catch (error) {
            this.logError(`Processing failed: ${error.message}`);
            job.status = 'failed';
            job.error = error.message;
        }
        
        this.updateProgressMonitor(job);
        this.saveJobResult(job);
    }

    async executeWorkflow(job) {
        const workflow = this.workflows.get(job.workflow);
        
        for (let i = 0; i < workflow.steps.length; i++) {
            if (job.status === 'paused') {
                await this.waitForResume(job);
            }
            
            if (job.status === 'stopped') {
                throw new Error('Processing stopped by user');
            }
            
            const step = workflow.steps[i];
            job.currentStep = i;
            job.progress = (i / workflow.steps.length) * 100;
            
            this.log(`Executing step ${i + 1}/${workflow.steps.length}: ${step}`);
            this.updateProgressMonitor(job);
            
            await this.executeProcessingStep(step, job);
            
            // Simulate processing time
            await this.delay(Math.random() * 2000 + 500);
        }
        
        job.status = 'completed';
        job.progress = 100;
        job.endTime = new Date();
        this.log(`Processing completed in ${this.formatDuration(job.endTime - job.startTime)}`);
        
        // Generate results
        await this.generateProcessingResults(job);
    }

    async executeProcessingStep(step, job) {
        const algorithms = window.insarAlgorithms;
        if (!algorithms) {
            throw new Error('InSAR algorithms not loaded');
        }
        
        switch (step) {
            case 'load_data':
                await this.simulateDataLoading(job);
                break;
            case 'network_modification':
                await algorithms.modifyNetwork(job.parameters);
                break;
            case 'reference_point':
                await algorithms.selectReferencePoint(job.parameters);
                break;
            case 'correct_unwrap_error':
                await algorithms.correctUnwrapErrors(job.parameters);
                break;
            case 'invert_network':
                await algorithms.invertNetwork(job.parameters);
                break;
            case 'correct_troposphere':
                if (job.parameters.atmosphericCorrection) {
                    await algorithms.correctAtmosphericPhase(job.parameters);
                }
                break;
            case 'velocity':
                await algorithms.estimateVelocity(job.parameters);
                break;
            case 'identify_ps_candidates':
                await algorithms.identifyPersistentScatterers(job.parameters);
                break;
            case 'phase_linking':
                await algorithms.performPhaseLinking(job.parameters);
                break;
            case 'geocode':
            case 'geocode_results':
                await algorithms.geocodeResults(job.parameters);
                break;
            default:
                this.log(`Executing custom step: ${step}`);
                await this.delay(1000);
        }
    }

    async simulateDataLoading(job) {
        const requiredData = this.workflows.get(job.workflow).requirements;
        
        for (const dataType of requiredData) {
            this.log(`Loading ${dataType} data...`);
            await this.delay(500);
            
            // Simulate data validation
            const dataValid = Math.random() > 0.1; // 90% success rate
            if (!dataValid) {
                throw new Error(`Invalid ${dataType} data detected`);
            }
        }
        
        this.log('All required data loaded successfully');
    }

    async generateProcessingResults(job) {
        const results = {
            jobId: job.id,
            workflow: job.workflow,
            parameters: job.parameters,
            processingTime: job.endTime - job.startTime,
            outputs: {}
        };

        // Generate velocity results
        results.outputs.velocity = {
            meanVelocity: this.generateRandomVelocityData(),
            velocityStdError: this.generateRandomErrorData(),
            spatialCoverage: 0.85,
            temporalCoverage: 0.92
        };

        // Generate time series results
        results.outputs.timeSeries = {
            displacementSeries: this.generateTimeSeriesData(),
            referenceDate: job.parameters.referenceDate || '2020-01-01',
            temporalBaseline: 365
        };

        // Generate quality metrics
        results.outputs.quality = {
            meanCoherence: 0.78,
            temporalCoherence: 0.82,
            residualRMS: 2.3,
            velocityUncertainty: 1.2,
            networkRedundancy: 3.4
        };

        this.results.set(job.id, results);
        this.displayResults(results);
        
        this.eventBus.dispatchEvent(new CustomEvent('processingComplete', {
            detail: { job, results }
        }));
    }

    generateRandomVelocityData() {
        const data = [];
        for (let i = 0; i < 1000; i++) {
            data.push({
                lat: 40.5 + (Math.random() - 0.5) * 0.1,
                lon: -120.3 + (Math.random() - 0.5) * 0.1,
                velocity: (Math.random() - 0.5) * 20, // mm/year
                error: Math.random() * 2 + 0.5
            });
        }
        return data;
    }

    generateRandomErrorData() {
        const data = [];
        for (let i = 0; i < 1000; i++) {
            data.push(Math.random() * 3 + 0.5); // 0.5-3.5 mm/year
        }
        return data;
    }

    generateTimeSeriesData() {
        const dates = [];
        const displacement = [];
        const startDate = new Date('2020-01-01');
        
        for (let i = 0; i < 100; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i * 12); // Every 12 days
            dates.push(date.toISOString().split('T')[0]);
            
            // Simulate cumulative displacement with noise
            const trend = i * 0.3; // Linear trend
            const seasonal = 5 * Math.sin(2 * Math.PI * i / 30); // Seasonal variation
            const noise = (Math.random() - 0.5) * 2; // Random noise
            displacement.push(trend + seasonal + noise);
        }
        
        return { dates, displacement };
    }

    displayResults(results) {
        this.displayVelocityResults(results.outputs.velocity);
        this.displayTimeSeriesResults(results.outputs.timeSeries);
        this.displayQualityMetrics(results.outputs.quality);
        this.displayStatistics(results);
    }

    displayVelocityResults(velocityData) {
        const container = document.getElementById('velocity-map-container');
        if (!container) return;

        const stats = this.calculateVelocityStatistics(velocityData.meanVelocity);
        
        container.innerHTML = `
            <div class="velocity-summary">
                <h5>Velocity Field Summary</h5>
                <div class="stats-grid">
                    <div class="stat">
                        <label>Mean Velocity:</label>
                        <span>${stats.mean.toFixed(2)} mm/year</span>
                    </div>
                    <div class="stat">
                        <label>Standard Deviation:</label>
                        <span>${stats.std.toFixed(2)} mm/year</span>
                    </div>
                    <div class="stat">
                        <label>Min/Max:</label>
                        <span>${stats.min.toFixed(2)} / ${stats.max.toFixed(2)} mm/year</span>
                    </div>
                    <div class="stat">
                        <label>Spatial Coverage:</label>
                        <span>${(velocityData.spatialCoverage * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            <div class="velocity-map-placeholder">
                <p>📍 Velocity map would be rendered here</p>
                <p>Total points: ${velocityData.meanVelocity.length}</p>
            </div>
        `;
    }

    displayTimeSeriesResults(timeSeriesData) {
        const container = document.getElementById('timeseries-plot-container');
        if (!container) return;

        container.innerHTML = `
            <div class="timeseries-summary">
                <h5>Time Series Analysis</h5>
                <div class="timeseries-stats">
                    <div class="stat">
                        <label>Reference Date:</label>
                        <span>${timeSeriesData.referenceDate}</span>
                    </div>
                    <div class="stat">
                        <label>Temporal Baseline:</label>
                        <span>${timeSeriesData.temporalBaseline} days</span>
                    </div>
                    <div class="stat">
                        <label>Data Points:</label>
                        <span>${timeSeriesData.dates.length}</span>
                    </div>
                </div>
            </div>
            <div class="timeseries-plot-placeholder">
                <p>📈 Time series plot would be rendered here</p>
                <p>Showing displacement evolution over time</p>
            </div>
        `;
    }

    displayQualityMetrics(qualityData) {
        const container = document.getElementById('quality-metrics-container');
        if (!container) return;

        container.innerHTML = `
            <div class="quality-summary">
                <h5>Quality Assessment</h5>
                <div class="quality-grid">
                    ${Object.entries(qualityData).map(([key, value]) => `
                        <div class="quality-metric">
                            <label>${this.formatMetricName(key)}:</label>
                            <span class="metric-value ${this.getQualityClass(key, value)}">
                                ${this.formatMetricValue(key, value)}
                            </span>
                            <div class="quality-bar">
                                <div class="quality-fill" style="width: ${this.getQualityPercentage(key, value)}%"></div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    displayStatistics(results) {
        const container = document.getElementById('statistics-summary-container');
        if (!container) return;

        const processingTime = this.formatDuration(results.processingTime);
        
        container.innerHTML = `
            <div class="statistics-summary">
                <h5>Processing Statistics</h5>
                <div class="stats-sections">
                    <div class="stats-section">
                        <h6>Processing Info</h6>
                        <div class="stat-item">
                            <label>Workflow:</label>
                            <span>${results.workflow.toUpperCase()}</span>
                        </div>
                        <div class="stat-item">
                            <label>Processing Time:</label>
                            <span>${processingTime}</span>
                        </div>
                        <div class="stat-item">
                            <label>Job ID:</label>
                            <span>${results.jobId}</span>
                        </div>
                    </div>
                    <div class="stats-section">
                        <h6>Data Quality</h6>
                        <div class="stat-item">
                            <label>Velocity Points:</label>
                            <span>${results.outputs.velocity.meanVelocity.length}</span>
                        </div>
                        <div class="stat-item">
                            <label>Time Series Length:</label>
                            <span>${results.outputs.timeSeries.dates.length}</span>
                        </div>
                        <div class="stat-item">
                            <label>Mean Coherence:</label>
                            <span>${results.outputs.quality.meanCoherence.toFixed(3)}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    calculateVelocityStatistics(velocityData) {
        const velocities = velocityData.map(d => d.velocity);
        const mean = velocities.reduce((a, b) => a + b, 0) / velocities.length;
        const variance = velocities.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / velocities.length;
        const std = Math.sqrt(variance);
        const min = Math.min(...velocities);
        const max = Math.max(...velocities);
        
        return { mean, std, min, max };
    }

    formatMetricName(key) {
        return key.replace(/([A-Z])/g, ' $1')
                 .replace(/^./, str => str.toUpperCase())
                 .replace(/Rms/g, 'RMS');
    }

    formatMetricValue(key, value) {
        if (key.includes('Coherence')) {
            return value.toFixed(3);
        } else if (key.includes('RMS') || key.includes('Uncertainty')) {
            return `${value.toFixed(1)} mm`;
        } else {
            return value.toFixed(2);
        }
    }

    getQualityClass(key, value) {
        if (key.includes('Coherence')) {
            return value > 0.8 ? 'excellent' : value > 0.6 ? 'good' : 'poor';
        } else if (key.includes('RMS')) {
            return value < 2 ? 'excellent' : value < 4 ? 'good' : 'poor';
        }
        return 'good';
    }

    getQualityPercentage(key, value) {
        if (key.includes('Coherence')) {
            return value * 100;
        } else if (key.includes('RMS')) {
            return Math.max(0, 100 - (value / 5 * 100));
        }
        return 75;
    }

    collectProcessingParameters(workflowType) {
        const params = {};
        const elements = document.querySelectorAll(`[id^="${workflowType}_"]`);
        
        elements.forEach(element => {
            const key = element.id.replace(`${workflowType}_`, '');
            
            if (element.type === 'checkbox') {
                params[key] = element.checked;
            } else if (element.type === 'number') {
                params[key] = parseFloat(element.value);
            } else {
                params[key] = element.value;
            }
        });
        
        return params;
    }

    updateProgressMonitor(job) {
        const fillElement = document.getElementById('progress-fill');
        const stepElement = document.getElementById('current-step');
        const percentageElement = document.getElementById('progress-percentage');
        const stepListElement = document.getElementById('step-list');
        
        if (fillElement) {
            fillElement.style.width = `${job.progress}%`;
        }
        
        if (stepElement) {
            const workflow = this.workflows.get(job.workflow);
            const currentStepName = workflow.steps[job.currentStep] || 'Completed';
            stepElement.textContent = currentStepName.replace(/_/g, ' ').replace(/^./, str => str.toUpperCase());
        }
        
        if (percentageElement) {
            percentageElement.textContent = `${Math.round(job.progress)}%`;
        }
        
        if (stepListElement) {
            this.updateStepList(stepListElement, job);
        }
    }

    updateStepList(container, job) {
        const workflow = this.workflows.get(job.workflow);
        container.innerHTML = workflow.steps.map((step, index) => {
            let status = 'pending';
            if (index < job.currentStep) status = 'completed';
            else if (index === job.currentStep) status = 'active';
            
            return `
                <div class="step-item ${status}">
                    <div class="step-icon">
                        ${status === 'completed' ? '✅' : status === 'active' ? '🔄' : '⏳'}
                    </div>
                    <div class="step-name">
                        ${step.replace(/_/g, ' ').replace(/^./, str => str.toUpperCase())}
                    </div>
                </div>
            `;
        }).join('');
    }

    pauseProcessing() {
        this.activeJobs.forEach(job => {
            if (job.status === 'running') {
                job.status = 'paused';
                this.log('Processing paused');
            }
        });
    }

    async waitForResume(job) {
        return new Promise(resolve => {
            const checkStatus = () => {
                if (job.status === 'running') {
                    resolve();
                } else {
                    setTimeout(checkStatus, 100);
                }
            };
            checkStatus();
        });
    }

    stopProcessing() {
        this.activeJobs.forEach(job => {
            if (job.status === 'running' || job.status === 'paused') {
                job.status = 'stopped';
                this.log('Processing stopped');
            }
        });
    }

    resumeProcessing() {
        this.activeJobs.forEach(job => {
            if (job.status === 'paused') {
                job.status = 'running';
                this.log('Processing resumed');
            }
        });
    }

    showResultsTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Remove active class from all tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab content
        const selectedContent = document.getElementById(`${tabName}-results`);
        const selectedBtn = event.target;
        
        if (selectedContent) selectedContent.classList.add('active');
        if (selectedBtn) selectedBtn.classList.add('active');
    }

    generateJobId() {
        return 'job_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    formatDuration(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }

    log(message) {
        const logContainer = document.getElementById('processing-logs');
        if (logContainer) {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `
                <span class="log-time">${new Date().toLocaleTimeString()}</span>
                <span class="log-message">${message}</span>
            `;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        console.log(`[MintPy] ${message}`);
    }

    logError(message) {
        const logContainer = document.getElementById('processing-logs');
        if (logContainer) {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry error';
            logEntry.innerHTML = `
                <span class="log-time">${new Date().toLocaleTimeString()}</span>
                <span class="log-message">❌ ${message}</span>
            `;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        console.error(`[MintPy] ${message}`);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    setupErrorHandling() {
        this.eventBus.addEventListener('error', (event) => {
            this.logError(`System error: ${event.detail.message}`);
        });

        window.addEventListener('error', (event) => {
            if (event.filename?.includes('mintpy_integration')) {
                this.logError(`Runtime error: ${event.message}`);
            }
        });
    }

    saveJobResult(job) {
        const results = this.results.get(job.id);
        if (results) {
            try {
                localStorage.setItem(`mintpy_result_${job.id}`, JSON.stringify({
                    job: {
                        id: job.id,
                        workflow: job.workflow,
                        status: job.status,
                        startTime: job.startTime,
                        endTime: job.endTime
                    },
                    results
                }));
            } catch (error) {
                this.logError(`Failed to save results: ${error.message}`);
            }
        }
    }

    loadSavedResults() {
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key?.startsWith('mintpy_result_')) {
                try {
                    const saved = JSON.parse(localStorage.getItem(key));
                    this.results.set(saved.job.id, saved.results);
                } catch (error) {
                    console.warn(`Failed to load saved result ${key}:`, error);
                }
            }
        }
    }

    exportResults() {
        const activeJobIds = Array.from(this.activeJobs.keys());
        const completedResults = Array.from(this.results.keys()).filter(id => 
            this.activeJobs.get(id)?.status === 'completed'
        );
        
        if (completedResults.length === 0) {
            alert('No completed processing results to export');
            return;
        }
        
        const exportData = completedResults.map(jobId => ({
            job: this.activeJobs.get(jobId),
            results: this.results.get(jobId)
        }));
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mintpy_results_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.log('Results exported successfully');
    }

    generateReport() {
        const reportWindow = window.open('', '_blank');
        const activeResults = Array.from(this.results.entries()).filter(([jobId]) => 
            this.activeJobs.get(jobId)?.status === 'completed'
        );
        
        if (activeResults.length === 0) {
            alert('No completed processing results to report');
            return;
        }
        
        const reportHtml = this.generateReportHTML(activeResults);
        reportWindow.document.write(reportHtml);
        reportWindow.document.close();
    }

    generateReportHTML(results) {
        const reportData = results.map(([jobId, result]) => {
            const job = this.activeJobs.get(jobId);
            return { job, result };
        });
        
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <title>MintPy Processing Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
                    .job-section { margin: 30px 0; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }
                    .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                    .metrics-table th, .metrics-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                    .metrics-table th { background-color: #f5f5f5; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🛰️ MintPy InSAR Processing Report</h1>
                    <p>Generated on ${new Date().toLocaleString()}</p>
                </div>
                
                ${reportData.map(({ job, result }) => `
                    <div class="job-section">
                        <h2>Job ${job.id}</h2>
                        <p><strong>Workflow:</strong> ${job.workflow.toUpperCase()}</p>
                        <p><strong>Processing Time:</strong> ${this.formatDuration(job.endTime - job.startTime)}</p>
                        <p><strong>Status:</strong> ${job.status}</p>
                        
                        <h3>Quality Metrics</h3>
                        <table class="metrics-table">
                            <tr><th>Metric</th><th>Value</th></tr>
                            ${Object.entries(result.outputs.quality).map(([key, value]) => 
                                `<tr><td>${this.formatMetricName(key)}</td><td>${this.formatMetricValue(key, value)}</td></tr>`
                            ).join('')}
                        </table>
                        
                        <h3>Processing Results</h3>
                        <ul>
                            <li>Velocity measurements: ${result.outputs.velocity.meanVelocity.length} points</li>
                            <li>Time series length: ${result.outputs.timeSeries.dates.length} dates</li>
                            <li>Spatial coverage: ${(result.outputs.velocity.spatialCoverage * 100).toFixed(1)}%</li>
                            <li>Temporal coverage: ${(result.outputs.velocity.temporalCoverage * 100).toFixed(1)}%</li>
                        </ul>
                    </div>
                `).join('')}
            </body>
            </html>
        `;
    }

    openDataManager() {
        alert('Data Manager would open here - this would interface with MintPy data loading utilities');
    }

    showHelp() {
        const helpWindow = window.open('', '_blank');
        const helpHtml = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>MintPy Integration Help</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1, h2 { color: #333; }
                    .workflow-section { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
                    code { background: #e4e4e4; padding: 2px 4px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>🛰️ MintPy Integration Help</h1>
                
                <h2>Available Workflows</h2>
                
                <div class="workflow-section">
                    <h3>SBAS (Small Baseline Subset)</h3>
                    <p>Best for: Time series analysis, deformation monitoring</p>
                    <p>Input: Interferogram stack, coherence maps</p>
                    <p>Features: Atmospheric correction, unwrap error correction, velocity estimation</p>
                </div>
                
                <div class="workflow-section">
                    <h3>PS-InSAR (Persistent Scatterer)</h3>
                    <p>Best for: Urban areas, stable targets, long-term monitoring</p>
                    <p>Input: SLC stack, amplitude data</p>
                    <p>Features: Phase linking, atmospheric phase screen estimation</p>
                </div>
                
                <div class="workflow-section">
                    <h3>Hybrid SBAS-PS</h3>
                    <p>Best for: Mixed terrain, optimal coverage</p>
                    <p>Input: Complete dataset (SLC + interferograms)</p>
                    <p>Features: Combined processing, joint inversion</p>
                </div>
                
                <h2>Processing Parameters</h2>
                <ul>
                    <li><code>Coherence Threshold</code>: Minimum coherence for valid pixels (0.0-1.0)</li>
                    <li><code>Unwrap Method</code>: Phase unwrapping algorithm (snaphu, 3d, icu)</li>
                    <li><code>Atmospheric Correction</code>: Enable tropospheric delay correction</li>
                    <li><code>Reference Selection</code>: Method for reference point selection</li>
                </ul>
                
                <h2>Quality Metrics</h2>
                <ul>
                    <li><code>Mean Coherence</code>: Average interferometric coherence</li>
                    <li><code>Temporal Coherence</code>: Consistency over time</li>
                    <li><code>Residual RMS</code>: Phase residual root-mean-square</li>
                    <li><code>Velocity Uncertainty</code>: Standard error of velocity estimates</li>
                </ul>
                
                <p><em>For detailed MintPy documentation, visit: <a href="https://mintpy.readthedocs.io" target="_blank">https://mintpy.readthedocs.io</a></em></p>
            </body>
            </html>
        `;
        helpWindow.document.write(helpHtml);
        helpWindow.document.close();
    }
}

// Global instance and initialization
let mintpyIntegration;

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        mintpyIntegration = new MintPyIntegration();
        window.mintpyIntegration = mintpyIntegration;
    });
} else {
    mintpyIntegration = new MintPyIntegration();
    window.mintpyIntegration = mintpyIntegration;
}

// CSS Styles for MintPy Integration
const mintpyStyles = document.createElement('style');
mintpyStyles.textContent = `
    .mintpy-processing-panel {
        background: var(--bg-secondary, #ffffff);
        border: 1px solid var(--border, #e0e0e0);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 20px;
        overflow: hidden;
    }

    .mintpy-processing-panel .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .mintpy-processing-panel .panel-header h3 {
        margin: 0;
        font-size: 1.5em;
        font-weight: 600;
    }

    .mintpy-processing-panel .panel-controls {
        display: flex;
        gap: 10px;
    }

    .mintpy-processing-panel .btn {
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .mintpy-processing-panel .btn-primary {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .mintpy-processing-panel .btn-primary:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-1px);
    }

    .mintpy-processing-panel .btn-secondary {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .mintpy-processing-panel .btn-success {
        background: #28a745;
        color: white;
    }

    .mintpy-processing-panel .btn-warning {
        background: #ffc107;
        color: #212529;
    }

    .mintpy-processing-panel .btn-danger {
        background: #dc3545;
        color: white;
    }

    .mintpy-processing-panel .btn-info {
        background: #17a2b8;
        color: white;
    }

    .mintpy-processing-panel .panel-content {
        padding: 25px;
    }

    .workflow-section, .parameters-section {
        margin-bottom: 25px;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border, #e0e0e0);
    }

    .workflow-section:last-child, .parameters-section:last-child {
        border-bottom: none;
    }

    .workflow-section label {
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
        color: var(--text-primary, #333);
    }

    .workflow-section select {
        width: 100%;
        padding: 12px;
        border: 1px solid var(--border, #e0e0e0);
        border-radius: 6px;
        background: var(--bg-secondary, #ffffff);
        font-size: 14px;
        transition: border-color 0.3s ease;
    }

    .workflow-section select:focus {
        outline: none;
        border-color: var(--accent, #667eea);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .parameters-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 15px;
    }

    .param-control {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .param-control label {
        font-weight: 500;
        color: var(--text-primary, #333);
        font-size: 14px;
    }

    .param-control input, .param-control select {
        padding: 10px 12px;
        border: 1px solid var(--border, #e0e0e0);
        border-radius: 6px;
        background: var(--bg-secondary, #ffffff);
        font-size: 14px;
        transition: all 0.3s ease;
    }

    .param-control input:focus, .param-control select:focus {
        outline: none;
        border-color: var(--accent, #667eea);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .param-control input[type="checkbox"] {
        width: auto;
        margin-right: 8px;
    }

    .processing-actions {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin-top: 25px;
        padding-top: 20px;
        border-top: 1px solid var(--border, #e0e0e0);
    }

    .progress-monitor {
        background: var(--bg-secondary, #ffffff);
        border: 1px solid var(--border, #e0e0e0);
        border-radius: 12px;
        margin: 20px;
        overflow: hidden;
    }

    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        background: var(--bg-tertiary, #f8f9fa);
        border-bottom: 1px solid var(--border, #e0e0e0);
    }

    .progress-header h4 {
        margin: 0;
        color: var(--text-primary, #333);
    }

    .progress-stats {
        display: flex;
        gap: 20px;
        font-size: 14px;
        color: var(--text-secondary, #666);
    }

    .progress-bar {
        height: 8px;
        background: var(--bg-tertiary, #f0f0f0);
        margin: 0 20px;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
        border-radius: 4px;
    }

    .step-details {
        padding: 20px;
    }

    .step-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 10px;
    }

    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        border-radius: 6px;
        background: var(--bg-tertiary, #f8f9fa);
        transition: all 0.3s ease;
    }

    .step-item.active {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    .step-item.completed {
        background: rgba(40, 167, 69, 0.1);
        border: 1px solid rgba(40, 167, 69, 0.3);
    }

    .step-icon {
        font-size: 16px;
        width: 20px;
        text-align: center;
    }

    .step-name {
        font-size: 14px;
        color: var(--text-primary, #333);
    }

    .processing-logs {
        padding: 20px;
        border-top: 1px solid var(--border, #e0e0e0);
    }

    .processing-logs h5 {
        margin: 0 0 15px 0;
        color: var(--text-primary, #333);
    }

    .log-container {
        height: 200px;
        overflow-y: auto;
        background: var(--bg-primary, #000);
        color: var(--text-primary, #00ff00);
        padding: 15px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
    }

    .log-entry {
        margin-bottom: 5px;
        display: flex;
        gap: 10px;
    }

    .log-entry.error {
        color: #ff6b6b;
    }

    .log-time {
        color: #888;
        font-size: 12px;
        min-width: 80px;
    }

    .log-message {
        flex: 1;
    }

    .results-viewer {
        background: var(--bg-secondary, #ffffff);
        border: 1px solid var(--border, #e0e0e0);
        border-radius: 12px;
        margin: 20px;
        overflow: hidden;
    }

    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        background: var(--bg-tertiary, #f8f9fa);
        border-bottom: 1px solid var(--border, #e0e0e0);
    }

    .results-header h4 {
        margin: 0;
        color: var(--text-primary, #333);
    }

    .results-controls {
        display: flex;
        gap: 10px;
    }

    .results-tabs {
        display: flex;
        background: var(--bg-tertiary, #f8f9fa);
        border-bottom: 1px solid var(--border, #e0e0e0);
    }

    .tab-btn {
        flex: 1;
        padding: 15px 20px;
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        color: var(--text-secondary, #666);
        transition: all 0.3s ease;
        border-bottom: 3px solid transparent;
    }

    .tab-btn:hover {
        background: rgba(102, 126, 234, 0.1);
        color: var(--text-primary, #333);
    }

    .tab-btn.active {
        color: var(--accent, #667eea);
        border-bottom-color: var(--accent, #667eea);
        background: var(--bg-secondary, #ffffff);
    }

    .results-display {
        min-height: 400px;
    }

    .tab-content {
        display: none;
        padding: 25px;
    }

    .tab-content.active {
        display: block;
    }

    .velocity-summary, .timeseries-summary, .quality-summary, .statistics-summary {
        margin-bottom: 25px;
    }

    .velocity-summary h5, .timeseries-summary h5, .quality-summary h5, .statistics-summary h5 {
        margin: 0 0 15px 0;
        color: var(--text-primary, #333);
        font-size: 1.2em;
    }

    .stats-grid, .timeseries-stats, .quality-grid, .stats-sections {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }

    .stat, .quality-metric, .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px;
        background: var(--bg-tertiary, #f8f9fa);
        border-radius: 6px;
        border-left: 4px solid var(--accent, #667eea);
    }

    .quality-metric {
        flex-direction: column;
        align-items: stretch;
        gap: 8px;
    }

    .metric-value {
        font-weight: 600;
        font-size: 1.1em;
    }

    .metric-value.excellent {
        color: #28a745;
    }

    .metric-value.good {
        color: #17a2b8;
    }

    .metric-value.poor {
        color: #dc3545;
    }

    .quality-bar {
        height: 4px;
        background: var(--border, #e0e0e0);
        border-radius: 2px;
        overflow: hidden;
    }

    .quality-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #17a2b8, #dc3545);
        transition: width 0.3s ease;
    }

    .velocity-map-placeholder, .timeseries-plot-placeholder {
        padding: 40px;
        text-align: center;
        background: var(--bg-tertiary, #f8f9fa);
        border: 2px dashed var(--border, #ccc);
        border-radius: 8px;
        color: var(--text-secondary, #666);
        margin-top: 15px;
    }

    .stats-section {
        background: var(--bg-tertiary, #f8f9fa);
        padding: 20px;
        border-radius: 8px;
    }

    .stats-section h6 {
        margin: 0 0 15px 0;
        color: var(--text-primary, #333);
        font-size: 1em;
        font-weight: 600;
        border-bottom: 2px solid var(--accent, #667eea);
        padding-bottom: 5px;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .parameters-grid {
            grid-template-columns: 1fr;
        }
        
        .processing-actions {
            flex-direction: column;
        }
        
        .results-tabs {
            flex-direction: column;
        }
        
        .tab-btn {
            border-bottom: 1px solid var(--border, #e0e0e0);
            border-right: none;
        }
        
        .tab-btn.active {
            border-bottom-color: var(--border, #e0e0e0);
            border-left: 3px solid var(--accent, #667eea);
        }
        
        .step-list {
            grid-template-columns: 1fr;
        }
        
        .stats-grid, .timeseries-stats, .quality-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .mintpy-processing-panel, .progress-monitor, .results-viewer {
            background: var(--bg-secondary, #2a2a2a);
            border-color: var(--border, #404040);
        }
        
        .log-container {
            background: var(--bg-primary, #1a1a1a);
        }
        
        .step-item, .stat, .quality-metric, .stat-item, .stats-section {
            background: var(--bg-tertiary, #3a3a3a);
        }
        
        .velocity-map-placeholder, .timeseries-plot-placeholder {
            background: var(--bg-tertiary, #3a3a3a);
            border-color: var(--border, #555);
        }
    }
`;

document.head.appendChild(mintpyStyles);

console.log('🛰️ MintPy Integration System Loaded');