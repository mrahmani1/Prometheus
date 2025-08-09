/**
 * Processing Queue Management System - Task 3.4
 * Advanced queue management for SAR data processing with priority scheduling,
 * resource allocation, job monitoring, and fault tolerance
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function() {
    'use strict';

    class ProcessingQueue {
        constructor(options = {}) {
            this.container = options.container || document.body;
            this.jobs = new Map();
            this.activeJobs = new Map();
            this.completedJobs = new Map();
            this.failedJobs = new Map();
            this.workers = [];
            this.ui = null;
            
            this.config = {
                maxConcurrentJobs: 4,
                maxRetryAttempts: 3,
                jobTimeout: 600000, // 10 minutes
                heartbeatInterval: 5000, // 5 seconds
                priorityLevels: ['low', 'normal', 'high', 'urgent'],
                resourceLimits: {
                    memory: 8192, // MB
                    cpu: 80,      // percentage
                    disk: 10240   // MB
                },
                supportedProcessors: [
                    'interferogram',
                    'coherence',
                    'geocoding',
                    'filtering',
                    'calibration',
                    'coregistration',
                    'unwrapping',
                    'subsidence',
                    'classification'
                ]
            };

            this.scheduler = new JobScheduler(this.config);
            this.resourceMonitor = new ResourceMonitor(this.config);
            this.processEngine = new ProcessEngine();
            this.jobLogger = new JobLogger();
            this.notificationSystem = new NotificationSystem();
            
            this.statistics = {
                totalJobs: 0,
                completedJobs: 0,
                failedJobs: 0,
                avgProcessingTime: 0,
                totalProcessingTime: 0,
                resourceUtilization: {
                    cpu: 0,
                    memory: 0,
                    disk: 0
                }
            };

            this.init();
        }

        init() {
            this.createQueueInterface();
            this.setupEventHandlers();
            this.initializeWorkers();
            this.startResourceMonitoring();
            this.setupHeartbeat();
            this.loadJobHistory();
            console.log('⚙️ Processing Queue System initialized');
        }

        createQueueInterface() {
            const container = document.createElement('div');
            container.className = 'processing-queue-container';
            container.innerHTML = `
                <div class="queue-header">
                    <h3><i class="fas fa-tasks"></i> Processing Queue</h3>
                    <div class="queue-controls">
                        <button class="btn btn-primary" id="addJob">Add Job</button>
                        <button class="btn btn-secondary" id="pauseQueue">Pause Queue</button>
                        <button class="btn btn-success" id="resumeQueue">Resume Queue</button>
                        <button class="btn btn-warning" id="clearCompleted">Clear Completed</button>
                    </div>
                </div>

                <div class="queue-dashboard">
                    <div class="dashboard-stats">
                        <div class="stat-card">
                            <div class="stat-icon"><i class="fas fa-play"></i></div>
                            <div class="stat-info">
                                <div class="stat-value" id="activeJobCount">0</div>
                                <div class="stat-label">Active Jobs</div>
                            </div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon"><i class="fas fa-clock"></i></div>
                            <div class="stat-info">
                                <div class="stat-value" id="queuedJobCount">0</div>
                                <div class="stat-label">Queued Jobs</div>
                            </div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon"><i class="fas fa-check"></i></div>
                            <div class="stat-info">
                                <div class="stat-value" id="completedJobCount">0</div>
                                <div class="stat-label">Completed</div>
                            </div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon"><i class="fas fa-times"></i></div>
                            <div class="stat-info">
                                <div class="stat-value" id="failedJobCount">0</div>
                                <div class="stat-label">Failed</div>
                            </div>
                        </div>
                    </div>

                    <div class="resource-monitor">
                        <h4>Resource Utilization</h4>
                        <div class="resource-meters">
                            <div class="resource-meter">
                                <label>CPU Usage</label>
                                <div class="meter-bar">
                                    <div class="meter-fill" id="cpuMeter"></div>
                                </div>
                                <span id="cpuValue">0%</span>
                            </div>
                            <div class="resource-meter">
                                <label>Memory Usage</label>
                                <div class="meter-bar">
                                    <div class="meter-fill" id="memoryMeter"></div>
                                </div>
                                <span id="memoryValue">0%</span>
                            </div>
                            <div class="resource-meter">
                                <label>Disk I/O</label>
                                <div class="meter-bar">
                                    <div class="meter-fill" id="diskMeter"></div>
                                </div>
                                <span id="diskValue">0%</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="queue-tabs">
                    <button class="tab-btn active" data-tab="active">Active Jobs</button>
                    <button class="tab-btn" data-tab="queue">Job Queue</button>
                    <button class="tab-btn" data-tab="history">History</button>
                    <button class="tab-btn" data-tab="config">Configuration</button>
                    <button class="tab-btn" data-tab="monitoring">Monitoring</button>
                </div>

                <div class="queue-content">
                    <!-- Active Jobs Tab -->
                    <div class="tab-content active" id="active-tab">
                        <div class="jobs-panel">
                            <div class="jobs-header">
                                <h4>Active Jobs</h4>
                                <div class="jobs-actions">
                                    <button id="pauseAllJobs" class="btn btn-sm btn-warning">Pause All</button>
                                    <button id="stopAllJobs" class="btn btn-sm btn-danger">Stop All</button>
                                </div>
                            </div>
                            <div class="jobs-list" id="activeJobsList">
                                <!-- Active jobs will be populated here -->
                            </div>
                        </div>
                    </div>

                    <!-- Queue Tab -->
                    <div class="tab-content" id="queue-tab">
                        <div class="jobs-panel">
                            <div class="jobs-header">
                                <h4>Job Queue</h4>
                                <div class="queue-actions">
                                    <select id="queueFilter">
                                        <option value="all">All Jobs</option>
                                        <option value="urgent">Urgent</option>
                                        <option value="high">High Priority</option>
                                        <option value="normal">Normal Priority</option>
                                        <option value="low">Low Priority</option>
                                    </select>
                                    <button id="reorderQueue" class="btn btn-sm btn-secondary">Reorder Queue</button>
                                </div>
                            </div>
                            <div class="jobs-list" id="queuedJobsList">
                                <!-- Queued jobs will be populated here -->
                            </div>
                        </div>
                    </div>

                    <!-- History Tab -->
                    <div class="tab-content" id="history-tab">
                        <div class="history-panel">
                            <div class="history-controls">
                                <div class="history-filters">
                                    <input type="date" id="historyFromDate" />
                                    <input type="date" id="historyToDate" />
                                    <select id="historyStatus">
                                        <option value="all">All Status</option>
                                        <option value="completed">Completed</option>
                                        <option value="failed">Failed</option>
                                        <option value="cancelled">Cancelled</option>
                                    </select>
                                    <button id="filterHistory" class="btn btn-sm">Filter</button>
                                </div>
                                <div class="history-actions">
                                    <button id="exportHistory" class="btn btn-sm btn-info">Export History</button>
                                    <button id="clearHistory" class="btn btn-sm btn-danger">Clear History</button>
                                </div>
                            </div>

                            <div class="history-stats">
                                <div class="stat-summary">
                                    <div class="summary-item">
                                        <span>Total Processed:</span>
                                        <span id="totalProcessed">0</span>
                                    </div>
                                    <div class="summary-item">
                                        <span>Success Rate:</span>
                                        <span id="successRate">0%</span>
                                    </div>
                                    <div class="summary-item">
                                        <span>Avg. Processing Time:</span>
                                        <span id="avgProcessingTime">0s</span>
                                    </div>
                                    <div class="summary-item">
                                        <span>Total CPU Hours:</span>
                                        <span id="totalCpuHours">0h</span>
                                    </div>
                                </div>
                            </div>

                            <div class="history-chart">
                                <canvas id="processingChart" width="800" height="300"></canvas>
                            </div>

                            <div class="history-list" id="historyList">
                                <!-- Historical jobs will be populated here -->
                            </div>
                        </div>
                    </div>

                    <!-- Configuration Tab -->
                    <div class="tab-content" id="config-tab">
                        <div class="config-panel">
                            <div class="config-sections">
                                <div class="config-section">
                                    <h4>Queue Settings</h4>
                                    <div class="config-group">
                                        <label>Max Concurrent Jobs:</label>
                                        <input type="number" id="maxConcurrentJobs" min="1" max="16" value="${this.config.maxConcurrentJobs}">
                                    </div>
                                    <div class="config-group">
                                        <label>Job Timeout (minutes):</label>
                                        <input type="number" id="jobTimeout" min="5" max="1440" value="${this.config.jobTimeout / 60000}">
                                    </div>
                                    <div class="config-group">
                                        <label>Max Retry Attempts:</label>
                                        <input type="number" id="maxRetryAttempts" min="0" max="10" value="${this.config.maxRetryAttempts}">
                                    </div>
                                    <div class="config-group">
                                        <label>Auto-start Queue:</label>
                                        <input type="checkbox" id="autoStartQueue" checked>
                                    </div>
                                </div>

                                <div class="config-section">
                                    <h4>Resource Limits</h4>
                                    <div class="config-group">
                                        <label>Memory Limit (MB):</label>
                                        <input type="number" id="memoryLimit" min="1024" max="32768" value="${this.config.resourceLimits.memory}">
                                    </div>
                                    <div class="config-group">
                                        <label>CPU Limit (%):</label>
                                        <input type="range" id="cpuLimit" min="10" max="100" value="${this.config.resourceLimits.cpu}">
                                        <span id="cpuLimitValue">${this.config.resourceLimits.cpu}%</span>
                                    </div>
                                    <div class="config-group">
                                        <label>Disk Space Limit (MB):</label>
                                        <input type="number" id="diskLimit" min="1024" max="102400" value="${this.config.resourceLimits.disk}">
                                    </div>
                                </div>

                                <div class="config-section">
                                    <h4>Scheduling</h4>
                                    <div class="config-group">
                                        <label>Scheduling Algorithm:</label>
                                        <select id="schedulingAlgorithm">
                                            <option value="priority">Priority-based</option>
                                            <option value="fifo">First-In-First-Out</option>
                                            <option value="sjf">Shortest Job First</option>
                                            <option value="round_robin">Round Robin</option>
                                        </select>
                                    </div>
                                    <div class="config-group">
                                        <label>Priority Aging:</label>
                                        <input type="checkbox" id="priorityAging" checked>
                                    </div>
                                    <div class="config-group">
                                        <label>Load Balancing:</label>
                                        <input type="checkbox" id="loadBalancing" checked>
                                    </div>
                                </div>

                                <div class="config-section">
                                    <h4>Notifications</h4>
                                    <div class="config-group">
                                        <label>Job Completion Notifications:</label>
                                        <input type="checkbox" id="completionNotifications" checked>
                                    </div>
                                    <div class="config-group">
                                        <label>Error Notifications:</label>
                                        <input type="checkbox" id="errorNotifications" checked>
                                    </div>
                                    <div class="config-group">
                                        <label>Resource Alert Threshold:</label>
                                        <input type="range" id="resourceAlertThreshold" min="50" max="95" value="85">
                                        <span id="resourceAlertValue">85%</span>
                                    </div>
                                </div>
                            </div>

                            <div class="config-actions">
                                <button id="saveConfig" class="btn btn-primary">Save Configuration</button>
                                <button id="resetConfig" class="btn btn-secondary">Reset to Defaults</button>
                                <button id="exportConfig" class="btn btn-info">Export Config</button>
                                <button id="importConfig" class="btn btn-warning">Import Config</button>
                            </div>
                        </div>
                    </div>

                    <!-- Monitoring Tab -->
                    <div class="tab-content" id="monitoring-tab">
                        <div class="monitoring-panel">
                            <div class="monitoring-overview">
                                <div class="system-health">
                                    <h4>System Health</h4>
                                    <div class="health-indicators">
                                        <div class="health-item">
                                            <span class="health-label">Queue Status:</span>
                                            <span class="health-value" id="queueStatus">Running</span>
                                            <span class="health-indicator healthy" id="queueIndicator"></span>
                                        </div>
                                        <div class="health-item">
                                            <span class="health-label">Worker Pool:</span>
                                            <span class="health-value" id="workerPoolStatus">4/4 Active</span>
                                            <span class="health-indicator healthy" id="workerIndicator"></span>
                                        </div>
                                        <div class="health-item">
                                            <span class="health-label">Resource Status:</span>
                                            <span class="health-value" id="resourceStatus">Normal</span>
                                            <span class="health-indicator healthy" id="resourceIndicator"></span>
                                        </div>
                                        <div class="health-item">
                                            <span class="health-label">Error Rate:</span>
                                            <span class="health-value" id="errorRate">0%</span>
                                            <span class="health-indicator healthy" id="errorIndicator"></span>
                                        </div>
                                    </div>
                                </div>

                                <div class="performance-metrics">
                                    <h4>Performance Metrics</h4>
                                    <div class="metrics-grid">
                                        <div class="metric-item">
                                            <div class="metric-value" id="throughput">0</div>
                                            <div class="metric-label">Jobs/Hour</div>
                                        </div>
                                        <div class="metric-item">
                                            <div class="metric-value" id="queueLatency">0</div>
                                            <div class="metric-label">Avg. Queue Time (s)</div>
                                        </div>
                                        <div class="metric-item">
                                            <div class="metric-value" id="processingLatency">0</div>
                                            <div class="metric-label">Avg. Process Time (s)</div>
                                        </div>
                                        <div class="metric-item">
                                            <div class="metric-value" id="resourceEfficiency">0</div>
                                            <div class="metric-label">Resource Efficiency (%)</div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="monitoring-charts">
                                <div class="chart-row">
                                    <div class="chart-panel">
                                        <h5>Job Throughput</h5>
                                        <canvas id="throughputChart" width="400" height="200"></canvas>
                                    </div>
                                    <div class="chart-panel">
                                        <h5>Resource Usage</h5>
                                        <canvas id="resourceChart" width="400" height="200"></canvas>
                                    </div>
                                </div>
                                <div class="chart-row">
                                    <div class="chart-panel">
                                        <h5>Queue Length</h5>
                                        <canvas id="queueChart" width="400" height="200"></canvas>
                                    </div>
                                    <div class="chart-panel">
                                        <h5>Error Rate</h5>
                                        <canvas id="errorChart" width="400" height="200"></canvas>
                                    </div>
                                </div>
                            </div>

                            <div class="log-viewer">
                                <h4>System Logs</h4>
                                <div class="log-controls">
                                    <select id="logLevel">
                                        <option value="all">All Levels</option>
                                        <option value="error">Errors</option>
                                        <option value="warning">Warnings</option>
                                        <option value="info">Info</option>
                                        <option value="debug">Debug</option>
                                    </select>
                                    <button id="clearLogs" class="btn btn-sm btn-secondary">Clear Logs</button>
                                    <button id="exportLogs" class="btn btn-sm btn-info">Export Logs</button>
                                </div>
                                <div class="log-content" id="logContent">
                                    <!-- Log entries will be populated here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Job Creation Modal -->
                <div class="modal" id="jobModal">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h4>Create New Job</h4>
                            <button class="modal-close" id="closeJobModal">&times;</button>
                        </div>
                        <div class="modal-body">
                            <div class="job-form">
                                <div class="form-group">
                                    <label>Job Name:</label>
                                    <input type="text" id="jobName" placeholder="Enter job name">
                                </div>
                                <div class="form-group">
                                    <label>Processing Type:</label>
                                    <select id="processingType">
                                        <option value="interferogram">Interferogram Generation</option>
                                        <option value="coherence">Coherence Analysis</option>
                                        <option value="geocoding">Geocoding</option>
                                        <option value="filtering">Speckle Filtering</option>
                                        <option value="calibration">Radiometric Calibration</option>
                                        <option value="coregistration">Image Coregistration</option>
                                        <option value="unwrapping">Phase Unwrapping</option>
                                        <option value="subsidence">Subsidence Analysis</option>
                                        <option value="classification">Land Cover Classification</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Priority:</label>
                                    <select id="jobPriority">
                                        <option value="low">Low</option>
                                        <option value="normal" selected>Normal</option>
                                        <option value="high">High</option>
                                        <option value="urgent">Urgent</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Input Data:</label>
                                    <input type="file" id="inputFiles" multiple accept=".tif,.zip,.h5,.nc">
                                </div>
                                <div class="form-group">
                                    <label>Processing Parameters:</label>
                                    <textarea id="processingParams" placeholder="JSON parameters" rows="4"></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Estimated Processing Time (minutes):</label>
                                    <input type="number" id="estimatedTime" min="1" max="1440" value="60">
                                </div>
                                <div class="form-group">
                                    <label>Output Format:</label>
                                    <select id="outputFormat">
                                        <option value="geotiff">GeoTIFF</option>
                                        <option value="netcdf">NetCDF</option>
                                        <option value="hdf5">HDF5</option>
                                        <option value="envi">ENVI</option>
                                        <option value="kmz">Google Earth KMZ</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button id="submitJob" class="btn btn-primary">Submit Job</button>
                            <button id="cancelJob" class="btn btn-secondary">Cancel</button>
                        </div>
                    </div>
                </div>

                <div class="queue-status">
                    <div class="status-bar">
                        <div class="status-progress">
                            <div class="progress-fill" id="queueProgress"></div>
                        </div>
                        <div class="status-text" id="queueStatusText">Processing Queue ready</div>
                    </div>
                </div>
            `;

            this.container.appendChild(container);
            this.ui = container;
            this.setupCanvases();
        }

        setupCanvases() {
            this.canvases = {
                processing: document.getElementById('processingChart'),
                throughput: document.getElementById('throughputChart'),
                resource: document.getElementById('resourceChart'),
                queue: document.getElementById('queueChart'),
                error: document.getElementById('errorChart')
            };

            this.contexts = {};
            Object.keys(this.canvases).forEach(key => {
                if (this.canvases[key]) {
                    this.contexts[key] = this.canvases[key].getContext('2d');
                }
            });
        }

        setupEventHandlers() {
            // Main control buttons
            document.getElementById('addJob')?.addEventListener('click', () => {
                this.showJobModal();
            });

            document.getElementById('pauseQueue')?.addEventListener('click', () => {
                this.pauseQueue();
            });

            document.getElementById('resumeQueue')?.addEventListener('click', () => {
                this.resumeQueue();
            });

            document.getElementById('clearCompleted')?.addEventListener('click', () => {
                this.clearCompletedJobs();
            });

            // Tab switching
            this.ui.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.switchTab(btn.dataset.tab);
                });
            });

            // Job modal
            document.getElementById('closeJobModal')?.addEventListener('click', () => {
                this.hideJobModal();
            });

            document.getElementById('submitJob')?.addEventListener('click', () => {
                this.submitNewJob();
            });

            document.getElementById('cancelJob')?.addEventListener('click', () => {
                this.hideJobModal();
            });

            // Configuration controls
            document.getElementById('saveConfig')?.addEventListener('click', () => {
                this.saveConfiguration();
            });

            document.getElementById('cpuLimit')?.addEventListener('input', (e) => {
                document.getElementById('cpuLimitValue').textContent = e.target.value + '%';
            });

            document.getElementById('resourceAlertThreshold')?.addEventListener('input', (e) => {
                document.getElementById('resourceAlertValue').textContent = e.target.value + '%';
            });

            // History controls
            document.getElementById('filterHistory')?.addEventListener('click', () => {
                this.filterJobHistory();
            });

            document.getElementById('exportHistory')?.addEventListener('click', () => {
                this.exportJobHistory();
            });
        }

        initializeWorkers() {
            this.workers = [];
            
            for (let i = 0; i < this.config.maxConcurrentJobs; i++) {
                const worker = this.createProcessingWorker(i);
                this.workers.push({
                    id: i,
                    worker: worker,
                    status: 'idle',
                    currentJob: null
                });
            }
        }

        createProcessingWorker(id) {
            const workerCode = `
                self.workerId = ${id};
                
                self.onmessage = function(e) {
                    const { action, jobData } = e.data;
                    
                    switch(action) {
                        case 'process':
                            processJob(jobData);
                            break;
                        case 'pause':
                            pauseJob();
                            break;
                        case 'resume':
                            resumeJob();
                            break;
                        case 'cancel':
                            cancelJob();
                            break;
                        default:
                            self.postMessage({ error: 'Unknown action', action });
                    }
                };

                let currentJob = null;
                let isPaused = false;
                let processingInterval = null;

                function processJob(job) {
                    currentJob = job;
                    
                    self.postMessage({
                        type: 'jobStarted',
                        workerId: self.workerId,
                        jobId: job.id,
                        timestamp: Date.now()
                    });

                    // Simulate processing based on job type
                    simulateProcessing(job);
                }

                function simulateProcessing(job) {
                    const totalSteps = getProcessingSteps(job.processingType);
                    let currentStep = 0;
                    let progress = 0;

                    processingInterval = setInterval(() => {
                        if (isPaused) return;

                        currentStep++;
                        progress = currentStep / totalSteps;

                        // Simulate resource usage
                        const cpuUsage = 60 + Math.random() * 30;
                        const memoryUsage = 1024 + Math.random() * 2048;

                        self.postMessage({
                            type: 'progress',
                            workerId: self.workerId,
                            jobId: job.id,
                            progress: progress,
                            currentStep: currentStep,
                            totalSteps: totalSteps,
                            stepName: getStepName(job.processingType, currentStep),
                            resources: {
                                cpu: cpuUsage,
                                memory: memoryUsage,
                                disk: Math.random() * 100
                            },
                            timestamp: Date.now()
                        });

                        // Random chance of error (2%)
                        if (Math.random() < 0.02) {
                            clearInterval(processingInterval);
                            self.postMessage({
                                type: 'error',
                                workerId: self.workerId,
                                jobId: job.id,
                                error: 'Simulated processing error',
                                timestamp: Date.now()
                            });
                            return;
                        }

                        if (progress >= 1) {
                            clearInterval(processingInterval);
                            self.postMessage({
                                type: 'completed',
                                workerId: self.workerId,
                                jobId: job.id,
                                outputFiles: generateOutputFiles(job),
                                timestamp: Date.now()
                            });
                        }
                    }, 1000 + Math.random() * 2000); // Variable step duration
                }

                function getProcessingSteps(processingType) {
                    const steps = {
                        'interferogram': 15,
                        'coherence': 8,
                        'geocoding': 10,
                        'filtering': 5,
                        'calibration': 6,
                        'coregistration': 12,
                        'unwrapping': 18,
                        'subsidence': 20,
                        'classification': 25
                    };
                    return steps[processingType] || 10;
                }

                function getStepName(processingType, step) {
                    const stepNames = {
                        'interferogram': [
                            'Loading master image',
                            'Loading slave image',
                            'Image coregistration',
                            'Resampling slave image',
                            'Computing interferogram',
                            'Phase filtering',
                            'Coherence calculation',
                            'Multilooking',
                            'Phase unwrapping',
                            'Geocoding',
                            'Terrain correction',
                            'Atmospheric correction',
                            'Noise filtering',
                            'Final processing',
                            'Export results'
                        ],
                        'coherence': [
                            'Loading input data',
                            'Preprocessing',
                            'Window selection',
                            'Coherence calculation',
                            'Statistical analysis',
                            'Quality assessment',
                            'Export coherence map',
                            'Generate report'
                        ]
                    };
                    
                    const steps = stepNames[processingType] || ['Processing step'];
                    return steps[Math.min(step - 1, steps.length - 1)] || 'Processing';
                }

                function generateOutputFiles(job) {
                    const outputs = [];
                    const baseFileName = job.name.replace(/\\s+/g, '_').toLowerCase();
                    
                    switch(job.processingType) {
                        case 'interferogram':
                            outputs.push(\`\${baseFileName}_interferogram.tif\`);
                            outputs.push(\`\${baseFileName}_coherence.tif\`);
                            outputs.push(\`\${baseFileName}_unwrapped.tif\`);
                            break;
                        case 'coherence':
                            outputs.push(\`\${baseFileName}_coherence.tif\`);
                            outputs.push(\`\${baseFileName}_stats.json\`);
                            break;
                        default:
                            outputs.push(\`\${baseFileName}_processed.tif\`);
                    }
                    
                    return outputs;
                }

                function pauseJob() {
                    isPaused = true;
                    self.postMessage({
                        type: 'paused',
                        workerId: self.workerId,
                        jobId: currentJob?.id,
                        timestamp: Date.now()
                    });
                }

                function resumeJob() {
                    isPaused = false;
                    self.postMessage({
                        type: 'resumed',
                        workerId: self.workerId,
                        jobId: currentJob?.id,
                        timestamp: Date.now()
                    });
                }

                function cancelJob() {
                    if (processingInterval) {
                        clearInterval(processingInterval);
                    }
                    
                    self.postMessage({
                        type: 'cancelled',
                        workerId: self.workerId,
                        jobId: currentJob?.id,
                        timestamp: Date.now()
                    });
                    
                    currentJob = null;
                    isPaused = false;
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const worker = new Worker(URL.createObjectURL(blob));
            
            worker.onmessage = (e) => {
                this.handleWorkerMessage(id, e.data);
            };

            return worker;
        }

        handleWorkerMessage(workerId, data) {
            const { type, jobId } = data;
            const job = this.jobs.get(jobId) || this.activeJobs.get(jobId);

            if (!job) return;

            switch (type) {
                case 'jobStarted':
                    this.onJobStarted(workerId, job, data);
                    break;
                case 'progress':
                    this.onJobProgress(workerId, job, data);
                    break;
                case 'completed':
                    this.onJobCompleted(workerId, job, data);
                    break;
                case 'error':
                    this.onJobError(workerId, job, data);
                    break;
                case 'paused':
                    this.onJobPaused(workerId, job, data);
                    break;
                case 'resumed':
                    this.onJobResumed(workerId, job, data);
                    break;
                case 'cancelled':
                    this.onJobCancelled(workerId, job, data);
                    break;
            }

            this.updateJobDisplays();
            this.updateStatistics();
        }

        onJobStarted(workerId, job, data) {
            job.status = 'processing';
            job.startTime = data.timestamp;
            job.workerId = workerId;
            
            this.jobs.delete(job.id);
            this.activeJobs.set(job.id, job);
            
            this.workers[workerId].status = 'busy';
            this.workers[workerId].currentJob = job.id;
            
            this.jobLogger.logJobStart(job);
            this.startNextJobInQueue();
        }

        onJobProgress(workerId, job, data) {
            job.progress = data.progress;
            job.currentStep = data.currentStep;
            job.totalSteps = data.totalSteps;
            job.stepName = data.stepName;
            job.resourceUsage = data.resources;
            job.lastUpdate = data.timestamp;
            
            this.resourceMonitor.updateUsage(data.resources);
        }

        onJobCompleted(workerId, job, data) {
            job.status = 'completed';
            job.endTime = data.timestamp;
            job.processingTime = job.endTime - job.startTime;
            job.outputFiles = data.outputFiles;
            job.progress = 1;
            
            this.activeJobs.delete(job.id);
            this.completedJobs.set(job.id, job);
            
            this.workers[workerId].status = 'idle';
            this.workers[workerId].currentJob = null;
            
            this.statistics.completedJobs++;
            this.statistics.totalProcessingTime += job.processingTime;
            this.statistics.avgProcessingTime = this.statistics.totalProcessingTime / this.statistics.completedJobs;
            
            this.jobLogger.logJobCompletion(job);
            this.notificationSystem.notifyJobCompleted(job);
            this.startNextJobInQueue();
        }

        onJobError(workerId, job, data) {
            job.status = 'failed';
            job.endTime = data.timestamp;
            job.error = data.error;
            job.retryCount = (job.retryCount || 0) + 1;
            
            if (job.retryCount < this.config.maxRetryAttempts) {
                // Retry job
                job.status = 'queued';
                job.priority = Math.min(job.priority + 1, this.config.priorityLevels.length - 1);
                this.jobs.set(job.id, job);
                this.activeJobs.delete(job.id);
            } else {
                // Move to failed jobs
                this.activeJobs.delete(job.id);
                this.failedJobs.set(job.id, job);
                this.statistics.failedJobs++;
            }
            
            this.workers[workerId].status = 'idle';
            this.workers[workerId].currentJob = null;
            
            this.jobLogger.logJobError(job, data.error);
            this.notificationSystem.notifyJobError(job, data.error);
            this.startNextJobInQueue();
        }

        onJobPaused(workerId, job, data) {
            job.status = 'paused';
            job.pausedTime = data.timestamp;
            this.jobLogger.logJobPause(job);
        }

        onJobResumed(workerId, job, data) {
            job.status = 'processing';
            if (job.pausedTime) {
                job.totalPausedTime = (job.totalPausedTime || 0) + (data.timestamp - job.pausedTime);
            }
            this.jobLogger.logJobResume(job);
        }

        onJobCancelled(workerId, job, data) {
            job.status = 'cancelled';
            job.endTime = data.timestamp;
            
            this.activeJobs.delete(job.id);
            this.failedJobs.set(job.id, job);
            
            this.workers[workerId].status = 'idle';
            this.workers[workerId].currentJob = null;
            
            this.jobLogger.logJobCancellation(job);
            this.startNextJobInQueue();
        }

        startResourceMonitoring() {
            setInterval(() => {
                this.updateResourceMeters();
                this.checkResourceAlerts();
            }, 1000);
        }

        updateResourceMeters() {
            // Aggregate resource usage from active jobs
            let totalCpu = 0;
            let totalMemory = 0;
            let totalDisk = 0;
            let activeJobCount = 0;

            this.activeJobs.forEach(job => {
                if (job.resourceUsage) {
                    totalCpu += job.resourceUsage.cpu;
                    totalMemory += job.resourceUsage.memory;
                    totalDisk += job.resourceUsage.disk;
                    activeJobCount++;
                }
            });

            const avgCpu = activeJobCount > 0 ? totalCpu / activeJobCount : 0;
            const memoryUsage = (totalMemory / this.config.resourceLimits.memory) * 100;
            const diskUsage = totalDisk;

            // Update UI meters
            document.getElementById('cpuMeter').style.width = Math.min(avgCpu, 100) + '%';
            document.getElementById('cpuValue').textContent = avgCpu.toFixed(1) + '%';
            
            document.getElementById('memoryMeter').style.width = Math.min(memoryUsage, 100) + '%';
            document.getElementById('memoryValue').textContent = memoryUsage.toFixed(1) + '%';
            
            document.getElementById('diskMeter').style.width = Math.min(diskUsage, 100) + '%';
            document.getElementById('diskValue').textContent = diskUsage.toFixed(1) + '%';

            // Update statistics
            this.statistics.resourceUtilization = {
                cpu: avgCpu,
                memory: memoryUsage,
                disk: diskUsage
            };
        }

        checkResourceAlerts() {
            const threshold = 85; // Could be configurable
            const { cpu, memory, disk } = this.statistics.resourceUtilization;

            if (cpu > threshold || memory > threshold || disk > threshold) {
                this.notificationSystem.notifyResourceAlert(this.statistics.resourceUtilization);
            }
        }

        setupHeartbeat() {
            setInterval(() => {
                this.updateStatistics();
                this.updateJobDisplays();
                this.saveJobState();
            }, this.config.heartbeatInterval);
        }

        showJobModal() {
            document.getElementById('jobModal').style.display = 'flex';
        }

        hideJobModal() {
            document.getElementById('jobModal').style.display = 'none';
            this.resetJobForm();
        }

        resetJobForm() {
            document.getElementById('jobName').value = '';
            document.getElementById('processingParams').value = '';
            document.getElementById('estimatedTime').value = '60';
            document.getElementById('inputFiles').value = '';
        }

        submitNewJob() {
            const jobData = this.getJobFormData();
            
            if (!jobData.name) {
                alert('Please enter a job name');
                return;
            }

            const job = this.createJob(jobData);
            this.addJobToQueue(job);
            this.hideJobModal();
            this.updateJobDisplays();
        }

        getJobFormData() {
            return {
                name: document.getElementById('jobName').value,
                processingType: document.getElementById('processingType').value,
                priority: document.getElementById('jobPriority').value,
                estimatedTime: parseInt(document.getElementById('estimatedTime').value) * 60000,
                outputFormat: document.getElementById('outputFormat').value,
                parameters: document.getElementById('processingParams').value,
                inputFiles: Array.from(document.getElementById('inputFiles').files)
            };
        }

        createJob(jobData) {
            const jobId = 'job_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            
            return {
                id: jobId,
                name: jobData.name,
                processingType: jobData.processingType,
                priority: this.config.priorityLevels.indexOf(jobData.priority),
                status: 'queued',
                progress: 0,
                estimatedTime: jobData.estimatedTime,
                outputFormat: jobData.outputFormat,
                parameters: jobData.parameters,
                inputFiles: jobData.inputFiles.map(file => file.name),
                createdTime: Date.now(),
                retryCount: 0,
                workerId: null
            };
        }

        addJobToQueue(job) {
            this.jobs.set(job.id, job);
            this.statistics.totalJobs++;
            this.jobLogger.logJobCreation(job);
            this.startNextJobInQueue();
        }

        startNextJobInQueue() {
            // Find available worker
            const availableWorker = this.workers.find(w => w.status === 'idle');
            if (!availableWorker) return;

            // Find highest priority job
            const queuedJobs = Array.from(this.jobs.values())
                .filter(job => job.status === 'queued')
                .sort((a, b) => {
                    // Higher priority first, then older jobs first
                    if (a.priority !== b.priority) {
                        return b.priority - a.priority;
                    }
                    return a.createdTime - b.createdTime;
                });

            if (queuedJobs.length === 0) return;

            const nextJob = queuedJobs[0];
            
            // Start job on worker
            availableWorker.worker.postMessage({
                action: 'process',
                jobData: nextJob
            });
        }

        updateJobDisplays() {
            this.updateActiveJobsList();
            this.updateQueuedJobsList();
            this.updateJobCounts();
        }

        updateActiveJobsList() {
            const activeJobsList = document.getElementById('activeJobsList');
            if (!activeJobsList) return;

            activeJobsList.innerHTML = '';

            this.activeJobs.forEach(job => {
                const jobElement = this.createJobElement(job, 'active');
                activeJobsList.appendChild(jobElement);
            });
        }

        updateQueuedJobsList() {
            const queuedJobsList = document.getElementById('queuedJobsList');
            if (!queuedJobsList) return;

            queuedJobsList.innerHTML = '';

            const queuedJobs = Array.from(this.jobs.values())
                .filter(job => job.status === 'queued')
                .sort((a, b) => {
                    if (a.priority !== b.priority) {
                        return b.priority - a.priority;
                    }
                    return a.createdTime - b.createdTime;
                });

            queuedJobs.forEach((job, index) => {
                const jobElement = this.createJobElement(job, 'queued', index + 1);
                queuedJobsList.appendChild(jobElement);
            });
        }

        createJobElement(job, type, queuePosition = null) {
            const jobElement = document.createElement('div');
            jobElement.className = `job-item job-${type}`;
            
            const priorityClass = this.config.priorityLevels[job.priority] || 'normal';
            
            jobElement.innerHTML = `
                <div class="job-header">
                    <div class="job-info">
                        <div class="job-title">
                            ${job.name}
                            <span class="job-type">${job.processingType}</span>
                            <span class="job-priority priority-${priorityClass}">${this.config.priorityLevels[job.priority]}</span>
                        </div>
                        <div class="job-details">
                            ID: ${job.id} • 
                            Created: ${new Date(job.createdTime).toLocaleString()}
                            ${queuePosition ? ` • Queue Position: #${queuePosition}` : ''}
                            ${job.workerId !== null ? ` • Worker: #${job.workerId}` : ''}
                        </div>
                    </div>
                    <div class="job-status">
                        <span class="status-badge status-${job.status}">${job.status}</span>
                    </div>
                </div>
                
                ${type === 'active' ? `
                    <div class="job-progress">
                        <div class="progress-info">
                            <span class="progress-text">${job.stepName || 'Processing'}</span>
                            <span class="progress-percent">${(job.progress * 100).toFixed(1)}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${(job.progress * 100).toFixed(1)}%"></div>
                        </div>
                        <div class="progress-details">
                            Step ${job.currentStep || 0} of ${job.totalSteps || 0}
                            ${job.resourceUsage ? `• CPU: ${job.resourceUsage.cpu.toFixed(1)}% • Memory: ${job.resourceUsage.memory.toFixed(0)}MB` : ''}
                        </div>
                    </div>
                ` : ''}
                
                <div class="job-actions">
                    ${type === 'active' ? `
                        <button class="btn btn-sm btn-warning" onclick="window.processingQueue.pauseJob('${job.id}')">Pause</button>
                        <button class="btn btn-sm btn-danger" onclick="window.processingQueue.cancelJob('${job.id}')">Cancel</button>
                    ` : `
                        <button class="btn btn-sm btn-info" onclick="window.processingQueue.editJob('${job.id}')">Edit</button>
                        <button class="btn btn-sm btn-danger" onclick="window.processingQueue.removeJob('${job.id}')">Remove</button>
                    `}
                    <button class="btn btn-sm btn-secondary" onclick="window.processingQueue.viewJobDetails('${job.id}')">Details</button>
                </div>
            `;

            return jobElement;
        }

        updateJobCounts() {
            document.getElementById('activeJobCount').textContent = this.activeJobs.size;
            document.getElementById('queuedJobCount').textContent = Array.from(this.jobs.values()).filter(job => job.status === 'queued').length;
            document.getElementById('completedJobCount').textContent = this.completedJobs.size;
            document.getElementById('failedJobCount').textContent = this.failedJobs.size;
        }

        updateStatistics() {
            // Calculate throughput (jobs per hour)
            const now = Date.now();
            const oneHourAgo = now - 3600000;
            const recentCompletedJobs = Array.from(this.completedJobs.values())
                .filter(job => job.endTime > oneHourAgo);
            
            const throughput = recentCompletedJobs.length;
            
            // Update UI
            document.getElementById('throughput').textContent = throughput;
            
            if (this.statistics.completedJobs > 0) {
                document.getElementById('avgProcessingTime').textContent = 
                    Math.round(this.statistics.avgProcessingTime / 1000) + 's';
            }
        }

        pauseJob(jobId) {
            const job = this.activeJobs.get(jobId);
            if (job && job.workerId !== null) {
                this.workers[job.workerId].worker.postMessage({ action: 'pause' });
            }
        }

        cancelJob(jobId) {
            const job = this.activeJobs.get(jobId);
            if (job && job.workerId !== null) {
                this.workers[job.workerId].worker.postMessage({ action: 'cancel' });
            }
        }

        removeJob(jobId) {
            if (confirm('Are you sure you want to remove this job?')) {
                this.jobs.delete(jobId);
                this.updateJobDisplays();
            }
        }

        viewJobDetails(jobId) {
            const job = this.jobs.get(jobId) || this.activeJobs.get(jobId) || 
                      this.completedJobs.get(jobId) || this.failedJobs.get(jobId);
            
            if (job) {
                // Show job details modal or panel
                console.log('Job details:', job);
            }
        }

        pauseQueue() {
            this.workers.forEach(worker => {
                if (worker.currentJob) {
                    worker.worker.postMessage({ action: 'pause' });
                }
            });
        }

        resumeQueue() {
            this.workers.forEach(worker => {
                if (worker.currentJob) {
                    worker.worker.postMessage({ action: 'resume' });
                }
            });
        }

        clearCompletedJobs() {
            if (confirm('Clear all completed jobs from history?')) {
                this.completedJobs.clear();
                this.updateJobDisplays();
            }
        }

        switchTab(tabId) {
            this.ui.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tab === tabId);
            });
            
            this.ui.querySelectorAll('.tab-content').forEach(content => {
                content.classList.toggle('active', content.id === `${tabId}-tab`);
            });

            // Update displays for specific tabs
            if (tabId === 'history') {
                this.updateHistoryDisplay();
            } else if (tabId === 'monitoring') {
                this.updateMonitoringCharts();
            }
        }

        updateHistoryDisplay() {
            this.updateHistoryStatistics();
            this.drawProcessingChart();
        }

        updateHistoryStatistics() {
            const totalProcessed = this.completedJobs.size + this.failedJobs.size;
            const successRate = totalProcessed > 0 ? 
                (this.completedJobs.size / totalProcessed * 100).toFixed(1) : 0;
            
            document.getElementById('totalProcessed').textContent = totalProcessed;
            document.getElementById('successRate').textContent = successRate + '%';
            
            const totalCpuHours = this.statistics.totalProcessingTime / (1000 * 60 * 60);
            document.getElementById('totalCpuHours').textContent = totalCpuHours.toFixed(1) + 'h';
        }

        drawProcessingChart() {
            const canvas = this.canvases.processing;
            const ctx = this.contexts.processing;
            
            if (!canvas || !ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Generate sample data for last 24 hours
            const now = Date.now();
            const data = [];
            
            for (let i = 23; i >= 0; i--) {
                const hourStart = now - (i * 60 * 60 * 1000);
                const hourEnd = hourStart + 60 * 60 * 1000;
                
                const completedInHour = Array.from(this.completedJobs.values())
                    .filter(job => job.endTime >= hourStart && job.endTime < hourEnd).length;
                
                data.push(completedInHour);
            }

            // Draw chart
            this.drawLineChart(ctx, canvas, data, 'Jobs Completed per Hour');
        }

        drawLineChart(ctx, canvas, data, title) {
            const margin = { top: 30, right: 20, bottom: 40, left: 50 };
            const plotWidth = canvas.width - margin.left - margin.right;
            const plotHeight = canvas.height - margin.top - margin.bottom;

            const maxValue = Math.max(...data, 1);
            const stepX = plotWidth / (data.length - 1);
            const stepY = plotHeight / maxValue;

            // Draw axes
            ctx.strokeStyle = '#666';
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, canvas.height - margin.bottom);
            ctx.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
            ctx.stroke();

            // Draw data line
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            data.forEach((value, index) => {
                const x = margin.left + index * stepX;
                const y = canvas.height - margin.bottom - (value * stepY);
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();

            // Draw title
            ctx.fillStyle = '#333';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(title, canvas.width / 2, 20);
        }

        loadJobHistory() {
            try {
                const historyData = localStorage.getItem('processingQueueHistory');
                if (historyData) {
                    const parsed = JSON.parse(historyData);
                    this.completedJobs = new Map(parsed.completed || []);
                    this.failedJobs = new Map(parsed.failed || []);
                }
            } catch (error) {
                console.warn('Error loading job history:', error);
            }
        }

        saveJobState() {
            try {
                const stateData = {
                    completed: Array.from(this.completedJobs.entries()),
                    failed: Array.from(this.failedJobs.entries()),
                    statistics: this.statistics
                };
                localStorage.setItem('processingQueueHistory', JSON.stringify(stateData));
            } catch (error) {
                console.warn('Error saving job state:', error);
            }
        }

        exportJobHistory() {
            const historyData = {
                completed: Array.from(this.completedJobs.values()),
                failed: Array.from(this.failedJobs.values()),
                statistics: this.statistics,
                exportTime: new Date().toISOString()
            };

            const blob = new Blob([JSON.stringify(historyData, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `processing_queue_history_${Date.now()}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
        }

        saveConfiguration() {
            // Update configuration from UI
            this.config.maxConcurrentJobs = parseInt(document.getElementById('maxConcurrentJobs').value);
            this.config.jobTimeout = parseInt(document.getElementById('jobTimeout').value) * 60000;
            this.config.maxRetryAttempts = parseInt(document.getElementById('maxRetryAttempts').value);
            
            // Save to storage
            localStorage.setItem('processingQueueConfig', JSON.stringify(this.config));
            
            console.log('Configuration saved');
        }

        destroy() {
            // Terminate all workers
            this.workers.forEach(workerInfo => {
                workerInfo.worker.terminate();
            });
            
            // Save state
            this.saveJobState();
            this.saveConfiguration();
            
            // Remove UI
            if (this.ui) {
                this.ui.remove();
            }
            
            console.log('🧹 Processing Queue destroyed');
        }
    }

    // Supporting classes
    class JobScheduler {
        constructor(config) {
            this.config = config;
        }

        scheduleJobs(jobs) {
            // Priority-based scheduling implementation
            return jobs.sort((a, b) => {
                if (a.priority !== b.priority) {
                    return b.priority - a.priority;
                }
                return a.createdTime - b.createdTime;
            });
        }
    }

    class ResourceMonitor {
        constructor(config) {
            this.config = config;
            this.currentUsage = { cpu: 0, memory: 0, disk: 0 };
        }

        updateUsage(usage) {
            this.currentUsage = usage;
        }

        checkLimits() {
            return this.currentUsage.cpu < this.config.resourceLimits.cpu &&
                   this.currentUsage.memory < this.config.resourceLimits.memory &&
                   this.currentUsage.disk < this.config.resourceLimits.disk;
        }
    }

    class ProcessEngine {
        constructor() {
            this.processors = new Map();
            this.initializeProcessors();
        }

        initializeProcessors() {
            // Initialize various processing algorithms
        }

        process(job) {
            const processor = this.processors.get(job.processingType);
            if (processor) {
                return processor.process(job);
            }
            throw new Error(`Unknown processing type: ${job.processingType}`);
        }
    }

    class JobLogger {
        constructor() {
            this.logs = [];
        }

        logJobCreation(job) {
            this.addLog('info', `Job created: ${job.name} (${job.id})`);
        }

        logJobStart(job) {
            this.addLog('info', `Job started: ${job.name} (${job.id})`);
        }

        logJobCompletion(job) {
            this.addLog('success', `Job completed: ${job.name} (${job.id}) in ${job.processingTime / 1000}s`);
        }

        logJobError(job, error) {
            this.addLog('error', `Job failed: ${job.name} (${job.id}) - ${error}`);
        }

        logJobPause(job) {
            this.addLog('warning', `Job paused: ${job.name} (${job.id})`);
        }

        logJobResume(job) {
            this.addLog('info', `Job resumed: ${job.name} (${job.id})`);
        }

        logJobCancellation(job) {
            this.addLog('warning', `Job cancelled: ${job.name} (${job.id})`);
        }

        addLog(level, message) {
            this.logs.push({
                level: level,
                message: message,
                timestamp: new Date().toISOString()
            });

            // Keep only last 1000 logs
            if (this.logs.length > 1000) {
                this.logs.shift();
            }
        }
    }

    class NotificationSystem {
        notifyJobCompleted(job) {
            console.log(`✅ Job completed: ${job.name}`);
        }

        notifyJobError(job, error) {
            console.error(`❌ Job failed: ${job.name} - ${error}`);
        }

        notifyResourceAlert(usage) {
            console.warn(`⚠️ High resource usage - CPU: ${usage.cpu.toFixed(1)}%, Memory: ${usage.memory.toFixed(1)}%`);
        }
    }

    // Create global instance
    window.ProcessingQueue = ProcessingQueue;
    
    // Auto-initialize if container exists
    const queueContainer = document.querySelector('.processing-queue-mount');
    if (queueContainer) {
        window.processingQueue = new ProcessingQueue({
            container: queueContainer
        });
    }
    
    console.log('⚙️ Processing Queue Management System loaded successfully');

})();