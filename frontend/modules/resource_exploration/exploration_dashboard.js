/**
 * Prometheus Platform - Interactive Mineral Exploration Dashboard
 * ==============================================================
 * 
 * Advanced web interface for AI-powered mineral prospectivity mapping and 
 * target generation with real-time visualization and analysis.
 * 
 * Key Features:
 * - Interactive prospectivity mapping with real-time ML predictions
 * - Multi-layer geological data visualization and integration
 * - Advanced target generation and ranking with MCDA
 * - Economic modeling with Monte Carlo simulation
 * - Risk assessment and portfolio optimization
 * - 3D/4D visualization of geological models
 * - Real-time collaboration and data sharing
 * - Export capabilities for reports and presentations
 * 
 * Author: Prometheus Development Team
 * Date: 2025-08-09
 * Version: 1.0.0
 */

class ExplorationDashboard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            apiBaseUrl: config.apiBaseUrl || '/api/exploration',
            mapProvider: config.mapProvider || 'mapbox',
            theme: config.theme || 'dark',
            autoUpdate: config.autoUpdate || true,
            updateInterval: config.updateInterval || 30000,
            maxTargets: config.maxTargets || 100,
            ...config
        };
        
        // State management
        this.state = {
            currentDataLayers: new Map(),
            prospectivityResults: null,
            explorationTargets: [],
            selectedTargets: new Set(),
            analysisResults: {},
            viewMode: 'map', // map, targets, analysis, economics
            isLoading: false,
            filters: {
                targetTypes: new Set(['all']),
                minProspectivity: 0.0,
                maxRisk: 5.0,
                minNPV: 0
            }
        };
        
        // UI components
        this.components = {};
        this.charts = {};
        this.map = null;
        this.scene3D = null;
        
        // Data management
        this.dataCache = new Map();
        this.eventListeners = new Map();
        
        // Performance monitoring
        this.performanceMetrics = {
            renderTime: 0,
            dataLoadTime: 0,
            analysisTime: 0
        };
        
        this.initialize();
    }
    
    async initialize() {
        try {
            console.log('Initializing Exploration Dashboard...');
            
            // Create UI structure
            this.createUIStructure();
            
            // Initialize components
            await this.initializeMap();
            this.initializeCharts();
            this.initializeDataManager();
            this.initializeEventListeners();
            
            // Load initial data
            await this.loadInitialData();
            
            // Start auto-update if enabled
            if (this.config.autoUpdate) {
                this.startAutoUpdate();
            }
            
            console.log('Exploration Dashboard initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize Exploration Dashboard:', error);
            this.showError('Dashboard initialization failed', error.message);
        }
    }
    
    createUIStructure() {
        this.container.innerHTML = `
            <div class="exploration-dashboard">
                <!-- Header -->
                <div class="dashboard-header">
                    <div class="header-left">
                        <h1><i class="fas fa-gem"></i> Mineral Exploration Dashboard</h1>
                        <div class="status-indicator">
                            <span class="status-light" id="connectionStatus"></span>
                            <span id="statusText">Ready</span>
                        </div>
                    </div>
                    <div class="header-center">
                        <div class="view-switcher">
                            <button class="view-btn active" data-view="map">
                                <i class="fas fa-map"></i> Map View
                            </button>
                            <button class="view-btn" data-view="targets">
                                <i class="fas fa-bullseye"></i> Targets
                            </button>
                            <button class="view-btn" data-view="analysis">
                                <i class="fas fa-chart-line"></i> Analysis
                            </button>
                            <button class="view-btn" data-view="economics">
                                <i class="fas fa-dollar-sign"></i> Economics
                            </button>
                        </div>
                    </div>
                    <div class="header-right">
                        <button class="btn btn-primary" id="generateTargetsBtn">
                            <i class="fas fa-magic"></i> Generate Targets
                        </button>
                        <button class="btn btn-secondary" id="exportBtn">
                            <i class="fas fa-download"></i> Export
                        </button>
                        <div class="settings-dropdown">
                            <button class="btn btn-icon" id="settingsBtn">
                                <i class="fas fa-cog"></i>
                            </button>
                            <div class="dropdown-menu" id="settingsMenu">
                                <div class="dropdown-item">
                                    <label>Theme:</label>
                                    <select id="themeSelect">
                                        <option value="dark">Dark</option>
                                        <option value="light">Light</option>
                                    </select>
                                </div>
                                <div class="dropdown-item">
                                    <label>Auto-update:</label>
                                    <input type="checkbox" id="autoUpdateCheck" ${this.config.autoUpdate ? 'checked' : ''}>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Main Content -->
                <div class="dashboard-main">
                    <!-- Sidebar -->
                    <div class="sidebar">
                        <div class="sidebar-section">
                            <h3><i class="fas fa-layer-group"></i> Data Layers</h3>
                            <div class="data-layers-panel" id="dataLayersPanel">
                                <div class="layer-controls">
                                    <button class="btn btn-sm btn-primary" id="addLayerBtn">
                                        <i class="fas fa-plus"></i> Add Layer
                                    </button>
                                    <button class="btn btn-sm btn-secondary" id="manageLayersBtn">
                                        <i class="fas fa-cog"></i> Manage
                                    </button>
                                </div>
                                <div class="layers-list" id="layersList"></div>
                            </div>
                        </div>
                        
                        <div class="sidebar-section">
                            <h3><i class="fas fa-filter"></i> Filters</h3>
                            <div class="filters-panel" id="filtersPanel">
                                <div class="filter-group">
                                    <label>Target Types:</label>
                                    <div class="checkbox-group" id="targetTypeFilters">
                                        <label><input type="checkbox" value="all" checked> All</label>
                                        <label><input type="checkbox" value="gold"> Gold</label>
                                        <label><input type="checkbox" value="copper"> Copper</label>
                                        <label><input type="checkbox" value="lithium"> Lithium</label>
                                        <label><input type="checkbox" value="rare_earth"> REE</label>
                                        <label><input type="checkbox" value="base_metals"> Base Metals</label>
                                    </div>
                                </div>
                                
                                <div class="filter-group">
                                    <label>Min Prospectivity: <span id="prospectivityValue">0.0</span></label>
                                    <input type="range" id="prospectivityFilter" min="0" max="1" step="0.1" value="0">
                                </div>
                                
                                <div class="filter-group">
                                    <label>Max Risk Level: <span id="riskValue">5</span></label>
                                    <input type="range" id="riskFilter" min="1" max="5" step="1" value="5">
                                </div>
                                
                                <div class="filter-group">
                                    <label>Min NPV (M$): <span id="npvValue">0</span></label>
                                    <input type="range" id="npvFilter" min="0" max="1000" step="10" value="0">
                                </div>
                                
                                <button class="btn btn-primary btn-block" id="applyFiltersBtn">Apply Filters</button>
                                <button class="btn btn-secondary btn-block" id="resetFiltersBtn">Reset</button>
                            </div>
                        </div>
                        
                        <div class="sidebar-section">
                            <h3><i class="fas fa-info-circle"></i> Statistics</h3>
                            <div class="stats-panel" id="statsPanel">
                                <div class="stat-item">
                                    <span class="stat-label">Total Targets:</span>
                                    <span class="stat-value" id="totalTargets">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">High Priority:</span>
                                    <span class="stat-value" id="highPriorityTargets">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Total NPV:</span>
                                    <span class="stat-value" id="totalNPV">$0M</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Avg Risk:</span>
                                    <span class="stat-value" id="avgRisk">-</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Content Area -->
                    <div class="content-area">
                        <!-- Map View -->
                        <div class="view-panel active" id="mapView">
                            <div class="map-container">
                                <div id="explorationMap"></div>
                                <div class="map-controls">
                                    <div class="control-group">
                                        <button class="btn btn-sm" id="zoomInBtn"><i class="fas fa-plus"></i></button>
                                        <button class="btn btn-sm" id="zoomOutBtn"><i class="fas fa-minus"></i></button>
                                        <button class="btn btn-sm" id="resetViewBtn"><i class="fas fa-home"></i></button>
                                    </div>
                                    <div class="control-group">
                                        <button class="btn btn-sm" id="toggle3DBtn"><i class="fas fa-cube"></i> 3D</button>
                                        <button class="btn btn-sm" id="measureBtn"><i class="fas fa-ruler"></i></button>
                                        <button class="btn btn-sm" id="drawBtn"><i class="fas fa-draw-polygon"></i></button>
                                    </div>
                                </div>
                                <div class="legend-panel" id="legendPanel"></div>
                            </div>
                        </div>
                        
                        <!-- Targets View -->
                        <div class="view-panel" id="targetsView">
                            <div class="targets-header">
                                <h2>Exploration Targets</h2>
                                <div class="targets-controls">
                                    <select id="sortTargetsBy">
                                        <option value="priority">Priority Score</option>
                                        <option value="npv">NPV</option>
                                        <option value="prospectivity">Prospectivity</option>
                                        <option value="risk">Risk Level</option>
                                    </select>
                                    <button class="btn btn-sm" id="exportTargetsBtn">Export</button>
                                </div>
                            </div>
                            <div class="targets-content">
                                <div class="targets-grid" id="targetsGrid"></div>
                                <div class="target-details" id="targetDetails"></div>
                            </div>
                        </div>
                        
                        <!-- Analysis View -->
                        <div class="view-panel" id="analysisView">
                            <div class="analysis-header">
                                <h2>Prospectivity Analysis</h2>
                                <div class="analysis-controls">
                                    <select id="analysisType">
                                        <option value="prospectivity">Prospectivity Mapping</option>
                                        <option value="uncertainty">Uncertainty Analysis</option>
                                        <option value="feature_importance">Feature Importance</option>
                                        <option value="model_performance">Model Performance</option>
                                    </select>
                                    <button class="btn btn-sm btn-primary" id="runAnalysisBtn">Run Analysis</button>
                                </div>
                            </div>
                            <div class="analysis-content">
                                <div class="analysis-charts" id="analysisCharts">
                                    <div class="chart-container">
                                        <canvas id="prospectivityChart"></canvas>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="uncertaintyChart"></canvas>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="featureImportanceChart"></canvas>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="modelPerformanceChart"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Economics View -->
                        <div class="view-panel" id="economicsView">
                            <div class="economics-header">
                                <h2>Economic Analysis</h2>
                                <div class="economics-controls">
                                    <button class="btn btn-sm btn-primary" id="runEconomicsBtn">Run Analysis</button>
                                    <button class="btn btn-sm btn-secondary" id="portfolioOptBtn">Optimize Portfolio</button>
                                </div>
                            </div>
                            <div class="economics-content">
                                <div class="economics-summary" id="economicsSummary">
                                    <div class="metric-card">
                                        <h4>Portfolio NPV</h4>
                                        <div class="metric-value" id="portfolioNPV">$0M</div>
                                        <div class="metric-trend" id="npvTrend">-</div>
                                    </div>
                                    <div class="metric-card">
                                        <h4>Expected Return</h4>
                                        <div class="metric-value" id="expectedReturn">0%</div>
                                        <div class="metric-trend" id="returnTrend">-</div>
                                    </div>
                                    <div class="metric-card">
                                        <h4>Risk Score</h4>
                                        <div class="metric-value" id="riskScore">-</div>
                                        <div class="metric-trend" id="riskTrend">-</div>
                                    </div>
                                    <div class="metric-card">
                                        <h4>Payback Period</h4>
                                        <div class="metric-value" id="paybackPeriod">- years</div>
                                        <div class="metric-trend" id="paybackTrend">-</div>
                                    </div>
                                </div>
                                <div class="economics-charts" id="economicsCharts">
                                    <div class="chart-container">
                                        <canvas id="npvDistributionChart"></canvas>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="sensitivityChart"></canvas>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="riskReturnChart"></canvas>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="portfolioCompositionChart"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Progress Overlay -->
                <div class="progress-overlay" id="progressOverlay" style="display: none;">
                    <div class="progress-content">
                        <div class="progress-spinner"></div>
                        <div class="progress-text" id="progressText">Processing...</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Modals -->
                <div class="modal" id="targetDetailsModal">
                    <div class="modal-content">
                        <span class="close">&times;</span>
                        <h2 id="modalTargetName">Target Details</h2>
                        <div id="modalTargetContent"></div>
                    </div>
                </div>
                
                <div class="modal" id="addLayerModal">
                    <div class="modal-content">
                        <span class="close">&times;</span>
                        <h2>Add Data Layer</h2>
                        <div class="modal-form" id="addLayerForm">
                            <div class="form-group">
                                <label>Layer Name:</label>
                                <input type="text" id="layerName" placeholder="Enter layer name">
                            </div>
                            <div class="form-group">
                                <label>Data Type:</label>
                                <select id="layerType">
                                    <option value="geophysics">Geophysics</option>
                                    <option value="geochemistry">Geochemistry</option>
                                    <option value="geology">Geology</option>
                                    <option value="remote_sensing">Remote Sensing</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Data Source:</label>
                                <input type="file" id="layerFile" accept=".csv,.tif,.geojson,.shp">
                            </div>
                            <div class="form-buttons">
                                <button class="btn btn-primary" id="uploadLayerBtn">Upload</button>
                                <button class="btn btn-secondary" id="cancelLayerBtn">Cancel</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Store component references
        this.components = {
            map: document.getElementById('explorationMap'),
            layersList: document.getElementById('layersList'),
            targetsGrid: document.getElementById('targetsGrid'),
            targetDetails: document.getElementById('targetDetails'),
            analysisCharts: document.getElementById('analysisCharts'),
            economicsCharts: document.getElementById('economicsCharts'),
            progressOverlay: document.getElementById('progressOverlay'),
            progressText: document.getElementById('progressText'),
            progressFill: document.getElementById('progressFill')
        };
    }
    
    async initializeMap() {
        try {
            // Initialize mapping library (Mapbox GL JS or Leaflet)
            if (this.config.mapProvider === 'mapbox' && window.mapboxgl) {
                await this.initializeMapbox();
            } else {
                await this.initializeLeaflet();
            }
            
            // Add custom controls
            this.addMapControls();
            
        } catch (error) {
            console.error('Failed to initialize map:', error);
            throw error;
        }
    }
    
    async initializeMapbox() {
        mapboxgl.accessToken = this.config.mapboxToken || 'pk.eyJ1IjoicHJvbWV0aGV1cy1kZXYiLCJhIjoiY2x2d2RlbnJyMDBhZDJrczBuYzJhNHJxaCJ9.example';
        
        this.map = new mapboxgl.Map({
            container: 'explorationMap',
            style: 'mapbox://styles/mapbox/satellite-v9',
            center: [-120, 45], // Default to western North America
            zoom: 6,
            projection: 'mercator'
        });
        
        await new Promise((resolve) => {
            this.map.on('load', resolve);
        });
        
        // Add 3D terrain
        this.map.addSource('mapbox-dem', {
            'type': 'raster-dem',
            'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
            'tileSize': 512,
            'maxzoom': 14
        });
        
        this.map.setTerrain({ 'source': 'mapbox-dem', 'exaggeration': 1.5 });
        
        // Add navigation controls
        this.map.addControl(new mapboxgl.NavigationControl());
        this.map.addControl(new mapboxgl.FullscreenControl());
    }
    
    async initializeLeaflet() {
        // Fallback to Leaflet if Mapbox is not available
        this.map = L.map('explorationMap').setView([45, -120], 6);
        
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '&copy; Esri &mdash; Source: Esri, Maxar, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
            maxZoom: 18
        }).addTo(this.map);
        
        // Add scale control
        L.control.scale().addTo(this.map);
    }
    
    initializeCharts() {
        // Initialize Chart.js charts
        this.charts = {
            prospectivity: this.createProspectivityChart(),
            uncertainty: this.createUncertaintyChart(),
            featureImportance: this.createFeatureImportanceChart(),
            modelPerformance: this.createModelPerformanceChart(),
            npvDistribution: this.createNPVDistributionChart(),
            sensitivity: this.createSensitivityChart(),
            riskReturn: this.createRiskReturnChart(),
            portfolioComposition: this.createPortfolioCompositionChart()
        };
    }
    
    createProspectivityChart() {
        const ctx = document.getElementById('prospectivityChart').getContext('2d');
        return new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'High Prospectivity',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)'
                }, {
                    label: 'Medium Prospectivity',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)'
                }, {
                    label: 'Low Prospectivity',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Prospectivity Distribution'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Longitude'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Latitude'
                        }
                    }
                }
            }
        });
    }
    
    createUncertaintyChart() {
        const ctx = document.getElementById('uncertaintyChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Prediction Uncertainty',
                    data: [],
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Uncertainty Analysis'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Uncertainty (%)'
                        }
                    }
                }
            }
        });
    }
    
    createFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart').getContext('2d');
        return new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Feature Importance',
                    data: [],
                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance Ranking'
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    }
                }
            }
        });
    }
    
    createModelPerformanceChart() {
        const ctx = document.getElementById('modelPerformanceChart').getContext('2d');
        return new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                datasets: [{
                    label: 'Random Forest',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                }, {
                    label: 'Neural Network',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Performance Comparison'
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                }
            }
        });
    }
    
    createNPVDistributionChart() {
        const ctx = document.getElementById('npvDistributionChart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'NPV Distribution',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'NPV Distribution (Monte Carlo)'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'NPV (Million USD)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Probability Density'
                        }
                    }
                }
            }
        });
    }
    
    createSensitivityChart() {
        const ctx = document.getElementById('sensitivityChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Sensitivity',
                    data: [],
                    backgroundColor: 'rgba(255, 205, 86, 0.6)',
                    borderColor: 'rgba(255, 205, 86, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Parameter Sensitivity Analysis'
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'NPV Impact (%)'
                        }
                    }
                }
            }
        });
    }
    
    createRiskReturnChart() {
        const ctx = document.getElementById('riskReturnChart').getContext('2d');
        return new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Exploration Targets',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Risk vs Return Analysis'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Risk Score'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Expected Return (%)'
                        }
                    }
                }
            }
        });
    }
    
    createPortfolioCompositionChart() {
        const ctx = document.getElementById('portfolioCompositionChart').getContext('2d');
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF',
                        '#FF9F40'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Portfolio Composition by Target Type'
                    }
                }
            }
        });
    }
    
    initializeDataManager() {
        // Data management system for caching and synchronization
        this.dataManager = {
            cache: new Map(),
            
            async fetchData(endpoint, options = {}) {
                const cacheKey = `${endpoint}_${JSON.stringify(options)}`;
                
                if (this.cache.has(cacheKey) && !options.forceRefresh) {
                    return this.cache.get(cacheKey);
                }
                
                try {
                    const response = await fetch(`${this.config.apiBaseUrl}${endpoint}`, {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                            ...options.headers
                        },
                        ...options
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    this.cache.set(cacheKey, data);
                    return data;
                    
                } catch (error) {
                    console.error(`Failed to fetch data from ${endpoint}:`, error);
                    throw error;
                }
            },
            
            async postData(endpoint, data, options = {}) {
                try {
                    const response = await fetch(`${this.config.apiBaseUrl}${endpoint}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            ...options.headers
                        },
                        body: JSON.stringify(data),
                        ...options
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    return await response.json();
                    
                } catch (error) {
                    console.error(`Failed to post data to ${endpoint}:`, error);
                    throw error;
                }
            },
            
            clearCache(pattern = null) {
                if (pattern) {
                    for (const key of this.cache.keys()) {
                        if (key.includes(pattern)) {
                            this.cache.delete(key);
                        }
                    }
                } else {
                    this.cache.clear();
                }
            }
        };
    }
    
    initializeEventListeners() {
        // View switching
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const viewName = e.target.closest('.view-btn').dataset.view;
                this.switchView(viewName);
            });
        });
        
        // Filter controls
        document.getElementById('prospectivityFilter').addEventListener('input', (e) => {
            document.getElementById('prospectivityValue').textContent = e.target.value;
            this.state.filters.minProspectivity = parseFloat(e.target.value);
        });
        
        document.getElementById('riskFilter').addEventListener('input', (e) => {
            document.getElementById('riskValue').textContent = e.target.value;
            this.state.filters.maxRisk = parseInt(e.target.value);
        });
        
        document.getElementById('npvFilter').addEventListener('input', (e) => {
            document.getElementById('npvValue').textContent = e.target.value;
            this.state.filters.minNPV = parseInt(e.target.value) * 1000000; // Convert to USD
        });
        
        // Filter application
        document.getElementById('applyFiltersBtn').addEventListener('click', () => {
            this.applyFilters();
        });
        
        document.getElementById('resetFiltersBtn').addEventListener('click', () => {
            this.resetFilters();
        });
        
        // Target generation
        document.getElementById('generateTargetsBtn').addEventListener('click', () => {
            this.generateTargets();
        });
        
        // Analysis controls
        document.getElementById('runAnalysisBtn').addEventListener('click', () => {
            const analysisType = document.getElementById('analysisType').value;
            this.runAnalysis(analysisType);
        });
        
        // Economics controls
        document.getElementById('runEconomicsBtn').addEventListener('click', () => {
            this.runEconomicAnalysis();
        });
        
        document.getElementById('portfolioOptBtn').addEventListener('click', () => {
            this.optimizePortfolio();
        });
        
        // Export functionality
        document.getElementById('exportBtn').addEventListener('click', () => {
            this.showExportDialog();
        });
        
        // Map controls
        if (document.getElementById('zoomInBtn')) {
            document.getElementById('zoomInBtn').addEventListener('click', () => {
                this.map.zoomIn();
            });
        }
        
        if (document.getElementById('zoomOutBtn')) {
            document.getElementById('zoomOutBtn').addEventListener('click', () => {
                this.map.zoomOut();
            });
        }
        
        // Modal controls
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', (e) => {
                e.target.closest('.modal').style.display = 'none';
            });
        });
        
        // Settings
        document.getElementById('themeSelect').addEventListener('change', (e) => {
            this.changeTheme(e.target.value);
        });
        
        document.getElementById('autoUpdateCheck').addEventListener('change', (e) => {
            this.config.autoUpdate = e.target.checked;
            if (this.config.autoUpdate) {
                this.startAutoUpdate();
            } else {
                this.stopAutoUpdate();
            }
        });
    }
    
    async loadInitialData() {
        try {
            this.showProgress('Loading initial data...', 0);
            
            // Load available data layers
            this.showProgress('Loading data layers...', 25);
            const layers = await this.dataManager.fetchData('/layers');
            this.updateDataLayers(layers);
            
            // Load existing targets if any
            this.showProgress('Loading exploration targets...', 50);
            const targets = await this.dataManager.fetchData('/targets');
            this.updateTargets(targets);
            
            // Load analysis results
            this.showProgress('Loading analysis results...', 75);
            const analysisResults = await this.dataManager.fetchData('/analysis/latest');
            this.updateAnalysisResults(analysisResults);
            
            this.showProgress('Complete!', 100);
            setTimeout(() => this.hideProgress(), 500);
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.hideProgress();
            this.showError('Failed to load initial data', error.message);
        }
    }
    
    switchView(viewName) {
        // Update button states
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
        
        // Update view panels
        document.querySelectorAll('.view-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`${viewName}View`).classList.add('active');
        
        // Update state
        this.state.viewMode = viewName;
        
        // Trigger view-specific updates
        switch (viewName) {
            case 'map':
                this.updateMapView();
                break;
            case 'targets':
                this.updateTargetsView();
                break;
            case 'analysis':
                this.updateAnalysisView();
                break;
            case 'economics':
                this.updateEconomicsView();
                break;
        }
    }
    
    async generateTargets() {
        try {
            this.showProgress('Generating exploration targets...', 0);
            
            // Prepare request data
            const requestData = {
                data_layers: Array.from(this.state.currentDataLayers.keys()),
                target_types: Array.from(this.state.filters.targetTypes),
                min_prospectivity: this.state.filters.minProspectivity,
                max_targets: this.config.maxTargets
            };
            
            this.showProgress('Running prospectivity mapping...', 25);
            
            // Generate targets via API
            const response = await this.dataManager.postData('/generate-targets', requestData);
            
            this.showProgress('Processing results...', 75);
            
            // Update targets
            this.updateTargets(response.targets);
            
            // Update prospectivity results
            if (response.prospectivity_data) {
                this.updateProspectivityResults(response.prospectivity_data);
            }
            
            this.showProgress('Complete!', 100);
            setTimeout(() => this.hideProgress(), 500);
            
            // Show success message
            this.showNotification('Success', `Generated ${response.targets.length} exploration targets`, 'success');
            
        } catch (error) {
            console.error('Failed to generate targets:', error);
            this.hideProgress();
            this.showError('Target generation failed', error.message);
        }
    }
    
    async runAnalysis(analysisType) {
        try {
            this.showProgress(`Running ${analysisType} analysis...`, 0);
            
            const response = await this.dataManager.postData('/analysis/run', {
                analysis_type: analysisType,
                targets: this.state.explorationTargets.map(t => t.id)
            });
            
            this.showProgress('Processing results...', 75);
            
            // Update analysis results
            this.state.analysisResults[analysisType] = response.results;
            this.updateAnalysisCharts(analysisType, response.results);
            
            this.showProgress('Complete!', 100);
            setTimeout(() => this.hideProgress(), 500);
            
        } catch (error) {
            console.error(`Failed to run ${analysisType} analysis:`, error);
            this.hideProgress();
            this.showError('Analysis failed', error.message);
        }
    }
    
    async runEconomicAnalysis() {
        try {
            this.showProgress('Running economic analysis...', 0);
            
            const selectedTargets = Array.from(this.state.selectedTargets);
            if (selectedTargets.length === 0) {
                this.showError('No targets selected', 'Please select targets for economic analysis');
                return;
            }
            
            const response = await this.dataManager.postData('/economics/analyze', {
                target_ids: selectedTargets,
                include_monte_carlo: true,
                n_simulations: 1000
            });
            
            this.showProgress('Processing results...', 75);
            
            // Update economics results
            this.updateEconomicsResults(response);
            
            this.showProgress('Complete!', 100);
            setTimeout(() => this.hideProgress(), 500);
            
        } catch (error) {
            console.error('Failed to run economic analysis:', error);
            this.hideProgress();
            this.showError('Economic analysis failed', error.message);
        }
    }
    
    updateTargets(targets) {
        this.state.explorationTargets = targets;
        this.updateTargetsView();
        this.updateMapTargets();
        this.updateStatistics();
    }
    
    updateTargetsView() {
        const targetsGrid = this.components.targetsGrid;
        
        // Filter targets based on current filters
        const filteredTargets = this.getFilteredTargets();
        
        // Sort targets
        const sortBy = document.getElementById('sortTargetsBy')?.value || 'priority';
        filteredTargets.sort((a, b) => {
            switch (sortBy) {
                case 'npv':
                    return (b.npv_estimate || 0) - (a.npv_estimate || 0);
                case 'prospectivity':
                    return b.prospectivity_score - a.prospectivity_score;
                case 'risk':
                    return a.average_risk - b.average_risk;
                default:
                    return (b.priority_score || 0) - (a.priority_score || 0);
            }
        });
        
        // Generate target cards
        targetsGrid.innerHTML = filteredTargets.map(target => `
            <div class="target-card" data-target-id="${target.id}" onclick="dashboard.selectTarget('${target.id}')">
                <div class="target-header">
                    <h4>${target.name}</h4>
                    <div class="target-type-badge ${target.target_type}">${target.target_type}</div>
                </div>
                <div class="target-metrics">
                    <div class="metric">
                        <span class="metric-label">Priority:</span>
                        <span class="metric-value">${(target.priority_score || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">NPV:</span>
                        <span class="metric-value">$${((target.npv_estimate || 0) / 1e6).toFixed(1)}M</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Risk:</span>
                        <span class="metric-value risk-${this.getRiskLevel(target.average_risk)}">
                            ${(target.average_risk || 0).toFixed(1)}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence:</span>
                        <span class="metric-value">${((target.confidence || 0) * 100).toFixed(0)}%</span>
                    </div>
                </div>
                <div class="target-location">
                    📍 ${target.location?.coordinates?.[1]?.toFixed(4)}, ${target.location?.coordinates?.[0]?.toFixed(4)}
                </div>
                <div class="target-recommendation">
                    ${target.recommendation || 'No recommendation available'}
                </div>
            </div>
        `).join('');
    }
    
    selectTarget(targetId) {
        // Toggle selection
        if (this.state.selectedTargets.has(targetId)) {
            this.state.selectedTargets.delete(targetId);
        } else {
            this.state.selectedTargets.add(targetId);
        }
        
        // Update UI
        this.updateTargetSelection();
        
        // Show target details
        const target = this.state.explorationTargets.find(t => t.id === targetId);
        if (target) {
            this.showTargetDetails(target);
        }
    }
    
    showTargetDetails(target) {
        const modal = document.getElementById('targetDetailsModal');
        const nameElement = document.getElementById('modalTargetName');
        const contentElement = document.getElementById('modalTargetContent');
        
        nameElement.textContent = target.name;
        
        contentElement.innerHTML = `
            <div class="target-details-content">
                <div class="details-section">
                    <h3>General Information</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <label>Target Type:</label>
                            <span>${target.target_type}</span>
                        </div>
                        <div class="detail-item">
                            <label>Location:</label>
                            <span>${target.location?.coordinates?.[1]?.toFixed(6)}, ${target.location?.coordinates?.[0]?.toFixed(6)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Area:</label>
                            <span>${target.area_km2?.toFixed(2)} km²</span>
                        </div>
                        <div class="detail-item">
                            <label>Depth Estimate:</label>
                            <span>${target.depth_estimate?.toFixed(0)} m</span>
                        </div>
                    </div>
                </div>
                
                <div class="details-section">
                    <h3>Geological Assessment</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <label>Prospectivity Score:</label>
                            <span class="score-value">${target.prospectivity_score?.toFixed(3)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Confidence:</label>
                            <span>${((target.confidence || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <label>Geological Risk:</label>
                            <span class="risk-value">${target.geological_risk || 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Technical Risk:</label>
                            <span class="risk-value">${target.technical_risk || 'N/A'}</span>
                        </div>
                    </div>
                </div>
                
                <div class="details-section">
                    <h3>Economic Assessment</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <label>Estimated Grade:</label>
                            <span>${target.estimated_grade?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Estimated Tonnage:</label>
                            <span>${((target.estimated_tonnage || 0) / 1e6).toFixed(1)}M tonnes</span>
                        </div>
                        <div class="detail-item">
                            <label>NPV Estimate:</label>
                            <span class="npv-value">$${((target.npv_estimate || 0) / 1e6).toFixed(1)}M</span>
                        </div>
                        <div class="detail-item">
                            <label>Payback Period:</label>
                            <span>${target.payback_period?.toFixed(1) || 'N/A'} years</span>
                        </div>
                    </div>
                </div>
                
                <div class="details-section">
                    <h3>Risk Assessment</h3>
                    <div class="risk-breakdown">
                        <div class="risk-item">
                            <label>Environmental:</label>
                            <div class="risk-bar">
                                <div class="risk-fill" style="width: ${(target.environmental_risk || 0) * 20}%"></div>
                            </div>
                            <span>${target.environmental_risk || 'N/A'}</span>
                        </div>
                        <div class="risk-item">
                            <label>Political:</label>
                            <div class="risk-bar">
                                <div class="risk-fill" style="width: ${(target.political_risk || 0) * 20}%"></div>
                            </div>
                            <span>${target.political_risk || 'N/A'}</span>
                        </div>
                        <div class="risk-item">
                            <label>Market:</label>
                            <div class="risk-bar">
                                <div class="risk-fill" style="width: ${(target.market_risk || 0) * 20}%"></div>
                            </div>
                            <span>${target.market_risk || 'N/A'}</span>
                        </div>
                    </div>
                </div>
                
                <div class="details-section">
                    <h3>Recommendation</h3>
                    <div class="recommendation-box">
                        <p>${target.recommendation || 'No recommendation available'}</p>
                    </div>
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }
    
    getFilteredTargets() {
        return this.state.explorationTargets.filter(target => {
            // Type filter
            const typeFilter = this.state.filters.targetTypes;
            if (!typeFilter.has('all') && !typeFilter.has(target.target_type)) {
                return false;
            }
            
            // Prospectivity filter
            if (target.prospectivity_score < this.state.filters.minProspectivity) {
                return false;
            }
            
            // Risk filter
            if (target.average_risk > this.state.filters.maxRisk) {
                return false;
            }
            
            // NPV filter
            if ((target.npv_estimate || 0) < this.state.filters.minNPV) {
                return false;
            }
            
            return true;
        });
    }
    
    updateStatistics() {
        const targets = this.getFilteredTargets();
        
        document.getElementById('totalTargets').textContent = targets.length;
        
        const highPriority = targets.filter(t => (t.priority_score || 0) > 0.7).length;
        document.getElementById('highPriorityTargets').textContent = highPriority;
        
        const totalNPV = targets.reduce((sum, t) => sum + (t.npv_estimate || 0), 0) / 1e6;
        document.getElementById('totalNPV').textContent = `$${totalNPV.toFixed(1)}M`;
        
        const avgRisk = targets.length > 0 ? 
            targets.reduce((sum, t) => sum + (t.average_risk || 0), 0) / targets.length : 0;
        document.getElementById('avgRisk').textContent = avgRisk.toFixed(1);
    }
    
    getRiskLevel(riskScore) {
        if (riskScore <= 2) return 'low';
        if (riskScore <= 3) return 'medium';
        return 'high';
    }
    
    showProgress(message, progress) {
        this.components.progressOverlay.style.display = 'flex';
        this.components.progressText.textContent = message;
        this.components.progressFill.style.width = `${progress}%`;
    }
    
    hideProgress() {
        this.components.progressOverlay.style.display = 'none';
    }
    
    showError(title, message) {
        // Simple error notification - could be enhanced with a proper notification system
        alert(`${title}: ${message}`);
    }
    
    showNotification(title, message, type = 'info') {
        // Simple notification - could be enhanced with a proper notification system
        console.log(`${type.toUpperCase()}: ${title} - ${message}`);
    }
    
    startAutoUpdate() {
        if (this.autoUpdateInterval) {
            clearInterval(this.autoUpdateInterval);
        }
        
        this.autoUpdateInterval = setInterval(() => {
            this.refreshData();
        }, this.config.updateInterval);
    }
    
    stopAutoUpdate() {
        if (this.autoUpdateInterval) {
            clearInterval(this.autoUpdateInterval);
            this.autoUpdateInterval = null;
        }
    }
    
    async refreshData() {
        try {
            // Refresh targets
            const targets = await this.dataManager.fetchData('/targets', { forceRefresh: true });
            this.updateTargets(targets);
            
            // Update status
            document.getElementById('statusText').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
            
        } catch (error) {
            console.error('Failed to refresh data:', error);
        }
    }
    
    destroy() {
        // Cleanup
        if (this.autoUpdateInterval) {
            clearInterval(this.autoUpdateInterval);
        }
        
        if (this.map && typeof this.map.remove === 'function') {
            this.map.remove();
        }
        
        // Clear event listeners
        this.eventListeners.clear();
        
        // Clear data cache
        this.dataCache.clear();
        
        console.log('Exploration Dashboard destroyed');
    }
}

// CSS Styles for the dashboard
const dashboardStyles = `
    .exploration-dashboard {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        height: 100vh;
        background: var(--bg-primary);
        color: var(--text-primary);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .dashboard-header {
        background: var(--bg-secondary);
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-left h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent-color);
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.25rem;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .status-light {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse 2s infinite;
    }
    
    .view-switcher {
        display: flex;
        background: var(--bg-tertiary);
        border-radius: 8px;
        padding: 4px;
    }
    
    .view-btn {
        padding: 0.75rem 1.5rem;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .view-btn:hover {
        background: var(--bg-hover);
        color: var(--text-primary);
    }
    
    .view-btn.active {
        background: var(--accent-color);
        color: white;
    }
    
    .dashboard-main {
        flex: 1;
        display: flex;
        overflow: hidden;
    }
    
    .sidebar {
        width: 320px;
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    
    .sidebar-section {
        padding: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .sidebar-section h3 {
        margin: 0 0 1rem 0;
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .content-area {
        flex: 1;
        position: relative;
        overflow: hidden;
    }
    
    .view-panel {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        overflow: auto;
    }
    
    .view-panel.active {
        opacity: 1;
        visibility: visible;
    }
    
    .map-container {
        position: relative;
        height: 100%;
    }
    
    #explorationMap {
        width: 100%;
        height: 100%;
    }
    
    .map-controls {
        position: absolute;
        top: 1rem;
        right: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        z-index: 1000;
    }
    
    .control-group {
        display: flex;
        gap: 0.25rem;
        background: rgba(255,255,255,0.9);
        padding: 0.25rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    .targets-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 1.5rem;
        padding: 2rem;
    }
    
    .target-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .target-card:hover {
        border-color: var(--accent-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .target-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .target-header h4 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .target-type-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .target-type-badge.gold { background: #fbbf24; color: #92400e; }
    .target-type-badge.copper { background: #f97316; color: #9a3412; }
    .target-type-badge.lithium { background: #06b6d4; color: #155e75; }
    .target-type-badge.rare_earth { background: #8b5cf6; color: #5b21b6; }
    .target-type-badge.base_metals { background: #6b7280; color: #374151; }
    
    .target-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .metric-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .metric-value.risk-low { color: #22c55e; }
    .metric-value.risk-medium { color: #f59e0b; }
    .metric-value.risk-high { color: #ef4444; }
    
    .analysis-charts {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        padding: 2rem;
    }
    
    .chart-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        height: 400px;
        position: relative;
    }
    
    .economics-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        padding: 2rem;
    }
    
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-card h4 {
        margin: 0 0 1rem 0;
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-color);
        margin-bottom: 0.5rem;
    }
    
    .progress-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    
    .progress-content {
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        min-width: 300px;
    }
    
    .progress-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--accent-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem auto;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: var(--border-color);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--accent-color);
        transition: width 0.3s;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Dark theme variables */
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --bg-hover: rgba(255,255,255,0.05);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --accent-color: #3b82f6;
        --border-color: #334155;
    }
    
    /* Light theme */
    [data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #e2e8f0;
        --bg-hover: rgba(0,0,0,0.05);
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --accent-color: #3b82f6;
        --border-color: #e2e8f0;
    }
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = dashboardStyles;
document.head.appendChild(styleSheet);

// Global dashboard instance
let dashboard = null;

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const dashboardContainer = document.getElementById('explorationDashboardContainer');
    if (dashboardContainer) {
        dashboard = new ExplorationDashboard('explorationDashboardContainer', {
            apiBaseUrl: '/api/exploration',
            theme: 'dark',
            autoUpdate: true
        });
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExplorationDashboard;
}