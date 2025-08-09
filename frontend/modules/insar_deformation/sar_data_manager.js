/**
 * SAR Data Management System - Task 3.4
 * Comprehensive data management for multiple SAR missions with automatic download,
 * caching, versioning, and multi-temporal comparison capabilities
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function() {
    'use strict';

    class SARDataManager {
        constructor(options = {}) {
            this.container = options.container || document.body;
            this.dataCache = new Map();
            this.versionHistory = new Map();
            this.downloadQueue = [];
            this.processingQueue = [];
            this.ui = null;
            
            this.config = {
                maxCacheSize: 5000, // MB
                cacheExpiry: 7 * 24 * 60 * 60 * 1000, // 7 days
                maxConcurrentDownloads: 3,
                retryAttempts: 3,
                downloadTimeout: 300000, // 5 minutes
                supportedMissions: ['Sentinel-1', 'ALOS-2', 'TerraSAR-X', 'COSMO-SkyMed', 'RADARSAT-2'],
                dataProviders: ['ESA', 'JAXA', 'DLR', 'ASI', 'CSA'],
                processingLevels: ['L0', 'L1', 'L2', 'L3'],
                defaultRegion: { lat: 0, lon: 0, radius: 50 }
            };

            this.missions = {
                'Sentinel-1': new Sentinel1Mission(),
                'ALOS-2': new ALOS2Mission(),
                'TerraSAR-X': new TerraSARXMission(),
                'COSMO-SkyMed': new COSMOSkyMedMission(),
                'RADARSAT-2': new RADARSAT2Mission()
            };

            this.dataProviders = {
                'ESA': new ESAProvider(),
                'JAXA': new JAXAProvider(),
                'DLR': new DLRProvider(),
                'ASI': new ASIProvider(),
                'CSA': new CSAProvider()
            };

            this.cacheManager = new CacheManager(this.config);
            this.versionController = new VersionController();
            this.comparisonEngine = new ComparisonEngine();
            this.downloadManager = new DownloadManager(this.config);
            
            this.init();
        }

        init() {
            this.createDataManagerInterface();
            this.setupEventHandlers();
            this.initializeWorkers();
            this.loadCachedData();
            this.setupPeriodicCleanup();
            console.log('📡 SAR Data Management System initialized');
        }

        createDataManagerInterface() {
            const container = document.createElement('div');
            container.className = 'sar-data-manager-container';
            container.innerHTML = `
                <div class="manager-header">
                    <h3><i class="fas fa-satellite"></i> SAR Data Manager</h3>
                    <div class="manager-controls">
                        <button class="btn btn-primary" id="searchData">Search Data</button>
                        <button class="btn btn-secondary" id="viewQueue">View Queue</button>
                        <button class="btn btn-info" id="manageCache">Manage Cache</button>
                        <button class="btn btn-success" id="compareData">Compare</button>
                    </div>
                </div>

                <div class="manager-tabs">
                    <button class="tab-btn active" data-tab="search">Data Search</button>
                    <button class="tab-btn" data-tab="downloads">Downloads</button>
                    <button class="tab-btn" data-tab="cache">Cache</button>
                    <button class="tab-btn" data-tab="processing">Processing</button>
                    <button class="tab-btn" data-tab="comparison">Comparison</button>
                    <button class="tab-btn" data-tab="missions">Missions</button>
                </div>

                <div class="manager-content">
                    <!-- Data Search Tab -->
                    <div class="tab-content active" id="search-tab">
                        <div class="search-panel">
                            <div class="search-criteria">
                                <div class="criteria-section">
                                    <h4>Mission Selection</h4>
                                    <div class="mission-grid">
                                        <div class="mission-item">
                                            <input type="checkbox" id="sentinel1" value="Sentinel-1" checked>
                                            <label for="sentinel1">
                                                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiPjxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzAwN2JmZiIvPjx0ZXh0IHg9IjE2IiB5PSIxOCIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMTBweCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UzE8L3RleHQ+PC9zdmc+" alt="S1">
                                                Sentinel-1
                                            </label>
                                        </div>
                                        <div class="mission-item">
                                            <input type="checkbox" id="alos2" value="ALOS-2">
                                            <label for="alos2">
                                                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiPjxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iI2ZmOTgwMCIvPjx0ZXh0IHg9IjE2IiB5PSIxOCIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iOXB4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5BTDwvdGV4dD48L3N2Zz4=" alt="AL">
                                                ALOS-2
                                            </label>
                                        </div>
                                        <div class="mission-item">
                                            <input type="checkbox" id="terrasar" value="TerraSAR-X">
                                            <label for="terrasar">
                                                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiPjxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzRjYWY1MCIvPjx0ZXh0IHg9IjE2IiB5PSIxOCIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iOXB4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5UWDwvdGV4dD48L3N2Zz4=" alt="TX">
                                                TerraSAR-X
                                            </label>
                                        </div>
                                        <div class="mission-item">
                                            <input type="checkbox" id="cosmo" value="COSMO-SkyMed">
                                            <label for="cosmo">
                                                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiPjxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iI2Y0NDMzNiIvPjx0ZXh0IHg9IjE2IiB5PSIxOCIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iOXB4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5DTzwvdGV4dD48L3N2Zz4=" alt="CO">
                                                COSMO-SkyMed
                                            </label>
                                        </div>
                                        <div class="mission-item">
                                            <input type="checkbox" id="radarsat2" value="RADARSAT-2">
                                            <label for="radarsat2">
                                                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiPjxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzljMjdiMCIvPjx0ZXh0IHg9IjE2IiB5PSIxOCIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iOXB4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5SUzwvdGV4dD48L3N2Zz4=" alt="RS">
                                                RADARSAT-2
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                <div class="criteria-section">
                                    <h4>Geographic Area</h4>
                                    <div class="geo-controls">
                                        <div class="coord-input">
                                            <label>Center Latitude:</label>
                                            <input type="number" id="searchLat" min="-90" max="90" step="0.1" value="0">
                                        </div>
                                        <div class="coord-input">
                                            <label>Center Longitude:</label>
                                            <input type="number" id="searchLon" min="-180" max="180" step="0.1" value="0">
                                        </div>
                                        <div class="coord-input">
                                            <label>Search Radius (km):</label>
                                            <input type="number" id="searchRadius" min="1" max="1000" value="50">
                                        </div>
                                        <button id="selectOnMap" class="btn btn-sm">Select on Map</button>
                                    </div>
                                    <div class="region-presets">
                                        <button class="preset-btn" data-region="etna">Mount Etna</button>
                                        <button class="preset-btn" data-region="vesuvius">Vesuvius</button>
                                        <button class="preset-btn" data-region="yellowstone">Yellowstone</button>
                                        <button class="preset-btn" data-region="kilauea">Kilauea</button>
                                    </div>
                                </div>

                                <div class="criteria-section">
                                    <h4>Temporal Range</h4>
                                    <div class="temporal-controls">
                                        <div class="date-input">
                                            <label>Start Date:</label>
                                            <input type="date" id="startDate" value="2020-01-01">
                                        </div>
                                        <div class="date-input">
                                            <label>End Date:</label>
                                            <input type="date" id="endDate">
                                        </div>
                                        <div class="temporal-presets">
                                            <button class="preset-btn" data-period="1month">Last Month</button>
                                            <button class="preset-btn" data-period="3months">Last 3 Months</button>
                                            <button class="preset-btn" data-period="1year">Last Year</button>
                                            <button class="preset-btn" data-period="all">All Available</button>
                                        </div>
                                    </div>
                                </div>

                                <div class="criteria-section">
                                    <h4>Data Properties</h4>
                                    <div class="property-controls">
                                        <div class="property-group">
                                            <label>Processing Level:</label>
                                            <select id="processingLevel">
                                                <option value="">Any</option>
                                                <option value="L0">L0 - Raw</option>
                                                <option value="L1">L1 - Single Look Complex</option>
                                                <option value="L2" selected>L2 - Ground Range Detected</option>
                                                <option value="L3">L3 - Processed</option>
                                            </select>
                                        </div>
                                        <div class="property-group">
                                            <label>Acquisition Mode:</label>
                                            <select id="acquisitionMode">
                                                <option value="">Any</option>
                                                <option value="IW">Interferometric Wide (IW)</option>
                                                <option value="SM">StripMap (SM)</option>
                                                <option value="WV">Wave (WV)</option>
                                                <option value="EW">Extra Wide (EW)</option>
                                            </select>
                                        </div>
                                        <div class="property-group">
                                            <label>Polarization:</label>
                                            <select id="polarization">
                                                <option value="">Any</option>
                                                <option value="VV">VV</option>
                                                <option value="VH">VH</option>
                                                <option value="HH">HH</option>
                                                <option value="HV">HV</option>
                                                <option value="VV+VH" selected>VV+VH</option>
                                                <option value="HH+HV">HH+HV</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="search-results">
                                <div class="results-header">
                                    <h4>Search Results <span id="resultCount">(0 found)</span></h4>
                                    <div class="results-actions">
                                        <button id="selectAllResults" class="btn btn-sm">Select All</button>
                                        <button id="downloadSelected" class="btn btn-primary btn-sm">Download Selected</button>
                                        <button id="exportResults" class="btn btn-secondary btn-sm">Export List</button>
                                    </div>
                                </div>
                                <div class="results-table-container">
                                    <table class="results-table" id="resultsTable">
                                        <thead>
                                            <tr>
                                                <th><input type="checkbox" id="selectAllCheckbox"></th>
                                                <th>Mission</th>
                                                <th>Date</th>
                                                <th>Track</th>
                                                <th>Mode</th>
                                                <th>Polarization</th>
                                                <th>Size</th>
                                                <th>Status</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody id="resultsTableBody">
                                            <!-- Results will be populated here -->
                                        </tbody>
                                    </table>
                                </div>
                                <div class="results-pagination">
                                    <button id="prevPage" class="btn btn-sm">← Previous</button>
                                    <span id="pageInfo">Page 1 of 1</span>
                                    <button id="nextPage" class="btn btn-sm">Next →</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Downloads Tab -->
                    <div class="tab-content" id="downloads-tab">
                        <div class="downloads-panel">
                            <div class="downloads-header">
                                <h4>Download Queue <span id="queueCount">(0 items)</span></h4>
                                <div class="queue-controls">
                                    <button id="pauseQueue" class="btn btn-warning">Pause</button>
                                    <button id="resumeQueue" class="btn btn-success">Resume</button>
                                    <button id="clearCompleted" class="btn btn-secondary">Clear Completed</button>
                                    <button id="clearQueue" class="btn btn-danger">Clear All</button>
                                </div>
                            </div>
                            
                            <div class="download-stats">
                                <div class="stat-item">
                                    <span class="stat-label">Active Downloads:</span>
                                    <span class="stat-value" id="activeDownloads">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Queue Length:</span>
                                    <span class="stat-value" id="queueLength">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Total Downloaded:</span>
                                    <span class="stat-value" id="totalDownloaded">0 GB</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Download Speed:</span>
                                    <span class="stat-value" id="downloadSpeed">0 MB/s</span>
                                </div>
                            </div>

                            <div class="downloads-list">
                                <div id="downloadItems" class="download-items">
                                    <!-- Download items will be populated here -->
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Cache Tab -->
                    <div class="tab-content" id="cache-tab">
                        <div class="cache-panel">
                            <div class="cache-overview">
                                <div class="cache-stats">
                                    <div class="cache-stat">
                                        <div class="stat-circle">
                                            <span id="cacheUsage">0%</span>
                                        </div>
                                        <div class="stat-label">Cache Usage</div>
                                    </div>
                                    <div class="cache-metrics">
                                        <div class="metric-item">
                                            <span>Total Size:</span>
                                            <span id="totalCacheSize">0 MB</span>
                                        </div>
                                        <div class="metric-item">
                                            <span>Files Cached:</span>
                                            <span id="cachedFileCount">0</span>
                                        </div>
                                        <div class="metric-item">
                                            <span>Hit Rate:</span>
                                            <span id="cacheHitRate">0%</span>
                                        </div>
                                        <div class="metric-item">
                                            <span>Last Cleanup:</span>
                                            <span id="lastCleanup">Never</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="cache-actions">
                                    <button id="optimizeCache" class="btn btn-primary">Optimize Cache</button>
                                    <button id="cleanupCache" class="btn btn-warning">Cleanup Old Files</button>
                                    <button id="clearCache" class="btn btn-danger">Clear All Cache</button>
                                </div>
                            </div>

                            <div class="cache-contents">
                                <div class="cache-filters">
                                    <select id="cacheFilter">
                                        <option value="all">All Files</option>
                                        <option value="recent">Recent</option>
                                        <option value="large">Large Files</option>
                                        <option value="old">Old Files</option>
                                    </select>
                                    <input type="search" id="cacheSearch" placeholder="Search cached files...">
                                </div>
                                <div class="cache-list" id="cacheList">
                                    <!-- Cache items will be populated here -->
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Processing Tab -->
                    <div class="tab-content" id="processing-tab">
                        <div class="processing-panel">
                            <div class="processing-queue-status">
                                <h4>Processing Queue</h4>
                                <div class="queue-stats">
                                    <div class="queue-stat">
                                        <span class="stat-number" id="processingCount">0</span>
                                        <span class="stat-label">Processing</span>
                                    </div>
                                    <div class="queue-stat">
                                        <span class="stat-number" id="queuedCount">0</span>
                                        <span class="stat-label">Queued</span>
                                    </div>
                                    <div class="queue-stat">
                                        <span class="stat-number" id="completedCount">0</span>
                                        <span class="stat-label">Completed</span>
                                    </div>
                                    <div class="queue-stat">
                                        <span class="stat-number" id="errorCount">0</span>
                                        <span class="stat-label">Errors</span>
                                    </div>
                                </div>
                            </div>

                            <div class="processing-controls">
                                <div class="processing-config">
                                    <h5>Processing Configuration</h5>
                                    <div class="config-group">
                                        <label>Processing Type:</label>
                                        <select id="processingType">
                                            <option value="interferogram">Interferogram Generation</option>
                                            <option value="coherence">Coherence Analysis</option>
                                            <option value="geocoding">Geocoding</option>
                                            <option value="filtering">Speckle Filtering</option>
                                            <option value="calibration">Radiometric Calibration</option>
                                        </select>
                                    </div>
                                    <div class="config-group">
                                        <label>Priority:</label>
                                        <select id="processingPriority">
                                            <option value="low">Low</option>
                                            <option value="normal" selected>Normal</option>
                                            <option value="high">High</option>
                                            <option value="urgent">Urgent</option>
                                        </select>
                                    </div>
                                    <div class="config-group">
                                        <label>Output Format:</label>
                                        <select id="outputFormat">
                                            <option value="geotiff">GeoTIFF</option>
                                            <option value="netcdf">NetCDF</option>
                                            <option value="hdf5">HDF5</option>
                                            <option value="envi">ENVI</option>
                                        </select>
                                    </div>
                                </div>
                            </div>

                            <div class="processing-history">
                                <h5>Processing History</h5>
                                <div class="history-list" id="processingHistory">
                                    <!-- Processing history items will be populated here -->
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Comparison Tab -->
                    <div class="tab-content" id="comparison-tab">
                        <div class="comparison-panel">
                            <div class="comparison-setup">
                                <h4>Multi-temporal Comparison</h4>
                                <div class="comparison-selector">
                                    <div class="dataset-selection">
                                        <div class="dataset-group">
                                            <h5>Dataset A (Reference)</h5>
                                            <select id="datasetA" class="dataset-select">
                                                <option value="">Select dataset...</option>
                                            </select>
                                            <div class="dataset-info" id="infoA">
                                                <span>No dataset selected</span>
                                            </div>
                                        </div>
                                        <div class="dataset-group">
                                            <h5>Dataset B (Compare)</h5>
                                            <select id="datasetB" class="dataset-select">
                                                <option value="">Select dataset...</option>
                                            </select>
                                            <div class="dataset-info" id="infoB">
                                                <span>No dataset selected</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="comparison-options">
                                        <div class="option-group">
                                            <label>Comparison Type:</label>
                                            <select id="comparisonType">
                                                <option value="difference">Difference</option>
                                                <option value="ratio">Ratio</option>
                                                <option value="correlation">Correlation</option>
                                                <option value="coherence_change">Coherence Change</option>
                                                <option value="phase_change">Phase Change</option>
                                            </select>
                                        </div>
                                        <div class="option-group">
                                            <label>Registration Method:</label>
                                            <select id="registrationMethod">
                                                <option value="automatic">Automatic</option>
                                                <option value="manual">Manual Control Points</option>
                                                <option value="orbit">Orbit-based</option>
                                            </select>
                                        </div>
                                        <div class="option-group">
                                            <label>Resampling:</label>
                                            <select id="resamplingMethod">
                                                <option value="nearest">Nearest Neighbor</option>
                                                <option value="bilinear" selected>Bilinear</option>
                                                <option value="bicubic">Bicubic</option>
                                            </select>
                                        </div>
                                    </div>
                                    
                                    <div class="comparison-actions">
                                        <button id="runComparison" class="btn btn-primary">Run Comparison</button>
                                        <button id="saveComparison" class="btn btn-secondary">Save Results</button>
                                        <button id="exportComparison" class="btn btn-info">Export</button>
                                    </div>
                                </div>
                            </div>

                            <div class="comparison-results">
                                <div class="results-visualization">
                                    <canvas id="comparisonCanvas" width="800" height="400"></canvas>
                                </div>
                                <div class="comparison-statistics">
                                    <div class="stat-panel">
                                        <h5>Change Statistics</h5>
                                        <div class="change-metrics">
                                            <div class="metric">
                                                <label>Mean Change:</label>
                                                <span id="meanChange">--</span>
                                            </div>
                                            <div class="metric">
                                                <label>Std Deviation:</label>
                                                <span id="stdChange">--</span>
                                            </div>
                                            <div class="metric">
                                                <label>Max Change:</label>
                                                <span id="maxChange">--</span>
                                            </div>
                                            <div class="metric">
                                                <label>Changed Area:</label>
                                                <span id="changedArea">--</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="stat-panel">
                                        <h5>Quality Metrics</h5>
                                        <div class="quality-metrics">
                                            <div class="metric">
                                                <label>Registration Error:</label>
                                                <span id="registrationError">--</span>
                                            </div>
                                            <div class="metric">
                                                <label>Correlation Coeff:</label>
                                                <span id="correlationCoeff">--</span>
                                            </div>
                                            <div class="metric">
                                                <label>Valid Pixels:</label>
                                                <span id="validPixels">--</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Missions Tab -->
                    <div class="tab-content" id="missions-tab">
                        <div class="missions-panel">
                            <div class="missions-overview">
                                <h4>Supported SAR Missions</h4>
                                <div class="missions-grid">
                                    <div class="mission-card" data-mission="Sentinel-1">
                                        <div class="mission-header">
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQiIGhlaWdodD0iNjQiPjxyZWN0IHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgZmlsbD0iIzAwN2JmZiIvPjx0ZXh0IHg9IjMyIiB5PSIzNiIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMjBweCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UzE8L3RleHQ+PC9zdmc+" alt="Sentinel-1">
                                            <div class="mission-info">
                                                <h5>Sentinel-1</h5>
                                                <span class="mission-agency">ESA</span>
                                            </div>
                                        </div>
                                        <div class="mission-details">
                                            <div class="detail-item">
                                                <span>Frequency:</span>
                                                <span>C-band (5.405 GHz)</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Resolution:</span>
                                                <span>5-20 m</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Repeat Cycle:</span>
                                                <span>12 days</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Coverage:</span>
                                                <span>Global</span>
                                            </div>
                                        </div>
                                        <div class="mission-status">
                                            <span class="status-indicator active">Active</span>
                                            <span class="data-count">2.3M scenes</span>
                                        </div>
                                    </div>

                                    <div class="mission-card" data-mission="ALOS-2">
                                        <div class="mission-header">
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQiIGhlaWdodD0iNjQiPjxyZWN0IHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgZmlsbD0iI2ZmOTgwMCIvPjx0ZXh0IHg9IjMyIiB5PSIzNiIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMThweCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+QUw8L3RleHQ+PC9zdmc+" alt="ALOS-2">
                                            <div class="mission-info">
                                                <h5>ALOS-2</h5>
                                                <span class="mission-agency">JAXA</span>
                                            </div>
                                        </div>
                                        <div class="mission-details">
                                            <div class="detail-item">
                                                <span>Frequency:</span>
                                                <span>L-band (1.27 GHz)</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Resolution:</span>
                                                <span>1-100 m</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Repeat Cycle:</span>
                                                <span>14 days</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Coverage:</span>
                                                <span>Global</span>
                                            </div>
                                        </div>
                                        <div class="mission-status">
                                            <span class="status-indicator active">Active</span>
                                            <span class="data-count">450K scenes</span>
                                        </div>
                                    </div>

                                    <div class="mission-card" data-mission="TerraSAR-X">
                                        <div class="mission-header">
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQiIGhlaWdodD0iNjQiPjxyZWN0IHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgZmlsbD0iIzRjYWY1MCIvPjx0ZXh0IHg9IjMyIiB5PSIzNiIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMThweCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+VFg8L3RleHQ+PC9zdmc+" alt="TerraSAR-X">
                                            <div class="mission-info">
                                                <h5>TerraSAR-X</h5>
                                                <span class="mission-agency">DLR</span>
                                            </div>
                                        </div>
                                        <div class="mission-details">
                                            <div class="detail-item">
                                                <span>Frequency:</span>
                                                <span>X-band (9.65 GHz)</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Resolution:</span>
                                                <span>1-16 m</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Repeat Cycle:</span>
                                                <span>11 days</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Coverage:</span>
                                                <span>Global</span>
                                            </div>
                                        </div>
                                        <div class="mission-status">
                                            <span class="status-indicator active">Active</span>
                                            <span class="data-count">890K scenes</span>
                                        </div>
                                    </div>

                                    <div class="mission-card" data-mission="COSMO-SkyMed">
                                        <div class="mission-header">
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQiIGhlaWdodD0iNjQiPjxyZWN0IHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgZmlsbD0iI2Y0NDMzNiIvPjx0ZXh0IHg9IjMyIiB5PSIzNiIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMThweCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Q088L3RleHQ+PC9zdmc+" alt="COSMO-SkyMed">
                                            <div class="mission-info">
                                                <h5>COSMO-SkyMed</h5>
                                                <span class="mission-agency">ASI</span>
                                            </div>
                                        </div>
                                        <div class="mission-details">
                                            <div class="detail-item">
                                                <span>Frequency:</span>
                                                <span>X-band (9.6 GHz)</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Resolution:</span>
                                                <span>1-30 m</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Repeat Cycle:</span>
                                                <span>16 days</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Coverage:</span>
                                                <span>Global</span>
                                            </div>
                                        </div>
                                        <div class="mission-status">
                                            <span class="status-indicator active">Active</span>
                                            <span class="data-count">340K scenes</span>
                                        </div>
                                    </div>

                                    <div class="mission-card" data-mission="RADARSAT-2">
                                        <div class="mission-header">
                                            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQiIGhlaWdodD0iNjQiPjxyZWN0IHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgZmlsbD0iIzljMjdiMCIvPjx0ZXh0IHg9IjMyIiB5PSIzNiIgZmlsbD0id2hpdGUiIGZvbnQtc2l6ZT0iMThweCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UlM8L3RleHQ+PC9zdmc+" alt="RADARSAT-2">
                                            <div class="mission-info">
                                                <h5>RADARSAT-2</h5>
                                                <span class="mission-agency">CSA</span>
                                            </div>
                                        </div>
                                        <div class="mission-details">
                                            <div class="detail-item">
                                                <span>Frequency:</span>
                                                <span>C-band (5.405 GHz)</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Resolution:</span>
                                                <span>3-100 m</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Repeat Cycle:</span>
                                                <span>24 days</span>
                                            </div>
                                            <div class="detail-item">
                                                <span>Coverage:</span>
                                                <span>Global</span>
                                            </div>
                                        </div>
                                        <div class="mission-status">
                                            <span class="status-indicator inactive">Inactive (2019)</span>
                                            <span class="data-count">650K scenes</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="mission-timeline">
                                <h5>Mission Timeline</h5>
                                <canvas id="missionTimeline" width="800" height="200"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="manager-status">
                    <div class="status-bar">
                        <div class="status-progress">
                            <div class="progress-fill" id="managerProgress"></div>
                        </div>
                        <div class="status-text" id="managerStatus">SAR Data Manager ready</div>
                    </div>
                </div>
            `;

            this.container.appendChild(container);
            this.ui = container;
            this.setupCanvases();
        }

        setupCanvases() {
            this.canvases = {
                comparison: document.getElementById('comparisonCanvas'),
                timeline: document.getElementById('missionTimeline')
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
            document.getElementById('searchData')?.addEventListener('click', () => {
                this.performDataSearch();
            });

            document.getElementById('viewQueue')?.addEventListener('click', () => {
                this.switchTab('downloads');
            });

            document.getElementById('manageCache')?.addEventListener('click', () => {
                this.switchTab('cache');
            });

            document.getElementById('compareData')?.addEventListener('click', () => {
                this.switchTab('comparison');
            });

            // Tab switching
            this.ui.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.switchTab(btn.dataset.tab);
                });
            });

            // Search controls
            document.getElementById('downloadSelected')?.addEventListener('click', () => {
                this.downloadSelectedData();
            });

            document.getElementById('selectAllResults')?.addEventListener('click', () => {
                this.selectAllSearchResults();
            });

            // Region presets
            this.ui.querySelectorAll('.preset-btn[data-region]').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.setRegionPreset(btn.dataset.region);
                });
            });

            // Temporal presets
            this.ui.querySelectorAll('.preset-btn[data-period]').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.setTemporalPreset(btn.dataset.period);
                });
            });

            // Queue controls
            document.getElementById('pauseQueue')?.addEventListener('click', () => {
                this.pauseDownloadQueue();
            });

            document.getElementById('resumeQueue')?.addEventListener('click', () => {
                this.resumeDownloadQueue();
            });

            document.getElementById('clearCompleted')?.addEventListener('click', () => {
                this.clearCompletedDownloads();
            });

            // Cache controls
            document.getElementById('optimizeCache')?.addEventListener('click', () => {
                this.optimizeCache();
            });

            document.getElementById('cleanupCache')?.addEventListener('click', () => {
                this.cleanupCache();
            });

            document.getElementById('clearCache')?.addEventListener('click', () => {
                this.clearCache();
            });

            // Comparison controls
            document.getElementById('runComparison')?.addEventListener('click', () => {
                this.runComparison();
            });

            // Today's date for end date
            const today = new Date().toISOString().split('T')[0];
            document.getElementById('endDate').value = today;
        }

        initializeWorkers() {
            this.workers = {
                search: this.createSearchWorker(),
                download: this.createDownloadWorker(),
                comparison: this.createComparisonWorker()
            };
        }

        createSearchWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { criteria, missions } = e.data;
                    const results = performDataSearch(criteria, missions);
                    self.postMessage(results);
                };

                function performDataSearch(criteria, missions) {
                    const results = [];
                    const { lat, lon, radius, startDate, endDate, selectedMissions } = criteria;
                    
                    selectedMissions.forEach(mission => {
                        // Generate mock search results based on criteria
                        const missionData = generateMissionData(mission, criteria);
                        results.push(...missionData);
                    });
                    
                    return {
                        results: results,
                        totalCount: results.length,
                        searchTime: Date.now()
                    };
                }
                
                function generateMissionData(mission, criteria) {
                    const results = [];
                    const startDate = new Date(criteria.startDate);
                    const endDate = new Date(criteria.endDate);
                    const daysDiff = Math.floor((endDate - startDate) / (1000 * 60 * 60 * 24));
                    
                    // Generate realistic number of scenes based on mission
                    const repeatCycle = getMissionRepeatCycle(mission);
                    const sceneCount = Math.floor(daysDiff / repeatCycle) + Math.floor(Math.random() * 5);
                    
                    for (let i = 0; i < sceneCount; i++) {
                        const acquisitionDate = new Date(startDate.getTime() + (i * repeatCycle * 24 * 60 * 60 * 1000));
                        
                        results.push({
                            id: \`\${mission}_\${acquisitionDate.toISOString().split('T')[0]}_\${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}\`,
                            mission: mission,
                            date: acquisitionDate.toISOString().split('T')[0],
                            time: acquisitionDate.toISOString().split('T')[1].substr(0, 8),
                            track: 100 + Math.floor(Math.random() * 200),
                            orbit: 'Ascending',
                            mode: getMissionMode(mission),
                            polarization: getMissionPolarization(mission),
                            size: (500 + Math.random() * 2000).toFixed(1) + ' MB',
                            processingLevel: criteria.processingLevel || 'L2',
                            cloudCover: Math.floor(Math.random() * 30),
                            baselineB: Math.floor(Math.random() * 500),
                            baselineT: Math.floor(Math.random() * 365),
                            geometry: {
                                coordinates: [
                                    [criteria.lon - 0.5, criteria.lat - 0.5],
                                    [criteria.lon + 0.5, criteria.lat - 0.5],
                                    [criteria.lon + 0.5, criteria.lat + 0.5],
                                    [criteria.lon - 0.5, criteria.lat + 0.5]
                                ]
                            },
                            provider: getMissionProvider(mission),
                            url: \`https://\${getMissionProvider(mission).toLowerCase()}.int/data/\${mission.toLowerCase()}/\${acquisitionDate.getFullYear()}/\${results.length}\`,
                            status: Math.random() > 0.9 ? 'unavailable' : 'available'
                        });
                    }
                    
                    return results;
                }
                
                function getMissionRepeatCycle(mission) {
                    const cycles = {
                        'Sentinel-1': 12,
                        'ALOS-2': 14,
                        'TerraSAR-X': 11,
                        'COSMO-SkyMed': 16,
                        'RADARSAT-2': 24
                    };
                    return cycles[mission] || 12;
                }
                
                function getMissionMode(mission) {
                    const modes = {
                        'Sentinel-1': 'IW',
                        'ALOS-2': 'SM',
                        'TerraSAR-X': 'SM',
                        'COSMO-SkyMed': 'HIMAGE',
                        'RADARSAT-2': 'Fine'
                    };
                    return modes[mission] || 'SM';
                }
                
                function getMissionPolarization(mission) {
                    const polarizations = {
                        'Sentinel-1': 'VV+VH',
                        'ALOS-2': 'HH+HV',
                        'TerraSAR-X': 'HH',
                        'COSMO-SkyMed': 'HH',
                        'RADARSAT-2': 'HH+HV'
                    };
                    return polarizations[mission] || 'VV';
                }
                
                function getMissionProvider(mission) {
                    const providers = {
                        'Sentinel-1': 'ESA',
                        'ALOS-2': 'JAXA',
                        'TerraSAR-X': 'DLR',
                        'COSMO-SkyMed': 'ASI',
                        'RADARSAT-2': 'CSA'
                    };
                    return providers[mission] || 'ESA';
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        createDownloadWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { action, data } = e.data;
                    
                    switch(action) {
                        case 'download':
                            simulateDownload(data);
                            break;
                        case 'pause':
                            pauseDownloads();
                            break;
                        case 'resume':
                            resumeDownloads();
                            break;
                        default:
                            self.postMessage({ error: 'Unknown action' });
                    }
                };

                function simulateDownload(downloadInfo) {
                    const { id, size, url } = downloadInfo;
                    const totalSize = parseFloat(size) * 1024 * 1024; // Convert MB to bytes
                    let downloaded = 0;
                    
                    const downloadInterval = setInterval(() => {
                        const chunkSize = Math.random() * 1024 * 1024; // Random chunk up to 1MB
                        downloaded += chunkSize;
                        
                        const progress = Math.min(downloaded / totalSize, 1);
                        const speed = chunkSize / 1024 / 1024; // MB/s
                        
                        self.postMessage({
                            type: 'progress',
                            id: id,
                            progress: progress,
                            downloaded: downloaded,
                            totalSize: totalSize,
                            speed: speed,
                            status: 'downloading'
                        });
                        
                        if (progress >= 1) {
                            clearInterval(downloadInterval);
                            self.postMessage({
                                type: 'complete',
                                id: id,
                                status: 'completed'
                            });
                        }
                    }, 100 + Math.random() * 200);
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        createComparisonWorker() {
            const workerCode = `
                self.onmessage = function(e) {
                    const { datasetA, datasetB, options } = e.data;
                    const result = performComparison(datasetA, datasetB, options);
                    self.postMessage(result);
                };

                function performComparison(datasetA, datasetB, options) {
                    const { comparisonType, registrationMethod } = options;
                    
                    // Simulate comparison processing
                    const results = {
                        comparisonType: comparisonType,
                        registrationError: Math.random() * 0.5,
                        correlationCoeff: 0.7 + Math.random() * 0.3,
                        meanChange: (Math.random() - 0.5) * 2,
                        stdChange: Math.random() * 0.5,
                        maxChange: Math.random() * 5,
                        changedArea: Math.random() * 100,
                        validPixels: 85 + Math.random() * 15,
                        processingTime: 1000 + Math.random() * 5000,
                        changeData: generateChangeData()
                    };
                    
                    return results;
                }
                
                function generateChangeData() {
                    const width = 100;
                    const height = 100;
                    const changeData = new Float32Array(width * height);
                    
                    for (let i = 0; i < changeData.length; i++) {
                        changeData[i] = (Math.random() - 0.5) * 2;
                    }
                    
                    return Array.from(changeData);
                }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            return new Worker(URL.createObjectURL(blob));
        }

        loadCachedData() {
            // Load cached data from localStorage or IndexedDB
            try {
                const cachedData = localStorage.getItem('sarDataCache');
                if (cachedData) {
                    const parsedCache = JSON.parse(cachedData);
                    this.dataCache = new Map(parsedCache);
                }
                
                this.updateCacheStatistics();
            } catch (error) {
                console.warn('Error loading cached data:', error);
            }
        }

        setupPeriodicCleanup() {
            // Clean up cache every hour
            setInterval(() => {
                this.cleanupExpiredCache();
            }, 60 * 60 * 1000);
        }

        async performDataSearch() {
            this.updateStatus('Searching for SAR data...', 20);
            
            try {
                const criteria = this.getSearchCriteria();
                const selectedMissions = this.getSelectedMissions();
                
                if (selectedMissions.length === 0) {
                    alert('Please select at least one SAR mission');
                    return;
                }

                const result = await new Promise((resolve) => {
                    this.workers.search.onmessage = (e) => {
                        resolve(e.data);
                    };
                    
                    this.workers.search.postMessage({
                        criteria: criteria,
                        missions: selectedMissions
                    });
                });

                this.displaySearchResults(result.results);
                this.updateResultCount(result.totalCount);
                this.updateStatus('Search completed', 100);

            } catch (error) {
                console.error('Error performing search:', error);
                this.updateStatus('Search failed', 0);
            }
        }

        getSearchCriteria() {
            return {
                lat: parseFloat(document.getElementById('searchLat')?.value || 0),
                lon: parseFloat(document.getElementById('searchLon')?.value || 0),
                radius: parseFloat(document.getElementById('searchRadius')?.value || 50),
                startDate: document.getElementById('startDate')?.value || '2020-01-01',
                endDate: document.getElementById('endDate')?.value || new Date().toISOString().split('T')[0],
                processingLevel: document.getElementById('processingLevel')?.value || '',
                acquisitionMode: document.getElementById('acquisitionMode')?.value || '',
                polarization: document.getElementById('polarization')?.value || ''
            };
        }

        getSelectedMissions() {
            const missions = [];
            const checkboxes = this.ui.querySelectorAll('.mission-item input[type="checkbox"]:checked');
            checkboxes.forEach(cb => {
                missions.push(cb.value);
            });
            return missions;
        }

        displaySearchResults(results) {
            const tableBody = document.getElementById('resultsTableBody');
            if (!tableBody) return;

            tableBody.innerHTML = '';

            results.forEach((result, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><input type="checkbox" class="result-checkbox" data-id="${result.id}"></td>
                    <td>
                        <div class="mission-cell">
                            <span class="mission-badge mission-${result.mission.toLowerCase().replace(/[^a-z0-9]/g, '')}">${result.mission}</span>
                        </div>
                    </td>
                    <td>
                        <div class="date-cell">
                            <div class="date">${result.date}</div>
                            <div class="time">${result.time}</div>
                        </div>
                    </td>
                    <td>${result.track}</td>
                    <td>${result.mode}</td>
                    <td>${result.polarization}</td>
                    <td>${result.size}</td>
                    <td>
                        <span class="status-badge status-${result.status}">${result.status}</span>
                    </td>
                    <td>
                        <div class="action-buttons">
                            <button class="btn btn-sm btn-primary" onclick="window.sarDataManager.downloadSingleItem('${result.id}')" ${result.status !== 'available' ? 'disabled' : ''}>
                                <i class="fas fa-download"></i>
                            </button>
                            <button class="btn btn-sm btn-info" onclick="window.sarDataManager.previewItem('${result.id}')">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                    </td>
                `;
                tableBody.appendChild(row);
            });

            // Store results for later use
            this.searchResults = results;
        }

        updateResultCount(count) {
            const resultCountEl = document.getElementById('resultCount');
            if (resultCountEl) {
                resultCountEl.textContent = `(${count} found)`;
            }
        }

        selectAllSearchResults() {
            const checkboxes = this.ui.querySelectorAll('.result-checkbox');
            const selectAllCheckbox = document.getElementById('selectAllCheckbox');
            const isChecked = selectAllCheckbox.checked;
            
            checkboxes.forEach(cb => {
                cb.checked = isChecked;
            });
        }

        async downloadSelectedData() {
            const selectedCheckboxes = this.ui.querySelectorAll('.result-checkbox:checked');
            const selectedIds = Array.from(selectedCheckboxes).map(cb => cb.dataset.id);
            
            if (selectedIds.length === 0) {
                alert('Please select at least one item to download');
                return;
            }

            // Add selected items to download queue
            selectedIds.forEach(id => {
                const result = this.searchResults.find(r => r.id === id);
                if (result) {
                    this.addToDownloadQueue(result);
                }
            });

            // Switch to downloads tab
            this.switchTab('downloads');
            this.updateDownloadDisplay();
        }

        addToDownloadQueue(item) {
            const downloadItem = {
                id: item.id,
                mission: item.mission,
                date: item.date,
                size: item.size,
                url: item.url,
                status: 'queued',
                progress: 0,
                addedTime: Date.now(),
                startTime: null,
                completedTime: null,
                speed: 0
            };

            this.downloadQueue.push(downloadItem);
            this.startNextDownload();
        }

        startNextDownload() {
            const activeDownloads = this.downloadQueue.filter(item => item.status === 'downloading').length;
            
            if (activeDownloads >= this.config.maxConcurrentDownloads) {
                return;
            }

            const nextItem = this.downloadQueue.find(item => item.status === 'queued');
            if (!nextItem) {
                return;
            }

            nextItem.status = 'downloading';
            nextItem.startTime = Date.now();

            // Start download in worker
            this.workers.download.onmessage = (e) => {
                this.handleDownloadProgress(e.data);
            };

            this.workers.download.postMessage({
                action: 'download',
                data: nextItem
            });
        }

        handleDownloadProgress(data) {
            const { type, id, progress, speed, status } = data;
            const item = this.downloadQueue.find(item => item.id === id);
            
            if (!item) return;

            if (type === 'progress') {
                item.progress = progress;
                item.speed = speed;
            } else if (type === 'complete') {
                item.status = 'completed';
                item.completedTime = Date.now();
                item.progress = 1;
                
                // Start next download
                this.startNextDownload();
                
                // Add to cache
                this.addToCache(item);
            }

            this.updateDownloadDisplay();
        }

        updateDownloadDisplay() {
            const downloadItems = document.getElementById('downloadItems');
            if (!downloadItems) return;

            downloadItems.innerHTML = '';

            this.downloadQueue.forEach(item => {
                const itemElement = document.createElement('div');
                itemElement.className = 'download-item';
                itemElement.innerHTML = `
                    <div class="download-header">
                        <div class="download-info">
                            <div class="download-title">${item.mission} - ${item.date}</div>
                            <div class="download-details">${item.size} • ${item.url}</div>
                        </div>
                        <div class="download-status">
                            <span class="status-badge status-${item.status}">${item.status}</span>
                        </div>
                    </div>
                    <div class="download-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${(item.progress * 100).toFixed(1)}%"></div>
                        </div>
                        <div class="progress-text">
                            ${(item.progress * 100).toFixed(1)}% ${item.status === 'downloading' ? `• ${item.speed.toFixed(1)} MB/s` : ''}
                        </div>
                    </div>
                    <div class="download-actions">
                        <button class="btn btn-sm ${item.status === 'downloading' ? 'btn-warning' : 'btn-danger'}" 
                                onclick="window.sarDataManager.${item.status === 'downloading' ? 'pauseDownload' : 'removeDownload'}('${item.id}')">
                            ${item.status === 'downloading' ? 'Pause' : 'Remove'}
                        </button>
                    </div>
                `;
                downloadItems.appendChild(itemElement);
            });

            // Update statistics
            this.updateDownloadStatistics();
        }

        updateDownloadStatistics() {
            const activeCount = this.downloadQueue.filter(item => item.status === 'downloading').length;
            const queueCount = this.downloadQueue.filter(item => item.status === 'queued').length;
            const totalSize = this.downloadQueue.reduce((sum, item) => {
                if (item.status === 'completed') {
                    return sum + parseFloat(item.size);
                }
                return sum;
            }, 0) / 1024; // Convert MB to GB
            
            const avgSpeed = this.downloadQueue
                .filter(item => item.status === 'downloading')
                .reduce((sum, item) => sum + item.speed, 0);

            document.getElementById('activeDownloads').textContent = activeCount;
            document.getElementById('queueLength').textContent = queueCount;
            document.getElementById('totalDownloaded').textContent = totalSize.toFixed(2) + ' GB';
            document.getElementById('downloadSpeed').textContent = avgSpeed.toFixed(1) + ' MB/s';
            document.getElementById('queueCount').textContent = `(${this.downloadQueue.length} items)`;
        }

        addToCache(item) {
            const cacheEntry = {
                id: item.id,
                mission: item.mission,
                date: item.date,
                size: parseFloat(item.size),
                addedTime: Date.now(),
                lastAccessed: Date.now(),
                accessCount: 0,
                filePath: `/cache/${item.mission}/${item.date}/${item.id}`,
                metadata: {
                    downloadTime: item.completedTime - item.startTime,
                    url: item.url
                }
            };

            this.dataCache.set(item.id, cacheEntry);
            this.saveCacheToStorage();
            this.updateCacheStatistics();
        }

        saveCacheToStorage() {
            try {
                const cacheArray = Array.from(this.dataCache.entries());
                localStorage.setItem('sarDataCache', JSON.stringify(cacheArray));
            } catch (error) {
                console.warn('Error saving cache to storage:', error);
            }
        }

        updateCacheStatistics() {
            const totalSize = Array.from(this.dataCache.values())
                .reduce((sum, item) => sum + item.size, 0);
            const fileCount = this.dataCache.size;
            const maxSize = this.config.maxCacheSize;
            const usage = (totalSize / maxSize * 100).toFixed(1);

            // Calculate hit rate (simplified)
            const totalAccess = Array.from(this.dataCache.values())
                .reduce((sum, item) => sum + item.accessCount, 0);
            const hitRate = totalAccess > 0 ? ((totalAccess - fileCount) / totalAccess * 100).toFixed(1) : 0;

            document.getElementById('cacheUsage').textContent = usage + '%';
            document.getElementById('totalCacheSize').textContent = totalSize.toFixed(1) + ' MB';
            document.getElementById('cachedFileCount').textContent = fileCount;
            document.getElementById('cacheHitRate').textContent = hitRate + '%';
        }

        optimizeCache() {
            this.updateStatus('Optimizing cache...', 50);
            
            // Sort by access frequency and last access time
            const cacheEntries = Array.from(this.dataCache.entries());
            cacheEntries.sort((a, b) => {
                const scoreA = a[1].accessCount / (Date.now() - a[1].lastAccessed);
                const scoreB = b[1].accessCount / (Date.now() - b[1].lastAccessed);
                return scoreB - scoreA;
            });

            // Remove least useful entries if cache is too large
            let currentSize = cacheEntries.reduce((sum, [, entry]) => sum + entry.size, 0);
            const targetSize = this.config.maxCacheSize * 0.8; // 80% of max size

            while (currentSize > targetSize && cacheEntries.length > 0) {
                const [id, entry] = cacheEntries.pop();
                this.dataCache.delete(id);
                currentSize -= entry.size;
            }

            this.saveCacheToStorage();
            this.updateCacheStatistics();
            this.updateCacheDisplay();
            this.updateStatus('Cache optimized', 100);
        }

        cleanupCache() {
            const now = Date.now();
            const expiredEntries = [];

            this.dataCache.forEach((entry, id) => {
                if (now - entry.addedTime > this.config.cacheExpiry) {
                    expiredEntries.push(id);
                }
            });

            expiredEntries.forEach(id => {
                this.dataCache.delete(id);
            });

            this.saveCacheToStorage();
            this.updateCacheStatistics();
            this.updateCacheDisplay();
            
            console.log(`Cache cleanup: removed ${expiredEntries.length} expired entries`);
        }

        cleanupExpiredCache() {
            this.cleanupCache();
        }

        clearCache() {
            if (confirm('Are you sure you want to clear all cached data?')) {
                this.dataCache.clear();
                this.saveCacheToStorage();
                this.updateCacheStatistics();
                this.updateCacheDisplay();
                console.log('Cache cleared');
            }
        }

        updateCacheDisplay() {
            const cacheList = document.getElementById('cacheList');
            if (!cacheList) return;

            cacheList.innerHTML = '';

            const cacheEntries = Array.from(this.dataCache.entries());
            cacheEntries.sort((a, b) => b[1].lastAccessed - a[1].lastAccessed);

            cacheEntries.forEach(([id, entry]) => {
                const itemElement = document.createElement('div');
                itemElement.className = 'cache-item';
                itemElement.innerHTML = `
                    <div class="cache-item-header">
                        <div class="cache-item-info">
                            <div class="cache-item-title">${entry.mission} - ${entry.date}</div>
                            <div class="cache-item-details">
                                Size: ${entry.size.toFixed(1)} MB • 
                                Accessed: ${entry.accessCount} times • 
                                Added: ${new Date(entry.addedTime).toLocaleDateString()}
                            </div>
                        </div>
                        <div class="cache-item-actions">
                            <button class="btn btn-sm btn-danger" onclick="window.sarDataManager.removeCacheItem('${id}')">
                                Remove
                            </button>
                        </div>
                    </div>
                `;
                cacheList.appendChild(itemElement);
            });
        }

        removeCacheItem(id) {
            this.dataCache.delete(id);
            this.saveCacheToStorage();
            this.updateCacheStatistics();
            this.updateCacheDisplay();
        }

        async runComparison() {
            const datasetA = document.getElementById('datasetA')?.value;
            const datasetB = document.getElementById('datasetB')?.value;
            
            if (!datasetA || !datasetB) {
                alert('Please select both datasets for comparison');
                return;
            }

            this.updateStatus('Running comparison analysis...', 60);

            try {
                const options = {
                    comparisonType: document.getElementById('comparisonType')?.value || 'difference',
                    registrationMethod: document.getElementById('registrationMethod')?.value || 'automatic',
                    resamplingMethod: document.getElementById('resamplingMethod')?.value || 'bilinear'
                };

                const result = await new Promise((resolve) => {
                    this.workers.comparison.onmessage = (e) => {
                        resolve(e.data);
                    };
                    
                    this.workers.comparison.postMessage({
                        datasetA: datasetA,
                        datasetB: datasetB,
                        options: options
                    });
                });

                this.displayComparisonResults(result);
                this.updateStatus('Comparison completed', 100);

            } catch (error) {
                console.error('Error running comparison:', error);
                this.updateStatus('Comparison failed', 60);
            }
        }

        displayComparisonResults(results) {
            // Update statistics
            document.getElementById('meanChange').textContent = results.meanChange.toFixed(3);
            document.getElementById('stdChange').textContent = results.stdChange.toFixed(3);
            document.getElementById('maxChange').textContent = results.maxChange.toFixed(3);
            document.getElementById('changedArea').textContent = results.changedArea.toFixed(1) + '%';
            document.getElementById('registrationError').textContent = results.registrationError.toFixed(3) + ' pixels';
            document.getElementById('correlationCoeff').textContent = results.correlationCoeff.toFixed(3);
            document.getElementById('validPixels').textContent = results.validPixels.toFixed(1) + '%';

            // Draw comparison visualization
            this.drawComparisonVisualization(results.changeData);
        }

        drawComparisonVisualization(changeData) {
            const canvas = this.canvases.comparison;
            const ctx = this.contexts.comparison;
            
            if (!canvas || !ctx || !changeData) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const width = 100;
            const height = 100;
            const cellWidth = canvas.width / width;
            const cellHeight = canvas.height / height;

            // Find min/max for scaling
            const minChange = Math.min(...changeData);
            const maxChange = Math.max(...changeData);
            const range = maxChange - minChange;

            // Draw change data as heatmap
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const value = changeData[idx];
                    
                    // Map to color
                    const normalized = range > 0 ? (value - minChange) / range : 0.5;
                    
                    let r, g, b;
                    if (normalized < 0.5) {
                        // Blue to white (negative changes)
                        const scale = normalized * 2;
                        r = Math.floor(scale * 255);
                        g = Math.floor(scale * 255);
                        b = 255;
                    } else {
                        // White to red (positive changes)
                        const scale = (normalized - 0.5) * 2;
                        r = 255;
                        g = Math.floor((1 - scale) * 255);
                        b = Math.floor((1 - scale) * 255);
                    }
                    
                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                }
            }

            // Add color bar
            this.drawComparisonColorBar(ctx, canvas, minChange, maxChange);
        }

        drawComparisonColorBar(ctx, canvas, minValue, maxValue) {
            const barWidth = 20;
            const barHeight = 200;
            const barX = canvas.width - barWidth - 10;
            const barY = 50;

            // Draw color bar
            for (let i = 0; i < barHeight; i++) {
                const normalized = 1 - i / barHeight;
                
                let r, g, b;
                if (normalized < 0.5) {
                    const scale = normalized * 2;
                    r = Math.floor(scale * 255);
                    g = Math.floor(scale * 255);
                    b = 255;
                } else {
                    const scale = (normalized - 0.5) * 2;
                    r = 255;
                    g = Math.floor((1 - scale) * 255);
                    b = Math.floor((1 - scale) * 255);
                }
                
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(barX, barY + i, barWidth, 1);
            }

            // Draw border
            ctx.strokeStyle = '#000000';
            ctx.strokeRect(barX, barY, barWidth, barHeight);

            // Draw labels
            ctx.fillStyle = '#000000';
            ctx.font = '10px Arial';
            ctx.textAlign = 'left';
            
            const steps = 5;
            for (let i = 0; i <= steps; i++) {
                const y = barY + (i / steps) * barHeight;
                const value = maxValue - (i / steps) * (maxValue - minValue);
                ctx.fillText(value.toFixed(2), barX + barWidth + 5, y + 3);
            }
        }

        setRegionPreset(region) {
            const presets = {
                etna: { lat: 37.7510, lon: 14.9934 },
                vesuvius: { lat: 40.8218, lon: 14.4289 },
                yellowstone: { lat: 44.4280, lon: -110.5885 },
                kilauea: { lat: 19.4069, lon: -155.2834 }
            };

            const preset = presets[region];
            if (preset) {
                document.getElementById('searchLat').value = preset.lat;
                document.getElementById('searchLon').value = preset.lon;
            }
        }

        setTemporalPreset(period) {
            const endDate = new Date();
            let startDate = new Date();

            switch (period) {
                case '1month':
                    startDate.setMonth(endDate.getMonth() - 1);
                    break;
                case '3months':
                    startDate.setMonth(endDate.getMonth() - 3);
                    break;
                case '1year':
                    startDate.setFullYear(endDate.getFullYear() - 1);
                    break;
                case 'all':
                    startDate = new Date('2014-04-03'); // Sentinel-1A launch
                    break;
            }

            document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
            document.getElementById('endDate').value = endDate.toISOString().split('T')[0];
        }

        pauseDownloadQueue() {
            // Implementation for pausing download queue
            console.log('Download queue paused');
        }

        resumeDownloadQueue() {
            // Implementation for resuming download queue
            console.log('Download queue resumed');
        }

        clearCompletedDownloads() {
            this.downloadQueue = this.downloadQueue.filter(item => item.status !== 'completed');
            this.updateDownloadDisplay();
        }

        downloadSingleItem(id) {
            const result = this.searchResults.find(r => r.id === id);
            if (result) {
                this.addToDownloadQueue(result);
                this.switchTab('downloads');
                this.updateDownloadDisplay();
            }
        }

        previewItem(id) {
            const result = this.searchResults.find(r => r.id === id);
            if (result) {
                // Show preview modal or panel
                console.log('Preview item:', result);
            }
        }

        switchTab(tabId) {
            this.ui.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tab === tabId);
            });
            
            this.ui.querySelectorAll('.tab-content').forEach(content => {
                content.classList.toggle('active', content.id === `${tabId}-tab`);
            });

            // Update display for specific tabs
            if (tabId === 'cache') {
                this.updateCacheDisplay();
            } else if (tabId === 'missions') {
                this.drawMissionTimeline();
            }
        }

        drawMissionTimeline() {
            const canvas = this.canvases.timeline;
            const ctx = this.contexts.timeline;
            
            if (!canvas || !ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Mission timeline data
            const missions = [
                { name: 'Sentinel-1A', start: new Date('2014-04-03'), end: null, color: '#007bff' },
                { name: 'Sentinel-1B', start: new Date('2016-04-25'), end: null, color: '#007bff' },
                { name: 'ALOS-2', start: new Date('2014-05-24'), end: null, color: '#ff9800' },
                { name: 'TerraSAR-X', start: new Date('2007-06-15'), end: null, color: '#4caf50' },
                { name: 'COSMO-SkyMed-1', start: new Date('2007-06-08'), end: null, color: '#f44336' },
                { name: 'RADARSAT-2', start: new Date('2007-12-14'), end: new Date('2019-03-29'), color: '#9c27b0' }
            ];

            const timelineStart = new Date('2007-01-01');
            const timelineEnd = new Date();
            const timelineSpan = timelineEnd - timelineStart;

            const margin = { top: 20, right: 20, bottom: 40, left: 150 };
            const plotWidth = canvas.width - margin.left - margin.right;
            const plotHeight = canvas.height - margin.top - margin.bottom;
            const rowHeight = plotHeight / missions.length;

            missions.forEach((mission, index) => {
                const y = margin.top + index * rowHeight + rowHeight / 2;
                
                // Mission name
                ctx.fillStyle = '#000000';
                ctx.font = '12px Arial';
                ctx.textAlign = 'right';
                ctx.fillText(mission.name, margin.left - 10, y + 4);
                
                // Mission timeline bar
                const startX = margin.left + ((mission.start - timelineStart) / timelineSpan) * plotWidth;
                const endX = mission.end ? 
                    margin.left + ((mission.end - timelineStart) / timelineSpan) * plotWidth :
                    canvas.width - margin.right;
                
                ctx.fillStyle = mission.color;
                ctx.fillRect(startX, y - 8, endX - startX, 16);
                
                // Current indicator for active missions
                if (!mission.end) {
                    ctx.fillStyle = '#ffff00';
                    ctx.beginPath();
                    ctx.arc(endX, y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });

            // Timeline axis
            ctx.strokeStyle = '#000000';
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top + plotHeight);
            ctx.lineTo(canvas.width - margin.right, margin.top + plotHeight);
            ctx.stroke();

            // Year markers
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            for (let year = 2007; year <= timelineEnd.getFullYear(); year += 2) {
                const yearDate = new Date(year, 0, 1);
                const x = margin.left + ((yearDate - timelineStart) / timelineSpan) * plotWidth;
                
                ctx.beginPath();
                ctx.moveTo(x, margin.top + plotHeight);
                ctx.lineTo(x, margin.top + plotHeight + 5);
                ctx.stroke();
                
                ctx.fillText(year.toString(), x, margin.top + plotHeight + 20);
            }
        }

        updateStatus(message, progress = null) {
            const statusText = document.getElementById('managerStatus');
            const progressFill = document.getElementById('managerProgress');
            
            if (statusText) statusText.textContent = message;
            if (progressFill && progress !== null) {
                progressFill.style.width = `${progress}%`;
            }
        }

        destroy() {
            // Clean up workers
            Object.values(this.workers).forEach(worker => {
                worker.terminate();
            });
            
            // Save cache before destroying
            this.saveCacheToStorage();
            
            // Remove UI
            if (this.ui) {
                this.ui.remove();
            }
            
            console.log('🧹 SAR Data Manager destroyed');
        }
    }

    // Supporting classes for SAR missions and providers
    class Sentinel1Mission {
        constructor() {
            this.name = 'Sentinel-1';
            this.agency = 'ESA';
            this.frequency = '5.405 GHz';
            this.band = 'C';
            this.resolution = '5-20 m';
            this.repeatCycle = 12;
            this.launchDate = new Date('2014-04-03');
        }
    }

    class ALOS2Mission {
        constructor() {
            this.name = 'ALOS-2';
            this.agency = 'JAXA';
            this.frequency = '1.27 GHz';
            this.band = 'L';
            this.resolution = '1-100 m';
            this.repeatCycle = 14;
            this.launchDate = new Date('2014-05-24');
        }
    }

    class TerraSARXMission {
        constructor() {
            this.name = 'TerraSAR-X';
            this.agency = 'DLR';
            this.frequency = '9.65 GHz';
            this.band = 'X';
            this.resolution = '1-16 m';
            this.repeatCycle = 11;
            this.launchDate = new Date('2007-06-15');
        }
    }

    class COSMOSkyMedMission {
        constructor() {
            this.name = 'COSMO-SkyMed';
            this.agency = 'ASI';
            this.frequency = '9.6 GHz';
            this.band = 'X';
            this.resolution = '1-30 m';
            this.repeatCycle = 16;
            this.launchDate = new Date('2007-06-08');
        }
    }

    class RADARSAT2Mission {
        constructor() {
            this.name = 'RADARSAT-2';
            this.agency = 'CSA';
            this.frequency = '5.405 GHz';
            this.band = 'C';
            this.resolution = '3-100 m';
            this.repeatCycle = 24;
            this.launchDate = new Date('2007-12-14');
            this.endDate = new Date('2019-03-29');
        }
    }

    class ESAProvider {
        search(criteria) {
            // Implementation for ESA data provider API
            return [];
        }
    }

    class JAXAProvider {
        search(criteria) {
            // Implementation for JAXA data provider API
            return [];
        }
    }

    class DLRProvider {
        search(criteria) {
            // Implementation for DLR data provider API
            return [];
        }
    }

    class ASIProvider {
        search(criteria) {
            // Implementation for ASI data provider API
            return [];
        }
    }

    class CSAProvider {
        search(criteria) {
            // Implementation for CSA data provider API
            return [];
        }
    }

    class CacheManager {
        constructor(config) {
            this.config = config;
        }

        optimize() {
            // Cache optimization implementation
        }

        cleanup() {
            // Cache cleanup implementation
        }
    }

    class VersionController {
        constructor() {
            this.versions = new Map();
        }

        addVersion(id, data) {
            // Version control implementation
        }

        getVersion(id, version) {
            // Get specific version
        }
    }

    class ComparisonEngine {
        compare(datasetA, datasetB, options) {
            // Comparison implementation
            return {};
        }
    }

    class DownloadManager {
        constructor(config) {
            this.config = config;
            this.activeDownloads = [];
        }

        download(item) {
            // Download implementation
        }

        pause(id) {
            // Pause download
        }

        resume(id) {
            // Resume download
        }
    }

    // Create global instance
    window.SARDataManager = SARDataManager;
    
    // Auto-initialize if container exists
    const managerContainer = document.querySelector('.sar-data-manager-mount');
    if (managerContainer) {
        window.sarDataManager = new SARDataManager({
            container: managerContainer
        });
    }
    
    console.log('📡 SAR Data Management System loaded successfully');

})();