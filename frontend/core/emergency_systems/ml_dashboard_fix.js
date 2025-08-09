/**
 * ML Dashboard Fix - Task 1.2
 * Fixes Chart.js rendering, panel layout, data layers, and error boundaries
 * @author Prometheus Platform
 * @version 2.0.0
 */

(function() {
    console.log('🚀 ML Dashboard Fix - Initializing comprehensive repairs...');
    
    // Global error boundary for chart failures
    window.chartErrorBoundary = {
        errors: [],
        maxRetries: 3,
        retryCount: {},
        
        handleError: function(chartId, error) {
            console.error(`Chart Error [${chartId}]:`, error);
            this.errors.push({ chartId, error, timestamp: Date.now() });
            
            // Retry logic
            if (!this.retryCount[chartId]) this.retryCount[chartId] = 0;
            if (this.retryCount[chartId] < this.maxRetries) {
                this.retryCount[chartId]++;
                console.log(`Retrying chart ${chartId} (${this.retryCount[chartId]}/${this.maxRetries})`);
                setTimeout(() => this.recoverChart(chartId), 1000 * this.retryCount[chartId]);
            } else {
                this.showFallback(chartId);
            }
        },
        
        recoverChart: function(chartId) {
            const canvas = document.getElementById(chartId);
            if (canvas) {
                try {
                    // Destroy existing chart if any
                    if (window.Chart && Chart.getChart(chartId)) {
                        Chart.getChart(chartId).destroy();
                    }
                    // Reinitialize chart
                    initializeChart(chartId);
                } catch (e) {
                    this.handleError(chartId, e);
                }
            }
        },
        
        showFallback: function(chartId) {
            const canvas = document.getElementById(chartId);
            if (canvas && canvas.parentElement) {
                const fallback = document.createElement('div');
                fallback.className = 'chart-error-fallback';
                fallback.innerHTML = `
                    <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.7);">
                        <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
                        <div>Chart temporarily unavailable</div>
                        <button onclick="chartErrorBoundary.recoverChart('${chartId}')" 
                                style="margin-top: 10px; padding: 5px 15px; background: rgba(255,255,255,0.2); 
                                       border: 1px solid rgba(255,255,255,0.3); border-radius: 5px; 
                                       color: white; cursor: pointer;">
                            Retry
                        </button>
                    </div>
                `;
                canvas.parentElement.appendChild(fallback);
                canvas.style.display = 'none';
            }
        }
    };
    
    // Fix 1: Modern Panel Layout System with Flexbox/Grid Hybrid
    function fixPanelLayout() {
        console.log('Fixing panel layout with flexbox/grid hybrid...');
        
        // Create improved CSS for dashboard grid
        const styleSheet = document.createElement('style');
        styleSheet.textContent = `
            /* Modern Dashboard Grid Layout */
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                padding: 20px;
                width: 100%;
                max-width: 1600px;
                margin: 0 auto;
            }
            
            @media (max-width: 1200px) {
                .dashboard-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            @media (min-width: 1201px) and (max-width: 1600px) {
                .dashboard-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            
            /* Panel Flex Container */
            .panel {
                display: flex;
                flex-direction: column;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                position: relative;
                min-height: 400px;
                transition: all 0.3s ease;
                overflow: hidden;
            }
            
            .panel:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            
            .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                flex-shrink: 0;
            }
            
            .panel-body {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
                overflow-x: hidden;
            }
            
            .panel-body::-webkit-scrollbar {
                width: 8px;
            }
            
            .panel-body::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            
            .panel-body::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
            }
            
            /* Chart Container Fixes */
            .chart-container {
                position: relative;
                height: 300px;
                width: 100%;
                margin-top: 15px;
            }
            
            .chart-container canvas {
                max-width: 100%;
                max-height: 100%;
            }
            
            /* Full-width panels */
            .panel.full-width {
                grid-column: 1 / -1;
            }
            
            /* Minimize animation */
            .panel.minimized {
                max-height: 60px;
                overflow: hidden;
            }
            
            .panel.minimized .panel-body {
                display: none;
            }
            
            /* Loading state */
            .panel.loading::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
            }
            
            .panel.loading::before {
                content: '⏳';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 48px;
                z-index: 1001;
                animation: spin 2s linear infinite;
            }
        `;
        document.head.appendChild(styleSheet);
        
        // Apply flex layout to existing panels
        const panels = document.querySelectorAll('.panel');
        panels.forEach(panel => {
            // Ensure panel has proper structure
            if (!panel.querySelector('.panel-body')) {
                const header = panel.querySelector('.panel-header');
                const content = Array.from(panel.children).filter(child => 
                    child !== header && !child.classList.contains('panel-header')
                );
                
                const body = document.createElement('div');
                body.className = 'panel-body';
                content.forEach(element => body.appendChild(element));
                panel.appendChild(body);
            }
            
            // Add minimize functionality
            const header = panel.querySelector('.panel-header');
            if (header) {
                const minimizeBtn = document.createElement('button');
                minimizeBtn.className = 'btn btn-minimize';
                minimizeBtn.innerHTML = '−';
                minimizeBtn.style.cssText = 'width: 30px; height: 30px; padding: 0; font-size: 20px;';
                minimizeBtn.onclick = function() {
                    panel.classList.toggle('minimized');
                    minimizeBtn.innerHTML = panel.classList.contains('minimized') ? '+' : '−';
                };
                
                const controls = header.querySelector('.panel-controls') || header;
                controls.appendChild(minimizeBtn);
            }
        });
        
        // Fix the full-width map panel
        const mapPanel = document.querySelector('.panel:has(#map)');
        if (mapPanel) {
            mapPanel.classList.add('full-width');
        }
    }
    
    // Fix 2: Chart.js v4.4.1 Initialization with Error Boundaries
    function initializeChart(chartId) {
        const canvas = document.getElementById(chartId);
        if (!canvas) return;
        
        try {
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Cannot get 2D context');
            
            // Destroy existing chart if any
            if (window.Chart && Chart.getChart(chartId)) {
                Chart.getChart(chartId).destroy();
            }
            
            // Chart configurations based on ID
            const configs = {
                'volcanic-chart': {
                    type: 'line',
                    data: {
                        labels: generateTimeLabels(7),
                        datasets: [{
                            label: 'Unrest Probability',
                            data: generateRandomData(7, 40, 80),
                            borderColor: 'rgb(255, 107, 107)',
                            backgroundColor: 'rgba(255, 107, 107, 0.1)',
                            tension: 0.4,
                            fill: true
                        }, {
                            label: 'Deformation Rate',
                            data: generateRandomData(7, 20, 60),
                            borderColor: 'rgb(255, 193, 7)',
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: getChartOptions('Volcanic Activity Trends')
                },
                
                'earthquake-chart': {
                    type: 'bar',
                    data: {
                        labels: ['1 day', '3 days', '7 days', '14 days', '30 days'],
                        datasets: [{
                            label: 'Probability (%)',
                            data: [5, 12, 18, 25, 35],
                            backgroundColor: createGradient(ctx, 'rgba(78, 205, 196, 0.6)', 'rgba(78, 205, 196, 0.2)'),
                            borderColor: 'rgb(78, 205, 196)',
                            borderWidth: 2
                        }]
                    },
                    options: getChartOptions('Earthquake Probability Forecast')
                },
                
                'performance-chart': {
                    type: 'radar',
                    data: {
                        labels: ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Lead Time'],
                        datasets: [{
                            label: 'Volcanic Models',
                            data: [85, 78, 81, 89, 72],
                            borderColor: 'rgb(255, 107, 107)',
                            backgroundColor: 'rgba(255, 107, 107, 0.2)',
                            pointBackgroundColor: 'rgb(255, 107, 107)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgb(255, 107, 107)'
                        }, {
                            label: 'Seismic Models',
                            data: [72, 68, 70, 82, 48],
                            borderColor: 'rgb(78, 205, 196)',
                            backgroundColor: 'rgba(78, 205, 196, 0.2)',
                            pointBackgroundColor: 'rgb(78, 205, 196)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgb(78, 205, 196)'
                        }]
                    },
                    options: getRadarChartOptions('Model Performance Metrics')
                }
            };
            
            const config = configs[chartId];
            if (config) {
                // Create chart with error handling
                const chart = new Chart(ctx, config);
                
                // Store chart instance
                window.mlDashboardCharts = window.mlDashboardCharts || {};
                window.mlDashboardCharts[chartId] = chart;
                
                console.log(`✅ Chart initialized: ${chartId}`);
            }
        } catch (error) {
            console.error(`Failed to initialize chart ${chartId}:`, error);
            chartErrorBoundary.handleError(chartId, error);
        }
    }
    
    // Helper: Generate time labels
    function generateTimeLabels(count) {
        const labels = [];
        for (let i = count - 1; i >= 0; i--) {
            if (i === 0) labels.push('Now');
            else labels.push(`${i}h ago`);
        }
        return labels;
    }
    
    // Helper: Generate random data
    function generateRandomData(count, min, max) {
        return Array.from({ length: count }, () => 
            Math.floor(Math.random() * (max - min + 1)) + min
        );
    }
    
    // Helper: Create gradient
    function createGradient(ctx, color1, color2) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, color1);
        gradient.addColorStop(1, color2);
        return gradient;
    }
    
    // Helper: Get standard chart options
    function getChartOptions(title) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: title,
                    color: 'white',
                    font: {
                        size: 14
                    }
                },
                legend: {
                    labels: {
                        color: 'white'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'white',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        };
    }
    
    // Helper: Get radar chart options
    function getRadarChartOptions(title) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    color: 'white',
                    font: {
                        size: 14
                    }
                },
                legend: {
                    labels: {
                        color: 'white'
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.8)',
                        backdropColor: 'transparent'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: 'white',
                        font: {
                            size: 11
                        }
                    }
                }
            }
        };
    }
    
    // Fix 3: Data Layer Toggle System
    function implementDataLayerToggle() {
        console.log('Implementing data layer toggle system...');
        
        const layerState = {
            volcanic: true,
            seismic: true,
            geodetic: true,
            alerts: true
        };
        
        // Create layer control panel
        const layerControlHTML = `
            <div class="data-layer-control" style="position: fixed; top: 80px; right: 20px; 
                 background: rgba(0,0,0,0.8); padding: 15px; border-radius: 10px; 
                 z-index: 1000; display: none;">
                <h3 style="color: white; margin-bottom: 10px;">Data Layers</h3>
                <div class="layer-toggles">
                    <label style="display: block; color: white; margin: 5px 0; cursor: pointer;">
                        <input type="checkbox" data-layer="volcanic" checked> 🌋 Volcanic
                    </label>
                    <label style="display: block; color: white; margin: 5px 0; cursor: pointer;">
                        <input type="checkbox" data-layer="seismic" checked> 📈 Seismic
                    </label>
                    <label style="display: block; color: white; margin: 5px 0; cursor: pointer;">
                        <input type="checkbox" data-layer="geodetic" checked> 📡 Geodetic
                    </label>
                    <label style="display: block; color: white; margin: 5px 0; cursor: pointer;">
                        <input type="checkbox" data-layer="alerts" checked> ⚠️ Alerts
                    </label>
                </div>
            </div>
        `;
        
        const controlDiv = document.createElement('div');
        controlDiv.innerHTML = layerControlHTML;
        document.body.appendChild(controlDiv.firstElementChild);
        
        // Add toggle button
        const header = document.querySelector('.header');
        if (header) {
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'btn';
            toggleBtn.innerHTML = '🗺️ Layers';
            toggleBtn.onclick = function() {
                const control = document.querySelector('.data-layer-control');
                control.style.display = control.style.display === 'none' ? 'block' : 'none';
            };
            header.querySelector('.panel-controls').appendChild(toggleBtn);
        }
        
        // Handle layer toggles
        document.querySelectorAll('.data-layer-control input').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const layer = this.dataset.layer;
                layerState[layer] = this.checked;
                updateDataLayers(layer, this.checked);
            });
        });
        
        // Global toggle layer function
        window.toggleLayer = function(layer) {
            layerState[layer] = !layerState[layer];
            updateDataLayers(layer, layerState[layer]);
            
            // Update button state
            if (event && event.target) {
                event.target.classList.toggle('active', layerState[layer]);
            }
        };
    }
    
    // Helper: Update data layers
    function updateDataLayers(layer, enabled) {
        console.log(`${enabled ? 'Enabling' : 'Disabling'} layer: ${layer}`);
        
        // Update map markers
        if (window.map) {
            window.map.eachLayer(function(mapLayer) {
                if (mapLayer.options && mapLayer.options.dataLayer === layer) {
                    if (enabled) {
                        mapLayer.setOpacity(1);
                    } else {
                        mapLayer.setOpacity(0);
                    }
                }
            });
        }
        
        // Update panels
        const layerPanels = {
            volcanic: '.panel:has(#volcanic-predictions)',
            seismic: '.panel:has(#earthquake-predictions)',
            alerts: '.panel:has(#alerts-container)'
        };
        
        if (layerPanels[layer]) {
            const panel = document.querySelector(layerPanels[layer]);
            if (panel) {
                panel.style.display = enabled ? 'flex' : 'none';
            }
        }
        
        // Update charts
        if (window.mlDashboardCharts) {
            const chartMap = {
                volcanic: 'volcanic-chart',
                seismic: 'earthquake-chart'
            };
            
            if (chartMap[layer] && window.mlDashboardCharts[chartMap[layer]]) {
                const chart = window.mlDashboardCharts[chartMap[layer]];
                chart.options.plugins.legend.display = enabled;
                chart.update();
            }
        }
    }
    
    // Fix 4: Real-time Data Updates with Error Handling
    function enhanceRealTimeUpdates() {
        console.log('Enhancing real-time data updates...');
        
        let updateInterval;
        const updateFrequency = 30000; // 30 seconds
        
        function safeUpdate() {
            try {
                // Update volcanic chart
                if (window.mlDashboardCharts && window.mlDashboardCharts['volcanic-chart']) {
                    const chart = window.mlDashboardCharts['volcanic-chart'];
                    chart.data.datasets.forEach(dataset => {
                        dataset.data.shift();
                        dataset.data.push(Math.floor(Math.random() * 40) + 40);
                    });
                    chart.update('none'); // No animation for smooth updates
                }
                
                // Update earthquake chart
                if (window.mlDashboardCharts && window.mlDashboardCharts['earthquake-chart']) {
                    const chart = window.mlDashboardCharts['earthquake-chart'];
                    chart.data.datasets[0].data = chart.data.datasets[0].data.map(
                        v => Math.max(0, Math.min(100, v + (Math.random() - 0.5) * 10))
                    );
                    chart.update('none');
                }
                
                // Update metrics
                updateMetrics();
                
                // Update alerts
                if (Math.random() > 0.8) {
                    addNewAlert();
                }
                
            } catch (error) {
                console.error('Error in real-time update:', error);
            }
        }
        
        // Clear existing interval
        if (window.mlDashboardUpdateInterval) {
            clearInterval(window.mlDashboardUpdateInterval);
        }
        
        // Start new interval
        window.mlDashboardUpdateInterval = setInterval(safeUpdate, updateFrequency);
        
        // Add control to pause/resume updates
        const pauseBtn = document.createElement('button');
        pauseBtn.className = 'btn';
        pauseBtn.innerHTML = '⏸️ Pause Updates';
        pauseBtn.onclick = function() {
            if (window.mlDashboardUpdateInterval) {
                clearInterval(window.mlDashboardUpdateInterval);
                window.mlDashboardUpdateInterval = null;
                pauseBtn.innerHTML = '▶️ Resume Updates';
            } else {
                window.mlDashboardUpdateInterval = setInterval(safeUpdate, updateFrequency);
                pauseBtn.innerHTML = '⏸️ Pause Updates';
            }
        };
        
        const header = document.querySelector('.header .panel-controls');
        if (header) {
            header.appendChild(pauseBtn);
        }
    }
    
    // Helper: Update metrics
    function updateMetrics() {
        const metrics = document.querySelectorAll('.metric-value');
        metrics.forEach(metric => {
            const current = parseFloat(metric.textContent);
            if (!isNaN(current)) {
                const change = (Math.random() - 0.5) * 5;
                const newValue = Math.max(0, current + change);
                
                if (metric.textContent.includes('%')) {
                    metric.textContent = Math.min(100, newValue).toFixed(0) + '%';
                } else if (metric.textContent.includes('.')) {
                    metric.textContent = newValue.toFixed(2);
                }
            }
        });
    }
    
    // Helper: Add new alert
    function addNewAlert() {
        const alerts = [
            { icon: '🌋', title: 'Volcanic Unrest Detected', desc: 'Increased activity at monitoring station' },
            { icon: '📈', title: 'Seismic Swarm', desc: 'Cluster of small earthquakes detected' },
            { icon: '📡', title: 'GPS Displacement', desc: 'Surface deformation measured' },
            { icon: '⚠️', title: 'Model Confidence Drop', desc: 'Prediction uncertainty increased' }
        ];
        
        const alert = alerts[Math.floor(Math.random() * alerts.length)];
        const container = document.getElementById('alerts-container');
        
        if (container && container.children.length < 10) {
            const alertBox = document.createElement('div');
            alertBox.className = 'alert-box';
            alertBox.style.animation = 'slideIn 0.5s ease';
            alertBox.innerHTML = `
                <div class="alert-header">
                    <span class="alert-icon">${alert.icon}</span>
                    <span class="alert-title">${alert.title}</span>
                    <span style="margin-left: auto; cursor: pointer;" onclick="this.parentElement.parentElement.remove()">✕</span>
                </div>
                <p>${alert.desc}</p>
            `;
            container.insertBefore(alertBox, container.firstChild);
            
            // Update count
            const count = container.children.length;
            const countElement = document.getElementById('alert-count');
            if (countElement) {
                countElement.textContent = count + ' Active';
            }
        }
    }
    
    // Fix 5: Enhanced Map Integration
    function fixMapIntegration() {
        console.log('Fixing map integration...');
        
        setTimeout(() => {
            if (window.map) {
                // Force resize
                window.map.invalidateSize();
                
                // Add layer groups
                const volcanoLayer = L.layerGroup().addTo(window.map);
                const seismicLayer = L.layerGroup().addTo(window.map);
                const geodeticLayer = L.layerGroup().addTo(window.map);
                
                // Store layers globally
                window.mapLayers = {
                    volcanic: volcanoLayer,
                    seismic: seismicLayer,
                    geodetic: geodeticLayer
                };
                
                // Add sample markers with proper layer assignment
                addEnhancedMarkers();
            }
        }, 1000);
    }
    
    // Helper: Add enhanced markers
    function addEnhancedMarkers() {
        // Volcanic markers
        const volcanoes = [
            { name: 'Kilauea', lat: 19.4069, lon: -155.2834, unrest: 72 },
            { name: 'Mount Fuji', lat: 35.3606, lon: 138.7274, unrest: 5 },
            { name: 'Vesuvius', lat: 40.8209, lon: 14.4289, unrest: 15 }
        ];
        
        volcanoes.forEach(v => {
            const color = v.unrest > 50 ? '#ff6b6b' : v.unrest > 20 ? '#ffc107' : '#4caf50';
            const marker = L.circleMarker([v.lat, v.lon], {
                radius: 8 + v.unrest / 10,
                fillColor: color,
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8,
                dataLayer: 'volcanic'
            }).bindPopup(`
                <b>${v.name}</b><br>
                Unrest: ${v.unrest}%<br>
                Status: ${v.unrest > 50 ? 'High Alert' : v.unrest > 20 ? 'Elevated' : 'Normal'}
            `);
            
            if (window.mapLayers.volcanic) {
                marker.addTo(window.mapLayers.volcanic);
            }
        });
        
        // Seismic markers
        const earthquakes = [
            { name: 'San Andreas', lat: 37.5, lon: -122.5, prob: 18 },
            { name: 'Tokyo', lat: 35.7, lon: 139.7, prob: 12 },
            { name: 'Chile', lat: -33.4, lon: -70.6, prob: 25 }
        ];
        
        earthquakes.forEach(eq => {
            const size = 5 + eq.prob / 2;
            const marker = L.circleMarker([eq.lat, eq.lon], {
                radius: size,
                fillColor: '#4ecdc4',
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.6,
                dataLayer: 'seismic'
            }).bindPopup(`
                <b>${eq.name}</b><br>
                7-day probability: ${eq.prob}%<br>
                Expected: M5.5-6.2
            `);
            
            if (window.mapLayers.seismic) {
                marker.addTo(window.mapLayers.seismic);
            }
        });
    }
    
    // Main initialization
    function initializeMLDashboardFix() {
        console.log('Initializing ML Dashboard fixes...');
        
        // Apply fixes in sequence
        try {
            // 1. Fix panel layout
            fixPanelLayout();
            
            // 2. Initialize charts with error boundaries
            setTimeout(() => {
                ['volcanic-chart', 'earthquake-chart', 'performance-chart'].forEach(chartId => {
                    initializeChart(chartId);
                });
            }, 500);
            
            // 3. Implement data layer toggles
            implementDataLayerToggle();
            
            // 4. Enhance real-time updates
            enhanceRealTimeUpdates();
            
            // 5. Fix map integration
            fixMapIntegration();
            
            // Add animation styles
            const animationStyles = document.createElement('style');
            animationStyles.textContent = `
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                .alert-box {
                    animation: slideIn 0.5s ease;
                }
                
                .panel {
                    animation: fadeIn 0.5s ease;
                }
            `;
            document.head.appendChild(animationStyles);
            
            console.log('✅ ML Dashboard fixes applied successfully');
            
        } catch (error) {
            console.error('❌ Error applying ML Dashboard fixes:', error);
        }
    }
    
    // Apply fixes when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeMLDashboardFix);
    } else {
        initializeMLDashboardFix();
    }
    
    // Export for manual triggering
    window.mlDashboardFix = {
        init: initializeMLDashboardFix,
        fixLayout: fixPanelLayout,
        fixCharts: () => {
            ['volcanic-chart', 'earthquake-chart', 'performance-chart'].forEach(initializeChart);
        },
        toggleLayers: implementDataLayerToggle,
        errorBoundary: chartErrorBoundary
    };
    
})();