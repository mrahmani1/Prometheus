/**
 * Model Metrics Dashboard for Deep Learning Visualization
 * Provides comprehensive training metrics, model comparison, and performance analysis
 */

class ModelMetricsDashboard {
    constructor() {
        this.container = null;
        this.models = new Map();
        this.activeModel = null;
        this.isAnimating = false;
        this.chartInstances = new Map();
        this.realTimeMode = false;
        this.updateInterval = null;
        
        // Color schemes for different metrics
        this.colors = {
            loss: ['#ff4757', '#ff6b81'],
            accuracy: ['#2ed573', '#7bed9f'],
            precision: ['#70a1ff', '#5352ed'],
            recall: ['#ffa502', '#ff6348'],
            f1Score: ['#ff9ff3', '#f368e0'],
            validation: ['#a4b0be', '#747d8c']
        };
        
        this.initializeModels();
        this.createDashboard();
        this.setupEventListeners();
    }
    
    initializeModels() {
        // Initialize volcano prediction models with sample training data
        this.models.set('vesuvius', {
            name: 'Mt. Vesuvius Predictor',
            type: 'Neural Network',
            architecture: 'Gradient Boosted Decision Trees + Neural',
            status: 'trained',
            accuracy: 96.5,
            lastTrained: new Date('2025-01-15'),
            metrics: {
                loss: this.generateMetricHistory('loss', 0.15, 100),
                accuracy: this.generateMetricHistory('accuracy', 96.5, 100),
                precision: this.generateMetricHistory('precision', 94.2, 100),
                recall: this.generateMetricHistory('recall', 98.1, 100),
                f1Score: this.generateMetricHistory('f1', 96.1, 100),
                valLoss: this.generateMetricHistory('val_loss', 0.18, 100),
                valAccuracy: this.generateMetricHistory('val_accuracy', 95.8, 100)
            },
            hyperparameters: {
                learningRate: 0.001,
                batchSize: 32,
                epochs: 100,
                dropout: 0.3,
                hiddenLayers: [256, 128, 64]
            },
            features: ['seismicity', 'deformation', 'gas_composition', 'temperature', 'gravity', 'neural_seismic_patterns']
        });
        
        this.models.set('yellowstone', {
            name: 'Yellowstone Caldera Model',
            type: 'PhaseNet-GENIE',
            architecture: 'Graph Neural Network',
            status: 'training',
            accuracy: 94.8,
            lastTrained: new Date('2025-01-10'),
            metrics: {
                loss: this.generateMetricHistory('loss', 0.22, 85),
                accuracy: this.generateMetricHistory('accuracy', 94.8, 85),
                precision: this.generateMetricHistory('precision', 92.5, 85),
                recall: this.generateMetricHistory('recall', 96.8, 85),
                f1Score: this.generateMetricHistory('f1', 94.6, 85),
                valLoss: this.generateMetricHistory('val_loss', 0.25, 85),
                valAccuracy: this.generateMetricHistory('val_accuracy', 93.2, 85)
            },
            hyperparameters: {
                learningRate: 0.0005,
                batchSize: 16,
                epochs: 150,
                dropout: 0.4,
                hiddenLayers: [512, 256, 128]
            },
            features: ['seismic_swarms', 'ground_deformation', 'hydrothermal', 'earthquake_patterns', 'magma_migration']
        });
        
        this.models.set('piton', {
            name: 'Piton de la Fournaise Model',
            type: 'Transfer Learning',
            architecture: 'Universal Transfer Network',
            status: 'trained',
            accuracy: 97.2,
            lastTrained: new Date('2025-01-20'),
            metrics: {
                loss: this.generateMetricHistory('loss', 0.12, 120),
                accuracy: this.generateMetricHistory('accuracy', 97.2, 120),
                precision: this.generateMetricHistory('precision', 96.8, 120),
                recall: this.generateMetricHistory('recall', 97.6, 120),
                f1Score: this.generateMetricHistory('f1', 97.2, 120),
                valLoss: this.generateMetricHistory('val_loss', 0.14, 120),
                valAccuracy: this.generateMetricHistory('val_accuracy', 96.5, 120)
            },
            hyperparameters: {
                learningRate: 0.0001,
                batchSize: 24,
                epochs: 120,
                dropout: 0.25,
                hiddenLayers: [384, 192, 96]
            },
            features: ['thermal_anomalies', 'tremor_patterns', 'lava_flow_velocity', 'gas_emissions', 'tilt_data']
        });
    }
    
    generateMetricHistory(type, finalValue, epochs) {
        const history = [];
        let currentValue;
        
        switch(type) {
            case 'loss':
            case 'val_loss':
                currentValue = Math.random() * 2 + 1; // Start high
                for (let i = 0; i < epochs; i++) {
                    const progress = i / epochs;
                    const noise = (Math.random() - 0.5) * 0.1;
                    currentValue = currentValue * (1 - progress * 0.8) + finalValue * progress + noise;
                    history.push(Math.max(0.01, currentValue));
                }
                break;
            case 'accuracy':
            case 'val_accuracy':
            case 'precision':
            case 'recall':
            case 'f1':
                currentValue = Math.random() * 30 + 20; // Start low
                for (let i = 0; i < epochs; i++) {
                    const progress = i / epochs;
                    const noise = (Math.random() - 0.5) * 2;
                    currentValue = currentValue + (finalValue - currentValue) * progress * 0.1 + noise;
                    history.push(Math.max(0, Math.min(100, currentValue)));
                }
                break;
        }
        
        return history;
    }
    
    createDashboard() {
        this.container = document.createElement('div');
        this.container.id = 'model-metrics-dashboard';
        this.container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            width: 450px;
            height: 600px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 10000;
            display: none;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        `;
        
        this.container.innerHTML = `
            <div class="dashboard-header" style="
                padding: 15px 20px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            ">
                <h3 style="margin: 0; font-size: 18px; font-weight: 600;">
                    📊 Model Metrics Dashboard
                </h3>
                <div class="dashboard-controls">
                    <button id="realtime-toggle" style="
                        background: rgba(46,213,115,0.2);
                        border: 1px solid #2ed573;
                        color: #2ed573;
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        cursor: pointer;
                        margin-right: 10px;
                    ">Real-time</button>
                    <button id="close-dashboard" style="
                        background: rgba(255,71,87,0.2);
                        border: 1px solid #ff4757;
                        color: #ff4757;
                        padding: 5px 10px;
                        border-radius: 15px;
                        cursor: pointer;
                    ">×</button>
                </div>
            </div>
            
            <div class="model-selector" style="
                padding: 15px 20px;
                background: rgba(0,0,0,0.1);
            ">
                <select id="model-select" style="
                    width: 100%;
                    padding: 10px;
                    background: rgba(255,255,255,0.1);
                    border: 1px solid rgba(255,255,255,0.3);
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                ">
                    <option value="">Select Model</option>
                </select>
            </div>
            
            <div class="metrics-container" style="
                padding: 0 20px;
                height: calc(100% - 140px);
                overflow-y: auto;
            ">
                <div id="model-info" style="margin-bottom: 20px;"></div>
                <div id="metrics-charts" style="height: 300px; margin-bottom: 20px;"></div>
                <div id="comparison-view" style="display: none;"></div>
                <div id="hyperparameters" style="margin-bottom: 20px;"></div>
            </div>
        `;
        
        document.body.appendChild(this.container);
        this.populateModelSelector();
    }
    
    populateModelSelector() {
        const select = this.container.querySelector('#model-select');
        this.models.forEach((model, key) => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = model.name;
            select.appendChild(option);
        });
    }
    
    setupEventListeners() {
        const closeBtn = this.container.querySelector('#close-dashboard');
        const modelSelect = this.container.querySelector('#model-select');
        const realtimeToggle = this.container.querySelector('#realtime-toggle');
        
        closeBtn.addEventListener('click', () => this.hide());
        modelSelect.addEventListener('change', (e) => this.selectModel(e.target.value));
        realtimeToggle.addEventListener('click', () => this.toggleRealTime());
        
        // Make draggable
        this.makeDraggable();
    }
    
    makeDraggable() {
        const header = this.container.querySelector('.dashboard-header');
        let isDragging = false;
        let startX, startY, startLeft, startTop;
        
        header.style.cursor = 'move';
        
        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = parseInt(window.getComputedStyle(this.container).left);
            startTop = parseInt(window.getComputedStyle(this.container).top);
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const newLeft = startLeft + e.clientX - startX;
            const newTop = startTop + e.clientY - startY;
            
            this.container.style.left = Math.max(0, Math.min(window.innerWidth - 450, newLeft)) + 'px';
            this.container.style.top = Math.max(0, Math.min(window.innerHeight - 600, newTop)) + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }
    
    selectModel(modelKey) {
        if (!modelKey) return;
        
        this.activeModel = modelKey;
        const model = this.models.get(modelKey);
        
        this.renderModelInfo(model);
        this.renderMetricsCharts(model);
        this.renderHyperparameters(model);
    }
    
    renderModelInfo(model) {
        const infoContainer = this.container.querySelector('#model-info');
        
        const statusColor = model.status === 'trained' ? '#2ed573' : 
                          model.status === 'training' ? '#ffa502' : '#ff4757';
        
        infoContainer.innerHTML = `
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #70a1ff;">${model.name}</h4>
                    <span style="
                        background: ${statusColor}20;
                        color: ${statusColor};
                        padding: 4px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: 600;
                    ">${model.status.toUpperCase()}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 13px;">
                    <div>
                        <strong>Architecture:</strong><br>
                        <span style="color: #a4b0be;">${model.architecture}</span>
                    </div>
                    <div>
                        <strong>Accuracy:</strong><br>
                        <span style="color: #2ed573; font-size: 16px; font-weight: 600;">${model.accuracy}%</span>
                    </div>
                    <div>
                        <strong>Last Trained:</strong><br>
                        <span style="color: #a4b0be;">${model.lastTrained.toLocaleDateString()}</span>
                    </div>
                    <div>
                        <strong>Features:</strong><br>
                        <span style="color: #a4b0be;">${model.features.length} parameters</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderMetricsCharts(model) {
        const chartsContainer = this.container.querySelector('#metrics-charts');
        chartsContainer.innerHTML = '';
        
        // Create canvas for metrics visualization
        const canvas = document.createElement('canvas');
        canvas.width = 410;
        canvas.height = 280;
        canvas.style.width = '100%';
        canvas.style.height = '280px';
        chartsContainer.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        this.drawMetricsCharts(ctx, model, canvas.width, canvas.height);
    }
    
    drawMetricsCharts(ctx, model, width, height) {
        ctx.clearRect(0, 0, width, height);
        
        // Chart dimensions
        const chartWidth = width - 60;
        const chartHeight = (height - 80) / 2;
        const startX = 40;
        const startY1 = 20;
        const startY2 = startY1 + chartHeight + 40;
        
        // Draw training metrics (loss)
        this.drawLineChart(ctx, {
            data: model.metrics.loss,
            valData: model.metrics.valLoss,
            x: startX,
            y: startY1,
            width: chartWidth,
            height: chartHeight,
            title: 'Training Loss',
            color: this.colors.loss[0],
            valColor: this.colors.validation[0],
            yMin: 0,
            yMax: Math.max(...model.metrics.loss, ...model.metrics.valLoss) * 1.1
        });
        
        // Draw accuracy metrics
        this.drawLineChart(ctx, {
            data: model.metrics.accuracy,
            valData: model.metrics.valAccuracy,
            x: startX,
            y: startY2,
            width: chartWidth,
            height: chartHeight,
            title: 'Accuracy',
            color: this.colors.accuracy[0],
            valColor: this.colors.validation[1],
            yMin: 0,
            yMax: 100
        });
    }
    
    drawLineChart(ctx, options) {
        const { data, valData, x, y, width, height, title, color, valColor, yMin, yMax } = options;
        
        // Draw background
        ctx.fillStyle = 'rgba(255,255,255,0.05)';
        ctx.fillRect(x, y, width, height);
        
        // Draw title
        ctx.fillStyle = 'white';
        ctx.font = '12px Segoe UI';
        ctx.fillText(title, x + 5, y - 5);
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const gridY = y + (height / 5) * i;
            ctx.beginPath();
            ctx.moveTo(x, gridY);
            ctx.lineTo(x + width, gridY);
            ctx.stroke();
        }
        
        // Draw data lines
        this.drawDataLine(ctx, data, x, y, width, height, color, yMin, yMax);
        if (valData) {
            this.drawDataLine(ctx, valData, x, y, width, height, valColor, yMin, yMax, true);
        }
        
        // Draw legend
        ctx.fillStyle = color;
        ctx.fillRect(x + width - 80, y + 10, 12, 3);
        ctx.fillStyle = 'white';
        ctx.font = '10px Segoe UI';
        ctx.fillText('Training', x + width - 65, y + 15);
        
        if (valData) {
            ctx.fillStyle = valColor;
            ctx.fillRect(x + width - 80, y + 25, 12, 3);
            ctx.fillStyle = 'white';
            ctx.fillText('Validation', x + width - 65, y + 30);
        }
    }
    
    drawDataLine(ctx, data, x, y, width, height, color, yMin, yMax, dashed = false) {
        if (data.length < 2) return;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        
        if (dashed) {
            ctx.setLineDash([5, 5]);
        } else {
            ctx.setLineDash([]);
        }
        
        ctx.beginPath();
        
        for (let i = 0; i < data.length; i++) {
            const pointX = x + (width / (data.length - 1)) * i;
            const normalizedY = (data[i] - yMin) / (yMax - yMin);
            const pointY = y + height - (normalizedY * height);
            
            if (i === 0) {
                ctx.moveTo(pointX, pointY);
            } else {
                ctx.lineTo(pointX, pointY);
            }
        }
        
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    renderHyperparameters(model) {
        const hyperContainer = this.container.querySelector('#hyperparameters');
        
        hyperContainer.innerHTML = `
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 15px;
            ">
                <h4 style="margin: 0 0 15px 0; color: #70a1ff;">Hyperparameters</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;">
                    ${Object.entries(model.hyperparameters).map(([key, value]) => `
                        <div>
                            <strong>${key.replace(/([A-Z])/g, ' $1').toLowerCase()}:</strong><br>
                            <span style="color: #ffa502;">${Array.isArray(value) ? value.join(', ') : value}</span>
                        </div>
                    `).join('')}
                </div>
                <div style="margin-top: 15px;">
                    <strong>Features (${model.features.length}):</strong><br>
                    <div style="
                        max-height: 60px;
                        overflow-y: auto;
                        color: #a4b0be;
                        font-size: 12px;
                        margin-top: 5px;
                    ">
                        ${model.features.map(feature => `<span style="
                            display: inline-block;
                            background: rgba(112,161,255,0.2);
                            color: #70a1ff;
                            padding: 2px 6px;
                            border-radius: 10px;
                            margin: 2px;
                            font-size: 11px;
                        ">${feature.replace(/_/g, ' ')}</span>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    toggleRealTime() {
        this.realTimeMode = !this.realTimeMode;
        const button = this.container.querySelector('#realtime-toggle');
        
        if (this.realTimeMode) {
            button.style.background = 'rgba(46,213,115,0.3)';
            button.textContent = 'Live';
            this.startRealTimeUpdates();
        } else {
            button.style.background = 'rgba(46,213,115,0.2)';
            button.textContent = 'Real-time';
            this.stopRealTimeUpdates();
        }
    }
    
    startRealTimeUpdates() {
        this.updateInterval = setInterval(() => {
            if (this.activeModel) {
                const model = this.models.get(this.activeModel);
                if (model && model.status === 'training') {
                    // Simulate new training data
                    this.simulateTrainingUpdate(model);
                    this.renderMetricsCharts(model);
                }
            }
        }, 2000);
    }
    
    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    simulateTrainingUpdate(model) {
        // Add new data points to metrics
        Object.keys(model.metrics).forEach(metric => {
            const currentData = model.metrics[metric];
            const lastValue = currentData[currentData.length - 1];
            let newValue;
            
            if (metric.includes('loss')) {
                newValue = Math.max(0.01, lastValue * (0.98 + Math.random() * 0.04));
            } else {
                newValue = Math.min(100, lastValue + (Math.random() - 0.3) * 2);
            }
            
            currentData.push(newValue);
            if (currentData.length > 150) {
                currentData.shift();
            }
        });
    }
    
    show() {
        this.container.style.display = 'block';
        this.container.style.animation = 'slideInRight 0.3s ease-out';
    }
    
    hide() {
        this.container.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            this.container.style.display = 'none';
            this.stopRealTimeUpdates();
        }, 300);
    }
    
    compareModels(modelKeys) {
        const comparisonView = this.container.querySelector('#comparison-view');
        comparisonView.style.display = 'block';
        
        const models = modelKeys.map(key => this.models.get(key)).filter(Boolean);
        
        comparisonView.innerHTML = `
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
            ">
                <h4 style="margin: 0 0 15px 0; color: #70a1ff;">Model Comparison</h4>
                <div style="overflow-x: auto;">
                    <table style="
                        width: 100%;
                        border-collapse: collapse;
                        font-size: 12px;
                    ">
                        <thead>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                                <th style="text-align: left; padding: 8px;">Model</th>
                                <th style="text-align: center; padding: 8px;">Accuracy</th>
                                <th style="text-align: center; padding: 8px;">Architecture</th>
                                <th style="text-align: center; padding: 8px;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${models.map(model => `
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                    <td style="padding: 8px; color: #70a1ff;">${model.name}</td>
                                    <td style="padding: 8px; text-align: center; color: #2ed573; font-weight: 600;">${model.accuracy}%</td>
                                    <td style="padding: 8px; text-align: center; color: #a4b0be;">${model.type}</td>
                                    <td style="padding: 8px; text-align: center;">
                                        <span style="
                                            color: ${model.status === 'trained' ? '#2ed573' : '#ffa502'};
                                            font-weight: 600;
                                        ">${model.status}</span>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    exportMetrics(format = 'json') {
        if (!this.activeModel) {
            alert('Please select a model first');
            return;
        }
        
        const model = this.models.get(this.activeModel);
        const data = {
            modelName: model.name,
            architecture: model.architecture,
            accuracy: model.accuracy,
            metrics: model.metrics,
            hyperparameters: model.hyperparameters,
            exportTime: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], 
                             { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${model.name.replace(/\s+/g, '_')}_metrics.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    #model-metrics-dashboard select option {
        background: #2a5298;
        color: white;
    }
    
    #model-metrics-dashboard::-webkit-scrollbar {
        width: 8px;
    }
    
    #model-metrics-dashboard::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    #model-metrics-dashboard::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    
    #model-metrics-dashboard::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
`;
document.head.appendChild(style);

// Global instance and interface
window.modelMetricsDashboard = new ModelMetricsDashboard();

// Add to global ML Visualizer integration
if (window.mlVisualizer) {
    window.mlVisualizer.metricsDashboard = window.modelMetricsDashboard;
}

// Keyboard shortcut to toggle dashboard
document.addEventListener('keydown', (e) => {
    if (e.key === 'M' && e.shiftKey && e.ctrlKey) {
        e.preventDefault();
        if (window.modelMetricsDashboard.container.style.display === 'none') {
            window.modelMetricsDashboard.show();
        } else {
            window.modelMetricsDashboard.hide();
        }
    }
});

console.log('📊 Model Metrics Dashboard loaded successfully!');
console.log('🔑 Press Ctrl+Shift+M to toggle dashboard');
console.log('🎯 Access via: window.modelMetricsDashboard.show()');