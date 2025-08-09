/**
 * Advanced Thermal Modeling System
 * Implements heat diffusion, thermal boundary conditions, cooling models,
 * magma chamber thermal evolution, and thermal anomaly detection
 */

class ThermalModel {
    constructor() {
        this.container = null;
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;
        this.isRunning = false;
        
        // Model parameters
        this.gridSize = { width: 150, height: 120 };
        this.physicalSize = { width: 50000, height: 40000 }; // 50x40 km
        this.timeStep = 3600; // 1 hour in seconds
        this.currentTime = 0; // seconds
        
        // Thermal properties
        this.thermalProperties = {
            rockThermalDiffusivity: 1e-6, // m²/s
            magmaThermalDiffusivity: 5e-7,
            waterThermalDiffusivity: 1.4e-7,
            rockThermalConductivity: 2.5, // W/m/K
            magmaThermalConductivity: 1.2,
            waterThermalConductivity: 0.6,
            rockDensity: 2700, // kg/m³
            magmaDensity: 2400,
            waterDensity: 1000,
            rockSpecificHeat: 1000, // J/kg/K
            magmaSpecificHeat: 1200,
            waterSpecificHeat: 4186,
            ambientTemperature: 15, // °C
            geothermalGradient: 0.025, // °C/m
            magmaTemperature: 1200,
            latentHeatFusion: 400000 // J/kg
        };
        
        // Model fields
        this.temperature = null;
        this.thermalDiffusivity = null;
        this.heatSources = [];
        this.magmaChambers = [];
        this.thermalAnomalies = [];
        this.coolingHistory = [];
        
        // Visualization settings
        this.temperatureColorScheme = [
            { temp: 0, color: [0, 0, 139] },      // Dark blue
            { temp: 50, color: [0, 100, 255] },   // Blue
            { temp: 100, color: [0, 255, 255] },  // Cyan
            { temp: 200, color: [0, 255, 0] },    // Green
            { temp: 400, color: [255, 255, 0] },  // Yellow
            { temp: 600, color: [255, 165, 0] },  // Orange
            { temp: 800, color: [255, 0, 0] },    // Red
            { temp: 1000, color: [139, 0, 0] },   // Dark red
            { temp: 1200, color: [255, 255, 255] } // White
        ];
        
        this.currentVisualization = 'temperature';
        this.showHeatSources = true;
        this.showMagmaChambers = true;
        this.showAnomalies = true;
        this.showIsotherms = true;
        
        this.initializeFields();
        this.createInterface();
        this.setupControls();
        
        console.log('🔥 Advanced Thermal Model initialized');
        console.log('🌡️ Heat diffusion with thermal boundary conditions');
    }
    
    initializeFields() {
        const { width, height } = this.gridSize;
        
        // Initialize temperature field with geothermal gradient
        this.temperature = Array(height).fill().map(() => Array(width).fill(0));
        this.thermalDiffusivity = Array(height).fill().map(() => Array(width).fill(0));
        
        const dx = this.physicalSize.width / width;
        const dy = this.physicalSize.height / height;
        
        for (let j = 0; j < height; j++) {
            const depth = j * dy;
            const geothermTemp = this.thermalProperties.ambientTemperature + 
                               depth * this.thermalProperties.geothermalGradient;
            
            for (let i = 0; i < width; i++) {
                this.temperature[j][i] = geothermTemp;
                this.thermalDiffusivity[j][i] = this.thermalProperties.rockThermalDiffusivity;
            }
        }
        
        // Initialize heat sources
        this.initializeHeatSources();
        
        // Initialize magma chambers
        this.initializeMagmaChambers();
        
        // Initialize thermal anomalies detection
        this.thermalAnomalies = [];
    }
    
    initializeHeatSources() {
        this.heatSources = [
            {
                id: 'geothermal_vent_1',
                type: 'point',
                position: { x: 75, y: 100 },
                intensity: 500000, // W/m²
                temperature: 350,
                active: true,
                timeVariation: {
                    enabled: true,
                    period: 86400 * 7, // 1 week
                    amplitude: 0.3
                }
            },
            {
                id: 'magma_intrusion',
                type: 'rectangular',
                position: { x: 50, y: 80 },
                dimensions: { width: 20, height: 15 },
                intensity: 800000,
                temperature: 1000,
                active: true,
                coolingRate: 1e-6
            },
            {
                id: 'hydrothermal_field',
                type: 'distributed',
                position: { x: 100, y: 60 },
                radius: 25,
                intensity: 300000,
                temperature: 200,
                active: true,
                distribution: 'gaussian'
            }
        ];
    }
    
    initializeMagmaChambers() {
        this.magmaChambers = [
            {
                id: 'main_chamber',
                position: { x: 60, y: 90 },
                dimensions: { width: 25, height: 20 },
                temperature: 1200,
                volume: 5e9, // m³
                crystallinity: 0.1,
                volatileContent: 0.05,
                composition: 'basaltic',
                coolingRate: 0,
                active: true,
                thermalHistory: [],
                convection: {
                    enabled: true,
                    vigour: 0.8,
                    cells: []
                }
            },
            {
                id: 'shallow_chamber',
                position: { x: 80, y: 40 },
                dimensions: { width: 15, height: 12 },
                temperature: 950,
                volume: 1e9,
                crystallinity: 0.3,
                volatileContent: 0.03,
                composition: 'andesitic',
                coolingRate: 0,
                active: true,
                thermalHistory: [],
                convection: {
                    enabled: false,
                    vigour: 0.2,
                    cells: []
                }
            }
        ];
        
        // Initialize magma chamber thermal history
        this.magmaChambers.forEach(chamber => {
            chamber.thermalHistory = [{
                time: 0,
                temperature: chamber.temperature,
                crystallinity: chamber.crystallinity,
                volume: chamber.volume
            }];
        });
    }
    
    createInterface() {
        this.container = document.createElement('div');
        this.container.id = 'thermal-model-panel';
        this.container.style.cssText = `
            position: fixed;
            top: 60px;
            right: 50px;
            width: 800px;
            height: 700px;
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            z-index: 9997;
            display: none;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        `;
        
        this.container.innerHTML = `
            <div class="thermal-header" style="
                padding: 20px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            ">
                <h3 style="margin: 0; font-size: 18px; font-weight: 600;">
                    🔥 Thermal Model - Heat Diffusion & Cooling
                </h3>
                <div class="header-controls">
                    <button id="thermal-snapshot" style="
                        background: rgba(241,196,15,0.2);
                        border: 1px solid #f1c40f;
                        color: #f1c40f;
                        padding: 8px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        cursor: pointer;
                        margin-right: 10px;
                    ">📸 Snapshot</button>
                    <button id="close-thermal" style="
                        background: rgba(255,71,87,0.2);
                        border: 1px solid #ff4757;
                        color: #ff4757;
                        padding: 8px 12px;
                        border-radius: 15px;
                        cursor: pointer;
                    ">×</button>
                </div>
            </div>
            
            <div class="thermal-controls" style="
                padding: 15px 20px;
                background: rgba(0,0,0,0.1);
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            ">
                <div>
                    <label style="font-size: 12px; margin-right: 8px;">Visualization:</label>
                    <select id="thermal-visualization" style="
                        padding: 6px 10px;
                        background: rgba(255,255,255,0.1);
                        border: 1px solid rgba(255,255,255,0.3);
                        border-radius: 6px;
                        color: white;
                        font-size: 12px;
                    ">
                        <option value="temperature">🌡️ Temperature</option>
                        <option value="heat_flow">🌊 Heat Flow</option>
                        <option value="thermal_gradient">📈 Thermal Gradient</option>
                        <option value="diffusivity">🔄 Thermal Diffusivity</option>
                    </select>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-heat-sources" checked style="margin-right: 5px;">
                        Heat Sources
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-chambers" checked style="margin-right: 5px;">
                        Magma Chambers
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-isotherms" checked style="margin-right: 5px;">
                        Isotherms
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-anomalies" checked style="margin-right: 5px;">
                        Anomalies
                    </label>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <button id="thermal-play-pause" style="
                        background: linear-gradient(45deg, #e74c3c, #c0392b);
                        border: none;
                        color: white;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        font-weight: 600;
                    ">▶️ Start</button>
                    
                    <button id="thermal-reset" style="
                        background: rgba(52,152,219,0.2);
                        border: 1px solid #3498db;
                        color: #3498db;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                    ">🔄 Reset</button>
                </div>
                
                <div style="flex: 1; text-align: right; font-size: 12px; color: #ecf0f1;">
                    Time: <span id="thermal-time">0 hours</span>
                </div>
            </div>
            
            <div class="thermal-visualization" style="
                position: relative;
                height: calc(100% - 140px);
                padding: 20px;
            ">
                <canvas id="thermal-canvas" style="
                    width: 500px;
                    height: 400px;
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 10px;
                    background: #2c3e50;
                    cursor: crosshair;
                "></canvas>
                
                <div id="thermal-colorbar" style="
                    position: absolute;
                    right: 40px;
                    top: 40px;
                    width: 25px;
                    height: 350px;
                    border: 1px solid rgba(255,255,255,0.3);
                    border-radius: 12px;
                    overflow: hidden;
                "></div>
                
                <div id="thermal-colorbar-labels" style="
                    position: absolute;
                    right: 70px;
                    top: 40px;
                    height: 350px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    font-size: 11px;
                    color: #ecf0f1;
                "></div>
                
                <div id="thermal-info" style="
                    position: absolute;
                    left: 540px;
                    top: 40px;
                    width: 200px;
                    background: rgba(0,0,0,0.7);
                    padding: 15px;
                    border-radius: 10px;
                    font-size: 12px;
                ">
                    <div><strong>Model Status:</strong></div>
                    <div id="thermal-status">Initialized</div>
                    <div style="margin-top: 10px;"><strong>Statistics:</strong></div>
                    <div id="thermal-stats">
                        Max Temp: 0°C<br>
                        Min Temp: 0°C<br>
                        Avg Temp: 0°C<br>
                        Heat Sources: ${this.heatSources.length}<br>
                        Magma Chambers: ${this.magmaChambers.length}
                    </div>
                    <div style="margin-top: 10px;"><strong>Cursor Info:</strong></div>
                    <div id="thermal-cursor-info">Move cursor over model</div>
                </div>
                
                <div id="chamber-evolution" style="
                    position: absolute;
                    left: 40px;
                    bottom: 40px;
                    width: 450px;
                    height: 100px;
                    background: rgba(0,0,0,0.7);
                    padding: 10px;
                    border-radius: 10px;
                    font-size: 11px;
                    overflow-y: auto;
                ">
                    <div><strong>Magma Chamber Evolution:</strong></div>
                    <div id="chamber-history" style="margin-top: 5px;"></div>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.container);
        this.canvas = this.container.querySelector('#thermal-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 500;
        this.canvas.height = 400;
        
        this.createThermalColorbar();
    }
    
    createThermalColorbar() {
        const colorbar = this.container.querySelector('#thermal-colorbar');
        const labels = this.container.querySelector('#thermal-colorbar-labels');
        
        // Create temperature gradient
        const canvas = document.createElement('canvas');
        canvas.width = 25;
        canvas.height = 350;
        const ctx = canvas.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 0, 350);
        this.temperatureColorScheme.forEach((item, index) => {
            const position = 1 - (item.temp / 1200); // Invert for top-to-bottom
            const color = `rgb(${item.color[0]}, ${item.color[1]}, ${item.color[2]})`;
            gradient.addColorStop(Math.max(0, Math.min(1, position)), color);
        });
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 25, 350);
        
        colorbar.style.backgroundImage = `url(${canvas.toDataURL()})`;
        
        // Update labels
        labels.innerHTML = '';
        const temperatures = [1200, 1000, 800, 600, 400, 200, 100, 50, 0];
        temperatures.forEach(temp => {
            const label = document.createElement('div');
            label.textContent = `${temp}°C`;
            labels.appendChild(label);
        });
    }
    
    setupControls() {
        const closeBtn = this.container.querySelector('#close-thermal');
        const playPauseBtn = this.container.querySelector('#thermal-play-pause');
        const resetBtn = this.container.querySelector('#thermal-reset');
        const snapshotBtn = this.container.querySelector('#thermal-snapshot');
        const visualizationSelect = this.container.querySelector('#thermal-visualization');
        
        const showHeatSourcesCheck = this.container.querySelector('#show-heat-sources');
        const showChambersCheck = this.container.querySelector('#show-chambers');
        const showIsothermsCheck = this.container.querySelector('#show-isotherms');
        const showAnomaliesCheck = this.container.querySelector('#show-anomalies');
        
        closeBtn.addEventListener('click', () => this.hide());
        playPauseBtn.addEventListener('click', () => this.toggleSimulation());
        resetBtn.addEventListener('click', () => this.resetModel());
        snapshotBtn.addEventListener('click', () => this.takeSnapshot());
        
        visualizationSelect.addEventListener('change', (e) => {
            this.currentVisualization = e.target.value;
            this.render();
        });
        
        showHeatSourcesCheck.addEventListener('change', (e) => {
            this.showHeatSources = e.target.checked;
            this.render();
        });
        
        showChambersCheck.addEventListener('change', (e) => {
            this.showMagmaChambers = e.target.checked;
            this.render();
        });
        
        showIsothermsCheck.addEventListener('change', (e) => {
            this.showIsotherms = e.target.checked;
            this.render();
        });
        
        showAnomaliesCheck.addEventListener('change', (e) => {
            this.showAnomalies = e.target.checked;
            this.render();
        });
        
        // Mouse interaction
        this.canvas.addEventListener('mousemove', (e) => this.updateCursorInfo(e));
        
        this.makeDraggable();
    }
    
    updateCursorInfo(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * this.gridSize.width);
        const y = Math.floor((e.clientY - rect.top) / rect.height * this.gridSize.height);
        
        if (x >= 0 && x < this.gridSize.width && y >= 0 && y < this.gridSize.height) {
            const temp = this.temperature[y][x];
            const diffusivity = this.thermalDiffusivity[y][x];
            const physicalX = (x / this.gridSize.width) * this.physicalSize.width / 1000; // km
            const physicalY = (y / this.gridSize.height) * this.physicalSize.height / 1000; // km
            
            const cursorInfo = this.container.querySelector('#thermal-cursor-info');
            cursorInfo.innerHTML = `
                Position: ${physicalX.toFixed(1)}, ${physicalY.toFixed(1)} km<br>
                Temperature: ${temp.toFixed(1)}°C<br>
                Thermal Diffusivity: ${(diffusivity * 1e6).toFixed(2)} mm²/s<br>
                Grid: (${x}, ${y})
            `;
        }
    }
    
    toggleSimulation() {
        const btn = this.container.querySelector('#thermal-play-pause');
        
        if (this.isRunning) {
            this.stopSimulation();
            btn.innerHTML = '▶️ Start';
            btn.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
        } else {
            this.startSimulation();
            btn.innerHTML = '⏸️ Pause';
            btn.style.background = 'linear-gradient(45deg, #27ae60, #2ecc71)';
        }
    }
    
    startSimulation() {
        this.isRunning = true;
        this.animate();
        
        const status = this.container.querySelector('#thermal-status');
        status.textContent = 'Running simulation...';
    }
    
    stopSimulation() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        const status = this.container.querySelector('#thermal-status');
        status.textContent = 'Simulation paused';
    }
    
    animate() {
        if (!this.isRunning) return;
        
        this.updateThermalModel();
        this.detectThermalAnomalies();
        this.updateMagmaChambers();
        this.render();
        this.updateStatistics();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    updateThermalModel() {
        const dt = this.timeStep;
        this.currentTime += dt;
        
        // Heat diffusion equation: ∂T/∂t = α∇²T + Q
        this.solveThermalDiffusion(dt);
        
        // Apply heat sources
        this.applyHeatSources(dt);
        
        // Apply boundary conditions
        this.applyBoundaryConditions();
        
        // Update time display
        const timeDisplay = this.container.querySelector('#thermal-time');
        const hours = Math.floor(this.currentTime / 3600);
        const days = Math.floor(hours / 24);
        
        if (days > 0) {
            timeDisplay.textContent = `${days} days, ${hours % 24} hours`;
        } else {
            timeDisplay.textContent = `${hours} hours`;
        }
    }
    
    solveThermalDiffusion(dt) {
        const { width, height } = this.gridSize;
        const dx = this.physicalSize.width / width;
        const dy = this.physicalSize.height / height;
        
        const newTemp = Array(height).fill().map(() => Array(width).fill(0));
        
        for (let j = 1; j < height - 1; j++) {
            for (let i = 1; i < width - 1; i++) {
                const T = this.temperature[j][i];
                const alpha = this.thermalDiffusivity[j][i];
                
                // Second derivatives (finite difference)
                const d2T_dx2 = (this.temperature[j][i+1] - 2*T + this.temperature[j][i-1]) / (dx*dx);
                const d2T_dy2 = (this.temperature[j+1][i] - 2*T + this.temperature[j-1][i]) / (dy*dy);
                
                // Heat diffusion equation
                const dT_dt = alpha * (d2T_dx2 + d2T_dy2);
                
                newTemp[j][i] = T + dT_dt * dt;
            }
        }
        
        // Copy boundary values
        for (let j = 0; j < height; j++) {
            newTemp[j][0] = this.temperature[j][0];
            newTemp[j][width-1] = this.temperature[j][width-1];
        }
        for (let i = 0; i < width; i++) {
            newTemp[0][i] = this.temperature[0][i];
            newTemp[height-1][i] = this.temperature[height-1][i];
        }
        
        this.temperature = newTemp;
    }
    
    applyHeatSources(dt) {
        this.heatSources.forEach(source => {
            if (source.active) {
                let intensity = source.intensity;
                
                // Apply time variation if enabled
                if (source.timeVariation && source.timeVariation.enabled) {
                    const phase = (2 * Math.PI * this.currentTime) / source.timeVariation.period;
                    const variation = 1 + source.timeVariation.amplitude * Math.sin(phase);
                    intensity *= variation;
                }
                
                this.applyHeatSource(source, intensity, dt);
            }
        });
    }
    
    applyHeatSource(source, intensity, dt) {
        const { width, height } = this.gridSize;
        const dx = this.physicalSize.width / width;
        const dy = this.physicalSize.height / height;
        
        switch (source.type) {
            case 'point':
                const x = Math.floor(source.position.x);
                const y = Math.floor(source.position.y);
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    const heatInput = (intensity * dt) / (this.thermalProperties.rockDensity * 
                                     this.thermalProperties.rockSpecificHeat * dx * dy);
                    this.temperature[y][x] += heatInput;
                }
                break;
                
            case 'rectangular':
                const x1 = Math.floor(source.position.x - source.dimensions.width/2);
                const y1 = Math.floor(source.position.y - source.dimensions.height/2);
                const x2 = x1 + source.dimensions.width;
                const y2 = y1 + source.dimensions.height;
                
                for (let j = Math.max(0, y1); j < Math.min(height, y2); j++) {
                    for (let i = Math.max(0, x1); i < Math.min(width, x2); i++) {
                        const heatInput = (intensity * dt) / (this.thermalProperties.rockDensity * 
                                         this.thermalProperties.rockSpecificHeat * dx * dy);
                        this.temperature[j][i] += heatInput;
                    }
                }
                break;
                
            case 'distributed':
                const centerX = source.position.x;
                const centerY = source.position.y;
                const radius = source.radius;
                
                for (let j = 0; j < height; j++) {
                    for (let i = 0; i < width; i++) {
                        const distance = Math.sqrt((i - centerX)**2 + (j - centerY)**2);
                        if (distance <= radius) {
                            let factor = 1;
                            if (source.distribution === 'gaussian') {
                                factor = Math.exp(-(distance**2) / (2 * (radius/3)**2));
                            } else if (source.distribution === 'linear') {
                                factor = 1 - distance / radius;
                            }
                            
                            const heatInput = (intensity * factor * dt) / (this.thermalProperties.rockDensity * 
                                             this.thermalProperties.rockSpecificHeat * dx * dy);
                            this.temperature[j][i] += heatInput;
                        }
                    }
                }
                break;
        }
    }
    
    applyBoundaryConditions() {
        const { width, height } = this.gridSize;
        
        // Surface temperature (constant)
        for (let i = 0; i < width; i++) {
            this.temperature[0][i] = this.thermalProperties.ambientTemperature;
        }
        
        // Sides (no-flux or constant temperature)
        for (let j = 0; j < height; j++) {
            // Left boundary
            this.temperature[j][0] = this.temperature[j][1];
            // Right boundary
            this.temperature[j][width-1] = this.temperature[j][width-2];
        }
        
        // Bottom boundary (geothermal gradient)
        const bottomDepth = this.physicalSize.height;
        const bottomTemp = this.thermalProperties.ambientTemperature + 
                          bottomDepth * this.thermalProperties.geothermalGradient;
        for (let i = 0; i < width; i++) {
            this.temperature[height-1][i] = bottomTemp;
        }
    }
    
    updateMagmaChambers() {
        this.magmaChambers.forEach(chamber => {
            if (chamber.active) {
                // Calculate cooling rate based on surroundings
                const avgSurroundingTemp = this.getAverageTemperatureAroundChamber(chamber);
                const tempDifference = chamber.temperature - avgSurroundingTemp;
                
                // Cooling rate proportional to temperature difference
                chamber.coolingRate = tempDifference * 1e-6; // Very slow cooling
                
                // Update chamber temperature
                chamber.temperature -= chamber.coolingRate * this.timeStep;
                
                // Update crystallinity (increases as temperature decreases)
                const solidusTemp = this.getSolidusTemperature(chamber.composition);
                if (chamber.temperature < chamber.temperature) {
                    const crystalizationRate = (solidusTemp - chamber.temperature) / solidusTemp * 1e-7;
                    chamber.crystallinity = Math.min(0.95, chamber.crystallinity + crystalizationRate * this.timeStep);
                }
                
                // Record thermal history
                if (this.currentTime % (3600 * 24) < this.timeStep) { // Daily recording
                    chamber.thermalHistory.push({
                        time: this.currentTime,
                        temperature: chamber.temperature,
                        crystallinity: chamber.crystallinity,
                        volume: chamber.volume * (1 - chamber.crystallinity * 0.1) // Volume reduction due to crystallization
                    });
                }
                
                // Apply chamber temperature to model grid
                this.applyChamberTemperature(chamber);
                
                // Deactivate if fully crystallized
                if (chamber.crystallinity > 0.9) {
                    chamber.active = false;
                }
            }
        });
        
        this.updateChamberEvolutionDisplay();
    }
    
    getAverageTemperatureAroundChamber(chamber) {
        const { width, height } = this.gridSize;
        const x = Math.floor(chamber.position.x);
        const y = Math.floor(chamber.position.y);
        const w = chamber.dimensions.width;
        const h = chamber.dimensions.height;
        
        let totalTemp = 0;
        let count = 0;
        
        // Sample temperatures around chamber perimeter
        const margin = 2;
        for (let j = Math.max(0, y - h/2 - margin); j < Math.min(height, y + h/2 + margin); j++) {
            for (let i = Math.max(0, x - w/2 - margin); i < Math.min(width, x + w/2 + margin); i++) {
                // Skip interior of chamber
                if (i >= x - w/2 && i < x + w/2 && j >= y - h/2 && j < y + h/2) continue;
                
                totalTemp += this.temperature[j][i];
                count++;
            }
        }
        
        return count > 0 ? totalTemp / count : this.thermalProperties.ambientTemperature;
    }
    
    getSolidusTemperature(composition) {
        const solidusTemps = {
            'basaltic': 1000,
            'andesitic': 950,
            'dacitic': 850,
            'rhyolitic': 750
        };
        
        return solidusTemps[composition] || 950;
    }
    
    applyChamberTemperature(chamber) {
        const { width, height } = this.gridSize;
        const x = Math.floor(chamber.position.x);
        const y = Math.floor(chamber.position.y);
        const w = chamber.dimensions.width;
        const h = chamber.dimensions.height;
        
        for (let j = Math.max(0, y - h/2); j < Math.min(height, y + h/2); j++) {
            for (let i = Math.max(0, x - w/2); i < Math.min(width, x + w/2); i++) {
                // Gradual temperature transition at edges
                const distFromEdge = Math.min(
                    i - (x - w/2), (x + w/2) - i,
                    j - (y - h/2), (y + h/2) - j
                );
                const edgeFactor = Math.min(1, distFromEdge / 2);
                
                const targetTemp = chamber.temperature * edgeFactor + 
                                 this.temperature[j][i] * (1 - edgeFactor);
                
                this.temperature[j][i] = targetTemp;
                
                // Update thermal diffusivity for magma
                if (edgeFactor > 0.5) {
                    this.thermalDiffusivity[j][i] = this.thermalProperties.magmaThermalDiffusivity;
                }
            }
        }
    }
    
    updateChamberEvolutionDisplay() {
        const historyDiv = this.container.querySelector('#chamber-history');
        
        let html = '';
        this.magmaChambers.forEach(chamber => {
            const recentHistory = chamber.thermalHistory.slice(-1)[0];
            if (recentHistory) {
                const age = Math.floor(this.currentTime / (3600 * 24));
                html += `<div style="margin-bottom: 5px;">
                    <strong>${chamber.id}:</strong> ${recentHistory.temperature.toFixed(0)}°C, 
                    ${(recentHistory.crystallinity * 100).toFixed(1)}% crystalized, 
                    ${(recentHistory.volume / 1e9).toFixed(1)} km³
                </div>`;
            }
        });
        
        historyDiv.innerHTML = html;
    }
    
    detectThermalAnomalies() {
        const { width, height } = this.gridSize;
        this.thermalAnomalies = [];
        
        // Detect temperature anomalies (areas significantly warmer than background)
        for (let j = 5; j < height - 5; j++) {
            for (let i = 5; i < width - 5; i++) {
                const centerTemp = this.temperature[j][i];
                
                // Calculate local average
                let localSum = 0;
                let localCount = 0;
                const radius = 5;
                
                for (let dy = -radius; dy <= radius; dy++) {
                    for (let dx = -radius; dx <= radius; dx++) {
                        const nx = i + dx;
                        const ny = j + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && (dx !== 0 || dy !== 0)) {
                            localSum += this.temperature[ny][nx];
                            localCount++;
                        }
                    }
                }
                
                const localAvg = localSum / localCount;
                const anomaly = centerTemp - localAvg;
                
                // Significant positive thermal anomaly
                if (anomaly > 50) { // 50°C above local average
                    this.thermalAnomalies.push({
                        position: { x: i, y: j },
                        intensity: anomaly,
                        temperature: centerTemp,
                        type: 'hot_spot'
                    });
                }
            }
        }
        
        // Remove overlapping anomalies (keep strongest)
        this.thermalAnomalies.sort((a, b) => b.intensity - a.intensity);
        this.thermalAnomalies = this.thermalAnomalies.filter((anomaly, index) => {
            for (let i = 0; i < index; i++) {
                const distance = Math.sqrt(
                    (anomaly.position.x - this.thermalAnomalies[i].position.x)**2 +
                    (anomaly.position.y - this.thermalAnomalies[i].position.y)**2
                );
                if (distance < 10) return false; // Too close to existing anomaly
            }
            return true;
        });
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Render main thermal field
        this.renderThermalField();
        
        // Render overlays
        if (this.showIsotherms) this.renderIsotherms();
        if (this.showHeatSources) this.renderHeatSources();
        if (this.showMagmaChambers) this.renderMagmaChambers();
        if (this.showAnomalies) this.renderThermalAnomalies();
    }
    
    renderThermalField() {
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        const imageData = this.ctx.createImageData(canvasWidth, canvasHeight);
        const data = imageData.data;
        
        for (let j = 0; j < canvasHeight; j++) {
            for (let i = 0; i < canvasWidth; i++) {
                const gridX = Math.floor(i / canvasWidth * width);
                const gridY = Math.floor(j / canvasHeight * height);
                
                let value;
                switch (this.currentVisualization) {
                    case 'temperature':
                        value = this.temperature[gridY][gridX];
                        break;
                    case 'heat_flow':
                        value = this.calculateHeatFlow(gridX, gridY);
                        break;
                    case 'thermal_gradient':
                        value = this.calculateThermalGradient(gridX, gridY);
                        break;
                    case 'diffusivity':
                        value = this.thermalDiffusivity[gridY][gridX] * 1e6; // Convert to mm²/s
                        break;
                }
                
                const color = this.temperatureToColor(value);
                const index = (j * canvasWidth + i) * 4;
                
                data[index] = color[0];     // R
                data[index + 1] = color[1]; // G
                data[index + 2] = color[2]; // B
                data[index + 3] = 255;      // A
            }
        }
        
        this.ctx.putImageData(imageData, 0, 0);
    }
    
    calculateHeatFlow(x, y) {
        const { width, height } = this.gridSize;
        if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return 0;
        
        const dx = this.physicalSize.width / width;
        const dy = this.physicalSize.height / height;
        
        // Heat flow = -k * ∇T
        const dT_dx = (this.temperature[y][x+1] - this.temperature[y][x-1]) / (2 * dx);
        const dT_dy = (this.temperature[y+1][x] - this.temperature[y-1][x]) / (2 * dy);
        
        const conductivity = this.thermalProperties.rockThermalConductivity;
        return conductivity * Math.sqrt(dT_dx*dT_dx + dT_dy*dT_dy);
    }
    
    calculateThermalGradient(x, y) {
        const { width, height } = this.gridSize;
        if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return 0;
        
        const dy = this.physicalSize.height / height;
        const dT_dy = (this.temperature[y+1][x] - this.temperature[y-1][x]) / (2 * dy);
        
        return Math.abs(dT_dy); // °C/m
    }
    
    temperatureToColor(temp) {
        // Clamp temperature to visualization range
        const clampedTemp = Math.max(0, Math.min(1200, temp));
        
        // Find appropriate color in scheme
        for (let i = 0; i < this.temperatureColorScheme.length - 1; i++) {
            const current = this.temperatureColorScheme[i];
            const next = this.temperatureColorScheme[i + 1];
            
            if (clampedTemp >= current.temp && clampedTemp <= next.temp) {
                const factor = (clampedTemp - current.temp) / (next.temp - current.temp);
                return [
                    Math.round(current.color[0] * (1 - factor) + next.color[0] * factor),
                    Math.round(current.color[1] * (1 - factor) + next.color[1] * factor),
                    Math.round(current.color[2] * (1 - factor) + next.color[2] * factor)
                ];
            }
        }
        
        return this.temperatureColorScheme[this.temperatureColorScheme.length - 1].color;
    }
    
    renderIsotherms() {
        const isotherms = [100, 200, 400, 600, 800, 1000];
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        this.ctx.lineWidth = 1;
        this.ctx.font = '10px Arial';
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        
        isotherms.forEach(isotherm => {
            this.ctx.beginPath();
            
            // Simple isotherm tracing
            for (let j = 1; j < height - 1; j++) {
                for (let i = 1; i < width - 1; i++) {
                    const temp = this.temperature[j][i];
                    
                    if (Math.abs(temp - isotherm) < 10) { // Within 10°C
                        const x = (i / width) * canvasWidth;
                        const y = (j / height) * canvasHeight;
                        
                        this.ctx.fillRect(x, y, 2, 2);
                        
                        // Label occasionally
                        if (i % 20 === 0 && j % 15 === 0) {
                            this.ctx.fillText(`${isotherm}°C`, x + 3, y - 3);
                        }
                    }
                }
            }
        });
    }
    
    renderHeatSources() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.heatSources.forEach(source => {
            if (source.active) {
                const x = (source.position.x / this.gridSize.width) * canvasWidth;
                const y = (source.position.y / this.gridSize.height) * canvasHeight;
                
                switch (source.type) {
                    case 'point':
                        this.ctx.fillStyle = '#FF6600';
                        this.ctx.beginPath();
                        this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
                        this.ctx.fill();
                        
                        this.ctx.strokeStyle = '#FF3300';
                        this.ctx.lineWidth = 2;
                        this.ctx.stroke();
                        break;
                        
                    case 'rectangular':
                        const w = (source.dimensions.width / this.gridSize.width) * canvasWidth;
                        const h = (source.dimensions.height / this.gridSize.height) * canvasHeight;
                        
                        this.ctx.fillStyle = 'rgba(255, 102, 0, 0.3)';
                        this.ctx.fillRect(x - w/2, y - h/2, w, h);
                        
                        this.ctx.strokeStyle = '#FF6600';
                        this.ctx.lineWidth = 2;
                        this.ctx.strokeRect(x - w/2, y - h/2, w, h);
                        break;
                        
                    case 'distributed':
                        const radius = (source.radius / this.gridSize.width) * canvasWidth;
                        
                        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius);
                        gradient.addColorStop(0, 'rgba(255, 102, 0, 0.4)');
                        gradient.addColorStop(1, 'rgba(255, 102, 0, 0.1)');
                        
                        this.ctx.fillStyle = gradient;
                        this.ctx.beginPath();
                        this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
                        this.ctx.fill();
                        break;
                }
                
                // Label
                this.ctx.fillStyle = 'white';
                this.ctx.font = 'bold 11px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(source.id.replace('_', ' '), x, y - 15);
            }
        });
    }
    
    renderMagmaChambers() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.magmaChambers.forEach(chamber => {
            if (chamber.active) {
                const x = (chamber.position.x / this.gridSize.width) * canvasWidth;
                const y = (chamber.position.y / this.gridSize.height) * canvasHeight;
                const w = (chamber.dimensions.width / this.gridSize.width) * canvasWidth;
                const h = (chamber.dimensions.height / this.gridSize.height) * canvasHeight;
                
                // Chamber outline
                this.ctx.strokeStyle = '#FFD700';
                this.ctx.lineWidth = 3;
                this.ctx.strokeRect(x - w/2, y - h/2, w, h);
                
                // Chamber fill based on crystallinity
                const alpha = 0.3 * (1 - chamber.crystallinity);
                this.ctx.fillStyle = `rgba(255, 215, 0, ${alpha})`;
                this.ctx.fillRect(x - w/2, y - h/2, w, h);
                
                // Chamber info
                this.ctx.fillStyle = 'white';
                this.ctx.font = 'bold 12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(chamber.id.replace('_', ' '), x, y - h/2 - 10);
                
                this.ctx.font = '10px Arial';
                this.ctx.fillText(`${chamber.temperature.toFixed(0)}°C`, x, y);
                this.ctx.fillText(`${(chamber.crystallinity * 100).toFixed(0)}% cryst.`, x, y + 12);
            }
        });
    }
    
    renderThermalAnomalies() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.thermalAnomalies.forEach(anomaly => {
            const x = (anomaly.position.x / this.gridSize.width) * canvasWidth;
            const y = (anomaly.position.y / this.gridSize.height) * canvasHeight;
            const radius = Math.min(15, anomaly.intensity / 10);
            
            // Pulsing effect
            const pulsePhase = (this.currentTime / 1000) % (2 * Math.PI);
            const pulseRadius = radius * (1 + 0.3 * Math.sin(pulsePhase * 3));
            
            this.ctx.strokeStyle = '#FF0066';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(x, y, pulseRadius, 0, 2 * Math.PI);
            this.ctx.stroke();
            
            // Anomaly intensity indicator
            this.ctx.fillStyle = 'rgba(255, 0, 102, 0.6)';
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Label
            this.ctx.fillStyle = '#FF0066';
            this.ctx.font = 'bold 9px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`+${anomaly.intensity.toFixed(0)}°C`, x, y + pulseRadius + 12);
        });
    }
    
    updateStatistics() {
        const { width, height } = this.gridSize;
        let maxTemp = -Infinity;
        let minTemp = Infinity;
        let totalTemp = 0;
        
        for (let j = 0; j < height; j++) {
            for (let i = 0; i < width; i++) {
                const temp = this.temperature[j][i];
                maxTemp = Math.max(maxTemp, temp);
                minTemp = Math.min(minTemp, temp);
                totalTemp += temp;
            }
        }
        
        const avgTemp = totalTemp / (width * height);
        
        const statsDiv = this.container.querySelector('#thermal-stats');
        statsDiv.innerHTML = `
            Max Temp: ${maxTemp.toFixed(0)}°C<br>
            Min Temp: ${minTemp.toFixed(0)}°C<br>
            Avg Temp: ${avgTemp.toFixed(0)}°C<br>
            Heat Sources: ${this.heatSources.filter(s => s.active).length}<br>
            Magma Chambers: ${this.magmaChambers.filter(c => c.active).length}<br>
            Anomalies: ${this.thermalAnomalies.length}
        `;
    }
    
    resetModel() {
        this.stopSimulation();
        this.currentTime = 0;
        this.initializeFields();
        this.render();
        
        const playPauseBtn = this.container.querySelector('#thermal-play-pause');
        playPauseBtn.innerHTML = '▶️ Start';
        playPauseBtn.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
        
        const timeDisplay = this.container.querySelector('#thermal-time');
        timeDisplay.textContent = '0 hours';
        
        const status = this.container.querySelector('#thermal-status');
        status.textContent = 'Model reset';
    }
    
    takeSnapshot() {
        const canvas = this.canvas;
        const link = document.createElement('a');
        link.download = `thermal_model_snapshot_${Math.floor(this.currentTime / 3600)}h.png`;
        link.href = canvas.toDataURL();
        link.click();
    }
    
    makeDraggable() {
        const header = this.container.querySelector('.thermal-header');
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
            
            this.container.style.left = Math.max(0, Math.min(window.innerWidth - 800, newLeft)) + 'px';
            this.container.style.top = Math.max(0, Math.min(window.innerHeight - 700, newTop)) + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }
    
    show() {
        this.container.style.display = 'block';
        this.container.style.animation = 'fadeIn 0.3s ease-out';
        this.render();
    }
    
    hide() {
        this.container.style.animation = 'fadeOut 0.3s ease-in';
        setTimeout(() => {
            this.container.style.display = 'none';
            this.stopSimulation();
        }, 300);
    }
}

// Add CSS animations
const thermalStyle = document.createElement('style');
thermalStyle.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.95); }
    }
    
    #thermal-model-panel select option {
        background: #c0392b;
        color: white;
    }
    
    #thermal-model-panel::-webkit-scrollbar {
        width: 8px;
    }
    
    #thermal-model-panel::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    #thermal-model-panel::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    
    #thermal-model-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
`;
document.head.appendChild(thermalStyle);

// Global instance
window.thermalModel = new ThermalModel();

// Integration with mantle convection model
if (window.mantleConvectionModel) {
    window.mantleConvectionModel.thermalModel = window.thermalModel;
}

// Keyboard shortcut to toggle thermal model
document.addEventListener('keydown', (e) => {
    if (e.key === 'T' && e.altKey && e.ctrlKey) {
        e.preventDefault();
        if (window.thermalModel.container.style.display === 'none') {
            window.thermalModel.show();
        } else {
            window.thermalModel.hide();
        }
    }
});

console.log('🔥 Advanced Thermal Model loaded successfully!');
console.log('🔑 Press Ctrl+Alt+T to toggle thermal model');
console.log('🎯 Access via: window.thermalModel.show()');
console.log('🌡️ Features: Heat diffusion, thermal boundary conditions, magma chamber cooling, anomaly detection');