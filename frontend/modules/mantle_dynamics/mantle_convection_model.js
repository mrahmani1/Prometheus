/**
 * ASPECT-inspired Mantle Convection Model
 * Implements convection cell visualization, temperature fields, viscosity variations,
 * plate motion vectors, and hotspot tracking
 */

class MantleConvectionModel {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;
        this.isRunning = false;
        
        // Model parameters
        this.gridSize = { width: 200, height: 100 };
        this.physicalDimensions = { width: 6371, height: 2890 }; // Earth's radius and mantle thickness (km)
        this.timeStep = 0.001; // Myr
        this.currentTime = 0;
        
        // Physical constants
        this.constants = {
            gravity: 9.81, // m/s²
            thermalExpansivity: 3e-5, // 1/K
            thermalDiffusivity: 1e-6, // m²/s
            referenceViscosity: 1e21, // Pa·s
            mantleDepth: 2890e3, // m
            surfaceTemperature: 300, // K
            coreTemperature: 4000, // K
            rayleighNumber: 1e6
        };
        
        // Model fields
        this.temperature = null;
        this.velocity = null;
        this.viscosity = null;
        this.pressure = null;
        this.convectionCells = [];
        this.hotspots = [];
        this.plateMotions = [];
        
        // Visualization settings
        this.colorSchemes = {
            temperature: {
                name: 'Temperature (K)',
                colors: ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#800000'],
                range: [300, 4000]
            },
            velocity: {
                name: 'Velocity (cm/yr)',
                colors: ['#FFFFFF', '#E0E0FF', '#C0C0FF', '#8080FF', '#4040FF', '#0000FF'],
                range: [0, 10]
            },
            viscosity: {
                name: 'Viscosity (log Pa·s)',
                colors: ['#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00', '#00FF80', '#00FFFF'],
                range: [18, 25]
            }
        };
        
        this.currentVisualization = 'temperature';
        this.showVelocityVectors = true;
        this.showConvectionCells = true;
        this.showHotspots = true;
        this.showPlateVectors = true;
        
        this.initializeFields();
        this.createInterface();
        this.setupControls();
        
        console.log('🌍 ASPECT-inspired Mantle Convection Model initialized');
        console.log('🔥 Temperature field modeling with convection cells');
    }
    
    initializeFields() {
        const { width, height } = this.gridSize;
        
        // Initialize temperature field with geotherm
        this.temperature = Array(height).fill().map(() => Array(width).fill(0));
        this.velocity = Array(height).fill().map(() => Array(width).fill({ vx: 0, vy: 0 }));
        this.viscosity = Array(height).fill().map(() => Array(width).fill(0));
        this.pressure = Array(height).fill().map(() => Array(width).fill(0));
        
        // Set initial temperature distribution (geotherm)
        for (let j = 0; j < height; j++) {
            const depth = (j / (height - 1)) * this.constants.mantleDepth;
            const baseTemp = this.constants.surfaceTemperature + 
                            (this.constants.coreTemperature - this.constants.surfaceTemperature) * (depth / this.constants.mantleDepth);
            
            for (let i = 0; i < width; i++) {
                // Add thermal perturbations for convection initiation
                const perturbation = 100 * Math.sin(2 * Math.PI * i / 20) * Math.sin(Math.PI * j / 10);
                this.temperature[j][i] = baseTemp + perturbation;
                
                // Initialize viscosity (temperature-dependent)
                this.viscosity[j][i] = this.calculateViscosity(this.temperature[j][i], depth);
                
                // Initialize pressure (lithostatic)
                this.pressure[j][i] = 3300 * 9.81 * depth; // ρgh
            }
        }
        
        // Initialize convection cells
        this.initializeConvectionCells();
        
        // Initialize hotspots
        this.initializeHotspots();
        
        // Initialize plate motions
        this.initializePlateMotions();
    }
    
    initializeConvectionCells() {
        this.convectionCells = [
            {
                id: 'cell_1',
                center: { x: 50, y: 40 },
                radius: 25,
                circulation: 'clockwise',
                strength: 5.0,
                age: 0,
                active: true
            },
            {
                id: 'cell_2',
                center: { x: 100, y: 35 },
                radius: 30,
                circulation: 'counterclockwise',
                strength: 6.2,
                age: 0,
                active: true
            },
            {
                id: 'cell_3',
                center: { x: 150, y: 45 },
                radius: 28,
                circulation: 'clockwise',
                strength: 4.8,
                age: 0,
                active: true
            }
        ];
    }
    
    initializeHotspots() {
        this.hotspots = [
            {
                id: 'hawaii',
                position: { x: 75, y: 85 },
                temperature: 4200,
                intensity: 8.5,
                age: 70, // Myr
                track: [],
                active: true,
                plumeDiameter: 200 // km
            },
            {
                id: 'yellowstone',
                position: { x: 45, y: 80 },
                temperature: 4000,
                intensity: 7.2,
                age: 16,
                track: [],
                active: true,
                plumeDiameter: 150
            },
            {
                id: 'reunion',
                position: { x: 125, y: 88 },
                temperature: 4100,
                intensity: 8.0,
                age: 65,
                track: [],
                active: true,
                plumeDiameter: 180
            }
        ];
        
        // Initialize hotspot tracks
        this.hotspots.forEach(hotspot => {
            for (let t = 0; t < hotspot.age; t += 1) {
                const trackX = hotspot.position.x - t * 0.5; // Simplified plate motion
                const trackY = hotspot.position.y + Math.sin(t * 0.1) * 2;
                hotspot.track.push({ x: trackX, y: trackY, age: t });
            }
        });
    }
    
    initializePlateMotions() {
        this.plateMotions = [
            {
                name: 'Pacific Plate',
                region: { x1: 0, y1: 0, x2: 80, y2: 20 },
                velocity: { vx: 8.5, vy: -1.2 }, // cm/yr
                color: '#FF6B6B'
            },
            {
                name: 'North American Plate',
                region: { x1: 40, y1: 0, x2: 120, y2: 25 },
                velocity: { vx: -2.3, vy: 0.5 },
                color: '#4ECDC4'
            },
            {
                name: 'Eurasian Plate',
                region: { x1: 100, y1: 0, x2: 200, y2: 30 },
                velocity: { vx: 1.8, vy: 1.1 },
                color: '#45B7D1'
            },
            {
                name: 'African Plate',
                region: { x1: 120, y1: 15, x2: 180, y2: 60 },
                velocity: { vx: 2.5, vy: 1.8 },
                color: '#96CEB4'
            }
        ];
    }
    
    calculateViscosity(temperature, depth) {
        // Temperature and pressure dependent viscosity
        const activationEnergy = 540000; // J/mol
        const gasConstant = 8.314; // J/mol/K
        const referenceTemp = 1600; // K
        const pressureDependence = Math.exp(depth * 1e-5);
        
        const viscosity = this.constants.referenceViscosity * 
                         Math.exp(activationEnergy / gasConstant * (1/temperature - 1/referenceTemp)) *
                         pressureDependence;
                         
        return Math.max(1e18, Math.min(1e26, viscosity));
    }
    
    createInterface() {
        this.container = document.createElement('div');
        this.container.id = 'mantle-convection-panel';
        this.container.style.cssText = `
            position: fixed;
            top: 50px;
            left: 50px;
            width: 900px;
            height: 750px;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            z-index: 9998;
            display: none;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        `;
        
        this.container.innerHTML = `
            <div class="convection-header" style="
                padding: 20px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            ">
                <h3 style="margin: 0; font-size: 20px; font-weight: 600;">
                    🌍 Mantle Convection Model (ASPECT-inspired)
                </h3>
                <div class="header-controls">
                    <button id="export-model" style="
                        background: rgba(52,152,219,0.2);
                        border: 1px solid #3498db;
                        color: #3498db;
                        padding: 8px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        cursor: pointer;
                        margin-right: 10px;
                    ">💾 Export</button>
                    <button id="close-convection" style="
                        background: rgba(231,76,60,0.2);
                        border: 1px solid #e74c3c;
                        color: #e74c3c;
                        padding: 8px 12px;
                        border-radius: 15px;
                        cursor: pointer;
                    ">×</button>
                </div>
            </div>
            
            <div class="model-controls" style="
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
                    <select id="visualization-type" style="
                        padding: 6px 10px;
                        background: rgba(255,255,255,0.1);
                        border: 1px solid rgba(255,255,255,0.3);
                        border-radius: 6px;
                        color: white;
                        font-size: 12px;
                    ">
                        <option value="temperature">🌡️ Temperature</option>
                        <option value="velocity">🌊 Velocity</option>
                        <option value="viscosity">🥞 Viscosity</option>
                    </select>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-vectors" checked style="margin-right: 5px;">
                        Velocity Vectors
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-cells" checked style="margin-right: 5px;">
                        Convection Cells
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-hotspots" checked style="margin-right: 5px;">
                        Hotspots
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-plates" checked style="margin-right: 5px;">
                        Plate Vectors
                    </label>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <button id="play-pause" style="
                        background: linear-gradient(45deg, #27ae60, #2ecc71);
                        border: none;
                        color: white;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        font-weight: 600;
                    ">▶️ Start</button>
                    
                    <button id="reset-model" style="
                        background: rgba(241,196,15,0.2);
                        border: 1px solid #f1c40f;
                        color: #f1c40f;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                    ">🔄 Reset</button>
                </div>
                
                <div style="flex: 1; text-align: right; font-size: 12px; color: #bdc3c7;">
                    Time: <span id="model-time">0.000 Myr</span>
                </div>
            </div>
            
            <div class="visualization-area" style="
                position: relative;
                height: calc(100% - 140px);
                padding: 20px;
            ">
                <canvas id="convection-canvas" style="
                    width: 100%;
                    height: 480px;
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 10px;
                    background: #1a252f;
                    cursor: crosshair;
                "></canvas>
                
                <div id="colorbar" style="
                    position: absolute;
                    right: 40px;
                    top: 40px;
                    width: 30px;
                    height: 400px;
                    border: 1px solid rgba(255,255,255,0.3);
                    border-radius: 15px;
                    overflow: hidden;
                "></div>
                
                <div id="colorbar-labels" style="
                    position: absolute;
                    right: 75px;
                    top: 40px;
                    height: 400px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    font-size: 11px;
                    color: #bdc3c7;
                "></div>
                
                <div id="model-info" style="
                    position: absolute;
                    left: 40px;
                    bottom: 40px;
                    background: rgba(0,0,0,0.7);
                    padding: 15px;
                    border-radius: 10px;
                    font-size: 12px;
                    min-width: 200px;
                ">
                    <div><strong>Model Parameters:</strong></div>
                    <div>Ra: ${this.constants.rayleighNumber.toExponential(1)}</div>
                    <div>Grid: ${this.gridSize.width}×${this.gridSize.height}</div>
                    <div>Depth: ${this.constants.mantleDepth/1000} km</div>
                    <div id="cursor-info"></div>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.container);
        this.canvas = this.container.querySelector('#convection-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 480;
        
        this.createColorbar();
    }
    
    createColorbar() {
        const colorbar = this.container.querySelector('#colorbar');
        const labels = this.container.querySelector('#colorbar-labels');
        
        const updateColorbar = () => {
            const scheme = this.colorSchemes[this.currentVisualization];
            
            // Create gradient
            const gradient = this.ctx.createLinearGradient(0, 0, 0, 400);
            scheme.colors.forEach((color, i) => {
                gradient.addColorStop(i / (scheme.colors.length - 1), color);
            });
            
            colorbar.style.background = `linear-gradient(to bottom, ${scheme.colors.join(', ')})`;
            
            // Update labels
            labels.innerHTML = '';
            const numLabels = 5;
            for (let i = 0; i < numLabels; i++) {
                const value = scheme.range[1] - (i / (numLabels - 1)) * (scheme.range[1] - scheme.range[0]);
                const label = document.createElement('div');
                label.textContent = value.toFixed(0);
                labels.appendChild(label);
            }
            
            // Add title
            const title = document.createElement('div');
            title.style.cssText = `
                position: absolute;
                right: 110px;
                top: 20px;
                transform: rotate(-90deg);
                transform-origin: center;
                font-size: 12px;
                color: #bdc3c7;
                white-space: nowrap;
            `;
            title.textContent = scheme.name;
            this.container.querySelector('.visualization-area').appendChild(title);
        };
        
        updateColorbar();
        this.updateColorbar = updateColorbar;
    }
    
    setupControls() {
        const closeBtn = this.container.querySelector('#close-convection');
        const playPauseBtn = this.container.querySelector('#play-pause');
        const resetBtn = this.container.querySelector('#reset-model');
        const exportBtn = this.container.querySelector('#export-model');
        const visualizationSelect = this.container.querySelector('#visualization-type');
        
        const showVectorsCheck = this.container.querySelector('#show-vectors');
        const showCellsCheck = this.container.querySelector('#show-cells');
        const showHotspotsCheck = this.container.querySelector('#show-hotspots');
        const showPlatesCheck = this.container.querySelector('#show-plates');
        
        closeBtn.addEventListener('click', () => this.hide());
        playPauseBtn.addEventListener('click', () => this.toggleSimulation());
        resetBtn.addEventListener('click', () => this.resetModel());
        exportBtn.addEventListener('click', () => this.exportModel());
        
        visualizationSelect.addEventListener('change', (e) => {
            this.currentVisualization = e.target.value;
            this.updateColorbar();
            this.render();
        });
        
        showVectorsCheck.addEventListener('change', (e) => {
            this.showVelocityVectors = e.target.checked;
            this.render();
        });
        
        showCellsCheck.addEventListener('change', (e) => {
            this.showConvectionCells = e.target.checked;
            this.render();
        });
        
        showHotspotsCheck.addEventListener('change', (e) => {
            this.showHotspots = e.target.checked;
            this.render();
        });
        
        showPlatesCheck.addEventListener('change', (e) => {
            this.showPlateVectors = e.target.checked;
            this.render();
        });
        
        // Mouse interaction for model info
        this.canvas.addEventListener('mousemove', (e) => this.updateCursorInfo(e));
        
        this.makeDraggable();
    }
    
    updateCursorInfo(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * this.gridSize.width);
        const y = Math.floor((e.clientY - rect.top) / rect.height * this.gridSize.height);
        
        if (x >= 0 && x < this.gridSize.width && y >= 0 && y < this.gridSize.height) {
            const temp = this.temperature[y][x];
            const visc = this.viscosity[y][x];
            const vel = this.velocity[y][x];
            const depth = (y / this.gridSize.height) * this.constants.mantleDepth / 1000;
            
            const cursorInfo = this.container.querySelector('#cursor-info');
            cursorInfo.innerHTML = `
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                    <strong>Point (${x}, ${y}):</strong><br>
                    Depth: ${depth.toFixed(0)} km<br>
                    Temp: ${temp.toFixed(0)} K<br>
                    Viscosity: ${(Math.log10(visc)).toFixed(1)} log Pa·s<br>
                    Velocity: ${(Math.sqrt(vel.vx*vel.vx + vel.vy*vel.vy)).toFixed(2)} cm/yr
                </div>
            `;
        }
    }
    
    toggleSimulation() {
        const btn = this.container.querySelector('#play-pause');
        
        if (this.isRunning) {
            this.stopSimulation();
            btn.innerHTML = '▶️ Start';
            btn.style.background = 'linear-gradient(45deg, #27ae60, #2ecc71)';
        } else {
            this.startSimulation();
            btn.innerHTML = '⏸️ Pause';
            btn.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
        }
    }
    
    startSimulation() {
        this.isRunning = true;
        this.animate();
    }
    
    stopSimulation() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    animate() {
        if (!this.isRunning) return;
        
        this.updateModel();
        this.render();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    updateModel() {
        const dt = this.timeStep;
        this.currentTime += dt;
        
        // Update temperature field (simplified advection-diffusion)
        this.updateTemperatureField(dt);
        
        // Update velocity field (simplified Stokes flow)
        this.updateVelocityField(dt);
        
        // Update viscosity based on new temperature
        this.updateViscosityField();
        
        // Update convection cells
        this.updateConvectionCells(dt);
        
        // Update hotspots
        this.updateHotspots(dt);
        
        // Update time display
        const timeDisplay = this.container.querySelector('#model-time');
        timeDisplay.textContent = `${this.currentTime.toFixed(3)} Myr`;
    }
    
    updateTemperatureField(dt) {
        const { width, height } = this.gridSize;
        const newTemp = Array(height).fill().map(() => Array(width).fill(0));
        const kappa = this.constants.thermalDiffusivity;
        const dx = this.physicalDimensions.width / width * 1000; // m
        const dy = this.physicalDimensions.height / height * 1000; // m
        
        for (let j = 1; j < height - 1; j++) {
            for (let i = 1; i < width - 1; i++) {
                const T = this.temperature[j][i];
                const vel = this.velocity[j][i];
                
                // Diffusion term (simplified)
                const diffusion = kappa * dt * (
                    (this.temperature[j][i+1] - 2*T + this.temperature[j][i-1]) / (dx*dx) +
                    (this.temperature[j+1][i] - 2*T + this.temperature[j-1][i]) / (dy*dy)
                );
                
                // Advection term (simplified)
                const advection = -dt * (
                    vel.vx * (this.temperature[j][i+1] - this.temperature[j][i-1]) / (2*dx) +
                    vel.vy * (this.temperature[j+1][i] - this.temperature[j-1][i]) / (2*dy)
                );
                
                newTemp[j][i] = T + diffusion + advection;
                
                // Apply thermal boundary conditions
                if (j === 0) newTemp[j][i] = this.constants.surfaceTemperature;
                if (j === height - 1) newTemp[j][i] = this.constants.coreTemperature;
            }
        }
        
        // Apply hotspot heating
        this.hotspots.forEach(hotspot => {
            if (hotspot.active) {
                const hx = Math.floor(hotspot.position.x);
                const hy = Math.floor(hotspot.position.y);
                const radius = hotspot.plumeDiameter / (this.physicalDimensions.width / width);
                
                for (let j = Math.max(0, hy - radius); j < Math.min(height, hy + radius); j++) {
                    for (let i = Math.max(0, hx - radius); i < Math.min(width, hx + radius); i++) {
                        const dist = Math.sqrt((i - hx)**2 + (j - hy)**2);
                        if (dist < radius) {
                            const heating = hotspot.intensity * Math.exp(-dist / radius * 3);
                            newTemp[j][i] = Math.min(hotspot.temperature, newTemp[j][i] + heating * dt * 1e6);
                        }
                    }
                }
            }
        });
        
        this.temperature = newTemp;
    }
    
    updateVelocityField(dt) {
        const { width, height } = this.gridSize;
        
        // Simplified velocity calculation based on convection cells
        for (let j = 0; j < height; j++) {
            for (let i = 0; i < width; i++) {
                let vx = 0, vy = 0;
                
                // Contribution from convection cells
                this.convectionCells.forEach(cell => {
                    if (cell.active) {
                        const dx = i - cell.center.x;
                        const dy = j - cell.center.y;
                        const distance = Math.sqrt(dx*dx + dy*dy);
                        
                        if (distance < cell.radius) {
                            const strength = cell.strength * (1 - distance / cell.radius);
                            const angle = Math.atan2(dy, dx);
                            
                            if (cell.circulation === 'clockwise') {
                                vx += strength * Math.sin(angle);
                                vy += -strength * Math.cos(angle);
                            } else {
                                vx += -strength * Math.sin(angle);
                                vy += strength * Math.cos(angle);
                            }
                        }
                    }
                });
                
                // Add thermal buoyancy
                const temp = this.temperature[j][i];
                const refTemp = this.constants.surfaceTemperature + 
                              (this.constants.coreTemperature - this.constants.surfaceTemperature) * 
                              (j / height);
                const buoyancy = this.constants.thermalExpansivity * (temp - refTemp);
                vy += buoyancy * this.constants.gravity * dt * 1e6; // Convert to cm/yr
                
                this.velocity[j][i] = { vx: vx, vy: vy };
            }
        }
    }
    
    updateViscosityField() {
        const { width, height } = this.gridSize;
        
        for (let j = 0; j < height; j++) {
            const depth = (j / height) * this.constants.mantleDepth;
            for (let i = 0; i < width; i++) {
                this.viscosity[j][i] = this.calculateViscosity(this.temperature[j][i], depth);
            }
        }
    }
    
    updateConvectionCells(dt) {
        this.convectionCells.forEach(cell => {
            if (cell.active) {
                cell.age += dt;
                
                // Cell evolution (simplified)
                const tempVariation = this.getTemperatureVariation(cell.center);
                cell.strength = Math.max(0, cell.strength + tempVariation * dt * 100);
                
                // Cell migration (very simplified)
                if (Math.random() < 0.001) {
                    cell.center.x += (Math.random() - 0.5) * 2;
                    cell.center.y += (Math.random() - 0.5);
                }
                
                // Cell deactivation
                if (cell.strength < 0.5) {
                    cell.active = false;
                }
            }
        });
        
        // Occasionally spawn new cells
        if (Math.random() < 0.0001 && this.convectionCells.filter(c => c.active).length < 5) {
            this.spawnNewConvectionCell();
        }
    }
    
    getTemperatureVariation(center) {
        const { width, height } = this.gridSize;
        const x = Math.floor(center.x);
        const y = Math.floor(center.y);
        
        if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return 0;
        
        const centerTemp = this.temperature[y][x];
        let surrounding = 0;
        let count = 0;
        
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                surrounding += this.temperature[y + dy][x + dx];
                count++;
            }
        }
        
        return (centerTemp - surrounding / count) / 100; // Normalized
    }
    
    spawnNewConvectionCell() {
        const { width, height } = this.gridSize;
        const newCell = {
            id: `cell_${Date.now()}`,
            center: { 
                x: Math.random() * (width - 60) + 30, 
                y: Math.random() * (height - 40) + 20 
            },
            radius: 15 + Math.random() * 20,
            circulation: Math.random() > 0.5 ? 'clockwise' : 'counterclockwise',
            strength: Math.random() * 3 + 2,
            age: 0,
            active: true
        };
        
        this.convectionCells.push(newCell);
    }
    
    updateHotspots(dt) {
        this.hotspots.forEach(hotspot => {
            if (hotspot.active) {
                // Update hotspot track
                if (this.currentTime > 0 && this.currentTime % 1 < dt) { // Every 1 Myr
                    hotspot.track.push({
                        x: hotspot.position.x,
                        y: hotspot.position.y,
                        age: this.currentTime
                    });
                    
                    // Limit track length
                    if (hotspot.track.length > hotspot.age) {
                        hotspot.track.shift();
                    }
                }
                
                // Hotspot evolution
                hotspot.intensity *= (1 - dt * 0.001); // Gradual weakening
                
                if (hotspot.intensity < 1.0) {
                    hotspot.active = false;
                }
            }
        });
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Render main field
        this.renderField();
        
        // Render overlays
        if (this.showVelocityVectors) this.renderVelocityVectors();
        if (this.showConvectionCells) this.renderConvectionCells();
        if (this.showHotspots) this.renderHotspots();
        if (this.showPlateVectors) this.renderPlateVectors();
    }
    
    renderField() {
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        const imageData = this.ctx.createImageData(canvasWidth, canvasHeight);
        const data = imageData.data;
        
        const scheme = this.colorSchemes[this.currentVisualization];
        
        for (let j = 0; j < canvasHeight; j++) {
            for (let i = 0; i < canvasWidth; i++) {
                const gridX = Math.floor(i / canvasWidth * width);
                const gridY = Math.floor(j / canvasHeight * height);
                
                let value;
                switch (this.currentVisualization) {
                    case 'temperature':
                        value = this.temperature[gridY][gridX];
                        break;
                    case 'velocity':
                        const vel = this.velocity[gridY][gridX];
                        value = Math.sqrt(vel.vx * vel.vx + vel.vy * vel.vy);
                        break;
                    case 'viscosity':
                        value = Math.log10(this.viscosity[gridY][gridX]);
                        break;
                }
                
                const color = this.valueToColor(value, scheme);
                const index = (j * canvasWidth + i) * 4;
                
                data[index] = color.r;
                data[index + 1] = color.g;
                data[index + 2] = color.b;
                data[index + 3] = 255;
            }
        }
        
        this.ctx.putImageData(imageData, 0, 0);
    }
    
    valueToColor(value, scheme) {
        const normalized = (value - scheme.range[0]) / (scheme.range[1] - scheme.range[0]);
        const clamped = Math.max(0, Math.min(1, normalized));
        
        const colorIndex = clamped * (scheme.colors.length - 1);
        const lowerIndex = Math.floor(colorIndex);
        const upperIndex = Math.ceil(colorIndex);
        const fraction = colorIndex - lowerIndex;
        
        const lowerColor = this.hexToRgb(scheme.colors[lowerIndex]);
        const upperColor = this.hexToRgb(scheme.colors[upperIndex]);
        
        return {
            r: Math.round(lowerColor.r * (1 - fraction) + upperColor.r * fraction),
            g: Math.round(lowerColor.g * (1 - fraction) + upperColor.g * fraction),
            b: Math.round(lowerColor.b * (1 - fraction) + upperColor.b * fraction)
        };
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }
    
    renderVelocityVectors() {
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        const step = 15; // Vector spacing
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.lineWidth = 1;
        
        for (let j = step; j < height; j += step) {
            for (let i = step; i < width; i += step) {
                const vel = this.velocity[j][i];
                const magnitude = Math.sqrt(vel.vx * vel.vx + vel.vy * vel.vy);
                
                if (magnitude > 0.1) {
                    const x = (i / width) * canvasWidth;
                    const y = (j / height) * canvasHeight;
                    const scale = Math.min(10, magnitude * 3);
                    
                    const dx = vel.vx / magnitude * scale;
                    const dy = vel.vy / magnitude * scale;
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(x, y);
                    this.ctx.lineTo(x + dx, y + dy);
                    this.ctx.stroke();
                    
                    // Arrow head
                    const angle = Math.atan2(dy, dx);
                    this.ctx.beginPath();
                    this.ctx.moveTo(x + dx, y + dy);
                    this.ctx.lineTo(x + dx - 3 * Math.cos(angle - 0.3), y + dy - 3 * Math.sin(angle - 0.3));
                    this.ctx.moveTo(x + dx, y + dy);
                    this.ctx.lineTo(x + dx - 3 * Math.cos(angle + 0.3), y + dy - 3 * Math.sin(angle + 0.3));
                    this.ctx.stroke();
                }
            }
        }
    }
    
    renderConvectionCells() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.convectionCells.forEach(cell => {
            if (cell.active) {
                const x = (cell.center.x / this.gridSize.width) * canvasWidth;
                const y = (cell.center.y / this.gridSize.height) * canvasHeight;
                const radius = (cell.radius / this.gridSize.width) * canvasWidth;
                
                // Cell boundary
                this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
                this.ctx.stroke();
                
                // Circulation indicator
                this.ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
                this.ctx.font = '14px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(
                    cell.circulation === 'clockwise' ? '↻' : '↺',
                    x, y + 5
                );
                
                // Cell info
                this.ctx.font = '10px Arial';
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                this.ctx.fillText(
                    `${cell.strength.toFixed(1)}`,
                    x, y - radius - 5
                );
            }
        });
    }
    
    renderHotspots() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.hotspots.forEach(hotspot => {
            if (hotspot.active) {
                const x = (hotspot.position.x / this.gridSize.width) * canvasWidth;
                const y = (hotspot.position.y / this.gridSize.height) * canvasHeight;
                
                // Hotspot plume
                const radius = (hotspot.plumeDiameter / this.physicalDimensions.width) * canvasWidth;
                const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius);
                gradient.addColorStop(0, 'rgba(255, 100, 0, 0.8)');
                gradient.addColorStop(0.5, 'rgba(255, 150, 0, 0.4)');
                gradient.addColorStop(1, 'rgba(255, 200, 0, 0.1)');
                
                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
                this.ctx.fill();
                
                // Hotspot center
                this.ctx.fillStyle = '#FF4500';
                this.ctx.beginPath();
                this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
                this.ctx.fill();
                
                // Hotspot track
                if (hotspot.track.length > 1) {
                    this.ctx.strokeStyle = 'rgba(255, 100, 0, 0.6)';
                    this.ctx.lineWidth = 2;
                    this.ctx.beginPath();
                    
                    hotspot.track.forEach((point, index) => {
                        const trackX = (point.x / this.gridSize.width) * canvasWidth;
                        const trackY = (point.y / this.gridSize.height) * canvasHeight;
                        
                        if (index === 0) {
                            this.ctx.moveTo(trackX, trackY);
                        } else {
                            this.ctx.lineTo(trackX, trackY);
                        }
                    });
                    
                    this.ctx.stroke();
                }
                
                // Label
                this.ctx.fillStyle = 'white';
                this.ctx.font = '12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(hotspot.id.toUpperCase(), x, y - radius - 10);
            }
        });
    }
    
    renderPlateVectors() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.plateMotions.forEach(plate => {
            const centerX = ((plate.region.x1 + plate.region.x2) / 2 / this.gridSize.width) * canvasWidth;
            const centerY = ((plate.region.y1 + plate.region.y2) / 2 / this.gridSize.height) * canvasHeight;
            
            const scale = 5;
            const dx = plate.velocity.vx * scale;
            const dy = plate.velocity.vy * scale;
            
            // Plate vector
            this.ctx.strokeStyle = plate.color;
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(centerX, centerY);
            this.ctx.lineTo(centerX + dx, centerY + dy);
            this.ctx.stroke();
            
            // Arrow head
            const angle = Math.atan2(dy, dx);
            this.ctx.beginPath();
            this.ctx.moveTo(centerX + dx, centerY + dy);
            this.ctx.lineTo(centerX + dx - 8 * Math.cos(angle - 0.3), centerY + dy - 8 * Math.sin(angle - 0.3));
            this.ctx.moveTo(centerX + dx, centerY + dy);
            this.ctx.lineTo(centerX + dx - 8 * Math.cos(angle + 0.3), centerY + dy - 8 * Math.sin(angle + 0.3));
            this.ctx.stroke();
            
            // Plate label
            this.ctx.fillStyle = plate.color;
            this.ctx.font = 'bold 11px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(plate.name, centerX, centerY - 15);
            
            // Velocity label
            const speed = Math.sqrt(plate.velocity.vx**2 + plate.velocity.vy**2);
            this.ctx.font = '9px Arial';
            this.ctx.fillText(`${speed.toFixed(1)} cm/yr`, centerX, centerY + 25);
        });
    }
    
    resetModel() {
        this.stopSimulation();
        this.currentTime = 0;
        this.initializeFields();
        this.render();
        
        const playPauseBtn = this.container.querySelector('#play-pause');
        playPauseBtn.innerHTML = '▶️ Start';
        playPauseBtn.style.background = 'linear-gradient(45deg, #27ae60, #2ecc71)';
        
        const timeDisplay = this.container.querySelector('#model-time');
        timeDisplay.textContent = '0.000 Myr';
    }
    
    exportModel() {
        const modelData = {
            timestamp: new Date().toISOString(),
            currentTime: this.currentTime,
            gridSize: this.gridSize,
            physicalDimensions: this.physicalDimensions,
            temperature: this.temperature,
            velocity: this.velocity,
            viscosity: this.viscosity,
            convectionCells: this.convectionCells,
            hotspots: this.hotspots,
            plateMotions: this.plateMotions,
            constants: this.constants
        };
        
        const blob = new Blob([JSON.stringify(modelData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mantle_convection_model_${this.currentTime.toFixed(3)}Myr.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    makeDraggable() {
        const header = this.container.querySelector('.convection-header');
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
            
            this.container.style.left = Math.max(0, Math.min(window.innerWidth - 900, newLeft)) + 'px';
            this.container.style.top = Math.max(0, Math.min(window.innerHeight - 750, newTop)) + 'px';
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
const mantleStyle = document.createElement('style');
mantleStyle.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.95); }
    }
    
    #mantle-convection-panel select option {
        background: #34495e;
        color: white;
    }
    
    #mantle-convection-panel::-webkit-scrollbar {
        width: 8px;
    }
    
    #mantle-convection-panel::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    #mantle-convection-panel::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    
    #mantle-convection-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
`;
document.head.appendChild(mantleStyle);

// Global instance
window.mantleConvectionModel = new MantleConvectionModel();

// Keyboard shortcut to toggle model
document.addEventListener('keydown', (e) => {
    if (e.key === 'M' && e.altKey && e.ctrlKey) {
        e.preventDefault();
        if (window.mantleConvectionModel.container.style.display === 'none') {
            window.mantleConvectionModel.show();
        } else {
            window.mantleConvectionModel.hide();
        }
    }
});

console.log('🌍 ASPECT-inspired Mantle Convection Model loaded successfully!');
console.log('🔑 Press Ctrl+Alt+M to toggle mantle model');
console.log('🎯 Access via: window.mantleConvectionModel.show()');
console.log('🔥 Features: Convection cells, temperature fields, viscosity, plate motions, hotspots');