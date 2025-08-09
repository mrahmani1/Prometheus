/**
 * Tectonic Stress Visualization System
 * Implements stress tensor visualization, Coulomb stress changes, fault plane solutions,
 * moment tensor beach balls, and strain rate maps
 */

class StressVisualizer {
    constructor() {
        this.container = null;
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;
        this.isRunning = false;
        
        // Model parameters
        this.gridSize = { width: 180, height: 120 };
        this.physicalSize = { width: 500000, height: 300000 }; // 500x300 km
        this.timeStep = 86400 * 30; // 30 days in seconds
        this.currentTime = 0;
        
        // Stress fields
        this.stressTensor = null;
        this.coulombStress = null;
        this.strainRate = null;
        this.faultPlanes = [];
        this.momentTensors = [];
        this.stressHistory = [];
        
        // Visualization settings
        this.currentVisualization = 'stress_tensor';
        this.showFaultPlanes = true;
        this.showBeachBalls = true;
        this.showStrainVectors = true;
        this.showCoulombContours = true;
        
        // Stress color schemes
        this.stressColorSchemes = {
            stress_tensor: {
                name: 'Principal Stress (MPa)',
                colors: ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000'],
                range: [-50, 50]
            },
            coulomb_stress: {
                name: 'Coulomb Stress Change (bars)',
                colors: ['#8B0000', '#FF0000', '#FF8080', '#FFFFFF', '#8080FF', '#0000FF', '#000080'],
                range: [-10, 10]
            },
            strain_rate: {
                name: 'Strain Rate (nanostrains/year)',
                colors: ['#FFFFFF', '#E0E0FF', '#C0C0FF', '#8080FF', '#4040FF', '#0000FF'],
                range: [0, 500]
            }
        };
        
        this.initializeStressFields();
        this.createInterface();
        this.setupControls();
        
        console.log('🗜️ Tectonic Stress Visualizer initialized');
        console.log('📊 Stress tensors, Coulomb stress, and fault analysis');
    }
    
    initializeStressFields() {
        const { width, height } = this.gridSize;
        
        // Initialize stress tensor field (σxx, σyy, σzz, σxy, σxz, σyz)
        this.stressTensor = Array(height).fill().map(() => 
            Array(width).fill().map(() => ({
                σxx: 0, σyy: 0, σzz: 0,
                σxy: 0, σxz: 0, σyz: 0
            }))
        );
        
        // Initialize Coulomb stress field
        this.coulombStress = Array(height).fill().map(() => Array(width).fill(0));
        
        // Initialize strain rate field
        this.strainRate = Array(height).fill().map(() => 
            Array(width).fill().map(() => ({
                exx: 0, eyy: 0, ezz: 0,
                exy: 0, exz: 0, eyz: 0
            }))
        );
        
        // Initialize background stress field
        this.initializeBackgroundStress();
        
        // Initialize fault planes
        this.initializeFaultPlanes();
        
        // Initialize moment tensors (earthquake sources)
        this.initializeMomentTensors();
    }
    
    initializeBackgroundStress() {
        const { width, height } = this.gridSize;
        
        // Regional stress field based on tectonic setting
        for (let j = 0; j < height; j++) {
            for (let i = 0; i < width; i++) {
                const x = (i / width) * this.physicalSize.width;
                const y = (j / height) * this.physicalSize.height;
                
                // Simulate regional stress field
                const depth = y; // Depth in meters
                const lithostatic = 2700 * 9.81 * depth / 1e6; // MPa
                
                // Tectonic stress components
                const tectonicStress = 20 + 10 * Math.sin(2 * Math.PI * i / 50); // Varying tectonic stress
                
                this.stressTensor[j][i] = {
                    σxx: tectonicStress + 5 * Math.cos(x / 50000),
                    σyy: lithostatic + 0.5 * tectonicStress,
                    σzz: lithostatic,
                    σxy: 2 * Math.sin(x / 30000) * Math.cos(y / 40000),
                    σxz: 0.5 * Math.sin(y / 20000),
                    σyz: 0.3 * Math.cos(x / 35000)
                };
            }
        }
    }
    
    initializeFaultPlanes() {
        this.faultPlanes = [
            {
                id: 'main_fault',
                type: 'normal',
                position: { x: 60, y: 40 },
                dimensions: { length: 80, width: 30 },
                orientation: {
                    strike: 45, // degrees
                    dip: 70,
                    rake: -90
                },
                friction: 0.6,
                cohesion: 5, // MPa
                active: true,
                slipHistory: []
            },
            {
                id: 'thrust_fault',
                type: 'reverse',
                position: { x: 120, y: 60 },
                dimensions: { length: 60, width: 25 },
                orientation: {
                    strike: 320,
                    dip: 30,
                    rake: 90
                },
                friction: 0.75,
                cohesion: 8,
                active: true,
                slipHistory: []
            },
            {
                id: 'strike_slip',
                type: 'strike_slip',
                position: { x: 90, y: 80 },
                dimensions: { length: 100, width: 15 },
                orientation: {
                    strike: 0,
                    dip: 90,
                    rake: 0
                },
                friction: 0.65,
                cohesion: 3,
                active: true,
                slipHistory: []
            }
        ];
    }
    
    initializeMomentTensors() {
        this.momentTensors = [
            {
                id: 'eq_1',
                position: { x: 75, y: 45 },
                magnitude: 6.5,
                depth: 15000, // meters
                mechanism: {
                    strike: 45,
                    dip: 70,
                    rake: -90
                },
                momentTensor: {
                    Mxx: 1.2e18, Myy: -0.8e18, Mzz: -0.4e18,
                    Mxy: 0.6e18, Mxz: 0.3e18, Myz: -0.2e18
                },
                time: 0,
                active: true,
                beachBall: null
            },
            {
                id: 'eq_2',
                position: { x: 125, y: 65 },
                magnitude: 7.2,
                depth: 8000,
                mechanism: {
                    strike: 320,
                    dip: 30,
                    rake: 90
                },
                momentTensor: {
                    Mxx: -2.1e18, Myy: 1.8e18, Mzz: 0.3e18,
                    Mxy: -1.2e18, Mxz: 0.1e18, Myz: 0.8e18
                },
                time: 86400 * 15, // 15 days later
                active: false,
                beachBall: null
            },
            {
                id: 'eq_3',
                position: { x: 95, y: 85 },
                magnitude: 5.8,
                depth: 12000,
                mechanism: {
                    strike: 0,
                    dip: 90,
                    rake: 0
                },
                momentTensor: {
                    Mxx: 0.5e18, Myy: -0.5e18, Mzz: 0,
                    Mxy: 0.8e18, Mxz: 0, Myz: 0
                },
                time: 86400 * 45, // 45 days later
                active: false,
                beachBall: null
            }
        ];
        
        // Generate beach balls for moment tensors
        this.momentTensors.forEach(mt => {
            mt.beachBall = this.generateBeachBall(mt.mechanism);
        });
    }
    
    createInterface() {
        this.container = document.createElement('div');
        this.container.id = 'stress-visualizer-panel';
        this.container.style.cssText = `
            position: fixed;
            top: 70px;
            left: 980px;
            width: 850px;
            height: 750px;
            background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            z-index: 9996;
            display: none;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        `;
        
        this.container.innerHTML = `
            <div class="stress-header" style="
                padding: 20px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            ">
                <h3 style="margin: 0; font-size: 20px; font-weight: 600;">
                    🗜️ Tectonic Stress Visualizer
                </h3>
                <div class="header-controls">
                    <button id="stress-export" style="
                        background: rgba(46,213,115,0.2);
                        border: 1px solid #2ed573;
                        color: #2ed573;
                        padding: 8px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        cursor: pointer;
                        margin-right: 10px;
                    ">💾 Export</button>
                    <button id="close-stress" style="
                        background: rgba(231,76,60,0.2);
                        border: 1px solid #e74c3c;
                        color: #e74c3c;
                        padding: 8px 12px;
                        border-radius: 15px;
                        cursor: pointer;
                    ">×</button>
                </div>
            </div>
            
            <div class="stress-controls" style="
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
                    <select id="stress-visualization" style="
                        padding: 6px 10px;
                        background: rgba(255,255,255,0.1);
                        border: 1px solid rgba(255,255,255,0.3);
                        border-radius: 6px;
                        color: white;
                        font-size: 12px;
                    ">
                        <option value="stress_tensor">📊 Stress Tensor</option>
                        <option value="coulomb_stress">⚡ Coulomb Stress</option>
                        <option value="strain_rate">🌊 Strain Rate</option>
                        <option value="principal_stress">🎯 Principal Stress</option>
                    </select>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-fault-planes" checked style="margin-right: 5px;">
                        Fault Planes
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-beach-balls" checked style="margin-right: 5px;">
                        Beach Balls
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-strain-vectors" checked style="margin-right: 5px;">
                        Strain Vectors
                    </label>
                    <label style="font-size: 12px;">
                        <input type="checkbox" id="show-coulomb-contours" checked style="margin-right: 5px;">
                        Coulomb Contours
                    </label>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <button id="stress-play-pause" style="
                        background: linear-gradient(45deg, #8e44ad, #9b59b6);
                        border: none;
                        color: white;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        font-weight: 600;
                    ">▶️ Start</button>
                    
                    <button id="stress-reset" style="
                        background: rgba(241,196,15,0.2);
                        border: 1px solid #f1c40f;
                        color: #f1c40f;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                    ">🔄 Reset</button>
                    
                    <button id="trigger-earthquake" style="
                        background: rgba(231,76,60,0.2);
                        border: 1px solid #e74c3c;
                        color: #e74c3c;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                    ">💥 Trigger EQ</button>
                </div>
                
                <div style="flex: 1; text-align: right; font-size: 12px; color: #ecf0f1;">
                    Time: <span id="stress-time">0 days</span>
                </div>
            </div>
            
            <div class="stress-visualization" style="
                position: relative;
                height: calc(100% - 140px);
                padding: 20px;
            ">
                <canvas id="stress-canvas" style="
                    width: 560px;
                    height: 400px;
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 10px;
                    background: #2c3e50;
                    cursor: crosshair;
                "></canvas>
                
                <div id="stress-colorbar" style="
                    position: absolute;
                    right: 40px;
                    top: 40px;
                    width: 25px;
                    height: 350px;
                    border: 1px solid rgba(255,255,255,0.3);
                    border-radius: 12px;
                    overflow: hidden;
                "></div>
                
                <div id="stress-colorbar-labels" style="
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
                
                <div id="stress-info" style="
                    position: absolute;
                    left: 580px;
                    top: 40px;
                    width: 220px;
                    background: rgba(0,0,0,0.7);
                    padding: 15px;
                    border-radius: 10px;
                    font-size: 12px;
                ">
                    <div><strong>Stress Analysis:</strong></div>
                    <div id="stress-status">Initialized</div>
                    <div style="margin-top: 10px;"><strong>Active Faults:</strong></div>
                    <div id="fault-status">${this.faultPlanes.filter(f => f.active).length} faults</div>
                    <div style="margin-top: 10px;"><strong>Earthquakes:</strong></div>
                    <div id="earthquake-count">0 events</div>
                    <div style="margin-top: 10px;"><strong>Cursor Info:</strong></div>
                    <div id="stress-cursor-info">Move cursor over model</div>
                </div>
                
                <div id="fault-analysis" style="
                    position: absolute;
                    left: 40px;
                    bottom: 40px;
                    width: 500px;
                    height: 120px;
                    background: rgba(0,0,0,0.7);
                    padding: 10px;
                    border-radius: 10px;
                    font-size: 11px;
                    overflow-y: auto;
                ">
                    <div><strong>Fault Analysis & Coulomb Stress Changes:</strong></div>
                    <div id="fault-details" style="margin-top: 5px;"></div>
                </div>
                
                <div id="beach-ball-legend" style="
                    position: absolute;
                    right: 40px;
                    bottom: 40px;
                    width: 120px;
                    background: rgba(0,0,0,0.7);
                    padding: 10px;
                    border-radius: 10px;
                    font-size: 10px;
                ">
                    <div><strong>Beach Ball Legend:</strong></div>
                    <div style="margin-top: 5px;">
                        <div>🔴 Compressive</div>
                        <div>⚪ Tensile</div>
                        <div>📐 Nodal Planes</div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.container);
        this.canvas = this.container.querySelector('#stress-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 560;
        this.canvas.height = 400;
        
        this.createStressColorbar();
    }
    
    createStressColorbar() {
        const colorbar = this.container.querySelector('#stress-colorbar');
        const labels = this.container.querySelector('#stress-colorbar-labels');
        
        const updateColorbar = () => {
            const scheme = this.stressColorSchemes[this.currentVisualization];
            
            // Create gradient
            const canvas = document.createElement('canvas');
            canvas.width = 25;
            canvas.height = 350;
            const ctx = canvas.getContext('2d');
            
            const gradient = ctx.createLinearGradient(0, 0, 0, 350);
            scheme.colors.forEach((color, index) => {
                const position = index / (scheme.colors.length - 1);
                gradient.addColorStop(1 - position, color); // Invert for top-to-bottom
            });
            
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 25, 350);
            
            colorbar.style.backgroundImage = `url(${canvas.toDataURL()})`;
            
            // Update labels
            labels.innerHTML = '';
            const numLabels = 7;
            for (let i = 0; i < numLabels; i++) {
                const value = scheme.range[1] - (i / (numLabels - 1)) * (scheme.range[1] - scheme.range[0]);
                const label = document.createElement('div');
                label.textContent = value.toFixed(1);
                labels.appendChild(label);
            }
        };
        
        updateColorbar();
        this.updateColorbar = updateColorbar;
    }
    
    setupControls() {
        const closeBtn = this.container.querySelector('#close-stress');
        const playPauseBtn = this.container.querySelector('#stress-play-pause');
        const resetBtn = this.container.querySelector('#stress-reset');
        const exportBtn = this.container.querySelector('#stress-export');
        const triggerEqBtn = this.container.querySelector('#trigger-earthquake');
        const visualizationSelect = this.container.querySelector('#stress-visualization');
        
        const showFaultPlanesCheck = this.container.querySelector('#show-fault-planes');
        const showBeachBallsCheck = this.container.querySelector('#show-beach-balls');
        const showStrainVectorsCheck = this.container.querySelector('#show-strain-vectors');
        const showCoulombContoursCheck = this.container.querySelector('#show-coulomb-contours');
        
        closeBtn.addEventListener('click', () => this.hide());
        playPauseBtn.addEventListener('click', () => this.toggleSimulation());
        resetBtn.addEventListener('click', () => this.resetModel());
        exportBtn.addEventListener('click', () => this.exportStressData());
        triggerEqBtn.addEventListener('click', () => this.triggerEarthquake());
        
        visualizationSelect.addEventListener('change', (e) => {
            this.currentVisualization = e.target.value;
            this.updateColorbar();
            this.render();
        });
        
        showFaultPlanesCheck.addEventListener('change', (e) => {
            this.showFaultPlanes = e.target.checked;
            this.render();
        });
        
        showBeachBallsCheck.addEventListener('change', (e) => {
            this.showBeachBalls = e.target.checked;
            this.render();
        });
        
        showStrainVectorsCheck.addEventListener('change', (e) => {
            this.showStrainVectors = e.target.checked;
            this.render();
        });
        
        showCoulombContoursCheck.addEventListener('change', (e) => {
            this.showCoulombContours = e.target.checked;
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
            const stress = this.stressTensor[y][x];
            const coulomb = this.coulombStress[y][x];
            const strain = this.strainRate[y][x];
            
            // Calculate principal stresses
            const principals = this.calculatePrincipalStresses(stress);
            
            const cursorInfo = this.container.querySelector('#stress-cursor-info');
            cursorInfo.innerHTML = `
                Grid: (${x}, ${y})<br>
                σ1: ${principals.σ1.toFixed(1)} MPa<br>
                σ2: ${principals.σ2.toFixed(1)} MPa<br>
                σ3: ${principals.σ3.toFixed(1)} MPa<br>
                Coulomb: ${coulomb.toFixed(3)} bars<br>
                Strain Rate: ${(Math.sqrt(strain.exx**2 + strain.eyy**2) * 1e9).toFixed(1)} ns/yr
            `;
        }
    }
    
    calculatePrincipalStresses(stress) {
        // Simplified 2D principal stress calculation
        const σavg = (stress.σxx + stress.σyy) / 2;
        const R = Math.sqrt(((stress.σxx - stress.σyy) / 2)**2 + stress.σxy**2);
        
        return {
            σ1: σavg + R,
            σ2: stress.σzz,
            σ3: σavg - R,
            angle: 0.5 * Math.atan2(2 * stress.σxy, stress.σxx - stress.σyy) * 180 / Math.PI
        };
    }
    
    toggleSimulation() {
        const btn = this.container.querySelector('#stress-play-pause');
        
        if (this.isRunning) {
            this.stopSimulation();
            btn.innerHTML = '▶️ Start';
            btn.style.background = 'linear-gradient(45deg, #8e44ad, #9b59b6)';
        } else {
            this.startSimulation();
            btn.innerHTML = '⏸️ Pause';
            btn.style.background = 'linear-gradient(45deg, #27ae60, #2ecc71)';
        }
    }
    
    startSimulation() {
        this.isRunning = true;
        this.animate();
        
        const status = this.container.querySelector('#stress-status');
        status.textContent = 'Running stress evolution...';
    }
    
    stopSimulation() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        const status = this.container.querySelector('#stress-status');
        status.textContent = 'Simulation paused';
    }
    
    animate() {
        if (!this.isRunning) return;
        
        this.updateStressEvolution();
        this.calculateCoulombStress();
        this.updateFaultAnalysis();
        this.checkEarthquakeTriggers();
        this.render();
        this.updateInterface();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    updateStressEvolution() {
        const dt = this.timeStep;
        this.currentTime += dt;
        
        // Update stress field due to tectonic loading
        this.applyTectonicLoading(dt);
        
        // Update strain rate field
        this.calculateStrainRate();
        
        // Update time display
        const timeDisplay = this.container.querySelector('#stress-time');
        const days = Math.floor(this.currentTime / 86400);
        const years = Math.floor(days / 365.25);
        
        if (years > 0) {
            timeDisplay.textContent = `${years} years, ${days % 365} days`;
        } else {
            timeDisplay.textContent = `${days} days`;
        }
    }
    
    applyTectonicLoading(dt) {
        const { width, height } = this.gridSize;
        const loadingRate = 1e-9; // MPa/s (very slow)
        
        for (let j = 0; j < height; j++) {
            for (let i = 0; i < width; i++) {
                const x = (i / width) * this.physicalSize.width;
                
                // Simulate regional tectonic loading
                const loadingX = loadingRate * dt * (1 + 0.5 * Math.sin(x / 100000));
                const loadingY = loadingRate * dt * 0.3;
                const loadingShear = loadingRate * dt * 0.1 * Math.cos(x / 80000);
                
                this.stressTensor[j][i].σxx += loadingX;
                this.stressTensor[j][i].σyy += loadingY;
                this.stressTensor[j][i].σxy += loadingShear;
            }
        }
    }
    
    calculateStrainRate() {
        const { width, height } = this.gridSize;
        const dx = this.physicalSize.width / width;
        const dy = this.physicalSize.height / height;
        
        for (let j = 1; j < height - 1; j++) {
            for (let i = 1; i < width - 1; i++) {
                // Calculate strain rate from stress gradients (simplified)
                const E = 70000; // Young's modulus (MPa)
                const nu = 0.25; // Poisson's ratio
                
                const stress = this.stressTensor[j][i];
                
                // Strain-stress relationship
                this.strainRate[j][i].exx = (stress.σxx - nu * (stress.σyy + stress.σzz)) / E;
                this.strainRate[j][i].eyy = (stress.σyy - nu * (stress.σxx + stress.σzz)) / E;
                this.strainRate[j][i].ezz = (stress.σzz - nu * (stress.σxx + stress.σyy)) / E;
                this.strainRate[j][i].exy = stress.σxy / (2 * E / (1 + nu));
            }
        }
    }
    
    calculateCoulombStress() {
        const { width, height } = this.gridSize;
        
        for (let j = 0; j < height; j++) {
            for (let i = 0; i < width; i++) {
                const stress = this.stressTensor[j][i];
                
                // Calculate Coulomb stress for a typical fault (μ = 0.6)
                const mu = 0.6; // Friction coefficient
                const principals = this.calculatePrincipalStresses(stress);
                
                // Simplified Coulomb stress calculation
                const shearStress = Math.abs(principals.σ1 - principals.σ3) / 2;
                const normalStress = (principals.σ1 + principals.σ3) / 2;
                
                // Coulomb stress = τ - μ(σn - P) where P is pore pressure (assumed 0)
                this.coulombStress[j][i] = (shearStress - mu * Math.max(0, normalStress)) / 10; // Convert to bars
            }
        }
    }
    
    updateFaultAnalysis() {
        const faultDetails = this.container.querySelector('#fault-details');
        let html = '';
        
        this.faultPlanes.forEach(fault => {
            if (fault.active) {
                const stress = this.getStressAtFault(fault);
                const coulombStress = this.getCoulombStressAtFault(fault);
                const failureProb = this.calculateFailureProbability(fault, coulombStress);
                
                html += `<div style="margin-bottom: 5px;">
                    <strong>${fault.id}</strong> (${fault.type}): 
                    Coulomb: ${coulombStress.toFixed(2)} bars, 
                    Failure Prob: ${(failureProb * 100).toFixed(1)}%
                </div>`;
                
                // Update fault slip history if near failure
                if (failureProb > 0.8) {
                    fault.slipHistory.push({
                        time: this.currentTime,
                        stress: coulombStress,
                        probability: failureProb
                    });
                }
            }
        });
        
        faultDetails.innerHTML = html;
    }
    
    getStressAtFault(fault) {
        const i = Math.floor(fault.position.x);
        const j = Math.floor(fault.position.y);
        return this.stressTensor[j][i];
    }
    
    getCoulombStressAtFault(fault) {
        const i = Math.floor(fault.position.x);
        const j = Math.floor(fault.position.y);
        return this.coulombStress[j][i];
    }
    
    calculateFailureProbability(fault, coulombStress) {
        // Simplified failure probability based on Coulomb stress
        const threshold = 1.0; // bars
        const probability = Math.max(0, Math.min(1, (coulombStress - fault.cohesion) / threshold));
        return probability;
    }
    
    checkEarthquakeTriggers() {
        // Check if any scheduled earthquakes should be triggered
        this.momentTensors.forEach(mt => {
            if (!mt.active && this.currentTime >= mt.time) {
                this.triggerMomentTensor(mt);
                mt.active = true;
            }
        });
    }
    
    triggerEarthquake() {
        // Manually trigger an earthquake at a random location
        const x = Math.floor(Math.random() * this.gridSize.width);
        const y = Math.floor(Math.random() * this.gridSize.height);
        
        const newEarthquake = {
            id: `manual_eq_${Date.now()}`,
            position: { x, y },
            magnitude: 5.0 + Math.random() * 2.0,
            depth: 5000 + Math.random() * 15000,
            mechanism: {
                strike: Math.random() * 360,
                dip: 30 + Math.random() * 60,
                rake: -180 + Math.random() * 360
            },
            momentTensor: this.generateMomentTensor(6.0),
            time: this.currentTime,
            active: true,
            beachBall: null
        };
        
        newEarthquake.beachBall = this.generateBeachBall(newEarthquake.mechanism);
        this.momentTensors.push(newEarthquake);
        
        this.triggerMomentTensor(newEarthquake);
        this.updateEarthquakeCount();
    }
    
    generateMomentTensor(magnitude) {
        const M0 = Math.pow(10, 1.5 * magnitude + 9.1); // Moment magnitude relation
        
        return {
            Mxx: (Math.random() - 0.5) * M0,
            Myy: (Math.random() - 0.5) * M0,
            Mzz: (Math.random() - 0.5) * M0,
            Mxy: (Math.random() - 0.5) * M0 * 0.5,
            Mxz: (Math.random() - 0.5) * M0 * 0.5,
            Myz: (Math.random() - 0.5) * M0 * 0.5
        };
    }
    
    triggerMomentTensor(mt) {
        // Apply stress changes due to earthquake
        const { width, height } = this.gridSize;
        const sourceX = Math.floor(mt.position.x);
        const sourceY = Math.floor(mt.position.y);
        const influenceRadius = Math.min(50, mt.magnitude * 8);
        
        for (let j = 0; j < height; j++) {
            for (let i = 0; i < width; i++) {
                const distance = Math.sqrt((i - sourceX)**2 + (j - sourceY)**2);
                
                if (distance <= influenceRadius) {
                    const factor = Math.exp(-distance / (influenceRadius / 3));
                    const stressChange = mt.magnitude * factor * 0.1; // MPa
                    
                    // Apply stress changes based on moment tensor
                    const angle = Math.atan2(j - sourceY, i - sourceX);
                    const cos2θ = Math.cos(2 * angle);
                    const sin2θ = Math.sin(2 * angle);
                    
                    this.stressTensor[j][i].σxx += stressChange * cos2θ;
                    this.stressTensor[j][i].σyy -= stressChange * cos2θ;
                    this.stressTensor[j][i].σxy += stressChange * sin2θ;
                    
                    // Update Coulomb stress
                    this.coulombStress[j][i] += stressChange * factor / 10; // bars
                }
            }
        }
    }
    
    generateBeachBall(mechanism) {
        const { strike, dip, rake } = mechanism;
        const size = 20;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        
        // Draw beach ball background
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(size/2, size/2, size/2 - 1, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw nodal planes (simplified)
        const plane1Angle = strike * Math.PI / 180;
        const plane2Angle = (strike + 90) * Math.PI / 180;
        
        // First nodal plane
        ctx.beginPath();
        ctx.moveTo(size/2 + (size/2 - 2) * Math.cos(plane1Angle), 
                   size/2 + (size/2 - 2) * Math.sin(plane1Angle));
        ctx.lineTo(size/2 - (size/2 - 2) * Math.cos(plane1Angle), 
                   size/2 - (size/2 - 2) * Math.sin(plane1Angle));
        ctx.stroke();
        
        // Second nodal plane
        ctx.beginPath();
        ctx.moveTo(size/2 + (size/2 - 2) * Math.cos(plane2Angle), 
                   size/2 + (size/2 - 2) * Math.sin(plane2Angle));
        ctx.lineTo(size/2 - (size/2 - 2) * Math.cos(plane2Angle), 
                   size/2 - (size/2 - 2) * Math.sin(plane2Angle));
        ctx.stroke();
        
        // Fill compressive quadrants (simplified based on rake)
        if (Math.abs(rake) < 45 || Math.abs(rake) > 135) {
            // Strike-slip mechanism
            ctx.fillStyle = 'black';
        } else if (rake > 0) {
            // Reverse/thrust mechanism
            ctx.fillStyle = 'black';
        } else {
            // Normal mechanism
            ctx.fillStyle = 'black';
        }
        
        // Fill appropriate quadrants (simplified)
        ctx.beginPath();
        ctx.arc(size/2, size/2, size/4, 0, Math.PI);
        ctx.fill();
        
        return canvas;
    }
    
    updateInterface() {
        this.updateEarthquakeCount();
        this.updateFaultStatus();
    }
    
    updateEarthquakeCount() {
        const count = this.momentTensors.filter(mt => mt.active).length;
        const eqCount = this.container.querySelector('#earthquake-count');
        eqCount.textContent = `${count} events`;
    }
    
    updateFaultStatus() {
        const activeFaults = this.faultPlanes.filter(f => f.active).length;
        const faultStatus = this.container.querySelector('#fault-status');
        faultStatus.textContent = `${activeFaults} active faults`;
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Render main stress field
        this.renderStressField();
        
        // Render overlays
        if (this.showCoulombContours) this.renderCoulombContours();
        if (this.showStrainVectors) this.renderStrainVectors();
        if (this.showFaultPlanes) this.renderFaultPlanes();
        if (this.showBeachBalls) this.renderBeachBalls();
    }
    
    renderStressField() {
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        const imageData = this.ctx.createImageData(canvasWidth, canvasHeight);
        const data = imageData.data;
        
        const scheme = this.stressColorSchemes[this.currentVisualization];
        
        for (let j = 0; j < canvasHeight; j++) {
            for (let i = 0; i < canvasWidth; i++) {
                const gridX = Math.floor(i / canvasWidth * width);
                const gridY = Math.floor(j / canvasHeight * height);
                
                let value;
                switch (this.currentVisualization) {
                    case 'stress_tensor':
                        const stress = this.stressTensor[gridY][gridX];
                        value = (stress.σxx + stress.σyy + stress.σzz) / 3; // Mean stress
                        break;
                    case 'coulomb_stress':
                        value = this.coulombStress[gridY][gridX];
                        break;
                    case 'strain_rate':
                        const strain = this.strainRate[gridY][gridX];
                        value = Math.sqrt(strain.exx**2 + strain.eyy**2) * 1e9; // nanostrain/year
                        break;
                    case 'principal_stress':
                        const principals = this.calculatePrincipalStresses(this.stressTensor[gridY][gridX]);
                        value = principals.σ1 - principals.σ3; // Differential stress
                        break;
                }
                
                const color = this.valueToColor(value, scheme);
                const index = (j * canvasWidth + i) * 4;
                
                data[index] = color[0];     // R
                data[index + 1] = color[1]; // G
                data[index + 2] = color[2]; // B
                data[index + 3] = 255;      // A
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
        
        return [
            Math.round(lowerColor.r * (1 - fraction) + upperColor.r * fraction),
            Math.round(lowerColor.g * (1 - fraction) + upperColor.g * fraction),
            Math.round(lowerColor.b * (1 - fraction) + upperColor.b * fraction)
        ];
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }
    
    renderCoulombContours() {
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        const contourLevels = [-5, -2, -1, 0, 1, 2, 5];
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.lineWidth = 1;
        this.ctx.font = '9px Arial';
        this.ctx.fillStyle = 'white';
        
        contourLevels.forEach((level, levelIndex) => {
            this.ctx.strokeStyle = level === 0 ? 'rgba(255, 255, 0, 0.9)' : 'rgba(255, 255, 255, 0.6)';
            this.ctx.lineWidth = level === 0 ? 2 : 1;
            
            // Simple contour tracing
            for (let j = 1; j < height - 1; j++) {
                for (let i = 1; i < width - 1; i++) {
                    const stress = this.coulombStress[j][i];
                    
                    if (Math.abs(stress - level) < 0.2) {
                        const x = (i / width) * canvasWidth;
                        const y = (j / height) * canvasHeight;
                        
                        this.ctx.beginPath();
                        this.ctx.arc(x, y, 1, 0, 2 * Math.PI);
                        this.ctx.stroke();
                        
                        // Label major contours
                        if (level === 0 && i % 30 === 0 && j % 20 === 0) {
                            this.ctx.fillText('0', x + 3, y - 3);
                        }
                    }
                }
            }
        });
    }
    
    renderStrainVectors() {
        const { width, height } = this.gridSize;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        const step = 12;
        
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
        this.ctx.lineWidth = 1;
        
        for (let j = step; j < height; j += step) {
            for (let i = step; i < width; i += step) {
                const strain = this.strainRate[j][i];
                const magnitude = Math.sqrt(strain.exx**2 + strain.eyy**2) * 1e12; // Scale up
                
                if (magnitude > 0.1) {
                    const x = (i / width) * canvasWidth;
                    const y = (j / height) * canvasHeight;
                    const scale = Math.min(8, magnitude * 50);
                    
                    // Extension direction
                    const angle = 0.5 * Math.atan2(2 * strain.exy, strain.exx - strain.eyy);
                    const dx = scale * Math.cos(angle);
                    const dy = scale * Math.sin(angle);
                    
                    // Draw strain vector
                    this.ctx.beginPath();
                    this.ctx.moveTo(x - dx, y - dy);
                    this.ctx.lineTo(x + dx, y + dy);
                    this.ctx.stroke();
                    
                    // Arrow heads
                    const arrowAngle = angle + Math.PI * 0.8;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x + dx, y + dy);
                    this.ctx.lineTo(x + dx - 3 * Math.cos(arrowAngle), y + dy - 3 * Math.sin(arrowAngle));
                    this.ctx.stroke();
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(x + dx, y + dy);
                    this.ctx.lineTo(x + dx - 3 * Math.cos(arrowAngle + 0.4), y + dy - 3 * Math.sin(arrowAngle + 0.4));
                    this.ctx.stroke();
                }
            }
        }
    }
    
    renderFaultPlanes() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.faultPlanes.forEach(fault => {
            if (fault.active) {
                const x = (fault.position.x / this.gridSize.width) * canvasWidth;
                const y = (fault.position.y / this.gridSize.height) * canvasHeight;
                const length = (fault.dimensions.length / this.gridSize.width) * canvasWidth;
                const width = (fault.dimensions.width / this.gridSize.height) * canvasHeight;
                
                // Fault type colors
                const colors = {
                    'normal': '#FF6B6B',
                    'reverse': '#4ECDC4',
                    'strike_slip': '#45B7D1'
                };
                
                const color = colors[fault.type] || '#FFFFFF';
                
                // Draw fault plane
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 3;
                const angle = fault.orientation.strike * Math.PI / 180;
                
                this.ctx.save();
                this.ctx.translate(x, y);
                this.ctx.rotate(angle);
                this.ctx.beginPath();
                this.ctx.moveTo(-length/2, 0);
                this.ctx.lineTo(length/2, 0);
                this.ctx.stroke();
                
                // Draw fault width indicator
                this.ctx.strokeStyle = `${color}80`;
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.moveTo(-length/2, -width/2);
                this.ctx.lineTo(length/2, -width/2);
                this.ctx.moveTo(-length/2, width/2);
                this.ctx.lineTo(length/2, width/2);
                this.ctx.stroke();
                
                this.ctx.restore();
                
                // Fault label
                this.ctx.fillStyle = color;
                this.ctx.font = 'bold 11px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(fault.id.replace('_', ' '), x, y - 20);
                
                // Fault info
                this.ctx.font = '9px Arial';
                this.ctx.fillText(`${fault.type}`, x, y + 25);
                this.ctx.fillText(`${fault.orientation.strike}°/${fault.orientation.dip}°`, x, y + 35);
            }
        });
    }
    
    renderBeachBalls() {
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        this.momentTensors.forEach(mt => {
            if (mt.active && mt.beachBall) {
                const x = (mt.position.x / this.gridSize.width) * canvasWidth;
                const y = (mt.position.y / this.gridSize.height) * canvasHeight;
                const size = Math.min(30, mt.magnitude * 4);
                
                // Draw beach ball
                this.ctx.save();
                this.ctx.translate(x, y);
                this.ctx.scale(size / 20, size / 20);
                this.ctx.drawImage(mt.beachBall, -10, -10);
                this.ctx.restore();
                
                // Magnitude label
                this.ctx.fillStyle = 'white';
                this.ctx.font = 'bold 10px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(`M${mt.magnitude.toFixed(1)}`, x, y + size/2 + 12);
                
                // Event ID
                this.ctx.font = '8px Arial';
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                this.ctx.fillText(mt.id, x, y + size/2 + 22);
            }
        });
    }
    
    resetModel() {
        this.stopSimulation();
        this.currentTime = 0;
        this.initializeStressFields();
        this.render();
        
        const playPauseBtn = this.container.querySelector('#stress-play-pause');
        playPauseBtn.innerHTML = '▶️ Start';
        playPauseBtn.style.background = 'linear-gradient(45deg, #8e44ad, #9b59b6)';
        
        const timeDisplay = this.container.querySelector('#stress-time');
        timeDisplay.textContent = '0 days';
        
        const status = this.container.querySelector('#stress-status');
        status.textContent = 'Model reset';
        
        this.updateInterface();
    }
    
    exportStressData() {
        const stressData = {
            timestamp: new Date().toISOString(),
            currentTime: this.currentTime,
            gridSize: this.gridSize,
            physicalSize: this.physicalSize,
            stressTensor: this.stressTensor,
            coulombStress: this.coulombStress,
            strainRate: this.strainRate,
            faultPlanes: this.faultPlanes,
            momentTensors: this.momentTensors.filter(mt => mt.active),
            stressHistory: this.stressHistory
        };
        
        const blob = new Blob([JSON.stringify(stressData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `stress_analysis_${Math.floor(this.currentTime / 86400)}days.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    makeDraggable() {
        const header = this.container.querySelector('.stress-header');
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
            
            this.container.style.left = Math.max(0, Math.min(window.innerWidth - 850, newLeft)) + 'px';
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
const stressStyle = document.createElement('style');
stressStyle.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.95); }
    }
    
    #stress-visualizer-panel select option {
        background: #9b59b6;
        color: white;
    }
    
    #stress-visualizer-panel::-webkit-scrollbar {
        width: 8px;
    }
    
    #stress-visualizer-panel::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    #stress-visualizer-panel::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    
    #stress-visualizer-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
`;
document.head.appendChild(stressStyle);

// Global instance
window.stressVisualizer = new StressVisualizer();

// Integration with other geological models
if (window.mantleConvectionModel) {
    window.mantleConvectionModel.stressVisualizer = window.stressVisualizer;
}

// Keyboard shortcut to toggle stress visualizer
document.addEventListener('keydown', (e) => {
    if (e.key === 'S' && e.altKey && e.ctrlKey) {
        e.preventDefault();
        if (window.stressVisualizer.container.style.display === 'none') {
            window.stressVisualizer.show();
        } else {
            window.stressVisualizer.hide();
        }
    }
});

console.log('🗜️ Tectonic Stress Visualizer loaded successfully!');
console.log('🔑 Press Ctrl+Alt+S to toggle stress visualizer');
console.log('🎯 Access via: window.stressVisualizer.show()');
console.log('📊 Features: Stress tensors, Coulomb stress, fault planes, beach balls, strain rates');