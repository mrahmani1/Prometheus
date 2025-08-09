/**
 * Advanced Core Dynamics 3D Visualization System
 * Real-time visualization of Earth's core convection, magnetic field generation,
 * and inner core growth with advanced fluid dynamics rendering
 */

class CoreDynamicsVisualization {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.config = this._mergeConfig(config);
        
        // Core visualization components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Core geometry and meshes
        this.innerCore = null;
        this.outerCore = null;
        this.mantleBoundary = null;
        
        // Fluid dynamics visualization
        this.convectionCells = [];
        this.flowVectors = [];
        this.temperatureField = null;
        this.velocityField = null;
        
        // Magnetic field visualization
        this.magneticFieldLines = [];
        this.currentLoops = [];
        this.inductionVectors = [];
        
        // Animation and data management
        this.simulationData = [];
        this.currentTimeIndex = 0;
        this.isPlaying = false;
        this.animationSpeed = 1.0;
        
        // Shader materials
        this.shaderMaterials = {};
        this.uniforms = {};
        
        // Analysis tools
        this.flowAnalyzer = new FlowAnalyzer();
        this.thermalAnalyzer = new ThermalAnalyzer();
        this.magneticAnalyzer = new MagneticAnalyzer();
        
        this.initialize();
    }
    
    _mergeConfig(config) {
        const defaults = {
            // Core dimensions (in meters)
            innerCoreRadius: 1216000,
            outerCoreRadius: 3480000,
            mantleBoundaryRadius: 6371000,
            
            // Visualization settings
            showConvection: true,
            showMagneticField: true,
            showTemperature: true,
            showVelocityField: true,
            showInnerCoreGrowth: true,
            
            // Rendering quality
            convectionResolution: 64,
            magneticFieldResolution: 32,
            particleCount: 10000,
            
            // Animation settings
            timeStep: 0.01,
            maxTimeSteps: 1000,
            
            // Visual effects
            useVolumeRendering: true,
            enableShaders: true,
            enableParticles: true,
            
            // Colors and opacity
            innerCoreColor: 0xffaa00,
            outerCoreColor: 0xff6600,
            temperatureColormap: 'hot',
            velocityColormap: 'viridis',
            magneticColormap: 'plasma'
        };
        
        return { ...defaults, ...config };
    }
    
    initialize() {
        this.setupThreeJS();
        this.setupCore();
        this.setupShaders();
        this.setupUI();
        this.loadSimulationData();
        this.startRenderLoop();
    }
    
    setupThreeJS() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000005);
        
        // Camera setup
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 100000, 50000000);
        this.camera.position.set(8000000, 6000000, 8000000);
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Enable extensions for advanced rendering
        this.renderer.extensions.get('OES_texture_float');
        this.renderer.extensions.get('OES_texture_float_linear');
        
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 2000000;
        this.controls.maxDistance = 20000000;
        this.controls.target.set(0, 0, 0);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(10000000, 10000000, 5000000);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Point light at core center
        const coreLight = new THREE.PointLight(0xff6600, 2.0, this.config.outerCoreRadius);
        coreLight.position.set(0, 0, 0);
        this.scene.add(coreLight);
    }
    
    setupCore() {
        // Inner core
        const innerCoreGeometry = new THREE.SphereGeometry(
            this.config.innerCoreRadius, 
            32, 16
        );
        
        const innerCoreMaterial = new THREE.MeshPhongMaterial({
            color: this.config.innerCoreColor,
            transparent: true,
            opacity: 0.9,
            emissive: new THREE.Color(0x441100),
            emissiveIntensity: 0.3
        });
        
        this.innerCore = new THREE.Mesh(innerCoreGeometry, innerCoreMaterial);
        this.innerCore.castShadow = true;
        this.scene.add(this.innerCore);
        
        // Outer core (fluid)
        const outerCoreGeometry = new THREE.SphereGeometry(
            this.config.outerCoreRadius,
            64, 32
        );
        
        this.outerCore = new THREE.Mesh(outerCoreGeometry, null); // Material set in shaders
        this.scene.add(this.outerCore);
        
        // Core-mantle boundary
        const mantleGeometry = new THREE.SphereGeometry(
            this.config.outerCoreRadius * 1.001,
            64, 32
        );
        
        const mantleMaterial = new THREE.MeshBasicMaterial({
            color: 0x664400,
            transparent: true,
            opacity: 0.1,
            wireframe: true
        });
        
        this.mantleBoundary = new THREE.Mesh(mantleGeometry, mantleMaterial);
        this.scene.add(this.mantleBoundary);
        
        // Reference axes
        this.setupAxes();
    }
    
    setupAxes() {
        // Rotation axis
        const axisLength = this.config.outerCoreRadius * 2.5;
        const axisGeometry = new THREE.CylinderGeometry(20000, 20000, axisLength);
        const axisMaterial = new THREE.MeshBasicMaterial({ 
            color: 0xffffff, 
            transparent: true, 
            opacity: 0.5 
        });
        
        this.rotationAxis = new THREE.Mesh(axisGeometry, axisMaterial);
        this.scene.add(this.rotationAxis);
        
        // Equatorial plane
        const planeGeometry = new THREE.RingGeometry(
            this.config.innerCoreRadius, 
            this.config.outerCoreRadius,
            32
        );
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0x666666,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });
        
        this.equatorialPlane = new THREE.Mesh(planeGeometry, planeMaterial);
        this.equatorialPlane.rotation.x = Math.PI / 2;
        this.scene.add(this.equatorialPlane);
    }
    
    setupShaders() {
        // Temperature field vertex shader
        this.temperatureVertexShader = `
            attribute float temperature;
            attribute vec3 velocity;
            varying float vTemperature;
            varying vec3 vVelocity;
            varying vec3 vPosition;
            varying vec3 vNormal;
            
            void main() {
                vTemperature = temperature;
                vVelocity = velocity;
                vPosition = position;
                vNormal = normal;
                
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        // Temperature field fragment shader
        this.temperatureFragmentShader = `
            uniform float time;
            uniform float minTemp;
            uniform float maxTemp;
            uniform vec3 cameraPosition;
            
            varying float vTemperature;
            varying vec3 vVelocity;
            varying vec3 vPosition;
            varying vec3 vNormal;
            
            vec3 temperatureToColor(float temp) {
                float normalized = (temp - minTemp) / (maxTemp - minTemp);
                
                // Hot colormap: black -> red -> orange -> yellow -> white
                if (normalized < 0.25) {
                    return mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), normalized * 4.0);
                } else if (normalized < 0.5) {
                    return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (normalized - 0.25) * 4.0);
                } else if (normalized < 0.75) {
                    return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), (normalized - 0.5) * 4.0);
                } else {
                    return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (normalized - 0.75) * 4.0);
                }
            }
            
            void main() {
                vec3 color = temperatureToColor(vTemperature);
                
                // Add convection animation
                float convectionPhase = sin(time * 2.0 + length(vPosition) * 0.001) * 0.5 + 0.5;
                color *= (0.8 + 0.4 * convectionPhase);
                
                // Add velocity-based effects
                float velocityMagnitude = length(vVelocity);
                color += vec3(0.1, 0.1, 0.2) * velocityMagnitude * 0.001;
                
                // Fresnel effect for core boundary
                vec3 viewDir = normalize(cameraPosition - vPosition);
                float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.0);
                color += vec3(0.2, 0.4, 0.8) * fresnel * 0.3;
                
                gl_FragColor = vec4(color, 0.7);
            }
        `;
        
        // Magnetic field line shader
        this.magneticVertexShader = `
            attribute float fieldStrength;
            attribute vec3 fieldDirection;
            varying float vFieldStrength;
            varying vec3 vFieldDirection;
            varying vec3 vPosition;
            
            void main() {
                vFieldStrength = fieldStrength;
                vFieldDirection = fieldDirection;
                vPosition = position;
                
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        this.magneticFragmentShader = `
            uniform float time;
            uniform float maxFieldStrength;
            varying float vFieldStrength;
            varying vec3 vFieldDirection;
            varying vec3 vPosition;
            
            vec3 magneticToColor(float strength) {
                float normalized = abs(strength) / maxFieldStrength;
                
                // Plasma colormap: blue -> purple -> red -> yellow
                if (normalized < 0.33) {
                    return mix(vec3(0.0, 0.0, 1.0), vec3(0.5, 0.0, 1.0), normalized * 3.0);
                } else if (normalized < 0.66) {
                    return mix(vec3(0.5, 0.0, 1.0), vec3(1.0, 0.0, 0.0), (normalized - 0.33) * 3.0);
                } else {
                    return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (normalized - 0.66) * 3.0);
                }
            }
            
            void main() {
                vec3 color = magneticToColor(vFieldStrength);
                
                // Add flowing magnetic field animation
                float flow = sin(time * 3.0 + dot(vPosition, vFieldDirection) * 0.0001) * 0.5 + 0.5;
                color *= (0.6 + 0.4 * flow);
                
                // Add polarity indication
                if (vFieldStrength < 0.0) {
                    color = mix(color, vec3(0.0, 0.8, 1.0), 0.3);
                }
                
                gl_FragColor = vec4(color, 0.8);
            }
        `;
        
        // Initialize uniforms
        this.uniforms = {
            time: { value: 0.0 },
            minTemp: { value: 4000.0 },
            maxTemp: { value: 6000.0 },
            maxFieldStrength: { value: 1e-3 },
            cameraPosition: { value: new THREE.Vector3() }
        };
    }
    
    setupUI() {
        // Create control panel
        const controlPanel = document.createElement('div');
        controlPanel.className = 'core-dynamics-controls';
        controlPanel.innerHTML = `
            <div class="control-section">
                <h3>Core Dynamics Simulation</h3>
                <div class="simulation-controls">
                    <button id="playPause">⏯️ Play/Pause</button>
                    <button id="reset">🔄 Reset</button>
                    <input type="range" id="timeSlider" min="0" max="1000" value="0">
                    <span id="timeDisplay">0 Myr</span>
                </div>
                <div class="speed-control">
                    <label>Speed: </label>
                    <input type="range" id="speedSlider" min="0.1" max="5" step="0.1" value="1">
                    <span id="speedDisplay">1.0x</span>
                </div>
            </div>
            
            <div class="control-section">
                <h4>Visualization Options</h4>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="showConvection" checked> Convection</label>
                    <label><input type="checkbox" id="showMagnetic" checked> Magnetic Field</label>
                    <label><input type="checkbox" id="showTemperature" checked> Temperature</label>
                    <label><input type="checkbox" id="showVelocity" checked> Velocity Field</label>
                    <label><input type="checkbox" id="showParticles"> Flow Particles</label>
                </div>
            </div>
            
            <div class="control-section">
                <h4>Core Properties</h4>
                <div class="property-sliders">
                    <div class="slider-group">
                        <label>Rayleigh Number:</label>
                        <input type="range" id="rayleighSlider" min="1000" max="100000" value="10000">
                        <span id="rayleighValue">10000</span>
                    </div>
                    <div class="slider-group">
                        <label>Magnetic Reynolds:</label>
                        <input type="range" id="magneticReSlider" min="100" max="10000" value="1000">
                        <span id="magneticReValue">1000</span>
                    </div>
                    <div class="slider-group">
                        <label>Inner Core Radius:</label>
                        <input type="range" id="innerCoreSlider" min="1000000" max="1500000" value="1216000">
                        <span id="innerCoreValue">1216 km</span>
                    </div>
                </div>
            </div>
            
            <div class="control-section">
                <h4>Analysis Tools</h4>
                <div class="analysis-buttons">
                    <button id="analyzeFlow">Flow Analysis</button>
                    <button id="analyzeThermal">Thermal Structure</button>
                    <button id="analyzeMagnetic">Magnetic Field</button>
                    <button id="exportData">Export Data</button>
                </div>
            </div>
            
            <div class="control-section">
                <h4>Core Statistics</h4>
                <div id="coreStats">
                    <div>Heat Flow: <span id="heatFlow">--</span> TW</div>
                    <div>Convection Vigor: <span id="convectionVigor">--</span></div>
                    <div>Magnetic Dipole: <span id="magneticDipole">--</span> Am²</div>
                    <div>Inner Core Age: <span id="innerCoreAge">--</span> Myr</div>
                </div>
            </div>
        `;
        
        this.container.appendChild(controlPanel);
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Animation controls
        document.getElementById('playPause').addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        document.getElementById('reset').addEventListener('click', () => {
            this.resetSimulation();
        });
        
        document.getElementById('timeSlider').addEventListener('input', (e) => {
            this.setTimeIndex(parseInt(e.target.value));
        });
        
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            this.animationSpeed = parseFloat(e.target.value);
            document.getElementById('speedDisplay').textContent = this.animationSpeed.toFixed(1) + 'x';
        });
        
        // Visualization toggles
        document.getElementById('showConvection').addEventListener('change', (e) => {
            this.toggleConvection(e.target.checked);
        });
        
        document.getElementById('showMagnetic').addEventListener('change', (e) => {
            this.toggleMagneticField(e.target.checked);
        });
        
        document.getElementById('showTemperature').addEventListener('change', (e) => {
            this.toggleTemperatureField(e.target.checked);
        });
        
        document.getElementById('showVelocity').addEventListener('change', (e) => {
            this.toggleVelocityField(e.target.checked);
        });
        
        document.getElementById('showParticles').addEventListener('change', (e) => {
            this.toggleFlowParticles(e.target.checked);
        });
        
        // Parameter sliders
        document.getElementById('rayleighSlider').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('rayleighValue').textContent = value;
            this.updateRayleighNumber(value);
        });
        
        document.getElementById('magneticReSlider').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('magneticReValue').textContent = value;
            this.updateMagneticReynolds(value);
        });
        
        document.getElementById('innerCoreSlider').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('innerCoreValue').textContent = (value / 1000).toFixed(0) + ' km';
            this.updateInnerCoreRadius(value);
        });
        
        // Analysis buttons
        document.getElementById('analyzeFlow').addEventListener('click', () => {
            this.analyzeFlowStructure();
        });
        
        document.getElementById('analyzeThermal').addEventListener('click', () => {
            this.analyzeThermalStructure();
        });
        
        document.getElementById('analyzeMagnetic').addEventListener('click', () => {
            this.analyzeMagneticField();
        });
        
        document.getElementById('exportData').addEventListener('click', () => {
            this.exportSimulationData();
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }
    
    loadSimulationData() {
        // Generate or load geodynamo simulation data
        console.log("Loading core dynamics simulation data...");
        
        // Generate synthetic data for demonstration
        this.generateSyntheticData();
        
        // Initialize visualization
        this.updateVisualization();
        
        console.log(`Loaded ${this.simulationData.length} time steps`);
    }
    
    generateSyntheticData() {
        const timeSteps = 1000;
        const timeSpan = 1e9; // 1 billion years
        
        for (let i = 0; i < timeSteps; i++) {
            const time = (i / timeSteps) * timeSpan;
            
            // Generate convection pattern
            const convectionData = this.generateConvectionPattern(time);
            
            // Generate magnetic field
            const magneticData = this.generateMagneticField(time, convectionData);
            
            // Generate temperature field
            const temperatureData = this.generateTemperatureField(time, convectionData);
            
            // Inner core growth
            const innerCoreRadius = this.calculateInnerCoreRadius(time);
            
            this.simulationData.push({
                time: time,
                convection: convectionData,
                magneticField: magneticData,
                temperature: temperatureData,
                innerCoreRadius: innerCoreRadius,
                statistics: this.calculateStatistics(convectionData, magneticData, temperatureData)
            });
        }
    }
    
    generateConvectionPattern(time) {
        // Generate realistic convection pattern
        const resolution = this.config.convectionResolution;
        const convectionCells = [];
        
        // Create convection cells at various scales
        const numCells = 6; // Primary convection cells
        
        for (let i = 0; i < numCells; i++) {
            const theta = (i / numCells) * 2 * Math.PI;
            const cellStrength = 0.01 * (1 + 0.5 * Math.sin(time * 1e-8 + i));
            
            convectionCells.push({
                center: {
                    x: Math.cos(theta) * this.config.outerCoreRadius * 0.7,
                    y: Math.sin(theta) * this.config.outerCoreRadius * 0.7,
                    z: 0
                },
                strength: cellStrength,
                radius: this.config.outerCoreRadius * 0.3,
                direction: i % 2 === 0 ? 1 : -1 // Alternating circulation
            });
        }
        
        return { cells: convectionCells };
    }
    
    generateMagneticField(time, convectionData) {
        // Generate magnetic field from convection
        const fieldLines = [];
        const currentLoops = [];
        
        // Dipole field component
        const dipoleStrength = 8e22 * (0.8 + 0.4 * Math.sin(time * 2e-8));
        
        // Generate field lines
        for (let i = 0; i < 20; i++) {
            const theta = (i / 20) * Math.PI;
            const fieldLine = this.generateDipoleFieldLine(theta, dipoleStrength);
            fieldLines.push(fieldLine);
        }
        
        // Generate current loops from convection
        convectionData.cells.forEach((cell, index) => {
            currentLoops.push({
                center: cell.center,
                radius: cell.radius * 0.8,
                current: cell.strength * 1e6,
                direction: cell.direction
            });
        });
        
        return {
            dipoleStrength: dipoleStrength,
            fieldLines: fieldLines,
            currentLoops: currentLoops,
            reversalProbability: this.calculateReversalProbability(time, dipoleStrength)
        };
    }
    
    generateDipoleFieldLine(theta, strength) {
        const points = [];
        const numPoints = 100;
        
        for (let i = 0; i < numPoints; i++) {
            const phi = (i / numPoints) * 2 * Math.PI;
            
            // Dipole field line equation
            const r = this.config.outerCoreRadius * Math.sin(theta)**2;
            
            if (r > this.config.innerCoreRadius && r < this.config.outerCoreRadius) {
                points.push({
                    x: r * Math.sin(theta) * Math.cos(phi),
                    y: r * Math.sin(theta) * Math.sin(phi),
                    z: r * Math.cos(theta),
                    strength: strength * Math.cos(theta)
                });
            }
        }
        
        return points;
    }
    
    generateTemperatureField(time, convectionData) {
        // Generate temperature field affected by convection
        const baseTemperature = 5000; // K
        const temperatureVariation = 1000; // K
        
        const field = {
            baseTemp: baseTemperature,
            variation: temperatureVariation,
            hotPlumes: [],
            coldDownwellings: []
        };
        
        // Add thermal plumes and downwellings
        convectionData.cells.forEach(cell => {
            if (cell.direction > 0) {
                // Hot upwelling
                field.hotPlumes.push({
                    center: cell.center,
                    temperature: baseTemperature + temperatureVariation,
                    radius: cell.radius * 0.5
                });
            } else {
                // Cold downwelling
                field.coldDownwellings.push({
                    center: cell.center,
                    temperature: baseTemperature - temperatureVariation * 0.5,
                    radius: cell.radius * 0.6
                });
            }
        });
        
        return field;
    }
    
    calculateInnerCoreRadius(time) {
        // Inner core growth model
        const initialRadius = 0; // Started with no inner core
        const currentAge = 4.5e9; // Earth's age
        const nucleationAge = 1.5e9; // Inner core nucleation
        
        if (time < (currentAge - nucleationAge)) {
            return 0; // No inner core yet
        }
        
        // Linear growth model (simplified)
        const growthTime = time - (currentAge - nucleationAge);
        const growthRate = this.config.innerCoreRadius / nucleationAge;
        
        return Math.min(growthRate * growthTime, this.config.innerCoreRadius);
    }
    
    calculateStatistics(convection, magnetic, temperature) {
        // Calculate various statistics
        const totalConvectionEnergy = convection.cells.reduce(
            (sum, cell) => sum + cell.strength * cell.radius**2, 0
        );
        
        const heatFlow = totalConvectionEnergy * 1e12; // Convert to TW
        
        return {
            heatFlow: heatFlow,
            convectionVigor: totalConvectionEnergy,
            magneticDipole: magnetic.dipoleStrength,
            innerCoreGrowthRate: 1e-4 // mm/year
        };
    }
    
    calculateReversalProbability(time, dipoleStrength) {
        // Simple reversal probability model
        const baseReversalRate = 1e-6; // per year
        const strengthFactor = Math.exp(-dipoleStrength / 4e22);
        
        return baseReversalRate * strengthFactor;
    }
    
    updateVisualization() {
        if (this.simulationData.length === 0) return;
        
        const currentData = this.simulationData[this.currentTimeIndex];
        
        // Update convection visualization
        this.updateConvectionVisualization(currentData.convection);
        
        // Update magnetic field visualization
        this.updateMagneticVisualization(currentData.magneticField);
        
        // Update temperature field
        this.updateTemperatureVisualization(currentData.temperature);
        
        // Update inner core
        this.updateInnerCore(currentData.innerCoreRadius);
        
        // Update statistics display
        this.updateStatisticsDisplay(currentData.statistics);
        
        // Update time display
        this.updateTimeDisplay(currentData.time);
    }
    
    updateConvectionVisualization(convectionData) {
        // Clear existing convection cells
        this.convectionCells.forEach(cell => {
            this.scene.remove(cell);
            if (cell.geometry) cell.geometry.dispose();
            if (cell.material) cell.material.dispose();
        });
        this.convectionCells = [];
        
        // Create new convection cells
        convectionData.cells.forEach(cellData => {
            const cell = this.createConvectionCell(cellData);
            this.convectionCells.push(cell);
            this.scene.add(cell);
        });
    }
    
    createConvectionCell(cellData) {
        // Create torus geometry for convection cell
        const geometry = new THREE.TorusGeometry(
            cellData.radius * 0.8,
            cellData.radius * 0.2,
            16, 32
        );
        
        const material = new THREE.MeshPhongMaterial({
            color: cellData.direction > 0 ? 0xff4444 : 0x4444ff,
            transparent: true,
            opacity: Math.abs(cellData.strength) * 50,
            wireframe: true
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(cellData.center);
        
        // Add rotation animation
        mesh.userData = {
            rotationSpeed: cellData.strength * cellData.direction * 100
        };
        
        return mesh;
    }
    
    updateMagneticVisualization(magneticData) {
        // Clear existing magnetic field lines
        this.magneticFieldLines.forEach(line => {
            this.scene.remove(line);
            if (line.geometry) line.geometry.dispose();
            if (line.material) line.material.dispose();
        });
        this.magneticFieldLines = [];
        
        // Create new field lines
        magneticData.fieldLines.forEach(fieldLineData => {
            if (fieldLineData.length > 2) {
                const fieldLine = this.createMagneticFieldLine(fieldLineData);
                this.magneticFieldLines.push(fieldLine);
                this.scene.add(fieldLine);
            }
        });
        
        // Update current loops
        this.updateCurrentLoops(magneticData.currentLoops);
    }
    
    createMagneticFieldLine(fieldLineData) {
        const positions = [];
        const strengths = [];
        
        fieldLineData.forEach(point => {
            positions.push(point.x, point.y, point.z);
            strengths.push(point.strength);
        });
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('fieldStrength', new THREE.Float32BufferAttribute(strengths, 1));
        
        const material = new THREE.ShaderMaterial({
            vertexShader: this.magneticVertexShader,
            fragmentShader: this.magneticFragmentShader,
            uniforms: {
                time: this.uniforms.time,
                maxFieldStrength: this.uniforms.maxFieldStrength
            },
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        return new THREE.Line(geometry, material);
    }
    
    updateCurrentLoops(currentLoops) {
        // Clear existing current loops
        this.currentLoops.forEach(loop => {
            this.scene.remove(loop);
            if (loop.geometry) loop.geometry.dispose();
            if (loop.material) loop.material.dispose();
        });
        this.currentLoops = [];
        
        // Create new current loops
        currentLoops.forEach(loopData => {
            const loop = this.createCurrentLoop(loopData);
            this.currentLoops.push(loop);
            this.scene.add(loop);
        });
    }
    
    createCurrentLoop(loopData) {
        const geometry = new THREE.RingGeometry(
            loopData.radius * 0.95,
            loopData.radius * 1.05,
            32
        );
        
        const intensity = Math.abs(loopData.current) / 1e6;
        const material = new THREE.MeshBasicMaterial({
            color: loopData.direction > 0 ? 0xff6600 : 0x0066ff,
            transparent: true,
            opacity: Math.min(intensity, 0.8),
            side: THREE.DoubleSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(loopData.center);
        
        return mesh;
    }
    
    updateTemperatureVisualization(temperatureData) {
        // Update outer core material with temperature shader
        if (!this.outerCore.material) {
            this.outerCore.material = new THREE.ShaderMaterial({
                vertexShader: this.temperatureVertexShader,
                fragmentShader: this.temperatureFragmentShader,
                uniforms: {
                    time: this.uniforms.time,
                    minTemp: this.uniforms.minTemp,
                    maxTemp: this.uniforms.maxTemp,
                    cameraPosition: this.uniforms.cameraPosition
                },
                transparent: true,
                side: THREE.BackSide
            });
        }
        
        // Update temperature uniforms
        this.uniforms.minTemp.value = temperatureData.baseTemp - temperatureData.variation;
        this.uniforms.maxTemp.value = temperatureData.baseTemp + temperatureData.variation;
    }
    
    updateInnerCore(radius) {
        // Update inner core size
        if (radius > 0) {
            this.innerCore.scale.setScalar(radius / this.config.innerCoreRadius);
            this.innerCore.visible = true;
        } else {
            this.innerCore.visible = false;
        }
    }
    
    updateStatisticsDisplay(stats) {
        document.getElementById('heatFlow').textContent = stats.heatFlow.toFixed(1);
        document.getElementById('convectionVigor').textContent = stats.convectionVigor.toExponential(2);
        document.getElementById('magneticDipole').textContent = stats.magneticDipole.toExponential(2);
        
        const currentTime = this.simulationData[this.currentTimeIndex].time;
        const innerCoreAge = Math.max(0, currentTime - 1e9) / 1e6; // Myr
        document.getElementById('innerCoreAge').textContent = innerCoreAge.toFixed(1);
    }
    
    updateTimeDisplay(time) {
        const timeMyr = time / 1e6; // Convert to Myr
        document.getElementById('timeDisplay').textContent = timeMyr.toFixed(1) + ' Myr';
        document.getElementById('timeSlider').value = this.currentTimeIndex;
    }
    
    toggleAnimation() {
        this.isPlaying = !this.isPlaying;
        const button = document.getElementById('playPause');
        button.textContent = this.isPlaying ? '⏸️ Pause' : '⏯️ Play';
    }
    
    resetSimulation() {
        this.currentTimeIndex = 0;
        this.isPlaying = false;
        document.getElementById('playPause').textContent = '⏯️ Play';
        this.updateVisualization();
    }
    
    setTimeIndex(index) {
        this.currentTimeIndex = Math.max(0, Math.min(index, this.simulationData.length - 1));
        this.updateVisualization();
    }
    
    // Analysis methods
    analyzeFlowStructure() {
        if (this.simulationData.length === 0) return;
        
        const results = this.flowAnalyzer.analyze(this.simulationData[this.currentTimeIndex]);
        this.showAnalysisResults('Flow Structure Analysis', results);
    }
    
    analyzeThermalStructure() {
        if (this.simulationData.length === 0) return;
        
        const results = this.thermalAnalyzer.analyze(this.simulationData[this.currentTimeIndex]);
        this.showAnalysisResults('Thermal Structure Analysis', results);
    }
    
    analyzeMagneticField() {
        if (this.simulationData.length === 0) return;
        
        const results = this.magneticAnalyzer.analyze(this.simulationData[this.currentTimeIndex]);
        this.showAnalysisResults('Magnetic Field Analysis', results);
    }
    
    showAnalysisResults(title, results) {
        // Create modal for analysis results
        const modal = document.createElement('div');
        modal.className = 'analysis-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <h3>${title}</h3>
                <div class="analysis-results">
                    ${this.formatAnalysisResults(results)}
                </div>
                <button onclick="this.closest('.analysis-modal').remove()">Close</button>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    formatAnalysisResults(results) {
        return Object.entries(results)
            .map(([key, value]) => `
                <div class="result-item">
                    <strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}
                </div>
            `).join('');
    }
    
    exportSimulationData() {
        const exportData = {
            simulationData: this.simulationData,
            config: this.config,
            currentTime: this.currentTimeIndex
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                             { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'core_dynamics_simulation.json';
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    startRenderLoop() {
        const animate = (time) => {
            requestAnimationFrame(animate);
            
            // Update uniforms
            this.uniforms.time.value = time * 0.001;
            this.uniforms.cameraPosition.value.copy(this.camera.position);
            
            // Animation updates
            if (this.isPlaying && this.simulationData.length > 0) {
                const newIndex = this.currentTimeIndex + Math.floor(this.animationSpeed);
                
                if (newIndex >= this.simulationData.length) {
                    this.currentTimeIndex = 0;
                } else {
                    this.setTimeIndex(newIndex);
                }
            }
            
            // Rotate convection cells
            this.convectionCells.forEach(cell => {
                if (cell.userData.rotationSpeed) {
                    cell.rotation.z += cell.userData.rotationSpeed * 0.01;
                }
            });
            
            // Update controls
            this.controls.update();
            
            // Render scene
            this.renderer.render(this.scene, this.camera);
        };
        
        animate(0);
    }
}

// Analysis classes
class FlowAnalyzer {
    analyze(data) {
        const convection = data.convection;
        
        return {
            totalConvectionEnergy: convection.cells.reduce(
                (sum, cell) => sum + Math.abs(cell.strength) * cell.radius**2, 0
            ),
            numberOfCells: convection.cells.length,
            averageCellStrength: convection.cells.reduce(
                (sum, cell) => sum + Math.abs(cell.strength), 0
            ) / convection.cells.length,
            flowSymmetry: this.calculateFlowSymmetry(convection.cells)
        };
    }
    
    calculateFlowSymmetry(cells) {
        // Calculate how symmetric the flow pattern is
        let symmetryScore = 0;
        
        for (let i = 0; i < cells.length; i++) {
            for (let j = i + 1; j < cells.length; j++) {
                const distance = Math.sqrt(
                    Math.pow(cells[i].center.x - cells[j].center.x, 2) +
                    Math.pow(cells[i].center.y - cells[j].center.y, 2) +
                    Math.pow(cells[i].center.z - cells[j].center.z, 2)
                );
                
                const strengthDiff = Math.abs(cells[i].strength - cells[j].strength);
                symmetryScore += strengthDiff / distance;
            }
        }
        
        return 1 / (1 + symmetryScore);
    }
}

class ThermalAnalyzer {
    analyze(data) {
        const temperature = data.temperature;
        
        return {
            baseTemperature: temperature.baseTemp,
            temperatureVariation: temperature.variation,
            numberOfHotPlumes: temperature.hotPlumes.length,
            numberOfColdDownwellings: temperature.coldDownwellings.length,
            thermalBoundaryLayerThickness: 100000 // meters, simplified
        };
    }
}

class MagneticAnalyzer {
    analyze(data) {
        const magnetic = data.magneticField;
        
        return {
            dipoleStrength: magnetic.dipoleStrength,
            numberOfCurrentLoops: magnetic.currentLoops.length,
            totalMagneticEnergy: magnetic.currentLoops.reduce(
                (sum, loop) => sum + Math.abs(loop.current) * loop.radius, 0
            ),
            reversalProbability: magnetic.reversalProbability,
            fieldComplexity: magnetic.fieldLines.length
        };
    }
}

// CSS styles
const coreDynamicsStyles = `
.core-dynamics-controls {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 320px;
    background: rgba(0, 0, 0, 0.85);
    color: white;
    padding: 20px;
    border-radius: 12px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 14px;
    z-index: 1000;
    max-height: 90vh;
    overflow-y: auto;
}

.control-section {
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.control-section:last-child {
    border-bottom: none;
}

.control-section h3, .control-section h4 {
    margin: 0 0 12px 0;
    color: #f59e0b;
    font-weight: 600;
}

.simulation-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.simulation-controls button {
    padding: 8px 12px;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.3s;
}

.simulation-controls button:hover {
    background: #2563eb;
}

.speed-control {
    display: flex;
    align-items: center;
    gap: 8px;
}

.checkbox-group {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.checkbox-group label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 13px;
}

.property-sliders {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.slider-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.slider-group label {
    font-size: 12px;
    color: #d1d5db;
}

.analysis-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.analysis-buttons button {
    padding: 8px 10px;
    background: #059669;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 11px;
    transition: background 0.3s;
}

.analysis-buttons button:hover {
    background: #047857;
}

#coreStats {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: rgba(0, 0, 0, 0.3);
    padding: 10px;
    border-radius: 6px;
}

#coreStats div {
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}

.analysis-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 2000;
}

.modal-content {
    background: white;
    padding: 30px;
    border-radius: 12px;
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
}

.analysis-results {
    margin: 20px 0;
}

.result-item {
    margin-bottom: 10px;
    padding: 8px;
    background: #f3f4f6;
    border-radius: 4px;
    border-left: 3px solid #3b82f6;
}

input[type="range"] {
    width: 100%;
    margin: 5px 0;
}
`;

// Inject CSS
const styleSheet = document.createElement('style');
styleSheet.textContent = coreDynamicsStyles;
document.head.appendChild(styleSheet);

// Export
window.CoreDynamicsVisualization = CoreDynamicsVisualization;