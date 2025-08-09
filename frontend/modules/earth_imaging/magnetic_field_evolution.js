/**
 * Advanced Magnetic Field Evolution Visualization System
 * Real-time 3D/4D visualization of geomagnetic field dynamics and reversals
 * Integrates with geodynamo simulation data for comprehensive analysis
 */

class MagneticFieldEvolution {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.config = this._mergeConfig(config);
        
        // Core visualization components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.earthMesh = null;
        this.fieldLines = [];
        this.dipoleLines = [];
        this.fieldVectors = [];
        
        // Data management
        this.timeSeriesData = [];
        this.currentTimeIndex = 0;
        this.isPlaying = false;
        this.animationSpeed = 1.0;
        this.fieldHistory = new Map();
        
        // Magnetic field analysis
        this.sphericalHarmonics = new SphericalHarmonicsAnalyzer();
        this.reversalDetector = new ReversalDetector();
        this.intensityAnalyzer = new IntensityAnalyzer();
        
        // UI components
        this.controlPanel = null;
        this.timelineSlider = null;
        this.playButton = null;
        this.speedControl = null;
        
        this.initialize();
    }
    
    _mergeConfig(config) {
        const defaults = {
            earthRadius: 6371000,
            coreRadius: 3480000,
            innerCoreRadius: 1216000,
            fieldLineCount: 200,
            vectorFieldResolution: 32,
            maxDegree: 13,
            colorScheme: 'magnetic',
            showVectors: true,
            showFieldLines: true,
            showDipole: true,
            showCore: true,
            animation: {
                autoPlay: false,
                speed: 1.0,
                loop: true
            },
            visualization: {
                fieldLineOpacity: 0.8,
                vectorOpacity: 0.6,
                coreOpacity: 0.3,
                useShaders: true,
                enableVolumeRendering: true
            }
        };
        
        return { ...defaults, ...config };
    }
    
    initialize() {
        this.setupThreeJS();
        this.setupEarth();
        this.setupUI();
        this.setupEventListeners();
        this.setupShaders();
        this.loadInitialData();
        this.startRenderLoop();
    }
    
    setupThreeJS() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000011);
        
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75, 
            this.container.clientWidth / this.container.clientHeight, 
            0.1, 
            1000000000
        );
        this.camera.position.set(15000000, 10000000, 15000000);
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 8000000;
        this.controls.maxDistance = 50000000;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(20000000, 20000000, 10000000);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
    }
    
    setupEarth() {
        // Earth geometry and materials
        const earthGeometry = new THREE.SphereGeometry(this.config.earthRadius, 64, 32);
        
        // Earth texture and material
        const earthMaterial = new THREE.MeshPhongMaterial({
            map: new THREE.TextureLoader().load('/assets/textures/earth_day.jpg'),
            transparent: true,
            opacity: 0.8
        });
        
        this.earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
        this.earthMesh.receiveShadow = true;
        this.scene.add(this.earthMesh);
        
        // Core visualization
        if (this.config.showCore) {
            this.setupCore();
        }
        
        // Coordinate system axes
        this.setupAxes();
    }
    
    setupCore() {
        // Outer core
        const outerCoreGeometry = new THREE.SphereGeometry(this.config.coreRadius, 32, 16);
        const outerCoreMaterial = new THREE.MeshPhongMaterial({
            color: 0xff6600,
            transparent: true,
            opacity: this.config.visualization.coreOpacity,
            wireframe: true
        });
        
        this.outerCoreMesh = new THREE.Mesh(outerCoreGeometry, outerCoreMaterial);
        this.scene.add(this.outerCoreMesh);
        
        // Inner core
        const innerCoreGeometry = new THREE.SphereGeometry(this.config.innerCoreRadius, 16, 8);
        const innerCoreMaterial = new THREE.MeshPhongMaterial({
            color: 0xffaa00,
            transparent: true,
            opacity: this.config.visualization.coreOpacity + 0.2
        });
        
        this.innerCoreMesh = new THREE.Mesh(innerCoreGeometry, innerCoreMaterial);
        this.scene.add(this.innerCoreMesh);
    }
    
    setupAxes() {
        // Rotation axis
        const axisGeometry = new THREE.CylinderGeometry(50000, 50000, this.config.earthRadius * 3);
        const axisMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
        
        this.rotationAxis = new THREE.Mesh(axisGeometry, axisMaterial);
        this.rotationAxis.rotation.z = Math.PI / 2;
        this.scene.add(this.rotationAxis);
        
        // Magnetic dipole axis (will be updated dynamically)
        this.magneticAxis = new THREE.Mesh(axisGeometry.clone(), 
            new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.5 })
        );
        this.scene.add(this.magneticAxis);
    }
    
    setupShaders() {
        // Vertex shader for magnetic field lines
        this.fieldLineVertexShader = `
            attribute float intensity;
            attribute vec3 direction;
            varying float vIntensity;
            varying vec3 vDirection;
            varying vec3 vPosition;
            
            void main() {
                vIntensity = intensity;
                vDirection = direction;
                vPosition = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        // Fragment shader for magnetic field lines
        this.fieldLineFragmentShader = `
            uniform float time;
            uniform float maxIntensity;
            varying float vIntensity;
            varying vec3 vDirection;
            varying vec3 vPosition;
            
            vec3 intensityToColor(float intensity) {
                float normalized = intensity / maxIntensity;
                if (normalized < 0.5) {
                    return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), normalized * 2.0);
                } else {
                    return mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 1.0, 0.0), (normalized - 0.5) * 2.0);
                }
            }
            
            void main() {
                vec3 color = intensityToColor(abs(vIntensity));
                
                // Add flowing animation
                float flow = sin(time * 2.0 + length(vPosition) * 0.001) * 0.5 + 0.5;
                color *= (0.7 + 0.3 * flow);
                
                // Add polarity indication
                if (vIntensity < 0.0) {
                    color = mix(color, vec3(1.0, 0.0, 0.0), 0.3);
                }
                
                gl_FragColor = vec4(color, ${this.config.visualization.fieldLineOpacity});
            }
        `;
        
        // Volume rendering shader for field intensity
        this.volumeVertexShader = `
            varying vec3 vPosition;
            varying vec3 vNormal;
            
            void main() {
                vPosition = position;
                vNormal = normal;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        this.volumeFragmentShader = `
            uniform sampler3D fieldTexture;
            uniform float threshold;
            uniform vec3 cameraPosition;
            varying vec3 vPosition;
            varying vec3 vNormal;
            
            void main() {
                vec3 rayDir = normalize(vPosition - cameraPosition);
                vec3 rayPos = vPosition;
                
                float stepSize = 0.01;
                float totalIntensity = 0.0;
                vec3 color = vec3(0.0);
                
                for (int i = 0; i < 100; i++) {
                    vec3 samplePos = (rayPos + vec3(1.0)) * 0.5; // Normalize to [0,1]
                    float intensity = texture3D(fieldTexture, samplePos).r;
                    
                    if (intensity > threshold) {
                        vec3 intensityColor = vec3(intensity, 1.0 - intensity, 0.5);
                        color += intensityColor * intensity * stepSize;
                        totalIntensity += intensity * stepSize;
                    }
                    
                    rayPos += rayDir * stepSize;
                    
                    if (length(rayPos) > 2.0) break; // Exit sphere bounds
                }
                
                gl_FragColor = vec4(color, min(totalIntensity, 1.0));
            }
        `;
    }
    
    setupUI() {
        // Create control panel
        this.controlPanel = document.createElement('div');
        this.controlPanel.className = 'magnetic-field-controls';
        this.controlPanel.innerHTML = `
            <div class="control-section">
                <h3>Magnetic Field Evolution</h3>
                <div class="time-controls">
                    <button id="playPause">⏯️</button>
                    <input type="range" id="timeSlider" min="0" max="100" value="0">
                    <span id="timeDisplay">0 ka</span>
                </div>
                <div class="speed-control">
                    <label>Speed: </label>
                    <input type="range" id="speedControl" min="0.1" max="5" step="0.1" value="1">
                    <span id="speedDisplay">1.0x</span>
                </div>
            </div>
            
            <div class="control-section">
                <h4>Visualization Options</h4>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="showFieldLines" checked> Field Lines</label>
                    <label><input type="checkbox" id="showVectors" checked> Vector Field</label>
                    <label><input type="checkbox" id="showDipole" checked> Dipole</label>
                    <label><input type="checkbox" id="showCore" checked> Core</label>
                </div>
                <div class="opacity-controls">
                    <label>Field Line Opacity: </label>
                    <input type="range" id="fieldOpacity" min="0.1" max="1" step="0.1" value="0.8">
                    <label>Vector Opacity: </label>
                    <input type="range" id="vectorOpacity" min="0.1" max="1" step="0.1" value="0.6">
                </div>
            </div>
            
            <div class="control-section">
                <h4>Analysis Tools</h4>
                <div class="analysis-buttons">
                    <button id="analyzeReversals">Detect Reversals</button>
                    <button id="computeSpectrum">Power Spectrum</button>
                    <button id="exportData">Export Data</button>
                </div>
            </div>
            
            <div class="control-section">
                <h4>Field Properties</h4>
                <div id="fieldProperties">
                    <div>Intensity: <span id="intensityValue">--</span> nT</div>
                    <div>Inclination: <span id="inclinationValue">--</span>°</div>
                    <div>Declination: <span id="declinationValue">--</span>°</div>
                    <div>Dipole Moment: <span id="dipoleValue">--</span> Am²</div>
                </div>
            </div>
        `;
        
        this.container.appendChild(this.controlPanel);
        
        // Get UI elements
        this.playButton = document.getElementById('playPause');
        this.timelineSlider = document.getElementById('timeSlider');
        this.speedControl = document.getElementById('speedControl');
        this.timeDisplay = document.getElementById('timeDisplay');
        this.speedDisplay = document.getElementById('speedDisplay');
    }
    
    setupEventListeners() {
        // Play/pause button
        this.playButton.addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        // Timeline slider
        this.timelineSlider.addEventListener('input', (e) => {
            this.setTimeIndex(parseInt(e.target.value));
        });
        
        // Speed control
        this.speedControl.addEventListener('input', (e) => {
            this.animationSpeed = parseFloat(e.target.value);
            this.speedDisplay.textContent = this.animationSpeed.toFixed(1) + 'x';
        });
        
        // Visualization toggles
        document.getElementById('showFieldLines').addEventListener('change', (e) => {
            this.toggleFieldLines(e.target.checked);
        });
        
        document.getElementById('showVectors').addEventListener('change', (e) => {
            this.toggleVectorField(e.target.checked);
        });
        
        document.getElementById('showDipole').addEventListener('change', (e) => {
            this.toggleDipole(e.target.checked);
        });
        
        document.getElementById('showCore').addEventListener('change', (e) => {
            this.toggleCore(e.target.checked);
        });
        
        // Opacity controls
        document.getElementById('fieldOpacity').addEventListener('input', (e) => {
            this.setFieldLineOpacity(parseFloat(e.target.value));
        });
        
        document.getElementById('vectorOpacity').addEventListener('input', (e) => {
            this.setVectorOpacity(parseFloat(e.target.value));
        });
        
        // Analysis buttons
        document.getElementById('analyzeReversals').addEventListener('click', () => {
            this.analyzeReversals();
        });
        
        document.getElementById('computeSpectrum').addEventListener('click', () => {
            this.computePowerSpectrum();
        });
        
        document.getElementById('exportData').addEventListener('click', () => {
            this.exportData();
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }
    
    loadInitialData() {
        // Load sample magnetic field data
        this.loadSampleData();
        
        // Initialize visualization
        this.updateVisualization();
    }
    
    loadSampleData() {
        // Generate sample data for demonstration
        // In production, this would load from the geodynamo simulator
        const timeSteps = 1000;
        const timeSpan = 1000000; // 1 million years
        
        for (let i = 0; i < timeSteps; i++) {
            const time = (i / timeSteps) * timeSpan;
            const fieldData = this.generateSampleFieldData(time);
            
            this.timeSeriesData.push({
                time: time,
                field: fieldData,
                intensity: this.calculateIntensity(fieldData),
                dipoleAxis: this.calculateDipoleAxis(fieldData),
                polarity: this.determinePolarity(fieldData)
            });
        }
        
        // Update timeline slider
        this.timelineSlider.max = this.timeSeriesData.length - 1;
    }
    
    generateSampleFieldData(time) {
        // Generate realistic magnetic field evolution
        const fieldData = {
            sphericalHarmonics: {},
            reversalProbability: 0,
            isReversing: false
        };
        
        // Main dipole component (varies with time)
        const dipoleStrength = 8e22 * (0.8 + 0.4 * Math.sin(time * 2 * Math.PI / 200000));
        const dipoleInclination = Math.PI * (0.5 + 0.1 * Math.sin(time * 2 * Math.PI / 50000));
        
        // Add reversals
        if (Math.random() < 0.001) { // Low probability of reversal
            fieldData.isReversing = true;
            fieldData.reversalProbability = 1.0;
        }
        
        // Generate spherical harmonic coefficients
        for (let n = 1; n <= this.config.maxDegree; n++) {
            for (let m = 0; m <= n; m++) {
                const baseAmplitude = dipoleStrength / Math.pow(n, 3);
                const timeVariation = Math.sin(time * 2 * Math.PI / (100000 * n));
                
                fieldData.sphericalHarmonics[`g_${n}_${m}`] = baseAmplitude * timeVariation;
                if (m > 0) {
                    fieldData.sphericalHarmonics[`h_${n}_${m}`] = baseAmplitude * timeVariation * 0.5;
                }
            }
        }
        
        return fieldData;
    }
    
    calculateIntensity(fieldData) {
        // Calculate total field intensity from spherical harmonics
        let intensity = 0;
        
        for (const [key, value] of Object.entries(fieldData.sphericalHarmonics)) {
            intensity += value * value;
        }
        
        return Math.sqrt(intensity);
    }
    
    calculateDipoleAxis(fieldData) {
        // Extract dipole axis from g₁⁰, g₁¹, h₁¹ coefficients
        const g10 = fieldData.sphericalHarmonics.g_1_0 || 0;
        const g11 = fieldData.sphericalHarmonics.g_1_1 || 0;
        const h11 = fieldData.sphericalHarmonics.h_1_1 || 0;
        
        // Calculate axis orientation
        const theta = Math.acos(-g10 / Math.sqrt(g10*g10 + g11*g11 + h11*h11));
        const phi = Math.atan2(h11, g11);
        
        return { theta, phi, strength: Math.sqrt(g10*g10 + g11*g11 + h11*h11) };
    }
    
    determinePolarity(fieldData) {
        const g10 = fieldData.sphericalHarmonics.g_1_0 || 0;
        return g10 > 0 ? 1 : -1; // Normal or reversed
    }
    
    updateVisualization() {
        if (this.timeSeriesData.length === 0) return;
        
        const currentData = this.timeSeriesData[this.currentTimeIndex];
        
        // Update field lines
        if (this.config.showFieldLines) {
            this.updateFieldLines(currentData.field);
        }
        
        // Update vector field
        if (this.config.showVectors) {
            this.updateVectorField(currentData.field);
        }
        
        // Update dipole visualization
        if (this.config.showDipole) {
            this.updateDipoleVisualization(currentData.dipoleAxis);
        }
        
        // Update UI displays
        this.updateFieldProperties(currentData);
        this.updateTimeDisplay(currentData.time);
    }
    
    updateFieldLines(fieldData) {
        // Clear existing field lines
        this.fieldLines.forEach(line => {
            this.scene.remove(line);
            line.geometry.dispose();
            line.material.dispose();
        });
        this.fieldLines = [];
        
        // Generate new field lines
        const fieldLineData = this.generateFieldLines(fieldData);
        
        fieldLineData.forEach(lineData => {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(lineData.positions, 3));
            geometry.setAttribute('intensity', new THREE.Float32BufferAttribute(lineData.intensities, 1));
            
            const material = new THREE.ShaderMaterial({
                vertexShader: this.fieldLineVertexShader,
                fragmentShader: this.fieldLineFragmentShader,
                uniforms: {
                    time: { value: 0 },
                    maxIntensity: { value: Math.max(...lineData.intensities) }
                },
                transparent: true
            });
            
            const fieldLine = new THREE.Line(geometry, material);
            this.fieldLines.push(fieldLine);
            this.scene.add(fieldLine);
        });
    }
    
    generateFieldLines(fieldData) {
        const fieldLines = [];
        const numLines = this.config.fieldLineCount;
        
        // Generate field line starting points
        for (let i = 0; i < numLines; i++) {
            const theta = Math.random() * Math.PI;
            const phi = Math.random() * 2 * Math.PI;
            
            const startPoint = this.sphericalToCartesian(
                this.config.earthRadius * 1.1, theta, phi
            );
            
            const fieldLine = this.traceFieldLine(startPoint, fieldData);
            if (fieldLine.positions.length > 6) { // Minimum line length
                fieldLines.push(fieldLine);
            }
        }
        
        return fieldLines;
    }
    
    traceFieldLine(startPoint, fieldData, maxSteps = 1000) {
        const positions = [];
        const intensities = [];
        
        let currentPoint = [...startPoint];
        let stepSize = 50000; // 50 km steps
        
        for (let step = 0; step < maxSteps; step++) {
            positions.push(...currentPoint);
            
            // Calculate magnetic field at current point
            const field = this.calculateFieldAtPoint(currentPoint, fieldData);
            const intensity = Math.sqrt(field.Br*field.Br + field.Btheta*field.Btheta + field.Bphi*field.Bphi);
            intensities.push(intensity);
            
            // Check termination conditions
            const radius = Math.sqrt(currentPoint[0]**2 + currentPoint[1]**2 + currentPoint[2]**2);
            if (radius < this.config.earthRadius * 0.9 || radius > this.config.earthRadius * 10) {
                break;
            }
            
            if (intensity < 1e-9) break; // Field too weak
            
            // Move along field line
            const fieldDirection = this.cartesianToSpherical(field.Br, field.Btheta, field.Bphi);
            const nextPoint = [
                currentPoint[0] + stepSize * fieldDirection.x / intensity,
                currentPoint[1] + stepSize * fieldDirection.y / intensity,
                currentPoint[2] + stepSize * fieldDirection.z / intensity
            ];
            
            currentPoint = nextPoint;
        }
        
        return { positions, intensities };
    }
    
    calculateFieldAtPoint(point, fieldData) {
        // Convert to spherical coordinates
        const [x, y, z] = point;
        const r = Math.sqrt(x*x + y*y + z*z);
        const theta = Math.acos(z / r);
        const phi = Math.atan2(y, x);
        
        // Calculate field from spherical harmonics
        let Br = 0, Btheta = 0, Bphi = 0;
        
        for (let n = 1; n <= this.config.maxDegree; n++) {
            for (let m = 0; m <= n; m++) {
                const gnm = fieldData.sphericalHarmonics[`g_${n}_${m}`] || 0;
                const hnm = fieldData.sphericalHarmonics[`h_${n}_${m}`] || 0;
                
                // Associated Legendre polynomial (simplified)
                const Pnm = this.legendrePolynomial(n, m, Math.cos(theta));
                const dPnm = this.legendreDerivative(n, m, theta);
                
                // Radial factor
                const rFactor = Math.pow(this.config.earthRadius / r, n + 2);
                
                // Trigonometric terms
                const cosmPhi = Math.cos(m * phi);
                const sinmPhi = Math.sin(m * phi);
                
                // Field components
                Br += rFactor * (n + 1) * Pnm * (gnm * cosmPhi + hnm * sinmPhi);
                Btheta += rFactor * dPnm * (gnm * cosmPhi + hnm * sinmPhi);
                
                if (m > 0) {
                    Bphi += rFactor * m * Pnm / Math.sin(theta) * (gnm * sinmPhi - hnm * cosmPhi);
                }
            }
        }
        
        return { Br, Btheta, Bphi };
    }
    
    legendrePolynomial(n, m, x) {
        // Simplified associated Legendre polynomial calculation
        // In production, use a proper implementation
        if (n === 1 && m === 0) return x;
        if (n === 1 && m === 1) return Math.sqrt(1 - x*x);
        if (n === 2 && m === 0) return 0.5 * (3*x*x - 1);
        
        // For higher orders, use recurrence relations
        return Math.pow(1 - x*x, m/2) * Math.pow(x, n-m);
    }
    
    legendreDerivative(n, m, theta) {
        // Derivative of associated Legendre polynomial
        // Simplified implementation
        const delta = 1e-6;
        const Pplus = this.legendrePolynomial(n, m, Math.cos(theta + delta));
        const Pminus = this.legendrePolynomial(n, m, Math.cos(theta - delta));
        
        return (Pplus - Pminus) / (2 * delta) * (-Math.sin(theta));
    }
    
    updateVectorField(fieldData) {
        // Clear existing vectors
        this.fieldVectors.forEach(vector => {
            this.scene.remove(vector);
            vector.geometry.dispose();
            vector.material.dispose();
        });
        this.fieldVectors = [];
        
        // Generate vector field
        const resolution = this.config.vectorFieldResolution;
        
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const theta = (i / (resolution - 1)) * Math.PI;
                const phi = (j / (resolution - 1)) * 2 * Math.PI;
                
                const position = this.sphericalToCartesian(
                    this.config.earthRadius * 1.5, theta, phi
                );
                
                const field = this.calculateFieldAtPoint(position, fieldData);
                const intensity = Math.sqrt(field.Br*field.Br + field.Btheta*field.Btheta + field.Bphi*field.Bphi);
                
                if (intensity > 1e-8) {
                    const vector = this.createFieldVector(position, field, intensity);
                    this.fieldVectors.push(vector);
                    this.scene.add(vector);
                }
            }
        }
    }
    
    createFieldVector(position, field, intensity) {
        // Create arrow geometry for field vector
        const direction = new THREE.Vector3(field.Br, field.Btheta, field.Bphi).normalize();
        const length = Math.log10(intensity + 1) * 500000; // Logarithmic scaling
        
        const arrowHelper = new THREE.ArrowHelper(
            direction,
            new THREE.Vector3(...position),
            length,
            this.intensityToColor(intensity),
            length * 0.2,
            length * 0.1
        );
        
        arrowHelper.line.material.transparent = true;
        arrowHelper.line.material.opacity = this.config.visualization.vectorOpacity;
        arrowHelper.cone.material.transparent = true;
        arrowHelper.cone.material.opacity = this.config.visualization.vectorOpacity;
        
        return arrowHelper;
    }
    
    intensityToColor(intensity) {
        // Convert field intensity to color
        const logIntensity = Math.log10(intensity + 1e-12);
        const normalized = (logIntensity + 12) / 6; // Normalize to [0,1]
        
        if (normalized < 0.5) {
            return new THREE.Color().setHSL(0.7, 1.0, 0.3 + normalized);
        } else {
            return new THREE.Color().setHSL(0.1, 1.0, normalized);
        }
    }
    
    updateDipoleVisualization(dipoleAxis) {
        // Update magnetic dipole axis visualization
        if (this.magneticAxis) {
            // Rotate axis to match dipole orientation
            this.magneticAxis.rotation.x = dipoleAxis.theta;
            this.magneticAxis.rotation.y = dipoleAxis.phi;
            
            // Scale by dipole strength
            const scale = Math.log10(dipoleAxis.strength + 1) * 0.1;
            this.magneticAxis.scale.set(scale, 1, scale);
        }
    }
    
    updateFieldProperties(currentData) {
        // Update field property displays
        document.getElementById('intensityValue').textContent = 
            (currentData.intensity * 1e9).toFixed(0);
        
        const inclination = Math.acos(currentData.dipoleAxis.theta) * 180 / Math.PI - 90;
        document.getElementById('inclinationValue').textContent = inclination.toFixed(1);
        
        const declination = currentData.dipoleAxis.phi * 180 / Math.PI;
        document.getElementById('declinationValue').textContent = declination.toFixed(1);
        
        document.getElementById('dipoleValue').textContent = 
            currentData.dipoleAxis.strength.toExponential(2);
    }
    
    updateTimeDisplay(time) {
        // Update time display
        const timeKa = (time / 1000).toFixed(1);
        this.timeDisplay.textContent = timeKa + ' ka';
        
        // Update slider position
        this.timelineSlider.value = this.currentTimeIndex;
    }
    
    toggleAnimation() {
        this.isPlaying = !this.isPlaying;
        this.playButton.textContent = this.isPlaying ? '⏸️' : '⏯️';
    }
    
    setTimeIndex(index) {
        this.currentTimeIndex = Math.max(0, Math.min(index, this.timeSeriesData.length - 1));
        this.updateVisualization();
    }
    
    toggleFieldLines(show) {
        this.fieldLines.forEach(line => {
            line.visible = show;
        });
        this.config.showFieldLines = show;
    }
    
    toggleVectorField(show) {
        this.fieldVectors.forEach(vector => {
            vector.visible = show;
        });
        this.config.showVectors = show;
    }
    
    toggleDipole(show) {
        if (this.magneticAxis) {
            this.magneticAxis.visible = show;
        }
        this.config.showDipole = show;
    }
    
    toggleCore(show) {
        if (this.outerCoreMesh) this.outerCoreMesh.visible = show;
        if (this.innerCoreMesh) this.innerCoreMesh.visible = show;
        this.config.showCore = show;
    }
    
    setFieldLineOpacity(opacity) {
        this.fieldLines.forEach(line => {
            line.material.uniforms.opacity = { value: opacity };
        });
        this.config.visualization.fieldLineOpacity = opacity;
    }
    
    setVectorOpacity(opacity) {
        this.fieldVectors.forEach(vector => {
            vector.line.material.opacity = opacity;
            vector.cone.material.opacity = opacity;
        });
        this.config.visualization.vectorOpacity = opacity;
    }
    
    analyzeReversals() {
        // Detect magnetic reversals in time series
        const reversals = this.reversalDetector.detectReversals(this.timeSeriesData);
        
        // Display results
        this.showAnalysisResults('Magnetic Reversals', {
            count: reversals.length,
            averageInterval: this.calculateAverageReversalInterval(reversals),
            reversalTimes: reversals.map(r => r.time / 1000) // Convert to ka
        });
    }
    
    calculateAverageReversalInterval(reversals) {
        if (reversals.length < 2) return 'N/A';
        
        let totalInterval = 0;
        for (let i = 1; i < reversals.length; i++) {
            totalInterval += reversals[i].time - reversals[i-1].time;
        }
        
        return (totalInterval / (reversals.length - 1) / 1000).toFixed(1) + ' ka';
    }
    
    computePowerSpectrum() {
        // Compute power spectrum of field variations
        const intensities = this.timeSeriesData.map(d => d.intensity);
        const spectrum = this.intensityAnalyzer.computePowerSpectrum(intensities);
        
        // Display results
        this.showAnalysisResults('Power Spectrum', spectrum);
    }
    
    exportData() {
        // Export current visualization data
        const exportData = {
            timeSeriesData: this.timeSeriesData,
            config: this.config,
            currentTime: this.timeSeriesData[this.currentTimeIndex]?.time || 0
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                             { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'magnetic_field_data.json';
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    showAnalysisResults(title, results) {
        // Create modal dialog for analysis results
        const modal = document.createElement('div');
        modal.className = 'analysis-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <h3>${title}</h3>
                <pre>${JSON.stringify(results, null, 2)}</pre>
                <button onclick="this.closest('.analysis-modal').remove()">Close</button>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    sphericalToCartesian(r, theta, phi) {
        return [
            r * Math.sin(theta) * Math.cos(phi),
            r * Math.sin(theta) * Math.sin(phi),
            r * Math.cos(theta)
        ];
    }
    
    cartesianToSpherical(x, y, z) {
        const r = Math.sqrt(x*x + y*y + z*z);
        const theta = Math.acos(z / r);
        const phi = Math.atan2(y, x);
        return { x, y, z, r, theta, phi };
    }
    
    handleResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    startRenderLoop() {
        const animate = (time) => {
            requestAnimationFrame(animate);
            
            // Update time-dependent effects
            if (this.fieldLines.length > 0) {
                this.fieldLines.forEach(line => {
                    if (line.material.uniforms) {
                        line.material.uniforms.time.value = time * 0.001;
                    }
                });
            }
            
            // Animation updates
            if (this.isPlaying && this.timeSeriesData.length > 0) {
                const newIndex = this.currentTimeIndex + Math.floor(this.animationSpeed);
                
                if (newIndex >= this.timeSeriesData.length) {
                    if (this.config.animation.loop) {
                        this.currentTimeIndex = 0;
                    } else {
                        this.isPlaying = false;
                        this.playButton.textContent = '⏯️';
                    }
                } else {
                    this.setTimeIndex(newIndex);
                }
            }
            
            // Update controls
            this.controls.update();
            
            // Render scene
            this.renderer.render(this.scene, this.camera);
        };
        
        animate(0);
    }
    
    loadGeodynamoData(dataUrl) {
        // Load data from geodynamo simulator
        fetch(dataUrl)
            .then(response => response.json())
            .then(data => {
                this.timeSeriesData = this.processGeodynamoData(data);
                this.timelineSlider.max = this.timeSeriesData.length - 1;
                this.updateVisualization();
            })
            .catch(error => {
                console.error('Error loading geodynamo data:', error);
            });
    }
    
    processGeodynamoData(rawData) {
        // Convert raw geodynamo simulation data to visualization format
        return rawData.timeSteps.map(step => ({
            time: step.time,
            field: {
                sphericalHarmonics: step.sphericalHarmonics,
                reversalProbability: step.reversalProbability || 0,
                isReversing: step.isReversing || false
            },
            intensity: this.calculateIntensity(step),
            dipoleAxis: this.calculateDipoleAxis(step),
            polarity: this.determinePolarity(step)
        }));
    }
}

// Supporting classes for analysis

class SphericalHarmonicsAnalyzer {
    constructor() {
        this.maxDegree = 13;
    }
    
    analyzeSpectrum(coefficients) {
        // Analyze spherical harmonic spectrum
        const spectrum = {};
        
        for (let n = 1; n <= this.maxDegree; n++) {
            let power = 0;
            for (let m = 0; m <= n; m++) {
                const gnm = coefficients[`g_${n}_${m}`] || 0;
                const hnm = coefficients[`h_${n}_${m}`] || 0;
                power += gnm*gnm + hnm*hnm;
            }
            spectrum[n] = Math.sqrt(power);
        }
        
        return spectrum;
    }
    
    computeMagneticDipoleVector(coefficients) {
        const g10 = coefficients.g_1_0 || 0;
        const g11 = coefficients.g_1_1 || 0;
        const h11 = coefficients.h_1_1 || 0;
        
        return {
            x: -g11,
            y: -h11,
            z: -g10,
            magnitude: Math.sqrt(g10*g10 + g11*g11 + h11*h11)
        };
    }
}

class ReversalDetector {
    constructor() {
        this.threshold = 0.1; // Reversal detection threshold
    }
    
    detectReversals(timeSeries) {
        const reversals = [];
        let lastPolarity = timeSeries[0]?.polarity || 1;
        
        for (let i = 1; i < timeSeries.length; i++) {
            const currentPolarity = timeSeries[i].polarity;
            
            if (currentPolarity !== lastPolarity) {
                reversals.push({
                    time: timeSeries[i].time,
                    index: i,
                    fromPolarity: lastPolarity,
                    toPolarity: currentPolarity,
                    intensity: timeSeries[i].intensity
                });
                
                lastPolarity = currentPolarity;
            }
        }
        
        return reversals;
    }
    
    analyzeReversalStatistics(reversals) {
        if (reversals.length < 2) {
            return { averageInterval: null, frequency: 0 };
        }
        
        const intervals = [];
        for (let i = 1; i < reversals.length; i++) {
            intervals.push(reversals[i].time - reversals[i-1].time);
        }
        
        const averageInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const frequency = 1 / averageInterval;
        
        return {
            averageInterval,
            frequency,
            intervals,
            standardDeviation: this.calculateStandardDeviation(intervals)
        };
    }
    
    calculateStandardDeviation(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
        return Math.sqrt(avgSquaredDiff);
    }
}

class IntensityAnalyzer {
    computePowerSpectrum(intensities) {
        // Simple FFT-based power spectrum computation
        const N = intensities.length;
        const spectrum = new Array(N/2);
        
        for (let k = 0; k < N/2; k++) {
            let real = 0, imag = 0;
            
            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                real += intensities[n] * Math.cos(angle);
                imag += intensities[n] * Math.sin(angle);
            }
            
            spectrum[k] = Math.sqrt(real*real + imag*imag);
        }
        
        return spectrum;
    }
    
    computeStatistics(intensities) {
        const mean = intensities.reduce((a, b) => a + b, 0) / intensities.length;
        const variance = intensities.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / intensities.length;
        const stdDev = Math.sqrt(variance);
        
        const sorted = [...intensities].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        
        return {
            mean,
            median,
            standardDeviation: stdDev,
            min: Math.min(...intensities),
            max: Math.max(...intensities),
            range: Math.max(...intensities) - Math.min(...intensities)
        };
    }
}

// CSS styles for the visualization
const magneticFieldStyles = `
.magnetic-field-controls {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 300px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 20px;
    border-radius: 10px;
    font-family: 'Arial', sans-serif;
    z-index: 1000;
}

.control-section {
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.3);
    padding-bottom: 15px;
}

.control-section:last-child {
    border-bottom: none;
}

.control-section h3, .control-section h4 {
    margin: 0 0 10px 0;
    color: #60a5fa;
}

.time-controls, .speed-control {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.checkbox-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.checkbox-group label {
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
}

.opacity-controls {
    margin-top: 10px;
}

.opacity-controls label {
    display: block;
    margin-bottom: 5px;
}

.analysis-buttons {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.analysis-buttons button {
    padding: 8px 12px;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.3s;
}

.analysis-buttons button:hover {
    background: #2563eb;
}

#fieldProperties {
    font-family: 'Courier New', monospace;
    font-size: 12px;
}

#fieldProperties div {
    margin-bottom: 5px;
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
    padding: 20px;
    border-radius: 10px;
    max-width: 80%;
    max-height: 80%;
    overflow: auto;
}

.modal-content pre {
    background: #f5f5f5;
    padding: 10px;
    border-radius: 5px;
    overflow: auto;
}
`;

// Inject CSS styles
const styleSheet = document.createElement('style');
styleSheet.textContent = magneticFieldStyles;
document.head.appendChild(styleSheet);

// Export for use
window.MagneticFieldEvolution = MagneticFieldEvolution;