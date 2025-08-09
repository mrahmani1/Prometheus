/**
 * InSAR Fringe Visualization System - Task 3.2
 * Interactive fringe pattern overlay with advanced visualization capabilities
 * @author Prometheus Platform
 * @version 3.0.0
 */

class InSARFringeVisualizer {
    constructor(options = {}) {
        this.container = options.container || document.body;
        this.mapInstance = options.mapInstance || null;
        this.canvas = null;
        this.ctx = null;
        this.webglCanvas = null;
        this.gl = null;
        this.shaderProgram = null;
        
        // Visualization parameters
        this.config = {
            fringeSpacing: 2.84, // cm (half wavelength for C-band)
            colorScale: 'phase',
            opacity: 0.8,
            showUnwrapped: false,
            showLOS: true,
            animate: false,
            animationSpeed: 1.0,
            coherenceThreshold: 0.3,
            wavelength: 5.66, // C-band SAR wavelength in cm
            lookAngle: 23, // degrees from vertical
            headingAngle: -13.3 // degrees from north (descending)
        };
        
        // Data storage
        this.interferogramData = new Map();
        this.coherenceData = new Map();
        this.unwrappedPhase = new Map();
        this.losVectors = new Map();
        this.deformationField = new Map();
        this.timeSeriesData = new Map();
        
        // Color scales
        this.colorScales = {
            phase: this.createPhaseColorScale(),
            displacement: this.createDisplacementColorScale(),
            velocity: this.createVelocityColorScale(),
            coherence: this.createCoherenceColorScale(),
            elevation: this.createElevationColorScale()
        };
        
        // UI elements
        this.controls = null;
        this.legend = null;
        this.progressBar = null;
        
        this.isInitialized = false;
        this.currentFrame = 0;
        this.maxFrames = 100;
        this.animationId = null;
        
        this.init();
    }
    
    async init() {
        try {
            await this.createUI();
            await this.setupCanvas();
            await this.initializeWebGL();
            await this.loadShaders();
            await this.generateSyntheticData();
            
            this.isInitialized = true;
            this.render();
            
            console.log('✅ InSAR Fringe Visualizer initialized');
        } catch (error) {
            console.error('❌ Failed to initialize fringe visualizer:', error);
            this.showError('Initialization failed: ' + error.message);
        }
    }
    
    createUI() {
        return new Promise(resolve => {
            // Main container
            const mainContainer = document.createElement('div');
            mainContainer.className = 'fringe-visualizer-container';
            mainContainer.innerHTML = `
                <style>
                    .fringe-visualizer-container {
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        width: 320px;
                        background: rgba(255, 255, 255, 0.95);
                        border-radius: 12px;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                        backdrop-filter: blur(10px);
                        z-index: 1000;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    }
                    
                    .fringe-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 16px 20px;
                        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                        cursor: move;
                    }
                    
                    .fringe-title {
                        font-size: 16px;
                        font-weight: 600;
                        color: #333;
                        margin: 0;
                    }
                    
                    .header-controls {
                        display: flex;
                        gap: 8px;
                    }
                    
                    .control-btn {
                        background: transparent;
                        border: none;
                        color: #666;
                        cursor: pointer;
                        padding: 4px;
                        border-radius: 4px;
                        transition: all 0.2s;
                    }
                    
                    .control-btn:hover {
                        background: rgba(0, 0, 0, 0.1);
                        color: #333;
                    }
                    
                    .fringe-content {
                        padding: 20px;
                        max-height: 500px;
                        overflow-y: auto;
                    }
                    
                    .control-group {
                        margin-bottom: 20px;
                    }
                    
                    .control-label {
                        display: block;
                        font-size: 12px;
                        font-weight: 500;
                        color: #666;
                        margin-bottom: 8px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    
                    .control-input {
                        width: 100%;
                        padding: 8px 12px;
                        border: 1px solid #e0e0e0;
                        border-radius: 6px;
                        background: #fff;
                        font-size: 14px;
                        transition: border-color 0.2s;
                    }
                    
                    .control-input:focus {
                        outline: none;
                        border-color: #007bff;
                        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
                    }
                    
                    .control-range {
                        width: 100%;
                        margin: 8px 0;
                    }
                    
                    .range-display {
                        display: flex;
                        justify-content: space-between;
                        font-size: 12px;
                        color: #666;
                        margin-top: 4px;
                    }
                    
                    .toggle-group {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 8px;
                    }
                    
                    .toggle-btn {
                        padding: 8px 12px;
                        border: 1px solid #e0e0e0;
                        background: #fff;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 12px;
                        text-align: center;
                        transition: all 0.2s;
                    }
                    
                    .toggle-btn:hover {
                        border-color: #007bff;
                    }
                    
                    .toggle-btn.active {
                        background: #007bff;
                        color: white;
                        border-color: #007bff;
                    }
                    
                    .color-scale-preview {
                        height: 20px;
                        border-radius: 4px;
                        margin-top: 8px;
                        border: 1px solid #e0e0e0;
                    }
                    
                    .animation-controls {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        margin-top: 12px;
                    }
                    
                    .play-btn {
                        width: 40px;
                        height: 40px;
                        border: none;
                        border-radius: 50%;
                        background: #007bff;
                        color: white;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 16px;
                        transition: all 0.2s;
                    }
                    
                    .play-btn:hover {
                        background: #0056b3;
                        transform: scale(1.05);
                    }
                    
                    .frame-info {
                        font-size: 12px;
                        color: #666;
                        flex: 1;
                        text-align: center;
                    }
                    
                    .status-indicator {
                        display: inline-block;
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        margin-right: 8px;
                    }
                    
                    .status-active { background: #28a745; }
                    .status-inactive { background: #dc3545; }
                    .status-processing { background: #ffc107; animation: pulse 1s infinite; }
                    
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                    
                    .legend-container {
                        margin-top: 20px;
                        padding: 16px;
                        background: rgba(0, 0, 0, 0.02);
                        border-radius: 8px;
                    }
                    
                    .legend-title {
                        font-size: 12px;
                        font-weight: 500;
                        margin-bottom: 8px;
                        color: #333;
                    }
                    
                    .legend-scale {
                        height: 12px;
                        border-radius: 2px;
                        position: relative;
                        margin-bottom: 8px;
                    }
                    
                    .legend-labels {
                        display: flex;
                        justify-content: space-between;
                        font-size: 10px;
                        color: #666;
                    }
                    
                    .processing-overlay {
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(255, 255, 255, 0.9);
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        border-radius: 12px;
                        z-index: 10;
                    }
                    
                    .spinner {
                        width: 40px;
                        height: 40px;
                        border: 3px solid #e0e0e0;
                        border-top: 3px solid #007bff;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-bottom: 12px;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    .canvas-container {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100vw;
                        height: 100vh;
                        pointer-events: none;
                        z-index: 500;
                    }
                    
                    .fringe-canvas {
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                    }
                </style>
                
                <div class="fringe-header">
                    <h3 class="fringe-title">
                        <span class="status-indicator status-active"></span>
                        InSAR Fringes
                    </h3>
                    <div class="header-controls">
                        <button class="control-btn" onclick="this.closest('.fringe-visualizer-container').style.display='none'">✕</button>
                    </div>
                </div>
                
                <div class="fringe-content">
                    <div class="control-group">
                        <label class="control-label">Visualization Mode</label>
                        <select class="control-input" id="colorScaleSelect">
                            <option value="phase">Phase (Wrapped)</option>
                            <option value="displacement">Displacement</option>
                            <option value="velocity">Velocity</option>
                            <option value="coherence">Coherence</option>
                            <option value="elevation">Elevation Change</option>
                        </select>
                        <div class="color-scale-preview" id="colorPreview"></div>
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">Opacity</label>
                        <input type="range" class="control-range" id="opacityRange" 
                               min="0" max="1" step="0.1" value="0.8">
                        <div class="range-display">
                            <span>0%</span>
                            <span id="opacityValue">80%</span>
                            <span>100%</span>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">Fringe Spacing</label>
                        <input type="range" class="control-range" id="spacingRange"
                               min="1" max="10" step="0.1" value="2.84">
                        <div class="range-display">
                            <span>1 cm</span>
                            <span id="spacingValue">2.84 cm</span>
                            <span>10 cm</span>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">Coherence Threshold</label>
                        <input type="range" class="control-range" id="coherenceRange"
                               min="0" max="1" step="0.05" value="0.3">
                        <div class="range-display">
                            <span>0.0</span>
                            <span id="coherenceValue">0.30</span>
                            <span>1.0</span>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">Display Options</label>
                        <div class="toggle-group">
                            <button class="toggle-btn active" id="unwrappedBtn">Unwrapped</button>
                            <button class="toggle-btn active" id="losBtn">LOS Vectors</button>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">Time Series Animation</label>
                        <div class="animation-controls">
                            <button class="play-btn" id="playBtn">▶</button>
                            <div class="frame-info">
                                Frame <span id="currentFrame">1</span> of <span id="maxFrames">100</span>
                            </div>
                        </div>
                        <input type="range" class="control-range" id="frameRange"
                               min="0" max="99" value="0">
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">Animation Speed</label>
                        <input type="range" class="control-range" id="speedRange"
                               min="0.1" max="3" step="0.1" value="1">
                        <div class="range-display">
                            <span>0.1x</span>
                            <span id="speedValue">1.0x</span>
                            <span>3.0x</span>
                        </div>
                    </div>
                    
                    <div class="legend-container">
                        <div class="legend-title">Color Scale Legend</div>
                        <div class="legend-scale" id="legendScale"></div>
                        <div class="legend-labels" id="legendLabels">
                            <span>-π</span>
                            <span>0</span>
                            <span>π</span>
                        </div>
                    </div>
                </div>
                
                <div class="processing-overlay" id="processingOverlay" style="display: none;">
                    <div class="spinner"></div>
                    <div>Processing interferogram...</div>
                </div>
            `;
            
            this.container.appendChild(mainContainer);
            this.controls = mainContainer;
            
            // Canvas container
            const canvasContainer = document.createElement('div');
            canvasContainer.className = 'canvas-container';
            canvasContainer.innerHTML = `
                <canvas class="fringe-canvas" id="fringeCanvas"></canvas>
                <canvas class="fringe-canvas" id="webglCanvas"></canvas>
            `;
            
            document.body.appendChild(canvasContainer);
            
            this.setupEventListeners();
            this.updateColorPreview();
            
            resolve();
        });
    }
    
    setupEventListeners() {
        // Color scale selection
        const colorScaleSelect = document.getElementById('colorScaleSelect');
        colorScaleSelect.addEventListener('change', (e) => {
            this.config.colorScale = e.target.value;
            this.updateColorPreview();
            this.render();
        });
        
        // Opacity control
        const opacityRange = document.getElementById('opacityRange');
        const opacityValue = document.getElementById('opacityValue');
        opacityRange.addEventListener('input', (e) => {
            this.config.opacity = parseFloat(e.target.value);
            opacityValue.textContent = Math.round(this.config.opacity * 100) + '%';
            this.render();
        });
        
        // Fringe spacing control
        const spacingRange = document.getElementById('spacingRange');
        const spacingValue = document.getElementById('spacingValue');
        spacingRange.addEventListener('input', (e) => {
            this.config.fringeSpacing = parseFloat(e.target.value);
            spacingValue.textContent = this.config.fringeSpacing + ' cm';
            this.render();
        });
        
        // Coherence threshold control
        const coherenceRange = document.getElementById('coherenceRange');
        const coherenceValue = document.getElementById('coherenceValue');
        coherenceRange.addEventListener('input', (e) => {
            this.config.coherenceThreshold = parseFloat(e.target.value);
            coherenceValue.textContent = this.config.coherenceThreshold.toFixed(2);
            this.render();
        });
        
        // Toggle buttons
        const unwrappedBtn = document.getElementById('unwrappedBtn');
        const losBtn = document.getElementById('losBtn');
        
        unwrappedBtn.addEventListener('click', () => {
            this.config.showUnwrapped = !this.config.showUnwrapped;
            unwrappedBtn.classList.toggle('active');
            this.render();
        });
        
        losBtn.addEventListener('click', () => {
            this.config.showLOS = !this.config.showLOS;
            losBtn.classList.toggle('active');
            this.render();
        });
        
        // Animation controls
        const playBtn = document.getElementById('playBtn');
        const frameRange = document.getElementById('frameRange');
        const speedRange = document.getElementById('speedRange');
        const speedValue = document.getElementById('speedValue');
        
        playBtn.addEventListener('click', () => {
            if (this.config.animate) {
                this.stopAnimation();
                playBtn.textContent = '▶';
            } else {
                this.startAnimation();
                playBtn.textContent = '⏸';
            }
        });
        
        frameRange.addEventListener('input', (e) => {
            this.currentFrame = parseInt(e.target.value);
            this.updateFrameInfo();
            this.render();
        });
        
        speedRange.addEventListener('input', (e) => {
            this.config.animationSpeed = parseFloat(e.target.value);
            speedValue.textContent = this.config.animationSpeed.toFixed(1) + 'x';
        });
        
        // Make draggable
        this.makeDraggable(this.controls.querySelector('.fringe-header'), this.controls);
    }
    
    makeDraggable(handle, element) {
        let isDragging = false;
        let startX, startY, startLeft, startTop;
        
        handle.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            const rect = element.getBoundingClientRect();
            startLeft = rect.left;
            startTop = rect.top;
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
        
        function onMouseMove(e) {
            if (!isDragging) return;
            
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            element.style.left = (startLeft + deltaX) + 'px';
            element.style.top = (startTop + deltaY) + 'px';
            element.style.right = 'auto';
        }
        
        function onMouseUp() {
            isDragging = false;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }
    }
    
    setupCanvas() {
        return new Promise(resolve => {
            this.canvas = document.getElementById('fringeCanvas');
            this.webglCanvas = document.getElementById('webglCanvas');
            
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            this.webglCanvas.width = window.innerWidth;
            this.webglCanvas.height = window.innerHeight;
            
            this.ctx = this.canvas.getContext('2d');
            this.gl = this.webglCanvas.getContext('webgl2') || this.webglCanvas.getContext('webgl');
            
            if (!this.gl) {
                console.warn('WebGL not supported, falling back to 2D canvas');
            }
            
            // Handle window resize
            window.addEventListener('resize', () => {
                this.canvas.width = window.innerWidth;
                this.canvas.height = window.innerHeight;
                this.webglCanvas.width = window.innerWidth;
                this.webglCanvas.height = window.innerHeight;
                
                if (this.gl) {
                    this.gl.viewport(0, 0, this.webglCanvas.width, this.webglCanvas.height);
                }
                
                this.render();
            });
            
            resolve();
        });
    }
    
    initializeWebGL() {
        return new Promise(resolve => {
            if (!this.gl) {
                resolve();
                return;
            }
            
            // Enable extensions
            const ext = this.gl.getExtension('OES_texture_float');
            if (!ext) {
                console.warn('Float textures not supported');
            }
            
            // Set viewport
            this.gl.viewport(0, 0, this.webglCanvas.width, this.webglCanvas.height);
            
            // Enable blending
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
            
            resolve();
        });
    }
    
    loadShaders() {
        return new Promise(resolve => {
            if (!this.gl) {
                resolve();
                return;
            }
            
            // Vertex shader
            const vertexShaderSource = `
                attribute vec2 a_position;
                attribute vec2 a_texCoord;
                
                uniform mat3 u_matrix;
                
                varying vec2 v_texCoord;
                
                void main() {
                    vec3 position = u_matrix * vec3(a_position, 1.0);
                    gl_Position = vec4(position.xy, 0.0, 1.0);
                    v_texCoord = a_texCoord;
                }
            `;
            
            // Fragment shader for phase visualization
            const fragmentShaderSource = `
                precision mediump float;
                
                uniform sampler2D u_phaseTexture;
                uniform sampler2D u_coherenceTexture;
                uniform float u_opacity;
                uniform float u_coherenceThreshold;
                uniform int u_colorMode;
                uniform vec2 u_phaseRange;
                
                varying vec2 v_texCoord;
                
                vec3 phaseToColor(float phase) {
                    phase = phase / 3.14159265; // Normalize to [-1, 1]
                    phase = fract(phase + 0.5) * 2.0 - 1.0; // Wrap to [-1, 1]
                    
                    float r = 0.5 + 0.5 * cos(3.14159265 * (phase + 1.0));
                    float g = 0.5 + 0.5 * cos(3.14159265 * (phase + 0.33));
                    float b = 0.5 + 0.5 * cos(3.14159265 * (phase - 0.33));
                    
                    return vec3(r, g, b);
                }
                
                vec3 displacementToColor(float displacement) {
                    displacement = clamp(displacement, u_phaseRange.x, u_phaseRange.y);
                    float normalized = (displacement - u_phaseRange.x) / (u_phaseRange.y - u_phaseRange.x);
                    
                    if (normalized < 0.25) {
                        return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), normalized * 4.0);
                    } else if (normalized < 0.5) {
                        return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (normalized - 0.25) * 4.0);
                    } else if (normalized < 0.75) {
                        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (normalized - 0.5) * 4.0);
                    } else {
                        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (normalized - 0.75) * 4.0);
                    }
                }
                
                void main() {
                    float phase = texture2D(u_phaseTexture, v_texCoord).r;
                    float coherence = texture2D(u_coherenceTexture, v_texCoord).r;
                    
                    if (coherence < u_coherenceThreshold) {
                        discard;
                    }
                    
                    vec3 color;
                    if (u_colorMode == 0) { // Phase
                        color = phaseToColor(phase);
                    } else { // Displacement
                        color = displacementToColor(phase);
                    }
                    
                    gl_FragColor = vec4(color, u_opacity * coherence);
                }
            `;
            
            // Create shaders
            const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexShaderSource);
            const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentShaderSource);
            
            // Create program
            this.shaderProgram = this.gl.createProgram();
            this.gl.attachShader(this.shaderProgram, vertexShader);
            this.gl.attachShader(this.shaderProgram, fragmentShader);
            this.gl.linkProgram(this.shaderProgram);
            
            if (!this.gl.getProgramParameter(this.shaderProgram, this.gl.LINK_STATUS)) {
                console.error('Shader program linking failed:', this.gl.getProgramInfoLog(this.shaderProgram));
            }
            
            resolve();
        });
    }
    
    createShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compilation failed:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    generateSyntheticData() {
        return new Promise(resolve => {
            const width = 512;
            const height = 512;
            
            // Generate synthetic interferogram data
            for (let frame = 0; frame < this.maxFrames; frame++) {
                const phaseData = new Float32Array(width * height);
                const coherenceData = new Float32Array(width * height);
                const losData = [];
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = y * width + x;
                        
                        // Simulate volcanic deformation with time progression
                        const centerX = width / 2 + Math.sin(frame * 0.1) * 20;
                        const centerY = height / 2 + Math.cos(frame * 0.1) * 10;
                        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
                        
                        // Mogi source deformation model
                        const depth = 5000; // meters
                        const volume = 1e6 * (1 + 0.5 * Math.sin(frame * 0.2)); // m³
                        const nu = 0.25; // Poisson's ratio
                        
                        let displacement = 0;
                        if (distance > 0) {
                            const r = distance * 30; // scale to meters
                            displacement = (volume * (1 - nu) / Math.PI) * (depth / (r * r + depth * depth) ** 1.5);
                        }
                        
                        // Convert to phase (4π displacement / wavelength)
                        const phase = (4 * Math.PI * displacement) / (this.config.wavelength / 100); // convert to meters
                        
                        // Add atmospheric noise
                        const atmosphere = 0.5 * Math.sin(x * 0.02 + frame * 0.1) * Math.cos(y * 0.02);
                        const noise = (Math.random() - 0.5) * 0.3;
                        
                        phaseData[idx] = phase + atmosphere + noise;
                        
                        // Coherence decreases with distance and noise
                        const coherence = Math.exp(-distance / 200) * (0.3 + 0.7 * Math.random());
                        coherenceData[idx] = Math.min(1, Math.max(0, coherence));
                        
                        // Line-of-sight vectors (every 10th pixel)
                        if (x % 10 === 0 && y % 10 === 0 && coherence > this.config.coherenceThreshold) {
                            const magnitude = displacement * 1000; // mm
                            if (magnitude > 1) { // Only show significant displacements
                                losData.push({
                                    x: x,
                                    y: y,
                                    magnitude: magnitude,
                                    direction: Math.atan2(y - centerY, x - centerX)
                                });
                            }
                        }
                    }
                }
                
                this.interferogramData.set(frame, phaseData);
                this.coherenceData.set(frame, coherenceData);
                this.losVectors.set(frame, losData);
            }
            
            resolve();
        });
    }
    
    createPhaseColorScale() {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        
        for (let i = 0; i < 256; i++) {
            const phase = (i / 255) * 2 * Math.PI - Math.PI;
            const r = Math.floor(127.5 + 127.5 * Math.cos(phase + Math.PI));
            const g = Math.floor(127.5 + 127.5 * Math.cos(phase + Math.PI / 3));
            const b = Math.floor(127.5 + 127.5 * Math.cos(phase - Math.PI / 3));
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(i, 0, 1, 1);
        }
        
        return canvas;
    }
    
    createDisplacementColorScale() {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        
        const colors = [
            [0, 0, 255],     // Blue (subsidence)
            [0, 255, 255],   // Cyan
            [0, 255, 0],     // Green
            [255, 255, 0],   // Yellow
            [255, 0, 0]      // Red (uplift)
        ];
        
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const scaledT = t * (colors.length - 1);
            const idx = Math.floor(scaledT);
            const frac = scaledT - idx;
            
            let r, g, b;
            if (idx >= colors.length - 1) {
                [r, g, b] = colors[colors.length - 1];
            } else {
                const c1 = colors[idx];
                const c2 = colors[idx + 1];
                r = Math.floor(c1[0] + (c2[0] - c1[0]) * frac);
                g = Math.floor(c1[1] + (c2[1] - c1[1]) * frac);
                b = Math.floor(c1[2] + (c2[2] - c1[2]) * frac);
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(i, 0, 1, 1);
        }
        
        return canvas;
    }
    
    createVelocityColorScale() {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        
        // Velocity color scale: purple (negative) to red (positive)
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const r = Math.floor(255 * Math.pow(t, 0.5));
            const g = Math.floor(255 * Math.pow(t, 2));
            const b = Math.floor(255 * Math.pow(1 - t, 0.5));
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(i, 0, 1, 1);
        }
        
        return canvas;
    }
    
    createCoherenceColorScale() {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        
        // Coherence: black (0) to white (1)
        for (let i = 0; i < 256; i++) {
            const value = i;
            ctx.fillStyle = `rgb(${value}, ${value}, ${value})`;
            ctx.fillRect(i, 0, 1, 1);
        }
        
        return canvas;
    }
    
    createElevationColorScale() {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        
        // Elevation: blue (low) to brown (high)
        const elevationColors = [
            [0, 0, 139],     // Dark blue
            [0, 191, 255],   // Deep sky blue
            [0, 255, 0],     // Green
            [255, 255, 0],   // Yellow
            [255, 165, 0],   // Orange
            [139, 69, 19]    // Saddle brown
        ];
        
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const scaledT = t * (elevationColors.length - 1);
            const idx = Math.floor(scaledT);
            const frac = scaledT - idx;
            
            let r, g, b;
            if (idx >= elevationColors.length - 1) {
                [r, g, b] = elevationColors[elevationColors.length - 1];
            } else {
                const c1 = elevationColors[idx];
                const c2 = elevationColors[idx + 1];
                r = Math.floor(c1[0] + (c2[0] - c1[0]) * frac);
                g = Math.floor(c1[1] + (c2[1] - c1[1]) * frac);
                b = Math.floor(c1[2] + (c2[2] - c1[2]) * frac);
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(i, 0, 1, 1);
        }
        
        return canvas;
    }
    
    updateColorPreview() {
        const preview = document.getElementById('colorPreview');
        const legend = document.getElementById('legendScale');
        const scale = this.colorScales[this.config.colorScale];
        
        if (scale && preview) {
            const ctx = scale.getContext('2d');
            const imageData = ctx.getImageData(0, 0, scale.width, scale.height);
            
            // Create CSS gradient from image data
            const colors = [];
            for (let i = 0; i < imageData.data.length; i += 4) {
                const r = imageData.data[i];
                const g = imageData.data[i + 1];
                const b = imageData.data[i + 2];
                colors.push(`rgb(${r}, ${g}, ${b})`);
            }
            
            const step = Math.ceil(colors.length / 20); // Sample every nth color
            const sampledColors = colors.filter((_, i) => i % step === 0);
            
            const gradient = `linear-gradient(to right, ${sampledColors.join(', ')})`;
            preview.style.background = gradient;
            
            if (legend) {
                legend.style.background = gradient;
            }
        }
        
        // Update legend labels
        const labels = document.getElementById('legendLabels');
        if (labels) {
            switch (this.config.colorScale) {
                case 'phase':
                    labels.innerHTML = '<span>-π</span><span>0</span><span>π</span>';
                    break;
                case 'displacement':
                    labels.innerHTML = '<span>-5 cm</span><span>0</span><span>+5 cm</span>';
                    break;
                case 'velocity':
                    labels.innerHTML = '<span>-10 mm/yr</span><span>0</span><span>+10 mm/yr</span>';
                    break;
                case 'coherence':
                    labels.innerHTML = '<span>0.0</span><span>0.5</span><span>1.0</span>';
                    break;
                case 'elevation':
                    labels.innerHTML = '<span>-50 m</span><span>0</span><span>+50 m</span>';
                    break;
            }
        }
    }
    
    startAnimation() {
        this.config.animate = true;
        
        const animate = () => {
            if (!this.config.animate) return;
            
            this.currentFrame = (this.currentFrame + this.config.animationSpeed) % this.maxFrames;
            this.updateFrameInfo();
            this.render();
            
            const frameRange = document.getElementById('frameRange');
            if (frameRange) {
                frameRange.value = Math.floor(this.currentFrame);
            }
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    stopAnimation() {
        this.config.animate = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    updateFrameInfo() {
        const currentFrameSpan = document.getElementById('currentFrame');
        const maxFramesSpan = document.getElementById('maxFrames');
        
        if (currentFrameSpan) {
            currentFrameSpan.textContent = Math.floor(this.currentFrame) + 1;
        }
        if (maxFramesSpan) {
            maxFramesSpan.textContent = this.maxFrames;
        }
    }
    
    render() {
        if (!this.isInitialized) return;
        
        // Clear canvases
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        
        if (this.gl) {
            this.renderWebGL();
        } else {
            this.renderCanvas();
        }
        
        // Render LOS vectors
        if (this.config.showLOS) {
            this.renderLOSVectors();
        }
    }
    
    renderWebGL() {
        if (!this.gl || !this.shaderProgram) return;
        
        const frameIndex = Math.floor(this.currentFrame);
        const phaseData = this.interferogramData.get(frameIndex);
        const coherenceData = this.coherenceData.get(frameIndex);
        
        if (!phaseData || !coherenceData) return;
        
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        this.gl.useProgram(this.shaderProgram);
        
        // Create and bind textures
        const phaseTexture = this.gl.createTexture();
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, phaseTexture);
        this.gl.texImage2D(
            this.gl.TEXTURE_2D, 0, this.gl.LUMINANCE,
            512, 512, 0, this.gl.LUMINANCE, this.gl.FLOAT, phaseData
        );
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        
        const coherenceTexture = this.gl.createTexture();
        this.gl.activeTexture(this.gl.TEXTURE1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, coherenceTexture);
        this.gl.texImage2D(
            this.gl.TEXTURE_2D, 0, this.gl.LUMINANCE,
            512, 512, 0, this.gl.LUMINANCE, this.gl.FLOAT, coherenceData
        );
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        
        // Set uniforms
        this.gl.uniform1i(this.gl.getUniformLocation(this.shaderProgram, 'u_phaseTexture'), 0);
        this.gl.uniform1i(this.gl.getUniformLocation(this.shaderProgram, 'u_coherenceTexture'), 1);
        this.gl.uniform1f(this.gl.getUniformLocation(this.shaderProgram, 'u_opacity'), this.config.opacity);
        this.gl.uniform1f(this.gl.getUniformLocation(this.shaderProgram, 'u_coherenceThreshold'), this.config.coherenceThreshold);
        
        const colorMode = this.config.colorScale === 'phase' ? 0 : 1;
        this.gl.uniform1i(this.gl.getUniformLocation(this.shaderProgram, 'u_colorMode'), colorMode);
        this.gl.uniform2f(this.gl.getUniformLocation(this.shaderProgram, 'u_phaseRange'), -0.05, 0.05);
        
        // Set up geometry
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]);
        
        const texCoords = new Float32Array([
            0, 0,
            1, 0,
            0, 1,
            1, 1
        ]);
        
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
        
        const positionLocation = this.gl.getAttribLocation(this.shaderProgram, 'a_position');
        this.gl.enableVertexAttribArray(positionLocation);
        this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
        
        const texCoordBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, texCoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);
        
        const texCoordLocation = this.gl.getAttribLocation(this.shaderProgram, 'a_texCoord');
        this.gl.enableVertexAttribArray(texCoordLocation);
        this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 0, 0);
        
        // Draw
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        
        // Clean up
        this.gl.deleteTexture(phaseTexture);
        this.gl.deleteTexture(coherenceTexture);
        this.gl.deleteBuffer(positionBuffer);
        this.gl.deleteBuffer(texCoordBuffer);
    }
    
    renderCanvas() {
        const frameIndex = Math.floor(this.currentFrame);
        const phaseData = this.interferogramData.get(frameIndex);
        const coherenceData = this.coherenceData.get(frameIndex);
        
        if (!phaseData || !coherenceData) return;
        
        const width = 512;
        const height = 512;
        const imageData = this.ctx.createImageData(width, height);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                const pixelIdx = idx * 4;
                
                const phase = phaseData[idx];
                const coherence = coherenceData[idx];
                
                if (coherence < this.config.coherenceThreshold) {
                    imageData.data[pixelIdx] = 0;
                    imageData.data[pixelIdx + 1] = 0;
                    imageData.data[pixelIdx + 2] = 0;
                    imageData.data[pixelIdx + 3] = 0;
                    continue;
                }
                
                let r, g, b;
                
                switch (this.config.colorScale) {
                    case 'phase':
                        const normalizedPhase = (phase / Math.PI + 1) / 2;
                        r = Math.floor(127.5 + 127.5 * Math.cos(phase + Math.PI));
                        g = Math.floor(127.5 + 127.5 * Math.cos(phase + Math.PI / 3));
                        b = Math.floor(127.5 + 127.5 * Math.cos(phase - Math.PI / 3));
                        break;
                        
                    case 'displacement':
                        const displacement = (phase * this.config.wavelength) / (4 * Math.PI) * 100; // cm
                        const normalized = Math.max(0, Math.min(1, (displacement + 5) / 10));
                        
                        if (normalized < 0.25) {
                            r = 0;
                            g = Math.floor(255 * normalized * 4);
                            b = 255;
                        } else if (normalized < 0.5) {
                            r = 0;
                            g = 255;
                            b = Math.floor(255 * (1 - (normalized - 0.25) * 4));
                        } else if (normalized < 0.75) {
                            r = Math.floor(255 * (normalized - 0.5) * 4);
                            g = 255;
                            b = 0;
                        } else {
                            r = 255;
                            g = Math.floor(255 * (1 - (normalized - 0.75) * 4));
                            b = 0;
                        }
                        break;
                        
                    case 'coherence':
                        const cohValue = Math.floor(coherence * 255);
                        r = g = b = cohValue;
                        break;
                        
                    default:
                        r = g = b = 0;
                }
                
                imageData.data[pixelIdx] = r;
                imageData.data[pixelIdx + 1] = g;
                imageData.data[pixelIdx + 2] = b;
                imageData.data[pixelIdx + 3] = Math.floor(255 * this.config.opacity * coherence);
            }
        }
        
        // Scale to canvas size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);
        
        // Draw scaled image
        const scale = Math.min(this.canvas.width / width, this.canvas.height / height);
        const scaledWidth = width * scale;
        const scaledHeight = height * scale;
        const offsetX = (this.canvas.width - scaledWidth) / 2;
        const offsetY = (this.canvas.height - scaledHeight) / 2;
        
        this.ctx.drawImage(tempCanvas, offsetX, offsetY, scaledWidth, scaledHeight);
    }
    
    renderLOSVectors() {
        const frameIndex = Math.floor(this.currentFrame);
        const losData = this.losVectors.get(frameIndex);
        
        if (!losData || !this.ctx) return;
        
        const scale = Math.min(this.canvas.width / 512, this.canvas.height / 512);
        const offsetX = (this.canvas.width - 512 * scale) / 2;
        const offsetY = (this.canvas.height - 512 * scale) / 2;
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.lineWidth = 1;
        
        for (const vector of losData) {
            const x = offsetX + vector.x * scale;
            const y = offsetY + vector.y * scale;
            const length = Math.min(20, vector.magnitude * 0.5); // Scale vector length
            
            // Draw vector arrow
            const endX = x + Math.cos(vector.direction) * length;
            const endY = y + Math.sin(vector.direction) * length;
            
            this.ctx.beginPath();
            this.ctx.moveTo(x, y);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
            
            // Draw arrowhead
            const arrowLength = 4;
            const arrowAngle = Math.PI / 6;
            
            this.ctx.beginPath();
            this.ctx.moveTo(endX, endY);
            this.ctx.lineTo(
                endX - arrowLength * Math.cos(vector.direction - arrowAngle),
                endY - arrowLength * Math.sin(vector.direction - arrowAngle)
            );
            this.ctx.moveTo(endX, endY);
            this.ctx.lineTo(
                endX - arrowLength * Math.cos(vector.direction + arrowAngle),
                endY - arrowLength * Math.sin(vector.direction + arrowAngle)
            );
            this.ctx.stroke();
        }
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <style>
                .error-message {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #ff4444;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    z-index: 10000;
                    max-width: 400px;
                    text-align: center;
                }
                .error-close {
                    margin-top: 10px;
                    padding: 8px 16px;
                    background: rgba(255, 255, 255, 0.2);
                    border: none;
                    color: white;
                    border-radius: 4px;
                    cursor: pointer;
                }
            </style>
            <div>❌ ${message}</div>
            <button class="error-close" onclick="this.parentElement.remove()">Close</button>
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 10000);
    }
    
    // Public API methods
    setMapInstance(mapInstance) {
        this.mapInstance = mapInstance;
    }
    
    loadInterferogram(data) {
        // Load external interferogram data
        this.interferogramData.set('external', data.phase);
        this.coherenceData.set('external', data.coherence);
        this.render();
    }
    
    exportVisualization() {
        const link = document.createElement('a');
        link.download = `insar_fringe_${new Date().toISOString().slice(0, 10)}.png`;
        link.href = this.canvas.toDataURL();
        link.click();
    }
    
    destroy() {
        this.stopAnimation();
        
        if (this.controls && this.controls.parentElement) {
            this.controls.parentElement.removeChild(this.controls);
        }
        
        const canvasContainer = document.querySelector('.canvas-container');
        if (canvasContainer && canvasContainer.parentElement) {
            canvasContainer.parentElement.removeChild(canvasContainer);
        }
    }
}

// Global initialization
if (typeof window !== 'undefined') {
    window.InSARFringeVisualizer = InSARFringeVisualizer;
    
    // Auto-initialize if container exists
    document.addEventListener('DOMContentLoaded', () => {
        if (document.getElementById('fringe-visualizer-target')) {
            window.fringeVisualizer = new InSARFringeVisualizer({
                container: document.getElementById('fringe-visualizer-target')
            });
        }
    });
    
    console.log('✅ InSAR Fringe Visualizer loaded - Interactive fringe patterns with WebGL acceleration');
}