/**
 * 3D Deformation Modeling System - Task 3.2
 * Advanced 3D visualization and modeling of volcanic deformation
 * @author Prometheus Platform
 * @version 3.0.0
 */

class Deformation3D {
    constructor(options = {}) {
        this.container = options.container || document.body;
        this.canvas = null;
        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.controls = null;
        
        // 3D modeling parameters
        this.config = {
            terrainResolution: 256,
            deformationScale: 100,
            timeScale: 1.0,
            showVectors: true,
            showTerrain: true,
            showSources: true,
            vectorScale: 50,
            animateTime: false,
            wireframeMode: false,
            lightingIntensity: 1.0,
            shadowsEnabled: true,
            fogEnabled: true
        };
        
        // Deformation sources
        this.sources = new Map();
        this.sourceTypes = {
            mogi: this.mogiSource,
            okada: this.okadaSource,
            yang: this.yangSource,
            penny: this.pennySource,
            sill: this.sillSource,
            dike: this.dikeSource
        };
        
        // 3D objects
        this.terrainMesh = null;
        this.deformationMesh = null;
        this.vectorField = null;
        this.sourceMeshes = [];
        this.timeSeriesMeshes = [];
        
        // Data storage
        this.elevationData = null;
        this.displacementData = new Map();
        this.velocityData = new Map();
        this.strainData = new Map();
        this.timeSeries = [];
        
        // UI elements
        this.controlPanel = null;
        this.timeline = null;
        this.stats = null;
        
        // Animation
        this.currentTime = 0;
        this.timeRange = [0, 365]; // days
        this.animationSpeed = 1.0;
        this.isAnimating = false;
        this.animationId = null;
        
        this.isInitialized = false;
        
        this.init();
    }
    
    async init() {
        try {
            await this.loadThreeJS();
            await this.setupScene();
            await this.createUI();
            await this.generateTerrain();
            await this.addDeformationSources();
            await this.calculateDeformation();
            await this.setupEventListeners();
            
            this.isInitialized = true;
            this.render();
            
            console.log('✅ 3D Deformation Modeling initialized');
        } catch (error) {
            console.error('❌ Failed to initialize 3D deformation modeling:', error);
            this.showError('Initialization failed: ' + error.message);
        }
    }
    
    loadThreeJS() {
        return new Promise((resolve, reject) => {
            if (window.THREE) {
                resolve();
                return;
            }
            
            // Load Three.js from CDN with fallback
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js';
            script.onload = () => {
                // Load OrbitControls
                const controlsScript = document.createElement('script');
                controlsScript.src = 'https://cdn.jsdelivr.net/npm/three@0.152.0/examples/js/controls/OrbitControls.js';
                controlsScript.onload = () => resolve();
                controlsScript.onerror = () => {
                    // Fallback OrbitControls implementation
                    this.createOrbitControls();
                    resolve();
                };
                document.head.appendChild(controlsScript);
            };
            script.onerror = () => {
                // Fallback Three.js implementation
                this.createThreejsFallback();
                resolve();
            };
            document.head.appendChild(script);
        });
    }
    
    createThreejsFallback() {
        // Minimal Three.js fallback for basic 3D rendering
        window.THREE = {
            Scene: function() {
                this.children = [];
                this.add = (obj) => this.children.push(obj);
                this.remove = (obj) => {
                    const index = this.children.indexOf(obj);
                    if (index > -1) this.children.splice(index, 1);
                };
            },
            
            PerspectiveCamera: function(fov, aspect, near, far) {
                this.fov = fov;
                this.aspect = aspect;
                this.near = near;
                this.far = far;
                this.position = { x: 0, y: 0, z: 5 };
                this.rotation = { x: 0, y: 0, z: 0 };
                this.updateProjectionMatrix = () => {};
            },
            
            WebGLRenderer: function(options = {}) {
                this.domElement = options.canvas || document.createElement('canvas');
                this.setSize = (w, h) => {
                    this.domElement.width = w;
                    this.domElement.height = h;
                };
                this.render = (scene, camera) => {
                    // Basic 2D canvas fallback rendering
                    const ctx = this.domElement.getContext('2d');
                    ctx.fillStyle = '#001122';
                    ctx.fillRect(0, 0, this.domElement.width, this.domElement.height);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '16px Arial';
                    ctx.fillText('3D Deformation Model', 20, 30);
                    ctx.fillText('(Fallback Mode)', 20, 50);
                };
            },
            
            Mesh: function(geometry, material) {
                this.geometry = geometry;
                this.material = material;
                this.position = { x: 0, y: 0, z: 0 };
                this.rotation = { x: 0, y: 0, z: 0 };
                this.scale = { x: 1, y: 1, z: 1 };
            },
            
            PlaneGeometry: function(width, height, widthSegments, heightSegments) {
                this.width = width;
                this.height = height;
                this.widthSegments = widthSegments;
                this.heightSegments = heightSegments;
            },
            
            MeshLambertMaterial: function(options = {}) {
                this.color = options.color || 0xffffff;
                this.wireframe = options.wireframe || false;
                this.transparent = options.transparent || false;
                this.opacity = options.opacity || 1.0;
            },
            
            DirectionalLight: function(color, intensity) {
                this.color = color;
                this.intensity = intensity;
                this.position = { x: 1, y: 1, z: 1 };
            },
            
            AmbientLight: function(color, intensity) {
                this.color = color;
                this.intensity = intensity;
            },
            
            Vector3: function(x = 0, y = 0, z = 0) {
                this.x = x;
                this.y = y;
                this.z = z;
                this.normalize = () => {
                    const length = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
                    if (length > 0) {
                        this.x /= length;
                        this.y /= length;
                        this.z /= length;
                    }
                    return this;
                };
            },
            
            Color: function(r, g, b) {
                if (typeof r === 'string') {
                    // Handle hex colors
                    this.setHex = (hex) => this;
                } else {
                    this.r = r || 0;
                    this.g = g || 0;
                    this.b = b || 0;
                }
                this.setHSL = (h, s, l) => this;
            }
        };
        
        console.log('Using Three.js fallback mode');
    }
    
    createOrbitControls() {
        if (!window.THREE.OrbitControls) {
            window.THREE.OrbitControls = function(camera, domElement) {
                this.camera = camera;
                this.domElement = domElement;
                this.enabled = true;
                
                // Basic mouse controls
                let isDragging = false;
                let previousMousePosition = { x: 0, y: 0 };
                
                domElement.addEventListener('mousedown', (e) => {
                    isDragging = true;
                    previousMousePosition = { x: e.clientX, y: e.clientY };
                });
                
                domElement.addEventListener('mousemove', (e) => {
                    if (!isDragging) return;
                    
                    const deltaX = e.clientX - previousMousePosition.x;
                    const deltaY = e.clientY - previousMousePosition.y;
                    
                    camera.rotation.y += deltaX * 0.01;
                    camera.rotation.x += deltaY * 0.01;
                    
                    previousMousePosition = { x: e.clientX, y: e.clientY };
                });
                
                domElement.addEventListener('mouseup', () => {
                    isDragging = false;
                });
                
                domElement.addEventListener('wheel', (e) => {
                    camera.position.z += e.deltaY * 0.01;
                    camera.position.z = Math.max(1, Math.min(50, camera.position.z));
                });
                
                this.update = () => {};
            };
        }
    }
    
    setupScene() {
        return new Promise(resolve => {
            // Create scene
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x001122);
            
            // Setup camera
            this.camera = new THREE.PerspectiveCamera(
                75,
                window.innerWidth / window.innerHeight,
                0.1,
                1000
            );
            this.camera.position.set(10, 10, 10);
            this.camera.lookAt(0, 0, 0);
            
            // Create renderer
            const canvas = document.createElement('canvas');
            canvas.id = 'deformation3d-canvas';
            this.renderer = new THREE.WebGLRenderer({
                canvas: canvas,
                antialias: true,
                alpha: true
            });
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.shadowMap.enabled = this.config.shadowsEnabled;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            // Add controls
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            
            // Add lighting
            const directionalLight = new THREE.DirectionalLight(0xffffff, this.config.lightingIntensity);
            directionalLight.position.set(5, 10, 5);
            directionalLight.castShadow = true;
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            this.scene.add(directionalLight);
            
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            this.scene.add(ambientLight);
            
            // Add fog
            if (this.config.fogEnabled) {
                this.scene.fog = new THREE.Fog(0x001122, 20, 100);
            }
            
            // Handle window resize
            window.addEventListener('resize', () => {
                this.camera.aspect = window.innerWidth / window.innerHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            resolve();
        });
    }
    
    createUI() {
        return new Promise(resolve => {
            // Main container
            const mainContainer = document.createElement('div');
            mainContainer.className = 'deformation3d-container';
            mainContainer.innerHTML = `
                <style>
                    .deformation3d-container {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100vw;
                        height: 100vh;
                        z-index: 100;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    }
                    
                    .deformation3d-controls {
                        position: absolute;
                        top: 20px;
                        left: 20px;
                        width: 300px;
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        border-radius: 8px;
                        padding: 16px;
                        backdrop-filter: blur(10px);
                        max-height: 80vh;
                        overflow-y: auto;
                    }
                    
                    .controls-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 16px;
                        padding-bottom: 8px;
                        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                    }
                    
                    .controls-title {
                        font-size: 16px;
                        font-weight: 600;
                        margin: 0;
                    }
                    
                    .close-btn {
                        background: transparent;
                        border: none;
                        color: white;
                        font-size: 18px;
                        cursor: pointer;
                        padding: 4px;
                    }
                    
                    .control-section {
                        margin-bottom: 20px;
                    }
                    
                    .section-title {
                        font-size: 12px;
                        font-weight: 500;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        margin-bottom: 8px;
                        color: #00ff88;
                    }
                    
                    .control-row {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        margin-bottom: 8px;
                    }
                    
                    .control-label {
                        font-size: 12px;
                        color: #ccc;
                        flex: 1;
                    }
                    
                    .control-input {
                        width: 60px;
                        padding: 4px 6px;
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 4px;
                        color: white;
                        font-size: 11px;
                    }
                    
                    .control-slider {
                        width: 120px;
                        margin-left: 8px;
                    }
                    
                    .toggle-btn {
                        padding: 6px 12px;
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        color: white;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 11px;
                        transition: all 0.2s;
                        margin: 2px;
                    }
                    
                    .toggle-btn:hover {
                        background: rgba(255, 255, 255, 0.2);
                    }
                    
                    .toggle-btn.active {
                        background: #00ff88;
                        color: black;
                        border-color: #00ff88;
                    }
                    
                    .source-list {
                        max-height: 150px;
                        overflow-y: auto;
                        margin-top: 8px;
                    }
                    
                    .source-item {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 6px 8px;
                        background: rgba(255, 255, 255, 0.05);
                        margin-bottom: 4px;
                        border-radius: 4px;
                        font-size: 11px;
                    }
                    
                    .source-type {
                        color: #00ff88;
                        font-weight: 500;
                    }
                    
                    .source-params {
                        color: #999;
                        font-size: 10px;
                    }
                    
                    .timeline-container {
                        position: absolute;
                        bottom: 20px;
                        left: 50%;
                        transform: translateX(-50%);
                        width: 80%;
                        max-width: 800px;
                        background: rgba(0, 0, 0, 0.8);
                        padding: 16px;
                        border-radius: 8px;
                        backdrop-filter: blur(10px);
                    }
                    
                    .timeline-controls {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        margin-bottom: 8px;
                    }
                    
                    .play-pause-btn {
                        width: 40px;
                        height: 40px;
                        border: none;
                        border-radius: 50%;
                        background: #00ff88;
                        color: black;
                        font-size: 16px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    
                    .time-display {
                        color: white;
                        font-size: 14px;
                        min-width: 100px;
                        text-align: center;
                    }
                    
                    .timeline-slider {
                        flex: 1;
                        height: 6px;
                        background: rgba(255, 255, 255, 0.2);
                        border-radius: 3px;
                        position: relative;
                        cursor: pointer;
                    }
                    
                    .timeline-progress {
                        height: 100%;
                        background: #00ff88;
                        border-radius: 3px;
                        transition: width 0.1s ease;
                    }
                    
                    .speed-control {
                        color: white;
                        font-size: 12px;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    .stats-panel {
                        position: absolute;
                        top: 20px;
                        right: 20px;
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 12px;
                        border-radius: 8px;
                        backdrop-filter: blur(10px);
                        font-family: monospace;
                        font-size: 11px;
                        line-height: 1.4;
                    }
                    
                    .canvas-container {
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        z-index: -1;
                    }
                </style>
                
                <div class="canvas-container"></div>
                
                <div class="deformation3d-controls">
                    <div class="controls-header">
                        <h3 class="controls-title">3D Deformation</h3>
                        <button class="close-btn" onclick="this.closest('.deformation3d-container').style.display='none'">✕</button>
                    </div>
                    
                    <div class="control-section">
                        <div class="section-title">Visualization</div>
                        <div class="control-row">
                            <button class="toggle-btn active" id="terrainBtn">Terrain</button>
                            <button class="toggle-btn active" id="vectorsBtn">Vectors</button>
                            <button class="toggle-btn active" id="sourcesBtn">Sources</button>
                        </div>
                        <div class="control-row">
                            <button class="toggle-btn" id="wireframeBtn">Wireframe</button>
                            <button class="toggle-btn active" id="shadowsBtn">Shadows</button>
                            <button class="toggle-btn active" id="fogBtn">Fog</button>
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <div class="section-title">Scaling</div>
                        <div class="control-row">
                            <span class="control-label">Deformation</span>
                            <input type="range" class="control-slider" id="deformationScale" 
                                   min="1" max="1000" value="100">
                            <input type="number" class="control-input" id="deformationValue" value="100">
                        </div>
                        <div class="control-row">
                            <span class="control-label">Vectors</span>
                            <input type="range" class="control-slider" id="vectorScale" 
                                   min="1" max="200" value="50">
                            <input type="number" class="control-input" id="vectorValue" value="50">
                        </div>
                        <div class="control-row">
                            <span class="control-label">Lighting</span>
                            <input type="range" class="control-slider" id="lightingScale" 
                                   min="0.1" max="3" step="0.1" value="1">
                            <input type="number" class="control-input" id="lightingValue" value="1.0" step="0.1">
                        </div>
                    </div>
                    
                    <div class="control-section">
                        <div class="section-title">Deformation Sources</div>
                        <div class="control-row">
                            <select id="sourceType" class="control-input" style="width: 120px;">
                                <option value="mogi">Mogi Point</option>
                                <option value="okada">Okada Fault</option>
                                <option value="yang">Yang Tensile</option>
                                <option value="penny">Penny Crack</option>
                                <option value="sill">Sill</option>
                                <option value="dike">Dike</option>
                            </select>
                            <button class="toggle-btn" id="addSourceBtn">Add Source</button>
                        </div>
                        <div class="source-list" id="sourceList"></div>
                    </div>
                    
                    <div class="control-section">
                        <div class="section-title">Export</div>
                        <div class="control-row">
                            <button class="toggle-btn" id="exportImageBtn">Image</button>
                            <button class="toggle-btn" id="exportDataBtn">Data</button>
                            <button class="toggle-btn" id="exportVideoBtn">Video</button>
                        </div>
                    </div>
                </div>
                
                <div class="timeline-container">
                    <div class="timeline-controls">
                        <button class="play-pause-btn" id="playPauseBtn">▶</button>
                        <div class="time-display" id="timeDisplay">Day 0</div>
                        <div class="timeline-slider" id="timelineSlider">
                            <div class="timeline-progress" id="timelineProgress"></div>
                        </div>
                        <div class="speed-control">
                            <span>Speed:</span>
                            <input type="range" id="speedSlider" min="0.1" max="5" step="0.1" value="1">
                            <span id="speedValue">1.0x</span>
                        </div>
                    </div>
                </div>
                
                <div class="stats-panel" id="statsPanel">
                    <div>FPS: <span id="fpsCounter">60</span></div>
                    <div>Sources: <span id="sourceCounter">0</span></div>
                    <div>Vertices: <span id="vertexCounter">0</span></div>
                    <div>Max Displacement: <span id="maxDisplacement">0.0</span> mm</div>
                    <div>Current Time: <span id="currentTimeStats">0</span> days</div>
                </div>
            `;
            
            this.container.appendChild(mainContainer);
            
            // Get canvas container and append renderer
            const canvasContainer = mainContainer.querySelector('.canvas-container');
            canvasContainer.appendChild(this.renderer.domElement);
            
            this.controlPanel = mainContainer.querySelector('.deformation3d-controls');
            this.timeline = mainContainer.querySelector('.timeline-container');
            this.stats = mainContainer.querySelector('.stats-panel');
            
            resolve();
        });
    }
    
    generateTerrain() {
        return new Promise(resolve => {
            const size = 20;
            const resolution = this.config.terrainResolution;
            
            // Create terrain geometry
            const geometry = new THREE.PlaneGeometry(size, size, resolution - 1, resolution - 1);
            
            // Generate elevation data (volcanic edifice)
            this.elevationData = new Float32Array(resolution * resolution);
            const positions = geometry.attributes.position.array;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const idx = i * resolution + j;
                    const x = (i / (resolution - 1) - 0.5) * size;
                    const z = (j / (resolution - 1) - 0.5) * size;
                    
                    // Create volcanic edifice with multiple peaks
                    const distance1 = Math.sqrt(x * x + z * z);
                    const distance2 = Math.sqrt((x - 2) * (x - 2) + (z - 1) * (z - 1));
                    const distance3 = Math.sqrt((x + 1) * (x + 1) + (z - 2) * (z - 2));
                    
                    let elevation = 0;
                    elevation += 3 * Math.exp(-distance1 * distance1 / 8); // Main peak
                    elevation += 1.5 * Math.exp(-distance2 * distance2 / 4); // Secondary peak
                    elevation += 1 * Math.exp(-distance3 * distance3 / 3); // Tertiary peak
                    
                    // Add noise
                    elevation += 0.1 * (Math.random() - 0.5);
                    
                    this.elevationData[idx] = elevation;
                    
                    // Update geometry
                    const vertexIndex = idx * 3 + 1; // Y coordinate
                    if (vertexIndex < positions.length) {
                        positions[vertexIndex] = elevation;
                    }
                }
            }
            
            // Update normals
            geometry.computeVertexNormals();
            
            // Create material with elevation-based coloring
            const material = new THREE.MeshLambertMaterial({
                vertexColors: true,
                wireframe: this.config.wireframeMode
            });
            
            // Add vertex colors based on elevation
            const colors = new Float32Array(positions.length);
            for (let i = 0; i < positions.length; i += 3) {
                const elevation = positions[i + 1]; // Y coordinate
                const normalizedElevation = elevation / 3; // Normalize to max height
                
                // Color gradient: blue (low) to green to brown (high)
                let r, g, b;
                if (normalizedElevation < 0.3) {
                    r = 0.2;
                    g = 0.4 + normalizedElevation;
                    b = 0.8 - normalizedElevation;
                } else if (normalizedElevation < 0.7) {
                    r = 0.3 + normalizedElevation * 0.3;
                    g = 0.7;
                    b = 0.2;
                } else {
                    r = 0.6 + normalizedElevation * 0.4;
                    g = 0.4 + normalizedElevation * 0.2;
                    b = 0.3;
                }
                
                colors[i] = r;
                colors[i + 1] = g;
                colors[i + 2] = b;
            }
            
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            // Create terrain mesh
            this.terrainMesh = new THREE.Mesh(geometry, material);
            this.terrainMesh.rotation.x = -Math.PI / 2;
            this.terrainMesh.receiveShadow = true;
            
            this.scene.add(this.terrainMesh);
            
            resolve();
        });
    }
    
    addDeformationSources() {
        return new Promise(resolve => {
            // Add default Mogi source at the main peak
            this.addSource('mogi', {
                x: 0,
                y: -2, // 2 km depth
                z: 0,
                volume: 1e6, // m³
                nu: 0.25 // Poisson's ratio
            });
            
            // Add secondary Okada fault source
            this.addSource('okada', {
                x: 2,
                y: -1,
                z: 1,
                length: 5000, // m
                width: 3000, // m
                strike: 45, // degrees
                dip: 60, // degrees
                rake: 90, // degrees
                slip: 0.5 // m
            });
            
            resolve();
        });
    }
    
    addSource(type, params) {
        const sourceId = Date.now().toString();
        const sourceData = {
            id: sourceId,
            type: type,
            params: params,
            active: true
        };
        
        this.sources.set(sourceId, sourceData);
        
        // Create visual representation
        this.createSourceMesh(sourceData);
        this.updateSourceList();
        
        return sourceId;
    }
    
    createSourceMesh(sourceData) {
        let geometry, material;
        
        switch (sourceData.type) {
            case 'mogi':
                geometry = new THREE.SphereGeometry(0.2, 16, 12);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0xff4444,
                    transparent: true,
                    opacity: 0.8
                });
                break;
                
            case 'okada':
                geometry = new THREE.BoxGeometry(1, 0.1, 0.6);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0x44ff44,
                    transparent: true,
                    opacity: 0.8
                });
                break;
                
            case 'dike':
                geometry = new THREE.BoxGeometry(0.1, 2, 1);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0x4444ff,
                    transparent: true,
                    opacity: 0.8
                });
                break;
                
            default:
                geometry = new THREE.SphereGeometry(0.15, 12, 8);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0xffff44,
                    transparent: true,
                    opacity: 0.8
                });
        }
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(sourceData.params.x, sourceData.params.y, sourceData.params.z);
        mesh.userData = { sourceId: sourceData.id };
        mesh.castShadow = true;
        
        this.scene.add(mesh);
        this.sourceMeshes.push(mesh);
    }
    
    calculateDeformation() {
        return new Promise(resolve => {
            const resolution = this.config.terrainResolution;
            const size = 20;
            const timeSteps = 365; // One year
            
            for (let t = 0; t < timeSteps; t++) {
                const displacements = new Float32Array(resolution * resolution * 3); // x, y, z
                const velocities = new Float32Array(resolution * resolution * 3);
                
                for (let i = 0; i < resolution; i++) {
                    for (let j = 0; j < resolution; j++) {
                        const idx = i * resolution + j;
                        const x = (i / (resolution - 1) - 0.5) * size;
                        const z = (j / (resolution - 1) - 0.5) * size;
                        const y = this.elevationData[idx];
                        
                        let totalDx = 0, totalDy = 0, totalDz = 0;
                        
                        // Calculate deformation from each source
                        for (const [sourceId, source] of this.sources) {
                            if (!source.active) continue;
                            
                            const deformation = this.calculateSourceDeformation(
                                source.type,
                                source.params,
                                x, y, z,
                                t / 365 // time in years
                            );
                            
                            totalDx += deformation.dx;
                            totalDy += deformation.dy;
                            totalDz += deformation.dz;
                        }
                        
                        const baseIdx = idx * 3;
                        displacements[baseIdx] = totalDx;
                        displacements[baseIdx + 1] = totalDy;
                        displacements[baseIdx + 2] = totalDz;
                        
                        // Calculate velocities (simple finite difference)
                        if (t > 0) {
                            const prevDisplacements = this.displacementData.get(t - 1);
                            if (prevDisplacements) {
                                velocities[baseIdx] = totalDx - prevDisplacements[baseIdx];
                                velocities[baseIdx + 1] = totalDy - prevDisplacements[baseIdx + 1];
                                velocities[baseIdx + 2] = totalDz - prevDisplacements[baseIdx + 2];
                            }
                        }
                    }
                }
                
                this.displacementData.set(t, displacements);
                this.velocityData.set(t, velocities);
            }
            
            resolve();
        });
    }
    
    calculateSourceDeformation(type, params, x, y, z, time) {
        switch (type) {
            case 'mogi':
                return this.mogiSource(params, x, y, z, time);
            case 'okada':
                return this.okadaSource(params, x, y, z, time);
            case 'yang':
                return this.yangSource(params, x, y, z, time);
            case 'penny':
                return this.pennySource(params, x, y, z, time);
            case 'sill':
                return this.sillSource(params, x, y, z, time);
            case 'dike':
                return this.dikeSource(params, x, y, z, time);
            default:
                return { dx: 0, dy: 0, dz: 0 };
        }
    }
    
    mogiSource(params, x, y, z, time) {
        const dx = x - params.x;
        const dy = y - params.y;
        const dz = z - params.z;
        
        const r = Math.sqrt(dx * dx + dz * dz);
        const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        if (R === 0) return { dx: 0, dy: 0, dz: 0 };
        
        // Mogi point source model
        const volume = params.volume * (1 + 0.1 * Math.sin(time * 2 * Math.PI)); // Time-varying volume
        const nu = params.nu || 0.25;
        const factor = (volume * (1 - nu)) / (Math.PI * R * R * R);
        
        return {
            dx: factor * dx,
            dy: factor * dy,
            dz: factor * dz
        };
    }
    
    okadaSource(params, x, y, z, time) {
        // Simplified Okada rectangular fault model
        const dx = x - params.x;
        const dy = y - params.y;
        const dz = z - params.z;
        
        // Convert angles to radians
        const strike = params.strike * Math.PI / 180;
        const dip = params.dip * Math.PI / 180;
        
        // Rotate coordinates to fault coordinate system
        const x_fault = dx * Math.cos(strike) + dz * Math.sin(strike);
        const y_fault = dy;
        const z_fault = -dx * Math.sin(strike) + dz * Math.cos(strike);
        
        // Simplified Green's function (actual Okada is more complex)
        const L = params.length / 1000; // km
        const W = params.width / 1000; // km
        const slip = params.slip * (1 + 0.05 * time); // Time-varying slip
        
        const factor = slip / (4 * Math.PI * Math.sqrt(x_fault * x_fault + y_fault * y_fault + 1));
        
        return {
            dx: factor * Math.cos(strike),
            dy: factor * 0.5,
            dz: factor * Math.sin(strike)
        };
    }
    
    yangSource(params, x, y, z, time) {
        // Yang et al. tensile crack model
        const dx = x - params.x;
        const dy = y - params.y;
        const dz = z - params.z;
        
        const r = Math.sqrt(dx * dx + dz * dz);
        const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        if (R === 0) return { dx: 0, dy: 0, dz: 0 };
        
        const opening = (params.opening || 1) * (1 + 0.1 * time);
        const factor = opening / (2 * Math.PI * R * R);
        
        return {
            dx: factor * dx,
            dy: factor * dy * 2, // Enhanced vertical component
            dz: factor * dz
        };
    }
    
    pennySource(params, x, y, z, time) {
        // Penny-shaped crack model
        const dx = x - params.x;
        const dy = y - params.y;
        const dz = z - params.z;
        
        const r = Math.sqrt(dx * dx + dz * dz);
        const radius = params.radius || 1;
        const pressure = (params.pressure || 1e6) * (1 + 0.2 * Math.sin(time * Math.PI));
        
        if (r > radius) return { dx: 0, dy: 0, dz: 0 };
        
        const factor = pressure * Math.sqrt(radius * radius - r * r) / (Math.PI * radius);
        
        return {
            dx: 0,
            dy: factor,
            dz: 0
        };
    }
    
    sillSource(params, x, y, z, time) {
        // Horizontal sill model
        const dx = x - params.x;
        const dy = y - params.y;
        const dz = z - params.z;
        
        const r = Math.sqrt(dx * dx + dz * dz);
        const thickness = (params.thickness || 0.1) * (1 + 0.15 * time);
        
        const factor = thickness * Math.exp(-r * r / 4) / (2 * Math.PI);
        
        return {
            dx: factor * dx / r,
            dy: factor * 2,
            dz: factor * dz / r
        };
    }
    
    dikeSource(params, x, y, z, time) {
        // Vertical dike model
        const dx = x - params.x;
        const dy = y - params.y;
        const dz = z - params.z;
        
        const opening = (params.opening || 0.5) * (1 + 0.1 * Math.cos(time * 2 * Math.PI));
        const length = params.length || 2;
        
        if (Math.abs(dz) > length / 2) return { dx: 0, dy: 0, dz: 0 };
        
        const factor = opening / (Math.PI * Math.sqrt(dx * dx + dy * dy + 1));
        
        return {
            dx: factor * Math.sign(dx),
            dy: factor * 0.3,
            dz: 0
        };
    }
    
    updateDeformationMesh() {
        if (!this.terrainMesh || !this.displacementData.has(Math.floor(this.currentTime))) {
            return;
        }
        
        const displacements = this.displacementData.get(Math.floor(this.currentTime));
        const geometry = this.terrainMesh.geometry;
        const positions = geometry.attributes.position.array;
        const originalPositions = geometry.userData.originalPositions || positions.slice();
        
        if (!geometry.userData.originalPositions) {
            geometry.userData.originalPositions = originalPositions;
        }
        
        const resolution = this.config.terrainResolution;
        const scale = this.config.deformationScale / 1000; // Convert to model scale
        
        for (let i = 0; i < resolution * resolution; i++) {
            const vertexIndex = i * 3;
            const dispIndex = i * 3;
            
            if (vertexIndex + 2 < positions.length && dispIndex + 2 < displacements.length) {
                positions[vertexIndex] = originalPositions[vertexIndex] + displacements[dispIndex] * scale;
                positions[vertexIndex + 1] = originalPositions[vertexIndex + 1] + displacements[dispIndex + 1] * scale;
                positions[vertexIndex + 2] = originalPositions[vertexIndex + 2] + displacements[dispIndex + 2] * scale;
            }
        }
        
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
    }
    
    updateVectorField() {
        // Remove existing vector field
        if (this.vectorField) {
            this.scene.remove(this.vectorField);
        }
        
        if (!this.config.showVectors || !this.velocityData.has(Math.floor(this.currentTime))) {
            return;
        }
        
        const velocities = this.velocityData.get(Math.floor(this.currentTime));
        const resolution = this.config.terrainResolution;
        const size = 20;
        const vectorScale = this.config.vectorScale / 1000;
        
        const vectorGeometry = new THREE.BufferGeometry();
        const vectorPositions = [];
        const vectorColors = [];
        
        // Sample every 8th point for performance
        const step = 8;
        
        for (let i = 0; i < resolution; i += step) {
            for (let j = 0; j < resolution; j += step) {
                const idx = i * resolution + j;
                const dispIdx = idx * 3;
                
                if (dispIdx + 2 >= velocities.length) continue;
                
                const x = (i / (resolution - 1) - 0.5) * size;
                const z = (j / (resolution - 1) - 0.5) * size;
                const y = this.elevationData[idx];
                
                const vx = velocities[dispIdx] * vectorScale;
                const vy = velocities[dispIdx + 1] * vectorScale;
                const vz = velocities[dispIdx + 2] * vectorScale;
                
                const magnitude = Math.sqrt(vx * vx + vy * vy + vz * vz);
                
                if (magnitude > 0.001) { // Only show significant vectors
                    // Start point
                    vectorPositions.push(x, y, z);
                    // End point
                    vectorPositions.push(x + vx, y + vy, z + vz);
                    
                    // Color based on magnitude
                    const normalizedMagnitude = Math.min(1, magnitude / 0.1);
                    const r = normalizedMagnitude;
                    const g = 1 - normalizedMagnitude;
                    const b = 0;
                    
                    vectorColors.push(r, g, b);
                    vectorColors.push(r, g, b);
                }
            }
        }
        
        vectorGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vectorPositions, 3));
        vectorGeometry.setAttribute('color', new THREE.Float32BufferAttribute(vectorColors, 3));
        
        const vectorMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 2
        });
        
        this.vectorField = new THREE.LineSegments(vectorGeometry, vectorMaterial);
        this.scene.add(this.vectorField);
    }
    
    setupEventListeners() {
        // Visualization toggles
        document.getElementById('terrainBtn').addEventListener('click', (e) => {
            this.config.showTerrain = !this.config.showTerrain;
            e.target.classList.toggle('active');
            this.terrainMesh.visible = this.config.showTerrain;
        });
        
        document.getElementById('vectorsBtn').addEventListener('click', (e) => {
            this.config.showVectors = !this.config.showVectors;
            e.target.classList.toggle('active');
            this.updateVectorField();
        });
        
        document.getElementById('sourcesBtn').addEventListener('click', (e) => {
            this.config.showSources = !this.config.showSources;
            e.target.classList.toggle('active');
            this.sourceMeshes.forEach(mesh => {
                mesh.visible = this.config.showSources;
            });
        });
        
        document.getElementById('wireframeBtn').addEventListener('click', (e) => {
            this.config.wireframeMode = !this.config.wireframeMode;
            e.target.classList.toggle('active');
            if (this.terrainMesh) {
                this.terrainMesh.material.wireframe = this.config.wireframeMode;
            }
        });
        
        // Scale controls
        const deformationScale = document.getElementById('deformationScale');
        const deformationValue = document.getElementById('deformationValue');
        
        deformationScale.addEventListener('input', (e) => {
            this.config.deformationScale = parseFloat(e.target.value);
            deformationValue.value = this.config.deformationScale;
            this.updateDeformationMesh();
        });
        
        const vectorScale = document.getElementById('vectorScale');
        const vectorValue = document.getElementById('vectorValue');
        
        vectorScale.addEventListener('input', (e) => {
            this.config.vectorScale = parseFloat(e.target.value);
            vectorValue.value = this.config.vectorScale;
            this.updateVectorField();
        });
        
        // Timeline controls
        const playPauseBtn = document.getElementById('playPauseBtn');
        const timelineSlider = document.getElementById('timelineSlider');
        const speedSlider = document.getElementById('speedSlider');
        const speedValue = document.getElementById('speedValue');
        
        playPauseBtn.addEventListener('click', () => {
            if (this.isAnimating) {
                this.stopTimeAnimation();
                playPauseBtn.textContent = '▶';
            } else {
                this.startTimeAnimation();
                playPauseBtn.textContent = '⏸';
            }
        });
        
        timelineSlider.addEventListener('click', (e) => {
            const rect = timelineSlider.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const progress = x / rect.width;
            this.currentTime = progress * (this.timeRange[1] - this.timeRange[0]) + this.timeRange[0];
            this.updateTimeDisplay();
            this.updateDeformationMesh();
            this.updateVectorField();
        });
        
        speedSlider.addEventListener('input', (e) => {
            this.animationSpeed = parseFloat(e.target.value);
            speedValue.textContent = this.animationSpeed.toFixed(1) + 'x';
        });
        
        // Source management
        document.getElementById('addSourceBtn').addEventListener('click', () => {
            const sourceType = document.getElementById('sourceType').value;
            this.showAddSourceDialog(sourceType);
        });
    }
    
    showAddSourceDialog(sourceType) {
        const dialog = document.createElement('div');
        dialog.className = 'source-dialog';
        dialog.innerHTML = `
            <style>
                .source-dialog {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    backdrop-filter: blur(10px);
                    z-index: 10000;
                    min-width: 300px;
                }
                .dialog-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                    padding-bottom: 8px;
                }
                .dialog-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }
                .dialog-input {
                    width: 100px;
                    padding: 4px 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 4px;
                    color: white;
                }
                .dialog-buttons {
                    display: flex;
                    gap: 8px;
                    justify-content: flex-end;
                    margin-top: 16px;
                }
                .dialog-btn {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .dialog-btn.primary {
                    background: #00ff88;
                    color: black;
                }
                .dialog-btn.secondary {
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                }
            </style>
            <div class="dialog-header">
                <h4>Add ${sourceType} Source</h4>
                <button class="close-btn" onclick="this.closest('.source-dialog').remove()">✕</button>
            </div>
            <div class="dialog-content" id="dialogContent">
                <!-- Content will be filled based on source type -->
            </div>
            <div class="dialog-buttons">
                <button class="dialog-btn secondary" onclick="this.closest('.source-dialog').remove()">Cancel</button>
                <button class="dialog-btn primary" id="addBtn">Add Source</button>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        // Fill content based on source type
        const content = dialog.querySelector('#dialogContent');
        this.fillSourceDialog(content, sourceType);
        
        // Handle add button
        dialog.querySelector('#addBtn').addEventListener('click', () => {
            const params = this.getSourceParams(content, sourceType);
            this.addSource(sourceType, params);
            dialog.remove();
            this.calculateDeformation();
        });
    }
    
    fillSourceDialog(content, sourceType) {
        let fields = [];
        
        switch (sourceType) {
            case 'mogi':
                fields = [
                    { name: 'x', label: 'X Position (km)', value: 0, step: 0.1 },
                    { name: 'y', label: 'Depth (km)', value: -2, step: 0.1 },
                    { name: 'z', label: 'Z Position (km)', value: 0, step: 0.1 },
                    { name: 'volume', label: 'Volume Change (m³)', value: 1000000, step: 100000 },
                    { name: 'nu', label: "Poisson's Ratio", value: 0.25, step: 0.01 }
                ];
                break;
            case 'okada':
                fields = [
                    { name: 'x', label: 'X Position (km)', value: 0, step: 0.1 },
                    { name: 'y', label: 'Depth (km)', value: -1, step: 0.1 },
                    { name: 'z', label: 'Z Position (km)', value: 0, step: 0.1 },
                    { name: 'length', label: 'Length (m)', value: 5000, step: 100 },
                    { name: 'width', label: 'Width (m)', value: 3000, step: 100 },
                    { name: 'strike', label: 'Strike (°)', value: 0, step: 1 },
                    { name: 'dip', label: 'Dip (°)', value: 90, step: 1 },
                    { name: 'slip', label: 'Slip (m)', value: 1, step: 0.1 }
                ];
                break;
            default:
                fields = [
                    { name: 'x', label: 'X Position (km)', value: 0, step: 0.1 },
                    { name: 'y', label: 'Depth (km)', value: -1, step: 0.1 },
                    { name: 'z', label: 'Z Position (km)', value: 0, step: 0.1 }
                ];
        }
        
        fields.forEach(field => {
            const row = document.createElement('div');
            row.className = 'dialog-row';
            row.innerHTML = `
                <label>${field.label}</label>
                <input type="number" class="dialog-input" name="${field.name}" 
                       value="${field.value}" step="${field.step}">
            `;
            content.appendChild(row);
        });
    }
    
    getSourceParams(content, sourceType) {
        const params = {};
        const inputs = content.querySelectorAll('input[name]');
        
        inputs.forEach(input => {
            params[input.name] = parseFloat(input.value);
        });
        
        return params;
    }
    
    updateSourceList() {
        const sourceList = document.getElementById('sourceList');
        sourceList.innerHTML = '';
        
        for (const [id, source] of this.sources) {
            const item = document.createElement('div');
            item.className = 'source-item';
            item.innerHTML = `
                <div>
                    <div class="source-type">${source.type.toUpperCase()}</div>
                    <div class="source-params">
                        ${source.params.x?.toFixed(1)}, ${source.params.y?.toFixed(1)}, ${source.params.z?.toFixed(1)}
                    </div>
                </div>
                <button class="toggle-btn" onclick="window.deformation3d.removeSource('${id}')">×</button>
            `;
            sourceList.appendChild(item);
        }
        
        // Update counter
        const counter = document.getElementById('sourceCounter');
        if (counter) {
            counter.textContent = this.sources.size;
        }
    }
    
    removeSource(sourceId) {
        this.sources.delete(sourceId);
        
        // Remove visual representation
        const meshIndex = this.sourceMeshes.findIndex(mesh => mesh.userData.sourceId === sourceId);
        if (meshIndex !== -1) {
            this.scene.remove(this.sourceMeshes[meshIndex]);
            this.sourceMeshes.splice(meshIndex, 1);
        }
        
        this.updateSourceList();
        this.calculateDeformation();
    }
    
    startTimeAnimation() {
        this.isAnimating = true;
        
        const animate = () => {
            if (!this.isAnimating) return;
            
            this.currentTime += this.animationSpeed * 0.5;
            if (this.currentTime > this.timeRange[1]) {
                this.currentTime = this.timeRange[0];
            }
            
            this.updateTimeDisplay();
            this.updateDeformationMesh();
            this.updateVectorField();
            this.updateStats();
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    stopTimeAnimation() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    updateTimeDisplay() {
        const timeDisplay = document.getElementById('timeDisplay');
        const timelineProgress = document.getElementById('timelineProgress');
        const currentTimeStats = document.getElementById('currentTimeStats');
        
        if (timeDisplay) {
            timeDisplay.textContent = `Day ${Math.floor(this.currentTime)}`;
        }
        
        if (timelineProgress) {
            const progress = (this.currentTime - this.timeRange[0]) / (this.timeRange[1] - this.timeRange[0]);
            timelineProgress.style.width = `${Math.max(0, Math.min(100, progress * 100))}%`;
        }
        
        if (currentTimeStats) {
            currentTimeStats.textContent = Math.floor(this.currentTime);
        }
    }
    
    updateStats() {
        // Calculate max displacement
        let maxDisplacement = 0;
        const currentDisplacements = this.displacementData.get(Math.floor(this.currentTime));
        if (currentDisplacements) {
            for (let i = 0; i < currentDisplacements.length; i += 3) {
                const dx = currentDisplacements[i];
                const dy = currentDisplacements[i + 1];
                const dz = currentDisplacements[i + 2];
                const magnitude = Math.sqrt(dx * dx + dy * dy + dz * dz) * 1000; // mm
                maxDisplacement = Math.max(maxDisplacement, magnitude);
            }
        }
        
        const maxDispElement = document.getElementById('maxDisplacement');
        if (maxDispElement) {
            maxDispElement.textContent = maxDisplacement.toFixed(1);
        }
        
        // Update vertex counter
        const vertexCounter = document.getElementById('vertexCounter');
        if (vertexCounter && this.terrainMesh) {
            const vertexCount = this.terrainMesh.geometry.attributes.position.count;
            vertexCounter.textContent = vertexCount.toLocaleString();
        }
    }
    
    render() {
        if (!this.isInitialized) return;
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        
        requestAnimationFrame(() => this.render());
    }
    
    exportImage() {
        const link = document.createElement('a');
        link.download = `deformation_3d_${new Date().toISOString().slice(0, 10)}.png`;
        link.href = this.renderer.domElement.toDataURL();
        link.click();
    }
    
    exportData() {
        const data = {
            sources: Array.from(this.sources.entries()),
            displacements: Array.from(this.displacementData.entries()),
            velocities: Array.from(this.velocityData.entries()),
            config: this.config,
            timeRange: this.timeRange
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.download = `deformation_data_${new Date().toISOString().slice(0, 10)}.json`;
        link.href = URL.createObjectURL(blob);
        link.click();
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `
            <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        background: #ff4444; color: white; padding: 20px; border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 10000; text-align: center;">
                ❌ ${message}
                <button onclick="this.parentElement.remove()" 
                        style="margin-top: 10px; padding: 8px 16px; background: rgba(255,255,255,0.2);
                               border: none; color: white; border-radius: 4px; cursor: pointer;">
                    Close
                </button>
            </div>
        `;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 10000);
    }
    
    destroy() {
        this.stopTimeAnimation();
        
        if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentElement) {
            this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
        }
        
        const container = document.querySelector('.deformation3d-container');
        if (container && container.parentElement) {
            container.parentElement.removeChild(container);
        }
    }
}

// Global initialization
if (typeof window !== 'undefined') {
    window.Deformation3D = Deformation3D;
    window.deformation3d = null;
    
    // Auto-initialize
    document.addEventListener('DOMContentLoaded', () => {
        window.deformation3d = new Deformation3D();
    });
    
    console.log('✅ 3D Deformation Modeling loaded - Advanced volcanic deformation visualization');
}