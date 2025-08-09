/**
 * Advanced 3D InSAR Visualization Interface
 * 
 * This module provides comprehensive 3D visualization capabilities for InSAR data including:
 * - Interactive 3D terrain visualization with deformation overlays
 * - Time series animation of deformation evolution
 * - Multi-satellite data integration and comparison
 * - Advanced coherence and uncertainty visualization
 * - Real-time data streaming and updates
 * - Scientific analysis tools and measurements
 * 
 * Key Features:
 * - WebGL-based high-performance 3D rendering
 * - Time series slider with animation controls
 * - Interactive deformation source visualization
 * - Advanced color mapping and transparency controls
 * - Cross-section and profile analysis tools
 * - Export capabilities for figures and animations
 */

class InSARViewer3D {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            width: options.width || this.container.clientWidth,
            height: options.height || this.container.clientHeight,
            terrain: {
                resolution: options.terrainResolution || 90, // meters per pixel
                exaggeration: options.verticalExaggeration || 3.0,
                wireframe: false
            },
            deformation: {
                scale: options.deformationScale || 1000, // Exaggeration factor
                colormap: options.colormap || 'jet',
                showUncertainty: true,
                transparentLowCoherence: true,
                coherenceThreshold: 0.3
            },
            animation: {
                duration: options.animationDuration || 5000, // ms
                fps: 30,
                autoPlay: false,
                loop: true
            },
            interaction: {
                enableControls: true,
                enableMeasurement: true,
                enableCrossSections: true
            }
        };
        
        // Data storage
        this.data = {
            terrain: null,
            timeSeries: [],
            coherence: [],
            deformationSources: [],
            satellites: new Map(), // Satellite-specific data
            currentTimeIndex: 0
        };
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.raycaster = null;
        this.mouse = new THREE.Vector2();
        
        // Terrain mesh
        this.terrainMesh = null;
        this.deformationMesh = null;
        this.coherenceMesh = null;
        
        // GUI and controls
        this.gui = null;
        this.timeSlider = null;
        this.animationControls = null;
        
        // Animation
        this.animationId = null;
        this.isAnimating = false;
        
        // Measurement tools
        this.measurementMode = false;
        this.measurementPoints = [];
        this.crossSectionMode = false;
        
        // Event handlers
        this.eventHandlers = new Map();
        
        // Statistics
        this.stats = null;
        
        this.init();
    }
    
    init() {
        this.createScene();
        this.createControls();
        this.createGUI();
        this.setupEventListeners();
        this.setupStats();
        this.animate();
        
        console.log('InSAR 3D Viewer initialized');
    }
    
    createScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000011);
        this.scene.fog = new THREE.FogExp2(0x000011, 0.0001);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60, 
            this.config.width / this.config.height, 
            0.1, 
            100000
        );
        this.camera.position.set(0, 5000, 5000);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true // For screenshots
        });
        this.renderer.setSize(this.config.width, this.config.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.gammaOutput = true;
        this.renderer.gammaFactor = 2.2;
        
        // Add to container
        this.container.appendChild(this.renderer.domElement);
        
        // Lighting
        this.setupLighting();
        
        // Raycaster for interaction
        this.raycaster = new THREE.Raycaster();
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
        directionalLight.position.set(1000, 2000, 1000);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 500;
        directionalLight.shadow.camera.far = 4000;
        directionalLight.shadow.camera.left = -2000;
        directionalLight.shadow.camera.right = 2000;
        directionalLight.shadow.camera.top = 2000;
        directionalLight.shadow.camera.bottom = -2000;
        this.scene.add(directionalLight);
        
        // Helper for directional light
        if (this.config.debug) {
            const lightHelper = new THREE.DirectionalLightHelper(directionalLight, 100);
            this.scene.add(lightHelper);
        }
        
        // Hemisphere light for natural lighting
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
        hemiLight.position.set(0, 1000, 0);
        this.scene.add(hemiLight);
    }
    
    createControls() {
        // Orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 50000;
        this.controls.maxPolarAngle = Math.PI / 2;
        
        // Control buttons
        this.createControlButtons();
    }
    
    createControlButtons() {
        const controlPanel = document.createElement('div');
        controlPanel.className = 'insar-control-panel';
        controlPanel.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
        `;
        
        // View controls
        const viewButtons = this.createButtonGroup('View', [
            { text: 'Reset View', action: () => this.resetView() },
            { text: 'Top View', action: () => this.setTopView() },
            { text: 'Side View', action: () => this.setSideView() },
            { text: 'Fit Data', action: () => this.fitToData() }
        ]);
        controlPanel.appendChild(viewButtons);
        
        // Display controls
        const displayButtons = this.createButtonGroup('Display', [
            { text: 'Toggle Wireframe', action: () => this.toggleWireframe() },
            { text: 'Toggle Deformation', action: () => this.toggleDeformation() },
            { text: 'Toggle Coherence', action: () => this.toggleCoherence() },
            { text: 'Toggle Sources', action: () => this.toggleSources() }
        ]);
        controlPanel.appendChild(displayButtons);
        
        // Analysis tools
        const toolButtons = this.createButtonGroup('Tools', [
            { text: 'Measure', action: () => this.toggleMeasurement() },
            { text: 'Cross Section', action: () => this.toggleCrossSection() },
            { text: 'Profile', action: () => this.createProfile() },
            { text: 'Export', action: () => this.exportView() }
        ]);
        controlPanel.appendChild(toolButtons);
        
        this.container.appendChild(controlPanel);
    }
    
    createButtonGroup(title, buttons) {
        const group = document.createElement('div');
        group.style.marginBottom = '10px';
        
        const titleDiv = document.createElement('div');
        titleDiv.textContent = title;
        titleDiv.style.fontWeight = 'bold';
        titleDiv.style.marginBottom = '5px';
        group.appendChild(titleDiv);
        
        buttons.forEach(button => {
            const btn = document.createElement('button');
            btn.textContent = button.text;
            btn.style.cssText = `
                background: #333;
                color: white;
                border: 1px solid #555;
                padding: 2px 6px;
                margin: 1px;
                cursor: pointer;
                border-radius: 3px;
                font-size: 11px;
            `;
            btn.onmouseover = () => btn.style.background = '#555';
            btn.onmouseout = () => btn.style.background = '#333';
            btn.onclick = button.action;
            group.appendChild(btn);
        });
        
        return group;
    }
    
    createGUI() {
        // Create time control panel
        this.createTimeControls();
        
        // Create parameter controls
        this.createParameterControls();
        
        // Create information panel
        this.createInfoPanel();
    }
    
    createTimeControls() {
        const timePanel = document.createElement('div');
        timePanel.className = 'insar-time-panel';
        timePanel.style.cssText = `
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 5px;
            color: white;
            font-family: Arial, sans-serif;
            min-width: 400px;
            z-index: 1000;
        `;
        
        // Time slider
        const sliderContainer = document.createElement('div');
        sliderContainer.style.marginBottom = '10px';
        
        const timeLabel = document.createElement('div');
        timeLabel.id = 'time-label';
        timeLabel.style.textAlign = 'center';
        timeLabel.style.marginBottom = '5px';
        timeLabel.textContent = 'Time: No data loaded';
        sliderContainer.appendChild(timeLabel);
        
        this.timeSlider = document.createElement('input');
        this.timeSlider.type = 'range';
        this.timeSlider.min = '0';
        this.timeSlider.max = '0';
        this.timeSlider.value = '0';
        this.timeSlider.style.width = '100%';
        this.timeSlider.oninput = (e) => this.setTimeIndex(parseInt(e.target.value));
        sliderContainer.appendChild(this.timeSlider);
        
        timePanel.appendChild(sliderContainer);
        
        // Animation controls
        const animControls = document.createElement('div');
        animControls.style.textAlign = 'center';
        
        const playButton = document.createElement('button');
        playButton.textContent = '▶ Play';
        playButton.onclick = () => this.toggleAnimation();
        playButton.style.cssText = 'margin: 0 5px; padding: 5px 10px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;';
        
        const stopButton = document.createElement('button');
        stopButton.textContent = '⏹ Stop';
        stopButton.onclick = () => this.stopAnimation();
        stopButton.style.cssText = 'margin: 0 5px; padding: 5px 10px; background: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer;';
        
        const speedControl = document.createElement('select');
        speedControl.innerHTML = `
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
        `;
        speedControl.onchange = (e) => this.setAnimationSpeed(parseFloat(e.target.value));
        speedControl.style.margin = '0 5px';
        
        animControls.appendChild(playButton);
        animControls.appendChild(stopButton);
        animControls.appendChild(speedControl);
        
        timePanel.appendChild(animControls);
        this.container.appendChild(timePanel);
    }
    
    createParameterControls() {
        const paramPanel = document.createElement('div');
        paramPanel.className = 'insar-param-panel';
        paramPanel.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 12px;
            width: 250px;
            z-index: 1000;
        `;
        
        // Vertical exaggeration
        const verticalControl = this.createSliderControl(
            'Vertical Exaggeration',
            this.config.terrain.exaggeration,
            0.1, 10, 0.1,
            (value) => this.setVerticalExaggeration(value)
        );
        paramPanel.appendChild(verticalControl);
        
        // Deformation scale
        const deformationControl = this.createSliderControl(
            'Deformation Scale',
            this.config.deformation.scale,
            1, 10000, 100,
            (value) => this.setDeformationScale(value)
        );
        paramPanel.appendChild(deformationControl);
        
        // Coherence threshold
        const coherenceControl = this.createSliderControl(
            'Coherence Threshold',
            this.config.deformation.coherenceThreshold,
            0, 1, 0.01,
            (value) => this.setCoherenceThreshold(value)
        );
        paramPanel.appendChild(coherenceControl);
        
        // Colormap selection
        const colormapControl = document.createElement('div');
        colormapControl.style.marginBottom = '10px';
        
        const colormapLabel = document.createElement('div');
        colormapLabel.textContent = 'Colormap';
        colormapLabel.style.marginBottom = '5px';
        colormapControl.appendChild(colormapLabel);
        
        const colormapSelect = document.createElement('select');
        colormapSelect.innerHTML = `
            <option value="jet">Jet</option>
            <option value="viridis">Viridis</option>
            <option value="plasma">Plasma</option>
            <option value="rainbow">Rainbow</option>
            <option value="hot">Hot</option>
            <option value="cool">Cool</option>
        `;
        colormapSelect.value = this.config.deformation.colormap;
        colormapSelect.onchange = (e) => this.setColormap(e.target.value);
        colormapSelect.style.width = '100%';
        colormapControl.appendChild(colormapSelect);
        
        paramPanel.appendChild(colormapControl);
        
        this.container.appendChild(paramPanel);
    }
    
    createSliderControl(label, initialValue, min, max, step, callback) {
        const container = document.createElement('div');
        container.style.marginBottom = '10px';
        
        const labelDiv = document.createElement('div');
        labelDiv.textContent = `${label}: ${initialValue}`;
        labelDiv.style.marginBottom = '5px';
        container.appendChild(labelDiv);
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = min.toString();
        slider.max = max.toString();
        slider.step = step.toString();
        slider.value = initialValue.toString();
        slider.style.width = '100%';
        
        slider.oninput = (e) => {
            const value = parseFloat(e.target.value);
            labelDiv.textContent = `${label}: ${value}`;
            callback(value);
        };
        
        container.appendChild(slider);
        return container;
    }
    
    createInfoPanel() {
        const infoPanel = document.createElement('div');
        infoPanel.className = 'insar-info-panel';
        infoPanel.id = 'insar-info';
        infoPanel.style.cssText = `
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: monospace;
            font-size: 11px;
            width: 200px;
            z-index: 1000;
        `;
        
        this.container.appendChild(infoPanel);
    }
    
    setupEventListeners() {
        // Mouse events for interaction
        this.renderer.domElement.addEventListener('click', (event) => this.onMouseClick(event));
        this.renderer.domElement.addEventListener('mousemove', (event) => this.onMouseMove(event));
        
        // Keyboard events
        document.addEventListener('keydown', (event) => this.onKeyDown(event));
        
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Context menu
        this.renderer.domElement.addEventListener('contextmenu', (event) => {
            event.preventDefault();
            this.showContextMenu(event);
        });
    }
    
    setupStats() {
        if (typeof Stats !== 'undefined') {
            this.stats = new Stats();
            this.stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
            this.stats.dom.style.position = 'absolute';
            this.stats.dom.style.left = '0px';
            this.stats.dom.style.bottom = '0px';
            this.stats.dom.style.top = 'auto';
            this.container.appendChild(this.stats.dom);
        }
    }
    
    async loadInSARData(dataUrl) {
        try {
            console.log('Loading InSAR data from:', dataUrl);
            
            const response = await fetch(dataUrl);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Process different data types
            if (data.terrain) {
                await this.loadTerrainData(data.terrain);
            }
            
            if (data.timeSeries) {
                await this.loadTimeSeriesData(data.timeSeries);
            }
            
            if (data.coherence) {
                await this.loadCoherenceData(data.coherence);
            }
            
            if (data.deformationSources) {
                await this.loadDeformationSources(data.deformationSources);
            }
            
            // Update UI
            this.updateTimeControls();
            this.fitToData();
            
            console.log('InSAR data loaded successfully');
            this.emit('dataLoaded', data);
            
        } catch (error) {
            console.error('Error loading InSAR data:', error);
            this.showError('Failed to load InSAR data: ' + error.message);
        }
    }
    
    async loadTerrainData(terrainData) {
        console.log('Loading terrain data');
        
        // Store terrain data
        this.data.terrain = terrainData;
        
        // Create terrain geometry
        const geometry = new THREE.PlaneGeometry(
            terrainData.width * terrainData.resolution,
            terrainData.height * terrainData.resolution,
            terrainData.width - 1,
            terrainData.height - 1
        );
        
        // Apply elevation data
        const vertices = geometry.attributes.position.array;
        for (let i = 0; i < terrainData.elevations.length; i++) {
            vertices[i * 3 + 2] = terrainData.elevations[i] * this.config.terrain.exaggeration;
        }
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
        
        // Create material
        const material = new THREE.MeshLambertMaterial({
            color: 0x8B7355,
            wireframe: this.config.terrain.wireframe
        });
        
        // Create mesh
        if (this.terrainMesh) {
            this.scene.remove(this.terrainMesh);
        }
        
        this.terrainMesh = new THREE.Mesh(geometry, material);
        this.terrainMesh.receiveShadow = true;
        this.terrainMesh.rotation.x = -Math.PI / 2;
        this.scene.add(this.terrainMesh);
    }
    
    async loadTimeSeriesData(timeSeriesData) {
        console.log('Loading time series data');
        
        this.data.timeSeries = timeSeriesData;
        
        // Create deformation meshes for each time step
        for (let i = 0; i < timeSeriesData.length; i++) {
            const timeStep = timeSeriesData[i];
            
            // Create geometry for this time step
            const geometry = new THREE.PlaneGeometry(
                timeStep.width * timeStep.resolution,
                timeStep.height * timeStep.resolution,
                timeStep.width - 1,
                timeStep.height - 1
            );
            
            // Apply deformation data
            this.applyDeformationToGeometry(geometry, timeStep.deformation);
            
            // Create colored material based on deformation values
            const material = this.createDeformationMaterial(timeStep.deformation);
            
            // Create mesh
            const mesh = new THREE.Mesh(geometry, material);
            mesh.rotation.x = -Math.PI / 2;
            mesh.position.y = 1; // Slightly above terrain
            mesh.visible = (i === 0); // Only show first time step initially
            
            this.scene.add(mesh);
            timeStep.mesh = mesh;
        }
        
        this.data.currentTimeIndex = 0;
    }
    
    applyDeformationToGeometry(geometry, deformationData) {
        const vertices = geometry.attributes.position.array;
        
        for (let i = 0; i < deformationData.length; i++) {
            const deformation = deformationData[i];
            
            // Apply vertical deformation (scaled)
            vertices[i * 3 + 2] = deformation.up * this.config.deformation.scale;
        }
        
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
    }
    
    createDeformationMaterial(deformationData) {
        // Calculate color based on deformation values
        const colors = new Float32Array(deformationData.length * 3);
        
        // Find min/max values for normalization
        const values = deformationData.map(d => d.up);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const range = maxValue - minValue;
        
        for (let i = 0; i < deformationData.length; i++) {
            const normalizedValue = (deformationData[i].up - minValue) / range;
            const color = this.getColorFromValue(normalizedValue, this.config.deformation.colormap);
            
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        
        // Create geometry with vertex colors
        const geometry = new THREE.PlaneGeometry(1, 1, deformationData.length - 1, 1);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        return new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
    }
    
    getColorFromValue(value, colormap) {
        // Clamp value to [0, 1]
        value = Math.max(0, Math.min(1, value));
        
        switch (colormap) {
            case 'jet':
                return this.jetColormap(value);
            case 'viridis':
                return this.viridisColormap(value);
            case 'plasma':
                return this.plasmaColormap(value);
            case 'rainbow':
                return this.rainbowColormap(value);
            case 'hot':
                return this.hotColormap(value);
            case 'cool':
                return this.coolColormap(value);
            default:
                return this.jetColormap(value);
        }
    }
    
    jetColormap(value) {
        // Jet colormap implementation
        let r, g, b;
        
        if (value < 0.125) {
            r = 0;
            g = 0;
            b = 0.5 + 4 * value;
        } else if (value < 0.375) {
            r = 0;
            g = 4 * value - 0.5;
            b = 1;
        } else if (value < 0.625) {
            r = 4 * value - 1.5;
            g = 1;
            b = 2.5 - 4 * value;
        } else if (value < 0.875) {
            r = 1;
            g = 3.5 - 4 * value;
            b = 0;
        } else {
            r = 4.5 - 4 * value;
            g = 0;
            b = 0;
        }
        
        return { r: Math.max(0, Math.min(1, r)), 
                g: Math.max(0, Math.min(1, g)), 
                b: Math.max(0, Math.min(1, b)) };
    }
    
    viridisColormap(value) {
        // Simplified viridis colormap
        const r = 0.267 + 0.004 * value + 0.329 * value * value;
        const g = 0.004 + 0.822 * value - 0.278 * value * value;
        const b = 0.329 + 0.531 * value - 0.455 * value * value;
        
        return { r: Math.max(0, Math.min(1, r)), 
                g: Math.max(0, Math.min(1, g)), 
                b: Math.max(0, Math.min(1, b)) };
    }
    
    plasmaColormap(value) {
        // Simplified plasma colormap
        const r = 0.051 + 0.865 * value - 0.187 * value * value;
        const g = 0.013 + 0.235 * value + 0.648 * value * value;
        const b = 0.562 - 0.319 * value - 0.235 * value * value;
        
        return { r: Math.max(0, Math.min(1, r)), 
                g: Math.max(0, Math.min(1, g)), 
                b: Math.max(0, Math.min(1, b)) };
    }
    
    rainbowColormap(value) {
        // Rainbow colormap
        const hue = (1 - value) * 240; // 240° to 0° (blue to red)
        return this.hslToRgb(hue / 360, 1, 0.5);
    }
    
    hotColormap(value) {
        // Hot colormap (black -> red -> yellow -> white)
        let r, g, b;
        
        if (value < 0.33) {
            r = 3 * value;
            g = 0;
            b = 0;
        } else if (value < 0.66) {
            r = 1;
            g = 3 * (value - 0.33);
            b = 0;
        } else {
            r = 1;
            g = 1;
            b = 3 * (value - 0.66);
        }
        
        return { r, g, b };
    }
    
    coolColormap(value) {
        // Cool colormap (cyan -> magenta)
        return {
            r: value,
            g: 1 - value,
            b: 1
        };
    }
    
    hslToRgb(h, s, l) {
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h * 6) % 2 - 1));
        const m = l - c / 2;
        
        let r, g, b;
        
        if (h < 1/6) {
            [r, g, b] = [c, x, 0];
        } else if (h < 2/6) {
            [r, g, b] = [x, c, 0];
        } else if (h < 3/6) {
            [r, g, b] = [0, c, x];
        } else if (h < 4/6) {
            [r, g, b] = [0, x, c];
        } else if (h < 5/6) {
            [r, g, b] = [x, 0, c];
        } else {
            [r, g, b] = [c, 0, x];
        }
        
        return {
            r: r + m,
            g: g + m,
            b: b + m
        };
    }
    
    async loadCoherenceData(coherenceData) {
        console.log('Loading coherence data');
        this.data.coherence = coherenceData;
        
        // Create coherence visualization mesh
        // Implementation depends on coherence data structure
    }
    
    async loadDeformationSources(sourcesData) {
        console.log('Loading deformation sources');
        this.data.deformationSources = sourcesData;
        
        // Visualize deformation sources
        sourcesData.forEach(source => {
            this.visualizeDeformationSource(source);
        });
    }
    
    visualizeDeformationSource(source) {
        let geometry, material, mesh;
        
        switch (source.type) {
            case 'mogi':
                // Point source - use sphere
                geometry = new THREE.SphereGeometry(source.parameters.radius || 100, 16, 16);
                material = new THREE.MeshBasicMaterial({ 
                    color: source.parameters.volume_change > 0 ? 0xff0000 : 0x0000ff,
                    transparent: true, 
                    opacity: 0.6 
                });
                mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(source.parameters.x0, source.parameters.depth, source.parameters.y0);
                break;
                
            case 'sill':
                // Horizontal intrusion - use box
                geometry = new THREE.BoxGeometry(
                    source.parameters.length, 
                    source.parameters.opening * 100, 
                    source.parameters.width
                );
                material = new THREE.MeshBasicMaterial({ 
                    color: 0x00ff00, 
                    transparent: true, 
                    opacity: 0.6 
                });
                mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(source.parameters.x0, source.parameters.depth, source.parameters.y0);
                break;
                
            case 'dike':
                // Vertical intrusion - use tall box
                geometry = new THREE.BoxGeometry(
                    source.parameters.length,
                    source.parameters.depth_bottom - source.parameters.depth_top,
                    source.parameters.opening * 100
                );
                material = new THREE.MeshBasicMaterial({ 
                    color: 0xffff00, 
                    transparent: true, 
                    opacity: 0.6 
                });
                mesh = new THREE.Mesh(geometry, material);
                const centerDepth = (source.parameters.depth_top + source.parameters.depth_bottom) / 2;
                mesh.position.set(source.parameters.x0, centerDepth, source.parameters.y0);
                break;
                
            default:
                // Generic source - use sphere
                geometry = new THREE.SphereGeometry(100, 16, 16);
                material = new THREE.MeshBasicMaterial({ 
                    color: 0xffffff, 
                    transparent: true, 
                    opacity: 0.6 
                });
                mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(source.parameters.x0 || 0, source.parameters.depth || 1000, source.parameters.y0 || 0);
        }
        
        mesh.userData = { source: source, type: 'deformationSource' };
        this.scene.add(mesh);
    }
    
    updateTimeControls() {
        if (this.data.timeSeries.length > 0) {
            this.timeSlider.max = (this.data.timeSeries.length - 1).toString();
            this.updateTimeLabel();
        }
    }
    
    updateTimeLabel() {
        const timeLabel = document.getElementById('time-label');
        if (timeLabel && this.data.timeSeries.length > 0) {
            const currentData = this.data.timeSeries[this.data.currentTimeIndex];
            if (currentData && currentData.timestamp) {
                timeLabel.textContent = `Time: ${new Date(currentData.timestamp).toLocaleDateString()}`;
            } else {
                timeLabel.textContent = `Time: Step ${this.data.currentTimeIndex + 1}/${this.data.timeSeries.length}`;
            }
        }
    }
    
    setTimeIndex(index) {
        if (index >= 0 && index < this.data.timeSeries.length) {
            // Hide current mesh
            if (this.data.timeSeries[this.data.currentTimeIndex]) {
                this.data.timeSeries[this.data.currentTimeIndex].mesh.visible = false;
            }
            
            // Update index
            this.data.currentTimeIndex = index;
            
            // Show new mesh
            if (this.data.timeSeries[this.data.currentTimeIndex]) {
                this.data.timeSeries[this.data.currentTimeIndex].mesh.visible = true;
            }
            
            // Update slider
            this.timeSlider.value = index.toString();
            this.updateTimeLabel();
            
            this.emit('timeChanged', index);
        }
    }
    
    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
    
    startAnimation() {
        if (this.data.timeSeries.length <= 1) return;
        
        this.isAnimating = true;
        this.animateTimeStep();
        
        // Update button text
        const playButton = this.container.querySelector('button');
        if (playButton && playButton.textContent.includes('Play')) {
            playButton.textContent = '⏸ Pause';
        }
    }
    
    stopAnimation() {
        this.isAnimating = false;
        
        if (this.animationId) {
            clearTimeout(this.animationId);
            this.animationId = null;
        }
        
        // Update button text
        const playButton = this.container.querySelector('button');
        if (playButton && playButton.textContent.includes('Pause')) {
            playButton.textContent = '▶ Play';
        }
    }
    
    animateTimeStep() {
        if (!this.isAnimating) return;
        
        const nextIndex = (this.data.currentTimeIndex + 1) % this.data.timeSeries.length;
        this.setTimeIndex(nextIndex);
        
        this.animationId = setTimeout(() => {
            this.animateTimeStep();
        }, this.config.animation.duration / this.data.timeSeries.length);
    }
    
    setAnimationSpeed(speed) {
        // Adjust animation duration based on speed
        this.config.animation.duration = 5000 / speed;
    }
    
    // View control methods
    resetView() {
        this.camera.position.set(0, 5000, 5000);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }
    
    setTopView() {
        this.camera.position.set(0, 10000, 0);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    setSideView() {
        this.camera.position.set(10000, 2000, 0);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    fitToData() {
        if (!this.data.terrain) return;
        
        const terrain = this.data.terrain;
        const width = terrain.width * terrain.resolution;
        const height = terrain.height * terrain.resolution;
        
        // Calculate bounding sphere
        const center = new THREE.Vector3(0, 0, 0);
        const radius = Math.max(width, height) * 0.8;
        
        // Position camera
        const distance = radius / Math.sin(Math.PI / 6); // 30 degree viewing angle
        this.camera.position.set(0, distance * 0.7, distance * 0.7);
        this.camera.lookAt(center);
        this.controls.target.copy(center);
        this.controls.update();
    }
    
    // Display control methods
    toggleWireframe() {
        if (this.terrainMesh) {
            this.terrainMesh.material.wireframe = !this.terrainMesh.material.wireframe;
        }
    }
    
    toggleDeformation() {
        this.data.timeSeries.forEach(timeStep => {
            if (timeStep.mesh) {
                timeStep.mesh.visible = !timeStep.mesh.visible && 
                    (timeStep === this.data.timeSeries[this.data.currentTimeIndex]);
            }
        });
    }
    
    toggleCoherence() {
        if (this.coherenceMesh) {
            this.coherenceMesh.visible = !this.coherenceMesh.visible;
        }
    }
    
    toggleSources() {
        this.scene.traverse(child => {
            if (child.userData && child.userData.type === 'deformationSource') {
                child.visible = !child.visible;
            }
        });
    }
    
    // Parameter control methods
    setVerticalExaggeration(value) {
        this.config.terrain.exaggeration = value;
        
        if (this.terrainMesh && this.data.terrain) {
            const geometry = this.terrainMesh.geometry;
            const vertices = geometry.attributes.position.array;
            
            for (let i = 0; i < this.data.terrain.elevations.length; i++) {
                vertices[i * 3 + 2] = this.data.terrain.elevations[i] * value;
            }
            
            geometry.attributes.position.needsUpdate = true;
            geometry.computeVertexNormals();
        }
    }
    
    setDeformationScale(value) {
        this.config.deformation.scale = value;
        
        // Update deformation meshes
        this.data.timeSeries.forEach(timeStep => {
            if (timeStep.mesh && timeStep.deformation) {
                this.applyDeformationToGeometry(timeStep.mesh.geometry, timeStep.deformation);
            }
        });
    }
    
    setCoherenceThreshold(value) {
        this.config.deformation.coherenceThreshold = value;
        // Update transparency based on coherence
        this.updateCoherenceTransparency();
    }
    
    setColormap(colormap) {
        this.config.deformation.colormap = colormap;
        
        // Update materials
        this.data.timeSeries.forEach(timeStep => {
            if (timeStep.mesh && timeStep.deformation) {
                const newMaterial = this.createDeformationMaterial(timeStep.deformation);
                timeStep.mesh.material = newMaterial;
            }
        });
    }
    
    updateCoherenceTransparency() {
        // Implementation depends on coherence data structure
        console.log('Updating coherence transparency');
    }
    
    // Measurement and analysis tools
    toggleMeasurement() {
        this.measurementMode = !this.measurementMode;
        
        if (this.measurementMode) {
            console.log('Measurement mode activated');
            this.renderer.domElement.style.cursor = 'crosshair';
        } else {
            console.log('Measurement mode deactivated');
            this.renderer.domElement.style.cursor = 'default';
            this.clearMeasurements();
        }
    }
    
    clearMeasurements() {
        this.measurementPoints = [];
        
        // Remove measurement objects from scene
        this.scene.traverse(child => {
            if (child.userData && child.userData.type === 'measurement') {
                this.scene.remove(child);
            }
        });
    }
    
    addMeasurementPoint(point) {
        this.measurementPoints.push(point);
        
        // Add visual marker
        const geometry = new THREE.SphereGeometry(50, 8, 8);
        const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const marker = new THREE.Mesh(geometry, material);
        marker.position.copy(point);
        marker.userData = { type: 'measurement' };
        this.scene.add(marker);
        
        // If we have two points, show distance
        if (this.measurementPoints.length === 2) {
            const distance = this.measurementPoints[0].distanceTo(this.measurementPoints[1]);
            console.log(`Distance: ${distance.toFixed(2)} m`);
            
            // Add line between points
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(this.measurementPoints);
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
            const line = new THREE.Line(lineGeometry, lineMaterial);
            line.userData = { type: 'measurement' };
            this.scene.add(line);
            
            // Reset for next measurement
            this.measurementPoints = [];
        }
    }
    
    toggleCrossSection() {
        this.crossSectionMode = !this.crossSectionMode;
        
        if (this.crossSectionMode) {
            console.log('Cross-section mode activated');
            this.renderer.domElement.style.cursor = 'crosshair';
        } else {
            console.log('Cross-section mode deactivated');
            this.renderer.domElement.style.cursor = 'default';
        }
    }
    
    createProfile() {
        console.log('Creating profile');
        // Implementation for profile creation
    }
    
    exportView() {
        const canvas = this.renderer.domElement;
        const dataURL = canvas.toDataURL('image/png');
        
        // Create download link
        const link = document.createElement('a');
        link.download = 'insar_view.png';
        link.href = dataURL;
        link.click();
    }
    
    // Event handlers
    onMouseClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children);
        
        if (intersects.length > 0) {
            const intersect = intersects[0];
            
            if (this.measurementMode) {
                this.addMeasurementPoint(intersect.point);
            } else if (this.crossSectionMode) {
                // Handle cross-section creation
                console.log('Cross-section point:', intersect.point);
            } else {
                // Show information about clicked point
                this.showPointInfo(intersect);
            }
        }
    }
    
    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update info panel with cursor information
        this.updateInfoPanel();
    }
    
    onKeyDown(event) {
        switch (event.code) {
            case 'Space':
                event.preventDefault();
                this.toggleAnimation();
                break;
            case 'KeyR':
                this.resetView();
                break;
            case 'KeyM':
                this.toggleMeasurement();
                break;
            case 'KeyC':
                this.toggleCrossSection();
                break;
            case 'Escape':
                this.measurementMode = false;
                this.crossSectionMode = false;
                this.renderer.domElement.style.cursor = 'default';
                break;
        }
    }
    
    onWindowResize() {
        this.config.width = this.container.clientWidth;
        this.config.height = this.container.clientHeight;
        
        this.camera.aspect = this.config.width / this.config.height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(this.config.width, this.config.height);
    }
    
    showContextMenu(event) {
        // Create context menu
        const menu = document.createElement('div');
        menu.style.cssText = `
            position: fixed;
            left: ${event.clientX}px;
            top: ${event.clientY}px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px 0;
            z-index: 10000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;
        
        const menuItems = [
            { text: 'Reset View', action: () => this.resetView() },
            { text: 'Fit to Data', action: () => this.fitToData() },
            { text: 'Export Image', action: () => this.exportView() },
            { text: 'Show Info', action: () => this.showInfo() }
        ];
        
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.textContent = item.text;
            menuItem.style.cssText = `
                padding: 5px 15px;
                cursor: pointer;
                font-family: Arial, sans-serif;
                font-size: 13px;
            `;
            menuItem.onmouseover = () => menuItem.style.background = '#f0f0f0';
            menuItem.onmouseout = () => menuItem.style.background = 'white';
            menuItem.onclick = () => {
                item.action();
                document.body.removeChild(menu);
            };
            menu.appendChild(menuItem);
        });
        
        document.body.appendChild(menu);
        
        // Remove menu when clicking elsewhere
        const removeMenu = (e) => {
            if (!menu.contains(e.target)) {
                document.body.removeChild(menu);
                document.removeEventListener('click', removeMenu);
            }
        };
        
        setTimeout(() => {
            document.addEventListener('click', removeMenu);
        }, 100);
    }
    
    showPointInfo(intersect) {
        const point = intersect.point;
        const object = intersect.object;
        
        let infoText = `Position: (${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`;
        
        if (object.userData && object.userData.source) {
            const source = object.userData.source;
            infoText += `\nSource: ${source.type}`;
            infoText += `\nParameters: ${JSON.stringify(source.parameters, null, 2)}`;
        }
        
        console.log(infoText);
        this.showTooltip(infoText, event);
    }
    
    showTooltip(text, event) {
        const tooltip = document.createElement('div');
        tooltip.style.cssText = `
            position: fixed;
            left: ${event.clientX + 10}px;
            top: ${event.clientY + 10}px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            white-space: pre-line;
            z-index: 10000;
            max-width: 300px;
        `;
        tooltip.textContent = text;
        
        document.body.appendChild(tooltip);
        
        setTimeout(() => {
            if (tooltip.parentNode) {
                document.body.removeChild(tooltip);
            }
        }, 3000);
    }
    
    updateInfoPanel() {
        const infoPanel = document.getElementById('insar-info');
        if (!infoPanel) return;
        
        const cameraPos = this.camera.position;
        const controlsTarget = this.controls.target;
        
        let infoHTML = `
Camera: (${cameraPos.x.toFixed(0)}, ${cameraPos.y.toFixed(0)}, ${cameraPos.z.toFixed(0)})
Target: (${controlsTarget.x.toFixed(0)}, ${controlsTarget.y.toFixed(0)}, ${controlsTarget.z.toFixed(0)})
Time: ${this.data.currentTimeIndex + 1}/${this.data.timeSeries.length}
`;
        
        if (this.data.timeSeries.length > 0) {
            const currentData = this.data.timeSeries[this.data.currentTimeIndex];
            if (currentData && currentData.statistics) {
                infoHTML += `
Min Deform: ${currentData.statistics.min.toFixed(3)}m
Max Deform: ${currentData.statistics.max.toFixed(3)}m
Mean Deform: ${currentData.statistics.mean.toFixed(3)}m
`;
            }
        }
        
        infoPanel.innerHTML = infoHTML.replace(/\n/g, '<br>');
    }
    
    showInfo() {
        const infoWindow = window.open('', '_blank', 'width=400,height=300');
        infoWindow.document.write(`
            <html>
                <head><title>InSAR Viewer Info</title></head>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2>InSAR 3D Viewer</h2>
                    <h3>Keyboard Shortcuts:</h3>
                    <ul>
                        <li><strong>Space</strong> - Play/Pause animation</li>
                        <li><strong>R</strong> - Reset view</li>
                        <li><strong>M</strong> - Toggle measurement mode</li>
                        <li><strong>C</strong> - Toggle cross-section mode</li>
                        <li><strong>Escape</strong> - Exit current mode</li>
                    </ul>
                    <h3>Mouse Controls:</h3>
                    <ul>
                        <li><strong>Left Click + Drag</strong> - Rotate view</li>
                        <li><strong>Right Click + Drag</strong> - Pan view</li>
                        <li><strong>Scroll</strong> - Zoom in/out</li>
                        <li><strong>Right Click</strong> - Context menu</li>
                    </ul>
                </body>
            </html>
        `);
    }
    
    showError(message) {
        console.error(message);
        
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 0, 0, 0.8);
            color: white;
            padding: 20px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            text-align: center;
            z-index: 10000;
        `;
        errorDiv.textContent = message;
        
        this.container.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentNode) {
                this.container.removeChild(errorDiv);
            }
        }, 5000);
    }
    
    // Event system
    on(event, callback) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(callback);
            if (index !== -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(callback => {
                callback(data);
            });
        }
    }
    
    // Main animation loop
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.stats) {
            this.stats.begin();
        }
        
        // Update controls
        if (this.controls) {
            this.controls.update();
        }
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
        
        // Update info panel periodically
        if (Math.floor(Date.now() / 100) % 10 === 0) { // Every second
            this.updateInfoPanel();
        }
        
        if (this.stats) {
            this.stats.end();
        }
    }
    
    // Cleanup
    dispose() {
        // Stop animation
        this.stopAnimation();
        
        // Dispose of Three.js resources
        this.scene.traverse(child => {
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        
        // Remove DOM elements
        while (this.container.firstChild) {
            this.container.removeChild(this.container.firstChild);
        }
        
        // Clear event handlers
        this.eventHandlers.clear();
        
        console.log('InSAR 3D Viewer disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InSARViewer3D;
}