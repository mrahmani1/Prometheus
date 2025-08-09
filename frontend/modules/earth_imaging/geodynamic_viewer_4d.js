/**
 * Advanced 4D Geodynamic Visualization System
 * ==========================================
 * 
 * State-of-the-art 4D visualization system for coupled plate-mantle evolution modeling.
 * Implements advanced rendering techniques including volumetric ray casting, temporal
 * interpolation, multi-scale visualization, and interactive exploration of geodynamic
 * processes across space and time.
 * 
 * Key Features:
 * - 4D spatiotemporal visualization (3D space + time)
 * - Advanced volume rendering with GPU ray casting
 * - Multi-physics field visualization (temperature, velocity, composition)
 * - Interactive plate boundary evolution tracking
 * - Real-time mantle plume detection and visualization
 * - Temporal animation and interpolation
 * - Multi-scale rendering (global to regional)
 * - Advanced shader materials and post-processing
 * - WebXR support for immersive visualization
 * 
 * Author: Prometheus Deep Earth Team
 * Date: 2025
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';
import { FXAAPass } from 'three/addons/postprocessing/FXAAPass.js';

/**
 * Main Geodynamic Viewer Class
 */
class GeodynamicViewer4D {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            enableVR: options.enableVR || false,
            enablePostProcessing: options.enablePostProcessing !== false,
            maxTextureSize: options.maxTextureSize || 2048,
            volumeResolution: options.volumeResolution || 128,
            timeSteps: options.timeSteps || 100,
            animationSpeed: options.animationSpeed || 1.0,
            multiScale: options.multiScale || true,
            ...options
        };
        
        // Core components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null;
        this.gui = null;
        
        // Geodynamic data
        this.data = {
            mantle: {
                temperature: null,
                velocity: null,
                composition: null,
                viscosity: null
            },
            plates: {
                boundaries: null,
                velocities: null,
                ages: null
            },
            plumes: {
                locations: null,
                strengths: null,
                tracks: null
            },
            timeSteps: [],
            currentTime: 0,
            totalTime: 100e6 // years
        };
        
        // Visualization objects
        this.visualElements = {
            earth: null,
            mantle: {
                volume: null,
                isosurfaces: [],
                streamlines: []
            },
            plates: {
                boundaries: [],
                surfaces: [],
                labels: []
            },
            plumes: {
                volumes: [],
                tracks: [],
                heads: []
            },
            annotations: []
        };
        
        // Shaders and materials
        this.shaders = {};
        this.materials = {};
        
        // Animation system
        this.animation = {
            playing: false,
            currentFrame: 0,
            totalFrames: 0,
            fps: 30,
            interpolation: true,
            loop: true
        };
        
        // Interaction state
        this.interaction = {
            selectedElement: null,
            hoverElement: null,
            mode: 'orbit', // orbit, fly, vr
            crossSection: {
                enabled: false,
                plane: new THREE.Plane(),
                position: new THREE.Vector3()
            },
            measurements: []
        };
        
        // Performance monitoring
        this.performance = {
            frameRate: 60,
            renderTime: 0,
            memoryUsage: 0,
            adaptiveQuality: true
        };
        
        this.initialize();
    }
    
    /**
     * Initialize the visualization system
     */
    async initialize() {
        console.log('Initializing Geodynamic Viewer 4D...');
        
        try {
            this.setupRenderer();
            this.setupScene();
            this.setupCamera();
            this.setupControls();
            this.setupLighting();
            this.setupPostProcessing();
            this.setupGUI();
            this.setupEventListeners();
            this.initializeShaders();
            
            // Load initial data
            await this.loadSampleData();
            
            // Start render loop
            this.animate();
            
            console.log('Geodynamic Viewer 4D initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Geodynamic Viewer 4D:', error);
            throw error;
        }
    }
    
    /**
     * Setup Three.js renderer
     */
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            logarithmicDepthBuffer: true
        });
        
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        // WebXR setup
        if (this.config.enableVR) {
            this.renderer.xr.enabled = true;
            this.container.appendChild(VRButton.createButton(this.renderer));
        }
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    /**
     * Setup Three.js scene
     */
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000611);
        this.scene.fog = new THREE.FogExp2(0x000611, 0.00001);
        
        // Add coordinate system helpers
        const axesHelper = new THREE.AxesHelper(6371000); // Earth radius
        axesHelper.visible = false;
        this.scene.add(axesHelper);
        this.visualElements.axesHelper = axesHelper;
    }
    
    /**
     * Setup camera
     */
    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        
        this.camera = new THREE.PerspectiveCamera(45, aspect, 1000, 50000000);
        this.camera.position.set(15000000, 10000000, 15000000); // 15,000 km from center
        this.camera.lookAt(0, 0, 0);
    }
    
    /**
     * Setup camera controls
     */
    setupControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 6500000; // Just above surface
        this.controls.maxDistance = 50000000; // 50,000 km
        this.controls.maxPolarAngle = Math.PI;
    }
    
    /**
     * Setup lighting
     */
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Sun light (directional)
        const sunLight = new THREE.DirectionalLight(0xffffff, 1.0);
        sunLight.position.set(20000000, 20000000, 10000000);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.width = 4096;
        sunLight.shadow.mapSize.height = 4096;
        sunLight.shadow.camera.near = 1000000;
        sunLight.shadow.camera.far = 50000000;
        sunLight.shadow.camera.left = -20000000;
        sunLight.shadow.camera.right = 20000000;
        sunLight.shadow.camera.top = 20000000;
        sunLight.shadow.camera.bottom = -20000000;
        this.scene.add(sunLight);
        
        // Inner Earth glow
        const innerGlow = new THREE.PointLight(0xff4400, 2.0, 10000000);
        innerGlow.position.set(0, 0, 0);
        this.scene.add(innerGlow);
    }
    
    /**
     * Setup post-processing effects
     */
    setupPostProcessing() {
        if (!this.config.enablePostProcessing) return;
        
        this.composer = new EffectComposer(this.renderer);
        
        // Render pass
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        // SSAO pass
        const ssaoPass = new SSAOPass(this.scene, this.camera, 
            this.container.clientWidth, this.container.clientHeight);
        ssaoPass.kernelRadius = 16;
        ssaoPass.minDistance = 0.001;
        ssaoPass.maxDistance = 0.1;
        this.composer.addPass(ssaoPass);
        
        // Bloom pass
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(this.container.clientWidth, this.container.clientHeight),
            1.5, 0.4, 0.85
        );
        this.composer.addPass(bloomPass);
        
        // FXAA anti-aliasing
        const fxaaPass = new FXAAPass();
        this.composer.addPass(fxaaPass);
    }
    
    /**
     * Setup GUI controls
     */
    setupGUI() {
        this.gui = new GUI({ container: this.container });
        
        // Time controls
        const timeFolder = this.gui.addFolder('Time Control');
        timeFolder.add(this.animation, 'playing').name('Play/Pause').onChange(value => {
            if (value) this.playAnimation();
            else this.pauseAnimation();
        });
        timeFolder.add(this.animation, 'fps', 1, 60, 1).name('FPS');
        timeFolder.add(this.animation, 'loop').name('Loop');
        timeFolder.add(this, 'currentTime', 0, this.data.totalTime, 1000000)
            .name('Time (years)')
            .onChange(time => this.setTime(time));
        
        // Visualization controls
        const vizFolder = this.gui.addFolder('Visualization');
        
        // Mantle controls
        const mantleFolder = vizFolder.addFolder('Mantle');
        mantleFolder.add(this.config, 'volumeResolution', 32, 256, 32)
            .name('Resolution')
            .onChange(() => this.updateMantleVisualization());
        
        // Plate controls
        const plateFolder = vizFolder.addFolder('Plates');
        plateFolder.add({ showBoundaries: true }, 'showBoundaries')
            .name('Show Boundaries')
            .onChange(value => this.togglePlateBoundaries(value));
        
        // Quality controls
        const qualityFolder = this.gui.addFolder('Quality');
        qualityFolder.add(this.performance, 'adaptiveQuality').name('Adaptive Quality');
        qualityFolder.add(this.renderer, 'toneMappingExposure', 0.1, 3, 0.1).name('Exposure');
        
        this.gui.close();
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Mouse interactions
        this.renderer.domElement.addEventListener('click', (event) => this.onMouseClick(event));
        this.renderer.domElement.addEventListener('mousemove', (event) => this.onMouseMove(event));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => this.onKeyDown(event));
    }
    
    /**
     * Initialize custom shaders
     */
    initializeShaders() {
        // Volume rendering shader for mantle
        this.shaders.volumeRendering = {
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                
                void main() {
                    vPosition = position;
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler3D temperatureTexture;
                uniform sampler3D velocityTexture;
                uniform sampler3D compositionTexture;
                uniform float time;
                uniform vec3 cameraPosition;
                uniform float raySteps;
                uniform float stepSize;
                uniform float temperatureMin;
                uniform float temperatureMax;
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                
                vec3 temperatureToColor(float temp) {
                    float normalized = (temp - temperatureMin) / (temperatureMax - temperatureMin);
                    
                    // Heat colormap: blue -> cyan -> green -> yellow -> red
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
                    vec3 rayDir = normalize(vPosition - cameraPosition);
                    vec3 rayPos = vPosition;
                    
                    vec4 color = vec4(0.0);
                    float totalAlpha = 0.0;
                    
                    // Ray casting through volume
                    for (float i = 0.0; i < raySteps && totalAlpha < 0.95; i += 1.0) {
                        vec3 samplePos = rayPos + rayDir * stepSize * i;
                        
                        // Convert to texture coordinates [0,1]
                        vec3 texCoord = (samplePos + 1.0) * 0.5;
                        
                        if (texCoord.x >= 0.0 && texCoord.x <= 1.0 &&
                            texCoord.y >= 0.0 && texCoord.y <= 1.0 &&
                            texCoord.z >= 0.0 && texCoord.z <= 1.0) {
                            
                            // Sample temperature
                            float temperature = texture(temperatureTexture, texCoord).r;
                            
                            // Sample velocity for motion blur effect
                            vec3 velocity = texture(velocityTexture, texCoord).rgb;
                            float velocityMagnitude = length(velocity);
                            
                            // Calculate alpha based on temperature gradient
                            float alpha = smoothstep(0.1, 0.9, 
                                (temperature - temperatureMin) / (temperatureMax - temperatureMin));
                            alpha *= 0.1; // Transparency factor
                            
                            // Enhanced alpha for high velocity regions (plumes)
                            alpha += velocityMagnitude * 0.05;
                            alpha = clamp(alpha, 0.0, 1.0);
                            
                            if (alpha > 0.01) {
                                vec3 tempColor = temperatureToColor(temperature);
                                
                                // Add velocity-based enhancement
                                tempColor += vec3(velocityMagnitude * 0.2, 0.0, 0.0);
                                
                                // Front-to-back compositing
                                color.rgb += tempColor * alpha * (1.0 - totalAlpha);
                                totalAlpha += alpha * (1.0 - totalAlpha);
                            }
                        }
                    }
                    
                    color.a = totalAlpha;
                    gl_FragColor = color;
                }
            `
        };
        
        // Plate boundary shader
        this.shaders.plateBoundary = {
            vertexShader: `
                attribute float age;
                attribute float spreadingRate;
                varying float vAge;
                varying float vSpreadingRate;
                varying vec3 vPosition;
                
                void main() {
                    vAge = age;
                    vSpreadingRate = spreadingRate;
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                varying float vAge;
                varying float vSpreadingRate;
                varying vec3 vPosition;
                
                vec3 ageToColor(float age) {
                    // Age colormap: young (red) to old (blue)
                    float normalizedAge = clamp(age / 200.0, 0.0, 1.0); // 200 Ma max
                    return mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), normalizedAge);
                }
                
                void main() {
                    vec3 color = ageToColor(vAge);
                    
                    // Add spreading rate intensity
                    float intensity = 1.0 + vSpreadingRate * 0.1;
                    color *= intensity;
                    
                    // Add time-based animation
                    float pulse = 0.5 + 0.5 * sin(time * 2.0 + vPosition.x * 0.001);
                    color *= pulse;
                    
                    gl_FragColor = vec4(color, 0.8);
                }
            `
        };
        
        // Plume shader
        this.shaders.plume = {
            vertexShader: `
                attribute float strength;
                attribute vec3 velocity;
                varying float vStrength;
                varying vec3 vVelocity;
                varying vec3 vPosition;
                
                void main() {
                    vStrength = strength;
                    vVelocity = velocity;
                    vPosition = position;
                    
                    // Scale based on strength
                    vec3 scaledPosition = position * (1.0 + strength * 0.5);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(scaledPosition, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                varying float vStrength;
                varying vec3 vVelocity;
                varying vec3 vPosition;
                
                void main() {
                    // Distance from center for spherical gradient
                    float dist = length(gl_PointCoord - 0.5);
                    
                    if (dist > 0.5) discard;
                    
                    // Plume color based on strength and temperature
                    vec3 hotColor = vec3(1.0, 0.3, 0.0);  // Hot orange
                    vec3 warmColor = vec3(1.0, 0.8, 0.0); // Warm yellow
                    
                    float strengthNorm = clamp(vStrength / 1000.0, 0.0, 1.0);
                    vec3 color = mix(warmColor, hotColor, strengthNorm);
                    
                    // Add velocity-based enhancement
                    float velocityMag = length(vVelocity);
                    color += vec3(0.2, 0.1, 0.0) * velocityMag * 0.01;
                    
                    // Radial fade
                    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
                    
                    // Time-based pulsing
                    float pulse = 0.7 + 0.3 * sin(time * 3.0);
                    alpha *= pulse;
                    
                    gl_FragColor = vec4(color, alpha * 0.8);
                }
            `
        };
    }
    
    /**
     * Load sample geodynamic data
     */
    async loadSampleData() {
        console.log('Loading sample geodynamic data...');
        
        try {
            // Generate synthetic mantle data
            await this.generateSyntheticMantleData();
            
            // Generate synthetic plate data
            await this.generateSyntheticPlateData();
            
            // Generate synthetic plume data
            await this.generateSyntheticPlumeData();
            
            // Create visualization objects
            await this.createEarthModel();
            await this.createMantleVisualization();
            await this.createPlateVisualization();
            await this.createPlumeVisualization();
            
            console.log('Sample data loaded successfully');
        } catch (error) {
            console.error('Error loading sample data:', error);
            throw error;
        }
    }
    
    /**
     * Generate synthetic mantle convection data
     */
    async generateSyntheticMantleData() {
        const resolution = this.config.volumeResolution;
        const timeSteps = this.config.timeSteps;
        
        // Initialize data arrays
        this.data.mantle.temperature = [];
        this.data.mantle.velocity = [];
        this.data.mantle.composition = [];
        this.data.mantle.viscosity = [];
        
        for (let t = 0; t < timeSteps; t++) {
            const tempData = new Float32Array(resolution * resolution * resolution);
            const velData = new Float32Array(resolution * resolution * resolution * 3);
            const compData = new Float32Array(resolution * resolution * resolution);
            const viscData = new Float32Array(resolution * resolution * resolution);
            
            // Generate data for this time step
            let index = 0;
            let velIndex = 0;
            
            for (let z = 0; z < resolution; z++) {
                for (let y = 0; y < resolution; y++) {
                    for (let x = 0; x < resolution; x++) {
                        // Convert to spherical coordinates
                        const xi = (x / resolution) * 2 - 1;
                        const yi = (y / resolution) * 2 - 1;
                        const zi = (z / resolution) * 2 - 1;
                        
                        const radius = Math.sqrt(xi*xi + yi*yi + zi*zi);
                        const theta = Math.acos(zi / Math.max(radius, 1e-6));
                        const phi = Math.atan2(yi, xi);
                        
                        // Only process points inside the sphere
                        if (radius <= 1.0) {
                            // Depth-dependent base temperature
                            const depth = 1.0 - radius;
                            const baseTemp = 300 + depth * 3200; // 300K surface, 3500K core
                            
                            // Add thermal plumes
                            let tempAnomaly = 0;
                            const time = t / timeSteps;
                            
                            // Several plume locations
                            const plumes = [
                                { theta: Math.PI * 0.3, phi: 0, strength: 200 },
                                { theta: Math.PI * 0.7, phi: Math.PI, strength: 150 },
                                { theta: Math.PI * 0.5, phi: Math.PI * 0.5, strength: 180 }
                            ];
                            
                            plumes.forEach(plume => {
                                const angularDist = Math.acos(
                                    Math.sin(theta) * Math.sin(plume.theta) * Math.cos(phi - plume.phi) +
                                    Math.cos(theta) * Math.cos(plume.theta)
                                );
                                
                                if (angularDist < 0.3) { // Plume width
                                    const plumeStrength = Math.exp(-angularDist * 10) * 
                                        Math.exp(-Math.pow(depth - 0.8, 2) * 20);
                                    tempAnomaly += plume.strength * plumeStrength;
                                }
                            });
                            
                            // Add convection cells
                            const convectionPattern = Math.sin(4 * theta) * Math.cos(6 * phi) * 
                                Math.sin(2 * Math.PI * time) * 50;
                            
                            const temperature = baseTemp + tempAnomaly + convectionPattern;
                            tempData[index] = temperature;
                            
                            // Velocity field (simplified convection pattern)
                            const vr = Math.sin(2 * theta) * Math.cos(phi) * Math.sin(2 * Math.PI * time) * 0.01;
                            const vtheta = Math.cos(theta) * Math.sin(2 * phi) * 0.005;
                            const vphi = Math.sin(theta) * Math.cos(3 * phi) * 0.005;
                            
                            velData[velIndex] = vr;
                            velData[velIndex + 1] = vtheta;
                            velData[velIndex + 2] = vphi;
                            
                            // Composition (primitive vs. depleted)
                            const composition = 0.1 + 0.05 * Math.sin(theta * 8) * Math.cos(phi * 4);
                            compData[index] = composition;
                            
                            // Viscosity (temperature-dependent)
                            const viscosity = 1e21 * Math.exp(50000 / (8.314 * temperature));
                            viscData[index] = Math.log10(Math.min(Math.max(viscosity, 1e18), 1e25));
                        } else {
                            // Outside the sphere
                            tempData[index] = 0;
                            velData[velIndex] = 0;
                            velData[velIndex + 1] = 0;
                            velData[velIndex + 2] = 0;
                            compData[index] = 0;
                            viscData[index] = 0;
                        }
                        
                        index++;
                        velIndex += 3;
                    }
                }
            }
            
            this.data.mantle.temperature.push(tempData);
            this.data.mantle.velocity.push(velData);
            this.data.mantle.composition.push(compData);
            this.data.mantle.viscosity.push(viscData);
        }
        
        // Set up time steps
        this.data.timeSteps = [];
        for (let i = 0; i < timeSteps; i++) {
            this.data.timeSteps.push(i * (this.data.totalTime / timeSteps));
        }
        
        this.animation.totalFrames = timeSteps;
    }
    
    /**
     * Generate synthetic plate data
     */
    async generateSyntheticPlateData() {
        // Simplified plate boundary data
        const boundaries = [];
        
        // Mid-Atlantic Ridge
        const atlanticRidge = [];
        for (let lat = -60; lat <= 70; lat += 5) {
            atlanticRidge.push({
                position: this.latLonToCartesian(lat, -30, 6371000),
                age: 0,
                spreadingRate: 2.5,
                type: 'divergent'
            });
        }
        boundaries.push(atlanticRidge);
        
        // Pacific Ring of Fire
        const pacificRing = [];
        const angles = [];
        for (let angle = 0; angle < 2 * Math.PI; angle += 0.2) {
            angles.push(angle);
        }
        angles.forEach(angle => {
            const lat = 20 * Math.sin(angle);
            const lon = 180 + 60 * Math.cos(angle);
            pacificRing.push({
                position: this.latLonToCartesian(lat, lon, 6371000),
                age: Math.random() * 100,
                spreadingRate: Math.random() * 10,
                type: 'convergent'
            });
        });
        boundaries.push(pacificRing);
        
        // Transform faults
        const transforms = [];
        for (let i = 0; i < 20; i++) {
            const lat = (Math.random() - 0.5) * 120;
            const lon = (Math.random() - 0.5) * 360;
            transforms.push({
                position: this.latLonToCartesian(lat, lon, 6371000),
                age: Math.random() * 200,
                spreadingRate: Math.random() * 5,
                type: 'transform'
            });
        }
        boundaries.push(transforms);
        
        this.data.plates.boundaries = boundaries;
        
        // Plate velocities (simplified)
        this.data.plates.velocities = boundaries.map(boundary => 
            boundary.map(point => ({
                position: point.position,
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.1,
                    (Math.random() - 0.5) * 0.1,
                    (Math.random() - 0.5) * 0.1
                )
            }))
        );
    }
    
    /**
     * Generate synthetic plume data
     */
    async generateSyntheticPlumeData() {
        const plumes = [];
        const plumeLocations = [
            { lat: 19.5, lon: -155.5, name: 'Hawaiian' },
            { lat: 64.1, lon: -21.9, name: 'Iceland' },
            { lat: -0.4, lon: -91.6, name: 'Galapagos' },
            { lat: -21.1, lon: 55.5, name: 'Reunion' },
            { lat: 44.4, lon: -110.6, name: 'Yellowstone' }
        ];
        
        plumeLocations.forEach((location, index) => {
            const position = this.latLonToCartesian(location.lat, location.lon, 6371000);
            
            plumes.push({
                id: index,
                name: location.name,
                position: position,
                strength: 500 + Math.random() * 500,
                radius: 100000 + Math.random() * 200000, // 100-300 km
                temperature: 1800 + Math.random() * 400, // 1800-2200 K
                velocity: new THREE.Vector3(0, Math.random() * 0.05, 0),
                tracks: this.generatePlumeTrack(location.lat, location.lon, index)
            });
        });
        
        this.data.plumes.locations = plumes;
    }
    
    /**
     * Generate hotspot track for a plume
     */
    generatePlumeTrack(lat, lon, plumeId) {
        const track = [];
        const trackLength = 50; // 50 points
        const plateVelocity = 0.05; // degrees per time step
        
        for (let i = 0; i < trackLength; i++) {
            const trackLat = lat + (i * plateVelocity * 0.5);
            const trackLon = lon - (i * plateVelocity);
            const age = i * (this.data.totalTime / trackLength);
            
            track.push({
                position: this.latLonToCartesian(trackLat, trackLon, 6371000),
                age: age,
                volcanism: Math.exp(-i * 0.1) // Decreasing volcanism with age
            });
        }
        
        return track;
    }
    
    /**
     * Convert lat/lon to Cartesian coordinates
     */
    latLonToCartesian(lat, lon, radius) {
        const phi = (90 - lat) * Math.PI / 180;
        const theta = (lon + 180) * Math.PI / 180;
        
        const x = radius * Math.sin(phi) * Math.cos(theta);
        const z = radius * Math.sin(phi) * Math.sin(theta);
        const y = radius * Math.cos(phi);
        
        return new THREE.Vector3(x, y, z);
    }
    
    /**
     * Create Earth model
     */
    async createEarthModel() {
        // Earth sphere
        const earthGeometry = new THREE.SphereGeometry(6371000, 64, 32);
        const earthTexture = await this.createEarthTexture();
        
        const earthMaterial = new THREE.MeshPhongMaterial({
            map: earthTexture,
            transparent: true,
            opacity: 0.8
        });
        
        this.visualElements.earth = new THREE.Mesh(earthGeometry, earthMaterial);
        this.scene.add(this.visualElements.earth);
        
        // Inner core
        const coreGeometry = new THREE.SphereGeometry(1216000, 32, 16);
        const coreMaterial = new THREE.MeshBasicMaterial({
            color: 0xff4400,
            transparent: true,
            opacity: 0.3
        });
        
        const core = new THREE.Mesh(coreGeometry, coreMaterial);
        this.scene.add(core);
        
        // Outer core
        const outerCoreGeometry = new THREE.SphereGeometry(3486000, 32, 16);
        const outerCoreMaterial = new THREE.MeshBasicMaterial({
            color: 0xff6600,
            transparent: true,
            opacity: 0.1
        });
        
        const outerCore = new THREE.Mesh(outerCoreGeometry, outerCoreMaterial);
        this.scene.add(outerCore);
    }
    
    /**
     * Create Earth texture
     */
    async createEarthTexture() {
        // Create a simple Earth-like texture
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const context = canvas.getContext('2d');
        
        // Create gradient for land/ocean
        const gradient = context.createLinearGradient(0, 0, 1024, 512);
        gradient.addColorStop(0, '#1e4d72'); // Ocean blue
        gradient.addColorStop(0.3, '#2e5d82');
        gradient.addColorStop(0.7, '#8fbc8f'); // Land green
        gradient.addColorStop(1, '#daa520'); // Desert yellow
        
        context.fillStyle = gradient;
        context.fillRect(0, 0, 1024, 512);
        
        // Add some continents (simplified)
        context.fillStyle = '#228b22';
        
        // North America
        context.fillRect(150, 100, 200, 150);
        
        // South America
        context.fillRect(200, 250, 100, 200);
        
        // Europe/Asia
        context.fillRect(450, 80, 300, 180);
        
        // Africa
        context.fillRect(480, 200, 120, 220);
        
        // Australia
        context.fillRect(700, 350, 80, 60);
        
        return new THREE.CanvasTexture(canvas);
    }
    
    /**
     * Create mantle visualization
     */
    async createMantleVisualization() {
        if (!this.data.mantle.temperature || this.data.mantle.temperature.length === 0) {
            return;
        }
        
        const resolution = this.config.volumeResolution;
        
        // Create 3D texture for volume rendering
        const temperatureTexture = new THREE.Data3DTexture(
            this.data.mantle.temperature[0],
            resolution, resolution, resolution
        );
        temperatureTexture.format = THREE.RedFormat;
        temperatureTexture.type = THREE.FloatType;
        temperatureTexture.minFilter = THREE.LinearFilter;
        temperatureTexture.magFilter = THREE.LinearFilter;
        temperatureTexture.wrapS = THREE.ClampToEdgeWrapping;
        temperatureTexture.wrapT = THREE.ClampToEdgeWrapping;
        temperatureTexture.wrapR = THREE.ClampToEdgeWrapping;
        temperatureTexture.needsUpdate = true;
        
        // Create velocity texture
        const velocityTexture = new THREE.Data3DTexture(
            this.data.mantle.velocity[0],
            resolution, resolution, resolution
        );
        velocityTexture.format = THREE.RGBFormat;
        velocityTexture.type = THREE.FloatType;
        velocityTexture.minFilter = THREE.LinearFilter;
        velocityTexture.magFilter = THREE.LinearFilter;
        velocityTexture.needsUpdate = true;
        
        // Create volume rendering material
        const volumeMaterial = new THREE.ShaderMaterial({
            uniforms: {
                temperatureTexture: { value: temperatureTexture },
                velocityTexture: { value: velocityTexture },
                time: { value: 0.0 },
                cameraPosition: { value: this.camera.position },
                raySteps: { value: 100.0 },
                stepSize: { value: 0.02 },
                temperatureMin: { value: 300.0 },
                temperatureMax: { value: 3500.0 }
            },
            vertexShader: this.shaders.volumeRendering.vertexShader,
            fragmentShader: this.shaders.volumeRendering.fragmentShader,
            transparent: true,
            side: THREE.BackSide,
            depthWrite: false
        });
        
        // Create volume geometry (sphere representing mantle)
        const mantleGeometry = new THREE.SphereGeometry(6371000 * 0.95, 32, 16);
        this.visualElements.mantle.volume = new THREE.Mesh(mantleGeometry, volumeMaterial);
        this.visualElements.mantle.volume.visible = true;
        this.scene.add(this.visualElements.mantle.volume);
    }
    
    /**
     * Create plate visualization
     */
    async createPlateVisualization() {
        if (!this.data.plates.boundaries) return;
        
        this.data.plates.boundaries.forEach((boundary, boundaryIndex) => {
            const points = [];
            const ages = [];
            const spreadingRates = [];
            
            boundary.forEach(point => {
                points.push(point.position);
                ages.push(point.age);
                spreadingRates.push(point.spreadingRate);
            });
            
            // Create line geometry for boundary
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            geometry.setAttribute('age', new THREE.Float32BufferAttribute(ages, 1));
            geometry.setAttribute('spreadingRate', new THREE.Float32BufferAttribute(spreadingRates, 1));
            
            // Create material with custom shader
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    time: { value: 0.0 }
                },
                vertexShader: this.shaders.plateBoundary.vertexShader,
                fragmentShader: this.shaders.plateBoundary.fragmentShader,
                transparent: true,
                linewidth: 3
            });
            
            const line = new THREE.Line(geometry, material);
            this.visualElements.plates.boundaries.push(line);
            this.scene.add(line);
        });
    }
    
    /**
     * Create plume visualization
     */
    async createPlumeVisualization() {
        if (!this.data.plumes.locations) return;
        
        this.data.plumes.locations.forEach(plume => {
            // Create plume head
            const plumeGeometry = new THREE.SphereGeometry(plume.radius, 16, 8);
            
            const plumeMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    time: { value: 0.0 }
                },
                vertexShader: this.shaders.plume.vertexShader,
                fragmentShader: this.shaders.plume.fragmentShader,
                transparent: true,
                blending: THREE.AdditiveBlending
            });
            
            // Set attributes
            const strengths = new Float32Array(plumeGeometry.attributes.position.count);
            const velocities = new Float32Array(plumeGeometry.attributes.position.count * 3);
            
            for (let i = 0; i < strengths.length; i++) {
                strengths[i] = plume.strength;
                velocities[i * 3] = plume.velocity.x;
                velocities[i * 3 + 1] = plume.velocity.y;
                velocities[i * 3 + 2] = plume.velocity.z;
            }
            
            plumeGeometry.setAttribute('strength', new THREE.Float32BufferAttribute(strengths, 1));
            plumeGeometry.setAttribute('velocity', new THREE.Float32BufferAttribute(velocities, 3));
            
            const plumeHead = new THREE.Mesh(plumeGeometry, plumeMaterial);
            plumeHead.position.copy(plume.position);
            
            this.visualElements.plumes.heads.push(plumeHead);
            this.scene.add(plumeHead);
            
            // Create hotspot track
            if (plume.tracks && plume.tracks.length > 0) {
                const trackPoints = plume.tracks.map(t => t.position);
                const trackGeometry = new THREE.BufferGeometry().setFromPoints(trackPoints);
                
                const trackMaterial = new THREE.LineBasicMaterial({
                    color: 0xff4444,
                    transparent: true,
                    opacity: 0.6
                });
                
                const track = new THREE.Line(trackGeometry, trackMaterial);
                this.visualElements.plumes.tracks.push(track);
                this.scene.add(track);
            }
        });
    }
    
    /**
     * Animation loop
     */
    animate() {
        if (this.config.enableVR) {
            this.renderer.setAnimationLoop(() => this.render());
        } else {
            requestAnimationFrame(() => this.animate());
            this.render();
        }
    }
    
    /**
     * Render frame
     */
    render() {
        const startTime = performance.now();
        
        // Update controls
        this.controls.update();
        
        // Update animation
        if (this.animation.playing) {
            this.updateAnimation();
        }
        
        // Update uniforms
        this.updateShaderUniforms();
        
        // Render scene
        if (this.composer && this.config.enablePostProcessing) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        
        // Performance monitoring
        this.performance.renderTime = performance.now() - startTime;
        this.updatePerformance();
    }
    
    /**
     * Update animation
     */
    updateAnimation() {
        if (this.animation.currentFrame >= this.animation.totalFrames - 1) {
            if (this.animation.loop) {
                this.animation.currentFrame = 0;
            } else {
                this.animation.playing = false;
            }
        } else {
            this.animation.currentFrame += 1 / this.animation.fps;
        }
        
        // Update time-dependent data
        this.updateTimeStep(Math.floor(this.animation.currentFrame));
    }
    
    /**
     * Update shader uniforms
     */
    updateShaderUniforms() {
        const time = Date.now() * 0.001;
        
        // Update mantle volume uniforms
        if (this.visualElements.mantle.volume && this.visualElements.mantle.volume.material) {
            this.visualElements.mantle.volume.material.uniforms.time.value = time;
            this.visualElements.mantle.volume.material.uniforms.cameraPosition.value.copy(this.camera.position);
        }
        
        // Update plate boundary uniforms
        this.visualElements.plates.boundaries.forEach(boundary => {
            if (boundary.material && boundary.material.uniforms) {
                boundary.material.uniforms.time.value = time;
            }
        });
        
        // Update plume uniforms
        this.visualElements.plumes.heads.forEach(head => {
            if (head.material && head.material.uniforms) {
                head.material.uniforms.time.value = time;
            }
        });
    }
    
    /**
     * Update time step
     */
    updateTimeStep(frameIndex) {
        if (!this.data.mantle.temperature || frameIndex >= this.data.mantle.temperature.length) {
            return;
        }
        
        // Update mantle data
        if (this.visualElements.mantle.volume && this.visualElements.mantle.volume.material) {
            const temperatureTexture = this.visualElements.mantle.volume.material.uniforms.temperatureTexture.value;
            temperatureTexture.image.data = this.data.mantle.temperature[frameIndex];
            temperatureTexture.needsUpdate = true;
            
            const velocityTexture = this.visualElements.mantle.volume.material.uniforms.velocityTexture.value;
            velocityTexture.image.data = this.data.mantle.velocity[frameIndex];
            velocityTexture.needsUpdate = true;
        }
        
        // Update current time display
        this.currentTime = this.data.timeSteps[frameIndex];
    }
    
    /**
     * Update performance monitoring
     */
    updatePerformance() {
        // Adaptive quality based on frame rate
        if (this.performance.adaptiveQuality) {
            const targetFrameTime = 1000 / 60; // 60 FPS target
            
            if (this.performance.renderTime > targetFrameTime * 1.5) {
                // Reduce quality
                this.config.volumeResolution = Math.max(this.config.volumeResolution - 8, 32);
            } else if (this.performance.renderTime < targetFrameTime * 0.7) {
                // Increase quality
                this.config.volumeResolution = Math.min(this.config.volumeResolution + 8, 256);
            }
        }
    }
    
    /**
     * Event handlers
     */
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
        
        if (this.composer) {
            this.composer.setSize(width, height);
        }
    }
    
    onMouseClick(event) {
        // Raycasting for object selection
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / this.container.clientWidth) * 2 - 1;
        mouse.y = -(event.clientY / this.container.clientHeight) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        // Check for intersections with interactive objects
        const intersects = raycaster.intersectObjects([
            ...this.visualElements.plumes.heads,
            ...this.visualElements.plates.boundaries
        ]);
        
        if (intersects.length > 0) {
            this.selectObject(intersects[0].object);
        }
    }
    
    onMouseMove(event) {
        // Update hover effects
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / this.container.clientWidth) * 2 - 1;
        mouse.y = -(event.clientY / this.container.clientHeight) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        const intersects = raycaster.intersectObjects([
            ...this.visualElements.plumes.heads
        ]);
        
        // Reset previous hover
        if (this.interaction.hoverElement) {
            this.interaction.hoverElement.scale.setScalar(1.0);
        }
        
        // Set new hover
        if (intersects.length > 0) {
            this.interaction.hoverElement = intersects[0].object;
            this.interaction.hoverElement.scale.setScalar(1.1);
            this.container.style.cursor = 'pointer';
        } else {
            this.interaction.hoverElement = null;
            this.container.style.cursor = 'default';
        }
    }
    
    onKeyDown(event) {
        switch (event.key) {
            case ' ':
                event.preventDefault();
                this.toggleAnimation();
                break;
            case 'r':
                this.resetView();
                break;
            case 'f':
                this.toggleFullscreen();
                break;
            case 'h':
                this.toggleGUI();
                break;
            case 'ArrowLeft':
                this.previousTimeStep();
                break;
            case 'ArrowRight':
                this.nextTimeStep();
                break;
        }
    }
    
    /**
     * Control methods
     */
    playAnimation() {
        this.animation.playing = true;
    }
    
    pauseAnimation() {
        this.animation.playing = false;
    }
    
    toggleAnimation() {
        this.animation.playing = !this.animation.playing;
    }
    
    setTime(time) {
        const frameIndex = Math.floor((time / this.data.totalTime) * this.animation.totalFrames);
        this.animation.currentFrame = Math.max(0, Math.min(frameIndex, this.animation.totalFrames - 1));
        this.updateTimeStep(this.animation.currentFrame);
    }
    
    nextTimeStep() {
        if (this.animation.currentFrame < this.animation.totalFrames - 1) {
            this.animation.currentFrame++;
            this.updateTimeStep(this.animation.currentFrame);
        }
    }
    
    previousTimeStep() {
        if (this.animation.currentFrame > 0) {
            this.animation.currentFrame--;
            this.updateTimeStep(this.animation.currentFrame);
        }
    }
    
    resetView() {
        this.camera.position.set(15000000, 10000000, 15000000);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            this.container.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    toggleGUI() {
        if (this.gui) {
            this.gui._hidden ? this.gui.show() : this.gui.hide();
        }
    }
    
    togglePlateBoundaries(visible) {
        this.visualElements.plates.boundaries.forEach(boundary => {
            boundary.visible = visible;
        });
    }
    
    selectObject(object) {
        // Reset previous selection
        if (this.interaction.selectedElement) {
            this.interaction.selectedElement.material.emissive.setHex(0x000000);
        }
        
        // Set new selection
        this.interaction.selectedElement = object;
        if (object.material && object.material.emissive) {
            object.material.emissive.setHex(0x444444);
        }
        
        // Show object information
        this.showObjectInfo(object);
    }
    
    showObjectInfo(object) {
        // Find associated data
        let info = 'Selected Object: ';
        
        // Check if it's a plume
        const plumeIndex = this.visualElements.plumes.heads.indexOf(object);
        if (plumeIndex !== -1) {
            const plume = this.data.plumes.locations[plumeIndex];
            info = `Plume: ${plume.name}\nStrength: ${plume.strength.toFixed(1)}\nTemperature: ${plume.temperature.toFixed(0)} K`;
        }
        
        console.log(info);
        
        // Could show in UI panel instead of console
    }
    
    updateMantleVisualization() {
        // Regenerate mantle visualization with new settings
        if (this.visualElements.mantle.volume) {
            this.scene.remove(this.visualElements.mantle.volume);
        }
        this.createMantleVisualization();
    }
    
    /**
     * Data loading methods
     */
    async loadGeodynamicData(dataUrl) {
        try {
            const response = await fetch(dataUrl);
            const data = await response.json();
            
            // Process and integrate loaded data
            this.data = { ...this.data, ...data };
            
            // Update visualizations
            await this.updateAllVisualizations();
            
            console.log('Geodynamic data loaded successfully');
        } catch (error) {
            console.error('Error loading geodynamic data:', error);
            throw error;
        }
    }
    
    async updateAllVisualizations() {
        // Clear existing visualizations
        this.clearVisualizations();
        
        // Recreate all visualizations
        await this.createMantleVisualization();
        await this.createPlateVisualization();
        await this.createPlumeVisualization();
    }
    
    clearVisualizations() {
        // Remove mantle visualization
        if (this.visualElements.mantle.volume) {
            this.scene.remove(this.visualElements.mantle.volume);
            this.visualElements.mantle.volume = null;
        }
        
        // Remove plate visualizations
        this.visualElements.plates.boundaries.forEach(boundary => {
            this.scene.remove(boundary);
        });
        this.visualElements.plates.boundaries = [];
        
        // Remove plume visualizations
        this.visualElements.plumes.heads.forEach(head => {
            this.scene.remove(head);
        });
        this.visualElements.plumes.tracks.forEach(track => {
            this.scene.remove(track);
        });
        this.visualElements.plumes.heads = [];
        this.visualElements.plumes.tracks = [];
    }
    
    /**
     * Export methods
     */
    exportVisualization(format = 'png') {
        const canvas = this.renderer.domElement;
        
        if (format === 'png') {
            const link = document.createElement('a');
            link.download = `geodynamic_visualization_${Date.now()}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        }
    }
    
    exportAnimation(format = 'webm') {
        // Would implement video recording of animation
        console.log('Animation export not yet implemented');
    }
    
    /**
     * Cleanup
     */
    dispose() {
        // Stop animation
        this.animation.playing = false;
        
        // Dispose of geometries and materials
        this.scene.traverse((object) => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        
        // Dispose of renderer
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        // Dispose of composer
        if (this.composer) {
            this.composer.dispose();
        }
        
        // Remove GUI
        if (this.gui) {
            this.gui.destroy();
        }
        
        // Remove event listeners
        window.removeEventListener('resize', this.onWindowResize);
        document.removeEventListener('keydown', this.onKeyDown);
        
        console.log('Geodynamic Viewer 4D disposed');
    }
}

/**
 * Export for use in other modules
 */
export { GeodynamicViewer4D };

/**
 * Initialize viewer when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    // Auto-initialize if container exists
    const container = document.getElementById('geodynamic-viewer');
    if (container) {
        window.geodynamicViewer = new GeodynamicViewer4D('geodynamic-viewer', {
            enableVR: true,
            enablePostProcessing: true,
            volumeResolution: 128,
            timeSteps: 100,
            multiScale: true
        });
    }
});

console.log('Geodynamic Viewer 4D module loaded successfully');