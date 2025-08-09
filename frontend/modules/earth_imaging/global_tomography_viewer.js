/**
 * Global Tomography Viewer - Advanced 3D Earth Visualization
 * ==========================================================
 * 
 * State-of-the-art 3D visualization system for global seismic tomography models
 * with real-time rendering, multi-scale visualization, and interactive analysis.
 * 
 * Author: Prometheus Platform
 * Date: 2025-08-08
 * Version: 2.0.0
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { SAOPass } from 'three/examples/jsm/postprocessing/SAOPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/examples/jsm/shaders/FXAAShader.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';

/**
 * Advanced shader materials for Earth visualization
 */
class EarthShaderMaterials {
    static createVelocityShader() {
        return {
            uniforms: {
                'velocityTexture': { value: null },
                'depthTexture': { value: null },
                'colorScale': { value: new THREE.Vector2(3000, 8000) }, // Velocity range
                'opacity': { value: 0.8 },
                'time': { value: 0.0 },
                'lightPosition': { value: new THREE.Vector3(10, 10, 10) },
                'ambientLight': { value: 0.3 },
                'diffuseStrength': { value: 0.7 },
                'specularStrength': { value: 0.5 },
                'shininess': { value: 32.0 },
                'contourLevels': { value: 10 },
                'showContours': { value: false },
                'useLogScale': { value: false }
            },
            
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec3 vWorldPosition;
                varying vec2 vUv;
                uniform float time;
                
                void main() {
                    vPosition = position;
                    vNormal = normalize(normalMatrix * normal);
                    vUv = uv;
                    
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            
            fragmentShader: `
                uniform sampler3D velocityTexture;
                uniform sampler3D depthTexture;
                uniform vec2 colorScale;
                uniform float opacity;
                uniform float time;
                uniform vec3 lightPosition;
                uniform float ambientLight;
                uniform float diffuseStrength;
                uniform float specularStrength;
                uniform float shininess;
                uniform int contourLevels;
                uniform bool showContours;
                uniform bool useLogScale;
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec3 vWorldPosition;
                varying vec2 vUv;
                
                // Advanced colormaps for seismic velocities
                vec3 velocityColormap(float value) {
                    // Normalize value to [0, 1]
                    float t = (value - colorScale.x) / (colorScale.y - colorScale.x);
                    t = clamp(t, 0.0, 1.0);
                    
                    // Scientific colormap (similar to seismic tomography conventions)
                    vec3 c1 = vec3(0.1, 0.0, 0.5); // Deep blue (low velocity)
                    vec3 c2 = vec3(0.0, 0.5, 1.0); // Blue
                    vec3 c3 = vec3(0.0, 1.0, 1.0); // Cyan  
                    vec3 c4 = vec3(1.0, 1.0, 0.0); // Yellow
                    vec3 c5 = vec3(1.0, 0.5, 0.0); // Orange
                    vec3 c6 = vec3(1.0, 0.0, 0.0); // Red (high velocity)
                    
                    if (t < 0.2) {
                        return mix(c1, c2, t * 5.0);
                    } else if (t < 0.4) {
                        return mix(c2, c3, (t - 0.2) * 5.0);
                    } else if (t < 0.6) {
                        return mix(c3, c4, (t - 0.4) * 5.0);
                    } else if (t < 0.8) {
                        return mix(c4, c5, (t - 0.6) * 5.0);
                    } else {
                        return mix(c5, c6, (t - 0.8) * 5.0);
                    }
                }
                
                // Ray casting for volume rendering
                vec4 raycast(vec3 rayStart, vec3 rayDir, float maxDistance) {
                    vec3 currentPos = rayStart;
                    vec4 accumColor = vec4(0.0);
                    float stepSize = 0.01;
                    int maxSteps = int(maxDistance / stepSize);
                    
                    for (int i = 0; i < maxSteps; i++) {
                        if (accumColor.a > 0.99) break;
                        
                        // Sample velocity at current position
                        vec3 samplePos = (currentPos + 1.0) * 0.5; // Normalize to [0,1]
                        float velocity = texture(velocityTexture, samplePos).r;
                        
                        if (velocity > 0.0) {
                            vec3 color = velocityColormap(velocity);
                            float alpha = 0.05 * opacity;
                            
                            // Apply contours if enabled
                            if (showContours) {
                                float contourValue = floor(velocity / 500.0) * 500.0;
                                float contourWeight = abs(velocity - contourValue) < 50.0 ? 1.5 : 1.0;
                                alpha *= contourWeight;
                            }
                            
                            // Front-to-back blending
                            accumColor.rgb += (1.0 - accumColor.a) * alpha * color;
                            accumColor.a += (1.0 - accumColor.a) * alpha;
                        }
                        
                        currentPos += rayDir * stepSize;
                        
                        // Check bounds
                        if (any(lessThan(currentPos, vec3(-1.0))) || any(greaterThan(currentPos, vec3(1.0)))) {
                            break;
                        }
                    }
                    
                    return accumColor;
                }
                
                void main() {
                    // Calculate ray direction in world space
                    vec3 rayDir = normalize(vWorldPosition - cameraPosition);
                    vec3 rayStart = vPosition;
                    
                    // Perform volume ray casting
                    vec4 volumeColor = raycast(rayStart, rayDir, 2.0);
                    
                    // Apply lighting
                    vec3 normal = normalize(vNormal);
                    vec3 lightDir = normalize(lightPosition - vWorldPosition);
                    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
                    vec3 reflectDir = reflect(-lightDir, normal);
                    
                    // Ambient component
                    vec3 ambient = ambientLight * volumeColor.rgb;
                    
                    // Diffuse component
                    float diffuse = max(dot(normal, lightDir), 0.0);
                    vec3 diffuseColor = diffuseStrength * diffuse * volumeColor.rgb;
                    
                    // Specular component
                    float specular = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                    vec3 specularColor = specularStrength * specular * vec3(1.0);
                    
                    vec3 finalColor = ambient + diffuseColor + specularColor;
                    
                    gl_FragColor = vec4(finalColor, volumeColor.a * opacity);
                }
            `
        };
    }
    
    static createDepthSliceShader() {
        return {
            uniforms: {
                'velocityTexture': { value: null },
                'depth': { value: 0.5 },
                'colorScale': { value: new THREE.Vector2(3000, 8000) },
                'opacity': { value: 1.0 },
                'showContours': { value: false },
                'contourSpacing': { value: 200.0 }
            },
            
            vertexShader: `
                varying vec2 vUv;
                varying vec3 vPosition;
                
                void main() {
                    vUv = uv;
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            
            fragmentShader: `
                uniform sampler3D velocityTexture;
                uniform float depth;
                uniform vec2 colorScale;
                uniform float opacity;
                uniform bool showContours;
                uniform float contourSpacing;
                
                varying vec2 vUv;
                varying vec3 vPosition;
                
                vec3 velocityColormap(float value) {
                    float t = (value - colorScale.x) / (colorScale.y - colorScale.x);
                    t = clamp(t, 0.0, 1.0);
                    
                    vec3 c1 = vec3(0.1, 0.0, 0.5);
                    vec3 c2 = vec3(0.0, 0.5, 1.0);
                    vec3 c3 = vec3(0.0, 1.0, 1.0);
                    vec3 c4 = vec3(1.0, 1.0, 0.0);
                    vec3 c5 = vec3(1.0, 0.5, 0.0);
                    vec3 c6 = vec3(1.0, 0.0, 0.0);
                    
                    if (t < 0.2) return mix(c1, c2, t * 5.0);
                    else if (t < 0.4) return mix(c2, c3, (t - 0.2) * 5.0);
                    else if (t < 0.6) return mix(c3, c4, (t - 0.4) * 5.0);
                    else if (t < 0.8) return mix(c4, c5, (t - 0.6) * 5.0);
                    else return mix(c5, c6, (t - 0.8) * 5.0);
                }
                
                void main() {
                    vec3 samplePos = vec3(vUv.x, vUv.y, depth);
                    float velocity = texture(velocityTexture, samplePos).r;
                    
                    vec3 color = velocityColormap(velocity);
                    
                    // Add contour lines if enabled
                    if (showContours) {
                        float contourValue = floor(velocity / contourSpacing) * contourSpacing;
                        float contourDist = abs(velocity - contourValue);
                        
                        if (contourDist < contourSpacing * 0.02) {
                            color = mix(color, vec3(0.0), 0.8);
                        }
                    }
                    
                    gl_FragColor = vec4(color, opacity);
                }
            `
        };
    }
    
    static createEarthSurfaceShader() {
        return {
            uniforms: {
                'earthTexture': { value: null },
                'bumpTexture': { value: null },
                'nightTexture': { value: null },
                'cloudsTexture': { value: null },
                'lightPosition': { value: new THREE.Vector3(10, 10, 10) },
                'time': { value: 0.0 },
                'showTopography': { value: true },
                'bumpScale': { value: 0.1 }
            },
            
            vertexShader: `
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec3 vWorldPosition;
                varying vec2 vUv;
                uniform float time;
                uniform float bumpScale;
                uniform sampler2D bumpTexture;
                
                void main() {
                    vUv = uv;
                    vPosition = position;
                    
                    // Sample bump map for displacement
                    float height = texture2D(bumpTexture, uv).r;
                    vec3 displacedPosition = position + normal * height * bumpScale;
                    
                    vNormal = normalize(normalMatrix * normal);
                    vec4 worldPosition = modelMatrix * vec4(displacedPosition, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPosition, 1.0);
                }
            `,
            
            fragmentShader: `
                uniform sampler2D earthTexture;
                uniform sampler2D bumpTexture;
                uniform sampler2D nightTexture;
                uniform sampler2D cloudsTexture;
                uniform vec3 lightPosition;
                uniform float time;
                uniform bool showTopography;
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec3 vWorldPosition;
                varying vec2 vUv;
                
                void main() {
                    // Sample textures
                    vec3 dayColor = texture2D(earthTexture, vUv).rgb;
                    vec3 nightColor = texture2D(nightTexture, vUv).rgb;
                    float clouds = texture2D(cloudsTexture, vUv + vec2(time * 0.0001, 0.0)).r;
                    
                    // Lighting calculations
                    vec3 normal = normalize(vNormal);
                    vec3 lightDir = normalize(lightPosition - vWorldPosition);
                    
                    float dayFactor = max(dot(normal, lightDir), 0.0);
                    float nightFactor = 1.0 - dayFactor;
                    
                    // Mix day and night textures
                    vec3 color = mix(nightColor, dayColor, dayFactor);
                    
                    // Add clouds with transparency
                    color = mix(color, vec3(1.0), clouds * 0.3);
                    
                    // Add atmospheric scattering effect at edges
                    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
                    float fresnel = 1.0 - abs(dot(viewDir, normal));
                    vec3 atmosphereColor = vec3(0.3, 0.6, 1.0);
                    color = mix(color, atmosphereColor, fresnel * 0.2);
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `
        };
    }
}

/**
 * Advanced data loading and processing
 */
class TomographyDataManager {
    constructor() {
        this.models = new Map();
        this.loadingPromises = new Map();
        this.cache = new Map();
        this.compressionEnabled = true;
        
        this.setupWorkers();
    }
    
    setupWorkers() {
        // Setup web workers for data processing
        this.workers = [];
        const numWorkers = navigator.hardwareConcurrency || 4;
        
        for (let i = 0; i < numWorkers; i++) {
            const worker = new Worker(new URL('./workers/tomography_processor.js', import.meta.url));
            worker.onmessage = this.handleWorkerMessage.bind(this);
            this.workers.push(worker);
        }
        
        this.workerQueue = [];
        this.activeWorkers = new Set();
    }
    
    async loadTomographyModel(modelId, url, format = 'netcdf') {
        console.log(`Loading tomography model: ${modelId} from ${url}`);
        
        if (this.loadingPromises.has(modelId)) {
            return this.loadingPromises.get(modelId);
        }
        
        const loadingPromise = this._loadModelData(url, format);
        this.loadingPromises.set(modelId, loadingPromise);
        
        try {
            const modelData = await loadingPromise;
            this.models.set(modelId, modelData);
            
            // Process model for GPU upload
            const processedData = await this.processModelForGPU(modelData);
            
            return {
                id: modelId,
                data: modelData,
                processed: processedData,
                metadata: this.extractMetadata(modelData)
            };
            
        } catch (error) {
            console.error(`Failed to load model ${modelId}:`, error);
            throw error;
        } finally {
            this.loadingPromises.delete(modelId);
        }
    }
    
    async _loadModelData(url, format) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        switch (format) {
            case 'netcdf':
                return this.parseNetCDF(await response.arrayBuffer());
            case 'hdf5':
                return this.parseHDF5(await response.arrayBuffer());
            case 'binary':
                return this.parseBinary(await response.arrayBuffer());
            case 'json':
                return response.json();
            default:
                throw new Error(`Unsupported format: ${format}`);
        }
    }
    
    async parseNetCDF(arrayBuffer) {
        // Parse NetCDF format (would use netcdfjs or similar library)
        // For demonstration, return mock data structure
        return {
            dimensions: { x: 360, y: 180, z: 50 },
            variables: {
                velocity_p: new Float32Array(360 * 180 * 50).fill(0).map(() => 
                    3000 + Math.random() * 5000),
                velocity_s: new Float32Array(360 * 180 * 50).fill(0).map(() => 
                    1500 + Math.random() * 2500),
                latitude: new Float32Array(180).fill(0).map((_, i) => -90 + i),
                longitude: new Float32Array(360).fill(0).map((_, i) => -180 + i),
                depth: new Float32Array(50).fill(0).map((_, i) => i * 50)
            },
            attributes: {
                title: 'Global Tomography Model',
                institution: 'Prometheus Platform',
                creation_date: new Date().toISOString()
            }
        };
    }
    
    async parseHDF5(arrayBuffer) {
        // Parse HDF5 format (would use h5wasm or similar library)
        // Mock implementation
        return this.parseNetCDF(arrayBuffer);
    }
    
    async parseBinary(arrayBuffer) {
        // Parse custom binary format
        const view = new DataView(arrayBuffer);
        let offset = 0;
        
        // Read header
        const dimensions = {
            x: view.getUint32(offset, true), offset += 4,
            y: view.getUint32(offset, true), offset += 4,
            z: view.getUint32(offset, true), offset += 4
        };
        
        const totalPoints = dimensions.x * dimensions.y * dimensions.z;
        
        // Read velocity data
        const velocity_p = new Float32Array(arrayBuffer, offset, totalPoints);
        offset += totalPoints * 4;
        
        const velocity_s = new Float32Array(arrayBuffer, offset, totalPoints);
        
        return {
            dimensions,
            variables: { velocity_p, velocity_s },
            attributes: { format: 'binary' }
        };
    }
    
    async processModelForGPU(modelData) {
        return new Promise((resolve) => {
            const worker = this.getAvailableWorker();
            const taskId = Math.random().toString(36).substr(2, 9);
            
            worker.postMessage({
                type: 'processForGPU',
                taskId: taskId,
                data: modelData
            });
            
            const handleResult = (event) => {
                if (event.data.taskId === taskId) {
                    worker.removeEventListener('message', handleResult);
                    this.releaseWorker(worker);
                    resolve(event.data.result);
                }
            };
            
            worker.addEventListener('message', handleResult);
        });
    }
    
    getAvailableWorker() {
        for (const worker of this.workers) {
            if (!this.activeWorkers.has(worker)) {
                this.activeWorkers.add(worker);
                return worker;
            }
        }
        
        // If no workers available, return first one (queue the task)
        return this.workers[0];
    }
    
    releaseWorker(worker) {
        this.activeWorkers.delete(worker);
    }
    
    handleWorkerMessage(event) {
        console.log('Worker message:', event.data);
    }
    
    extractMetadata(modelData) {
        const { dimensions, variables, attributes } = modelData;
        
        return {
            dimensions,
            bounds: this.calculateBounds(variables),
            statistics: this.calculateStatistics(variables),
            attributes
        };
    }
    
    calculateBounds(variables) {
        const bounds = {};
        
        for (const [name, data] of Object.entries(variables)) {
            if (data instanceof Float32Array || data instanceof Float64Array) {
                bounds[name] = {
                    min: Math.min(...data),
                    max: Math.max(...data),
                    mean: data.reduce((a, b) => a + b, 0) / data.length
                };
            }
        }
        
        return bounds;
    }
    
    calculateStatistics(variables) {
        const stats = {};
        
        for (const [name, data] of Object.entries(variables)) {
            if (data instanceof Float32Array || data instanceof Float64Array) {
                const sorted = [...data].sort((a, b) => a - b);
                const n = sorted.length;
                
                stats[name] = {
                    min: sorted[0],
                    max: sorted[n - 1],
                    mean: sorted.reduce((a, b) => a + b, 0) / n,
                    median: n % 2 === 0 ? 
                        (sorted[n/2 - 1] + sorted[n/2]) / 2 : 
                        sorted[Math.floor(n/2)],
                    std: Math.sqrt(sorted.reduce((acc, val) => 
                        acc + Math.pow(val - stats[name].mean, 2), 0) / n)
                };
            }
        }
        
        return stats;
    }
    
    createTexture3D(data, dimensions, renderer) {
        const { x, y, z } = dimensions;
        const texture = new THREE.Data3DTexture(data, x, y, z);
        
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.wrapR = THREE.ClampToEdgeWrapping;
        texture.generateMipmaps = false;
        texture.needsUpdate = true;
        
        return texture;
    }
}

/**
 * Interactive controls and user interface
 */
class TomographyControls {
    constructor(viewer) {
        this.viewer = viewer;
        this.gui = null;
        this.params = {
            // Visualization parameters
            opacity: 0.8,
            colorScale: { min: 3000, max: 8000 },
            showContours: false,
            contourLevels: 10,
            useLogScale: false,
            
            // Slicing parameters
            depthSlice: 0.5,
            showDepthSlice: true,
            latitudeSlice: 0.0,
            longitudeSlice: 0.0,
            
            // Animation parameters
            animationSpeed: 1.0,
            autoRotate: false,
            
            // Earth parameters
            showEarth: true,
            showTopography: true,
            earthOpacity: 1.0,
            
            // Quality parameters
            renderQuality: 'high',
            volumeSteps: 256,
            enableBloom: true,
            enableAO: false
        };
        
        this.setupGUI();
        this.setupKeyboardControls();
    }
    
    setupGUI() {
        this.gui = new GUI({ width: 350 });
        
        // Visualization folder
        const visualFolder = this.gui.addFolder('Visualization');
        visualFolder.add(this.params, 'opacity', 0, 1, 0.01)
            .onChange(() => this.updateVisualization());
        
        const colorFolder = visualFolder.addFolder('Color Scale');
        colorFolder.add(this.params.colorScale, 'min', 1000, 10000, 100)
            .onChange(() => this.updateColorScale());
        colorFolder.add(this.params.colorScale, 'max', 1000, 10000, 100)
            .onChange(() => this.updateColorScale());
        colorFolder.add(this.params, 'useLogScale')
            .onChange(() => this.updateColorScale());
        
        const contourFolder = visualFolder.addFolder('Contours');
        contourFolder.add(this.params, 'showContours')
            .onChange(() => this.updateVisualization());
        contourFolder.add(this.params, 'contourLevels', 1, 50, 1)
            .onChange(() => this.updateVisualization());
        
        visualFolder.open();
        
        // Slicing folder
        const sliceFolder = this.gui.addFolder('Cross Sections');
        sliceFolder.add(this.params, 'showDepthSlice')
            .onChange(() => this.updateSlicing());
        sliceFolder.add(this.params, 'depthSlice', 0, 1, 0.01)
            .name('Depth')
            .onChange(() => this.updateSlicing());
        sliceFolder.add(this.params, 'latitudeSlice', -90, 90, 1)
            .name('Latitude')
            .onChange(() => this.updateSlicing());
        sliceFolder.add(this.params, 'longitudeSlice', -180, 180, 1)
            .name('Longitude')
            .onChange(() => this.updateSlicing());
        
        sliceFolder.open();
        
        // Earth folder
        const earthFolder = this.gui.addFolder('Earth');
        earthFolder.add(this.params, 'showEarth')
            .onChange(() => this.updateEarth());
        earthFolder.add(this.params, 'showTopography')
            .onChange(() => this.updateEarth());
        earthFolder.add(this.params, 'earthOpacity', 0, 1, 0.01)
            .onChange(() => this.updateEarth());
        
        // Animation folder
        const animFolder = this.gui.addFolder('Animation');
        animFolder.add(this.params, 'animationSpeed', 0, 5, 0.1);
        animFolder.add(this.params, 'autoRotate')
            .onChange(() => this.updateAnimation());
        
        // Quality folder
        const qualityFolder = this.gui.addFolder('Rendering Quality');
        qualityFolder.add(this.params, 'renderQuality', ['low', 'medium', 'high', 'ultra'])
            .onChange(() => this.updateRenderQuality());
        qualityFolder.add(this.params, 'volumeSteps', 64, 512, 16)
            .onChange(() => this.updateRenderQuality());
        qualityFolder.add(this.params, 'enableBloom')
            .onChange(() => this.updatePostProcessing());
        qualityFolder.add(this.params, 'enableAO')
            .name('Ambient Occlusion')
            .onChange(() => this.updatePostProcessing());
        
        // Model selection
        this.modelDropdown = this.gui.add({ model: 'Global-P' }, 'model', 
            ['Global-P', 'Global-S', 'Regional-P', 'Regional-S'])
            .name('Tomography Model')
            .onChange((value) => this.switchModel(value));
        
        // Export controls
        const exportFolder = this.gui.addFolder('Export');
        exportFolder.add({ exportImage: () => this.exportImage() }, 'exportImage')
            .name('Export Image');
        exportFolder.add({ exportAnimation: () => this.exportAnimation() }, 'exportAnimation')
            .name('Export Animation');
        exportFolder.add({ exportData: () => this.exportData() }, 'exportData')
            .name('Export Model Data');
    }
    
    setupKeyboardControls() {
        document.addEventListener('keydown', (event) => {
            switch(event.code) {
                case 'Space':
                    event.preventDefault();
                    this.params.autoRotate = !this.params.autoRotate;
                    this.updateAnimation();
                    break;
                    
                case 'KeyH':
                    this.gui.show(this.gui._hidden);
                    break;
                    
                case 'KeyR':
                    this.resetView();
                    break;
                    
                case 'KeyS':
                    this.exportImage();
                    break;
                    
                case 'KeyT':
                    this.params.showTopography = !this.params.showTopography;
                    this.updateEarth();
                    break;
                    
                case 'KeyC':
                    this.params.showContours = !this.params.showContours;
                    this.updateVisualization();
                    break;
                    
                case 'ArrowUp':
                    this.params.depthSlice = Math.min(1, this.params.depthSlice + 0.05);
                    this.updateSlicing();
                    break;
                    
                case 'ArrowDown':
                    this.params.depthSlice = Math.max(0, this.params.depthSlice - 0.05);
                    this.updateSlicing();
                    break;
            }
        });
    }
    
    updateVisualization() {
        this.viewer.updateVolumeRendering(this.params);
    }
    
    updateColorScale() {
        this.viewer.updateColorScale(this.params.colorScale, this.params.useLogScale);
    }
    
    updateSlicing() {
        this.viewer.updateCrossSections(this.params);
    }
    
    updateEarth() {
        this.viewer.updateEarthVisualization(this.params);
    }
    
    updateAnimation() {
        this.viewer.setAutoRotate(this.params.autoRotate, this.params.animationSpeed);
    }
    
    updateRenderQuality() {
        this.viewer.setRenderQuality(this.params.renderQuality, this.params.volumeSteps);
    }
    
    updatePostProcessing() {
        this.viewer.updatePostProcessing(this.params);
    }
    
    switchModel(modelName) {
        this.viewer.switchTomographyModel(modelName);
    }
    
    resetView() {
        this.viewer.resetCamera();
    }
    
    exportImage() {
        this.viewer.exportImage();
    }
    
    exportAnimation() {
        this.viewer.exportAnimation();
    }
    
    exportData() {
        this.viewer.exportCurrentModel();
    }
}

/**
 * Main Global Tomography Viewer Class
 */
export class GlobalTomographyViewer {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            enableStats: true,
            enableVR: false,
            initialModel: 'Global-P',
            renderQuality: 'high',
            ...config
        };
        
        // Core Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null;
        
        // Visualization objects
        this.volumeGroup = new THREE.Group();
        this.earthGroup = new THREE.Group();
        this.sliceGroup = new THREE.Group();
        
        // Data management
        this.dataManager = new TomographyDataManager();
        this.currentModel = null;
        this.velocityTexture = null;
        
        // Animation
        this.clock = new THREE.Clock();
        this.animationFrame = null;
        this.isAnimating = false;
        
        // Performance monitoring
        this.stats = null;
        this.frameCount = 0;
        this.lastFPSUpdate = 0;
        
        this.init();
    }
    
    async init() {
        console.log('Initializing Global Tomography Viewer...');
        
        try {
            this.setupScene();
            this.setupRenderer();
            this.setupCamera();
            this.setupControls();
            this.setupLighting();
            this.setupPostProcessing();
            
            if (this.config.enableStats) {
                this.setupStats();
            }
            
            // Setup UI controls
            this.uiControls = new TomographyControls(this);
            
            // Create Earth sphere
            this.createEarthSphere();
            
            // Load initial model
            if (this.config.initialModel) {
                await this.loadModel(this.config.initialModel);
            }
            
            // Start render loop
            this.startRenderLoop();
            
            // Setup event listeners
            this.setupEventListeners();
            
            console.log('Global Tomography Viewer initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize viewer:', error);
            throw error;
        }
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000011);
        
        // Add groups to scene
        this.scene.add(this.volumeGroup);
        this.scene.add(this.earthGroup);
        this.scene.add(this.sliceGroup);
    }
    
    setupRenderer() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            throw new Error(`Container with id '${this.containerId}' not found`);
        }
        
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            powerPreference: 'high-performance',
            precision: 'highp'
        });
        
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        container.appendChild(this.renderer.domElement);
    }
    
    setupCamera() {
        const container = document.getElementById(this.containerId);
        const aspect = container.clientWidth / container.clientHeight;
        
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 10000);
        this.camera.position.set(15, 10, 15);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 8;
        this.controls.maxDistance = 100;
        this.controls.maxPolarAngle = Math.PI;
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Directional light (Sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(10, 10, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = -20;
        directionalLight.shadow.camera.right = 20;
        directionalLight.shadow.camera.top = 20;
        directionalLight.shadow.camera.bottom = -20;
        this.scene.add(directionalLight);
        
        // Point lights for rim lighting
        const rimLight1 = new THREE.PointLight(0x4488ff, 0.5, 100);
        rimLight1.position.set(-20, 0, 0);
        this.scene.add(rimLight1);
        
        const rimLight2 = new THREE.PointLight(0xff8844, 0.5, 100);
        rimLight2.position.set(20, 0, 0);
        this.scene.add(rimLight2);
    }
    
    setupPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        
        // Base render pass
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        // SAO (Screen-space Ambient Occlusion)
        this.saoPass = new SAOPass(this.scene, this.camera, false, true);
        this.saoPass.params.output = SAOPass.OUTPUT.Beauty;
        this.saoPass.params.saoBias = 0.5;
        this.saoPass.params.saoIntensity = 0.1;
        this.saoPass.params.saoScale = 1.0;
        this.saoPass.params.saoKernelRadius = 100;
        this.composer.addPass(this.saoPass);
        
        // Bloom pass
        this.bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5, 0.4, 0.85
        );
        this.bloomPass.threshold = 0.1;
        this.bloomPass.strength = 0.3;
        this.bloomPass.radius = 0.5;
        this.composer.addPass(this.bloomPass);
        
        // FXAA anti-aliasing
        this.fxaaPass = new ShaderPass(FXAAShader);
        this.fxaaPass.material.uniforms['resolution'].value.x = 1 / window.innerWidth;
        this.fxaaPass.material.uniforms['resolution'].value.y = 1 / window.innerHeight;
        this.composer.addPass(this.fxaaPass);
    }
    
    setupStats() {
        this.stats = new Stats();
        this.stats.showPanel(0); // FPS panel
        document.getElementById(this.containerId).appendChild(this.stats.dom);
        this.stats.dom.style.position = 'absolute';
        this.stats.dom.style.top = '10px';
        this.stats.dom.style.left = '10px';
        this.stats.dom.style.zIndex = '1000';
    }
    
    createEarthSphere() {
        const earthGeometry = new THREE.SphereGeometry(6.371, 128, 64); // Earth radius
        
        // Load Earth textures
        const textureLoader = new THREE.TextureLoader();
        const earthTexture = textureLoader.load('assets/textures/earth_day.jpg');
        const bumpTexture = textureLoader.load('assets/textures/earth_bump.jpg');
        const nightTexture = textureLoader.load('assets/textures/earth_night.jpg');
        const cloudsTexture = textureLoader.load('assets/textures/earth_clouds.jpg');
        
        // Create Earth shader material
        const earthShader = EarthShaderMaterials.createEarthSurfaceShader();
        const earthMaterial = new THREE.ShaderMaterial({
            ...earthShader,
            transparent: true
        });
        
        earthMaterial.uniforms.earthTexture.value = earthTexture;
        earthMaterial.uniforms.bumpTexture.value = bumpTexture;
        earthMaterial.uniforms.nightTexture.value = nightTexture;
        earthMaterial.uniforms.cloudsTexture.value = cloudsTexture;
        
        this.earthSphere = new THREE.Mesh(earthGeometry, earthMaterial);
        this.earthGroup.add(this.earthSphere);
    }
    
    async loadModel(modelName) {
        console.log(`Loading tomography model: ${modelName}`);
        
        try {
            // Show loading indicator
            this.showLoadingIndicator(true);
            
            // Construct model URL
            const modelUrl = `/data/tomography/${modelName.toLowerCase()}.nc`;
            
            // Load model data
            const model = await this.dataManager.loadTomographyModel(
                modelName, modelUrl, 'netcdf'
            );
            
            this.currentModel = model;
            
            // Create 3D texture from model data
            this.createVolumeTexture(model);
            
            // Create volume rendering mesh
            this.createVolumeVisualization(model);
            
            // Update UI with model metadata
            this.updateModelInfo(model.metadata);
            
            console.log(`Model ${modelName} loaded successfully`);
            
        } catch (error) {
            console.error(`Failed to load model ${modelName}:`, error);
            this.showErrorMessage(`Failed to load ${modelName}: ${error.message}`);
        } finally {
            this.showLoadingIndicator(false);
        }
    }
    
    createVolumeTexture(model) {
        const { dimensions, processed } = model;
        
        // Create 3D texture
        this.velocityTexture = this.dataManager.createTexture3D(
            processed.velocity_p, dimensions, this.renderer
        );
        
        console.log('Volume texture created:', dimensions);
    }
    
    createVolumeVisualization(model) {
        // Clear existing volume objects
        this.volumeGroup.clear();
        
        const { dimensions, metadata } = model;
        const bounds = metadata.bounds;
        
        // Create volume rendering geometry
        const volumeGeometry = new THREE.BoxGeometry(
            dimensions.x * 0.1, 
            dimensions.y * 0.1, 
            dimensions.z * 0.1
        );
        
        // Create volume shader material
        const volumeShader = EarthShaderMaterials.createVelocityShader();
        const volumeMaterial = new THREE.ShaderMaterial({
            ...volumeShader,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        volumeMaterial.uniforms.velocityTexture.value = this.velocityTexture;
        volumeMaterial.uniforms.colorScale.value.set(
            bounds.velocity_p.min, 
            bounds.velocity_p.max
        );
        
        this.volumeMesh = new THREE.Mesh(volumeGeometry, volumeMaterial);
        this.volumeGroup.add(this.volumeMesh);
        
        // Create depth slices
        this.createDepthSlices(model);
    }
    
    createDepthSlices(model) {
        const { dimensions } = model;
        const numSlices = 10;
        
        this.sliceGroup.clear();
        
        for (let i = 0; i < numSlices; i++) {
            const depth = i / (numSlices - 1);
            
            const sliceGeometry = new THREE.PlaneGeometry(
                dimensions.x * 0.1, 
                dimensions.y * 0.1
            );
            
            const sliceShader = EarthShaderMaterials.createDepthSliceShader();
            const sliceMaterial = new THREE.ShaderMaterial({
                ...sliceShader,
                transparent: true
            });
            
            sliceMaterial.uniforms.velocityTexture.value = this.velocityTexture;
            sliceMaterial.uniforms.depth.value = depth;
            
            const slice = new THREE.Mesh(sliceGeometry, sliceMaterial);
            slice.position.z = (depth - 0.5) * dimensions.z * 0.1;
            slice.visible = false; // Initially hidden
            
            this.sliceGroup.add(slice);
        }
    }
    
    startRenderLoop() {
        this.isAnimating = true;
        this.animate();
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        this.animationFrame = requestAnimationFrame(() => this.animate());
        
        if (this.stats) this.stats.begin();
        
        // Update controls
        this.controls.update();
        
        // Update uniforms
        this.updateUniforms();
        
        // Render scene
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        
        if (this.stats) this.stats.end();
        
        this.frameCount++;
    }
    
    updateUniforms() {
        const time = this.clock.getElapsedTime();
        
        // Update Earth shader uniforms
        if (this.earthSphere && this.earthSphere.material.uniforms) {
            this.earthSphere.material.uniforms.time.value = time;
        }
        
        // Update volume shader uniforms
        if (this.volumeMesh && this.volumeMesh.material.uniforms) {
            this.volumeMesh.material.uniforms.time.value = time;
        }
    }
    
    // Public API methods
    
    updateVolumeRendering(params) {
        if (!this.volumeMesh) return;
        
        const material = this.volumeMesh.material;
        material.uniforms.opacity.value = params.opacity;
        material.uniforms.showContours.value = params.showContours;
        material.uniforms.contourLevels.value = params.contourLevels;
        material.uniforms.useLogScale.value = params.useLogScale;
    }
    
    updateColorScale(colorScale, useLogScale = false) {
        if (!this.volumeMesh) return;
        
        const material = this.volumeMesh.material;
        material.uniforms.colorScale.value.set(colorScale.min, colorScale.max);
        material.uniforms.useLogScale.value = useLogScale;
        
        // Update slice materials too
        this.sliceGroup.children.forEach(slice => {
            if (slice.material.uniforms) {
                slice.material.uniforms.colorScale.value.set(colorScale.min, colorScale.max);
            }
        });
    }
    
    updateCrossSections(params) {
        this.sliceGroup.children.forEach((slice, index) => {
            const depth = index / (this.sliceGroup.children.length - 1);
            slice.visible = params.showDepthSlice && 
                           Math.abs(depth - params.depthSlice) < 0.1;
        });
    }
    
    updateEarthVisualization(params) {
        if (this.earthSphere) {
            this.earthSphere.visible = params.showEarth;
            
            if (this.earthSphere.material.uniforms) {
                this.earthSphere.material.uniforms.showTopography.value = params.showTopography;
                this.earthSphere.material.opacity = params.earthOpacity;
            }
        }
    }
    
    setAutoRotate(autoRotate, speed = 1.0) {
        if (this.controls) {
            this.controls.autoRotate = autoRotate;
            this.controls.autoRotateSpeed = speed;
        }
    }
    
    setRenderQuality(quality, volumeSteps = 256) {
        const qualitySettings = {
            'low': { pixelRatio: 0.5, shadows: false },
            'medium': { pixelRatio: 1.0, shadows: true },
            'high': { pixelRatio: Math.min(window.devicePixelRatio, 2), shadows: true },
            'ultra': { pixelRatio: window.devicePixelRatio, shadows: true }
        };
        
        const settings = qualitySettings[quality] || qualitySettings['high'];
        
        this.renderer.setPixelRatio(settings.pixelRatio);
        this.renderer.shadowMap.enabled = settings.shadows;
        
        // Update volume rendering steps
        if (this.volumeMesh && this.volumeMesh.material.uniforms) {
            this.volumeMesh.material.uniforms.volumeSteps = { value: volumeSteps };
        }
    }
    
    updatePostProcessing(params) {
        if (this.bloomPass) {
            this.bloomPass.enabled = params.enableBloom;
        }
        
        if (this.saoPass) {
            this.saoPass.enabled = params.enableAO;
        }
    }
    
    async switchTomographyModel(modelName) {
        await this.loadModel(modelName);
    }
    
    resetCamera() {
        this.camera.position.set(15, 10, 15);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }
    
    exportImage(filename = 'tomography_view.png') {
        // Render at high resolution
        const originalSize = this.renderer.getSize(new THREE.Vector2());
        const exportSize = { width: 1920, height: 1080 };
        
        this.renderer.setSize(exportSize.width, exportSize.height);
        this.composer.render();
        
        // Export canvas
        const canvas = this.renderer.domElement;
        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();
        
        // Restore original size
        this.renderer.setSize(originalSize.x, originalSize.y);
    }
    
    async exportAnimation(duration = 10, fps = 30) {
        console.log(`Exporting ${duration}s animation at ${fps} fps`);
        
        // This would require additional libraries like CCapture.js
        // For now, just log the request
        this.showInfoMessage(`Animation export requested: ${duration}s at ${fps}fps`);
    }
    
    exportCurrentModel() {
        if (!this.currentModel) {
            this.showWarningMessage('No model loaded to export');
            return;
        }
        
        const modelData = {
            id: this.currentModel.id,
            metadata: this.currentModel.metadata,
            exportTime: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(modelData, null, 2)], 
                            { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `${this.currentModel.id}_metadata.json`;
        link.click();
        
        URL.revokeObjectURL(url);
    }
    
    // Utility methods
    
    setupEventListeners() {
        window.addEventListener('resize', () => this.handleResize());
        
        // Handle visibility change for performance
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseRendering();
            } else {
                this.resumeRendering();
            }
        });
    }
    
    handleResize() {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
        
        if (this.composer) {
            this.composer.setSize(width, height);
        }
        
        if (this.fxaaPass) {
            this.fxaaPass.material.uniforms['resolution'].value.x = 1 / width;
            this.fxaaPass.material.uniforms['resolution'].value.y = 1 / height;
        }
    }
    
    pauseRendering() {
        this.isAnimating = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
    
    resumeRendering() {
        if (!this.isAnimating) {
            this.isAnimating = true;
            this.animate();
        }
    }
    
    showLoadingIndicator(show) {
        const containerId = this.containerId;
        let indicator = document.getElementById(`${containerId}-loading`);
        
        if (show && !indicator) {
            indicator = document.createElement('div');
            indicator.id = `${containerId}-loading`;
            indicator.innerHTML = `
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                           background: rgba(0,0,0,0.8); color: white; padding: 20px; border-radius: 10px;
                           z-index: 1000; text-align: center;">
                    <div>Loading Tomography Model...</div>
                    <div style="margin-top: 10px; width: 200px; height: 4px; background: #333; border-radius: 2px;">
                        <div style="height: 100%; background: #4CAF50; border-radius: 2px; animation: loading 1.5s ease-in-out infinite;"></div>
                    </div>
                </div>
                <style>
                    @keyframes loading {
                        0% { width: 0%; }
                        50% { width: 100%; }
                        100% { width: 0%; }
                    }
                </style>
            `;
            document.getElementById(containerId).appendChild(indicator);
        } else if (!show && indicator) {
            indicator.remove();
        }
    }
    
    showErrorMessage(message) {
        console.error(message);
        // Could implement a proper notification system here
    }
    
    showInfoMessage(message) {
        console.info(message);
        // Could implement a proper notification system here
    }
    
    showWarningMessage(message) {
        console.warn(message);
        // Could implement a proper notification system here
    }
    
    updateModelInfo(metadata) {
        // Update any UI elements showing model information
        console.log('Model metadata:', metadata);
    }
    
    dispose() {
        this.pauseRendering();
        
        // Clean up Three.js resources
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        // Clean up GUI
        if (this.uiControls && this.uiControls.gui) {
            this.uiControls.gui.destroy();
        }
        
        // Clean up data manager
        if (this.dataManager) {
            // Clean up workers
            this.dataManager.workers.forEach(worker => worker.terminate());
        }
        
        console.log('Global Tomography Viewer disposed');
    }
}

// Export for use in other modules
export default GlobalTomographyViewer;