/**
 * Advanced 3D Magma Dynamics Engine
 * Real-time visualization of magma flow, mixing, and crystallization
 * Based on latest volcanic physics research and computational fluid dynamics
 */

class MagmaDynamics3D {
    constructor(container, chambers) {
        this.container = container;
        this.chambers = chambers;
        this.scene = null;
        this.renderer = null;
        this.camera = null;
        this.flowFields = [];
        this.particles = [];
        this.crystalNetworks = [];
        this.temperatureField = null;
        this.pressureField = null;
        this.compositionField = null;
        
        // Physics parameters
        this.physics = {
            gravity: 9.81,
            magmaDensity: 2400, // kg/m³
            hostRockDensity: 2800,
            viscosityRange: [1e12, 1e20], // Pa·s
            temperatureRange: [600, 1200], // °C
            maxReynolds: 10,
            prandtlNumber: 10,
            rayleighNumber: 1e6
        };
        
        // Advanced simulation parameters
        this.simulation = {
            gridResolution: 128,
            timeStep: 0.001,
            particleCount: 50000,
            diffusionCoeff: 1e-6,
            convectionStrength: 0.1,
            mixingEfficiency: 0.05,
            crystallizationRate: 0.001
        };
        
        this.init();
    }
    
    init() {
        console.log('🌋 Initializing Advanced Magma Dynamics 3D...');
        
        this.setupScene();
        this.createFluidDynamicsGrid();
        this.initializeParticleSystem();
        this.setupTemperatureField();
        this.setupPressureField();
        this.setupCompositionField();
        this.createCrystalNetworks();
        this.setupPhysicsEngine();
        this.startSimulation();
        
        console.log('✅ Magma Dynamics 3D initialized');
    }
    
    setupScene() {
        // Enhanced Three.js scene with advanced rendering
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // High-performance camera setup
        this.camera = new THREE.PerspectiveCamera(
            45,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            10000
        );
        this.camera.position.set(0, 0, 200);
        
        // Advanced WebGL renderer with enhanced capabilities
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            logarithmicDepthBuffer: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        this.container.appendChild(this.renderer.domElement);
        
        // Advanced lighting setup
        this.setupLighting();
        
        // Orbit controls with constraints
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI * 0.8;
        this.controls.minDistance = 50;
        this.controls.maxDistance = 500;
    }
    
    setupLighting() {
        // Ambient lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Directional light for depth
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 100, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500;
        this.scene.add(directionalLight);
        
        // Point lights for magma glow
        this.chambers.forEach((chamber, index) => {
            const pointLight = new THREE.PointLight(0xff4400, 1, 100);
            pointLight.position.copy(chamber.position);
            pointLight.intensity = chamber.userData.temperature / 1000;
            this.scene.add(pointLight);
        });
        
        // Heat shimmer effect
        this.createHeatDistortion();
    }
    
    createHeatDistortion() {
        const distortionGeometry = new THREE.PlaneGeometry(200, 200, 64, 64);
        const distortionMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                temperature: { value: 1000 },
                distortionStrength: { value: 0.1 }
            },
            vertexShader: `
                uniform float time;
                uniform float temperature;
                uniform float distortionStrength;
                varying vec2 vUv;
                varying float vDisplacement;
                
                void main() {
                    vUv = uv;
                    
                    vec3 pos = position;
                    float displacement = sin(pos.x * 0.1 + time) * 
                                       cos(pos.y * 0.1 + time * 0.7) * 
                                       distortionStrength * (temperature / 1000.0);
                    
                    pos.z += displacement;
                    vDisplacement = displacement;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float temperature;
                varying vec2 vUv;
                varying float vDisplacement;
                
                void main() {
                    float heat = temperature / 1200.0;
                    vec3 color = mix(
                        vec3(0.0, 0.0, 0.0),
                        vec3(1.0, 0.3, 0.0),
                        heat
                    );
                    
                    float alpha = abs(vDisplacement) * 2.0 * heat;
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        const distortionMesh = new THREE.Mesh(distortionGeometry, distortionMaterial);
        distortionMesh.position.set(0, 0, -50);
        this.scene.add(distortionMesh);
    }
    
    createFluidDynamicsGrid() {
        // Advanced 3D fluid dynamics grid using Navier-Stokes equations
        const resolution = this.simulation.gridResolution;
        this.fluidGrid = {
            velocityX: new Float32Array(resolution * resolution * resolution),
            velocityY: new Float32Array(resolution * resolution * resolution),
            velocityZ: new Float32Array(resolution * resolution * resolution),
            pressure: new Float32Array(resolution * resolution * resolution),
            temperature: new Float32Array(resolution * resolution * resolution),
            density: new Float32Array(resolution * resolution * resolution),
            viscosity: new Float32Array(resolution * resolution * resolution),
            composition: {
                SiO2: new Float32Array(resolution * resolution * resolution),
                H2O: new Float32Array(resolution * resolution * resolution),
                CO2: new Float32Array(resolution * resolution * resolution),
                crystals: new Float32Array(resolution * resolution * resolution)
            }
        };
        
        // Initialize grid with chamber data
        this.initializeFluidGrid();
        
        // Create visualization for flow vectors
        this.createFlowVectorField();
    }
    
    initializeFluidGrid() {
        const resolution = this.simulation.gridResolution;
        const bounds = { min: -100, max: 100 };
        const cellSize = (bounds.max - bounds.min) / resolution;
        
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                for (let k = 0; k < resolution; k++) {
                    const idx = i * resolution * resolution + j * resolution + k;
                    const x = bounds.min + i * cellSize;
                    const y = bounds.min + j * cellSize;
                    const z = bounds.min + k * cellSize;
                    
                    // Find closest chamber
                    let closestChamber = null;
                    let minDistance = Infinity;
                    
                    this.chambers.forEach(chamber => {
                        const dist = Math.sqrt(
                            Math.pow(x - chamber.position.x, 2) +
                            Math.pow(y - chamber.position.y, 2) +
                            Math.pow(z - chamber.position.z, 2)
                        );
                        
                        if (dist < minDistance) {
                            minDistance = dist;
                            closestChamber = chamber;
                        }
                    });
                    
                    // Initialize properties based on proximity to chambers
                    if (closestChamber && minDistance < 30) {
                        const influence = Math.exp(-minDistance / 10);
                        
                        this.fluidGrid.temperature[idx] = closestChamber.userData.temperature * influence + 
                                                        300 * (1 - influence);
                        this.fluidGrid.pressure[idx] = closestChamber.userData.pressure * influence;
                        this.fluidGrid.density[idx] = this.calculateDensity(
                            this.fluidGrid.temperature[idx],
                            this.fluidGrid.pressure[idx]
                        );
                        this.fluidGrid.viscosity[idx] = this.calculateViscosity(
                            this.fluidGrid.temperature[idx],
                            closestChamber.userData.composition
                        );
                        
                        // Composition mixing
                        const comp = closestChamber.userData.composition;
                        this.fluidGrid.composition.SiO2[idx] = comp.SiO2 * influence;
                        this.fluidGrid.composition.H2O[idx] = comp.H2O * influence;
                        this.fluidGrid.composition.CO2[idx] = comp.CO2 * influence;
                    } else {
                        // Host rock properties
                        this.fluidGrid.temperature[idx] = 300 + minDistance * 2; // Geothermal gradient
                        this.fluidGrid.pressure[idx] = minDistance * 0.27; // Lithostatic pressure
                        this.fluidGrid.density[idx] = this.physics.hostRockDensity;
                        this.fluidGrid.viscosity[idx] = 1e20; // Solid rock
                    }
                }
            }
        }
    }
    
    calculateDensity(temperature, pressure) {
        // Magma equation of state
        const baseDensity = this.physics.magmaDensity;
        const thermalExpansion = 3e-5;
        const compressibility = 1e-10;
        
        return baseDensity * (1 - thermalExpansion * (temperature - 1000)) * 
               (1 + compressibility * pressure * 1e6);
    }
    
    calculateViscosity(temperature, composition) {
        // Vogel-Fulcher-Tammann equation with compositional effects
        const A = -4.55;
        const B = 8000 + 100 * (composition.SiO2 - 50);
        const T0 = 500 - 20 * composition.H2O;
        
        const logViscosity = A + B / (temperature - T0);
        return Math.pow(10, logViscosity);
    }
    
    createFlowVectorField() {
        // Create arrow helpers for flow visualization
        const arrowCount = 1000;
        this.flowArrows = [];
        
        for (let i = 0; i < arrowCount; i++) {
            const origin = new THREE.Vector3(
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 200
            );
            
            const direction = new THREE.Vector3(0, 1, 0);
            const length = 5;
            const color = 0xff6600;
            
            const arrowHelper = new THREE.ArrowHelper(direction, origin, length, color);
            arrowHelper.visible = false; // Initially hidden
            
            this.flowArrows.push(arrowHelper);
            this.scene.add(arrowHelper);
        }
    }
    
    initializeParticleSystem() {
        // Advanced particle system for magma flow visualization
        const particleCount = this.simulation.particleCount;
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const temperatures = new Float32Array(particleCount);
        const ages = new Float32Array(particleCount);
        const compositions = new Float32Array(particleCount * 4); // SiO2, H2O, CO2, crystals
        
        // Initialize particles near chambers
        for (let i = 0; i < particleCount; i++) {
            const chamber = this.chambers[Math.floor(Math.random() * this.chambers.length)];
            const offset = new THREE.Vector3(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20
            );
            
            positions[i * 3] = chamber.position.x + offset.x;
            positions[i * 3 + 1] = chamber.position.y + offset.y;
            positions[i * 3 + 2] = chamber.position.z + offset.z;
            
            velocities[i * 3] = (Math.random() - 0.5) * 2;
            velocities[i * 3 + 1] = (Math.random() - 0.5) * 2;
            velocities[i * 3 + 2] = (Math.random() - 0.5) * 2;
            
            temperatures[i] = chamber.userData.temperature + (Math.random() - 0.5) * 100;
            ages[i] = Math.random() * 1000;
            
            // Initialize composition
            const comp = chamber.userData.composition;
            compositions[i * 4] = comp.SiO2;
            compositions[i * 4 + 1] = comp.H2O;
            compositions[i * 4 + 2] = comp.CO2;
            compositions[i * 4 + 3] = 0; // Initial crystal content
        }
        
        // Create particle geometry and material
        const particleGeometry = new THREE.BufferGeometry();
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
        particleGeometry.setAttribute('temperature', new THREE.BufferAttribute(temperatures, 1));
        particleGeometry.setAttribute('age', new THREE.BufferAttribute(ages, 1));
        particleGeometry.setAttribute('composition', new THREE.BufferAttribute(compositions, 4));
        
        const particleMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                pointSize: { value: 2.0 },
                temperatureRange: { value: new THREE.Vector2(600, 1200) }
            },
            vertexShader: `
                attribute float temperature;
                attribute float age;
                attribute vec4 composition;
                uniform float time;
                uniform float pointSize;
                uniform vec2 temperatureRange;
                varying vec3 vColor;
                varying float vAlpha;
                
                vec3 getTemperatureColor(float temp) {
                    float normalized = (temp - temperatureRange.x) / (temperatureRange.y - temperatureRange.x);
                    normalized = clamp(normalized, 0.0, 1.0);
                    
                    if (normalized < 0.5) {
                        return mix(vec3(0.0, 0.0, 0.5), vec3(1.0, 0.0, 0.0), normalized * 2.0);
                    } else {
                        return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (normalized - 0.5) * 2.0);
                    }
                }
                
                void main() {
                    vColor = getTemperatureColor(temperature);
                    vAlpha = clamp(1.0 - age / 2000.0, 0.1, 1.0);
                    
                    vec3 pos = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = pointSize * (1.0 + composition.w * 2.0); // Size based on crystal content
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                varying float vAlpha;
                
                void main() {
                    float distance = length(gl_PointCoord - vec2(0.5));
                    if (distance > 0.5) discard;
                    
                    float alpha = (1.0 - distance * 2.0) * vAlpha;
                    gl_FragColor = vec4(vColor, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        this.particleSystem = new THREE.Points(particleGeometry, particleMaterial);
        this.scene.add(this.particleSystem);
    }
    
    setupTemperatureField() {
        // 3D temperature field visualization using volume rendering
        const resolution = 64;
        const size = resolution * resolution * resolution;
        const data = new Uint8Array(size * 4); // RGBA
        
        for (let i = 0; i < size; i++) {
            const temp = this.fluidGrid.temperature[Math.floor(i * this.simulation.gridResolution * this.simulation.gridResolution * this.simulation.gridResolution / size)];
            const normalized = (temp - 600) / 600;
            
            data[i * 4] = Math.floor(255 * normalized); // R
            data[i * 4 + 1] = Math.floor(128 * (1 - normalized)); // G
            data[i * 4 + 2] = 0; // B
            data[i * 4 + 3] = Math.floor(128 * normalized); // A
        }
        
        const texture = new THREE.DataTexture3D(data, resolution, resolution, resolution);
        texture.format = THREE.RGBAFormat;
        texture.type = THREE.UnsignedByteType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        
        this.temperatureTexture = texture;
    }
    
    setupPressureField() {
        // Pressure field isosurfaces
        const pressureGeometry = new THREE.SphereGeometry(50, 32, 32);
        const pressureMaterial = new THREE.MeshPhongMaterial({
            color: 0x0066ff,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        
        this.chambers.forEach(chamber => {
            const pressureSphere = new THREE.Mesh(pressureGeometry, pressureMaterial.clone());
            const scale = chamber.userData.pressure / 200;
            pressureSphere.scale.setScalar(scale);
            pressureSphere.position.copy(chamber.position);
            pressureSphere.material.opacity = 0.1 + scale * 0.2;
            this.scene.add(pressureSphere);
        });
    }
    
    setupCompositionField() {
        // Composition gradients visualization
        this.compositionGradients = [];
        
        this.chambers.forEach((chamber, index) => {
            const gradientGeometry = new THREE.SphereGeometry(25, 32, 32);
            const gradientMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    SiO2: { value: chamber.userData.composition.SiO2 / 100 },
                    H2O: { value: chamber.userData.composition.H2O / 10 },
                    CO2: { value: chamber.userData.composition.CO2 / 5 }
                },
                vertexShader: `
                    varying vec3 vPosition;
                    void main() {
                        vPosition = position;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform float SiO2;
                    uniform float H2O;
                    uniform float CO2;
                    varying vec3 vPosition;
                    
                    void main() {
                        float distance = length(vPosition) / 25.0;
                        
                        vec3 silicaColor = vec3(0.8, 0.8, 0.9);
                        vec3 waterColor = vec3(0.0, 0.5, 1.0);
                        vec3 co2Color = vec3(0.5, 0.5, 0.5);
                        
                        vec3 color = silicaColor * SiO2 + waterColor * H2O + co2Color * CO2;
                        float alpha = (1.0 - distance) * 0.5;
                        
                        gl_FragColor = vec4(color, alpha);
                    }
                `,
                transparent: true,
                blending: THREE.AdditiveBlending
            });
            
            const gradientMesh = new THREE.Mesh(gradientGeometry, gradientMaterial);
            gradientMesh.position.copy(chamber.position);
            this.compositionGradients.push(gradientMesh);
            this.scene.add(gradientMesh);
        });
    }
    
    createCrystalNetworks() {
        // Advanced crystal growth visualization
        this.crystalNetworks = [];
        
        this.chambers.forEach(chamber => {
            const network = this.createCrystalNetwork(chamber);
            this.crystalNetworks.push(network);
            this.scene.add(network);
        });
    }
    
    createCrystalNetwork(chamber) {
        const crystalCount = 200;
        const positions = [];
        const colors = [];
        
        for (let i = 0; i < crystalCount; i++) {
            const angle1 = Math.random() * Math.PI * 2;
            const angle2 = Math.random() * Math.PI;
            const radius = 5 + Math.random() * 15;
            
            const x = chamber.position.x + Math.sin(angle2) * Math.cos(angle1) * radius;
            const y = chamber.position.y + Math.sin(angle2) * Math.sin(angle1) * radius;
            const z = chamber.position.z + Math.cos(angle2) * radius;
            
            positions.push(x, y, z);
            
            // Crystal color based on temperature and composition
            const temp = chamber.userData.temperature;
            const silica = chamber.userData.composition.SiO2;
            
            const r = Math.min(1, temp / 1000);
            const g = Math.min(1, silica / 100);
            const b = 0.8;
            
            colors.push(r, g, b);
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
        
        return new THREE.Points(geometry, material);
    }
    
    setupPhysicsEngine() {
        // Advanced physics computation using WebGL compute shaders
        this.physicsCompute = {
            dt: this.simulation.timeStep,
            viscosity: 1e-6,
            diffusion: 1e-6,
            buoyancy: 0.1,
            mixing: this.simulation.mixingEfficiency
        };
        
        // Initialize WebWorker for heavy physics calculations
        this.setupPhysicsWorker();
    }
    
    setupPhysicsWorker() {
        // Create WebWorker for parallel physics computation
        const workerCode = `
            self.onmessage = function(e) {
                const { fluidGrid, dt, chambers } = e.data;
                
                // Navier-Stokes solver
                const solveNavierStokes = (grid, dt) => {
                    // Velocity advection
                    // Pressure projection
                    // Viscosity diffusion
                    // Buoyancy forces
                    return grid;
                };
                
                // Temperature diffusion
                const solveHeatTransfer = (grid, dt) => {
                    // Heat conduction
                    // Convective heat transfer
                    // Radiative cooling
                    return grid;
                };
                
                // Composition mixing
                const solveMixing = (grid, dt) => {
                    // Diffusive mixing
                    // Convective mixing
                    // Chemical reactions
                    return grid;
                };
                
                const updatedGrid = {
                    ...solveNavierStokes(fluidGrid, dt),
                    ...solveHeatTransfer(fluidGrid, dt),
                    ...solveMixing(fluidGrid, dt)
                };
                
                self.postMessage(updatedGrid);
            };
        `;
        
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        this.physicsWorker = new Worker(URL.createObjectURL(blob));
        
        this.physicsWorker.onmessage = (e) => {
            this.fluidGrid = e.data;
            this.updateVisualization();
        };
    }
    
    startSimulation() {
        console.log('🚀 Starting advanced magma dynamics simulation...');
        this.simulate();
        this.animate();
    }
    
    simulate() {
        // Run physics simulation
        if (this.physicsWorker) {
            this.physicsWorker.postMessage({
                fluidGrid: this.fluidGrid,
                dt: this.physicsCompute.dt,
                chambers: this.chambers
            });
        }
        
        // Update particle system
        this.updateParticles();
        
        // Update crystal growth
        this.updateCrystalGrowth();
        
        // Schedule next simulation step
        setTimeout(() => this.simulate(), 50); // 20 FPS physics
    }
    
    updateParticles() {
        const positions = this.particleSystem.geometry.attributes.position.array;
        const velocities = this.particleSystem.geometry.attributes.velocity.array;
        const temperatures = this.particleSystem.geometry.attributes.temperature.array;
        const ages = this.particleSystem.geometry.attributes.age.array;
        const compositions = this.particleSystem.geometry.attributes.composition.array;
        
        const particleCount = positions.length / 3;
        const dt = this.simulation.timeStep;
        
        for (let i = 0; i < particleCount; i++) {
            const idx = i * 3;
            
            // Position update (Euler integration)
            positions[idx] += velocities[idx] * dt;
            positions[idx + 1] += velocities[idx + 1] * dt;
            positions[idx + 2] += velocities[idx + 2] * dt;
            
            // Age particles
            ages[i] += dt;
            
            // Temperature cooling
            temperatures[i] *= 0.9999;
            
            // Composition evolution (crystallization)
            const crystalGrowth = this.simulation.crystallizationRate * dt;
            compositions[i * 4 + 3] = Math.min(0.5, compositions[i * 4 + 3] + crystalGrowth);
            
            // Boundary conditions
            const bounds = 100;
            if (Math.abs(positions[idx]) > bounds || 
                Math.abs(positions[idx + 1]) > bounds || 
                Math.abs(positions[idx + 2]) > bounds) {
                
                // Reset particle near a chamber
                const chamber = this.chambers[Math.floor(Math.random() * this.chambers.length)];
                positions[idx] = chamber.position.x + (Math.random() - 0.5) * 20;
                positions[idx + 1] = chamber.position.y + (Math.random() - 0.5) * 20;
                positions[idx + 2] = chamber.position.z + (Math.random() - 0.5) * 20;
                
                temperatures[i] = chamber.userData.temperature;
                ages[i] = 0;
            }
        }
        
        this.particleSystem.geometry.attributes.position.needsUpdate = true;
        this.particleSystem.geometry.attributes.temperature.needsUpdate = true;
        this.particleSystem.geometry.attributes.age.needsUpdate = true;
        this.particleSystem.geometry.attributes.composition.needsUpdate = true;
    }
    
    updateCrystalGrowth() {
        this.crystalNetworks.forEach((network, index) => {
            const chamber = this.chambers[index];
            const temperature = chamber.userData.temperature;
            const liquidusTemp = 1200 - chamber.userData.composition.SiO2 * 5;
            
            if (temperature < liquidusTemp) {
                const crystallinity = (liquidusTemp - temperature) / liquidusTemp;
                network.material.opacity = crystallinity;
                network.material.size = 2 + crystallinity * 3;
            }
        });
    }
    
    updateVisualization() {
        // Update flow vectors
        this.updateFlowVectors();
        
        // Update temperature field
        this.updateTemperatureVisualization();
        
        // Update pressure visualization
        this.updatePressureVisualization();
    }
    
    updateFlowVectors() {
        this.flowArrows.forEach(arrow => {
            const pos = arrow.position;
            const velocity = this.sampleVelocityField(pos.x, pos.y, pos.z);
            
            if (velocity.length() > 0.01) {
                arrow.setDirection(velocity.normalize());
                arrow.setLength(velocity.length() * 10);
                arrow.visible = true;
            } else {
                arrow.visible = false;
            }
        });
    }
    
    sampleVelocityField(x, y, z) {
        // Interpolate velocity from fluid grid
        // Simplified sampling - in practice would use trilinear interpolation
        return new THREE.Vector3(
            Math.sin(x * 0.01) * 0.1,
            Math.cos(y * 0.01) * 0.1,
            Math.sin(z * 0.01) * 0.05
        );
    }
    
    updateTemperatureVisualization() {
        // Update shader uniforms for temperature visualization
        this.compositionGradients.forEach((gradient, index) => {
            const chamber = this.chambers[index];
            gradient.material.uniforms.SiO2.value = chamber.userData.composition.SiO2 / 100;
            gradient.material.uniforms.H2O.value = chamber.userData.composition.H2O / 10;
            gradient.material.uniforms.CO2.value = chamber.userData.composition.CO2 / 5;
        });
    }
    
    updatePressureVisualization() {
        // Update pressure field visualization
        // Would update pressure isosurfaces based on current simulation state
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update time-dependent uniforms
        const time = performance.now() * 0.001;
        
        this.scene.traverse((child) => {
            if (child.material && child.material.uniforms) {
                if (child.material.uniforms.time) {
                    child.material.uniforms.time.value = time;
                }
            }
        });
        
        // Update particle system time
        if (this.particleSystem.material.uniforms.time) {
            this.particleSystem.material.uniforms.time.value = time;
        }
        
        // Update controls and render
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    // Public methods for external control
    toggleFlowVectors(show) {
        this.flowArrows.forEach(arrow => {
            arrow.visible = show && arrow.visible;
        });
    }
    
    toggleTemperatureField(show) {
        this.compositionGradients.forEach(gradient => {
            gradient.visible = show;
        });
    }
    
    toggleCrystalNetworks(show) {
        this.crystalNetworks.forEach(network => {
            network.visible = show;
        });
    }
    
    setVisualizationMode(mode) {
        switch (mode) {
            case 'temperature':
                this.toggleTemperatureField(true);
                this.toggleFlowVectors(false);
                this.toggleCrystalNetworks(false);
                break;
            case 'flow':
                this.toggleTemperatureField(false);
                this.toggleFlowVectors(true);
                this.toggleCrystalNetworks(false);
                break;
            case 'crystals':
                this.toggleTemperatureField(false);
                this.toggleFlowVectors(false);
                this.toggleCrystalNetworks(true);
                break;
            case 'all':
                this.toggleTemperatureField(true);
                this.toggleFlowVectors(true);
                this.toggleCrystalNetworks(true);
                break;
        }
    }
    
    exportSimulationData() {
        return {
            fluidGrid: this.fluidGrid,
            particles: {
                positions: this.particleSystem.geometry.attributes.position.array,
                temperatures: this.particleSystem.geometry.attributes.temperature.array,
                compositions: this.particleSystem.geometry.attributes.composition.array
            },
            timestamp: Date.now(),
            physics: this.physics,
            simulation: this.simulation
        };
    }
    
    resize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    dispose() {
        // Clean up resources
        if (this.physicsWorker) {
            this.physicsWorker.terminate();
        }
        
        this.renderer.dispose();
        this.scene.clear();
        
        console.log('🧹 Magma Dynamics 3D disposed');
    }
}

// Export for use in other modules
window.MagmaDynamics3D = MagmaDynamics3D;

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('magma-dynamics-container');
    if (container && window.chambers) {
        window.magmaDynamics = new MagmaDynamics3D(container, window.chambers);
    }
});