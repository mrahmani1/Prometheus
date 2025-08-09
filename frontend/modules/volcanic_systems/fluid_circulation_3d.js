/**
 * Advanced 3D Fluid Circulation Visualization System
 * Interactive 3D visualization of hydrothermal fluid flow and circulation patterns
 * Based on latest fluid dynamics research and advanced rendering techniques
 */

class FluidCirculation3D {
    constructor(container, hydrodata) {
        this.container = container;
        this.hydrodata = hydrodata;
        this.scene = null;
        this.renderer = null;
        this.camera = null;
        this.controls = null;
        
        // Visualization components
        this.fluidParticles = null;
        this.flowVectors = [];
        this.temperatureField = null;
        this.pressureField = null;
        this.permeabilityField = null;
        this.chemicalConcentrations = [];
        this.alteration_zones = [];
        
        // Simulation parameters
        this.simulation = {
            timeStep: 0.1,
            particleCount: 100000,
            flowSpeed: 1.0,
            temperatureRange: [273, 873],
            pressureRange: [1e5, 100e6],
            visualizationMode: 'particles',
            showVectors: true,
            showTemperature: true,
            showPressure: false,
            showChemistry: true,
            showAlteration: true
        };
        
        // Fluid properties
        this.fluidProperties = {
            density: 800.0,          // kg/m³
            viscosity: 1e-4,         // Pa·s
            thermalConductivity: 0.6, // W/m/K
            heatCapacity: 4200,      // J/kg/K
            compressibility: 4e-10    // Pa^-1
        };
        
        // Animation parameters
        this.animationState = {
            time: 0,
            playing: true,
            speed: 1.0,
            frame: 0
        };
        
        this.init();
    }
    
    init() {
        console.log('💧 Initializing 3D Fluid Circulation System...');
        
        this.setupScene();
        this.setupLighting();
        this.createFluidParticleSystem();
        this.createFlowVectorField();
        this.createTemperatureVisualization();
        this.createPressureVisualization();
        this.createPermeabilityVisualization();
        this.createChemicalVisualization();
        this.createAlterationVisualization();
        this.setupControls();
        this.startAnimation();
        
        console.log('✅ 3D Fluid Circulation System initialized');
    }
    
    setupScene() {
        // Create Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x001122);
        this.scene.fog = new THREE.Fog(0x001122, 100, 2000);
        
        // Setup camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            5000
        );
        this.camera.position.set(200, 200, 200);
        
        // Setup renderer with advanced features
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
        this.renderer.toneMappingExposure = 1.0;
        
        this.container.appendChild(this.renderer.domElement);
        
        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 1000;
        this.controls.minDistance = 10;
    }
    
    setupLighting() {
        // Ambient lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light for depth perception
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(100, 100, 100);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Point lights for hydrothermal sources
        this.createHydrothermalLights();
        
        // Volumetric lighting effects
        this.createVolumetricLighting();
    }
    
    createHydrothermalLights() {
        // Create point lights at hydrothermal sources
        const sources = [
            {position: new THREE.Vector3(0, -50, 0), intensity: 2.0, color: 0xff4400},
            {position: new THREE.Vector3(30, -30, 20), intensity: 1.5, color: 0xff6600},
            {position: new THREE.Vector3(-20, -40, -10), intensity: 1.8, color: 0xff5500}
        ];
        
        sources.forEach(source => {
            const light = new THREE.PointLight(source.color, source.intensity, 150);
            light.position.copy(source.position);
            
            // Add flickering animation
            light.userData = {
                originalIntensity: source.intensity,
                flickerSpeed: 0.5 + Math.random() * 0.5
            };
            
            this.scene.add(light);
        });
    }
    
    createVolumetricLighting() {
        // Create volumetric lighting using shader material
        const volumetricGeometry = new THREE.BoxGeometry(200, 200, 200);
        const volumetricMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                lightPosition: { value: new THREE.Vector3(0, -50, 0) },
                lightColor: { value: new THREE.Color(0xff4400) },
                density: { value: 0.02 },
                samples: { value: 32 }
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                varying vec3 vLocalPosition;
                
                void main() {
                    vLocalPosition = position;
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 lightPosition;
                uniform vec3 lightColor;
                uniform float density;
                uniform int samples;
                
                varying vec3 vWorldPosition;
                varying vec3 vLocalPosition;
                
                float noise3D(vec3 p) {
                    return sin(p.x * 0.5 + time * 0.1) * 
                           cos(p.y * 0.3 + time * 0.15) * 
                           sin(p.z * 0.4 + time * 0.12);
                }
                
                void main() {
                    vec3 rayDir = normalize(vWorldPosition - cameraPosition);
                    vec3 rayStart = vWorldPosition;
                    
                    vec3 color = vec3(0.0);
                    float alpha = 0.0;
                    
                    float stepSize = 2.0 / float(samples);
                    
                    for(int i = 0; i < 32; i++) {
                        if(i >= samples) break;
                        
                        vec3 currentPos = rayStart + rayDir * float(i) * stepSize;
                        
                        // Distance from light
                        float distanceToLight = length(currentPos - lightPosition);
                        float lightIntensity = 1.0 / (1.0 + distanceToLight * 0.01);
                        
                        // Volumetric noise
                        float noiseValue = noise3D(currentPos * 0.1) * 0.5 + 0.5;
                        
                        // Accumulate color and alpha
                        vec3 sampleColor = lightColor * lightIntensity * noiseValue * density;
                        color += sampleColor * (1.0 - alpha);
                        alpha += density * noiseValue * lightIntensity * (1.0 - alpha);
                        
                        if(alpha > 0.99) break;
                    }
                    
                    gl_FragColor = vec4(color, alpha * 0.3);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            side: THREE.BackSide
        });
        
        const volumetricMesh = new THREE.Mesh(volumetricGeometry, volumetricMaterial);
        volumetricMesh.position.set(0, -25, 0);
        this.scene.add(volumetricMesh);
        
        this.volumetricLight = volumetricMesh;
    }
    
    createFluidParticleSystem() {
        console.log('Creating advanced fluid particle system...');
        
        const particleCount = this.simulation.particleCount;
        
        // Create geometry with multiple attributes
        const geometry = new THREE.BufferGeometry();
        
        // Positions
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const temperatures = new Float32Array(particleCount);
        const pressures = new Float32Array(particleCount);
        const ages = new Float32Array(particleCount);
        const sizes = new Float32Array(particleCount);
        const phases = new Float32Array(particleCount); // 0=liquid, 1=vapor, 2=supercritical
        
        // Initialize particles
        for (let i = 0; i < particleCount; i++) {
            // Distribute particles in hydrothermal sources
            const source = Math.floor(Math.random() * 3);
            let sourcePos;
            
            switch (source) {
                case 0:
                    sourcePos = new THREE.Vector3(0, -50, 0);
                    break;
                case 1:
                    sourcePos = new THREE.Vector3(30, -30, 20);
                    break;
                case 2:
                    sourcePos = new THREE.Vector3(-20, -40, -10);
                    break;
            }
            
            // Add random offset
            const offset = new THREE.Vector3(
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 5,
                (Math.random() - 0.5) * 10
            );
            sourcePos.add(offset);
            
            positions[i * 3] = sourcePos.x;
            positions[i * 3 + 1] = sourcePos.y;
            positions[i * 3 + 2] = sourcePos.z;
            
            // Initial velocities (upward with some turbulence)
            velocities[i * 3] = (Math.random() - 0.5) * 2;
            velocities[i * 3 + 1] = 1 + Math.random() * 3;
            velocities[i * 3 + 2] = (Math.random() - 0.5) * 2;
            
            // Fluid properties
            temperatures[i] = 573 + Math.random() * 300; // 300-600°C
            pressures[i] = 1e6 + Math.random() * 50e6;   // 10-500 bar
            ages[i] = Math.random() * 1000;
            sizes[i] = 0.5 + Math.random() * 2.0;
            
            // Determine phase based on T-P conditions
            const T = temperatures[i];
            const P = pressures[i];
            
            if (T > 647 && P > 22e6) {
                phases[i] = 2; // Supercritical
            } else if (T > 373 && P < this.vaporPressure(T)) {
                phases[i] = 1; // Vapor
            } else {
                phases[i] = 0; // Liquid
            }
        }
        
        // Set geometry attributes
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
        geometry.setAttribute('temperature', new THREE.BufferAttribute(temperatures, 1));
        geometry.setAttribute('pressure', new THREE.BufferAttribute(pressures, 1));
        geometry.setAttribute('age', new THREE.BufferAttribute(ages, 1));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));
        
        // Advanced shader material for particles
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                temperatureMin: { value: 273 },
                temperatureMax: { value: 873 },
                pressureMin: { value: 1e5 },
                pressureMax: { value: 100e6 },
                pointSize: { value: 2.0 },
                opacity: { value: 0.8 }
            },
            vertexShader: `
                attribute float temperature;
                attribute float pressure;
                attribute float age;
                attribute float size;
                attribute float phase;
                
                uniform float time;
                uniform float temperatureMin;
                uniform float temperatureMax;
                uniform float pressureMin;
                uniform float pressureMax;
                uniform float pointSize;
                
                varying float vTemperature;
                varying float vPressure;
                varying float vAge;
                varying float vPhase;
                varying vec3 vColor;
                
                vec3 getTemperatureColor(float temp) {
                    float normalized = (temp - temperatureMin) / (temperatureMax - temperatureMin);
                    normalized = clamp(normalized, 0.0, 1.0);
                    
                    // Blue (cold) -> Red (hot) -> Yellow (very hot)
                    if (normalized < 0.5) {
                        return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), normalized * 2.0);
                    } else {
                        return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (normalized - 0.5) * 2.0);
                    }
                }
                
                vec3 getPhaseColor(float phase) {
                    if (phase < 0.5) return vec3(0.2, 0.6, 1.0);      // Liquid - blue
                    else if (phase < 1.5) return vec3(0.8, 0.8, 0.8); // Vapor - white
                    else return vec3(1.0, 0.4, 0.0);                  // Supercritical - orange
                }
                
                void main() {
                    vTemperature = temperature;
                    vPressure = pressure;
                    vAge = age;
                    vPhase = phase;
                    
                    // Blend temperature and phase colors
                    vec3 tempColor = getTemperatureColor(temperature);
                    vec3 phaseColor = getPhaseColor(phase);
                    vColor = mix(tempColor, phaseColor, 0.3);
                    
                    // Particle size based on phase and pressure
                    float pressureNorm = (pressure - pressureMin) / (pressureMax - pressureMin);
                    float finalSize = pointSize * size * (1.0 + phase * 0.5) * (0.5 + pressureNorm * 0.5);
                    
                    // Age fading
                    float ageFactor = clamp(1.0 - age / 2000.0, 0.1, 1.0);
                    finalSize *= ageFactor;
                    
                    gl_PointSize = finalSize;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float opacity;
                varying float vTemperature;
                varying float vPressure;
                varying float vAge;
                varying float vPhase;
                varying vec3 vColor;
                
                void main() {
                    // Circular particle shape
                    vec2 center = gl_PointCoord - 0.5;
                    float distance = length(center);
                    
                    if (distance > 0.5) discard;
                    
                    // Soft edges
                    float alpha = (1.0 - distance * 2.0) * opacity;
                    
                    // Age-based transparency
                    float ageFactor = clamp(1.0 - vAge / 2000.0, 0.1, 1.0);
                    alpha *= ageFactor;
                    
                    // Phase-based effects
                    if (vPhase > 1.5) {
                        // Supercritical - add glow
                        alpha *= 1.5;
                    } else if (vPhase > 0.5) {
                        // Vapor - more transparent
                        alpha *= 0.7;
                    }
                    
                    gl_FragColor = vec4(vColor, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            vertexColors: true
        });
        
        // Create particle system
        this.fluidParticles = new THREE.Points(geometry, material);
        this.scene.add(this.fluidParticles);
        
        console.log(`Created ${particleCount:,} fluid particles`);
    }
    
    vaporPressure(temperature) {
        // Antoine equation for water vapor pressure
        const A = 8.07131;
        const B = 1730.63;
        const C = -39.724;
        
        const T_celsius = temperature - 273.15;
        const log_p = A - B / (T_celsius + C);
        const p_mmHg = Math.pow(10, log_p);
        const p_Pa = p_mmHg * 133.322;
        
        return p_Pa;
    }
    
    createFlowVectorField() {
        console.log('Creating flow vector field...');
        
        const gridSize = 20;
        const spacing = 10;
        
        for (let x = -100; x <= 100; x += spacing) {
            for (let y = -100; y <= 50; y += spacing) {
                for (let z = -100; z <= 100; z += spacing) {
                    // Calculate flow direction based on position
                    const flowDirection = this.calculateFlowDirection(x, y, z);
                    
                    if (flowDirection.length() > 0.1) {
                        const arrow = this.createFlowArrow(
                            new THREE.Vector3(x, y, z),
                            flowDirection,
                            flowDirection.length()
                        );
                        
                        arrow.visible = this.simulation.showVectors;
                        this.flowVectors.push(arrow);
                        this.scene.add(arrow);
                    }
                }
            }
        }
        
        console.log(`Created ${this.flowVectors.length} flow vectors`);
    }
    
    calculateFlowDirection(x, y, z) {
        // Simplified flow field calculation
        // In reality, this would be computed from the fluid dynamics simulation
        
        const pos = new THREE.Vector3(x, y, z);
        const flow = new THREE.Vector3(0, 0, 0);
        
        // Hydrothermal sources create upward flow
        const sources = [
            {pos: new THREE.Vector3(0, -50, 0), strength: 5.0},
            {pos: new THREE.Vector3(30, -30, 20), strength: 3.0},
            {pos: new THREE.Vector3(-20, -40, -10), strength: 4.0}
        ];
        
        sources.forEach(source => {
            const distance = pos.distanceTo(source.pos);
            if (distance < 50) {
                const direction = pos.clone().sub(source.pos).normalize();
                const strength = source.strength / (1 + distance * 0.1);
                
                // Upward bias
                direction.y = Math.abs(direction.y) + 0.5;
                direction.normalize();
                
                flow.add(direction.multiplyScalar(strength));
            }
        });
        
        // Add convection patterns
        const convectionFlow = new THREE.Vector3(
            Math.sin(x * 0.02) * Math.cos(z * 0.02),
            Math.cos(x * 0.01 + z * 0.01) * 0.5,
            Math.sin(z * 0.02) * Math.cos(x * 0.02)
        );
        
        flow.add(convectionFlow.multiplyScalar(0.5));
        
        // Gravity effect (downward for cooled fluids)
        if (y > 0) {
            flow.y -= 0.5;
        }
        
        return flow;
    }
    
    createFlowArrow(position, direction, magnitude) {
        const arrowLength = Math.min(magnitude * 2, 8);
        const arrowColor = new THREE.Color().setHSL(
            0.6 - magnitude * 0.1,  // Blue to red based on magnitude
            0.8,
            0.5
        );
        
        const arrowGeometry = new THREE.ConeGeometry(0.5, arrowLength, 8);
        const arrowMaterial = new THREE.MeshLambertMaterial({ color: arrowColor });
        
        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
        arrow.position.copy(position);
        
        // Orient arrow in flow direction
        const up = new THREE.Vector3(0, 1, 0);
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(up, direction.normalize());
        arrow.setRotationFromQuaternion(quaternion);
        
        return arrow;
    }
    
    createTemperatureVisualization() {
        console.log('Creating temperature field visualization...');
        
        // Create temperature isosurfaces
        const temperatureLevels = [373, 473, 573, 673, 773]; // Kelvin
        const colors = [0x0066ff, 0x00ff66, 0xffff00, 0xff6600, 0xff0000];
        
        temperatureLevels.forEach((temp, index) => {
            const geometry = new THREE.SphereGeometry(20 + index * 10, 32, 32);
            const material = new THREE.MeshPhongMaterial({
                color: colors[index],
                transparent: true,
                opacity: 0.2,
                side: THREE.DoubleSide
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(0, -30, 0);
            sphere.userData.temperature = temp;
            sphere.visible = this.simulation.showTemperature;
            
            this.scene.add(sphere);
        });
        
        // Create temperature volume rendering
        this.createTemperatureVolume();
    }
    
    createTemperatureVolume() {
        const size = 64;
        const data = new Uint8Array(size * size * size * 4); // RGBA
        
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                for (let k = 0; k < size; k++) {
                    const idx = (i * size * size + j * size + k) * 4;
                    
                    // Calculate position
                    const x = (i / size - 0.5) * 200;
                    const y = (j / size - 0.5) * 200;
                    const z = (k / size - 0.5) * 200;
                    
                    // Calculate temperature based on distance from heat sources
                    let temperature = 293; // Base temperature
                    
                    const sources = [
                        {pos: [0, -50, 0], temp: 773},
                        {pos: [30, -30, 20], temp: 673},
                        {pos: [-20, -40, -10], temp: 723}
                    ];
                    
                    sources.forEach(source => {
                        const dx = x - source.pos[0];
                        const dy = y - source.pos[1];
                        const dz = z - source.pos[2];
                        const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
                        
                        const influence = Math.exp(-distance / 30);
                        temperature += (source.temp - 293) * influence;
                    });
                    
                    // Convert to color
                    const normalized = (temperature - 273) / 500;
                    const r = Math.floor(255 * Math.min(1, normalized));
                    const g = Math.floor(255 * Math.max(0, 1 - Math.abs(normalized - 0.5) * 2));
                    const b = Math.floor(255 * Math.max(0, 1 - normalized));
                    const a = Math.floor(128 * Math.min(1, normalized));
                    
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                    data[idx + 3] = a;
                }
            }
        }
        
        const texture = new THREE.DataTexture3D(data, size, size, size);
        texture.format = THREE.RGBAFormat;
        texture.type = THREE.UnsignedByteType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
        texture.needsUpdate = true;
        
        this.temperatureTexture = texture;
    }
    
    createPressureVisualization() {
        console.log('Creating pressure field visualization...');
        
        // Create pressure contour lines
        const pressureLevels = [1e6, 5e6, 10e6, 25e6, 50e6]; // Pa
        
        pressureLevels.forEach((pressure, index) => {
            const geometry = new THREE.RingGeometry(15 + index * 8, 16 + index * 8, 32);
            const material = new THREE.MeshBasicMaterial({
                color: 0x0088ff,
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide
            });
            
            const ring = new THREE.Mesh(geometry, material);
            ring.position.set(0, -25 - index * 5, 0);
            ring.rotation.x = Math.PI / 2;
            ring.userData.pressure = pressure;
            ring.visible = this.simulation.showPressure;
            
            this.scene.add(ring);
        });
    }
    
    createPermeabilityVisualization() {
        // Create permeability field using point cloud
        const permeabilityPoints = [];
        const permeabilityColors = [];
        
        for (let x = -80; x <= 80; x += 20) {
            for (let y = -80; y <= 20; y += 20) {
                for (let z = -80; z <= 80; z += 20) {
                    // Calculate permeability based on rock type and fracturing
                    const permeability = this.calculatePermeability(x, y, z);
                    
                    permeabilityPoints.push(x, y, z);
                    
                    // Color based on permeability (log scale)
                    const logPerm = Math.log10(permeability);
                    const normalized = (logPerm + 18) / 6; // -18 to -12 log(m²)
                    
                    const color = new THREE.Color();
                    color.setHSL(0.7 - normalized * 0.7, 1.0, 0.5);
                    permeabilityColors.push(color.r, color.g, color.b);
                }
            }
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(permeabilityPoints, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(permeabilityColors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            transparent: true,
            opacity: 0.6
        });
        
        this.permeabilityField = new THREE.Points(geometry, material);
        this.permeabilityField.visible = false;
        this.scene.add(this.permeabilityField);
    }
    
    calculatePermeability(x, y, z) {
        // Base permeability
        let permeability = 1e-15; // m²
        
        // Increase near heat sources (thermal fracturing)
        const sources = [
            {pos: [0, -50, 0], factor: 100},
            {pos: [30, -30, 20], factor: 50},
            {pos: [-20, -40, -10], factor: 75}
        ];
        
        sources.forEach(source => {
            const dx = x - source.pos[0];
            const dy = y - source.pos[1];
            const dz = z - source.pos[2];
            const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            const influence = Math.exp(-distance / 25);
            permeability *= (1 + source.factor * influence);
        });
        
        // Depth effect (decreasing permeability with depth)
        const depth = Math.max(0, -y);
        permeability *= Math.exp(-depth / 100);
        
        return Math.max(1e-18, Math.min(1e-12, permeability));
    }
    
    createChemicalVisualization() {
        console.log('Creating chemical concentration visualization...');
        
        const chemicals = [
            {name: 'SiO2', color: 0xaaaaaa, concentration: 0.6},
            {name: 'H2S', color: 0xffff00, concentration: 0.02},
            {name: 'CO2', color: 0x888888, concentration: 0.05},
            {name: 'SO2', color: 0xff4400, concentration: 0.01}
        ];
        
        chemicals.forEach((chemical, index) => {
            const particleCount = 5000;
            const positions = new Float32Array(particleCount * 3);
            const concentrations = new Float32Array(particleCount);
            
            for (let i = 0; i < particleCount; i++) {
                // Distribute particles based on chemical transport patterns
                const angle = Math.random() * Math.PI * 2;
                const radius = 20 + Math.random() * 60;
                const height = -50 + Math.random() * 80;
                
                positions[i * 3] = Math.cos(angle) * radius;
                positions[i * 3 + 1] = height;
                positions[i * 3 + 2] = Math.sin(angle) * radius;
                
                concentrations[i] = chemical.concentration * (0.5 + Math.random() * 0.5);
            }
            
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('concentration', new THREE.BufferAttribute(concentrations, 1));
            
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    color: { value: new THREE.Color(chemical.color) },
                    time: { value: 0 },
                    pointSize: { value: 1.5 }
                },
                vertexShader: `
                    attribute float concentration;
                    uniform float time;
                    uniform float pointSize;
                    varying float vConcentration;
                    varying vec3 vColor;
                    
                    void main() {
                        vConcentration = concentration;
                        vColor = color.rgb;
                        
                        gl_PointSize = pointSize * (0.5 + concentration);
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    varying float vConcentration;
                    varying vec3 vColor;
                    
                    void main() {
                        vec2 center = gl_PointCoord - 0.5;
                        float distance = length(center);
                        
                        if (distance > 0.5) discard;
                        
                        float alpha = (1.0 - distance * 2.0) * vConcentration;
                        gl_FragColor = vec4(vColor, alpha);
                    }
                `,
                transparent: true,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            });
            
            const chemicalField = new THREE.Points(geometry, material);
            chemicalField.userData.chemical = chemical.name;
            chemicalField.visible = this.simulation.showChemistry;
            
            this.chemicalConcentrations.push(chemicalField);
            this.scene.add(chemicalField);
        });
    }
    
    createAlterationVisualization() {
        console.log('Creating alteration zone visualization...');
        
        const alterationTypes = [
            {name: 'Propylitic', color: 0x00aa00, radius: 40, depth: -30},
            {name: 'Phyllic', color: 0xaaaa00, radius: 25, depth: -20},
            {name: 'Argillic', color: 0xaa4400, radius: 15, depth: -15},
            {name: 'Potassic', color: 0xaa0044, radius: 8, depth: -10}
        ];
        
        alterationTypes.forEach(alteration => {
            const geometry = new THREE.SphereGeometry(alteration.radius, 32, 32);
            const material = new THREE.MeshPhongMaterial({
                color: alteration.color,
                transparent: true,
                opacity: 0.15,
                side: THREE.DoubleSide,
                wireframe: true
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(0, alteration.depth, 0);
            sphere.userData.alterationType = alteration.name;
            sphere.visible = this.simulation.showAlteration;
            
            this.alteration_zones.push(sphere);
            this.scene.add(sphere);
        });
    }
    
    setupControls() {
        // Create control panel
        const controlPanel = document.createElement('div');
        controlPanel.id = 'fluid-controls';
        controlPanel.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            min-width: 200px;
        `;
        
        controlPanel.innerHTML = `
            <h3 style="margin: 0 0 10px 0;">Fluid Circulation Controls</h3>
            
            <div style="margin: 10px 0;">
                <label>Visualization Mode:</label>
                <select id="viz-mode" style="width: 100%; margin-top: 5px;">
                    <option value="particles">Fluid Particles</option>
                    <option value="vectors">Flow Vectors</option>
                    <option value="temperature">Temperature Field</option>
                    <option value="chemistry">Chemical Transport</option>
                    <option value="all">All Combined</option>
                </select>
            </div>
            
            <div style="margin: 10px 0;">
                <label>
                    <input type="checkbox" id="show-vectors" ${this.simulation.showVectors ? 'checked' : ''}> 
                    Flow Vectors
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label>
                    <input type="checkbox" id="show-temperature" ${this.simulation.showTemperature ? 'checked' : ''}> 
                    Temperature Field
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label>
                    <input type="checkbox" id="show-chemistry" ${this.simulation.showChemistry ? 'checked' : ''}> 
                    Chemical Concentrations
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label>
                    <input type="checkbox" id="show-alteration" ${this.simulation.showAlteration ? 'checked' : ''}> 
                    Alteration Zones
                </label>
            </div>
            
            <div style="margin: 10px 0;">
                <label>Flow Speed:</label>
                <input type="range" id="flow-speed" min="0.1" max="5.0" step="0.1" value="${this.simulation.flowSpeed}" style="width: 100%;">
                <span id="speed-value">${this.simulation.flowSpeed}</span>
            </div>
            
            <div style="margin: 10px 0;">
                <label>Animation:</label>
                <button id="play-pause">${this.animationState.playing ? 'Pause' : 'Play'}</button>
                <button id="reset">Reset</button>
            </div>
        `;
        
        this.container.appendChild(controlPanel);
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Visualization mode
        document.getElementById('viz-mode').addEventListener('change', (e) => {
            this.simulation.visualizationMode = e.target.value;
            this.updateVisualization();
        });
        
        // Checkboxes
        document.getElementById('show-vectors').addEventListener('change', (e) => {
            this.simulation.showVectors = e.target.checked;
            this.flowVectors.forEach(arrow => arrow.visible = e.target.checked);
        });
        
        document.getElementById('show-temperature').addEventListener('change', (e) => {
            this.simulation.showTemperature = e.target.checked;
            // Update temperature visualization visibility
        });
        
        document.getElementById('show-chemistry').addEventListener('change', (e) => {
            this.simulation.showChemistry = e.target.checked;
            this.chemicalConcentrations.forEach(field => field.visible = e.target.checked);
        });
        
        document.getElementById('show-alteration').addEventListener('change', (e) => {
            this.simulation.showAlteration = e.target.checked;
            this.alteration_zones.forEach(zone => zone.visible = e.target.checked);
        });
        
        // Flow speed slider
        document.getElementById('flow-speed').addEventListener('input', (e) => {
            this.simulation.flowSpeed = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = this.simulation.flowSpeed;
        });
        
        // Animation controls
        document.getElementById('play-pause').addEventListener('click', () => {
            this.animationState.playing = !this.animationState.playing;
            document.getElementById('play-pause').textContent = this.animationState.playing ? 'Pause' : 'Play';
        });
        
        document.getElementById('reset').addEventListener('click', () => {
            this.resetSimulation();
        });
        
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    updateVisualization() {
        const mode = this.simulation.visualizationMode;
        
        // Hide all components first
        if (this.fluidParticles) this.fluidParticles.visible = false;
        this.flowVectors.forEach(arrow => arrow.visible = false);
        this.chemicalConcentrations.forEach(field => field.visible = false);
        this.alteration_zones.forEach(zone => zone.visible = false);
        
        // Show components based on mode
        switch (mode) {
            case 'particles':
                if (this.fluidParticles) this.fluidParticles.visible = true;
                break;
            case 'vectors':
                this.flowVectors.forEach(arrow => arrow.visible = true);
                break;
            case 'temperature':
                // Show temperature visualizations
                break;
            case 'chemistry':
                this.chemicalConcentrations.forEach(field => field.visible = true);
                break;
            case 'all':
                if (this.fluidParticles) this.fluidParticles.visible = this.simulation.showVectors;
                this.flowVectors.forEach(arrow => arrow.visible = this.simulation.showVectors);
                this.chemicalConcentrations.forEach(field => field.visible = this.simulation.showChemistry);
                this.alteration_zones.forEach(zone => zone.visible = this.simulation.showAlteration);
                break;
        }
    }
    
    updateParticleSystem(deltaTime) {
        if (!this.fluidParticles) return;
        
        const positions = this.fluidParticles.geometry.attributes.position.array;
        const velocities = this.fluidParticles.geometry.attributes.velocity.array;
        const temperatures = this.fluidParticles.geometry.attributes.temperature.array;
        const ages = this.fluidParticles.geometry.attributes.age.array;
        
        const particleCount = positions.length / 3;
        const dt = deltaTime * this.simulation.flowSpeed;
        
        for (let i = 0; i < particleCount; i++) {
            const idx = i * 3;
            
            // Current position
            const x = positions[idx];
            const y = positions[idx + 1];
            const z = positions[idx + 2];
            
            // Calculate flow velocity at current position
            const flowVelocity = this.calculateFlowDirection(x, y, z);
            
            // Update velocities (with some inertia)
            const inertia = 0.9;
            velocities[idx] = inertia * velocities[idx] + (1 - inertia) * flowVelocity.x;
            velocities[idx + 1] = inertia * velocities[idx + 1] + (1 - inertia) * flowVelocity.y;
            velocities[idx + 2] = inertia * velocities[idx + 2] + (1 - inertia) * flowVelocity.z;
            
            // Update positions
            positions[idx] += velocities[idx] * dt;
            positions[idx + 1] += velocities[idx + 1] * dt;
            positions[idx + 2] += velocities[idx + 2] * dt;
            
            // Update age
            ages[i] += dt * 10;
            
            // Temperature cooling
            temperatures[i] *= 0.9999;
            
            // Boundary conditions and particle recycling
            const distance = Math.sqrt(x*x + y*y + z*z);
            if (distance > 150 || ages[i] > 2000) {
                // Reset particle at a source
                const sourceIdx = Math.floor(Math.random() * 3);
                const sources = [
                    [0, -50, 0],
                    [30, -30, 20],
                    [-20, -40, -10]
                ];
                
                const source = sources[sourceIdx];
                positions[idx] = source[0] + (Math.random() - 0.5) * 10;
                positions[idx + 1] = source[1] + (Math.random() - 0.5) * 5;
                positions[idx + 2] = source[2] + (Math.random() - 0.5) * 10;
                
                velocities[idx] = (Math.random() - 0.5) * 2;
                velocities[idx + 1] = 1 + Math.random() * 3;
                velocities[idx + 2] = (Math.random() - 0.5) * 2;
                
                temperatures[i] = 573 + Math.random() * 300;
                ages[i] = 0;
            }
        }
        
        // Mark attributes as needing update
        this.fluidParticles.geometry.attributes.position.needsUpdate = true;
        this.fluidParticles.geometry.attributes.velocity.needsUpdate = true;
        this.fluidParticles.geometry.attributes.temperature.needsUpdate = true;
        this.fluidParticles.geometry.attributes.age.needsUpdate = true;
    }
    
    startAnimation() {
        this.animate();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.animationState.playing) {
            const deltaTime = 0.016; // ~60 FPS
            this.animationState.time += deltaTime;
            this.animationState.frame++;
            
            // Update particle system
            this.updateParticleSystem(deltaTime);
            
            // Update uniforms
            if (this.fluidParticles) {
                this.fluidParticles.material.uniforms.time.value = this.animationState.time;
            }
            
            if (this.volumetricLight) {
                this.volumetricLight.material.uniforms.time.value = this.animationState.time;
            }
            
            // Update chemical concentrations
            this.chemicalConcentrations.forEach(field => {
                if (field.material.uniforms.time) {
                    field.material.uniforms.time.value = this.animationState.time;
                }
            });
            
            // Animate hydrothermal lights (flickering)
            this.scene.traverse((child) => {
                if (child instanceof THREE.PointLight && child.userData.originalIntensity) {
                    const flicker = Math.sin(this.animationState.time * child.userData.flickerSpeed) * 0.1 + 1.0;
                    child.intensity = child.userData.originalIntensity * flicker;
                }
            });
        }
        
        // Update controls and render
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    resetSimulation() {
        this.animationState.time = 0;
        this.animationState.frame = 0;
        
        // Reset particle positions and properties
        if (this.fluidParticles) {
            this.createFluidParticleSystem();
        }
        
        console.log('Simulation reset');
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    exportData() {
        const data = {
            simulation_parameters: this.simulation,
            fluid_properties: this.fluidProperties,
            particle_count: this.simulation.particleCount,
            timestamp: Date.now(),
            frame: this.animationState.frame
        };
        
        return data;
    }
    
    dispose() {
        // Clean up resources
        this.renderer.dispose();
        this.scene.clear();
        
        // Remove controls
        const controlPanel = document.getElementById('fluid-controls');
        if (controlPanel) {
            controlPanel.remove();
        }
        
        console.log('🧹 Fluid Circulation 3D disposed');
    }
}

// Export for use in other modules
window.FluidCirculation3D = FluidCirculation3D;

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('fluid-circulation-container');
    if (container && window.hydrodata) {
        window.fluidCirculation3D = new FluidCirculation3D(container, window.hydrodata);
    }
});