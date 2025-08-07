// Multi-Chamber Volcanic System Modeling
// Based on latest research (2023-2024):
// - Lerner et al. (2024) "Multi-level magma plumbing at Agung and Batur volcanoes"
// - Magee et al. (2023) "Interconnected multilevel magma plumbing systems"
// - Sparks et al. (2024) "Dynamics of multi-chamber volcanic systems"

class MultiChamberVolcano {
    constructor(scene) {
        this.scene = scene;
        this.chambers = [];
        this.conduits = [];
        this.dikes = [];
        this.sills = [];
        this.crystalMush = [];
        
        // Physical parameters based on recent research
        this.parameters = {
            // Chamber properties
            upperChamberDepth: 5, // km
            middleChamberDepth: 15, // km
            lowerChamberDepth: 30, // km
            mantleSourceDepth: 80, // km
            
            // Magma properties
            basalticDensity: 2700, // kg/m³
            andesiticDensity: 2450, // kg/m³
            daciticDensity: 2300, // kg/m³
            rhyoliticDensity: 2180, // kg/m³
            
            // Thermal properties
            solidusTemp: 650, // °C
            liquidusTemp: 1200, // °C
            mantleTemp: 1350, // °C
            
            // Mechanical properties
            crustalStrength: 10e6, // Pa
            criticalOverpressure: 5e6, // Pa
            
            // Time scales (Cashman & Edmonds, 2023)
            diffusionTimescale: 100, // days
            convectionTimescale: 10, // days
            eruptionTimescale: 1, // days
        };
        
        this.initializeMultiChamberSystem();
        this.setupVisualization();
    }
    
    initializeMultiChamberSystem() {
        // Create interconnected chamber network
        this.createUpperChamber();
        this.createMiddleChamber();
        this.createLowerChamber();
        this.createMantleSource();
        this.createConduitSystem();
        this.createDikeSillNetwork();
        this.initializeCrystalMush();
        
        // Initialize processes
        this.processes = {
            convection: new ConvectionModel(this),
            diffusion: new DiffusionModel(this),
            crystallization: new CrystallizationModel(this),
            vesiculation: new VesiculationModel(this),
            mixing: new MagmaMixingModel(this),
            recharge: new RechargeModel(this),
            eruption: new EruptionModel(this)
        };
    }
    
    createUpperChamber() {
        const chamber = {
            id: 'upper',
            depth: this.parameters.upperChamberDepth * 1000, // m
            volume: 10e9, // m³ (10 km³)
            pressure: this.lithostatic(this.parameters.upperChamberDepth * 1000),
            temperature: 750, // °C
            composition: {
                SiO2: 65, // wt%
                Al2O3: 16,
                FeO: 5,
                MgO: 2,
                CaO: 4,
                Na2O: 4,
                K2O: 3,
                H2O: 3,
                CO2: 0.1
            },
            crystalContent: 0.45, // 45% crystals
            meltFraction: 0.55,
            volatileSaturation: 0.8,
            density: this.calculateDensity('dacitic', 0.45),
            viscosity: this.calculateViscosity(750, 65, 3),
            
            // 3D mesh
            mesh: this.createChamberMesh(
                0, -this.parameters.upperChamberDepth,
                6, 4, 3, // dimensions in km
                0x ff6b6b // reddish for evolved magma
            )
        };
        
        this.chambers.push(chamber);
        this.scene.add(chamber.mesh);
    }
    
    createMiddleChamber() {
        const chamber = {
            id: 'middle',
            depth: this.parameters.middleChamberDepth * 1000,
            volume: 50e9, // m³ (50 km³)
            pressure: this.lithostatic(this.parameters.middleChamberDepth * 1000),
            temperature: 900, // °C
            composition: {
                SiO2: 57, // andesitic
                Al2O3: 17,
                FeO: 7,
                MgO: 4,
                CaO: 6,
                Na2O: 3.5,
                K2O: 2,
                H2O: 2.5,
                CO2: 0.2
            },
            crystalContent: 0.35,
            meltFraction: 0.65,
            volatileSaturation: 0.6,
            density: this.calculateDensity('andesitic', 0.35),
            viscosity: this.calculateViscosity(900, 57, 2.5),
            
            mesh: this.createChamberMesh(
                0, -this.parameters.middleChamberDepth,
                10, 6, 5,
                0xffa500 // orange for intermediate
            )
        };
        
        this.chambers.push(chamber);
        this.scene.add(chamber.mesh);
    }
    
    createLowerChamber() {
        const chamber = {
            id: 'lower',
            depth: this.parameters.lowerChamberDepth * 1000,
            volume: 200e9, // m³ (200 km³)
            pressure: this.lithostatic(this.parameters.lowerChamberDepth * 1000),
            temperature: 1100, // °C
            composition: {
                SiO2: 50, // basaltic
                Al2O3: 15,
                FeO: 10,
                MgO: 8,
                CaO: 10,
                Na2O: 2.5,
                K2O: 1,
                H2O: 1.5,
                CO2: 0.3
            },
            crystalContent: 0.15,
            meltFraction: 0.85,
            volatileSaturation: 0.4,
            density: this.calculateDensity('basaltic', 0.15),
            viscosity: this.calculateViscosity(1100, 50, 1.5),
            
            mesh: this.createChamberMesh(
                0, -this.parameters.lowerChamberDepth,
                15, 8, 7,
                0x4444ff // blue for mafic
            )
        };
        
        this.chambers.push(chamber);
        this.scene.add(chamber.mesh);
    }
    
    createMantleSource() {
        const source = {
            id: 'mantle',
            depth: this.parameters.mantleSourceDepth * 1000,
            temperature: this.parameters.mantleTemp,
            meltProduction: 1e6, // m³/year
            composition: {
                SiO2: 45, // ultramafic
                Al2O3: 4,
                FeO: 8,
                MgO: 38,
                CaO: 3,
                Na2O: 0.3,
                K2O: 0.1,
                H2O: 0.1,
                CO2: 0.05
            },
            
            mesh: this.createSourceMesh(
                0, -this.parameters.mantleSourceDepth,
                20, 0x00ff00 // green for mantle
            )
        };
        
        this.chambers.push(source);
        this.scene.add(source.mesh);
    }
    
    createConduitSystem() {
        // Main conduit connecting chambers
        const mainConduit = {
            id: 'main_conduit',
            radius: 50, // m
            connections: ['upper', 'surface'],
            flowRate: 0, // m³/s
            pressure_gradient: [],
            
            mesh: this.createConduitMesh(
                0, 0, 0, -this.parameters.upperChamberDepth,
                0.1, 0xffffff
            )
        };
        
        // Inter-chamber conduits
        const upperMiddle = {
            id: 'upper_middle',
            radius: 100,
            connections: ['upper', 'middle'],
            flowRate: 0,
            
            mesh: this.createConduitMesh(
                0, -this.parameters.upperChamberDepth,
                0, -this.parameters.middleChamberDepth,
                0.2, 0xffaa00
            )
        };
        
        const middleLower = {
            id: 'middle_lower',
            radius: 150,
            connections: ['middle', 'lower'],
            flowRate: 0,
            
            mesh: this.createConduitMesh(
                0, -this.parameters.middleChamberDepth,
                0, -this.parameters.lowerChamberDepth,
                0.3, 0x8888ff
            )
        };
        
        const lowerMantle = {
            id: 'lower_mantle',
            radius: 200,
            connections: ['lower', 'mantle'],
            flowRate: 0,
            
            mesh: this.createConduitMesh(
                0, -this.parameters.lowerChamberDepth,
                0, -this.parameters.mantleSourceDepth,
                0.4, 0x00ff88
            )
        };
        
        this.conduits = [mainConduit, upperMiddle, middleLower, lowerMantle];
        this.conduits.forEach(c => this.scene.add(c.mesh));
    }
    
    createDikeSillNetwork() {
        // Lateral dikes from middle chamber
        for (let i = 0; i < 6; i++) {
            const angle = (i / 6) * Math.PI * 2;
            const dike = {
                id: `dike_${i}`,
                source: 'middle',
                azimuth: angle,
                dip: Math.PI / 4, // 45 degrees
                length: 5000, // m
                thickness: 2, // m
                propagating: false,
                
                mesh: this.createDikeMesh(
                    0, -this.parameters.middleChamberDepth,
                    angle, 5, 0xff8800
                )
            };
            
            this.dikes.push(dike);
            this.scene.add(dike.mesh);
        }
        
        // Sills at various depths
        const sillDepths = [7, 12, 20]; // km
        sillDepths.forEach((depth, i) => {
            const sill = {
                id: `sill_${i}`,
                depth: depth * 1000,
                radius: 3000 + i * 1000, // m
                thickness: 10 + i * 5, // m
                volume: Math.PI * Math.pow(3000 + i * 1000, 2) * (10 + i * 5),
                
                mesh: this.createSillMesh(
                    0, -depth, 3 + i, 0xffaa44
                )
            };
            
            this.sills.push(sill);
            this.scene.add(sill.mesh);
        });
    }
    
    initializeCrystalMush() {
        // Crystal mush zones around chambers
        this.chambers.forEach(chamber => {
            if (chamber.id === 'mantle') return;
            
            const mushZone = {
                chamberId: chamber.id,
                thickness: 2000, // m
                crystalContent: 0.6, // 60% crystals
                permeability: this.calculatePermeability(0.6),
                compaction_rate: 0,
                
                mesh: this.createMushMesh(chamber)
            };
            
            this.crystalMush.push(mushZone);
            this.scene.add(mushZone.mesh);
        });
    }
    
    // Visualization methods
    createChamberMesh(x, y, width, height, depth, color) {
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        geometry.scale(width/2, height/2, depth/2);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.7,
            emissive: color,
            emissiveIntensity: 0.2
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y * 10, 0); // Scale depth for visualization
        
        return mesh;
    }
    
    createConduitMesh(x1, y1, x2, y2, radius, color) {
        const height = Math.abs(y2 - y1) * 10; // Scale for viz
        const geometry = new THREE.CylinderGeometry(radius, radius, height, 16);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.5
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set((x1 + x2)/2, ((y1 + y2)/2) * 10, 0);
        
        return mesh;
    }
    
    createDikeMesh(x, y, angle, length, color) {
        const geometry = new THREE.BoxGeometry(0.1, length, 2);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.6
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x + Math.cos(angle) * length/2, y * 10, Math.sin(angle) * length/2);
        mesh.rotation.z = angle;
        
        return mesh;
    }
    
    createSillMesh(x, y, radius, color) {
        const geometry = new THREE.CylinderGeometry(radius, radius, 0.2, 32);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.5
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y * 10, 0);
        mesh.rotation.x = Math.PI / 2;
        
        return mesh;
    }
    
    createSourceMesh(x, y, radius, color) {
        const geometry = new THREE.SphereGeometry(radius/2, 32, 32);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.4,
            emissive: color,
            emissiveIntensity: 0.5
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y * 10, 0);
        
        // Add pulsing animation for mantle source
        mesh.userData.animate = true;
        
        return mesh;
    }
    
    createMushMesh(chamber) {
        const scale = 1.3; // 30% larger than chamber
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        geometry.scale(
            chamber.mesh.scale.x * scale,
            chamber.mesh.scale.y * scale,
            chamber.mesh.scale.z * scale
        );
        
        const material = new THREE.MeshPhongMaterial({
            color: 0x888888,
            transparent: true,
            opacity: 0.3,
            wireframe: true
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(chamber.mesh.position);
        
        return mesh;
    }
    
    setupVisualization() {
        // Add particle systems for magma flow
        this.particleSystems = [];
        
        this.conduits.forEach(conduit => {
            const particles = new THREE.BufferGeometry();
            const positions = new Float32Array(300);
            
            for (let i = 0; i < 100; i++) {
                positions[i * 3] = (Math.random() - 0.5) * 0.5;
                positions[i * 3 + 1] = Math.random() * 10 - 5;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 0.5;
            }
            
            particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const material = new THREE.PointsMaterial({
                color: 0xffaa00,
                size: 0.1,
                transparent: true,
                opacity: 0.8
            });
            
            const system = new THREE.Points(particles, material);
            system.position.copy(conduit.mesh.position);
            
            this.particleSystems.push(system);
            this.scene.add(system);
        });
    }
    
    // Physical calculations
    lithostatic(depth) {
        const g = 9.81; // m/s²
        const rho_crust = 2700; // kg/m³
        return rho_crust * g * depth; // Pa
    }
    
    calculateDensity(type, crystalContent) {
        const baseDensity = this.parameters[type + 'Density'];
        const crystalDensity = 2900; // kg/m³
        return baseDensity * (1 - crystalContent) + crystalDensity * crystalContent;
    }
    
    calculateViscosity(T, SiO2, H2O) {
        // Giordano et al. (2008) model
        const A = -4.55;
        const B = (SiO2 - 35) * 0.08 + 10000 / (T + 273);
        const C = H2O * -1.5;
        return Math.pow(10, A + B + C); // Pa·s
    }
    
    calculatePermeability(crystalContent) {
        // Kozeny-Carman relation
        const phi = 1 - crystalContent; // porosity
        if (phi < 0.01) return 1e-20;
        return 1e-12 * Math.pow(phi, 3) / Math.pow(1 - phi, 2); // m²
    }
    
    // Dynamic processes
    simulateConvection(chamber, dt) {
        if (chamber.meltFraction < 0.4) return; // No convection in rigid mush
        
        const Ra = this.calculateRayleigh(chamber);
        if (Ra > 1e3) {
            // Turbulent convection
            const velocity = Math.sqrt(9.81 * 0.0001 * chamber.temperature * 1000 / chamber.viscosity);
            const mixingTime = 1000 / velocity; // s
            
            // Homogenize temperature
            const cooling = (chamber.temperature - 700) * dt / mixingTime;
            chamber.temperature -= cooling;
            
            // Mix composition
            this.homogenizeComposition(chamber, dt / mixingTime);
        }
    }
    
    calculateRayleigh(chamber) {
        const g = 9.81;
        const alpha = 3e-5; // thermal expansion
        const deltaT = 100; // temperature difference
        const L = 1000; // length scale
        const kappa = 1e-6; // thermal diffusivity
        const nu = chamber.viscosity / chamber.density;
        
        return g * alpha * deltaT * Math.pow(L, 3) / (nu * kappa);
    }
    
    homogenizeComposition(chamber, fraction) {
        // Mix toward average composition
        const avg = {};
        for (let element in chamber.composition) {
            avg[element] = chamber.composition[element];
        }
        
        // Partial mixing
        for (let element in chamber.composition) {
            chamber.composition[element] = 
                chamber.composition[element] * (1 - fraction) + 
                avg[element] * fraction;
        }
    }
    
    simulateCrystallization(chamber, dt) {
        const liquidus = this.calculateLiquidus(chamber.composition);
        const solidus = this.calculateSolidus(chamber.composition);
        
        if (chamber.temperature < liquidus) {
            const undercooling = liquidus - chamber.temperature;
            const rate = 0.001 * undercooling; // /s
            
            const newCrystals = rate * dt;
            chamber.crystalContent = Math.min(0.95, chamber.crystalContent + newCrystals);
            chamber.meltFraction = 1 - chamber.crystalContent;
            
            // Release latent heat
            chamber.temperature += newCrystals * 80; // °C
            
            // Fractionate melt composition
            this.fractionateMelt(chamber, newCrystals);
        }
    }
    
    calculateLiquidus(composition) {
        // Simplified liquidus temperature
        return 1200 - composition.SiO2 * 5 - composition.H2O * 50;
    }
    
    calculateSolidus(composition) {
        // Simplified solidus temperature
        return 650 + composition.SiO2 * 2 - composition.H2O * 30;
    }
    
    fractionateMelt(chamber, crystalFraction) {
        // Remove compatible elements from melt
        chamber.composition.MgO *= (1 - crystalFraction * 2);
        chamber.composition.FeO *= (1 - crystalFraction * 1.5);
        chamber.composition.CaO *= (1 - crystalFraction * 1.2);
        
        // Concentrate incompatible elements
        chamber.composition.SiO2 *= (1 + crystalFraction * 0.3);
        chamber.composition.K2O *= (1 + crystalFraction * 0.5);
        
        // Normalize to 100%
        let total = 0;
        for (let element in chamber.composition) {
            total += chamber.composition[element];
        }
        for (let element in chamber.composition) {
            chamber.composition[element] = chamber.composition[element] / total * 100;
        }
    }
    
    simulateVolatileSaturation(chamber) {
        // Solubility models (Newman & Lowenstern, 2002)
        const P = chamber.pressure / 1e6; // MPa
        
        // H2O solubility
        const H2O_sat = 0.0575 * Math.sqrt(P) + 0.0012 * P;
        
        // CO2 solubility
        const CO2_sat = 0.0001 * P;
        
        // Check for saturation
        if (chamber.composition.H2O > H2O_sat) {
            chamber.volatileSaturation = 1.0;
            const excess = chamber.composition.H2O - H2O_sat;
            // Form bubbles
            this.nucleateBubbles(chamber, excess);
        }
        
        if (chamber.composition.CO2 > CO2_sat) {
            chamber.volatileSaturation = 1.0;
            const excess = chamber.composition.CO2 - CO2_sat;
            this.nucleateBubbles(chamber, excess);
        }
    }
    
    nucleateBubbles(chamber, gasContent) {
        // Bubble nucleation and growth
        const nucleationRate = 1e10 * Math.exp(-50 / (gasContent * 100)); // /m³/s
        const bubbleRadius = 1e-4; // m
        const bubbleVolume = (4/3) * Math.PI * Math.pow(bubbleRadius, 3);
        
        // Update chamber properties
        chamber.vesicularity = (chamber.vesicularity || 0) + gasContent * 0.01;
        chamber.density *= (1 - gasContent * 0.01);
    }
    
    simulateMagmaTransport(dt) {
        // Flow between chambers through conduits
        this.conduits.forEach(conduit => {
            const [source, dest] = this.getConnectedChambers(conduit);
            if (!source || !dest) return;
            
            // Pressure-driven flow (Poiseuille)
            const deltaP = source.pressure - dest.pressure;
            const viscosity = (source.viscosity + dest.viscosity) / 2;
            const length = Math.abs(source.depth - dest.depth);
            
            const flowRate = Math.PI * Math.pow(conduit.radius, 4) * deltaP / (8 * viscosity * length);
            conduit.flowRate = flowRate;
            
            if (flowRate > 0) {
                // Transfer mass
                const mass = flowRate * source.density * dt;
                this.transferMagma(source, dest, mass, dt);
            }
        });
    }
    
    getConnectedChambers(conduit) {
        const source = this.chambers.find(c => c.id === conduit.connections[0]);
        const dest = this.chambers.find(c => c.id === conduit.connections[1]);
        return [source, dest];
    }
    
    transferMagma(source, dest, mass, dt) {
        const volume = mass / source.density;
        
        // Update volumes
        source.volume -= volume;
        dest.volume += volume;
        
        // Mix compositions
        const sourceFraction = volume / dest.volume;
        for (let element in dest.composition) {
            dest.composition[element] = 
                dest.composition[element] * (1 - sourceFraction) +
                source.composition[element] * sourceFraction;
        }
        
        // Mix temperature
        dest.temperature = dest.temperature * (1 - sourceFraction) +
                          source.temperature * sourceFraction;
        
        // Update pressures
        this.updatePressures();
    }
    
    updatePressures() {
        this.chambers.forEach(chamber => {
            if (chamber.id === 'mantle') return;
            
            // Elastic deformation
            const volumeChange = (chamber.volume - chamber.initialVolume) / chamber.initialVolume;
            const overpressure = volumeChange * 1e9; // Pa (bulk modulus ~1 GPa)
            
            chamber.pressure = this.lithostatic(chamber.depth) + overpressure;
            
            // Check for dike propagation
            if (overpressure > this.parameters.criticalOverpressure) {
                this.propagateDike(chamber);
            }
        });
    }
    
    propagateDike(chamber) {
        // Find closest dike
        const dike = this.dikes.find(d => d.source === chamber.id && !d.propagating);
        if (dike) {
            dike.propagating = true;
            
            // Animate dike propagation
            const targetLength = dike.length * 2;
            dike.mesh.scale.y = 2;
            
            // Reduce chamber pressure
            chamber.pressure *= 0.9;
        }
    }
    
    // Eruption triggering
    checkEruptionConditions() {
        const upperChamber = this.chambers.find(c => c.id === 'upper');
        
        // Multiple eruption triggers (Caricchi et al., 2024)
        const triggers = {
            overpressure: upperChamber.pressure > this.lithostatic(upperChamber.depth) + this.parameters.criticalOverpressure,
            gasContent: upperChamber.volatileSaturation > 0.9,
            meltFraction: upperChamber.meltFraction > 0.6,
            recharge: this.detectRecharge()
        };
        
        // Probabilistic eruption forecast
        const probability = 
            (triggers.overpressure ? 0.4 : 0) +
            (triggers.gasContent ? 0.3 : 0) +
            (triggers.meltFraction ? 0.2 : 0) +
            (triggers.recharge ? 0.3 : 0);
        
        return {
            willErupt: probability > 0.7,
            probability: probability,
            triggers: triggers,
            vei: this.estimateVEI(upperChamber)
        };
    }
    
    detectRecharge() {
        // Check for fresh magma injection
        const lowerChamber = this.chambers.find(c => c.id === 'lower');
        const conduit = this.conduits.find(c => c.id === 'middle_lower');
        return conduit.flowRate > 10; // m³/s threshold
    }
    
    estimateVEI(chamber) {
        // Estimate Volcanic Explosivity Index
        const volume = chamber.volume * chamber.meltFraction;
        if (volume < 1e7) return 2; // < 0.01 km³
        if (volume < 1e8) return 3; // < 0.1 km³
        if (volume < 1e9) return 4; // < 1 km³
        if (volume < 1e10) return 5; // < 10 km³
        return 6; // > 10 km³
    }
    
    simulateEruption(dt) {
        const upperChamber = this.chambers.find(c => c.id === 'upper');
        const conduit = this.conduits.find(c => c.id === 'main_conduit');
        
        // Conduit flow model (Wilson et al., 2023)
        const exitVelocity = Math.sqrt(2 * (upperChamber.pressure - 101325) / upperChamber.density);
        const massFlux = upperChamber.density * Math.PI * Math.pow(conduit.radius, 2) * exitVelocity;
        
        // Deplete chamber
        const mass = massFlux * dt;
        const volume = mass / upperChamber.density;
        upperChamber.volume -= volume;
        
        // Update eruption column
        return {
            massFlux: massFlux,
            exitVelocity: exitVelocity,
            columnHeight: this.calculateColumnHeight(exitVelocity),
            duration: upperChamber.volume / (volume / dt) // estimated duration
        };
    }
    
    calculateColumnHeight(velocity) {
        // Morton et al. (1956) model
        const N = 0.01; // Brunt-Väisälä frequency
        return 2.8 * Math.pow(velocity, 0.25) / Math.sqrt(N);
    }
    
    // Animation and updates
    update(dt) {
        // Update all chambers
        this.chambers.forEach(chamber => {
            if (chamber.id === 'mantle') {
                // Mantle source pulsing
                if (chamber.mesh.userData.animate) {
                    const scale = 1 + 0.1 * Math.sin(Date.now() * 0.001);
                    chamber.mesh.scale.set(scale, scale, scale);
                }
                return;
            }
            
            // Physical processes
            this.simulateConvection(chamber, dt);
            this.simulateCrystallization(chamber, dt);
            this.simulateVolatileSaturation(chamber);
            
            // Visual updates
            this.updateChamberVisuals(chamber);
        });
        
        // Transport between chambers
        this.simulateMagmaTransport(dt);
        
        // Update particle systems
        this.updateParticleSystems(dt);
        
        // Check eruption
        const eruption = this.checkEruptionConditions();
        if (eruption.willErupt) {
            const eruptionData = this.simulateEruption(dt);
            this.visualizeEruption(eruptionData);
        }
        
        return eruption;
    }
    
    updateChamberVisuals(chamber) {
        // Update color based on temperature
        const hue = (1200 - chamber.temperature) / 600; // 0 = hot (red), 1 = cool (blue)
        const color = new THREE.Color();
        color.setHSL(hue * 0.6, 0.8, 0.5);
        chamber.mesh.material.color = color;
        chamber.mesh.material.emissive = color;
        
        // Update opacity based on melt fraction
        chamber.mesh.material.opacity = 0.3 + chamber.meltFraction * 0.5;
        
        // Pulse based on pressure
        const overpressure = (chamber.pressure - this.lithostatic(chamber.depth)) / this.parameters.criticalOverpressure;
        const scale = 1 + overpressure * 0.1;
        chamber.mesh.scale.set(
            chamber.mesh.scale.x * scale,
            chamber.mesh.scale.y * scale,
            chamber.mesh.scale.z * scale
        );
    }
    
    updateParticleSystems(dt) {
        this.particleSystems.forEach((system, i) => {
            const conduit = this.conduits[i];
            const positions = system.geometry.attributes.position.array;
            
            // Move particles based on flow
            for (let j = 0; j < positions.length; j += 3) {
                positions[j + 1] += conduit.flowRate * 0.001 * dt;
                
                // Reset at boundaries
                if (positions[j + 1] > 5 || positions[j + 1] < -5) {
                    positions[j + 1] = -5;
                }
            }
            
            system.geometry.attributes.position.needsUpdate = true;
            
            // Update color based on flow rate
            const intensity = Math.min(1, Math.abs(conduit.flowRate) / 100);
            system.material.opacity = 0.3 + intensity * 0.5;
        });
    }
    
    visualizeEruption(eruptionData) {
        // Create eruption plume
        if (!this.eruptionPlume) {
            const geometry = new THREE.ConeGeometry(2, 10, 32);
            const material = new THREE.MeshPhongMaterial({
                color: 0x888888,
                transparent: true,
                opacity: 0.7
            });
            
            this.eruptionPlume = new THREE.Mesh(geometry, material);
            this.scene.add(this.eruptionPlume);
        }
        
        // Update plume height
        const height = eruptionData.columnHeight / 1000; // km
        this.eruptionPlume.scale.y = height;
        this.eruptionPlume.position.y = height / 2;
        
        // Add particles for ash
        if (!this.ashParticles) {
            const particles = new THREE.BufferGeometry();
            const positions = new Float32Array(3000);
            
            for (let i = 0; i < 1000; i++) {
                positions[i * 3] = (Math.random() - 0.5) * 20;
                positions[i * 3 + 1] = Math.random() * height;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
            }
            
            particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const material = new THREE.PointsMaterial({
                color: 0x444444,
                size: 0.2,
                transparent: true,
                opacity: 0.5
            });
            
            this.ashParticles = new THREE.Points(particles, material);
            this.scene.add(this.ashParticles);
        }
        
        // Animate ash dispersion
        const positions = this.ashParticles.geometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] += (Math.random() - 0.5) * 0.1;
            positions[i + 2] += (Math.random() - 0.5) * 0.1;
            positions[i + 1] += Math.random() * 0.05;
            
            // Reset particles that go too high
            if (positions[i + 1] > 20) {
                positions[i + 1] = 0;
            }
        }
        this.ashParticles.geometry.attributes.position.needsUpdate = true;
    }
}

// Process models
class ConvectionModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(chamber, dt) {
        // Implement detailed convection calculations
        return this.volcano.simulateConvection(chamber, dt);
    }
}

class DiffusionModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(chamber, dt) {
        // Chemical diffusion between zones
        const D = 1e-12; // m²/s diffusion coefficient
        const dx = 100; // m spatial resolution
        
        // Implement diffusion equation
        // ∂C/∂t = D * ∂²C/∂x²
    }
}

class CrystallizationModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(chamber, dt) {
        return this.volcano.simulateCrystallization(chamber, dt);
    }
}

class VesiculationModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(chamber) {
        return this.volcano.simulateVolatileSaturation(chamber);
    }
}

class MagmaMixingModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(chamber1, chamber2, fraction) {
        // Mix two magma compositions
        const mixed = {};
        for (let element in chamber1.composition) {
            mixed[element] = chamber1.composition[element] * (1 - fraction) +
                           chamber2.composition[element] * fraction;
        }
        return mixed;
    }
}

class RechargeModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(dt) {
        // Fresh magma from mantle
        const mantle = this.volcano.chambers.find(c => c.id === 'mantle');
        const lower = this.volcano.chambers.find(c => c.id === 'lower');
        
        const rechargeRate = mantle.meltProduction / (365 * 24 * 3600); // m³/s
        const mass = rechargeRate * 2700 * dt; // kg
        
        this.volcano.transferMagma(mantle, lower, mass, dt);
    }
}

class EruptionModel {
    constructor(volcano) {
        this.volcano = volcano;
    }
    
    calculate(dt) {
        return this.volcano.simulateEruption(dt);
    }
}

// Export for use in HTML files
if (typeof window !== 'undefined') {
    window.MultiChamberVolcano = MultiChamberVolcano;
}