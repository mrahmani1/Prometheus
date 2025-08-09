/**
 * Multi-Chamber Volcanic System
 * Advanced 3D modeling and ML prediction for interconnected magma chambers
 */

class MultiChamberVolcanicSystem {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.renderer = null;
        this.camera = null;
        this.chambers = [];
        this.connections = [];
        this.simulationData = null;
        this.mlPredictions = null;
        
        // API endpoints
        this.API_BASE = window.API_URL || 'http://localhost:8000/api/v1';
        this.WS_URL = window.WS_URL || 'ws://localhost:8000/ws';
        
        // WebSocket for real-time updates
        this.ws = null;
        
        // Simulation parameters
        this.parameters = {
            timeRange: { start: 0, end: 365, step: 1 }, // days
            physics: {
                viscosity: 1e19, // Pa·s
                thermalDiffusivity: 1e-6, // m²/s
                gasExsolution: true,
                crystallization: true
            },
            ml: {
                useTransferLearning: true,
                uncertaintyQuantification: true,
                ensembleSize: 10
            }
        };
        
        this.init();
    }
    
    async init() {
        console.log('🌋 Initializing Multi-Chamber Volcanic System...');
        
        // Initialize 3D scene
        this.initScene();
        
        // Connect to backend
        await this.connectToBackend();
        
        // Load volcano data
        await this.loadVolcanoData();
        
        // Initialize UI controls
        this.initControls();
        
        // Start real-time monitoring
        this.startMonitoring();
        
        console.log('✅ Multi-Chamber System initialized');
    }
    
    initScene() {
        // Three.js scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            10000
        );
        this.camera.position.set(0, -50, 100);
        
        // Renderer with enhanced capabilities
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            logarithmicDepthBuffer: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 0);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Start animation loop
        this.animate();
    }
    
    async connectToBackend() {
        try {
            // Test API connection
            const response = await fetch(`${this.API_BASE}/health`);
            if (response.ok) {
                console.log('✅ Connected to backend API');
            }
            
            // Establish WebSocket connection
            this.ws = new WebSocket(`${this.WS_URL}/volcanic-monitoring`);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.ws.send(JSON.stringify({
                    type: 'subscribe',
                    topics: ['chamber_pressure', 'temperature', 'deformation', 'seismic']
                }));
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealtimeUpdate(data);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
        } catch (error) {
            console.warn('⚠️ Backend connection failed, using demo mode:', error);
            this.useDemoMode = true;
        }
    }
    
    async loadVolcanoData() {
        try {
            if (this.useDemoMode) {
                // Use demo data
                this.createDemoChambers();
            } else {
                // Load from backend
                const response = await fetch(`${this.API_BASE}/volcanic/chambers`);
                const data = await response.json();
                this.createChambersFrom3DModel(data);
            }
        } catch (error) {
            console.error('Error loading volcano data:', error);
            this.createDemoChambers();
        }
    }
    
    createDemoChambers() {
        // Create demo multi-chamber system
        const chamberData = [
            {
                id: 'upper',
                position: { x: 0, y: 0, z: -5 },
                volume: 1e8, // m³
                temperature: 850, // °C
                pressure: 150, // MPa
                composition: { SiO2: 65, H2O: 3, CO2: 0.5 },
                depth: 5000 // m
            },
            {
                id: 'middle',
                position: { x: -5, y: 2, z: -15 },
                volume: 5e8,
                temperature: 950,
                pressure: 300,
                composition: { SiO2: 60, H2O: 4, CO2: 1 },
                depth: 15000
            },
            {
                id: 'lower',
                position: { x: 0, y: 0, z: -30 },
                volume: 1e9,
                temperature: 1100,
                pressure: 500,
                composition: { SiO2: 55, H2O: 2, CO2: 0.3 },
                depth: 30000
            }
        ];
        
        chamberData.forEach(data => {
            this.createChamber(data);
        });
        
        // Create connections between chambers
        this.createConnection('upper', 'middle', 0.3);
        this.createConnection('middle', 'lower', 0.5);
    }
    
    createChamber(data) {
        // Calculate radius from volume (assuming spherical)
        const radius = Math.cbrt((3 * data.volume) / (4 * Math.PI)) / 1000; // Convert to km
        
        // Create chamber geometry
        const geometry = new THREE.SphereGeometry(radius, 32, 32);
        
        // Create material with temperature-based color
        const color = this.getTemperatureColor(data.temperature);
        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.7,
            side: THREE.DoubleSide
        });
        
        // Create mesh
        const chamber = new THREE.Mesh(geometry, material);
        chamber.position.set(data.position.x, data.position.y, data.position.z);
        chamber.castShadow = true;
        chamber.receiveShadow = true;
        
        // Store chamber data
        chamber.userData = data;
        this.chambers.push(chamber);
        this.scene.add(chamber);
        
        // Add label
        this.addChamberLabel(chamber, data);
        
        // Add pressure field visualization
        this.addPressureField(chamber, data);
    }
    
    createConnection(fromId, toId, conductivity) {
        const fromChamber = this.chambers.find(c => c.userData.id === fromId);
        const toChamber = this.chambers.find(c => c.userData.id === toId);
        
        if (!fromChamber || !toChamber) return;
        
        // Create conduit geometry
        const points = [];
        points.push(fromChamber.position);
        
        // Add control points for curved conduit
        const midPoint = new THREE.Vector3().lerpVectors(
            fromChamber.position,
            toChamber.position,
            0.5
        );
        midPoint.x += (Math.random() - 0.5) * 5;
        midPoint.y += (Math.random() - 0.5) * 5;
        points.push(midPoint);
        
        points.push(toChamber.position);
        
        const curve = new THREE.CatmullRomCurve3(points);
        const geometry = new THREE.TubeGeometry(curve, 20, conductivity, 8, false);
        
        const material = new THREE.MeshPhongMaterial({
            color: 0xff6600,
            emissive: 0xff3300,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.6
        });
        
        const connection = new THREE.Mesh(geometry, material);
        connection.userData = {
            from: fromId,
            to: toId,
            conductivity: conductivity,
            flowRate: 0
        };
        
        this.connections.push(connection);
        this.scene.add(connection);
    }
    
    getTemperatureColor(temperature) {
        // Map temperature to color (600°C to 1200°C)
        const normalized = (temperature - 600) / 600;
        const r = Math.min(255, Math.floor(255 * normalized));
        const g = Math.min(100, Math.floor(100 * (1 - normalized)));
        const b = Math.min(50, Math.floor(50 * (1 - normalized)));
        return new THREE.Color(`rgb(${r}, ${g}, ${b})`);
    }
    
    addChamberLabel(chamber, data) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 128;
        
        context.fillStyle = 'rgba(0, 0, 0, 0.7)';
        context.fillRect(0, 0, 256, 128);
        
        context.fillStyle = 'white';
        context.font = '16px Arial';
        context.fillText(`Chamber: ${data.id}`, 10, 25);
        context.fillText(`Depth: ${data.depth}m`, 10, 45);
        context.fillText(`T: ${data.temperature}°C`, 10, 65);
        context.fillText(`P: ${data.pressure} MPa`, 10, 85);
        context.fillText(`SiO₂: ${data.composition.SiO2}%`, 10, 105);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(10, 5, 1);
        sprite.position.copy(chamber.position);
        sprite.position.y += 10;
        
        this.scene.add(sprite);
    }
    
    addPressureField(chamber, data) {
        // Create pressure field visualization
        const fieldGeometry = new THREE.SphereGeometry(
            chamber.geometry.parameters.radius * 2,
            16,
            16
        );
        
        const fieldMaterial = new THREE.ShaderMaterial({
            uniforms: {
                pressure: { value: data.pressure / 1000 },
                time: { value: 0 }
            },
            vertexShader: `
                varying vec3 vPosition;
                uniform float time;
                
                void main() {
                    vPosition = position;
                    vec3 pos = position;
                    pos += normal * sin(time) * 0.1;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform float pressure;
                varying vec3 vPosition;
                
                void main() {
                    float intensity = pressure * 0.5;
                    vec3 color = vec3(intensity, 0.0, 1.0 - intensity);
                    float alpha = 0.2 * (1.0 - length(vPosition) / 20.0);
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide
        });
        
        const field = new THREE.Mesh(fieldGeometry, fieldMaterial);
        field.position.copy(chamber.position);
        this.scene.add(field);
    }
    
    async runSimulation() {
        console.log('🔄 Running multi-chamber simulation...');
        
        const simulationParams = {
            chambers: this.chambers.map(c => ({
                id: c.userData.id,
                volume: c.userData.volume,
                temperature: c.userData.temperature,
                pressure: c.userData.pressure,
                composition: c.userData.composition,
                position: c.position
            })),
            connections: this.connections.map(c => ({
                from: c.userData.from,
                to: c.userData.to,
                conductivity: c.userData.conductivity
            })),
            timeRange: this.parameters.timeRange,
            physics: this.parameters.physics,
            ml: this.parameters.ml
        };
        
        try {
            if (this.useDemoMode) {
                // Generate demo simulation
                this.simulationData = this.generateDemoSimulation(simulationParams);
                this.visualizeSimulation();
            } else {
                // Send to backend
                const response = await fetch(`${this.API_BASE}/volcanic/multi-chamber/simulate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(simulationParams)
                });
                
                const result = await response.json();
                this.simulationData = result.results;
                this.mlPredictions = result.ml_predictions;
                
                this.visualizeSimulation();
                this.displayPredictions();
            }
        } catch (error) {
            console.error('Simulation error:', error);
            // Fallback to demo
            this.simulationData = this.generateDemoSimulation(simulationParams);
            this.visualizeSimulation();
        }
    }
    
    generateDemoSimulation(params) {
        // Generate realistic demo simulation data
        const steps = params.timeRange.end / params.timeRange.step;
        const simulation = {
            timeSteps: [],
            chambers: {},
            flowRates: [],
            eruptionProbability: []
        };
        
        // Initialize chamber states
        params.chambers.forEach(chamber => {
            simulation.chambers[chamber.id] = {
                pressure: [],
                temperature: [],
                gasContent: [],
                crystallinity: []
            };
        });
        
        // Generate time evolution
        for (let i = 0; i < steps; i++) {
            const t = i * params.timeRange.step;
            simulation.timeSteps.push(t);
            
            // Simulate pressure evolution with periodic injection
            params.chambers.forEach(chamber => {
                const basePressure = chamber.pressure;
                const injection = Math.sin(t / 30) * 20 + Math.sin(t / 100) * 50;
                const noise = (Math.random() - 0.5) * 10;
                const pressure = basePressure + injection + noise;
                
                simulation.chambers[chamber.id].pressure.push(pressure);
                
                // Temperature evolution
                const baseTemp = chamber.temperature;
                const cooling = -t * 0.01;
                const heating = injection * 0.5;
                simulation.chambers[chamber.id].temperature.push(baseTemp + cooling + heating);
                
                // Gas evolution
                const gasContent = 3 + Math.sin(t / 50) * 1 + Math.random() * 0.5;
                simulation.chambers[chamber.id].gasContent.push(gasContent);
                
                // Crystallinity
                const crystallinity = Math.min(50, t * 0.1 + Math.random() * 5);
                simulation.chambers[chamber.id].crystallinity.push(crystallinity);
            });
            
            // Flow rates between chambers
            const flowRate = Math.sin(t / 40) * 100 + 50;
            simulation.flowRates.push(flowRate);
            
            // Eruption probability (increases with pressure)
            const avgPressure = params.chambers.reduce((sum, c) => 
                sum + simulation.chambers[c.id].pressure[i], 0) / params.chambers.length;
            const probability = Math.min(1, Math.max(0, (avgPressure - 200) / 300));
            simulation.eruptionProbability.push(probability);
        }
        
        // Add ML predictions
        this.mlPredictions = {
            eruption_probability: simulation.eruptionProbability[simulation.eruptionProbability.length - 1],
            time_to_eruption: this.predictTimeToEruption(simulation),
            expected_vei: this.predictVEI(simulation),
            confidence: 0.75 + Math.random() * 0.2,
            uncertainty_bounds: {
                lower: simulation.eruptionProbability.map(p => Math.max(0, p - 0.2)),
                upper: simulation.eruptionProbability.map(p => Math.min(1, p + 0.2))
            }
        };
        
        return simulation;
    }
    
    predictTimeToEruption(simulation) {
        // Simple prediction based on pressure trend
        const lastPressures = Object.values(simulation.chambers).map(c => 
            c.pressure.slice(-10).reduce((a, b) => a + b, 0) / 10
        );
        const avgPressure = lastPressures.reduce((a, b) => a + b, 0) / lastPressures.length;
        
        if (avgPressure > 400) return Math.random() * 10 + 5; // 5-15 days
        if (avgPressure > 300) return Math.random() * 30 + 20; // 20-50 days
        if (avgPressure > 200) return Math.random() * 100 + 50; // 50-150 days
        return null; // No eruption expected
    }
    
    predictVEI(simulation) {
        // Predict Volcanic Explosivity Index based on chamber volumes and gas content
        const totalVolume = this.chambers.reduce((sum, c) => sum + c.userData.volume, 0);
        const avgGas = Object.values(simulation.chambers).map(c => 
            c.gasContent[c.gasContent.length - 1]
        ).reduce((a, b) => a + b, 0) / this.chambers.length;
        
        if (totalVolume > 1e9 && avgGas > 4) return 5 + Math.floor(Math.random() * 2);
        if (totalVolume > 5e8 && avgGas > 3) return 4 + Math.floor(Math.random() * 2);
        if (totalVolume > 1e8 && avgGas > 2) return 3 + Math.floor(Math.random() * 2);
        return 2 + Math.floor(Math.random() * 2);
    }
    
    visualizeSimulation() {
        // Create time-series plots
        this.createPressurePlot();
        this.createTemperaturePlot();
        this.createFlowRatePlot();
        this.createEruptionProbabilityPlot();
        
        // Animate chamber evolution
        this.animateChamberEvolution();
    }
    
    createPressurePlot() {
        const plotContainer = document.createElement('div');
        plotContainer.id = 'pressure-plot';
        plotContainer.style.position = 'absolute';
        plotContainer.style.top = '10px';
        plotContainer.style.right = '10px';
        plotContainer.style.width = '400px';
        plotContainer.style.height = '200px';
        plotContainer.style.background = 'rgba(0, 0, 0, 0.8)';
        plotContainer.style.borderRadius = '8px';
        plotContainer.style.padding = '10px';
        this.container.appendChild(plotContainer);
        
        const traces = Object.entries(this.simulationData.chambers).map(([id, data]) => ({
            x: this.simulationData.timeSteps,
            y: data.pressure,
            type: 'scatter',
            name: `Chamber ${id}`,
            line: { width: 2 }
        }));
        
        const layout = {
            title: 'Chamber Pressure Evolution',
            xaxis: { title: 'Time (days)', color: '#fff' },
            yaxis: { title: 'Pressure (MPa)', color: '#fff' },
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            plot_bgcolor: 'rgba(0, 0, 0, 0.2)',
            font: { color: '#fff' },
            margin: { t: 30, r: 20, b: 40, l: 50 }
        };
        
        if (window.Plotly) {
            Plotly.newPlot('pressure-plot', traces, layout);
        }
    }
    
    createEruptionProbabilityPlot() {
        const plotContainer = document.createElement('div');
        plotContainer.id = 'eruption-plot';
        plotContainer.style.position = 'absolute';
        plotContainer.style.bottom = '10px';
        plotContainer.style.right = '10px';
        plotContainer.style.width = '400px';
        plotContainer.style.height = '200px';
        plotContainer.style.background = 'rgba(0, 0, 0, 0.8)';
        plotContainer.style.borderRadius = '8px';
        plotContainer.style.padding = '10px';
        this.container.appendChild(plotContainer);
        
        const traces = [
            {
                x: this.simulationData.timeSteps,
                y: this.simulationData.eruptionProbability,
                type: 'scatter',
                name: 'Eruption Probability',
                line: { color: 'red', width: 3 }
            }
        ];
        
        if (this.mlPredictions && this.mlPredictions.uncertainty_bounds) {
            traces.push({
                x: this.simulationData.timeSteps,
                y: this.mlPredictions.uncertainty_bounds.upper,
                type: 'scatter',
                name: 'Upper bound',
                line: { color: 'orange', width: 1, dash: 'dash' }
            });
            traces.push({
                x: this.simulationData.timeSteps,
                y: this.mlPredictions.uncertainty_bounds.lower,
                type: 'scatter',
                name: 'Lower bound',
                line: { color: 'orange', width: 1, dash: 'dash' }
            });
        }
        
        const layout = {
            title: 'ML Eruption Probability Forecast',
            xaxis: { title: 'Time (days)', color: '#fff' },
            yaxis: { title: 'Probability', range: [0, 1], color: '#fff' },
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            plot_bgcolor: 'rgba(0, 0, 0, 0.2)',
            font: { color: '#fff' },
            margin: { t: 30, r: 20, b: 40, l: 50 }
        };
        
        if (window.Plotly) {
            Plotly.newPlot('eruption-plot', traces, layout);
        }
    }
    
    animateChamberEvolution() {
        let currentStep = 0;
        const totalSteps = this.simulationData.timeSteps.length;
        
        const animate = () => {
            if (currentStep < totalSteps) {
                // Update chamber properties based on simulation
                this.chambers.forEach(chamber => {
                    const id = chamber.userData.id;
                    const pressure = this.simulationData.chambers[id].pressure[currentStep];
                    const temperature = this.simulationData.chambers[id].temperature[currentStep];
                    
                    // Update chamber color based on temperature
                    const color = this.getTemperatureColor(temperature);
                    chamber.material.color = color;
                    chamber.material.emissive = color;
                    
                    // Update chamber size based on pressure
                    const scaleFactor = 1 + (pressure - chamber.userData.pressure) / 1000;
                    chamber.scale.setScalar(scaleFactor);
                });
                
                // Update connection flow visualization
                this.connections.forEach((connection, index) => {
                    const flowRate = this.simulationData.flowRates[currentStep];
                    const intensity = Math.abs(flowRate) / 200;
                    connection.material.emissiveIntensity = intensity;
                    connection.material.opacity = 0.3 + intensity * 0.4;
                });
                
                currentStep++;
                setTimeout(animate, 100); // 100ms per step
            }
        };
        
        animate();
    }
    
    displayPredictions() {
        if (!this.mlPredictions) return;
        
        const predictionPanel = document.createElement('div');
        predictionPanel.style.position = 'absolute';
        predictionPanel.style.top = '10px';
        predictionPanel.style.left = '10px';
        predictionPanel.style.width = '300px';
        predictionPanel.style.background = 'rgba(0, 0, 0, 0.9)';
        predictionPanel.style.borderRadius = '8px';
        predictionPanel.style.padding = '15px';
        predictionPanel.style.color = 'white';
        predictionPanel.style.fontFamily = 'monospace';
        
        predictionPanel.innerHTML = `
            <h3 style="margin-top: 0; color: #ff6600;">ML Predictions</h3>
            <div style="margin: 10px 0;">
                <strong>Eruption Probability:</strong>
                <div style="background: #333; border-radius: 4px; height: 20px; margin: 5px 0;">
                    <div style="background: linear-gradient(90deg, green, yellow, orange, red); 
                               width: ${this.mlPredictions.eruption_probability * 100}%; 
                               height: 100%; border-radius: 4px;"></div>
                </div>
                <span>${(this.mlPredictions.eruption_probability * 100).toFixed(1)}%</span>
            </div>
            <div style="margin: 10px 0;">
                <strong>Time to Eruption:</strong> 
                ${this.mlPredictions.time_to_eruption ? 
                    `${this.mlPredictions.time_to_eruption.toFixed(0)} days` : 
                    'No eruption expected'}
            </div>
            <div style="margin: 10px 0;">
                <strong>Expected VEI:</strong> ${this.mlPredictions.expected_vei}
            </div>
            <div style="margin: 10px 0;">
                <strong>Model Confidence:</strong> ${(this.mlPredictions.confidence * 100).toFixed(1)}%
            </div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #666;">
                <strong>Alert Level:</strong> 
                <span style="color: ${this.getAlertColor()}; font-weight: bold;">
                    ${this.getAlertLevel()}
                </span>
            </div>
        `;
        
        this.container.appendChild(predictionPanel);
    }
    
    getAlertLevel() {
        const prob = this.mlPredictions.eruption_probability;
        if (prob > 0.8) return 'CRITICAL';
        if (prob > 0.6) return 'WARNING';
        if (prob > 0.4) return 'WATCH';
        if (prob > 0.2) return 'ADVISORY';
        return 'NORMAL';
    }
    
    getAlertColor() {
        const level = this.getAlertLevel();
        const colors = {
            'CRITICAL': '#ff0000',
            'WARNING': '#ff6600',
            'WATCH': '#ffaa00',
            'ADVISORY': '#ffff00',
            'NORMAL': '#00ff00'
        };
        return colors[level];
    }
    
    handleRealtimeUpdate(data) {
        console.log('📡 Real-time update:', data);
        
        switch(data.type) {
            case 'chamber_pressure':
                this.updateChamberPressure(data.chamber_id, data.pressure);
                break;
            case 'temperature':
                this.updateChamberTemperature(data.chamber_id, data.temperature);
                break;
            case 'deformation':
                this.updateDeformation(data);
                break;
            case 'seismic':
                this.updateSeismicActivity(data);
                break;
            case 'alert':
                this.showAlert(data);
                break;
        }
    }
    
    updateChamberPressure(chamberId, pressure) {
        const chamber = this.chambers.find(c => c.userData.id === chamberId);
        if (chamber) {
            chamber.userData.pressure = pressure;
            // Visual feedback
            const scaleFactor = 1 + (pressure - 200) / 1000;
            new TWEEN.Tween(chamber.scale)
                .to({ x: scaleFactor, y: scaleFactor, z: scaleFactor }, 1000)
                .easing(TWEEN.Easing.Quadratic.Out)
                .start();
        }
    }
    
    initControls() {
        const controlPanel = document.createElement('div');
        controlPanel.style.position = 'absolute';
        controlPanel.style.bottom = '10px';
        controlPanel.style.left = '10px';
        controlPanel.style.background = 'rgba(0, 0, 0, 0.8)';
        controlPanel.style.borderRadius = '8px';
        controlPanel.style.padding = '15px';
        controlPanel.style.color = 'white';
        
        controlPanel.innerHTML = `
            <h4 style="margin-top: 0;">Simulation Controls</h4>
            <button id="run-simulation" style="
                background: #ff6600;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                margin: 5px;
            ">Run Simulation</button>
            <button id="reset-view" style="
                background: #0066ff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                margin: 5px;
            ">Reset View</button>
            <div style="margin-top: 10px;">
                <label>
                    <input type="checkbox" id="show-pressure" checked> Show Pressure Fields
                </label>
            </div>
            <div>
                <label>
                    <input type="checkbox" id="show-flow" checked> Show Flow Paths
                </label>
            </div>
        `;
        
        this.container.appendChild(controlPanel);
        
        // Event listeners
        document.getElementById('run-simulation').addEventListener('click', () => {
            this.runSimulation();
        });
        
        document.getElementById('reset-view').addEventListener('click', () => {
            this.camera.position.set(0, -50, 100);
            this.camera.lookAt(0, 0, 0);
            this.controls.reset();
        });
    }
    
    startMonitoring() {
        // Start continuous monitoring
        setInterval(() => {
            // Update time-dependent uniforms
            this.scene.traverse((child) => {
                if (child.material && child.material.uniforms && child.material.uniforms.time) {
                    child.material.uniforms.time.value += 0.01;
                }
            });
        }, 16); // 60 FPS
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Update TWEEN animations
        if (window.TWEEN) {
            TWEEN.update();
        }
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    resize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

// Export for use in other modules
window.MultiChamberVolcanicSystem = MultiChamberVolcanicSystem;

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('volcanic-system-container');
    if (container) {
        window.volcanicSystem = new MultiChamberVolcanicSystem(container);
    }
});