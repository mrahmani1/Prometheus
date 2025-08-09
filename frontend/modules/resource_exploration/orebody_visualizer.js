/**
 * 3D/4D Ore Body Visualization System
 * Interactive visualization of geological models, mining sequences, and temporal evolution
 */

class OreBodyVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }
        
        // Default options
        this.options = {
            width: options.width || this.container.clientWidth || 800,
            height: options.height || this.container.clientHeight || 600,
            backgroundColor: options.backgroundColor || '#1a1a1a',
            cameraPosition: options.cameraPosition || { x: 100, y: 100, z: 100 },
            enableControls: options.enableControls !== false,
            showAxes: options.showAxes !== false,
            showGrid: options.showGrid !== false,
            timeAnimation: options.timeAnimation !== false,
            ...options
        };
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.raycaster = null;
        this.mouse = null;
        
        // Visualization data
        this.geologicalModel = null;
        this.miningSchedule = null;
        this.processData = null;
        this.currentTimeStep = 0;
        this.timeSteps = [];
        
        // Object groups
        this.drillholeGroup = new THREE.Group();
        this.blockModelGroup = new THREE.Group();
        this.surfaceGroup = new THREE.Group();
        this.miningGroup = new THREE.Group();
        this.alteration = new THREE.Group();
        this.fluidFlowGroup = new THREE.Group();
        
        // Materials
        this.materials = {};
        this.colorScales = {};
        
        // UI elements
        this.timeSlider = null;
        this.layerControls = null;
        this.colorbarContainer = null;
        
        // Animation
        this.animationId = null;
        this.isAnimating = false;
        
        this.initialize();
    }
    
    initialize() {
        this.setupThreeJS();
        this.setupMaterials();
        this.setupUI();
        this.setupEventListeners();
        
        console.log('OreBody Visualizer initialized');
    }
    
    setupThreeJS() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.options.width / this.options.height,
            0.1,
            10000
        );
        this.camera.position.set(
            this.options.cameraPosition.x,
            this.options.cameraPosition.y,
            this.options.cameraPosition.z
        );
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        this.renderer.setSize(this.options.width, this.options.height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        if (this.options.enableControls) {
            if (typeof THREE.OrbitControls !== 'undefined') {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
            }
        }
        
        // Raycaster for interaction
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Lighting
        this.setupLighting();
        
        // Add object groups to scene
        this.scene.add(this.drillholeGroup);
        this.scene.add(this.blockModelGroup);
        this.scene.add(this.surfaceGroup);
        this.scene.add(this.miningGroup);
        this.scene.add(this.alterationGroup);
        this.scene.add(this.fluidFlowGroup);
        
        // Add helpers
        if (this.options.showAxes) {
            const axesHelper = new THREE.AxesHelper(50);
            this.scene.add(axesHelper);
        }
        
        if (this.options.showGrid) {
            const gridHelper = new THREE.GridHelper(1000, 20);
            gridHelper.rotateX(Math.PI / 2);
            this.scene.add(gridHelper);
        }
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(200, 300, 200);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 1000;
        this.scene.add(directionalLight);
        
        // Point lights for underground illumination
        const pointLight1 = new THREE.PointLight(0xffffff, 0.5, 500);
        pointLight1.position.set(0, -100, 0);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0xff8844, 0.3, 300);
        pointLight2.position.set(100, -200, 100);
        this.scene.add(pointLight2);
    }
    
    setupMaterials() {
        this.materials = {
            // Rock types
            ore: new THREE.MeshLambertMaterial({ 
                color: 0xcc6600, 
                transparent: true, 
                opacity: 0.8 
            }),
            waste: new THREE.MeshLambertMaterial({ 
                color: 0x888888, 
                transparent: true, 
                opacity: 0.6 
            }),
            overburden: new THREE.MeshLambertMaterial({ 
                color: 0x8B4513, 
                transparent: true, 
                opacity: 0.7 
            }),
            
            // Alteration zones
            potassic: new THREE.MeshLambertMaterial({ color: 0xFF4500 }),
            phyllic: new THREE.MeshLambertMaterial({ color: 0x32CD32 }),
            argillic: new THREE.MeshLambertMaterial({ color: 0xFFFF00 }),
            propylitic: new THREE.MeshLambertMaterial({ color: 0x00CED1 }),
            
            // Mining materials
            mined: new THREE.MeshLambertMaterial({ 
                color: 0x000000, 
                transparent: true, 
                opacity: 0.3 
            }),
            planned: new THREE.MeshLambertMaterial({ 
                color: 0xffff00, 
                wireframe: true 
            }),
            
            // Surfaces
            geologicalSurface: new THREE.MeshPhongMaterial({
                color: 0x8B4513,
                transparent: true,
                opacity: 0.7,
                side: THREE.DoubleSide
            }),
            
            // Drillholes
            drillhole: new THREE.MeshBasicMaterial({ color: 0x0000ff }),
            drillholeCore: new THREE.MeshBasicMaterial({ color: 0xff0000 })
        };
        
        // Setup color scales
        this.setupColorScales();
    }
    
    setupColorScales() {
        this.colorScales = {
            grade: {
                Cu: { min: 0, max: 5, colors: ['#000080', '#0000ff', '#00ffff', '#ffff00', '#ff8000', '#ff0000'] },
                Au: { min: 0, max: 10, colors: ['#000040', '#000080', '#0080ff', '#ffff00', '#ffd700'] },
                Fe: { min: 0, max: 60, colors: ['#f0f0f0', '#ff8000', '#800000'] },
                Zn: { min: 0, max: 15, colors: ['#ffffff', '#c0c0c0', '#404040'] }
            },
            temperature: {
                min: 273,
                max: 1273,
                colors: ['#000080', '#0000ff', '#00ff00', '#ffff00', '#ff0000', '#ffffff']
            },
            pressure: {
                min: 1e5,
                max: 1e8,
                colors: ['#ffffff', '#ffff00', '#ff8000', '#ff0000', '#800000']
            },
            time: {
                min: 0,
                max: 100,
                colors: ['#000000', '#404040', '#808080', '#c0c0c0', '#ffffff']
            }
        };
    }
    
    setupUI() {
        // Create control panel
        const controlPanel = document.createElement('div');
        controlPanel.className = 'orebody-controls';
        controlPanel.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
        `;
        
        // Time slider
        if (this.options.timeAnimation) {
            const timeControls = document.createElement('div');
            timeControls.innerHTML = `
                <label>Time Step:</label><br>
                <input type="range" id="timeSlider" min="0" max="100" value="0" style="width: 200px;">
                <span id="timeDisplay">0</span><br>
                <button id="playButton">Play</button>
                <button id="pauseButton">Pause</button>
                <button id="resetButton">Reset</button>
            `;
            controlPanel.appendChild(timeControls);
            
            this.timeSlider = controlPanel.querySelector('#timeSlider');
            this.timeDisplay = controlPanel.querySelector('#timeDisplay');
        }
        
        // Layer controls
        const layerControls = document.createElement('div');
        layerControls.innerHTML = `
            <h4>Layers</h4>
            <label><input type="checkbox" id="showDrillholes" checked> Drillholes</label><br>
            <label><input type="checkbox" id="showBlocks" checked> Block Model</label><br>
            <label><input type="checkbox" id="showSurfaces" checked> Geological Surfaces</label><br>
            <label><input type="checkbox" id="showMining" checked> Mining</label><br>
            <label><input type="checkbox" id="showAlteration"> Alteration Zones</label><br>
            <label><input type="checkbox" id="showFluidFlow"> Fluid Flow</label><br>
        `;
        controlPanel.appendChild(layerControls);
        
        // Visualization options
        const vizOptions = document.createElement('div');
        vizOptions.innerHTML = `
            <h4>Visualization</h4>
            <label>Color by:</label><br>
            <select id="colorBySelect">
                <option value="rockType">Rock Type</option>
                <option value="grade_Cu">Cu Grade</option>
                <option value="grade_Au">Au Grade</option>
                <option value="grade_Fe">Fe Grade</option>
                <option value="temperature">Temperature</option>
                <option value="pressure">Pressure</option>
                <option value="alteration">Alteration</option>
            </select><br>
            <label>Opacity:</label><br>
            <input type="range" id="opacitySlider" min="0" max="100" value="80" style="width: 150px;"><br>
            <label>Clipping:</label><br>
            <input type="range" id="clippingSlider" min="0" max="100" value="0" style="width: 150px;">
        `;
        controlPanel.appendChild(vizOptions);
        
        this.container.appendChild(controlPanel);
        
        // Color bar
        this.createColorBar();
        
        // Stats panel
        this.createStatsPanel();
    }
    
    createColorBar() {
        this.colorbarContainer = document.createElement('div');
        this.colorbarContainer.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 20px;
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, blue, cyan, yellow, red);
            border: 1px solid white;
            z-index: 1000;
        `;
        
        const colorbarLabel = document.createElement('div');
        colorbarLabel.style.cssText = `
            position: absolute;
            bottom: -20px;
            left: 0;
            color: white;
            font-size: 10px;
            width: 100%;
        `;
        colorbarLabel.innerHTML = '<span style="float:left">Min</span><span style="float:right">Max</span>';
        
        this.colorbarContainer.appendChild(colorbarLabel);
        this.container.appendChild(this.colorbarContainer);
    }
    
    createStatsPanel() {
        const statsPanel = document.createElement('div');
        statsPanel.id = 'statsPanel';
        statsPanel.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            min-width: 200px;
        `;
        
        this.container.appendChild(statsPanel);
        this.statsPanel = statsPanel;
    }
    
    setupEventListeners() {
        // Time slider
        if (this.timeSlider) {
            this.timeSlider.addEventListener('input', (e) => {
                this.currentTimeStep = parseInt(e.target.value);
                this.updateVisualization();
                this.timeDisplay.textContent = this.currentTimeStep;
            });
            
            // Play/Pause buttons
            document.getElementById('playButton')?.addEventListener('click', () => this.playAnimation());
            document.getElementById('pauseButton')?.addEventListener('click', () => this.pauseAnimation());
            document.getElementById('resetButton')?.addEventListener('click', () => this.resetAnimation());
        }
        
        // Layer controls
        document.getElementById('showDrillholes')?.addEventListener('change', (e) => {
            this.drillholeGroup.visible = e.target.checked;
        });
        document.getElementById('showBlocks')?.addEventListener('change', (e) => {
            this.blockModelGroup.visible = e.target.checked;
        });
        document.getElementById('showSurfaces')?.addEventListener('change', (e) => {
            this.surfaceGroup.visible = e.target.checked;
        });
        document.getElementById('showMining')?.addEventListener('change', (e) => {
            this.miningGroup.visible = e.target.checked;
        });
        document.getElementById('showAlteration')?.addEventListener('change', (e) => {
            this.alterationGroup.visible = e.target.checked;
        });
        document.getElementById('showFluidFlow')?.addEventListener('change', (e) => {
            this.fluidFlowGroup.visible = e.target.checked;
        });
        
        // Visualization options
        document.getElementById('colorBySelect')?.addEventListener('change', (e) => {
            this.updateColorMapping(e.target.value);
        });
        
        document.getElementById('opacitySlider')?.addEventListener('input', (e) => {
            this.updateOpacity(e.target.value / 100);
        });
        
        document.getElementById('clippingSlider')?.addEventListener('input', (e) => {
            this.updateClipping(e.target.value / 100);
        });
        
        // Mouse events for interaction
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            this.onMouseMove(event);
        });
        
        this.renderer.domElement.addEventListener('click', (event) => {
            this.onMouseClick(event);
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
    }
    
    // Data loading methods
    loadGeologicalModel(modelData) {
        this.geologicalModel = modelData;
        this.visualizeGeologicalModel();
        this.updateStats();
    }
    
    loadMiningSchedule(scheduleData) {
        this.miningSchedule = scheduleData;
        this.timeSteps = scheduleData.time_steps || [];
        
        if (this.timeSlider) {
            this.timeSlider.max = Math.max(0, this.timeSteps.length - 1);
        }
        
        this.visualizeMiningSchedule();
        this.updateStats();
    }
    
    loadProcessData(processData) {
        this.processData = processData;
        this.visualizeProcessData();
        this.updateStats();
    }
    
    // Visualization methods
    visualizeGeologicalModel() {
        if (!this.geologicalModel) return;
        
        // Clear existing surfaces
        this.clearGroup(this.surfaceGroup);
        
        // Visualize drillholes
        if (this.geologicalModel.drillholes) {
            this.visualizeDrillholes(this.geologicalModel.drillholes);
        }
        
        // Visualize block model
        if (this.geologicalModel.blocks) {
            this.visualizeBlockModel(this.geologicalModel.blocks);
        }
        
        // Visualize geological surfaces
        if (this.geologicalModel.surfaces) {
            this.visualizeGeologicalSurfaces(this.geologicalModel.surfaces);
        }
        
        // Visualize alteration zones
        if (this.geologicalModel.alteration_zones) {
            this.visualizeAlterationZones(this.geologicalModel.alteration_zones);
        }
    }
    
    visualizeDrillholes(drillholeData) {
        this.clearGroup(this.drillholeGroup);
        
        drillholeData.forEach((hole, index) => {
            const collar = hole.collar || [0, 0, 0];
            const totalDepth = hole.total_depth || 100;
            const azimuth = hole.azimuth || 0;
            const dip = hole.dip || -90;
            
            // Calculate end position
            const azimuthRad = (azimuth * Math.PI) / 180;
            const dipRad = (dip * Math.PI) / 180;
            
            const endX = collar[0] + totalDepth * Math.sin(dipRad) * Math.cos(azimuthRad);
            const endY = collar[1] + totalDepth * Math.sin(dipRad) * Math.sin(azimuthRad);
            const endZ = collar[2] - totalDepth * Math.cos(dipRad);
            
            // Create hole trace
            const geometry = new THREE.CylinderGeometry(0.5, 0.5, totalDepth, 8);
            const mesh = new THREE.Mesh(geometry, this.materials.drillhole.clone());
            
            // Position and orient
            mesh.position.set(
                (collar[0] + endX) / 2,
                (collar[1] + endY) / 2,
                (collar[2] + endZ) / 2
            );
            
            // Rotate to match azimuth and dip
            mesh.lookAt(endX, endY, endZ);
            mesh.rotateX(Math.PI / 2);
            
            mesh.userData = {
                type: 'drillhole',
                hole_id: hole.hole_id,
                collar: collar,
                total_depth: totalDepth
            };
            
            this.drillholeGroup.add(mesh);
            
            // Add samples as colored segments
            if (hole.samples) {
                this.visualizeDrillholeSamples(hole, collar, azimuthRad, dipRad);
            }
        });
    }
    
    visualizeDrillholeSamples(hole, collar, azimuthRad, dipRad) {
        hole.samples.forEach(sample => {
            const fromDepth = sample.from || 0;
            const toDepth = sample.to || fromDepth + 1;
            const length = toDepth - fromDepth;
            const midDepth = (fromDepth + toDepth) / 2;
            
            // Calculate position
            const x = collar[0] + midDepth * Math.sin(-dipRad) * Math.cos(azimuthRad);
            const y = collar[1] + midDepth * Math.sin(-dipRad) * Math.sin(azimuthRad);
            const z = collar[2] + midDepth * Math.cos(-dipRad);
            
            // Create sample segment
            const geometry = new THREE.CylinderGeometry(0.7, 0.7, length, 6);
            
            // Color by grade or lithology
            let material;
            if (sample.Cu && sample.Cu > 0.1) {
                const color = this.getColorForValue(sample.Cu, this.colorScales.grade.Cu);
                material = new THREE.MeshLambertMaterial({ color: color });
            } else {
                material = this.materials.drillholeCore.clone();
            }
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(x, y, z);
            mesh.lookAt(
                x + Math.sin(-dipRad) * Math.cos(azimuthRad),
                y + Math.sin(-dipRad) * Math.sin(azimuthRad),
                z + Math.cos(-dipRad)
            );
            mesh.rotateX(Math.PI / 2);
            
            mesh.userData = {
                type: 'sample',
                hole_id: hole.hole_id,
                from: fromDepth,
                to: toDepth,
                grade: sample
            };
            
            this.drillholeGroup.add(mesh);
        });
    }
    
    visualizeBlockModel(blockData) {
        this.clearGroup(this.blockModelGroup);
        
        blockData.forEach(block => {
            const coords = block.coordinates || [0, 0, 0];
            const size = block.size || [10, 10, 10];
            
            const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
            
            // Material based on rock type or grade
            let material;
            if (block.rock_type === 'ore') {
                if (block.grades && block.grades.Cu) {
                    const color = this.getColorForValue(block.grades.Cu, this.colorScales.grade.Cu);
                    material = new THREE.MeshLambertMaterial({ 
                        color: color, 
                        transparent: true, 
                        opacity: 0.8 
                    });
                } else {
                    material = this.materials.ore.clone();
                }
            } else if (block.rock_type === 'waste') {
                material = this.materials.waste.clone();
            } else {
                material = this.materials.overburden.clone();
            }
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(coords[0], coords[1], coords[2]);
            
            mesh.userData = {
                type: 'block',
                block_id: block.block_id,
                rock_type: block.rock_type,
                grades: block.grades,
                tonnage: block.tonnage
            };
            
            this.blockModelGroup.add(mesh);
        });
    }
    
    visualizeGeologicalSurfaces(surfaceData) {
        Object.keys(surfaceData).forEach(surfaceName => {
            const surface = surfaceData[surfaceName];
            if (surface.points) {
                this.createSurfaceFromPoints(surface.points, surfaceName);
            }
        });
    }
    
    createSurfaceFromPoints(points, surfaceName) {
        // Create triangulated surface from points
        const vertices = [];
        const indices = [];
        
        points.forEach(point => {
            vertices.push(point[0], point[1], point[2]);
        });
        
        // Simple triangulation (in practice, use more sophisticated methods)
        for (let i = 0; i < points.length - 2; i += 3) {
            indices.push(i, i + 1, i + 2);
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setIndex(indices);
        geometry.computeNormals();
        
        const material = this.materials.geologicalSurface.clone();
        const mesh = new THREE.Mesh(geometry, material);
        
        mesh.userData = {
            type: 'surface',
            surface_name: surfaceName
        };
        
        this.surfaceGroup.add(mesh);
    }
    
    visualizeAlterationZones(alterationData) {
        this.clearGroup(this.alterationGroup);
        
        if (alterationData.zones) {
            alterationData.zones.forEach(zone => {
                const center = alterationData.center_location || [0, 0, 0];
                const radius = zone.distance_from_center || 100;
                
                // Create spherical zone representation
                const geometry = new THREE.SphereGeometry(radius, 16, 16);
                const material = this.materials[zone.zone_name] || 
                               new THREE.MeshLambertMaterial({ color: Math.random() * 0xffffff });
                
                material.transparent = true;
                material.opacity = 0.3;
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(center[0], center[1], center[2]);
                
                mesh.userData = {
                    type: 'alteration_zone',
                    zone_name: zone.zone_name,
                    distance: zone.distance_from_center
                };
                
                this.alterationGroup.add(mesh);
            });
        }
    }
    
    visualizeMiningSchedule() {
        if (!this.miningSchedule) return;
        
        this.visualizeMiningSequence();
        this.updateMiningVisualization();
    }
    
    visualizeMiningSequence() {
        // This will show the planned mining sequence
        if (this.miningSchedule.mining_schedule) {
            this.miningSchedule.mining_schedule.forEach((yearData, yearIndex) => {
                yearData.blocks.forEach(blockId => {
                    // Find the corresponding block in the model
                    const block = this.findBlockById(blockId);
                    if (block) {
                        this.addMiningIndicator(block, yearIndex);
                    }
                });
            });
        }
    }
    
    updateMiningVisualization() {
        // Update visualization based on current time step
        this.miningGroup.children.forEach(child => {
            if (child.userData.miningYear <= this.currentTimeStep) {
                child.visible = true;
                child.material = this.materials.mined;
            } else {
                child.visible = true;
                child.material = this.materials.planned;
            }
        });
    }
    
    visualizeProcessData() {
        if (!this.processData) return;
        
        // Visualize fluid flow if available
        if (this.processData.flow_evolution) {
            this.visualizeFluidFlow();
        }
        
        // Visualize temperature evolution
        if (this.processData.thermal_evolution) {
            this.visualizeThermalEvolution();
        }
        
        // Visualize precipitation
        if (this.processData.precipitation_evolution) {
            this.visualizePrecipitation();
        }
    }
    
    visualizeFluidFlow() {
        this.clearGroup(this.fluidFlowGroup);
        
        const flowData = this.processData.flow_evolution[this.currentTimeStep];
        if (!flowData) return;
        
        // Create flow vectors (simplified visualization)
        if (flowData.velocity_field) {
            this.createFlowVectors(flowData.velocity_field);
        }
    }
    
    createFlowVectors(velocityField) {
        // Create arrow helpers to show flow direction
        const geometry = new THREE.ConeGeometry(1, 5, 8);
        
        velocityField.forEach(vector => {
            const position = vector.position || [0, 0, 0];
            const velocity = vector.velocity || [0, 0, 0];
            const magnitude = Math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2);
            
            if (magnitude > 0.001) {  // Only show significant flow
                const material = new THREE.MeshLambertMaterial({ 
                    color: this.getColorForValue(magnitude, this.colorScales.pressure)
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(position[0], position[1], position[2]);
                
                // Orient arrow in flow direction
                const direction = new THREE.Vector3(velocity[0], velocity[1], velocity[2]).normalize();
                mesh.lookAt(
                    position[0] + direction.x,
                    position[1] + direction.y,
                    position[2] + direction.z
                );
                mesh.rotateX(Math.PI / 2);
                
                mesh.userData = {
                    type: 'flow_vector',
                    velocity: velocity,
                    magnitude: magnitude
                };
                
                this.fluidFlowGroup.add(mesh);
            }
        });
    }
    
    // Utility methods
    getColorForValue(value, colorScale) {
        const normalized = Math.max(0, Math.min(1, (value - colorScale.min) / (colorScale.max - colorScale.min)));
        const colorIndex = Math.floor(normalized * (colorScale.colors.length - 1));
        return new THREE.Color(colorScale.colors[colorIndex]);
    }
    
    clearGroup(group) {
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
    }
    
    findBlockById(blockId) {
        let foundBlock = null;
        this.blockModelGroup.traverse(child => {
            if (child.userData && child.userData.block_id === blockId) {
                foundBlock = child;
            }
        });
        return foundBlock;
    }
    
    addMiningIndicator(block, yearIndex) {
        const indicator = block.clone();
        indicator.material = yearIndex <= this.currentTimeStep ? 
                           this.materials.mined : this.materials.planned;
        indicator.userData.miningYear = yearIndex;
        this.miningGroup.add(indicator);
    }
    
    updateColorMapping(colorBy) {
        // Update materials based on selected color mapping
        this.blockModelGroup.children.forEach(child => {
            if (child.userData && child.userData.type === 'block') {
                let newColor;
                
                switch (colorBy) {
                    case 'grade_Cu':
                        if (child.userData.grades && child.userData.grades.Cu) {
                            newColor = this.getColorForValue(
                                child.userData.grades.Cu, 
                                this.colorScales.grade.Cu
                            );
                        }
                        break;
                    case 'grade_Au':
                        if (child.userData.grades && child.userData.grades.Au) {
                            newColor = this.getColorForValue(
                                child.userData.grades.Au, 
                                this.colorScales.grade.Au
                            );
                        }
                        break;
                    case 'rockType':
                    default:
                        // Keep original rock type colors
                        return;
                }
                
                if (newColor) {
                    child.material.color = newColor;
                }
            }
        });
        
        this.updateColorBar(colorBy);
    }
    
    updateColorBar(colorBy) {
        let gradient;
        switch (colorBy) {
            case 'grade_Cu':
                gradient = 'linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff8000, #ff0000)';
                break;
            case 'grade_Au':
                gradient = 'linear-gradient(to right, #000040, #000080, #0080ff, #ffff00, #ffd700)';
                break;
            case 'temperature':
                gradient = 'linear-gradient(to right, #000080, #0000ff, #00ff00, #ffff00, #ff0000, #ffffff)';
                break;
            default:
                gradient = 'linear-gradient(to right, blue, cyan, yellow, red)';
        }
        
        this.colorbarContainer.style.background = gradient;
    }
    
    updateOpacity(opacity) {
        [this.blockModelGroup, this.surfaceGroup, this.alterationGroup].forEach(group => {
            group.children.forEach(child => {
                if (child.material && child.material.transparent !== undefined) {
                    child.material.opacity = opacity;
                }
            });
        });
    }
    
    updateClipping(clipLevel) {
        const clipHeight = clipLevel * 200 - 100;  // Assume model height ~200
        
        [this.blockModelGroup, this.surfaceGroup].forEach(group => {
            group.children.forEach(child => {
                child.visible = child.position.z >= clipHeight;
            });
        });
    }
    
    updateVisualization() {
        if (this.miningSchedule) {
            this.updateMiningVisualization();
        }
        
        if (this.processData) {
            this.visualizeFluidFlow();
        }
        
        this.updateStats();
    }
    
    updateStats() {
        if (!this.statsPanel) return;
        
        let stats = '<h4>Statistics</h4>';
        
        if (this.geologicalModel) {
            const blockCount = this.blockModelGroup.children.length;
            const drillholeCount = this.drillholeGroup.children.filter(child => 
                child.userData.type === 'drillhole').length;
            
            stats += `Blocks: ${blockCount}<br>`;
            stats += `Drillholes: ${drillholeCount}<br>`;
        }
        
        if (this.miningSchedule) {
            stats += `<br><h5>Mining (Year ${this.currentTimeStep})</h5>`;
            const yearData = this.miningSchedule.mining_schedule && 
                           this.miningSchedule.mining_schedule[this.currentTimeStep];
            
            if (yearData) {
                stats += `Blocks: ${yearData.blocks.length}<br>`;
                stats += `Tonnage: ${(yearData.tonnage / 1000).toFixed(0)}kt<br>`;
            }
        }
        
        stats += `<br>Time Step: ${this.currentTimeStep}`;
        
        this.statsPanel.innerHTML = stats;
    }
    
    // Animation methods
    playAnimation() {
        this.isAnimating = true;
        this.animate();
    }
    
    pauseAnimation() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    resetAnimation() {
        this.pauseAnimation();
        this.currentTimeStep = 0;
        if (this.timeSlider) {
            this.timeSlider.value = 0;
            this.timeDisplay.textContent = '0';
        }
        this.updateVisualization();
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        // Auto-advance time step
        if (this.currentTimeStep < this.timeSteps.length - 1) {
            this.currentTimeStep++;
            if (this.timeSlider) {
                this.timeSlider.value = this.currentTimeStep;
                this.timeDisplay.textContent = this.currentTimeStep;
            }
            this.updateVisualization();
        } else {
            this.pauseAnimation();
        }
        
        setTimeout(() => {
            if (this.isAnimating) {
                this.animate();
            }
        }, 500);  // 0.5 second per step
    }
    
    // Event handlers
    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Raycasting for hover effects
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const allObjects = [];
        this.scene.traverse(child => {
            if (child.isMesh && child.visible) {
                allObjects.push(child);
            }
        });
        
        const intersects = this.raycaster.intersectObjects(allObjects);
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            this.showTooltip(event, object.userData);
        } else {
            this.hideTooltip();
        }
    }
    
    onMouseClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const allObjects = [];
        this.scene.traverse(child => {
            if (child.isMesh && child.visible) {
                allObjects.push(child);
            }
        });
        
        const intersects = this.raycaster.intersectObjects(allObjects);
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            this.onObjectClick(object);
        }
    }
    
    onObjectClick(object) {
        console.log('Clicked object:', object.userData);
        
        // Highlight selected object
        this.clearSelection();
        object.material = object.material.clone();
        object.material.emissive = new THREE.Color(0x444444);
        object.userData.selected = true;
        
        // Show detailed info
        this.showDetailedInfo(object.userData);
    }
    
    clearSelection() {
        this.scene.traverse(child => {
            if (child.userData && child.userData.selected) {
                child.material.emissive = new THREE.Color(0x000000);
                child.userData.selected = false;
            }
        });
    }
    
    showTooltip(event, userData) {
        let tooltip = document.getElementById('tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'tooltip';
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 5px;
                border-radius: 3px;
                font-size: 12px;
                pointer-events: none;
                z-index: 2000;
            `;
            document.body.appendChild(tooltip);
        }
        
        let content = `Type: ${userData.type || 'Unknown'}`;
        if (userData.block_id) content += `<br>ID: ${userData.block_id}`;
        if (userData.grades) {
            content += '<br>Grades:';
            Object.entries(userData.grades).forEach(([element, grade]) => {
                content += `<br>&nbsp;&nbsp;${element}: ${grade.toFixed(2)}%`;
            });
        }
        
        tooltip.innerHTML = content;
        tooltip.style.left = event.clientX + 10 + 'px';
        tooltip.style.top = event.clientY - 10 + 'px';
        tooltip.style.display = 'block';
    }
    
    hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    showDetailedInfo(userData) {
        let infoPanel = document.getElementById('detailInfo');
        if (!infoPanel) {
            infoPanel = document.createElement('div');
            infoPanel.id = 'detailInfo';
            infoPanel.style.cssText = `
                position: absolute;
                bottom: 10px;
                right: 10px;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 15px;
                border-radius: 5px;
                font-size: 12px;
                max-width: 250px;
                z-index: 1000;
            `;
            this.container.appendChild(infoPanel);
        }
        
        let content = `<h4>${userData.type || 'Object'} Details</h4>`;
        
        Object.entries(userData).forEach(([key, value]) => {
            if (key !== 'type' && key !== 'selected') {
                if (typeof value === 'object' && value !== null) {
                    content += `<strong>${key}:</strong><br>`;
                    Object.entries(value).forEach(([subKey, subValue]) => {
                        content += `&nbsp;&nbsp;${subKey}: ${subValue}<br>`;
                    });
                } else {
                    content += `<strong>${key}:</strong> ${value}<br>`;
                }
            }
        });
        
        infoPanel.innerHTML = content;
        infoPanel.style.display = 'block';
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    // Render loop
    render() {
        requestAnimationFrame(() => this.render());
        
        if (this.controls) {
            this.controls.update();
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    // Public methods for integration
    start() {
        this.render();
    }
    
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Clean up Three.js resources
        this.scene.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
        
        this.renderer.dispose();
        
        // Remove DOM elements
        while (this.container.firstChild) {
            this.container.removeChild(this.container.firstChild);
        }
    }
    
    // Export methods
    exportImage(filename = 'orebody_visualization.png') {
        const dataURL = this.renderer.domElement.toDataURL();
        const link = document.createElement('a');
        link.download = filename;
        link.href = dataURL;
        link.click();
    }
    
    exportData() {
        return {
            geologicalModel: this.geologicalModel,
            miningSchedule: this.miningSchedule,
            processData: this.processData,
            currentTimeStep: this.currentTimeStep,
            cameraPosition: this.camera.position,
            cameraTarget: this.controls ? this.controls.target : null
        };
    }
}

// Factory function for easy instantiation
window.createOreBodyVisualizer = function(containerId, options) {
    return new OreBodyVisualizer(containerId, options);
};

console.log('OreBody Visualizer loaded successfully');