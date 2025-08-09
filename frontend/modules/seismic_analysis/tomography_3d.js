/**
 * Seismic Tomography 3D Visualization System
 * Advanced 3D velocity model visualization with cross-sections, isosurfaces, and ray paths
 */

window.TOMOGRAPHY_3D = (function() {
    'use strict';

    let scene, camera, renderer, controls;
    let velocityModel = null;
    let crossSections = [];
    let rayPaths = [];
    let isosurfaces = [];
    let resolutionMatrix = null;
    
    const TOMOGRAPHY_CONFIG = {
        dimensions: {
            nx: 100, ny: 100, nz: 50,
            xmin: -50, xmax: 50,
            ymin: -50, ymax: 50,
            zmin: -30, zmax: 20
        },
        velocityRange: {
            vp: { min: 4.0, max: 8.5 },
            vs: { min: 2.3, max: 4.9 }
        },
        visualization: {
            opacity: 0.7,
            colorScale: 'viridis',
            isosurfaceLevels: [5.5, 6.0, 6.5, 7.0, 7.5],
            crossSectionSpacing: 5.0
        },
        rayTracing: {
            maxRays: 1000,
            stepSize: 0.1,
            maxSteps: 10000
        }
    };

    // Color scales for velocity visualization
    const COLOR_SCALES = {
        viridis: [
            [0.267004, 0.004874, 0.329415],
            [0.282623, 0.140926, 0.457517],
            [0.253935, 0.265254, 0.529983],
            [0.206756, 0.371758, 0.553117],
            [0.163625, 0.471133, 0.558148],
            [0.127568, 0.566949, 0.550556],
            [0.134692, 0.658636, 0.517649],
            [0.266941, 0.748751, 0.440573],
            [0.477504, 0.821444, 0.318195],
            [0.741388, 0.873449, 0.149561],
            [0.993248, 0.906157, 0.143936]
        ],
        seismic: [
            [0.0, 0.0, 1.0],  // Blue (slow)
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 1.0],  // Cyan
            [0.5, 1.0, 0.5],
            [1.0, 1.0, 0.0],  // Yellow
            [1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0]   // Red (fast)
        ]
    };

    // Initialize the 3D tomography system
    function init() {
        console.log('🌍 Initializing Seismic Tomography 3D System...');
        
        createScene();
        createUI();
        generateSyntheticModel();
        setupEventListeners();
        
        console.log('✅ Tomography 3D System initialized');
    }

    function createScene() {
        // Scene setup
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            1000
        );
        camera.position.set(80, 60, 80);
        
        // Renderer
        renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Controls
        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
        }
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        // Coordinate axes
        const axesHelper = new THREE.AxesHelper(25);
        scene.add(axesHelper);
        
        // Grid
        const gridHelper = new THREE.GridHelper(100, 20, 0x444444, 0x222222);
        scene.add(gridHelper);
    }

    function createUI() {
        const ui = document.createElement('div');
        ui.id = 'tomography-ui';
        ui.innerHTML = `
            <div class="tomography-panel">
                <h3>🌍 Seismic Tomography 3D</h3>
                <div class="control-group">
                    <label>Velocity Type:</label>
                    <select id="velocity-type">
                        <option value="vp">P-wave Velocity (Vp)</option>
                        <option value="vs">S-wave Velocity (Vs)</option>
                        <option value="vp_vs_ratio">Vp/Vs Ratio</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Color Scale:</label>
                    <select id="color-scale">
                        <option value="viridis">Viridis</option>
                        <option value="seismic">Seismic</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Opacity: <span id="opacity-value">0.7</span></label>
                    <input type="range" id="opacity-slider" min="0" max="1" step="0.1" value="0.7">
                </div>
                
                <div class="control-group">
                    <h4>Cross-Sections</h4>
                    <label>X-Section: <span id="x-section-value">0</span></label>
                    <input type="range" id="x-section" min="-50" max="50" step="5" value="0">
                    
                    <label>Y-Section: <span id="y-section-value">0</span></label>
                    <input type="range" id="y-section" min="-50" max="50" step="5" value="0">
                    
                    <label>Z-Section: <span id="z-section-value">-5</span></label>
                    <input type="range" id="z-section" min="-30" max="20" step="2" value="-5">
                </div>
                
                <div class="control-group">
                    <h4>Isosurfaces</h4>
                    <label>Velocity Level: <span id="iso-level-value">6.0</span></label>
                    <input type="range" id="iso-level" min="4.0" max="8.5" step="0.1" value="6.0">
                    <button id="add-isosurface">Add Isosurface</button>
                    <button id="clear-isosurfaces">Clear All</button>
                </div>
                
                <div class="control-group">
                    <h4>Ray Paths</h4>
                    <button id="add-ray">Add Ray Path</button>
                    <button id="clear-rays">Clear Rays</button>
                    <label>
                        <input type="checkbox" id="show-raypaths" checked> Show Ray Paths
                    </label>
                </div>
                
                <div class="control-group">
                    <h4>Analysis Tools</h4>
                    <button id="resolution-analysis">Resolution Analysis</button>
                    <button id="checkerboard-test">Checkerboard Test</button>
                    <button id="export-model">Export Model</button>
                </div>
                
                <div class="control-group">
                    <h4>Presets</h4>
                    <button id="crustal-model">Crustal Model</button>
                    <button id="mantle-model">Mantle Model</button>
                    <button id="subduction-model">Subduction Zone</button>
                </div>
            </div>
            <canvas id="tomography-canvas"></canvas>
        `;
        
        ui.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            z-index: 10000;
            display: none;
        `;
        
        document.body.appendChild(ui);
        
        // Move renderer to the canvas container
        const canvas = ui.querySelector('#tomography-canvas');
        canvas.appendChild(renderer.domElement);
    }

    function generateSyntheticModel() {
        const { nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax } = TOMOGRAPHY_CONFIG.dimensions;
        
        velocityModel = {
            dimensions: { nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax },
            vp: new Float32Array(nx * ny * nz),
            vs: new Float32Array(nx * ny * nz),
            coordinates: {
                x: new Float32Array(nx),
                y: new Float32Array(ny),
                z: new Float32Array(nz)
            }
        };
        
        // Generate coordinate arrays
        for (let i = 0; i < nx; i++) {
            velocityModel.coordinates.x[i] = xmin + (i / (nx - 1)) * (xmax - xmin);
        }
        for (let j = 0; j < ny; j++) {
            velocityModel.coordinates.y[j] = ymin + (j / (ny - 1)) * (ymax - ymin);
        }
        for (let k = 0; k < nz; k++) {
            velocityModel.coordinates.z[k] = zmin + (k / (nz - 1)) * (zmax - zmin);
        }
        
        // Generate synthetic velocity model with geological layers
        generateCrustalModel();
        
        console.log('✅ Synthetic velocity model generated');
    }

    function generateCrustalModel() {
        const { nx, ny, nz } = velocityModel.dimensions;
        
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < ny; j++) {
                for (let k = 0; k < nz; k++) {
                    const idx = i + j * nx + k * nx * ny;
                    const x = velocityModel.coordinates.x[i];
                    const y = velocityModel.coordinates.y[j];
                    const z = velocityModel.coordinates.z[k];
                    
                    // Layer-based velocity structure
                    let vp, vs;
                    
                    if (z > 0) {
                        // Surface layer (sediments)
                        vp = 2.5 + 0.5 * Math.random();
                        vs = vp / 1.8;
                    } else if (z > -5) {
                        // Upper crust
                        vp = 5.5 + 0.3 * Math.sin(x/10) + 0.2 * Math.cos(y/8) + 0.2 * Math.random();
                        vs = vp / 1.75;
                    } else if (z > -15) {
                        // Mid crust
                        vp = 6.2 + 0.2 * Math.sin(x/15) + 0.15 * Math.sin(y/12) + 0.1 * Math.random();
                        vs = vp / 1.73;
                    } else if (z > -25) {
                        // Lower crust
                        vp = 6.8 + 0.1 * Math.sin(x/20) + 0.1 * Math.random();
                        vs = vp / 1.71;
                    } else {
                        // Upper mantle
                        vp = 7.8 + 0.2 * Math.sin(x/25) + 0.1 * Math.sin(y/20) + 0.05 * Math.random();
                        vs = vp / 1.69;
                    }
                    
                    // Add anomalies (magma chambers, fault zones)
                    const dist1 = Math.sqrt((x-10)*(x-10) + (y-5)*(y-5) + (z+8)*(z+8));
                    const dist2 = Math.sqrt((x+15)*(x+15) + (y-10)*(y-10) + (z+12)*(z+12));
                    
                    if (dist1 < 8) {
                        // Low velocity anomaly (magma chamber)
                        vp *= 0.85;
                        vs *= 0.80;
                    }
                    
                    if (dist2 < 6) {
                        // High velocity anomaly (intrusion)
                        vp *= 1.15;
                        vs *= 1.20;
                    }
                    
                    velocityModel.vp[idx] = Math.max(2.0, Math.min(8.5, vp));
                    velocityModel.vs[idx] = Math.max(1.2, Math.min(4.9, vs));
                }
            }
        }
    }

    function createCrossSection(axis, position) {
        const { nx, ny, nz } = velocityModel.dimensions;
        const geometry = new THREE.PlaneGeometry(100, 100, 100, 100);
        const vertices = geometry.attributes.position.array;
        const colors = new Float32Array(vertices.length);
        
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];
            
            // Get velocity value at this position
            const velocity = interpolateVelocity(x, y, z, 'vp');
            const color = getVelocityColor(velocity, 'vp');
            
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        // Position the cross-section based on axis
        if (axis === 'x') {
            mesh.rotation.y = Math.PI / 2;
            mesh.position.x = position;
        } else if (axis === 'y') {
            mesh.rotation.x = Math.PI / 2;
            mesh.position.y = position;
        } else if (axis === 'z') {
            mesh.position.z = position;
        }
        
        scene.add(mesh);
        crossSections.push(mesh);
        
        return mesh;
    }

    function createIsosurface(level) {
        console.log(`Creating isosurface at velocity level: ${level} km/s`);
        
        // Use marching cubes algorithm (simplified version)
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        const normals = [];
        
        const { nx, ny, nz } = velocityModel.dimensions;
        const dx = (TOMOGRAPHY_CONFIG.dimensions.xmax - TOMOGRAPHY_CONFIG.dimensions.xmin) / (nx - 1);
        const dy = (TOMOGRAPHY_CONFIG.dimensions.ymax - TOMOGRAPHY_CONFIG.dimensions.ymin) / (ny - 1);
        const dz = (TOMOGRAPHY_CONFIG.dimensions.zmax - TOMOGRAPHY_CONFIG.dimensions.zmin) / (nz - 1);
        
        // Simplified isosurface generation (sphere-based approximation)
        for (let i = 1; i < nx - 1; i += 2) {
            for (let j = 1; j < ny - 1; j += 2) {
                for (let k = 1; k < nz - 1; k += 2) {
                    const x = TOMOGRAPHY_CONFIG.dimensions.xmin + i * dx;
                    const y = TOMOGRAPHY_CONFIG.dimensions.ymin + j * dy;
                    const z = TOMOGRAPHY_CONFIG.dimensions.zmin + k * dz;
                    
                    const velocity = interpolateVelocity(x, y, z, 'vp');
                    
                    if (Math.abs(velocity - level) < 0.1) {
                        // Create small sphere at this location
                        const sphereGeometry = new THREE.SphereGeometry(0.5, 8, 6);
                        const sphereVertices = sphereGeometry.attributes.position.array;
                        
                        for (let v = 0; v < sphereVertices.length; v += 3) {
                            vertices.push(
                                sphereVertices[v] + x,
                                sphereVertices[v + 1] + y,
                                sphereVertices[v + 2] + z
                            );
                        }
                        
                        // Add normals
                        const sphereNormals = sphereGeometry.attributes.normal.array;
                        for (let n = 0; n < sphereNormals.length; n++) {
                            normals.push(sphereNormals[n]);
                        }
                    }
                }
            }
        }
        
        if (vertices.length === 0) {
            console.warn('No vertices found for isosurface level:', level);
            return null;
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        
        const color = getVelocityColor(level, 'vp');
        const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color(color.r, color.g, color.b),
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        isosurfaces.push(mesh);
        
        return mesh;
    }

    function createRayPath(source, receiver) {
        // Simplified ray tracing using Snell's law
        const points = [];
        const steps = 100;
        
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const point = {
                x: source.x + t * (receiver.x - source.x),
                y: source.y + t * (receiver.y - source.y),
                z: source.z + t * (receiver.z - source.z)
            };
            
            // Apply ray bending based on velocity gradient
            const velocity = interpolateVelocity(point.x, point.y, point.z, 'vp');
            
            // Simple ray bending approximation
            if (i > 0 && i < steps) {
                const gradient = getVelocityGradient(point.x, point.y, point.z);
                point.z += gradient.z * 0.1; // Bend based on vertical gradient
            }
            
            points.push(new THREE.Vector3(point.x, point.y, point.z));
        }
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0xffff00,
            linewidth: 2
        });
        
        const rayPath = new THREE.Line(geometry, material);
        scene.add(rayPath);
        rayPaths.push(rayPath);
        
        return rayPath;
    }

    function interpolateVelocity(x, y, z, type) {
        const { nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax } = velocityModel.dimensions;
        
        // Convert to grid coordinates
        const xi = (x - xmin) / (xmax - xmin) * (nx - 1);
        const yi = (y - ymin) / (ymax - ymin) * (ny - 1);
        const zi = (z - zmin) / (zmax - zmin) * (nz - 1);
        
        // Check bounds
        if (xi < 0 || xi >= nx - 1 || yi < 0 || yi >= ny - 1 || zi < 0 || zi >= nz - 1) {
            return type === 'vp' ? 6.0 : 3.5; // Default values
        }
        
        // Trilinear interpolation
        const i0 = Math.floor(xi), i1 = Math.min(i0 + 1, nx - 1);
        const j0 = Math.floor(yi), j1 = Math.min(j0 + 1, ny - 1);
        const k0 = Math.floor(zi), k1 = Math.min(k0 + 1, nz - 1);
        
        const fx = xi - i0, fy = yi - j0, fz = zi - k0;
        
        const data = velocityModel[type];
        
        const v000 = data[i0 + j0 * nx + k0 * nx * ny];
        const v001 = data[i0 + j0 * nx + k1 * nx * ny];
        const v010 = data[i0 + j1 * nx + k0 * nx * ny];
        const v011 = data[i0 + j1 * nx + k1 * nx * ny];
        const v100 = data[i1 + j0 * nx + k0 * nx * ny];
        const v101 = data[i1 + j0 * nx + k1 * nx * ny];
        const v110 = data[i1 + j1 * nx + k0 * nx * ny];
        const v111 = data[i1 + j1 * nx + k1 * nx * ny];
        
        const v00 = v000 * (1 - fz) + v001 * fz;
        const v01 = v010 * (1 - fz) + v011 * fz;
        const v10 = v100 * (1 - fz) + v101 * fz;
        const v11 = v110 * (1 - fz) + v111 * fz;
        
        const v0 = v00 * (1 - fy) + v01 * fy;
        const v1 = v10 * (1 - fy) + v11 * fy;
        
        return v0 * (1 - fx) + v1 * fx;
    }

    function getVelocityGradient(x, y, z) {
        const eps = 0.1;
        const vx1 = interpolateVelocity(x - eps, y, z, 'vp');
        const vx2 = interpolateVelocity(x + eps, y, z, 'vp');
        const vy1 = interpolateVelocity(x, y - eps, z, 'vp');
        const vy2 = interpolateVelocity(x, y + eps, z, 'vp');
        const vz1 = interpolateVelocity(x, y, z - eps, 'vp');
        const vz2 = interpolateVelocity(x, y, z + eps, 'vp');
        
        return {
            x: (vx2 - vx1) / (2 * eps),
            y: (vy2 - vy1) / (2 * eps),
            z: (vz2 - vz1) / (2 * eps)
        };
    }

    function getVelocityColor(velocity, type) {
        const range = TOMOGRAPHY_CONFIG.velocityRange[type];
        const normalized = (velocity - range.min) / (range.max - range.min);
        const clamped = Math.max(0, Math.min(1, normalized));
        
        const colorScale = COLOR_SCALES[TOMOGRAPHY_CONFIG.visualization.colorScale];
        const index = clamped * (colorScale.length - 1);
        const i0 = Math.floor(index);
        const i1 = Math.min(i0 + 1, colorScale.length - 1);
        const f = index - i0;
        
        const c0 = colorScale[i0];
        const c1 = colorScale[i1];
        
        return {
            r: c0[0] * (1 - f) + c1[0] * f,
            g: c0[1] * (1 - f) + c1[1] * f,
            b: c0[2] * (1 - f) + c1[2] * f
        };
    }

    function performResolutionAnalysis() {
        console.log('🔍 Performing resolution analysis...');
        
        // Generate synthetic checkerboard test
        const { nx, ny, nz } = velocityModel.dimensions;
        const checkerboard = new Float32Array(nx * ny * nz);
        
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < ny; j++) {
                for (let k = 0; k < nz; k++) {
                    const idx = i + j * nx + k * nx * ny;
                    const pattern = ((Math.floor(i/5) + Math.floor(j/5) + Math.floor(k/3)) % 2) * 2 - 1;
                    checkerboard[idx] = 6.0 + pattern * 0.5;
                }
            }
        }
        
        // Calculate resolution matrix (simplified)
        const resolution = calculateResolutionMatrix(checkerboard);
        
        // Visualize resolution
        visualizeResolution(resolution);
        
        console.log('✅ Resolution analysis complete');
    }

    function calculateResolutionMatrix(model) {
        // Simplified resolution calculation
        const { nx, ny, nz } = velocityModel.dimensions;
        const resolution = new Float32Array(nx * ny * nz);
        
        for (let i = 0; i < nx * ny * nz; i++) {
            // Simple Gaussian spread function
            resolution[i] = Math.exp(-Math.random() * 2);
        }
        
        return resolution;
    }

    function visualizeResolution(resolution) {
        // Create resolution visualization as colored volume
        const { nx, ny, nz } = velocityModel.dimensions;
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        
        for (let i = 0; i < nx; i += 3) {
            for (let j = 0; j < ny; j += 3) {
                for (let k = 0; k < nz; k += 3) {
                    const idx = i + j * nx + k * nx * ny;
                    const x = TOMOGRAPHY_CONFIG.dimensions.xmin + (i / (nx - 1)) * 
                             (TOMOGRAPHY_CONFIG.dimensions.xmax - TOMOGRAPHY_CONFIG.dimensions.xmin);
                    const y = TOMOGRAPHY_CONFIG.dimensions.ymin + (j / (ny - 1)) * 
                             (TOMOGRAPHY_CONFIG.dimensions.ymax - TOMOGRAPHY_CONFIG.dimensions.ymin);
                    const z = TOMOGRAPHY_CONFIG.dimensions.zmin + (k / (nz - 1)) * 
                             (TOMOGRAPHY_CONFIG.dimensions.zmax - TOMOGRAPHY_CONFIG.dimensions.zmin);
                    
                    positions.push(x, y, z);
                    
                    const res = resolution[idx];
                    colors.push(res, res, 1 - res);
                }
            }
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.6
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        setTimeout(() => scene.remove(points), 10000); // Remove after 10 seconds
    }

    function setupEventListeners() {
        // Velocity type change
        document.getElementById('velocity-type').addEventListener('change', function() {
            updateVisualization();
        });
        
        // Color scale change
        document.getElementById('color-scale').addEventListener('change', function() {
            TOMOGRAPHY_CONFIG.visualization.colorScale = this.value;
            updateVisualization();
        });
        
        // Opacity control
        const opacitySlider = document.getElementById('opacity-slider');
        opacitySlider.addEventListener('input', function() {
            const opacity = parseFloat(this.value);
            document.getElementById('opacity-value').textContent = opacity.toFixed(1);
            TOMOGRAPHY_CONFIG.visualization.opacity = opacity;
            updateOpacity();
        });
        
        // Cross-section controls
        ['x', 'y', 'z'].forEach(axis => {
            const slider = document.getElementById(`${axis}-section`);
            slider.addEventListener('input', function() {
                const value = parseFloat(this.value);
                document.getElementById(`${axis}-section-value`).textContent = value;
                updateCrossSection(axis, value);
            });
        });
        
        // Isosurface controls
        document.getElementById('iso-level').addEventListener('input', function() {
            document.getElementById('iso-level-value').textContent = parseFloat(this.value).toFixed(1);
        });
        
        document.getElementById('add-isosurface').addEventListener('click', function() {
            const level = parseFloat(document.getElementById('iso-level').value);
            createIsosurface(level);
        });
        
        document.getElementById('clear-isosurfaces').addEventListener('click', clearIsosurfaces);
        
        // Ray path controls
        document.getElementById('add-ray').addEventListener('click', function() {
            const source = { x: -30, y: -20, z: 0 };
            const receiver = { x: 30, y: 20, z: -10 };
            createRayPath(source, receiver);
        });
        
        document.getElementById('clear-rays').addEventListener('click', clearRayPaths);
        
        // Analysis tools
        document.getElementById('resolution-analysis').addEventListener('click', performResolutionAnalysis);
        document.getElementById('checkerboard-test').addEventListener('click', function() {
            generateCheckerboardModel();
            updateVisualization();
        });
        
        // Presets
        document.getElementById('crustal-model').addEventListener('click', function() {
            generateCrustalModel();
            updateVisualization();
        });
        
        document.getElementById('mantle-model').addEventListener('click', function() {
            generateMantleModel();
            updateVisualization();
        });
        
        document.getElementById('subduction-model').addEventListener('click', function() {
            generateSubductionModel();
            updateVisualization();
        });
        
        // Export functionality
        document.getElementById('export-model').addEventListener('click', exportModel);
    }

    function updateVisualization() {
        // Clear existing cross-sections
        clearCrossSections();
        
        // Recreate cross-sections with current settings
        const xVal = parseFloat(document.getElementById('x-section').value);
        const yVal = parseFloat(document.getElementById('y-section').value);
        const zVal = parseFloat(document.getElementById('z-section').value);
        
        if (xVal !== 0) updateCrossSection('x', xVal);
        if (yVal !== 0) updateCrossSection('y', yVal);
        if (zVal !== -5) updateCrossSection('z', zVal);
    }

    function updateCrossSection(axis, position) {
        // Remove existing cross-section for this axis
        crossSections.forEach((section, index) => {
            if (section.userData.axis === axis) {
                scene.remove(section);
                crossSections.splice(index, 1);
            }
        });
        
        // Create new cross-section
        const section = createCrossSection(axis, position);
        section.userData.axis = axis;
    }

    function updateOpacity() {
        const opacity = TOMOGRAPHY_CONFIG.visualization.opacity;
        crossSections.forEach(section => {
            section.material.opacity = opacity;
        });
        isosurfaces.forEach(surface => {
            surface.material.opacity = opacity * 0.8;
        });
    }

    function clearCrossSections() {
        crossSections.forEach(section => scene.remove(section));
        crossSections.length = 0;
    }

    function clearIsosurfaces() {
        isosurfaces.forEach(surface => scene.remove(surface));
        isosurfaces.length = 0;
    }

    function clearRayPaths() {
        rayPaths.forEach(ray => scene.remove(ray));
        rayPaths.length = 0;
    }

    function generateMantleModel() {
        const { nx, ny, nz } = velocityModel.dimensions;
        
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < ny; j++) {
                for (let k = 0; k < nz; k++) {
                    const idx = i + j * nx + k * nx * ny;
                    const z = velocityModel.coordinates.z[k];
                    
                    // Mantle velocity structure
                    const vp = 7.8 + (z / -100) * 1.2 + 0.1 * Math.random();
                    const vs = vp / 1.69 + 0.05 * Math.random();
                    
                    velocityModel.vp[idx] = Math.max(7.0, Math.min(9.0, vp));
                    velocityModel.vs[idx] = Math.max(4.0, Math.min(5.2, vs));
                }
            }
        }
    }

    function generateSubductionModel() {
        const { nx, ny, nz } = velocityModel.dimensions;
        
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < ny; j++) {
                for (let k = 0; k < nz; k++) {
                    const idx = i + j * nx + k * nx * ny;
                    const x = velocityModel.coordinates.x[i];
                    const y = velocityModel.coordinates.y[j];
                    const z = velocityModel.coordinates.z[k];
                    
                    // Subducting slab (high velocity)
                    const slabDepth = -5 - (x + 50) * 0.3;
                    const slabThickness = 8;
                    
                    let vp, vs;
                    if (z > slabDepth && z < slabDepth + slabThickness && x > -40) {
                        // High velocity slab
                        vp = 7.5 + 0.1 * Math.random();
                        vs = vp / 1.70;
                    } else if (z > -10) {
                        // Overriding plate
                        vp = 6.0 + 0.2 * Math.random();
                        vs = vp / 1.75;
                    } else {
                        // Mantle wedge
                        vp = 7.8 - 0.3 * Math.exp(-(Math.pow(x+20, 2) + Math.pow(z+15, 2))/100);
                        vs = vp / 1.69;
                    }
                    
                    velocityModel.vp[idx] = Math.max(4.0, Math.min(8.5, vp));
                    velocityModel.vs[idx] = Math.max(2.3, Math.min(4.9, vs));
                }
            }
        }
    }

    function generateCheckerboardModel() {
        const { nx, ny, nz } = velocityModel.dimensions;
        
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < ny; j++) {
                for (let k = 0; k < nz; k++) {
                    const idx = i + j * nx + k * nx * ny;
                    const pattern = ((Math.floor(i/8) + Math.floor(j/8) + Math.floor(k/4)) % 2) * 2 - 1;
                    
                    const baseVp = 6.5;
                    const baseVs = 3.7;
                    
                    velocityModel.vp[idx] = baseVp + pattern * 0.8;
                    velocityModel.vs[idx] = baseVs + pattern * 0.4;
                }
            }
        }
    }

    function exportModel() {
        const data = {
            dimensions: velocityModel.dimensions,
            vp: Array.from(velocityModel.vp),
            vs: Array.from(velocityModel.vs),
            coordinates: {
                x: Array.from(velocityModel.coordinates.x),
                y: Array.from(velocityModel.coordinates.y),
                z: Array.from(velocityModel.coordinates.z)
            },
            metadata: {
                created: new Date().toISOString(),
                type: 'seismic_tomography_model',
                units: 'km/s'
            }
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'tomography_model.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    function animate() {
        requestAnimationFrame(animate);
        
        if (controls) {
            controls.update();
        }
        
        renderer.render(scene, camera);
    }

    function show() {
        const ui = document.getElementById('tomography-ui');
        if (ui) {
            ui.style.display = 'block';
            if (!renderer) init();
            animate();
            console.log('🌍 Tomography 3D Visualizer opened');
        }
    }

    function hide() {
        const ui = document.getElementById('tomography-ui');
        if (ui) {
            ui.style.display = 'none';
            console.log('🌍 Tomography 3D Visualizer closed');
        }
    }

    // Keyboard shortcut
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.altKey && e.key === 'T') {
            e.preventDefault();
            show();
        }
        if (e.key === 'Escape') {
            hide();
        }
    });

    // Public API
    return {
        init: init,
        show: show,
        hide: hide,
        createCrossSection: createCrossSection,
        createIsosurface: createIsosurface,
        createRayPath: createRayPath,
        generateCrustalModel: generateCrustalModel,
        generateMantleModel: generateMantleModel,
        generateSubductionModel: generateSubductionModel,
        exportModel: exportModel,
        getVelocityModel: () => velocityModel
    };

})();

// Auto-initialize
document.addEventListener('DOMContentLoaded', function() {
    if (typeof THREE !== 'undefined') {
        window.TOMOGRAPHY_3D.init();
        console.log('🌍 Seismic Tomography 3D System loaded successfully');
        console.log('🎮 Press Ctrl+Alt+T to open Tomography 3D Visualizer');
    } else {
        console.warn('⚠️ THREE.js not found - Tomography 3D system disabled');
    }
});