/**
 * Velocity Model Management System
 * Advanced velocity model handling with format conversion, interpolation, and analysis
 */

window.VELOCITY_MODEL = (function() {
    'use strict';

    let models = new Map();
    let activeModel = null;
    let modelHistory = [];
    let interpolationCache = new Map();

    const MODEL_FORMATS = {
        FANG: 'fang',
        VTK: 'vtk',
        NETCDF: 'netcdf',
        SEGY: 'segy',
        SEISAN: 'seisan',
        HYPOCENTER: 'hypocenter',
        CUSTOM_JSON: 'json'
    };

    const EARTH_MODELS = {
        IASP91: 'iasp91',
        PREM: 'prem',
        AK135: 'ak135',
        CRUST1: 'crust1.0',
        LITHO1: 'litho1.0'
    };

    const INTERPOLATION_METHODS = {
        TRILINEAR: 'trilinear',
        TRICUBIC: 'tricubic',
        SHEPARD: 'shepard',
        KRIGING: 'kriging',
        RBF: 'rbf'
    };

    // Initialize the velocity model system
    function init() {
        console.log('🌍 Initializing Velocity Model Management System...');
        
        loadStandardEarthModels();
        setupModelCache();
        createModelBrowser();
        
        console.log('✅ Velocity Model System initialized');
    }

    function loadStandardEarthModels() {
        // IASP91 Model
        const iasp91 = createIASP91Model();
        registerModel('IASP91', iasp91);
        
        // PREM Model
        const prem = createPREMModel();
        registerModel('PREM', prem);
        
        // AK135 Model
        const ak135 = createAK135Model();
        registerModel('AK135', ak135);
        
        // CRUST1.0 Model (simplified)
        const crust1 = createCrust1Model();
        registerModel('CRUST1.0', crust1);
        
        console.log('📚 Standard Earth models loaded');
    }

    function createIASP91Model() {
        return {
            name: 'IASP91',
            type: '1D',
            description: 'International Association of Seismology and Physics of the Earth Interior 1991',
            reference: 'Kennett & Engdahl (1991)',
            layers: [
                { depth: 0.0, vp: 5.8, vs: 3.36, density: 2.72 },
                { depth: 20.0, vp: 5.8, vs: 3.36, density: 2.72 },
                { depth: 20.0, vp: 6.5, vs: 3.75, density: 2.92 },
                { depth: 35.0, vp: 6.5, vs: 3.75, density: 2.92 },
                { depth: 35.0, vp: 8.04, vs: 4.47, density: 3.32 },
                { depth: 210.0, vp: 8.045, vs: 4.49, density: 3.32 },
                { depth: 410.0, vp: 8.14, vs: 4.52, density: 3.32 },
                { depth: 660.0, vp: 8.3, vs: 4.61, density: 3.45 },
                { depth: 660.0, vp: 9.03, vs: 4.87, density: 3.99 },
                { depth: 2740.0, vp: 10.2, vs: 5.61, density: 4.38 },
                { depth: 2740.0, vp: 10.79, vs: 0.0, density: 9.90 },
                { depth: 2889.0, vp: 10.79, vs: 0.0, density: 9.90 },
                { depth: 2889.0, vp: 11.05, vs: 3.5, density: 12.8 },
                { depth: 5153.0, vp: 11.05, vs: 3.5, density: 12.8 },
                { depth: 5153.0, vp: 11.26, vs: 3.85, density: 13.01 },
                { depth: 6371.0, vp: 11.26, vs: 3.85, density: 13.01 }
            ]
        };
    }

    function createPREMModel() {
        return {
            name: 'PREM',
            type: '1D',
            description: 'Preliminary Reference Earth Model',
            reference: 'Dziewonski & Anderson (1981)',
            layers: [
                { depth: 0.0, vp: 5.8, vs: 3.2, density: 2.6 },
                { depth: 15.0, vp: 5.8, vs: 3.2, density: 2.6 },
                { depth: 15.0, vp: 6.8, vs: 3.9, density: 2.9 },
                { depth: 24.4, vp: 6.8, vs: 3.9, density: 2.9 },
                { depth: 24.4, vp: 8.11, vs: 4.49, density: 3.38 },
                { depth: 40.0, vp: 8.11, vs: 4.49, density: 3.38 },
                { depth: 60.0, vp: 8.04, vs: 4.48, density: 3.37 },
                { depth: 80.0, vp: 8.01, vs: 4.47, density: 3.37 },
                { depth: 120.0, vp: 8.03, vs: 4.48, density: 3.37 },
                { depth: 165.0, vp: 8.12, vs: 4.51, density: 3.37 },
                { depth: 210.0, vp: 8.15, vs: 4.52, density: 3.37 },
                { depth: 260.0, vp: 8.18, vs: 4.53, density: 3.37 },
                { depth: 310.0, vp: 8.20, vs: 4.55, density: 3.37 },
                { depth: 360.0, vp: 8.25, vs: 4.56, density: 3.37 },
                { depth: 410.0, vp: 8.30, vs: 4.59, density: 3.37 },
                { depth: 460.0, vp: 8.48, vs: 4.64, density: 3.49 },
                { depth: 510.0, vp: 8.68, vs: 4.69, density: 3.58 },
                { depth: 560.0, vp: 8.85, vs: 4.73, density: 3.65 },
                { depth: 610.0, vp: 9.03, vs: 4.78, density: 3.72 },
                { depth: 660.0, vp: 9.36, vs: 5.08, density: 3.99 },
                { depth: 710.0, vp: 9.53, vs: 5.18, density: 4.07 },
                { depth: 760.0, vp: 9.68, vs: 5.27, density: 4.14 },
                { depth: 860.0, vp: 9.93, vs: 5.42, density: 4.26 },
                { depth: 960.0, vp: 10.16, vs: 5.57, density: 4.37 },
                { depth: 1060.0, vp: 10.35, vs: 5.69, density: 4.46 },
                { depth: 2891.0, vp: 13.69, vs: 7.27, density: 5.57 },
                { depth: 2891.0, vp: 8.06, vs: 0.0, density: 9.90 },
                { depth: 5149.5, vp: 10.36, vs: 0.0, density: 12.17 },
                { depth: 5149.5, vp: 11.03, vs: 3.50, density: 12.76 },
                { depth: 6371.0, vp: 11.26, vs: 3.67, density: 13.01 }
            ]
        };
    }

    function createAK135Model() {
        return {
            name: 'AK135',
            type: '1D',
            description: 'AK135 Reference Earth Model',
            reference: 'Kennett et al. (1995)',
            layers: [
                { depth: 0.0, vp: 5.8, vs: 3.46, density: 2.72 },
                { depth: 20.0, vp: 5.8, vs: 3.46, density: 2.72 },
                { depth: 20.0, vp: 6.5, vs: 3.85, density: 2.92 },
                { depth: 35.0, vp: 6.5, vs: 3.85, density: 2.92 },
                { depth: 35.0, vp: 8.04, vs: 4.48, density: 3.32 },
                { depth: 77.5, vp: 8.045, vs: 4.49, density: 3.32 },
                { depth: 120.0, vp: 8.05, vs: 4.50, density: 3.37 },
                { depth: 165.0, vp: 8.175, vs: 4.51, density: 3.37 },
                { depth: 210.0, vp: 8.301, vs: 4.52, density: 3.37 },
                { depth: 260.0, vp: 8.482, vs: 4.609, density: 3.368 },
                { depth: 310.0, vp: 8.665, vs: 4.696, density: 3.549 },
                { depth: 360.0, vp: 8.848, vs: 4.783, density: 3.727 },
                { depth: 410.0, vp: 9.03, vs: 4.87, density: 3.902 },
                { depth: 460.0, vp: 9.36, vs: 5.08, density: 3.99 },
                { depth: 510.0, vp: 9.528, vs: 5.186, density: 4.076 },
                { depth: 560.0, vp: 9.696, vs: 5.292, density: 4.159 },
                { depth: 610.0, vp: 9.864, vs: 5.398, density: 4.241 },
                { depth: 660.0, vp: 10.032, vs: 5.504, density: 4.32 },
                { depth: 710.0, vp: 10.2, vs: 5.61, density: 4.396 },
                { depth: 2891.5, vp: 13.717, vs: 7.265, density: 5.566 },
                { depth: 2891.5, vp: 8.065, vs: 0.0, density: 9.903 },
                { depth: 5149.5, vp: 10.355, vs: 0.0, density: 12.166 },
                { depth: 5149.5, vp: 11.028, vs: 3.504, density: 12.763 },
                { depth: 6371.0, vp: 11.262, vs: 3.667, density: 13.013 }
            ]
        };
    }

    function createCrust1Model() {
        // Simplified global crustal model
        return {
            name: 'CRUST1.0',
            type: '3D',
            description: 'Global 1-degree crustal model',
            reference: 'Laske et al. (2013)',
            resolution: { lat: 1.0, lon: 1.0 },
            crustalTypes: {
                'Archean': { vp: 6.4, vs: 3.6, density: 2.8, thickness: 45 },
                'Early Proterozoic': { vp: 6.3, vs: 3.5, density: 2.75, thickness: 42 },
                'Late Proterozoic': { vp: 6.2, vs: 3.4, density: 2.7, thickness: 40 },
                'Phanerozoic': { vp: 6.0, vs: 3.3, density: 2.65, thickness: 35 },
                'Extended Crust': { vp: 5.8, vs: 3.2, density: 2.6, thickness: 25 },
                'Oceanic Crust': { vp: 6.8, vs: 3.9, density: 2.9, thickness: 7 },
                'Oceanic Plateau': { vp: 7.0, vs: 4.0, density: 3.0, thickness: 15 }
            }
        };
    }

    function registerModel(name, model) {
        model.id = generateModelId();
        model.created = new Date().toISOString();
        models.set(name, model);
        
        console.log(`📊 Model registered: ${name}`);
    }

    function generateModelId() {
        return 'model_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function setActiveModel(name) {
        if (models.has(name)) {
            activeModel = models.get(name);
            console.log(`🎯 Active model set to: ${name}`);
            updateModelHistory(name);
            return true;
        }
        console.warn(`⚠️ Model not found: ${name}`);
        return false;
    }

    function getActiveModel() {
        return activeModel;
    }

    function updateModelHistory(name) {
        modelHistory.unshift({ name, timestamp: Date.now() });
        if (modelHistory.length > 10) {
            modelHistory.pop();
        }
    }

    function interpolateVelocity(lat, lon, depth, property = 'vp', method = 'trilinear') {
        if (!activeModel) {
            console.warn('⚠️ No active model for interpolation');
            return null;
        }

        const cacheKey = `${lat}_${lon}_${depth}_${property}_${method}`;
        if (interpolationCache.has(cacheKey)) {
            return interpolationCache.get(cacheKey);
        }

        let value;

        if (activeModel.type === '1D') {
            value = interpolate1D(depth, property);
        } else if (activeModel.type === '3D') {
            value = interpolate3D(lat, lon, depth, property, method);
        }

        interpolationCache.set(cacheKey, value);
        return value;
    }

    function interpolate1D(depth, property) {
        const layers = activeModel.layers;
        
        // Find surrounding layers
        let upperLayer = null, lowerLayer = null;
        
        for (let i = 0; i < layers.length - 1; i++) {
            if (depth >= layers[i].depth && depth <= layers[i + 1].depth) {
                upperLayer = layers[i];
                lowerLayer = layers[i + 1];
                break;
            }
        }
        
        if (!upperLayer || !lowerLayer) {
            // Extrapolate from nearest layer
            if (depth < layers[0].depth) {
                return layers[0][property];
            } else {
                return layers[layers.length - 1][property];
            }
        }
        
        // Linear interpolation
        const ratio = (depth - upperLayer.depth) / (lowerLayer.depth - upperLayer.depth);
        return upperLayer[property] + ratio * (lowerLayer[property] - upperLayer[property]);
    }

    function interpolate3D(lat, lon, depth, property, method) {
        // Simplified 3D interpolation for CRUST1.0
        if (activeModel.name === 'CRUST1.0') {
            return interpolateCrust1(lat, lon, depth, property);
        }
        
        // Default fallback
        return 6.0; // km/s
    }

    function interpolateCrust1(lat, lon, depth, property) {
        // Determine crustal type based on location
        let crustalType;
        
        if (Math.abs(lat) < 30 && depth < 10) {
            crustalType = 'Oceanic Crust';
        } else if (Math.abs(lat) > 60) {
            crustalType = 'Archean';
        } else if (depth > 30) {
            crustalType = 'Extended Crust';
        } else {
            crustalType = 'Phanerozoic';
        }
        
        const crustProps = activeModel.crustalTypes[crustalType];
        return crustProps[property] || 6.0;
    }

    function loadModelFromFile(file, format) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                try {
                    let model;
                    
                    switch (format) {
                        case MODEL_FORMATS.CUSTOM_JSON:
                            model = parseJSONModel(e.target.result);
                            break;
                        case MODEL_FORMATS.VTK:
                            model = parseVTKModel(e.target.result);
                            break;
                        case MODEL_FORMATS.FANG:
                            model = parseFangModel(e.target.result);
                            break;
                        default:
                            throw new Error(`Unsupported format: ${format}`);
                    }
                    
                    resolve(model);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    function parseJSONModel(content) {
        const data = JSON.parse(content);
        
        return {
            name: data.name || 'Custom Model',
            type: data.type || '3D',
            description: data.description || 'Custom velocity model',
            dimensions: data.dimensions,
            vp: new Float32Array(data.vp),
            vs: new Float32Array(data.vs || []),
            density: new Float32Array(data.density || []),
            coordinates: data.coordinates
        };
    }

    function parseVTKModel(content) {
        // Simple VTK parser for structured grids
        const lines = content.split('\n');
        
        let dimensions = null;
        let spacing = null;
        let origin = null;
        let vp = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            if (line.startsWith('DIMENSIONS')) {
                const dims = line.split(' ').slice(1).map(Number);
                dimensions = { nx: dims[0], ny: dims[1], nz: dims[2] };
            } else if (line.startsWith('SPACING')) {
                const sp = line.split(' ').slice(1).map(Number);
                spacing = { dx: sp[0], dy: sp[1], dz: sp[2] };
            } else if (line.startsWith('ORIGIN')) {
                const orig = line.split(' ').slice(1).map(Number);
                origin = { x: orig[0], y: orig[1], z: orig[2] };
            } else if (line.includes('SCALARS vp')) {
                // Read velocity data
                i += 2; // Skip lookup table line
                for (let j = i; j < lines.length; j++) {
                    const values = lines[j].trim().split(/\s+/).map(Number);
                    vp.push(...values);
                }
                break;
            }
        }
        
        return {
            name: 'VTK Model',
            type: '3D',
            description: 'Model loaded from VTK file',
            dimensions: dimensions,
            spacing: spacing,
            origin: origin,
            vp: new Float32Array(vp)
        };
    }

    function parseFangModel(content) {
        // Parser for FANG format (commonly used in seismic tomography)
        const lines = content.split('\n');
        const model = {
            name: 'FANG Model',
            type: '3D',
            description: 'Model loaded from FANG format',
            nodes: [],
            elements: [],
            velocities: []
        };
        
        let section = '';
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('#NODES')) {
                section = 'nodes';
                continue;
            } else if (trimmed.startsWith('#ELEMENTS')) {
                section = 'elements';
                continue;
            } else if (trimmed.startsWith('#VELOCITIES')) {
                section = 'velocities';
                continue;
            }
            
            if (trimmed === '' || trimmed.startsWith('#')) continue;
            
            const values = trimmed.split(/\s+/).map(Number);
            
            if (section === 'nodes') {
                model.nodes.push({ x: values[0], y: values[1], z: values[2] });
            } else if (section === 'elements') {
                model.elements.push(values);
            } else if (section === 'velocities') {
                model.velocities.push(values[0]);
            }
        }
        
        return model;
    }

    function exportModel(name, format) {
        const model = models.get(name);
        if (!model) {
            console.warn(`⚠️ Model not found: ${name}`);
            return null;
        }
        
        let content;
        let filename;
        
        switch (format) {
            case MODEL_FORMATS.CUSTOM_JSON:
                content = JSON.stringify(model, null, 2);
                filename = `${name.replace(/\s+/g, '_')}.json`;
                break;
            case MODEL_FORMATS.VTK:
                content = exportToVTK(model);
                filename = `${name.replace(/\s+/g, '_')}.vtk`;
                break;
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
        
        downloadFile(content, filename);
    }

    function exportToVTK(model) {
        if (model.type !== '3D' || !model.dimensions) {
            throw new Error('VTK export requires 3D model with dimensions');
        }
        
        const { nx, ny, nz } = model.dimensions;
        const { xmin, xmax, ymin, ymax, zmin, zmax } = model.dimensions;
        
        const dx = (xmax - xmin) / (nx - 1);
        const dy = (ymax - ymin) / (ny - 1);
        const dz = (zmax - zmin) / (nz - 1);
        
        let vtk = `# vtk DataFile Version 3.0
Velocity Model: ${model.name}
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS ${nx} ${ny} ${nz}
ORIGIN ${xmin} ${ymin} ${zmin}
SPACING ${dx} ${dy} ${dz}
POINT_DATA ${nx * ny * nz}
SCALARS vp float 1
LOOKUP_TABLE default
`;
        
        // Add velocity data
        for (let i = 0; i < model.vp.length; i++) {
            vtk += model.vp[i].toFixed(3);
            if ((i + 1) % 10 === 0) vtk += '\n';
            else vtk += ' ';
        }
        
        return vtk;
    }

    function downloadFile(content, filename) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function createModelBrowser() {
        const browser = document.createElement('div');
        browser.id = 'velocity-model-browser';
        browser.innerHTML = `
            <div class="model-browser-panel">
                <h3>🌍 Velocity Model Browser</h3>
                
                <div class="model-list">
                    <h4>Available Models</h4>
                    <div id="model-list-container"></div>
                </div>
                
                <div class="model-details">
                    <h4>Model Details</h4>
                    <div id="model-details-container">
                        <p>Select a model to view details</p>
                    </div>
                </div>
                
                <div class="model-controls">
                    <h4>Model Operations</h4>
                    <button id="load-model-file">Load Model File</button>
                    <input type="file" id="model-file-input" accept=".json,.vtk,.fang" style="display: none;">
                    <button id="export-active-model">Export Active Model</button>
                    <button id="create-synthetic-model">Create Synthetic Model</button>
                </div>
                
                <div class="interpolation-test">
                    <h4>Interpolation Test</h4>
                    <label>Latitude: <input type="number" id="test-lat" value="40.0" step="0.1"></label>
                    <label>Longitude: <input type="number" id="test-lon" value="-120.0" step="0.1"></label>
                    <label>Depth (km): <input type="number" id="test-depth" value="10.0" step="0.1"></label>
                    <button id="test-interpolation">Test Interpolation</button>
                    <div id="interpolation-result"></div>
                </div>
            </div>
        `;
        
        browser.style.cssText = `
            position: fixed;
            top: 50px;
            right: 20px;
            width: 400px;
            max-height: 80vh;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 10px;
            padding: 20px;
            z-index: 10001;
            overflow-y: auto;
            display: none;
        `;
        
        document.body.appendChild(browser);
        
        setupModelBrowserEvents();
        updateModelList();
    }

    function setupModelBrowserEvents() {
        document.getElementById('load-model-file').addEventListener('click', function() {
            document.getElementById('model-file-input').click();
        });
        
        document.getElementById('model-file-input').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const format = getFormatFromExtension(file.name);
                loadModelFromFile(file, format).then(model => {
                    registerModel(file.name.split('.')[0], model);
                    updateModelList();
                }).catch(error => {
                    console.error('Failed to load model:', error);
                });
            }
        });
        
        document.getElementById('export-active-model').addEventListener('click', function() {
            if (activeModel) {
                exportModel(activeModel.name, MODEL_FORMATS.CUSTOM_JSON);
            } else {
                alert('No active model to export');
            }
        });
        
        document.getElementById('test-interpolation').addEventListener('click', function() {
            testInterpolation();
        });
    }

    function getFormatFromExtension(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        switch (ext) {
            case 'json': return MODEL_FORMATS.CUSTOM_JSON;
            case 'vtk': return MODEL_FORMATS.VTK;
            case 'fang': return MODEL_FORMATS.FANG;
            default: return MODEL_FORMATS.CUSTOM_JSON;
        }
    }

    function updateModelList() {
        const container = document.getElementById('model-list-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        models.forEach((model, name) => {
            const item = document.createElement('div');
            item.className = 'model-item';
            item.innerHTML = `
                <div class="model-name">${name}</div>
                <div class="model-info">${model.type} - ${model.description}</div>
                <button onclick="VELOCITY_MODEL.setActiveModel('${name}')">Set Active</button>
            `;
            item.style.cssText = `
                padding: 10px;
                margin: 5px 0;
                background: ${activeModel && activeModel.name === name ? '#444' : '#222'};
                border-radius: 5px;
                cursor: pointer;
            `;
            
            item.addEventListener('click', function() {
                showModelDetails(model);
            });
            
            container.appendChild(item);
        });
    }

    function showModelDetails(model) {
        const container = document.getElementById('model-details-container');
        if (!container) return;
        
        container.innerHTML = `
            <h5>${model.name}</h5>
            <p><strong>Type:</strong> ${model.type}</p>
            <p><strong>Description:</strong> ${model.description}</p>
            ${model.reference ? `<p><strong>Reference:</strong> ${model.reference}</p>` : ''}
            ${model.layers ? `<p><strong>Layers:</strong> ${model.layers.length}</p>` : ''}
            ${model.dimensions ? `<p><strong>Dimensions:</strong> ${model.dimensions.nx}×${model.dimensions.ny}×${model.dimensions.nz}</p>` : ''}
            <p><strong>Created:</strong> ${model.created}</p>
        `;
    }

    function testInterpolation() {
        if (!activeModel) {
            document.getElementById('interpolation-result').innerHTML = 'No active model';
            return;
        }
        
        const lat = parseFloat(document.getElementById('test-lat').value);
        const lon = parseFloat(document.getElementById('test-lon').value);
        const depth = parseFloat(document.getElementById('test-depth').value);
        
        const vp = interpolateVelocity(lat, lon, depth, 'vp');
        const vs = interpolateVelocity(lat, lon, depth, 'vs');
        const density = interpolateVelocity(lat, lon, depth, 'density');
        
        document.getElementById('interpolation-result').innerHTML = `
            <p><strong>Vp:</strong> ${vp ? vp.toFixed(2) : 'N/A'} km/s</p>
            <p><strong>Vs:</strong> ${vs ? vs.toFixed(2) : 'N/A'} km/s</p>
            <p><strong>Density:</strong> ${density ? density.toFixed(2) : 'N/A'} g/cm³</p>
            <p><strong>Vp/Vs:</strong> ${(vp && vs) ? (vp/vs).toFixed(2) : 'N/A'}</p>
        `;
    }

    function setupModelCache() {
        // Clear cache periodically to prevent memory issues
        setInterval(() => {
            if (interpolationCache.size > 10000) {
                interpolationCache.clear();
                console.log('🗑️ Interpolation cache cleared');
            }
        }, 300000); // 5 minutes
    }

    function showModelBrowser() {
        const browser = document.getElementById('velocity-model-browser');
        if (browser) {
            browser.style.display = 'block';
            updateModelList();
        }
    }

    function hideModelBrowser() {
        const browser = document.getElementById('velocity-model-browser');
        if (browser) {
            browser.style.display = 'none';
        }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.altKey && e.key === 'M') {
            e.preventDefault();
            showModelBrowser();
        }
        if (e.key === 'Escape') {
            hideModelBrowser();
        }
    });

    // Public API
    return {
        init: init,
        registerModel: registerModel,
        setActiveModel: setActiveModel,
        getActiveModel: getActiveModel,
        interpolateVelocity: interpolateVelocity,
        loadModelFromFile: loadModelFromFile,
        exportModel: exportModel,
        getModels: () => Array.from(models.keys()),
        showModelBrowser: showModelBrowser,
        hideModelBrowser: hideModelBrowser,
        FORMATS: MODEL_FORMATS,
        EARTH_MODELS: EARTH_MODELS,
        INTERPOLATION_METHODS: INTERPOLATION_METHODS
    };

})();

// Auto-initialize
document.addEventListener('DOMContentLoaded', function() {
    window.VELOCITY_MODEL.init();
    console.log('🌍 Velocity Model Management System loaded');
    console.log('🎮 Press Ctrl+Alt+M to open Model Browser');
});