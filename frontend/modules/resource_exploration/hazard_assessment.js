/**
 * Integrated Hazard Assessment System
 * Multi-hazard analysis, probabilistic assessment, and emergency response planning
 */

window.HAZARD_ASSESSMENT = (function() {
    'use strict';

    let hazardData = new Map();
    let activeScenario = null;
    let hazardMaps = [];
    let probabilisticModels = new Map();
    let emergencyPlans = new Map();
    
    const HAZARD_TYPES = {
        VOLCANIC: {
            name: 'Volcanic Hazards',
            subtypes: ['lava_flows', 'pyroclastic_flows', 'ash_fall', 'lahars', 'debris_avalanche', 'volcanic_gas']
        },
        SEISMIC: {
            name: 'Seismic Hazards',
            subtypes: ['ground_shaking', 'surface_rupture', 'liquefaction', 'landslides', 'tsunami']
        },
        GEOTECHNICAL: {
            name: 'Geotechnical Hazards',
            subtypes: ['slope_instability', 'rockfall', 'subsidence', 'expansion_soils', 'foundation_failure']
        },
        HYDROLOGICAL: {
            name: 'Hydrological Hazards',
            subtypes: ['flooding', 'dam_failure', 'erosion', 'sedimentation', 'groundwater_depletion']
        }
    };

    const VULNERABILITY_CLASSES = {
        POPULATION: {
            high: 'dense_urban_areas',
            medium: 'suburban_residential', 
            low: 'rural_scattered'
        },
        INFRASTRUCTURE: {
            critical: 'hospitals_schools_emergency',
            important: 'utilities_transportation',
            standard: 'commercial_industrial'
        },
        ECONOMIC: {
            high_value: 'financial_centers_tourism',
            medium_value: 'manufacturing_agriculture',
            low_value: 'vacant_undeveloped'
        }
    };

    const INTENSITY_SCALES = {
        VOLCANIC: {
            VEI: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            descriptions: [
                'Non-explosive', 'Gentle', 'Explosive', 'Severe', 
                'Cataclysmic', 'Paroxysmal', 'Colossal', 'Super-colossal', 'Mega-colossal'
            ]
        },
        SEISMIC: {
            MMI: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            PGA: [0.01, 0.03, 0.07, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0]
        }
    };

    // Initialize the hazard assessment system
    function init() {
        console.log('🚨 Initializing Integrated Hazard Assessment System...');
        
        createHazardUI();
        loadHazardData();
        initializeProbabilisticModels();
        setupEmergencyPlans();
        
        console.log('✅ Hazard Assessment System initialized');
    }

    function createHazardUI() {
        const ui = document.createElement('div');
        ui.id = 'hazard-assessment-ui';
        ui.innerHTML = `
            <div class="hazard-panel">
                <h3>🚨 Integrated Hazard Assessment</h3>
                
                <div class="hazard-tabs">
                    <button class="tab-button active" data-tab="multi-hazard">Multi-Hazard Map</button>
                    <button class="tab-button" data-tab="probabilistic">Probabilistic Analysis</button>
                    <button class="tab-button" data-tab="scenarios">Scenario Modeling</button>
                    <button class="tab-button" data-tab="risk">Risk Assessment</button>
                    <button class="tab-button" data-tab="emergency">Emergency Planning</button>
                </div>
                
                <!-- Multi-Hazard Map Tab -->
                <div id="multi-hazard" class="tab-content active">
                    <h4>Multi-Hazard Mapping</h4>
                    <div class="hazard-selector">
                        <label>Select Hazards:</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" id="volcanic-hazard" checked> Volcanic</label>
                            <label><input type="checkbox" id="seismic-hazard" checked> Seismic</label>
                            <label><input type="checkbox" id="geotechnical-hazard"> Geotechnical</label>
                            <label><input type="checkbox" id="hydrological-hazard"> Hydrological</label>
                        </div>
                    </div>
                    
                    <div class="map-controls">
                        <label>Aggregation Method:</label>
                        <select id="aggregation-method">
                            <option value="maximum">Maximum Hazard</option>
                            <option value="weighted">Weighted Sum</option>
                            <option value="multiplicative">Multiplicative</option>
                            <option value="fuzzy">Fuzzy Logic</option>
                        </select>
                        
                        <button id="generate-multi-hazard">Generate Multi-Hazard Map</button>
                    </div>
                    
                    <div class="hazard-legend">
                        <h5>Hazard Intensity Legend</h5>
                        <div class="legend-items" id="hazard-legend"></div>
                    </div>
                </div>
                
                <!-- Probabilistic Analysis Tab -->
                <div id="probabilistic" class="tab-content">
                    <h4>Probabilistic Hazard Analysis</h4>
                    <div class="probability-controls">
                        <label>Return Periods (years):</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" id="return-10" checked> 10</label>
                            <label><input type="checkbox" id="return-50" checked> 50</label>
                            <label><input type="checkbox" id="return-100" checked> 100</label>
                            <label><input type="checkbox" id="return-500" checked> 500</label>
                            <label><input type="checkbox" id="return-1000"> 1000</label>
                            <label><input type="checkbox" id="return-2500"> 2500</label>
                        </div>
                        
                        <label>Confidence Level:</label>
                        <select id="confidence-level">
                            <option value="0.5">50% (Median)</option>
                            <option value="0.84">84th Percentile</option>
                            <option value="0.95" selected>95th Percentile</option>
                        </select>
                        
                        <button id="compute-hazard-curves">Compute Hazard Curves</button>
                    </div>
                    
                    <div class="hazard-curves">
                        <canvas id="hazard-curve-chart" width="600" height="400"></canvas>
                    </div>
                </div>
                
                <!-- Scenario Modeling Tab -->
                <div id="scenarios" class="tab-content">
                    <h4>Scenario Modeling</h4>
                    <div class="scenario-builder">
                        <label>Scenario Type:</label>
                        <select id="scenario-type">
                            <option value="deterministic">Deterministic</option>
                            <option value="stochastic">Stochastic</option>
                            <option value="worst_case">Worst Case</option>
                            <option value="historical">Historical Analog</option>
                        </select>
                        
                        <div class="volcanic-scenario">
                            <h5>Volcanic Eruption Scenario</h5>
                            <label>VEI: <select id="scenario-vei">
                                <option value="2">2 - Explosive</option>
                                <option value="3">3 - Severe</option>
                                <option value="4" selected>4 - Cataclysmic</option>
                                <option value="5">5 - Paroxysmal</option>
                            </select></label>
                            
                            <label>Wind Direction: <select id="wind-direction">
                                <option value="N">North</option>
                                <option value="NE">Northeast</option>
                                <option value="E">East</option>
                                <option value="SE">Southeast</option>
                                <option value="S">South</option>
                                <option value="SW" selected>Southwest</option>
                                <option value="W">West</option>
                                <option value="NW">Northwest</option>
                            </select></label>
                            
                            <label>Wind Speed: <input type="range" id="wind-speed" min="5" max="50" value="20"> <span id="wind-speed-value">20</span> m/s</label>
                        </div>
                        
                        <div class="seismic-scenario">
                            <h5>Earthquake Scenario</h5>
                            <label>Magnitude: <input type="range" id="earthquake-magnitude" min="5.0" max="8.0" step="0.1" value="6.5"> <span id="magnitude-value">6.5</span></label>
                            <label>Depth: <input type="range" id="earthquake-depth" min="1" max="30" value="10"> <span id="depth-value">10</span> km</label>
                            <label>Fault Type: <select id="fault-type">
                                <option value="strike_slip">Strike-Slip</option>
                                <option value="normal">Normal</option>
                                <option value="reverse" selected>Reverse</option>
                                <option value="thrust">Thrust</option>
                            </select></label>
                        </div>
                        
                        <button id="run-scenario">Run Scenario</button>
                        <button id="save-scenario">Save Scenario</button>
                    </div>
                    
                    <div class="scenario-results">
                        <h5>Scenario Results</h5>
                        <div id="scenario-output"></div>
                    </div>
                </div>
                
                <!-- Risk Assessment Tab -->
                <div id="risk" class="tab-content">
                    <h4>Risk Assessment</h4>
                    <div class="risk-parameters">
                        <h5>Vulnerability Assessment</h5>
                        <label>Population Density: <select id="population-density">
                            <option value="low">Low (< 100/km²)</option>
                            <option value="medium" selected>Medium (100-1000/km²)</option>
                            <option value="high">High (> 1000/km²)</option>
                        </select></label>
                        
                        <label>Building Stock: <select id="building-stock">
                            <option value="unreinforced">Unreinforced Masonry</option>
                            <option value="wood_frame">Wood Frame</option>
                            <option value="concrete" selected>Reinforced Concrete</option>
                            <option value="steel">Steel Frame</option>
                        </select></label>
                        
                        <label>Critical Facilities:</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" id="hospitals" checked> Hospitals</label>
                            <label><input type="checkbox" id="schools" checked> Schools</label>
                            <label><input type="checkbox" id="fire-stations"> Fire Stations</label>
                            <label><input type="checkbox" id="power-plants"> Power Plants</label>
                        </div>
                        
                        <button id="calculate-risk">Calculate Risk</button>
                    </div>
                    
                    <div class="risk-results">
                        <h5>Risk Metrics</h5>
                        <div id="risk-metrics"></div>
                        
                        <h5>Loss Estimation</h5>
                        <div id="loss-estimates"></div>
                    </div>
                </div>
                
                <!-- Emergency Planning Tab -->
                <div id="emergency" class="tab-content">
                    <h4>Emergency Response Planning</h4>
                    <div class="emergency-zones">
                        <h5>Evacuation Zones</h5>
                        <label>Zone Type: <select id="zone-type">
                            <option value="volcanic">Volcanic Hazard Zones</option>
                            <option value="seismic">Seismic Safety Zones</option>
                            <option value="tsunami">Tsunami Evacuation Zones</option>
                            <option value="multi">Multi-Hazard Zones</option>
                        </select></label>
                        
                        <button id="generate-zones">Generate Evacuation Zones</button>
                    </div>
                    
                    <div class="response-resources">
                        <h5>Emergency Resources</h5>
                        <div class="resource-inventory">
                            <label>Shelters: <input type="number" id="shelter-count" value="5"></label>
                            <label>Emergency Services: <input type="number" id="emergency-services" value="3"></label>
                            <label>Medical Facilities: <input type="number" id="medical-facilities" value="2"></label>
                        </div>
                        
                        <button id="optimize-response">Optimize Response Plan</button>
                    </div>
                    
                    <div class="communication-plan">
                        <h5>Communication Strategy</h5>
                        <div class="alert-systems">
                            <label><input type="checkbox" id="sms-alerts" checked> SMS Alerts</label>
                            <label><input type="checkbox" id="radio-broadcasts" checked> Radio Broadcasts</label>
                            <label><input type="checkbox" id="social-media"> Social Media</label>
                            <label><input type="checkbox" id="sirens"> Emergency Sirens</label>
                        </div>
                        
                        <button id="test-communication">Test Communication System</button>
                    </div>
                </div>
            </div>
            
            <div class="hazard-map-container">
                <canvas id="hazard-map-canvas" width="800" height="600"></canvas>
                <div class="map-overlay">
                    <div class="map-legend" id="map-legend"></div>
                    <div class="map-controls" id="map-controls">
                        <button id="zoom-in">🔍+</button>
                        <button id="zoom-out">🔍-</button>
                        <button id="reset-view">🎯</button>
                    </div>
                </div>
            </div>
        `;
        
        ui.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            z-index: 10000;
            display: none;
            overflow: hidden;
        `;
        
        document.body.appendChild(ui);
        setupHazardUIEvents();
    }

    function setupHazardUIEvents() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', function() {
                const tabName = this.dataset.tab;
                switchTab(tabName);
            });
        });
        
        // Multi-hazard map generation
        document.getElementById('generate-multi-hazard').addEventListener('click', generateMultiHazardMap);
        
        // Probabilistic analysis
        document.getElementById('compute-hazard-curves').addEventListener('click', computeHazardCurves);
        
        // Scenario modeling
        document.getElementById('run-scenario').addEventListener('click', runScenario);
        document.getElementById('save-scenario').addEventListener('click', saveScenario);
        
        // Risk assessment
        document.getElementById('calculate-risk').addEventListener('click', calculateRisk);
        
        // Emergency planning
        document.getElementById('generate-zones').addEventListener('click', generateEvacuationZones);
        document.getElementById('optimize-response').addEventListener('click', optimizeResponse);
        document.getElementById('test-communication').addEventListener('click', testCommunication);
        
        // Slider updates
        document.getElementById('wind-speed').addEventListener('input', function() {
            document.getElementById('wind-speed-value').textContent = this.value;
        });
        
        document.getElementById('earthquake-magnitude').addEventListener('input', function() {
            document.getElementById('magnitude-value').textContent = parseFloat(this.value).toFixed(1);
        });
        
        document.getElementById('earthquake-depth').addEventListener('input', function() {
            document.getElementById('depth-value').textContent = this.value;
        });
    }

    function switchTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Remove active class from all buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active');
        });
        
        // Show selected tab and activate button
        document.getElementById(tabName).classList.add('active');
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    }

    function loadHazardData() {
        // Load volcanic hazard data
        hazardData.set('volcanic', {
            lava_flows: generateHazardGrid('lava', 0.8, 15),
            pyroclastic_flows: generateHazardGrid('pyroclastic', 0.9, 25),
            ash_fall: generateHazardGrid('ash', 0.6, 50),
            lahars: generateHazardGrid('lahar', 0.7, 30),
            debris_avalanche: generateHazardGrid('debris', 0.85, 20),
            volcanic_gas: generateHazardGrid('gas', 0.5, 10)
        });
        
        // Load seismic hazard data
        hazardData.set('seismic', {
            ground_shaking: generateHazardGrid('shake', 0.7, 40),
            surface_rupture: generateHazardGrid('rupture', 0.9, 5),
            liquefaction: generateHazardGrid('liquefaction', 0.6, 20),
            landslides: generateHazardGrid('landslide', 0.75, 15),
            tsunami: generateHazardGrid('tsunami', 0.8, 100)
        });
        
        // Load geotechnical hazard data
        hazardData.set('geotechnical', {
            slope_instability: generateHazardGrid('slope', 0.65, 25),
            rockfall: generateHazardGrid('rockfall', 0.7, 10),
            subsidence: generateHazardGrid('subsidence', 0.55, 30),
            expansion_soils: generateHazardGrid('expansion', 0.4, 35),
            foundation_failure: generateHazardGrid('foundation', 0.6, 20)
        });
        
        console.log('📊 Hazard data loaded');
    }

    function generateHazardGrid(type, maxIntensity, radius) {
        const grid = [];
        const centerX = 400, centerY = 300; // Canvas center
        
        for (let y = 0; y < 600; y += 10) {
            for (let x = 0; x < 800; x += 10) {
                const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
                let intensity = 0;
                
                if (distance < radius) {
                    // Exponential decay with random variations
                    intensity = maxIntensity * Math.exp(-distance / (radius * 0.5)) * (0.7 + 0.6 * Math.random());
                    
                    // Add terrain and geological factors
                    if (type === 'lava') {
                        // Lava flows follow topography
                        intensity *= (1 + 0.3 * Math.sin(x / 50) * Math.cos(y / 40));
                    } else if (type === 'ash') {
                        // Wind effects on ash distribution
                        intensity *= (1 + 0.5 * Math.cos((x - centerX) / 30));
                    } else if (type === 'shake') {
                        // Amplification in sedimentary basins
                        intensity *= (1 + 0.4 * Math.sin(x / 80) * Math.sin(y / 60));
                    }
                    
                    intensity = Math.max(0, Math.min(1, intensity));
                }
                
                grid.push({ x, y, intensity });
            }
        }
        
        return grid;
    }

    function generateMultiHazardMap() {
        console.log('🗺️ Generating multi-hazard map...');
        
        const canvas = document.getElementById('hazard-map-canvas');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Get selected hazards
        const selectedHazards = [];
        if (document.getElementById('volcanic-hazard').checked) selectedHazards.push('volcanic');
        if (document.getElementById('seismic-hazard').checked) selectedHazards.push('seismic');
        if (document.getElementById('geotechnical-hazard').checked) selectedHazards.push('geotechnical');
        if (document.getElementById('hydrological-hazard').checked) selectedHazards.push('hydrological');
        
        const aggregationMethod = document.getElementById('aggregation-method').value;
        
        // Create combined hazard map
        const combinedMap = aggregateHazards(selectedHazards, aggregationMethod);
        
        // Render hazard map
        renderHazardMap(ctx, combinedMap);
        
        // Update legend
        updateHazardLegend(selectedHazards);
        
        console.log('✅ Multi-hazard map generated');
    }

    function aggregateHazards(hazardTypes, method) {
        const combinedMap = [];
        
        for (let y = 0; y < 600; y += 10) {
            for (let x = 0; x < 800; x += 10) {
                let combinedIntensity = 0;
                const intensities = [];
                
                // Collect intensities from all hazard types
                hazardTypes.forEach(hazardType => {
                    if (hazardData.has(hazardType)) {
                        const hazards = hazardData.get(hazardType);
                        let maxHazardIntensity = 0;
                        
                        // Find maximum intensity for this hazard type at this location
                        Object.values(hazards).forEach(hazard => {
                            const point = hazard.find(p => p.x === x && p.y === y);
                            if (point) {
                                maxHazardIntensity = Math.max(maxHazardIntensity, point.intensity);
                            }
                        });
                        
                        intensities.push(maxHazardIntensity);
                    }
                });
                
                // Apply aggregation method
                switch (method) {
                    case 'maximum':
                        combinedIntensity = Math.max(...intensities, 0);
                        break;
                    case 'weighted':
                        const weights = [0.4, 0.3, 0.2, 0.1]; // Volcanic, Seismic, Geotechnical, Hydrological
                        combinedIntensity = intensities.reduce((sum, intensity, i) => 
                            sum + intensity * (weights[i] || 0.1), 0);
                        break;
                    case 'multiplicative':
                        combinedIntensity = intensities.reduce((product, intensity) => 
                            product * (1 - intensity), 1);
                        combinedIntensity = 1 - combinedIntensity;
                        break;
                    case 'fuzzy':
                        // Fuzzy logic combination
                        combinedIntensity = fuzzyOr(intensities);
                        break;
                }
                
                combinedMap.push({ x, y, intensity: Math.min(1, combinedIntensity) });
            }
        }
        
        return combinedMap;
    }

    function fuzzyOr(values) {
        return values.reduce((result, value) => 
            result + value - result * value, 0);
    }

    function renderHazardMap(ctx, hazardMap) {
        // Create image data
        const imageData = ctx.createImageData(800, 600);
        const data = imageData.data;
        
        hazardMap.forEach(point => {
            const index = (point.y * 800 + point.x) * 4;
            const intensity = point.intensity;
            
            // Color mapping based on intensity
            let r, g, b;
            if (intensity < 0.2) {
                // Green (low hazard)
                r = 0; g = 255 * (intensity / 0.2); b = 0;
            } else if (intensity < 0.5) {
                // Yellow (moderate hazard)
                r = 255 * ((intensity - 0.2) / 0.3); g = 255; b = 0;
            } else if (intensity < 0.8) {
                // Orange (high hazard)
                r = 255; g = 255 * (1 - (intensity - 0.5) / 0.3); b = 0;
            } else {
                // Red (very high hazard)
                r = 255; g = 0; b = 0;
            }
            
            // Fill 10x10 pixel block
            for (let dy = 0; dy < 10 && point.y + dy < 600; dy++) {
                for (let dx = 0; dx < 10 && point.x + dx < 800; dx++) {
                    const pixelIndex = ((point.y + dy) * 800 + (point.x + dx)) * 4;
                    data[pixelIndex] = r;
                    data[pixelIndex + 1] = g;
                    data[pixelIndex + 2] = b;
                    data[pixelIndex + 3] = 255 * Math.min(1, intensity + 0.1); // Alpha
                }
            }
        });
        
        ctx.putImageData(imageData, 0, 0);
        
        // Add contour lines
        drawContours(ctx, hazardMap);
        
        // Add scale and north arrow
        drawMapElements(ctx);
    }

    function drawContours(ctx, hazardMap) {
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = 1;
        
        const contourLevels = [0.2, 0.4, 0.6, 0.8];
        
        contourLevels.forEach(level => {
            ctx.beginPath();
            
            for (let y = 0; y < 590; y += 10) {
                for (let x = 0; x < 790; x += 10) {
                    const point = hazardMap.find(p => p.x === x && p.y === y);
                    const pointRight = hazardMap.find(p => p.x === x + 10 && p.y === y);
                    const pointDown = hazardMap.find(p => p.x === x && p.y === y + 10);
                    
                    if (point && pointRight && 
                        ((point.intensity <= level && pointRight.intensity > level) ||
                         (point.intensity > level && pointRight.intensity <= level))) {
                        // Vertical edge
                        ctx.moveTo(x + 10, y);
                        ctx.lineTo(x + 10, y + 10);
                    }
                    
                    if (point && pointDown && 
                        ((point.intensity <= level && pointDown.intensity > level) ||
                         (point.intensity > level && pointDown.intensity <= level))) {
                        // Horizontal edge
                        ctx.moveTo(x, y + 10);
                        ctx.lineTo(x + 10, y + 10);
                    }
                }
            }
            
            ctx.stroke();
        });
    }

    function drawMapElements(ctx) {
        // North arrow
        ctx.save();
        ctx.translate(750, 50);
        ctx.fillStyle = 'black';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        
        // Arrow
        ctx.beginPath();
        ctx.moveTo(0, -20);
        ctx.lineTo(-8, 8);
        ctx.lineTo(0, 0);
        ctx.lineTo(8, 8);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // N label
        ctx.font = '14px bold sans-serif';
        ctx.fillText('N', -5, 25);
        ctx.restore();
        
        // Scale bar
        ctx.save();
        ctx.translate(50, 550);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(100, 0);
        ctx.stroke();
        
        // Scale marks
        [0, 25, 50, 75, 100].forEach(x => {
            ctx.beginPath();
            ctx.moveTo(x, -3);
            ctx.lineTo(x, 3);
            ctx.stroke();
        });
        
        ctx.font = '12px sans-serif';
        ctx.fillStyle = 'black';
        ctx.fillText('0', -3, 15);
        ctx.fillText('5 km', 40, 15);
        ctx.fillText('10 km', 90, 15);
        ctx.restore();
    }

    function updateHazardLegend(selectedHazards) {
        const legend = document.getElementById('hazard-legend');
        legend.innerHTML = '';
        
        const legendItems = [
            { level: 'Very High', color: '#ff0000', range: '0.8 - 1.0' },
            { level: 'High', color: '#ff8000', range: '0.5 - 0.8' },
            { level: 'Moderate', color: '#ffff00', range: '0.2 - 0.5' },
            { level: 'Low', color: '#80ff00', range: '0.0 - 0.2' }
        ];
        
        legendItems.forEach(item => {
            const div = document.createElement('div');
            div.innerHTML = `
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background: ${item.color}; margin-right: 10px;"></div>
                    <span>${item.level} (${item.range})</span>
                </div>
            `;
            legend.appendChild(div);
        });
    }

    function computeHazardCurves() {
        console.log('📈 Computing probabilistic hazard curves...');
        
        const canvas = document.getElementById('hazard-curve-chart');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Get selected return periods
        const returnPeriods = [];
        ['10', '50', '100', '500', '1000', '2500'].forEach(period => {
            if (document.getElementById(`return-${period}`).checked) {
                returnPeriods.push(parseInt(period));
            }
        });
        
        const confidenceLevel = parseFloat(document.getElementById('confidence-level').value);
        
        // Generate hazard curves for different intensity levels
        const intensityLevels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        const hazardCurves = generateHazardCurves(intensityLevels, returnPeriods, confidenceLevel);
        
        // Plot hazard curves
        plotHazardCurves(ctx, hazardCurves, intensityLevels);
        
        console.log('✅ Hazard curves computed');
    }

    function generateHazardCurves(intensityLevels, returnPeriods, confidenceLevel) {
        const curves = [];
        
        intensityLevels.forEach(intensity => {
            const curve = [];
            
            returnPeriods.forEach(returnPeriod => {
                // Simple probabilistic model: exceedance probability
                const lambda = 1 / returnPeriod; // Annual rate
                const probability = 1 - Math.exp(-lambda); // Annual exceedance probability
                
                // Apply uncertainty based on confidence level
                const uncertainty = (1 - confidenceLevel) * 0.5;
                const adjustedProbability = probability * (1 + uncertainty * (Math.random() - 0.5));
                
                curve.push({ returnPeriod, probability: Math.max(0, Math.min(1, adjustedProbability)) });
            });
            
            curves.push({ intensity, curve });
        });
        
        return curves;
    }

    function plotHazardCurves(ctx, hazardCurves, intensityLevels) {
        const margin = { top: 40, right: 40, bottom: 60, left: 80 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        
        // Set up coordinate system
        ctx.save();
        ctx.translate(margin.left, margin.top);
        
        // Background
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        
        // Grid lines
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        
        // Vertical grid lines (return periods)
        [10, 50, 100, 500, 1000, 2500].forEach(period => {
            const x = width * Math.log10(period) / Math.log10(2500);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        });
        
        // Horizontal grid lines (probability)
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0].forEach(prob => {
            const y = height * (1 - Math.log10(prob + 0.001) / Math.log10(1.001));
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        });
        
        // Plot curves
        const colors = ['#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00', 
                       '#bfff00', '#80ff00', '#40ff00', '#00ff00', '#00ff40'];
        
        hazardCurves.forEach((hazardCurve, index) => {
            ctx.strokeStyle = colors[index % colors.length];
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            hazardCurve.curve.forEach((point, i) => {
                const x = width * Math.log10(point.returnPeriod) / Math.log10(2500);
                const y = height * (1 - Math.log10(point.probability + 0.001) / Math.log10(1.001));
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        });
        
        // Axes
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(0, height);
        ctx.lineTo(width, height);
        ctx.stroke();
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, height);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = 'black';
        ctx.font = '12px sans-serif';
        
        // X-axis labels
        [10, 50, 100, 500, 1000, 2500].forEach(period => {
            const x = width * Math.log10(period) / Math.log10(2500);
            ctx.fillText(period.toString(), x - 10, height + 20);
        });
        
        // Y-axis labels
        [0.01, 0.1, 1.0].forEach(prob => {
            const y = height * (1 - Math.log10(prob + 0.001) / Math.log10(1.001));
            ctx.fillText(prob.toFixed(2), -30, y + 5);
        });
        
        // Axis titles
        ctx.font = '14px sans-serif';
        ctx.fillText('Return Period (years)', width/2 - 60, height + 50);
        
        ctx.save();
        ctx.translate(-60, height/2);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('Annual Exceedance Probability', -100, 0);
        ctx.restore();
        
        // Title
        ctx.font = '16px bold sans-serif';
        ctx.fillText('Probabilistic Hazard Curves', width/2 - 100, -20);
        
        ctx.restore();
    }

    function runScenario() {
        console.log('🎬 Running hazard scenario...');
        
        const scenarioType = document.getElementById('scenario-type').value;
        const vei = parseInt(document.getElementById('scenario-vei').value);
        const windDirection = document.getElementById('wind-direction').value;
        const windSpeed = parseInt(document.getElementById('wind-speed').value);
        const magnitude = parseFloat(document.getElementById('earthquake-magnitude').value);
        const depth = parseInt(document.getElementById('earthquake-depth').value);
        const faultType = document.getElementById('fault-type').value;
        
        // Create scenario object
        activeScenario = {
            type: scenarioType,
            volcanic: { vei, windDirection, windSpeed },
            seismic: { magnitude, depth, faultType },
            timestamp: new Date().toISOString()
        };
        
        // Simulate scenario impacts
        const impacts = simulateScenarioImpacts(activeScenario);
        
        // Display results
        displayScenarioResults(impacts);
        
        console.log('✅ Scenario simulation complete');
    }

    function simulateScenarioImpacts(scenario) {
        const impacts = {
            volcanic: {},
            seismic: {},
            affected_area: 0,
            population_at_risk: 0,
            economic_losses: 0
        };
        
        if (scenario.volcanic.vei > 0) {
            // Volcanic impacts based on VEI
            const vei = scenario.volcanic.vei;
            impacts.volcanic = {
                ash_fall_radius: Math.pow(10, vei * 0.5) * 5, // km
                pyroclastic_flow_radius: Math.pow(2, vei) * 2, // km
                lava_flow_distance: vei * 8, // km
                evacuation_radius: Math.pow(2, vei) * 3 // km
            };
            
            impacts.affected_area = Math.PI * Math.pow(impacts.volcanic.evacuation_radius, 2);
            impacts.population_at_risk = impacts.affected_area * 150; // people/km²
            impacts.economic_losses = impacts.affected_area * 2000000; // $2M/km²
        }
        
        if (scenario.seismic.magnitude > 0) {
            // Seismic impacts based on magnitude
            const mag = scenario.seismic.magnitude;
            const depth = scenario.seismic.depth;
            
            impacts.seismic = {
                felt_radius: Math.pow(10, 0.5 * mag - 2.8), // km
                damage_radius: Math.pow(10, 0.5 * mag - 3.5), // km
                peak_ground_acceleration: Math.exp(mag - 3.5) / depth, // g
                liquefaction_susceptibility: mag > 5.5 ? 'high' : 'low'
            };
            
            const seismicArea = Math.PI * Math.pow(impacts.seismic.damage_radius, 2);
            impacts.affected_area += seismicArea;
            impacts.population_at_risk += seismicArea * 200;
            impacts.economic_losses += seismicArea * 5000000;
        }
        
        return impacts;
    }

    function displayScenarioResults(impacts) {
        const output = document.getElementById('scenario-output');
        
        let html = '<div class="scenario-summary">';
        html += `<h6>Scenario Impact Assessment</h6>`;
        html += `<p><strong>Total Affected Area:</strong> ${impacts.affected_area.toFixed(1)} km²</p>`;
        html += `<p><strong>Population at Risk:</strong> ${impacts.population_at_risk.toLocaleString()}</p>`;
        html += `<p><strong>Estimated Economic Losses:</strong> $${(impacts.economic_losses/1000000).toFixed(1)}M</p>`;
        
        if (impacts.volcanic.evacuation_radius) {
            html += '<h6>Volcanic Hazard Zones</h6>';
            html += `<p>Evacuation Radius: ${impacts.volcanic.evacuation_radius.toFixed(1)} km</p>`;
            html += `<p>Ash Fall Radius: ${impacts.volcanic.ash_fall_radius.toFixed(1)} km</p>`;
            html += `<p>Pyroclastic Flow Radius: ${impacts.volcanic.pyroclastic_flow_radius.toFixed(1)} km</p>`;
        }
        
        if (impacts.seismic.felt_radius) {
            html += '<h6>Seismic Hazard Assessment</h6>';
            html += `<p>Felt Radius: ${impacts.seismic.felt_radius.toFixed(1)} km</p>`;
            html += `<p>Damage Radius: ${impacts.seismic.damage_radius.toFixed(1)} km</p>`;
            html += `<p>Peak Ground Acceleration: ${impacts.seismic.peak_ground_acceleration.toFixed(2)} g</p>`;
        }
        
        html += '</div>';
        output.innerHTML = html;
    }

    function calculateRisk() {
        console.log('⚠️ Calculating risk assessment...');
        
        if (!window.RISK_CALCULATOR) {
            console.warn('Risk calculator not available');
            return;
        }
        
        const populationDensity = document.getElementById('population-density').value;
        const buildingStock = document.getElementById('building-stock').value;
        
        // Get vulnerability factors
        const vulnerabilityFactors = {
            population: getPopulationVulnerability(populationDensity),
            buildings: getBuildingVulnerability(buildingStock),
            critical_facilities: getCriticalFacilityCount()
        };
        
        // Calculate risk using external risk calculator
        const riskMetrics = window.RISK_CALCULATOR.calculateComprehensiveRisk(vulnerabilityFactors);
        
        displayRiskResults(riskMetrics);
        
        console.log('✅ Risk assessment complete');
    }

    function getPopulationVulnerability(density) {
        const vulnerabilities = {
            low: { factor: 0.3, count: 5000 },
            medium: { factor: 0.6, count: 25000 },
            high: { factor: 0.9, count: 100000 }
        };
        return vulnerabilities[density] || vulnerabilities.medium;
    }

    function getBuildingVulnerability(stock) {
        const vulnerabilities = {
            unreinforced: { factor: 0.9, cost_per_unit: 150000 },
            wood_frame: { factor: 0.6, cost_per_unit: 200000 },
            concrete: { factor: 0.4, cost_per_unit: 300000 },
            steel: { factor: 0.2, cost_per_unit: 400000 }
        };
        return vulnerabilities[stock] || vulnerabilities.concrete;
    }

    function getCriticalFacilityCount() {
        let count = 0;
        if (document.getElementById('hospitals').checked) count += 2;
        if (document.getElementById('schools').checked) count += 5;
        if (document.getElementById('fire-stations').checked) count += 3;
        if (document.getElementById('power-plants').checked) count += 1;
        return count;
    }

    function displayRiskResults(metrics) {
        const metricsDiv = document.getElementById('risk-metrics');
        const lossesDiv = document.getElementById('loss-estimates');
        
        metricsDiv.innerHTML = `
            <div class="risk-metric">
                <span>Overall Risk Level:</span>
                <span class="risk-${metrics.overall_risk.toLowerCase()}">${metrics.overall_risk}</span>
            </div>
            <div class="risk-metric">
                <span>Casualty Risk:</span>
                <span>${metrics.casualty_risk}</span>
            </div>
            <div class="risk-metric">
                <span>Economic Risk:</span>
                <span>${metrics.economic_risk}</span>
            </div>
        `;
        
        lossesDiv.innerHTML = `
            <div class="loss-estimate">
                <span>Expected Casualties:</span>
                <span>${metrics.expected_casualties.toLocaleString()}</span>
            </div>
            <div class="loss-estimate">
                <span>Economic Losses:</span>
                <span>$${(metrics.economic_losses/1000000).toFixed(1)}M</span>
            </div>
            <div class="loss-estimate">
                <span>Recovery Time:</span>
                <span>${metrics.recovery_time} months</span>
            </div>
        `;
    }

    function generateEvacuationZones() {
        console.log('🏃 Generating evacuation zones...');
        
        const zoneType = document.getElementById('zone-type').value;
        const canvas = document.getElementById('hazard-map-canvas');
        const ctx = canvas.getContext('2d');
        
        // Clear existing zones
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Generate evacuation zones based on hazard type
        const zones = createEvacuationZones(zoneType);
        
        // Render zones on map
        renderEvacuationZones(ctx, zones);
        
        console.log('✅ Evacuation zones generated');
    }

    function createEvacuationZones(type) {
        const centerX = 400, centerY = 300;
        const zones = [];
        
        switch (type) {
            case 'volcanic':
                zones.push(
                    { name: 'Red Zone', radius: 15, color: '#ff0000', action: 'Immediate Evacuation' },
                    { name: 'Orange Zone', radius: 25, color: '#ff8000', action: 'Prepare for Evacuation' },
                    { name: 'Yellow Zone', radius: 40, color: '#ffff00', action: 'Enhanced Monitoring' }
                );
                break;
            case 'seismic':
                zones.push(
                    { name: 'High Damage', radius: 20, color: '#ff0000', action: 'Search and Rescue' },
                    { name: 'Moderate Damage', radius: 35, color: '#ff8000', action: 'Emergency Response' },
                    { name: 'Light Damage', radius: 50, color: '#ffff00', action: 'Assessment' }
                );
                break;
            case 'tsunami':
                zones.push(
                    { name: 'Inundation Zone', radius: 30, color: '#0080ff', action: 'Immediate Evacuation' },
                    { name: 'Buffer Zone', radius: 45, color: '#80c0ff', action: 'Evacuation Preparation' }
                );
                break;
            case 'multi':
                zones.push(
                    { name: 'Extreme Risk', radius: 18, color: '#800000', action: 'Immediate Evacuation' },
                    { name: 'High Risk', radius: 30, color: '#ff0000', action: 'Evacuation Required' },
                    { name: 'Moderate Risk', radius: 45, color: '#ff8000', action: 'Evacuation Preparation' },
                    { name: 'Low Risk', radius: 60, color: '#ffff00', action: 'Enhanced Monitoring' }
                );
                break;
        }
        
        return zones.map(zone => ({
            ...zone,
            centerX,
            centerY,
            area: Math.PI * zone.radius * zone.radius,
            population: Math.PI * zone.radius * zone.radius * 150 // 150 people/km²
        }));
    }

    function renderEvacuationZones(ctx, zones) {
        // Draw zones from largest to smallest
        zones.sort((a, b) => b.radius - a.radius).forEach(zone => {
            ctx.fillStyle = zone.color + '40'; // Semi-transparent
            ctx.strokeStyle = zone.color;
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(zone.centerX, zone.centerY, zone.radius * 8, 0, 2 * Math.PI); // Scale for visualization
            ctx.fill();
            ctx.stroke();
            
            // Add zone label
            ctx.fillStyle = 'black';
            ctx.font = 'bold 12px sans-serif';
            const labelX = zone.centerX + zone.radius * 6;
            const labelY = zone.centerY - zone.radius * 6;
            ctx.fillText(zone.name, labelX, labelY);
        });
    }

    function optimizeResponse() {
        console.log('🚑 Optimizing emergency response plan...');
        
        const shelters = parseInt(document.getElementById('shelter-count').value);
        const emergencyServices = parseInt(document.getElementById('emergency-services').value);
        const medicalFacilities = parseInt(document.getElementById('medical-facilities').value);
        
        // Simple optimization: distribute resources based on population density
        const optimization = {
            shelters: distributeResources(shelters, 'shelter'),
            emergency_services: distributeResources(emergencyServices, 'emergency'),
            medical_facilities: distributeResources(medicalFacilities, 'medical'),
            total_capacity: shelters * 500 + emergencyServices * 100 + medicalFacilities * 200
        };
        
        displayOptimizationResults(optimization);
        
        console.log('✅ Response plan optimized');
    }

    function distributeResources(count, type) {
        const zones = ['high_risk', 'moderate_risk', 'low_risk'];
        const distribution = [];
        
        for (let i = 0; i < count; i++) {
            const zone = zones[i % zones.length];
            distribution.push({
                id: i + 1,
                zone: zone,
                capacity: getResourceCapacity(type),
                priority: zones.indexOf(zone) + 1
            });
        }
        
        return distribution;
    }

    function getResourceCapacity(type) {
        const capacities = {
            shelter: 500,
            emergency: 100,
            medical: 200
        };
        return capacities[type] || 100;
    }

    function displayOptimizationResults(optimization) {
        // Display results in the UI (simplified)
        console.log('Resource optimization:', optimization);
        
        // You would typically show this in a dedicated results area
        alert(`Response plan optimized!\nTotal capacity: ${optimization.total_capacity} people\nShelters: ${optimization.shelters.length}\nEmergency Services: ${optimization.emergency_services.length}\nMedical Facilities: ${optimization.medical_facilities.length}`);
    }

    function testCommunication() {
        console.log('📢 Testing communication systems...');
        
        const systems = [];
        if (document.getElementById('sms-alerts').checked) systems.push('SMS');
        if (document.getElementById('radio-broadcasts').checked) systems.push('Radio');
        if (document.getElementById('social-media').checked) systems.push('Social Media');
        if (document.getElementById('sirens').checked) systems.push('Sirens');
        
        // Simulate communication system test
        const testResults = systems.map(system => ({
            system: system,
            status: Math.random() > 0.1 ? 'Operational' : 'Failed',
            coverage: (85 + Math.random() * 15).toFixed(1) + '%',
            response_time: (Math.random() * 30 + 5).toFixed(1) + ' seconds'
        }));
        
        displayCommunicationResults(testResults);
        
        console.log('✅ Communication test complete');
    }

    function displayCommunicationResults(results) {
        let message = 'Communication System Test Results:\n\n';
        results.forEach(result => {
            message += `${result.system}: ${result.status}\n`;
            message += `  Coverage: ${result.coverage}\n`;
            message += `  Response Time: ${result.response_time}\n\n`;
        });
        
        alert(message);
    }

    function saveScenario() {
        if (activeScenario) {
            const scenarioJson = JSON.stringify(activeScenario, null, 2);
            const blob = new Blob([scenarioJson], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `hazard_scenario_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            console.log('💾 Scenario saved');
        }
    }

    function show() {
        const ui = document.getElementById('hazard-assessment-ui');
        if (ui) {
            ui.style.display = 'block';
            console.log('🚨 Hazard Assessment System opened');
        }
    }

    function hide() {
        const ui = document.getElementById('hazard-assessment-ui');
        if (ui) {
            ui.style.display = 'none';
            console.log('🚨 Hazard Assessment System closed');
        }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.altKey && e.key === 'H') {
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
        generateMultiHazardMap: generateMultiHazardMap,
        computeHazardCurves: computeHazardCurves,
        runScenario: runScenario,
        calculateRisk: calculateRisk,
        generateEvacuationZones: generateEvacuationZones,
        getHazardData: () => hazardData,
        getActiveScenario: () => activeScenario,
        HAZARD_TYPES: HAZARD_TYPES,
        VULNERABILITY_CLASSES: VULNERABILITY_CLASSES
    };

})();

// Auto-initialize
document.addEventListener('DOMContentLoaded', function() {
    window.HAZARD_ASSESSMENT.init();
    console.log('🚨 Integrated Hazard Assessment System loaded');
    console.log('🎮 Press Ctrl+Alt+H to open Hazard Assessment');
});