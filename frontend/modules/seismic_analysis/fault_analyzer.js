/**
 * Advanced Fault Analysis System
 * Implements fault geometry analysis, slip tendency, dilation tendency,
 * fracture stability, and fault network connectivity analysis
 */

class FaultAnalyzer {
    constructor() {
        this.container = null;
        this.canvas = null;
        this.ctx = null;
        this.selectedFault = null;
        this.analysisResults = new Map();
        
        // Analysis parameters
        this.frictionCoefficient = 0.6;
        this.cohesion = 5; // MPa
        this.fluidPressure = 0; // MPa
        this.tensileStrength = 3; // MPa
        
        // Fault database
        this.faultDatabase = [];
        this.faultNetworks = [];
        this.stressTensor = null;
        
        this.initializeFaultDatabase();
        this.createInterface();
        this.setupControls();
        
        console.log('🔍 Advanced Fault Analyzer initialized');
        console.log('⚖️ Slip tendency, dilation, fracture stability analysis');
    }
    
    initializeFaultDatabase() {
        this.faultDatabase = [
            {
                id: 'SAF_segment_1',
                name: 'San Andreas Fault - Northern Segment',
                type: 'strike_slip',
                location: { lat: 37.5, lon: -122.0 },
                geometry: {
                    strike: 315,
                    dip: 90,
                    rake: 0,
                    length: 150000, // meters
                    width: 15000,
                    depth: 12000
                },
                properties: {
                    friction: 0.65,
                    cohesion: 3,
                    permeability: 1e-15, // m²
                    porosity: 0.05
                },
                kinematic: {
                    slipRate: 25, // mm/yr
                    lockingDepth: 15000,
                    lastRupture: new Date('1906-04-18'),
                    recurrenceInterval: 150 // years
                },
                stressState: {
                    resolved: 0,
                    normal: 0,
                    shear: 0,
                    slipTendency: 0,
                    dilationTendency: 0
                },
                active: true,
                segments: []
            },
            {
                id: 'thrust_fault_1',
                name: 'Reverse Fault Zone',
                type: 'reverse',
                location: { lat: 36.8, lon: -121.5 },
                geometry: {
                    strike: 45,
                    dip: 30,
                    rake: 90,
                    length: 80000,
                    width: 20000,
                    depth: 18000
                },
                properties: {
                    friction: 0.75,
                    cohesion: 8,
                    permeability: 5e-16,
                    porosity: 0.03
                },
                kinematic: {
                    slipRate: 2,
                    lockingDepth: 20000,
                    lastRupture: null,
                    recurrenceInterval: 500
                },
                stressState: {
                    resolved: 0,
                    normal: 0,
                    shear: 0,
                    slipTendency: 0,
                    dilationTendency: 0
                },
                active: true,
                segments: []
            },
            {
                id: 'normal_fault_1',
                name: 'Basin-Bounding Normal Fault',
                type: 'normal',
                location: { lat: 37.2, lon: -121.8 },
                geometry: {
                    strike: 0,
                    dip: 70,
                    rake: -90,
                    length: 60000,
                    width: 12000,
                    depth: 10000
                },
                properties: {
                    friction: 0.6,
                    cohesion: 2,
                    permeability: 2e-15,
                    porosity: 0.08
                },
                kinematic: {
                    slipRate: 0.5,
                    lockingDepth: 8000,
                    lastRupture: null,
                    recurrenceInterval: 1000
                },
                stressState: {
                    resolved: 0,
                    normal: 0,
                    shear: 0,
                    slipTendency: 0,
                    dilationTendency: 0
                },
                active: true,
                segments: []
            }
        ];
        
        // Initialize fault networks
        this.initializeFaultNetworks();
    }
    
    initializeFaultNetworks() {
        // Create fault network connectivity
        this.faultNetworks = [
            {
                id: 'network_1',
                name: 'San Andreas Fault System',
                faults: ['SAF_segment_1'],
                connectivity: 'serial',
                totalLength: 150000,
                averageStrike: 315,
                complexity: 'low'
            },
            {
                id: 'network_2',
                name: 'Thrust Belt Network',
                faults: ['thrust_fault_1'],
                connectivity: 'parallel',
                totalLength: 80000,
                averageStrike: 45,
                complexity: 'medium'
            }
        ];
    }
    
    createInterface() {
        this.container = document.createElement('div');
        this.container.id = 'fault-analyzer-panel';
        this.container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 900px;
            width: 750px;
            height: 680px;
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            z-index: 9995;
            display: none;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
        `;
        
        this.container.innerHTML = `
            <div class="fault-header" style="
                padding: 20px;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            ">
                <h3 style="margin: 0; font-size: 20px; font-weight: 600;">
                    🔍 Advanced Fault Analyzer
                </h3>
                <div class="header-controls">
                    <button id="fault-report" style="
                        background: rgba(52,152,219,0.2);
                        border: 1px solid #3498db;
                        color: #3498db;
                        padding: 8px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        cursor: pointer;
                        margin-right: 10px;
                    ">📋 Report</button>
                    <button id="close-fault-analyzer" style="
                        background: rgba(231,76,60,0.2);
                        border: 1px solid #e74c3c;
                        color: #e74c3c;
                        padding: 8px 12px;
                        border-radius: 15px;
                        cursor: pointer;
                    ">×</button>
                </div>
            </div>
            
            <div class="fault-controls" style="
                padding: 15px 20px;
                background: rgba(0,0,0,0.1);
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            ">
                <div>
                    <label style="font-size: 12px; margin-right: 8px;">Select Fault:</label>
                    <select id="fault-selector" style="
                        padding: 6px 10px;
                        background: rgba(255,255,255,0.1);
                        border: 1px solid rgba(255,255,255,0.3);
                        border-radius: 6px;
                        color: white;
                        font-size: 12px;
                        width: 200px;
                    ">
                        <option value="">Choose fault for analysis</option>
                    </select>
                </div>
                
                <div>
                    <label style="font-size: 12px; margin-right: 8px;">Analysis Type:</label>
                    <select id="analysis-type" style="
                        padding: 6px 10px;
                        background: rgba(255,255,255,0.1);
                        border: 1px solid rgba(255,255,255,0.3);
                        border-radius: 6px;
                        color: white;
                        font-size: 12px;
                    ">
                        <option value="slip_tendency">Slip Tendency</option>
                        <option value="dilation_tendency">Dilation Tendency</option>
                        <option value="fracture_stability">Fracture Stability</option>
                        <option value="stress_ratio">Stress Ratio Analysis</option>
                        <option value="connectivity">Network Connectivity</option>
                    </select>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <button id="analyze-fault" style="
                        background: linear-gradient(45deg, #34495e, #2c3e50);
                        border: none;
                        color: white;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        font-weight: 600;
                    ">🔍 Analyze</button>
                    
                    <button id="analyze-all" style="
                        background: rgba(46,213,115,0.2);
                        border: 1px solid #2ed573;
                        color: #2ed573;
                        padding: 8px 15px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                    ">🔄 Analyze All</button>
                </div>
                
                <div style="flex: 1; text-align: right;">
                    <button id="stress-parameters" style="
                        background: rgba(155,89,182,0.2);
                        border: 1px solid #9b59b6;
                        color: #9b59b6;
                        padding: 8px 12px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 12px;
                    ">⚙️ Stress Parameters</button>
                </div>
            </div>
            
            <div class="fault-visualization" style="
                display: flex;
                height: calc(100% - 140px);
                padding: 20px;
                gap: 20px;
            ">
                <div class="fault-canvas-container" style="
                    flex: 1;
                    position: relative;
                ">
                    <canvas id="fault-canvas" style="
                        width: 400px;
                        height: 350px;
                        border: 1px solid rgba(255,255,255,0.2);
                        border-radius: 10px;
                        background: #1a252f;
                    "></canvas>
                    
                    <div id="stereonet-canvas" style="
                        position: absolute;
                        top: 360px;
                        left: 0;
                        width: 200px;
                        height: 200px;
                        border: 1px solid rgba(255,255,255,0.2);
                        border-radius: 10px;
                        background: #1a252f;
                    ">
                        <canvas width="200" height="200" style="width: 100%; height: 100%;"></canvas>
                    </div>
                    
                    <div id="rose-diagram" style="
                        position: absolute;
                        top: 360px;
                        right: 0;
                        width: 180px;
                        height: 200px;
                        border: 1px solid rgba(255,255,255,0.2);
                        border-radius: 10px;
                        background: #1a252f;
                    ">
                        <canvas width="180" height="200" style="width: 100%; height: 100%;"></canvas>
                    </div>
                </div>
                
                <div class="fault-analysis-panel" style="
                    width: 280px;
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                ">
                    <div id="fault-properties" style="
                        background: rgba(0,0,0,0.3);
                        border-radius: 10px;
                        padding: 15px;
                        font-size: 12px;
                    ">
                        <div><strong>Fault Properties:</strong></div>
                        <div id="fault-info">Select a fault to view properties</div>
                    </div>
                    
                    <div id="stress-analysis-results" style="
                        background: rgba(0,0,0,0.3);
                        border-radius: 10px;
                        padding: 15px;
                        font-size: 12px;
                        flex: 1;
                        overflow-y: auto;
                    ">
                        <div><strong>Analysis Results:</strong></div>
                        <div id="analysis-output">Run analysis to see results</div>
                    </div>
                    
                    <div id="stability-assessment" style="
                        background: rgba(0,0,0,0.3);
                        border-radius: 10px;
                        padding: 15px;
                        font-size: 12px;
                    ">
                        <div><strong>Stability Assessment:</strong></div>
                        <div id="stability-output">-</div>
                    </div>
                    
                    <div id="hazard-evaluation" style="
                        background: rgba(231,76,60,0.1);
                        border: 1px solid #e74c3c;
                        border-radius: 10px;
                        padding: 10px;
                        font-size: 11px;
                    ">
                        <div><strong>🚨 Hazard Evaluation:</strong></div>
                        <div id="hazard-output">No active hazard</div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.container);
        this.canvas = this.container.querySelector('#fault-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 400;
        this.canvas.height = 350;
        
        this.populateFaultSelector();
    }
    
    populateFaultSelector() {
        const selector = this.container.querySelector('#fault-selector');
        
        this.faultDatabase.forEach(fault => {
            const option = document.createElement('option');
            option.value = fault.id;
            option.textContent = fault.name;
            selector.appendChild(option);
        });
    }
    
    setupControls() {
        const closeBtn = this.container.querySelector('#close-fault-analyzer');
        const faultSelector = this.container.querySelector('#fault-selector');
        const analyzeBtn = this.container.querySelector('#analyze-fault');
        const analyzeAllBtn = this.container.querySelector('#analyze-all');
        const reportBtn = this.container.querySelector('#fault-report');
        const stressParamsBtn = this.container.querySelector('#stress-parameters');
        
        closeBtn.addEventListener('click', () => this.hide());
        faultSelector.addEventListener('change', (e) => this.selectFault(e.target.value));
        analyzeBtn.addEventListener('click', () => this.analyzeSelectedFault());
        analyzeAllBtn.addEventListener('click', () => this.analyzeAllFaults());
        reportBtn.addEventListener('click', () => this.generateReport());
        stressParamsBtn.addEventListener('click', () => this.showStressParameters());
        
        this.makeDraggable();
    }
    
    selectFault(faultId) {
        if (!faultId) {
            this.selectedFault = null;
            this.clearFaultInfo();
            return;
        }
        
        this.selectedFault = this.faultDatabase.find(f => f.id === faultId);
        if (this.selectedFault) {
            this.displayFaultProperties();
            this.renderFaultVisualization();
            this.drawStereonet();
            this.drawRoseDiagram();
        }
    }
    
    displayFaultProperties() {
        const faultInfo = this.container.querySelector('#fault-info');
        const fault = this.selectedFault;
        
        faultInfo.innerHTML = `
            <div><strong>Name:</strong> ${fault.name}</div>
            <div><strong>Type:</strong> ${fault.type.replace('_', ' ')}</div>
            <div><strong>Location:</strong> ${fault.location.lat.toFixed(2)}°, ${fault.location.lon.toFixed(2)}°</div>
            <div style="margin-top: 10px;"><strong>Geometry:</strong></div>
            <div>Strike: ${fault.geometry.strike}°</div>
            <div>Dip: ${fault.geometry.dip}°</div>
            <div>Rake: ${fault.geometry.rake}°</div>
            <div>Length: ${(fault.geometry.length/1000).toFixed(1)} km</div>
            <div>Width: ${(fault.geometry.width/1000).toFixed(1)} km</div>
            <div>Depth: ${(fault.geometry.depth/1000).toFixed(1)} km</div>
            <div style="margin-top: 10px;"><strong>Properties:</strong></div>
            <div>Friction: ${fault.properties.friction}</div>
            <div>Cohesion: ${fault.properties.cohesion} MPa</div>
            <div>Slip Rate: ${fault.kinematic.slipRate} mm/yr</div>
            <div>Recurrence: ${fault.kinematic.recurrenceInterval} years</div>
        `;
    }
    
    clearFaultInfo() {
        const faultInfo = this.container.querySelector('#fault-info');
        faultInfo.textContent = 'Select a fault to view properties';
        
        const analysisOutput = this.container.querySelector('#analysis-output');
        analysisOutput.textContent = 'Run analysis to see results';
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    analyzeSelectedFault() {
        if (!this.selectedFault) {
            alert('Please select a fault first');
            return;
        }
        
        const analysisType = this.container.querySelector('#analysis-type').value;
        this.performFaultAnalysis(this.selectedFault, analysisType);
    }
    
    analyzeAllFaults() {
        const analysisType = this.container.querySelector('#analysis-type').value;
        
        this.faultDatabase.forEach(fault => {
            this.performFaultAnalysis(fault, analysisType);
        });
        
        this.displayBulkResults();
    }
    
    performFaultAnalysis(fault, analysisType) {
        // Set background stress tensor (would come from stress visualizer in real implementation)
        const stressTensor = {
            σ1: 50, // MPa
            σ2: 30,
            σ3: 10,
            σ1_trend: 45,
            σ1_plunge: 10,
            σ2_trend: 135,
            σ2_plunge: 20,
            σ3_trend: 225,
            σ3_plunge: 70
        };
        
        switch (analysisType) {
            case 'slip_tendency':
                this.calculateSlipTendency(fault, stressTensor);
                break;
            case 'dilation_tendency':
                this.calculateDilationTendency(fault, stressTensor);
                break;
            case 'fracture_stability':
                this.calculateFractureStability(fault, stressTensor);
                break;
            case 'stress_ratio':
                this.calculateStressRatio(fault, stressTensor);
                break;
            case 'connectivity':
                this.analyzeNetworkConnectivity(fault);
                break;
        }
        
        this.analysisResults.set(fault.id, fault.stressState);
    }
    
    calculateSlipTendency(fault, stressTensor) {
        // Transform stress tensor to fault coordinate system
        const faultNormal = this.calculateFaultNormal(fault.geometry);
        const faultStrike = this.calculateStrikeVector(fault.geometry);
        
        // Resolve stress on fault plane
        const normalStress = this.resolveNormalStress(stressTensor, faultNormal);
        const shearStress = this.resolveShearStress(stressTensor, faultNormal, faultStrike);
        
        // Calculate slip tendency
        const effectiveNormalStress = Math.max(0.1, normalStress - this.fluidPressure);
        const slipTendency = Math.abs(shearStress) / effectiveNormalStress;
        
        fault.stressState.normal = normalStress;
        fault.stressState.shear = shearStress;
        fault.stressState.slipTendency = slipTendency;
        
        // Evaluate slip tendency
        let tendency_assessment;
        if (slipTendency > fault.properties.friction) {
            tendency_assessment = 'HIGH - Critical for slip';
        } else if (slipTendency > fault.properties.friction * 0.8) {
            tendency_assessment = 'MODERATE - Close to failure';
        } else {
            tendency_assessment = 'LOW - Stable';
        }
        
        this.displayAnalysisResults(`
            <div><strong>Slip Tendency Analysis:</strong></div>
            <div>Normal Stress: ${normalStress.toFixed(2)} MPa</div>
            <div>Shear Stress: ${Math.abs(shearStress).toFixed(2)} MPa</div>
            <div>Slip Tendency: ${slipTendency.toFixed(3)}</div>
            <div>Friction Coefficient: ${fault.properties.friction}</div>
            <div style="margin-top: 10px; color: ${slipTendency > fault.properties.friction ? '#e74c3c' : '#2ecc71'};">
                <strong>Assessment: ${tendency_assessment}</strong>
            </div>
            <div style="margin-top: 10px;">
                <div>Mohr-Coulomb Criterion:</div>
                <div>τ = ${Math.abs(shearStress).toFixed(2)} MPa</div>
                <div>μσn = ${(fault.properties.friction * effectiveNormalStress).toFixed(2)} MPa</div>
                <div>Factor of Safety: ${(fault.properties.friction * effectiveNormalStress / Math.abs(shearStress)).toFixed(2)}</div>
            </div>
        `);
    }
    
    calculateDilationTendency(fault, stressTensor) {
        const faultNormal = this.calculateFaultNormal(fault.geometry);
        const normalStress = this.resolveNormalStress(stressTensor, faultNormal);
        
        // Dilation tendency = (σ1 - σn) / (σ1 - σ3)
        const dilationTendency = (stressTensor.σ1 - normalStress) / (stressTensor.σ1 - stressTensor.σ3);
        
        fault.stressState.dilationTendency = dilationTendency;
        
        let dilation_assessment;
        if (dilationTendency > 0.7) {
            dilation_assessment = 'HIGH - Significant dilation potential';
        } else if (dilationTendency > 0.4) {
            dilation_assessment = 'MODERATE - Some dilation expected';
        } else {
            dilation_assessment = 'LOW - Minimal dilation';
        }
        
        this.displayAnalysisResults(`
            <div><strong>Dilation Tendency Analysis:</strong></div>
            <div>σ1: ${stressTensor.σ1} MPa</div>
            <div>σ3: ${stressTensor.σ3} MPa</div>
            <div>σn: ${normalStress.toFixed(2)} MPa</div>
            <div>Dilation Tendency: ${dilationTendency.toFixed(3)}</div>
            <div style="margin-top: 10px; color: ${dilationTendency > 0.7 ? '#e67e22' : '#2ecc71'};">
                <strong>Assessment: ${dilation_assessment}</strong>
            </div>
            <div style="margin-top: 10px;">
                <div>Fluid pressure effects:</div>
                <div>Current Pf: ${this.fluidPressure} MPa</div>
                <div>Critical Pf for failure: ${(normalStress - Math.abs(fault.stressState.shear) / fault.properties.friction).toFixed(2)} MPa</div>
            </div>
        `);
    }
    
    calculateFractureStability(fault, stressTensor) {
        const faultNormal = this.calculateFaultNormal(fault.geometry);
        const normalStress = this.resolveNormalStress(stressTensor, faultNormal);
        const shearStress = this.resolveShearStress(stressTensor, faultNormal, this.calculateStrikeVector(fault.geometry));
        
        // Fracture stability analysis
        const effectiveNormalStress = normalStress - this.fluidPressure;
        const shearStrength = fault.properties.cohesion + fault.properties.friction * effectiveNormalStress;
        const stabilityFactor = shearStrength / Math.abs(shearStress);
        
        // Tensile failure check
        const tensileStress = -normalStress + this.fluidPressure;
        const tensileStabilityFactor = tensileStress > 0 ? this.tensileStrength / tensileStress : Infinity;
        
        const overallStability = Math.min(stabilityFactor, tensileStabilityFactor);
        
        let stability_assessment;
        if (overallStability < 1.0) {
            stability_assessment = 'UNSTABLE - Failure expected';
        } else if (overallStability < 1.5) {
            stability_assessment = 'MARGINALLY STABLE - Monitor closely';
        } else {
            stability_assessment = 'STABLE - No immediate failure risk';
        }
        
        this.displayAnalysisResults(`
            <div><strong>Fracture Stability Analysis:</strong></div>
            <div>Shear Stress: ${Math.abs(shearStress).toFixed(2)} MPa</div>
            <div>Shear Strength: ${shearStrength.toFixed(2)} MPa</div>
            <div>Shear Stability Factor: ${stabilityFactor.toFixed(2)}</div>
            <div>Tensile Stability Factor: ${tensileStabilityFactor.toFixed(2)}</div>
            <div style="margin-top: 10px; color: ${overallStability < 1.5 ? '#e74c3c' : '#2ecc71'};">
                <strong>Overall Stability: ${overallStability.toFixed(2)}</strong>
            </div>
            <div style="margin-top: 10px; color: ${overallStability < 1.5 ? '#e74c3c' : '#2ecc71'};">
                <strong>Assessment: ${stability_assessment}</strong>
            </div>
            <div style="margin-top: 10px;">
                <div>Critical parameters:</div>
                <div>Cohesion: ${fault.properties.cohesion} MPa</div>
                <div>Friction: ${fault.properties.friction}</div>
                <div>Tensile Strength: ${this.tensileStrength} MPa</div>
            </div>
        `);
        
        this.displayStabilityAssessment(stability_assessment, overallStability);
    }
    
    calculateStressRatio(fault, stressTensor) {
        // Calculate various stress ratios
        const R = (stressTensor.σ1 - stressTensor.σ2) / (stressTensor.σ1 - stressTensor.σ3); // Stress ratio
        const Φ = Math.sqrt(((stressTensor.σ1 - stressTensor.σ2)**2 + (stressTensor.σ2 - stressTensor.σ3)**2 + (stressTensor.σ3 - stressTensor.σ1)**2) / 2) / ((stressTensor.σ1 + stressTensor.σ2 + stressTensor.σ3) / 3); // Stress ratio
        
        const faultNormal = this.calculateFaultNormal(fault.geometry);
        const normalStress = this.resolveNormalStress(stressTensor, faultNormal);
        const shearStress = this.resolveShearStress(stressTensor, faultNormal, this.calculateStrikeVector(fault.geometry));
        
        const stressRatio = Math.abs(shearStress) / Math.abs(normalStress);
        
        this.displayAnalysisResults(`
            <div><strong>Stress Ratio Analysis:</strong></div>
            <div>Principal Stress Ratio (R): ${R.toFixed(3)}</div>
            <div>Shape Factor (Φ): ${Φ.toFixed(3)}</div>
            <div>Fault Stress Ratio (τ/σn): ${stressRatio.toFixed(3)}</div>
            <div style="margin-top: 10px;">
                <div>Stress regime interpretation:</div>
                <div>${this.interpretStressRegime(R, stressTensor)}</div>
            </div>
            <div style="margin-top: 10px;">
                <div>Fault orientation favorability:</div>
                <div>${this.assessFaultOrientation(fault, stressTensor)}</div>
            </div>
        `);
    }
    
    analyzeNetworkConnectivity(fault) {
        const network = this.faultNetworks.find(n => n.faults.includes(fault.id));
        
        if (network) {
            const connectivityMetrics = this.calculateConnectivityMetrics(network);
            
            this.displayAnalysisResults(`
                <div><strong>Network Connectivity Analysis:</strong></div>
                <div>Network: ${network.name}</div>
                <div>Connectivity Type: ${network.connectivity}</div>
                <div>Total Length: ${(network.totalLength/1000).toFixed(1)} km</div>
                <div>Average Strike: ${network.averageStrike}°</div>
                <div>Complexity: ${network.complexity}</div>
                <div style="margin-top: 10px;">
                    <div>Connectivity Metrics:</div>
                    <div>Node Degree: ${connectivityMetrics.nodeDegree}</div>
                    <div>Branch Intensity: ${connectivityMetrics.branchIntensity.toFixed(3)} /km²</div>
                    <div>Network Efficiency: ${connectivityMetrics.efficiency.toFixed(3)}</div>
                </div>
                <div style="margin-top: 10px;">
                    <div>Seismic Hazard Implications:</div>
                    <div>${connectivityMetrics.hazardAssessment}</div>
                </div>
            `);
        }
    }
    
    calculateConnectivityMetrics(network) {
        // Simplified connectivity metrics
        const nodeDegree = network.faults.length;
        const area = (network.totalLength / 1000) * 20; // Simplified area calculation
        const branchIntensity = nodeDegree / area;
        const efficiency = network.connectivity === 'serial' ? 0.8 : 0.6;
        
        let hazardAssessment;
        if (branchIntensity > 0.1 && efficiency > 0.7) {
            hazardAssessment = 'HIGH - Well-connected network with high rupture potential';
        } else if (branchIntensity > 0.05) {
            hazardAssessment = 'MODERATE - Some connectivity, isolated ruptures likely';
        } else {
            hazardAssessment = 'LOW - Poorly connected, individual fault behavior';
        }
        
        return {
            nodeDegree,
            branchIntensity,
            efficiency,
            hazardAssessment
        };
    }
    
    calculateFaultNormal(geometry) {
        const strike = geometry.strike * Math.PI / 180;
        const dip = geometry.dip * Math.PI / 180;
        
        return {
            x: Math.sin(dip) * Math.sin(strike + Math.PI / 2),
            y: Math.sin(dip) * Math.cos(strike + Math.PI / 2),
            z: Math.cos(dip)
        };
    }
    
    calculateStrikeVector(geometry) {
        const strike = geometry.strike * Math.PI / 180;
        
        return {
            x: Math.cos(strike),
            y: Math.sin(strike),
            z: 0
        };
    }
    
    resolveNormalStress(stressTensor, normal) {
        // Simplified normal stress calculation
        // In full implementation, would use complete stress tensor transformation
        return stressTensor.σ1 * (normal.z**2) + stressTensor.σ2 * (normal.y**2) + stressTensor.σ3 * (normal.x**2);
    }
    
    resolveShearStress(stressTensor, normal, strike) {
        // Simplified shear stress calculation
        const shearMagnitude = Math.sqrt((stressTensor.σ1 - stressTensor.σ3)**2 - this.resolveNormalStress(stressTensor, normal)**2);
        return shearMagnitude * 0.5; // Simplified
    }
    
    interpretStressRegime(R, stressTensor) {
        if (R > 0.75) {
            return 'Strike-slip regime (σ2 ≈ σ3)';
        } else if (R < 0.25) {
            return 'Extensional regime (σ1 ≈ σ2)';
        } else {
            return 'Compressional regime (intermediate)';
        }
    }
    
    assessFaultOrientation(fault, stressTensor) {
        const optimalStrike = stressTensor.σ1_trend + 30; // Simplified optimal orientation
        const strikeDiff = Math.abs(fault.geometry.strike - optimalStrike);
        const orientationDiff = Math.min(strikeDiff, 180 - strikeDiff);
        
        if (orientationDiff < 15) {
            return 'OPTIMAL - Well oriented for failure';
        } else if (orientationDiff < 30) {
            return 'GOOD - Reasonably oriented';
        } else {
            return 'POOR - Unfavorably oriented';
        }
    }
    
    displayAnalysisResults(html) {
        const output = this.container.querySelector('#analysis-output');
        output.innerHTML = html;
    }
    
    displayStabilityAssessment(assessment, factor) {
        const stability = this.container.querySelector('#stability-output');
        const hazard = this.container.querySelector('#hazard-output');
        
        stability.textContent = `Stability Factor: ${factor.toFixed(2)}`;
        
        if (factor < 1.0) {
            hazard.innerHTML = '🔴 HIGH RISK - Failure likely';
            hazard.style.borderColor = '#e74c3c';
        } else if (factor < 1.5) {
            hazard.innerHTML = '🟡 MODERATE RISK - Monitor closely';
            hazard.style.borderColor = '#f39c12';
        } else {
            hazard.innerHTML = '🟢 LOW RISK - Stable';
            hazard.style.borderColor = '#27ae60';
        }
    }
    
    displayBulkResults() {
        // Display summary of all fault analyses
        const criticalFaults = Array.from(this.analysisResults.entries())
            .filter(([id, state]) => state.slipTendency > 0.6)
            .map(([id]) => this.faultDatabase.find(f => f.id === id));
        
        let bulkSummary = `<div><strong>Bulk Analysis Summary:</strong></div>`;
        bulkSummary += `<div>Total Faults Analyzed: ${this.analysisResults.size}</div>`;
        bulkSummary += `<div>Critical Faults: ${criticalFaults.length}</div>`;
        
        if (criticalFaults.length > 0) {
            bulkSummary += `<div style="margin-top: 10px; color: #e74c3c;"><strong>Critical Faults:</strong></div>`;
            criticalFaults.forEach(fault => {
                bulkSummary += `<div>• ${fault.name}</div>`;
            });
        }
        
        this.displayAnalysisResults(bulkSummary);
    }
    
    renderFaultVisualization() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.selectedFault) return;
        
        const fault = this.selectedFault;
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Draw fault plane in 3D perspective (simplified)
        this.drawFaultPlane3D(fault, centerX, centerY);
        
        // Draw stress indicators
        this.drawStressIndicators(fault, centerX, centerY);
        
        // Draw coordinate system
        this.drawCoordinateSystem(centerX, centerY);
    }
    
    drawFaultPlane3D(fault, centerX, centerY) {
        const strike = fault.geometry.strike * Math.PI / 180;
        const dip = fault.geometry.dip * Math.PI / 180;
        
        // Fault plane dimensions (scaled for visualization)
        const length = Math.min(150, fault.geometry.length / 1000);
        const width = Math.min(100, fault.geometry.width / 1000);
        
        // Draw fault plane
        this.ctx.strokeStyle = '#e74c3c';
        this.ctx.fillStyle = 'rgba(231, 76, 60, 0.3)';
        this.ctx.lineWidth = 2;
        
        this.ctx.save();
        this.ctx.translate(centerX, centerY);
        this.ctx.rotate(strike);
        
        // Draw fault surface
        this.ctx.beginPath();
        this.ctx.moveTo(-length/2, -width/2 * Math.cos(dip));
        this.ctx.lineTo(length/2, -width/2 * Math.cos(dip));
        this.ctx.lineTo(length/2, width/2 * Math.cos(dip));
        this.ctx.lineTo(-length/2, width/2 * Math.cos(dip));
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
        
        // Draw dip direction indicator
        this.ctx.strokeStyle = '#3498db';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(0, 0);
        this.ctx.lineTo(0, 40);
        this.ctx.stroke();
        
        // Draw dip angle arc
        this.ctx.strokeStyle = '#f39c12';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 30, -Math.PI/2, -Math.PI/2 + dip, false);
        this.ctx.stroke();
        
        this.ctx.restore();
        
        // Labels
        this.ctx.fillStyle = 'white';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`Strike: ${fault.geometry.strike}°`, centerX, 30);
        this.ctx.fillText(`Dip: ${fault.geometry.dip}°`, centerX, 45);
        this.ctx.fillText(`Rake: ${fault.geometry.rake}°`, centerX, 60);
    }
    
    drawStressIndicators(fault, centerX, centerY) {
        if (fault.stressState.slipTendency > 0) {
            // Draw stress arrows
            const stressScale = fault.stressState.slipTendency * 50;
            
            // Normal stress
            this.ctx.strokeStyle = '#9b59b6';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(centerX + 80, centerY - 20);
            this.ctx.lineTo(centerX + 80 + stressScale * 0.5, centerY - 20);
            this.ctx.stroke();
            
            // Shear stress
            this.ctx.strokeStyle = '#e67e22';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(centerX + 80, centerY + 20);
            this.ctx.lineTo(centerX + 80 + stressScale, centerY + 20);
            this.ctx.stroke();
            
            // Labels
            this.ctx.fillStyle = '#9b59b6';
            this.ctx.font = '10px Arial';
            this.ctx.fillText('σn', centerX + 90 + stressScale * 0.5, centerY - 25);
            
            this.ctx.fillStyle = '#e67e22';
            this.ctx.fillText('τ', centerX + 90 + stressScale, centerY + 15);
        }
    }
    
    drawCoordinateSystem(centerX, centerY) {
        const axisLength = 30;
        
        // X axis
        this.ctx.strokeStyle = '#e74c3c';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - axisLength, centerY + 120);
        this.ctx.lineTo(centerX, centerY + 120);
        this.ctx.stroke();
        
        // Y axis
        this.ctx.strokeStyle = '#27ae60';
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - axisLength, centerY + 120);
        this.ctx.lineTo(centerX - axisLength, centerY + 120 - axisLength);
        this.ctx.stroke();
        
        // Z axis
        this.ctx.strokeStyle = '#3498db';
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - axisLength, centerY + 120);
        this.ctx.lineTo(centerX - axisLength - 20, centerY + 120 - 15);
        this.ctx.stroke();
        
        // Labels
        this.ctx.fillStyle = 'white';
        this.ctx.font = '10px Arial';
        this.ctx.fillText('X', centerX + 5, centerY + 125);
        this.ctx.fillText('Y', centerX - axisLength - 5, centerY + 120 - axisLength - 5);
        this.ctx.fillText('Z', centerX - axisLength - 25, centerY + 120 - 20);
    }
    
    drawStereonet() {
        const stereoCanvas = this.container.querySelector('#stereonet-canvas canvas');
        const ctx = stereoCanvas.getContext('2d');
        ctx.clearRect(0, 0, 200, 200);
        
        if (!this.selectedFault) return;
        
        const centerX = 100;
        const centerY = 100;
        const radius = 90;
        
        // Draw stereonet circle
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Draw great circles (simplified)
        for (let i = 0; i < 180; i += 30) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius * Math.sin(i * Math.PI / 180), 0, Math.PI);
            ctx.stroke();
        }
        
        // Plot fault plane
        const fault = this.selectedFault;
        const strike = fault.geometry.strike * Math.PI / 180;
        const dip = fault.geometry.dip * Math.PI / 180;
        
        const poleX = centerX + radius * Math.sin(dip) * Math.sin(strike);
        const poleY = centerY - radius * Math.sin(dip) * Math.cos(strike);
        
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.arc(poleX, poleY, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Label
        ctx.fillStyle = 'white';
        ctx.font = '10px Arial';
        ctx.fillText('Stereonet', centerX - 25, 15);
        ctx.fillText('Fault Pole', centerX - 25, 190);
    }
    
    drawRoseDiagram() {
        const roseCanvas = this.container.querySelector('#rose-diagram canvas');
        const ctx = roseCanvas.getContext('2d');
        ctx.clearRect(0, 0, 180, 200);
        
        const centerX = 90;
        const centerY = 100;
        const maxRadius = 70;
        
        // Draw rose diagram circles
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        for (let r = 20; r <= maxRadius; r += 20) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
            ctx.stroke();
        }
        
        // Draw radial lines
        for (let angle = 0; angle < 360; angle += 30) {
            const radian = angle * Math.PI / 180;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(centerX + maxRadius * Math.cos(radian), centerY + maxRadius * Math.sin(radian));
            ctx.stroke();
        }
        
        // Plot fault strikes (simplified - would use actual fault database)
        const strikes = this.faultDatabase.map(f => f.geometry.strike);
        const binSize = 30;
        const bins = new Array(360 / binSize).fill(0);
        
        strikes.forEach(strike => {
            const binIndex = Math.floor(strike / binSize);
            bins[binIndex]++;
        });
        
        const maxCount = Math.max(...bins);
        
        ctx.fillStyle = 'rgba(231, 76, 60, 0.7)';
        bins.forEach((count, index) => {
            if (count > 0) {
                const angle1 = (index * binSize) * Math.PI / 180;
                const angle2 = ((index + 1) * binSize) * Math.PI / 180;
                const radius = (count / maxCount) * maxRadius;
                
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.arc(centerX, centerY, radius, angle1, angle2);
                ctx.fill();
            }
        });
        
        // Label
        ctx.fillStyle = 'white';
        ctx.font = '10px Arial';
        ctx.fillText('Rose Diagram', centerX - 30, 15);
        ctx.fillText('Strike Distribution', centerX - 40, 190);
    }
    
    showStressParameters() {
        const params = prompt(`Enter stress parameters (comma-separated):
Current: μ=${this.frictionCoefficient}, C=${this.cohesion}MPa, Pf=${this.fluidPressure}MPa, T=${this.tensileStrength}MPa
Format: friction,cohesion,fluid_pressure,tensile_strength`);
        
        if (params) {
            const values = params.split(',').map(v => parseFloat(v.trim()));
            if (values.length === 4 && values.every(v => !isNaN(v))) {
                this.frictionCoefficient = values[0];
                this.cohesion = values[1];
                this.fluidPressure = values[2];
                this.tensileStrength = values[3];
                
                alert('Stress parameters updated successfully!');
            } else {
                alert('Invalid input format. Please use: friction,cohesion,fluid_pressure,tensile_strength');
            }
        }
    }
    
    generateReport() {
        let report = `FAULT ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}

ANALYSIS PARAMETERS:
- Friction Coefficient: ${this.frictionCoefficient}
- Cohesion: ${this.cohesion} MPa
- Fluid Pressure: ${this.fluidPressure} MPa
- Tensile Strength: ${this.tensileStrength} MPa

FAULT DATABASE SUMMARY:
Total Faults: ${this.faultDatabase.length}
Active Faults: ${this.faultDatabase.filter(f => f.active).length}

INDIVIDUAL FAULT RESULTS:
`;
        
        this.faultDatabase.forEach(fault => {
            const results = this.analysisResults.get(fault.id);
            if (results) {
                report += `
${fault.name}:
- Type: ${fault.type}
- Strike/Dip/Rake: ${fault.geometry.strike}°/${fault.geometry.dip}°/${fault.geometry.rake}°
- Slip Tendency: ${results.slipTendency ? results.slipTendency.toFixed(3) : 'N/A'}
- Dilation Tendency: ${results.dilationTendency ? results.dilationTendency.toFixed(3) : 'N/A'}
- Normal Stress: ${results.normal ? results.normal.toFixed(2) : 'N/A'} MPa
- Shear Stress: ${results.shear ? Math.abs(results.shear).toFixed(2) : 'N/A'} MPa
`;
            }
        });
        
        // Download report
        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fault_analysis_report_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    makeDraggable() {
        const header = this.container.querySelector('.fault-header');
        let isDragging = false;
        let startX, startY, startLeft, startTop;
        
        header.style.cursor = 'move';
        
        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = parseInt(window.getComputedStyle(this.container).left);
            startTop = parseInt(window.getComputedStyle(this.container).top);
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const newLeft = startLeft + e.clientX - startX;
            const newTop = startTop + e.clientY - startY;
            
            this.container.style.left = Math.max(0, Math.min(window.innerWidth - 750, newLeft)) + 'px';
            this.container.style.top = Math.max(0, Math.min(window.innerHeight - 680, newTop)) + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }
    
    show() {
        this.container.style.display = 'block';
        this.container.style.animation = 'fadeIn 0.3s ease-out';
        
        if (this.selectedFault) {
            this.renderFaultVisualization();
            this.drawStereonet();
            this.drawRoseDiagram();
        }
    }
    
    hide() {
        this.container.style.animation = 'fadeOut 0.3s ease-in';
        setTimeout(() => {
            this.container.style.display = 'none';
        }, 300);
    }
}

// Add CSS animations
const faultAnalyzerStyle = document.createElement('style');
faultAnalyzerStyle.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.95); }
    }
    
    #fault-analyzer-panel select option {
        background: #2c3e50;
        color: white;
    }
    
    #fault-analyzer-panel::-webkit-scrollbar {
        width: 8px;
    }
    
    #fault-analyzer-panel::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    #fault-analyzer-panel::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }
    
    #fault-analyzer-panel::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
`;
document.head.appendChild(faultAnalyzerStyle);

// Global instance
window.faultAnalyzer = new FaultAnalyzer();

// Integration with stress visualizer
if (window.stressVisualizer) {
    window.stressVisualizer.faultAnalyzer = window.faultAnalyzer;
}

// Keyboard shortcut to toggle fault analyzer
document.addEventListener('keydown', (e) => {
    if (e.key === 'F' && e.altKey && e.ctrlKey) {
        e.preventDefault();
        if (window.faultAnalyzer.container.style.display === 'none') {
            window.faultAnalyzer.show();
        } else {
            window.faultAnalyzer.hide();
        }
    }
});

console.log('🔍 Advanced Fault Analyzer loaded successfully!');
console.log('🔑 Press Ctrl+Alt+F to toggle fault analyzer');
console.log('🎯 Access via: window.faultAnalyzer.show()');
console.log('⚖️ Features: Slip tendency, dilation tendency, fracture stability, stress ratios, network connectivity');