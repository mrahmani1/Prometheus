/**
 * Seismic Analysis Module Index
 * Exports all seismic analysis components and utilities
 */

// Machine learning models for seismic analysis
export { SeismicMLModels } from './seismic_ml_models.js';
export { MLEarthquakePrediction } from './ml_earthquake_prediction.js';
export { EQTransformerIntegration } from './eqtransformer_integration.js';

// Stress and fault analysis
export { StressVisualizer } from './stress_visualizer.js';
export { FaultAnalyzer } from './fault_analyzer.js';

// 3D visualization and modeling
export { Tomography3D } from './tomography_3d.js';
export { VelocityModel } from './velocity_model.js';

/**
 * Main seismic analysis initialization
 * @returns {Promise<Object>} Initialized seismic analysis systems
 */
export async function initSeismicAnalysis() {
    const mlModels = new SeismicMLModels();
    const stressVisualizer = new StressVisualizer();
    const faultAnalyzer = new FaultAnalyzer();
    const tomography = new Tomography3D();
    
    await Promise.all([
        mlModels.init(),
        stressVisualizer.init(),
        faultAnalyzer.init(),
        tomography.init()
    ]);
    
    return {
        mlModels,
        stressVisualizer,
        faultAnalyzer,
        tomography
    };
}