/**
 * Volcanic Systems Module Index
 * Exports all volcanic system components and utilities
 */

// Core volcanic database and analysis
export { ComprehensiveVolcanoDatabase } from './comprehensive_volcano_database.js';
export { VolcanoSelector } from './comprehensive_volcano_selector.js';
export { SmithsonianVolcanoes } from './smithsonian_volcanoes_2025.js';

// Volcanic modeling and simulation
export { MultiChamberVolcano } from './multi_chamber_volcano.js';
export { VolcanoMLPredictor } from './volcano_ml_predictor.js';
export { EruptionForecaster } from './eruption_forecaster.js';
export { MLVolcanicPrediction } from './ml_volcanic_prediction.js';

// Holocene volcano data
export { HoloceneVolcanoes } from './holocene_volcanoes.js';
export { HoloceneVolcanoesFixed } from './holocene_volcanoes_fixed.js';

// System fixes and utilities
export { VolcanoSystemFix } from './volcano_system_fix.js';

/**
 * Main volcanic systems initialization
 * @returns {Promise<Object>} Initialized volcanic systems
 */
export async function initVolcanicSystems() {
    const database = new ComprehensiveVolcanoDatabase();
    const predictor = new VolcanoMLPredictor();
    const forecaster = new EruptionForecaster();
    
    await Promise.all([
        database.init(),
        predictor.init(),
        forecaster.init()
    ]);
    
    return {
        database,
        predictor,
        forecaster
    };
}