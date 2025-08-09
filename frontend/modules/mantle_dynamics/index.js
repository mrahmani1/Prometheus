/**
 * Mantle Dynamics Module Index
 * Exports all mantle convection and plate tectonic components
 */

// Mantle convection and thermal modeling
export { MantleConvectionModel } from './mantle_convection_model.js';
export { MantleTomography } from './mantle_tomography.js';
export { ThermalModel } from './thermal_model.js';

// Plate tectonics and reconstruction
export { PlateReconstruction } from './plate_reconstruction.js';
export { MLPlatePredictor } from './ml_plate_predictor.js';
export { GPlatesEngine } from './gplates_engine.js';

// User interface components
export { MantleUIComponents } from './mantle_ui_components.js';

/**
 * Main mantle dynamics initialization
 * @returns {Promise<Object>} Initialized mantle dynamics systems
 */
export async function initMantleDynamics() {
    const convectionModel = new MantleConvectionModel();
    const tomography = new MantleTomography();
    const thermalModel = new ThermalModel();
    const plateReconstruction = new PlateReconstruction();
    
    await Promise.all([
        convectionModel.init(),
        tomography.init(),
        thermalModel.init(),
        plateReconstruction.init()
    ]);
    
    return {
        convectionModel,
        tomography,
        thermalModel,
        plateReconstruction
    };
}