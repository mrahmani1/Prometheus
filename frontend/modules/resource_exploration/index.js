/**
 * Resource Exploration Module Index
 * Exports all resource exploration and hazard assessment components
 */

// GPS and satellite monitoring
export { GPSVelocityField } from './gps_velocity_field.js';
export { SatelliteRealtime } from './satellite_realtime.js';

// Hazard and risk assessment
export { HazardAssessment } from './hazard_assessment.js';
export { RiskCalculator } from './risk_calculator.js';

/**
 * Main resource exploration initialization
 * @returns {Promise<Object>} Initialized resource exploration systems
 */
export async function initResourceExploration() {
    const gpsField = new GPSVelocityField();
    const satelliteMonitor = new SatelliteRealtime();
    const hazardAssessment = new HazardAssessment();
    const riskCalculator = new RiskCalculator();
    
    await Promise.all([
        gpsField.init(),
        satelliteMonitor.init(),
        hazardAssessment.init(),
        riskCalculator.init()
    ]);
    
    return {
        gpsField,
        satelliteMonitor,
        hazardAssessment,
        riskCalculator
    };
}