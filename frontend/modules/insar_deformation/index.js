/**
 * InSAR Deformation Module Index
 * Exports all InSAR processing and deformation analysis components
 */

// Core InSAR algorithms and processing
export { InSARAlgorithms } from './insar_algorithms.js';
export { InSARAnalytics } from './insar_analytics.js';
export { MintPyIntegration } from './mintpy_integration.js';

// Data management and processing
export { SARDataManager } from './sar_data_manager.js';
export { CoherenceAnalyzer } from './coherence_analyzer.js';

// Visualization and modeling
export { FringeVisualizer } from './fringe_visualizer.js';
export { Deformation3D } from './deformation_3d.js';

// Monitoring and utilities
export { InSARMonitor } from './insar_monitor.js';
export { InSARIssueFix } from './fix_insar_issues.js';

/**
 * Main InSAR deformation analysis initialization
 * @returns {Promise<Object>} Initialized InSAR systems
 */
export async function initInSARDeformation() {
    const algorithms = new InSARAlgorithms();
    const analytics = new InSARAnalytics();
    const dataManager = new SARDataManager();
    const coherenceAnalyzer = new CoherenceAnalyzer();
    const visualizer = new FringeVisualizer();
    
    await Promise.all([
        algorithms.init(),
        analytics.init(),
        dataManager.init(),
        coherenceAnalyzer.init(),
        visualizer.init()
    ]);
    
    return {
        algorithms,
        analytics,
        dataManager,
        coherenceAnalyzer,
        visualizer
    };
}