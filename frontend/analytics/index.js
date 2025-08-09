/**
 * Analytics Module Index
 * Exports all machine learning and analytics components
 */

// ML visualization and dashboard
export { MLVisualizer } from './ml_visualizer.js';
export { ModelMetricsDashboard } from './model_metrics_dashboard.js';

// Model optimization and ensemble methods
export { AutoOptimizer } from './auto_optimizer.js';
export { ModelEnsemble } from './model_ensemble.js';

// Processing and workflow management
export { ProcessingQueue } from './processing_queue.js';

/**
 * Main analytics initialization
 * @returns {Promise<Object>} Initialized analytics systems
 */
export async function initAnalytics() {
    const visualizer = new MLVisualizer();
    const dashboard = new ModelMetricsDashboard();
    const optimizer = new AutoOptimizer();
    const ensemble = new ModelEnsemble();
    const queue = new ProcessingQueue();
    
    await Promise.all([
        visualizer.init(),
        dashboard.init(),
        optimizer.init(),
        ensemble.init(),
        queue.init()
    ]);
    
    return {
        visualizer,
        dashboard,
        optimizer,
        ensemble,
        queue
    };
}