/**
 * Frontend Core Module Index
 * Exports all core framework components and utilities
 */

// Core framework components
export { PrometheusComponents } from './prometheus_components.js';
export { ModernDashboardFramework } from './modern_dashboard_framework.js';
export { ThemeManager } from './theme_manager.js';

// Data and visualization components
export { DataTableAdvanced } from './data_table_advanced.js';
export { MapSystemAdvanced } from './map_system_advanced.js';
export { MapControls } from './map_controls.js';

// System utilities
export { GlobalFunctionsUtility } from './global_functions_utility.js';
export { AutomaticRecoverySystem } from './automatic_recovery_system.js';
export { PerformanceManager } from './performance_manager.js';
export { CacheSystem } from './cache_system.js';
export { RemoteMonitor } from './remote_monitor.js';

// Testing utilities
export { TestJS } from './test_js.js';

/**
 * Main core framework initialization
 * @returns {Promise<Object>} Initialized core systems
 */
export async function initCore() {
    const components = new PrometheusComponents();
    const dashboard = new ModernDashboardFramework();
    const themeManager = new ThemeManager();
    const dataTable = new DataTableAdvanced();
    const mapSystem = new MapSystemAdvanced();
    const performanceManager = new PerformanceManager();
    const cacheSystem = new CacheSystem();
    
    await Promise.all([
        components.init(),
        dashboard.init(),
        themeManager.init(),
        dataTable.init(),
        mapSystem.init(),
        performanceManager.init(),
        cacheSystem.init()
    ]);
    
    return {
        components,
        dashboard,
        themeManager,
        dataTable,
        mapSystem,
        performanceManager,
        cacheSystem
    };
}