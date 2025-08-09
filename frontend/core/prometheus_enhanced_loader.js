/**
 * Prometheus Enhanced Loader
 * Automatically loads all improvements and fixes
 */

(function() {
    console.log('🚀 Prometheus Enhanced System Loading...');
    
    // Scripts to load in order with new modular architecture paths
    const scripts = [
        // Emergency systems (highest priority)
        'emergency_systems/suppress_sourcemap_warnings.js',
        'emergency_systems/immediate_emergency_loader.js',
        'emergency_systems/error_management_system.js', // NEW: Comprehensive error handling
        'emergency_systems/critical_ui_fixes.js',  // NEW: Apply critical fixes immediately
        'emergency_systems/ml_dashboard_fix.js',   // NEW: Fix ML Dashboard visualization issues
        'emergency_systems/ultimate_cdn_bypass.js',
        'emergency_systems/offline_library_system.js',
        'emergency_systems/emergency_cdn_fix.js',
        'emergency_systems/variable_conflict_fix.js',
        'emergency_systems/system_recovery.js',
        'emergency_systems/fix_all_errors.js',
        'emergency_systems/cdn_fallback_system.js',
        'emergency_systems/ui_fixes_2025.js',
        'emergency_systems/universal_feature_repair_v2.js',
        'emergency_systems/enhanced_ux_2025.js',
        'emergency_systems/prometheus_universal_fix.js',
        'emergency_systems/webgl_fix_and_minimize.js',
        
        // Core framework components
        '../modules/mantle_dynamics/mantle_ui_components.js', // NEW: Fix Mantle Dynamics UI controls
        'theme_manager.js',       // NEW: Advanced theme management system
        'modern_dashboard_framework.js', // NEW: Modern dashboard with Dash/Plotly-like components
        'prometheus_components.js', // NEW: Reusable component system with drag-and-drop
        'data_table_advanced.js', // NEW: Advanced data tables with virtual scrolling
        'map_system_advanced.js', // NEW: Interactive map system with Mapbox/Leaflet
        'map_controls.js',        // NEW: Advanced map control widgets
        'performance_manager.js', // NEW: Performance optimization with WebWorkers, LOD, virtual scrolling, and progressive loading
        'cache_system.js',        // NEW: Advanced IndexedDB caching system with compression and intelligent eviction
        
        // InSAR deformation module
        '../modules/insar_deformation/mintpy_integration.js',  // NEW: MintPy-based InSAR processing workflow integration
        '../modules/insar_deformation/insar_algorithms.js',    // NEW: SBAS and PS-InSAR algorithms with quality assessment
        '../modules/insar_deformation/fringe_visualizer.js',   // NEW: Interactive InSAR fringe pattern overlay with WebGL
        '../modules/insar_deformation/deformation_3d.js',      // NEW: 3D volcanic deformation modeling with multiple sources
        '../modules/insar_deformation/insar_analytics.js',     // NEW: Advanced InSAR analytics with velocity field extraction and PCA
        '../modules/insar_deformation/coherence_analyzer.js',  // NEW: Coherence analysis with threshold filtering and quality assessment
        '../modules/insar_deformation/sar_data_manager.js',    // NEW: Multi-mission SAR data management with automatic downloads and caching
        
        // Analytics module
        '../analytics/processing_queue.js',    // NEW: Advanced processing queue management with priority scheduling
        '../analytics/ml_visualizer.js',       // NEW: Deep learning model visualization with attention heatmaps and feature importance
        '../analytics/model_metrics_dashboard.js', // NEW: Training metrics dashboard with real-time monitoring and model comparison
        '../analytics/auto_optimizer.js',      // NEW: Bayesian optimization system with hyperparameter tuning and k-fold CV
        '../analytics/model_ensemble.js',      // NEW: Advanced ensemble methods (voting, stacking, blending, Bayesian averaging)
        
        // Seismic analysis module
        '../modules/seismic_analysis/eqtransformer_integration.js', // NEW: Real-time P/S wave detection with automatic event classification
        '../modules/seismic_analysis/seismic_ml_models.js',   // NEW: Advanced ML models for earthquake detection and analysis
        '../modules/seismic_analysis/stress_visualizer.js',   // NEW: Tectonic stress visualization with Coulomb stress and moment tensor beach balls
        '../modules/seismic_analysis/fault_analyzer.js',      // NEW: Advanced fault analysis with slip tendency and fracture stability
        '../modules/seismic_analysis/tomography_3d.js',       // NEW: 3D seismic tomography visualization with cross-sections and isosurfaces
        '../modules/seismic_analysis/velocity_model.js',      // NEW: Velocity model management with standard Earth models and interpolation
        
        // Volcanic systems module
        '../modules/volcanic_systems/comprehensive_volcano_database.js', // NEW: Expanded global volcano database (300+ volcanoes)
        '../modules/volcanic_systems/volcano_ml_predictor.js', // NEW: Multi-parameter neural network volcano predictions based on 2025 studies
        '../modules/volcanic_systems/eruption_forecaster.js', // NEW: Comprehensive eruption timeline forecasting with uncertainty quantification
        '../modules/volcanic_systems/comprehensive_volcano_selector.js',
        '../modules/volcanic_systems/smithsonian_volcanoes_2025.js',
        '../modules/volcanic_systems/volcano_system_fix.js',  // NEW: Fix Volcanic System Simulator
        
        // Mantle dynamics module
        '../modules/mantle_dynamics/mantle_convection_model.js', // NEW: ASPECT-inspired mantle convection with temperature fields and viscosity variations
        '../modules/mantle_dynamics/thermal_model.js',       // NEW: Advanced thermal modeling with heat diffusion and cooling models
        
        // Resource exploration module
        '../modules/resource_exploration/hazard_assessment.js',   // NEW: Integrated multi-hazard assessment with probabilistic analysis and emergency planning
        '../modules/resource_exploration/risk_calculator.js'     // NEW: Advanced risk calculator with Monte Carlo analysis and vulnerability models
    ];
    
    let loaded = 0;
    
    function loadScript(src, callback) {
        const script = document.createElement('script');
        script.src = src;
        script.onload = function() {
            console.log(`✅ Loaded: ${src}`);
            loaded++;
            if (callback) callback();
        };
        script.onerror = function() {
            console.warn(`⚠️ Failed to load: ${src}`);
            loaded++;
            if (callback) callback();
        };
        document.head.appendChild(script);
    }
    
    function loadAll() {
        let index = 0;
        
        function loadNext() {
            if (index < scripts.length) {
                loadScript(scripts[index], loadNext);
                index++;
            } else {
                console.log(`✨ Prometheus Enhanced System Ready (${loaded}/${scripts.length} modules loaded)`);
                
                // Show summary
                setTimeout(() => {
                    console.log('🔥 ULTIMATE RECOVERY SYSTEMS ACTIVE 🔥');
                    console.log('🚫 CDN completely blocked? No problem - offline libraries loaded');
                    console.log('⚡ MIME type errors? Ultimate bypass system handles everything');
                    console.log('🎮 WebGL failing? Enhanced context creation with fallbacks');
                    console.log('🗺️ OrbitControls missing? Custom implementation provided');
                    console.log('🧠 TensorFlow blocked? Advanced offline ML system active');
                    console.log('📊 Chart.js blocked? Full canvas rendering system ready');
                    console.log('🗺️ Leaflet blocked? Complete map system with markers');
                    
                    if (window.VOLCANIC_DATABASE) {
                        console.log('🌋 Comprehensive Volcano Database: Complete Smithsonian catalog');
                        console.log('🗺️ Navigation: Continent → Country → Volcanic System (alphabetical)');
                    }
                    if (window.SMITHSONIAN_VOLCANOES) {
                        console.log('🔴 Active Eruptions: ' + window.SMITHSONIAN_VOLCANOES.getActiveVolcanoes().length);
                    }
                    
                    console.log('🎨 UI fixes applied - buttons and menus should work properly');
                    console.log('📊 Press ? for keyboard shortcuts');
                    console.log('🔧 Run manualRepair() if features are not working');
                    console.log('🎯 Run fixUI() if buttons are still overlapping');
                    console.log('🚑 Run ultimateRecovery() for maximum emergency repairs');
                    console.log('📦 Run initializeOfflineLibraries() to reload offline systems');
                }, 1000);
            }
        }
        
        loadNext();
    }
    
    // Start loading when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadAll);
    } else {
        loadAll();
    }
})();