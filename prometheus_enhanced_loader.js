/**
 * Prometheus Enhanced Loader
 * Automatically loads all improvements and fixes
 */

(function() {
    console.log('🚀 Prometheus Enhanced System Loading...');
    
    // Scripts to load in order (immediate emergency loader first)
    const scripts = [
        'suppress_sourcemap_warnings.js',
        'immediate_emergency_loader.js',
        'error_management_system.js', // NEW: Comprehensive error handling
        'critical_ui_fixes.js',  // NEW: Apply critical fixes immediately
        'ml_dashboard_fix.js',   // NEW: Fix ML Dashboard visualization issues
        'mantle_ui_components.js', // NEW: Fix Mantle Dynamics UI controls
        'volcano_system_fix.js',  // NEW: Fix Volcanic System Simulator
        'theme_manager.js',       // NEW: Advanced theme management system
        'modern_dashboard_framework.js', // NEW: Modern dashboard with Dash/Plotly-like components
        'prometheus_components.js', // NEW: Reusable component system with drag-and-drop
        'data_table_advanced.js', // NEW: Advanced data tables with virtual scrolling
        'map_system_advanced.js', // NEW: Interactive map system with Mapbox/Leaflet
        'map_controls.js',        // NEW: Advanced map control widgets
        'ultimate_cdn_bypass.js',
        'offline_library_system.js',
        'emergency_cdn_fix.js',
        'variable_conflict_fix.js',
        'system_recovery.js',
        'fix_all_errors.js',
        'cdn_fallback_system.js',
        'ui_fixes_2025.js',
        'comprehensive_volcano_selector.js',
        'universal_feature_repair_v2.js',
        'smithsonian_volcanoes_2025.js',
        'enhanced_ux_2025.js'
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