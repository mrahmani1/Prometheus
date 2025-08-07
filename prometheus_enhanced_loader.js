/**
 * Prometheus Enhanced Loader
 * Automatically loads all improvements and fixes
 */

(function() {
    console.log('🚀 Prometheus Enhanced System Loading...');
    
    // Scripts to load in order
    const scripts = [
        'suppress_sourcemap_warnings.js',
        'fix_all_errors.js',
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
                    if (window.SMITHSONIAN_VOLCANOES) {
                        console.log('🌋 Volcano Database: ' + window.SMITHSONIAN_VOLCANOES.metadata.total_volcanoes + ' volcanoes');
                        console.log('🔴 Active Eruptions: ' + window.SMITHSONIAN_VOLCANOES.getActiveVolcanoes().length);
                    }
                    console.log('📊 Press ? for keyboard shortcuts');
                    console.log('🔧 Run manualRepair() if features are not working');
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