// Variable Conflict Resolution System
// Prevents redeclaration errors and manages global variable conflicts

(function() {
    'use strict';
    
    console.log('🔧 Variable Conflict Fix loading...');
    
    // Store original variables before conflicts occur
    const originalVariables = {};
    
    // Check and backup existing variables
    if (typeof window.HOLOCENE_VOLCANOES !== 'undefined') {
        originalVariables.HOLOCENE_VOLCANOES = window.HOLOCENE_VOLCANOES;
        console.log('📦 Backed up existing HOLOCENE_VOLCANOES');
    }
    
    if (typeof window.VOLCANIC_DATABASE !== 'undefined') {
        originalVariables.VOLCANIC_DATABASE = window.VOLCANIC_DATABASE;
        console.log('📦 Backed up existing VOLCANIC_DATABASE');
    }
    
    // Override const redeclaration errors
    const originalEval = window.eval;
    window.eval = function(code) {
        try {
            // Check for const redeclarations
            if (code.includes('const HOLOCENE_VOLCANOES') && typeof window.HOLOCENE_VOLCANOES !== 'undefined') {
                console.warn('⚠️ Preventing HOLOCENE_VOLCANOES redeclaration');
                return; // Skip redeclaration
            }
            return originalEval.call(this, code);
        } catch (e) {
            if (e.message.includes('redeclaration')) {
                console.warn('🚫 Prevented variable redeclaration:', e.message);
                return;
            }
            throw e;
        }
    };
    
    // Global variable manager
    window.safeDefine = function(varName, value) {
        try {
            if (typeof window[varName] === 'undefined') {
                window[varName] = value;
                console.log(`✅ Safely defined ${varName}`);
            } else {
                console.log(`ℹ️ ${varName} already exists, skipping redefinition`);
            }
        } catch (e) {
            console.warn(`⚠️ Failed to define ${varName}:`, e);
        }
    };
    
    // Merge volcanic databases if multiple exist
    window.mergeVolcanicDatabases = function() {
        const merged = {};
        
        // Merge HOLOCENE_VOLCANOES if it exists
        if (originalVariables.HOLOCENE_VOLCANOES) {
            Object.assign(merged, originalVariables.HOLOCENE_VOLCANOES);
        }
        
        // Merge VOLCANIC_DATABASE if it exists
        if (originalVariables.VOLCANIC_DATABASE) {
            Object.assign(merged, originalVariables.VOLCANIC_DATABASE);
        }
        
        // Set the merged database
        window.UNIFIED_VOLCANIC_DATABASE = merged;
        console.log('🔄 Created unified volcanic database');
        
        return merged;
    };
    
    // Fix missing selectVolcano function
    if (typeof window.selectVolcano === 'undefined') {
        window.selectVolcano = function(volcanoId) {
            console.log('🌋 selectVolcano called for:', volcanoId);
            
            try {
                // Try to find volcano in any available database
                let volcano = null;
                
                // Check comprehensive volcano selector
                if (typeof VOLCANIC_DATABASE !== 'undefined') {
                    volcano = findVolcanoInDatabase(volcanoId, VOLCANIC_DATABASE);
                }
                
                // Check HOLOCENE_VOLCANOES
                if (!volcano && typeof HOLOCENE_VOLCANOES !== 'undefined') {
                    volcano = findVolcanoInHolocene(volcanoId, HOLOCENE_VOLCANOES);
                }
                
                // Check unified database
                if (!volcano && typeof UNIFIED_VOLCANIC_DATABASE !== 'undefined') {
                    volcano = findVolcanoInDatabase(volcanoId, UNIFIED_VOLCANIC_DATABASE);
                }
                
                if (volcano) {
                    console.log('✅ Found volcano:', volcano);
                    
                    // Update display if function exists
                    if (typeof updateVolcanoDisplay === 'function') {
                        updateVolcanoDisplay(volcano);
                    }
                    
                    // Update map if function exists
                    if (typeof updateMapView === 'function') {
                        updateMapView(volcano);
                    }
                    
                    return volcano;
                } else {
                    console.warn('⚠️ Volcano not found:', volcanoId);
                }
                
            } catch (e) {
                console.error('❌ selectVolcano error:', e);
            }
        };
    }
    
    // Helper function to find volcano in comprehensive database
    function findVolcanoInDatabase(volcanoId, database) {
        for (const continent of Object.values(database)) {
            if (typeof continent === 'object') {
                for (const country of Object.values(continent)) {
                    if (typeof country === 'object') {
                        for (const region of Object.values(country)) {
                            if (Array.isArray(region)) {
                                const found = region.find(v => 
                                    (typeof v === 'string' && v.toLowerCase().includes(volcanoId.toLowerCase())) ||
                                    (typeof v === 'object' && v.name && v.name.toLowerCase().includes(volcanoId.toLowerCase()))
                                );
                                if (found) return found;
                            }
                        }
                    }
                }
            }
        }
        return null;
    }
    
    // Helper function to find volcano in HOLOCENE_VOLCANOES
    function findVolcanoInHolocene(volcanoId, holocene) {
        if (Array.isArray(holocene)) {
            return holocene.find(v => 
                (typeof v === 'string' && v.toLowerCase().includes(volcanoId.toLowerCase())) ||
                (typeof v === 'object' && v.name && v.name.toLowerCase().includes(volcanoId.toLowerCase()))
            );
        }
        return null;
    }
    
    // Add window resize handler for map invalidation
    if (typeof window.handleResize === 'undefined') {
        window.handleResize = function() {
            try {
                // Fix map invalidateSize errors
                if (typeof map !== 'undefined' && map && typeof map.invalidateSize === 'function') {
                    map.invalidateSize();
                }
                
                // Fix camera projection matrix updates
                if (typeof window.camera !== 'undefined' && window.camera && typeof window.camera.updateProjectionMatrix === 'function') {
                    window.camera.updateProjectionMatrix();
                }
                
                console.log('🔄 Resize handled successfully');
            } catch (e) {
                console.warn('⚠️ Resize handling failed:', e);
            }
        };
        
        // Add resize listener
        window.addEventListener('resize', window.handleResize);
    }
    
    // Initialize merged databases
    setTimeout(() => {
        window.mergeVolcanicDatabases();
    }, 1000);
    
    console.log('✅ Variable Conflict Fix initialized');
})();