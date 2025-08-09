// Holocene Volcano Database - Fixed Version
// Prevents redeclaration errors

(function() {
    'use strict';
    
    // Check if already defined
    if (typeof window !== 'undefined' && window.HOLOCENE_VOLCANOES) {
        console.log('HOLOCENE_VOLCANOES already loaded, skipping...');
        return;
    }
    
    // Define the database
    const HOLOCENE_VOLCANOES = {
        'United States': [
            { name: 'Mount St. Helens', lat: 46.1914, lon: -122.1956, vnum: '321050', type: 'Stratovolcano', vei: 5, lastEruption: 2008 },
            { name: 'Kilauea', lat: 19.4069, lon: -155.2834, vnum: '332010', type: 'Shield', vei: 4, lastEruption: 2023 },
            { name: 'Mount Rainier', lat: 46.8523, lon: -121.7603, vnum: '321030', type: 'Stratovolcano', vei: 4, lastEruption: 1894 },
            { name: 'Mount Shasta', lat: 41.4092, lon: -122.1949, vnum: '323010', type: 'Stratovolcano', vei: 3, lastEruption: 1786 },
            { name: 'Mount Baker', lat: 48.7767, lon: -121.8144, vnum: '321010', type: 'Stratovolcano', vei: 3, lastEruption: 1880 },
            { name: 'Yellowstone', lat: 44.4300, lon: -110.5885, vnum: '325010', type: 'Caldera', vei: 8, lastEruption: -70000 }
        ],
        'Japan': [
            { name: 'Mount Fuji', lat: 35.3606, lon: 138.7274, vnum: '283030', type: 'Stratovolcano', vei: 5, lastEruption: 1708 },
            { name: 'Sakurajima', lat: 31.5833, lon: 130.6500, vnum: '282080', type: 'Stratovolcano', vei: 4, lastEruption: 2023 },
            { name: 'Mount Aso', lat: 32.8842, lon: 131.1017, vnum: '282110', type: 'Caldera', vei: 4, lastEruption: 2021 },
            { name: 'Unzen', lat: 32.7608, lon: 130.2986, vnum: '282100', type: 'Complex', vei: 4, lastEruption: 1996 }
        ],
        'Indonesia': [
            { name: 'Krakatoa', lat: -6.1021, lon: 105.4230, vnum: '262000', type: 'Caldera', vei: 6, lastEruption: 2020 },
            { name: 'Mount Merapi', lat: -7.5407, lon: 110.4457, vnum: '263250', type: 'Stratovolcano', vei: 4, lastEruption: 2023 },
            { name: 'Tambora', lat: -8.2500, lon: 118.0000, vnum: '264040', type: 'Stratovolcano', vei: 7, lastEruption: 1967 },
            { name: 'Toba', lat: 2.6845, lon: 98.8756, vnum: '261090', type: 'Caldera', vei: 8, lastEruption: -74000 }
        ],
        'Italy': [
            { name: 'Mount Vesuvius', lat: 40.8224, lon: 14.4289, vnum: '211020', type: 'Complex', vei: 5, lastEruption: 1944 },
            { name: 'Mount Etna', lat: 37.7510, lon: 14.9934, vnum: '211060', type: 'Stratovolcano', vei: 4, lastEruption: 2023 },
            { name: 'Stromboli', lat: 38.7892, lon: 15.2133, vnum: '211040', type: 'Stratovolcano', vei: 3, lastEruption: 2023 },
            { name: 'Campi Flegrei', lat: 40.8270, lon: 14.1390, vnum: '211010', type: 'Caldera', vei: 7, lastEruption: 1538 }
        ],
        'Chile': [
            { name: 'Villarrica', lat: -39.4203, lon: -71.9397, vnum: '357120', type: 'Stratovolcano', vei: 4, lastEruption: 2023 },
            { name: 'Calbuco', lat: -41.3264, lon: -72.6147, vnum: '358020', type: 'Stratovolcano', vei: 4, lastEruption: 2015 },
            { name: 'Chaiten', lat: -42.8330, lon: -72.6460, vnum: '358041', type: 'Caldera', vei: 4, lastEruption: 2008 }
        ],
        'New Zealand': [
            { name: 'White Island', lat: -37.5208, lon: 177.1817, vnum: '241040', type: 'Stratovolcano', vei: 3, lastEruption: 2019 },
            { name: 'Ruapehu', lat: -39.2817, lon: 175.5681, vnum: '241100', type: 'Stratovolcano', vei: 3, lastEruption: 2007 },
            { name: 'Taupo', lat: -38.8200, lon: 176.0000, vnum: '241070', type: 'Caldera', vei: 7, lastEruption: 260 }
        ],
        'Iceland': [
            { name: 'Eyjafjallajokull', lat: 63.6314, lon: -19.6083, vnum: '372020', type: 'Stratovolcano', vei: 4, lastEruption: 2010 },
            { name: 'Katla', lat: 63.6330, lon: -19.0550, vnum: '372030', type: 'Subglacial', vei: 5, lastEruption: 1918 },
            { name: 'Hekla', lat: 63.9830, lon: -19.7000, vnum: '372070', type: 'Stratovolcano', vei: 5, lastEruption: 2000 },
            { name: 'Grimsvotn', lat: 64.4200, lon: -17.3300, vnum: '373010', type: 'Subglacial', vei: 4, lastEruption: 2011 }
        ]
    };
    
    // Make it globally available
    if (typeof window !== 'undefined') {
        window.HOLOCENE_VOLCANOES = HOLOCENE_VOLCANOES;
        console.log('✓ Holocene volcano database loaded successfully');
    }
    
    // Export for module systems
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = HOLOCENE_VOLCANOES;
    }
})();

// Global switchView function fix
if (typeof window !== 'undefined' && typeof window.switchView === 'undefined') {
    window.switchView = function(viewType) {
        console.log('Switching to view:', viewType);
        
        const views = {
            '2D': 'map-container',
            '3D': 'threejs-container',
            'satellite': 'satellite-container',
            'terrain': 'terrain-container'
        };
        
        // Hide all views
        Object.values(views).forEach(id => {
            const elem = document.getElementById(id);
            if (elem) elem.style.display = 'none';
        });
        
        // Show selected view
        const targetId = views[viewType];
        if (targetId) {
            const targetElem = document.getElementById(targetId);
            if (targetElem) {
                targetElem.style.display = 'block';
            }
        }
        
        // Update button states
        document.querySelectorAll('.view-btn, .view-toggle, [onclick*="switchView"]').forEach(btn => {
            btn.classList.remove('active');
            if (btn.textContent.includes(viewType) || btn.getAttribute('onclick')?.includes(viewType)) {
                btn.classList.add('active');
            }
        });
        
        // Trigger resize event for map/3D updates
        window.dispatchEvent(new Event('resize'));
        
        // Store preference
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('preferredView', viewType);
        }
    };
    
    console.log('✓ switchView function initialized');
}