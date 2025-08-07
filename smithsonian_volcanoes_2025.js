/**
 * Smithsonian Volcanoes Database 2025
 * Enhanced volcano data from Global Volcanism Program
 * Including recent 2024-2025 volcanic activity updates
 */

window.SMITHSONIAN_VOLCANOES = {
    metadata: {
        source: "Smithsonian Institution Global Volcanism Program",
        updated: "2025-01-07",
        total_volcanoes: 1350,
        active_since_2024: 48
    },
    
    // Recent 2024-2025 eruptions
    recent_activity: [
        { name: "Reykjanes", country: "Iceland", lat: 63.88, lon: -22.40, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Kilauea", country: "USA (Hawaii)", lat: 19.421, lon: -155.287, last_eruption: "2024-12", vei: 0, status: "erupting" },
        { name: "Marapi", country: "Indonesia", lat: -0.38, lon: 100.47, last_eruption: "2024-11", vei: 3, status: "unrest" },
        { name: "Lewotobi", country: "Indonesia", lat: -8.542, lon: 122.775, last_eruption: "2024-11", vei: 3, status: "erupting" },
        { name: "Etna", country: "Italy", lat: 37.734, lon: 15.004, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Stromboli", country: "Italy", lat: 38.789, lon: 15.213, last_eruption: "2024-12", vei: 1, status: "erupting" },
        { name: "Popocatepetl", country: "Mexico", lat: 19.023, lon: -98.622, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Fuego", country: "Guatemala", lat: 14.473, lon: -90.880, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Sangay", country: "Ecuador", lat: -2.005, lon: -78.341, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Villarrica", country: "Chile", lat: -39.42, lon: -71.93, last_eruption: "2024-11", vei: 1, status: "unrest" },
        { name: "Semeru", country: "Indonesia", lat: -8.108, lon: 112.922, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Merapi", country: "Indonesia", lat: -7.54, lon: 110.44, last_eruption: "2024-11", vei: 2, status: "unrest" },
        { name: "Shiveluch", country: "Russia", lat: 56.653, lon: 161.360, last_eruption: "2024-12", vei: 3, status: "erupting" },
        { name: "Karymsky", country: "Russia", lat: 54.049, lon: 159.443, last_eruption: "2024-11", vei: 2, status: "erupting" },
        { name: "Sakurajima", country: "Japan", lat: 31.585, lon: 130.657, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Suwanosejima", country: "Japan", lat: 29.638, lon: 129.714, last_eruption: "2024-12", vei: 1, status: "erupting" },
        { name: "Ibu", country: "Indonesia", lat: 1.488, lon: 127.63, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Dukono", country: "Indonesia", lat: 1.693, lon: 127.894, last_eruption: "2024-12", vei: 2, status: "erupting" },
        { name: "Mayon", country: "Philippines", lat: 13.257, lon: 123.685, last_eruption: "2024-06", vei: 2, status: "unrest" },
        { name: "Taal", country: "Philippines", lat: 14.002, lon: 120.993, last_eruption: "2024-09", vei: 1, status: "unrest" }
    ],
    
    // Major volcanic regions with comprehensive data
    regions: {
        "Ring of Fire": {
            description: "Pacific Ring of Fire - 75% of world's volcanoes",
            volcano_count: 452,
            countries: ["Japan", "Indonesia", "Philippines", "Papua New Guinea", "New Zealand", "Chile", "Peru", "Ecuador", "Colombia", "Central America", "Mexico", "USA", "Canada", "Russia"],
            major_volcanoes: [
                { name: "Mount Fuji", lat: 35.361, lon: 138.731, elevation: 3776, type: "Stratovolcano", last_eruption: "1708" },
                { name: "Mount St. Helens", lat: 46.200, lon: -122.180, elevation: 2549, type: "Stratovolcano", last_eruption: "2008" },
                { name: "Krakatoa", lat: -6.102, lon: 105.423, elevation: 813, type: "Caldera", last_eruption: "2023" },
                { name: "Mount Pinatubo", lat: 15.143, lon: 120.350, elevation: 1486, type: "Stratovolcano", last_eruption: "2021" },
                { name: "Cotopaxi", lat: -0.677, lon: -78.436, elevation: 5897, type: "Stratovolcano", last_eruption: "2016" }
            ]
        },
        
        "Mediterranean": {
            description: "Mediterranean and Southern Europe",
            volcano_count: 23,
            countries: ["Italy", "Greece", "Spain", "Turkey"],
            major_volcanoes: [
                { name: "Vesuvius", lat: 40.821, lon: 14.426, elevation: 1281, type: "Complex", last_eruption: "1944" },
                { name: "Santorini", lat: 36.404, lon: 25.396, elevation: 367, type: "Caldera", last_eruption: "1950" },
                { name: "Campi Flegrei", lat: 40.827, lon: 14.139, elevation: 458, type: "Caldera", last_eruption: "1538", status: "major_unrest" }
            ]
        },
        
        "East Africa": {
            description: "East African Rift System",
            volcano_count: 85,
            countries: ["Ethiopia", "Kenya", "Tanzania", "Uganda", "DRC"],
            major_volcanoes: [
                { name: "Nyiragongo", lat: -1.520, lon: 29.250, elevation: 3470, type: "Stratovolcano", last_eruption: "2023" },
                { name: "Kilimanjaro", lat: -3.076, lon: 37.353, elevation: 5895, type: "Stratovolcano", last_eruption: "Pleistocene" },
                { name: "Erta Ale", lat: 13.601, lon: 40.667, elevation: 613, type: "Shield", last_eruption: "2024" },
                { name: "Ol Doinyo Lengai", lat: -2.764, lon: 35.914, elevation: 2962, type: "Stratovolcano", last_eruption: "2024" }
            ]
        },
        
        "Caribbean": {
            description: "Caribbean Volcanic Arc",
            volcano_count: 27,
            countries: ["Montserrat", "Martinique", "Guadeloupe", "St. Vincent", "Dominica", "St. Lucia", "Grenada"],
            major_volcanoes: [
                { name: "Soufrière Hills", lat: 16.720, lon: -62.180, elevation: 915, type: "Stratovolcano", last_eruption: "2013" },
                { name: "Mount Pelée", lat: 14.809, lon: -61.165, elevation: 1397, type: "Stratovolcano", last_eruption: "1932" },
                { name: "La Soufrière (St. Vincent)", lat: 13.330, lon: -61.180, elevation: 1234, type: "Stratovolcano", last_eruption: "2021" }
            ]
        },
        
        "Iceland": {
            description: "Mid-Atlantic Ridge and Iceland Hotspot",
            volcano_count: 35,
            countries: ["Iceland"],
            major_volcanoes: [
                { name: "Eyjafjallajökull", lat: 63.630, lon: -19.620, elevation: 1666, type: "Stratovolcano", last_eruption: "2010" },
                { name: "Katla", lat: 63.630, lon: -19.050, elevation: 1512, type: "Subglacial", last_eruption: "1918", status: "high_risk" },
                { name: "Hekla", lat: 63.980, lon: -19.700, elevation: 1491, type: "Stratovolcano", last_eruption: "2000" },
                { name: "Bárðarbunga", lat: 64.641, lon: -17.528, elevation: 2009, type: "Stratovolcano", last_eruption: "2015" },
                { name: "Grímsvötn", lat: 64.416, lon: -17.316, elevation: 1725, type: "Subglacial", last_eruption: "2011" }
            ]
        }
    },
    
    // Volcanic hazard monitoring priorities (2025)
    high_risk_volcanoes: [
        { name: "Vesuvius", country: "Italy", population_at_risk: 3000000, hazard_level: "very_high" },
        { name: "Campi Flegrei", country: "Italy", population_at_risk: 1500000, hazard_level: "very_high", note: "Bradyseismic crisis 2024" },
        { name: "Mount Rainier", country: "USA", population_at_risk: 2000000, hazard_level: "high" },
        { name: "Popocatepetl", country: "Mexico", population_at_risk: 25000000, hazard_level: "very_high" },
        { name: "Galeras", country: "Colombia", population_at_risk: 500000, hazard_level: "high" },
        { name: "Merapi", country: "Indonesia", population_at_risk: 24000000, hazard_level: "very_high" },
        { name: "Nyiragongo", country: "DRC", population_at_risk: 2000000, hazard_level: "very_high" },
        { name: "Sakurajima", country: "Japan", population_at_risk: 700000, hazard_level: "high" },
        { name: "Taal", country: "Philippines", population_at_risk: 25000000, hazard_level: "high" },
        { name: "Yellowstone", country: "USA", population_at_risk: 100000000, hazard_level: "low", note: "Supervolcano - closely monitored" }
    ],
    
    // New monitoring technologies (2024-2025)
    monitoring_advances: {
        satellite_systems: [
            "Sentinel-1 InSAR (ESA)",
            "NISAR (NASA/ISRO) - Launched 2024",
            "COSMO-SkyMed Second Generation",
            "ALOS-4 (JAXA) - 2024"
        ],
        ai_systems: [
            "Deep learning for eruption forecasting",
            "Pattern recognition in seismic data",
            "Automated ash plume tracking",
            "Multi-parameter anomaly detection"
        ],
        ground_networks: [
            "Real-time GPS networks",
            "Infrasound arrays",
            "Multi-gas sensors",
            "Thermal camera networks",
            "Drone surveillance systems"
        ]
    },
    
    // Helper functions
    getVolcanoByName: function(name) {
        for (const activity of this.recent_activity) {
            if (activity.name.toLowerCase() === name.toLowerCase()) {
                return activity;
            }
        }
        for (const region in this.regions) {
            for (const volcano of this.regions[region].major_volcanoes) {
                if (volcano.name.toLowerCase() === name.toLowerCase()) {
                    return volcano;
                }
            }
        }
        return null;
    },
    
    getActiveVolcanoes: function() {
        return this.recent_activity.filter(v => v.status === 'erupting');
    },
    
    getVolcanoesByCountry: function(country) {
        const results = [];
        this.recent_activity.forEach(v => {
            if (v.country.toLowerCase().includes(country.toLowerCase())) {
                results.push(v);
            }
        });
        return results;
    },
    
    getHighRiskVolcanoes: function() {
        return this.high_risk_volcanoes.filter(v => v.hazard_level === 'very_high');
    }
};

// Initialize enhanced volcano data globally
if (typeof window.HOLOCENE_VOLCANOES === 'undefined') {
    window.HOLOCENE_VOLCANOES = {};
}

// Merge Smithsonian data with existing data
window.VOLCANO_DATABASE = {
    ...window.HOLOCENE_VOLCANOES,
    smithsonian: window.SMITHSONIAN_VOLCANOES
};

console.log('🌋 Smithsonian Volcanoes Database 2025 loaded');
console.log(`📊 Total volcanoes: ${window.SMITHSONIAN_VOLCANOES.metadata.total_volcanoes}`);
console.log(`🔴 Currently erupting: ${window.SMITHSONIAN_VOLCANOES.getActiveVolcanoes().length}`);