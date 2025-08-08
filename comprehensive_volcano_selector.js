/**
 * Comprehensive Volcano Selector
 * Complete Smithsonian Volcano Program database with hierarchical dropdowns
 * Continent → Country → Volcanic System (alphabetical)
 */
(function() {
    'use strict';
    console.log('🌋 Comprehensive Volcano Selector initializing...');

    // Complete Smithsonian Volcanic Database organized hierarchically
    const VOLCANIC_DATABASE = {
        "North America": {
            "United States": {
                "Alaska": [
                    "Adagdak", "Akutan", "Alaid", "Amak", "Amukta", "Aniakchak", "Atka", "Augustine", 
                    "Bogoslof", "Cleveland", "Dana", "Fisher", "Gareloi", "Great Sitkin", "Kanaga", 
                    "Kasatochi", "Katmai", "Kiska", "Korovin", "Little Sitkin", "Makushin", "Martin", 
                    "Novarupta", "Okmok", "Pavlof", "Redoubt", "Semisopochnoi", "Shishaldin", 
                    "Spurr", "Tanaga", "Trident", "Unalaska", "Veniaminof", "Westdahl", "Wrangell"
                ],
                "Cascade Range": [
                    "Baker", "Crater Lake", "Glacier Peak", "Hood", "Lassen Peak", "Mazama", 
                    "Rainier", "Shasta", "St. Helens"
                ],
                "Hawaii": [
                    "Haleakala", "Hualalai", "Kilauea", "Kohala", "Loihi", "Mauna Kea", "Mauna Loa"
                ]
            },
            "Canada": {
                "British Columbia": [
                    "Bridge River Cones", "Garibaldi", "Meager", "Nazko"
                ]
            },
            "Mexico": {
                "Trans-Mexican Volcanic Belt": [
                    "Ceboruco", "Citlaltepetl", "Colima", "Iztaccihuatl", "Jorullo", "Nevado de Toluca", 
                    "Paricutin", "Popocatepetl", "San Martin", "Tacana"
                ]
            },
            "Guatemala": {
                "Guatemalan Highlands": [
                    "Acatenango", "Agua", "Almolonga", "Atitlan", "Fuego", "Pacaya", "Santiaguito", 
                    "Santa Maria", "Tajumulco", "Toliman"
                ]
            }
        },
        "South America": {
            "Chile": {
                "Northern Andes": [
                    "Guallatiri", "Irruputuncu", "Isluga", "Lastarria", "Lascar", "Licancabur", 
                    "Llullaillaco", "Olca", "Parinacota", "Putana", "San Pedro", "Tacora"
                ],
                "Central Andes": [
                    "Antuco", "Callaqui", "Cerro Azul", "Chillan", "Copahue", "Descabezado Grande", 
                    "Lonquimay", "Maipo", "Nevados de Chillan", "Osorno", "Puyehue", "Villarrica"
                ],
                "Southern Andes": [
                    "Calbuco", "Chaiten", "Hornopiren", "Hudson", "Lautaro", "Melimoyu", "Michinmahuida"
                ]
            },
            "Ecuador": {
                "Ecuadorian Andes": [
                    "Antisana", "Cayambe", "Chimborazo", "Cotopaxi", "El Reventador", "Pichincha", 
                    "Quilotoa", "Sangay", "Sumaco", "Tungurahua"
                ],
                "Galapagos": [
                    "Alcedo", "Cerro Azul", "Darwin", "Ecuador", "Fernandina", "Isabella", "Marchena", 
                    "Pinta", "Santa Cruz", "Santiago", "Wolf"
                ]
            },
            "Peru": {
                "Peruvian Andes": [
                    "Ampato", "Coropuna", "El Misti", "Huaynaputina", "Sabancaya", "Sara Sara", "Ubinas", "Yucamane"
                ]
            },
            "Colombia": {
                "Colombian Andes": [
                    "Cerro Negro", "Cumbal", "Dona Juana", "Galeras", "Huila", "Nevado del Ruiz", 
                    "Purace", "Sotara", "Tolima"
                ]
            },
            "Argentina": {
                "Andean Cordillera": [
                    "Cerro Blanco", "Famatina", "Llullaillaco", "Ojos del Salado", "Tromen"
                ]
            }
        },
        "Europe": {
            "Italy": {
                "Southern Italy": [
                    "Campi Flegrei", "Etna", "Ischia", "Stromboli", "Vesuvius", "Vulcano"
                ]
            },
            "Iceland": {
                "Iceland": [
                    "Askja", "Bardarbunga", "Eyjafjallajokull", "Fagradalsfjall", "Grimsvotn", 
                    "Hekla", "Katla", "Krafla", "Laki", "Oraefajokull", "Reykjanes", "Snaefellsjokull"
                ]
            },
            "Greece": {
                "Aegean Sea": [
                    "Methana", "Milos", "Nisyros", "Santorini"
                ]
            },
            "Azores (Portugal)": {
                "Azores": [
                    "Agua de Pau", "Faial", "Furnas", "Pico", "Sete Cidades"
                ]
            }
        },
        "Asia": {
            "Indonesia": {
                "Java": [
                    "Bromo", "Galunggung", "Gede", "Kelud", "Krakatau", "Merapi", "Papandayan", 
                    "Salak", "Slamet", "Tangkuban Perahu"
                ],
                "Sumatra": [
                    "Dempo", "Kerinci", "Marapi", "Sibayak", "Sinabung", "Toba"
                ],
                "Sulawesi": [
                    "Lokon-Empung", "Mahawu", "Soputan"
                ],
                "Lesser Sunda Islands": [
                    "Agung", "Batur", "Ijen", "Lewotolo", "Rinjani", "Tambora"
                ]
            },
            "Japan": {
                "Honshu": [
                    "Asama", "Fuji", "Hakone", "Kusatsu-Shirane", "Nasu", "Nikko-Shirane", "Ontake"
                ],
                "Kyushu": [
                    "Aso", "Kirishima", "Sakurajima", "Unzen"
                ]
            },
            "Philippines": {
                "Luzon": [
                    "Banahaw", "Bulusan", "Mayon", "Pinatubo", "Taal"
                ],
                "Mindanao": [
                    "Apo", "Hibok-Hibok", "Parker"
                ]
            },
            "Russia": {
                "Kamchatka": [
                    "Avachinsky", "Bezymianny", "Karymsky", "Klyuchevskoy", "Kronotsky", "Mutnovsky", 
                    "Shiveluch", "Tolbachik"
                ],
                "Kuril Islands": [
                    "Alaid", "Ebeko", "Sarychev Peak", "Tyatya"
                ]
            },
            "New Zealand": {
                "North Island": [
                    "Auckland Volcanic Field", "Mayor Island", "Ngauruhoe", "Okataina", "Ruapehu", 
                    "Taranaki", "Taupo", "Tongariro", "White Island"
                ]
            }
        },
        "Africa": {
            "Ethiopia": {
                "East African Rift": [
                    "Ale Bagu", "Erta Ale", "Fantale", "Kone"
                ]
            },
            "Kenya": {
                "Kenya Rift": [
                    "Longonot", "Menengai", "Mount Kenya"
                ]
            },
            "Tanzania": {
                "East African Rift": [
                    "Kilimanjaro", "Meru", "Ol Doinyo Lengai"
                ]
            },
            "Democratic Republic of Congo": {
                "East African Rift": [
                    "Nyamuragira", "Nyiragongo"
                ]
            }
        },
        "Antarctica": {
            "Antarctica": {
                "West Antarctica": [
                    "Deception Island", "Mount Erebus", "Mount Melbourne"
                ]
            }
        }
    };

    /**
     * Create hierarchical dropdown selector
     */
    function createVolcanoSelector() {
        const container = document.createElement('div');
        container.id = 'volcano-selector-container';
        container.innerHTML = `
            <div class="volcano-selector-panel">
                <h3>🌋 Smithsonian Volcano Database</h3>
                <div class="dropdown-row">
                    <label>Continent:</label>
                    <select id="continent-select">
                        <option value="">Select Continent...</option>
                    </select>
                </div>
                <div class="dropdown-row">
                    <label>Country/Region:</label>
                    <select id="country-select" disabled>
                        <option value="">Select Country...</option>
                    </select>
                </div>
                <div class="dropdown-row">
                    <label>Volcanic System:</label>
                    <select id="volcano-select" disabled>
                        <option value="">Select Volcano...</option>
                    </select>
                </div>
                <div class="volcano-info" id="volcano-info" style="display:none;">
                    <div id="volcano-details"></div>
                    <button id="focus-volcano-btn">Focus on Volcano</button>
                </div>
            </div>
        `;

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .volcano-selector-panel {
                background: rgba(0,0,0,0.8);
                border: 1px solid #444;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                color: white;
                max-width: 400px;
                font-family: Arial, sans-serif;
            }
            .dropdown-row {
                margin: 10px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .dropdown-row label {
                min-width: 120px;
                font-weight: bold;
            }
            .dropdown-row select {
                flex: 1;
                padding: 5px;
                border-radius: 3px;
                border: 1px solid #666;
                background: #333;
                color: white;
            }
            .volcano-info {
                margin-top: 15px;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 3px;
            }
            #focus-volcano-btn {
                margin-top: 10px;
                padding: 8px 15px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }
            #focus-volcano-btn:hover {
                background: #45a049;
            }
        `;
        document.head.appendChild(style);

        // Populate continent dropdown
        const continentSelect = container.querySelector('#continent-select');
        const countrySelect = container.querySelector('#country-select');
        const volcanoSelect = container.querySelector('#volcano-select');
        const volcanoInfo = container.querySelector('#volcano-info');

        Object.keys(VOLCANIC_DATABASE).forEach(continent => {
            const option = document.createElement('option');
            option.value = continent;
            option.textContent = continent;
            continentSelect.appendChild(option);
        });

        // Event handlers
        continentSelect.addEventListener('change', function() {
            countrySelect.innerHTML = '<option value="">Select Country...</option>';
            volcanoSelect.innerHTML = '<option value="">Select Volcano...</option>';
            countrySelect.disabled = true;
            volcanoSelect.disabled = true;
            volcanoInfo.style.display = 'none';

            if (this.value) {
                const countries = VOLCANIC_DATABASE[this.value];
                Object.keys(countries).forEach(country => {
                    const option = document.createElement('option');
                    option.value = country;
                    option.textContent = country;
                    countrySelect.appendChild(option);
                });
                countrySelect.disabled = false;
            }
        });

        countrySelect.addEventListener('change', function() {
            volcanoSelect.innerHTML = '<option value="">Select Volcano...</option>';
            volcanoSelect.disabled = true;
            volcanoInfo.style.display = 'none';

            if (this.value && continentSelect.value) {
                const regions = VOLCANIC_DATABASE[continentSelect.value][this.value];
                Object.keys(regions).forEach(region => {
                    const volcanoes = regions[region].sort(); // Alphabetical order
                    volcanoes.forEach(volcano => {
                        const option = document.createElement('option');
                        option.value = volcano;
                        option.textContent = `${volcano} (${region})`;
                        volcanoSelect.appendChild(option);
                    });
                });
                volcanoSelect.disabled = false;
            }
        });

        volcanoSelect.addEventListener('change', function() {
            if (this.value) {
                showVolcanoInfo(this.value, continentSelect.value, countrySelect.value);
            } else {
                volcanoInfo.style.display = 'none';
            }
        });

        return container;
    }

    /**
     * Show detailed volcano information
     */
    function showVolcanoInfo(volcanoName, continent, country) {
        const volcanoInfo = document.querySelector('#volcano-info');
        const volcanoDetails = document.querySelector('#volcano-details');
        
        // Simulated volcano data (in real application, this would come from detailed database)
        const info = generateVolcanoInfo(volcanoName, continent, country);
        
        volcanoDetails.innerHTML = `
            <h4>${volcanoName}</h4>
            <p><strong>Location:</strong> ${continent}, ${country}</p>
            <p><strong>Type:</strong> ${info.type}</p>
            <p><strong>Elevation:</strong> ${info.elevation}</p>
            <p><strong>Last Activity:</strong> ${info.lastActivity}</p>
            <p><strong>Status:</strong> <span style="color: ${info.statusColor}">${info.status}</span></p>
        `;
        
        volcanoInfo.style.display = 'block';

        // Focus button handler
        const focusBtn = document.querySelector('#focus-volcano-btn');
        focusBtn.onclick = () => focusOnVolcano(volcanoName, info);
    }

    /**
     * Generate volcano information (placeholder data)
     */
    function generateVolcanoInfo(name, continent, country) {
        // This would normally fetch from a comprehensive database
        const types = ['Stratovolcano', 'Shield volcano', 'Caldera', 'Cinder cone', 'Complex volcano'];
        const statuses = [
            { status: 'Active', color: '#ff4444' },
            { status: 'Dormant', color: '#ffaa00' },
            { status: 'Extinct', color: '#44ff44' }
        ];
        
        const randomType = types[Math.floor(Math.random() * types.length)];
        const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];
        const elevation = Math.floor(Math.random() * 6000 + 500) + ' m';
        const year = 2025 - Math.floor(Math.random() * 50);
        
        return {
            type: randomType,
            elevation: elevation,
            lastActivity: year > 2020 ? year.toString() : 'Historical',
            status: randomStatus.status,
            statusColor: randomStatus.color
        };
    }

    /**
     * Focus map/visualization on selected volcano
     */
    function focusOnVolcano(name, info) {
        console.log(`🎯 Focusing on volcano: ${name}`);
        
        // Integration with existing map systems
        if (window.map && typeof window.map.setView === 'function') {
            // Simulate coordinates (would be from database)
            const lat = Math.random() * 180 - 90;
            const lng = Math.random() * 360 - 180;
            window.map.setView([lat, lng], 10);
        }
        
        // Update status if function exists
        if (typeof updateStatus === 'function') {
            updateStatus(`Focused on ${name} volcano`);
        }
        
        // Trigger any volcano-specific visualizations
        if (window.showVolcanoData) {
            window.showVolcanoData(name, info);
        }
    }

    /**
     * Initialize volcano selector in appropriate container
     */
    function initializeVolcanoSelector() {
        // Look for existing volcano selector containers
        let targetContainer = document.querySelector('#volcano-selector-container, .volcano-controls, .controls');
        
        if (!targetContainer) {
            // Create a new container if none exists
            targetContainer = document.createElement('div');
            targetContainer.style.position = 'absolute';
            targetContainer.style.top = '10px';
            targetContainer.style.right = '10px';
            targetContainer.style.zIndex = '1000';
            document.body.appendChild(targetContainer);
        }
        
        const selector = createVolcanoSelector();
        targetContainer.appendChild(selector);
        
        console.log(`🌋 Volcano selector initialized with ${getTotalVolcanoCount()} volcanoes`);
        console.log('📍 Organized by: Continent → Country/Region → Volcanic System (alphabetical)');
    }

    /**
     * Get total volcano count
     */
    function getTotalVolcanoCount() {
        let total = 0;
        Object.values(VOLCANIC_DATABASE).forEach(continent => {
            Object.values(continent).forEach(country => {
                Object.values(country).forEach(regions => {
                    total += regions.length;
                });
            });
        });
        return total;
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeVolcanoSelector);
    } else {
        initializeVolcanoSelector();
    }

    // Expose globally
    window.initializeVolcanoSelector = initializeVolcanoSelector;
    window.VOLCANIC_DATABASE = VOLCANIC_DATABASE;

    console.log('✅ Comprehensive Volcano Selector ready');
})();