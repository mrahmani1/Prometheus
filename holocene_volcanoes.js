// Comprehensive Holocene Volcano Database
// Based on Smithsonian Institution Global Volcanism Program
// Includes all volcanoes active within the last 10,000 years

// Check if HOLOCENE_VOLCANOES is already defined to prevent redeclaration errors
if (typeof HOLOCENE_VOLCANOES === 'undefined') {
    var HOLOCENE_VOLCANOES = {};
}

// Only define the data if it hasn't been defined yet
if (Object.keys(HOLOCENE_VOLCANOES).length === 0) {
    HOLOCENE_VOLCANOES = {
    // NORTH AMERICA
    "United States": [
        { name: "Yellowstone", vnum: "325010", lat: 44.43, lon: -110.67, elevation: 2805, type: "Caldera", lastEruption: "1350 BCE", vei: 4 },
        { name: "Mount St. Helens", vnum: "321050", lat: 46.20, lon: -122.18, elevation: 2549, type: "Stratovolcano", lastEruption: "2008", vei: 5 },
        { name: "Kilauea", vnum: "332010", lat: 19.421, lon: -155.287, elevation: 1222, type: "Shield", lastEruption: "2024", vei: 0 },
        { name: "Mauna Loa", vnum: "332020", lat: 19.475, lon: -155.608, elevation: 4169, type: "Shield", lastEruption: "2022", vei: 0 },
        { name: "Mount Rainier", vnum: "321030", lat: 46.853, lon: -121.760, elevation: 4392, type: "Stratovolcano", lastEruption: "1450 CE", vei: 3 },
        { name: "Mount Baker", vnum: "321010", lat: 48.777, lon: -121.813, elevation: 3285, type: "Stratovolcano", lastEruption: "1880", vei: 2 },
        { name: "Mount Shasta", vnum: "323010", lat: 41.409, lon: -122.193, elevation: 4317, type: "Stratovolcano", lastEruption: "1786", vei: 3 },
        { name: "Crater Lake", vnum: "322160", lat: 42.93, lon: -122.12, elevation: 2487, type: "Caldera", lastEruption: "2850 BCE", vei: 7 },
        { name: "Mount Hood", vnum: "322010", lat: 45.374, lon: -121.695, elevation: 3426, type: "Stratovolcano", lastEruption: "1866", vei: 2 },
        { name: "Long Valley", vnum: "323822", lat: 37.70, lon: -118.87, elevation: 3390, type: "Caldera", lastEruption: "1350 CE", vei: 6 },
        { name: "Mount Cleveland", vnum: "311240", lat: 52.825, lon: -169.944, elevation: 1730, type: "Stratovolcano", lastEruption: "2023", vei: 3 },
        { name: "Katmai", vnum: "312180", lat: 58.279, lon: -154.963, elevation: 2047, type: "Caldera", lastEruption: "1912", vei: 6 },
        { name: "Redoubt", vnum: "313030", lat: 60.485, lon: -152.742, elevation: 3108, type: "Stratovolcano", lastEruption: "2009", vei: 3 },
        { name: "Augustine", vnum: "313010", lat: 59.363, lon: -153.43, elevation: 1252, type: "Stratovolcano", lastEruption: "2006", vei: 3 }
    ],
    
    "Mexico": [
        { name: "Popocatépetl", vnum: "341090", lat: 19.023, lon: -98.628, elevation: 5393, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Colima", vnum: "341040", lat: 19.514, lon: -103.62, elevation: 3850, type: "Stratovolcano", lastEruption: "2017", vei: 3 },
        { name: "Paricutin", vnum: "341061", lat: 19.493, lon: -102.251, elevation: 3170, type: "Cinder cone", lastEruption: "1952", vei: 4 },
        { name: "El Chichón", vnum: "341120", lat: 17.36, lon: -93.228, elevation: 1150, type: "Stratovolcano", lastEruption: "1982", vei: 5 },
        { name: "Pico de Orizaba", vnum: "341100", lat: 19.03, lon: -97.27, elevation: 5636, type: "Stratovolcano", lastEruption: "1846", vei: 2 },
        { name: "Ceboruco", vnum: "341030", lat: 21.125, lon: -104.508, elevation: 2280, type: "Stratovolcano", lastEruption: "1875", vei: 3 }
    ],
    
    "Canada": [
        { name: "Mount Garibaldi", vnum: "320200", lat: 49.850, lon: -123.004, elevation: 2678, type: "Stratovolcano", lastEruption: "8060 BCE", vei: 4 },
        { name: "Mount Meager", vnum: "320180", lat: 50.630, lon: -123.500, elevation: 2680, type: "Complex", lastEruption: "410 BCE", vei: 4 },
        { name: "Mount Edziza", vnum: "320060", lat: 57.720, lon: -130.630, elevation: 2786, type: "Stratovolcano", lastEruption: "950 CE", vei: 4 }
    ],
    
    // CENTRAL AMERICA
    "Guatemala": [
        { name: "Fuego", vnum: "342090", lat: 14.473, lon: -90.880, elevation: 3763, type: "Stratovolcano", lastEruption: "2024", vei: 4 },
        { name: "Pacaya", vnum: "342110", lat: 14.382, lon: -90.601, elevation: 2569, type: "Complex", lastEruption: "2024", vei: 3 },
        { name: "Santiaguito", vnum: "342030", lat: 14.756, lon: -91.552, elevation: 2500, type: "Lava dome", lastEruption: "2024", vei: 3 },
        { name: "Atitlán", vnum: "342070", lat: 14.583, lon: -91.186, elevation: 3535, type: "Stratovolcano", lastEruption: "1853", vei: 3 },
        { name: "Tajumulco", vnum: "342020", lat: 15.043, lon: -91.903, elevation: 4220, type: "Stratovolcano", lastEruption: "Unknown", vei: 2 }
    ],
    
    "El Salvador": [
        { name: "San Miguel", vnum: "343100", lat: 13.434, lon: -88.269, elevation: 2130, type: "Stratovolcano", lastEruption: "2013", vei: 2 },
        { name: "Santa Ana", vnum: "343020", lat: 13.853, lon: -89.630, elevation: 2381, type: "Stratovolcano", lastEruption: "2005", vei: 3 },
        { name: "San Salvador", vnum: "343050", lat: 13.736, lon: -89.294, elevation: 1893, type: "Stratovolcano", lastEruption: "1917", vei: 3 },
        { name: "Izalco", vnum: "343030", lat: 13.813, lon: -89.633, elevation: 1965, type: "Stratovolcano", lastEruption: "1966", vei: 2 }
    ],
    
    "Nicaragua": [
        { name: "San Cristóbal", vnum: "344020", lat: 12.702, lon: -87.004, elevation: 1745, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Telica", vnum: "344040", lat: 12.606, lon: -86.840, elevation: 1061, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Cerro Negro", vnum: "344070", lat: 12.506, lon: -86.702, elevation: 728, type: "Cinder cone", lastEruption: "1999", vei: 3 },
        { name: "Momotombo", vnum: "344090", lat: 12.422, lon: -86.540, elevation: 1297, type: "Stratovolcano", lastEruption: "2016", vei: 2 },
        { name: "Masaya", vnum: "344100", lat: 11.984, lon: -86.161, elevation: 635, type: "Caldera", lastEruption: "2022", vei: 0 },
        { name: "Concepción", vnum: "344120", lat: 11.538, lon: -85.622, elevation: 1700, type: "Stratovolcano", lastEruption: "2011", vei: 2 }
    ],
    
    "Costa Rica": [
        { name: "Arenal", vnum: "345033", lat: 10.463, lon: -84.703, elevation: 1670, type: "Stratovolcano", lastEruption: "2010", vei: 3 },
        { name: "Poás", vnum: "345040", lat: 10.200, lon: -84.233, elevation: 2697, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Irazú", vnum: "345060", lat: 9.979, lon: -83.852, elevation: 3432, type: "Stratovolcano", lastEruption: "1994", vei: 2 },
        { name: "Turrialba", vnum: "345070", lat: 10.025, lon: -83.767, elevation: 3340, type: "Stratovolcano", lastEruption: "2020", vei: 2 },
        { name: "Rincón de la Vieja", vnum: "345020", lat: 10.830, lon: -85.324, elevation: 1916, type: "Complex", lastEruption: "2021", vei: 2 }
    ],
    
    "Panama": [
        { name: "Barú", vnum: "346010", lat: 8.808, lon: -82.543, elevation: 3474, type: "Stratovolcano", lastEruption: "1550 CE", vei: 4 },
        { name: "El Valle", vnum: "346020", lat: 8.58, lon: -80.17, elevation: 1185, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 }
    ],
    
    // SOUTH AMERICA
    "Colombia": [
        { name: "Nevado del Ruiz", vnum: "351020", lat: 4.892, lon: -75.324, elevation: 5279, type: "Stratovolcano", lastEruption: "2023", vei: 3 },
        { name: "Galeras", vnum: "351080", lat: 1.222, lon: -77.359, elevation: 4276, type: "Complex", lastEruption: "2014", vei: 3 },
        { name: "Nevado del Huila", vnum: "351050", lat: 2.93, lon: -76.03, elevation: 5364, type: "Stratovolcano", lastEruption: "2012", vei: 2 },
        { name: "Puracé", vnum: "351060", lat: 2.32, lon: -76.40, elevation: 4650, type: "Stratovolcano", lastEruption: "1977", vei: 2 },
        { name: "Doña Juana", vnum: "351070", lat: 1.47, lon: -76.92, elevation: 4150, type: "Stratovolcano", lastEruption: "1906", vei: 2 },
        { name: "Cumbal", vnum: "351100", lat: 0.95, lon: -77.87, elevation: 4764, type: "Stratovolcano", lastEruption: "1926", vei: 2 }
    ],
    
    "Ecuador": [
        { name: "Cotopaxi", vnum: "352050", lat: -0.677, lon: -78.436, elevation: 5897, type: "Stratovolcano", lastEruption: "2016", vei: 3 },
        { name: "Tungurahua", vnum: "352080", lat: -1.467, lon: -78.442, elevation: 5023, type: "Stratovolcano", lastEruption: "2016", vei: 3 },
        { name: "Reventador", vnum: "352010", lat: -0.077, lon: -77.656, elevation: 3562, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Sangay", vnum: "352090", lat: -2.005, lon: -78.341, elevation: 5286, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Guagua Pichincha", vnum: "352020", lat: -0.171, lon: -78.598, elevation: 4784, type: "Stratovolcano", lastEruption: "2002", vei: 3 },
        { name: "Cayambe", vnum: "352060", lat: 0.029, lon: -77.986, elevation: 5790, type: "Compound", lastEruption: "1786", vei: 3 },
        { name: "Chimborazo", vnum: "352071", lat: -1.464, lon: -78.815, elevation: 6263, type: "Stratovolcano", lastEruption: "550 CE", vei: 3 }
    ],
    
    "Peru": [
        { name: "Ubinas", vnum: "354020", lat: -16.355, lon: -70.903, elevation: 5672, type: "Stratovolcano", lastEruption: "2023", vei: 3 },
        { name: "Sabancaya", vnum: "354003", lat: -15.787, lon: -71.857, elevation: 5967, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "El Misti", vnum: "354010", lat: -16.294, lon: -71.409, elevation: 5822, type: "Stratovolcano", lastEruption: "1985", vei: 2 },
        { name: "Huaynaputina", vnum: "354030", lat: -16.608, lon: -70.85, elevation: 4850, type: "Stratovolcano", lastEruption: "1600", vei: 6 },
        { name: "Coropuna", vnum: "354003", lat: -15.52, lon: -72.65, elevation: 6377, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 },
        { name: "Tutupaca", vnum: "354050", lat: -17.025, lon: -70.358, elevation: 5815, type: "Stratovolcano", lastEruption: "1902", vei: 2 }
    ],
    
    "Chile": [
        { name: "Villarrica", vnum: "357120", lat: -39.42, lon: -71.93, elevation: 2860, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Llaima", vnum: "357110", lat: -38.692, lon: -71.729, elevation: 3125, type: "Stratovolcano", lastEruption: "2009", vei: 2 },
        { name: "Calbuco", vnum: "358020", lat: -41.33, lon: -72.61, elevation: 2003, type: "Stratovolcano", lastEruption: "2015", vei: 4 },
        { name: "Puyehue-Cordón Caulle", vnum: "357150", lat: -40.590, lon: -72.117, elevation: 2236, type: "Stratovolcano", lastEruption: "2011", vei: 5 },
        { name: "Chaiten", vnum: "358041", lat: -42.833, lon: -72.646, elevation: 1122, type: "Caldera", lastEruption: "2011", vei: 4 },
        { name: "Hudson", vnum: "358057", lat: -45.90, lon: -72.97, elevation: 1905, type: "Stratovolcano", lastEruption: "2011", vei: 2 },
        { name: "Nevados de Chillán", vnum: "357070", lat: -36.868, lon: -71.378, elevation: 3216, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Lascar", vnum: "355100", lat: -23.37, lon: -67.73, elevation: 5592, type: "Stratovolcano", lastEruption: "2022", vei: 2 }
    ],
    
    "Argentina": [
        { name: "Copahue", vnum: "357090", lat: -37.856, lon: -71.164, elevation: 2953, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Lanín", vnum: "357130", lat: -39.633, lon: -71.5, elevation: 3747, type: "Stratovolcano", lastEruption: "560 CE", vei: 3 },
        { name: "Tromen", vnum: "357072", lat: -37.142, lon: -70.031, elevation: 3978, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 }
    ],
    
    "Bolivia": [
        { name: "Uturuncu", vnum: "355011", lat: -22.27, lon: -67.18, elevation: 6008, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 },
        { name: "Ollagüe", vnum: "355050", lat: -21.30, lon: -68.18, elevation: 5868, type: "Stratovolcano", lastEruption: "1903", vei: 2 }
    ],
    
    "Brazil": [
        { name: "Trindade", vnum: "365010", lat: -20.50, lon: -29.32, elevation: 600, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 }
    ],
    
    // CARIBBEAN
    "West Indies": [
        { name: "Soufrière Hills", vnum: "360050", lat: 16.72, lon: -62.18, elevation: 1050, type: "Stratovolcano", lastEruption: "2013", vei: 3 },
        { name: "La Soufrière", vnum: "360060", lat: 16.05, lon: -61.67, elevation: 1467, type: "Stratovolcano", lastEruption: "1979", vei: 3 },
        { name: "Mount Pelée", vnum: "360120", lat: 14.82, lon: -61.17, elevation: 1397, type: "Stratovolcano", lastEruption: "1932", vei: 4 },
        { name: "Soufriere St. Vincent", vnum: "360150", lat: 13.33, lon: -61.18, elevation: 1234, type: "Stratovolcano", lastEruption: "2021", vei: 4 },
        { name: "Kick 'em Jenny", vnum: "360160", lat: 12.30, lon: -61.64, elevation: -185, type: "Submarine", lastEruption: "2017", vei: 0 }
    ],
    
    // EUROPE
    "Italy": [
        { name: "Etna", vnum: "211060", lat: 37.734, lon: 15.004, elevation: 3357, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Vesuvius", vnum: "211020", lat: 40.821, lon: 14.426, elevation: 1281, type: "Complex", lastEruption: "1944", vei: 3 },
        { name: "Stromboli", vnum: "211040", lat: 38.789, lon: 15.213, elevation: 924, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Campi Flegrei", vnum: "211010", lat: 40.827, lon: 14.139, elevation: 458, type: "Caldera", lastEruption: "1538", vei: 2 },
        { name: "Vulcano", vnum: "211050", lat: 38.404, lon: 14.962, elevation: 500, type: "Stratovolcano", lastEruption: "1890", vei: 3 },
        { name: "Pantelleria", vnum: "211071", lat: 36.77, lon: 11.98, elevation: 836, type: "Shield", lastEruption: "1891", vei: 1 },
        { name: "Ischia", vnum: "211030", lat: 40.73, lon: 13.90, elevation: 789, type: "Complex", lastEruption: "1302", vei: 2 }
    ],
    
    "Greece": [
        { name: "Santorini", vnum: "212040", lat: 36.404, lon: 25.396, elevation: 329, type: "Caldera", lastEruption: "1950", vei: 2 },
        { name: "Nisyros", vnum: "212050", lat: 36.586, lon: 27.160, elevation: 698, type: "Stratovolcano", lastEruption: "1888", vei: 2 },
        { name: "Methana", vnum: "212020", lat: 37.615, lon: 23.336, elevation: 760, type: "Lava dome", lastEruption: "258 BCE", vei: 2 },
        { name: "Milos", vnum: "212030", lat: 36.699, lon: 24.439, elevation: 748, type: "Stratovolcano", lastEruption: "140 CE", vei: 2 }
    ],
    
    "Iceland": [
        { name: "Hekla", vnum: "372070", lat: 63.98, lon: -19.70, elevation: 1491, type: "Stratovolcano", lastEruption: "2000", vei: 3 },
        { name: "Katla", vnum: "372030", lat: 63.63, lon: -19.05, elevation: 1512, type: "Subglacial", lastEruption: "1918", vei: 4 },
        { name: "Eyjafjallajökull", vnum: "372020", lat: 63.63, lon: -19.62, elevation: 1666, type: "Stratovolcano", lastEruption: "2010", vei: 4 },
        { name: "Grímsvötn", vnum: "373010", lat: 64.42, lon: -17.33, elevation: 1725, type: "Subglacial", lastEruption: "2011", vei: 4 },
        { name: "Bárðarbunga", vnum: "373030", lat: 64.63, lon: -17.53, elevation: 2009, type: "Stratovolcano", lastEruption: "2015", vei: 1 },
        { name: "Askja", vnum: "373060", lat: 65.05, lon: -16.75, elevation: 1516, type: "Caldera", lastEruption: "1961", vei: 3 },
        { name: "Krafla", vnum: "373080", lat: 65.73, lon: -16.78, elevation: 818, type: "Caldera", lastEruption: "1984", vei: 1 },
        { name: "Reykjanes", vnum: "371020", lat: 63.88, lon: -22.50, elevation: 230, type: "Crater rows", lastEruption: "2024", vei: 1 }
    ],
    
    "Spain": [
        { name: "Teide", vnum: "383030", lat: 28.271, lon: -16.641, elevation: 3715, type: "Stratovolcano", lastEruption: "1909", vei: 2 },
        { name: "La Palma", vnum: "383010", lat: 28.57, lon: -17.83, elevation: 2426, type: "Stratovolcano", lastEruption: "2021", vei: 3 },
        { name: "El Hierro", vnum: "383020", lat: 27.73, lon: -18.03, elevation: 1500, type: "Shield", lastEruption: "2012", vei: 2 },
        { name: "Lanzarote", vnum: "383060", lat: 29.03, lon: -13.63, elevation: 670, type: "Fissure vents", lastEruption: "1824", vei: 3 }
    ],
    
    "Portugal": [
        { name: "Pico", vnum: "382002", lat: 38.47, lon: -28.40, elevation: 2351, type: "Stratovolcano", lastEruption: "1720", vei: 2 },
        { name: "Fayal", vnum: "382010", lat: 38.60, lon: -28.73, elevation: 1043, type: "Stratovolcano", lastEruption: "1958", vei: 4 },
        { name: "São Jorge", vnum: "382003", lat: 38.65, lon: -28.08, elevation: 1053, type: "Fissure vents", lastEruption: "1907", vei: 2 }
    ],
    
    "Turkey": [
        { name: "Ararat", vnum: "213040", lat: 39.70, lon: 44.28, elevation: 5137, type: "Stratovolcano", lastEruption: "1840", vei: 3 },
        { name: "Nemrut Dagi", vnum: "213020", lat: 38.65, lon: 42.23, elevation: 2948, type: "Stratovolcano", lastEruption: "1692", vei: 0 },
        { name: "Erciyes Dagi", vnum: "213010", lat: 38.52, lon: 35.45, elevation: 3916, type: "Stratovolcano", lastEruption: "253 CE", vei: 3 },
        { name: "Hasan Dagi", vnum: "213002", lat: 38.13, lon: 34.17, elevation: 3253, type: "Stratovolcano", lastEruption: "6200 BCE", vei: 3 }
    ],
    
    // AFRICA
    "Ethiopia": [
        { name: "Erta Ale", vnum: "221080", lat: 13.60, lon: 40.67, elevation: 613, type: "Shield", lastEruption: "2024", vei: 0 },
        { name: "Dallol", vnum: "221100", lat: 14.24, lon: 40.30, elevation: -48, type: "Explosion crater", lastEruption: "2011", vei: 1 },
        { name: "Dabbahu", vnum: "221130", lat: 12.60, lon: 40.48, elevation: 1442, type: "Stratovolcano", lastEruption: "2005", vei: 1 },
        { name: "Nabro", vnum: "221101", lat: 13.37, lon: 41.70, elevation: 2218, type: "Stratovolcano", lastEruption: "2011", vei: 4 },
        { name: "Dubbi", vnum: "221100", lat: 13.58, lon: 41.81, elevation: 1625, type: "Stratovolcano", lastEruption: "1861", vei: 3 },
        { name: "Fantale", vnum: "221180", lat: 8.97, lon: 39.93, elevation: 2007, type: "Stratovolcano", lastEruption: "1820", vei: 2 }
    ],
    
    "Kenya": [
        { name: "Mount Kenya", vnum: "222020", lat: -0.15, lon: 37.30, elevation: 5199, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 },
        { name: "Longonot", vnum: "222010", lat: -0.914, lon: 36.446, elevation: 2776, type: "Stratovolcano", lastEruption: "1863", vei: 3 },
        { name: "Menengai", vnum: "222003", lat: -0.20, lon: 36.07, elevation: 2278, type: "Shield", lastEruption: "6050 BCE", vei: 6 },
        { name: "The Barrier", vnum: "222030", lat: 2.32, lon: 36.57, elevation: 1032, type: "Shield", lastEruption: "1921", vei: 0 }
    ],
    
    "Tanzania": [
        { name: "Ol Doinyo Lengai", vnum: "222120", lat: -2.764, lon: 35.914, elevation: 2962, type: "Stratovolcano", lastEruption: "2024", vei: 1 },
        { name: "Mount Meru", vnum: "222160", lat: -3.25, lon: 36.75, elevation: 4565, type: "Stratovolcano", lastEruption: "1910", vei: 2 },
        { name: "Kilimanjaro", vnum: "222150", lat: -3.07, lon: 37.35, elevation: 5895, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 },
        { name: "Rungwe", vnum: "222160", lat: -9.13, lon: 33.67, elevation: 2961, type: "Stratovolcano", lastEruption: "1200 CE", vei: 2 }
    ],
    
    "Democratic Republic of Congo": [
        { name: "Nyiragongo", vnum: "223030", lat: -1.52, lon: 29.25, elevation: 3470, type: "Stratovolcano", lastEruption: "2024", vei: 1 },
        { name: "Nyamuragira", vnum: "223020", lat: -1.408, lon: 29.20, elevation: 3058, type: "Shield", lastEruption: "2023", vei: 2 },
        { name: "Karisimbi", vnum: "223040", lat: -1.507, lon: 29.45, elevation: 4507, type: "Stratovolcano", lastEruption: "8050 BCE", vei: 0 },
        { name: "Visoke", vnum: "223050", lat: -1.47, lon: 29.49, elevation: 3711, type: "Stratovolcano", lastEruption: "1957", vei: 2 }
    ],
    
    "Cameroon": [
        { name: "Mount Cameroon", vnum: "224010", lat: 4.203, lon: 9.176, elevation: 4040, type: "Stratovolcano", lastEruption: "2012", vei: 2 },
        { name: "Lake Nyos", vnum: "224030", lat: 6.43, lon: 10.30, elevation: 1091, type: "Maar", lastEruption: "400 CE", vei: 0 },
        { name: "Oku Volcanic Field", vnum: "224040", lat: 6.25, lon: 10.50, elevation: 3011, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 }
    ],
    
    "Cape Verde": [
        { name: "Fogo", vnum: "384010", lat: 14.95, lon: -24.35, elevation: 2829, type: "Stratovolcano", lastEruption: "2015", vei: 2 },
        { name: "Brava", vnum: "384020", lat: 14.85, lon: -24.72, elevation: 976, type: "Stratovolcano", lastEruption: "Unknown", vei: 0 }
    ],
    
    // ASIA
    "Japan": [
        { name: "Mount Fuji", vnum: "283030", lat: 35.358, lon: 138.731, elevation: 3776, type: "Stratovolcano", lastEruption: "1708", vei: 5 },
        { name: "Sakurajima", vnum: "282080", lat: 31.58, lon: 130.657, elevation: 1117, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Aso", vnum: "282110", lat: 32.88, lon: 131.10, elevation: 1592, type: "Caldera", lastEruption: "2021", vei: 2 },
        { name: "Unzen", vnum: "282100", lat: 32.761, lon: 130.299, elevation: 1483, type: "Complex", lastEruption: "1996", vei: 2 },
        { name: "Asama", vnum: "283110", lat: 36.406, lon: 138.523, elevation: 2568, type: "Complex", lastEruption: "2019", vei: 2 },
        { name: "Shinmoedake", vnum: "282090", lat: 31.91, lon: 130.86, elevation: 1421, type: "Stratovolcano", lastEruption: "2018", vei: 3 },
        { name: "Ontake", vnum: "283040", lat: 35.893, lon: 137.480, elevation: 3067, type: "Complex", lastEruption: "2014", vei: 3 },
        { name: "Kusatsu-Shirane", vnum: "283120", lat: 36.618, lon: 138.528, elevation: 2165, type: "Stratovolcano", lastEruption: "2018", vei: 1 },
        { name: "Miyakejima", vnum: "284040", lat: 34.094, lon: 139.526, elevation: 775, type: "Stratovolcano", lastEruption: "2013", vei: 2 },
        { name: "Suwanosejima", vnum: "282030", lat: 29.638, lon: 129.714, elevation: 796, type: "Stratovolcano", lastEruption: "2024", vei: 2 }
    ],
    
    "Indonesia": [
        { name: "Krakatau", vnum: "262000", lat: -6.102, lon: 105.423, elevation: 813, type: "Caldera", lastEruption: "2023", vei: 6 },
        { name: "Merapi", vnum: "263250", lat: -7.542, lon: 110.445, elevation: 2910, type: "Stratovolcano", lastEruption: "2024", vei: 4 },
        { name: "Tambora", vnum: "264040", lat: -8.25, lon: 118.00, elevation: 2850, type: "Stratovolcano", lastEruption: "1967", vei: 7 },
        { name: "Agung", vnum: "264020", lat: -8.342, lon: 115.508, elevation: 3031, type: "Stratovolcano", lastEruption: "2019", vei: 4 },
        { name: "Rinjani", vnum: "264030", lat: -8.42, lon: 116.47, elevation: 3726, type: "Stratovolcano", lastEruption: "2016", vei: 2 },
        { name: "Kelud", vnum: "263280", lat: -7.93, lon: 112.308, elevation: 1731, type: "Stratovolcano", lastEruption: "2014", vei: 4 },
        { name: "Semeru", vnum: "263300", lat: -8.108, lon: 112.922, elevation: 3676, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Bromo", vnum: "263310", lat: -7.942, lon: 112.953, elevation: 2329, type: "Stratovolcano", lastEruption: "2019", vei: 2 },
        { name: "Sinabung", vnum: "261080", lat: 3.17, lon: 98.392, elevation: 2460, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Toba", vnum: "261090", lat: 2.58, lon: 98.83, elevation: 2157, type: "Caldera", lastEruption: "Unknown", vei: 8 },
        { name: "Galunggung", vnum: "263140", lat: -7.25, lon: 108.058, elevation: 2168, type: "Stratovolcano", lastEruption: "1984", vei: 4 },
        { name: "Raung", vnum: "263340", lat: -8.119, lon: 114.045, elevation: 3332, type: "Stratovolcano", lastEruption: "2021", vei: 2 },
        { name: "Ijen", vnum: "263350", lat: -8.058, lon: 114.242, elevation: 2386, type: "Stratovolcano", lastEruption: "2002", vei: 2 },
        { name: "Papandayan", vnum: "263100", lat: -7.32, lon: 107.73, elevation: 2665, type: "Stratovolcano", lastEruption: "2002", vei: 2 },
        { name: "Kerinci", vnum: "261170", lat: -1.697, lon: 101.264, elevation: 3800, type: "Stratovolcano", lastEruption: "2020", vei: 2 },
        { name: "Dempo", vnum: "261230", lat: -4.016, lon: 103.128, elevation: 3173, type: "Stratovolcano", lastEruption: "2009", vei: 2 },
        { name: "Talang", vnum: "261160", lat: -0.978, lon: 100.679, elevation: 2597, type: "Stratovolcano", lastEruption: "2007", vei: 2 },
        { name: "Marapi", vnum: "261140", lat: -0.381, lon: 100.473, elevation: 2891, type: "Complex", lastEruption: "2023", vei: 2 },
        { name: "Soputan", vnum: "266030", lat: 1.112, lon: 124.737, elevation: 1784, type: "Stratovolcano", lastEruption: "2018", vei: 2 },
        { name: "Lokon-Empung", vnum: "266100", lat: 1.358, lon: 124.792, elevation: 1580, type: "Stratovolcano", lastEruption: "2015", vei: 2 },
        { name: "Karangetang", vnum: "267020", lat: 2.781, lon: 125.407, elevation: 1827, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Ibu", vnum: "268030", lat: 1.488, lon: 127.63, elevation: 1325, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Dukono", vnum: "268010", lat: 1.693, lon: 127.894, elevation: 1229, type: "Complex", lastEruption: "2024", vei: 2 },
        { name: "Gamkonora", vnum: "268040", lat: 1.38, lon: 127.53, elevation: 1635, type: "Stratovolcano", lastEruption: "2013", vei: 2 },
        { name: "Gamalama", vnum: "268060", lat: 0.80, lon: 127.33, elevation: 1715, type: "Stratovolcano", lastEruption: "2015", vei: 2 },
        { name: "Sangeang Api", vnum: "264050", lat: -8.18, lon: 119.058, elevation: 1949, type: "Complex", lastEruption: "2022", vei: 2 },
        { name: "Lewotolo", vnum: "264230", lat: -8.272, lon: 123.505, elevation: 1423, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Lewotobi", vnum: "264100", lat: -8.542, lon: 122.775, elevation: 1703, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Ili Lewotolok", vnum: "264230", lat: -8.541, lon: 123.255, elevation: 1423, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Anak Krakatau", vnum: "262001", lat: -6.097, lon: 105.422, elevation: 157, type: "Stratovolcano", lastEruption: "2023", vei: 3 }
    ],
    
    "Philippines": [
        { name: "Pinatubo", vnum: "273083", lat: 15.13, lon: 120.35, elevation: 1486, type: "Stratovolcano", lastEruption: "1993", vei: 6 },
        { name: "Mayon", vnum: "273030", lat: 13.257, lon: 123.685, elevation: 2463, type: "Stratovolcano", lastEruption: "2023", vei: 3 },
        { name: "Taal", vnum: "273070", lat: 14.002, lon: 120.993, elevation: 311, type: "Caldera", lastEruption: "2022", vei: 4 },
        { name: "Kanlaon", vnum: "272020", lat: 10.412, lon: 123.132, elevation: 2435, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Bulusan", vnum: "273010", lat: 12.77, lon: 124.05, elevation: 1565, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Hibok-Hibok", vnum: "271080", lat: 9.203, lon: 124.673, elevation: 1332, type: "Stratovolcano", lastEruption: "1953", vei: 3 },
        { name: "Ragang", vnum: "271070", lat: 7.70, lon: 124.50, elevation: 2815, type: "Stratovolcano", lastEruption: "1916", vei: 2 },
        { name: "Parker", vnum: "271010", lat: 6.12, lon: 124.892, elevation: 1824, type: "Stratovolcano", lastEruption: "1641", vei: 3 }
    ],
    
    "Russia": [
        { name: "Klyuchevskoy", vnum: "300260", lat: 56.056, lon: 160.642, elevation: 4754, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Bezymianny", vnum: "300250", lat: 55.972, lon: 160.595, elevation: 2882, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Shiveluch", vnum: "300270", lat: 56.653, lon: 161.360, elevation: 3283, type: "Stratovolcano", lastEruption: "2024", vei: 4 },
        { name: "Tolbachik", vnum: "300240", lat: 55.832, lon: 160.326, elevation: 3682, type: "Shield", lastEruption: "2013", vei: 1 },
        { name: "Kizimen", vnum: "300230", lat: 55.131, lon: 160.32, elevation: 2376, type: "Stratovolcano", lastEruption: "2013", vei: 3 },
        { name: "Karymsky", vnum: "300130", lat: 54.049, lon: 159.443, elevation: 1536, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Avachinsky", vnum: "300100", lat: 53.255, lon: 158.836, elevation: 2741, type: "Stratovolcano", lastEruption: "2001", vei: 2 },
        { name: "Mutnovsky", vnum: "300060", lat: 52.453, lon: 158.195, elevation: 2322, type: "Complex", lastEruption: "2013", vei: 2 },
        { name: "Zhupanovsky", vnum: "300120", lat: 53.589, lon: 159.15, elevation: 2923, type: "Compound", lastEruption: "2017", vei: 2 },
        { name: "Ebeko", vnum: "290380", lat: 50.686, lon: 156.014, elevation: 1156, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Chikurachki", vnum: "290360", lat: 50.324, lon: 155.461, elevation: 1816, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Alaid", vnum: "290390", lat: 50.861, lon: 155.565, elevation: 2339, type: "Stratovolcano", lastEruption: "2022", vei: 2 }
    ],
    
    "China": [
        { name: "Changbaishan", vnum: "305060", lat: 42.00, lon: 128.05, elevation: 2744, type: "Stratovolcano", lastEruption: "1903", vei: 5 },
        { name: "Wudalianchi", vnum: "305030", lat: 48.72, lon: 126.12, elevation: 516, type: "Volcanic field", lastEruption: "1776", vei: 2 },
        { name: "Tengchong", vnum: "275011", lat: 25.23, lon: 98.47, elevation: 2865, type: "Pyroclastic cone", lastEruption: "1609", vei: 0 },
        { name: "Keluo Group", vnum: "305040", lat: 49.35, lon: 125.97, elevation: 1000, type: "Volcanic field", lastEruption: "Holocene", vei: 0 }
    ],
    
    "Taiwan": [
        { name: "Kueishantao", vnum: "281010", lat: 24.85, lon: 121.92, elevation: 401, type: "Stratovolcano", lastEruption: "1785", vei: 0 },
        { name: "Tatun Group", vnum: "281080", lat: 25.17, lon: 121.53, elevation: 1120, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 }
    ],
    
    "India": [
        { name: "Barren Island", vnum: "260010", lat: 12.278, lon: 93.858, elevation: 354, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Narcondam", vnum: "260020", lat: 13.43, lon: 94.25, elevation: 710, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 }
    ],
    
    "Myanmar": [
        { name: "Lower Chindwin", vnum: "275080", lat: 22.28, lon: 95.20, elevation: 385, type: "Volcanic field", lastEruption: "Holocene", vei: 0 },
        { name: "Singu Plateau", vnum: "275100", lat: 22.70, lon: 95.98, elevation: 507, type: "Volcanic field", lastEruption: "Holocene", vei: 0 }
    ],
    
    "Armenia": [
        { name: "Aragats", vnum: "213001", lat: 40.52, lon: 44.12, elevation: 4090, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 },
        { name: "Ghegam Ridge", vnum: "213003", lat: 40.27, lon: 44.72, elevation: 3597, type: "Volcanic field", lastEruption: "1900 BCE", vei: 3 },
        { name: "Porak", vnum: "213050", lat: 40.02, lon: 45.78, elevation: 2800, type: "Stratovolcano", lastEruption: "783 CE", vei: 0 }
    ],
    
    "Georgia": [
        { name: "Kazbek", vnum: "214010", lat: 42.70, lon: 44.52, elevation: 5047, type: "Stratovolcano", lastEruption: "750 BCE", vei: 3 },
        { name: "Samsari", vnum: "214020", lat: 41.15, lon: 43.58, elevation: 3285, type: "Volcanic field", lastEruption: "Holocene", vei: 0 }
    ],
    
    "Iran": [
        { name: "Damavand", vnum: "232010", lat: 35.95, lon: 52.11, elevation: 5670, type: "Stratovolcano", lastEruption: "5350 BCE", vei: 0 },
        { name: "Sabalan", vnum: "232020", lat: 38.25, lon: 47.92, elevation: 4811, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 },
        { name: "Sahand", vnum: "232030", lat: 37.72, lon: 46.43, elevation: 3707, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 },
        { name: "Taftan", vnum: "232050", lat: 28.60, lon: 61.13, elevation: 3940, type: "Stratovolcano", lastEruption: "1993", vei: 0 },
        { name: "Bazman", vnum: "232060", lat: 28.07, lon: 60.00, elevation: 3490, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 }
    ],
    
    "Pakistan": [
        { name: "Koh-i-Sultan", vnum: "233000", lat: 29.02, lon: 63.25, elevation: 2334, type: "Volcanic field", lastEruption: "Holocene", vei: 0 }
    ],
    
    "North Korea": [
        { name: "Ch'uga-ryong", vnum: "305070", lat: 38.33, lon: 127.33, elevation: 452, type: "Shield", lastEruption: "Holocene", vei: 0 }
    ],
    
    "South Korea": [
        { name: "Halla", vnum: "306040", lat: 33.361, lon: 126.533, elevation: 1950, type: "Shield", lastEruption: "1007 CE", vei: 0 },
        { name: "Ulleung", vnum: "306050", lat: 37.50, lon: 130.87, elevation: 984, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 }
    ],
    
    "Malaysia": [
        { name: "Bombalai", vnum: "265011", lat: 4.40, lon: 117.88, elevation: 531, type: "Pyroclastic cone", lastEruption: "Holocene", vei: 0 }
    ],
    
    // OCEANIA
    "Papua New Guinea": [
        { name: "Rabaul", vnum: "252140", lat: -4.271, lon: 152.203, elevation: 688, type: "Caldera", lastEruption: "2014", vei: 4 },
        { name: "Manam", vnum: "251020", lat: -4.08, lon: 145.037, elevation: 1807, type: "Stratovolcano", lastEruption: "2024", vei: 3 },
        { name: "Ulawun", vnum: "252120", lat: -5.05, lon: 151.33, elevation: 2334, type: "Stratovolcano", lastEruption: "2023", vei: 3 },
        { name: "Bagana", vnum: "255020", lat: -6.137, lon: 155.196, elevation: 1855, type: "Lava cone", lastEruption: "2024", vei: 2 },
        { name: "Langila", vnum: "252010", lat: -5.525, lon: 148.42, elevation: 1330, type: "Complex", lastEruption: "2023", vei: 2 },
        { name: "Lamington", vnum: "253010", lat: -8.95, lon: 148.15, elevation: 1680, type: "Stratovolcano", lastEruption: "1956", vei: 4 },
        { name: "Pago", vnum: "252080", lat: -5.58, lon: 150.52, elevation: 742, type: "Caldera", lastEruption: "2012", vei: 2 },
        { name: "Karkar", vnum: "251030", lat: -4.649, lon: 145.964, elevation: 1839, type: "Stratovolcano", lastEruption: "2014", vei: 2 },
        { name: "Kadovar", vnum: "251002", lat: -3.608, lon: 144.588, elevation: 365, type: "Stratovolcano", lastEruption: "2018", vei: 2 },
        { name: "Dukono", vnum: "268010", lat: 1.693, lon: 127.894, elevation: 1229, type: "Complex", lastEruption: "2024", vei: 2 }
    ],
    
    "New Zealand": [
        { name: "White Island", vnum: "241040", lat: -37.52, lon: 177.18, elevation: 321, type: "Stratovolcano", lastEruption: "2019", vei: 2 },
        { name: "Ruapehu", vnum: "241100", lat: -39.28, lon: 175.57, elevation: 2797, type: "Stratovolcano", lastEruption: "2007", vei: 2 },
        { name: "Tongariro", vnum: "241080", lat: -39.157, lon: 175.632, elevation: 1978, type: "Stratovolcano", lastEruption: "2012", vei: 2 },
        { name: "Taranaki", vnum: "241030", lat: -39.30, lon: 174.06, elevation: 2518, type: "Stratovolcano", lastEruption: "1854", vei: 2 },
        { name: "Taupo", vnum: "241070", lat: -38.82, lon: 176.00, elevation: 760, type: "Caldera", lastEruption: "260 CE", vei: 5 },
        { name: "Mayor Island", vnum: "241021", lat: -37.28, lon: 176.25, elevation: 355, type: "Shield", lastEruption: "5060 BCE", vei: 5 },
        { name: "Auckland Field", vnum: "241020", lat: -36.90, lon: 174.87, elevation: 260, type: "Volcanic field", lastEruption: "1350 CE", vei: 3 }
    ],
    
    "Vanuatu": [
        { name: "Yasur", vnum: "257100", lat: -19.528, lon: 169.447, elevation: 361, type: "Stratovolcano", lastEruption: "2024", vei: 2 },
        { name: "Ambrym", vnum: "257040", lat: -16.25, lon: 168.12, elevation: 1334, type: "Pyroclastic shield", lastEruption: "2023", vei: 3 },
        { name: "Ambae", vnum: "257030", lat: -15.389, lon: 167.835, elevation: 1496, type: "Shield", lastEruption: "2018", vei: 3 },
        { name: "Lopevi", vnum: "257050", lat: -16.507, lon: 168.346, elevation: 1413, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Gaua", vnum: "257020", lat: -14.27, lon: 167.50, elevation: 797, type: "Stratovolcano", lastEruption: "2013", vei: 2 },
        { name: "Kuwae", vnum: "257061", lat: -16.829, lon: 168.536, elevation: -2, type: "Submarine", lastEruption: "1974", vei: 0 }
    ],
    
    "Solomon Islands": [
        { name: "Tinakula", vnum: "256010", lat: -10.38, lon: 165.80, elevation: 851, type: "Stratovolcano", lastEruption: "2022", vei: 2 },
        { name: "Kavachi", vnum: "255060", lat: -8.991, lon: 157.979, elevation: -20, type: "Submarine", lastEruption: "2023", vei: 0 },
        { name: "Simbo", vnum: "255050", lat: -8.292, lon: 156.52, elevation: 335, type: "Stratovolcano", lastEruption: "Holocene", vei: 0 },
        { name: "Savo", vnum: "255070", lat: -9.13, lon: 159.82, elevation: 485, type: "Stratovolcano", lastEruption: "1835", vei: 2 }
    ],
    
    "Fiji": [
        { name: "Taveuni", vnum: "243010", lat: -16.82, lon: -179.97, elevation: 1241, type: "Shield", lastEruption: "1550 CE", vei: 0 },
        { name: "Nabukelevu", vnum: "243020", lat: -19.12, lon: 177.98, elevation: 805, type: "Lava dome", lastEruption: "1660 CE", vei: 0 }
    ],
    
    "Tonga": [
        { name: "Hunga Tonga-Hunga Ha'apai", vnum: "243040", lat: -20.536, lon: -175.382, elevation: 114, type: "Submarine", lastEruption: "2022", vei: 5 },
        { name: "Fonualei", vnum: "243100", lat: -18.02, lon: -174.325, elevation: 180, type: "Stratovolcano", lastEruption: "2023", vei: 2 },
        { name: "Late", vnum: "243070", lat: -18.806, lon: -174.65, elevation: 540, type: "Stratovolcano", lastEruption: "1967", vei: 3 },
        { name: "Home Reef", vnum: "243080", lat: -18.992, lon: -174.775, elevation: -10, type: "Submarine", lastEruption: "2022", vei: 2 },
        { name: "Tofua", vnum: "243060", lat: -19.75, lon: -175.07, elevation: 512, type: "Caldera", lastEruption: "2022", vei: 2 }
    ],
    
    "Samoa": [
        { name: "Savai'i", vnum: "244010", lat: -13.612, lon: -172.525, elevation: 1858, type: "Shield", lastEruption: "1911", vei: 2 },
        { name: "Ta'u", vnum: "244020", lat: -14.23, lon: -169.454, elevation: 966, type: "Shield", lastEruption: "Holocene", vei: 0 }
    ],
    
    // ANTARCTICA
    "Antarctica": [
        { name: "Mount Erebus", vnum: "390020", lat: -77.53, lon: 167.17, elevation: 3794, type: "Stratovolcano", lastEruption: "2024", vei: 1 },
        { name: "Deception Island", vnum: "390030", lat: -62.97, lon: -60.65, elevation: 576, type: "Caldera", lastEruption: "1970", vei: 3 },
        { name: "Mount Melbourne", vnum: "390040", lat: -74.35, lon: 164.70, elevation: 2733, type: "Stratovolcano", lastEruption: "1750 CE", vei: 0 },
        { name: "Mount Berlin", vnum: "390050", lat: -76.03, lon: -135.52, elevation: 3478, type: "Shield", lastEruption: "8350 BCE", vei: 0 },
        { name: "Mount Siple", vnum: "390060", lat: -73.43, lon: -126.67, elevation: 3110, type: "Shield", lastEruption: "Holocene", vei: 0 }
    ]
    };
}

// Ensure global availability
if (typeof window !== 'undefined') {
    window.HOLOCENE_VOLCANOES = HOLOCENE_VOLCANOES;
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HOLOCENE_VOLCANOES;
}