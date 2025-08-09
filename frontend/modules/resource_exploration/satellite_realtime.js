// Real-Time Satellite Data Integration Module
// Integrates multiple satellite constellations for Earth observation
// Based on 2023-2024 research in satellite geodesy and remote sensing

class SatelliteDataIntegration {
    constructor() {
        this.satellites = {
            sentinel1: {
                name: 'Sentinel-1A/B',
                agency: 'ESA',
                type: 'SAR',
                band: 'C-band',
                wavelength: 5.6, // cm
                revisit: 6, // days
                resolution: 5, // m
                swath: 250, // km
                orbits: [],
                lastUpdate: null
            },
            alos2: {
                name: 'ALOS-2 PALSAR-2',
                agency: 'JAXA',
                type: 'SAR',
                band: 'L-band',
                wavelength: 23.6, // cm
                revisit: 14,
                resolution: 3,
                swath: 350,
                orbits: [],
                lastUpdate: null
            },
            terrasar: {
                name: 'TerraSAR-X',
                agency: 'DLR',
                type: 'SAR',
                band: 'X-band',
                wavelength: 3.1, // cm
                revisit: 11,
                resolution: 1,
                swath: 100,
                orbits: [],
                lastUpdate: null
            },
            cosmo: {
                name: 'COSMO-SkyMed',
                agency: 'ASI',
                type: 'SAR',
                band: 'X-band',
                wavelength: 3.1,
                revisit: 8,
                resolution: 1,
                swath: 200,
                orbits: [],
                lastUpdate: null
            },
            nisar: {
                name: 'NISAR',
                agency: 'NASA/ISRO',
                type: 'SAR',
                band: 'L/S-band',
                wavelength: 24, // cm (L-band)
                revisit: 12,
                resolution: 7,
                swath: 240,
                orbits: [],
                lastUpdate: null,
                status: 'Launching 2024'
            },
            modis: {
                name: 'MODIS Terra/Aqua',
                agency: 'NASA',
                type: 'Optical/Thermal',
                bands: 36,
                resolution: 250, // m (best)
                swath: 2330,
                thermalBands: [20, 22, 23, 29, 31, 32],
                lastUpdate: null
            },
            viirs: {
                name: 'VIIRS Suomi-NPP',
                agency: 'NASA/NOAA',
                type: 'Optical/Thermal',
                bands: 22,
                resolution: 375,
                swath: 3060,
                nighttime: true,
                lastUpdate: null
            },
            landsat9: {
                name: 'Landsat-9',
                agency: 'NASA/USGS',
                type: 'Multispectral',
                bands: 11,
                resolution: 30,
                revisit: 16,
                swath: 185,
                lastUpdate: null
            }
        };
        
        this.dataStreams = new Map();
        this.processingQueue = [];
        this.websockets = new Map();
        
        // InSAR processing parameters
        this.insarParams = {
            multilook: { range: 4, azimuth: 1 },
            filtering: 'goldstein',
            unwrapping: 'snaphu',
            referencePoint: null,
            coherenceThreshold: 0.3,
            atmosphericCorrection: true
        };
        
        // Thermal anomaly detection
        this.thermalParams = {
            backgroundWindow: 60, // days
            anomalyThreshold: 3, // standard deviations
            minPixels: 4,
            maxCloudCover: 0.2
        };
        
        this.initializeConnections();
        this.setupDataProcessing();
    }
    
    initializeConnections() {
        // Simulate WebSocket connections to satellite data providers
        const providers = [
            'wss://scihub.copernicus.eu/stream', // Sentinel
            'wss://gportal.jaxa.jp/stream', // ALOS
            'wss://earthdata.nasa.gov/stream', // NASA
            'wss://asi.it/cosmo-stream' // COSMO-SkyMed
        ];
        
        providers.forEach(url => {
            this.connectToProvider(url);
        });
    }
    
    connectToProvider(url) {
        // Simulate WebSocket connection
        const ws = {
            url: url,
            status: 'connected',
            reconnectAttempts: 0,
            send: (data) => console.log(`Sending to ${url}:`, data),
            close: () => console.log(`Closing ${url}`),
            
            // Simulate incoming data
            simulateData: () => {
                const data = this.generateSimulatedData(url);
                this.handleIncomingData(data);
            }
        };
        
        this.websockets.set(url, ws);
        
        // Start simulating data stream
        setInterval(() => ws.simulateData(), 5000);
    }
    
    generateSimulatedData(source) {
        const timestamp = new Date().toISOString();
        const satelliteType = source.includes('copernicus') ? 'sentinel1' :
                            source.includes('jaxa') ? 'alos2' :
                            source.includes('nasa') ? 'modis' : 'cosmo';
        
        return {
            source: source,
            satellite: satelliteType,
            timestamp: timestamp,
            orbit: {
                number: Math.floor(Math.random() * 1000) + 1000,
                direction: Math.random() > 0.5 ? 'ascending' : 'descending',
                inclination: 98.18, // degrees (sun-synchronous)
                altitude: 693000 + Math.random() * 1000, // meters
                velocity: 7500 + Math.random() * 10, // m/s
                groundTrack: {
                    start: { lat: -80 + Math.random() * 160, lon: -180 + Math.random() * 360 },
                    end: { lat: -80 + Math.random() * 160, lon: -180 + Math.random() * 360 }
                }
            },
            acquisition: {
                mode: this.getAcquisitionMode(satelliteType),
                polarization: this.getPolarization(satelliteType),
                incidenceAngle: 20 + Math.random() * 40, // degrees
                pixelSpacing: {
                    range: this.satellites[satelliteType].resolution,
                    azimuth: this.satellites[satelliteType].resolution
                },
                processingLevel: Math.random() > 0.5 ? 'L1' : 'L2',
                dataVolume: Math.floor(Math.random() * 1000) + 500 // MB
            },
            coverage: {
                polygon: this.generateCoveragePolygon(),
                area: Math.floor(Math.random() * 10000) + 5000, // km²
                cloudCover: Math.random() * 0.3, // 0-30%
                snowCover: Math.random() * 0.1 // 0-10%
            },
            quicklook: this.generateQuicklookUrl(satelliteType, timestamp)
        };
    }
    
    getAcquisitionMode(satellite) {
        const modes = {
            sentinel1: ['IW', 'EW', 'SM', 'WV'],
            alos2: ['FBS', 'FBD', 'WBS', 'WWS'],
            terrasar: ['SM', 'HS', 'SL', 'SC'],
            cosmo: ['HIMAGE', 'PINGPONG', 'WIDEREGION', 'HUGEREGION'],
            modis: ['DAYTIME', 'NIGHTTIME', 'AQUA', 'TERRA']
        };
        
        const satelliteModes = modes[satellite] || ['STANDARD'];
        return satelliteModes[Math.floor(Math.random() * satelliteModes.length)];
    }
    
    getPolarization(satellite) {
        if (satellite === 'modis' || satellite === 'viirs') return 'N/A';
        
        const polarizations = ['VV', 'VH', 'HH', 'HV', 'VV+VH', 'HH+HV'];
        return polarizations[Math.floor(Math.random() * polarizations.length)];
    }
    
    generateCoveragePolygon() {
        const centerLat = -90 + Math.random() * 180;
        const centerLon = -180 + Math.random() * 360;
        const size = 1 + Math.random() * 3; // degrees
        
        return [
            [centerLon - size/2, centerLat - size/2],
            [centerLon + size/2, centerLat - size/2],
            [centerLon + size/2, centerLat + size/2],
            [centerLon - size/2, centerLat + size/2],
            [centerLon - size/2, centerLat - size/2]
        ];
    }
    
    generateQuicklookUrl(satellite, timestamp) {
        return `https://quicklook.${satellite}.example.com/${timestamp.replace(/[:.]/g, '')}.jpg`;
    }
    
    handleIncomingData(data) {
        // Store in data stream
        if (!this.dataStreams.has(data.satellite)) {
            this.dataStreams.set(data.satellite, []);
        }
        
        const stream = this.dataStreams.get(data.satellite);
        stream.push(data);
        
        // Keep only recent data (last 100 acquisitions)
        if (stream.length > 100) {
            stream.shift();
        }
        
        // Update satellite info
        this.satellites[data.satellite].orbits.push(data.orbit);
        this.satellites[data.satellite].lastUpdate = data.timestamp;
        
        // Add to processing queue if it covers area of interest
        if (this.coversAreaOfInterest(data.coverage)) {
            this.processingQueue.push(data);
            this.processNextInQueue();
        }
    }
    
    coversAreaOfInterest(coverage) {
        // Check if coverage intersects with monitored volcanoes or fault zones
        // Simplified check - in reality would use proper polygon intersection
        return coverage.cloudCover < 0.5 && Math.random() > 0.7;
    }
    
    setupDataProcessing() {
        // InSAR processor
        this.insarProcessor = new InSARProcessor(this);
        
        // Thermal processor
        this.thermalProcessor = new ThermalProcessor(this);
        
        // Optical processor
        this.opticalProcessor = new OpticalProcessor(this);
        
        // Change detection
        this.changeDetector = new ChangeDetector(this);
    }
    
    async processNextInQueue() {
        if (this.processingQueue.length === 0) return;
        
        const data = this.processingQueue.shift();
        
        // Route to appropriate processor
        if (data.satellite.includes('sentinel') || data.satellite.includes('alos')) {
            await this.processInSAR(data);
        } else if (data.satellite === 'modis' || data.satellite === 'viirs') {
            await this.processThermal(data);
        } else if (data.satellite === 'landsat9') {
            await this.processOptical(data);
        }
    }
    
    async processInSAR(data) {
        // Find matching pair for interferometry
        const pair = this.findInterferometricPair(data);
        if (!pair) {
            console.log('No suitable pair found for interferometry');
            return;
        }
        
        // Simulate InSAR processing
        const result = await this.insarProcessor.process(data, pair);
        
        return {
            type: 'InSAR',
            timestamp: new Date().toISOString(),
            master: data,
            slave: pair,
            interferogram: result.interferogram,
            coherence: result.coherence,
            displacement: result.displacement,
            los: result.los, // Line of sight vector
            quality: this.assessQuality(result)
        };
    }
    
    findInterferometricPair(acquisition) {
        const stream = this.dataStreams.get(acquisition.satellite) || [];
        
        // Find acquisitions with same track and frame
        const candidates = stream.filter(acq => 
            acq.orbit.direction === acquisition.orbit.direction &&
            Math.abs(acq.orbit.number - acquisition.orbit.number) % this.satellites[acquisition.satellite].revisit === 0 &&
            acq.timestamp !== acquisition.timestamp
        );
        
        if (candidates.length === 0) return null;
        
        // Select best pair based on temporal and perpendicular baseline
        return candidates.reduce((best, candidate) => {
            const temporalBaseline = Math.abs(new Date(candidate.timestamp) - new Date(acquisition.timestamp));
            const perpendicularBaseline = Math.abs(candidate.orbit.altitude - acquisition.orbit.altitude);
            
            // Optimal baselines: temporal < 30 days, perpendicular < 300m
            const score = (temporalBaseline / (30 * 24 * 3600 * 1000)) + (perpendicularBaseline / 300);
            
            if (!best || score < best.score) {
                return { ...candidate, score };
            }
            return best;
        }, null);
    }
    
    async processThermal(data) {
        const result = await this.thermalProcessor.process(data);
        
        return {
            type: 'Thermal',
            timestamp: new Date().toISOString(),
            acquisition: data,
            hotspots: result.hotspots,
            anomalies: result.anomalies,
            temperature: result.temperature,
            radiance: result.radiance,
            volcanicThermalOutput: result.vto // MW
        };
    }
    
    async processOptical(data) {
        const result = await this.opticalProcessor.process(data);
        
        return {
            type: 'Optical',
            timestamp: new Date().toISOString(),
            acquisition: data,
            indices: result.indices, // NDVI, NDWI, etc.
            classification: result.classification,
            changeMap: result.changeMap,
            ashDetection: result.ashDetection
        };
    }
    
    assessQuality(result) {
        // Quality assessment based on coherence and coverage
        const avgCoherence = result.coherence.reduce((a, b) => a + b, 0) / result.coherence.length;
        const coverage = result.validPixels / result.totalPixels;
        
        if (avgCoherence > 0.7 && coverage > 0.8) return 'excellent';
        if (avgCoherence > 0.5 && coverage > 0.6) return 'good';
        if (avgCoherence > 0.3 && coverage > 0.4) return 'fair';
        return 'poor';
    }
    
    // Real-time data access methods
    getLatestAcquisitions(satellite = null, limit = 10) {
        if (satellite) {
            const stream = this.dataStreams.get(satellite) || [];
            return stream.slice(-limit);
        }
        
        // Return latest from all satellites
        const allData = [];
        this.dataStreams.forEach((stream, sat) => {
            allData.push(...stream.slice(-limit).map(d => ({...d, satellite: sat})));
        });
        
        return allData.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, limit);
    }
    
    getProcessingStatus() {
        return {
            queueLength: this.processingQueue.length,
            activeProcessors: {
                insar: this.insarProcessor?.isProcessing || false,
                thermal: this.thermalProcessor?.isProcessing || false,
                optical: this.opticalProcessor?.isProcessing || false
            },
            totalProcessed: this.getTotalProcessed(),
            lastUpdate: new Date().toISOString()
        };
    }
    
    getTotalProcessed() {
        let total = 0;
        this.dataStreams.forEach(stream => {
            total += stream.length;
        });
        return total;
    }
    
    // Subscribe to specific area updates
    subscribeToArea(bounds, callback) {
        const subscription = {
            id: Math.random().toString(36).substr(2, 9),
            bounds: bounds,
            callback: callback,
            active: true
        };
        
        // Check incoming data against subscription
        this.areaSubscriptions = this.areaSubscriptions || [];
        this.areaSubscriptions.push(subscription);
        
        return subscription.id;
    }
    
    unsubscribe(subscriptionId) {
        if (this.areaSubscriptions) {
            const index = this.areaSubscriptions.findIndex(s => s.id === subscriptionId);
            if (index !== -1) {
                this.areaSubscriptions.splice(index, 1);
            }
        }
    }
}

// InSAR Processor
class InSARProcessor {
    constructor(parent) {
        this.parent = parent;
        this.isProcessing = false;
    }
    
    async process(master, slave) {
        this.isProcessing = true;
        
        // Simulate processing steps
        await this.simulateDelay(100);
        
        // Coregistration
        const coregistered = await this.coregister(master, slave);
        
        // Interferogram generation
        const interferogram = await this.generateInterferogram(coregistered);
        
        // Filtering
        const filtered = await this.filter(interferogram);
        
        // Phase unwrapping
        const unwrapped = await this.unwrap(filtered);
        
        // Displacement calculation
        const displacement = await this.calculateDisplacement(unwrapped);
        
        this.isProcessing = false;
        
        return {
            interferogram: filtered,
            coherence: this.calculateCoherence(interferogram),
            displacement: displacement,
            los: this.calculateLOS(master),
            validPixels: Math.floor(Math.random() * 1000000),
            totalPixels: 1000000
        };
    }
    
    async coregister(master, slave) {
        // Simulate coregistration
        return { master, slave, shifts: { x: Math.random() * 10, y: Math.random() * 10 } };
    }
    
    async generateInterferogram(coregistered) {
        // Generate synthetic interferogram
        const size = 1000;
        const interferogram = new Float32Array(size * size);
        
        for (let i = 0; i < interferogram.length; i++) {
            // Simulate interference pattern
            const x = i % size;
            const y = Math.floor(i / size);
            interferogram[i] = Math.sin(x * 0.1) * Math.cos(y * 0.1) + Math.random() * 0.2;
        }
        
        return interferogram;
    }
    
    async filter(interferogram) {
        // Goldstein filter simulation
        return interferogram.map(val => val * 0.9 + Math.random() * 0.05);
    }
    
    async unwrap(filtered) {
        // SNAPHU unwrapping simulation
        return filtered.map(val => val * 2 * Math.PI);
    }
    
    async calculateDisplacement(unwrapped) {
        // Convert phase to displacement
        const wavelength = 5.6; // cm for Sentinel-1
        return unwrapped.map(phase => (phase * wavelength) / (4 * Math.PI));
    }
    
    calculateCoherence(interferogram) {
        // Simulate coherence calculation
        const coherence = new Float32Array(interferogram.length);
        for (let i = 0; i < coherence.length; i++) {
            coherence[i] = 0.3 + Math.random() * 0.7;
        }
        return coherence;
    }
    
    calculateLOS(acquisition) {
        // Line of sight vector
        const inc = acquisition.acquisition.incidenceAngle * Math.PI / 180;
        const azi = acquisition.orbit.direction === 'ascending' ? 348 : 192; // degrees
        const aziRad = azi * Math.PI / 180;
        
        return {
            east: -Math.sin(inc) * Math.sin(aziRad),
            north: -Math.sin(inc) * Math.cos(aziRad),
            up: Math.cos(inc)
        };
    }
    
    simulateDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Thermal Processor
class ThermalProcessor {
    constructor(parent) {
        this.parent = parent;
        this.isProcessing = false;
        this.backgroundModel = new Map();
    }
    
    async process(data) {
        this.isProcessing = true;
        
        // Extract thermal bands
        const thermalData = await this.extractThermalBands(data);
        
        // Atmospheric correction
        const corrected = await this.atmosphericCorrection(thermalData);
        
        // Temperature retrieval
        const temperature = await this.retrieveTemperature(corrected);
        
        // Anomaly detection
        const anomalies = await this.detectAnomalies(temperature);
        
        // Hotspot detection
        const hotspots = await this.detectHotspots(temperature);
        
        // Calculate volcanic thermal output
        const vto = await this.calculateVTO(hotspots, temperature);
        
        this.isProcessing = false;
        
        return {
            hotspots: hotspots,
            anomalies: anomalies,
            temperature: temperature,
            radiance: corrected,
            vto: vto
        };
    }
    
    async extractThermalBands(data) {
        // Simulate thermal band extraction
        const size = 500;
        const thermal = new Float32Array(size * size);
        
        for (let i = 0; i < thermal.length; i++) {
            // Simulate thermal radiance values
            thermal[i] = 280 + Math.random() * 40; // Kelvin
        }
        
        return thermal;
    }
    
    async atmosphericCorrection(thermal) {
        // Simple atmospheric correction
        return thermal.map(val => val - 5 + Math.random() * 2);
    }
    
    async retrieveTemperature(radiance) {
        // Split-window algorithm for temperature retrieval
        return radiance.map(val => {
            // Convert radiance to temperature using Planck's law
            const c1 = 1.19104e8; // W·μm⁴/m²/sr
            const c2 = 14387.7; // μm·K
            const lambda = 11; // μm (thermal band)
            
            return c2 / (lambda * Math.log(c1 / (Math.pow(lambda, 5) * val) + 1));
        });
    }
    
    async detectAnomalies(temperature) {
        // Statistical anomaly detection
        const mean = temperature.reduce((a, b) => a + b, 0) / temperature.length;
        const std = Math.sqrt(temperature.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / temperature.length);
        
        const anomalies = [];
        const threshold = mean + this.parent.thermalParams.anomalyThreshold * std;
        
        for (let i = 0; i < temperature.length; i++) {
            if (temperature[i] > threshold) {
                anomalies.push({
                    index: i,
                    temperature: temperature[i],
                    anomaly: (temperature[i] - mean) / std
                });
            }
        }
        
        return anomalies;
    }
    
    async detectHotspots(temperature) {
        // MODVOLC algorithm adaptation
        const hotspots = [];
        const size = Math.sqrt(temperature.length);
        
        for (let i = 0; i < temperature.length; i++) {
            const x = i % size;
            const y = Math.floor(i / size);
            
            // Check if pixel is significantly hotter than neighbors
            const neighbors = this.getNeighbors(temperature, x, y, size);
            const neighborMean = neighbors.reduce((a, b) => a + b, 0) / neighbors.length;
            
            if (temperature[i] > neighborMean + 10) {
                hotspots.push({
                    x: x,
                    y: y,
                    temperature: temperature[i],
                    intensity: temperature[i] - neighborMean
                });
            }
        }
        
        return hotspots;
    }
    
    getNeighbors(data, x, y, size) {
        const neighbors = [];
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                if (dx === 0 && dy === 0) continue;
                const nx = x + dx;
                const ny = y + dy;
                if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                    neighbors.push(data[ny * size + nx]);
                }
            }
        }
        return neighbors;
    }
    
    async calculateVTO(hotspots, temperature) {
        // Volcanic Thermal Output calculation
        if (hotspots.length === 0) return 0;
        
        const pixelArea = 1000 * 1000; // m² (1km x 1km for MODIS)
        const stefanBoltzmann = 5.67e-8; // W/m²/K⁴
        const emissivity = 0.95;
        
        let totalPower = 0;
        hotspots.forEach(hotspot => {
            const T = hotspot.temperature;
            const power = emissivity * stefanBoltzmann * Math.pow(T, 4) * pixelArea;
            totalPower += power;
        });
        
        return totalPower / 1e6; // Convert to MW
    }
}

// Optical Processor
class OpticalProcessor {
    constructor(parent) {
        this.parent = parent;
        this.isProcessing = false;
    }
    
    async process(data) {
        this.isProcessing = true;
        
        // Extract bands
        const bands = await this.extractBands(data);
        
        // Calculate indices
        const indices = await this.calculateIndices(bands);
        
        // Land cover classification
        const classification = await this.classify(bands, indices);
        
        // Change detection
        const changeMap = await this.detectChanges(bands);
        
        // Volcanic ash detection
        const ashDetection = await this.detectAsh(bands);
        
        this.isProcessing = false;
        
        return {
            indices: indices,
            classification: classification,
            changeMap: changeMap,
            ashDetection: ashDetection
        };
    }
    
    async extractBands(data) {
        // Simulate multispectral bands
        const size = 500;
        return {
            red: new Float32Array(size * size).fill(0).map(() => Math.random()),
            green: new Float32Array(size * size).fill(0).map(() => Math.random()),
            blue: new Float32Array(size * size).fill(0).map(() => Math.random()),
            nir: new Float32Array(size * size).fill(0).map(() => Math.random()),
            swir1: new Float32Array(size * size).fill(0).map(() => Math.random()),
            swir2: new Float32Array(size * size).fill(0).map(() => Math.random())
        };
    }
    
    async calculateIndices(bands) {
        const indices = {
            ndvi: [], // Normalized Difference Vegetation Index
            ndwi: [], // Normalized Difference Water Index
            nbr: [],  // Normalized Burn Ratio
            bai: []   // Burn Area Index
        };
        
        for (let i = 0; i < bands.red.length; i++) {
            // NDVI = (NIR - Red) / (NIR + Red)
            indices.ndvi[i] = (bands.nir[i] - bands.red[i]) / (bands.nir[i] + bands.red[i] + 0.001);
            
            // NDWI = (Green - NIR) / (Green + NIR)
            indices.ndwi[i] = (bands.green[i] - bands.nir[i]) / (bands.green[i] + bands.nir[i] + 0.001);
            
            // NBR = (NIR - SWIR2) / (NIR + SWIR2)
            indices.nbr[i] = (bands.nir[i] - bands.swir2[i]) / (bands.nir[i] + bands.swir2[i] + 0.001);
            
            // BAI = 1 / ((0.1 - Red)² + (0.06 - NIR)²)
            indices.bai[i] = 1 / (Math.pow(0.1 - bands.red[i], 2) + Math.pow(0.06 - bands.nir[i], 2) + 0.001);
        }
        
        return indices;
    }
    
    async classify(bands, indices) {
        // Simple classification based on indices
        const classification = new Uint8Array(bands.red.length);
        
        for (let i = 0; i < classification.length; i++) {
            if (indices.ndwi[i] > 0.3) {
                classification[i] = 1; // Water
            } else if (indices.ndvi[i] > 0.5) {
                classification[i] = 2; // Dense vegetation
            } else if (indices.ndvi[i] > 0.2) {
                classification[i] = 3; // Sparse vegetation
            } else if (indices.nbr[i] < -0.1) {
                classification[i] = 4; // Burned area
            } else {
                classification[i] = 5; // Bare soil/rock
            }
        }
        
        return classification;
    }
    
    async detectChanges(bands) {
        // Simulate change detection
        const changeMap = new Float32Array(bands.red.length);
        
        for (let i = 0; i < changeMap.length; i++) {
            // Random change magnitude
            changeMap[i] = Math.random() * 0.3 - 0.15;
        }
        
        return changeMap;
    }
    
    async detectAsh(bands) {
        // Volcanic ash detection using brightness temperature difference
        const ashPixels = [];
        const size = Math.sqrt(bands.red.length);
        
        for (let i = 0; i < bands.red.length; i++) {
            // Simplified ash detection algorithm
            const btd = bands.swir1[i] - bands.swir2[i]; // Brightness temp difference
            
            if (btd < -0.5) {
                ashPixels.push({
                    x: i % size,
                    y: Math.floor(i / size),
                    confidence: Math.abs(btd)
                });
            }
        }
        
        return {
            detected: ashPixels.length > 0,
            pixels: ashPixels,
            coverage: (ashPixels.length / bands.red.length) * 100 // percentage
        };
    }
}

// Change Detector
class ChangeDetector {
    constructor(parent) {
        this.parent = parent;
        this.baseline = new Map();
    }
    
    async detectChanges(current, previous) {
        if (!previous) return null;
        
        // Calculate difference
        const changes = {
            magnitude: [],
            direction: [],
            significance: []
        };
        
        // Pixel-wise comparison
        for (let i = 0; i < current.length; i++) {
            const diff = current[i] - previous[i];
            changes.magnitude[i] = Math.abs(diff);
            changes.direction[i] = Math.sign(diff);
            changes.significance[i] = Math.abs(diff) > 0.1 ? 1 : 0;
        }
        
        return changes;
    }
    
    updateBaseline(data, location) {
        const key = `${location.lat}_${location.lon}`;
        
        if (!this.baseline.has(key)) {
            this.baseline.set(key, []);
        }
        
        const history = this.baseline.get(key);
        history.push(data);
        
        // Keep last 30 days of data
        const cutoff = Date.now() - 30 * 24 * 3600 * 1000;
        const filtered = history.filter(d => new Date(d.timestamp) > cutoff);
        this.baseline.set(key, filtered);
    }
}

// Export for use in HTML files
if (typeof window !== 'undefined') {
    window.SatelliteDataIntegration = SatelliteDataIntegration;
}