// GPS Velocity Field Visualization Module
// Real-time GNSS data integration and strain rate calculation
// Based on 2023-2024 geodetic research

class GPSVelocityField {
    constructor(map, scene) {
        this.map = map; // Leaflet map instance
        this.scene = scene; // Three.js scene for 3D visualization
        
        // GPS networks
        this.networks = {
            unavco: {
                name: 'UNAVCO',
                url: 'https://www.unavco.org/data/',
                stations: [],
                color: '#ff6b6b'
            },
            igs: {
                name: 'IGS',
                url: 'https://igs.org/data/',
                stations: [],
                color: '#4ecdc4'
            },
            euref: {
                name: 'EUREF',
                url: 'https://www.epncb.oma.be/',
                stations: [],
                color: '#45b7d1'
            },
            geonet: {
                name: 'GeoNet NZ',
                url: 'https://www.geonet.org.nz/',
                stations: [],
                color: '#96ceb4'
            },
            jgn: {
                name: 'Japan GEONET',
                url: 'https://www.gsi.go.jp/',
                stations: [],
                color: '#ffd93d'
            }
        };
        
        // Visualization layers
        this.layers = {
            vectors: null,
            interpolated: null,
            strain: null,
            rotation: null,
            dilatation: null
        };
        
        // Processing parameters
        this.parameters = {
            vectorScale: 100,
            interpolationMethod: 'kriging',
            strainCalculation: 'least_squares',
            referenceFrame: 'ITRF2020',
            epoch: '2024.0',
            minStations: 3,
            maxDistance: 100, // km
            smoothingRadius: 50 // km
        };
        
        this.initializeGPSData();
        this.setupVisualization();
    }
    
    initializeGPSData() {
        // Generate synthetic GPS station data
        this.generateGlobalNetwork();
        this.generateRegionalNetworks();
        
        // Initialize real-time processing
        this.processor = new VelocityProcessor(this);
        this.strainCalculator = new StrainRateCalculator(this);
        
        // Start data updates
        this.startDataStream();
    }
    
    generateGlobalNetwork() {
        // Generate global IGS stations
        const igsStations = [];
        
        // Major tectonic boundaries
        const boundaries = [
            { lat: 37.5, lon: -122.5, name: 'P149' }, // San Andreas
            { lat: 35.7, lon: 139.7, name: 'TSK2' }, // Japan
            { lat: -41.3, lon: 174.8, name: 'WGTN' }, // New Zealand
            { lat: 64.1, lon: -21.9, name: 'REYK' }, // Iceland
            { lat: 38.0, lon: 23.7, name: 'ATH1' }, // Greece
            { lat: -33.9, lon: 18.4, name: 'SUTH' }, // South Africa
            { lat: 19.4, lon: -99.1, name: 'MEX1' }, // Mexico
            { lat: -23.5, lon: -70.4, name: 'ANTC' } // Chile
        ];
        
        boundaries.forEach(station => {
            igsStations.push(this.createGPSStation(station));
        });
        
        // Add more stations on a grid
        for (let lat = -60; lat <= 60; lat += 30) {
            for (let lon = -180; lon <= 180; lon += 45) {
                if (Math.random() > 0.5) {
                    igsStations.push(this.createGPSStation({
                        lat: lat + Math.random() * 10 - 5,
                        lon: lon + Math.random() * 10 - 5,
                        name: `IGS${Math.abs(lat)}${Math.abs(lon)}`
                    }));
                }
            }
        }
        
        this.networks.igs.stations = igsStations;
    }
    
    generateRegionalNetworks() {
        // Western US (UNAVCO)
        for (let lat = 32; lat <= 49; lat += 2) {
            for (let lon = -125; lon <= -104; lon += 2) {
                if (Math.random() > 0.3) {
                    this.networks.unavco.stations.push(this.createGPSStation({
                        lat: lat + Math.random() - 0.5,
                        lon: lon + Math.random() - 0.5,
                        name: `UN${Math.floor(Math.random() * 1000)}`
                    }));
                }
            }
        }
        
        // Japan (GEONET)
        for (let lat = 30; lat <= 46; lat += 1) {
            for (let lon = 128; lon <= 146; lon += 1) {
                if (Math.random() > 0.2) {
                    this.networks.jgn.stations.push(this.createGPSStation({
                        lat: lat + Math.random() * 0.5,
                        lon: lon + Math.random() * 0.5,
                        name: `JP${Math.floor(Math.random() * 10000)}`
                    }));
                }
            }
        }
        
        // Europe (EUREF)
        for (let lat = 35; lat <= 70; lat += 3) {
            for (let lon = -10; lon <= 40; lon += 3) {
                if (Math.random() > 0.4) {
                    this.networks.euref.stations.push(this.createGPSStation({
                        lat: lat + Math.random() - 0.5,
                        lon: lon + Math.random() - 0.5,
                        name: `EU${Math.floor(Math.random() * 1000)}`
                    }));
                }
            }
        }
        
        // New Zealand (GeoNet)
        for (let lat = -47; lat <= -34; lat += 1) {
            for (let lon = 166; lon <= 179; lon += 1) {
                if (Math.random() > 0.3) {
                    this.networks.geonet.stations.push(this.createGPSStation({
                        lat: lat + Math.random() * 0.5,
                        lon: lon + Math.random() * 0.5,
                        name: `NZ${Math.floor(Math.random() * 100)}`
                    }));
                }
            }
        }
    }
    
    createGPSStation(config) {
        // Simulate plate motion based on location
        const velocity = this.getPlateVelocity(config.lat, config.lon);
        
        return {
            id: config.name,
            name: config.name,
            lat: config.lat,
            lon: config.lon,
            height: Math.random() * 1000, // meters
            
            // Velocities in mm/year
            ve: velocity.east + (Math.random() - 0.5) * 5,
            vn: velocity.north + (Math.random() - 0.5) * 5,
            vu: (Math.random() - 0.5) * 2,
            
            // Uncertainties in mm/year
            se: 0.5 + Math.random() * 0.5,
            sn: 0.5 + Math.random() * 0.5,
            su: 1.0 + Math.random(),
            
            // Time series
            timeSeries: this.generateTimeSeries(),
            
            // Quality metrics
            quality: {
                dataSpan: 5 + Math.random() * 15, // years
                completeness: 0.8 + Math.random() * 0.2,
                stability: Math.random() > 0.1
            },
            
            // Additional data
            lastUpdate: Date.now(),
            status: 'active'
        };
    }
    
    getPlateVelocity(lat, lon) {
        // Simplified plate motion model
        // Based on MORVEL-2010 plate motion model
        
        // Pacific Plate
        if (lon < -130 && lat > 20 && lat < 60) {
            return { east: -50, north: 35 };
        }
        
        // North American Plate
        if (lon > -130 && lon < -60 && lat > 25) {
            return { east: -20, north: -5 };
        }
        
        // Eurasian Plate
        if (lon > -10 && lon < 140 && lat > 35) {
            return { east: 20, north: 5 };
        }
        
        // Australian Plate
        if (lon > 110 && lon < 160 && lat < -10) {
            return { east: 65, north: 35 };
        }
        
        // South American Plate
        if (lon > -80 && lon < -35 && lat < 10) {
            return { east: 10, north: -8 };
        }
        
        // African Plate
        if (lon > -20 && lon < 50 && lat < 35 && lat > -35) {
            return { east: 15, north: 20 };
        }
        
        // Antarctic Plate
        if (lat < -60) {
            return { east: 5, north: 5 };
        }
        
        // Default (mid-ocean or boundary)
        return { east: Math.random() * 20 - 10, north: Math.random() * 20 - 10 };
    }
    
    generateTimeSeries() {
        // Generate synthetic time series data
        const series = {
            times: [],
            east: [],
            north: [],
            up: []
        };
        
        const startTime = Date.now() - 365 * 24 * 3600 * 1000; // 1 year ago
        const interval = 24 * 3600 * 1000; // Daily
        
        for (let i = 0; i < 365; i++) {
            series.times.push(startTime + i * interval);
            
            // Linear trend + seasonal + noise
            const t = i / 365;
            series.east.push(
                20 * t + // Linear trend
                5 * Math.sin(2 * Math.PI * t) + // Annual
                2 * Math.sin(4 * Math.PI * t) + // Semi-annual
                (Math.random() - 0.5) * 2 // Noise
            );
            series.north.push(
                15 * t +
                3 * Math.sin(2 * Math.PI * t + Math.PI/4) +
                1 * Math.sin(4 * Math.PI * t + Math.PI/2) +
                (Math.random() - 0.5) * 2
            );
            series.up.push(
                -2 * t +
                10 * Math.sin(2 * Math.PI * t + Math.PI/2) +
                (Math.random() - 0.5) * 5
            );
        }
        
        return series;
    }
    
    setupVisualization() {
        // Create Leaflet layers
        this.createVectorLayer();
        this.createInterpolationLayer();
        this.createStrainLayer();
        
        // Create Three.js 3D visualization
        this.create3DVelocityField();
    }
    
    createVectorLayer() {
        // Velocity vector layer
        this.layers.vectors = L.layerGroup();
        
        // Add all stations
        Object.values(this.networks).forEach(network => {
            network.stations.forEach(station => {
                this.addVelocityVector(station, network.color);
            });
        });
        
        if (this.map) {
            this.layers.vectors.addTo(this.map);
        }
    }
    
    addVelocityVector(station, color) {
        // Create arrow marker for velocity
        const magnitude = Math.sqrt(station.ve * station.ve + station.vn * station.vn);
        const angle = Math.atan2(station.ve, station.vn) * 180 / Math.PI;
        
        // Scale factor for visualization
        const scale = this.parameters.vectorScale / 1000;
        
        // End point of vector
        const endLat = station.lat + station.vn * scale / 111; // degrees
        const endLon = station.lon + station.ve * scale / (111 * Math.cos(station.lat * Math.PI / 180));
        
        // Create polyline
        const vector = L.polyline(
            [[station.lat, station.lon], [endLat, endLon]],
            {
                color: color,
                weight: 2,
                opacity: 0.8
            }
        );
        
        // Add arrowhead
        const arrowHead = L.polylineDecorator(vector, {
            patterns: [
                {
                    offset: '100%',
                    repeat: 0,
                    symbol: L.Symbol.arrowHead({
                        pixelSize: 10,
                        polygon: true,
                        pathOptions: { 
                            fillOpacity: 1,
                            weight: 0,
                            color: color
                        }
                    })
                }
            ]
        });
        
        // Station marker
        const marker = L.circleMarker([station.lat, station.lon], {
            radius: 4,
            fillColor: color,
            fillOpacity: 0.8,
            color: 'white',
            weight: 1
        });
        
        // Popup with station info
        marker.bindPopup(`
            <strong>${station.name}</strong><br>
            Ve: ${station.ve.toFixed(1)} ± ${station.se.toFixed(1)} mm/yr<br>
            Vn: ${station.vn.toFixed(1)} ± ${station.sn.toFixed(1)} mm/yr<br>
            Vu: ${station.vu.toFixed(1)} ± ${station.su.toFixed(1)} mm/yr<br>
            Magnitude: ${magnitude.toFixed(1)} mm/yr<br>
            Azimuth: ${angle.toFixed(1)}°
        `);
        
        if (this.layers.vectors) {
            this.layers.vectors.addLayer(vector);
            this.layers.vectors.addLayer(arrowHead);
            this.layers.vectors.addLayer(marker);
        }
    }
    
    createInterpolationLayer() {
        // Create interpolated velocity field
        const bounds = this.map ? this.map.getBounds() : 
            L.latLngBounds([[-90, -180], [90, 180]]);
        
        const gridSize = 50;
        const latStep = (bounds.getNorth() - bounds.getSouth()) / gridSize;
        const lonStep = (bounds.getEast() - bounds.getWest()) / gridSize;
        
        // Interpolation grid
        const grid = [];
        
        for (let i = 0; i <= gridSize; i++) {
            for (let j = 0; j <= gridSize; j++) {
                const lat = bounds.getSouth() + i * latStep;
                const lon = bounds.getWest() + j * lonStep;
                
                const velocity = this.interpolateVelocity(lat, lon);
                grid.push({
                    lat: lat,
                    lon: lon,
                    ve: velocity.east,
                    vn: velocity.north,
                    vu: velocity.up
                });
            }
        }
        
        // Create heat map layer
        this.layers.interpolated = this.createHeatMap(grid);
    }
    
    interpolateVelocity(lat, lon) {
        // Kriging interpolation
        const stations = this.getAllStations();
        const nearbyStations = this.findNearbyStations(lat, lon, stations);
        
        if (nearbyStations.length < this.parameters.minStations) {
            return { east: 0, north: 0, up: 0 };
        }
        
        // Simple inverse distance weighting for now
        let totalWeight = 0;
        let ve = 0, vn = 0, vu = 0;
        
        nearbyStations.forEach(station => {
            const distance = this.calculateDistance(lat, lon, station.lat, station.lon);
            const weight = 1 / (distance + 1);
            
            ve += station.ve * weight;
            vn += station.vn * weight;
            vu += station.vu * weight;
            totalWeight += weight;
        });
        
        return {
            east: ve / totalWeight,
            north: vn / totalWeight,
            up: vu / totalWeight
        };
    }
    
    getAllStations() {
        const allStations = [];
        Object.values(this.networks).forEach(network => {
            allStations.push(...network.stations);
        });
        return allStations;
    }
    
    findNearbyStations(lat, lon, stations) {
        const nearby = [];
        const maxDist = this.parameters.maxDistance;
        
        stations.forEach(station => {
            const distance = this.calculateDistance(lat, lon, station.lat, station.lon);
            if (distance < maxDist) {
                nearby.push({ ...station, distance });
            }
        });
        
        // Sort by distance and return closest stations
        return nearby.sort((a, b) => a.distance - b.distance).slice(0, 10);
    }
    
    calculateDistance(lat1, lon1, lat2, lon2) {
        // Haversine formula
        const R = 6371; // Earth radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }
    
    createHeatMap(grid) {
        // Create heat map from velocity magnitude
        const heatData = grid.map(point => {
            const magnitude = Math.sqrt(point.ve * point.ve + point.vn * point.vn);
            return [point.lat, point.lon, magnitude];
        });
        
        if (typeof L !== 'undefined' && L.heatLayer) {
            return L.heatLayer(heatData, {
                radius: 25,
                blur: 15,
                maxZoom: 10,
                gradient: {
                    0.0: 'blue',
                    0.25: 'cyan',
                    0.5: 'green',
                    0.75: 'yellow',
                    1.0: 'red'
                }
            });
        }
        
        return null;
    }
    
    createStrainLayer() {
        // Calculate and visualize strain rate
        const strainField = this.strainCalculator.calculateStrainField();
        
        // Create contour layer for strain
        this.layers.strain = L.layerGroup();
        
        // Add principal strain axes
        strainField.forEach(point => {
            this.addStrainTensor(point);
        });
        
        // Create dilatation layer
        this.layers.dilatation = this.createDilatationLayer(strainField);
        
        // Create rotation layer
        this.layers.rotation = this.createRotationLayer(strainField);
    }
    
    addStrainTensor(point) {
        // Visualize strain tensor as crosses
        const scale = 1000; // Scaling factor
        
        // Principal strain directions
        const theta = 0.5 * Math.atan2(2 * point.exy, point.exx - point.eyy);
        
        // Principal strain magnitudes
        const e1 = 0.5 * (point.exx + point.eyy) + 
                   Math.sqrt(0.25 * Math.pow(point.exx - point.eyy, 2) + point.exy * point.exy);
        const e2 = 0.5 * (point.exx + point.eyy) - 
                   Math.sqrt(0.25 * Math.pow(point.exx - point.eyy, 2) + point.exy * point.exy);
        
        // Extension (red) and compression (blue)
        if (e1 > 0) {
            const extensionLine = this.createStrainLine(point, theta, e1 * scale, 'red');
            this.layers.strain.addLayer(extensionLine);
        }
        
        if (e2 < 0) {
            const compressionLine = this.createStrainLine(point, theta + Math.PI/2, Math.abs(e2) * scale, 'blue');
            this.layers.strain.addLayer(compressionLine);
        }
    }
    
    createStrainLine(point, angle, length, color) {
        const dx = length * Math.cos(angle) / 111000; // Convert to degrees
        const dy = length * Math.sin(angle) / 111000;
        
        return L.polyline([
            [point.lat - dy/2, point.lon - dx/2],
            [point.lat + dy/2, point.lon + dx/2]
        ], {
            color: color,
            weight: 2,
            opacity: 0.6
        });
    }
    
    createDilatationLayer(strainField) {
        // Dilatation (volume change) visualization
        const dilatationData = strainField.map(point => {
            const dilatation = point.exx + point.eyy; // Trace of strain tensor
            return {
                lat: point.lat,
                lon: point.lon,
                value: dilatation
            };
        });
        
        // Create color-coded markers
        const layer = L.layerGroup();
        dilatationData.forEach(point => {
            const color = point.value > 0 ? '#ff0000' : '#0000ff';
            const opacity = Math.min(Math.abs(point.value) * 10000, 1);
            
            const marker = L.circleMarker([point.lat, point.lon], {
                radius: 5,
                fillColor: color,
                fillOpacity: opacity,
                weight: 0
            });
            
            layer.addLayer(marker);
        });
        
        return layer;
    }
    
    createRotationLayer(strainField) {
        // Rotation rate visualization
        const rotationData = strainField.map(point => {
            const rotation = 0.5 * (point.dVn_dx - point.dVe_dy); // Vorticity
            return {
                lat: point.lat,
                lon: point.lon,
                value: rotation
            };
        });
        
        // Create rotation symbols
        const layer = L.layerGroup();
        rotationData.forEach(point => {
            if (Math.abs(point.value) > 1e-9) {
                const icon = L.divIcon({
                    html: point.value > 0 ? '↻' : '↺',
                    className: 'rotation-icon',
                    iconSize: [20, 20]
                });
                
                const marker = L.marker([point.lat, point.lon], { icon });
                layer.addLayer(marker);
            }
        });
        
        return layer;
    }
    
    create3DVelocityField() {
        if (!this.scene) return;
        
        // Create 3D velocity field visualization
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        
        const allStations = this.getAllStations();
        
        allStations.forEach(station => {
            // Convert lat/lon to 3D coordinates
            const pos = this.latLonToVector3(station.lat, station.lon, 100);
            
            // Start point
            positions.push(pos.x, pos.y, pos.z);
            
            // End point (scaled velocity)
            const velocity = new THREE.Vector3(
                station.ve * 0.1,
                station.vn * 0.1,
                station.vu * 0.1
            );
            positions.push(
                pos.x + velocity.x,
                pos.y + velocity.y,
                pos.z + velocity.z
            );
            
            // Color based on magnitude
            const magnitude = velocity.length();
            const color = new THREE.Color();
            color.setHSL((1 - magnitude / 10) * 0.7, 1, 0.5);
            
            colors.push(color.r, color.g, color.b);
            colors.push(color.r, color.g, color.b);
        });
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 2
        });
        
        const velocityField = new THREE.LineSegments(geometry, material);
        this.scene.add(velocityField);
        
        // Add station points
        const pointGeometry = new THREE.BufferGeometry();
        const pointPositions = [];
        
        allStations.forEach(station => {
            const pos = this.latLonToVector3(station.lat, station.lon, 100);
            pointPositions.push(pos.x, pos.y, pos.z);
        });
        
        pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(pointPositions, 3));
        
        const pointMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 3,
            sizeAttenuation: false
        });
        
        const stationPoints = new THREE.Points(pointGeometry, pointMaterial);
        this.scene.add(stationPoints);
    }
    
    latLonToVector3(lat, lon, radius) {
        const phi = (90 - lat) * Math.PI / 180;
        const theta = (lon + 180) * Math.PI / 180;
        
        return new THREE.Vector3(
            -radius * Math.sin(phi) * Math.cos(theta),
            radius * Math.cos(phi),
            radius * Math.sin(phi) * Math.sin(theta)
        );
    }
    
    startDataStream() {
        // Simulate real-time updates
        setInterval(() => {
            this.updateStations();
            this.processData();
        }, 5000); // Update every 5 seconds
    }
    
    updateStations() {
        // Update station positions with simulated real-time data
        Object.values(this.networks).forEach(network => {
            network.stations.forEach(station => {
                // Add small random changes to simulate real-time updates
                const dt = 5 / (365 * 24 * 3600); // 5 seconds in years
                
                // Update position
                station.timeSeries.east.push(
                    station.timeSeries.east[station.timeSeries.east.length - 1] + 
                    station.ve * dt + (Math.random() - 0.5) * 0.1
                );
                station.timeSeries.north.push(
                    station.timeSeries.north[station.timeSeries.north.length - 1] + 
                    station.vn * dt + (Math.random() - 0.5) * 0.1
                );
                station.timeSeries.up.push(
                    station.timeSeries.up[station.timeSeries.up.length - 1] + 
                    station.vu * dt + (Math.random() - 0.5) * 0.2
                );
                
                station.lastUpdate = Date.now();
            });
        });
    }
    
    processData() {
        // Process velocity field and strain
        this.processor.process();
        this.strainCalculator.update();
        
        // Trigger events for anomalies
        this.detectAnomalies();
    }
    
    detectAnomalies() {
        // Detect velocity anomalies
        const allStations = this.getAllStations();
        
        allStations.forEach(station => {
            // Check for sudden velocity changes
            const recentVelocity = this.calculateRecentVelocity(station);
            const velocityChange = Math.sqrt(
                Math.pow(recentVelocity.east - station.ve, 2) +
                Math.pow(recentVelocity.north - station.vn, 2)
            );
            
            if (velocityChange > 10) { // mm/yr threshold
                this.triggerAnomaly({
                    type: 'velocity_change',
                    station: station,
                    magnitude: velocityChange,
                    timestamp: Date.now()
                });
            }
        });
    }
    
    calculateRecentVelocity(station) {
        // Calculate velocity from recent time series
        const n = 10; // Use last 10 points
        const recent = {
            east: station.timeSeries.east.slice(-n),
            north: station.timeSeries.north.slice(-n)
        };
        
        // Linear fit to get velocity
        const dt = 1 / 365; // Daily data in years
        const veRecent = (recent.east[n-1] - recent.east[0]) / (n * dt);
        const vnRecent = (recent.north[n-1] - recent.north[0]) / (n * dt);
        
        return {
            east: veRecent,
            north: vnRecent
        };
    }
    
    triggerAnomaly(anomaly) {
        // Emit anomaly event
        if (this.onAnomaly) {
            this.onAnomaly(anomaly);
        }
        
        // Visual alert
        if (anomaly.station && this.map) {
            L.circle([anomaly.station.lat, anomaly.station.lon], {
                radius: 50000,
                color: 'red',
                fillOpacity: 0.3,
                weight: 2
            }).addTo(this.map);
        }
    }
}

// Velocity Processor
class VelocityProcessor {
    constructor(parent) {
        this.parent = parent;
    }
    
    process() {
        // Process velocity field
        this.removeOutliers();
        this.applyReferenceFrame();
        this.calculateUncertainties();
    }
    
    removeOutliers() {
        // Statistical outlier detection
        const allStations = this.parent.getAllStations();
        const velocities = allStations.map(s => Math.sqrt(s.ve * s.ve + s.vn * s.vn));
        
        const mean = velocities.reduce((a, b) => a + b, 0) / velocities.length;
        const std = Math.sqrt(velocities.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / velocities.length);
        
        allStations.forEach((station, i) => {
            if (Math.abs(velocities[i] - mean) > 3 * std) {
                station.quality.outlier = true;
            }
        });
    }
    
    applyReferenceFrame() {
        // Transform to specified reference frame
        // Simplified - in reality would apply full transformation
        const rotation = { x: 0, y: 0, z: 0.1 }; // mas/yr
        const translation = { x: 0, y: 0, z: 0 }; // mm/yr
        
        const allStations = this.parent.getAllStations();
        allStations.forEach(station => {
            // Apply rotation
            station.ve += rotation.z * station.lat / 206265;
            station.vn -= rotation.z * station.lon / 206265;
        });
    }
    
    calculateUncertainties() {
        // Propagate uncertainties
        const allStations = this.parent.getAllStations();
        allStations.forEach(station => {
            // Formal error propagation
            const correlationCoeff = 0.3;
            station.sMagnitude = Math.sqrt(
                Math.pow(station.ve * station.se / Math.sqrt(station.ve * station.ve + station.vn * station.vn), 2) +
                Math.pow(station.vn * station.sn / Math.sqrt(station.ve * station.ve + station.vn * station.vn), 2) +
                2 * correlationCoeff * station.se * station.sn * station.ve * station.vn / 
                (station.ve * station.ve + station.vn * station.vn)
            );
        });
    }
}

// Strain Rate Calculator
class StrainRateCalculator {
    constructor(parent) {
        this.parent = parent;
        this.strainGrid = [];
    }
    
    calculateStrainField() {
        // Calculate strain rate field from GPS velocities
        const bounds = this.parent.map ? this.parent.map.getBounds() : 
            L.latLngBounds([[-90, -180], [90, 180]]);
        
        const gridSize = 20;
        const latStep = (bounds.getNorth() - bounds.getSouth()) / gridSize;
        const lonStep = (bounds.getEast() - bounds.getWest()) / gridSize;
        
        this.strainGrid = [];
        
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const lat = bounds.getSouth() + (i + 0.5) * latStep;
                const lon = bounds.getWest() + (j + 0.5) * lonStep;
                
                const strain = this.calculateStrainAtPoint(lat, lon);
                if (strain) {
                    this.strainGrid.push(strain);
                }
            }
        }
        
        return this.strainGrid;
    }
    
    calculateStrainAtPoint(lat, lon) {
        // Least squares strain calculation
        const stations = this.parent.findNearbyStations(lat, lon, this.parent.getAllStations());
        
        if (stations.length < this.parent.parameters.minStations) {
            return null;
        }
        
        // Build design matrix
        const A = [];
        const b = [];
        
        stations.forEach(station => {
            const dx = (station.lon - lon) * 111 * Math.cos(lat * Math.PI / 180); // km
            const dy = (station.lat - lat) * 111; // km
            
            // East velocity equation
            A.push([1, 0, dx, dy, 0]);
            b.push(station.ve);
            
            // North velocity equation
            A.push([0, 1, 0, dx, dy]);
            b.push(station.vn);
        });
        
        // Solve least squares
        const solution = this.leastSquares(A, b);
        
        if (!solution) return null;
        
        return {
            lat: lat,
            lon: lon,
            ve0: solution[0], // East velocity at point
            vn0: solution[1], // North velocity at point
            dVe_dx: solution[2] * 1e-6, // Strain rate components (1/year)
            dVe_dy: solution[3] * 1e-6,
            dVn_dx: solution[3] * 1e-6,
            dVn_dy: solution[4] * 1e-6,
            exx: solution[2] * 1e-6, // Normal strain rates
            eyy: solution[4] * 1e-6,
            exy: 0.5 * (solution[3] + solution[3]) * 1e-6 // Shear strain rate
        };
    }
    
    leastSquares(A, b) {
        // Simple least squares solver
        // In reality, would use more robust numerical methods
        try {
            const At = this.transpose(A);
            const AtA = this.multiply(At, A);
            const Atb = this.multiplyVector(At, b);
            return this.solve(AtA, Atb);
        } catch (e) {
            return null;
        }
    }
    
    transpose(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = [];
        
        for (let j = 0; j < cols; j++) {
            result[j] = [];
            for (let i = 0; i < rows; i++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    multiply(a, b) {
        const result = [];
        const rows = a.length;
        const cols = b[0].length;
        const inner = b.length;
        
        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                let sum = 0;
                for (let k = 0; k < inner; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        return result;
    }
    
    multiplyVector(matrix, vector) {
        const result = [];
        for (let i = 0; i < matrix.length; i++) {
            let sum = 0;
            for (let j = 0; j < vector.length; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result.push(sum);
        }
        return result;
    }
    
    solve(A, b) {
        // Gaussian elimination (simplified)
        const n = A.length;
        const augmented = A.map((row, i) => [...row, b[i]]);
        
        // Forward elimination
        for (let i = 0; i < n; i++) {
            // Pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
            
            // Eliminate
            for (let k = i + 1; k < n; k++) {
                const factor = augmented[k][i] / augmented[i][i];
                for (let j = i; j <= n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        // Back substitution
        const solution = new Array(n);
        for (let i = n - 1; i >= 0; i--) {
            solution[i] = augmented[i][n];
            for (let j = i + 1; j < n; j++) {
                solution[i] -= augmented[i][j] * solution[j];
            }
            solution[i] /= augmented[i][i];
        }
        
        return solution;
    }
    
    update() {
        // Update strain calculations
        this.calculateStrainField();
        
        // Check for high strain areas
        this.detectHighStrain();
    }
    
    detectHighStrain() {
        // Detect areas of high strain rate
        const threshold = 1e-7; // 1/year
        
        this.strainGrid.forEach(point => {
            const maxStrain = Math.max(Math.abs(point.exx), Math.abs(point.eyy), Math.abs(point.exy));
            
            if (maxStrain > threshold) {
                this.parent.triggerAnomaly({
                    type: 'high_strain',
                    location: { lat: point.lat, lon: point.lon },
                    strain: maxStrain,
                    timestamp: Date.now()
                });
            }
        });
    }
}

// Export for use in HTML files
if (typeof window !== 'undefined') {
    window.GPSVelocityField = GPSVelocityField;
}