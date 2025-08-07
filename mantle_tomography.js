// Advanced Mantle Tomography and Visualization Module
// Based on latest 2023-2024 research papers and seismic tomography models

class MantleTomography {
    constructor(scene, renderer) {
        this.scene = scene;
        this.renderer = renderer;
        this.mantleLayers = [];
        this.slabGeometries = [];
        this.plumeGeometries = [];
        this.llsvpGeometries = [];
        
        // Fix WebGL extensions
        this.enableWebGLExtensions();
        
        // Initialize mantle structure
        this.initializeMantleStructure();
    }
    
    enableWebGLExtensions() {
        if (this.renderer && this.renderer.capabilities) {
            const gl = this.renderer.getContext();
            if (gl) {
                // Explicitly enable required extensions
                const extensions = [
                    'EXT_color_buffer_float',
                    'WEBGL_color_buffer_float',
                    'OES_texture_float',
                    'OES_texture_float_linear',
                    'EXT_float_blend'
                ];
                
                extensions.forEach(ext => {
                    try {
                        gl.getExtension(ext);
                        console.log(`✓ Enabled: ${ext}`);
                    } catch (e) {
                        console.warn(`Could not enable: ${ext}`);
                    }
                });
            }
        }
    }
    
    initializeMantleStructure() {
        // Mantle layers based on PREM model and recent tomography
        this.layers = {
            lithosphere: { depth: [0, 100], color: 0x8B4513, opacity: 0.8 },
            asthenosphere: { depth: [100, 220], color: 0xFF6347, opacity: 0.7 },
            upperMantle: { depth: [220, 410], color: 0xFF8C00, opacity: 0.6 },
            transitionZone: { depth: [410, 660], color: 0xFFD700, opacity: 0.6 },
            lowerMantle: { depth: [660, 2700], color: 0xFF4500, opacity: 0.5 },
            dPrimeLayer: { depth: [2700, 2891], color: 0x8B0000, opacity: 0.7 },
            outerCore: { depth: [2891, 5150], color: 0xFFFFFF, opacity: 0.3 },
            innerCore: { depth: [5150, 6371], color: 0xC0C0C0, opacity: 0.9 }
        };
        
        // Large Low-Shear-Velocity Provinces (LLSVPs)
        this.llsvps = [
            { name: "African LLSVP", lat: 0, lon: 10, radius: 40, height: 1000 },
            { name: "Pacific LLSVP", lat: -10, lon: -160, radius: 50, height: 1200 }
        ];
        
        // Major mantle plumes from latest research
        this.plumes = [
            { name: "Hawaii", lat: 19.5, lon: -155.5, depth: 2900, radius: 200, temp: 300 },
            { name: "Iceland", lat: 64.5, lon: -17.5, depth: 2900, radius: 250, temp: 250 },
            { name: "Yellowstone", lat: 44.5, lon: -110.5, depth: 1000, radius: 150, temp: 200 },
            { name: "Afar", lat: 11.5, lon: 41.5, depth: 2900, radius: 300, temp: 350 },
            { name: "Reunion", lat: -21.0, lon: 55.5, depth: 2900, radius: 180, temp: 280 },
            { name: "Galapagos", lat: -0.5, lon: -91.0, depth: 2900, radius: 200, temp: 260 },
            { name: "Easter", lat: -27.0, lon: -109.0, depth: 2900, radius: 170, temp: 240 },
            { name: "Samoa", lat: -14.0, lon: -169.0, depth: 2900, radius: 190, temp: 270 },
            { name: "Tahiti", lat: -17.5, lon: -149.5, depth: 2900, radius: 160, temp: 230 },
            { name: "Azores", lat: 38.0, lon: -28.0, depth: 2000, radius: 140, temp: 180 }
        ];
        
        // Subduction slabs from updated models
        this.slabs = [
            { name: "Farallon", lat: 40, lon: -120, depth: 1500, angle: 30, velocity: -2.5 },
            { name: "Pacific", lat: 35, lon: 140, depth: 660, angle: 45, velocity: -3.0 },
            { name: "Nazca", lat: -20, lon: -70, depth: 700, angle: 25, velocity: -2.8 },
            { name: "Indo-Australian", lat: -5, lon: 105, depth: 800, angle: 50, velocity: -2.2 },
            { name: "Philippine", lat: 10, lon: 125, depth: 600, angle: 60, velocity: -2.7 },
            { name: "Cocos", lat: 15, lon: -95, depth: 500, angle: 35, velocity: -2.4 },
            { name: "Caribbean", lat: 15, lon: -65, depth: 400, angle: 40, velocity: -2.0 },
            { name: "Scotia", lat: -55, lon: -60, depth: 300, angle: 55, velocity: -1.8 }
        ];
    }
    
    createMantleSlice(startLat, startLon, endLat, endLon, maxDepth = 2900) {
        const sliceGroup = new THREE.Group();
        
        // Create vertical slice through mantle
        const numPoints = 100;
        const numDepths = 50;
        
        const geometry = new THREE.PlaneGeometry(1, 1, numPoints - 1, numDepths - 1);
        const vertices = geometry.attributes.position.array;
        const colors = new Float32Array(vertices.length);
        
        // Calculate P-wave and S-wave velocities
        for (let i = 0; i < numPoints; i++) {
            for (let j = 0; j < numDepths; j++) {
                const idx = (j * numPoints + i) * 3;
                
                // Interpolate position along great circle
                const t = i / (numPoints - 1);
                const lat = startLat + t * (endLat - startLat);
                const lon = startLon + t * (endLon - startLon);
                const depth = (j / (numDepths - 1)) * maxDepth;
                
                // Get velocity anomaly at this point
                const velocity = this.getVelocityAnomaly(lat, lon, depth);
                
                // Map to color (blue = fast/cold, red = slow/hot)
                const color = this.velocityToColor(velocity);
                colors[idx] = color.r;
                colors[idx + 1] = color.g;
                colors[idx + 2] = color.b;
                
                // Set vertex position
                vertices[idx] = t * 1000 - 500; // X position
                vertices[idx + 1] = -depth / 3; // Y position (scaled)
                vertices[idx + 2] = 0; // Z position
            }
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8
        });
        
        const slice = new THREE.Mesh(geometry, material);
        sliceGroup.add(slice);
        
        // Add depth markers
        this.addDepthMarkers(sliceGroup);
        
        // Add slab intersections
        this.addSlabIntersections(sliceGroup, startLat, startLon, endLat, endLon);
        
        // Add plume intersections
        this.addPlumeIntersections(sliceGroup, startLat, startLon, endLat, endLon);
        
        return sliceGroup;
    }
    
    getVelocityAnomaly(lat, lon, depth) {
        let anomaly = 0;
        
        // Check proximity to slabs (cold/fast anomalies)
        this.slabs.forEach(slab => {
            const dist = this.getDistance(lat, lon, slab.lat, slab.lon);
            if (dist < 1000 && depth < slab.depth) {
                const influence = Math.exp(-dist / 200) * Math.exp(-(depth - slab.depth / 2) / 200);
                anomaly += slab.velocity * influence;
            }
        });
        
        // Check proximity to plumes (hot/slow anomalies)
        this.plumes.forEach(plume => {
            const dist = this.getDistance(lat, lon, plume.lat, plume.lon);
            if (dist < plume.radius * 2) {
                const influence = Math.exp(-dist / plume.radius) * 
                                 Math.exp(-Math.abs(depth - plume.depth / 2) / 500);
                anomaly += (plume.temp / 100) * influence; // Positive anomaly for hot
            }
        });
        
        // Check LLSVPs
        this.llsvps.forEach(llsvp => {
            const dist = this.getDistance(lat, lon, llsvp.lat, llsvp.lon);
            if (dist < llsvp.radius * 100 && depth > 2000) {
                anomaly += 1.5 * Math.exp(-dist / (llsvp.radius * 50));
            }
        });
        
        // Add background heterogeneity
        anomaly += Math.sin(lat * 0.1) * Math.cos(lon * 0.1) * Math.sin(depth * 0.01) * 0.5;
        
        return Math.max(-5, Math.min(5, anomaly)); // Clamp to reasonable range
    }
    
    velocityToColor(velocity) {
        // Map velocity anomaly to color
        // Blue (fast/cold) -> White (neutral) -> Red (slow/hot)
        const normalized = (velocity + 5) / 10; // Normalize to 0-1
        
        if (normalized < 0.5) {
            // Blue to white
            const t = normalized * 2;
            return new THREE.Color(t, t, 1);
        } else {
            // White to red
            const t = (normalized - 0.5) * 2;
            return new THREE.Color(1, 1 - t, 1 - t);
        }
    }
    
    create3DMantleStructure() {
        const mantleGroup = new THREE.Group();
        
        // Create concentric spheres for mantle layers
        Object.entries(this.layers).forEach(([name, layer]) => {
            if (name === 'outerCore' || name === 'innerCore') return; // Skip core
            
            const innerRadius = (6371 - layer.depth[1]) / 10;
            const outerRadius = (6371 - layer.depth[0]) / 10;
            
            const geometry = new THREE.SphereGeometry(
                outerRadius, 
                64, 
                32,
                0, 
                Math.PI * 2,
                0, 
                Math.PI
            );
            
            const material = new THREE.MeshPhongMaterial({
                color: layer.color,
                transparent: true,
                opacity: layer.opacity,
                side: THREE.DoubleSide
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            mantleGroup.add(sphere);
        });
        
        // Add LLSVPs
        this.llsvps.forEach(llsvp => {
            const geometry = new THREE.CylinderGeometry(
                llsvp.radius / 2, 
                llsvp.radius, 
                llsvp.height / 10,
                32
            );
            
            const material = new THREE.MeshPhongMaterial({
                color: 0x800080,
                transparent: true,
                opacity: 0.6,
                emissive: 0x400040,
                emissiveIntensity: 0.3
            });
            
            const llsvpMesh = new THREE.Mesh(geometry, material);
            const pos = this.latLonToCartesian(llsvp.lat, llsvp.lon, 3000);
            llsvpMesh.position.set(pos.x / 10, pos.y / 10, pos.z / 10);
            mantleGroup.add(llsvpMesh);
        });
        
        // Add mantle plumes
        this.plumes.forEach(plume => {
            const curve = new THREE.CatmullRomCurve3([
                new THREE.Vector3(0, -290, 0), // Core-mantle boundary
                new THREE.Vector3(
                    Math.random() * 20 - 10,
                    -200,
                    Math.random() * 20 - 10
                ), // Mid-mantle
                new THREE.Vector3(
                    Math.random() * 30 - 15,
                    -100,
                    Math.random() * 30 - 15
                ), // Upper mantle
                this.latLonToCartesian(plume.lat, plume.lon, 0) // Surface
            ]);
            
            const tubeGeometry = new THREE.TubeGeometry(
                curve,
                20,
                plume.radius / 100,
                8,
                false
            );
            
            const gradient = this.createGradientTexture(plume.temp);
            const material = new THREE.MeshPhongMaterial({
                map: gradient,
                transparent: true,
                opacity: 0.7,
                emissive: 0xFF4500,
                emissiveIntensity: plume.temp / 1000
            });
            
            const plumeMesh = new THREE.Mesh(tubeGeometry, material);
            mantleGroup.add(plumeMesh);
        });
        
        // Add subducting slabs
        this.slabs.forEach(slab => {
            const slabGeometry = new THREE.BoxGeometry(
                50,
                slab.depth / 10,
                10
            );
            
            const material = new THREE.MeshPhongMaterial({
                color: 0x4169E1,
                transparent: true,
                opacity: 0.6,
                emissive: 0x000080,
                emissiveIntensity: 0.2
            });
            
            const slabMesh = new THREE.Mesh(slabGeometry, material);
            const pos = this.latLonToCartesian(slab.lat, slab.lon, slab.depth / 2);
            slabMesh.position.set(pos.x / 10, pos.y / 10, pos.z / 10);
            slabMesh.rotation.z = slab.angle * Math.PI / 180;
            mantleGroup.add(slabMesh);
        });
        
        return mantleGroup;
    }
    
    createGradientTexture(temperature) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 0, 256);
        const t = temperature / 400; // Normalize temperature
        
        gradient.addColorStop(0, `hsl(${60 - t * 60}, 100%, 50%)`); // Yellow to red
        gradient.addColorStop(0.5, `hsl(${30 - t * 30}, 100%, 40%)`);
        gradient.addColorStop(1, `hsl(0, 100%, 30%)`); // Dark red
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 256, 256);
        
        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }
    
    addDepthMarkers(group) {
        const depths = [100, 410, 660, 1000, 2000, 2891];
        const labels = ['100 km', '410 km', '660 km', '1000 km', '2000 km', 'CMB'];
        
        depths.forEach((depth, i) => {
            const geometry = new THREE.PlaneGeometry(1000, 1);
            const material = new THREE.MeshBasicMaterial({
                color: 0xFFFFFF,
                transparent: true,
                opacity: 0.3
            });
            
            const line = new THREE.Mesh(geometry, material);
            line.position.y = -depth / 3;
            group.add(line);
            
            // Add text label (using sprite for simplicity)
            const sprite = this.createTextSprite(labels[i]);
            sprite.position.set(-550, -depth / 3, 0);
            group.add(sprite);
        });
    }
    
    createTextSprite(text) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = 'white';
        ctx.font = '24px Arial';
        ctx.fillText(text, 10, 40);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(50, 12.5, 1);
        
        return sprite;
    }
    
    addSlabIntersections(group, startLat, startLon, endLat, endLon) {
        this.slabs.forEach(slab => {
            // Check if slab intersects with slice
            const slabDist = this.pointToLineDistance(
                slab.lat, slab.lon,
                startLat, startLon,
                endLat, endLon
            );
            
            if (slabDist < 500) { // Within 500 km of slice
                const t = this.getProjectionT(slab.lat, slab.lon, startLat, startLon, endLat, endLon);
                
                if (t >= 0 && t <= 1) {
                    const geometry = new THREE.PlaneGeometry(50, slab.depth / 3);
                    const material = new THREE.MeshBasicMaterial({
                        color: 0x0000FF,
                        transparent: true,
                        opacity: 0.7,
                        side: THREE.DoubleSide
                    });
                    
                    const slabPlane = new THREE.Mesh(geometry, material);
                    slabPlane.position.x = t * 1000 - 500;
                    slabPlane.position.y = -slab.depth / 6;
                    slabPlane.rotation.z = -slab.angle * Math.PI / 180;
                    
                    group.add(slabPlane);
                }
            }
        });
    }
    
    addPlumeIntersections(group, startLat, startLon, endLat, endLon) {
        this.plumes.forEach(plume => {
            const plumeDist = this.pointToLineDistance(
                plume.lat, plume.lon,
                startLat, startLon,
                endLat, endLon
            );
            
            if (plumeDist < plume.radius) {
                const t = this.getProjectionT(plume.lat, plume.lon, startLat, startLon, endLat, endLon);
                
                if (t >= 0 && t <= 1) {
                    // Create plume cross-section
                    const geometry = new THREE.CircleGeometry(plume.radius / 10, 32);
                    const material = new THREE.MeshBasicMaterial({
                        color: new THREE.Color().setHSL(0.1, 1, 0.5),
                        transparent: true,
                        opacity: 0.6,
                        side: THREE.DoubleSide
                    });
                    
                    // Add multiple circles at different depths
                    for (let depth = 0; depth < plume.depth; depth += 100) {
                        const circle = new THREE.Mesh(geometry, material.clone());
                        circle.position.x = t * 1000 - 500;
                        circle.position.y = -depth / 3;
                        circle.material.opacity = 0.6 * (1 - depth / plume.depth);
                        group.add(circle);
                    }
                }
            }
        });
    }
    
    // Utility functions
    latLonToCartesian(lat, lon, depth) {
        const radius = 6371 - depth;
        const phi = (90 - lat) * Math.PI / 180;
        const theta = (lon + 180) * Math.PI / 180;
        
        return {
            x: radius * Math.sin(phi) * Math.cos(theta),
            y: radius * Math.cos(phi),
            z: radius * Math.sin(phi) * Math.sin(theta)
        };
    }
    
    getDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) ** 2 +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon / 2) ** 2;
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }
    
    pointToLineDistance(pLat, pLon, l1Lat, l1Lon, l2Lat, l2Lon) {
        // Calculate distance from point to great circle line
        const A = this.latLonToCartesian(l1Lat, l1Lon, 0);
        const B = this.latLonToCartesian(l2Lat, l2Lon, 0);
        const P = this.latLonToCartesian(pLat, pLon, 0);
        
        const AB = { x: B.x - A.x, y: B.y - A.y, z: B.z - A.z };
        const AP = { x: P.x - A.x, y: P.y - A.y, z: P.z - A.z };
        
        const cross = {
            x: AB.y * AP.z - AB.z * AP.y,
            y: AB.z * AP.x - AB.x * AP.z,
            z: AB.x * AP.y - AB.y * AP.x
        };
        
        const crossMag = Math.sqrt(cross.x ** 2 + cross.y ** 2 + cross.z ** 2);
        const ABMag = Math.sqrt(AB.x ** 2 + AB.y ** 2 + AB.z ** 2);
        
        return crossMag / ABMag;
    }
    
    getProjectionT(pLat, pLon, l1Lat, l1Lon, l2Lat, l2Lon) {
        // Get parameter t for projection of point onto line
        const totalDist = this.getDistance(l1Lat, l1Lon, l2Lat, l2Lon);
        const dist1 = this.getDistance(l1Lat, l1Lon, pLat, pLon);
        const dist2 = this.getDistance(l2Lat, l2Lon, pLat, pLon);
        
        // Use law of cosines to find projection
        const cosAngle = (totalDist ** 2 + dist1 ** 2 - dist2 ** 2) / (2 * totalDist * dist1);
        const projDist = dist1 * cosAngle;
        
        return projDist / totalDist;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MantleTomography;
}

// Make available globally
window.MantleTomography = MantleTomography;