/**
 * Plate Reconstruction Module
 */

window.PlateReconstruction = {
    currentTime: 0,
    plates: {},
    
    init: function() {
        console.log('🗺️ Plate Reconstruction module initialized');
        this.generatePlates();
        return this;
    },
    
    generatePlates: function() {
        const plateNames = ['Pacific', 'North American', 'Eurasian', 'African', 'Antarctic', 'Australian', 'South American'];
        plateNames.forEach(name => {
            this.plates[name] = {
                id: name.toLowerCase().replace(' ', '_'),
                name: name,
                vertices: this.generatePlateVertices(),
                velocity: { x: Math.random() * 10 - 5, y: Math.random() * 10 - 5 },
                color: `hsl(${Math.random() * 360}, 70%, 50%)`
            };
        });
    },
    
    generatePlateVertices: function() {
        const vertices = [];
        const numVertices = 6 + Math.floor(Math.random() * 4);
        for (let i = 0; i < numVertices; i++) {
            const angle = (i / numVertices) * Math.PI * 2;
            vertices.push({
                lat: Math.sin(angle) * 30 + (Math.random() - 0.5) * 20,
                lon: Math.cos(angle) * 60 + (Math.random() - 0.5) * 40
            });
        }
        return vertices;
    },
    
    reconstructAtTime: function(time) {
        this.currentTime = time;
        console.log(`Reconstructing plates at ${time} Ma`);
        
        // Simulate plate movement
        for (const plateName in this.plates) {
            const plate = this.plates[plateName];
            plate.vertices = plate.vertices.map(v => ({
                lat: v.lat + plate.velocity.y * time * 0.01,
                lon: v.lon + plate.velocity.x * time * 0.01
            }));
        }
        
        return this.plates;
    },
    
    getPlateAtLocation: function(lat, lon) {
        // Simple approximation
        for (const plateName in this.plates) {
            return this.plates[plateName];
        }
        return null;
    }
};

console.log('✅ Plate Reconstruction module loaded');