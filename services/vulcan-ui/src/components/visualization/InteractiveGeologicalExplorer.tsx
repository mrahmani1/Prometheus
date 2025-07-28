import React, { useRef, useState, useMemo, useCallback } from 'react';
import { useFrame, useThree, extend } from '@react-three/fiber';
import { 
  Html, 
  Sphere
} from '@react-three/drei';
import * as THREE from 'three';
import { 
  Box, 
  Typography, 
  FormControlLabel, 
  Switch, 
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  IconButton
} from '@mui/material';
import { 
  ExpandMore, 
  Visibility, 
  VisibilityOff
} from '@mui/icons-material';
import { GeologicalLocation, SeismicEvent, ModelPrediction } from '../../types/geological';

// Extend Three.js with custom materials
extend({ 
  ShaderMaterial: THREE.ShaderMaterial,
  RawShaderMaterial: THREE.RawShaderMaterial
});

// Custom geological layer shader
const geologicalLayerVertexShader = `
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  
  void main() {
    vPosition = position;
    vNormal = normal;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const geologicalLayerFragmentShader = `
  uniform float time;
  uniform float opacity;
  uniform vec3 color1;
  uniform vec3 color2;
  uniform float layerDepth;
  uniform float noiseScale;
  
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  
  // Simple noise function
  float noise(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
  }
  
  void main() {
    vec3 pos = vPosition * noiseScale;
    float n = noise(pos + time * 0.1);
    
    // Mix colors based on depth and noise
    vec3 finalColor = mix(color1, color2, n * 0.5 + 0.5);
    
    // Add depth-based transparency
    float depthFactor = 1.0 - (vPosition.y + layerDepth) / (layerDepth * 2.0);
    float finalOpacity = opacity * clamp(depthFactor, 0.1, 1.0);
    
    gl_FragColor = vec4(finalColor, finalOpacity);
  }
`;

// Geological layer component
interface GeologicalLayerProps {
  name: string;
  depth: number;
  thickness: number;
  color1: string;
  color2: string;
  opacity: number;
  visible: boolean;
  noiseScale: number;
}

const GeologicalLayer: React.FC<GeologicalLayerProps> = ({
  name,
  depth,
  thickness,
  color1,
  color2,
  opacity,
  visible,
  noiseScale
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  const uniforms = useMemo(() => ({
    time: { value: 0 },
    opacity: { value: opacity },
    color1: { value: new THREE.Color(color1) },
    color2: { value: new THREE.Color(color2) },
    layerDepth: { value: depth },
    noiseScale: { value: noiseScale }
  }), [opacity, color1, color2, depth, noiseScale]);
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
    }
  });
  
  if (!visible) return null;
  
  return (
    <mesh 
      ref={meshRef} 
      position={[0, -depth, 0]}
    >
      <boxGeometry args={[36, thickness, 18]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={geologicalLayerVertexShader}
        fragmentShader={geologicalLayerFragmentShader}
        uniforms={uniforms}
        transparent
        side={THREE.DoubleSide}
      />
      
      {/* Layer label */}
      <Html position={[18, thickness/2, 0]}>
        <div style={{
          background: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '12px',
          pointerEvents: 'none'
        }}>
          {name}
        </div>
      </Html>
    </mesh>
  );
};

// Cross-section visualization
interface CrossSectionProps {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  size: [number, number];
  data: any[];
  visible: boolean;
}

const CrossSection: React.FC<CrossSectionProps> = ({
  position,
  // normal,
  size,
  data,
  visible
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Create cross-section texture from data
  const crossSectionTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return null;
    
    // Create gradient visualization of geological data
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const pixels = imageData.data;
    
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const index = (y * canvas.width + x) * 4;
        
        // Sample data based on position
        const depth = (y / canvas.height) * 10; // 10km depth
        // const lateral = (x / canvas.width) * 36 - 18; // Scene width
        
        // Simple geological stratification visualization
        const layerIndex = Math.floor(depth / 2);
        const colors = [
          [139, 69, 19],   // Brown - sedimentary
          [105, 105, 105], // Gray - metamorphic
          [255, 69, 0],    // Red - igneous
          [255, 140, 0],   // Orange - mantle
          [255, 0, 0]      // Red - core
        ];
        
        const color = colors[Math.min(layerIndex, colors.length - 1)];
        
        pixels[index] = color[0];     // Red
        pixels[index + 1] = color[1]; // Green
        pixels[index + 2] = color[2]; // Blue
        pixels[index + 3] = 200;      // Alpha
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  }, [data]);
  
  if (!visible || !crossSectionTexture) return null;
  
  return (
    <mesh 
      ref={meshRef}
      position={position}
      rotation={[0, 0, 0]}
    >
      <planeGeometry args={size} />
      <meshBasicMaterial 
        map={crossSectionTexture}
        transparent
        opacity={0.8}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// 3D geological model with interactive features
interface InteractiveGeologicalModelProps {
  layers: Array<{
    name: string;
    depth: number;
    thickness: number;
    color1: string;
    color2: string;
    properties: Record<string, any>;
  }>;
  seismicEvents: SeismicEvent[];
  predictions: ModelPrediction[];
  onLayerSelect: (layerName: string) => void;
  onLocationSelect: (location: GeologicalLocation) => void;
}

const InteractiveGeologicalModel: React.FC<InteractiveGeologicalModelProps> = ({
  layers,
  seismicEvents,
  // predictions,
  // onLayerSelect,
  onLocationSelect
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const raycaster = useRef(new THREE.Raycaster());
  const mouse = useRef(new THREE.Vector2());
  const { camera, gl } = useThree();
  
  const [selectedLayer] = useState<string | null>(null);
  const [crossSectionVisible] = useState(false);
  const [crossSectionPosition, setCrossSectionPosition] = useState(new THREE.Vector3(0, 0, 0));
  const [hoveredObject, setHoveredObject] = useState<any>(null);
  
  // Handle mouse interactions
  const handlePointerMove = useCallback((event: any) => {
    const rect = gl.domElement.getBoundingClientRect();
    mouse.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.current.setFromCamera(mouse.current, camera);
    
    if (groupRef.current) {
      const intersects = raycaster.current.intersectObjects(groupRef.current.children, true);
      
      if (intersects.length > 0) {
        const intersected = intersects[0];
        setHoveredObject({
          object: intersected.object,
          point: intersected.point,
          distance: intersected.distance
        });
        
        // Change cursor
        gl.domElement.style.cursor = 'pointer';
      } else {
        setHoveredObject(null);
        gl.domElement.style.cursor = 'default';
      }
    }
  }, [camera, gl]);
  
  const handleClick = useCallback((_event: any) => {
    if (hoveredObject) {
      const point = hoveredObject.point;
      
      // Convert 3D point to geological coordinates
      const longitude = (point.x + 18) / 0.1 - 180;
      const latitude = (point.z + 9) / 0.1 - 90;
      const elevation = point.y / 0.001;
      
      onLocationSelect({
        latitude,
        longitude,
        elevation,
        depth: Math.abs(point.y)
      });
      
      // Update cross-section position
      setCrossSectionPosition(point);
    }
  }, [hoveredObject, onLocationSelect]);
  
  // Seismic event clustering for better visualization
  const clusteredEvents = useMemo(() => {
    // Simple clustering algorithm
    const clusters: Array<{
      center: THREE.Vector3;
      events: SeismicEvent[];
      averageMagnitude: number;
    }> = [];
    
    const clusterRadius = 2.0; // Scene units
    
    seismicEvents.forEach(event => {
      const eventPos = new THREE.Vector3(
        (event.location.longitude + 180) * 0.1 - 18,
        (event.location.elevation || 0) * 0.001,
        (event.location.latitude + 90) * 0.1 - 9
      );
      
      // Find existing cluster
      let foundCluster = false;
      for (const cluster of clusters) {
        if (cluster.center.distanceTo(eventPos) < clusterRadius) {
          cluster.events.push(event);
          cluster.averageMagnitude = cluster.events.reduce((sum, e) => sum + e.magnitude, 0) / cluster.events.length;
          foundCluster = true;
          break;
        }
      }
      
      // Create new cluster if none found
      if (!foundCluster) {
        clusters.push({
          center: eventPos,
          events: [event],
          averageMagnitude: event.magnitude
        });
      }
    });
    
    return clusters;
  }, [seismicEvents]);
  
  return (
    <group 
      ref={groupRef}
      onPointerMove={handlePointerMove}
      onClick={handleClick}
    >
      {/* Geological layers */}
      {layers.map((layer) => (
        <GeologicalLayer
          key={layer.name}
          name={layer.name}
          depth={layer.depth}
          thickness={layer.thickness}
          color1={layer.color1}
          color2={layer.color2}
          opacity={selectedLayer === layer.name ? 0.9 : 0.6}
          visible={true}
          noiseScale={1.0}
        />
      ))}
      
      {/* Seismic event clusters */}
      {clusteredEvents.map((cluster, index) => (
        <group key={`cluster-${index}`}>
          <Sphere
            position={cluster.center}
            args={[Math.max(0.1, cluster.averageMagnitude * 0.1), 16, 16]}
          >
            <meshStandardMaterial 
              color={cluster.events.length > 5 ? '#ff4444' : '#4444ff'}
              transparent
              opacity={0.7}
              emissive={cluster.events.length > 5 ? '#ff2222' : '#2222ff'}
              emissiveIntensity={0.2}
            />
          </Sphere>
          
          {/* Cluster info */}
          <Html position={cluster.center.clone().add(new THREE.Vector3(0, 0.5, 0))}>
            <div style={{
              background: 'rgba(0,0,0,0.8)',
              color: 'white',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '10px',
              textAlign: 'center',
              pointerEvents: 'none'
            }}>
              <div>{cluster.events.length} events</div>
              <div>Avg M{cluster.averageMagnitude.toFixed(1)}</div>
            </div>
          </Html>
        </group>
      ))}
      
      {/* Cross-section plane */}
      {crossSectionVisible && (
        <CrossSection
          position={crossSectionPosition}
          normal={new THREE.Vector3(1, 0, 0)}
          size={[18, 10]}
          data={[]}
          visible={crossSectionVisible}
        />
      )}
      
      {/* Hover indicator */}
      {hoveredObject && (
        <Sphere
          position={hoveredObject.point}
          args={[0.1, 8, 8]}
        >
          <meshBasicMaterial color="#ffff00" />
        </Sphere>
      )}
    </group>
  );
};

// Main interactive geological explorer component
interface InteractiveGeologicalExplorerProps {
  data: {
    layers?: any[];
    seismicEvents?: SeismicEvent[];
    predictions?: ModelPrediction[];
  };
  onDataRequest?: (params: any) => void;
}

export const InteractiveGeologicalExplorer: React.FC<InteractiveGeologicalExplorerProps> = ({
  data,
  onDataRequest
}) => {
  const [layerVisibility, setLayerVisibility] = useState<Record<string, boolean>>({});
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [crossSectionMode, setCrossSectionMode] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<GeologicalLocation | null>(null);
  const [viewMode, setViewMode] = useState<'surface' | 'subsurface' | 'cross-section'>('subsurface');
  
  // Default geological layers
  const defaultLayers = [
    {
      name: 'Sedimentary Cover',
      depth: 0,
      thickness: 2,
      color1: '#8B4513',
      color2: '#D2B48C',
      properties: { density: 2.3, porosity: 0.15 }
    },
    {
      name: 'Metamorphic Basement',
      depth: 2,
      thickness: 3,
      color1: '#696969',
      color2: '#A9A9A9',
      properties: { density: 2.7, porosity: 0.05 }
    },
    {
      name: 'Igneous Intrusion',
      depth: 5,
      thickness: 2,
      color1: '#FF4500',
      color2: '#FF6347',
      properties: { density: 2.9, porosity: 0.02 }
    },
    {
      name: 'Upper Mantle',
      depth: 7,
      thickness: 8,
      color1: '#FF8C00',
      color2: '#FFA500',
      properties: { density: 3.3, porosity: 0.01 }
    }
  ];
  
  const layers = data.layers || defaultLayers;
  
  const handleLayerToggle = (layerName: string) => {
    setLayerVisibility(prev => ({
      ...prev,
      [layerName]: !prev[layerName]
    }));
  };
  
  const handleLayerSelect = (layerName: string) => {
    setSelectedLayer(layerName);
    if (onDataRequest) {
      onDataRequest({ layer: layerName, type: 'layer_details' });
    }
  };
  
  const handleLocationSelect = (location: GeologicalLocation) => {
    setSelectedLocation(location);
    if (onDataRequest) {
      onDataRequest({ location, type: 'location_details' });
    }
  };
  
  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Scene */}
      <Box sx={{ width: '100%', height: '100%' }}>
        <InteractiveGeologicalModel
          layers={layers}
          seismicEvents={data.seismicEvents || []}
          predictions={data.predictions || []}
          onLayerSelect={handleLayerSelect}
          onLocationSelect={handleLocationSelect}
        />
      </Box>
      
      {/* Control Panel */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          width: 300,
          maxHeight: '80%',
          overflow: 'auto',
          background: 'rgba(0,0,0,0.8)',
          borderRadius: 2,
          p: 2,
          color: 'white'
        }}
      >
        <Typography variant="h6" gutterBottom>
          Geological Explorer
        </Typography>
        
        {/* View Mode Controls */}
        <Accordion sx={{ mb: 2, bgcolor: 'rgba(255,255,255,0.1)' }}>
          <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
            <Typography sx={{ color: 'white' }}>View Mode</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControlLabel
              control={
                <Switch
                  checked={viewMode === 'subsurface'}
                  onChange={(e) => setViewMode(e.target.checked ? 'subsurface' : 'surface')}
                />
              }
              label="Subsurface View"
              sx={{ color: 'white' }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={crossSectionMode}
                  onChange={(e) => setCrossSectionMode(e.target.checked)}
                />
              }
              label="Cross-Section Mode"
              sx={{ color: 'white' }}
            />
          </AccordionDetails>
        </Accordion>
        
        {/* Layer Controls */}
        <Accordion sx={{ mb: 2, bgcolor: 'rgba(255,255,255,0.1)' }}>
          <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
            <Typography sx={{ color: 'white' }}>Geological Layers</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {layers.map((layer) => (
              <Box key={layer.name} sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <IconButton
                    size="small"
                    onClick={() => handleLayerToggle(layer.name)}
                    sx={{ color: 'white' }}
                  >
                    {layerVisibility[layer.name] !== false ? <Visibility /> : <VisibilityOff />}
                  </IconButton>
                  
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      flex: 1, 
                      cursor: 'pointer',
                      color: selectedLayer === layer.name ? '#ffff00' : 'white'
                    }}
                    onClick={() => handleLayerSelect(layer.name)}
                  >
                    {layer.name}
                  </Typography>
                  
                  <Chip
                    label={`${layer.depth}km`}
                    size="small"
                    sx={{ bgcolor: layer.color1, color: 'white' }}
                  />
                </Box>
                
                {selectedLayer === layer.name && (
                  <Box sx={{ mt: 1, pl: 2, fontSize: '12px' }}>
                    <div>Depth: {layer.depth} - {layer.depth + layer.thickness} km</div>
                    <div>Density: {layer.properties.density} g/cm³</div>
                    <div>Porosity: {(layer.properties.porosity * 100).toFixed(1)}%</div>
                  </Box>
                )}
              </Box>
            ))}
          </AccordionDetails>
        </Accordion>
        
        {/* Selected Location Info */}
        {selectedLocation && (
          <Accordion sx={{ mb: 2, bgcolor: 'rgba(255,255,255,0.1)' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
              <Typography sx={{ color: 'white' }}>Selected Location</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" sx={{ color: 'white' }}>
                Lat: {selectedLocation.latitude.toFixed(4)}°<br/>
                Lon: {selectedLocation.longitude.toFixed(4)}°<br/>
                Elevation: {selectedLocation.elevation.toFixed(0)}m<br/>
                {selectedLocation.depth && `Depth: ${selectedLocation.depth.toFixed(1)}km`}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}
        
        {/* Seismic Activity Summary */}
        {data.seismicEvents && data.seismicEvents.length > 0 && (
          <Accordion sx={{ bgcolor: 'rgba(255,255,255,0.1)' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: 'white' }} />}>
              <Typography sx={{ color: 'white' }}>Seismic Activity</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" sx={{ color: 'white' }}>
                Total Events: {data.seismicEvents.length}<br/>
                Max Magnitude: {Math.max(...data.seismicEvents.map(e => e.magnitude)).toFixed(1)}<br/>
                Volcanic Events: {data.seismicEvents.filter(e => e.eventType === 'volcanic').length}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}
      </Box>
    </Box>
  );
};