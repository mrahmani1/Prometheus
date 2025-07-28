import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Stats, 
  Grid, 
  Environment,
  Html
} from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, Slider, IconButton } from '@mui/material';
import { 
  PlayArrow, 
  Pause, 
  Stop
} from '@mui/icons-material';
import { GeologicalLocation, SeismicEvent, DeformationMeasurement } from '../../types/geological';

interface ThreeVisualizationProps {
  data: {
    seismicEvents?: SeismicEvent[];
    deformationData?: DeformationMeasurement[];
    thermalData?: any[];
    topographyData?: any[];
  };
  timeRange: [string, string];
  currentTime: string;
  onTimeChange: (time: string) => void;
  isPlaying: boolean;
  onPlayStateChange: (playing: boolean) => void;
  viewMode: '3D' | '4D';
  showPerformanceStats?: boolean;
}

// Convert geological coordinates to 3D scene coordinates
const geoToScene = (location: GeologicalLocation): [number, number, number] => {
  // Simple conversion - in production, this would use proper map projections
  const x = (location.longitude + 180) * 0.1 - 18; // Scale longitude to scene
  const z = (location.latitude + 90) * 0.1 - 9; // Scale latitude to scene
  const y = (location.elevation || 0) * 0.001; // Scale elevation
  return [x, y, z];
};

// Seismic event visualization component
const SeismicEventMarkers: React.FC<{ 
  events: SeismicEvent[]; 
  currentTime: string;
  timeWindow: number;
}> = ({ events, currentTime, timeWindow }) => {
  const currentTimestamp = new Date(currentTime).getTime();
  
  const visibleEvents = useMemo(() => {
    return events.filter(event => {
      const eventTime = new Date(event.timestamp).getTime();
      return Math.abs(currentTimestamp - eventTime) <= timeWindow;
    });
  }, [events, currentTimestamp, timeWindow]);

  return (
    <group>
      {visibleEvents.map((event, index) => {
        const [x, y, z] = geoToScene(event.location);
        const size = Math.max(0.1, event.magnitude * 0.05);
        const age = (currentTimestamp - new Date(event.timestamp).getTime()) / timeWindow;
        const opacity = Math.max(0.1, 1 - age);
        
        const color = event.eventType === 'volcanic' ? '#ff4444' : 
                     event.eventType === 'earthquake' ? '#4444ff' : '#44ff44';

        return (
          <mesh key={`${event.id}-${index}`} position={[x, y, z]}>
            <sphereGeometry args={[size, 16, 16]} />
            <meshStandardMaterial 
              color={color} 
              transparent 
              opacity={opacity}
              emissive={color}
              emissiveIntensity={0.2}
            />
            <Html distanceFactor={10}>
              <div style={{ 
                background: 'rgba(0,0,0,0.8)', 
                color: 'white', 
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                pointerEvents: 'none'
              }}>
                M{event.magnitude.toFixed(1)}
              </div>
            </Html>
          </mesh>
        );
      })}
    </group>
  );
};

// Deformation visualization component
const DeformationVectors: React.FC<{ 
  deformationData: DeformationMeasurement[];
  currentTime: string;
  scale: number;
}> = ({ deformationData, currentTime, scale }) => {
  const currentTimestamp = new Date(currentTime).getTime();
  
  const visibleDeformation = useMemo(() => {
    // Get the most recent measurement for each location
    const locationMap = new Map<string, DeformationMeasurement>();
    
    deformationData.forEach(measurement => {
      const measurementTime = new Date(measurement.timestamp).getTime();
      if (measurementTime <= currentTimestamp) {
        const key = `${measurement.location.latitude}-${measurement.location.longitude}`;
        const existing = locationMap.get(key);
        if (!existing || new Date(existing.timestamp).getTime() < measurementTime) {
          locationMap.set(key, measurement);
        }
      }
    });
    
    return Array.from(locationMap.values());
  }, [deformationData, currentTimestamp]);

  return (
    <group>
      {visibleDeformation.map((measurement, index) => {
        const [x, y, z] = geoToScene(measurement.location);
        const [dx, dy, dz] = measurement.displacementVector;
        
        // Scale displacement vectors for visibility
        const scaledDisplacement = [dx * scale, dy * scale, dz * scale];
        const magnitude = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        if (magnitude < 0.001) return null; // Skip very small displacements
        
        return (
          <group key={`deformation-${measurement.id}-${index}`}>
            {/* Base point */}
            <mesh position={[x, y, z]}>
              <sphereGeometry args={[0.02, 8, 8]} />
              <meshStandardMaterial color="#ffaa00" />
            </mesh>
            
            {/* Displacement vector */}
            <arrowHelper
              args={[
                new THREE.Vector3(...scaledDisplacement).normalize(),
                new THREE.Vector3(x, y, z),
                Math.min(magnitude * scale, 2), // Clamp arrow length
                '#ffaa00',
                0.1,
                0.05
              ]}
            />
          </group>
        );
      })}
    </group>
  );
};

// Terrain/topography component
const TerrainMesh: React.FC<{ 
  topographyData?: any[];
  opacity?: number;
}> = ({ topographyData, opacity = 0.3 }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Generate a simple terrain mesh - in production, this would use real DEM data
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(36, 18, 100, 50);
    const vertices = geo.attributes.position.array as Float32Array;
    
    // Add some noise for terrain variation
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const z = vertices[i + 2];
      vertices[i + 1] = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 2 + 
                        Math.random() * 0.5;
    }
    
    geo.computeVertexNormals();
    return geo;
  }, [topographyData]);

  return (
    <mesh 
      ref={meshRef} 
      geometry={geometry} 
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, -2, 0]}
    >
      <meshStandardMaterial 
        color="#8B4513" 
        transparent 
        opacity={opacity}
        wireframe={false}
      />
    </mesh>
  );
};

// Performance monitoring component
const PerformanceMonitor: React.FC = () => {
  const { gl } = useThree();
  const [stats, setStats] = useState({ fps: 0, triangles: 0, calls: 0 });
  
  useFrame(() => {
    if (gl.info) {
      setStats({
        fps: Math.round(1000 / (performance.now() % 1000)),
        triangles: gl.info.render.triangles,
        calls: gl.info.render.calls
      });
    }
  });

  return (
    <Html position={[-17, 8, 0]}>
      <div style={{
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontFamily: 'monospace'
      }}>
        <div>FPS: {stats.fps}</div>
        <div>Triangles: {stats.triangles.toLocaleString()}</div>
        <div>Draw Calls: {stats.calls}</div>
      </div>
    </Html>
  );
};

// Main 3D visualization component
export const ThreeVisualization: React.FC<ThreeVisualizationProps> = ({
  data,
  timeRange,
  currentTime,
  onTimeChange,
  isPlaying,
  onPlayStateChange,
  // viewMode,
  showPerformanceStats = false
}) => {
  const [deformationScale, setDeformationScale] = useState(1000);
  const [timeWindow, setTimeWindow] = useState(86400000); // 24 hours in ms
  const [showTerrain] = useState(true);
  
  // Time animation
  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      const current = new Date(currentTime);
      const end = new Date(timeRange[1]);
      
      if (current >= end) {
        onPlayStateChange(false);
        return;
      }
      
      const next = new Date(current.getTime() + 3600000); // 1 hour step
      onTimeChange(next.toISOString());
    }, 100); // Update every 100ms for smooth animation
    
    return () => clearInterval(interval);
  }, [isPlaying, currentTime, timeRange, onTimeChange, onPlayStateChange]);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [10, 10, 10], fov: 60 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: "high-performance"
        }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight 
          position={[10, 10, 5]} 
          intensity={1}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <pointLight position={[-10, -10, -5]} intensity={0.5} />
        
        {/* Environment */}
        <Environment preset="sunset" />
        
        {/* Grid and reference */}
        <Grid 
          args={[36, 18]} 
          position={[0, -2, 0]}
          cellSize={1}
          cellThickness={0.5}
          cellColor="#666666"
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#888888"
        />
        
        {/* Terrain */}
        {showTerrain && <TerrainMesh topographyData={data.topographyData} />}
        
        {/* Geological data visualization */}
        {data.seismicEvents && (
          <SeismicEventMarkers 
            events={data.seismicEvents}
            currentTime={currentTime}
            timeWindow={timeWindow}
          />
        )}
        
        {data.deformationData && (
          <DeformationVectors
            deformationData={data.deformationData}
            currentTime={currentTime}
            scale={deformationScale}
          />
        )}
        
        {/* Performance monitoring */}
        {showPerformanceStats && <PerformanceMonitor />}
        
        {/* Controls */}
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI}
          minDistance={1}
          maxDistance={100}
        />
        
        {showPerformanceStats && <Stats />}
      </Canvas>
      
      {/* Time controls overlay */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          right: 16,
          background: 'rgba(0,0,0,0.8)',
          borderRadius: 2,
          p: 2,
          color: 'white'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <IconButton 
            onClick={() => onPlayStateChange(!isPlaying)}
            sx={{ color: 'white' }}
          >
            {isPlaying ? <Pause /> : <PlayArrow />}
          </IconButton>
          
          <IconButton 
            onClick={() => onPlayStateChange(false)}
            sx={{ color: 'white' }}
          >
            <Stop />
          </IconButton>
          
          <Typography variant="body2" sx={{ minWidth: 200 }}>
            {new Date(currentTime).toLocaleString()}
          </Typography>
          
          <Box sx={{ flex: 1 }}>
            <Slider
              value={new Date(currentTime).getTime()}
              min={new Date(timeRange[0]).getTime()}
              max={new Date(timeRange[1]).getTime()}
              onChange={(_: any, value: any) => {
                onTimeChange(new Date(value as number).toISOString());
              }}
              sx={{ color: 'white' }}
            />
          </Box>
        </Box>
        
        {/* Visualization controls */}
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Typography variant="caption">Deformation Scale:</Typography>
          <Slider
            value={deformationScale}
            min={100}
            max={10000}
            step={100}
            onChange={(_: any, value: any) => setDeformationScale(value as number)}
            sx={{ width: 150, color: 'white' }}
          />
          
          <Typography variant="caption">Time Window (hrs):</Typography>
          <Slider
            value={timeWindow / 3600000}
            min={1}
            max={168} // 1 week
            step={1}
            onChange={(_: any, value: any) => setTimeWindow((value as number) * 3600000)}
            sx={{ width: 150, color: 'white' }}
          />
        </Box>
      </Box>
    </Box>
  );
};