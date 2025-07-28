import React, { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { 
  SeismicEvent, 
  DeformationMeasurement, 
  ThermalMeasurement,
  ModelPrediction 
} from '../../types/geological';
// import { TemporalDataCache } from './PerformanceOptimizer';

// Time-based interpolation utilities
export class TemporalInterpolator {
  static interpolateScalar(
    value1: number, 
    value2: number, 
    time1: number, 
    time2: number, 
    currentTime: number
  ): number {
    if (time1 === time2) return value1;
    const t = (currentTime - time1) / (time2 - time1);
    return value1 + (value2 - value1) * Math.max(0, Math.min(1, t));
  }
  
  static interpolateVector3(
    v1: THREE.Vector3, 
    v2: THREE.Vector3, 
    time1: number, 
    time2: number, 
    currentTime: number
  ): THREE.Vector3 {
    if (time1 === time2) return v1.clone();
    const t = (currentTime - time1) / (time2 - time1);
    const clampedT = Math.max(0, Math.min(1, t));
    return v1.clone().lerp(v2, clampedT);
  }
  
  static interpolateColor(
    color1: THREE.Color, 
    color2: THREE.Color, 
    time1: number, 
    time2: number, 
    currentTime: number
  ): THREE.Color {
    if (time1 === time2) return color1.clone();
    const t = (currentTime - time1) / (time2 - time1);
    const clampedT = Math.max(0, Math.min(1, t));
    return color1.clone().lerp(color2, clampedT);
  }
}

// Animated seismic wave propagation
interface SeismicWavePropagationProps {
  events: SeismicEvent[];
  currentTime: string;
  waveSpeed: number; // km/s
  maxWaveRadius: number; // km
}

export const SeismicWavePropagation: React.FC<SeismicWavePropagationProps> = ({
  events,
  currentTime,
  waveSpeed = 6.0, // P-wave speed in km/s
  maxWaveRadius = 100
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const currentTimestamp = new Date(currentTime).getTime();
  
  const activeWaves = useMemo(() => {
    return events
      .map(event => {
        const eventTime = new Date(event.timestamp).getTime();
        const timeDiff = (currentTimestamp - eventTime) / 1000; // seconds
        
        if (timeDiff < 0 || timeDiff > maxWaveRadius / waveSpeed) {
          return null; // Wave hasn't started or has dissipated
        }
        
        const radius = (timeDiff * waveSpeed) / 100; // Scale for visualization
        const opacity = Math.max(0, 1 - (timeDiff / (maxWaveRadius / waveSpeed)));
        
        return {
          event,
          radius,
          opacity,
          timeDiff
        };
      })
      .filter(Boolean);
  }, [events, currentTimestamp, waveSpeed, maxWaveRadius]);
  
  return (
    <group ref={groupRef}>
      {activeWaves.map((wave, index) => {
        if (!wave) return null;
        
        const [x, y, z] = [
          (wave.event.location.longitude + 180) * 0.1 - 18,
          (wave.event.location.elevation || 0) * 0.001,
          (wave.event.location.latitude + 90) * 0.1 - 9
        ];
        
        return (
          <mesh 
            key={`wave-${wave.event.id}-${index}`}
            position={[x, y, z]}
          >
            <ringGeometry args={[wave.radius * 0.9, wave.radius, 32]} />
            <meshBasicMaterial 
              color={wave.event.eventType === 'volcanic' ? '#ff4444' : '#4444ff'}
              transparent
              opacity={wave.opacity * 0.5}
              side={THREE.DoubleSide}
            />
          </mesh>
        );
      })}
    </group>
  );
};

// Animated deformation field visualization
interface DeformationFieldProps {
  deformationData: DeformationMeasurement[];
  currentTime: string;
  timeWindow: number;
  gridResolution: number;
}

export const DeformationField: React.FC<DeformationFieldProps> = ({
  deformationData,
  currentTime,
  timeWindow,
  gridResolution = 20
}) => {
  // const meshRef = useRef<THREE.Mesh>(null);
  const currentTimestamp = new Date(currentTime).getTime();
  
  const deformationGrid = useMemo(() => {
    // Create a grid of interpolated deformation values
    const grid: Array<{
      position: THREE.Vector3;
      displacement: THREE.Vector3;
      magnitude: number;
    }> = [];
    
    // Get recent deformation measurements
    const recentData = deformationData.filter(measurement => {
      const measurementTime = new Date(measurement.timestamp).getTime();
      return Math.abs(currentTimestamp - measurementTime) <= timeWindow;
    });
    
    if (recentData.length === 0) return grid;
    
    // Create grid points
    for (let i = 0; i < gridResolution; i++) {
      for (let j = 0; j < gridResolution; j++) {
        const x = (i / (gridResolution - 1)) * 36 - 18; // Scene coordinates
        const z = (j / (gridResolution - 1)) * 18 - 9;
        const y = 0;
        
        const gridPoint = new THREE.Vector3(x, y, z);
        
        // Interpolate deformation at this grid point using inverse distance weighting
        let totalWeight = 0;
        const weightedDisplacement = new THREE.Vector3(0, 0, 0);
        
        recentData.forEach(measurement => {
          const measurementPos = new THREE.Vector3(
            (measurement.location.longitude + 180) * 0.1 - 18,
            (measurement.location.elevation || 0) * 0.001,
            (measurement.location.latitude + 90) * 0.1 - 9
          );
          
          const distance = gridPoint.distanceTo(measurementPos);
          const weight = distance > 0 ? 1 / (distance * distance) : 1000;
          
          const displacement = new THREE.Vector3(
            measurement.displacementVector[0] * 1000, // Scale for visibility
            measurement.displacementVector[1] * 1000,
            measurement.displacementVector[2] * 1000
          );
          
          weightedDisplacement.add(displacement.multiplyScalar(weight));
          totalWeight += weight;
        });
        
        if (totalWeight > 0) {
          weightedDisplacement.divideScalar(totalWeight);
          const magnitude = weightedDisplacement.length();
          
          if (magnitude > 0.01) { // Only include significant deformation
            grid.push({
              position: gridPoint,
              displacement: weightedDisplacement,
              magnitude
            });
          }
        }
      }
    }
    
    return grid;
  }, [deformationData, currentTimestamp, timeWindow, gridResolution]);
  
  return (
    <group>
      {deformationGrid.map((gridPoint, index) => (
        <group key={`deformation-grid-${index}`}>
          {/* Deformation vector */}
          <arrowHelper
            args={[
              gridPoint.displacement.clone().normalize(),
              gridPoint.position,
              Math.min(gridPoint.magnitude * 0.1, 2),
              `hsl(${Math.min(gridPoint.magnitude * 10, 360)}, 70%, 50%)`,
              0.1,
              0.05
            ]}
          />
          
          {/* Base point */}
          <mesh position={gridPoint.position}>
            <sphereGeometry args={[0.02, 8, 8]} />
            <meshStandardMaterial 
              color={`hsl(${Math.min(gridPoint.magnitude * 10, 360)}, 70%, 50%)`}
            />
          </mesh>
        </group>
      ))}
    </group>
  );
};

// Thermal evolution visualization
interface ThermalEvolutionProps {
  thermalData: ThermalMeasurement[];
  currentTime: string;
  timeWindow: number;
}

export const ThermalEvolution: React.FC<ThermalEvolutionProps> = ({
  thermalData,
  currentTime,
  timeWindow
}) => {
  const [heatmapTexture, setHeatmapTexture] = useState<THREE.Texture | null>(null);
  const currentTimestamp = new Date(currentTime).getTime();
  
  const thermalField = useMemo(() => {
    const recentData = thermalData.filter(measurement => {
      const measurementTime = new Date(measurement.timestamp).getTime();
      return Math.abs(currentTimestamp - measurementTime) <= timeWindow;
    });
    
    if (recentData.length === 0) return null;
    
    // Create thermal field visualization
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return null;
    
    // Create gradient for temperature visualization
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;
    
    // Find temperature range
    const temperatures = recentData.map(d => d.temperature);
    const minTemp = Math.min(...temperatures);
    const maxTemp = Math.max(...temperatures);
    const tempRange = maxTemp - minTemp;
    
    // Fill with interpolated temperature data
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const worldX = (x / canvas.width) * 36 - 18;
        const worldZ = (y / canvas.height) * 18 - 9;
        
        // Interpolate temperature at this pixel
        let totalWeight = 0;
        let weightedTemp = 0;
        
        recentData.forEach(measurement => {
          const measurementX = (measurement.location.longitude + 180) * 0.1 - 18;
          const measurementZ = (measurement.location.latitude + 90) * 0.1 - 9;
          
          const distance = Math.sqrt(
            (worldX - measurementX) ** 2 + (worldZ - measurementZ) ** 2
          );
          
          const weight = distance > 0 ? 1 / (distance + 0.1) : 1000;
          weightedTemp += measurement.temperature * weight;
          totalWeight += weight;
        });
        
        const temperature = totalWeight > 0 ? weightedTemp / totalWeight : minTemp;
        const normalizedTemp = tempRange > 0 ? (temperature - minTemp) / tempRange : 0;
        
        // Convert to color (blue = cold, red = hot)
        const pixelIndex = (y * canvas.width + x) * 4;
        data[pixelIndex] = Math.floor(normalizedTemp * 255); // Red
        data[pixelIndex + 1] = Math.floor((1 - normalizedTemp) * 128); // Green
        data[pixelIndex + 2] = Math.floor((1 - normalizedTemp) * 255); // Blue
        data[pixelIndex + 3] = Math.floor(normalizedTemp * 128 + 64); // Alpha
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    
    return texture;
  }, [thermalData, currentTimestamp, timeWindow]);
  
  useEffect(() => {
    setHeatmapTexture(thermalField);
    
    return () => {
      if (thermalField) {
        thermalField.dispose();
      }
    };
  }, [thermalField]);
  
  if (!heatmapTexture) return null;
  
  return (
    <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[36, 18]} />
      <meshBasicMaterial 
        map={heatmapTexture}
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
};

// Magma chamber evolution visualization
interface MagmaChamberEvolutionProps {
  predictions: ModelPrediction[];
  currentTime: string;
  timeWindow: number;
}

export const MagmaChamberEvolution: React.FC<MagmaChamberEvolutionProps> = ({
  predictions,
  currentTime,
  timeWindow
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const currentTimestamp = new Date(currentTime).getTime();
  
  const chamberStates = useMemo(() => {
    // Filter predictions for magma chamber parameters
    const chamberPredictions = predictions.filter(p => 
      p.predictionType.includes('magma_chamber') ||
      p.predictionType.includes('reservoir')
    );
    
    const recentPredictions = chamberPredictions.filter(prediction => {
      const predictionTime = new Date(prediction.timestamp).getTime();
      return Math.abs(currentTimestamp - predictionTime) <= timeWindow;
    });
    
    // Group by chamber ID and get latest state for each
    const chamberMap = new Map<string, ModelPrediction>();
    
    recentPredictions.forEach(prediction => {
      const chamberId = prediction.id.split('-')[0]; // Assume ID format: chamberId-timestamp
      const existing = chamberMap.get(chamberId);
      
      if (!existing || new Date(prediction.timestamp).getTime() > new Date(existing.timestamp).getTime()) {
        chamberMap.set(chamberId, prediction);
      }
    });
    
    return Array.from(chamberMap.values());
  }, [predictions, currentTimestamp, timeWindow]);
  
  return (
    <group ref={groupRef}>
      {chamberStates.map((chamber, index) => {
        // Extract chamber parameters from prediction values
        // Assuming values array contains [x, y, z, volume, pressure, temperature]
        const [x, y, z, volume, pressure, temperature] = chamber.values;
        
        const radius = Math.pow(volume / (4/3 * Math.PI), 1/3) * 0.001; // Scale for visualization
        const color = new THREE.Color().setHSL(
          Math.min(temperature / 1500, 1) * 0.8, // Hue based on temperature
          0.8,
          0.5
        );
        
        // Pressure-based pulsation
        const pulsationScale = 1 + Math.sin(Date.now() * 0.001 * pressure * 0.001) * 0.1;
        
        return (
          <group key={`chamber-${chamber.id}-${index}`}>
            {/* Main chamber body */}
            <mesh position={[x * 0.01, y * 0.001, z * 0.01]}>
              <sphereGeometry args={[radius * pulsationScale, 16, 16]} />
              <meshStandardMaterial 
                color={color}
                transparent
                opacity={0.7}
                emissive={color}
                emissiveIntensity={0.3}
              />
            </mesh>
            
            {/* Pressure visualization */}
            <mesh position={[x * 0.01, y * 0.001, z * 0.01]}>
              <sphereGeometry args={[radius * pulsationScale * 1.2, 16, 16]} />
              <meshBasicMaterial 
                color={color}
                transparent
                opacity={Math.min(pressure / 100000, 0.3)} // Pressure in Pa
                wireframe
              />
            </mesh>
            
            {/* Chamber info */}
            <Html position={[x * 0.01, y * 0.001 + radius + 0.5, z * 0.01]}>
              <div style={{
                background: 'rgba(0,0,0,0.8)',
                color: 'white',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '10px',
                textAlign: 'center',
                pointerEvents: 'none'
              }}>
                <div>Vol: {(volume / 1e9).toFixed(1)}km³</div>
                <div>T: {temperature.toFixed(0)}°C</div>
                <div>P: {(pressure / 1e6).toFixed(1)}MPa</div>
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
};

// Time-series particle system for geological processes
interface GeologicalParticleSystemProps {
  processType: 'volcanic_ash' | 'gas_emission' | 'hydrothermal_flow';
  sourceLocation: THREE.Vector3;
  currentTime: string;
  intensity: number;
  windDirection?: THREE.Vector3;
}

export const GeologicalParticleSystem: React.FC<GeologicalParticleSystemProps> = ({
  processType,
  sourceLocation,
  // currentTime,
  intensity,
  windDirection = new THREE.Vector3(1, 0, 0)
}) => {
  const particlesRef = useRef<THREE.Points>(null);
  const particleCount = Math.floor(intensity * 1000);
  
  const { positions, velocities, colors, sizes } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      
      // Initial positions around source
      positions[i3] = sourceLocation.x + (Math.random() - 0.5) * 0.5;
      positions[i3 + 1] = sourceLocation.y + Math.random() * 0.2;
      positions[i3 + 2] = sourceLocation.z + (Math.random() - 0.5) * 0.5;
      
      // Velocities based on process type
      switch (processType) {
        case 'volcanic_ash':
          velocities[i3] = windDirection.x * (0.5 + Math.random() * 0.5);
          velocities[i3 + 1] = 0.2 + Math.random() * 0.3; // Upward
          velocities[i3 + 2] = windDirection.z * (0.5 + Math.random() * 0.5);
          
          colors[i3] = 0.3 + Math.random() * 0.2; // Gray
          colors[i3 + 1] = 0.2 + Math.random() * 0.2;
          colors[i3 + 2] = 0.1 + Math.random() * 0.1;
          break;
          
        case 'gas_emission':
          velocities[i3] = (Math.random() - 0.5) * 0.2;
          velocities[i3 + 1] = 0.1 + Math.random() * 0.2;
          velocities[i3 + 2] = (Math.random() - 0.5) * 0.2;
          
          colors[i3] = 0.8 + Math.random() * 0.2; // White/yellow
          colors[i3 + 1] = 0.8 + Math.random() * 0.2;
          colors[i3 + 2] = 0.6 + Math.random() * 0.2;
          break;
          
        case 'hydrothermal_flow':
          velocities[i3] = (Math.random() - 0.5) * 0.1;
          velocities[i3 + 1] = 0.05 + Math.random() * 0.1;
          velocities[i3 + 2] = (Math.random() - 0.5) * 0.1;
          
          colors[i3] = 0.2 + Math.random() * 0.3; // Blue
          colors[i3 + 1] = 0.4 + Math.random() * 0.3;
          colors[i3 + 2] = 0.8 + Math.random() * 0.2;
          break;
      }
      
      sizes[i] = 0.5 + Math.random() * 1.5;
    }
    
    return { positions, velocities, colors, sizes };
  }, [particleCount, sourceLocation, processType, windDirection]);
  
  useFrame((_state, delta) => {
    if (!particlesRef.current) return;
    
    const positionAttribute = particlesRef.current.geometry.attributes.position;
    const positions = positionAttribute.array as Float32Array;
    
    // Update particle positions
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      
      // Apply velocities
      positions[i3] += velocities[i3] * delta;
      positions[i3 + 1] += velocities[i3 + 1] * delta;
      positions[i3 + 2] += velocities[i3 + 2] * delta;
      
      // Apply gravity and wind effects
      velocities[i3 + 1] -= 0.1 * delta; // Gravity
      
      // Reset particles that have moved too far
      const distance = Math.sqrt(
        (positions[i3] - sourceLocation.x) ** 2 +
        (positions[i3 + 1] - sourceLocation.y) ** 2 +
        (positions[i3 + 2] - sourceLocation.z) ** 2
      );
      
      if (distance > 10 || positions[i3 + 1] < -5) {
        // Reset to source
        positions[i3] = sourceLocation.x + (Math.random() - 0.5) * 0.5;
        positions[i3 + 1] = sourceLocation.y + Math.random() * 0.2;
        positions[i3 + 2] = sourceLocation.z + (Math.random() - 0.5) * 0.5;
        
        // Reset velocity
        velocities[i3 + 1] = 0.1 + Math.random() * 0.2;
      }
    }
    
    positionAttribute.needsUpdate = true;
  });
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particleCount}
          array={colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={particleCount}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.1}
        sizeAttenuation
        vertexColors
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};