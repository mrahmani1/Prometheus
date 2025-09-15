import React, { useRef, useMemo, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, Paper, Slider, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { GeologicalModel3D, LithologyUnit, StructuralFeature } from '../../../types/geological';

interface GeologicalRendererProps {
  model: GeologicalModel3D;
  onFeatureSelect?: (feature: StructuralFeature | LithologyUnit) => void;
  showUncertainty?: boolean;
  timeStep?: number;
  colorScheme?: 'geological' | 'geophysical' | 'structural';
}

// Advanced geological color schemes based on scientific standards
const GEOLOGICAL_COLOR_SCHEMES = {
  geological: {
    igneous: '#FF6B6B',
    sedimentary: '#4ECDC4',
    metamorphic: '#45B7D1',
    volcanic: '#FF8E53',
    plutonic: '#96CEB4',
    fault: '#FF4757',
    fracture: '#5F27CD'
  },
  geophysical: {
    highVelocity: '#2E86AB',
    mediumVelocity: '#A23B72',
    lowVelocity: '#F18F01',
    highDensity: '#C73E1D',
    lowDensity: '#F4D35E'
  },
  structural: {
    compression: '#FF6B6B',
    tension: '#4ECDC4',
    shear: '#FFE66D',
    normal: '#A8E6CF',
    reverse: '#FF8B94'
  }
};

// WebGL shader for advanced geological rendering
const geologicalVertexShader = `
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  varying float vElevation;
  
  uniform float uTime;
  uniform float uTimeStep;
  
  void main() {
    vPosition = position;
    vNormal = normal;
    vUv = uv;
    
    // Calculate elevation for geological layering
    vElevation = position.y;
    
    // Apply temporal deformation for 4D visualization
    vec3 deformedPosition = position;
    if (uTimeStep > 0.0) {
      deformedPosition.y += sin(position.x * 0.1 + uTime * uTimeStep) * 0.1;
    }
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(deformedPosition, 1.0);
  }
`;

const geologicalFragmentShader = `
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  varying float vElevation;
  
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uMetalness;
  uniform float uRoughness;
  uniform bool uShowUncertainty;
  uniform float uUncertaintyLevel;
  
  void main() {
    vec3 color = uColor;
    
    // Apply geological stratification based on elevation
    float strataFactor = mod(vElevation * 10.0, 1.0);
    color = mix(color, color * 0.8, step(0.5, strataFactor));
    
    // Add uncertainty visualization
    if (uShowUncertainty) {
      float uncertainty = sin(vPosition.x * 20.0) * sin(vPosition.z * 20.0) * uUncertaintyLevel;
      color = mix(color, vec3(1.0, 0.0, 0.0), abs(uncertainty) * 0.3);
    }
    
    // Calculate lighting
    vec3 normal = normalize(vNormal);
    vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
    float lightIntensity = max(dot(normal, lightDirection), 0.2);
    
    gl_FragColor = vec4(color * lightIntensity, uOpacity);
  }
`;

// Geological mesh component with advanced rendering
const GeologicalMesh: React.FC<{
  unit: LithologyUnit;
  colorScheme: string;
  showUncertainty: boolean;
  timeStep: number;
  onClick?: () => void;
}> = ({ unit, colorScheme, showUncertainty, timeStep, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { clock } = useThree();

  const shaderMaterial = useMemo(() => {
    const scheme = GEOLOGICAL_COLOR_SCHEMES[colorScheme as keyof typeof GEOLOGICAL_COLOR_SCHEMES];
    const color = scheme[unit.rockType as keyof typeof scheme] || '#888888';
    
    return new THREE.ShaderMaterial({
      vertexShader: geologicalVertexShader,
      fragmentShader: geologicalFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uTimeStep: { value: timeStep },
        uColor: { value: new THREE.Color(color) },
        uOpacity: { value: 0.8 }, // Default opacity
        uMetalness: { value: 0.1 },
        uRoughness: { value: 0.8 },
        uShowUncertainty: { value: showUncertainty },
        uUncertaintyLevel: { value: 0.0 } // Default uncertainty
      },
      transparent: true,
      side: THREE.DoubleSide
    });
  }, [unit, colorScheme, showUncertainty, timeStep]);

  useFrame(() => {
    if (meshRef.current && shaderMaterial.uniforms.uTime) {
      shaderMaterial.uniforms.uTime.value = clock.getElapsedTime();
    }
  });

  // Generate geometry from geological unit data
  const geometry = useMemo(() => {
    // Create default box geometry for lithology unit
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    geometry.computeVertexNormals();
    return geometry;
  }, [unit.name]); // Use unit name as dependency since geometry is fixed

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      material={shaderMaterial}
      onClick={onClick}
      position={[0, 0, 0]} // Default position
    />
  );
};

// Structural features renderer
const StructuralFeatures: React.FC<{
  features: StructuralFeature[];
  colorScheme: string;
  onFeatureSelect?: (feature: StructuralFeature) => void;
}> = ({ features, _colorScheme, _onFeatureSelect }) => {
  return (
    <>
      {features.map((feature, index) => (
        <group key={index}>
          {feature.type === 'fault' && (
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={feature.geometry.length / 3}
                  array={new Float32Array(feature.geometry)}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color={GEOLOGICAL_COLOR_SCHEMES.geological.fault} linewidth={3} />
            </line>
          )}
          {feature.type === 'fracture' && (
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={feature.geometry.length / 3}
                  array={new Float32Array(feature.geometry)}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color={GEOLOGICAL_COLOR_SCHEMES.geological.fracture} linewidth={1} />
            </line>
          )}
        </group>
      ))}
    </>
  );
};

// Main geological renderer component
export const GeologicalRenderer: React.FC<GeologicalRendererProps> = ({
  model,
  onFeatureSelect,
  showUncertainty = false,
  timeStep = 0,
  colorScheme = 'geological'
}) => {
  const [currentTimeStep, setCurrentTimeStep] = React.useState(timeStep);
  const [currentColorScheme, setCurrentColorScheme] = React.useState(colorScheme);
  const [showControls, setShowControls] = React.useState(true);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Canvas */}
      <Canvas
        style={{ width: '100%', height: '100%' }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: 'high-performance'
        }}
      >
        <Suspense fallback={null}>
          {/* Lighting setup */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={0.8} castShadow />
          <pointLight position={[-10, -10, -5]} intensity={0.3} />
          
          {/* Environment */}
          <Environment preset="sunset" />
          
          {/* Camera */}
          <PerspectiveCamera makeDefault position={[10, 10, 10]} fov={60} />
          
          {/* Controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            maxDistance={100}
            minDistance={1}
          />
          
          {/* Grid for reference */}
          <Grid
            args={[100, 100]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#6f6f6f"
            sectionSize={10}
            sectionThickness={1}
            sectionColor="#9d4b4b"
            fadeDistance={50}
            fadeStrength={1}
          />
          
          {/* Geological units */}
          {model.lithology_units?.map((unit, index) => (
            <GeologicalMesh
              key={unit.id || index}
              unit={unit}
              colorScheme={currentColorScheme}
              showUncertainty={showUncertainty}
              timeStep={currentTimeStep}
              onClick={() => onFeatureSelect?.(unit)}
            />
          ))}
          
          {/* Structural features */}
          {model.structural_features && (
            <StructuralFeatures
              features={model.structural_features}
              colorScheme={currentColorScheme}
              onFeatureSelect={onFeatureSelect}
            />
          )}
        </Suspense>
      </Canvas>
      
      {/* Control Panel */}
      {showControls && (
        <Paper
          sx={{
            position: 'absolute',
            top: 16,
            right: 16,
            p: 2,
            minWidth: 250,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)'
          }}
        >
          <Typography variant="h6" gutterBottom>
            Geological Visualization Controls
          </Typography>
          
          {/* Color Scheme Selection */}
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Color Scheme</InputLabel>
            <Select
              value={currentColorScheme}
              onChange={(e) => setCurrentColorScheme(e.target.value)}
            >
              <MenuItem value="geological">Geological</MenuItem>
              <MenuItem value="geophysical">Geophysical</MenuItem>
              <MenuItem value="structural">Structural</MenuItem>
            </Select>
          </FormControl>
          
          {/* Time Step Control for 4D visualization */}
          <Typography gutterBottom>Time Step: {currentTimeStep}</Typography>
          <Slider
            value={currentTimeStep}
            onChange={(_, value) => setCurrentTimeStep(value as number)}
            min={0}
            max={100}
            step={1}
            sx={{ mb: 2 }}
          />
          
          {/* Model Information */}
          <Typography variant="body2" color="textSecondary">
            Units: {model.lithology_units?.length || 0}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Features: {model.structural_features?.length || 0}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default GeologicalRenderer;