import React, { useState, useEffect, useMemo, useCallback, useRef, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Grid } from '@react-three/drei'; // Text, Box as ThreeBox - unused
import * as THREE from 'three';
import {
  Box,
  Paper,
  Typography,
  Grid,
  // Card, CardContent - unused
  Slider,
  // FormControl, InputLabel, Select, MenuItem - unused
  Switch,
  FormControlLabel,
  Button,
  IconButton,
  // Tooltip - unused
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Chip,
  // Alert - unused
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  Settings,
  Download,
  Fullscreen,
  ExpandMore,
  Layers,
  Speed,
  Thermostat,
  Timeline,
  RotateLeft,
  ZoomIn,
  ZoomOut
} from '@mui/icons-material';
import { format } from 'date-fns';

interface MantleDynamics3DProps {
  mantleData?: MantleConvectionData;
  plateData?: PlateMotionData;
  onParameterChange?: (parameter: string, value: number) => void;
  onSimulationControl?: (action: 'play' | 'pause' | 'stop' | 'reset') => void;
  realTimeMode?: boolean;
}

interface MantleConvectionData {
  temperatureField: Float32Array;
  velocityField: {
    x: Float32Array;
    y: Float32Array;
    z: Float32Array;
  };
  pressureField: Float32Array;
  viscosityField: Float32Array;
  dimensions: {
    width: number;
    height: number;
    depth: number;
  };
  resolution: {
    x: number;
    y: number;
    z: number;
  };
  timeStep: number;
  currentTime: number;
  physicalProperties: {
    rayleighNumber: number;
    prandtlNumber: number;
    thermalExpansion: number;
    viscosityContrast: number;
  };
}

interface PlateMotionData {
  plates: PlateSegment[];
  boundaries: PlateBoundary[];
  hotspots: Hotspot[];
  subductionZones: SubductionZone[];
}

interface PlateSegment {
  id: string;
  name: string;
  vertices: THREE.Vector3[];
  velocity: THREE.Vector3;
  age: number;
  thickness: number;
  density: number;
}

interface PlateBoundary {
  id: string;
  type: 'divergent' | 'convergent' | 'transform';
  points: THREE.Vector3[];
  spreadingRate?: number;
  subductionRate?: number;
}

interface Hotspot {
  id: string;
  position: THREE.Vector3;
  intensity: number;
  radius: number;
  temperature: number;
}

interface SubductionZone {
  id: string;
  trench: THREE.Vector3[];
  slabGeometry: THREE.Vector3[];
  dip: number;
  depth: number;
  age: number;
}

// Mantle convection visualization component
const MantleConvectionVisualization: React.FC<{
  data: MantleConvectionData;
  showTemperature: boolean;
  showVelocity: boolean;
  showPressure: boolean;
  timeStep: number;
  colorScheme: string;
}> = ({ data, showTemperature, showVelocity, showPressure, timeStep, colorScheme }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { scene } = useThree();

  // Create temperature field visualization
  const temperatureGeometry = useMemo(() => {
    if (!showTemperature || !data.temperatureField) return null;

    const geometry = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];
    const { x: resX, y: resY, z: resZ } = data.resolution;
    const { width, height, depth } = data.dimensions;

    // Create color scale for temperature
    const colorScale = d3.scaleSequential(d3.interpolateInferno)
      .domain([0, 1]); // Normalized temperature range

    for (let i = 0; i < resX; i++) {
      for (let j = 0; j < resY; j++) {
        for (let k = 0; k < resZ; k++) {
          const index = i * resY * resZ + j * resZ + k;
          const temperature = data.temperatureField[index];

          // Skip if temperature is below threshold for performance
          if (temperature < 0.3) continue;

          const x = (i / resX - 0.5) * width;
          const y = (j / resY - 0.5) * height;
          const z = (k / resZ - 0.5) * depth;

          positions.push(x, y, z);

          const color = new THREE.Color(colorScale(temperature));
          colors.push(color.r, color.g, color.b);
        }
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    return geometry;
  }, [data, showTemperature, colorScheme]);

  // Create velocity field visualization
  const velocityArrows = useMemo(() => {
    if (!showVelocity || !data.velocityField) return [];

    const arrows: JSX.Element[] = [];
    const { x: resX, y: resY, z: resZ } = data.resolution;
    const { width, height, depth } = data.dimensions;
    const skipFactor = 4; // Show every 4th arrow for performance

    for (let i = 0; i < resX; i += skipFactor) {
      for (let j = 0; j < resY; j += skipFactor) {
        for (let k = 0; k < resZ; k += skipFactor) {
          const index = i * resY * resZ + j * resZ + k;
          
          const vx = data.velocityField.x[index];
          const vy = data.velocityField.y[index];
          const vz = data.velocityField.z[index];
          
          const magnitude = Math.sqrt(vx * vx + vy * vy + vz * vz);
          if (magnitude < 0.1) continue; // Skip small velocities

          const x = (i / resX - 0.5) * width;
          const y = (j / resY - 0.5) * height;
          const z = (k / resZ - 0.5) * depth;

          const direction = new THREE.Vector3(vx, vy, vz).normalize();
          const length = Math.min(magnitude * 10, 2); // Scale arrow length

          arrows.push(
            <group key={`arrow-${i}-${j}-${k}`} position={[x, y, z]}>
              <mesh>
                <cylinderGeometry args={[0.02, 0.02, length, 8]} />
                <meshBasicMaterial color="#00ff00" />
              </mesh>
              <mesh position={[0, length / 2, 0]}>
                <coneGeometry args={[0.05, 0.2, 8]} />
                <meshBasicMaterial color="#00ff00" />
              </mesh>
            </group>
          );
        }
      }
    }

    return arrows;
  }, [data, showVelocity]);

  return (
    <group>
      {/* Temperature field */}
      {temperatureGeometry && (
        <points geometry={temperatureGeometry}>
          <pointsMaterial
            size={0.1}
            vertexColors
            transparent
            opacity={0.8}
            sizeAttenuation={false}
          />
        </points>
      )}

      {/* Velocity field */}
      {velocityArrows}

      {/* Mantle boundaries */}
      <group>
        {/* Core-mantle boundary */}
        <mesh>
          <sphereGeometry args={[data.dimensions.width * 0.3, 32, 32]} />
          <meshBasicMaterial
            color="#ff4444"
            transparent
            opacity={0.2}
            wireframe
          />
        </mesh>

        {/* Upper mantle boundary */}
        <mesh>
          <sphereGeometry args={[data.dimensions.width * 0.45, 32, 32]} />
          <meshBasicMaterial
            color="#ffaa44"
            transparent
            opacity={0.1}
            wireframe
          />
        </mesh>
      </group>
    </group>
  );
};

// Plate tectonics visualization component
const PlateVisualization: React.FC<{
  data: PlateMotionData;
  showPlates: boolean;
  showBoundaries: boolean;
  showHotspots: boolean;
  animationSpeed: number;
}> = ({ data, showPlates, showBoundaries, showHotspots, animationSpeed }) => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current && animationSpeed > 0) {
      // Animate plate motion
      groupRef.current.rotation.y += animationSpeed * 0.001;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Plate segments */}
      {showPlates && data.plates.map((plate) => (
        <group key={plate.id}>
          <mesh>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={plate.vertices.length}
                array={new Float32Array(plate.vertices.flatMap(v => [v.x, v.y, v.z]))}
                itemSize={3}
              />
            </bufferGeometry>
            <meshLambertMaterial
              color={`hsl(${plate.age * 10 % 360}, 70%, 50%)`}
              transparent
              opacity={0.7}
            />
          </mesh>
          
          {/* Plate velocity vector */}
          <arrowHelper
            args={[
              plate.velocity.normalize(),
              new THREE.Vector3(0, 0, 0),
              plate.velocity.length() * 100,
              0x00ff00
            ]}
          />
        </group>
      ))}

      {/* Plate boundaries */}
      {showBoundaries && data.boundaries.map((boundary) => {
        const color = boundary.type === 'divergent' ? '#ff0000' :
                     boundary.type === 'convergent' ? '#0000ff' : '#ffff00';
        
        return (
          <line key={boundary.id}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={boundary.points.length}
                array={new Float32Array(boundary.points.flatMap(p => [p.x, p.y, p.z]))}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color={color} linewidth={3} />
          </line>
        );
      })}

      {/* Hotspots */}
      {showHotspots && data.hotspots.map((hotspot) => (
        <mesh key={hotspot.id} position={hotspot.position}>
          <sphereGeometry args={[hotspot.radius, 16, 16]} />
          <meshBasicMaterial
            color="#ff6600"
            transparent
            opacity={hotspot.intensity}
          />
        </mesh>
      ))}

      {/* Subduction zones */}
      {data.subductionZones.map((zone) => (
        <group key={zone.id}>
          {/* Trench */}
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={zone.trench.length}
                array={new Float32Array(zone.trench.flatMap(p => [p.x, p.y, p.z]))}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#000080" linewidth={5} />
          </line>
          
          {/* Slab geometry */}
          <mesh>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={zone.slabGeometry.length}
                array={new Float32Array(zone.slabGeometry.flatMap(p => [p.x, p.y, p.z]))}
                itemSize={3}
              />
            </bufferGeometry>
            <meshLambertMaterial
              color="#4444ff"
              transparent
              opacity={0.5}
              side={THREE.DoubleSide}
            />
          </mesh>
        </group>
      ))}
    </group>
  );
};

export const MantleDynamics3D: React.FC<MantleDynamics3DProps> = ({
  mantleData,
  plateData,
  onParameterChange,
  onSimulationControl,
  realTimeMode = false
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);
  const [timeStep, setTimeStep] = useState(0);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [cameraPosition, setCameraPosition] = useState([10, 10, 10]);
  
  // Visualization controls
  const [showTemperature, setShowTemperature] = useState(true);
  const [showVelocity, setShowVelocity] = useState(false);
  const [showPressure, setShowPressure] = useState(false);
  const [showPlates, setShowPlates] = useState(true);
  const [showBoundaries, setShowBoundaries] = useState(true);
  const [showHotspots, setShowHotspots] = useState(true);
  
  // Simulation parameters
  const [rayleighNumber, setRayleighNumber] = useState(mantleData?.physicalProperties.rayleighNumber || 1e6);
  const [viscosityContrast, setViscosityContrast] = useState(mantleData?.physicalProperties.viscosityContrast || 1000);
  const [thermalExpansion, setThermalExpansion] = useState(mantleData?.physicalProperties.thermalExpansion || 3e-5);

  // Generate synthetic mantle data if not provided
  const defaultMantleData: MantleConvectionData = useMemo(() => {
    if (mantleData) return mantleData;

    const resX = 32, resY = 32, resZ = 32;
    const size = resX * resY * resZ;
    
    return {
      temperatureField: new Float32Array(size).map(() => Math.random()),
      velocityField: {
        x: new Float32Array(size).map(() => (Math.random() - 0.5) * 0.1),
        y: new Float32Array(size).map(() => (Math.random() - 0.5) * 0.1),
        z: new Float32Array(size).map(() => (Math.random() - 0.5) * 0.1)
      },
      pressureField: new Float32Array(size).map(() => Math.random()),
      viscosityField: new Float32Array(size).map(() => Math.random() * 1000),
      dimensions: { width: 10, height: 10, depth: 10 },
      resolution: { x: resX, y: resY, z: resZ },
      timeStep: 0.001,
      currentTime: 0,
      physicalProperties: {
        rayleighNumber: 1e6,
        prandtlNumber: 1,
        thermalExpansion: 3e-5,
        viscosityContrast: 1000
      }
    };
  }, [mantleData]);

  // Generate synthetic plate data if not provided
  const defaultPlateData: PlateMotionData = useMemo(() => {
    if (plateData) return plateData;

    return {
      plates: [
        {
          id: 'pacific',
          name: 'Pacific Plate',
          vertices: [
            new THREE.Vector3(-5, 0, -5),
            new THREE.Vector3(5, 0, -5),
            new THREE.Vector3(5, 0, 5),
            new THREE.Vector3(-5, 0, 5)
          ],
          velocity: new THREE.Vector3(0.1, 0, 0.05),
          age: 100,
          thickness: 100,
          density: 3300
        }
      ],
      boundaries: [
        {
          id: 'mid-atlantic',
          type: 'divergent',
          points: [
            new THREE.Vector3(0, 0, -5),
            new THREE.Vector3(0, 0, 5)
          ],
          spreadingRate: 2.5
        }
      ],
      hotspots: [
        {
          id: 'hawaii',
          position: new THREE.Vector3(2, -1, 2),
          intensity: 0.8,
          radius: 0.5,
          temperature: 1600
        }
      ],
      subductionZones: [
        {
          id: 'japan-trench',
          trench: [
            new THREE.Vector3(3, 0, -3),
            new THREE.Vector3(3, 0, 3)
          ],
          slabGeometry: [
            new THREE.Vector3(3, 0, -3),
            new THREE.Vector3(2, -2, -3),
            new THREE.Vector3(2, -2, 3),
            new THREE.Vector3(3, 0, 3)
          ],
          dip: 45,
          depth: 600,
          age: 150
        }
      ]
    };
  }, [plateData]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setTimeStep(prev => prev + animationSpeed * 0.1);
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, animationSpeed]);

  // Handle simulation controls
  const handleSimulationControl = useCallback((action: 'play' | 'pause' | 'stop' | 'reset') => {
    switch (action) {
      case 'play':
        setIsPlaying(true);
        break;
      case 'pause':
        setIsPlaying(false);
        break;
      case 'stop':
        setIsPlaying(false);
        setTimeStep(0);
        break;
      case 'reset':
        setIsPlaying(false);
        setTimeStep(0);
        // Reset other parameters
        break;
    }
    onSimulationControl?.(action);
  }, [onSimulationControl]);

  // Handle parameter changes
  const handleParameterChange = useCallback((parameter: string, value: number) => {
    switch (parameter) {
      case 'rayleighNumber':
        setRayleighNumber(value);
        break;
      case 'viscosityContrast':
        setViscosityContrast(value);
        break;
      case 'thermalExpansion':
        setThermalExpansion(value);
        break;
    }
    onParameterChange?.(parameter, value);
  }, [onParameterChange]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5">Mantle Dynamics 3D Visualization</Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={`Time: ${timeStep.toFixed(2)} Myr`}
              color="primary"
              variant="outlined"
            />
            
            <IconButton
              onClick={() => handleSimulationControl(isPlaying ? 'pause' : 'play')}
              color="primary"
            >
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>
            
            <IconButton onClick={() => handleSimulationControl('stop')}>
              <Stop />
            </IconButton>
            
            <IconButton onClick={() => handleSimulationControl('reset')}>
              <RotateLeft />
            </IconButton>
          </Box>
        </Box>

        {/* Simulation Parameters */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>
              Animation Speed: {animationSpeed}x
            </Typography>
            <Slider
              value={animationSpeed}
              onChange={(_, value) => setAnimationSpeed(value as number)}
              min={0}
              max={5}
              step={0.1}
              marks={[
                { value: 0, label: '0x' },
                { value: 1, label: '1x' },
                { value: 5, label: '5x' }
              ]}
            />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>
              Rayleigh Number: {rayleighNumber.toExponential(1)}
            </Typography>
            <Slider
              value={Math.log10(rayleighNumber)}
              onChange={(_, value) => handleParameterChange('rayleighNumber', Math.pow(10, value as number))}
              min={4}
              max={8}
              step={0.1}
              marks={[
                { value: 4, label: '10⁴' },
                { value: 6, label: '10⁶' },
                { value: 8, label: '10⁸' }
              ]}
            />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>
              Viscosity Contrast: {viscosityContrast}
            </Typography>
            <Slider
              value={Math.log10(viscosityContrast)}
              onChange={(_, value) => handleParameterChange('viscosityContrast', Math.pow(10, value as number))}
              min={1}
              max={6}
              step={0.1}
              marks={[
                { value: 1, label: '10¹' },
                { value: 3, label: '10³' },
                { value: 6, label: '10⁶' }
              ]}
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Main Content */}
      <Box sx={{ flex: 1, display: 'flex', gap: 2 }}>
        {/* 3D Visualization */}
        <Paper sx={{ flex: 1, position: 'relative' }}>
          <Canvas
            style={{ width: '100%', height: '100%' }}
            gl={{ antialias: true, alpha: true }}
          >
            <Suspense fallback={null}>
              {/* Lighting */}
              <ambientLight intensity={0.4} />
              <directionalLight position={[10, 10, 5]} intensity={0.8} />
              <pointLight position={[-10, -10, -5]} intensity={0.3} />

              {/* Camera */}
              <PerspectiveCamera
                makeDefault
                position={cameraPosition as [number, number, number]}
                fov={60}
              />

              {/* Controls */}
              <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                maxDistance={50}
                minDistance={5}
              />

              {/* Environment */}
              <Environment preset="sunset" />

              {/* Grid */}
              <Grid
                args={[20, 20]}
                cellSize={1}
                cellThickness={0.5}
                cellColor="#6f6f6f"
                sectionSize={5}
                sectionThickness={1}
                sectionColor="#9d4b4b"
                fadeDistance={25}
                fadeStrength={1}
              />

              {/* Mantle Convection */}
              {currentTab === 0 && (
                <MantleConvectionVisualization
                  data={defaultMantleData}
                  showTemperature={showTemperature}
                  showVelocity={showVelocity}
                  showPressure={showPressure}
                  timeStep={timeStep}
                  colorScheme="inferno"
                />
              )}

              {/* Plate Tectonics */}
              {currentTab === 1 && (
                <PlateVisualization
                  data={defaultPlateData}
                  showPlates={showPlates}
                  showBoundaries={showBoundaries}
                  showHotspots={showHotspots}
                  animationSpeed={animationSpeed}
                />
              )}

              {/* Coordinate axes */}
              <axesHelper args={[5]} />
            </Suspense>
          </Canvas>

          {/* Overlay Controls */}
          <Box sx={{ position: 'absolute', top: 16, right: 16 }}>
            <Paper sx={{ p: 1 }}>
              <Tabs
                value={currentTab}
                onChange={(_, value) => setCurrentTab(value)}
                orientation="vertical"
              >
                <Tab label="Mantle" />
                <Tab label="Plates" />
              </Tabs>
            </Paper>
          </Box>
        </Paper>

        {/* Control Panel */}
        <Paper sx={{ width: 300, p: 2, overflow: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Visualization Controls
          </Typography>

          {currentTab === 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Mantle Convection
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={showTemperature}
                    onChange={(e) => setShowTemperature(e.target.checked)}
                  />
                }
                label="Temperature Field"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={showVelocity}
                    onChange={(e) => setShowVelocity(e.target.checked)}
                  />
                }
                label="Velocity Vectors"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={showPressure}
                    onChange={(e) => setShowPressure(e.target.checked)}
                  />
                }
                label="Pressure Field"
              />
            </Box>
          )}

          {currentTab === 1 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Plate Tectonics
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={showPlates}
                    onChange={(e) => setShowPlates(e.target.checked)}
                  />
                }
                label="Plate Segments"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={showBoundaries}
                    onChange={(e) => setShowBoundaries(e.target.checked)}
                  />
                }
                label="Plate Boundaries"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={showHotspots}
                    onChange={(e) => setShowHotspots(e.target.checked)}
                  />
                }
                label="Hotspots"
              />
            </Box>
          )}

          <Divider sx={{ my: 2 }} />

          {/* Physical Parameters */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">Physical Parameters</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Typography variant="body2">
                  Thermal Expansion: {thermalExpansion.toExponential(1)} K⁻¹
                </Typography>
                <Slider
                  value={Math.log10(thermalExpansion * 1e5)}
                  onChange={(_, value) => handleParameterChange('thermalExpansion', Math.pow(10, value as number) / 1e5)}
                  min={0}
                  max={2}
                  step={0.1}
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Statistics */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">Statistics</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box>
                <Typography variant="body2">
                  Current Time: {timeStep.toFixed(2)} Myr
                </Typography>
                <Typography variant="body2">
                  Max Temperature: {defaultMantleData.temperatureField ? Math.max(...defaultMantleData.temperatureField).toFixed(2) : 'N/A'}
                </Typography>
                <Typography variant="body2">
                  Max Velocity: {defaultMantleData.velocityField ? Math.max(...defaultMantleData.velocityField.x).toFixed(4) : 'N/A'} m/s
                </Typography>
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Export Options */}
          <Box sx={{ mt: 2 }}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<Download />}
              onClick={() => {
                // Export functionality
              }}
            >
              Export Animation
            </Button>
          </Box>
        </Paper>
      </Box>

      {/* Status Bar */}
      <Paper sx={{ p: 1, mt: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'between', gap: 2 }}>
          <Typography variant="caption">
            Status: {isPlaying ? 'Running' : 'Paused'}
          </Typography>
          <Typography variant="caption">
            FPS: {realTimeMode ? '60' : '30'}
          </Typography>
          <Typography variant="caption">
            Particles: {defaultMantleData.temperatureField?.length || 0}
          </Typography>
          {isPlaying && <LinearProgress sx={{ flex: 1 }} />}
        </Box>
      </Paper>
    </Box>
  );
};

export default MantleDynamics3D;