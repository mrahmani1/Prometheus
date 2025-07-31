import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree, extend } from '@react-three/fiber';
import { OrbitControls, Text, Html, useTexture, Sphere, Line } from '@react-three/drei';
import { EffectComposer, Bloom, DepthOfField } from '@react-three/postprocessing';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Slider,
  Switch,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Badge,
  Progress,
  Input,
  Label
} from '@/components/ui';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ZoomIn, 
  ZoomOut,
  Globe,
  Mountain,
  Flame,
  Activity,
  Settings,
  Download,
  Eye,
  EyeOff,
  Clock,
  MapPin,
  Layers
} from 'lucide-react';

// Extend Three.js with custom materials
extend({ 
  ShaderMaterial: THREE.ShaderMaterial,
  BufferGeometry: THREE.BufferGeometry,
  BufferAttribute: THREE.BufferAttribute
});

interface TectonicPlate {
  id: string;
  name: string;
  type: 'oceanic' | 'continental' | 'mixed';
  vertices: number[][];
  age: number;
  velocity: [number, number, number];
  color: string;
  elevation: number[];
  thickness: number[];
}

interface VolcanicSystem {
  id: string;
  name: string;
  type: 'hotspot' | 'arc' | 'ridge' | 'flood_basalt';
  location: [number, number, number];
  activity: number;
  temperature: number;
  magmaProduction: number;
  plumeSource?: [number, number, number];
}

interface SimulationState {
  currentTime: number;
  timeRange: [number, number];
  isPlaying: boolean;
  playbackSpeed: number;
  plates: TectonicPlate[];
  volcanicSystems: VolcanicSystem[];
  mantlePlumes: any[];
  selectedRegion?: string;
  zoomLevel: number;
  viewMode: '3d' | '4d' | 'cross_section';
}

// Custom Earth shader for realistic rendering
const EarthShaderMaterial = () => {
  const earthVertexShader = `
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vNormal = normalize(normalMatrix * normal);
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;
  
  const earthFragmentShader = `
    uniform sampler2D earthTexture;
    uniform sampler2D bumpTexture;
    uniform sampler2D specularTexture;
    uniform float time;
    uniform vec3 lightPosition;
    
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
      vec3 earthColor = texture2D(earthTexture, vUv).rgb;
      vec3 bump = texture2D(bumpTexture, vUv).rgb;
      float specular = texture2D(specularTexture, vUv).r;
      
      // Lighting calculations
      vec3 lightDir = normalize(lightPosition - vPosition);
      float diff = max(dot(vNormal, lightDir), 0.0);
      
      // Specular reflection for oceans
      vec3 viewDir = normalize(-vPosition);
      vec3 reflectDir = reflect(-lightDir, vNormal);
      float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * specular;
      
      // Atmospheric scattering effect
      float atmosphere = pow(1.0 - dot(vNormal, viewDir), 3.0);
      
      vec3 finalColor = earthColor * diff + vec3(spec) + vec3(0.3, 0.6, 1.0) * atmosphere * 0.3;
      
      gl_FragColor = vec4(finalColor, 1.0);
    }
  `;
  
  return (
    <shaderMaterial
      vertexShader={earthVertexShader}
      fragmentShader={earthFragmentShader}
      uniforms={{
        earthTexture: { value: null },
        bumpTexture: { value: null },
        specularTexture: { value: null },
        time: { value: 0 },
        lightPosition: { value: new THREE.Vector3(10, 10, 10) }
      }}
    />
  );
};

// Tectonic plate component with dynamic geometry
const TectonicPlateComponent: React.FC<{
  plate: TectonicPlate;
  time: number;
  showVelocity: boolean;
  showAge: boolean;
}> = ({ plate, time, showVelocity, showAge }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Convert lat/lon vertices to 3D coordinates
  const vertices = React.useMemo(() => {
    const earthRadius = 6.371;
    return plate.vertices.map(([lat, lon]) => {
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = (lon + 180) * (Math.PI / 180);
      
      return [
        earthRadius * Math.sin(phi) * Math.cos(theta),
        earthRadius * Math.cos(phi),
        earthRadius * Math.sin(phi) * Math.sin(theta)
      ];
    });
  }, [plate.vertices]);
  
  // Create plate geometry
  const geometry = React.useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(vertices.flat());
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Add elevation data
    const elevations = new Float32Array(plate.elevation);
    geom.setAttribute('elevation', new THREE.BufferAttribute(elevations, 1));
    
    return geom;
  }, [vertices, plate.elevation]);
  
  // Animate plate motion
  useFrame((state) => {
    if (meshRef.current) {
      // Apply plate motion based on velocity vector
      const [vx, vy, vz] = plate.velocity;
      meshRef.current.rotation.x += vx * 0.0001;
      meshRef.current.rotation.y += vy * 0.0001;
      meshRef.current.rotation.z += vz * 0.0001;
    }
  });
  
  return (
    <group>
      <mesh ref={meshRef} geometry={geometry}>
        <meshPhongMaterial
          color={plate.color}
          transparent
          opacity={0.8}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Velocity vectors */}
      {showVelocity && (
        <Line
          points={[
            [0, 0, 0],
            plate.velocity.map(v => v * 0.1) as [number, number, number]
          ]}
          color="yellow"
          lineWidth={2}
        />
      )}
      
      {/* Age information */}
      {showAge && (
        <Html position={[0, 0, 6.5]}>
          <div className="bg-black bg-opacity-75 text-white p-2 rounded text-xs">
            {plate.name}<br />
            Age: {plate.age.toFixed(0)} Ma<br />
            Type: {plate.type}
          </div>
        </Html>
      )}
    </group>
  );
};

// Volcanic system component with dynamic effects
const VolcanicSystemComponent: React.FC<{
  system: VolcanicSystem;
  time: number;
  showPlumes: boolean;
}> = ({ system, time, showPlumes }) => {
  const [lat, lon, depth] = system.location;
  const earthRadius = 6.371;
  
  // Convert to 3D coordinates
  const position = React.useMemo(() => {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const r = earthRadius + (depth / 1000); // Convert depth to radius offset
    
    return [
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    ] as [number, number, number];
  }, [lat, lon, depth, earthRadius]);
  
  // Animate volcanic activity
  const activityRef = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    if (activityRef.current) {
      const intensity = system.activity * (0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 2));
      activityRef.current.scale.setScalar(0.5 + intensity * 0.5);
    }
  });
  
  return (
    <group position={position}>
      {/* Volcanic center */}
      <mesh ref={activityRef}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial
          color={system.type === 'hotspot' ? '#ff4444' : '#ff8844'}
          transparent
          opacity={0.8}
        />
      </mesh>
      
      {/* Mantle plume visualization */}
      {showPlumes && system.plumeSource && (
        <Line
          points={[
            position,
            system.plumeSource.map((coord, i) => 
              i === 2 ? coord / 1000 : coord * Math.PI / 180
            ) as [number, number, number]
          ]}
          color="#ff6600"
          lineWidth={3}
          transparent
          opacity={0.6}
        />
      )}
      
      {/* System information */}
      <Html>
        <div className="bg-red-900 bg-opacity-75 text-white p-1 rounded text-xs">
          {system.name}<br />
          {system.type}<br />
          Activity: {(system.activity * 100).toFixed(0)}%
        </div>
      </Html>
    </group>
  );
};

// Main visualization component
const GlobalTectonicVisualization: React.FC = () => {
  const [simulationState, setSimulationState] = useState<SimulationState>({
    currentTime: 0,
    timeRange: [-200, 0], // 200 Ma ago to present
    isPlaying: false,
    playbackSpeed: 1.0,
    plates: [],
    volcanicSystems: [],
    mantlePlumes: [],
    zoomLevel: 1.0,
    viewMode: '3d'
  });
  
  const [visualizationSettings, setVisualizationSettings] = useState({
    showPlates: true,
    showBoundaries: true,
    showVolcanoes: true,
    showPlumes: true,
    showVelocityVectors: false,
    showAgeColors: true,
    showTopography: true,
    showHeatFlow: false,
    atmosphereEffect: true,
    qualityLevel: 'high'
  });
  
  const [selectedRegion, setSelectedRegion] = useState<string>('global');
  const [isLoading, setIsLoading] = useState(false);
  
  // Initialize simulation data
  useEffect(() => {
    initializeSimulation();
  }, []);
  
  const initializeSimulation = async () => {
    setIsLoading(true);
    
    // Generate mock tectonic plates
    const mockPlates: TectonicPlate[] = [
      {
        id: 'pacific',
        name: 'Pacific Plate',
        type: 'oceanic',
        vertices: generatePlateVertices('pacific'),
        age: 180,
        velocity: [10.2, -6.8, 0.65],
        color: '#4a90e2',
        elevation: generateElevationData(100),
        thickness: generateThicknessData(100)
      },
      {
        id: 'north_american',
        name: 'North American Plate',
        type: 'continental',
        vertices: generatePlateVertices('north_american'),
        age: 3800,
        velocity: [2.5, 1.8, 0.22],
        color: '#8b4513',
        elevation: generateElevationData(80),
        thickness: generateThicknessData(80)
      },
      {
        id: 'eurasian',
        name: 'Eurasian Plate',
        type: 'continental',
        vertices: generatePlateVertices('eurasian'),
        age: 3800,
        velocity: [0.9, 2.1, 0.26],
        color: '#228b22',
        elevation: generateElevationData(90),
        thickness: generateThicknessData(90)
      }
    ];
    
    // Generate mock volcanic systems
    const mockVolcanicSystems: VolcanicSystem[] = [
      {
        id: 'hawaii',
        name: 'Hawaiian Hotspot',
        type: 'hotspot',
        location: [19.5, -155.5, 0],
        activity: 0.8,
        temperature: 1200,
        magmaProduction: 0.15,
        plumeSource: [19.5, -155.5, -2890]
      },
      {
        id: 'yellowstone',
        name: 'Yellowstone Hotspot',
        type: 'hotspot',
        location: [44.4, -110.6, 0],
        activity: 0.3,
        temperature: 1000,
        magmaProduction: 0.08,
        plumeSource: [44.4, -110.6, -2890]
      },
      {
        id: 'ring_of_fire',
        name: 'Ring of Fire',
        type: 'arc',
        location: [40, -120, 0],
        activity: 0.9,
        temperature: 1100,
        magmaProduction: 0.5
      }
    ];
    
    setSimulationState(prev => ({
      ...prev,
      plates: mockPlates,
      volcanicSystems: mockVolcanicSystems
    }));
    
    setIsLoading(false);
  };  // H
elper functions for generating mock data
  const generatePlateVertices = (plateId: string): number[][] => {
    const baseVertices: { [key: string]: number[][] } = {
      pacific: [
        [60, -180], [60, -120], [40, -120], [20, -140],
        [0, -160], [-20, -180], [-40, -160], [-60, -140],
        [-60, -180], [60, -180]
      ],
      north_american: [
        [70, -170], [70, -60], [50, -60], [30, -80],
        [25, -100], [30, -120], [50, -140], [70, -170]
      ],
      eurasian: [
        [80, -10], [80, 180], [60, 180], [40, 160],
        [50, 100], [60, 60], [70, 20], [80, -10]
      ]
    };
    
    return baseVertices[plateId] || [];
  };
  
  const generateElevationData = (count: number): number[] => {
    return Array.from({ length: count }, () => Math.random() * 8000 - 4000);
  };
  
  const generateThicknessData = (count: number): number[] => {
    return Array.from({ length: count }, () => 50 + Math.random() * 100);
  };
  
  // Animation controls
  const togglePlayback = () => {
    setSimulationState(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  };
  
  const resetSimulation = () => {
    setSimulationState(prev => ({
      ...prev,
      currentTime: prev.timeRange[0],
      isPlaying: false
    }));
  };
  
  const handleTimeChange = (newTime: number[]) => {
    setSimulationState(prev => ({ ...prev, currentTime: newTime[0] }));
  };
  
  const handleRegionSelect = (region: string) => {
    setSelectedRegion(region);
    // Implement region-specific camera positioning
  };
  
  // Time evolution effect
  useEffect(() => {
    if (!simulationState.isPlaying) return;
    
    const interval = setInterval(() => {
      setSimulationState(prev => {
        const newTime = prev.currentTime + (prev.playbackSpeed * 0.1);
        if (newTime >= prev.timeRange[1]) {
          return { ...prev, currentTime: prev.timeRange[1], isPlaying: false };
        }
        return { ...prev, currentTime: newTime };
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [simulationState.isPlaying, simulationState.playbackSpeed]);
  
  return (
    <div className="w-full h-full bg-gray-900 text-white">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Globe className="h-6 w-6 text-blue-500" />
          Global Tectonic Evolution Simulator
        </h1>
        <p className="text-gray-400 mt-1">
          Interactive 3D/4D visualization of Earth's tectonic history
        </p>
      </div>
      
      <div className="flex h-full">
        {/* Control Panel */}
        <div className="w-80 bg-gray-800 p-4 overflow-y-auto">
          <Tabs defaultValue="time" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="time">Time</TabsTrigger>
              <TabsTrigger value="view">View</TabsTrigger>
              <TabsTrigger value="data">Data</TabsTrigger>
            </TabsList>
            
            <TabsContent value="time" className="space-y-4">
              <Card className="bg-gray-700 border-gray-600">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Time Controls
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Current Time: {simulationState.currentTime.toFixed(1)} Ma</Label>
                    <Slider
                      value={[simulationState.currentTime]}
                      onValueChange={handleTimeChange}
                      min={simulationState.timeRange[0]}
                      max={simulationState.timeRange[1]}
                      step={0.1}
                      className="mt-2"
                    />
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      onClick={togglePlayback}
                      variant={simulationState.isPlaying ? "secondary" : "default"}
                      size="sm"
                    >
                      {simulationState.isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    </Button>
                    <Button onClick={resetSimulation} variant="outline" size="sm">
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  <div>
                    <Label>Playback Speed: {simulationState.playbackSpeed}x</Label>
                    <Slider
                      value={[simulationState.playbackSpeed]}
                      onValueChange={(value) => 
                        setSimulationState(prev => ({ ...prev, playbackSpeed: value[0] }))
                      }
                      min={0.1}
                      max={5.0}
                      step={0.1}
                      className="mt-2"
                    />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-700 border-gray-600">
                <CardHeader>
                  <CardTitle>Time Periods</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full justify-start"
                      onClick={() => handleTimeChange([-66])}
                    >
                      K-Pg Boundary (66 Ma)
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full justify-start"
                      onClick={() => handleTimeChange([-252])}
                    >
                      P-T Boundary (252 Ma)
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full justify-start"
                      onClick={() => handleTimeChange([-200])}
                    >
                      Pangaea (200 Ma)
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="view" className="space-y-4">
              <Card className="bg-gray-700 border-gray-600">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Eye className="h-4 w-4" />
                    Visualization Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Show Plates</Label>
                    <Switch
                      checked={visualizationSettings.showPlates}
                      onCheckedChange={(checked) =>
                        setVisualizationSettings(prev => ({ ...prev, showPlates: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Show Boundaries</Label>
                    <Switch
                      checked={visualizationSettings.showBoundaries}
                      onCheckedChange={(checked) =>
                        setVisualizationSettings(prev => ({ ...prev, showBoundaries: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Show Volcanoes</Label>
                    <Switch
                      checked={visualizationSettings.showVolcanoes}
                      onCheckedChange={(checked) =>
                        setVisualizationSettings(prev => ({ ...prev, showVolcanoes: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Show Plumes</Label>
                    <Switch
                      checked={visualizationSettings.showPlumes}
                      onCheckedChange={(checked) =>
                        setVisualizationSettings(prev => ({ ...prev, showPlumes: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Velocity Vectors</Label>
                    <Switch
                      checked={visualizationSettings.showVelocityVectors}
                      onCheckedChange={(checked) =>
                        setVisualizationSettings(prev => ({ ...prev, showVelocityVectors: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Age Colors</Label>
                    <Switch
                      checked={visualizationSettings.showAgeColors}
                      onCheckedChange={(checked) =>
                        setVisualizationSettings(prev => ({ ...prev, showAgeColors: checked }))
                      }
                    />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-700 border-gray-600">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MapPin className="h-4 w-4" />
                    Region Focus
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Select value={selectedRegion} onValueChange={handleRegionSelect}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="global">Global View</SelectItem>
                      <SelectItem value="pacific">Pacific Region</SelectItem>
                      <SelectItem value="atlantic">Atlantic Region</SelectItem>
                      <SelectItem value="mediterranean">Mediterranean</SelectItem>
                      <SelectItem value="ring_of_fire">Ring of Fire</SelectItem>
                      <SelectItem value="himalaya">Himalayan Region</SelectItem>
                      <SelectItem value="andes">Andes Mountains</SelectItem>
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="data" className="space-y-4">
              <Card className="bg-gray-700 border-gray-600">
                <CardHeader>
                  <CardTitle>Simulation Data</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-sm">
                    <div className="flex justify-between">
                      <span>Active Plates:</span>
                      <Badge variant="secondary">{simulationState.plates.length}</Badge>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span>Volcanic Systems:</span>
                      <Badge variant="secondary">{simulationState.volcanicSystems.length}</Badge>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span>Mantle Plumes:</span>
                      <Badge variant="secondary">{simulationState.mantlePlumes.length}</Badge>
                    </div>
                  </div>
                  
                  {isLoading && (
                    <div className="space-y-2">
                      <div className="text-sm">Loading simulation data...</div>
                      <Progress value={75} className="h-2" />
                    </div>
                  )}
                </CardContent>
              </Card>
              
              <Card className="bg-gray-700 border-gray-600">
                <CardHeader>
                  <CardTitle>Export Options</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button variant="outline" size="sm" className="w-full">
                    <Download className="h-4 w-4 mr-2" />
                    Export Animation
                  </Button>
                  <Button variant="outline" size="sm" className="w-full">
                    <Download className="h-4 w-4 mr-2" />
                    Export Data
                  </Button>
                  <Button variant="outline" size="sm" className="w-full">
                    <Download className="h-4 w-4 mr-2" />
                    Export Image
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* 3D Visualization */}
        <div className="flex-1 relative">
          <Canvas
            camera={{ position: [0, 0, 15], fov: 60 }}
            style={{ background: '#000011' }}
          >
            {/* Lighting */}
            <ambientLight intensity={0.3} />
            <directionalLight
              position={[10, 10, 10]}
              intensity={1}
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
            />
            <pointLight position={[-10, -10, -10]} intensity={0.5} />
            
            {/* Earth sphere */}
            <Sphere args={[6.371, 64, 64]} position={[0, 0, 0]}>
              <EarthShaderMaterial />
            </Sphere>
            
            {/* Tectonic plates */}
            {visualizationSettings.showPlates && simulationState.plates.map(plate => (
              <TectonicPlateComponent
                key={plate.id}
                plate={plate}
                time={simulationState.currentTime}
                showVelocity={visualizationSettings.showVelocityVectors}
                showAge={visualizationSettings.showAgeColors}
              />
            ))}
            
            {/* Volcanic systems */}
            {visualizationSettings.showVolcanoes && simulationState.volcanicSystems.map(system => (
              <VolcanicSystemComponent
                key={system.id}
                system={system}
                time={simulationState.currentTime}
                showPlumes={visualizationSettings.showPlumes}
              />
            ))}
            
            {/* Controls */}
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              minDistance={8}
              maxDistance={50}
            />
            
            {/* Post-processing effects */}
            <EffectComposer>
              <Bloom intensity={0.5} luminanceThreshold={0.9} />
              <DepthOfField focusDistance={0} focalLength={0.02} bokehScale={2} />
            </EffectComposer>
          </Canvas>
          
          {/* Overlay information */}
          <div className="absolute top-4 right-4 bg-black bg-opacity-75 p-3 rounded">
            <div className="text-sm space-y-1">
              <div>Time: {simulationState.currentTime.toFixed(1)} Ma</div>
              <div>View: {selectedRegion.replace('_', ' ').toUpperCase()}</div>
              <div>Quality: {visualizationSettings.qualityLevel.toUpperCase()}</div>
            </div>
          </div>
          
          {/* Legend */}
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 p-3 rounded">
            <div className="text-sm space-y-2">
              <div className="font-semibold">Legend</div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded"></div>
                <span>Oceanic Plates</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-600 rounded"></div>
                <span>Continental Plates</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded"></div>
                <span>Volcanic Systems</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 bg-orange-500"></div>
                <span>Mantle Plumes</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GlobalTectonicVisualization;