import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
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
  Alert,
  AlertDescription,
  Progress,
  Badge,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Slider,
  Switch,
  Input,
  Label
} from '@/components/ui';
import { 
  Activity, 
  AlertTriangle, 
  TrendingUp, 
  Thermometer, 
  Droplets,
  Zap,
  Mountain,
  Eye,
  Settings,
  Download,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';

interface MagmaProperties {
  type: 'basaltic' | 'andesitic' | 'dacitic' | 'rhyolitic';
  sio2Content: number;
  temperature: number;
  viscosity: number;
  gasContent: {
    h2o: number;
    co2: number;
    so2: number;
    h2s: number;
  };
  crystallinity: number;
  density: number;
}

interface IntrusionGeometry {
  type: 'sill' | 'dike' | 'laccolith' | 'stock' | 'batholith';
  depth: number;
  volume: number;
  dimensions: {
    length: number;
    width: number;
    thickness: number;
  };
  orientation: {
    strike: number;
    dip: number;
  };
}

interface DeformationData {
  timestamp: string;
  displacement: number[];
  velocity: number[];
  strain: number[];
  tilt: number[];
  coordinates: [number, number, number][];
}

interface EruptionForecast {
  probability: number;
  timeframe: string;
  eruptionStyle: 'effusive' | 'explosive' | 'mixed';
  vei: number;
  hazardZones: {
    pyroclasticFlow: number;
    ashfall: number;
    lahar: number;
    lavaFlow: number;
  };
}

interface AnalysisResult {
  magmaProperties: MagmaProperties;
  intrusionGeometry: IntrusionGeometry;
  deformationAnalysis: {
    totalVolume: number;
    pressureChange: number;
    sourceDepth: number;
    migrationRate: number;
  };
  eruptionForecast: EruptionForecast;
  confidence: number;
  recommendations: string[];
}

// 3D Magma Chamber Visualization Component
const MagmaChamberVisualization: React.FC<{
  analysisResult: AnalysisResult | null;
  deformationData: DeformationData | null;
  isAnimating: boolean;
}> = ({ analysisResult, deformationData, isAnimating }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [animationTime, setAnimationTime] = useState(0);

  useFrame((state, delta) => {
    if (isAnimating) {
      setAnimationTime(prev => prev + delta);
      
      if (meshRef.current) {
        // Animate magma chamber pulsation
        const pulsation = 1 + 0.1 * Math.sin(animationTime * 2);
        meshRef.current.scale.setScalar(pulsation);
        
        // Rotate slowly
        meshRef.current.rotation.y += delta * 0.1;
      }
    }
  });

  if (!analysisResult) {
    return (
      <Text
        position={[0, 0, 0]}
        fontSize={0.5}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        No Analysis Data
      </Text>
    );
  }

  const { intrusionGeometry, magmaProperties } = analysisResult;

  // Color based on magma type
  const getMagmaColor = (type: string) => {
    switch (type) {
      case 'basaltic': return '#ff4444';
      case 'andesitic': return '#ff8844';
      case 'dacitic': return '#ffaa44';
      case 'rhyolitic': return '#ffdd44';
      default: return '#ff6644';
    }
  };

  const magmaColor = getMagmaColor(magmaProperties.type);

  return (
    <group>
      {/* Main magma chamber */}
      <mesh ref={meshRef} position={[0, -intrusionGeometry.depth, 0]}>
        <sphereGeometry args={[
          intrusionGeometry.volume / 1000,
          32,
          32
        ]} />
        <meshPhongMaterial
          color={magmaColor}
          transparent
          opacity={0.8}
          emissive={magmaColor}
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* Intrusion conduits */}
      {intrusionGeometry.type === 'dike' && (
        <mesh position={[0, -intrusionGeometry.depth / 2, 0]}>
          <boxGeometry args={[
            intrusionGeometry.dimensions.thickness,
            intrusionGeometry.depth,
            intrusionGeometry.dimensions.width
          ]} />
          <meshPhongMaterial
            color={magmaColor}
            transparent
            opacity={0.6}
          />
        </mesh>
      )}

      {intrusionGeometry.type === 'sill' && (
        <mesh position={[0, -intrusionGeometry.depth, 0]}>
          <boxGeometry args={[
            intrusionGeometry.dimensions.length,
            intrusionGeometry.dimensions.thickness,
            intrusionGeometry.dimensions.width
          ]} />
          <meshPhongMaterial
            color={magmaColor}
            transparent
            opacity={0.6}
          />
        </mesh>
      )}

      {/* Deformation field visualization */}
      {deformationData && deformationData.coordinates.map((coord, index) => (
        <mesh key={index} position={coord}>
          <coneGeometry args={[0.1, deformationData.displacement[index] * 100, 8]} />
          <meshPhongMaterial
            color={deformationData.displacement[index] > 0 ? '#00ff00' : '#ff0000'}
            transparent
            opacity={0.7}
          />
        </mesh>
      ))}

      {/* Ground surface */}
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshPhongMaterial
          color="#8B4513"
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* Coordinate system */}
      <axesHelper args={[5]} />

      {/* Labels */}
      <Text
        position={[0, -intrusionGeometry.depth - 2, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {`${magmaProperties.type} magma`}
      </Text>

      <Text
        position={[0, -intrusionGeometry.depth - 2.5, 0]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {`Depth: ${intrusionGeometry.depth.toFixed(1)} km`}
      </Text>

      <Text
        position={[0, -intrusionGeometry.depth - 3, 0]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {`Volume: ${(intrusionGeometry.volume / 1e6).toFixed(2)} km³`}
      </Text>
    </group>
  );
};

// Main Magmatic Intrusion Analyzer Component
const MagmaticIntrusionAnalyzer: React.FC = () => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [deformationData, setDeformationData] = useState<DeformationData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAnimating, setIsAnimating] = useState(true);
  const [selectedVolcano, setSelectedVolcano] = useState('yellowstone');
  const [analysisParameters, setAnalysisParameters] = useState({
    temperatureRange: [800, 1200],
    pressureRange: [50, 500],
    gasContentThreshold: 0.05,
    viscosityModel: 'giordano2008',
    deformationModel: 'mogi',
    timeWindow: 30
  });

  const volcanoes = [
    { id: 'yellowstone', name: 'Yellowstone Caldera', type: 'caldera' },
    { id: 'long_valley', name: 'Long Valley Caldera', type: 'caldera' },
    { id: 'kilauea', name: 'Kilauea Volcano', type: 'shield' },
    { id: 'mount_st_helens', name: 'Mount St. Helens', type: 'stratovolcano' },
    { id: 'vesuvius', name: 'Mount Vesuvius', type: 'stratovolcano' },
    { id: 'etna', name: 'Mount Etna', type: 'stratovolcano' }
  ];

  // Simulate analysis process
  const runAnalysis = async () => {
    setIsAnalyzing(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Generate realistic analysis results
    const mockResult: AnalysisResult = {
      magmaProperties: {
        type: selectedVolcano === 'kilauea' ? 'basaltic' : 
              selectedVolcano === 'yellowstone' ? 'rhyolitic' : 'andesitic',
        sio2Content: selectedVolcano === 'kilauea' ? 48 : 
                     selectedVolcano === 'yellowstone' ? 72 : 58,
        temperature: selectedVolcano === 'kilauea' ? 1150 : 
                     selectedVolcano === 'yellowstone' ? 750 : 950,
        viscosity: selectedVolcano === 'kilauea' ? 1e3 : 
                   selectedVolcano === 'yellowstone' ? 1e11 : 1e6,
        gasContent: {
          h2o: selectedVolcano === 'kilauea' ? 0.5 : 4.0,
          co2: selectedVolcano === 'kilauea' ? 0.1 : 0.05,
          so2: 0.02,
          h2s: 0.001
        },
        crystallinity: selectedVolcano === 'kilauea' ? 0.1 : 0.3,
        density: selectedVolcano === 'kilauea' ? 2800 : 2400
      },
      intrusionGeometry: {
        type: selectedVolcano === 'yellowstone' ? 'sill' : 'dike',
        depth: selectedVolcano === 'yellowstone' ? 8 : 
               selectedVolcano === 'kilauea' ? 3 : 6,
        volume: selectedVolcano === 'yellowstone' ? 5e8 : 1e7,
        dimensions: {
          length: selectedVolcano === 'yellowstone' ? 15000 : 5000,
          width: selectedVolcano === 'yellowstone' ? 10000 : 2000,
          thickness: selectedVolcano === 'yellowstone' ? 500 : 100
        },
        orientation: {
          strike: 45,
          dip: selectedVolcano === 'yellowstone' ? 10 : 85
        }
      },
      deformationAnalysis: {
        totalVolume: selectedVolcano === 'yellowstone' ? 2e7 : 5e6,
        pressureChange: selectedVolcano === 'yellowstone' ? 50 : 20,
        sourceDepth: selectedVolcano === 'yellowstone' ? 8 : 4,
        migrationRate: selectedVolcano === 'kilauea' ? 0.5 : 0.1
      },
      eruptionForecast: {
        probability: selectedVolcano === 'kilauea' ? 0.8 : 
                     selectedVolcano === 'yellowstone' ? 0.1 : 0.3,
        timeframe: selectedVolcano === 'kilauea' ? 'days-weeks' : 
                   selectedVolcano === 'yellowstone' ? 'decades-centuries' : 'months-years',
        eruptionStyle: selectedVolcano === 'kilauea' ? 'effusive' : 
                       selectedVolcano === 'yellowstone' ? 'explosive' : 'mixed',
        vei: selectedVolcano === 'kilauea' ? 2 : 
             selectedVolcano === 'yellowstone' ? 8 : 4,
        hazardZones: {
          pyroclasticFlow: selectedVolcano === 'yellowstone' ? 100 : 10,
          ashfall: selectedVolcano === 'yellowstone' ? 1000 : 50,
          lahar: 20,
          lavaFlow: selectedVolcano === 'kilauea' ? 30 : 5
        }
      },
      confidence: 0.85,
      recommendations: [
        'Increase monitoring frequency to daily',
        'Deploy additional GPS stations',
        'Monitor gas emissions continuously',
        'Prepare evacuation plans',
        'Coordinate with emergency services'
      ]
    };

    // Generate mock deformation data
    const mockDeformation: DeformationData = {
      timestamp: new Date().toISOString(),
      displacement: Array.from({ length: 20 }, () => Math.random() * 0.1 - 0.05),
      velocity: Array.from({ length: 20 }, () => Math.random() * 0.02 - 0.01),
      strain: Array.from({ length: 20 }, () => Math.random() * 1e-6),
      tilt: Array.from({ length: 20 }, () => Math.random() * 1e-6),
      coordinates: Array.from({ length: 20 }, (_, i) => [
        (Math.random() - 0.5) * 10,
        Math.random() * 2,
        (Math.random() - 0.5) * 10
      ] as [number, number, number])
    };

    setAnalysisResult(mockResult);
    setDeformationData(mockDeformation);
    setIsAnalyzing(false);
  };

  const getAlertLevel = (probability: number) => {
    if (probability > 0.7) return { level: 'high', color: 'destructive' };
    if (probability > 0.4) return { level: 'medium', color: 'warning' };
    return { level: 'low', color: 'default' };
  };

  const formatViscosity = (viscosity: number) => {
    const log = Math.log10(viscosity);
    return `10^${log.toFixed(1)} Pa·s`;
  };

  return (
    <div className="w-full h-full p-6 bg-gray-900 text-white">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Mountain className="h-8 w-8 text-orange-500" />
          Magmatic Intrusion Analyzer
        </h1>
        <p className="text-gray-400">
          Advanced analysis of magmatic intrusions with real-time deformation monitoring
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        {/* Control Panel */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Analysis Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Target Volcano</Label>
                <Select value={selectedVolcano} onValueChange={setSelectedVolcano}>
                  <SelectTrigger className="bg-gray-700 border-gray-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    {volcanoes.map(volcano => (
                      <SelectItem key={volcano.id} value={volcano.id}>
                        {volcano.name} ({volcano.type})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Temperature Range (°C)</Label>
                <Slider
                  value={analysisParameters.temperatureRange}
                  onValueChange={(value) => 
                    setAnalysisParameters(prev => ({ ...prev, temperatureRange: value }))
                  }
                  min={600}
                  max={1300}
                  step={10}
                  className="mt-2"
                />
                <div className="text-sm text-gray-400 mt-1">
                  {analysisParameters.temperatureRange[0]}°C - {analysisParameters.temperatureRange[1]}°C
                </div>
              </div>

              <div>
                <Label>Pressure Range (MPa)</Label>
                <Slider
                  value={analysisParameters.pressureRange}
                  onValueChange={(value) => 
                    setAnalysisParameters(prev => ({ ...prev, pressureRange: value }))
                  }
                  min={10}
                  max={1000}
                  step={10}
                  className="mt-2"
                />
                <div className="text-sm text-gray-400 mt-1">
                  {analysisParameters.pressureRange[0]} - {analysisParameters.pressureRange[1]} MPa
                </div>
              </div>

              <div>
                <Label>Viscosity Model</Label>
                <Select 
                  value={analysisParameters.viscosityModel} 
                  onValueChange={(value) => 
                    setAnalysisParameters(prev => ({ ...prev, viscosityModel: value }))
                  }
                >
                  <SelectTrigger className="bg-gray-700 border-gray-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    <SelectItem value="giordano2008">Giordano et al. (2008)</SelectItem>
                    <SelectItem value="shaw1972">Shaw (1972)</SelectItem>
                    <SelectItem value="bottinga1995">Bottinga & Richet (1995)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Deformation Model</Label>
                <Select 
                  value={analysisParameters.deformationModel} 
                  onValueChange={(value) => 
                    setAnalysisParameters(prev => ({ ...prev, deformationModel: value }))
                  }
                >
                  <SelectTrigger className="bg-gray-700 border-gray-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    <SelectItem value="mogi">Mogi Point Source</SelectItem>
                    <SelectItem value="yang">Yang et al. (1988)</SelectItem>
                    <SelectItem value="mctigue">McTigue (1987)</SelectItem>
                    <SelectItem value="fialko">Fialko et al. (2001)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button 
                onClick={runAnalysis} 
                disabled={isAnalyzing}
                className="w-full bg-orange-600 hover:bg-orange-700"
              >
                {isAnalyzing ? (
                  <>
                    <Activity className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run Analysis
                  </>
                )}
              </Button>

              <div className="flex items-center justify-between">
                <Label>Animation</Label>
                <Switch
                  checked={isAnimating}
                  onCheckedChange={setIsAnimating}
                />
              </div>
            </CardContent>
          </Card>

          {/* Analysis Progress */}
          {isAnalyzing && (
            <Card className="bg-gray-800 border-gray-700">
              <CardContent className="pt-6">
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span>Processing InSAR data...</span>
                    <span>45%</span>
                  </div>
                  <Progress value={45} className="h-2" />
                  
                  <div className="flex justify-between text-sm">
                    <span>Analyzing magma properties...</span>
                    <span>78%</span>
                  </div>
                  <Progress value={78} className="h-2" />
                  
                  <div className="flex justify-between text-sm">
                    <span>Modeling deformation...</span>
                    <span>92%</span>
                  </div>
                  <Progress value={92} className="h-2" />
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 3D Visualization */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="bg-gray-800 border-gray-700 h-96">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Eye className="h-5 w-5" />
                  3D Magma Chamber Visualization
                </span>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setIsAnimating(!isAnimating)}
                  >
                    {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </Button>
                  <Button size="sm" variant="outline">
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="h-full p-0">
              <Canvas
                camera={{ position: [10, 10, 10], fov: 60 }}
                style={{ background: '#1a1a1a' }}
              >
                <ambientLight intensity={0.4} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <pointLight position={[-10, -10, -10]} intensity={0.5} />
                
                <MagmaChamberVisualization
                  analysisResult={analysisResult}
                  deformationData={deformationData}
                  isAnimating={isAnimating}
                />
                
                <OrbitControls
                  enablePan={true}
                  enableZoom={true}
                  enableRotate={true}
                />
              </Canvas>
            </CardContent>
          </Card>

          {/* Analysis Results */}
          {analysisResult && (
            <Tabs defaultValue="properties" className="w-full">
              <TabsList className="grid w-full grid-cols-4 bg-gray-800">
                <TabsTrigger value="properties">Magma Properties</TabsTrigger>
                <TabsTrigger value="geometry">Intrusion Geometry</TabsTrigger>
                <TabsTrigger value="deformation">Deformation</TabsTrigger>
                <TabsTrigger value="forecast">Eruption Forecast</TabsTrigger>
              </TabsList>

              <TabsContent value="properties">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Thermometer className="h-5 w-5 text-red-500" />
                      Magma Properties Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm text-gray-400">Magma Type</Label>
                          <div className="text-lg font-semibold capitalize">
                            {analysisResult.magmaProperties.type}
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">SiO₂ Content</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.magmaProperties.sio2Content.toFixed(1)}%
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Temperature</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.magmaProperties.temperature}°C
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Viscosity</Label>
                          <div className="text-lg font-semibold">
                            {formatViscosity(analysisResult.magmaProperties.viscosity)}
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm text-gray-400">Gas Content</Label>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>H₂O:</span>
                              <span>{analysisResult.magmaProperties.gasContent.h2o.toFixed(2)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>CO₂:</span>
                              <span>{analysisResult.magmaProperties.gasContent.co2.toFixed(3)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>SO₂:</span>
                              <span>{analysisResult.magmaProperties.gasContent.so2.toFixed(3)}%</span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Crystallinity</Label>
                          <div className="text-lg font-semibold">
                            {(analysisResult.magmaProperties.crystallinity * 100).toFixed(1)}%
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Density</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.magmaProperties.density} kg/m³
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="geometry">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Mountain className="h-5 w-5 text-orange-500" />
                      Intrusion Geometry
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm text-gray-400">Intrusion Type</Label>
                          <div className="text-lg font-semibold capitalize">
                            {analysisResult.intrusionGeometry.type}
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Depth</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.intrusionGeometry.depth.toFixed(1)} km
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Volume</Label>
                          <div className="text-lg font-semibold">
                            {(analysisResult.intrusionGeometry.volume / 1e6).toFixed(2)} km³
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm text-gray-400">Dimensions</Label>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>Length:</span>
                              <span>{(analysisResult.intrusionGeometry.dimensions.length / 1000).toFixed(1)} km</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Width:</span>
                              <span>{(analysisResult.intrusionGeometry.dimensions.width / 1000).toFixed(1)} km</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Thickness:</span>
                              <span>{analysisResult.intrusionGeometry.dimensions.thickness} m</span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Orientation</Label>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>Strike:</span>
                              <span>{analysisResult.intrusionGeometry.orientation.strike}°</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Dip:</span>
                              <span>{analysisResult.intrusionGeometry.orientation.dip}°</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="deformation">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-blue-500" />
                      Deformation Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm text-gray-400">Volume Change</Label>
                          <div className="text-lg font-semibold">
                            {(analysisResult.deformationAnalysis.totalVolume / 1e6).toFixed(2)} km³
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Pressure Change</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.deformationAnalysis.pressureChange.toFixed(1)} MPa
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm text-gray-400">Source Depth</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.deformationAnalysis.sourceDepth.toFixed(1)} km
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm text-gray-400">Migration Rate</Label>
                          <div className="text-lg font-semibold">
                            {analysisResult.deformationAnalysis.migrationRate.toFixed(2)} km/day
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="forecast">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-yellow-500" />
                      Eruption Forecast
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <Alert className={`border-${getAlertLevel(analysisResult.eruptionForecast.probability).color}`}>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          <div className="flex items-center justify-between">
                            <span>Eruption Probability</span>
                            <Badge variant={getAlertLevel(analysisResult.eruptionForecast.probability).color as any}>
                              {(analysisResult.eruptionForecast.probability * 100).toFixed(1)}%
                            </Badge>
                          </div>
                        </AlertDescription>
                      </Alert>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <div>
                            <Label className="text-sm text-gray-400">Timeframe</Label>
                            <div className="text-lg font-semibold capitalize">
                              {analysisResult.eruptionForecast.timeframe}
                            </div>
                          </div>
                          
                          <div>
                            <Label className="text-sm text-gray-400">Eruption Style</Label>
                            <div className="text-lg font-semibold capitalize">
                              {analysisResult.eruptionForecast.eruptionStyle}
                            </div>
                          </div>
                          
                          <div>
                            <Label className="text-sm text-gray-400">VEI (Volcanic Explosivity Index)</Label>
                            <div className="text-lg font-semibold">
                              {analysisResult.eruptionForecast.vei}
                            </div>
                          </div>
                        </div>
                        
                        <div className="space-y-3">
                          <div>
                            <Label className="text-sm text-gray-400">Hazard Zones (km)</Label>
                            <div className="space-y-1">
                              <div className="flex justify-between">
                                <span>Pyroclastic Flow:</span>
                                <span>{analysisResult.eruptionForecast.hazardZones.pyroclasticFlow}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Ashfall:</span>
                                <span>{analysisResult.eruptionForecast.hazardZones.ashfall}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Lahar:</span>
                                <span>{analysisResult.eruptionForecast.hazardZones.lahar}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Lava Flow:</span>
                                <span>{analysisResult.eruptionForecast.hazardZones.lavaFlow}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <Label className="text-sm text-gray-400">Recommendations</Label>
                        <ul className="mt-2 space-y-1">
                          {analysisResult.recommendations.map((rec, index) => (
                            <li key={index} className="text-sm flex items-start gap-2">
                              <span className="text-orange-500 mt-1">•</span>
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  );
};

export default MagmaticIntrusionAnalyzer;