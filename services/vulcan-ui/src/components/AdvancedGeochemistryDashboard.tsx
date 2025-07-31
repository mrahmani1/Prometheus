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
  RotateCcw,
  BarChart3,
  Scatter3D,
  PieChart,
  LineChart
} from 'lucide-react';
import Plot from 'react-plotly.js';

interface GeochemicalSample {
  id: string;
  name: string;
  location: [number, number, number];
  majorElements: { [key: string]: number };
  traceElements: { [key: string]: number };
  rareEarthElements: { [key: string]: number };
  isotopeRatios: { [key: string]: number };
  rockType: string;
  tectonicSetting: string;
  age: number;
  uncertainty: { [key: string]: number };
}

interface GeochemicalAnalysis {
  classification: {
    rockType: string;
    confidence: number;
    method: string;
  };
  tectonicDiscrimination: {
    setting: string;
    confidence: number;
    indices: { [key: string]: number };
  };
  petrogenesis: {
    processes: string[];
    temperature: number;
    pressure: number;
    parentMagma: string;
  };
  alteration: {
    degree: number;
    type: string;
    minerals: string[];
  };
}

const AdvancedGeochemistryDashboard: React.FC = () => {
  const [samples, setSamples] = useState<GeochemicalSample[]>([]);
  const [selectedSamples, setSelectedSamples] = useState<string[]>([]);
  const [analysisResults, setAnalysisResults] = useState<{ [key: string]: GeochemicalAnalysis }>({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [visualizationType, setVisualizationType] = useState('harker');
  const [normalization, setNormalization] = useState('chondrite');
  const [showUncertainty, setShowUncertainty] = useState(true);
  const [colorBy, setColorBy] = useState('rockType');

  // Generate mock geochemical data
  useEffect(() => {
    const mockSamples: GeochemicalSample[] = Array.from({ length: 50 }, (_, i) => ({
      id: `sample_${i + 1}`,
      name: `Sample ${i + 1}`,
      location: [
        Math.random() * 180 - 90,  // latitude
        Math.random() * 360 - 180, // longitude
        Math.random() * 5000       // elevation
      ],
      majorElements: {
        SiO2: 45 + Math.random() * 30,
        Al2O3: 10 + Math.random() * 10,
        Fe2O3: 5 + Math.random() * 15,
        MgO: Math.random() * 20,
        CaO: Math.random() * 15,
        Na2O: Math.random() * 8,
        K2O: Math.random() * 6,
        TiO2: Math.random() * 3,
        P2O5: Math.random() * 2,
        MnO: Math.random() * 0.5
      },
      traceElements: {
        Rb: Math.random() * 200,
        Sr: Math.random() * 1000,
        Y: Math.random() * 100,
        Zr: Math.random() * 500,
        Nb: Math.random() * 50,
        Ba: Math.random() * 2000,
        Th: Math.random() * 20,
        U: Math.random() * 10,
        Pb: Math.random() * 50,
        Cr: Math.random() * 500,
        Ni: Math.random() * 300,
        V: Math.random() * 400
      },
      rareEarthElements: {
        La: Math.random() * 100,
        Ce: Math.random() * 200,
        Pr: Math.random() * 25,
        Nd: Math.random() * 100,
        Sm: Math.random() * 20,
        Eu: Math.random() * 5,
        Gd: Math.random() * 20,
        Tb: Math.random() * 3,
        Dy: Math.random() * 15,
        Ho: Math.random() * 3,
        Er: Math.random() * 8,
        Tm: Math.random() * 1,
        Yb: Math.random() * 8,
        Lu: Math.random() * 1
      },
      isotopeRatios: {
        '87Sr/86Sr': 0.703 + Math.random() * 0.02,
        '143Nd/144Nd': 0.5125 + Math.random() * 0.001,
        '206Pb/204Pb': 17 + Math.random() * 5,
        'δ18O': -10 + Math.random() * 20
      },
      rockType: ['basalt', 'andesite', 'dacite', 'rhyolite', 'granite'][Math.floor(Math.random() * 5)],
      tectonicSetting: ['MORB', 'OIB', 'VAB', 'WPB'][Math.floor(Math.random() * 4)],
      age: Math.random() * 4000,
      uncertainty: {
        SiO2: 0.5,
        Al2O3: 0.3,
        Fe2O3: 0.4
      }
    }));
    
    setSamples(mockSamples);
    setSelectedSamples(mockSamples.slice(0, 10).map(s => s.id));
  }, []);

  const runGeochemicalAnalysis = async () => {
    setIsAnalyzing(true);
    
    // Simulate analysis delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const results: { [key: string]: GeochemicalAnalysis } = {};
    
    selectedSamples.forEach(sampleId => {
      const sample = samples.find(s => s.id === sampleId);
      if (sample) {
        results[sampleId] = {
          classification: {
            rockType: sample.rockType,
            confidence: 0.85 + Math.random() * 0.1,
            method: 'TAS + ML'
          },
          tectonicDiscrimination: {
            setting: sample.tectonicSetting,
            confidence: 0.75 + Math.random() * 0.2,
            indices: {
              'Ti/Y': sample.traceElements.Y > 0 ? 8000 / sample.traceElements.Y : 0,
              'Nb/Y': sample.traceElements.Y > 0 ? sample.traceElements.Nb / sample.traceElements.Y : 0,
              'Th/Yb': 2 + Math.random() * 8
            }
          },
          petrogenesis: {
            processes: ['fractional_crystallization', 'crustal_assimilation'],
            temperature: 800 + Math.random() * 400,
            pressure: 1 + Math.random() * 10,
            parentMagma: 'basaltic'
          },
          alteration: {
            degree: Math.random() * 100,
            type: 'hydrothermal',
            minerals: ['chlorite', 'epidote', 'calcite']
          }
        };
      }
    });
    
    setAnalysisResults(results);
    setIsAnalyzing(false);
  };  const 
generateHarkerDiagram = () => {
    const selectedSampleData = samples.filter(s => selectedSamples.includes(s.id));
    
    const traces = selectedSampleData.map(sample => ({
      x: [sample.majorElements.SiO2],
      y: [sample.majorElements.Al2O3],
      mode: 'markers',
      type: 'scatter',
      name: sample.name,
      marker: {
        size: 8,
        color: getColorForSample(sample),
        opacity: 0.8
      },
      text: [`${sample.name}<br>Rock Type: ${sample.rockType}<br>SiO2: ${sample.majorElements.SiO2.toFixed(2)}%<br>Al2O3: ${sample.majorElements.Al2O3.toFixed(2)}%`],
      hovertemplate: '%{text}<extra></extra>'
    }));

    return {
      data: traces,
      layout: {
        title: 'Harker Diagram (SiO2 vs Al2O3)',
        xaxis: { title: 'SiO2 (wt%)' },
        yaxis: { title: 'Al2O3 (wt%)' },
        showlegend: false,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: 'white' }
      }
    };
  };

  const generateREEDiagram = () => {
    const selectedSampleData = samples.filter(s => selectedSamples.includes(s.id));
    const chondriteValues = {
      La: 0.237, Ce: 0.613, Pr: 0.095, Nd: 0.467, Sm: 0.153,
      Eu: 0.058, Gd: 0.204, Tb: 0.037, Dy: 0.246, Ho: 0.0546,
      Er: 0.160, Tm: 0.0247, Yb: 0.161, Lu: 0.0246
    };
    
    const rareEarthOrder = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'];
    
    const traces = selectedSampleData.slice(0, 10).map(sample => {
      const normalizedREE = rareEarthOrder.map(element => 
        sample.rareEarthElements[element] / chondriteValues[element]
      );
      
      return {
        x: rareEarthOrder,
        y: normalizedREE,
        mode: 'lines+markers',
        type: 'scatter',
        name: sample.name,
        line: { width: 2 },
        marker: { size: 6 }
      };
    });

    return {
      data: traces,
      layout: {
        title: 'REE Spider Diagram (Chondrite Normalized)',
        xaxis: { title: 'Rare Earth Elements' },
        yaxis: { 
          title: 'Sample/Chondrite',
          type: 'log'
        },
        showlegend: true,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: 'white' }
      }
    };
  };

  const generateTASDiagram = () => {
    const selectedSampleData = samples.filter(s => selectedSamples.includes(s.id));
    
    const trace = {
      x: selectedSampleData.map(s => s.majorElements.SiO2),
      y: selectedSampleData.map(s => s.majorElements.Na2O + s.majorElements.K2O),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 10,
        color: selectedSampleData.map(s => getColorForSample(s)),
        opacity: 0.8
      },
      text: selectedSampleData.map(s => `${s.name}<br>Rock Type: ${s.rockType}`),
      hovertemplate: '%{text}<extra></extra>'
    };

    // TAS classification boundaries
    const boundaries = [
      // Simplified TAS boundaries
      { x: [41, 45, 48.4, 52.5, 57.6, 63, 77.3], y: [0, 0, 0, 0, 0, 0, 0], name: 'Silica axis' },
      { x: [41, 41], y: [0, 7], name: 'Left boundary' },
      { x: [45, 49.4], y: [5, 7.3], name: 'Basalt-Andesite' }
    ];

    return {
      data: [trace, ...boundaries.map(b => ({
        x: b.x,
        y: b.y,
        mode: 'lines',
        type: 'scatter',
        name: b.name,
        line: { color: 'rgba(255,255,255,0.3)', width: 1 },
        showlegend: false
      }))],
      layout: {
        title: 'TAS (Total Alkali-Silica) Diagram',
        xaxis: { title: 'SiO2 (wt%)', range: [40, 80] },
        yaxis: { title: 'Na2O + K2O (wt%)', range: [0, 16] },
        showlegend: false,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: 'white' }
      }
    };
  };

  const generateTectonicDiscrimination = () => {
    const selectedSampleData = samples.filter(s => selectedSamples.includes(s.id));
    
    const trace = {
      x: selectedSampleData.map(s => s.traceElements.Y > 0 ? s.traceElements.Nb / s.traceElements.Y : 0),
      y: selectedSampleData.map(s => s.traceElements.Y > 0 ? 8000 / s.traceElements.Y : 0),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 10,
        color: selectedSampleData.map(s => getTectonicColor(s.tectonicSetting)),
        opacity: 0.8
      },
      text: selectedSampleData.map(s => `${s.name}<br>Setting: ${s.tectonicSetting}`),
      hovertemplate: '%{text}<extra></extra>'
    };

    return {
      data: [trace],
      layout: {
        title: 'Tectonic Discrimination (Nb/Y vs Ti/Y)',
        xaxis: { title: 'Nb/Y', type: 'log' },
        yaxis: { title: 'Ti/Y', type: 'log' },
        showlegend: false,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: 'white' }
      }
    };
  };

  const getColorForSample = (sample: GeochemicalSample) => {
    const colorMap = {
      basalt: '#FF6B6B',
      andesite: '#4ECDC4',
      dacite: '#45B7D1',
      rhyolite: '#96CEB4',
      granite: '#FFEAA7'
    };
    return colorMap[sample.rockType] || '#DDA0DD';
  };

  const getTectonicColor = (setting: string) => {
    const colorMap = {
      MORB: '#FF6B6B',
      OIB: '#4ECDC4',
      VAB: '#45B7D1',
      WPB: '#96CEB4'
    };
    return colorMap[setting] || '#DDA0DD';
  };

  const renderVisualization = () => {
    switch (visualizationType) {
      case 'harker':
        return <Plot {...generateHarkerDiagram()} style={{ width: '100%', height: '400px' }} />;
      case 'ree':
        return <Plot {...generateREEDiagram()} style={{ width: '100%', height: '400px' }} />;
      case 'tas':
        return <Plot {...generateTASDiagram()} style={{ width: '100%', height: '400px' }} />;
      case 'tectonic':
        return <Plot {...generateTectonicDiscrimination()} style={{ width: '100%', height: '400px' }} />;
      default:
        return <div>Select a visualization type</div>;
    }
  };

  return (
    <div className="w-full h-full p-6 bg-gray-900 text-white">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <BarChart3 className="h-8 w-8 text-blue-500" />
          Advanced Geochemistry Dashboard
        </h1>
        <p className="text-gray-400">
          Comprehensive geochemical analysis and visualization platform
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full">
        {/* Control Panel */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Analysis Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Samples Selected: {selectedSamples.length}</Label>
                <div className="text-sm text-gray-400 mt-1">
                  Total available: {samples.length}
                </div>
              </div>

              <div>
                <Label>Visualization Type</Label>
                <Select value={visualizationType} onValueChange={setVisualizationType}>
                  <SelectTrigger className="bg-gray-700 border-gray-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    <SelectItem value="harker">Harker Diagrams</SelectItem>
                    <SelectItem value="ree">REE Spider Diagram</SelectItem>
                    <SelectItem value="tas">TAS Classification</SelectItem>
                    <SelectItem value="tectonic">Tectonic Discrimination</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Normalization</Label>
                <Select value={normalization} onValueChange={setNormalization}>
                  <SelectTrigger className="bg-gray-700 border-gray-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-700 border-gray-600">
                    <SelectItem value="chondrite">Chondrite</SelectItem>
                    <SelectItem value="primitive_mantle">Primitive Mantle</SelectItem>
                    <SelectItem value="morb">N-MORB</SelectItem>
                    <SelectItem value="oib">OIB</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <Label>Show Uncertainty</Label>
                <Switch
                  checked={showUncertainty}
                  onCheckedChange={setShowUncertainty}
                />
              </div>

              <Button 
                onClick={runGeochemicalAnalysis} 
                disabled={isAnalyzing || selectedSamples.length === 0}
                className="w-full bg-blue-600 hover:bg-blue-700"
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
            </CardContent>
          </Card>

          {/* Sample Statistics */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle>Sample Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Basalt:</span>
                  <span>{samples.filter(s => s.rockType === 'basalt').length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Andesite:</span>
                  <span>{samples.filter(s => s.rockType === 'andesite').length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Dacite:</span>
                  <span>{samples.filter(s => s.rockType === 'dacite').length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Rhyolite:</span>
                  <span>{samples.filter(s => s.rockType === 'rhyolite').length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Granite:</span>
                  <span>{samples.filter(s => s.rockType === 'granite').length}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Visualization Area */}
        <div className="lg:col-span-3 space-y-4">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Scatter3D className="h-5 w-5" />
                  Geochemical Visualization
                </span>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="h-96">
              {renderVisualization()}
            </CardContent>
          </Card>

          {/* Analysis Results */}
          {Object.keys(analysisResults).length > 0 && (
            <Tabs defaultValue="classification" className="w-full">
              <TabsList className="grid w-full grid-cols-4 bg-gray-800">
                <TabsTrigger value="classification">Classification</TabsTrigger>
                <TabsTrigger value="tectonic">Tectonic Setting</TabsTrigger>
                <TabsTrigger value="petrogenesis">Petrogenesis</TabsTrigger>
                <TabsTrigger value="alteration">Alteration</TabsTrigger>
              </TabsList>

              <TabsContent value="classification">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Rock Classification Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(analysisResults).slice(0, 6).map(([sampleId, analysis]) => {
                        const sample = samples.find(s => s.id === sampleId);
                        return (
                          <div key={sampleId} className="p-4 bg-gray-700 rounded-lg">
                            <h4 className="font-semibold mb-2">{sample?.name}</h4>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span>Rock Type:</span>
                                <Badge variant="secondary">{analysis.classification.rockType}</Badge>
                              </div>
                              <div className="flex justify-between">
                                <span>Confidence:</span>
                                <span>{(analysis.classification.confidence * 100).toFixed(1)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Method:</span>
                                <span>{analysis.classification.method}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="tectonic">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Tectonic Setting Discrimination</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(analysisResults).slice(0, 6).map(([sampleId, analysis]) => {
                        const sample = samples.find(s => s.id === sampleId);
                        return (
                          <div key={sampleId} className="p-4 bg-gray-700 rounded-lg">
                            <h4 className="font-semibold mb-2">{sample?.name}</h4>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span>Setting:</span>
                                <Badge variant="secondary">{analysis.tectonicDiscrimination.setting}</Badge>
                              </div>
                              <div className="flex justify-between">
                                <span>Confidence:</span>
                                <span>{(analysis.tectonicDiscrimination.confidence * 100).toFixed(1)}%</span>
                              </div>
                              <div className="space-y-1">
                                <div className="text-xs text-gray-400">Discrimination Indices:</div>
                                {Object.entries(analysis.tectonicDiscrimination.indices).map(([key, value]) => (
                                  <div key={key} className="flex justify-between text-xs">
                                    <span>{key}:</span>
                                    <span>{value.toFixed(2)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="petrogenesis">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Petrogenetic Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(analysisResults).slice(0, 6).map(([sampleId, analysis]) => {
                        const sample = samples.find(s => s.id === sampleId);
                        return (
                          <div key={sampleId} className="p-4 bg-gray-700 rounded-lg">
                            <h4 className="font-semibold mb-2">{sample?.name}</h4>
                            <div className="space-y-1 text-sm">
                              <div>
                                <span className="text-gray-400">Processes:</span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {analysis.petrogenesis.processes.map(process => (
                                    <Badge key={process} variant="outline" className="text-xs">
                                      {process.replace('_', ' ')}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                              <div className="flex justify-between">
                                <span>Temperature:</span>
                                <span>{analysis.petrogenesis.temperature.toFixed(0)}°C</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Pressure:</span>
                                <span>{analysis.petrogenesis.pressure.toFixed(1)} kbar</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Parent Magma:</span>
                                <span>{analysis.petrogenesis.parentMagma}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="alteration">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Alteration Assessment</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(analysisResults).slice(0, 6).map(([sampleId, analysis]) => {
                        const sample = samples.find(s => s.id === sampleId);
                        return (
                          <div key={sampleId} className="p-4 bg-gray-700 rounded-lg">
                            <h4 className="font-semibold mb-2">{sample?.name}</h4>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span>Degree:</span>
                                <span>{analysis.alteration.degree.toFixed(1)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Type:</span>
                                <Badge variant="secondary">{analysis.alteration.type}</Badge>
                              </div>
                              <div>
                                <span className="text-gray-400">Alteration Minerals:</span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {analysis.alteration.minerals.map(mineral => (
                                    <Badge key={mineral} variant="outline" className="text-xs">
                                      {mineral}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
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

export default AdvancedGeochemistryDashboard;