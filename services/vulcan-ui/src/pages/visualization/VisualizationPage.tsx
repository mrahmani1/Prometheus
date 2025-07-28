import React, { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { 
  Box, 
  Typography, 
  Paper, 
  Tabs, 
  Tab, 
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  CircularProgress
} from '@mui/material';
import { 
  ThreeDRotation, 
  Timeline, 
  Explore, 
  Settings,
  Download,
  Fullscreen
} from '@mui/icons-material';
import { ThreeVisualization } from '../../components/visualization/ThreeVisualization';
import { InteractiveGeologicalExplorer } from '../../components/visualization/InteractiveGeologicalExplorer';
import { 
  SeismicWavePropagation,
  DeformationField,
  ThermalEvolution,
  MagmaChamberEvolution,
  GeologicalParticleSystem
} from '../../components/visualization/TimeEvolvingVisualization';
import { 
  usePerformanceMonitoring,
  AdaptiveQualityManager
} from '../../components/visualization/PerformanceOptimizer';
import { 
  SeismicEvent, 
  DeformationMeasurement, 
  ThermalMeasurement,
  ModelPrediction,
  // GeologicalLocation
} from '../../types/geological';
import { geologicalDataService } from '../../services/geologicalDataService';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} style={{ height: '100%' }}>
    {value === index && children}
  </div>
);

export const VisualizationPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [viewMode, setViewMode] = useState<'3D' | '4D'>('3D');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date().toISOString());
  const [timeRange] = useState<[string, string]>([
    new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days ago
    new Date().toISOString()
  ]);
  const [showPerformanceStats, setShowPerformanceStats] = useState(false);
  const [adaptiveQuality, setAdaptiveQuality] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [_isFullscreen, setIsFullscreen] = useState(false);
  
  // Data state
  const [geologicalData, setGeologicalData] = useState<{
    seismicEvents: SeismicEvent[];
    deformationData: DeformationMeasurement[];
    thermalData: ThermalMeasurement[];
    predictions: ModelPrediction[];
    topographyData: any[];
  }>({
    seismicEvents: [],
    deformationData: [],
    thermalData: [],
    predictions: [],
    topographyData: []
  });
  
  // Performance monitoring
  const performanceMetrics = usePerformanceMonitoring();
  const qualityManager = useMemo(() => new AdaptiveQualityManager(), []);
  
  // Load geological data
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Load sample data - in production, this would come from the API
        const [seismicEvents, deformationData, thermalData, predictions] = await Promise.all([
          geologicalDataService.getSeismicEvents(timeRange[0], timeRange[1]),
          geologicalDataService.getDeformationData(timeRange[0], timeRange[1]),
          geologicalDataService.getThermalData(timeRange[0], timeRange[1]),
          geologicalDataService.getModelPredictions(timeRange[0], timeRange[1])
        ]);
        
        setGeologicalData({
          seismicEvents: seismicEvents || generateSampleSeismicEvents(),
          deformationData: deformationData || generateSampleDeformationData(),
          thermalData: thermalData || generateSampleThermalData(),
          predictions: predictions || generateSamplePredictions(),
          topographyData: []
        });
      } catch (err) {
        setError('Failed to load geological data');
        console.error('Data loading error:', err);
        
        // Use sample data as fallback
        setGeologicalData({
          seismicEvents: generateSampleSeismicEvents(),
          deformationData: generateSampleDeformationData(),
          thermalData: generateSampleThermalData(),
          predictions: generateSamplePredictions(),
          topographyData: []
        });
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [timeRange]);
  
  // Adaptive quality management
  useEffect(() => {
    if (adaptiveQuality && performanceMetrics.fps > 0) {
      qualityManager.updateQuality(performanceMetrics.fps);
    }
  }, [performanceMetrics.fps, adaptiveQuality, qualityManager]);
  
  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };
  
  // Handle data export
  const handleExport = () => {
    const dataToExport = {
      timeRange,
      currentTime,
      geologicalData,
      viewMode,
      exportedAt: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `geological-visualization-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Handle location selection from interactive explorer
  // const _handleLocationSelect = (_location: GeologicalLocation) => {
  //   console.log('Selected location:', location);
  //   // Could trigger additional data loading or analysis
  // };
  
  // Handle data requests from components
  const handleDataRequest = (params: any) => {
    console.log('Data request:', params);
    // Could trigger specific data loading based on user interaction
  };
  
  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h4" component="h1">
            3D/4D Geological Visualization
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            {/* View Mode Toggle */}
            <FormControl size="small">
              <InputLabel>View Mode</InputLabel>
              <Select
                value={viewMode}
                label="View Mode"
                onChange={(e) => setViewMode(e.target.value as '3D' | '4D')}
              >
                <MenuItem value="3D">3D Spatial</MenuItem>
                <MenuItem value="4D">4D Temporal</MenuItem>
              </Select>
            </FormControl>
            
            {/* Performance Settings */}
            <FormControlLabel
              control={
                <Switch
                  checked={showPerformanceStats}
                  onChange={(e) => setShowPerformanceStats(e.target.checked)}
                />
              }
              label="Performance Stats"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={adaptiveQuality}
                  onChange={(e) => setAdaptiveQuality(e.target.checked)}
                />
              }
              label="Adaptive Quality"
            />
            
            {/* Action Buttons */}
            <Button
              startIcon={<Download />}
              onClick={handleExport}
              variant="outlined"
              size="small"
            >
              Export
            </Button>
            
            <Button
              startIcon={<Fullscreen />}
              onClick={toggleFullscreen}
              variant="outlined"
              size="small"
            >
              Fullscreen
            </Button>
          </Box>
        </Box>
        
        {/* Performance Metrics */}
        {showPerformanceStats && (
          <Box sx={{ mt: 1, display: 'flex', gap: 2, fontSize: '12px', color: 'text.secondary' }}>
            <span>FPS: {performanceMetrics.fps}</span>
            <span>Frame Time: {performanceMetrics.frameTime}ms</span>
            <span>Triangles: {performanceMetrics.triangles.toLocaleString()}</span>
            <span>Draw Calls: {performanceMetrics.drawCalls}</span>
            {adaptiveQuality && (
              <span>Quality: {(qualityManager.getCurrentQuality() * 100).toFixed(0)}%</span>
            )}
          </Box>
        )}
      </Box>
      
      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ m: 2 }}>
          {error}
        </Alert>
      )}
      
      {/* Loading Indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress />
        </Box>
      )}
      
      {/* Main Content */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Tabs */}
        <Tabs 
          value={activeTab} 
          onChange={(_: any, newValue: any) => setActiveTab(newValue)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab icon={<ThreeDRotation />} label="3D Visualization" />
          <Tab icon={<Timeline />} label="Time Evolution" />
          <Tab icon={<Explore />} label="Interactive Explorer" />
          <Tab icon={<Settings />} label="Advanced Settings" />
        </Tabs>
        
        {/* Tab Panels */}
        <Box sx={{ flex: 1, position: 'relative' }}>
          {/* 3D Visualization Tab */}
          <TabPanel value={activeTab} index={0}>
            <Canvas
              camera={{ position: [10, 10, 10], fov: 60 }}
              style={{ width: '100%', height: '100%' }}
              gl={{ 
                antialias: true, 
                alpha: true,
                powerPreference: "high-performance"
              }}
            >
              <ThreeVisualization
                data={geologicalData}
                timeRange={timeRange}
                currentTime={currentTime}
                onTimeChange={setCurrentTime}
                isPlaying={isPlaying}
                onPlayStateChange={setIsPlaying}
                viewMode={viewMode}
                showPerformanceStats={showPerformanceStats}
              />
            </Canvas>
          </TabPanel>
          
          {/* Time Evolution Tab */}
          <TabPanel value={activeTab} index={1}>
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
              <directionalLight position={[10, 10, 5]} intensity={1} />
              
              {/* Time-evolving visualizations */}
              <SeismicWavePropagation
                events={geologicalData.seismicEvents}
                currentTime={currentTime}
                waveSpeed={6.0}
                maxWaveRadius={100}
              />
              
              <DeformationField
                deformationData={geologicalData.deformationData}
                currentTime={currentTime}
                timeWindow={86400000} // 24 hours
                gridResolution={20}
              />
              
              <ThermalEvolution
                thermalData={geologicalData.thermalData}
                currentTime={currentTime}
                timeWindow={86400000}
              />
              
              <MagmaChamberEvolution
                predictions={geologicalData.predictions}
                currentTime={currentTime}
                timeWindow={86400000}
              />
              
              {/* Particle systems for active processes */}
              {geologicalData.seismicEvents
                .filter(event => event.eventType === 'volcanic')
                .slice(0, 3) // Limit for performance
                .map((event) => (
                  <GeologicalParticleSystem
                    key={`particles-${event.id}`}
                    processType="volcanic_ash"
                    sourceLocation={new THREE.Vector3(
                      (event.location.longitude + 180) * 0.1 - 18,
                      (event.location.elevation || 0) * 0.001,
                      (event.location.latitude + 90) * 0.1 - 9
                    )}
                    currentTime={currentTime}
                    intensity={event.magnitude / 10}
                    windDirection={new THREE.Vector3(1, 0, 0.5)}
                  />
                ))}
            </Canvas>
          </TabPanel>
          
          {/* Interactive Explorer Tab */}
          <TabPanel value={activeTab} index={2}>
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
              <directionalLight position={[10, 10, 5]} intensity={1} />
              
              <InteractiveGeologicalExplorer
                data={geologicalData}
                onDataRequest={handleDataRequest}
              />
            </Canvas>
          </TabPanel>
          
          {/* Advanced Settings Tab */}
          <TabPanel value={activeTab} index={3}>
            <Paper sx={{ p: 3, m: 2 }}>
              <Typography variant="h6" gutterBottom>
                Advanced Visualization Settings
              </Typography>
              
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Performance Settings
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={adaptiveQuality}
                        onChange={(e) => setAdaptiveQuality(e.target.checked)}
                      />
                    }
                    label="Adaptive Quality Management"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={showPerformanceStats}
                        onChange={(e) => setShowPerformanceStats(e.target.checked)}
                      />
                    }
                    label="Show Performance Statistics"
                  />
                  
                  {adaptiveQuality && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Current Quality: {(qualityManager.getCurrentQuality() * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Target FPS: 60
                      </Typography>
                    </Box>
                  )}
                </Box>
                
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Data Settings
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary">
                    Seismic Events: {geologicalData.seismicEvents.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Deformation Points: {geologicalData.deformationData.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Thermal Measurements: {geologicalData.thermalData.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Model Predictions: {geologicalData.predictions.length}
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </TabPanel>
        </Box>
      </Box>
    </Box>
  );
};

// Sample data generators for development/testing
const generateSampleSeismicEvents = (): SeismicEvent[] => {
  const events: SeismicEvent[] = [];
  const now = Date.now();
  
  for (let i = 0; i < 50; i++) {
    events.push({
      id: `event-${i}`,
      timestamp: new Date(now - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      location: {
        latitude: -15 + Math.random() * 30, // Roughly centered on volcanic regions
        longitude: -30 + Math.random() * 60,
        elevation: Math.random() * 3000,
        depth: Math.random() * 50
      },
      magnitude: 1 + Math.random() * 6,
      depth: Math.random() * 50,
      eventType: Math.random() > 0.7 ? 'volcanic' : 'earthquake',
      confidence: 0.7 + Math.random() * 0.3
    });
  }
  
  return events;
};

const generateSampleDeformationData = (): DeformationMeasurement[] => {
  const measurements: DeformationMeasurement[] = [];
  const now = Date.now();
  
  for (let i = 0; i < 30; i++) {
    measurements.push({
      id: `deformation-${i}`,
      timestamp: new Date(now - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      location: {
        latitude: -15 + Math.random() * 30,
        longitude: -30 + Math.random() * 60,
        elevation: Math.random() * 3000
      },
      displacementVector: [
        (Math.random() - 0.5) * 0.1, // x displacement in meters
        (Math.random() - 0.5) * 0.1, // y displacement
        (Math.random() - 0.5) * 0.1  // z displacement
      ],
      measurementType: Math.random() > 0.5 ? 'GPS' : 'InSAR',
      uncertainty: [0.01, 0.01, 0.01],
      referenceFrame: 'ITRF2014'
    });
  }
  
  return measurements;
};

const generateSampleThermalData = (): ThermalMeasurement[] => {
  const measurements: ThermalMeasurement[] = [];
  const now = Date.now();
  
  for (let i = 0; i < 20; i++) {
    measurements.push({
      id: `thermal-${i}`,
      timestamp: new Date(now - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      location: {
        latitude: -15 + Math.random() * 30,
        longitude: -30 + Math.random() * 60,
        elevation: Math.random() * 3000
      },
      temperature: 20 + Math.random() * 200, // 20-220°C
      measurementMethod: Math.random() > 0.5 ? 'satellite' : 'ground-based',
      spatialResolution: 10 + Math.random() * 90, // 10-100m
      uncertainty: 1 + Math.random() * 5 // 1-6°C uncertainty
    });
  }
  
  return measurements;
};

const generateSamplePredictions = (): ModelPrediction[] => {
  const predictions: ModelPrediction[] = [];
  const now = Date.now();
  
  for (let i = 0; i < 10; i++) {
    predictions.push({
      id: `prediction-${i}`,
      timestamp: new Date(now - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      predictionType: 'magma_chamber_evolution',
      values: [
        Math.random() * 1000, // x position
        -Math.random() * 10,  // y position (depth)
        Math.random() * 1000, // z position
        1e9 + Math.random() * 1e10, // volume in m³
        1e6 + Math.random() * 1e8,  // pressure in Pa
        800 + Math.random() * 700   // temperature in °C
      ],
      uncertainty: [50, 1, 50, 1e8, 1e6, 50],
      confidenceInterval: [0.7, 0.9],
      modelVersion: 'helios-v1.0',
      inputDataHash: `hash-${i}`
    });
  }
  
  return predictions;
};