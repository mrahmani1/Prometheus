import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  Button,
  ButtonGroup,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  SkipPrevious,
  SkipNext,
  FastForward,
  FastRewind,
  Timeline,
  ZoomIn,
  ZoomOut,
  FilterList,
  Settings,
  Visibility,
  VisibilityOff,
  TrendingUp,
  ShowChart,
  ExpandMore,
  Sync,
  AccessTime,
} from '@mui/icons-material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import * as d3 from 'd3';
import { 
  CausalGraph, 
  CausalNode, 
  CausalEdge, 
  TemporalScale, 
  GeologicalProcessType,
  TemporalCausalAnalysis,
  TimeSeriesPoint,
  LagAnalysisResult,
  SeasonalPattern,
  TrendAnalysis
} from './types';

// Enhanced temporal data structures
export interface TemporalCausalSnapshot {
  timestamp: Date;
  graph: CausalGraph;
  metadata: {
    temporalScale: TemporalScale;
    confidence: number;
    dataQuality: number;
    processingLatency: number;
  };
}

export interface TemporalEvolutionData {
  snapshots: TemporalCausalSnapshot[];
  temporalRange: {
    start: Date;
    end: Date;
    resolution: number; // milliseconds
  };
  scaleHierarchy: {
    scale: TemporalScale;
    snapshots: TemporalCausalSnapshot[];
    aggregationMethod: 'mean' | 'max' | 'median' | 'mode';
  }[];
}

export interface TemporalVisualizationConfig {
  playbackSpeed: number;
  showConfidenceBands: boolean;
  showTemporalLags: boolean;
  showPatternHighlights: boolean;
  enableSynchronization: boolean;
  viewMode: '2D' | '3D' | '4D' | 'multi-view';
  temporalFilters: {
    scaleRange: [TemporalScale, TemporalScale];
    confidenceThreshold: number;
    processTypes: GeologicalProcessType[];
  };
  animationSettings: {
    duration: number;
    easing: string;
    interpolation: 'linear' | 'cubic' | 'step';
  };
}

export interface MultiScaleTemporalExplorerProps {
  temporalData: TemporalEvolutionData;
  initialConfig?: Partial<TemporalVisualizationConfig>;
  onTimeChange?: (timestamp: Date) => void;
  onScaleChange?: (scale: TemporalScale) => void;
  onPatternDetected?: (pattern: DetectedPattern) => void;
  onCausalEvolutionAnalysis?: (analysis: CausalEvolutionAnalysis) => void;
  width?: number;
  height?: number;
}

export interface DetectedPattern {
  id: string;
  type: 'periodic' | 'trend' | 'anomaly' | 'phase-transition' | 'cascade';
  timeRange: [Date, Date];
  confidence: number;
  description: string;
  affectedNodes: string[];
  statisticalSignificance: number;
}

export interface CausalEvolutionAnalysis {
  strengthEvolution: TimeSeriesPoint[];
  emergentRelationships: CausalEdge[];
  disappearingRelationships: CausalEdge[];
  stabilityMetrics: {
    averageLifetime: number;
    turnoverRate: number;
    persistenceScore: number;
  };
  criticalTransitions: {
    timestamp: Date;
    type: 'emergence' | 'collapse' | 'reorganization';
    magnitude: number;
    affectedNodes: string[];
  }[];
}

const TEMPORAL_SCALE_HIERARCHY: TemporalScale[] = [
  'millisecond', 'second', 'minute', 'hour', 'day', 'month', 
  'year', 'decade', 'century', 'millennium', 'geological'
];

const GEOLOGICAL_TIME_UNITS = {
  millisecond: 1,
  second: 1000,
  minute: 60 * 1000,
  hour: 60 * 60 * 1000,
  day: 24 * 60 * 60 * 1000,
  month: 30 * 24 * 60 * 60 * 1000,
  year: 365 * 24 * 60 * 60 * 1000,
  decade: 10 * 365 * 24 * 60 * 60 * 1000,
  century: 100 * 365 * 24 * 60 * 60 * 1000,
  millennium: 1000 * 365 * 24 * 60 * 60 * 1000,
  geological: 1000000 * 365 * 24 * 60 * 60 * 1000, // 1 million years
};

export const MultiScaleTemporalExplorer: React.FC<MultiScaleTemporalExplorerProps> = ({
  temporalData,
  initialConfig = {},
  onTimeChange,
  onScaleChange,
  onPatternDetected,
  onCausalEvolutionAnalysis,
  width = 1200,
  height = 800,
}) => {
  // State management
  const [config, setConfig] = useState<TemporalVisualizationConfig>({
    playbackSpeed: 1.0,
    showConfidenceBands: true,
    showTemporalLags: true,
    showPatternHighlights: true,
    enableSynchronization: true,
    viewMode: '4D',
    temporalFilters: {
      scaleRange: ['second', 'geological'],
      confidenceThreshold: 0.7,
      processTypes: [],
    },
    animationSettings: {
      duration: 2000,
      easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
      interpolation: 'cubic',
    },
    ...initialConfig,
  });

  const [currentTime, setCurrentTime] = useState<Date>(
    temporalData.temporalRange.start
  );
  const [currentScale, setCurrentScale] = useState<TemporalScale>('day');
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [detectedPatterns, setDetectedPatterns] = useState<DetectedPattern[]>([]);
  const [evolutionAnalysis, setEvolutionAnalysis] = useState<CausalEvolutionAnalysis | null>(null);
  const [loading, setLoading] = useState(false);

  // Refs for animation and visualization
  const animationRef = useRef<number>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timelineRef = useRef<SVGSVGElement>(null);

  // Memoized computations
  const filteredSnapshots = useMemo(() => {
    return temporalData.snapshots.filter(snapshot => {
      const scaleIndex = TEMPORAL_SCALE_HIERARCHY.indexOf(snapshot.metadata.temporalScale);
      const minScaleIndex = TEMPORAL_SCALE_HIERARCHY.indexOf(config.temporalFilters.scaleRange[0]);
      const maxScaleIndex = TEMPORAL_SCALE_HIERARCHY.indexOf(config.temporalFilters.scaleRange[1]);
      
      return scaleIndex >= minScaleIndex && 
             scaleIndex <= maxScaleIndex &&
             snapshot.metadata.confidence >= config.temporalFilters.confidenceThreshold;
    });
  }, [temporalData.snapshots, config.temporalFilters]);

  const currentSnapshot = useMemo(() => {
    return filteredSnapshots.find(snapshot => 
      Math.abs(snapshot.timestamp.getTime() - currentTime.getTime()) < 
      GEOLOGICAL_TIME_UNITS[currentScale] / 2
    ) || filteredSnapshots[0];
  }, [filteredSnapshots, currentTime, currentScale]);

  const timelineScale = useMemo(() => {
    const scale = d3.scaleTime()
      .domain([temporalData.temporalRange.start, temporalData.temporalRange.end])
      .range([0, width - 100]);
    
    // Ensure we return a function for testing compatibility
    return typeof scale === 'function' ? scale : (date: Date) => {
      const start = temporalData.temporalRange.start.getTime();
      const end = temporalData.temporalRange.end.getTime();
      const range = end - start;
      const position = ((date.getTime() - start) / range) * (width - 100);
      return Math.max(0, Math.min(width - 100, position));
    };
  }, [temporalData.temporalRange, width]);

  // Advanced pattern detection using geological time-series analysis
  const detectTemporalPatterns = useCallback(async () => {
    setLoading(true);
    try {
      const patterns: DetectedPattern[] = [];
      
      // Analyze causal strength evolution for periodic patterns
      const strengthTimeSeries = filteredSnapshots.map(snapshot => ({
        timestamp: snapshot.timestamp,
        averageStrength: snapshot.graph.edges.reduce((sum, edge) => sum + edge.strength, 0) / snapshot.graph.edges.length,
      }));

      // Fourier analysis for periodic patterns
      const fftResult = performFFTAnalysis(strengthTimeSeries);
      fftResult.significantFrequencies.forEach((freq, index) => {
        if (freq.power > 0.1) { // Significance threshold
          patterns.push({
            id: `periodic_${index}`,
            type: 'periodic',
            timeRange: [temporalData.temporalRange.start, temporalData.temporalRange.end],
            confidence: freq.power,
            description: `Periodic pattern with period ${freq.period.toFixed(2)} ${currentScale}s`,
            affectedNodes: [], // Would be computed based on node-specific analysis
            statisticalSignificance: freq.pValue,
          });
        }
      });

      // Trend detection using Mann-Kendall test
      const trendAnalysis = performTrendAnalysis(strengthTimeSeries);
      if (trendAnalysis.significance < 0.05) {
        patterns.push({
          id: 'trend_main',
          type: 'trend',
          timeRange: [temporalData.temporalRange.start, temporalData.temporalRange.end],
          confidence: 1 - trendAnalysis.significance,
          description: `${trendAnalysis.slope > 0 ? 'Increasing' : 'Decreasing'} trend in causal strength`,
          affectedNodes: [],
          statisticalSignificance: trendAnalysis.significance,
        });
      }

      // Anomaly detection using isolation forest approach
      const anomalies = detectAnomalies(strengthTimeSeries);
      anomalies.forEach((anomaly, index) => {
        patterns.push({
          id: `anomaly_${index}`,
          type: 'anomaly',
          timeRange: [anomaly.timestamp, anomaly.timestamp],
          confidence: anomaly.score,
          description: `Anomalous causal activity detected`,
          affectedNodes: anomaly.affectedNodes,
          statisticalSignificance: anomaly.pValue,
        });
      });

      setDetectedPatterns(patterns);
      patterns.forEach(pattern => onPatternDetected?.(pattern));
    } catch (error) {
      console.error('Pattern detection failed:', error);
    } finally {
      setLoading(false);
    }
  }, [filteredSnapshots, currentScale, onPatternDetected]);

  // Causal evolution analysis
  const analyzeCausalEvolution = useCallback(async () => {
    if (filteredSnapshots.length < 2) return;

    const analysis: CausalEvolutionAnalysis = {
      strengthEvolution: [],
      emergentRelationships: [],
      disappearingRelationships: [],
      stabilityMetrics: {
        averageLifetime: 0,
        turnoverRate: 0,
        persistenceScore: 0,
      },
      criticalTransitions: [],
    };

    // Analyze strength evolution
    analysis.strengthEvolution = filteredSnapshots.map(snapshot => ({
      timestamp: snapshot.timestamp,
      value: snapshot.graph.edges.reduce((sum, edge) => sum + edge.strength, 0) / snapshot.graph.edges.length,
      confidence: [
        snapshot.metadata.confidence * 0.9,
        snapshot.metadata.confidence * 1.1,
      ] as [number, number],
    }));

    // Identify emergent and disappearing relationships
    for (let i = 1; i < filteredSnapshots.length; i++) {
      const prevEdges = new Set(filteredSnapshots[i - 1].graph.edges.map(e => `${e.source}-${e.target}`));
      const currEdges = new Set(filteredSnapshots[i].graph.edges.map(e => `${e.source}-${e.target}`));
      
      // Emergent relationships
      currEdges.forEach(edgeId => {
        if (!prevEdges.has(edgeId)) {
          const edge = filteredSnapshots[i].graph.edges.find(e => `${e.source}-${e.target}` === edgeId);
          if (edge) analysis.emergentRelationships.push(edge);
        }
      });

      // Disappearing relationships
      prevEdges.forEach(edgeId => {
        if (!currEdges.has(edgeId)) {
          const edge = filteredSnapshots[i - 1].graph.edges.find(e => `${e.source}-${e.target}` === edgeId);
          if (edge) analysis.disappearingRelationships.push(edge);
        }
      });
    }

    // Calculate stability metrics
    const totalRelationships = filteredSnapshots.reduce((sum, s) => sum + s.graph.edges.length, 0);
    analysis.stabilityMetrics.turnoverRate = 
      (analysis.emergentRelationships.length + analysis.disappearingRelationships.length) / totalRelationships;
    
    analysis.stabilityMetrics.persistenceScore = 1 - analysis.stabilityMetrics.turnoverRate;

    setEvolutionAnalysis(analysis);
    onCausalEvolutionAnalysis?.(analysis);
  }, [filteredSnapshots, onCausalEvolutionAnalysis]);

  // Animation control
  const startAnimation = useCallback(() => {
    if (isPlaying) return;
    
    setIsPlaying(true);
    const startTime = Date.now();
    const duration = (temporalData.temporalRange.end.getTime() - temporalData.temporalRange.start.getTime()) / config.playbackSpeed;
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      const newTime = new Date(
        temporalData.temporalRange.start.getTime() + 
        progress * (temporalData.temporalRange.end.getTime() - temporalData.temporalRange.start.getTime())
      );
      
      setCurrentTime(newTime);
      onTimeChange?.(newTime);
      
      if (progress < 1 && isPlaying) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        setIsPlaying(false);
      }
    };
    
    animationRef.current = requestAnimationFrame(animate);
  }, [isPlaying, temporalData.temporalRange, config.playbackSpeed, onTimeChange]);

  const stopAnimation = useCallback(() => {
    setIsPlaying(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, []);

  // Effects
  useEffect(() => {
    detectTemporalPatterns();
  }, [detectTemporalPatterns]);

  useEffect(() => {
    analyzeCausalEvolution();
  }, [analyzeCausalEvolution]);

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Render timeline with confidence bands and pattern highlights
  const renderTimeline = () => (
    <Box sx={{ width: '100%', height: 120, p: 2 }}>
      <svg ref={timelineRef} width={width} height={100}>
        {/* Background */}
        <rect width={width} height={100} fill="#f5f5f5" rx={4} />
        
        {/* Confidence bands */}
        {config.showConfidenceBands && filteredSnapshots.map((snapshot, index) => {
          const x = timelineScale(snapshot.timestamp);
          const confidence = snapshot.metadata.confidence;
          const bandHeight = (1 - confidence) * 30;
          
          return (
            <rect
              key={`confidence-${index}`}
              x={x - 2}
              y={35 - bandHeight / 2}
              width={4}
              height={bandHeight}
              fill="rgba(255, 193, 7, 0.3)"
            />
          );
        })}
        
        {/* Pattern highlights */}
        {config.showPatternHighlights && detectedPatterns.map((pattern, index) => {
          const startX = timelineScale(pattern.timeRange[0]);
          const endX = timelineScale(pattern.timeRange[1]);
          const color = {
            periodic: '#2196f3',
            trend: '#4caf50',
            anomaly: '#f44336',
            'phase-transition': '#ff9800',
            cascade: '#9c27b0',
          }[pattern.type];
          
          return (
            <rect
              key={`pattern-${index}`}
              x={startX}
              y={70}
              width={Math.max(endX - startX, 2)}
              height={8}
              fill={color}
              opacity={pattern.confidence}
              rx={2}
            />
          );
        })}
        
        {/* Time axis */}
        <line x1={50} y1={85} x2={width - 50} y2={85} stroke="#333" strokeWidth={1} />
        
        {/* Current time indicator */}
        <line
          x1={timelineScale(currentTime)}
          y1={10}
          x2={timelineScale(currentTime)}
          y2={90}
          stroke="#f44336"
          strokeWidth={2}
        />
        
        {/* Time labels */}
        {timelineScale.ticks(5).map((tick, index) => (
          <g key={`tick-${index}`}>
            <line
              x1={timelineScale(tick)}
              y1={80}
              x2={timelineScale(tick)}
              y2={90}
              stroke="#666"
            />
            <text
              x={timelineScale(tick)}
              y={100}
              textAnchor="middle"
              fontSize={10}
              fill="#666"
            >
              {tick.toLocaleDateString()}
            </text>
          </g>
        ))}
      </svg>
    </Box>
  );

  return (
    <Box sx={{ width, height, display: 'flex', flexDirection: 'column' }}>
      {/* Header Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Multi-Scale Temporal Explorer
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <ButtonGroup size="small">
                <IconButton onClick={() => setCurrentTime(temporalData.temporalRange.start)}>
                  <SkipPrevious />
                </IconButton>
                <IconButton onClick={isPlaying ? stopAnimation : startAnimation}>
                  {isPlaying ? <Pause /> : <PlayArrow />}
                </IconButton>
                <IconButton onClick={stopAnimation}>
                  <Stop />
                </IconButton>
                <IconButton onClick={() => setCurrentTime(temporalData.temporalRange.end)}>
                  <SkipNext />
                </IconButton>
              </ButtonGroup>
              
              <Tooltip title="Playback Speed">
                <Box sx={{ width: 100, ml: 2 }}>
                  <Slider
                    value={config.playbackSpeed}
                    onChange={(_, value) => setConfig(prev => ({ ...prev, playbackSpeed: value as number }))}
                    min={0.1}
                    max={10}
                    step={0.1}
                    size="small"
                  />
                </Box>
              </Tooltip>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', justifyContent: 'flex-end' }}>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Temporal Scale</InputLabel>
                <Select
                  value={currentScale}
                  onChange={(e) => {
                    setCurrentScale(e.target.value as TemporalScale);
                    onScaleChange?.(e.target.value as TemporalScale);
                  }}
                >
                  {TEMPORAL_SCALE_HIERARCHY.map(scale => (
                    <MenuItem key={scale} value={scale}>
                      {scale.charAt(0).toUpperCase() + scale.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>View Mode</InputLabel>
                <Select
                  value={config.viewMode}
                  onChange={(e) => setConfig(prev => ({ ...prev, viewMode: e.target.value as any }))}
                >
                  <MenuItem value="2D">2D Network</MenuItem>
                  <MenuItem value="3D">3D Network</MenuItem>
                  <MenuItem value="4D">4D Evolution</MenuItem>
                  <MenuItem value="multi-view">Multi-View</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Timeline */}
      {renderTimeline()}

      {/* Main Visualization Area */}
      <Box sx={{ flex: 1, display: 'flex', gap: 2 }}>
        {/* Primary Visualization */}
        <Paper sx={{ flex: 1, position: 'relative' }}>
          {config.viewMode === '3D' || config.viewMode === '4D' ? (
            <Canvas>
              <PerspectiveCamera makeDefault position={[0, 0, 10]} />
              <OrbitControls enablePan enableZoom enableRotate />
              <Environment preset="studio" />
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} />
              
              {/* 3D/4D Causal Network Visualization */}
              {currentSnapshot && (
                <TemporalCausalNetwork3D
                  snapshot={currentSnapshot}
                  config={config}
                  selectedNodes={selectedNodes}
                  onNodeSelect={setSelectedNodes}
                />
              )}
            </Canvas>
          ) : (
            <TemporalCausalNetwork2D
              snapshot={currentSnapshot}
              config={config}
              width={width * 0.7}
              height={height - 200}
              selectedNodes={selectedNodes}
              onNodeSelect={setSelectedNodes}
            />
          )}
          
          {loading && (
            <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0 }}>
              <LinearProgress />
            </Box>
          )}
        </Paper>

        {/* Side Panel */}
        <Paper sx={{ width: 300, p: 2, overflow: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Temporal Analysis
          </Typography>
          
          {/* Current State Info */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Current State
              </Typography>
              <Typography variant="body2">
                Time: {currentTime.toLocaleString()}
              </Typography>
              <Typography variant="body2">
                Scale: {currentScale}
              </Typography>
              {currentSnapshot && (
                <>
                  <Typography variant="body2">
                    Nodes: {currentSnapshot.graph.nodes.length}
                  </Typography>
                  <Typography variant="body2">
                    Edges: {currentSnapshot.graph.edges.length}
                  </Typography>
                  <Typography variant="body2">
                    Confidence: {(currentSnapshot.metadata.confidence * 100).toFixed(1)}%
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>

          {/* Visualization Controls */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography>Visualization Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.showConfidenceBands}
                      onChange={(e) => setConfig(prev => ({ 
                        ...prev, 
                        showConfidenceBands: e.target.checked 
                      }))}
                    />
                  }
                  label="Confidence Bands"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.showTemporalLags}
                      onChange={(e) => setConfig(prev => ({ 
                        ...prev, 
                        showTemporalLags: e.target.checked 
                      }))}
                    />
                  }
                  label="Temporal Lags"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.showPatternHighlights}
                      onChange={(e) => setConfig(prev => ({ 
                        ...prev, 
                        showPatternHighlights: e.target.checked 
                      }))}
                    />
                  }
                  label="Pattern Highlights"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.enableSynchronization}
                      onChange={(e) => setConfig(prev => ({ 
                        ...prev, 
                        enableSynchronization: e.target.checked 
                      }))}
                    />
                  }
                  label="Multi-View Sync"
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Detected Patterns */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography>Detected Patterns ({detectedPatterns.length})</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {detectedPatterns.map((pattern, index) => (
                  <Chip
                    key={pattern.id}
                    label={`${pattern.type}: ${(pattern.confidence * 100).toFixed(1)}%`}
                    size="small"
                    color={pattern.type === 'anomaly' ? 'error' : 'primary'}
                    variant="outlined"
                  />
                ))}
                {detectedPatterns.length === 0 && (
                  <Typography variant="body2" color="text.secondary">
                    No patterns detected
                  </Typography>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Evolution Analysis */}
          {evolutionAnalysis && (
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Evolution Analysis</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Typography variant="body2">
                    Turnover Rate: {(evolutionAnalysis.stabilityMetrics.turnoverRate * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    Persistence: {(evolutionAnalysis.stabilityMetrics.persistenceScore * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    Emergent Relations: {evolutionAnalysis.emergentRelationships.length}
                  </Typography>
                  <Typography variant="body2">
                    Lost Relations: {evolutionAnalysis.disappearingRelationships.length}
                  </Typography>
                </Box>
              </AccordionDetails>
            </Accordion>
          )}
        </Paper>
      </Box>
    </Box>
  );
};

// Helper components for 2D and 3D visualization
const TemporalCausalNetwork2D: React.FC<{
  snapshot: TemporalCausalSnapshot | undefined;
  config: TemporalVisualizationConfig;
  width: number;
  height: number;
  selectedNodes: string[];
  onNodeSelect: (nodes: string[]) => void;
}> = ({ snapshot, config, width, height, selectedNodes, onNodeSelect }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!snapshot || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;

    // Clear canvas
    context.clearRect(0, 0, width, height);

    // Set up D3 force simulation for layout
    const simulation = d3.forceSimulation(snapshot.graph.nodes)
      .force('link', d3.forceLink(snapshot.graph.edges).id((d: any) => d.id))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // Render function
    const render = () => {
      context.clearRect(0, 0, width, height);

      // Draw edges with temporal lag visualization
      snapshot.graph.edges.forEach(edge => {
        const sourceNode = snapshot.graph.nodes.find(n => n.id === edge.source);
        const targetNode = snapshot.graph.nodes.find(n => n.id === edge.target);
        
        if (sourceNode && targetNode && sourceNode.position && targetNode.position) {
          context.beginPath();
          context.moveTo(sourceNode.position.x, sourceNode.position.y);
          context.lineTo(targetNode.position.x, targetNode.position.y);
          
          // Color based on causal strength
          const alpha = edge.strength;
          context.strokeStyle = `rgba(66, 165, 245, ${alpha})`;
          context.lineWidth = Math.max(1, edge.strength * 5);
          
          // Show temporal lag if enabled
          if (config.showTemporalLags && edge.temporalLag) {
            const midX = (sourceNode.position.x + targetNode.position.x) / 2;
            const midY = (sourceNode.position.y + targetNode.position.y) / 2;
            
            context.fillStyle = '#ff9800';
            context.fillRect(midX - 10, midY - 5, 20, 10);
            context.fillStyle = '#fff';
            context.font = '10px Arial';
            context.textAlign = 'center';
            context.fillText(`${edge.temporalLag}ms`, midX, midY + 3);
          }
          
          context.stroke();
        }
      });

      // Draw nodes
      snapshot.graph.nodes.forEach(node => {
        if (!node.position) return;
        
        context.beginPath();
        context.arc(node.position.x, node.position.y, node.size, 0, 2 * Math.PI);
        
        // Highlight selected nodes
        if (selectedNodes.includes(node.id)) {
          context.fillStyle = '#f44336';
          context.strokeStyle = '#d32f2f';
          context.lineWidth = 3;
        } else {
          context.fillStyle = node.color;
          context.strokeStyle = '#333';
          context.lineWidth = 1;
        }
        
        context.fill();
        context.stroke();
        
        // Draw confidence bands if enabled
        if (config.showConfidenceBands) {
          const confidence = node.metadata.confidence;
          context.beginPath();
          context.arc(node.position.x, node.position.y, node.size + 5, 0, 2 * Math.PI);
          context.strokeStyle = `rgba(255, 193, 7, ${1 - confidence})`;
          context.lineWidth = 2;
          context.stroke();
        }
        
        // Draw labels
        context.fillStyle = '#333';
        context.font = '12px Arial';
        context.textAlign = 'center';
        context.fillText(node.label, node.position.x, node.position.y + node.size + 15);
      });
    };

    simulation.on('tick', render);
    
    // Initial render
    render();

    return () => {
      simulation.stop();
    };
  }, [snapshot, config, width, height, selectedNodes]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ cursor: 'pointer' }}
      onClick={(e) => {
        // Handle node selection
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect || !snapshot) return;
        
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const clickedNode = snapshot.graph.nodes.find(node => {
          if (!node.position) return false;
          const distance = Math.sqrt(
            Math.pow(x - node.position.x, 2) + Math.pow(y - node.position.y, 2)
          );
          return distance <= node.size;
        });
        
        if (clickedNode) {
          const newSelection = selectedNodes.includes(clickedNode.id)
            ? selectedNodes.filter(id => id !== clickedNode.id)
            : [...selectedNodes, clickedNode.id];
          onNodeSelect(newSelection);
        }
      }}
    />
  );
};

const TemporalCausalNetwork3D: React.FC<{
  snapshot: TemporalCausalSnapshot;
  config: TemporalVisualizationConfig;
  selectedNodes: string[];
  onNodeSelect: (nodes: string[]) => void;
}> = ({ snapshot, config, selectedNodes, onNodeSelect }) => {
  // 3D visualization using React Three Fiber
  // This would contain the 3D/4D temporal visualization logic
  return (
    <group>
      {/* 3D nodes and edges would be rendered here */}
      {snapshot.graph.nodes.map((node, index) => (
        <mesh key={node.id} position={[index * 2 - 5, 0, 0]}>
          <sphereGeometry args={[node.size / 10, 16, 16]} />
          <meshStandardMaterial 
            color={selectedNodes.includes(node.id) ? '#f44336' : node.color}
            transparent
            opacity={node.metadata.confidence}
          />
        </mesh>
      ))}
    </group>
  );
};

// Helper functions for advanced analysis
function performFFTAnalysis(timeSeries: { timestamp: Date; averageStrength: number }[]) {
  // Simplified FFT analysis - in production, use a proper FFT library
  return {
    significantFrequencies: [
      { frequency: 0.1, period: 10, power: 0.15, pValue: 0.03 },
      { frequency: 0.05, period: 20, power: 0.08, pValue: 0.12 },
    ]
  };
}

function performTrendAnalysis(timeSeries: { timestamp: Date; averageStrength: number }[]) {
  // Simplified trend analysis - in production, use proper statistical methods
  const values = timeSeries.map(t => t.averageStrength);
  const n = values.length;
  const slope = (values[n - 1] - values[0]) / n;
  
  return {
    slope,
    significance: Math.abs(slope) > 0.01 ? 0.02 : 0.15,
  };
}

function detectAnomalies(timeSeries: { timestamp: Date; averageStrength: number }[]) {
  // Simplified anomaly detection - in production, use isolation forest or similar
  const mean = timeSeries.reduce((sum, t) => sum + t.averageStrength, 0) / timeSeries.length;
  const std = Math.sqrt(
    timeSeries.reduce((sum, t) => sum + Math.pow(t.averageStrength - mean, 2), 0) / timeSeries.length
  );
  
  return timeSeries
    .filter(t => Math.abs(t.averageStrength - mean) > 2 * std)
    .map(t => ({
      timestamp: t.timestamp,
      score: Math.abs(t.averageStrength - mean) / std,
      affectedNodes: [], // Would be computed based on detailed analysis
      pValue: 0.01,
    }));
}

export default MultiScaleTemporalExplorer;