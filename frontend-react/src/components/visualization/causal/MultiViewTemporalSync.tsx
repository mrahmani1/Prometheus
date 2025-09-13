import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  Chip,
  Badge,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  ButtonGroup,
} from '@mui/material';
import {
  Sync,
  SyncDisabled,
  Visibility,
  VisibilityOff,
  ZoomIn,
  ZoomOut,
  Fullscreen,
  FullscreenExit,
  Settings,
  Timeline,
  ShowChart,
  TrendingUp,
  ExpandMore,
  GridView,
  ViewModule,
  AspectRatio,
} from '@mui/icons-material';
import { Canvas } from '@react-three/fiber';
import * as d3 from 'd3';
import { 
  TemporalCausalSnapshot, 
  CausalGraph, 
  TemporalScale, 
  TemporalVisualizationConfig,
  DetectedPattern,
  CausalNode,
  CausalEdge
} from './types';
import { TemporalCausalMetricsCalculator } from './temporalAnalysis';

export interface MultiViewTemporalSyncProps {
  snapshots: TemporalCausalSnapshot[];
  currentTime: Date;
  config: TemporalVisualizationConfig;
  detectedPatterns: DetectedPattern[];
  onTimeChange: (time: Date) => void;
  onConfigChange: (config: Partial<TemporalVisualizationConfig>) => void;
  onViewInteraction: (viewId: string, interaction: ViewInteraction) => void;
  width?: number;
  height?: number;
}

export interface ViewInteraction {
  type: 'zoom' | 'pan' | 'select' | 'hover' | 'filter';
  data: any;
  timestamp: Date;
}

interface TemporalView {
  id: string;
  title: string;
  scale: TemporalScale;
  timeRange: [Date, Date];
  snapshots: TemporalCausalSnapshot[];
  isActive: boolean;
  isSynced: boolean;
  viewType: 'network' | 'timeline' | 'metrics' | 'patterns';
  position: { row: number; col: number };
  size: { width: number; height: number };
}

interface SynchronizationState {
  enabled: boolean;
  masterView: string | null;
  syncedProperties: ('time' | 'zoom' | 'selection' | 'filters')[];
  crossScaleMapping: Map<TemporalScale, number>; // Scale factor for time synchronization
}

const DEFAULT_VIEWS: Omit<TemporalView, 'snapshots' | 'timeRange'>[] = [
  {
    id: 'realtime',
    title: 'Real-time Scale',
    scale: 'second',
    isActive: true,
    isSynced: true,
    viewType: 'network',
    position: { row: 0, col: 0 },
    size: { width: 50, height: 50 },
  },
  {
    id: 'daily',
    title: 'Daily Scale',
    scale: 'day',
    isActive: true,
    isSynced: true,
    viewType: 'timeline',
    position: { row: 0, col: 1 },
    size: { width: 50, height: 50 },
  },
  {
    id: 'geological',
    title: 'Geological Scale',
    scale: 'geological',
    isActive: true,
    isSynced: true,
    viewType: 'metrics',
    position: { row: 1, col: 0 },
    size: { width: 50, height: 50 },
  },
  {
    id: 'patterns',
    title: 'Pattern Analysis',
    scale: 'year',
    isActive: true,
    isSynced: false,
    viewType: 'patterns',
    position: { row: 1, col: 1 },
    size: { width: 50, height: 50 },
  },
];

const SCALE_TIME_FACTORS = {
  millisecond: 1,
  second: 1000,
  minute: 60000,
  hour: 3600000,
  day: 86400000,
  month: 2592000000,
  year: 31536000000,
  decade: 315360000000,
  century: 3153600000000,
  millennium: 31536000000000,
  geological: 31536000000000000,
};

export const MultiViewTemporalSync: React.FC<MultiViewTemporalSyncProps> = ({
  snapshots,
  currentTime,
  config,
  detectedPatterns,
  onTimeChange,
  onConfigChange,
  onViewInteraction,
  width = 1200,
  height = 800,
}) => {
  // State management
  const [views, setViews] = useState<TemporalView[]>([]);
  const [syncState, setSyncState] = useState<SynchronizationState>({
    enabled: config.enableSynchronization,
    masterView: 'realtime',
    syncedProperties: ['time', 'selection'],
    crossScaleMapping: new Map(
      Object.entries(SCALE_TIME_FACTORS).map(([scale, factor]) => [scale as TemporalScale, factor])
    ),
  });
  const [fullscreenView, setFullscreenView] = useState<string | null>(null);
  const [layoutMode, setLayoutMode] = useState<'grid' | 'cascade' | 'focus'>('grid');

  // Refs for view synchronization
  const viewRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const metricsCalculator = useRef(new TemporalCausalMetricsCalculator());

  // Initialize views with processed data
  useEffect(() => {
    const processedViews = DEFAULT_VIEWS.map(viewTemplate => {
      const viewSnapshots = filterSnapshotsForScale(snapshots, viewTemplate.scale);
      const timeRange = viewSnapshots.length > 0 
        ? [viewSnapshots[0].timestamp, viewSnapshots[viewSnapshots.length - 1].timestamp] as [Date, Date]
        : [new Date(), new Date()] as [Date, Date];

      return {
        ...viewTemplate,
        snapshots: viewSnapshots,
        timeRange,
      };
    });

    setViews(processedViews);
  }, [snapshots]);

  // Synchronization logic
  const synchronizeViews = useCallback((sourceViewId: string, interaction: ViewInteraction) => {
    if (!syncState.enabled || sourceViewId === syncState.masterView) return;

    setViews(prevViews => 
      prevViews.map(view => {
        if (view.id === sourceViewId || !view.isSynced) return view;

        // Apply synchronization based on interaction type and synced properties
        let updatedView = { ...view };

        if (syncState.syncedProperties.includes('time') && interaction.type === 'zoom') {
          // Synchronize time ranges across scales
          const sourceView = prevViews.find(v => v.id === sourceViewId);
          if (sourceView) {
            updatedView.timeRange = mapTimeRangeAcrossScales(
              interaction.data.timeRange,
              sourceView.scale,
              view.scale
            );
          }
        }

        return updatedView;
      })
    );

    // Notify parent component
    onViewInteraction(sourceViewId, interaction);
  }, [syncState, onViewInteraction]);

  // Handle view configuration changes
  const handleViewConfigChange = useCallback((viewId: string, changes: Partial<TemporalView>) => {
    setViews(prevViews =>
      prevViews.map(view =>
        view.id === viewId ? { ...view, ...changes } : view
      )
    );
  }, []);

  // Handle synchronization state changes
  const handleSyncStateChange = useCallback((changes: Partial<SynchronizationState>) => {
    setSyncState(prev => ({ ...prev, ...changes }));
    
    if (changes.enabled !== undefined) {
      onConfigChange({ enableSynchronization: changes.enabled });
    }
  }, [onConfigChange]);

  // Calculate layout based on mode and view sizes
  const calculateLayout = useMemo(() => {
    const activeViews = views.filter(view => view.isActive);
    
    switch (layoutMode) {
      case 'grid':
        return calculateGridLayout(activeViews, width, height);
      case 'cascade':
        return calculateCascadeLayout(activeViews, width, height);
      case 'focus':
        return calculateFocusLayout(activeViews, fullscreenView, width, height);
      default:
        return calculateGridLayout(activeViews, width, height);
    }
  }, [views, layoutMode, fullscreenView, width, height]);

  // Render individual view based on type
  const renderView = useCallback((view: TemporalView, layout: ViewLayout) => {
    const viewProps = {
      snapshots: view.snapshots,
      currentTime,
      config,
      detectedPatterns: detectedPatterns.filter(p => 
        isPatternRelevantForScale(p, view.scale)
      ),
      onInteraction: (interaction: ViewInteraction) => synchronizeViews(view.id, interaction),
      width: layout.width,
      height: layout.height,
    };

    switch (view.viewType) {
      case 'network':
        return <NetworkView {...viewProps} scale={view.scale} />;
      case 'timeline':
        return <TimelineView {...viewProps} scale={view.scale} />;
      case 'metrics':
        return <MetricsView {...viewProps} scale={view.scale} />;
      case 'patterns':
        return <PatternsView {...viewProps} scale={view.scale} />;
      default:
        return <NetworkView {...viewProps} scale={view.scale} />;
    }
  }, [currentTime, config, detectedPatterns, synchronizeViews]);

  return (
    <Box sx={{ width, height, position: 'relative' }}>
      {/* Control Panel */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Multi-Scale Temporal Views
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={syncState.enabled}
                    onChange={(e) => handleSyncStateChange({ enabled: e.target.checked })}
                  />
                }
                label="Synchronization"
              />
              
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Layout</InputLabel>
                <Select
                  value={layoutMode}
                  onChange={(e) => setLayoutMode(e.target.value as any)}
                >
                  <MenuItem value="grid">Grid</MenuItem>
                  <MenuItem value="cascade">Cascade</MenuItem>
                  <MenuItem value="focus">Focus</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end', alignItems: 'center' }}>
              {syncState.enabled && (
                <Chip
                  icon={<Sync />}
                  label={`Master: ${views.find(v => v.id === syncState.masterView)?.title || 'None'}`}
                  size="small"
                  color="primary"
                />
              )}
              
              <ButtonGroup size="small">
                <Button
                  startIcon={<GridView />}
                  variant={layoutMode === 'grid' ? 'contained' : 'outlined'}
                  onClick={() => setLayoutMode('grid')}
                >
                  Grid
                </Button>
                <Button
                  startIcon={<ViewModule />}
                  variant={layoutMode === 'cascade' ? 'contained' : 'outlined'}
                  onClick={() => setLayoutMode('cascade')}
                >
                  Cascade
                </Button>
                <Button
                  startIcon={<AspectRatio />}
                  variant={layoutMode === 'focus' ? 'contained' : 'outlined'}
                  onClick={() => setLayoutMode('focus')}
                >
                  Focus
                </Button>
              </ButtonGroup>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Views Container */}
      <Box sx={{ position: 'relative', width: '100%', height: height - 100 }}>
        {views.filter(view => view.isActive).map(view => {
          const layout = calculateLayout.find(l => l.viewId === view.id);
          if (!layout) return null;

          return (
            <Paper
              key={view.id}
              ref={(el) => {
                if (el) viewRefs.current.set(view.id, el);
              }}
              sx={{
                position: 'absolute',
                left: layout.x,
                top: layout.y,
                width: layout.width,
                height: layout.height,
                border: view.isSynced ? '2px solid #2196f3' : '1px solid #ddd',
                borderRadius: 1,
                overflow: 'hidden',
                zIndex: fullscreenView === view.id ? 1000 : 1,
              }}
            >
              {/* View Header */}
              <Box
                sx={{
                  p: 1,
                  borderBottom: '1px solid #ddd',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  bgcolor: view.isSynced ? 'primary.50' : 'grey.50',
                }}
              >
                <Typography variant="subtitle2">
                  {view.title}
                </Typography>
                
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <Tooltip title={view.isSynced ? 'Synchronized' : 'Not synchronized'}>
                    <IconButton
                      size="small"
                      onClick={() => handleViewConfigChange(view.id, { isSynced: !view.isSynced })}
                    >
                      {view.isSynced ? <Sync fontSize="small" /> : <SyncDisabled fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title={fullscreenView === view.id ? 'Exit fullscreen' : 'Fullscreen'}>
                    <IconButton
                      size="small"
                      onClick={() => setFullscreenView(fullscreenView === view.id ? null : view.id)}
                    >
                      {fullscreenView === view.id ? 
                        <FullscreenExit fontSize="small" /> : 
                        <Fullscreen fontSize="small" />
                      }
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Hide view">
                    <IconButton
                      size="small"
                      onClick={() => handleViewConfigChange(view.id, { isActive: false })}
                    >
                      <VisibilityOff fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>

              {/* View Content */}
              <Box sx={{ height: layout.height - 50, overflow: 'hidden' }}>
                {renderView(view, layout)}
              </Box>

              {/* View Status */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 4,
                  right: 4,
                  display: 'flex',
                  gap: 0.5,
                }}
              >
                <Chip
                  label={view.scale}
                  size="small"
                  variant="outlined"
                />
                {view.snapshots.length > 0 && (
                  <Badge badgeContent={view.snapshots.length} color="primary">
                    <Timeline fontSize="small" />
                  </Badge>
                )}
              </Box>
            </Paper>
          );
        })}
      </Box>

      {/* Synchronization Status */}
      {syncState.enabled && (
        <Paper
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            p: 1,
            bgcolor: 'primary.50',
            border: '1px solid',
            borderColor: 'primary.200',
          }}
        >
          <Typography variant="caption" display="block">
            Synchronization Active
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {syncState.syncedProperties.join(', ')} synchronized
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

// View Layout Interface
interface ViewLayout {
  viewId: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

// Layout calculation functions
function calculateGridLayout(views: TemporalView[], width: number, height: number): ViewLayout[] {
  const cols = Math.ceil(Math.sqrt(views.length));
  const rows = Math.ceil(views.length / cols);
  const viewWidth = width / cols;
  const viewHeight = height / rows;

  return views.map((view, index) => ({
    viewId: view.id,
    x: (index % cols) * viewWidth,
    y: Math.floor(index / cols) * viewHeight,
    width: viewWidth - 8, // Account for margins
    height: viewHeight - 8,
  }));
}

function calculateCascadeLayout(views: TemporalView[], width: number, height: number): ViewLayout[] {
  const baseWidth = width * 0.6;
  const baseHeight = height * 0.6;
  const offset = 40;

  return views.map((view, index) => ({
    viewId: view.id,
    x: index * offset,
    y: index * offset,
    width: baseWidth,
    height: baseHeight,
  }));
}

function calculateFocusLayout(
  views: TemporalView[], 
  focusViewId: string | null, 
  width: number, 
  height: number
): ViewLayout[] {
  if (!focusViewId) {
    return calculateGridLayout(views, width, height);
  }

  const focusView = views.find(v => v.id === focusViewId);
  const otherViews = views.filter(v => v.id !== focusViewId);

  const layouts: ViewLayout[] = [];

  // Focus view takes most of the space
  if (focusView) {
    layouts.push({
      viewId: focusView.id,
      x: 0,
      y: 0,
      width: width * 0.75,
      height: height,
    });
  }

  // Other views are arranged vertically on the right
  const sideWidth = width * 0.25;
  const sideHeight = height / otherViews.length;

  otherViews.forEach((view, index) => {
    layouts.push({
      viewId: view.id,
      x: width * 0.75,
      y: index * sideHeight,
      width: sideWidth,
      height: sideHeight - 8,
    });
  });

  return layouts;
}

// Helper functions
function filterSnapshotsForScale(snapshots: TemporalCausalSnapshot[], scale: TemporalScale): TemporalCausalSnapshot[] {
  // Filter and potentially aggregate snapshots based on temporal scale
  const scaleFactor = SCALE_TIME_FACTORS[scale];
  const aggregationWindow = scaleFactor / 1000; // Convert to seconds

  if (aggregationWindow <= 1) {
    // No aggregation needed for fine scales
    return snapshots;
  }

  // Aggregate snapshots for coarser scales
  const aggregated: TemporalCausalSnapshot[] = [];
  let currentWindow: TemporalCausalSnapshot[] = [];
  let windowStart = snapshots[0]?.timestamp.getTime() || 0;

  snapshots.forEach(snapshot => {
    if (snapshot.timestamp.getTime() - windowStart > aggregationWindow * 1000) {
      // Process current window
      if (currentWindow.length > 0) {
        aggregated.push(aggregateSnapshots(currentWindow, scale));
      }
      
      // Start new window
      currentWindow = [snapshot];
      windowStart = snapshot.timestamp.getTime();
    } else {
      currentWindow.push(snapshot);
    }
  });

  // Process final window
  if (currentWindow.length > 0) {
    aggregated.push(aggregateSnapshots(currentWindow, scale));
  }

  return aggregated;
}

function aggregateSnapshots(snapshots: TemporalCausalSnapshot[], scale: TemporalScale): TemporalCausalSnapshot {
  // Aggregate multiple snapshots into one for the given scale
  const representativeTime = new Date(
    snapshots.reduce((sum, s) => sum + s.timestamp.getTime(), 0) / snapshots.length
  );

  // Merge nodes (keep unique nodes, average properties)
  const nodeMap = new Map<string, CausalNode>();
  snapshots.forEach(snapshot => {
    snapshot.graph.nodes.forEach(node => {
      if (nodeMap.has(node.id)) {
        const existing = nodeMap.get(node.id)!;
        existing.metadata.confidence = (existing.metadata.confidence + node.metadata.confidence) / 2;
      } else {
        nodeMap.set(node.id, { ...node });
      }
    });
  });

  // Merge edges (average strengths)
  const edgeMap = new Map<string, CausalEdge>();
  snapshots.forEach(snapshot => {
    snapshot.graph.edges.forEach(edge => {
      const edgeKey = `${edge.source}-${edge.target}`;
      if (edgeMap.has(edgeKey)) {
        const existing = edgeMap.get(edgeKey)!;
        existing.strength = (existing.strength + edge.strength) / 2;
      } else {
        edgeMap.set(edgeKey, { ...edge });
      }
    });
  });

  return {
    timestamp: representativeTime,
    graph: {
      nodes: Array.from(nodeMap.values()),
      edges: Array.from(edgeMap.values()),
      metadata: {
        createdAt: representativeTime,
        lastUpdated: representativeTime,
        totalNodes: nodeMap.size,
        totalEdges: edgeMap.size,
        averageCausalStrength: Array.from(edgeMap.values()).reduce((sum, e) => sum + e.strength, 0) / edgeMap.size,
        spatialExtent: snapshots[0].graph.metadata.spatialExtent,
        temporalExtent: {
          start: snapshots[0].timestamp,
          end: snapshots[snapshots.length - 1].timestamp,
        },
      },
    },
    metadata: {
      temporalScale: scale,
      confidence: snapshots.reduce((sum, s) => sum + s.metadata.confidence, 0) / snapshots.length,
      dataQuality: snapshots.reduce((sum, s) => sum + (s.metadata.dataQuality || 1), 0) / snapshots.length,
      processingLatency: Math.max(...snapshots.map(s => s.metadata.processingLatency || 0)),
    },
  };
}

function mapTimeRangeAcrossScales(
  timeRange: [Date, Date],
  sourceScale: TemporalScale,
  targetScale: TemporalScale
): [Date, Date] {
  // Map time range from one scale to another
  const sourceFactor = SCALE_TIME_FACTORS[sourceScale];
  const targetFactor = SCALE_TIME_FACTORS[targetScale];
  const scalingRatio = targetFactor / sourceFactor;

  const duration = timeRange[1].getTime() - timeRange[0].getTime();
  const scaledDuration = duration * scalingRatio;
  const center = timeRange[0].getTime() + duration / 2;

  return [
    new Date(center - scaledDuration / 2),
    new Date(center + scaledDuration / 2),
  ];
}

function isPatternRelevantForScale(pattern: DetectedPattern, scale: TemporalScale): boolean {
  // Determine if a pattern is relevant for the given temporal scale
  const patternDuration = pattern.timeRange[1].getTime() - pattern.timeRange[0].getTime();
  const scaleFactor = SCALE_TIME_FACTORS[scale];
  
  // Pattern should be visible at this scale (not too short or too long)
  return patternDuration >= scaleFactor / 10 && patternDuration <= scaleFactor * 10;
}

// Individual view components (simplified implementations)
const NetworkView: React.FC<any> = ({ snapshots, currentTime, scale, width, height, onInteraction }) => (
  <Box sx={{ width, height, bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography variant="body2">Network View ({scale})</Typography>
  </Box>
);

const TimelineView: React.FC<any> = ({ snapshots, currentTime, scale, width, height, onInteraction }) => (
  <Box sx={{ width, height, bgcolor: '#f0f8ff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography variant="body2">Timeline View ({scale})</Typography>
  </Box>
);

const MetricsView: React.FC<any> = ({ snapshots, currentTime, scale, width, height, onInteraction }) => (
  <Box sx={{ width, height, bgcolor: '#f0fff0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography variant="body2">Metrics View ({scale})</Typography>
  </Box>
);

const PatternsView: React.FC<any> = ({ snapshots, currentTime, scale, width, height, onInteraction }) => (
  <Box sx={{ width, height, bgcolor: '#fff8f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography variant="body2">Patterns View ({scale})</Typography>
  </Box>
);

export default MultiViewTemporalSync;