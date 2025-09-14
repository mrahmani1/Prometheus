import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Box,
  Grid,
  // Paper, Tabs, Tab - unused
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Chip,
  Badge,
  Tooltip,
  // Fab - unused
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  Alert,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Switch,
  FormControlLabel,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  // Card, CardContent, CardHeader - unused
  // CardActions - unused
  LinearProgress,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  // Autocomplete, ButtonGroup, ToggleButton, ToggleButtonGroup - unused
} from '@mui/material';
import {
  Menu as MenuIcon,
  // Dashboard as DashboardIcon - unused
  Timeline as TimelineIcon,
  AccountTree as NetworkIcon,
  Science as HypothesisIcon,
  Group as CollaborativeIcon,
  MonitorHeart as MonitorIcon,
  Settings as SettingsIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  Share as ShareIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  // PlayArrow as PlayIcon, Pause as PauseIcon - unused
  // Stop as StopIcon - unused
  ExpandMore as ExpandMoreIcon,
  Notifications as NotificationsIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
  Info as InfoIcon,
  // FilterList as FilterIcon, Search as SearchIcon, ZoomIn as ZoomInIcon, ZoomOut as ZoomOutIcon, CenterFocusStrong as CenterIcon, Layers as LayersIcon, Visibility as VisibilityIcon, VisibilityOff as VisibilityOffIcon - unused
} from '@mui/icons-material';
import { useTheme, alpha } from '@mui/material/styles';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../../store';
import { 
  setActiveTab, 
  setDrawerOpen, 
  setFullscreen,
  updateDashboardConfig,
  addNotification,
  removeNotification,
} from '../../store/slices/dashboardSlice';

// Import all the completed subtask components
import { CausalNetworkVisualization } from '../visualization/causal/CausalNetworkVisualization';
import { MultiScaleTemporalExplorer } from '../visualization/causal/MultiScaleTemporalExplorer';
import { CausalFilterPanel } from '../visualization/causal/CausalFilterPanel';
import { CausalSearchInterface } from '../visualization/causal/CausalSearchInterface';
// import { CausalPathwayExplorer } from '../visualization/causal/CausalPathwayExplorer'; // Unused
import { HypothesisTestingInterface } from '../hypothesis-testing/HypothesisTestingInterface';
import { CollaborativeAnalysisInterface } from '../collaborative/CollaborativeAnalysisInterface';
import { RealTimeProcessDashboard } from '../RealTimeProcessDashboard';

// Import types and API client
import {
  CausalGraph,
  CausalNode,
  CausalEdge,
  CausalLayoutConfig,
  CausalFilterConfig,
  CausalPath,
  InteractionMode,
  RenderMode,
  LayoutAlgorithm,
  // GeologicalProcessType - unused
  CausalDiscoveryRequest,
  CausalAnalysisResponse,
  CausalMetrics,
  RealTimeUpdate,
  HypothesisTestRequest,
  // TemporalCausalModel, MultiScaleCausalModel - unused
  CausalAlert,
  AlertSeverity,
  // SpatialScale, TemporalScale - unused
  CausalDiscoveryMethod,
  // ValidationStatus - unused
} from '../../types/causal';

import { causalAnalysisApi } from '../../services/causalAnalysisApi';

interface ComprehensiveCausalDashboardProps {
  initialGraph?: CausalGraph;
  realTimeEnabled?: boolean;
  collaborativeMode?: boolean;
  onExport?: (format: string, data: any) => void;
  onShare?: (shareConfig: any) => void;
  className?: string;
}

interface DashboardTab {
  id: string;
  label: string;
  icon: React.ReactNode;
  component: React.ReactNode;
  badge?: number;
  disabled?: boolean;
}

// interface DashboardNotification { // Unused
//   id: string;
//   type: 'info' | 'warning' | 'error' | 'success';
//   title: string;
//   message: string;
//   timestamp: Date;
//   autoHide?: boolean;
//   duration?: number;
// }

// interface DashboardConfig { // Unused
//   layout: 'grid' | 'tabs' | 'split' | 'custom';
//   theme: 'light' | 'dark' | 'auto';
//   realTimeUpdates: boolean;
//   autoRefresh: boolean;
//   refreshInterval: number;
//   notifications: boolean;
//   soundAlerts: boolean;
//   performanceMode: 'high' | 'balanced' | 'low';
//   renderQuality: 'high' | 'medium' | 'low';
//   animationsEnabled: boolean;
//   collaborativeFeatures: boolean;
// }

const ComprehensiveCausalDashboard: React.FC<ComprehensiveCausalDashboardProps> = ({
  initialGraph,
  realTimeEnabled = true,
  collaborativeMode = false,
  onExport,
  onShare,
  className,
}) => {
  const theme = useTheme();
  const dispatch = useDispatch();
  
  // Redux state
  const {
    activeTab,
    drawerOpen,
    fullscreen,
    config,
    notifications,
  } = useSelector((state: RootState) => state.dashboard);

  // Local state
  const [causalGraph, setCausalGraph] = useState<CausalGraph | null>(initialGraph || null);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<string[]>([]);
  const [filterConfig, setFilterConfig] = useState<CausalFilterConfig>({
    strengthRange: [0, 1],
    confidenceThreshold: 0.5,
    processTypes: [],
    spatialScales: [],
    temporalScales: [],
    edgeTypes: [],
    validationStatus: [],
    discoveryMethods: [],
  });
  
  // Real data integration state
  const [availableDataSources] = useState<string[]>([
    'seismic', 'deformation', 'thermal', 'geochemical', 'gas_emissions'
  ]);
  const [selectedDataSources, setSelectedDataSources] = useState<string[]>(['seismic', 'deformation']);
  const [discoveryInProgress, setDiscoveryInProgress] = useState(false);
  const [discoveryProgress, setDiscoveryProgress] = useState(0);
  const [lastAnalysisResults, setLastAnalysisResults] = useState<CausalAnalysisResponse | null>(null);
  
  // Advanced analysis state
  // const [temporalModel] = useState<TemporalCausalModel | null>(null); // Unused
  // const [multiScaleModel] = useState<MultiScaleCausalModel | null>(null); // Unused
  const [, setHypothesisTests] = useState<any[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<CausalAlert[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<CausalMetrics | null>(null);
  
  // Real-time monitoring state
  // const [realTimeEnabledState, setRealTimeEnabledState] = useState(realTimeEnabled); // Unused
  const [wsConnected, setWsConnected] = useState(false);
  const wsUnsubscribe = useRef<(() => void) | null>(null);
  
  // Discovery configuration state
  const [discoveryConfig, setDiscoveryConfig] = useState<Partial<CausalDiscoveryRequest>>({
    algorithms: [CausalDiscoveryMethod.PC, CausalDiscoveryMethod.GES],
    confidenceThreshold: 0.05,
    maxLag: 24, // hours
  });
  
  // Spatial and temporal bounds
  const [spatialBounds] = useState<{
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  } | null>(null);
  const [timeRange, setTimeRange] = useState<[Date, Date]>([
    new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
    new Date()
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightedPaths] = useState<CausalPath[]>([]);
  const [interactionMode] = useState<InteractionMode>(InteractionMode.EXPLORE);
  const [renderMode] = useState<RenderMode>(RenderMode.WEBGL);
  const [layoutConfig] = useState<CausalLayoutConfig>({
    algorithm: LayoutAlgorithm.FORCE_DIRECTED,
    forceStrength: 0.5,
    linkDistance: 100,
    nodeRepulsion: 300,
    centerForce: 0.1,
    collisionRadius: 10,
    alphaDecay: 0.02,
    velocityDecay: 0.4,
    clustering: {
      enabled: true,
      clusterBy: 'processType',
      clusterStrength: 0.3,
    },
    hierarchical: {
      enabled: false,
      direction: 'TB',
      levelSeparation: 150,
      nodeSeparation: 100,
    },
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [realTimeStatus, setRealTimeStatus] = useState<'connected' | 'disconnected' | 'error'>('disconnected');
  const [performanceMetrics] = useState({
    fps: 60,
    renderTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    memoryUsage: 0,
  });

  // Dashboard tabs configuration
  const dashboardTabs: DashboardTab[] = useMemo(() => [
    {
      id: 'network',
      label: 'Causal Network',
      icon: <NetworkIcon />,
      component: (
        <CausalNetworkVisualization
          graph={causalGraph as any}
          width={window.innerWidth - (drawerOpen ? 280 : 80)}
          height={window.innerHeight - 200}
          layoutConfig={layoutConfig as any}
          onNodeClick={handleNodeClick as any}
          onEdgeClick={handleEdgeClick as any}
          onNodeHover={handleNodeHover as any}
          onEdgeHover={handleEdgeHover as any}
          onSelectionChange={handleSelectionChange}
          onLayoutComplete={handleLayoutComplete}
          filterConfig={filterConfig as any}
          searchQuery={searchQuery}
          highlightedPaths={highlightedPaths as any}
          interactionMode={interactionMode as any}
          renderMode={renderMode as any}
          showLabels={true}
          showConfidenceIntervals={true}
          showTemporalLags={true}
          animationEnabled={config.animationsEnabled}
        />
      ),
      badge: causalGraph?.nodes.length || 0,
    },
    {
      id: 'temporal',
      label: 'Temporal Analysis',
      icon: <TimelineIcon />,
      component: (
        <MultiScaleTemporalExplorer
          temporalData={null as any}
          initialConfig={{}}
          onPatternDetected={() => {}}
          onTimeRangeChange={handleTimeRangeChange}
        />
      ),
    },
    {
      id: 'monitor',
      label: 'Real-Time Monitor',
      icon: <MonitorIcon />,
      component: (
        <RealTimeProcessDashboard />
      ),
      badge: notifications.filter(n => n.type === 'warning' || n.type === 'error').length,
    },
    {
      id: 'hypothesis',
      label: 'Hypothesis Testing',
      icon: <HypothesisIcon />,
      component: (
        <HypothesisTestingInterface
          hypotheses={[]}
          onCreateHypothesis={() => {}}
          onTestHypothesis={handleHypothesisTest as any}
          onUpdateHypothesis={() => {}}
        />
      ),
    },
    {
      id: 'collaborative',
      label: 'Collaboration',
      icon: <CollaborativeIcon />,
      component: (
        <CollaborativeAnalysisInterface
          causalGraph={causalGraph}
          onGraphUpdate={handleCollaborativeUpdate}
          onAnnotation={handleAnnotation}
          onConflictResolution={handleConflictResolution}
          enabled={collaborativeMode}
        />
      ),
      disabled: !collaborativeMode,
    },
  ], [
    causalGraph,
    drawerOpen,
    layoutConfig,
    filterConfig,
    searchQuery,
    highlightedPaths,
    interactionMode,
    renderMode,
    config.animationsEnabled,
    realTimeEnabled,
    collaborativeMode,
    notifications,
  ]);

  // Initialize real-time connection
  useEffect(() => {
    if (realTimeEnabled && !wsConnected) {
      const unsubscribe = causalAnalysisApi.subscribeToRealTimeUpdates({
        onConnect: () => {
          setWsConnected(true);
          setRealTimeStatus('connected');
        },
        onDisconnect: () => {
          setWsConnected(false);
          setRealTimeStatus('disconnected');
        },
        onGraphUpdate: (update: RealTimeUpdate) => {
          if (update.graphId === causalGraph?.id) {
            // Handle real-time graph updates
            handleRealTimeGraphUpdate(update);
          }
        },
        onDiscoveryComplete: (_graphId: string, nodeCount: number, edgeCount: number) => {
          dispatch(addNotification({
            id: `discovery-complete-${Date.now()}`,
            type: 'success',
            title: 'Causal Discovery Complete',
            message: `Discovered ${nodeCount} nodes and ${edgeCount} relationships`,
            timestamp: new Date(),
            autoHide: true,
            duration: 5000,
          }));
        },
        onError: (error: string) => {
          setRealTimeStatus('error');
          dispatch(addNotification({
            id: `ws-error-${Date.now()}`,
            type: 'error',
            title: 'Real-time Connection Error',
            message: error,
            timestamp: new Date(),
            autoHide: false,
          }));
        },
      });
      
      wsUnsubscribe.current = unsubscribe;
    }
    
    return () => {
      if (wsUnsubscribe.current) {
        wsUnsubscribe.current();
        wsUnsubscribe.current = null;
      }
    };
  }, [realTimeEnabled, causalGraph?.id, dispatch]);

  // Load system metrics periodically
  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const metrics = await causalAnalysisApi.getMetrics();
        setSystemMetrics(metrics);
      } catch (error) {
        console.error('Failed to load system metrics:', error);
      }
    };

    loadMetrics();
    const interval = setInterval(loadMetrics, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Event handlers
  const handleNodeClick = useCallback((node: CausalNode) => {
    setSelectedNodes(prev => 
      prev.includes(node.id) 
        ? prev.filter(id => id !== node.id)
        : [...prev, node.id]
    );
    
    dispatch(addNotification({
      id: `node-${node.id}-${Date.now()}`,
      type: 'info',
      title: 'Node Selected',
      message: `Selected ${node.label} (${node.processType})`,
      timestamp: new Date(),
      autoHide: true,
      duration: 3000,
    }));
  }, [dispatch]);

  const handleRealTimeGraphUpdate = useCallback((update: RealTimeUpdate) => {
    if (update.changes && causalGraph) {
      // Apply real-time changes to the graph
      const updatedGraph = { ...causalGraph };
      
      update.changes.forEach(change => {
        switch (change.changeType) {
          case 'node_added':
            if (change.details.node) {
              updatedGraph.nodes.push(change.details.node);
            }
            break;
          case 'node_removed':
            updatedGraph.nodes = updatedGraph.nodes.filter(n => n.id !== change.elementId);
            break;
          case 'edge_added':
            if (change.details.edge) {
              updatedGraph.edges.push(change.details.edge);
            }
            break;
          case 'edge_removed':
            updatedGraph.edges = updatedGraph.edges.filter(e => e.id !== change.elementId);
            break;
          // case 'strength_changed': // Not in the enum, commenting out
          //   const edge = updatedGraph.edges.find(e => e.id === change.elementId);
          //   if (edge && change.details.newStrength !== undefined) {
          //     edge.causalStrength = change.details.newStrength;
          //   }
          //   break;
        }
      });
      
      setCausalGraph(updatedGraph);
    }
    
    // Handle alerts
    if (update.alerts) {
      setActiveAlerts(prev => [...prev, ...update.alerts!]);
      
      // Show high-severity alerts as notifications
      update.alerts.forEach(alert => {
        if (alert.severity === AlertSeverity.ERROR || alert.severity === AlertSeverity.CRITICAL) {
          dispatch(addNotification({
            id: alert.id,
            type: alert.severity === AlertSeverity.CRITICAL ? 'error' : 'warning',
            title: alert.title,
            message: alert.message,
            timestamp: alert.timestamp,
            autoHide: alert.severity !== AlertSeverity.CRITICAL,
            duration: 10000,
          }));
        }
      });
    }
  }, [causalGraph, dispatch]);

  const handleDiscoverCausalRelationships = useCallback(async () => {
    if (!selectedDataSources.length || !timeRange) {
      dispatch(addNotification({
        id: `discovery-error-${Date.now()}`,
        type: 'error',
        title: 'Discovery Configuration Error',
        message: 'Please select data sources and time range',
        timestamp: new Date(),
        autoHide: true,
        duration: 5000,
      }));
      return;
    }

    setDiscoveryInProgress(true);
    setDiscoveryProgress(0);

    try {
      const request: CausalDiscoveryRequest = {
        dataSources: selectedDataSources,
        timeRange: timeRange,
        spatialBounds: spatialBounds || undefined,
        algorithms: discoveryConfig.algorithms || [CausalDiscoveryMethod.PC],
        confidenceThreshold: discoveryConfig.confidenceThreshold || 0.05,
        maxLag: discoveryConfig.maxLag,
        backgroundKnowledge: discoveryConfig.backgroundKnowledge,
      };

      const response = await causalAnalysisApi.discoverCausalRelationships(
        request,
        (progress) => setDiscoveryProgress(progress)
      );

      setCausalGraph(response.graph);
      setLastAnalysisResults(response);
      
      dispatch(addNotification({
        id: `discovery-success-${Date.now()}`,
        type: 'success',
        title: 'Causal Discovery Successful',
        message: `Discovered ${response.graph.nodes.length} nodes and ${response.graph.edges.length} causal relationships`,
        timestamp: new Date(),
        autoHide: true,
        duration: 5000,
      }));

    } catch (error) {
      dispatch(addNotification({
        id: `discovery-error-${Date.now()}`,
        type: 'error',
        title: 'Causal Discovery Failed',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        timestamp: new Date(),
        autoHide: false,
      }));
    } finally {
      setDiscoveryInProgress(false);
      setDiscoveryProgress(0);
    }
  }, [selectedDataSources, timeRange, spatialBounds, discoveryConfig, dispatch]);

  const handleExportGraph = useCallback(async (format: 'json' | 'csv' | 'png' | 'svg') => {
    if (!causalGraph) {
      dispatch(addNotification({
        id: `export-error-${Date.now()}`,
        type: 'error',
        title: 'Export Error',
        message: 'No causal graph to export',
        timestamp: new Date(),
        autoHide: true,
        duration: 3000,
      }));
      return;
    }

    try {
      const blob = await causalAnalysisApi.exportGraph(causalGraph.id, format);
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `causal-graph-${causalGraph.id}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      dispatch(addNotification({
        id: `export-success-${Date.now()}`,
        type: 'success',
        title: 'Export Successful',
        message: `Causal graph exported as ${format.toUpperCase()}`,
        timestamp: new Date(),
        autoHide: true,
        duration: 3000,
      }));

    } catch (error) {
      dispatch(addNotification({
        id: `export-error-${Date.now()}`,
        type: 'error',
        title: 'Export Failed',
        message: error instanceof Error ? error.message : 'Export failed',
        timestamp: new Date(),
        autoHide: true,
        duration: 5000,
      }));
    }
  }, [causalGraph, dispatch]);

  const handleTestHypothesis = useCallback(async (hypothesisGraph: CausalGraph) => {
    try {
      const request: HypothesisTestRequest = {
        hypothesisGraph,
        testData: {}, // Would be populated with actual test data
        testMethods: ['bootstrap', 'permutation'],
        significanceLevel: 0.05,
      };

      const result = await causalAnalysisApi.testHypothesis(request);
      
      setHypothesisTests(prev => [...prev, result]);
      
      dispatch(addNotification({
        id: `hypothesis-test-${Date.now()}`,
        type: 'success',
        title: 'Hypothesis Test Complete',
        message: result.conclusion,
        timestamp: new Date(),
        autoHide: true,
        duration: 5000,
      }));

    } catch (error) {
      dispatch(addNotification({
        id: `hypothesis-error-${Date.now()}`,
        type: 'error',
        title: 'Hypothesis Test Failed',
        message: error instanceof Error ? error.message : 'Test failed',
        timestamp: new Date(),
        autoHide: true,
        duration: 5000,
      }));
    }
  }, [dispatch]);

  const handleEdgeClick = useCallback((edge: CausalEdge) => {
    setSelectedEdges(prev => 
      prev.includes(edge.id) 
        ? prev.filter(id => id !== edge.id)
        : [...prev, edge.id]
    );
  }, []);

  const handleNodeHover = useCallback((_node: CausalNode | null) => {
    // Handle node hover for tooltips and highlighting
  }, []);

  const handleEdgeHover = useCallback((_edge: CausalEdge | null) => {
    // Handle edge hover for tooltips and highlighting
  }, []);

  const handleSelectionChange = useCallback((nodes: string[], edges: string[]) => {
    setSelectedNodes(nodes);
    setSelectedEdges(edges);
  }, []);

  const handleLayoutComplete = useCallback(() => {
    setIsLoading(false);
    dispatch(addNotification({
      id: `layout-complete-${Date.now()}`,
      type: 'success',
      title: 'Layout Complete',
      message: 'Causal network layout has been updated',
      timestamp: new Date(),
      autoHide: true,
      duration: 2000,
    }));
  }, [dispatch]);

  const handleTimeRangeChange = useCallback((_start: Date, _end: Date) => {
    // Handle temporal range changes
  }, []);

  // const handleScaleChange = useCallback((_scale: string) => { // Unused
  //   // Handle temporal scale changes
  // }, []);

  // const handleTemporalFilterChange = useCallback((_filter: any) => { // Unused
  //   // Handle temporal filter changes
  // }, []);

  // const handleRealTimeAlert = useCallback((alert: any) => { // Unused function
  //   dispatch(addNotification({
  //     id: `alert-${Date.now()}`,
  //     type: alert.severity || 'warning',
  //     title: alert.title || 'Real-Time Alert',
  //     message: alert.message,
  //     timestamp: new Date(),
  //     autoHide: alert.severity !== 'error',
  //     duration: alert.severity === 'error' ? undefined : 5000,
  //   }));
  // }, [dispatch]);

  const handleHypothesisTest = useCallback((_hypothesis: any) => {
    // Handle hypothesis testing
  }, []);

  // const handleModelUpdate = useCallback((_model: any) => { // Unused
  //   // Handle model updates from hypothesis testing
  // }, []);

  const handleCollaborativeUpdate = useCallback((update: any) => {
    // Handle collaborative graph updates
    setCausalGraph(update.graph);
  }, []);

  const handleAnnotation = useCallback((_annotation: any) => {
    // Handle collaborative annotations
  }, []);

  const handleConflictResolution = useCallback((_resolution: any) => {
    // Handle collaborative conflict resolution
  }, []);

  // const handleTabChange = useCallback((_event: React.SyntheticEvent, newValue: string) => { // Unused
  //   dispatch(setActiveTab(newValue));
  // }, [dispatch]);

  const handleDrawerToggle = useCallback(() => {
    dispatch(setDrawerOpen(!drawerOpen));
  }, [dispatch, drawerOpen]);

  const handleFullscreenToggle = useCallback(() => {
    dispatch(setFullscreen(!fullscreen));
    
    if (!fullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, [dispatch, fullscreen]);

  const handleExport = useCallback((format: string) => {
    if (onExport) {
      const exportData = {
        graph: causalGraph,
        config: { layoutConfig, filterConfig },
        selection: { nodes: selectedNodes, edges: selectedEdges },
        timestamp: new Date(),
      };
      onExport(format, exportData);
    }
    setExportDialogOpen(false);
  }, [onExport, causalGraph, layoutConfig, filterConfig, selectedNodes, selectedEdges]);

  const handleShare = useCallback((shareConfig: any) => {
    if (onShare) {
      onShare({
        ...shareConfig,
        graph: causalGraph,
        config: { layoutConfig, filterConfig },
        timestamp: new Date(),
      });
    }
    setShareDialogOpen(false);
  }, [onShare, causalGraph, layoutConfig, filterConfig]);

  const handleRefresh = useCallback(async () => {
    setIsLoading(true);
    try {
      // Refresh causal graph data
      // This would typically fetch from an API
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      
      dispatch(addNotification({
        id: `refresh-${Date.now()}`,
        type: 'success',
        title: 'Data Refreshed',
        message: 'Causal graph data has been updated',
        timestamp: new Date(),
        autoHide: true,
        duration: 2000,
      }));
    } catch (error) {
      setError('Failed to refresh data');
      dispatch(addNotification({
        id: `refresh-error-${Date.now()}`,
        type: 'error',
        title: 'Refresh Failed',
        message: 'Failed to update causal graph data',
        timestamp: new Date(),
        autoHide: false,
      }));
    } finally {
      setIsLoading(false);
    }
  }, [dispatch]);

  // Speed dial actions
  const speedDialActions = [
    {
      icon: <NetworkIcon />,
      name: 'Discover Relationships',
      onClick: handleDiscoverCausalRelationships,
      disabled: discoveryInProgress || !selectedDataSources.length,
    },
    {
      icon: <RefreshIcon />,
      name: 'Refresh Data',
      onClick: handleRefresh,
    },
    {
      icon: <ShareIcon />,
      name: 'Share Dashboard',
      onClick: () => setShareDialogOpen(true),
    },
    {
      icon: <DownloadIcon />,
      name: 'Export Graph',
      onClick: () => setExportDialogOpen(true),
      disabled: !causalGraph,
    },
    {
      icon: fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />,
      name: fullscreen ? 'Exit Fullscreen' : 'Fullscreen',
      onClick: handleFullscreenToggle,
    },
    {
      icon: <SettingsIcon />,
      name: 'Settings',
      onClick: () => setSettingsOpen(true),
    },
  ];

  // Render notification icon with badge
  const renderNotificationIcon = () => {
    const unreadCount = notifications.filter(n => !n.read).length;
    return (
      <Badge badgeContent={unreadCount} color="error">
        <NotificationsIcon />
      </Badge>
    );
  };

  // Render performance indicator
  const renderPerformanceIndicator = () => {
    const getPerformanceColor = (fps: number) => {
      if (fps >= 50) return 'success';
      if (fps >= 30) return 'warning';
      return 'error';
    };

    return (
      <Tooltip title={`FPS: ${performanceMetrics.fps}, Nodes: ${performanceMetrics.nodeCount}, Edges: ${performanceMetrics.edgeCount}`}>
        <Chip
          size="small"
          label={`${performanceMetrics.fps} FPS`}
          color={getPerformanceColor(performanceMetrics.fps)}
          variant="outlined"
        />
      </Tooltip>
    );
  };

  // Render real-time status indicator
  const renderRealTimeStatus = () => {
    const statusConfig = {
      connected: { color: 'success', icon: <SuccessIcon />, label: 'Connected' },
      disconnected: { color: 'warning', icon: <WarningIcon />, label: 'Disconnected' },
      error: { color: 'error', icon: <ErrorIcon />, label: 'Error' },
    };

    const config = statusConfig[realTimeStatus];
    
    return (
      <Tooltip title={`Real-time status: ${config.label}`}>
        <Chip
          size="small"
          icon={config.icon}
          label={config.label}
          color={config.color as any}
          variant="outlined"
        />
      </Tooltip>
    );
  };

  return (
    <Box className={className} sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* App Bar */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: theme.zIndex.drawer + 1,
          backgroundColor: alpha(theme.palette.background.paper, 0.9),
          backdropFilter: 'blur(10px)',
          color: theme.palette.text.primary,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Comprehensive Causal Dashboard
          </Typography>

          {/* Status indicators */}
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mr: 2 }}>
            {renderPerformanceIndicator()}
            {realTimeEnabled && renderRealTimeStatus()}
          </Box>

          {/* Action buttons */}
          <IconButton color="inherit" onClick={() => setSettingsOpen(true)}>
            <SettingsIcon />
          </IconButton>
          
          <IconButton color="inherit">
            {renderNotificationIcon()}
          </IconButton>
        </Toolbar>
        
        {isLoading && <LinearProgress />}
      </AppBar>

      {/* Navigation Drawer */}
      <Drawer
        variant="permanent"
        open={drawerOpen}
        sx={{
          width: drawerOpen ? 280 : 60,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerOpen ? 280 : 60,
            boxSizing: 'border-box',
            transition: theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
            overflowX: 'hidden',
          },
        }}
      >
        <Toolbar />
        
        <Box sx={{ overflow: 'auto', flex: 1 }}>
          <List>
            {dashboardTabs.map((tab) => (
              <ListItem
                key={tab.id}
                button
                selected={activeTab === tab.id}
                onClick={() => dispatch(setActiveTab(tab.id))}
                disabled={tab.disabled}
                sx={{
                  minHeight: 48,
                  justifyContent: drawerOpen ? 'initial' : 'center',
                  px: 2.5,
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 0,
                    mr: drawerOpen ? 3 : 'auto',
                    justifyContent: 'center',
                  }}
                >
                  <Badge badgeContent={tab.badge} color="primary">
                    {tab.icon}
                  </Badge>
                </ListItemIcon>
                <ListItemText 
                  primary={tab.label} 
                  sx={{ opacity: drawerOpen ? 1 : 0 }} 
                />
              </ListItem>
            ))}
          </List>
          
          <Divider />
          
          {/* Discovery Configuration and Filter Panel */}
          {drawerOpen && (
            <Box sx={{ p: 2 }}>
              {/* Real Data Discovery Configuration */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle2">Causal Discovery</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {/* Data Sources Selection */}
                    <Grid item xs={12}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Data Sources</InputLabel>
                        <Select
                          multiple
                          value={selectedDataSources}
                          onChange={(e) => setSelectedDataSources(e.target.value as string[])}
                          renderValue={(selected) => (
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                              {selected.map((value) => (
                                <Chip key={value} label={value} size="small" />
                              ))}
                            </Box>
                          )}
                        >
                          {availableDataSources.map((source) => (
                            <MenuItem key={source} value={source}>
                              {source.replace('_', ' ').toUpperCase()}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    {/* Time Range */}
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="Start Date"
                        type="datetime-local"
                        value={timeRange[0].toISOString().slice(0, 16)}
                        onChange={(e) => setTimeRange([new Date(e.target.value), timeRange[1]])}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="End Date"
                        type="datetime-local"
                        value={timeRange[1].toISOString().slice(0, 16)}
                        onChange={(e) => setTimeRange([timeRange[0], new Date(e.target.value)])}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>

                    {/* Algorithms Selection */}
                    <Grid item xs={12}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Algorithms</InputLabel>
                        <Select
                          multiple
                          value={discoveryConfig.algorithms || []}
                          onChange={(e) => setDiscoveryConfig(prev => ({
                            ...prev,
                            algorithms: e.target.value as CausalDiscoveryMethod[]
                          }))}
                          renderValue={(selected) => (
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                              {selected.map((value) => (
                                <Chip key={value} label={value.toUpperCase()} size="small" />
                              ))}
                            </Box>
                          )}
                        >
                          {Object.values(CausalDiscoveryMethod).map((method) => (
                            <MenuItem key={method} value={method}>
                              {method.toUpperCase()}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    {/* Confidence Threshold */}
                    <Grid item xs={12}>
                      <Typography variant="caption">
                        Confidence Threshold: {discoveryConfig.confidenceThreshold}
                      </Typography>
                      <Slider
                        size="small"
                        value={discoveryConfig.confidenceThreshold || 0.05}
                        onChange={(_: Event, value: number | number[]) => setDiscoveryConfig(prev => ({
                          ...prev,
                          confidenceThreshold: value as number
                        }))}
                        min={0.001}
                        max={0.1}
                        step={0.001}
                        valueLabelDisplay="auto"
                      />
                    </Grid>

                    {/* Max Temporal Lag */}
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        size="small"
                        label="Max Temporal Lag (hours)"
                        type="number"
                        value={discoveryConfig.maxLag || 24}
                        onChange={(e) => setDiscoveryConfig(prev => ({
                          ...prev,
                          maxLag: parseInt(e.target.value)
                        }))}
                      />
                    </Grid>

                    {/* Discovery Button */}
                    <Grid item xs={12}>
                      <Button
                        fullWidth
                        variant="contained"
                        onClick={handleDiscoverCausalRelationships}
                        disabled={discoveryInProgress || !selectedDataSources.length}
                        startIcon={discoveryInProgress ? <CircularProgress size={16} /> : <NetworkIcon />}
                      >
                        {discoveryInProgress ? 'Discovering...' : 'Discover Relationships'}
                      </Button>
                      {discoveryInProgress && (
                        <LinearProgress 
                          variant="determinate" 
                          value={discoveryProgress} 
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Search and Filter Panel */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle2">Search & Filter</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <CausalSearchInterface
                    onSearch={setSearchQuery}
                    onFilterChange={setFilterConfig}
                    currentFilter={filterConfig}
                  />
                  
                  <CausalFilterPanel
                    filterConfig={filterConfig}
                    onFilterChange={setFilterConfig}
                    availableProcessTypes={Object.values(causalGraph?.nodes.reduce((acc, node) => {
                      acc[node.processType] = true;
                      return acc;
                    }, {} as Record<string, boolean>) || {})}
                  />
                </AccordionDetails>
              </Accordion>

              {/* System Status */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle2">System Status</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={1}>
                    <Grid item xs={12}>
                      <Chip
                        size="small"
                        icon={wsConnected ? <SuccessIcon /> : <ErrorIcon />}
                        label={`Real-time: ${wsConnected ? 'Connected' : 'Disconnected'}`}
                        color={wsConnected ? 'success' : 'error'}
                        variant="outlined"
                      />
                    </Grid>
                    {systemMetrics && (
                      <>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            Active Graphs: {systemMetrics.activeGraphs}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            CPU: {systemMetrics.systemPerformance.cpuUsage.toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            Memory: {systemMetrics.systemPerformance.memoryUsage.toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption">
                            Avg Discovery: {systemMetrics.averageDiscoveryTime.toFixed(1)}s
                          </Typography>
                        </Grid>
                      </>
                    )}
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Active Alerts */}
              {activeAlerts.length > 0 && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2">
                      Active Alerts ({activeAlerts.length})
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {activeAlerts.slice(0, 5).map((alert) => (
                        <ListItem key={alert.id}>
                          <ListItemIcon>
                            {alert.severity === AlertSeverity.CRITICAL ? <ErrorIcon color="error" /> :
                             alert.severity === AlertSeverity.ERROR ? <ErrorIcon color="warning" /> :
                             alert.severity === AlertSeverity.WARNING ? <WarningIcon color="warning" /> :
                             <InfoIcon color="info" />}
                          </ListItemIcon>
                          <ListItemText
                            primary={alert.title}
                            secondary={alert.message}
                            primaryTypographyProps={{ variant: 'caption' }}
                            secondaryTypographyProps={{ variant: 'caption' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          )}
        </Box>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 0,
          mt: 8,
          height: 'calc(100vh - 64px)',
          overflow: 'hidden',
        }}
      >
        {error && (
          <Alert 
            severity="error" 
            onClose={() => setError(null)}
            sx={{ m: 2 }}
          >
            {error}
          </Alert>
        )}

        {/* Tab Content */}
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {dashboardTabs.find(tab => tab.id === activeTab)?.component}
        </Box>
      </Box>

      {/* Speed Dial */}
      <SpeedDial
        ariaLabel="Dashboard Actions"
        sx={{ position: 'absolute', bottom: 16, right: 16 }}
        icon={<SpeedDialIcon />}
      >
        {speedDialActions.map((action) => (
          <SpeedDialAction
            key={action.name}
            icon={action.icon}
            tooltipTitle={action.name}
            onClick={action.onClick}
            disabled={action.disabled}
          />
        ))}
      </SpeedDial>

      {/* Notifications */}
      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={notification.autoHide ? notification.duration : null}
          onClose={() => dispatch(removeNotification(notification.id))}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert 
            severity={notification.type}
            onClose={() => dispatch(removeNotification(notification.id))}
          >
            <Typography variant="subtitle2">{notification.title}</Typography>
            <Typography variant="body2">{notification.message}</Typography>
          </Alert>
        </Snackbar>
      ))}

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Dashboard Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.realTimeUpdates}
                    onChange={(e) => dispatch(updateDashboardConfig({ realTimeUpdates: e.target.checked }))}
                  />
                }
                label="Real-time Updates"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.animationsEnabled}
                    onChange={(e) => dispatch(updateDashboardConfig({ animationsEnabled: e.target.checked }))}
                  />
                }
                label="Animations"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Performance Mode</InputLabel>
                <Select
                  value={config.performanceMode}
                  onChange={(e) => dispatch(updateDashboardConfig({ performanceMode: e.target.value as any }))}
                >
                  <MenuItem value="high">High Performance</MenuItem>
                  <MenuItem value="balanced">Balanced</MenuItem>
                  <MenuItem value="low">Low Performance</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Render Quality</InputLabel>
                <Select
                  value={config.renderQuality}
                  onChange={(e) => dispatch(updateDashboardConfig({ renderQuality: e.target.value as any }))}
                >
                  <MenuItem value="high">High Quality</MenuItem>
                  <MenuItem value="medium">Medium Quality</MenuItem>
                  <MenuItem value="low">Low Quality</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button onClick={() => setSettingsOpen(false)} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Export Causal Graph</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Choose the format to export your causal graph data and visualization.
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Button 
                fullWidth 
                variant="outlined"
                onClick={() => handleExportGraph('json')}
                disabled={!causalGraph}
              >
                JSON Data
              </Button>
            </Grid>
            <Grid item xs={6}>
              <Button 
                fullWidth 
                variant="outlined"
                onClick={() => handleExportGraph('csv')}
                disabled={!causalGraph}
              >
                CSV Data
              </Button>
            </Grid>
            <Grid item xs={6}>
              <Button 
                fullWidth 
                variant="outlined"
                onClick={() => handleExportGraph('png')}
                disabled={!causalGraph}
              >
                PNG Image
              </Button>
            </Grid>
            <Grid item xs={6}>
              <Button 
                fullWidth 
                variant="outlined"
                onClick={() => handleExportGraph('svg')}
                disabled={!causalGraph}
              >
                SVG Vector
              </Button>
            </Grid>
          </Grid>
          
          {lastAnalysisResults && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2">Analysis Summary</Typography>
              <Typography variant="caption" display="block">
                Nodes: {lastAnalysisResults.graph.nodes.length}
              </Typography>
              <Typography variant="caption" display="block">
                Edges: {lastAnalysisResults.graph.edges.length}
              </Typography>
              <Typography variant="caption" display="block">
                Algorithms: {lastAnalysisResults.graph.metadata.algorithmsUsed.join(', ')}
              </Typography>
              <Typography variant="caption" display="block">
                Confidence: {lastAnalysisResults.graph.metadata.confidenceThreshold}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      {/* Share Dialog */}
      <Dialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)}>
        <DialogTitle>Share Dashboard</DialogTitle>
        <DialogContent>
          <Typography>Share options will be implemented here</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => handleShare({})} variant="contained">Share</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ComprehensiveCausalDashboard;