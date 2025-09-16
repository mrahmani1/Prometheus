import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { Box, Container, Typography, Paper, Grid, Card, CardContent, Button, CircularProgress, Alert } from '@mui/material';
import { CausalNetworkVisualization } from './CausalNetworkVisualization';
import { CausalNetworkControls } from './CausalNetworkControls';
import { CausalPathwayExplorer } from './CausalPathwayExplorer';
import DatabaseService from '../../../services/DatabaseService';
import {
  CausalGraph,
  CausalNode,
  CausalEdge,
  CausalLayoutConfig,
  CausalFilterConfig,
  CausalPath,
  InteractionMode,
  RenderMode,
  // GeologicalProcessType, SpatialScale, TemporalScale, CausalEdgeType, ValidationStatus - unused
} from './types';

export const CausalDashboardDemo: React.FC = () => {
  // Real data state
  const [causalGraph, setCausalGraph] = useState<CausalGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const databaseService = DatabaseService.getInstance();

  // Demo state
  const [layoutConfig, setLayoutConfig] = useState<CausalLayoutConfig>({
    algorithm: 'force-directed',
    forceStrength: 0.5,
    linkDistance: 120,
    nodeRepulsion: 400,
    centerForce: 0.1,
    collisionRadius: 8,
    alphaDecay: 0.0228,
    velocityDecay: 0.4,
    clustering: {
      enabled: true,
      clusterBy: 'processType',
      clusterStrength: 0.3
    },
    hierarchical: {
      enabled: false,
      direction: 'TB',
      levelSeparation: 100,
      nodeSeparation: 50
    }
  });

  const [filterConfig, _setFilterConfig] = useState<CausalFilterConfig>({
    strengthRange: [0.2, 1.0],
    confidenceThreshold: 0.5,
    processTypes: [],
    spatialScales: [],
    temporalScales: [],
    edgeTypes: [],
    validationStatus: ['validated', 'pending'],
    discoveryMethods: []
  });

  const [interactionMode, setInteractionMode] = useState<InteractionMode>('explore');
  const [renderMode, setRenderMode] = useState<RenderMode>('canvas');
  const [showLabels, setShowLabels] = useState(true);
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);
  const [showTemporalLags, setShowTemporalLags] = useState(true);
  const [animationEnabled, setAnimationEnabled] = useState(true);
  const [isLayoutRunning, setIsLayoutRunning] = useState(false);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<string[]>([]);
  const [highlightedPaths, setHighlightedPaths] = useState<CausalPath[]>([]);
  const [searchQuery, _setSearchQuery] = useState('');
  const [showControls, setShowControls] = useState(false);
  const [showPathExplorer, setShowPathExplorer] = useState(false);

  // Load real causal graph data from database
  useEffect(() => {
    const loadCausalGraph = async () => {
      try {
        setLoading(true);
        setError(null);

        // Query real causal nodes from Neo4j
        const causalNodes = await databaseService.queryCausalNodes({
          processTypes: ['seismic', 'magmatic', 'deformation', 'geochemical', 'thermal', 'hydrothermal', 'tectonic'],
          confidenceThreshold: 0.5,
          dataSources: ['IRIS/EarthScope', 'USGS Volcano Hazards Program', 'ASF-Hyp3 MT-SAR', 'NASA FIRMS', 'Volcanic Observatory Network', 'GPS Network']
        });

        // Query real causal relationships from Neo4j
        const causalRelationships = await databaseService.queryCausalRelationships({
          strengthThreshold: 0.3,
          confidenceThreshold: 0.5,
          validationStatus: ['validated', 'pending', 'uncertain']
        });

        // Transform database records to CausalGraph format
        const nodes: CausalNode[] = causalNodes.map(node => ({
          id: node.id,
          label: node.label,
          processType: node.process_type as any,
          position: {
            x: node.longitude ? (node.longitude + 122) * 400 : calculateDeterministicPosition(node.id, 'x', 600, 100),
            y: node.latitude ? (45 - node.latitude) * 400 : calculateDeterministicPosition(node.id, 'y', 400, 100)
          },
          size: Math.max(15, Math.min(35, node.confidence * 40)),
          color: getProcessTypeColor(node.process_type),
          metadata: {
            description: node.metadata.description || `${node.process_type} process`,
            dataSource: node.data_source,
            lastUpdated: node.updated_at,
            confidence: node.confidence,
            spatialScale: (node.spatial_scale as any) || 'local',
            temporalScale: (node.temporal_scale as any) || 'hours'
          },
          properties: {
            geologicalContext: node.metadata.geologicalContext || 'unknown',
            ...node.metadata
          }
        }));

        const edges: CausalEdge[] = causalRelationships.map(edge => ({
          id: edge.id,
          source: edge.source_node_id,
          target: edge.target_node_id,
          strength: edge.strength,
          confidence: [edge.confidence_lower, edge.confidence_upper],
          temporalLag: edge.temporal_lag || 0,
          edgeType: edge.relationship_type as any,
          metadata: {
            discoveryMethod: edge.discovery_method,
            statisticalSignificance: edge.statistical_significance,
            validationStatus: (edge.validation_status as any) || 'pending',
            geologicalContext: edge.evidence.geologicalContext || 'unknown'
          },
          properties: {
            mechanism: edge.evidence.mechanism || 'unknown',
            evidenceStrength: edge.evidence.evidenceStrength || 'unknown'
          }
        }));

        const graph: CausalGraph = {
          nodes,
          edges,
          metadata: {
            createdAt: new Date(),
            lastUpdated: new Date(),
            totalNodes: nodes.length,
            totalEdges: edges.length,
            averageCausalStrength: edges.length > 0 ? edges.reduce((sum, e) => sum + e.strength, 0) / edges.length : 0,
            spatialExtent: {
              north: nodes.length > 0 ? Math.max(...nodes.filter(n => n.position).map(n => n.position!.y)) : 45.0,
              south: nodes.length > 0 ? Math.min(...nodes.filter(n => n.position).map(n => n.position!.y)) : 44.0,
              east: nodes.length > 0 ? Math.max(...nodes.filter(n => n.position).map(n => n.position!.x)) : -121.0,
              west: nodes.length > 0 ? Math.min(...nodes.filter(n => n.position).map(n => n.position!.x)) : -122.0
            },
            temporalExtent: {
              start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
              end: new Date()
            }
          }
        };

        setCausalGraph(graph);
      } catch (err) {
        console.error('Failed to load causal graph:', err);
        setError(err instanceof Error ? err.message : 'Failed to load causal graph data');
      } finally {
        setLoading(false);
      }
    };

    loadCausalGraph();
  }, [databaseService]);

  // Helper function to get color based on process type
  const getProcessTypeColor = (processType: string): string => {
    const colorMap: Record<string, string> = {
      seismic: '#e74c3c',
      magmatic: '#e67e22',
      deformation: '#f39c12',
      geochemical: '#27ae60',
      thermal: '#e74c3c',
      hydrothermal: '#16a085',
      tectonic: '#34495e'
    };
    return colorMap[processType] || '#95a5a6';
  };

  // Deterministic positioning based on node properties instead of random
  const calculateDeterministicPosition = (nodeId: string, axis: 'x' | 'y', range: number, offset: number): number => {
    // Create deterministic hash from node ID
    let hash = 0;
    for (let i = 0; i < nodeId.length; i++) {
      const char = nodeId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Use different hash seeds for x and y axes
    const seed = axis === 'x' ? hash : hash * 31;
    const normalized = Math.abs(seed) / 2147483647; // Normalize to 0-1
    
    return normalized * range + offset;
  };

  // Use real data or fallback to empty graph
  const displayGraph = useMemo((): CausalGraph => {
    if (causalGraph) {
      return causalGraph;
    }

    // Return empty graph structure if no data loaded yet
    return {
      nodes: [],
      edges: [],
      metadata: {
        createdAt: new Date(),
        lastUpdated: new Date(),
        totalNodes: 0,
        totalEdges: 0,
        averageCausalStrength: 0,
        spatialExtent: { north: 45.0, south: 44.0, east: -121.0, west: -122.0 },
        temporalExtent: { start: new Date(), end: new Date() }
      }
    };
  }, [causalGraph]);

  // Event handlers
  const handleNodeClick = useCallback((node: CausalNode) => {
    console.log('Node clicked:', node);
  }, []);

  const handleEdgeClick = useCallback((edge: CausalEdge) => {
    console.log('Edge clicked:', edge);
  }, []);

  const handleSelectionChange = useCallback((nodes: string[], edges: string[]) => {
    setSelectedNodes(nodes);
    setSelectedEdges(edges);
  }, []);

  const handleLayoutComplete = useCallback(() => {
    console.log('Layout computation complete');
    setIsLayoutRunning(false);
  }, []);

  const handlePathHighlight = useCallback((paths: CausalPath[]) => {
    setHighlightedPaths(paths);
  }, []);

  if (loading) {
    return (
      <Container maxWidth={false} sx={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box textAlign="center">
          <CircularProgress size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            Loading Real Causal Network Data...
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Querying Neo4j graph database for causal relationships
          </Typography>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth={false} sx={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Alert severity="error" sx={{ maxWidth: 600 }}>
          <Typography variant="h6">Failed to Load Causal Network</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth={false} sx={{ height: '100vh', p: 2 }}>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Real-Time Causal Network Visualization
        </Typography>
        <Typography variant="subtitle1" color="textSecondary" gutterBottom>
          Live geological process causality analysis from Neo4j graph database
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
          <Button
            variant={showControls ? 'contained' : 'outlined'}
            onClick={() => setShowControls(!showControls)}
          >
            Network Controls
          </Button>
          <Button
            variant={showPathExplorer ? 'contained' : 'outlined'}
            onClick={() => setShowPathExplorer(!showPathExplorer)}
          >
            Pathway Explorer
          </Button>
          <Button
            variant="outlined"
            onClick={() => setLayoutConfig(prev => ({ ...prev, algorithm: 'force-directed' }))}
          >
            Force Layout
          </Button>
          <Button
            variant="outlined"
            onClick={() => setLayoutConfig(prev => ({ ...prev, algorithm: 'hierarchical' }))}
          >
            Hierarchical Layout
          </Button>
        </Box>
      </Box>

      <Grid container spacing={2} sx={{ height: 'calc(100vh - 120px)' }}>
        {/* Main Visualization */}
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ height: '100%', position: 'relative' }}>
            <CausalNetworkVisualization
              graph={displayGraph}
              width={800}
              height={600}
              layoutConfig={layoutConfig}
              onNodeClick={handleNodeClick}
              onEdgeClick={handleEdgeClick}
              onSelectionChange={handleSelectionChange}
              onLayoutComplete={handleLayoutComplete}
              filterConfig={filterConfig}
              searchQuery={searchQuery}
              highlightedPaths={highlightedPaths}
              interactionMode={interactionMode}
              renderMode={renderMode}
              showLabels={showLabels}
              showConfidenceIntervals={showConfidenceIntervals}
              showTemporalLags={showTemporalLags}
              animationEnabled={animationEnabled}
            />

            {/* Network Controls */}
            {showControls && (
              <CausalNetworkControls
                layoutConfig={layoutConfig}
                interactionMode={interactionMode}
                renderMode={renderMode}
                showLabels={showLabels}
                showConfidenceIntervals={showConfidenceIntervals}
                showTemporalLags={showTemporalLags}
                animationEnabled={animationEnabled}
                isLayoutRunning={isLayoutRunning}
                onLayoutConfigChange={setLayoutConfig}
                onInteractionModeChange={setInteractionMode}
                onRenderModeChange={setRenderMode}
                onShowLabelsChange={setShowLabels}
                onShowConfidenceIntervalsChange={setShowConfidenceIntervals}
                onShowTemporalLagsChange={setShowTemporalLags}
                onAnimationEnabledChange={setAnimationEnabled}
                onLayoutStart={() => setIsLayoutRunning(true)}
                onLayoutStop={() => setIsLayoutRunning(false)}
                onLayoutReset={() => {
                  setIsLayoutRunning(true);
                  setTimeout(() => setIsLayoutRunning(false), 2000);
                }}
                onClose={() => setShowControls(false)}
              />
            )}

            {/* Pathway Explorer */}
            {showPathExplorer && (
              <CausalPathwayExplorer
                graph={displayGraph}
                selectedNodes={selectedNodes}
                onPathHighlight={handlePathHighlight}
                onClose={() => setShowPathExplorer(false)}
              />
            )}
          </Paper>
        </Grid>

        {/* Information Panel */}
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
            {/* Network Statistics */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Network Statistics
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Nodes: {displayGraph.nodes.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Edges: {displayGraph.edges.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Avg. Causal Strength: {displayGraph.metadata.averageCausalStrength.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Selected Nodes: {selectedNodes.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Selected Edges: {selectedEdges.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Highlighted Paths: {highlightedPaths.length}
                </Typography>
              </CardContent>
            </Card>

            {/* Process Types Legend */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Geological Process Types
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {Array.from(new Set(displayGraph.nodes.map(n => n.processType))).map(type => (
                    <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          borderRadius: '50%',
                          bgcolor: displayGraph.nodes.find(n => n.processType === type)?.color || '#ccc'
                        }}
                      />
                      <Typography variant="body2">
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>

            {/* Current Selection */}
            {(selectedNodes.length > 0 || selectedEdges.length > 0) && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Current Selection
                  </Typography>
                  {selectedNodes.length > 0 && (
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="subtitle2">Selected Nodes:</Typography>
                      {selectedNodes.map(nodeId => {
                        const node = displayGraph.nodes.find(n => n.id === nodeId);
                        return (
                          <Typography key={nodeId} variant="body2" color="textSecondary">
                            • {node?.label || nodeId}
                          </Typography>
                        );
                      })}
                    </Box>
                  )}
                  {selectedEdges.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2">Selected Edges:</Typography>
                      {selectedEdges.map(edgeId => {
                        const edge = displayGraph.edges.find(e => e.id === edgeId);
                        const sourceNode = displayGraph.nodes.find(n => n.id === edge?.source);
                        const targetNode = displayGraph.nodes.find(n => n.id === edge?.target);
                        return (
                          <Typography key={edgeId} variant="body2" color="textSecondary">
                            • {sourceNode?.label} → {targetNode?.label}
                          </Typography>
                        );
                      })}
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}
          </Box>
        </Grid>
      </Grid>
    </Container>
  );
};