import React, { useState, useMemo, useCallback } from 'react';
import { Box, Container, Typography, Paper, Grid, Card, CardContent, Button } from '@mui/material';
import { CausalNetworkVisualization } from './CausalNetworkVisualization';
import { CausalNetworkControls } from './CausalNetworkControls';
import { CausalPathwayExplorer } from './CausalPathwayExplorer';
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

  // Generate demo causal graph data
  const demoGraph = useMemo((): CausalGraph => {
    const nodes: CausalNode[] = [
      {
        id: 'seismic-activity',
        label: 'Seismic Activity',
        processType: 'seismic',
        position: { x: 200, y: 150 },
        size: 25,
        color: '#e74c3c',
        metadata: {
          description: 'Earthquake and microseismic events',
          dataSource: 'IRIS/EarthScope',
          lastUpdated: new Date(),
          confidence: 0.92,
          spatialScale: 'regional',
          temporalScale: 'second'
        },
        properties: {
          geologicalContext: 'active-fault-zone',
          monitoringNetwork: 'regional-seismic-network'
        }
      },
      {
        id: 'magma-intrusion',
        label: 'Magma Intrusion',
        processType: 'magmatic',
        position: { x: 400, y: 200 },
        size: 30,
        color: '#e67e22',
        metadata: {
          description: 'Magma injection and chamber pressurization',
          dataSource: 'USGS Volcano Hazards Program',
          lastUpdated: new Date(),
          confidence: 0.85,
          spatialScale: 'local',
          temporalScale: 'day'
        },
        properties: {
          geologicalContext: 'volcanic-system',
          depth: '5-15 km'
        }
      },
      {
        id: 'ground-deformation',
        label: 'Ground Deformation',
        processType: 'deformation',
        position: { x: 300, y: 100 },
        size: 22,
        color: '#f39c12',
        metadata: {
          description: 'Surface uplift and subsidence',
          dataSource: 'ASF-Hyp3 MT-SAR',
          lastUpdated: new Date(),
          confidence: 0.88,
          spatialScale: 'local',
          temporalScale: 'day'
        },
        properties: {
          geologicalContext: 'volcanic-edifice',
          measurementTechnique: 'InSAR'
        }
      },
      {
        id: 'gas-emissions',
        label: 'Volcanic Gas Emissions',
        processType: 'geochemical',
        position: { x: 500, y: 150 },
        size: 20,
        color: '#27ae60',
        metadata: {
          description: 'SO2, CO2, and other volcanic gas release',
          dataSource: 'Volcanic Observatory Network',
          lastUpdated: new Date(),
          confidence: 0.78,
          spatialScale: 'local',
          temporalScale: 'hour'
        },
        properties: {
          geologicalContext: 'volcanic-crater',
          gasTypes: ['SO2', 'CO2', 'H2S']
        }
      },
      {
        id: 'thermal-anomalies',
        label: 'Thermal Anomalies',
        processType: 'thermal',
        position: { x: 350, y: 300 },
        size: 18,
        color: '#e74c3c',
        metadata: {
          description: 'Surface temperature increases',
          dataSource: 'NASA FIRMS',
          lastUpdated: new Date(),
          confidence: 0.82,
          spatialScale: 'local',
          temporalScale: 'hour'
        },
        properties: {
          geologicalContext: 'volcanic-surface',
          detectionMethod: 'satellite-thermal'
        }
      },
      {
        id: 'hydrothermal-activity',
        label: 'Hydrothermal Activity',
        processType: 'hydrothermal',
        position: { x: 150, y: 250 },
        size: 16,
        color: '#16a085',
        metadata: {
          description: 'Hot spring and fumarole activity',
          dataSource: 'Field Observations',
          lastUpdated: new Date(),
          confidence: 0.75,
          spatialScale: 'outcrop',
          temporalScale: 'day'
        },
        properties: {
          geologicalContext: 'geothermal-field',
          temperature: '80-95°C'
        }
      },
      {
        id: 'crustal-stress',
        label: 'Crustal Stress Changes',
        processType: 'tectonic',
        position: { x: 100, y: 150 },
        size: 24,
        color: '#34495e',
        metadata: {
          description: 'Regional stress field variations',
          dataSource: 'GPS Network',
          lastUpdated: new Date(),
          confidence: 0.90,
          spatialScale: 'regional',
          temporalScale: 'month'
        },
        properties: {
          geologicalContext: 'tectonic-plate-boundary',
          stressRegime: 'extensional'
        }
      }
    ];

    const edges: CausalEdge[] = [
      {
        id: 'stress-seismic',
        source: 'crustal-stress',
        target: 'seismic-activity',
        strength: 0.85,
        confidence: [0.78, 0.92],
        temporalLag: 86400000, // 1 day in ms
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'Granger Causality',
          statisticalSignificance: 0.001,
          validationStatus: 'validated',
          geologicalContext: 'stress-induced seismicity'
        },
        properties: {
          mechanism: 'stress-transfer',
          evidenceStrength: 'strong'
        }
      },
      {
        id: 'magma-deformation',
        source: 'magma-intrusion',
        target: 'ground-deformation',
        strength: 0.92,
        confidence: [0.88, 0.96],
        temporalLag: 3600000, // 1 hour in ms
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'PC Algorithm',
          statisticalSignificance: 0.0001,
          validationStatus: 'validated',
          geologicalContext: 'magma-induced deformation'
        },
        properties: {
          mechanism: 'volume-expansion',
          evidenceStrength: 'very-strong'
        }
      },
      {
        id: 'magma-seismic',
        source: 'magma-intrusion',
        target: 'seismic-activity',
        strength: 0.78,
        confidence: [0.70, 0.86],
        temporalLag: 1800000, // 30 minutes in ms
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'Transfer Entropy',
          statisticalSignificance: 0.005,
          validationStatus: 'validated',
          geologicalContext: 'magma-induced seismicity'
        },
        properties: {
          mechanism: 'fracturing-pressurization',
          evidenceStrength: 'strong'
        }
      },
      {
        id: 'magma-gas',
        source: 'magma-intrusion',
        target: 'gas-emissions',
        strength: 0.88,
        confidence: [0.82, 0.94],
        temporalLag: 7200000, // 2 hours in ms
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'LiNGAM',
          statisticalSignificance: 0.0005,
          validationStatus: 'validated',
          geologicalContext: 'magma-degassing'
        },
        properties: {
          mechanism: 'volatile-exsolution',
          evidenceStrength: 'very-strong'
        }
      },
      {
        id: 'magma-thermal',
        source: 'magma-intrusion',
        target: 'thermal-anomalies',
        strength: 0.82,
        confidence: [0.75, 0.89],
        temporalLag: 10800000, // 3 hours in ms
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'Causal Forests',
          statisticalSignificance: 0.002,
          validationStatus: 'validated',
          geologicalContext: 'thermal-conduction'
        },
        properties: {
          mechanism: 'heat-transfer',
          evidenceStrength: 'strong'
        }
      },
      {
        id: 'thermal-hydrothermal',
        source: 'thermal-anomalies',
        target: 'hydrothermal-activity',
        strength: 0.65,
        confidence: [0.55, 0.75],
        temporalLag: 21600000, // 6 hours in ms
        edgeType: 'indirect',
        metadata: {
          discoveryMethod: 'Cross-Correlation',
          statisticalSignificance: 0.02,
          validationStatus: 'pending',
          geologicalContext: 'thermal-fluid-interaction'
        },
        properties: {
          mechanism: 'fluid-heating',
          evidenceStrength: 'moderate'
        }
      },
      {
        id: 'deformation-seismic-feedback',
        source: 'ground-deformation',
        target: 'seismic-activity',
        strength: 0.45,
        confidence: [0.35, 0.55],
        temporalLag: 43200000, // 12 hours in ms
        edgeType: 'bidirectional',
        metadata: {
          discoveryMethod: 'GES',
          statisticalSignificance: 0.05,
          validationStatus: 'uncertain',
          geologicalContext: 'stress-redistribution'
        },
        properties: {
          mechanism: 'feedback-loop',
          evidenceStrength: 'weak'
        }
      }
    ];

    return {
      nodes,
      edges,
      metadata: {
        createdAt: new Date(),
        lastUpdated: new Date(),
        totalNodes: nodes.length,
        totalEdges: edges.length,
        averageCausalStrength: edges.reduce((sum, e) => sum + e.strength, 0) / edges.length,
        spatialExtent: {
          north: 45.0,
          south: 44.0,
          east: -121.0,
          west: -122.0
        },
        temporalExtent: {
          start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
          end: new Date()
        }
      }
    };
  }, []);

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

  return (
    <Container maxWidth={false} sx={{ height: '100vh', p: 2 }}>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Interactive Causal Network Visualization Demo
        </Typography>
        <Typography variant="subtitle1" color="textSecondary" gutterBottom>
          Advanced geological process causality analysis with real-time interaction
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
              graph={demoGraph}
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
                graph={demoGraph}
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
                  Nodes: {demoGraph.nodes.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Edges: {demoGraph.edges.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Avg. Causal Strength: {demoGraph.metadata.averageCausalStrength.toFixed(2)}
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
                  {Array.from(new Set(demoGraph.nodes.map(n => n.processType))).map(type => (
                    <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          borderRadius: '50%',
                          bgcolor: demoGraph.nodes.find(n => n.processType === type)?.color || '#ccc'
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
                        const node = demoGraph.nodes.find(n => n.id === nodeId);
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
                        const edge = demoGraph.edges.find(e => e.id === edgeId);
                        const sourceNode = demoGraph.nodes.find(n => n.id === edge?.source);
                        const targetNode = demoGraph.nodes.find(n => n.id === edge?.target);
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