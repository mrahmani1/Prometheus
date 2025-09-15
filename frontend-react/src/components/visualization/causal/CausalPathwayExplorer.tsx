import React, { useState, useMemo, useCallback } from 'react';
import {
  Paper,
  Box,
  Typography,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Button,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip
} from '@mui/material';
import {
  Close,
  AccountTree,
  Timeline,
  TrendingUp,
  Loop,
  CallSplit,
  CallMerge,
  ExpandMore,
  Visibility,
  VisibilityOff,
  Info
} from '@mui/icons-material';
import {
  CausalNode,
  CausalEdge,
  CausalGraph,
  CausalPath,
  PathType
} from './types';

interface CausalPathwayExplorerProps {
  graph: CausalGraph;
  selectedNodes: string[];
  onPathHighlight: (paths: CausalPath[]) => void;
  onClose: () => void;
}

interface PathAnalysisResult {
  path: CausalPath;
  nodes: CausalNode[];
  edges: CausalEdge[];
  statistics: {
    averageStrength: number;
    totalLag: number;
    confidenceScore: number;
    significanceScore: number;
  };
}

export const CausalPathwayExplorer: React.FC<CausalPathwayExplorerProps> = ({
  graph,
  selectedNodes,
  onPathHighlight,
  onClose
}) => {
  const [pathOptions, setPathOptions] = useState({
    maxPathLength: 5,
    minStrength: 0.3,
    pathTypes: ['causal-chain', 'feedback-loop', 'convergent', 'divergent'] as PathType[],
    includeIndirect: true,
    sortBy: 'strength' as 'strength' | 'length' | 'confidence' | 'significance'
  });

  const [highlightedPaths, setHighlightedPaths] = useState<string[]>([]);

  // Discover causal pathways
  const discoveredPaths = useMemo(() => {
    const paths: PathAnalysisResult[] = [];

    // If nodes are selected, find paths between them
    if (selectedNodes.length >= 2) {
      for (let i = 0; i < selectedNodes.length; i++) {
        for (let j = i + 1; j < selectedNodes.length; j++) {
          const pathsBetween = findPathsBetweenNodes(
            graph,
            selectedNodes[i],
            selectedNodes[j],
            pathOptions
          );
          paths.push(...pathsBetween);
        }
      }
    } else {
      // Find all significant paths in the graph
      const allPaths = findAllSignificantPaths(graph, pathOptions);
      paths.push(...allPaths);
    }

    // Sort paths based on selected criteria
    return paths.sort((a, b) => {
      switch (pathOptions.sortBy) {
        case 'strength':
          return b.statistics.averageStrength - a.statistics.averageStrength;
        case 'length':
          return a.path.pathLength - b.path.pathLength;
        case 'confidence':
          return b.statistics.confidenceScore - a.statistics.confidenceScore;
        case 'significance':
          return b.statistics.significanceScore - a.statistics.significanceScore;
        default:
          return b.statistics.averageStrength - a.statistics.averageStrength;
      }
    });
  }, [graph, selectedNodes, pathOptions]);

  const handlePathToggle = useCallback((pathId: string) => {
    setHighlightedPaths(prev => {
      const newHighlighted = prev.includes(pathId)
        ? prev.filter(id => id !== pathId)
        : [...prev, pathId];
      
      const pathsToHighlight = discoveredPaths
        .filter(p => newHighlighted.includes(p.path.id))
        .map(p => p.path);
      
      onPathHighlight(pathsToHighlight);
      return newHighlighted;
    });
  }, [discoveredPaths, onPathHighlight]);

  const handleHighlightAll = useCallback(() => {
    const allPathIds = discoveredPaths.map(p => p.path.id);
    setHighlightedPaths(allPathIds);
    onPathHighlight(discoveredPaths.map(p => p.path));
  }, [discoveredPaths, onPathHighlight]);

  const handleClearAll = useCallback(() => {
    setHighlightedPaths([]);
    onPathHighlight([]);
  }, [onPathHighlight]);

  const getPathTypeIcon = (pathType: PathType) => {
    switch (pathType) {
      case 'causal-chain':
        return <TrendingUp />;
      case 'feedback-loop':
        return <Loop />;
      case 'convergent':
        return <CallMerge />;
      case 'divergent':
        return <CallSplit />;
      case 'mediating':
        return <AccountTree />;
      case 'confounding':
        return <Timeline />;
      default:
        return <AccountTree />;
    }
  };

  const getPathTypeColor = (pathType: PathType): 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success' => {
    switch (pathType) {
      case 'causal-chain':
        return 'primary';
      case 'feedback-loop':
        return 'warning';
      case 'convergent':
        return 'info';
      case 'divergent':
        return 'success';
      case 'mediating':
        return 'secondary';
      case 'confounding':
        return 'error';
      default:
        return 'primary';
    }
  };

  const formatPathDescription = (pathResult: PathAnalysisResult): string => {
    const { path, statistics } = pathResult;
    const nodeLabels = pathResult.nodes.map(n => n.label);
    
    return `${nodeLabels.join(' → ')} (Strength: ${statistics.averageStrength.toFixed(2)}, Length: ${path.pathLength})`;
  };

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'absolute',
        top: 16,
        right: 380,
        width: 450,
        maxHeight: 'calc(100% - 32px)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 999
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AccountTree color="primary" />
            <Typography variant="h6">Causal Pathway Explorer</Typography>
          </Box>
          <IconButton size="small" onClick={onClose}>
            <Close />
          </IconButton>
        </Box>

        {/* Quick Actions */}
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button
            size="small"
            variant="outlined"
            onClick={handleHighlightAll}
            disabled={discoveredPaths.length === 0}
          >
            Highlight All
          </Button>
          <Button
            size="small"
            variant="outlined"
            onClick={handleClearAll}
            disabled={highlightedPaths.length === 0}
          >
            Clear All
          </Button>
        </Box>

        {/* Path Statistics */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            label={`${discoveredPaths.length} paths found`}
            size="small"
            color="primary"
            variant="outlined"
          />
          <Chip
            label={`${highlightedPaths.length} highlighted`}
            size="small"
            color="secondary"
            variant="outlined"
          />
        </Box>
      </Box>

      {/* Path Options */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="subtitle2">Path Discovery Options</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Max Path Length */}
            <Box>
              <Typography variant="caption" gutterBottom>
                Maximum Path Length: {pathOptions.maxPathLength}
              </Typography>
              <Slider
                value={pathOptions.maxPathLength}
                onChange={(_event: Event, value: number | number[]) => setPathOptions(prev => ({ ...prev, maxPathLength: value as number }))}
                min={2}
                max={10}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Minimum Strength */}
            <Box>
              <Typography variant="caption" gutterBottom>
                Minimum Strength: {pathOptions.minStrength.toFixed(2)}
              </Typography>
              <Slider
                value={pathOptions.minStrength}
                onChange={(_event: Event, value: number | number[]) => setPathOptions(prev => ({ ...prev, minStrength: value as number }))}
                min={0}
                max={1}
                step={0.05}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Sort By */}
            <FormControl fullWidth size="small">
              <InputLabel>Sort By</InputLabel>
              <Select
                value={pathOptions.sortBy}
                onChange={(e) => setPathOptions(prev => ({ ...prev, sortBy: e.target.value as any }))}
                label="Sort By"
              >
                <MenuItem value="strength">Causal Strength</MenuItem>
                <MenuItem value="length">Path Length</MenuItem>
                <MenuItem value="confidence">Confidence Score</MenuItem>
                <MenuItem value="significance">Statistical Significance</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </AccordionDetails>
      </Accordion>

      <Divider />

      {/* Discovered Paths */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {discoveredPaths.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="textSecondary">
              {selectedNodes.length >= 2 
                ? `No causal paths found between selected nodes`
                : `No significant causal paths found in the network`
              }
            </Typography>
          </Box>
        ) : (
          <List dense>
            {discoveredPaths.map((pathResult, index) => {
              const isHighlighted = highlightedPaths.includes(pathResult.path.id);
              
              return (
                <React.Fragment key={pathResult.path.id}>
                  <ListItem
                    button
                    onClick={() => handlePathToggle(pathResult.path.id)}
                    sx={{ 
                      cursor: 'pointer',
                      bgcolor: isHighlighted ? 'action.selected' : 'transparent'
                    }}
                  >
                    <ListItemIcon>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getPathTypeIcon(pathResult.path.pathType)}
                        <IconButton size="small">
                          {isHighlighted ? <Visibility /> : <VisibilityOff />}
                        </IconButton>
                      </Box>
                    </ListItemIcon>
                    
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Typography variant="body2" noWrap>
                            Path {index + 1}
                          </Typography>
                          <Chip
                            label={pathResult.path.pathType}
                            size="small"
                            color={getPathTypeColor(pathResult.path.pathType)}
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" color="textSecondary" sx={{ display: 'block' }}>
                            {formatPathDescription(pathResult)}
                          </Typography>
                          
                          <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
                            <Chip
                              label={`Strength: ${pathResult.statistics.averageStrength.toFixed(2)}`}
                              size="small"
                              variant="filled"
                              color="primary"
                            />
                            <Chip
                              label={`Confidence: ${(pathResult.statistics.confidenceScore * 100).toFixed(0)}%`}
                              size="small"
                              variant="filled"
                              color="secondary"
                            />
                            {pathResult.statistics.totalLag > 0 && (
                              <Chip
                                label={`Lag: ${formatLag(pathResult.statistics.totalLag)}`}
                                size="small"
                                variant="outlined"
                              />
                            )}
                          </Box>
                        </Box>
                      }
                    />
                    
                    <Tooltip title="Path Information">
                      <IconButton size="small">
                        <Info />
                      </IconButton>
                    </Tooltip>
                  </ListItem>
                  {index < discoveredPaths.length - 1 && <Divider />}
                </React.Fragment>
              );
            })}
          </List>
        )}
      </Box>
    </Paper>
  );
};

// Helper functions for path discovery

function findPathsBetweenNodes(
  graph: CausalGraph,
  sourceId: string,
  targetId: string,
  options: any
): PathAnalysisResult[] {
  const paths: PathAnalysisResult[] = [];
  const visited = new Set<string>();
  
  const dfs = (currentId: string, currentPath: string[], currentEdges: string[], depth: number) => {
    if (depth > options.maxPathLength || visited.has(currentId)) return;
    
    if (currentId === targetId && currentPath.length > 1) {
      // Found a path
      const pathNodes = currentPath.map(id => graph.nodes.find(n => n.id === id)!).filter(Boolean);
      const pathEdges = currentEdges.map(id => graph.edges.find(e => e.id === id)!).filter(Boolean);
      
      const statistics = calculatePathStatistics(pathNodes, pathEdges);
      
      if (statistics.averageStrength >= options.minStrength) {
        const pathType = determinePathType(pathNodes, pathEdges);
        
        paths.push({
          path: {
            id: `path-${currentPath.join('-')}`,
            nodes: currentPath,
            edges: currentEdges,
            pathStrength: statistics.averageStrength,
            pathLength: currentPath.length,
            totalLag: statistics.totalLag,
            pathType
          },
          nodes: pathNodes,
          edges: pathEdges,
          statistics
        });
      }
      return;
    }
    
    visited.add(currentId);
    
    const outgoingEdges = graph.edges.filter(edge => edge.source === currentId);
    
    for (const edge of outgoingEdges) {
      if (!visited.has(edge.target)) {
        dfs(
          edge.target,
          [...currentPath, edge.target],
          [...currentEdges, edge.id],
          depth + 1
        );
      }
    }
    
    visited.delete(currentId);
  };
  
  dfs(sourceId, [sourceId], [], 0);
  
  return paths;
}

function findAllSignificantPaths(
  graph: CausalGraph,
  options: any
): PathAnalysisResult[] {
  const paths: PathAnalysisResult[] = [];
  
  // Find paths starting from high-centrality nodes
  const centralityScores = calculateNodeCentrality(graph);
  const topNodes = Object.entries(centralityScores)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([id]) => id);
  
  for (const startNodeId of topNodes) {
    const nodePaths = findPathsFromNode(graph, startNodeId, options);
    paths.push(...nodePaths);
  }
  
  // Remove duplicates and sort by significance
  const uniquePaths = paths.filter((path, index, arr) => 
    arr.findIndex(p => p.path.id === path.path.id) === index
  );
  
  return uniquePaths.slice(0, 50); // Limit to top 50 paths
}

function findPathsFromNode(
  graph: CausalGraph,
  startNodeId: string,
  options: any
): PathAnalysisResult[] {
  const paths: PathAnalysisResult[] = [];
  const visited = new Set<string>();
  
  const dfs = (currentId: string, currentPath: string[], currentEdges: string[], depth: number) => {
    if (depth > options.maxPathLength || visited.has(currentId)) return;
    
    if (currentPath.length >= 2) {
      const pathNodes = currentPath.map(id => graph.nodes.find(n => n.id === id)!).filter(Boolean);
      const pathEdges = currentEdges.map(id => graph.edges.find(e => e.id === id)!).filter(Boolean);
      
      const statistics = calculatePathStatistics(pathNodes, pathEdges);
      
      if (statistics.averageStrength >= options.minStrength) {
        const pathType = determinePathType(pathNodes, pathEdges);
        
        paths.push({
          path: {
            id: `path-${currentPath.join('-')}`,
            nodes: currentPath,
            edges: currentEdges,
            pathStrength: statistics.averageStrength,
            pathLength: currentPath.length,
            totalLag: statistics.totalLag,
            pathType
          },
          nodes: pathNodes,
          edges: pathEdges,
          statistics
        });
      }
    }
    
    visited.add(currentId);
    
    const outgoingEdges = graph.edges.filter(edge => edge.source === currentId);
    
    for (const edge of outgoingEdges) {
      if (!visited.has(edge.target)) {
        dfs(
          edge.target,
          [...currentPath, edge.target],
          [...currentEdges, edge.id],
          depth + 1
        );
      }
    }
    
    visited.delete(currentId);
  };
  
  dfs(startNodeId, [startNodeId], [], 0);
  
  return paths;
}

function calculatePathStatistics(_nodes: CausalNode[], edges: CausalEdge[]) {
  const averageStrength = edges.reduce((sum, edge) => sum + edge.strength, 0) / edges.length;
  const totalLag = edges.reduce((sum, edge) => sum + (edge.temporalLag || 0), 0);
  const confidenceScore = edges.reduce((sum, edge) => {
    const [lower, upper] = edge.confidence;
    return sum + (upper - lower); // Narrower confidence interval = higher score
  }, 0) / edges.length;
  const significenceScore = edges.reduce((sum, edge) => sum + edge.metadata.statisticalSignificance, 0) / edges.length;
  
  return {
    averageStrength,
    totalLag,
    confidenceScore: 1 - confidenceScore, // Invert so higher is better
    significanceScore: significenceScore
  };
}

function determinePathType(nodes: CausalNode[], edges: CausalEdge[]): PathType {
  // Check for feedback loops
  const nodeIds = nodes.map(n => n.id);
  const hasLoop = edges.some(edge => 
    nodeIds.includes(edge.source) && nodeIds.includes(edge.target) &&
    nodeIds.indexOf(edge.target) < nodeIds.indexOf(edge.source)
  );
  
  if (hasLoop) return 'feedback-loop';
  
  // Check for convergent/divergent patterns
  const inDegrees = new Map<string, number>();
  const outDegrees = new Map<string, number>();
  
  edges.forEach(edge => {
    outDegrees.set(edge.source, (outDegrees.get(edge.source) || 0) + 1);
    inDegrees.set(edge.target, (inDegrees.get(edge.target) || 0) + 1);
  });
  
  const hasHighOutDegree = Array.from(outDegrees.values()).some(degree => degree > 2);
  const hasHighInDegree = Array.from(inDegrees.values()).some(degree => degree > 2);
  
  if (hasHighOutDegree) return 'divergent';
  if (hasHighInDegree) return 'convergent';
  
  // Check for mediating relationships
  const hasMediatingEdge = edges.some(edge => edge.edgeType === 'mediating');
  if (hasMediatingEdge) return 'mediating';
  
  // Check for confounding relationships
  const hasConfoundingEdge = edges.some(edge => edge.edgeType === 'confounding');
  if (hasConfoundingEdge) return 'confounding';
  
  // Default to causal chain
  return 'causal-chain';
}

function calculateNodeCentrality(graph: CausalGraph): Record<string, number> {
  const centrality: Record<string, number> = {};
  
  graph.nodes.forEach(node => {
    const inDegree = graph.edges.filter(edge => edge.target === node.id).length;
    const outDegree = graph.edges.filter(edge => edge.source === node.id).length;
    const strengthSum = graph.edges
      .filter(edge => edge.source === node.id || edge.target === node.id)
      .reduce((sum, edge) => sum + edge.strength, 0);
    
    centrality[node.id] = (inDegree + outDegree) * strengthSum;
  });
  
  return centrality;
}

function formatLag(lagMs: number): string {
  const seconds = lagMs / 1000;
  const minutes = seconds / 60;
  const hours = minutes / 60;
  const days = hours / 24;
  
  if (days >= 1) return `${days.toFixed(1)}d`;
  if (hours >= 1) return `${hours.toFixed(1)}h`;
  if (minutes >= 1) return `${minutes.toFixed(1)}m`;
  return `${seconds.toFixed(1)}s`;
}