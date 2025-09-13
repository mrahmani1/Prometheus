import React, { useState, useMemo } from 'react';
import {
  Paper,
  Box,
  Typography,
  IconButton,
  TextField,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Switch
} from '@mui/material';
import {
  Close,
  Search,
  Clear,
  ExpandMore,
  AccountTree,
  Timeline,
  Science,
  TrendingUp
} from '@mui/icons-material';
import { CausalNode, CausalEdge, GeologicalProcessType } from './types';

interface CausalSearchInterfaceProps {
  searchQuery?: string;
  nodes?: CausalNode[];
  edges?: CausalEdge[];
  onSearchChange: (query: string) => void;
  onNodeSelect?: (node: CausalNode) => void;
  onEdgeSelect?: (edge: CausalEdge) => void;
  onClose: () => void;
}

interface SearchResult {
  type: 'node' | 'edge' | 'path';
  id: string;
  label: string;
  description: string;
  processType?: GeologicalProcessType;
  strength?: number;
  confidence?: number;
  relevanceScore: number;
}

export const CausalSearchInterface: React.FC<CausalSearchInterfaceProps> = ({
  searchQuery = '',
  nodes = [],
  edges = [],
  onSearchChange,
  onNodeSelect,
  onEdgeSelect,
  onClose
}) => {
  const [localQuery, setLocalQuery] = useState(searchQuery);
  const [searchOptions, setSearchOptions] = useState({
    includeNodes: true,
    includeEdges: true,
    includePaths: true,
    fuzzySearch: true,
    caseSensitive: false
  });

  // Advanced search functionality
  const searchResults = useMemo(() => {
    if (!localQuery.trim()) return [];

    const query = searchOptions.caseSensitive ? localQuery : localQuery.toLowerCase();
    const results: SearchResult[] = [];

    // Search nodes
    if (searchOptions.includeNodes) {
      nodes.forEach(node => {
        const searchText = searchOptions.caseSensitive 
          ? `${node.label} ${node.processType} ${node.metadata.description}`
          : `${node.label} ${node.processType} ${node.metadata.description}`.toLowerCase();

        let relevanceScore = 0;

        // Exact match in label (highest priority)
        if (searchText.includes(query)) {
          if (node.label.toLowerCase().includes(query.toLowerCase())) {
            relevanceScore += 100;
          }
          if (node.processType.toLowerCase().includes(query.toLowerCase())) {
            relevanceScore += 80;
          }
          if (node.metadata.description.toLowerCase().includes(query.toLowerCase())) {
            relevanceScore += 60;
          }
        }

        // Fuzzy search
        if (searchOptions.fuzzySearch && relevanceScore === 0) {
          const fuzzyScore = calculateFuzzyScore(query, searchText);
          if (fuzzyScore > 0.3) {
            relevanceScore = fuzzyScore * 50;
          }
        }

        if (relevanceScore > 0) {
          results.push({
            type: 'node',
            id: node.id,
            label: node.label,
            description: `${node.processType} process - ${node.metadata.description}`,
            processType: node.processType,
            confidence: node.metadata.confidence,
            relevanceScore
          });
        }
      });
    }

    // Search edges
    if (searchOptions.includeEdges) {
      edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (!sourceNode || !targetNode) return;

        const searchText = searchOptions.caseSensitive
          ? `${sourceNode.label} ${targetNode.label} ${edge.edgeType} ${edge.metadata.discoveryMethod} ${edge.metadata.geologicalContext}`
          : `${sourceNode.label} ${targetNode.label} ${edge.edgeType} ${edge.metadata.discoveryMethod} ${edge.metadata.geologicalContext}`.toLowerCase();

        let relevanceScore = 0;

        if (searchText.includes(query)) {
          relevanceScore += 70;
        }

        if (searchOptions.fuzzySearch && relevanceScore === 0) {
          const fuzzyScore = calculateFuzzyScore(query, searchText);
          if (fuzzyScore > 0.3) {
            relevanceScore = fuzzyScore * 40;
          }
        }

        if (relevanceScore > 0) {
          results.push({
            type: 'edge',
            id: edge.id,
            label: `${sourceNode.label} → ${targetNode.label}`,
            description: `${edge.edgeType} relationship (strength: ${edge.strength.toFixed(2)})`,
            strength: edge.strength,
            relevanceScore
          });
        }
      });
    }

    // Search causal paths (simplified implementation)
    if (searchOptions.includePaths) {
      const paths = findCausalPaths(nodes, edges, query);
      paths.forEach(path => {
        results.push({
          type: 'path',
          id: path.id,
          label: path.label,
          description: path.description,
          relevanceScore: path.relevanceScore
        });
      });
    }

    // Sort by relevance score
    return results.sort((a, b) => b.relevanceScore - a.relevanceScore).slice(0, 50);
  }, [localQuery, nodes, edges, searchOptions]);

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = event.target.value;
    setLocalQuery(newQuery);
    onSearchChange(newQuery);
  };

  const handleClearSearch = () => {
    setLocalQuery('');
    onSearchChange('');
  };

  const handleResultClick = (result: SearchResult) => {
    if (result.type === 'node') {
      const node = nodes.find(n => n.id === result.id);
      if (node && onNodeSelect) {
        onNodeSelect(node);
      }
    } else if (result.type === 'edge') {
      const edge = edges.find(e => e.id === result.id);
      if (edge && onEdgeSelect) {
        onEdgeSelect(edge);
      }
    }
  };

  const getResultIcon = (type: string) => {
    switch (type) {
      case 'node':
        return <Science />;
      case 'edge':
        return <TrendingUp />;
      case 'path':
        return <AccountTree />;
      default:
        return <Search />;
    }
  };

  const getProcessTypeColor = (processType?: GeologicalProcessType) => {
    if (!processType) return 'default';
    
    const colorMap: Record<GeologicalProcessType, 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success'> = {
      seismic: 'error',
      volcanic: 'warning',
      deformation: 'info',
      geochemical: 'success',
      thermal: 'error',
      electromagnetic: 'secondary',
      hydrological: 'info',
      atmospheric: 'primary',
      tectonic: 'secondary',
      magmatic: 'warning',
      hydrothermal: 'info',
      metamorphic: 'secondary',
      sedimentary: 'primary'
    };

    return colorMap[processType] || 'default';
  };

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'absolute',
        top: 16,
        left: 16,
        width: 400,
        maxHeight: 'calc(100% - 32px)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 1000
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Search color="primary" />
            <Typography variant="h6">Search Causal Network</Typography>
          </Box>
          <IconButton size="small" onClick={onClose}>
            <Close />
          </IconButton>
        </Box>

        {/* Search Input */}
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search nodes, edges, or causal pathways..."
          value={localQuery}
          onChange={handleSearchChange}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
            endAdornment: localQuery && (
              <InputAdornment position="end">
                <IconButton size="small" onClick={handleClearSearch}>
                  <Clear />
                </IconButton>
              </InputAdornment>
            )
          }}
        />
      </Box>

      {/* Search Options */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="subtitle2">Search Options</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={searchOptions.includeNodes}
                  onChange={(e) => setSearchOptions(prev => ({ ...prev, includeNodes: e.target.checked }))}
                />
              }
              label="Include Nodes"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={searchOptions.includeEdges}
                  onChange={(e) => setSearchOptions(prev => ({ ...prev, includeEdges: e.target.checked }))}
                />
              }
              label="Include Edges"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={searchOptions.includePaths}
                  onChange={(e) => setSearchOptions(prev => ({ ...prev, includePaths: e.target.checked }))}
                />
              }
              label="Include Causal Paths"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={searchOptions.fuzzySearch}
                  onChange={(e) => setSearchOptions(prev => ({ ...prev, fuzzySearch: e.target.checked }))}
                />
              }
              label="Fuzzy Search"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={searchOptions.caseSensitive}
                  onChange={(e) => setSearchOptions(prev => ({ ...prev, caseSensitive: e.target.checked }))}
                />
              }
              label="Case Sensitive"
            />
          </Box>
        </AccordionDetails>
      </Accordion>

      <Divider />

      {/* Search Results */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {localQuery.trim() === '' ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="textSecondary">
              Enter a search term to find nodes, edges, or causal pathways
            </Typography>
          </Box>
        ) : searchResults.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="textSecondary">
              No results found for "{localQuery}"
            </Typography>
          </Box>
        ) : (
          <List dense>
            {searchResults.map((result, index) => (
              <React.Fragment key={result.id}>
                <ListItem
                  button
                  onClick={() => handleResultClick(result)}
                  sx={{ cursor: 'pointer' }}
                >
                  <ListItemIcon>
                    {getResultIcon(result.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" noWrap>
                          {result.label}
                        </Typography>
                        {result.processType && (
                          <Chip
                            label={result.processType}
                            size="small"
                            color={getProcessTypeColor(result.processType)}
                            variant="outlined"
                          />
                        )}
                        <Chip
                          label={result.type}
                          size="small"
                          variant="filled"
                          sx={{ ml: 'auto' }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" color="textSecondary" noWrap>
                          {result.description}
                        </Typography>
                        {(result.strength !== undefined || result.confidence !== undefined) && (
                          <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                            {result.strength !== undefined && (
                              <Chip
                                label={`Strength: ${result.strength.toFixed(2)}`}
                                size="small"
                                variant="outlined"
                              />
                            )}
                            {result.confidence !== undefined && (
                              <Chip
                                label={`Confidence: ${(result.confidence * 100).toFixed(0)}%`}
                                size="small"
                                variant="outlined"
                              />
                            )}
                          </Box>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
                {index < searchResults.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Box>

      {/* Results Summary */}
      {localQuery.trim() !== '' && (
        <>
          <Divider />
          <Box sx={{ p: 1, bgcolor: 'grey.50' }}>
            <Typography variant="caption" color="textSecondary">
              {searchResults.length} results found
            </Typography>
          </Box>
        </>
      )}
    </Paper>
  );
};

// Helper functions

function calculateFuzzyScore(query: string, text: string): number {
  // Simple fuzzy matching algorithm
  const queryLower = query.toLowerCase();
  const textLower = text.toLowerCase();
  
  let score = 0;
  let queryIndex = 0;
  
  for (let i = 0; i < textLower.length && queryIndex < queryLower.length; i++) {
    if (textLower[i] === queryLower[queryIndex]) {
      score++;
      queryIndex++;
    }
  }
  
  return queryIndex === queryLower.length ? score / query.length : 0;
}

function findCausalPaths(
  nodes: CausalNode[], 
  edges: CausalEdge[], 
  query: string
): Array<{ id: string; label: string; description: string; relevanceScore: number }> {
  // Simplified causal path finding
  const paths: Array<{ id: string; label: string; description: string; relevanceScore: number }> = [];
  
  // Find nodes that match the query
  const matchingNodes = nodes.filter(node => 
    node.label.toLowerCase().includes(query.toLowerCase()) ||
    node.processType.toLowerCase().includes(query.toLowerCase())
  );
  
  // For each matching node, find paths starting from it
  matchingNodes.forEach(startNode => {
    const visited = new Set<string>();
    const currentPath: string[] = [];
    
    const findPathsRecursive = (nodeId: string, depth: number) => {
      if (depth > 3 || visited.has(nodeId)) return; // Limit path depth
      
      visited.add(nodeId);
      currentPath.push(nodeId);
      
      const outgoingEdges = edges.filter(edge => edge.source === nodeId);
      
      if (outgoingEdges.length === 0 && currentPath.length > 1) {
        // End of path found
        const pathNodes = currentPath.map(id => nodes.find(n => n.id === id)?.label).filter(Boolean);
        const pathStrength = outgoingEdges.reduce((sum, edge) => sum + edge.strength, 0) / outgoingEdges.length;
        
        paths.push({
          id: `path-${currentPath.join('-')}`,
          label: pathNodes.join(' → '),
          description: `Causal pathway with ${currentPath.length} steps`,
          relevanceScore: 30 + pathStrength * 20
        });
      }
      
      outgoingEdges.forEach(edge => {
        findPathsRecursive(edge.target, depth + 1);
      });
      
      currentPath.pop();
      visited.delete(nodeId);
    };
    
    findPathsRecursive(startNode.id, 0);
  });
  
  return paths.slice(0, 10); // Limit number of paths
}