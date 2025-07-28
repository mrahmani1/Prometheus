import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  InputAdornment,
  Chip,
  Card,
  CardContent,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore,
  Search,
  Info,
  DataObject,
  Analytics,
  Terrain,
  Timeline,
  Thermostat,
  Psychology,
  Visibility,
  SaveAlt,
  Settings
} from '@mui/icons-material';
import { WorkflowNodeCategory, WorkflowNodeDefinition } from '@/types/workflow';
import { useWorkflowStore } from '@/store/workflow/workflowStore';
import { geologicalNodeDefinitions } from '@/data/workflow/nodeDefinitions';

interface ComponentPaletteProps {
  onAddNode: (nodeType: string, position: { x: number; y: number }) => void;
}

const categoryIcons: Record<WorkflowNodeCategory, React.ReactElement> = {
  'data-input': <DataObject />,
  'data-processing': <Analytics />,
  'geological-analysis': <Terrain />,
  'seismic-analysis': <Timeline />,
  'deformation-analysis': <Timeline />,
  'geochemical-analysis': <Analytics />,
  'thermal-analysis': <Thermostat />,
  'ai-modeling': <Psychology />,
  'visualization': <Visibility />,
  'data-output': <SaveAlt />,
  'control-flow': <Settings />,
  'utilities': <Settings />
};

const categoryColors: Record<WorkflowNodeCategory, string> = {
  'data-input': '#2196F3',
  'data-processing': '#FF9800',
  'geological-analysis': '#8BC34A',
  'seismic-analysis': '#E91E63',
  'deformation-analysis': '#9C27B0',
  'geochemical-analysis': '#00BCD4',
  'thermal-analysis': '#FF5722',
  'ai-modeling': '#673AB7',
  'visualization': '#4CAF50',
  'data-output': '#795548',
  'control-flow': '#607D8B',
  'utilities': '#9E9E9E'
};

export const ComponentPalette: React.FC<ComponentPaletteProps> = ({ onAddNode }) => {
  const { nodeDefinitions, registerNodeDefinition } = useWorkflowStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['data-input', 'geological-analysis'])
  );

  // Register geological node definitions on mount
  React.useEffect(() => {
    geologicalNodeDefinitions.forEach(definition => {
      registerNodeDefinition(definition);
    });
  }, [registerNodeDefinition]);

  const filteredNodes = useMemo(() => {
    const allNodes = Object.values(nodeDefinitions);
    
    if (!searchTerm) return allNodes;
    
    return allNodes.filter(node =>
      node.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.category.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [nodeDefinitions, searchTerm]);

  const nodesByCategory = useMemo(() => {
    const categories: Record<WorkflowNodeCategory, WorkflowNodeDefinition[]> = {
      'data-input': [],
      'data-processing': [],
      'geological-analysis': [],
      'seismic-analysis': [],
      'deformation-analysis': [],
      'geochemical-analysis': [],
      'thermal-analysis': [],
      'ai-modeling': [],
      'visualization': [],
      'data-output': [],
      'control-flow': [],
      'utilities': []
    };

    filteredNodes.forEach(node => {
      if (categories[node.category]) {
        categories[node.category].push(node);
      }
    });

    return categories;
  }, [filteredNodes]);

  const handleCategoryToggle = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const handleNodeDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/json', JSON.stringify({
      type: 'workflow-node',
      nodeType
    }));
    event.dataTransfer.effectAllowed = 'copy';
  };

  const handleNodeClick = (nodeType: string) => {
    // Add node at center of canvas
    onAddNode(nodeType, { x: 400, y: 300 });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" gutterBottom>
          Components
        </Typography>
        
        <TextField
          fullWidth
          size="small"
          placeholder="Search components..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            )
          }}
        />
      </Box>

      {/* Component Categories */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {Object.entries(nodesByCategory).map(([category, nodes]) => {
          if (nodes.length === 0) return null;
          
          const categoryKey = category as WorkflowNodeCategory;
          const isExpanded = expandedCategories.has(category);
          
          return (
            <Accordion
              key={category}
              expanded={isExpanded}
              onChange={() => handleCategoryToggle(category)}
              disableGutters
              elevation={0}
              sx={{
                '&:before': { display: 'none' },
                borderBottom: 1,
                borderColor: 'divider'
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  minHeight: 48,
                  '& .MuiAccordionSummary-content': {
                    alignItems: 'center',
                    gap: 1
                  }
                }}
              >
                <Box sx={{ color: categoryColors[categoryKey] }}>
                  {categoryIcons[categoryKey]}
                </Box>
                <Typography variant="subtitle2">
                  {category.split('-').map(word => 
                    word.charAt(0).toUpperCase() + word.slice(1)
                  ).join(' ')}
                </Typography>
                <Chip 
                  label={nodes.length} 
                  size="small" 
                  sx={{ ml: 'auto', mr: 1 }}
                />
              </AccordionSummary>
              
              <AccordionDetails sx={{ p: 1 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {nodes.map(node => (
                    <Card
                      key={node.type}
                      variant="outlined"
                      draggable
                      onDragStart={(e: any) => handleNodeDragStart(e, node.type)}
                      onClick={() => handleNodeClick(node.type)}
                      sx={{
                        cursor: 'grab',
                        transition: 'all 0.2s',
                        '&:hover': {
                          boxShadow: 2,
                          transform: 'translateY(-1px)'
                        },
                        '&:active': {
                          cursor: 'grabbing'
                        }
                      }}
                    >
                      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                          <Box
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              backgroundColor: categoryColors[categoryKey],
                              mt: 0.5,
                              flexShrink: 0
                            }}
                          />
                          
                          <Box sx={{ flex: 1, minWidth: 0 }}>
                            <Typography 
                              variant="body2" 
                              fontWeight="medium"
                              sx={{ 
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}
                            >
                              {node.name}
                            </Typography>
                            
                            <Typography 
                              variant="caption" 
                              color="text.secondary"
                              sx={{
                                display: '-webkit-box',
                                WebkitLineClamp: 2,
                                WebkitBoxOrient: 'vertical',
                                overflow: 'hidden',
                                lineHeight: 1.2,
                                mt: 0.5
                              }}
                            >
                              {node.description}
                            </Typography>
                          </Box>
                          
                          <Tooltip title={node.description} placement="left">
                            <IconButton size="small" sx={{ p: 0.5 }}>
                              <Info fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                        
                        {/* Input/Output indicators */}
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            In: {node.inputs.length}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Out: {node.outputs.length}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              </AccordionDetails>
            </Accordion>
          );
        })}
      </Box>

      {/* Footer */}
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary" align="center">
          Drag components to canvas or click to add
        </Typography>
      </Box>
    </Box>
  );
};