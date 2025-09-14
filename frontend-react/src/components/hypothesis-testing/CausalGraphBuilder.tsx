import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDrag, useDrop } from 'react-dnd';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Slider,
  Switch,
  FormControlLabel,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Link as LinkIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  Timeline as TimelineIcon,
  Place as PlaceIcon
} from '@mui/icons-material';
import * as d3 from 'd3';
import {
  CausalHypothesis,
  CausalVariable,
  CausalEdge,
  GeologicalProcess,
  CollaborativeSession,
  VariableType,
  CausalRelationshipType,
  TemporalConstraint,
  SpatialConstraint
} from '../../types/causal-analysis';

interface CausalGraphBuilderProps {
  availableVariables: CausalVariable[];
  geologicalProcesses: GeologicalProcess[];
  hypothesis: CausalHypothesis | null;
  onHypothesisChange: (hypothesis: CausalHypothesis) => void;
  collaborativeSession?: CollaborativeSession;
}

interface DraggedVariable {
  variable: CausalVariable;
  type: 'variable';
}

interface GraphNode extends CausalVariable {
  x: number;
  y: number;
  fx?: number;
  fy?: number;
}

interface GraphLink extends CausalEdge {
  source: GraphNode;
  target: GraphNode;
}

export const CausalGraphBuilder: React.FC<CausalGraphBuilderProps> = ({
  availableVariables,
  geologicalProcesses,
  hypothesis,
  onHypothesisChange,
  collaborativeSession
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());
  const [selectedEdge, setSelectedEdge] = useState<CausalEdge | null>(null);
  const [showEdgeDialog, setShowEdgeDialog] = useState(false);
  const [showVariableDialog, setShowVariableDialog] = useState(false);
  const [editingVariable, setEditingVariable] = useState<CausalVariable | null>(null);
  const [newEdge, setNewEdge] = useState<Partial<CausalEdge>>({});
  const [graphDimensions, setGraphDimensions] = useState({ width: 600, height: 400 });
  const [simulation, setSimulation] = useState<d3.Simulation<GraphNode, GraphLink> | null>(null);

  // Initialize or update hypothesis
  const currentHypothesis = hypothesis || {
    id: `hypothesis_${Date.now()}`,
    name: 'New Causal Hypothesis',
    description: '',
    variables: [],
    edges: [],
    constraints: {
      temporal: [],
      spatial: [],
      geological: []
    },
    metadata: {
      created: new Date(),
      lastModified: new Date(),
      author: collaborativeSession?.currentUser || 'anonymous',
      version: 1
    }
  };

  // D3 force simulation setup
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = graphDimensions.width;
    const height = graphDimensions.height;

    // Clear previous content
    svg.selectAll('*').remove();

    // Create simulation
    const sim = d3.forceSimulation<GraphNode>()
      .force('link', d3.forceLink<GraphNode, GraphLink>().id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    setSimulation(sim);

    return () => {
      sim.stop();
    };
  }, [graphDimensions]);

  // Update graph when hypothesis changes
  useEffect(() => {
    if (!simulation || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const nodes: GraphNode[] = currentHypothesis.variables.map(v => ({
      ...v,
      x: v.x || Math.random() * graphDimensions.width,
      y: v.y || Math.random() * graphDimensions.height
    }));

    const links: GraphLink[] = currentHypothesis.edges.map(edge => ({
      ...edge,
      source: nodes.find(n => n.id === edge.sourceId)!,
      target: nodes.find(n => n.id === edge.targetId)!
    })).filter(link => link.source && link.target);

    // Update simulation
    simulation.nodes(nodes);
    simulation.force<d3.ForceLink<GraphNode, GraphLink>>('link')?.links(links);

    // Draw links
    const linkSelection = svg.selectAll<SVGLineElement, GraphLink>('.link')
      .data(links, d => `${d.sourceId}-${d.targetId}`);

    linkSelection.exit().remove();

    const linkEnter = linkSelection.enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke', d => getEdgeColor(d.relationshipType))
      .attr('stroke-width', d => Math.max(1, d.strength * 5))
      .attr('stroke-dasharray', d => d.relationshipType === 'inhibitory' ? '5,5' : 'none')
      .attr('marker-end', 'url(#arrowhead)')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedEdge(d);
        setShowEdgeDialog(true);
      });

    // Draw nodes
    const nodeSelection = svg.selectAll<SVGGElement, GraphNode>('.node')
      .data(nodes, d => d.id);

    nodeSelection.exit().remove();

    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      )
      .on('click', (event, d) => {
        if (event.ctrlKey || event.metaKey) {
          // Multi-select for edge creation
          const newSelected = new Set(selectedNodes);
          if (newSelected.has(d.id)) {
            newSelected.delete(d.id);
          } else {
            newSelected.add(d.id);
          }
          setSelectedNodes(newSelected);
        } else {
          // Single select for editing
          setEditingVariable(d);
          setShowVariableDialog(true);
        }
      });

    // Node circles
    nodeEnter.append('circle')
      .attr('r', 20)
      .attr('fill', d => getVariableColor(d.type))
      .attr('stroke', d => selectedNodes.has(d.id) ? '#ff6b6b' : '#333')
      .attr('stroke-width', d => selectedNodes.has(d.id) ? 3 : 1);

    // Node labels
    nodeEnter.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '10px')
      .attr('fill', 'white')
      .text(d => d.name.substring(0, 8));

    // Update positions on simulation tick
    simulation.on('tick', () => {
      svg.selectAll<SVGLineElement, GraphLink>('.link')
        .attr('x1', d => d.source.x!)
        .attr('y1', d => d.source.y!)
        .attr('x2', d => d.target.x!)
        .attr('y2', d => d.target.y!);

      svg.selectAll<SVGGElement, GraphNode>('.node')
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Add arrowhead marker
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#666');

    simulation.restart();
  }, [currentHypothesis, selectedNodes, simulation, graphDimensions]);

  // Drop zone for adding variables
  const [{ isOver }, drop] = useDrop<DraggedVariable, void, { isOver: boolean }>({
    accept: 'variable',
    drop: (item, monitor) => {
      const offset = monitor.getClientOffset();
      if (offset && svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        const x = offset.x - rect.left;
        const y = offset.y - rect.top;
        
        addVariableToHypothesis({
          ...item.variable,
          x,
          y
        });
      }
    },
    collect: (monitor) => ({
      isOver: monitor.isOver()
    })
  });

  const addVariableToHypothesis = useCallback((variable: CausalVariable) => {
    if (currentHypothesis.variables.find(v => v.id === variable.id)) return;

    const updatedHypothesis = {
      ...currentHypothesis,
      variables: [...currentHypothesis.variables, variable],
      metadata: {
        ...currentHypothesis.metadata,
        lastModified: new Date()
      }
    };
    onHypothesisChange(updatedHypothesis);
  }, [currentHypothesis, onHypothesisChange]);

  const removeVariable = useCallback((variableId: string) => {
    const updatedHypothesis = {
      ...currentHypothesis,
      variables: currentHypothesis.variables.filter(v => v.id !== variableId),
      edges: currentHypothesis.edges.filter(e => e.sourceId !== variableId && e.targetId !== variableId),
      metadata: {
        ...currentHypothesis.metadata,
        lastModified: new Date()
      }
    };
    onHypothesisChange(updatedHypothesis);
  }, [currentHypothesis, onHypothesisChange]);

  const createEdge = useCallback(() => {
    if (selectedNodes.size === 2) {
      const [sourceId, targetId] = Array.from(selectedNodes);
      const edge: CausalEdge = {
        id: `edge_${sourceId}_${targetId}`,
        sourceId,
        targetId,
        relationshipType: 'causal',
        strength: 0.5,
        confidence: 0.8,
        temporalLag: 0,
        spatialScale: 'local',
        constraints: [],
        metadata: {
          created: new Date(),
          discoveryMethod: 'manual',
          validationStatus: 'proposed'
        }
      };

      const updatedHypothesis = {
        ...currentHypothesis,
        edges: [...currentHypothesis.edges, edge],
        metadata: {
          ...currentHypothesis.metadata,
          lastModified: new Date()
        }
      };
      onHypothesisChange(updatedHypothesis);
      setSelectedNodes(new Set());
    }
  }, [selectedNodes, currentHypothesis, onHypothesisChange]);

  const updateEdge = useCallback((updatedEdge: CausalEdge) => {
    const updatedHypothesis = {
      ...currentHypothesis,
      edges: currentHypothesis.edges.map(e => e.id === updatedEdge.id ? updatedEdge : e),
      metadata: {
        ...currentHypothesis.metadata,
        lastModified: new Date()
      }
    };
    onHypothesisChange(updatedHypothesis);
  }, [currentHypothesis, onHypothesisChange]);

  const removeEdge = useCallback((edgeId: string) => {
    const updatedHypothesis = {
      ...currentHypothesis,
      edges: currentHypothesis.edges.filter(e => e.id !== edgeId),
      metadata: {
        ...currentHypothesis.metadata,
        lastModified: new Date()
      }
    };
    onHypothesisChange(updatedHypothesis);
  }, [currentHypothesis, onHypothesisChange]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PsychologyIcon />
          Causal Graph Builder
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button
            variant="outlined"
            startIcon={<LinkIcon />}
            onClick={createEdge}
            disabled={selectedNodes.size !== 2}
            size="small"
          >
            Create Edge ({selectedNodes.size}/2)
          </Button>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setShowVariableDialog(true)}
            size="small"
          >
            Add Variable
          </Button>
        </Box>

        {selectedNodes.size > 0 && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Selected {selectedNodes.size} node(s). {selectedNodes.size === 2 ? 'Click "Create Edge" to connect them.' : 'Select one more node to create an edge.'}
          </Alert>
        )}
      </Box>

      {/* Variable Palette */}
      <Accordion sx={{ mb: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">Available Variables ({availableVariables.length})</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, maxHeight: 150, overflow: 'auto' }}>
            {availableVariables.map(variable => (
              <DraggableVariable key={variable.id} variable={variable} />
            ))}
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Graph Canvas */}
      <Paper 
        ref={drop}
        sx={{ 
          flex: 1, 
          position: 'relative',
          border: isOver ? '2px dashed #2196f3' : '1px solid #e0e0e0',
          backgroundColor: isOver ? 'rgba(33, 150, 243, 0.1)' : 'white'
        }}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`0 0 ${graphDimensions.width} ${graphDimensions.height}`}
          style={{ display: 'block' }}
        />
        
        {currentHypothesis.variables.length === 0 && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              color: 'text.secondary'
            }}
          >
            <Typography variant="h6">Drag variables here to build your hypothesis</Typography>
            <Typography variant="body2">Or click "Add Variable" to create custom variables</Typography>
          </Box>
        )}
      </Paper>

      {/* Current Hypothesis Summary */}
      {currentHypothesis.variables.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Current Hypothesis: {currentHypothesis.variables.length} variables, {currentHypothesis.edges.length} edges
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {currentHypothesis.variables.map(variable => (
              <Chip
                key={variable.id}
                label={variable.name}
                size="small"
                onDelete={() => removeVariable(variable.id)}
                color={selectedNodes.has(variable.id) ? 'primary' : 'default'}
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Edge Configuration Dialog */}
      <Dialog open={showEdgeDialog} onClose={() => setShowEdgeDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Configure Causal Relationship</DialogTitle>
        <DialogContent>
          {selectedEdge && (
            <EdgeConfigurationPanel
              edge={selectedEdge}
              variables={currentHypothesis.variables}
              onEdgeUpdate={updateEdge}
              onEdgeDelete={() => {
                removeEdge(selectedEdge.id);
                setShowEdgeDialog(false);
              }}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowEdgeDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Variable Configuration Dialog */}
      <Dialog open={showVariableDialog} onClose={() => setShowVariableDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingVariable ? 'Edit Variable' : 'Add Custom Variable'}
        </DialogTitle>
        <DialogContent>
          <VariableConfigurationPanel
            variable={editingVariable}
            onVariableCreate={addVariableToHypothesis}
            onVariableUpdate={(variable) => {
              const updatedHypothesis = {
                ...currentHypothesis,
                variables: currentHypothesis.variables.map(v => v.id === variable.id ? variable : v)
              };
              onHypothesisChange(updatedHypothesis);
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowVariableDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Draggable Variable Component
const DraggableVariable: React.FC<{ variable: CausalVariable }> = ({ variable }) => {
  const [{ isDragging }, drag] = useDrag<DraggedVariable, void, { isDragging: boolean }>({
    type: 'variable',
    item: { variable, type: 'variable' },
    collect: (monitor) => ({
      isDragging: monitor.isDragging()
    })
  });

  return (
    <Chip
      ref={drag}
      label={variable.name}
      size="small"
      sx={{
        opacity: isDragging ? 0.5 : 1,
        cursor: 'grab',
        '&:active': { cursor: 'grabbing' }
      }}
      color={getVariableChipColor(variable.type)}
    />
  );
};

// Edge Configuration Panel
const EdgeConfigurationPanel: React.FC<{
  edge: CausalEdge;
  variables: CausalVariable[];
  onEdgeUpdate: (edge: CausalEdge) => void;
  onEdgeDelete: () => void;
}> = ({ edge, variables, onEdgeUpdate, onEdgeDelete }) => {
  const [localEdge, setLocalEdge] = useState(edge);

  const sourceVar = variables.find(v => v.id === edge.sourceId);
  const targetVar = variables.find(v => v.id === edge.targetId);

  const handleUpdate = (field: keyof CausalEdge, value: any) => {
    const updated = { ...localEdge, [field]: value };
    setLocalEdge(updated);
    onEdgeUpdate(updated);
  };

  return (
    <Box sx={{ pt: 2 }}>
      <Typography variant="subtitle1" gutterBottom>
        {sourceVar?.name} → {targetVar?.name}
      </Typography>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Relationship Type</InputLabel>
        <Select
          value={localEdge.relationshipType}
          onChange={(e) => handleUpdate('relationshipType', e.target.value)}
        >
          <MenuItem value="causal">Causal</MenuItem>
          <MenuItem value="inhibitory">Inhibitory</MenuItem>
          <MenuItem value="correlational">Correlational</MenuItem>
          <MenuItem value="confounding">Confounding</MenuItem>
        </Select>
      </FormControl>

      <Typography gutterBottom>Causal Strength: {localEdge.strength.toFixed(2)}</Typography>
      <Slider
        value={localEdge.strength}
        onChange={(_, value) => handleUpdate('strength', value)}
        min={0}
        max={1}
        step={0.01}
        sx={{ mb: 2 }}
      />

      <Typography gutterBottom>Confidence: {localEdge.confidence.toFixed(2)}</Typography>
      <Slider
        value={localEdge.confidence}
        onChange={(_, value) => handleUpdate('confidence', value)}
        min={0}
        max={1}
        step={0.01}
        sx={{ mb: 2 }}
      />

      <TextField
        fullWidth
        label="Temporal Lag (days)"
        type="number"
        value={localEdge.temporalLag}
        onChange={(e) => handleUpdate('temporalLag', parseFloat(e.target.value) || 0)}
        sx={{ mb: 2 }}
      />

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Spatial Scale</InputLabel>
        <Select
          value={localEdge.spatialScale}
          onChange={(e) => handleUpdate('spatialScale', e.target.value)}
        >
          <MenuItem value="microscopic">Microscopic</MenuItem>
          <MenuItem value="local">Local</MenuItem>
          <MenuItem value="regional">Regional</MenuItem>
          <MenuItem value="global">Global</MenuItem>
        </Select>
      </FormControl>

      <Button
        variant="outlined"
        color="error"
        startIcon={<DeleteIcon />}
        onClick={onEdgeDelete}
        fullWidth
      >
        Delete Relationship
      </Button>
    </Box>
  );
};

// Variable Configuration Panel
const VariableConfigurationPanel: React.FC<{
  variable?: CausalVariable | null;
  onVariableCreate: (variable: CausalVariable) => void;
  onVariableUpdate: (variable: CausalVariable) => void;
}> = ({ variable, onVariableCreate, onVariableUpdate }) => {
  const [localVariable, setLocalVariable] = useState<Partial<CausalVariable>>(
    variable || {
      name: '',
      type: 'continuous',
      unit: '',
      description: '',
      geologicalContext: 'volcanic'
    }
  );

  const handleSave = () => {
    if (localVariable.name) {
      const fullVariable: CausalVariable = {
        id: variable?.id || `var_${Date.now()}`,
        name: localVariable.name!,
        type: localVariable.type as VariableType,
        unit: localVariable.unit || '',
        description: localVariable.description || '',
        geologicalContext: localVariable.geologicalContext || 'volcanic',
        constraints: localVariable.constraints || [],
        metadata: {
          created: new Date(),
          source: 'manual',
          quality: 'high'
        }
      };

      if (variable) {
        onVariableUpdate(fullVariable);
      } else {
        onVariableCreate(fullVariable);
      }
    }
  };

  return (
    <Box sx={{ pt: 2 }}>
      <TextField
        fullWidth
        label="Variable Name"
        value={localVariable.name || ''}
        onChange={(e) => setLocalVariable(prev => ({ ...prev, name: e.target.value }))}
        sx={{ mb: 2 }}
      />

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Variable Type</InputLabel>
        <Select
          value={localVariable.type || 'continuous'}
          onChange={(e) => setLocalVariable(prev => ({ ...prev, type: e.target.value as VariableType }))}
        >
          <MenuItem value="continuous">Continuous</MenuItem>
          <MenuItem value="discrete">Discrete</MenuItem>
          <MenuItem value="categorical">Categorical</MenuItem>
          <MenuItem value="binary">Binary</MenuItem>
        </Select>
      </FormControl>

      <TextField
        fullWidth
        label="Unit"
        value={localVariable.unit || ''}
        onChange={(e) => setLocalVariable(prev => ({ ...prev, unit: e.target.value }))}
        sx={{ mb: 2 }}
      />

      <TextField
        fullWidth
        label="Description"
        multiline
        rows={3}
        value={localVariable.description || ''}
        onChange={(e) => setLocalVariable(prev => ({ ...prev, description: e.target.value }))}
        sx={{ mb: 2 }}
      />

      <Button
        variant="contained"
        onClick={handleSave}
        disabled={!localVariable.name}
        fullWidth
      >
        {variable ? 'Update Variable' : 'Create Variable'}
      </Button>
    </Box>
  );
};

// Utility functions
function getVariableColor(type: VariableType): string {
  switch (type) {
    case 'continuous': return '#2196f3';
    case 'discrete': return '#4caf50';
    case 'categorical': return '#ff9800';
    case 'binary': return '#9c27b0';
    default: return '#757575';
  }
}

function getVariableChipColor(type: VariableType): 'primary' | 'secondary' | 'success' | 'warning' {
  switch (type) {
    case 'continuous': return 'primary';
    case 'discrete': return 'success';
    case 'categorical': return 'warning';
    case 'binary': return 'secondary';
    default: return 'primary';
  }
}

function getEdgeColor(type: CausalRelationshipType): string {
  switch (type) {
    case 'causal': return '#4caf50';
    case 'inhibitory': return '#f44336';
    case 'correlational': return '#2196f3';
    case 'confounding': return '#ff9800';
    default: return '#757575';
  }
}