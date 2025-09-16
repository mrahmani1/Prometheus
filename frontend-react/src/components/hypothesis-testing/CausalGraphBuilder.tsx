import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Card,
  CardContent,
  Slider
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Link as LinkIcon,
  Science as ScienceIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  CausalVariable,
  CausalEdge,
  GeologicalProcess,
  CollaborativeSession,
  VariableType,
  CausalRelationshipType,
  SpatialScale,
  GeologicalContext
} from '../../types/causal-analysis';

interface CausalGraphBuilderProps {
  availableVariables: CausalVariable[];
  geologicalProcesses: GeologicalProcess[];
  hypothesis: CausalHypothesis | null;
  onHypothesisChange: (hypothesis: CausalHypothesis) => void;
  collaborativeSession?: CollaborativeSession;
}

export const CausalGraphBuilder: React.FC<CausalGraphBuilderProps> = ({
  availableVariables: _availableVariables,
  geologicalProcesses: _geologicalProcesses,
  hypothesis,
  onHypothesisChange,
  collaborativeSession
}) => {
  const [showVariableDialog, setShowVariableDialog] = useState(false);
  const [showEdgeDialog, setShowEdgeDialog] = useState(false);
  const [editingVariable, setEditingVariable] = useState<CausalVariable | null>(null);
  const [editingEdge, setEditingEdge] = useState<CausalEdge | null>(null);
  const [newVariable, setNewVariable] = useState<Partial<CausalVariable>>({
    name: '',
    type: 'continuous',
    unit: '',
    description: '',
    geologicalContext: 'volcanic'
  });
  const [newEdge, setNewEdge] = useState<Partial<CausalEdge>>({
    sourceId: '',
    targetId: '',
    relationshipType: 'causal',
    strength: 0.5,
    confidence: 0.8,
    temporalLag: 0,
    spatialScale: 'local'
  });

  const handleAddVariable = useCallback(() => {
    if (!hypothesis || !newVariable.name) return;

    const variable: CausalVariable = {
      id: `var_${Date.now()}`,
      name: newVariable.name,
      type: newVariable.type as VariableType,
      unit: newVariable.unit || '',
      description: newVariable.description || '',
      geologicalContext: newVariable.geologicalContext as GeologicalContext,
      constraints: [],
      metadata: {
        created: new Date(),
        source: 'manual',
        quality: 'high'
      }
    };

    const updatedHypothesis: CausalHypothesis = {
      ...hypothesis,
      variables: [...hypothesis.variables, variable]
    };

    onHypothesisChange(updatedHypothesis);
    setNewVariable({
      name: '',
      type: 'continuous',
      unit: '',
      description: '',
      geologicalContext: 'volcanic'
    });
    setShowVariableDialog(false);
  }, [hypothesis, newVariable, onHypothesisChange]);

  const handleAddEdge = useCallback(() => {
    if (!hypothesis || !newEdge.sourceId || !newEdge.targetId) return;

    const edge: CausalEdge = {
      id: `edge_${Date.now()}`,
      sourceId: newEdge.sourceId,
      targetId: newEdge.targetId,
      relationshipType: newEdge.relationshipType as CausalRelationshipType,
      strength: newEdge.strength || 0.5,
      confidence: newEdge.confidence || 0.8,
      temporalLag: newEdge.temporalLag || 0,
      spatialScale: newEdge.spatialScale as SpatialScale,
      constraints: [],
      metadata: {
        created: new Date(),
        discoveryMethod: 'manual',
        validationStatus: 'proposed'
      }
    };

    const updatedHypothesis: CausalHypothesis = {
      ...hypothesis,
      edges: [...hypothesis.edges, edge]
    };

    onHypothesisChange(updatedHypothesis);
    setNewEdge({
      sourceId: '',
      targetId: '',
      relationshipType: 'causal',
      strength: 0.5,
      confidence: 0.8,
      temporalLag: 0,
      spatialScale: 'local'
    });
    setShowEdgeDialog(false);
  }, [hypothesis, newEdge, onHypothesisChange]);

  const handleRemoveVariable = useCallback((variableId: string) => {
    if (!hypothesis) return;

    const updatedHypothesis: CausalHypothesis = {
      ...hypothesis,
      variables: hypothesis.variables.filter(v => v.id !== variableId),
      edges: hypothesis.edges.filter(e => e.sourceId !== variableId && e.targetId !== variableId)
    };

    onHypothesisChange(updatedHypothesis);
  }, [hypothesis, onHypothesisChange]);

  const handleRemoveEdge = useCallback((edgeId: string) => {
    if (!hypothesis) return;

    const updatedHypothesis: CausalHypothesis = {
      ...hypothesis,
      edges: hypothesis.edges.filter(e => e.id !== edgeId)
    };

    onHypothesisChange(updatedHypothesis);
  }, [hypothesis, onHypothesisChange]);

  if (!hypothesis) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <ScienceIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          Create a new hypothesis to start building causal relationships
        </Typography>
        <Button
          variant="contained"
          onClick={() => {
            const newHypothesis: CausalHypothesis = {
              id: `hyp_${Date.now()}`,
              name: 'New Hypothesis',
              description: 'A new causal hypothesis',
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
            onHypothesisChange(newHypothesis);
          }}
          sx={{ mt: 2 }}
        >
          Create New Hypothesis
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Causal Graph Builder
      </Typography>

      {/* Variables Section */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1">Variables ({hypothesis.variables.length})</Typography>
            <Button
              startIcon={<AddIcon />}
              onClick={() => setShowVariableDialog(true)}
              size="small"
            >
              Add Variable
            </Button>
          </Box>
          
          <List dense>
            {hypothesis.variables.map((variable) => (
              <ListItem key={variable.id}>
                <ListItemText
                  primary={variable.name}
                  secondary={`${variable.type} | ${variable.geologicalContext} | ${variable.unit}`}
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    onClick={() => {
                      setEditingVariable(variable);
                      setShowVariableDialog(true);
                    }}
                    size="small"
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    edge="end"
                    onClick={() => handleRemoveVariable(variable.id)}
                    size="small"
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Edges Section */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1">Causal Relationships ({hypothesis.edges.length})</Typography>
            <Button
              startIcon={<LinkIcon />}
              onClick={() => setShowEdgeDialog(true)}
              size="small"
              disabled={hypothesis.variables.length < 2}
            >
              Add Relationship
            </Button>
          </Box>
          
          <List dense>
            {hypothesis.edges.map((edge) => {
              const sourceVar = hypothesis.variables.find(v => v.id === edge.sourceId);
              const targetVar = hypothesis.variables.find(v => v.id === edge.targetId);
              
              return (
                <ListItem key={edge.id}>
                  <ListItemText
                    primary={`${sourceVar?.name} → ${targetVar?.name}`}
                    secondary={`${edge.relationshipType} | Strength: ${edge.strength.toFixed(2)} | Confidence: ${edge.confidence.toFixed(2)}`}
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => {
                        setEditingEdge(edge);
                        setShowEdgeDialog(true);
                      }}
                      size="small"
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      edge="end"
                      onClick={() => handleRemoveEdge(edge.id)}
                      size="small"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              );
            })}
          </List>
        </CardContent>
      </Card>

      {/* Variable Dialog */}
      <Dialog open={showVariableDialog} onClose={() => setShowVariableDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingVariable ? 'Edit Variable' : 'Add Variable'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Variable Name"
                value={editingVariable?.name || newVariable.name}
                onChange={(e) => {
                  if (editingVariable) {
                    setEditingVariable({ ...editingVariable, name: e.target.value });
                  } else {
                    setNewVariable({ ...newVariable, name: e.target.value });
                  }
                }}
              />
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Type</InputLabel>
                <Select
                  value={editingVariable?.type || newVariable.type}
                  onChange={(e) => {
                    if (editingVariable) {
                      setEditingVariable({ ...editingVariable, type: e.target.value as VariableType });
                    } else {
                      setNewVariable({ ...newVariable, type: e.target.value as VariableType });
                    }
                  }}
                  label="Type"
                >
                  <MenuItem value="continuous">Continuous</MenuItem>
                  <MenuItem value="discrete">Discrete</MenuItem>
                  <MenuItem value="categorical">Categorical</MenuItem>
                  <MenuItem value="binary">Binary</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Unit"
                value={editingVariable?.unit || newVariable.unit}
                onChange={(e) => {
                  if (editingVariable) {
                    setEditingVariable({ ...editingVariable, unit: e.target.value });
                  } else {
                    setNewVariable({ ...newVariable, unit: e.target.value });
                  }
                }}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Geological Context</InputLabel>
                <Select
                  value={editingVariable?.geologicalContext || newVariable.geologicalContext}
                  onChange={(e) => {
                    if (editingVariable) {
                      setEditingVariable({ ...editingVariable, geologicalContext: e.target.value as GeologicalContext });
                    } else {
                      setNewVariable({ ...newVariable, geologicalContext: e.target.value as GeologicalContext });
                    }
                  }}
                  label="Geological Context"
                >
                  <MenuItem value="volcanic">Volcanic</MenuItem>
                  <MenuItem value="seismic">Seismic</MenuItem>
                  <MenuItem value="tectonic">Tectonic</MenuItem>
                  <MenuItem value="hydrothermal">Hydrothermal</MenuItem>
                  <MenuItem value="magmatic">Magmatic</MenuItem>
                  <MenuItem value="structural">Structural</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={editingVariable?.description || newVariable.description}
                onChange={(e) => {
                  if (editingVariable) {
                    setEditingVariable({ ...editingVariable, description: e.target.value });
                  } else {
                    setNewVariable({ ...newVariable, description: e.target.value });
                  }
                }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setShowVariableDialog(false);
            setEditingVariable(null);
          }}>
            Cancel
          </Button>
          <Button onClick={handleAddVariable} variant="contained">
            {editingVariable ? 'Update' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edge Dialog */}
      <Dialog open={showEdgeDialog} onClose={() => setShowEdgeDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingEdge ? 'Edit Relationship' : 'Add Relationship'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Source Variable</InputLabel>
                <Select
                  value={editingEdge?.sourceId || newEdge.sourceId}
                  onChange={(e) => {
                    if (editingEdge) {
                      setEditingEdge({ ...editingEdge, sourceId: e.target.value });
                    } else {
                      setNewEdge({ ...newEdge, sourceId: e.target.value });
                    }
                  }}
                  label="Source Variable"
                >
                  {hypothesis.variables.map((variable) => (
                    <MenuItem key={variable.id} value={variable.id}>
                      {variable.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Target Variable</InputLabel>
                <Select
                  value={editingEdge?.targetId || newEdge.targetId}
                  onChange={(e) => {
                    if (editingEdge) {
                      setEditingEdge({ ...editingEdge, targetId: e.target.value });
                    } else {
                      setNewEdge({ ...newEdge, targetId: e.target.value });
                    }
                  }}
                  label="Target Variable"
                >
                  {hypothesis.variables.map((variable) => (
                    <MenuItem key={variable.id} value={variable.id}>
                      {variable.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Relationship Type</InputLabel>
                <Select
                  value={editingEdge?.relationshipType || newEdge.relationshipType}
                  onChange={(e) => {
                    if (editingEdge) {
                      setEditingEdge({ ...editingEdge, relationshipType: e.target.value as CausalRelationshipType });
                    } else {
                      setNewEdge({ ...newEdge, relationshipType: e.target.value as CausalRelationshipType });
                    }
                  }}
                  label="Relationship Type"
                >
                  <MenuItem value="causal">Causal</MenuItem>
                  <MenuItem value="inhibitory">Inhibitory</MenuItem>
                  <MenuItem value="correlational">Correlational</MenuItem>
                  <MenuItem value="confounding">Confounding</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Typography gutterBottom>Causal Strength: {(editingEdge?.strength || newEdge.strength || 0.5).toFixed(2)}</Typography>
              <Slider
                value={editingEdge?.strength || newEdge.strength || 0.5}
                onChange={(_e: Event, value: number | number[]) => {
                  if (editingEdge) {
                    setEditingEdge({ ...editingEdge, strength: value as number });
                  } else {
                    setNewEdge({ ...newEdge, strength: value as number });
                  }
                }}
                min={0}
                max={1}
                step={0.01}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography gutterBottom>Confidence: {(editingEdge?.confidence || newEdge.confidence || 0.8).toFixed(2)}</Typography>
              <Slider
                value={editingEdge?.confidence || newEdge.confidence || 0.8}
                onChange={(_e: Event, value: number | number[]) => {
                  if (editingEdge) {
                    setEditingEdge({ ...editingEdge, confidence: value as number });
                  } else {
                    setNewEdge({ ...newEdge, confidence: value as number });
                  }
                }}
                min={0}
                max={1}
                step={0.01}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' }
                ]}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                type="number"
                label="Temporal Lag (days)"
                value={editingEdge?.temporalLag || newEdge.temporalLag || 0}
                onChange={(e) => {
                  const value = parseInt(e.target.value) || 0;
                  if (editingEdge) {
                    setEditingEdge({ ...editingEdge, temporalLag: value });
                  } else {
                    setNewEdge({ ...newEdge, temporalLag: value });
                  }
                }}
              />
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Spatial Scale</InputLabel>
                <Select
                  value={editingEdge?.spatialScale || newEdge.spatialScale}
                  onChange={(e) => {
                    if (editingEdge) {
                      setEditingEdge({ ...editingEdge, spatialScale: e.target.value as SpatialScale });
                    } else {
                      setNewEdge({ ...newEdge, spatialScale: e.target.value as SpatialScale });
                    }
                  }}
                  label="Spatial Scale"
                >
                  <MenuItem value="microscopic">Microscopic</MenuItem>
                  <MenuItem value="local">Local</MenuItem>
                  <MenuItem value="regional">Regional</MenuItem>
                  <MenuItem value="global">Global</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setShowEdgeDialog(false);
            setEditingEdge(null);
          }}>
            Cancel
          </Button>
          <Button onClick={handleAddEdge} variant="contained">
            {editingEdge ? 'Update' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};