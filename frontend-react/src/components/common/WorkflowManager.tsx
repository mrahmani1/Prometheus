/**
 * Workflow Manager Component
 * Manages complex geological analysis workflows with multiple modules
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Paper,
  List,

  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,

  Checkbox,
  ListItemButton,

  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Add,
  Delete,
  CheckCircle,
  Error as ErrorIcon,
  Schedule,
  Timeline,
} from '@mui/icons-material';
// Removed unused useTheme import

interface AnalysisModule {
  id: string;
  name: string;
  icon: React.ReactNode;
  enabled: boolean;
  realTimeCapable: boolean;
  dataTypes: string[];
  status: 'active' | 'inactive' | 'error' | 'loading';
}

interface AnalysisWorkflow {
  id: string;
  name: string;
  description: string;
  modules: string[];
  parameters: Record<string, any>;
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  results?: any;
  createdAt: Date;
  lastRun?: Date;
  duration?: number;
}

interface WorkflowStep {
  id: string;
  name: string;
  moduleId: string;
  parameters: Record<string, any>;
  dependencies: string[];
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
  results?: any;
}

interface WorkflowManagerProps {
  workflows: AnalysisWorkflow[];
  availableModules: AnalysisModule[];
  onStartWorkflow: (workflow: Partial<AnalysisWorkflow>) => Promise<void>;
  onStopWorkflow?: (workflowId: string) => void;
  onDeleteWorkflow?: (workflowId: string) => void;
}

const WorkflowManager: React.FC<WorkflowManagerProps> = ({
  workflows,
  availableModules,
  onStartWorkflow,
  onStopWorkflow,
  onDeleteWorkflow,
}) => {
  // Removed unused theme

  // State management
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [newWorkflow, setNewWorkflow] = useState<Partial<AnalysisWorkflow>>({
    name: '',
    description: '',
    modules: [],
    parameters: {},
  });
  const [workflowSteps, setWorkflowSteps] = useState<WorkflowStep[]>([]);

  // Predefined workflow templates
  const workflowTemplates = [
    {
      name: 'Comprehensive Volcanic Analysis',
      description: 'Complete volcanic monitoring workflow with seismic, deformation, and gas analysis',
      modules: ['seismic', 'volcanic', 'deformation'],
      parameters: {
        timeWindow: 24,
        alertThreshold: 0.8,
        realTimeMode: true,
      },
    },
    {
      name: 'Geological Survey Integration',
      description: 'Integrate geological modeling with geochemical analysis',
      modules: ['geological', 'geochemical'],
      parameters: {
        modelResolution: 'high',
        sampleFilter: 'all',
      },
    },
    {
      name: 'Multi-Physics Coupling Analysis',
      description: 'Advanced multi-physics analysis with causal inference',
      modules: ['seismic', 'volcanic', 'deformation', 'geochemical'],
      parameters: {
        couplingStrength: 0.5,
        temporalWindow: 168, // 1 week
        spatialRadius: 50, // km
      },
    },
  ];

  // Handle workflow creation
  const handleCreateWorkflow = useCallback(async () => {
    if (!newWorkflow.name || !newWorkflow.modules || newWorkflow.modules.length === 0) {
      return;
    }

    try {
      await onStartWorkflow({
        ...newWorkflow,
        id: `workflow_${Date.now()}`,
        status: 'idle',
        progress: 0,
        createdAt: new Date(),
      });
      
      setShowCreateDialog(false);
      setNewWorkflow({
        name: '',
        description: '',
        modules: [],
        parameters: {},
      });
    } catch (error) {
      console.error('Failed to create workflow:', error);
    }
  }, [newWorkflow, onStartWorkflow]);

  // Handle workflow template selection
  const handleSelectTemplate = useCallback((template: typeof workflowTemplates[0]) => {
    setNewWorkflow({
      name: template.name,
      description: template.description,
      modules: template.modules,
      parameters: template.parameters,
    });
  }, []);

  // Get workflow status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'primary';
      case 'error': return 'error';
      case 'idle': return 'default';
      default: return 'default';
    }
  };

  // Get workflow status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'running': return <Schedule />;
      case 'error': return <ErrorIcon />;
      case 'idle': return <Pause />;
      default: return <Pause />;
    }
  };

  // Generate workflow steps
  const generateWorkflowSteps = useCallback((workflow: AnalysisWorkflow): WorkflowStep[] => {
    return workflow.modules.map((moduleId, index) => {
      const module = availableModules.find(m => m.id === moduleId);
      return {
        id: `step_${index}`,
        name: module?.name || moduleId,
        moduleId,
        parameters: workflow.parameters,
        dependencies: index > 0 ? [`step_${index - 1}`] : [],
        status: workflow.status === 'running' && index === 0 ? 'running' : 'pending',
        progress: workflow.status === 'completed' ? 100 : 0,
      };
    });
  }, [availableModules]);

  return (
    <Box>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h6">Workflow Manager</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setShowCreateDialog(true)}
        >
          Create Workflow
        </Button>
      </Box>

      {/* Active Workflows */}
      <Grid container spacing={2}>
        {workflows.map((workflow) => (
          <Grid item xs={12} md={6} lg={4} key={workflow.id}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
                  <Box flexGrow={1}>
                    <Typography variant="h6" gutterBottom>
                      {workflow.name}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      {workflow.description}
                    </Typography>
                  </Box>
                  <Chip
                    icon={getStatusIcon(workflow.status)}
                    label={workflow.status}
                    color={getStatusColor(workflow.status)}
                    size="small"
                  />
                </Box>

                {/* Progress */}
                {workflow.status === 'running' && (
                  <Box mb={2}>
                    <LinearProgress
                      variant="determinate"
                      value={workflow.progress}
                      sx={{ mb: 1 }}
                    />
                    <Typography variant="caption">
                      Progress: {workflow.progress}%
                    </Typography>
                  </Box>
                )}

                {/* Modules */}
                <Box mb={2}>
                  <Typography variant="caption" display="block" gutterBottom>
                    Modules:
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={0.5}>
                    {workflow.modules.map((moduleId) => {
                      const module = availableModules.find(m => m.id === moduleId);
                      return (
                        <Chip
                          key={moduleId}
                          label={module?.name || moduleId}
                          size="small"
                          variant="outlined"
                        />
                      );
                    })}
                  </Box>
                </Box>

                {/* Metadata */}
                <Box mb={2}>
                  <Typography variant="caption" display="block">
                    Created: {workflow.createdAt.toLocaleString()}
                  </Typography>
                  {workflow.lastRun && (
                    <Typography variant="caption" display="block">
                      Last Run: {workflow.lastRun.toLocaleString()}
                    </Typography>
                  )}
                  {workflow.duration && (
                    <Typography variant="caption" display="block">
                      Duration: {Math.round(workflow.duration / 1000)}s
                    </Typography>
                  )}
                </Box>

                {/* Actions */}
                <Box display="flex" gap={1}>
                  {workflow.status === 'idle' && (
                    <Button
                      size="small"
                      startIcon={<PlayArrow />}
                      onClick={() => onStartWorkflow(workflow)}
                    >
                      Start
                    </Button>
                  )}
                  {workflow.status === 'running' && onStopWorkflow && (
                    <Button
                      size="small"
                      startIcon={<Stop />}
                      onClick={() => onStopWorkflow(workflow.id)}
                    >
                      Stop
                    </Button>
                  )}
                  <Button
                    size="small"
                    startIcon={<Timeline />}
                    onClick={() => {
                      setSelectedWorkflow(workflow.id);
                      setWorkflowSteps(generateWorkflowSteps(workflow));
                    }}
                  >
                    Details
                  </Button>
                  {onDeleteWorkflow && (
                    <IconButton
                      size="small"
                      onClick={() => onDeleteWorkflow(workflow.id)}
                    >
                      <Delete />
                    </IconButton>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {workflows.length === 0 && (
        <Paper elevation={1} sx={{ p: 4, textAlign: 'center', mt: 2 }}>
          <Typography variant="h6" color="textSecondary" gutterBottom>
            No workflows created yet
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Create your first analysis workflow to get started
          </Typography>
          <Button
            variant="outlined"
            startIcon={<Add />}
            onClick={() => setShowCreateDialog(true)}
          >
            Create Workflow
          </Button>
        </Paper>
      )}

      {/* Create Workflow Dialog */}
      <Dialog open={showCreateDialog} onClose={() => setShowCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Analysis Workflow</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            {/* Templates */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Templates
              </Typography>
              <Grid container spacing={1}>
                {workflowTemplates.map((template, index) => (
                  <Grid item xs={12} md={4} key={index}>
                    <Card
                      variant="outlined"
                      sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                      onClick={() => handleSelectTemplate(template)}
                    >
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          {template.name}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {template.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Grid>

            {/* Workflow Details */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Workflow Name"
                value={newWorkflow.name || ''}
                onChange={(e) => setNewWorkflow(prev => ({ ...prev, name: e.target.value }))}
                margin="normal"
              />
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={newWorkflow.description || ''}
                onChange={(e) => setNewWorkflow(prev => ({ ...prev, description: e.target.value }))}
                margin="normal"
              />
            </Grid>

            {/* Module Selection */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Select Modules
              </Typography>
              <List>
                {availableModules.map((module) => (
                  <ListItemButton
                    key={module.id}
                    onClick={() => {
                      const modules = newWorkflow.modules || [];
                      const isSelected = modules.includes(module.id);
                      setNewWorkflow(prev => ({
                        ...prev,
                        modules: isSelected
                          ? modules.filter(id => id !== module.id)
                          : [...modules, module.id]
                      }));
                    }}
                  >
                    <ListItemIcon>
                      <Checkbox
                        checked={(newWorkflow.modules || []).includes(module.id)}
                      />
                    </ListItemIcon>
                    <ListItemIcon>
                      {module.icon}
                    </ListItemIcon>
                    <ListItemText
                      primary={module.name}
                      secondary={`Data types: ${module.dataTypes.join(', ')}`}
                    />
                    <Chip
                      size="small"
                      label={module.status}
                      color={module.status === 'active' ? 'success' : 'default'}
                    />
                  </ListItemButton>
                ))}
              </List>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreateDialog(false)}>Cancel</Button>
          <Button
            onClick={handleCreateWorkflow}
            variant="contained"
            disabled={!newWorkflow.name || !newWorkflow.modules || newWorkflow.modules.length === 0}
          >
            Create Workflow
          </Button>
        </DialogActions>
      </Dialog>

      {/* Workflow Details Dialog */}
      <Dialog
        open={selectedWorkflow !== null}
        onClose={() => setSelectedWorkflow(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Workflow Details</DialogTitle>
        <DialogContent>
          {selectedWorkflow && (
            <Box>
              <Stepper orientation="vertical">
                {workflowSteps.map((step) => (
                  <Step key={step.id} active={step.status === 'running'} completed={step.status === 'completed'}>
                    <StepLabel
                      error={step.status === 'error'}
                      icon={step.status === 'completed' ? <CheckCircle /> : undefined}
                    >
                      {step.name}
                    </StepLabel>
                    <StepContent>
                      <Typography variant="body2" color="textSecondary">
                        Module: {step.moduleId}
                      </Typography>
                      {step.status === 'running' && (
                        <LinearProgress
                          variant="determinate"
                          value={step.progress}
                          sx={{ mt: 1, mb: 1 }}
                        />
                      )}
                      {step.status === 'error' && (
                        <Alert severity="error" sx={{ mt: 1 }}>
                          Step failed to execute
                        </Alert>
                      )}
                    </StepContent>
                  </Step>
                ))}
              </Stepper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedWorkflow(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default WorkflowManager;