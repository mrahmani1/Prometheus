/**
 * MLOps Integration Interface Component
 * Visual workflow builder and model management for geological ML pipelines
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  IconButton,
  Chip,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  Paper,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Stepper,
  Step,
  StepLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
} from '@mui/material';
import {
  ModelTraining,
  Timeline,
  PlayArrow,
  Pause,
  Delete,
  Edit,
  Visibility,
  CheckCircle,
  Error as ErrorIcon,
  Info,
  Speed,
  Memory,
  Compare,
  Tune,
} from '@mui/icons-material';
// Removed unused useTheme import
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface MLModel {
  id: string;
  name: string;
  type: 'classification' | 'regression' | 'clustering' | 'anomaly_detection' | 'time_series';
  algorithm: string;
  version: string;
  status: 'training' | 'deployed' | 'failed' | 'archived';
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    rmse?: number;
    mae?: number;
    r2Score?: number;
  };
  trainingData: {
    samples: number;
    features: number;
    classes?: number;
  };
  performance: {
    trainingTime: number; // seconds
    inferenceTime: number; // milliseconds
    memoryUsage: number; // MB
    cpuUsage: number; // percentage
  };
  hyperparameters: Record<string, any>;
  createdAt: Date;
  lastTrained: Date;
  deployedAt?: Date;
}

interface MLPipeline {
  id: string;
  name: string;
  description: string;
  steps: MLPipelineStep[];
  status: 'idle' | 'running' | 'completed' | 'failed';
  progress: number;
  models: string[]; // Model IDs
  schedule?: {
    enabled: boolean;
    cron: string;
    nextRun: Date;
  };
  metadata: {
    createdBy: string;
    createdAt: Date;
    lastRun?: Date;
    totalRuns: number;
  };
}

interface MLPipelineStep {
  id: string;
  name: string;
  type: 'data_preprocessing' | 'feature_engineering' | 'model_training' | 'model_evaluation' | 'deployment';
  config: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration?: number;
  outputs?: any;
}

// Removed unused ModelComparison interface

interface MLOpsIntegrationInterfaceProps {
  models: MLModel[];
  pipelines: MLPipeline[];
  onModelAction?: (modelId: string, action: string) => void;
  onPipelineAction?: (pipelineId: string, action: string) => void;
  onCreatePipeline?: (pipeline: Partial<MLPipeline>) => void;
  realTimeMonitoring?: boolean;
}

const MLOpsIntegrationInterface: React.FC<MLOpsIntegrationInterfaceProps> = ({
  models,
  pipelines,
  onModelAction,
  onPipelineAction,
  onCreatePipeline,
  realTimeMonitoring = false,
}) => {
  // Removed unused theme

  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [showModelComparison, setShowModelComparison] = useState(false);
  const [showPipelineBuilder, setShowPipelineBuilder] = useState(false);
  const [newPipeline, setNewPipeline] = useState<Partial<MLPipeline>>({
    name: '',
    description: '',
    steps: [],
  });
  const [monitoringMetrics, setMonitoringMetrics] = useState<Record<string, number[]>>({});
  // Removed unused isLoading state
  const [error, setError] = useState<string | null>(null);

  // Model performance metrics over time
  useEffect(() => {
    if (realTimeMonitoring) {
      const interval = setInterval(() => {
        // Simulate real-time metrics updates
        const newMetrics: Record<string, number[]> = {};
        models.forEach(model => {
          if (!monitoringMetrics[model.id]) {
            newMetrics[model.id] = [];
          } else {
            newMetrics[model.id] = [...monitoringMetrics[model.id]];
          }
          
          // Add new metric point
          newMetrics[model.id].push(
            (model.metrics.accuracy || model.metrics.r2Score || Math.random()) + 
            (Math.random() - 0.5) * 0.1
          );
          
          // Keep only last 50 points
          if (newMetrics[model.id].length > 50) {
            newMetrics[model.id] = newMetrics[model.id].slice(-50);
          }
        });
        setMonitoringMetrics(newMetrics);
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [realTimeMonitoring, models, monitoringMetrics]);

  // Generate model performance comparison chart
  const modelComparisonChart = useMemo(() => {
    if (selectedModels.length === 0) return { data: [], layout: {} };

    const selectedModelData = models.filter(m => selectedModels.includes(m.id));
    const metrics = ['accuracy', 'precision', 'recall', 'f1Score', 'rmse', 'mae', 'r2Score'];
    
    const traces: Data[] = metrics.map(metric => ({
      type: 'bar',
      name: metric.toUpperCase(),
      x: selectedModelData.map(m => m.name),
      y: selectedModelData.map(m => (m.metrics as any)[metric] || 0),
      hovertemplate: `<b>%{x}</b><br>${metric}: %{y:.3f}<extra></extra>`,
    }));

    return {
      data: traces,
      layout: {
        title: 'Model Performance Comparison',
        xaxis: { title: 'Models' },
        yaxis: { title: 'Metric Value' },
        barmode: 'group',
        height: 400,
      } as Partial<Layout>,
    };
  }, [selectedModels, models]);

  // Generate real-time monitoring chart
  const monitoringChart = useMemo(() => {
    const traces: Data[] = Object.entries(monitoringMetrics).map(([modelId, values]) => {
      const model = models.find(m => m.id === modelId);
      return {
        type: 'scatter',
        mode: 'lines+markers',
        name: model?.name || modelId,
        x: values.map((_, i) => i),
        y: values,
        line: { width: 2 },
        marker: { size: 4 },
        hovertemplate: `<b>${model?.name}</b><br>Time: %{x}<br>Performance: %{y:.3f}<extra></extra>`,
      };
    });

    return {
      data: traces,
      layout: {
        title: 'Real-time Model Performance Monitoring',
        xaxis: { title: 'Time Steps' },
        yaxis: { title: 'Performance Metric' },
        height: 400,
      } as Partial<Layout>,
    };
  }, [monitoringMetrics, models]);

  // Handle model selection for comparison
  const handleModelSelection = useCallback((modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  }, []);

  // Handle pipeline step addition
  const addPipelineStep = useCallback((stepType: MLPipelineStep['type']) => {
    const newStep: MLPipelineStep = {
      id: `step_${Date.now()}`,
      name: stepType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      type: stepType,
      config: {},
      status: 'pending',
    };

    setNewPipeline(prev => ({
      ...prev,
      steps: [...(prev.steps || []), newStep],
    }));
  }, []);

  // Handle pipeline creation
  const handleCreatePipeline = useCallback(() => {
    if (!newPipeline.name || !newPipeline.steps || newPipeline.steps.length === 0) {
      setError('Pipeline name and steps are required');
      return;
    }

    const pipeline: Partial<MLPipeline> = {
      ...newPipeline,
      id: `pipeline_${Date.now()}`,
      status: 'idle',
      progress: 0,
      models: [],
      metadata: {
        createdBy: 'current_user',
        createdAt: new Date(),
        totalRuns: 0,
      },
    };

    onCreatePipeline?.(pipeline);
    setNewPipeline({ name: '', description: '', steps: [] });
    setShowPipelineBuilder(false);
  }, [newPipeline, onCreatePipeline]);

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': case 'completed': return 'success';
      case 'training': case 'running': return 'primary';
      case 'failed': return 'error';
      case 'archived': case 'idle': return 'default';
      default: return 'default';
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': case 'completed': return <CheckCircle />;
      case 'training': case 'running': return <Timeline />;
      case 'failed': return <ErrorIcon />;
      case 'archived': case 'idle': return <Pause />;
      default: return <Info />;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="h5">MLOps Integration</Typography>
            <Typography variant="body2" color="textSecondary">
              Machine Learning Operations and Model Management
            </Typography>
          </Grid>
          
          <Grid item>
            <Button
              variant="contained"
              startIcon={<ModelTraining />}
              onClick={() => setShowPipelineBuilder(true)}
            >
              Create Pipeline
            </Button>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Compare />}
              onClick={() => setShowModelComparison(true)}
              disabled={selectedModels.length < 2}
            >
              Compare Models
            </Button>
          </Grid>

          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={realTimeMonitoring}
                  onChange={() => {}}
                  disabled
                />
              }
              label="Real-time Monitoring"
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Removed isLoading check as state was removed */}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Main Content */}
      <Paper elevation={1}>
        <Tabs
          value={activeTab}
          onChange={(_: React.SyntheticEvent, newValue: number) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Models" />
          <Tab label="Pipelines" />
          <Tab label="Monitoring" />
          <Tab label="Experiments" />
        </Tabs>

        <Box p={3}>
          {/* Models Tab */}
          {activeTab === 0 && (
            <Grid container spacing={2}>
              {models.map((model) => (
                <Grid item xs={12} md={6} lg={4} key={model.id}>
                  <Card 
                    variant={selectedModels.includes(model.id) ? "outlined" : "elevation"}
                    sx={{ 
                      border: selectedModels.includes(model.id) ? 2 : 0,
                      borderColor: 'primary.main',
                      cursor: 'pointer',
                    }}
                    onClick={() => handleModelSelection(model.id)}
                  >
                    <CardContent>
                      <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
                        <Box flexGrow={1}>
                          <Typography variant="h6" gutterBottom>
                            {model.name}
                          </Typography>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            {model.algorithm} • v{model.version}
                          </Typography>
                        </Box>
                        <Chip
                          icon={getStatusIcon(model.status)}
                          label={model.status}
                          color={getStatusColor(model.status)}
                          size="small"
                        />
                      </Box>

                      {/* Model Metrics */}
                      <Grid container spacing={1} mb={2}>
                        {Object.entries(model.metrics).map(([metric, value]) => (
                          <Grid item xs={6} key={metric}>
                            <Typography variant="caption" display="block">
                              {metric.toUpperCase()}
                            </Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {value?.toFixed(3) || 'N/A'}
                            </Typography>
                          </Grid>
                        ))}
                      </Grid>

                      {/* Performance Indicators */}
                      <Box mb={2}>
                        <Typography variant="caption" display="block">
                          Performance
                        </Typography>
                        <Grid container spacing={1}>
                          <Grid item xs={4}>
                            <Tooltip title="Training Time">
                              <Box display="flex" alignItems="center" gap={0.5}>
                                <Timeline fontSize="small" />
                                <Typography variant="caption">
                                  {model.performance.trainingTime}s
                                </Typography>
                              </Box>
                            </Tooltip>
                          </Grid>
                          <Grid item xs={4}>
                            <Tooltip title="Memory Usage">
                              <Box display="flex" alignItems="center" gap={0.5}>
                                <Memory fontSize="small" />
                                <Typography variant="caption">
                                  {model.performance.memoryUsage}MB
                                </Typography>
                              </Box>
                            </Tooltip>
                          </Grid>
                          <Grid item xs={4}>
                            <Tooltip title="Inference Time">
                              <Box display="flex" alignItems="center" gap={0.5}>
                                <Speed fontSize="small" />
                                <Typography variant="caption">
                                  {model.performance.inferenceTime}ms
                                </Typography>
                              </Box>
                            </Tooltip>
                          </Grid>
                        </Grid>
                      </Box>

                      {/* Actions */}
                      <Box display="flex" gap={1}>
                        <Button
                          size="small"
                          startIcon={<PlayArrow />}
                          onClick={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            onModelAction?.(model.id, 'deploy');
                          }}
                          disabled={model.status === 'deployed'}
                        >
                          Deploy
                        </Button>
                        <Button
                          size="small"
                          startIcon={<Tune />}
                          onClick={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            onModelAction?.(model.id, 'tune');
                          }}
                        >
                          Tune
                        </Button>
                        <IconButton
                          size="small"
                          onClick={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            onModelAction?.(model.id, 'delete');
                          }}
                        >
                          <Delete />
                        </IconButton>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}

          {/* Pipelines Tab */}
          {activeTab === 1 && (
            <Box>
              <Grid container spacing={2}>
                {pipelines.map((pipeline) => (
                  <Grid item xs={12} key={pipeline.id}>
                    <Card>
                      <CardContent>
                        <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
                          <Box flexGrow={1}>
                            <Typography variant="h6" gutterBottom>
                              {pipeline.name}
                            </Typography>
                            <Typography variant="body2" color="textSecondary" gutterBottom>
                              {pipeline.description}
                            </Typography>
                          </Box>
                          <Box display="flex" alignItems="center" gap={1}>
                            <Chip
                              icon={getStatusIcon(pipeline.status)}
                              label={pipeline.status}
                              color={getStatusColor(pipeline.status)}
                              size="small"
                            />
                            {pipeline.status === 'running' && (
                              <LinearProgress
                                variant="determinate"
                                value={pipeline.progress}
                                sx={{ width: 100 }}
                              />
                            )}
                          </Box>
                        </Box>

                        {/* Pipeline Steps */}
                        <Stepper orientation="horizontal" sx={{ mb: 2 }}>
                          {pipeline.steps.map((step) => (
                            <Step key={step.id} completed={step.status === 'completed'}>
                              <StepLabel
                                error={step.status === 'failed'}
                                icon={step.status === 'running' ? <Timeline /> : undefined}
                              >
                                {step.name}
                              </StepLabel>
                            </Step>
                          ))}
                        </Stepper>

                        {/* Pipeline Metadata */}
                        <Grid container spacing={2} mb={2}>
                          <Grid item xs={3}>
                            <Typography variant="caption">Created</Typography>
                            <Typography variant="body2">
                              {pipeline.metadata.createdAt.toLocaleDateString()}
                            </Typography>
                          </Grid>
                          <Grid item xs={3}>
                            <Typography variant="caption">Total Runs</Typography>
                            <Typography variant="body2">
                              {pipeline.metadata.totalRuns}
                            </Typography>
                          </Grid>
                          <Grid item xs={3}>
                            <Typography variant="caption">Models</Typography>
                            <Typography variant="body2">
                              {pipeline.models.length}
                            </Typography>
                          </Grid>
                          <Grid item xs={3}>
                            <Typography variant="caption">Schedule</Typography>
                            <Typography variant="body2">
                              {pipeline.schedule?.enabled ? 'Enabled' : 'Disabled'}
                            </Typography>
                          </Grid>
                        </Grid>

                        {/* Actions */}
                        <Box display="flex" gap={1}>
                          <Button
                            size="small"
                            startIcon={<PlayArrow />}
                            onClick={() => onPipelineAction?.(pipeline.id, 'run')}
                            disabled={pipeline.status === 'running'}
                          >
                            Run
                          </Button>
                          <Button
                            size="small"
                            startIcon={<Edit />}
                            onClick={() => onPipelineAction?.(pipeline.id, 'edit')}
                          >
                            Edit
                          </Button>
                          <Button
                            size="small"
                            startIcon={<Visibility />}
                            onClick={() => onPipelineAction?.(pipeline.id, 'view')}
                          >
                            View Results
                          </Button>
                          <IconButton
                            size="small"
                            onClick={() => onPipelineAction?.(pipeline.id, 'delete')}
                          >
                            <Delete />
                          </IconButton>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}

          {/* Monitoring Tab */}
          {activeTab === 2 && (
            <Box>
              {realTimeMonitoring && Object.keys(monitoringMetrics).length > 0 ? (
                <Plot
                  data={monitoringChart.data}
                  layout={monitoringChart.layout}
                  config={{ displayModeBar: true, displaylogo: false }}
                  style={{ width: '100%' }}
                />
              ) : (
                <Alert severity="info">
                  Real-time monitoring is not enabled or no data available
                </Alert>
              )}

              {/* Model Health Status */}
              <Paper elevation={1} sx={{ mt: 2, p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Model Health Status
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>CPU Usage</TableCell>
                        <TableCell>Memory Usage</TableCell>
                        <TableCell>Inference Time</TableCell>
                        <TableCell>Last Updated</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {models.filter(m => m.status === 'deployed').map((model) => (
                        <TableRow key={model.id}>
                          <TableCell>{model.name}</TableCell>
                          <TableCell>
                            <Chip
                              icon={<CheckCircle />}
                              label="Healthy"
                              color="success"
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{model.performance.cpuUsage}%</TableCell>
                          <TableCell>{model.performance.memoryUsage}MB</TableCell>
                          <TableCell>{model.performance.inferenceTime}ms</TableCell>
                          <TableCell>{model.lastTrained.toLocaleString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Box>
          )}

          {/* Experiments Tab */}
          {activeTab === 3 && (
            <Box>
              <Alert severity="info" sx={{ mb: 2 }}>
                Experiment tracking and hyperparameter optimization results
              </Alert>
              
              {/* Hyperparameter Optimization Results */}
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Recent Experiments
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  No experiments available. Create a pipeline with hyperparameter tuning to see results here.
                </Typography>
              </Paper>
            </Box>
          )}
        </Box>
      </Paper>

      {/* Model Comparison Dialog */}
      <Dialog open={showModelComparison} onClose={() => setShowModelComparison(false)} maxWidth="lg" fullWidth>
        <DialogTitle>Model Performance Comparison</DialogTitle>
        <DialogContent>
          {selectedModels.length >= 2 ? (
            <Plot
              data={modelComparisonChart.data}
              layout={modelComparisonChart.layout}
              config={{ displayModeBar: true, displaylogo: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <Alert severity="warning">
              Please select at least 2 models to compare
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowModelComparison(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Pipeline Builder Dialog */}
      <Dialog open={showPipelineBuilder} onClose={() => setShowPipelineBuilder(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create ML Pipeline</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Pipeline Name"
                value={newPipeline.name || ''}
                onChange={(e) => setNewPipeline(prev => ({ ...prev, name: e.target.value }))}
                margin="normal"
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={newPipeline.description || ''}
                onChange={(e) => setNewPipeline(prev => ({ ...prev, description: e.target.value }))}
                margin="normal"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Pipeline Steps
              </Typography>
              <Box display="flex" gap={1} mb={2}>
                <Button
                  size="small"
                  onClick={() => addPipelineStep('data_preprocessing')}
                >
                  Data Preprocessing
                </Button>
                <Button
                  size="small"
                  onClick={() => addPipelineStep('feature_engineering')}
                >
                  Feature Engineering
                </Button>
                <Button
                  size="small"
                  onClick={() => addPipelineStep('model_training')}
                >
                  Model Training
                </Button>
                <Button
                  size="small"
                  onClick={() => addPipelineStep('model_evaluation')}
                >
                  Model Evaluation
                </Button>
                <Button
                  size="small"
                  onClick={() => addPipelineStep('deployment')}
                >
                  Deployment
                </Button>
              </Box>

              {newPipeline.steps && newPipeline.steps.length > 0 && (
                <List>
                  {newPipeline.steps.map((step, index) => (
                    <ListItem key={step.id}>
                      <ListItemIcon>
                        <Typography variant="body2">{index + 1}</Typography>
                      </ListItemIcon>
                      <ListItemText
                        primary={step.name}
                        secondary={step.type}
                      />
                      <IconButton
                        size="small"
                        onClick={() => {
                          setNewPipeline(prev => ({
                            ...prev,
                            steps: prev.steps?.filter(s => s.id !== step.id) || []
                          }));
                        }}
                      >
                        <Delete />
                      </IconButton>
                    </ListItem>
                  ))}
                </List>
              )}
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPipelineBuilder(false)}>Cancel</Button>
          <Button
            onClick={handleCreatePipeline}
            variant="contained"
            disabled={!newPipeline.name || !newPipeline.steps || newPipeline.steps.length === 0}
          >
            Create Pipeline
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MLOpsIntegrationInterface;