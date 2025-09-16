import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  AutoAwesome as AutoGenerateIcon,
  Visibility as ViewIcon,
  Refresh as RefreshIcon,
  Science as ScienceIcon
} from '@mui/icons-material';
import {
  CausalVariable,
  GeologicalProcess,
  CausalHypothesis,
  HypothesisGenerationConfig,
  GenerationAlgorithm,
  GeneratedHypothesisMetrics
} from '../../types/causal-analysis';

interface AutoHypothesisGeneratorProps {
  availableVariables: CausalVariable[];
  geologicalProcesses: GeologicalProcess[];
  onHypothesesGenerated: (hypotheses: CausalHypothesis[]) => void;
  onSelectHypothesis: (hypothesis: CausalHypothesis) => void;
  generatedHypotheses: CausalHypothesis[];
}

export const AutoHypothesisGenerator: React.FC<AutoHypothesisGeneratorProps> = ({
  availableVariables,
  geologicalProcesses: _geologicalProcesses,
  onHypothesesGenerated,
  onSelectHypothesis,
  generatedHypotheses
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [config, setConfig] = useState<HypothesisGenerationConfig>({
    algorithms: ['correlation-based', 'granger-causality'],
    maxHypotheses: 5,
    minVariables: 3,
    maxVariables: 8,
    minEdges: 2,
    maxEdges: 10,
    significanceThreshold: 0.05,
    effectSizeThreshold: 0.3,
    temporalConstraints: true,
    spatialConstraints: true,
    geologicalConstraints: true,
    domainKnowledge: true,
    noveltyWeight: 0.4,
    plausibilityWeight: 0.4,
    simplicityWeight: 0.2
  });

  // Mock generated hypotheses with metrics
  const mockGeneratedHypotheses: Array<CausalHypothesis & { metrics: GeneratedHypothesisMetrics }> = [
    {
      id: 'gen_hyp_1',
      name: 'Seismic-Volcanic Coupling Model',
      description: 'Auto-generated hypothesis exploring causal relationships between seismic activity and volcanic unrest',
      variables: [],
      edges: [],
      constraints: { temporal: [], spatial: [], geological: [] },
      metadata: {
        created: new Date(),
        lastModified: new Date(),
        author: 'AI Generator',
        version: 1,
        generationMetrics: {
          algorithm: 'correlation-based',
          overallScore: 0.85,
          noveltyScore: 0.7,
          plausibilityScore: 0.9,
          complexity: 6,
          patterns: ['volcanic-unrest', 'seismic-precursors'],
          confidence: 0.82
        }
      },
      metrics: {
        algorithm: 'correlation-based',
        overallScore: 0.85,
        noveltyScore: 0.7,
        plausibilityScore: 0.9,
        complexity: 6,
        patterns: ['volcanic-unrest', 'seismic-precursors'],
        confidence: 0.82
      }
    },
    {
      id: 'gen_hyp_2',
      name: 'Deformation-Gas Emission Network',
      description: 'Causal network linking ground deformation patterns to volcanic gas emissions',
      variables: [],
      edges: [],
      constraints: { temporal: [], spatial: [], geological: [] },
      metadata: {
        created: new Date(),
        lastModified: new Date(),
        author: 'AI Generator',
        version: 1,
        generationMetrics: {
          algorithm: 'granger-causality',
          overallScore: 0.78,
          noveltyScore: 0.8,
          plausibilityScore: 0.75,
          complexity: 8,
          patterns: ['deformation-patterns', 'gas-emissions'],
          confidence: 0.76
        }
      },
      metrics: {
        algorithm: 'granger-causality',
        overallScore: 0.78,
        noveltyScore: 0.8,
        plausibilityScore: 0.75,
        complexity: 8,
        patterns: ['deformation-patterns', 'gas-emissions'],
        confidence: 0.76
      }
    }
  ];

  const handleGenerateHypotheses = async () => {
    setIsGenerating(true);
    
    // Simulate generation process
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // In a real implementation, this would call the backend API
    const generatedHypotheses = mockGeneratedHypotheses.map(h => ({
      id: h.id,
      name: h.name,
      description: h.description,
      variables: h.variables,
      edges: h.edges,
      constraints: h.constraints,
      metadata: h.metadata
    }));
    
    onHypothesesGenerated(generatedHypotheses);
    setIsGenerating(false);
  };

  const getAlgorithmColor = (algorithm: GenerationAlgorithm) => {
    const colors = {
      'correlation-based': 'primary',
      'mutual-information': 'secondary',
      'granger-causality': 'success',
      'pc-algorithm': 'warning',
      'ges-algorithm': 'error',
      'lingam': 'info',
      'causal-forests': 'default'
    } as const;
    return colors[algorithm] || 'default';
  };

  const getPatternColor = (pattern: string) => {
    const colors: Record<string, 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info' | 'default'> = {
      'volcanic-unrest': 'error',
      'seismic-precursors': 'warning',
      'deformation-patterns': 'info',
      'gas-emissions': 'success',
      'thermal-anomalies': 'secondary',
      'magma-intrusion': 'primary'
    };
    return colors[pattern] || 'default';
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AutoGenerateIcon />
        Auto-Hypothesis Generator
      </Typography>

      {/* Generation Configuration */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Generation Configuration
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Algorithms</InputLabel>
                <Select
                  multiple
                  value={config.algorithms}
                  onChange={(e) => setConfig({ ...config, algorithms: e.target.value as GenerationAlgorithm[] })}
                  label="Algorithms"
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  <MenuItem value="correlation-based">Correlation-based</MenuItem>
                  <MenuItem value="mutual-information">Mutual Information</MenuItem>
                  <MenuItem value="granger-causality">Granger Causality</MenuItem>
                  <MenuItem value="pc-algorithm">PC Algorithm</MenuItem>
                  <MenuItem value="ges-algorithm">GES Algorithm</MenuItem>
                  <MenuItem value="lingam">LiNGAM</MenuItem>
                  <MenuItem value="causal-forests">Causal Forests</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Max Hypotheses: {config.maxHypotheses}</Typography>
              <Slider
                value={config.maxHypotheses}
                onChange={(_e: Event, value: number | number[]) => setConfig({ ...config, maxHypotheses: value as number })}
                min={1}
                max={20}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                  { value: 20, label: '20' }
                ]}
              />
            </Grid>
            
            <Grid item xs={6}>
              <Typography gutterBottom>Variables: {config.minVariables} - {config.maxVariables}</Typography>
              <Slider
                value={[config.minVariables, config.maxVariables]}
                onChange={(_e: Event, value: number | number[]) => {
                  const [min, max] = value as number[];
                  setConfig({ ...config, minVariables: min, maxVariables: max });
                }}
                min={2}
                max={15}
                step={1}
                marks={[
                  { value: 2, label: '2' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                  { value: 15, label: '15' }
                ]}
              />
            </Grid>
            
            <Grid item xs={6}>
              <Typography gutterBottom>Relationships: {config.minEdges} - {config.maxEdges}</Typography>
              <Slider
                value={[config.minEdges, config.maxEdges]}
                onChange={(_e: Event, value: number | number[]) => {
                  const [min, max] = value as number[];
                  setConfig({ ...config, minEdges: min, maxEdges: max });
                }}
                min={1}
                max={20}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                  { value: 20, label: '20' }
                ]}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>Scoring Weights</Typography>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography gutterBottom>Novelty: {config.noveltyWeight.toFixed(1)}</Typography>
                  <Slider
                    value={config.noveltyWeight}
                    onChange={(_e: Event, value: number | number[]) => setConfig({ ...config, noveltyWeight: value as number })}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Grid>
                <Grid item xs={4}>
                  <Typography gutterBottom>Plausibility: {config.plausibilityWeight.toFixed(1)}</Typography>
                  <Slider
                    value={config.plausibilityWeight}
                    onChange={(_e: Event, value: number | number[]) => setConfig({ ...config, plausibilityWeight: value as number })}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Grid>
                <Grid item xs={4}>
                  <Typography gutterBottom>Simplicity: {config.simplicityWeight.toFixed(1)}</Typography>
                  <Slider
                    value={config.simplicityWeight}
                    onChange={(_e: Event, value: number | number[]) => setConfig({ ...config, simplicityWeight: value as number })}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Grid>
              </Grid>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.temporalConstraints}
                      onChange={(e) => setConfig({ ...config, temporalConstraints: e.target.checked })}
                    />
                  }
                  label="Temporal Constraints"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.spatialConstraints}
                      onChange={(e) => setConfig({ ...config, spatialConstraints: e.target.checked })}
                    />
                  }
                  label="Spatial Constraints"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.geologicalConstraints}
                      onChange={(e) => setConfig({ ...config, geologicalConstraints: e.target.checked })}
                    />
                  }
                  label="Geological Constraints"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.domainKnowledge}
                      onChange={(e) => setConfig({ ...config, domainKnowledge: e.target.checked })}
                    />
                  }
                  label="Domain Knowledge"
                />
              </Box>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
            <Button
              variant="contained"
              startIcon={<AutoGenerateIcon />}
              onClick={handleGenerateHypotheses}
              disabled={isGenerating || availableVariables.length < config.minVariables}
            >
              Generate Hypotheses
            </Button>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={() => onHypothesesGenerated([])}
              disabled={generatedHypotheses.length === 0}
            >
              Clear Results
            </Button>
          </Box>
          
          {isGenerating && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Generating hypotheses using {config.algorithms.join(', ')}...
              </Typography>
              <LinearProgress />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Generated Hypotheses */}
      {mockGeneratedHypotheses.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Generated Hypotheses ({mockGeneratedHypotheses.length})
            </Typography>
            
            <List>
              {mockGeneratedHypotheses.map((hypothesis) => (
                <ListItem key={hypothesis.id} divider>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <ScienceIcon color="primary" />
                        <Typography variant="subtitle2">{hypothesis.name}</Typography>
                        <Chip
                          label={hypothesis.metrics.algorithm}
                          size="small"
                          color={getAlgorithmColor(hypothesis.metrics.algorithm)}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          {hypothesis.description}
                        </Typography>
                        
                        <Box sx={{ display: 'flex', gap: 2, mb: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="caption">Overall Score:</Typography>
                            <LinearProgress
                              variant="determinate"
                              value={hypothesis.metrics.overallScore * 100}
                              sx={{ width: 60, height: 4 }}
                            />
                            <Typography variant="caption">
                              {(hypothesis.metrics.overallScore * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                          
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="caption">Novelty:</Typography>
                            <LinearProgress
                              variant="determinate"
                              value={hypothesis.metrics.noveltyScore * 100}
                              sx={{ width: 60, height: 4 }}
                              color="secondary"
                            />
                            <Typography variant="caption">
                              {(hypothesis.metrics.noveltyScore * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                          
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="caption">Plausibility:</Typography>
                            <LinearProgress
                              variant="determinate"
                              value={hypothesis.metrics.plausibilityScore * 100}
                              sx={{ width: 60, height: 4 }}
                              color="success"
                            />
                            <Typography variant="caption">
                              {(hypothesis.metrics.plausibilityScore * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        </Box>
                        
                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                          {hypothesis.metrics.patterns.map((pattern) => (
                            <Chip
                              key={pattern}
                              label={pattern.replace('-', ' ')}
                              size="small"
                              variant="outlined"
                              color={getPatternColor(pattern)}
                            />
                          ))}
                        </Box>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => onSelectHypothesis(hypothesis)}
                      title="View/Edit Hypothesis"
                    >
                      <ViewIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {availableVariables.length < config.minVariables && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Need at least {config.minVariables} variables to generate hypotheses. 
          Currently have {availableVariables.length} variables available.
        </Alert>
      )}
    </Box>
  );
};