import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Card,
  CardContent,
  CardActions,
  Grid,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Switch,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction
} from '@mui/material';
import {
  AutoAwesome as AutoAwesomeIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  FilterList as FilterListIcon,
  ExpandMore as ExpandMoreIcon,
  Science as ScienceIcon,
  Lightbulb as LightbulbIcon,
  Assessment as AssessmentIcon,
  Info as InfoIcon,
  Star as StarIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  CausalVariable,
  GeologicalProcess,
  GenerationAlgorithm,
  HypothesisGenerationConfig,
  GeneratedHypothesisMetrics,
  PatternType,
  ConstraintType
} from '../../types/causal-analysis';

interface AutoHypothesisGeneratorProps {
  availableVariables: CausalVariable[];
  geologicalProcesses: GeologicalProcess[];
  onHypothesesGenerated: (hypotheses: CausalHypothesis[]) => void;
  onSelectHypothesis: (hypothesis: CausalHypothesis) => void;
  generatedHypotheses: CausalHypothesis[];
}

interface GenerationProgress {
  stage: string;
  progress: number;
  currentAlgorithm: string;
  hypothesesFound: number;
}

export const AutoHypothesisGenerator: React.FC<AutoHypothesisGeneratorProps> = ({
  availableVariables,
  geologicalProcesses,
  onHypothesesGenerated,
  onSelectHypothesis,
  generatedHypotheses
}) => {
  const [generationConfig, setGenerationConfig] = useState<HypothesisGenerationConfig>({
    algorithms: ['correlation-based', 'mutual-information', 'granger-causality', 'pc-algorithm'],
    maxHypotheses: 20,
    minVariables: 3,
    maxVariables: 8,
    minEdges: 2,
    maxEdges: 12,
    significanceThreshold: 0.05,
    effectSizeThreshold: 0.2,
    temporalConstraints: true,
    spatialConstraints: true,
    geologicalConstraints: true,
    domainKnowledge: true,
    noveltyWeight: 0.3,
    plausibilityWeight: 0.4,
    simplicityWeight: 0.3
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState<GenerationProgress | null>(null);
  const [selectedPatterns, setSelectedPatterns] = useState<Set<PatternType>>(new Set(['volcanic-unrest', 'seismic-precursors', 'deformation-patterns']));
  const [filterCriteria, setFilterCriteria] = useState({
    minScore: 0.5,
    maxComplexity: 0.8,
    requireNovelty: false,
    geologicalContext: 'all'
  });
  const [showConfigDialog, setShowConfigDialog] = useState(false);
  const [sortBy, setSortBy] = useState<'score' | 'novelty' | 'plausibility' | 'simplicity'>('score');

  // Enhanced hypothesis generation using multiple algorithms
  const generateHypotheses = useCallback(async () => {
    if (availableVariables.length < generationConfig.minVariables) {
      alert(`Need at least ${generationConfig.minVariables} variables to generate hypotheses`);
      return;
    }

    setIsGenerating(true);
    setGenerationProgress({
      stage: 'Initializing',
      progress: 0,
      currentAlgorithm: '',
      hypothesesFound: 0
    });

    try {
      const newHypotheses: CausalHypothesis[] = [];
      const totalSteps = generationConfig.algorithms.length * 4; // 4 steps per algorithm
      let currentStep = 0;

      for (const algorithm of generationConfig.algorithms) {
        setGenerationProgress(prev => prev ? {
          ...prev,
          stage: `Running ${algorithm}`,
          currentAlgorithm: algorithm,
          progress: (currentStep / totalSteps) * 100
        } : null);

        // Simulate algorithm execution with realistic delays
        await new Promise(resolve => setTimeout(resolve, 1000));

        const algorithmHypotheses = await runGenerationAlgorithm(
          algorithm as GenerationAlgorithm,
          availableVariables,
          geologicalProcesses,
          generationConfig,
          selectedPatterns
        );

        newHypotheses.push(...algorithmHypotheses);
        currentStep += 4;

        setGenerationProgress(prev => prev ? {
          ...prev,
          hypothesesFound: newHypotheses.length,
          progress: (currentStep / totalSteps) * 100
        } : null);
      }

      // Post-processing: ranking, filtering, and deduplication
      setGenerationProgress(prev => prev ? {
        ...prev,
        stage: 'Post-processing and ranking',
        progress: 90
      } : null);

      await new Promise(resolve => setTimeout(resolve, 500));

      const processedHypotheses = await postProcessHypotheses(
        newHypotheses,
        generationConfig,
        filterCriteria
      );

      setGenerationProgress(prev => prev ? {
        ...prev,
        stage: 'Complete',
        progress: 100
      } : null);

      onHypothesesGenerated(processedHypotheses);

    } catch (error) {
      console.error('Hypothesis generation failed:', error);
      alert('Hypothesis generation failed. Please try again.');
    } finally {
      setIsGenerating(false);
      setTimeout(() => setGenerationProgress(null), 2000);
    }
  }, [availableVariables, geologicalProcesses, generationConfig, selectedPatterns, filterCriteria, onHypothesesGenerated]);

  // Filter and sort generated hypotheses
  const filteredAndSortedHypotheses = useMemo(() => {
    let filtered = generatedHypotheses.filter(h => {
      const metrics = h.metadata.generationMetrics;
      if (!metrics) return true;

      return (
        metrics.overallScore >= filterCriteria.minScore &&
        metrics.complexity <= filterCriteria.maxComplexity &&
        (!filterCriteria.requireNovelty || metrics.noveltyScore > 0.5) &&
        (filterCriteria.geologicalContext === 'all' || 
         h.variables.some(v => v.geologicalContext === filterCriteria.geologicalContext))
      );
    });

    return filtered.sort((a, b) => {
      const aMetrics = a.metadata.generationMetrics;
      const bMetrics = b.metadata.generationMetrics;
      
      if (!aMetrics || !bMetrics) return 0;

      switch (sortBy) {
        case 'score':
          return bMetrics.overallScore - aMetrics.overallScore;
        case 'novelty':
          return bMetrics.noveltyScore - aMetrics.noveltyScore;
        case 'plausibility':
          return bMetrics.plausibilityScore - aMetrics.plausibilityScore;
        case 'simplicity':
          return aMetrics.complexity - bMetrics.complexity; // Lower complexity = higher simplicity
        default:
          return 0;
      }
    });
  }, [generatedHypotheses, filterCriteria, sortBy]);

  const handleConfigChange = useCallback((field: keyof HypothesisGenerationConfig, value: any) => {
    setGenerationConfig(prev => ({
      ...prev,
      [field]: value
    }));
  }, []);

  const handlePatternToggle = useCallback((pattern: PatternType) => {
    setSelectedPatterns(prev => {
      const newSet = new Set(prev);
      if (newSet.has(pattern)) {
        newSet.delete(pattern);
      } else {
        newSet.add(pattern);
      }
      return newSet;
    });
  }, []);

  const handleExportHypotheses = useCallback(() => {
    const exportData = {
      timestamp: new Date().toISOString(),
      generationConfig,
      selectedPatterns: Array.from(selectedPatterns),
      hypotheses: filteredAndSortedHypotheses.map(h => ({
        name: h.name,
        description: h.description,
        variables: h.variables.map(v => ({ name: v.name, type: v.type })),
        edges: h.edges.map(e => ({
          source: h.variables.find(v => v.id === e.sourceId)?.name,
          target: h.variables.find(v => v.id === e.targetId)?.name,
          strength: e.strength,
          type: e.relationshipType
        })),
        metrics: h.metadata.generationMetrics
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `auto-generated-hypotheses-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filteredAndSortedHypotheses, generationConfig, selectedPatterns]);

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AutoAwesomeIcon />
          Auto-Generate Hypotheses
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setShowConfigDialog(true)}
            size="small"
          >
            Configure
          </Button>
          <Button
            variant="contained"
            startIcon={<PsychologyIcon />}
            onClick={generateHypotheses}
            disabled={isGenerating || availableVariables.length < 3}
          >
            Generate
          </Button>
        </Box>
      </Box>

      {/* Generation Progress */}
      {generationProgress && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              {generationProgress.stage}
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={generationProgress.progress} 
              sx={{ mb: 1 }}
            />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                {generationProgress.currentAlgorithm && `Algorithm: ${generationProgress.currentAlgorithm}`}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Found: {generationProgress.hypothesesFound} hypotheses
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Quick Configuration */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">Quick Configuration</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Max Hypotheses: {generationConfig.maxHypotheses}
              </Typography>
              <Slider
                value={generationConfig.maxHypotheses}
                onChange={(_, value) => handleConfigChange('maxHypotheses', value)}
                min={5}
                max={50}
                step={5}
                marks={[
                  { value: 10, label: '10' },
                  { value: 25, label: '25' },
                  { value: 50, label: '50' }
                ]}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Significance Threshold: {generationConfig.significanceThreshold}
              </Typography>
              <Slider
                value={generationConfig.significanceThreshold}
                onChange={(_, value) => handleConfigChange('significanceThreshold', value)}
                min={0.001}
                max={0.1}
                step={0.001}
                marks={[
                  { value: 0.01, label: '0.01' },
                  { value: 0.05, label: '0.05' },
                  { value: 0.1, label: '0.1' }
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Geological Patterns to Explore
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {[
                  'volcanic-unrest',
                  'seismic-precursors',
                  'deformation-patterns',
                  'gas-emissions',
                  'thermal-anomalies',
                  'magma-intrusion',
                  'hydrothermal-activity',
                  'tectonic-stress'
                ].map(pattern => (
                  <Chip
                    key={pattern}
                    label={pattern.replace('-', ' ')}
                    onClick={() => handlePatternToggle(pattern as PatternType)}
                    color={selectedPatterns.has(pattern as PatternType) ? 'primary' : 'default'}
                    variant={selectedPatterns.has(pattern as PatternType) ? 'filled' : 'outlined'}
                    size="small"
                  />
                ))}
              </Box>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Results Section */}
      {generatedHypotheses.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Generated Hypotheses ({filteredAndSortedHypotheses.length})
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 1 }}>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                >
                  <MenuItem value="score">Overall Score</MenuItem>
                  <MenuItem value="novelty">Novelty</MenuItem>
                  <MenuItem value="plausibility">Plausibility</MenuItem>
                  <MenuItem value="simplicity">Simplicity</MenuItem>
                </Select>
              </FormControl>
              
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleExportHypotheses}
                size="small"
              >
                Export
              </Button>
            </Box>
          </Box>

          {/* Filters */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <FilterListIcon />
                Filters
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography gutterBottom>
                    Min Score: {filterCriteria.minScore.toFixed(2)}
                  </Typography>
                  <Slider
                    value={filterCriteria.minScore}
                    onChange={(_, value) => setFilterCriteria(prev => ({ ...prev, minScore: value as number }))}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Typography gutterBottom>
                    Max Complexity: {filterCriteria.maxComplexity.toFixed(2)}
                  </Typography>
                  <Slider
                    value={filterCriteria.maxComplexity}
                    onChange={(_, value) => setFilterCriteria(prev => ({ ...prev, maxComplexity: value as number }))}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Grid>

                <Grid item xs={12} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={filterCriteria.requireNovelty}
                        onChange={(e) => setFilterCriteria(prev => ({ ...prev, requireNovelty: e.target.checked }))}
                      />
                    }
                    label="Require Novelty"
                  />
                </Grid>

                <Grid item xs={12} md={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Geological Context</InputLabel>
                    <Select
                      value={filterCriteria.geologicalContext}
                      onChange={(e) => setFilterCriteria(prev => ({ ...prev, geologicalContext: e.target.value }))}
                    >
                      <MenuItem value="all">All</MenuItem>
                      <MenuItem value="volcanic">Volcanic</MenuItem>
                      <MenuItem value="seismic">Seismic</MenuItem>
                      <MenuItem value="tectonic">Tectonic</MenuItem>
                      <MenuItem value="hydrothermal">Hydrothermal</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Hypothesis Cards */}
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {filteredAndSortedHypotheses.map((hypothesis, index) => (
              <Grid item xs={12} md={6} key={hypothesis.id}>
                <GeneratedHypothesisCard
                  hypothesis={hypothesis}
                  rank={index + 1}
                  onSelect={() => onSelectHypothesis(hypothesis)}
                />
              </Grid>
            ))}
          </Grid>

          {filteredAndSortedHypotheses.length === 0 && (
            <Alert severity="info" sx={{ mt: 2 }}>
              No hypotheses match the current filter criteria. Try adjusting the filters above.
            </Alert>
          )}
        </Box>
      )}

      {generatedHypotheses.length === 0 && !isGenerating && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Click "Generate" to automatically discover causal hypotheses based on available data patterns and geological domain knowledge.
        </Alert>
      )}

      {/* Configuration Dialog */}
      <Dialog open={showConfigDialog} onClose={() => setShowConfigDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Advanced Generation Configuration</DialogTitle>
        <DialogContent>
          <AdvancedConfigurationPanel
            config={generationConfig}
            onConfigChange={handleConfigChange}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowConfigDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Generated Hypothesis Card Component
const GeneratedHypothesisCard: React.FC<{
  hypothesis: CausalHypothesis;
  rank: number;
  onSelect: () => void;
}> = ({ hypothesis, rank, onSelect }) => {
  const metrics = hypothesis.metadata.generationMetrics;

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flex: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6" component="div">
            #{rank} {hypothesis.name}
          </Typography>
          {metrics && (
            <Chip
              label={`Score: ${metrics.overallScore.toFixed(2)}`}
              color={metrics.overallScore > 0.8 ? 'success' : 
                     metrics.overallScore > 0.6 ? 'warning' : 'default'}
              size="small"
            />
          )}
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {hypothesis.description}
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip label={`${hypothesis.variables.length} variables`} size="small" />
          <Chip label={`${hypothesis.edges.length} edges`} size="small" />
          {metrics?.algorithm && (
            <Chip label={metrics.algorithm} size="small" variant="outlined" />
          )}
        </Box>

        {metrics && (
          <Grid container spacing={1}>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Novelty
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.noveltyScore * 100}
                sx={{ height: 4 }}
              />
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Plausibility
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.plausibilityScore * 100}
                sx={{ height: 4 }}
              />
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Simplicity
              </Typography>
              <LinearProgress
                variant="determinate"
                value={(1 - metrics.complexity) * 100}
                sx={{ height: 4 }}
              />
            </Grid>
          </Grid>
        )}
      </CardContent>

      <CardActions>
        <Button size="small" onClick={onSelect} variant="contained">
          Select Hypothesis
        </Button>
        <Button size="small" startIcon={<InfoIcon />}>
          Details
        </Button>
      </CardActions>
    </Card>
  );
};

// Advanced Configuration Panel Component
const AdvancedConfigurationPanel: React.FC<{
  config: HypothesisGenerationConfig;
  onConfigChange: (field: keyof HypothesisGenerationConfig, value: any) => void;
}> = ({ config, onConfigChange }) => {
  return (
    <Box sx={{ pt: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Generation Algorithms
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {[
              'correlation-based',
              'mutual-information',
              'granger-causality',
              'pc-algorithm',
              'ges-algorithm',
              'lingam',
              'causal-forests'
            ].map(algorithm => (
              <Chip
                key={algorithm}
                label={algorithm}
                onClick={() => {
                  const newAlgorithms = config.algorithms.includes(algorithm as GenerationAlgorithm)
                    ? config.algorithms.filter(a => a !== algorithm)
                    : [...config.algorithms, algorithm as GenerationAlgorithm];
                  onConfigChange('algorithms', newAlgorithms);
                }}
                color={config.algorithms.includes(algorithm as GenerationAlgorithm) ? 'primary' : 'default'}
                variant={config.algorithms.includes(algorithm as GenerationAlgorithm) ? 'filled' : 'outlined'}
              />
            ))}
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography gutterBottom>
            Variable Range: {config.minVariables} - {config.maxVariables}
          </Typography>
          <Slider
            value={[config.minVariables, config.maxVariables]}
            onChange={(_, value) => {
              const [min, max] = value as number[];
              onConfigChange('minVariables', min);
              onConfigChange('maxVariables', max);
            }}
            min={2}
            max={15}
            step={1}
            valueLabelDisplay="auto"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography gutterBottom>
            Edge Range: {config.minEdges} - {config.maxEdges}
          </Typography>
          <Slider
            value={[config.minEdges, config.maxEdges]}
            onChange={(_, value) => {
              const [min, max] = value as number[];
              onConfigChange('minEdges', min);
              onConfigChange('maxEdges', max);
            }}
            min={1}
            max={20}
            step={1}
            valueLabelDisplay="auto"
          />
        </Grid>

        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Scoring Weights
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Typography gutterBottom>
                Novelty: {config.noveltyWeight.toFixed(2)}
              </Typography>
              <Slider
                value={config.noveltyWeight}
                onChange={(_, value) => onConfigChange('noveltyWeight', value)}
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            <Grid item xs={4}>
              <Typography gutterBottom>
                Plausibility: {config.plausibilityWeight.toFixed(2)}
              </Typography>
              <Slider
                value={config.plausibilityWeight}
                onChange={(_, value) => onConfigChange('plausibilityWeight', value)}
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            <Grid item xs={4}>
              <Typography gutterBottom>
                Simplicity: {config.simplicityWeight.toFixed(2)}
              </Typography>
              <Slider
                value={config.simplicityWeight}
                onChange={(_, value) => onConfigChange('simplicityWeight', value)}
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
          </Grid>
        </Grid>

        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Constraints
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.temporalConstraints}
                    onChange={(e) => onConfigChange('temporalConstraints', e.target.checked)}
                  />
                }
                label="Temporal"
              />
            </Grid>
            <Grid item xs={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.spatialConstraints}
                    onChange={(e) => onConfigChange('spatialConstraints', e.target.checked)}
                  />
                }
                label="Spatial"
              />
            </Grid>
            <Grid item xs={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.geologicalConstraints}
                    onChange={(e) => onConfigChange('geologicalConstraints', e.target.checked)}
                  />
                }
                label="Geological"
              />
            </Grid>
            <Grid item xs={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.domainKnowledge}
                    onChange={(e) => onConfigChange('domainKnowledge', e.target.checked)}
                  />
                }
                label="Domain Knowledge"
              />
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

// Algorithm implementations (simplified for demo)
async function runGenerationAlgorithm(
  algorithm: GenerationAlgorithm,
  variables: CausalVariable[],
  processes: GeologicalProcess[],
  config: HypothesisGenerationConfig,
  patterns: Set<PatternType>
): Promise<CausalHypothesis[]> {
  // Simulate algorithm execution
  await new Promise(resolve => setTimeout(resolve, 800));

  const hypotheses: CausalHypothesis[] = [];
  const numHypotheses = Math.min(config.maxHypotheses / config.algorithms.length, 10);

  for (let i = 0; i < numHypotheses; i++) {
    const numVars = Math.floor(Math.random() * (config.maxVariables - config.minVariables + 1)) + config.minVariables;
    const selectedVars = variables.slice(0, numVars);
    const numEdges = Math.floor(Math.random() * (config.maxEdges - config.minEdges + 1)) + config.minEdges;

    const hypothesis: CausalHypothesis = {
      id: `generated_${algorithm}_${i}_${Date.now()}`,
      name: `${algorithm.toUpperCase()} Hypothesis ${i + 1}`,
      description: `Auto-generated hypothesis using ${algorithm} algorithm focusing on ${Array.from(patterns).join(', ')}`,
      variables: selectedVars,
      edges: generateRandomEdges(selectedVars, numEdges),
      constraints: {
        temporal: [],
        spatial: [],
        geological: []
      },
      metadata: {
        created: new Date(),
        lastModified: new Date(),
        author: 'auto-generator',
        version: 1,
        generationMetrics: {
          algorithm,
          overallScore: Math.random() * 0.4 + 0.6, // 0.6-1.0
          noveltyScore: Math.random(),
          plausibilityScore: Math.random() * 0.3 + 0.7, // 0.7-1.0
          complexity: Math.random() * 0.6 + 0.2, // 0.2-0.8
          patterns: Array.from(patterns),
          confidence: Math.random() * 0.3 + 0.7
        }
      }
    };

    hypotheses.push(hypothesis);
  }

  return hypotheses;
}

function generateRandomEdges(variables: CausalVariable[], numEdges: number) {
  const edges = [];
  const usedPairs = new Set<string>();

  for (let i = 0; i < numEdges && edges.length < numEdges; i++) {
    const sourceIdx = Math.floor(Math.random() * variables.length);
    const targetIdx = Math.floor(Math.random() * variables.length);
    
    if (sourceIdx !== targetIdx) {
      const pairKey = `${sourceIdx}-${targetIdx}`;
      if (!usedPairs.has(pairKey)) {
        usedPairs.add(pairKey);
        
        edges.push({
          id: `edge_${sourceIdx}_${targetIdx}`,
          sourceId: variables[sourceIdx].id,
          targetId: variables[targetIdx].id,
          relationshipType: Math.random() > 0.8 ? 'inhibitory' : 'causal',
          strength: Math.random() * 0.6 + 0.4,
          confidence: Math.random() * 0.3 + 0.7,
          temporalLag: Math.floor(Math.random() * 30),
          spatialScale: ['local', 'regional', 'global'][Math.floor(Math.random() * 3)],
          constraints: [],
          metadata: {
            created: new Date(),
            discoveryMethod: 'auto-generated',
            validationStatus: 'proposed'
          }
        });
      }
    }
  }

  return edges;
}

async function postProcessHypotheses(
  hypotheses: CausalHypothesis[],
  config: HypothesisGenerationConfig,
  filterCriteria: any
): Promise<CausalHypothesis[]> {
  // Remove duplicates, rank by score, and apply filters
  const uniqueHypotheses = removeDuplicateHypotheses(hypotheses);
  const rankedHypotheses = rankHypotheses(uniqueHypotheses, config);
  
  return rankedHypotheses
    .filter(h => h.metadata.generationMetrics!.overallScore >= filterCriteria.minScore)
    .slice(0, config.maxHypotheses);
}

function removeDuplicateHypotheses(hypotheses: CausalHypothesis[]): CausalHypothesis[] {
  const seen = new Set<string>();
  return hypotheses.filter(h => {
    const signature = createHypothesisSignature(h);
    if (seen.has(signature)) {
      return false;
    }
    seen.add(signature);
    return true;
  });
}

function createHypothesisSignature(hypothesis: CausalHypothesis): string {
  const varNames = hypothesis.variables.map(v => v.name).sort().join(',');
  const edgeSignatures = hypothesis.edges
    .map(e => `${e.sourceId}-${e.targetId}`)
    .sort()
    .join(',');
  return `${varNames}|${edgeSignatures}`;
}

function rankHypotheses(hypotheses: CausalHypothesis[], config: HypothesisGenerationConfig): CausalHypothesis[] {
  return hypotheses.sort((a, b) => {
    const aScore = a.metadata.generationMetrics!.overallScore;
    const bScore = b.metadata.generationMetrics!.overallScore;
    return bScore - aScore;
  });
}