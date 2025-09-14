import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Button,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  TestParameters,
  HypothesisTestResult,
  StatisticalTest,
  MultipleTestingCorrection,
  BootstrapMethod,
  CrossValidationMethod
} from '../../types/causal-analysis';

interface StatisticalTestPanelProps {
  hypothesis: CausalHypothesis | null;
  testParameters: TestParameters;
  onParameterChange: (paramName: keyof TestParameters, value: any) => void;
  testResult?: HypothesisTestResult;
  isTestingInProgress: boolean;
}

export const StatisticalTestPanel: React.FC<StatisticalTestPanelProps> = ({
  hypothesis,
  testParameters,
  onParameterChange,
  testResult,
  isTestingInProgress
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['basic']));
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const toggleSection = useCallback((section: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  }, []);

  const handleParameterChange = useCallback((param: keyof TestParameters, value: any) => {
    onParameterChange(param, value);
  }, [onParameterChange]);

  // Validate parameters and provide warnings
  const parameterWarnings = useMemo(() => {
    const warnings: string[] = [];
    
    if (testParameters.bootstrapSamples < 100) {
      warnings.push('Bootstrap samples too low for reliable confidence intervals');
    }
    if (testParameters.crossValidationFolds < 3) {
      warnings.push('Cross-validation folds too low for robust validation');
    }
    if (testParameters.significanceLevel > 0.1) {
      warnings.push('Significance level is quite high, consider lowering for stricter testing');
    }
    if (testParameters.temporalLagRange[1] - testParameters.temporalLagRange[0] < 7) {
      warnings.push('Temporal lag range might be too narrow for geological processes');
    }
    
    return warnings;
  }, [testParameters]);

  const estimatedTestTime = useMemo(() => {
    if (!hypothesis) return 0;
    
    const baseTime = hypothesis.edges.length * 0.5; // 0.5s per edge
    const bootstrapTime = testParameters.robustnessTests ? testParameters.bootstrapSamples * 0.001 : 0;
    const cvTime = testParameters.crossValidationFolds * 0.2;
    
    return Math.max(1, baseTime + bootstrapTime + cvTime);
  }, [hypothesis, testParameters]);

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AssessmentIcon />
        Statistical Testing Configuration
      </Typography>

      {!hypothesis && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Build a causal hypothesis first to configure statistical tests.
        </Alert>
      )}

      {parameterWarnings.length > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Parameter Warnings:</Typography>
          {parameterWarnings.map((warning, index) => (
            <Typography key={index} variant="body2">• {warning}</Typography>
          ))}
        </Alert>
      )}

      {/* Basic Parameters */}
      <Accordion 
        expanded={expandedSections.has('basic')}
        onChange={() => toggleSection('basic')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon />
            Basic Test Parameters
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Typography gutterBottom>
                  Significance Level (α): {testParameters.significanceLevel}
                </Typography>
                <Tooltip title="Probability of Type I error (false positive)">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={testParameters.significanceLevel}
                onChange={(_, value) => handleParameterChange('significanceLevel', value)}
                min={0.001}
                max={0.1}
                step={0.001}
                marks={[
                  { value: 0.001, label: '0.001' },
                  { value: 0.01, label: '0.01' },
                  { value: 0.05, label: '0.05' },
                  { value: 0.1, label: '0.1' }
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Typography gutterBottom>
                  Confidence Interval: {(testParameters.confidenceInterval * 100).toFixed(1)}%
                </Typography>
                <Tooltip title="Confidence level for effect size estimation">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={testParameters.confidenceInterval}
                onChange={(_, value) => handleParameterChange('confidenceInterval', value)}
                min={0.8}
                max={0.999}
                step={0.001}
                marks={[
                  { value: 0.9, label: '90%' },
                  { value: 0.95, label: '95%' },
                  { value: 0.99, label: '99%' }
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Multiple Testing Correction</InputLabel>
                <Select
                  value={testParameters.multipleTestingCorrection}
                  onChange={(e) => handleParameterChange('multipleTestingCorrection', e.target.value)}
                >
                  <MenuItem value="none">None</MenuItem>
                  <MenuItem value="bonferroni">Bonferroni</MenuItem>
                  <MenuItem value="holm">Holm-Bonferroni</MenuItem>
                  <MenuItem value="benjamini-hochberg">Benjamini-Hochberg (FDR)</MenuItem>
                  <MenuItem value="benjamini-yekutieli">Benjamini-Yekutieli</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Temporal Analysis */}
      <Accordion 
        expanded={expandedSections.has('temporal')}
        onChange={() => toggleSection('temporal')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TimelineIcon />
            Temporal Analysis Parameters
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography gutterBottom>
                Temporal Lag Range: {testParameters.temporalLagRange[0]} - {testParameters.temporalLagRange[1]} days
              </Typography>
              <Slider
                value={testParameters.temporalLagRange}
                onChange={(_, value) => handleParameterChange('temporalLagRange', value)}
                min={0}
                max={3650} // 10 years
                step={1}
                marks={[
                  { value: 0, label: '0' },
                  { value: 30, label: '1 month' },
                  { value: 365, label: '1 year' },
                  { value: 3650, label: '10 years' }
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>
                Spatial Scale Range: {testParameters.spatialScaleRange[0]} - {testParameters.spatialScaleRange[1]} km
              </Typography>
              <Slider
                value={testParameters.spatialScaleRange}
                onChange={(_, value) => handleParameterChange('spatialScaleRange', value)}
                min={0.001}
                max={10000}
                step={0.1}
                scale={(x) => Math.log10(x)}
                marks={[
                  { value: 0.001, label: '1m' },
                  { value: 1, label: '1km' },
                  { value: 100, label: '100km' },
                  { value: 10000, label: '10,000km' }
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Robustness Testing */}
      <Accordion 
        expanded={expandedSections.has('robustness')}
        onChange={() => toggleSection('robustness')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityIcon />
            Robustness & Validation
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={testParameters.robustnessTests}
                    onChange={(e) => handleParameterChange('robustnessTests', e.target.checked)}
                  />
                }
                label="Enable Robustness Testing"
              />
            </Grid>

            {testParameters.robustnessTests && (
              <>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Bootstrap Samples"
                    type="number"
                    value={testParameters.bootstrapSamples}
                    onChange={(e) => handleParameterChange('bootstrapSamples', parseInt(e.target.value) || 1000)}
                    inputProps={{ min: 100, max: 10000, step: 100 }}
                    helperText="More samples = better confidence intervals but slower computation"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Cross-Validation Folds"
                    type="number"
                    value={testParameters.crossValidationFolds}
                    onChange={(e) => handleParameterChange('crossValidationFolds', parseInt(e.target.value) || 5)}
                    inputProps={{ min: 3, max: 20, step: 1 }}
                    helperText="Number of folds for cross-validation"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Bootstrap Method</InputLabel>
                    <Select
                      value={testParameters.bootstrapMethod || 'percentile'}
                      onChange={(e) => handleParameterChange('bootstrapMethod', e.target.value)}
                    >
                      <MenuItem value="percentile">Percentile</MenuItem>
                      <MenuItem value="bias-corrected">Bias-Corrected</MenuItem>
                      <MenuItem value="accelerated">Bias-Corrected and Accelerated (BCa)</MenuItem>
                      <MenuItem value="studentized">Studentized</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Cross-Validation Method</InputLabel>
                    <Select
                      value={testParameters.crossValidationMethod || 'k-fold'}
                      onChange={(e) => handleParameterChange('crossValidationMethod', e.target.value)}
                    >
                      <MenuItem value="k-fold">K-Fold</MenuItem>
                      <MenuItem value="stratified">Stratified K-Fold</MenuItem>
                      <MenuItem value="time-series">Time Series Split</MenuItem>
                      <MenuItem value="leave-one-out">Leave-One-Out</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </>
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Advanced Options */}
      <Accordion 
        expanded={expandedSections.has('advanced')}
        onChange={() => toggleSection('advanced')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUpIcon />
            Advanced Statistical Options
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={testParameters.bayesianAnalysis || false}
                    onChange={(e) => handleParameterChange('bayesianAnalysis', e.target.checked)}
                  />
                }
                label="Bayesian Analysis"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={testParameters.nonParametricTests || false}
                    onChange={(e) => handleParameterChange('nonParametricTests', e.target.checked)}
                  />
                }
                label="Non-parametric Tests"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={testParameters.permutationTests || false}
                    onChange={(e) => handleParameterChange('permutationTests', e.target.checked)}
                  />
                }
                label="Permutation Tests"
              />
            </Grid>

            {testParameters.permutationTests && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Permutation Iterations"
                  type="number"
                  value={testParameters.permutationIterations || 10000}
                  onChange={(e) => handleParameterChange('permutationIterations', parseInt(e.target.value) || 10000)}
                  inputProps={{ min: 1000, max: 100000, step: 1000 }}
                />
              </Grid>
            )}

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Random Seed"
                type="number"
                value={testParameters.randomSeed || 42}
                onChange={(e) => handleParameterChange('randomSeed', parseInt(e.target.value) || 42)}
                helperText="For reproducible results"
              />
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Minimum Effect Size"
                type="number"
                value={testParameters.minimumEffectSize || 0.1}
                onChange={(e) => handleParameterChange('minimumEffectSize', parseFloat(e.target.value) || 0.1)}
                inputProps={{ min: 0, max: 2, step: 0.01 }}
                helperText="Minimum meaningful effect size (Cohen's d)"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Test Summary */}
      {hypothesis && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Test Configuration Summary
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Hypothesis Edges: {hypothesis.edges.length}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Estimated Time: {estimatedTestTime.toFixed(1)}s
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Significance Level: {testParameters.significanceLevel}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Bootstrap Samples: {testParameters.robustnessTests ? testParameters.bootstrapSamples : 'Disabled'}
                </Typography>
              </Grid>
            </Grid>

            {isTestingInProgress && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Testing in progress...
                </Typography>
                <LinearProgress />
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Current Test Result Summary */}
      {testResult && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Latest Test Result
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip 
                label={`p = ${testResult.pValue.toFixed(4)}`}
                color={testResult.pValue < testParameters.significanceLevel ? 'success' : 'error'}
                size="small"
              />
              <Chip 
                label={`d = ${testResult.effectSize.toFixed(3)}`}
                color="primary"
                size="small"
              />
              <Chip 
                label={`CI: [${testResult.confidenceInterval.lower.toFixed(3)}, ${testResult.confidenceInterval.upper.toFixed(3)}]`}
                color="secondary"
                size="small"
              />
              {testResult.robustnessTests && (
                <Chip 
                  label={`Stability: ${(testResult.robustnessTests.bootstrapStability * 100).toFixed(1)}%`}
                  color="info"
                  size="small"
                />
              )}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};