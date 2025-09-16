import React from 'react';
import {
  Box,
  Typography,
  Slider,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Switch,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  CausalHypothesis,
  TestParameters,
  HypothesisTestResult,
  MultipleTestingCorrection,
  BootstrapMethod,
  CrossValidationMethod
} from '../../types/causal-analysis';

interface StatisticalTestPanelProps {
  hypothesis: CausalHypothesis | null;
  testParameters: TestParameters;
  onParameterChange: (paramName: keyof TestParameters, value: any) => void;
  testResult: HypothesisTestResult | null;
  isTestingInProgress: boolean;
}

export const StatisticalTestPanel: React.FC<StatisticalTestPanelProps> = ({
  hypothesis,
  testParameters,
  onParameterChange,
  testResult,
  isTestingInProgress
}) => {
  if (!hypothesis) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Create a hypothesis with variables and relationships to configure statistical tests.
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Statistical Test Configuration
      </Typography>

      {isTestingInProgress && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Running statistical tests...
          </Typography>
          <LinearProgress />
        </Box>
      )}

      <Grid container spacing={2}>
        {/* Basic Parameters */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Basic Parameters
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography gutterBottom>
                    Significance Level: {testParameters.significanceLevel}
                  </Typography>
                  <Slider
                    value={testParameters.significanceLevel}
                    onChange={(_e: Event, value: number | number[]) => onParameterChange('significanceLevel', value)}
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
                
                <Grid item xs={6}>
                  <Typography gutterBottom>
                    Confidence Interval: {(testParameters.confidenceInterval * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={testParameters.confidenceInterval}
                    onChange={(_e: Event, value: number | number[]) => onParameterChange('confidenceInterval', value)}
                    min={0.8}
                    max={0.99}
                    step={0.01}
                    marks={[
                      { value: 0.8, label: '80%' },
                      { value: 0.9, label: '90%' },
                      { value: 0.95, label: '95%' },
                      { value: 0.99, label: '99%' }
                    ]}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Bootstrap Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Bootstrap Analysis
              </Typography>
              
              <TextField
                fullWidth
                type="number"
                label="Bootstrap Samples"
                value={testParameters.bootstrapSamples}
                onChange={(e) => onParameterChange('bootstrapSamples', parseInt(e.target.value) || 1000)}
                sx={{ mb: 2 }}
                inputProps={{ min: 100, max: 10000, step: 100 }}
              />
              
              <FormControl fullWidth>
                <InputLabel>Bootstrap Method</InputLabel>
                <Select
                  value={testParameters.bootstrapMethod || 'percentile'}
                  onChange={(e) => onParameterChange('bootstrapMethod', e.target.value as BootstrapMethod)}
                  label="Bootstrap Method"
                >
                  <MenuItem value="percentile">Percentile</MenuItem>
                  <MenuItem value="bias-corrected">Bias-Corrected</MenuItem>
                  <MenuItem value="accelerated">Accelerated</MenuItem>
                  <MenuItem value="studentized">Studentized</MenuItem>
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        {/* Cross-Validation Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Cross-Validation
              </Typography>
              
              <TextField
                fullWidth
                type="number"
                label="CV Folds"
                value={testParameters.crossValidationFolds}
                onChange={(e) => onParameterChange('crossValidationFolds', parseInt(e.target.value) || 5)}
                sx={{ mb: 2 }}
                inputProps={{ min: 2, max: 20 }}
              />
              
              <FormControl fullWidth>
                <InputLabel>CV Method</InputLabel>
                <Select
                  value={testParameters.crossValidationMethod || 'k-fold'}
                  onChange={(e) => onParameterChange('crossValidationMethod', e.target.value as CrossValidationMethod)}
                  label="CV Method"
                >
                  <MenuItem value="k-fold">K-Fold</MenuItem>
                  <MenuItem value="stratified">Stratified</MenuItem>
                  <MenuItem value="time-series">Time-Series</MenuItem>
                  <MenuItem value="leave-one-out">Leave-One-Out</MenuItem>
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        {/* Temporal and Spatial Constraints */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Temporal and Spatial Constraints
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography gutterBottom>
                    Temporal Lag Range: {testParameters.temporalLagRange[0]} - {testParameters.temporalLagRange[1]} days
                  </Typography>
                  <Slider
                    value={testParameters.temporalLagRange}
                    onChange={(_e: Event, value: number | number[]) => onParameterChange('temporalLagRange', value)}
                    min={0}
                    max={365}
                    step={1}
                    marks={[
                      { value: 0, label: '0' },
                      { value: 30, label: '30d' },
                      { value: 90, label: '90d' },
                      { value: 365, label: '1y' }
                    ]}
                  />
                </Grid>
                
                <Grid item xs={6}>
                  <Typography gutterBottom>
                    Spatial Scale Range: {testParameters.spatialScaleRange[0]} - {testParameters.spatialScaleRange[1]} km
                  </Typography>
                  <Slider
                    value={testParameters.spatialScaleRange}
                    onChange={(_e: Event, value: number | number[]) => onParameterChange('spatialScaleRange', value)}
                    min={1}
                    max={1000}
                    step={1}
                    marks={[
                      { value: 1, label: '1km' },
                      { value: 10, label: '10km' },
                      { value: 100, label: '100km' },
                      { value: 1000, label: '1000km' }
                    ]}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Advanced Options */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Advanced Options
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Multiple Testing Correction</InputLabel>
                    <Select
                      value={testParameters.multipleTestingCorrection}
                      onChange={(e) => onParameterChange('multipleTestingCorrection', e.target.value as MultipleTestingCorrection)}
                      label="Multiple Testing Correction"
                    >
                      <MenuItem value="none">None</MenuItem>
                      <MenuItem value="bonferroni">Bonferroni</MenuItem>
                      <MenuItem value="holm">Holm</MenuItem>
                      <MenuItem value="benjamini-hochberg">Benjamini-Hochberg</MenuItem>
                      <MenuItem value="benjamini-yekutieli">Benjamini-Yekutieli</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Minimum Effect Size"
                    value={testParameters.minimumEffectSize || 0.1}
                    onChange={(e) => onParameterChange('minimumEffectSize', parseFloat(e.target.value) || 0.1)}
                    sx={{ mb: 2 }}
                    inputProps={{ min: 0, max: 2, step: 0.01 }}
                  />
                </Grid>
              </Grid>
              
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={testParameters.robustnessTests}
                        onChange={(e) => onParameterChange('robustnessTests', e.target.checked)}
                      />
                    }
                    label="Robustness Tests"
                  />
                </Grid>
                
                <Grid item xs={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={testParameters.bayesianAnalysis || false}
                        onChange={(e) => onParameterChange('bayesianAnalysis', e.target.checked)}
                      />
                    }
                    label="Bayesian Analysis"
                  />
                </Grid>
                
                <Grid item xs={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={testParameters.nonParametricTests || false}
                        onChange={(e) => onParameterChange('nonParametricTests', e.target.checked)}
                      />
                    }
                    label="Non-parametric Tests"
                  />
                </Grid>
                
                <Grid item xs={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={testParameters.permutationTests || false}
                        onChange={(e) => onParameterChange('permutationTests', e.target.checked)}
                      />
                    }
                    label="Permutation Tests"
                  />
                </Grid>
              </Grid>
              
              {testParameters.permutationTests && (
                <TextField
                  fullWidth
                  type="number"
                  label="Permutation Iterations"
                  value={testParameters.permutationIterations || 1000}
                  onChange={(e) => onParameterChange('permutationIterations', parseInt(e.target.value) || 1000)}
                  sx={{ mt: 2 }}
                  inputProps={{ min: 100, max: 10000, step: 100 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Test Results Summary */}
        {testResult && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Latest Test Results
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color={testResult.pValue < testParameters.significanceLevel ? 'success.main' : 'error.main'}>
                        {testResult.pValue.toFixed(4)}
                      </Typography>
                      <Typography variant="caption">p-value</Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        {testResult.effectSize.toFixed(3)}
                      </Typography>
                      <Typography variant="caption">Effect Size</Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6">
                        {testResult.testStatistic.toFixed(3)}
                      </Typography>
                      <Typography variant="caption">Test Statistic</Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6">
                        {testResult.degreesOfFreedom}
                      </Typography>
                      <Typography variant="caption">Degrees of Freedom</Typography>
                    </Box>
                  </Grid>
                </Grid>
                
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={testResult.pValue < testParameters.significanceLevel ? 'Statistically Significant' : 'Not Significant'}
                    color={testResult.pValue < testParameters.significanceLevel ? 'success' : 'error'}
                    sx={{ mr: 1 }}
                  />
                  
                  <Chip
                    label={`CI: [${testResult.confidenceInterval.lower.toFixed(3)}, ${testResult.confidenceInterval.upper.toFixed(3)}]`}
                    variant="outlined"
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};