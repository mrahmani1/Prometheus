import React, { useState, useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Grid,
  Alert,
  // Accordion, AccordionSummary, AccordionDetails - unused
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  Compare as CompareIcon,
  TrendingUp as TrendingUpIcon,
  // Assessment as AssessmentIcon, ExpandMore as ExpandMoreIcon - unused
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Download as DownloadIcon,
  Share as _ShareIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  HypothesisTestResult,
  HypothesisComparison as _HypothesisComparison,
  ComparisonMetric as _ComparisonMetric,
  ModelSelectionCriterion as _ModelSelectionCriterion
} from '../../types/causal-analysis';
// import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend, ScatterPlot, Scatter } from 'recharts'; // Module not available

interface HypothesisComparatorProps {
  hypotheses: CausalHypothesis[];
  onRemoveHypothesis: (id: string) => void;
  onSelectHypothesis: (hypothesis: CausalHypothesis) => void;
  testResults?: Map<string, HypothesisTestResult>;
}

type SortCriterion = 'pValue' | 'effectSize' | 'complexity' | 'robustness' | 'parsimony' | 'timestamp';
type ComparisonView = 'table' | 'radar' | 'scatter' | 'ranking';

export const HypothesisComparator: React.FC<HypothesisComparatorProps> = ({
  hypotheses,
  onRemoveHypothesis,
  onSelectHypothesis,
  testResults = new Map()
}) => {
  const [sortBy, setSortBy] = useState<SortCriterion>('pValue');
  const [selectedHypotheses, setSelectedHypotheses] = useState<Set<string>>(new Set());
  const [comparisonView, setComparisonView] = useState<ComparisonView>('table');
  const [showDetailDialog, setShowDetailDialog] = useState(false);
  const [detailHypothesis, setDetailHypothesis] = useState<CausalHypothesis | null>(null);
  const [favoriteHypotheses, setFavoriteHypotheses] = useState<Set<string>>(new Set());

  // Enhanced comparison metrics calculation
  const comparisonData = useMemo(() => {
    return hypotheses.map(hypothesis => {
      const testResult = testResults.get(hypothesis.id);
      const complexity = calculateComplexity(hypothesis);
      const parsimony = calculateParsimony(hypothesis);
      const robustness = testResult?.robustnessTests ? 
        (testResult.robustnessTests.bootstrapStability + testResult.robustnessTests.crossValidationScore) / 2 : 0;

      return {
        hypothesis,
        testResult,
        metrics: {
          pValue: testResult?.pValue || 1,
          effectSize: Math.abs(testResult?.effectSize || 0),
          complexity,
          parsimony,
          robustness,
          aic: calculateAIC(hypothesis, testResult),
          bic: calculateBIC(hypothesis, testResult),
          evidenceRatio: testResult ? calculateEvidenceRatio(testResult) : 0,
          powerAnalysis: testResult ? calculateStatisticalPower(testResult) : 0
        }
      };
    });
  }, [hypotheses, testResults]);

  // Sort hypotheses based on selected criterion
  const sortedData = useMemo(() => {
    return [...comparisonData].sort((a, b) => {
      const aValue = (a.metrics as any)[sortBy];
      const bValue = (b.metrics as any)[sortBy];
      
      // For p-values and complexity, lower is better
      if (sortBy === 'pValue' || sortBy === 'complexity') {
        return aValue - bValue;
      }
      // For other metrics, higher is better
      return bValue - aValue;
    });
  }, [comparisonData, sortBy]);

  // Model ranking using multiple criteria
  const rankedHypotheses = useMemo(() => {
    return comparisonData.map(data => {
      const scores = {
        statistical: data.metrics.pValue < 0.05 ? (1 - data.metrics.pValue) * 20 : 0,
        effect: Math.min(data.metrics.effectSize * 10, 10),
        robustness: data.metrics.robustness * 10,
        parsimony: data.metrics.parsimony * 10,
        evidence: data.metrics.evidenceRatio * 2
      };
      
      const totalScore = Object.values(scores).reduce((sum, score) => sum + score, 0);
      
      return {
        ...data,
        scores,
        totalScore,
        rank: 0 // Will be set after sorting
      };
    }).sort((a, b) => b.totalScore - a.totalScore)
      .map((item, index) => ({ ...item, rank: index + 1 }));
  }, [comparisonData]);

  // Radar chart data for selected hypotheses
  // const _radarData = useMemo(() => {
  //   const selectedData = comparisonData.filter(d => selectedHypotheses.has(d.hypothesis.id));
  //   if (selectedData.length === 0) return [];

  //   const metrics = ['Statistical Significance', 'Effect Size', 'Robustness', 'Parsimony', 'Evidence Ratio'];
    
  //   return metrics.map(metric => {
  //     const dataPoint: any = { metric };
  //     selectedData.forEach(d => {
  //       const value = getMetricValue(d.metrics, metric);
  //       dataPoint[d.hypothesis.name] = value;
  //     });
  //     return dataPoint;
  //   });
  // }, [comparisonData, selectedHypotheses]);

  const handleHypothesisSelect = useCallback((hypothesisId: string, selected: boolean) => {
    setSelectedHypotheses(prev => {
      const newSet = new Set(prev);
      if (selected) {
        newSet.add(hypothesisId);
      } else {
        newSet.delete(hypothesisId);
      }
      return newSet;
    });
  }, []);

  const handleToggleFavorite = useCallback((hypothesisId: string) => {
    setFavoriteHypotheses(prev => {
      const newSet = new Set(prev);
      if (newSet.has(hypothesisId)) {
        newSet.delete(hypothesisId);
      } else {
        newSet.add(hypothesisId);
      }
      return newSet;
    });
  }, []);

  const handleShowDetails = useCallback((hypothesis: CausalHypothesis) => {
    setDetailHypothesis(hypothesis);
    setShowDetailDialog(true);
  }, []);

  const handleExportComparison = useCallback(() => {
    const exportData = {
      timestamp: new Date().toISOString(),
      hypotheses: comparisonData.map(d => ({
        name: d.hypothesis.name,
        description: d.hypothesis.description,
        variables: d.hypothesis.variables.length,
        edges: d.hypothesis.edges.length,
        metrics: d.metrics,
        testResult: d.testResult
      })),
      ranking: rankedHypotheses.map(r => ({
        rank: r.rank,
        name: r.hypothesis.name,
        totalScore: r.totalScore,
        scores: r.scores
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hypothesis-comparison-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [comparisonData, rankedHypotheses]);

  if (hypotheses.length === 0) {
    return (
      <Alert severity="info">
        No hypotheses to compare. Add hypotheses to the comparison by clicking "Add to Comparison" in the test results.
      </Alert>
    );
  }

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CompareIcon />
          Hypothesis Comparison ({hypotheses.length})
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>View</InputLabel>
            <Select
              value={comparisonView}
              onChange={(e) => setComparisonView(e.target.value as ComparisonView)}
            >
              <MenuItem value="table">Table</MenuItem>
              <MenuItem value="radar">Radar Chart</MenuItem>
              <MenuItem value="scatter">Scatter Plot</MenuItem>
              <MenuItem value="ranking">Ranking</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortCriterion)}
            >
              <MenuItem value="pValue">P-Value</MenuItem>
              <MenuItem value="effectSize">Effect Size</MenuItem>
              <MenuItem value="complexity">Complexity</MenuItem>
              <MenuItem value="robustness">Robustness</MenuItem>
              <MenuItem value="parsimony">Parsimony</MenuItem>
            </Select>
          </FormControl>

          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExportComparison}
            size="small"
          >
            Export
          </Button>
        </Box>
      </Box>

      {/* Table View */}
      {comparisonView === 'table' && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">Select</TableCell>
                <TableCell>Hypothesis</TableCell>
                <TableCell align="right">Variables</TableCell>
                <TableCell align="right">Edges</TableCell>
                <TableCell align="right">P-Value</TableCell>
                <TableCell align="right">Effect Size</TableCell>
                <TableCell align="right">Robustness</TableCell>
                <TableCell align="right">AIC</TableCell>
                <TableCell align="right">BIC</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedData.map((data) => (
                <TableRow key={data.hypothesis.id} hover>
                  <TableCell padding="checkbox">
                    <input
                      type="checkbox"
                      checked={selectedHypotheses.has(data.hypothesis.id)}
                      onChange={(e) => handleHypothesisSelect(data.hypothesis.id, e.target.checked)}
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <IconButton
                        size="small"
                        onClick={() => handleToggleFavorite(data.hypothesis.id)}
                      >
                        {favoriteHypotheses.has(data.hypothesis.id) ? 
                          <StarIcon color="primary" /> : 
                          <StarBorderIcon />
                        }
                      </IconButton>
                      <Box>
                        <Typography variant="body2" fontWeight="medium">
                          {data.hypothesis.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {data.hypothesis.description?.substring(0, 50)}...
                        </Typography>
                      </Box>
                    </Box>
                  </TableCell>
                  <TableCell align="right">{data.hypothesis.variables.length}</TableCell>
                  <TableCell align="right">{data.hypothesis.edges.length}</TableCell>
                  <TableCell align="right">
                    <Chip
                      label={data.metrics.pValue.toFixed(4)}
                      size="small"
                      color={data.metrics.pValue < 0.05 ? 'success' : 'error'}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Chip
                      label={data.metrics.effectSize.toFixed(3)}
                      size="small"
                      color={data.metrics.effectSize > 0.5 ? 'success' : 
                             data.metrics.effectSize > 0.2 ? 'warning' : 'default'}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={data.metrics.robustness * 100}
                        sx={{ width: 50, height: 4 }}
                      />
                      <Typography variant="caption">
                        {(data.metrics.robustness * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">{data.metrics.aic.toFixed(2)}</TableCell>
                  <TableCell align="right">{data.metrics.bic.toFixed(2)}</TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleShowDetails(data.hypothesis)}
                        >
                          <VisibilityIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Select for Testing">
                        <IconButton
                          size="small"
                          onClick={() => onSelectHypothesis(data.hypothesis)}
                        >
                          <TrendingUpIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Remove from Comparison">
                        <IconButton
                          size="small"
                          onClick={() => onRemoveHypothesis(data.hypothesis.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Radar Chart View */}
      {comparisonView === 'radar' && (
        <Box>
          {selectedHypotheses.size === 0 ? (
            <Alert severity="info">
              Select hypotheses from the table to compare them in the radar chart.
            </Alert>
          ) : (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Multi-Dimensional Comparison
                </Typography>
                <Alert severity="info">
                  Radar chart visualization would be displayed here (recharts module not available)
                </Alert>
              </CardContent>
            </Card>
          )}
        </Box>
      )}

      {/* Ranking View */}
      {comparisonView === 'ranking' && (
        <Grid container spacing={2}>
          {rankedHypotheses.map((rankedHypothesis) => (
            <Grid item xs={12} md={6} key={rankedHypothesis.hypothesis.id}>
              <Card sx={{ 
                border: rankedHypothesis.rank <= 3 ? '2px solid' : '1px solid',
                borderColor: rankedHypothesis.rank === 1 ? 'gold' : 
                            rankedHypothesis.rank === 2 ? 'silver' : 
                            rankedHypothesis.rank === 3 ? '#CD7F32' : 'grey.300'
              }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      #{rankedHypothesis.rank} {rankedHypothesis.hypothesis.name}
                    </Typography>
                    <Chip
                      label={`Score: ${rankedHypothesis.totalScore.toFixed(1)}`}
                      color={rankedHypothesis.rank <= 3 ? 'primary' : 'default'}
                    />
                  </Box>
                  
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Statistical: {rankedHypothesis.scores.statistical.toFixed(1)}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(rankedHypothesis.scores.statistical / 20) * 100}
                        sx={{ height: 4, mb: 1 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Effect: {rankedHypothesis.scores.effect.toFixed(1)}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(rankedHypothesis.scores.effect / 10) * 100}
                        sx={{ height: 4, mb: 1 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Robustness: {rankedHypothesis.scores.robustness.toFixed(1)}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(rankedHypothesis.scores.robustness / 10) * 100}
                        sx={{ height: 4, mb: 1 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Parsimony: {rankedHypothesis.scores.parsimony.toFixed(1)}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(rankedHypothesis.scores.parsimony / 10) * 100}
                        sx={{ height: 4, mb: 1 }}
                      />
                    </Grid>
                  </Grid>

                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => onSelectHypothesis(rankedHypothesis.hypothesis)}
                    >
                      Select
                    </Button>
                    <Button
                      size="small"
                      variant="text"
                      onClick={() => handleShowDetails(rankedHypothesis.hypothesis)}
                    >
                      Details
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Hypothesis Detail Dialog */}
      <Dialog open={showDetailDialog} onClose={() => setShowDetailDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Hypothesis Details: {detailHypothesis?.name}
        </DialogTitle>
        <DialogContent>
          {detailHypothesis && (
            <HypothesisDetailView
              hypothesis={detailHypothesis}
              testResult={testResults.get(detailHypothesis.id)}
              comparisonMetrics={comparisonData.find(d => d.hypothesis.id === detailHypothesis.id)?.metrics}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetailDialog(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={() => {
              if (detailHypothesis) {
                onSelectHypothesis(detailHypothesis);
                setShowDetailDialog(false);
              }
            }}
          >
            Select for Testing
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Hypothesis Detail View Component
const HypothesisDetailView: React.FC<{
  hypothesis: CausalHypothesis;
  testResult?: HypothesisTestResult;
  comparisonMetrics?: any;
}> = ({ hypothesis, testResult, comparisonMetrics }) => {
  return (
    <Box sx={{ pt: 2 }}>
      <Typography variant="body1" paragraph>
        {hypothesis.description}
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>Structure</Typography>
          <Typography variant="body2">Variables: {hypothesis.variables.length}</Typography>
          <Typography variant="body2">Causal Edges: {hypothesis.edges.length}</Typography>
          <Typography variant="body2">
            Created: {new Date(hypothesis.metadata.created).toLocaleDateString()}
          </Typography>
        </Grid>

        {testResult && (
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>Test Results</Typography>
            <Typography variant="body2">P-Value: {testResult.pValue.toFixed(4)}</Typography>
            <Typography variant="body2">Effect Size: {testResult.effectSize.toFixed(3)}</Typography>
            <Typography variant="body2">
              Confidence Interval: [{testResult.confidenceInterval.lower.toFixed(3)}, {testResult.confidenceInterval.upper.toFixed(3)}]
            </Typography>
          </Grid>
        )}

        {comparisonMetrics && (
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>Comparison Metrics</Typography>
            <Grid container spacing={1}>
              <Grid item xs={3}>
                <Typography variant="body2">AIC: {comparisonMetrics.aic.toFixed(2)}</Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="body2">BIC: {comparisonMetrics.bic.toFixed(2)}</Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="body2">Complexity: {comparisonMetrics.complexity.toFixed(2)}</Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="body2">Parsimony: {comparisonMetrics.parsimony.toFixed(2)}</Typography>
              </Grid>
            </Grid>
          </Grid>
        )}

        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom>Variables</Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {hypothesis.variables.map(variable => (
              <Chip
                key={variable.id}
                label={`${variable.name} (${variable.type})`}
                size="small"
                variant="outlined"
              />
            ))}
          </Box>
        </Grid>

        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom>Causal Relationships</Typography>
          {hypothesis.edges.map(edge => {
            const sourceVar = hypothesis.variables.find(v => v.id === edge.sourceId);
            const targetVar = hypothesis.variables.find(v => v.id === edge.targetId);
            return (
              <Typography key={edge.id} variant="body2">
                {sourceVar?.name} → {targetVar?.name} (strength: {edge.strength.toFixed(2)})
              </Typography>
            );
          })}
        </Grid>
      </Grid>
    </Box>
  );
};

// Utility functions
function calculateComplexity(hypothesis: CausalHypothesis): number {
  const variableCount = hypothesis.variables.length;
  const edgeCount = hypothesis.edges.length;
  const maxPossibleEdges = variableCount * (variableCount - 1);
  
  return maxPossibleEdges > 0 ? edgeCount / maxPossibleEdges : 0;
}

function calculateParsimony(hypothesis: CausalHypothesis): number {
  const variableCount = hypothesis.variables.length;
  const edgeCount = hypothesis.edges.length;
  
  // Parsimony is inversely related to complexity
  return variableCount > 0 ? Math.max(0, 1 - (edgeCount / (variableCount * 2))) : 0;
}

function calculateAIC(hypothesis: CausalHypothesis, testResult?: HypothesisTestResult): number {
  if (!testResult) return Infinity;
  
  const k = hypothesis.edges.length; // Number of parameters
  const logLikelihood = -Math.log(testResult.pValue + 1e-10); // Approximate log-likelihood
  
  return 2 * k - 2 * logLikelihood;
}

function calculateBIC(hypothesis: CausalHypothesis, testResult?: HypothesisTestResult): number {
  if (!testResult) return Infinity;
  
  const k = hypothesis.edges.length; // Number of parameters
  const n = 100; // Assume sample size (should be passed from actual data)
  const logLikelihood = -Math.log(testResult.pValue + 1e-10);
  
  return Math.log(n) * k - 2 * logLikelihood;
}

function calculateEvidenceRatio(testResult: HypothesisTestResult): number {
  // Bayes factor approximation
  const bf = Math.exp(-0.5 * Math.pow(testResult.effectSize, 2));
  return Math.min(bf, 10); // Cap at 10 for display purposes
}

function calculateStatisticalPower(testResult: HypothesisTestResult): number {
  // Approximate power calculation based on effect size
  const effectSize = Math.abs(testResult.effectSize);
  return Math.min(1, effectSize * 0.8 + 0.2);
}

// function _getMetricValue(metrics: any, metricName: string): number {
//   switch (metricName) {
//     case 'Statistical Significance':
//       return Math.max(0, 1 - metrics.pValue);
//     case 'Effect Size':
//       return Math.min(1, metrics.effectSize);
//     case 'Robustness':
//       return metrics.robustness;
//     case 'Parsimony':
//       return metrics.parsimony;
//     case 'Evidence Ratio':
//       return Math.min(1, metrics.evidenceRatio / 10);
//     default:
//       return 0;
//   }
// }

// function _getColorForIndex(index: number): string {
//   const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0'];
//   return colors[index % colors.length];
// }