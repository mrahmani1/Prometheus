import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  Compare as CompareIcon
} from '@mui/icons-material';
import { CausalHypothesis } from '../../types/causal-analysis';

interface HypothesisComparatorProps {
  hypotheses: CausalHypothesis[];
  onRemoveHypothesis: (id: string) => void;
  onSelectHypothesis: (hypothesis: CausalHypothesis) => void;
}

export const HypothesisComparator: React.FC<HypothesisComparatorProps> = ({
  hypotheses,
  onRemoveHypothesis,
  onSelectHypothesis
}) => {
  if (hypotheses.length === 0) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Add hypotheses to comparison by clicking "Add to Comparison" after testing them.
      </Alert>
    );
  }

  const getComplexityScore = (hypothesis: CausalHypothesis): number => {
    return hypothesis.variables.length + hypothesis.edges.length * 2;
  };

  const getAverageStrength = (hypothesis: CausalHypothesis): number => {
    if (hypothesis.edges.length === 0) return 0;
    return hypothesis.edges.reduce((sum, edge) => sum + edge.strength, 0) / hypothesis.edges.length;
  };

  const getAverageConfidence = (hypothesis: CausalHypothesis): number => {
    if (hypothesis.edges.length === 0) return 0;
    return hypothesis.edges.reduce((sum, edge) => sum + edge.confidence, 0) / hypothesis.edges.length;
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CompareIcon />
        Hypothesis Comparison ({hypotheses.length})
      </Typography>

      <TableContainer component={Card}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell align="center">Variables</TableCell>
              <TableCell align="center">Relationships</TableCell>
              <TableCell align="center">Complexity</TableCell>
              <TableCell align="center">Avg Strength</TableCell>
              <TableCell align="center">Avg Confidence</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {hypotheses.map((hypothesis) => (
              <TableRow key={hypothesis.id} hover>
                <TableCell>
                  <Box>
                    <Typography variant="body2" fontWeight="medium">
                      {hypothesis.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {hypothesis.description}
                    </Typography>
                  </Box>
                </TableCell>
                
                <TableCell align="center">
                  <Chip
                    label={hypothesis.variables.length}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                </TableCell>
                
                <TableCell align="center">
                  <Chip
                    label={hypothesis.edges.length}
                    size="small"
                    color="secondary"
                    variant="outlined"
                  />
                </TableCell>
                
                <TableCell align="center">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(getComplexityScore(hypothesis) * 10, 100)}
                      sx={{ width: 60, height: 6 }}
                    />
                    <Typography variant="caption">
                      {getComplexityScore(hypothesis)}
                    </Typography>
                  </Box>
                </TableCell>
                
                <TableCell align="center">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={getAverageStrength(hypothesis) * 100}
                      sx={{ width: 60, height: 6 }}
                      color="success"
                    />
                    <Typography variant="caption">
                      {getAverageStrength(hypothesis).toFixed(2)}
                    </Typography>
                  </Box>
                </TableCell>
                
                <TableCell align="center">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={getAverageConfidence(hypothesis) * 100}
                      sx={{ width: 60, height: 6 }}
                      color="info"
                    />
                    <Typography variant="caption">
                      {getAverageConfidence(hypothesis).toFixed(2)}
                    </Typography>
                  </Box>
                </TableCell>
                
                <TableCell align="center">
                  <IconButton
                    size="small"
                    onClick={() => onSelectHypothesis(hypothesis)}
                    title="View/Edit Hypothesis"
                  >
                    <ViewIcon />
                  </IconButton>
                  <IconButton
                    size="small"
                    onClick={() => onRemoveHypothesis(hypothesis.id)}
                    title="Remove from Comparison"
                    color="error"
                  >
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {hypotheses.length >= 2 && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Comparison Summary
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Most Complex
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {hypotheses.reduce((max, h) => 
                    getComplexityScore(h) > getComplexityScore(max) ? h : max
                  ).name}
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Highest Avg Strength
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {hypotheses.reduce((max, h) => 
                    getAverageStrength(h) > getAverageStrength(max) ? h : max
                  ).name}
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Highest Avg Confidence
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {hypotheses.reduce((max, h) => 
                    getAverageConfidence(h) > getAverageConfidence(max) ? h : max
                  ).name}
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};