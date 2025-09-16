/**
 * Multi-Parameter Correlation Matrix Component
 * Interactive correlation analysis between geological parameters
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Chip,
  Tooltip,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';

interface MultiParameterCorrelationMatrixProps {
  data: number[][];
  parameters: string[];
  title?: string;
  showValues?: boolean;
  colorScale?: string;
}

const MultiParameterCorrelationMatrix: React.FC<MultiParameterCorrelationMatrixProps> = ({
  data,
  parameters,
  title = 'Parameter Correlation Matrix',
  showValues = true,
  colorScale = 'RdBu',
}) => {
  const theme = useTheme();

  // State for correlation matrix
  const [correlationMatrix, setCorrelationMatrix] = useState<{
    matrix: number[][];
    labels: string[];
  }>({ matrix: [], labels: [] });

  // Load correlation matrix
  useEffect(() => {
    const loadCorrelationMatrix = async () => {
      if (data.length === 0 || data[0].length === 0) {
        // Generate correlation data from real backend
        const realParams = parameters.length > 0 ? parameters : [
          'Seismic Activity', 'SO2 Emissions', 'GPS Displacement', 'Thermal Anomaly',
          'Deformation Rate', 'Gas Temperature', 'Tilt Change', 'Earthquake Count'
        ];
        
        try {
          // Calculate real correlations from backend causal analysis
          const correlationMatrix = await Promise.all(realParams.map(async (param1, i) =>
            await Promise.all(realParams.map(async (param2, j) => {
              if (i === j) return 1;
              
              try {
                // Fetch real correlation from causal analysis backend
                const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/causal/correlation`, {
                  method: 'POST',
                  headers: {
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
                    'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({
                    parameter1: param1,
                    parameter2: param2,
                    timeRange: {
                      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
                      end: new Date(),
                    },
                    method: 'pearson', // Pearson correlation coefficient
                  }),
                });
                
                if (response.ok) {
                  const correlationData = await response.json();
                  return correlationData.correlation;
                } else {
                  console.warn(`Failed to fetch correlation for ${param1} vs ${param2}`);
                  return 0; // Default to no correlation if data unavailable
                }
              } catch (error) {
                console.warn(`Error fetching correlation for ${param1} vs ${param2}:`, error);
                return 0; // Default to no correlation on error
              }
            }))
          ));
          
          setCorrelationMatrix({
            matrix: correlationMatrix,
            labels: realParams,
          });
        } catch (error) {
          console.error('Failed to load correlation matrix:', error);
          // Fallback to identity matrix
          const identityMatrix = realParams.map((_, i) =>
            realParams.map((_, j) => i === j ? 1 : 0)
          );
          setCorrelationMatrix({
            matrix: identityMatrix,
            labels: realParams,
          });
        }
      } else {
        setCorrelationMatrix({
          matrix: data,
          labels: parameters,
        });
      }
    };

    loadCorrelationMatrix();
  }, [data, parameters]);

  // Create heatmap data
  const heatmapData = useMemo(() => {
    const { matrix, labels } = correlationMatrix;
    
    return [{
      z: matrix,
      x: labels,
      y: labels,
      type: 'heatmap',
      colorscale: colorScale,
      zmin: -1,
      zmax: 1,
      showscale: true,
      colorbar: {
        title: 'Correlation Coefficient',
        titleside: 'right',
      },
      hoverongaps: false,
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>' +
                    'Correlation: %{z:.3f}<br>' +
                    '<extra></extra>',
      text: showValues ? matrix.map(row => 
        row.map(val => val.toFixed(3))
      ) : undefined,
      texttemplate: showValues ? '%{text}' : undefined,
      textfont: {
        color: matrix.map(row => 
          row.map(val => Math.abs(val) > 0.5 ? 'white' : 'black')
        ),
        size: 10,
      },
    }];
  }, [correlationMatrix, colorScale, showValues]);

  // Get correlation strength description
  const getCorrelationStrength = (value: number): { label: string; color: string } => {
    const abs = Math.abs(value);
    if (abs >= 0.8) return { label: 'Very Strong', color: theme.palette.error.main };
    if (abs >= 0.6) return { label: 'Strong', color: theme.palette.warning.main };
    if (abs >= 0.4) return { label: 'Moderate', color: theme.palette.info.main };
    if (abs >= 0.2) return { label: 'Weak', color: theme.palette.success.main };
    return { label: 'Very Weak', color: theme.palette.grey[500] };
  };

  // Find strongest correlations
  const strongestCorrelations = useMemo(() => {
    const { matrix, labels } = correlationMatrix;
    const correlations: Array<{ param1: string; param2: string; value: number }> = [];
    
    for (let i = 0; i < matrix.length; i++) {
      for (let j = i + 1; j < matrix[i].length; j++) {
        correlations.push({
          param1: labels[i],
          param2: labels[j],
          value: matrix[i][j],
        });
      }
    }
    
    return correlations
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5);
  }, [correlationMatrix]);

  return (
    <Box>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">{title}</Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <FormControlLabel
                control={<Switch defaultChecked={showValues} />}
                label="Show Values"
              />
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Color Scale</InputLabel>
                <Select defaultValue={colorScale}>
                  <MenuItem value="RdBu">Red-Blue</MenuItem>
                  <MenuItem value="Viridis">Viridis</MenuItem>
                  <MenuItem value="Plasma">Plasma</MenuItem>
                  <MenuItem value="RdYlBu">Red-Yellow-Blue</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Box>

          <Box height={500}>
            <Plot
              data={heatmapData as any}
              layout={{
                title: '',
                height: 500,
                margin: { l: 120, r: 60, t: 20, b: 120 },
                xaxis: {
                  tickangle: -45,
                  side: 'bottom',
                },
                yaxis: {
                  tickangle: 0,
                  autorange: 'reversed',
                },
                font: { size: 11 },
              }}
              config={{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
              }}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>

          {/* Strongest Correlations Summary */}
          <Box mt={2}>
            <Typography variant="subtitle2" gutterBottom>
              Strongest Correlations
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {strongestCorrelations.map((corr: any, index: number) => {
                const strength = getCorrelationStrength(corr.value);
                return (
                  <Tooltip
                    key={index}
                    title={`${corr.param1} ↔ ${corr.param2}: ${corr.value.toFixed(3)}`}
                  >
                    <Chip
                      label={`${strength.label} (${corr.value.toFixed(2)})`}
                      size="small"
                      sx={{
                        backgroundColor: strength.color,
                        color: 'white',
                        '&:hover': {
                          backgroundColor: strength.color,
                          opacity: 0.8,
                        },
                      }}
                    />
                  </Tooltip>
                );
              })}
            </Box>
          </Box>

          {/* Legend */}
          <Box mt={2}>
            <Typography variant="caption" color="textSecondary">
              Correlation strength: |r| ≥ 0.8 (Very Strong), 0.6-0.8 (Strong), 0.4-0.6 (Moderate), 0.2-0.4 (Weak), &lt; 0.2 (Very Weak)
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MultiParameterCorrelationMatrix;