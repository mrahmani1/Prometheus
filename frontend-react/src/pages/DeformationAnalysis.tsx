/**
 * Deformation Analysis page - Real MT-SAR deformation analysis with database integration
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  CircularProgress, 
  Alert,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import DatabaseService from '../services/DatabaseService';
import GeologicalMap from '../components/visualization/maps/GeologicalMap';
import Plot from 'react-plotly.js';

interface DeformationData {
  id: string;
  location: {
    latitude: number;
    longitude: number;
    elevation: number;
  };
  timestamp: Date;
  displacement: {
    vertical: number;
    horizontal: number;
    lineOfSight: number;
  };
  coherence: number;
  satellite: 'Sentinel-1' | 'ALOS-2' | 'TerraSAR-X';
  orbitDirection: 'ascending' | 'descending';
  processingMethod: 'MT-SAR' | 'DInSAR' | 'PSInSAR';
  quality: number;
  uncertainty: number;
}

interface DeformationTimeSeries {
  location: string;
  data: Array<{
    timestamp: Date;
    displacement: number;
    uncertainty: number;
  }>;
}

const DeformationAnalysis: React.FC = () => {
  const { t } = useTranslation();
  const [deformationData, setDeformationData] = useState<DeformationData[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<DeformationTimeSeries[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSatellite, setSelectedSatellite] = useState<string>('all');
  const [selectedMethod, setSelectedMethod] = useState<string>('all');
  const databaseService = DatabaseService.getInstance();

  useEffect(() => {
    const loadDeformationData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Query real MT-SAR deformation data from database
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/deformation/mt-sar`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            satellite: selectedSatellite !== 'all' ? selectedSatellite : undefined,
            processingMethod: selectedMethod !== 'all' ? selectedMethod : undefined,
            timeRange: {
              start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), // Last year
              end: new Date(),
            },
            qualityThreshold: 0.5,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          setDeformationData(data.deformationPoints || []);
          setTimeSeriesData(data.timeSeries || []);
        } else {
          throw new Error(`Failed to load deformation data: ${response.status}`);
        }
      } catch (err) {
        console.error('Failed to load deformation data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load deformation data');
      } finally {
        setLoading(false);
      }
    };

    loadDeformationData();
  }, [databaseService, selectedSatellite, selectedMethod]);

  // Removed unused getDisplacementColor function

  const getSatelliteColor = (satellite: string): 'primary' | 'secondary' | 'success' => {
    switch (satellite) {
      case 'Sentinel-1': return 'primary';
      case 'ALOS-2': return 'secondary';
      case 'TerraSAR-X': return 'success';
      default: return 'primary';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading MT-SAR Deformation Data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          <Typography variant="h6">Failed to Load Deformation Analysis</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('deformation.title', 'Deformation Analysis')}
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Real-time MT-SAR deformation analysis with multi-temporal satellite data
      </Typography>

      {/* Filters */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth>
            <InputLabel>Satellite Platform</InputLabel>
            <Select
              value={selectedSatellite}
              onChange={(e) => setSelectedSatellite(e.target.value)}
            >
              <MenuItem value="all">All Satellites</MenuItem>
              <MenuItem value="Sentinel-1">Sentinel-1</MenuItem>
              <MenuItem value="ALOS-2">ALOS-2</MenuItem>
              <MenuItem value="TerraSAR-X">TerraSAR-X</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth>
            <InputLabel>Processing Method</InputLabel>
            <Select
              value={selectedMethod}
              onChange={(e) => setSelectedMethod(e.target.value)}
            >
              <MenuItem value="all">All Methods</MenuItem>
              <MenuItem value="MT-SAR">Multi-Temporal SAR</MenuItem>
              <MenuItem value="DInSAR">Differential InSAR</MenuItem>
              <MenuItem value="PSInSAR">Persistent Scatterer InSAR</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Button
            variant="outlined"
            onClick={() => window.location.reload()}
            fullWidth
          >
            Refresh Data
          </Button>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Deformation Map ({deformationData.length} points)
              </Typography>
              <Box sx={{ height: 500 }}>
                <GeologicalMap
                  data={deformationData.map(point => ({
                    id: point.id,
                    location: point.location,
                    dataType: 'deformation' as any,
                    value: point.displacement.vertical,
                    timestamp: point.timestamp,
                    qualityMetrics: {
                      completeness: point.quality,
                      accuracy: point.coherence,
                      consistency: point.quality * 0.9,
                      timeliness: 0.95,
                      validity: point.quality,
                      uniqueness: 1.0,
                      overall: point.quality
                    },
                    metadata: {
                      satellite: point.satellite,
                      coherence: point.coherence,
                      displacement: point.displacement,
                      processingMethod: point.processingMethod,
                      quality: point.quality
                    }
                  }))}
                  seismicEvents={[]}
                  volcanicStates={[]}
                  initialViewport={{ center: { lat: 0, lng: 0 }, zoom: 2 }}
                  onFeatureClick={(feature) => console.log('Selected deformation point:', feature)}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Summary
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Total Measurements: {deformationData.length}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    By Satellite:
                  </Typography>
                  {Object.entries(
                    deformationData.reduce((acc, point) => {
                      acc[point.satellite] = (acc[point.satellite] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).map(([satellite, count]) => (
                    <Chip
                      key={satellite}
                      label={`${satellite}: ${count}`}
                      size="small"
                      color={getSatelliteColor(satellite)}
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Average Coherence: {
                      deformationData.length > 0 
                        ? (deformationData.reduce((sum, p) => sum + p.coherence, 0) / deformationData.length).toFixed(3)
                        : 'N/A'
                    }
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Max Vertical Displacement: {
                      deformationData.length > 0
                        ? Math.max(...deformationData.map(p => Math.abs(p.displacement.vertical))).toFixed(2) + ' mm'
                        : 'N/A'
                    }
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Methods
              </Typography>
              {Object.entries(
                deformationData.reduce((acc, point) => {
                  acc[point.processingMethod] = (acc[point.processingMethod] || 0) + 1;
                  return acc;
                }, {} as Record<string, number>)
              ).map(([method, count]) => (
                <Box key={method} sx={{ mb: 1 }}>
                  <Typography variant="body2">
                    {method}: {count} points ({((count / deformationData.length) * 100).toFixed(1)}%)
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {timeSeriesData.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Deformation Time Series
                </Typography>
                <Plot
                  data={timeSeriesData.map(series => ({
                    x: series.data.map(d => d.timestamp.getTime()),
                    y: series.data.map(d => d.displacement),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: series.location,
                    error_y: {
                      type: 'data',
                      array: series.data.map(d => d.uncertainty),
                      visible: true
                    }
                  }))}
                  layout={{
                    title: 'Vertical Displacement Over Time',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Displacement (mm)' },
                    height: 400
                  }}
                  style={{ width: '100%', height: '400px' }}
                />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default DeformationAnalysis;