/**
 * Resource Assessment page - Real geological resource assessment with database integration
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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import DatabaseService from '../services/DatabaseService';
import GeologicalMap from '../components/visualization/maps/GeologicalMap';

interface ResourceDeposit {
  id: string;
  name: string;
  type: 'mineral' | 'hydrocarbon' | 'geothermal' | 'groundwater';
  location: {
    latitude: number;
    longitude: number;
    elevation: number;
  };
  estimatedReserves: number;
  confidence: number;
  status: 'exploration' | 'development' | 'production' | 'depleted';
  lastAssessment: Date;
  economicViability: 'high' | 'medium' | 'low' | 'unknown';
}

const ResourceAssessment: React.FC = () => {
  const { t } = useTranslation();
  const [deposits, setDeposits] = useState<ResourceDeposit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const databaseService = DatabaseService.getInstance();

  useEffect(() => {
    const loadResourceDeposits = async () => {
      try {
        setLoading(true);
        setError(null);

        // Query real resource deposits from database
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/resources/deposits`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const depositData = await response.json();
          setDeposits(depositData.deposits || []);
        } else {
          throw new Error(`Failed to load resource deposits: ${response.status}`);
        }
      } catch (err) {
        console.error('Failed to load resource deposits:', err);
        setError(err instanceof Error ? err.message : 'Failed to load resource deposits');
      } finally {
        setLoading(false);
      }
    };

    loadResourceDeposits();
  }, [databaseService]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'production': return 'success';
      case 'development': return 'primary';
      case 'exploration': return 'warning';
      case 'depleted': return 'error';
      default: return 'default';
    }
  };

  const getViabilityColor = (viability: string) => {
    switch (viability) {
      case 'high': return 'success';
      case 'medium': return 'warning';
      case 'low': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading Resource Assessment Data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          <Typography variant="h6">Failed to Load Resource Assessment</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('resources.title', 'Resource Assessment')}
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Real-time geological resource assessment with production database integration
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Resource Deposit Locations
              </Typography>
              <Box sx={{ height: 500 }}>
                <GeologicalMap
                  data={deposits.map(deposit => ({
                    id: deposit.id,
                    location: deposit.location,
                    dataType: deposit.type as any,
                    value: deposit.estimatedReserves,
                    timestamp: deposit.lastAssessment,
                    qualityMetrics: {
                      completeness: deposit.confidence,
                      accuracy: deposit.confidence * 0.9,
                      consistency: deposit.confidence * 0.95,
                      timeliness: 0.9,
                      validity: deposit.confidence,
                      uniqueness: 1.0,
                      overall: deposit.confidence
                    },
                    metadata: {
                      name: deposit.name,
                      status: deposit.status,
                      confidence: deposit.confidence,
                      economicViability: deposit.economicViability
                    }
                  }))}
                  seismicEvents={[]}
                  volcanicStates={[]}
                  initialViewport={{ center: { lat: 0, lng: 0 }, zoom: 2 }}
                  onFeatureClick={(feature) => console.log('Selected deposit:', feature)}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Assessment Summary
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Total Deposits: {deposits.length}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    By Type:
                  </Typography>
                  {Object.entries(
                    deposits.reduce((acc, deposit) => {
                      acc[deposit.type] = (acc[deposit.type] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).map(([type, count]) => (
                    <Chip
                      key={type}
                      label={`${type}: ${count}`}
                      size="small"
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    By Status:
                  </Typography>
                  {Object.entries(
                    deposits.reduce((acc, deposit) => {
                      acc[deposit.status] = (acc[deposit.status] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).map(([status, count]) => (
                    <Chip
                      key={status}
                      label={`${status}: ${count}`}
                      size="small"
                      color={getStatusColor(status) as any}
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Resource Deposits ({deposits.length})
              </Typography>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Location</TableCell>
                      <TableCell>Reserves</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Economic Viability</TableCell>
                      <TableCell>Last Assessment</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {deposits.map((deposit) => (
                      <TableRow key={deposit.id}>
                        <TableCell>{deposit.name}</TableCell>
                        <TableCell>
                          <Chip label={deposit.type} size="small" />
                        </TableCell>
                        <TableCell>
                          {deposit.location.latitude.toFixed(4)}, {deposit.location.longitude.toFixed(4)}
                        </TableCell>
                        <TableCell>{deposit.estimatedReserves.toLocaleString()}</TableCell>
                        <TableCell>{(deposit.confidence * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          <Chip 
                            label={deposit.status} 
                            size="small" 
                            color={getStatusColor(deposit.status) as any}
                          />
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={deposit.economicViability} 
                            size="small" 
                            color={getViabilityColor(deposit.economicViability) as any}
                          />
                        </TableCell>
                        <TableCell>{deposit.lastAssessment.toLocaleDateString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResourceAssessment;