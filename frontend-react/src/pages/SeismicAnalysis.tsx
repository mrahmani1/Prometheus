/**
 * Seismic Analysis page - Real-time seismic data analysis and visualization
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Card, 
  CardContent, 
  CircularProgress,
  Alert,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { Refresh, Timeline, Map, Assessment } from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import { Helmet } from 'react-helmet-async';
import RealDataService from '../services/RealDataService';
import DatabaseService from '../services/DatabaseService';

interface SeismicEvent {
  id: string;
  time: Date;
  latitude: number;
  longitude: number;
  depth: number;
  magnitude: number;
  magnitudeType: string;
  location: string;
  catalog: string;
}

interface SeismicStation {
  id: string;
  network: string;
  station: string;
  latitude: number;
  longitude: number;
  elevation: number;
  instrumentType: string;
  dataQuality: number;
}

const SeismicAnalysis: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [seismicEvents, setSeismicEvents] = useState<SeismicEvent[]>([]);
  const [seismicStations, setSeismicStations] = useState<SeismicStation[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const loadSeismicData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const realDataService = RealDataService.getInstance();
      const databaseService = DatabaseService.getInstance();

      // Fetch recent seismic events from USGS/IRIS
      const events = await realDataService.fetchSeismicEvents({
        startTime: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // Last 7 days
        endTime: new Date(),
        minMagnitude: 2.0,
        bounds: {
          minLatitude: 32.0,
          maxLatitude: 49.0,
          minLongitude: -125.0,
          maxLongitude: -110.0, // Western US
        },
      });

      // Fetch seismic stations from database
      const stations = await databaseService.querySeismicStations({
        bounds: {
          minLatitude: 32.0,
          maxLatitude: 49.0,
          minLongitude: -125.0,
          maxLongitude: -110.0,
        },
        networks: ['US', 'CI', 'NC', 'UW'],
        active: true,
      });

      setSeismicEvents(events);
      // Map SeismicStationRecord to SeismicStation
      const mappedStations = stations.map(station => ({
        ...station,
        instrumentType: (station as any).instrumentType || 'broadband',
        dataQuality: (station as any).dataQuality || 0.9
      }));
      setSeismicStations(mappedStations);
      setLastUpdate(new Date());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load seismic data';
      setError(errorMessage);
      console.error('Seismic data loading error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSeismicData();
    
    // Set up auto-refresh every 5 minutes
    const interval = setInterval(loadSeismicData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [loadSeismicData]);

  const getMagnitudeColor = (magnitude: number): string => {
    if (magnitude >= 7.0) return '#d32f2f'; // Red
    if (magnitude >= 6.0) return '#f57c00'; // Orange
    if (magnitude >= 5.0) return '#fbc02d'; // Yellow
    if (magnitude >= 4.0) return '#689f38'; // Light green
    return '#388e3c'; // Green
  };

  const getStationStatusColor = (quality: number): string => {
    if (quality >= 0.9) return '#4caf50'; // Green
    if (quality >= 0.7) return '#ff9800'; // Orange
    return '#f44336'; // Red
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading real-time seismic data...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Helmet>
        <title>Seismic Analysis - Prometheus AI Platform</title>
        <meta name="description" content="Real-time seismic data analysis and earthquake monitoring" />
      </Helmet>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {t('seismic.title', 'Real-time Seismic Analysis')}
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {lastUpdate && (
            <Typography variant="caption" color="textSecondary">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </Typography>
          )}
          <Tooltip title="Refresh Data">
            <IconButton onClick={loadSeismicData} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Recent Events Summary */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Timeline sx={{ mr: 1 }} />
                <Typography variant="h6">Recent Seismic Events</Typography>
              </Box>
              
              <Typography variant="h3" color="primary" gutterBottom>
                {seismicEvents.length}
              </Typography>
              
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Events in the last 7 days (M ≥ 2.0)
              </Typography>

              {seismicEvents.slice(0, 5).map((event) => (
                <Box key={event.id} sx={{ mb: 1, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" fontWeight="bold">
                      M {event.magnitude.toFixed(1)}
                    </Typography>
                    <Chip 
                      size="small" 
                      label={event.magnitudeType}
                      sx={{ bgcolor: getMagnitudeColor(event.magnitude), color: 'white' }}
                    />
                  </Box>
                  <Typography variant="caption" color="textSecondary">
                    {event.location}
                  </Typography>
                  <Typography variant="caption" display="block" color="textSecondary">
                    {event.time.toLocaleString()}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Station Network Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Map sx={{ mr: 1 }} />
                <Typography variant="h6">Monitoring Network</Typography>
              </Box>
              
              <Typography variant="h3" color="primary" gutterBottom>
                {seismicStations.length}
              </Typography>
              
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Active seismic stations
              </Typography>

              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Network Status:
                </Typography>
                {['US', 'CI', 'NC', 'UW'].map((network) => {
                  const networkStations = seismicStations.filter(s => s.network === network);
                  const avgQuality = networkStations.length > 0 
                    ? networkStations.reduce((sum, s) => sum + s.dataQuality, 0) / networkStations.length 
                    : 0;
                  
                  return (
                    <Box key={network} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{network}</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="caption">{networkStations.length} stations</Typography>
                        <Box 
                          sx={{ 
                            width: 12, 
                            height: 12, 
                            borderRadius: '50%', 
                            bgcolor: getStationStatusColor(avgQuality) 
                          }} 
                        />
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Tools */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Assessment sx={{ mr: 1 }} />
              <Typography variant="h6">Analysis Tools</Typography>
            </Box>
            
            <Typography variant="body1" color="textSecondary">
              Real-time seismic analysis tools including waveform analysis, event detection, 
              magnitude calculation, and seismic hazard assessment are available through the 
              integrated IRIS/USGS data streams.
            </Typography>
            
            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <Chip label="Real-time Waveforms" variant="outlined" />
              <Chip label="Automatic Detection" variant="outlined" />
              <Chip label="Magnitude Analysis" variant="outlined" />
              <Chip label="Hazard Assessment" variant="outlined" />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SeismicAnalysis;