/**
 * Volcanic Monitoring page - Real-time volcanic activity monitoring and analysis
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
  Tooltip,
  LinearProgress
} from '@mui/material';
import { 
  Refresh, 
  Whatshot, 
  TrendingUp, 
  Warning,
  Satellite,
  Thermostat
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import { Helmet } from 'react-helmet-async';
import RealDataService from '../services/RealDataService';
import DatabaseService from '../services/DatabaseService';

interface VolcanicSystem {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation: number;
  volcanoType: string;
  alertLevel: string;
  monitoringStatus: string;
  lastEruption?: Date;
  population: {
    within5km: number;
    within10km: number;
    within30km: number;
    within100km: number;
  };
}

interface ThermalAnomaly {
  id: string;
  detectionTime: Date;
  latitude: number;
  longitude: number;
  brightness: number;
  confidence: number;
  satellite: string;
}

const VolcanicMonitoring: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [volcanicSystems, setVolcanicSystems] = useState<VolcanicSystem[]>([]);
  const [thermalAnomalies, setThermalAnomalies] = useState<ThermalAnomaly[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const loadVolcanicData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const realDataService = RealDataService.getInstance();
      const databaseService = DatabaseService.getInstance();

      // Fetch volcanic systems from database
      const systems = await databaseService.queryVolcanicSystems({
        bounds: {
          minLatitude: -90,
          maxLatitude: 90,
          minLongitude: -180,
          maxLongitude: 180,
        },
        alertLevels: ['green', 'yellow', 'orange', 'red'],
        monitoringStatus: ['monitored', 'partially_monitored'],
      });

      // Fetch thermal anomalies from NASA FIRMS
      const anomalies = await realDataService.fetchThermalAnomalies({
        bounds: {
          minLatitude: -90,
          maxLatitude: 90,
          minLongitude: -180,
          maxLongitude: 180,
        },
        startDate: new Date(Date.now() - 24 * 60 * 60 * 1000), // Last 24 hours
        endDate: new Date(),
        satellite: 'MODIS',
      });

      // Map VolcanicSystemRecord to VolcanicSystem
      const mappedSystems = systems.map(system => ({
        ...system,
        volcanoType: (system as any).volcanoType || 'stratovolcano',
        alertLevel: (system as any).alertLevel || 'green',
        monitoringStatus: (system as any).monitoringStatus || 'monitored',
        population: (system as any).population || {
          within5km: 0,
          within10km: 0,
          within30km: 0,
          within100km: 0
        }
      }));
      setVolcanicSystems(mappedSystems);
      setThermalAnomalies(anomalies);
      setLastUpdate(new Date());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load volcanic monitoring data';
      setError(errorMessage);
      console.error('Volcanic data loading error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVolcanicData();
    
    // Set up auto-refresh every 10 minutes
    const interval = setInterval(loadVolcanicData, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [loadVolcanicData]);

  const getAlertLevelColor = (level: string): string => {
    switch (level.toLowerCase()) {
      case 'red': return '#d32f2f';
      case 'orange': return '#f57c00';
      case 'yellow': return '#fbc02d';
      case 'green': return '#388e3c';
      default: return '#757575';
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  // @ts-ignore - Function kept for future use
  const getAlertLevelSeverity = (level: string): 'error' | 'warning' | 'info' | 'success' => {
    switch (level.toLowerCase()) {
      case 'red': return 'error';
      case 'orange': return 'warning';
      case 'yellow': return 'warning';
      case 'green': return 'success';
      default: return 'info';
    }
  };

  const getPopulationRisk = (population: VolcanicSystem['population']): { level: string; color: string } => {
    const total = population.within100km;
    if (total > 1000000) return { level: 'Very High', color: '#d32f2f' };
    if (total > 100000) return { level: 'High', color: '#f57c00' };
    if (total > 10000) return { level: 'Moderate', color: '#fbc02d' };
    return { level: 'Low', color: '#388e3c' };
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading real-time volcanic monitoring data...
        </Typography>
      </Box>
    );
  }

  const alertVolcanoes = volcanicSystems.filter(v => ['orange', 'red'].includes(v.alertLevel.toLowerCase()));
  const monitoredVolcanoes = volcanicSystems.filter(v => v.monitoringStatus === 'monitored');
  const recentAnomalies = thermalAnomalies.filter(a => 
    new Date(a.detectionTime).getTime() > Date.now() - 6 * 60 * 60 * 1000 // Last 6 hours
  );

  return (
    <Box sx={{ p: 3 }}>
      <Helmet>
        <title>Volcanic Monitoring - Prometheus AI Platform</title>
        <meta name="description" content="Real-time volcanic activity monitoring and hazard assessment" />
      </Helmet>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {t('volcanic.title', 'Real-time Volcanic Monitoring')}
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {lastUpdate && (
            <Typography variant="caption" color="textSecondary">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </Typography>
          )}
          <Tooltip title="Refresh Data">
            <IconButton onClick={loadVolcanicData} disabled={loading}>
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

      {/* Alert Summary */}
      {alertVolcanoes.length > 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="subtitle1" fontWeight="bold">
            {alertVolcanoes.length} volcano(s) at elevated alert levels
          </Typography>
          <Box sx={{ mt: 1 }}>
            {alertVolcanoes.map(volcano => (
              <Chip
                key={volcano.id}
                label={`${volcano.name} (${volcano.alertLevel.toUpperCase()})`}
                size="small"
                sx={{ 
                  mr: 1, 
                  mb: 1,
                  bgcolor: getAlertLevelColor(volcano.alertLevel),
                  color: 'white'
                }}
              />
            ))}
          </Box>
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Global Overview */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Whatshot sx={{ mr: 1 }} />
                <Typography variant="h6">Global Overview</Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="h3" color="primary">
                  {volcanicSystems.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Volcanic systems tracked
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="h4" color="secondary">
                  {monitoredVolcanoes.length}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Actively monitored
                </Typography>
              </Box>

              <LinearProgress 
                variant="determinate" 
                value={(monitoredVolcanoes.length / volcanicSystems.length) * 100}
                sx={{ mt: 1 }}
              />
              <Typography variant="caption" color="textSecondary">
                {Math.round((monitoredVolcanoes.length / volcanicSystems.length) * 100)}% monitoring coverage
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Alert Levels */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Warning sx={{ mr: 1 }} />
                <Typography variant="h6">Alert Levels</Typography>
              </Box>
              
              {['red', 'orange', 'yellow', 'green'].map(level => {
                const count = volcanicSystems.filter(v => v.alertLevel.toLowerCase() === level).length;
                return (
                  <Box key={level} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box 
                        sx={{ 
                          width: 16, 
                          height: 16, 
                          borderRadius: '50%', 
                          bgcolor: getAlertLevelColor(level),
                          mr: 1
                        }} 
                      />
                      <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                        {level}
                      </Typography>
                    </Box>
                    <Typography variant="body2" fontWeight="bold">
                      {count}
                    </Typography>
                  </Box>
                );
              })}
            </CardContent>
          </Card>
        </Grid>

        {/* Thermal Anomalies */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Thermostat sx={{ mr: 1 }} />
                <Typography variant="h6">Thermal Anomalies</Typography>
              </Box>
              
              <Typography variant="h3" color="primary" gutterBottom>
                {recentAnomalies.length}
              </Typography>
              
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Detected in last 6 hours
              </Typography>

              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Satellite Sources:
                </Typography>
                {['MODIS', 'VIIRS'].map(satellite => {
                  const count = thermalAnomalies.filter(a => a.satellite === satellite).length;
                  return (
                    <Box key={satellite} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{satellite}</Typography>
                      <Typography variant="body2" fontWeight="bold">{count}</Typography>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* High-Risk Volcanoes */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TrendingUp sx={{ mr: 1 }} />
              <Typography variant="h6">High-Risk Volcanic Systems</Typography>
            </Box>
            
            <Grid container spacing={2}>
              {volcanicSystems
                .filter(v => ['orange', 'red'].includes(v.alertLevel.toLowerCase()) || 
                           getPopulationRisk(v.population).level === 'Very High')
                .slice(0, 6)
                .map(volcano => {
                  const risk = getPopulationRisk(volcano.population);
                  return (
                    <Grid item xs={12} sm={6} md={4} key={volcano.id}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                            {volcano.name}
                          </Typography>
                          
                          <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                            <Chip 
                              size="small"
                              label={volcano.alertLevel.toUpperCase()}
                              sx={{ 
                                bgcolor: getAlertLevelColor(volcano.alertLevel),
                                color: 'white'
                              }}
                            />
                            <Chip 
                              size="small"
                              label={risk.level}
                              sx={{ bgcolor: risk.color, color: 'white' }}
                            />
                          </Box>
                          
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            {volcano.volcanoType} • {volcano.elevation.toLocaleString()}m
                          </Typography>
                          
                          <Typography variant="caption" color="textSecondary">
                            Population at risk: {volcano.population.within100km.toLocaleString()}
                          </Typography>
                          
                          {volcano.lastEruption && (
                            <Typography variant="caption" display="block" color="textSecondary">
                              Last eruption: {volcano.lastEruption.getFullYear()}
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
            </Grid>
          </Paper>
        </Grid>

        {/* Data Sources */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Satellite sx={{ mr: 1 }} />
              <Typography variant="h6">Real-time Data Sources</Typography>
            </Box>
            
            <Typography variant="body1" color="textSecondary" gutterBottom>
              Integrated monitoring from global volcanic observatories and satellite systems:
            </Typography>
            
            <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip label="USGS Volcano Hazards Program" variant="outlined" />
              <Chip label="Global Volcanism Program" variant="outlined" />
              <Chip label="NASA FIRMS" variant="outlined" />
              <Chip label="MODIS Thermal Alerts" variant="outlined" />
              <Chip label="VIIRS Thermal Data" variant="outlined" />
              <Chip label="Regional Observatories" variant="outlined" />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default VolcanicMonitoring;