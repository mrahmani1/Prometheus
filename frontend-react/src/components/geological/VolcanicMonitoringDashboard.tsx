import React, { useState, useEffect, useMemo, useCallback } from 'react';
import DatabaseService from '../../services/DatabaseService';
import RealDataService from '../../services/RealDataService';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Alert,
  // LinearProgress - unused
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  // Dialog, DialogTitle, DialogContent, DialogActions - unused
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider
} from '@mui/material';
import {
  // Warning, TrendingUp, TrendingDown, Fullscreen, Settings, LocationOn, Timeline, Speed, Notifications, Map as MapIcon - unused
  Refresh,
  Thermostat,
  BarChart as _BarChart,
  ShowChart as _ShowChart
} from '@mui/icons-material';
import { format, subHours as _subHours, subDays } from 'date-fns';
import {
  VolcanicState,
  AlertLevel,
  ThermalAnomaly as _ThermalAnomaly,
  GasComposition,
  CalderaClassification as _CalderaClassification,
  SupereruptionRiskAssessment as _SupereruptionRiskAssessment
} from '../../types/geological';
import { ScientificCharts } from '../visualization/charts/ScientificCharts';
import { GeologicalMap } from '../visualization/maps/GeologicalMap';

interface VolcanicMonitoringDashboardProps {
  volcanicStates: VolcanicState[];
  onAlertGenerated?: (alert: any) => void;
  onVolcanoSelect?: (volcanoId: string) => void;
  realTimeUpdates?: boolean;
  selectedVolcano?: string;
  timeRange?: {
    start: Date;
    end: Date;
  };
}

// Alert level configuration with geological context
const ALERT_LEVEL_CONFIG = {
  green: {
    label: 'Normal',
    color: 'success' as const,
    description: 'Background activity, no immediate threat',
    icon: '🟢',
    priority: 1
  },
  yellow: {
    label: 'Advisory',
    color: 'warning' as const,
    description: 'Elevated unrest, increased monitoring',
    icon: '🟡',
    priority: 2
  },
  orange: {
    label: 'Watch',
    color: 'warning' as const,
    description: 'Heightened unrest, eruption possible',
    icon: '🟠',
    priority: 3
  },
  red: {
    label: 'Warning',
    color: 'error' as const,
    description: 'Eruption imminent or in progress',
    icon: '🔴',
    priority: 4
  }
};

// Volcanic parameter thresholds
const PARAMETER_THRESHOLDS = {
  magmaChamberPressure: {
    normal: [0, 50],
    elevated: [50, 100],
    high: [100, 200],
    critical: [200, Infinity]
  },
  deformationRate: {
    normal: [0, 5],
    elevated: [5, 15],
    high: [15, 30],
    critical: [30, Infinity]
  },
  eruptionProbability: {
    low: [0, 0.1],
    moderate: [0.1, 0.3],
    high: [0.3, 0.7],
    critical: [0.7, 1.0]
  }
};

export const VolcanicMonitoringDashboard: React.FC<VolcanicMonitoringDashboardProps> = ({
  volcanicStates,
  onAlertGenerated,
  onVolcanoSelect,
  realTimeUpdates = false,
  selectedVolcano,
  timeRange = {
    start: subDays(new Date(), 7),
    end: new Date()
  }
}) => {
  const [autoRefresh, setAutoRefresh] = useState(realTimeUpdates);
  const [refreshInterval, _setRefreshInterval] = useState(30); // seconds
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'map'>('overview');
  const [_selectedParameter, _setSelectedParameter] = useState('eruptionProbability');
  const [_alertDialog, _setAlertDialog] = useState(false);
  const [_currentAlert, _setCurrentAlert] = useState<any>(null);

  // Filter volcanic states by time range and selection
  const filteredVolcanicStates = useMemo(() => {
    let filtered = volcanicStates.filter(state => 
      state.timestamp >= timeRange.start && state.timestamp <= timeRange.end
    );

    if (selectedVolcano) {
      filtered = filtered.filter(state => state.volcanoId === selectedVolcano);
    }

    return filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }, [volcanicStates, timeRange, selectedVolcano]);

  // Get latest state for each volcano
  const latestVolcanicStates = useMemo(() => {
    const stateMap = new Map<string, VolcanicState>();
    
    filteredVolcanicStates.forEach(state => {
      const existing = stateMap.get(state.volcanoId);
      if (!existing || state.timestamp > existing.timestamp) {
        stateMap.set(state.volcanoId, state);
      }
    });
    
    return Array.from(stateMap.values()).sort((a, b) => 
      ALERT_LEVEL_CONFIG[b.alertLevel].priority - ALERT_LEVEL_CONFIG[a.alertLevel].priority
    );
  }, [filteredVolcanicStates]);

  // Calculate dashboard statistics
  const dashboardStats = useMemo(() => {
    const stats = {
      totalVolcanoes: latestVolcanicStates.length,
      activeAlerts: latestVolcanicStates.filter(s => s.alertLevel !== 'green').length,
      criticalAlerts: latestVolcanicStates.filter(s => s.alertLevel === 'red').length,
      averageEruptionProbability: 0,
      highestRiskVolcano: null as VolcanicState | null,
      recentThermalAnomalies: 0,
      alertLevelDistribution: {
        green: 0,
        yellow: 0,
        orange: 0,
        red: 0
      }
    };

    if (latestVolcanicStates.length > 0) {
      stats.averageEruptionProbability = latestVolcanicStates.reduce(
        (sum, state) => sum + state.eruptionProbability, 0
      ) / latestVolcanicStates.length;

      stats.highestRiskVolcano = latestVolcanicStates.reduce((highest, current) => 
        current.eruptionProbability > (highest?.eruptionProbability || 0) ? current : highest
      );

      stats.recentThermalAnomalies = latestVolcanicStates.reduce(
        (sum, state) => sum + (state.thermalAnomalies?.length || 0), 0
      );

      latestVolcanicStates.forEach(state => {
        stats.alertLevelDistribution[state.alertLevel]++;
      });
    }

    return stats;
  }, [latestVolcanicStates]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(async () => {
      // Trigger real data refresh from volcanic monitoring networks
      try {
        const databaseService = DatabaseService.getInstance();
        const realDataService = RealDataService.getInstance();
        
        // Refresh volcanic systems data
        const updatedVolcanicSystems = await databaseService.queryVolcanicSystems({
          alertLevels: ['yellow', 'orange', 'red'],
          monitoringStatus: ['monitored', 'partially_monitored'],
        });
        
        // Refresh thermal anomalies from NASA FIRMS
        const thermalAnomalies = await realDataService.fetchThermalAnomalies({
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
        
        console.log(`Refreshed ${updatedVolcanicSystems.length} volcanic systems and ${thermalAnomalies.length} thermal anomalies`);
        
        // Update component state with new data
        // This would trigger re-render with fresh data
        
      } catch (error) {
        console.error('Failed to refresh volcanic monitoring data:', error);
      }
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  // Alert generation based on thresholds
  useEffect(() => {
    latestVolcanicStates.forEach(state => {
      // Check for critical conditions
      if (state.alertLevel === 'red' && state.eruptionProbability > 0.8) {
        const alert = {
          id: `volcanic-critical-${state.volcanoId}-${Date.now()}`,
          type: 'volcanic_activity',
          level: 'critical',
          title: `Critical Volcanic Activity - ${state.volcanoId}`,
          message: `Eruption probability: ${(state.eruptionProbability * 100).toFixed(1)}%`,
          timestamp: new Date(),
          source: state.volcanoId,
          metadata: { volcanoId: state.volcanoId, state }
        };
        
        onAlertGenerated?.(alert);
      }

      // Check for rapid deformation
      if (state.deformationRate > PARAMETER_THRESHOLDS.deformationRate.critical[0]) {
        const alert = {
          id: `deformation-alert-${state.volcanoId}-${Date.now()}`,
          type: 'volcanic_activity',
          level: 'warning',
          title: `Rapid Deformation Detected - ${state.volcanoId}`,
          message: `Deformation rate: ${state.deformationRate.toFixed(2)} mm/day`,
          timestamp: new Date(),
          source: state.volcanoId,
          metadata: { volcanoId: state.volcanoId, parameter: 'deformation', value: state.deformationRate }
        };
        
        onAlertGenerated?.(alert);
      }
    });
  }, [latestVolcanicStates, onAlertGenerated]);

  // Get parameter status color
  const getParameterStatus = useCallback((parameter: string, value: number) => {
    const thresholds = PARAMETER_THRESHOLDS[parameter as keyof typeof PARAMETER_THRESHOLDS];
    if (!thresholds) return 'default';

    if (value >= thresholds.critical[0]) return 'error';
    if (value >= thresholds.high[0]) return 'warning';
    // Handle different threshold structures
    if ('elevated' in thresholds && value >= (thresholds as any).elevated[0]) return 'info';
    if ('moderate' in thresholds && value >= (thresholds as any).moderate[0]) return 'info';
    return 'success';
  }, []);

  // Render volcano status card
  const renderVolcanoCard = useCallback((state: VolcanicState) => {
    const alertConfig = ALERT_LEVEL_CONFIG[state.alertLevel];
    
    return (
      <Card
        key={state.volcanoId}
        sx={{
          cursor: 'pointer',
          border: '2px solid',
          borderColor: selectedVolcano === state.volcanoId ? 'primary.main' : 'transparent',
          '&:hover': { borderColor: 'primary.light' }
        }}
        onClick={() => onVolcanoSelect?.(state.volcanoId)}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" noWrap>
              {state.volcanoId}
            </Typography>
            <Chip
              label={alertConfig.label}
              color={alertConfig.color}
              size="small"
              icon={<span>{alertConfig.icon}</span>}
            />
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2" color="textSecondary">
                Eruption Probability
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="h6">
                  {(state.eruptionProbability * 100).toFixed(1)}%
                </Typography>
                <Chip
                  size="small"
                  color={getParameterStatus('eruptionProbability', state.eruptionProbability)}
                  label={
                    state.eruptionProbability > 0.7 ? 'Critical' :
                    state.eruptionProbability > 0.3 ? 'High' :
                    state.eruptionProbability > 0.1 ? 'Moderate' : 'Low'
                  }
                />
              </Box>
            </Grid>

            <Grid item xs={6}>
              <Typography variant="body2" color="textSecondary">
                Deformation Rate
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="h6">
                  {state.deformationRate.toFixed(1)}
                </Typography>
                <Typography variant="caption">mm/day</Typography>
              </Box>
            </Grid>

            <Grid item xs={6}>
              <Typography variant="body2" color="textSecondary">
                Chamber Pressure
              </Typography>
              <Typography variant="body1">
                {state.magmaChamberPressure.toFixed(1)} MPa
              </Typography>
            </Grid>

            <Grid item xs={6}>
              <Typography variant="body2" color="textSecondary">
                Thermal Anomalies
              </Typography>
              <Typography variant="body1">
                {state.thermalAnomalies?.length || 0}
              </Typography>
            </Grid>
          </Grid>

          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="textSecondary">
              Last Updated: {format(state.timestamp, 'MMM dd, HH:mm')}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }, [selectedVolcano, onVolcanoSelect, getParameterStatus]);

  // Render gas composition display
  const renderGasComposition = useCallback((gasEmissions: GasComposition) => {
    return (
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Gas Emissions
        </Typography>
        <Grid container spacing={1}>
          <Grid item xs={3}>
            <Typography variant="caption">CO₂</Typography>
            <Typography variant="body2">{gasEmissions.co2} ppm</Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="caption">SO₂</Typography>
            <Typography variant="body2">{gasEmissions.so2} ppm</Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="caption">H₂S</Typography>
            <Typography variant="body2">{gasEmissions.h2s} ppm</Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="caption">H₂O</Typography>
            <Typography variant="body2">{gasEmissions.h2o} ppm</Typography>
          </Grid>
        </Grid>
        <Box sx={{ mt: 1 }}>
          <Typography variant="caption">
            Temperature: {gasEmissions.temperature}°C | 
            Flux: {gasEmissions.flux} t/day
          </Typography>
        </Box>
      </Box>
    );
  }, []);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', p: 2 }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5">Volcanic Monitoring Dashboard</Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>View Mode</InputLabel>
              <Select
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value as any)}
              >
                <MenuItem value="overview">Overview</MenuItem>
                <MenuItem value="detailed">Detailed</MenuItem>
                <MenuItem value="map">Map View</MenuItem>
              </Select>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
              }
              label="Auto Refresh"
            />
            
            <Tooltip title="Refresh Data">
              <IconButton>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Statistics */}
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {dashboardStats.totalVolcanoes}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Volcanoes
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {dashboardStats.activeAlerts}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Active Alerts
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="error.main">
                {dashboardStats.criticalAlerts}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Critical Alerts
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main">
                {(dashboardStats.averageEruptionProbability * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Avg. Eruption Risk
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Alert Level Distribution */}
        <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {Object.entries(dashboardStats.alertLevelDistribution).map(([level, count]) => {
            const config = ALERT_LEVEL_CONFIG[level as AlertLevel];
            return (
              <Chip
                key={level}
                label={`${config.label}: ${count}`}
                color={config.color}
                size="small"
                icon={<span>{config.icon}</span>}
              />
            );
          })}
        </Box>
      </Paper>

      {/* Main Content */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {viewMode === 'overview' && (
          <Grid container spacing={2}>
            {latestVolcanicStates.map(renderVolcanoCard)}
          </Grid>
        )}

        {viewMode === 'detailed' && selectedVolcano && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2, height: 400 }}>
                <Typography variant="h6" gutterBottom>
                  Volcanic Parameters - {selectedVolcano}
                </Typography>
                <ScientificCharts
                  data={filteredVolcanicStates.filter(s => s.volcanoId === selectedVolcano) as any}
                  chartType="timeseries"
                  realTimeUpdate={autoRefresh}
                />
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: 400 }}>
                <Typography variant="h6" gutterBottom>
                  Current Status
                </Typography>
                {(() => {
                  const currentState = latestVolcanicStates.find(s => s.volcanoId === selectedVolcano);
                  if (!currentState) return <Typography>No data available</Typography>;
                  
                  return (
                    <Box>
                      <Box sx={{ mb: 2 }}>
                        <Chip
                          label={ALERT_LEVEL_CONFIG[currentState.alertLevel].label}
                          color={ALERT_LEVEL_CONFIG[currentState.alertLevel].color}
                          icon={<span>{ALERT_LEVEL_CONFIG[currentState.alertLevel].icon}</span>}
                        />
                      </Box>
                      
                      {renderGasComposition(currentState.gasEmissions)}
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Typography variant="subtitle2" gutterBottom>
                        Thermal Anomalies
                      </Typography>
                      <List dense>
                        {currentState.thermalAnomalies?.slice(0, 3).map((anomaly, index) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <Thermostat color="error" />
                            </ListItemIcon>
                            <ListItemText
                              primary={`${anomaly.temperature}°C`}
                              secondary={`Area: ${anomaly.area} m²`}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  );
                })()}
              </Paper>
            </Grid>
          </Grid>
        )}

        {viewMode === 'map' && (
          <Paper sx={{ height: '100%' }}>
            <GeologicalMap
              data={[]}
              volcanicStates={latestVolcanicStates}
              onFeatureClick={(feature) => {
                if (feature.properties?.volcanoId) {
                  onVolcanoSelect?.(feature.properties.volcanoId);
                }
              }}
              showRealTimeData={autoRefresh}
            />
          </Paper>
        )}
      </Box>

      {/* High Risk Alert */}
      {dashboardStats.highestRiskVolcano && dashboardStats.highestRiskVolcano.eruptionProbability > 0.7 && (
        <Alert
          severity="error"
          sx={{ mt: 2 }}
          action={
            <Button
              color="inherit"
              size="small"
              onClick={() => onVolcanoSelect?.(dashboardStats.highestRiskVolcano!.volcanoId)}
            >
              VIEW DETAILS
            </Button>
          }
        >
          High eruption risk detected at {dashboardStats.highestRiskVolcano.volcanoId} - 
          {(dashboardStats.highestRiskVolcano.eruptionProbability * 100).toFixed(1)}% probability
        </Alert>
      )}
    </Box>
  );
};

export default VolcanicMonitoringDashboard;