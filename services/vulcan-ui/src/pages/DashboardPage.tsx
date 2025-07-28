import React, { useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  Warning as WarningIcon,
  Science as ScienceIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '@/store';
import { fetchSeismicEvents } from '@/store/slices/geologicalDataSlice';
import { useNavigate } from 'react-router-dom';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  
  const { user } = useAppSelector((state) => state.auth);
  const { seismicEvents, riskAssessments, activeSimulation } = useAppSelector(
    (state) => state.geologicalData
  );
  const { notifications } = useAppSelector((state) => state.ui);

  useEffect(() => {
    // Load recent data for dashboard
    dispatch(fetchSeismicEvents({ 
      timeRange: [
        new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), // Last 24 hours
        new Date().toISOString()
      ]
    }));
  }, [dispatch]);

  const recentAlerts = notifications.filter(n => n.type === 'warning' || n.type === 'error');
  const recentSeismicEvents = seismicEvents.slice(0, 5);
  const activeRiskAssessments = riskAssessments.filter(r => r.alertLevel.level > 2);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Welcome Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Welcome back, {user?.firstName}
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Prometheus AI Geodynamic Platform Dashboard
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* System Status Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TimelineIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Real-time Monitoring</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                {seismicEvents.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Events in last 24h
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" onClick={() => navigate('/monitoring')}>
                View Details
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <WarningIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">Active Alerts</Typography>
              </Box>
              <Typography variant="h4" color="warning.main">
                {recentAlerts.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Requiring attention
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" onClick={() => navigate('/monitoring/alerts')}>
                View Alerts
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <ScienceIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Active Simulations</Typography>
              </Box>
              <Typography variant="h4" color="success.main">
                {activeSimulation.isRunning ? 1 : 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Currently running
              </Typography>
              {activeSimulation.isRunning && (
                <LinearProgress 
                  variant="determinate" 
                  value={activeSimulation.progress} 
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
            <CardActions>
              <Button size="small" onClick={() => navigate('/simulations')}>
                View Simulations
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Risk Assessments</Typography>
              </Box>
              <Typography variant="h4" color="info.main">
                {activeRiskAssessments.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                High risk areas
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" onClick={() => navigate('/simulations/risk')}>
                View Assessments
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Seismic Activity
            </Typography>
            {recentSeismicEvents.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                No recent seismic events detected.
              </Typography>
            ) : (
              <Box>
                {recentSeismicEvents.map((event) => (
                  <Box
                    key={event.id}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      py: 1,
                      borderBottom: 1,
                      borderColor: 'divider',
                      '&:last-child': { borderBottom: 0 },
                    }}
                  >
                    <Box>
                      <Typography variant="subtitle2">
                        Magnitude {event.magnitude} {event.eventType}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {event.location.latitude.toFixed(3)}°N, {event.location.longitude.toFixed(3)}°W
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Chip
                        label={`${event.depth}km deep`}
                        size="small"
                        variant="outlined"
                      />
                      <Typography variant="caption" display="block" color="text.secondary">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<ScienceIcon />}
                onClick={() => navigate('/simulations/helios')}
              >
                Start New Simulation
              </Button>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<TimelineIcon />}
                onClick={() => navigate('/workflows/builder')}
              >
                Create Workflow
              </Button>
              <Button
                variant="outlined"
                fullWidth
                onClick={() => navigate('/data/import')}
              >
                Import Data
              </Button>
              <Button
                variant="outlined"
                fullWidth
                onClick={() => navigate('/visualization/maps')}
              >
                View 3D Visualization
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* System Health */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Health & Performance
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Data Ingestion
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={95}
                      sx={{ flexGrow: 1, mr: 2 }}
                    />
                    <Typography variant="body2">95%</Typography>
                  </Box>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    AI Model Performance
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={88}
                      sx={{ flexGrow: 1, mr: 2 }}
                    />
                    <Typography variant="body2">88%</Typography>
                  </Box>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    System Resources
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={72}
                      sx={{ flexGrow: 1, mr: 2 }}
                    />
                    <Typography variant="body2">72%</Typography>
                  </Box>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};