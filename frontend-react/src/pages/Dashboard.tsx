/**
 * Main dashboard page component
 */

import React from 'react';
import { Box, Typography, Grid, Card, CardContent } from '@mui/material';
import { useTranslation } from 'react-i18next';

const Dashboard: React.FC = () => {
  const { t } = useTranslation();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('dashboard.title')}
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {t('seismic.events')}
              </Typography>
              <Typography variant="h3" color="primary">
                42
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last 24 hours
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {t('volcanic.volcanoes')}
              </Typography>
              <Typography variant="h3" color="secondary">
                8
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Active monitoring
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {t('resources.prospects')}
              </Typography>
              <Typography variant="h3" color="success.main">
                156
              </Typography>
              <Typography variant="body2" color="textSecondary">
                High confidence
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                {t('monitoring.alerts')}
              </Typography>
              <Typography variant="h3" color="warning.main">
                3
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Requires attention
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;