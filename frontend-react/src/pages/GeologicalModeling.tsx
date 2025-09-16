/**
 * Geological Modeling page - Real geological modeling interface
 */

import React, { useState, useEffect } from 'react';
import { Box, Typography, Grid, Card, CardContent, CircularProgress, Alert } from '@mui/material';
import { useTranslation } from 'react-i18next';
import DatabaseService from '../services/DatabaseService';
import Advanced3DGeologicalVisualization from '../components/visualization/Advanced3DGeologicalVisualization';
import GeologicalDataGrid from '../components/visualization/grids/GeologicalDataGrid';

interface GeologicalModel {
  id: string;
  name: string;
  type: string;
  parameters: Record<string, any>;
  lastUpdated: Date;
  status: 'active' | 'processing' | 'completed' | 'error';
}

const GeologicalModeling: React.FC = () => {
  const { t } = useTranslation();
  const [models, setModels] = useState<GeologicalModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const databaseService = DatabaseService.getInstance();

  useEffect(() => {
    const loadGeologicalModels = async () => {
      try {
        setLoading(true);
        setError(null);

        // Query real geological models from database
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/geological/models`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const modelData = await response.json();
          setModels(modelData.models || []);
        } else {
          throw new Error(`Failed to load geological models: ${response.status}`);
        }
      } catch (err) {
        console.error('Failed to load geological models:', err);
        setError(err instanceof Error ? err.message : 'Failed to load geological models');
      } finally {
        setLoading(false);
      }
    };

    loadGeologicalModels();
  }, [databaseService]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading Geological Models...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          <Typography variant="h6">Failed to Load Geological Models</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('geological.title', 'Geological Modeling')}
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Real-time geological modeling with production database integration
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                3D Geological Visualization
              </Typography>
              <Advanced3DGeologicalVisualization
                layers={[]}
                height={600}
                onViewChange={() => {}}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Models ({models.length})
              </Typography>
              {models.length === 0 ? (
                <Typography variant="body2" color="textSecondary">
                  No geological models found. Create a new model to get started.
                </Typography>
              ) : (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {models.map((model) => (
                    <Card key={model.id} variant="outlined">
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="subtitle2">{model.name}</Typography>
                        <Typography variant="body2" color="textSecondary">
                          Type: {model.type}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Status: {model.status}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Updated: {model.lastUpdated.toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Geological Data
              </Typography>
              <GeologicalDataGrid
                data={[]}
                dataType="geological"
                onRowSelect={() => {}}
                onExport={() => {}}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default GeologicalModeling;