/**
 * Settings page - Real application settings with database integration
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  Switch, 
  FormControlLabel, 
  TextField, 
  Button,
  Alert,
  Divider
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import DatabaseService from '../services/DatabaseService';

interface UserSettings {
  id: string;
  userId: string;
  realTimeUpdates: boolean;
  dataQualityThreshold: number;
  notificationPreferences: {
    email: boolean;
    push: boolean;
    sms: boolean;
  };
  visualizationSettings: {
    defaultColorScheme: string;
    animationsEnabled: boolean;
    highContrastMode: boolean;
  };
  databaseSettings: {
    cacheTimeout: number;
    maxQueryResults: number;
    autoRefreshInterval: number;
  };
}

const Settings: React.FC = () => {
  const { t } = useTranslation();
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const databaseService = DatabaseService.getInstance();

  useEffect(() => {
    const loadSettings = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/user/settings`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const userSettings = await response.json();
          setSettings(userSettings);
        } else {
          throw new Error('Failed to load user settings');
        }
      } catch (error) {
        console.error('Error loading settings:', error);
        setMessage({ type: 'error', text: 'Failed to load settings' });
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, [databaseService]);

  const handleSaveSettings = async () => {
    if (!settings) return;

    try {
      setSaving(true);
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/user/settings`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });

      if (response.ok) {
        setMessage({ type: 'success', text: 'Settings saved successfully' });
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (error) {
      console.error('Error saving settings:', error);
      setMessage({ type: 'error', text: 'Failed to save settings' });
    } finally {
      setSaving(false);
    }
  };

  const updateSettings = (path: string, value: any) => {
    if (!settings) return;

    const keys = path.split('.');
    const newSettings = { ...settings };
    let current: any = newSettings;

    for (let i = 0; i < keys.length - 1; i++) {
      current[keys[i]] = { ...current[keys[i]] };
      current = current[keys[i]];
    }

    current[keys[keys.length - 1]] = value;
    setSettings(newSettings);
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {t('settings.title', 'Settings')}
        </Typography>
        <Typography>Loading settings...</Typography>
      </Box>
    );
  }

  if (!settings) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">Failed to load user settings</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('settings.title', 'Settings')}
      </Typography>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        Configure your application preferences and database connections
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 3 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Real-Time Data Settings
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.realTimeUpdates}
                    onChange={(e) => updateSettings('realTimeUpdates', e.target.checked)}
                  />
                }
                label="Enable real-time data updates"
              />
              <TextField
                fullWidth
                label="Data Quality Threshold"
                type="number"
                value={settings.dataQualityThreshold}
                onChange={(e) => updateSettings('dataQualityThreshold', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                sx={{ mt: 2 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Database Settings
              </Typography>
              <TextField
                fullWidth
                label="Cache Timeout (seconds)"
                type="number"
                value={settings.databaseSettings.cacheTimeout}
                onChange={(e) => updateSettings('databaseSettings.cacheTimeout', parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Max Query Results"
                type="number"
                value={settings.databaseSettings.maxQueryResults}
                onChange={(e) => updateSettings('databaseSettings.maxQueryResults', parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Auto Refresh Interval (ms)"
                type="number"
                value={settings.databaseSettings.autoRefreshInterval}
                onChange={(e) => updateSettings('databaseSettings.autoRefreshInterval', parseInt(e.target.value))}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Visualization Settings
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.visualizationSettings.animationsEnabled}
                        onChange={(e) => updateSettings('visualizationSettings.animationsEnabled', e.target.checked)}
                      />
                    }
                    label="Enable animations"
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.visualizationSettings.highContrastMode}
                        onChange={(e) => updateSettings('visualizationSettings.highContrastMode', e.target.checked)}
                      />
                    }
                    label="High contrast mode"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Divider sx={{ my: 2 }} />
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              onClick={handleSaveSettings}
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;