import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Switch,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  TextField,
  Button,
  // Divider - unused
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Chip,
  Alert,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  // List, ListItem, ListItemText, ListItemSecondaryAction, IconButton - unused
  // Tooltip - unused
} from '@mui/material';
import {
  ExpandMore,
  Save,
  Restore,
  Download,
  Upload,
  // Delete - unused
  Palette,
  Language,
  Accessibility,
  Speed,
  Notifications,
  // Security - unused
  Storage,
  // Sync - unused
} from '@mui/icons-material';
import { UserPreferences } from '../../types/dashboard'; // NotificationSettings, AccessibilitySettings, PerformanceSettings - unused

interface UserPreferencesProps {
  preferences: UserPreferences;
  onPreferencesChange: (preferences: UserPreferences) => void;
  onExport?: () => void;
  onImport?: (preferences: UserPreferences) => void;
  onReset?: () => void;
}

// Available themes
const THEMES = [
  { id: 'light', name: 'Light', description: 'Light theme for daytime use' },
  { id: 'dark', name: 'Dark', description: 'Dark theme for low-light environments' },
  { id: 'auto', name: 'Auto', description: 'Automatically switch based on system preference' }
];

// Available languages
const LANGUAGES = [
  { id: 'en', name: 'English', nativeName: 'English' },
  { id: 'es', name: 'Spanish', nativeName: 'Español' },
  { id: 'fr', name: 'French', nativeName: 'Français' },
  { id: 'de', name: 'German', nativeName: 'Deutsch' },
  { id: 'it', name: 'Italian', nativeName: 'Italiano' },
  { id: 'pt', name: 'Portuguese', nativeName: 'Português' },
  { id: 'ja', name: 'Japanese', nativeName: '日本語' },
  { id: 'zh', name: 'Chinese', nativeName: '中文' }
];

// Available timezones (subset)
const TIMEZONES = [
  'UTC',
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'Europe/London',
  'Europe/Paris',
  'Europe/Berlin',
  'Asia/Tokyo',
  'Asia/Shanghai',
  'Australia/Sydney'
];

// Date format options
const DATE_FORMATS = [
  { id: 'MM/dd/yyyy', name: 'MM/DD/YYYY', example: '12/31/2023' },
  { id: 'dd/MM/yyyy', name: 'DD/MM/YYYY', example: '31/12/2023' },
  { id: 'yyyy-MM-dd', name: 'YYYY-MM-DD', example: '2023-12-31' },
  { id: 'MMM dd, yyyy', name: 'MMM DD, YYYY', example: 'Dec 31, 2023' },
  { id: 'dd MMM yyyy', name: 'DD MMM YYYY', example: '31 Dec 2023' }
];

// Number format options
const NUMBER_FORMATS = [
  { id: 'en-US', name: 'US (1,234.56)', example: '1,234.56' },
  { id: 'en-GB', name: 'UK (1,234.56)', example: '1,234.56' },
  { id: 'de-DE', name: 'German (1.234,56)', example: '1.234,56' },
  { id: 'fr-FR', name: 'French (1 234,56)', example: '1 234,56' },
  { id: 'es-ES', name: 'Spanish (1.234,56)', example: '1.234,56' }
];

export const UserPreferencesComponent: React.FC<UserPreferencesProps> = ({
  preferences,
  onPreferencesChange,
  onExport: _onExport,
  onImport,
  onReset
}) => {
  const [localPreferences, setLocalPreferences] = useState<UserPreferences>(preferences);
  const [hasChanges, setHasChanges] = useState(false);
  const [notification, setNotification] = useState<{ message: string; severity: 'success' | 'error' | 'info' } | null>(null);
  const [resetDialog, setResetDialog] = useState(false);
  const [importDialog, setImportDialog] = useState(false);
  const [importData, setImportData] = useState('');

  // Update local preferences when props change
  useEffect(() => {
    setLocalPreferences(preferences);
    setHasChanges(false);
  }, [preferences]);

  // Handle preference changes
  const handlePreferenceChange = useCallback((path: string, value: any) => {
    setLocalPreferences(prev => {
      const newPrefs = { ...prev };
      const keys = path.split('.');
      let current: any = newPrefs;
      
      for (let i = 0; i < keys.length - 1; i++) {
        if (!(keys[i] in current)) {
          current[keys[i]] = {};
        }
        current = current[keys[i]];
      }
      
      current[keys[keys.length - 1]] = value;
      return newPrefs;
    });
    setHasChanges(true);
  }, []);

  // Save preferences
  const handleSave = useCallback(() => {
    onPreferencesChange(localPreferences);
    setHasChanges(false);
    setNotification({ message: 'Preferences saved successfully', severity: 'success' });
  }, [localPreferences, onPreferencesChange]);

  // Reset preferences
  const handleReset = useCallback(() => {
    onReset?.();
    setResetDialog(false);
    setNotification({ message: 'Preferences reset to defaults', severity: 'info' });
  }, [onReset]);

  // Export preferences
  const handleExport = useCallback(() => {
    const dataStr = JSON.stringify(localPreferences, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `prometheus-preferences-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    setNotification({ message: 'Preferences exported successfully', severity: 'success' });
  }, [localPreferences]);

  // Import preferences
  const handleImport = useCallback(() => {
    try {
      const importedPrefs = JSON.parse(importData);
      onImport?.(importedPrefs);
      setImportDialog(false);
      setImportData('');
      setNotification({ message: 'Preferences imported successfully', severity: 'success' });
    } catch (error) {
      setNotification({ message: 'Invalid preferences file', severity: 'error' });
    }
  }, [importData, onImport]);

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5">User Preferences</Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleExport}
            >
              Export
            </Button>
            <Button
              variant="outlined"
              startIcon={<Upload />}
              onClick={() => setImportDialog(true)}
            >
              Import
            </Button>
            <Button
              variant="outlined"
              startIcon={<Restore />}
              onClick={() => setResetDialog(true)}
            >
              Reset
            </Button>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={handleSave}
              disabled={!hasChanges}
            >
              Save Changes
            </Button>
          </Box>
        </Box>
        
        {hasChanges && (
          <Alert severity="info">
            You have unsaved changes. Click "Save Changes" to apply them.
          </Alert>
        )}
      </Paper>

      {/* Appearance Settings */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Palette />
            <Typography variant="h6">Appearance</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Theme</InputLabel>
                <Select
                  value={localPreferences.theme}
                  onChange={(e) => handlePreferenceChange('theme', e.target.value)}
                >
                  {THEMES.map(theme => (
                    <MenuItem key={theme.id} value={theme.id}>
                      <Box>
                        <Typography>{theme.name}</Typography>
                        <Typography variant="caption" color="textSecondary">
                          {theme.description}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Localization Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Language />
            <Typography variant="h6">Localization</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select
                  value={localPreferences.language}
                  onChange={(e) => handlePreferenceChange('language', e.target.value)}
                >
                  {LANGUAGES.map(lang => (
                    <MenuItem key={lang.id} value={lang.id}>
                      {lang.name} ({lang.nativeName})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Timezone</InputLabel>
                <Select
                  value={localPreferences.timezone}
                  onChange={(e) => handlePreferenceChange('timezone', e.target.value)}
                >
                  {TIMEZONES.map(tz => (
                    <MenuItem key={tz} value={tz}>
                      {tz}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Date Format</InputLabel>
                <Select
                  value={localPreferences.dateFormat}
                  onChange={(e) => handlePreferenceChange('dateFormat', e.target.value)}
                >
                  {DATE_FORMATS.map(format => (
                    <MenuItem key={format.id} value={format.id}>
                      <Box>
                        <Typography>{format.name}</Typography>
                        <Typography variant="caption" color="textSecondary">
                          Example: {format.example}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Number Format</InputLabel>
                <Select
                  value={localPreferences.numberFormat}
                  onChange={(e) => handlePreferenceChange('numberFormat', e.target.value)}
                >
                  {NUMBER_FORMATS.map(format => (
                    <MenuItem key={format.id} value={format.id}>
                      <Box>
                        <Typography>{format.name}</Typography>
                        <Typography variant="caption" color="textSecondary">
                          Example: {format.example}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Notifications Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Notifications />
            <Typography variant="h6">Notifications</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.notifications.enabled}
                    onChange={(e) => handlePreferenceChange('notifications.enabled', e.target.checked)}
                  />
                }
                label="Enable Notifications"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.notifications.sound}
                    onChange={(e) => handlePreferenceChange('notifications.sound', e.target.checked)}
                  />
                }
                label="Sound Notifications"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.notifications.desktop}
                    onChange={(e) => handlePreferenceChange('notifications.desktop', e.target.checked)}
                  />
                }
                label="Desktop Notifications"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.notifications.email}
                    onChange={(e) => handlePreferenceChange('notifications.email', e.target.checked)}
                  />
                }
                label="Email Notifications"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography gutterBottom>Alert Levels</Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {['info', 'warning', 'error', 'critical'].map(level => (
                  <Chip
                    key={level}
                    label={level.toUpperCase()}
                    clickable
                    color={localPreferences.notifications.alertLevels.includes(level as any) ? 'primary' : 'default'}
                    onClick={() => {
                      const currentLevels = localPreferences.notifications.alertLevels;
                      const newLevels = currentLevels.includes(level as any)
                        ? currentLevels.filter(l => l !== level)
                        : [...currentLevels, level as any];
                      handlePreferenceChange('notifications.alertLevels', newLevels);
                    }}
                  />
                ))}
              </Box>
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.notifications.quietHours?.enabled || false}
                    onChange={(e) => handlePreferenceChange('notifications.quietHours.enabled', e.target.checked)}
                  />
                }
                label="Enable Quiet Hours"
              />
            </Grid>
            
            {localPreferences.notifications.quietHours?.enabled && (
              <>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Quiet Hours Start"
                    type="time"
                    value={localPreferences.notifications.quietHours?.start || '22:00'}
                    onChange={(e) => handlePreferenceChange('notifications.quietHours.start', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Quiet Hours End"
                    type="time"
                    value={localPreferences.notifications.quietHours?.end || '08:00'}
                    onChange={(e) => handlePreferenceChange('notifications.quietHours.end', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
              </>
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Accessibility Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Accessibility />
            <Typography variant="h6">Accessibility</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.accessibility.highContrast}
                    onChange={(e) => handlePreferenceChange('accessibility.highContrast', e.target.checked)}
                  />
                }
                label="High Contrast Mode"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.accessibility.largeText}
                    onChange={(e) => handlePreferenceChange('accessibility.largeText', e.target.checked)}
                  />
                }
                label="Large Text"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.accessibility.reducedMotion}
                    onChange={(e) => handlePreferenceChange('accessibility.reducedMotion', e.target.checked)}
                  />
                }
                label="Reduced Motion"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.accessibility.screenReader}
                    onChange={(e) => handlePreferenceChange('accessibility.screenReader', e.target.checked)}
                  />
                }
                label="Screen Reader Support"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.accessibility.keyboardNavigation}
                    onChange={(e) => handlePreferenceChange('accessibility.keyboardNavigation', e.target.checked)}
                  />
                }
                label="Enhanced Keyboard Navigation"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Performance Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Speed />
            <Typography variant="h6">Performance</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.performance.enableVirtualization}
                    onChange={(e) => handlePreferenceChange('performance.enableVirtualization', e.target.checked)}
                  />
                }
                label="Enable Virtualization"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.performance.enableWebGL}
                    onChange={(e) => handlePreferenceChange('performance.enableWebGL', e.target.checked)}
                  />
                }
                label="Enable WebGL"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.performance.enableWorkers}
                    onChange={(e) => handlePreferenceChange('performance.enableWorkers', e.target.checked)}
                  />
                }
                label="Enable Web Workers"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography gutterBottom>
                Max Data Points: {localPreferences.performance.maxDataPoints.toLocaleString()}
              </Typography>
              <Slider
                value={localPreferences.performance.maxDataPoints}
                onChange={(_, value) => handlePreferenceChange('performance.maxDataPoints', value)}
                min={100}
                max={100000}
                step={100}
                marks={[
                  { value: 1000, label: '1K' },
                  { value: 10000, label: '10K' },
                  { value: 50000, label: '50K' },
                  { value: 100000, label: '100K' }
                ]}
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Render Quality</InputLabel>
                <Select
                  value={localPreferences.performance.renderQuality}
                  onChange={(e) => handlePreferenceChange('performance.renderQuality', e.target.value)}
                >
                  <MenuItem value="low">Low (Better Performance)</MenuItem>
                  <MenuItem value="medium">Medium (Balanced)</MenuItem>
                  <MenuItem value="high">High (Better Quality)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Typography gutterBottom>
                Cache Size: {localPreferences.performance.cacheSize} MB
              </Typography>
              <Slider
                value={localPreferences.performance.cacheSize}
                onChange={(_, value) => handlePreferenceChange('performance.cacheSize', value)}
                min={10}
                max={1000}
                step={10}
                marks={[
                  { value: 50, label: '50MB' },
                  { value: 200, label: '200MB' },
                  { value: 500, label: '500MB' },
                  { value: 1000, label: '1GB' }
                ]}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* General Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Storage />
            <Typography variant="h6">General</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localPreferences.autoSave}
                    onChange={(e) => handlePreferenceChange('autoSave', e.target.checked)}
                  />
                }
                label="Auto-save Changes"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Default Dashboard"
                value={localPreferences.defaultDashboard || ''}
                onChange={(e) => handlePreferenceChange('defaultDashboard', e.target.value)}
                placeholder="Enter dashboard ID"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Reset Confirmation Dialog */}
      <Dialog open={resetDialog} onClose={() => setResetDialog(false)}>
        <DialogTitle>Reset Preferences</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to reset all preferences to their default values? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialog(false)}>Cancel</Button>
          <Button onClick={handleReset} color="error" variant="contained">
            Reset
          </Button>
        </DialogActions>
      </Dialog>

      {/* Import Dialog */}
      <Dialog open={importDialog} onClose={() => setImportDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Import Preferences</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Paste your exported preferences JSON data below:
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={10}
            value={importData}
            onChange={(e) => setImportData(e.target.value)}
            placeholder="Paste JSON data here..."
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImportDialog(false)}>Cancel</Button>
          <Button onClick={handleImport} variant="contained" disabled={!importData.trim()}>
            Import
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notification Snackbar */}
      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={3000}
        onClose={() => setNotification(null)}
      >
        {notification && (
          <Alert severity={notification.severity} onClose={() => setNotification(null)}>
            {notification.message}
          </Alert>
        )}
      </Snackbar>
    </Box>
  );
};

export default UserPreferencesComponent;