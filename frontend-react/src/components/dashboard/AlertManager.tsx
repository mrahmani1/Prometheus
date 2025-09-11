import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Badge,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  Snackbar,
  Alert as MuiAlert,
  Divider,
  Menu,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Warning,
  Error,
  Info,
  CheckCircle,
  ExpandMore,
  Notifications,
  NotificationsOff,
  FilterList,
  Sort,
  MoreVert,
  Check,
  Close,
  Refresh,
  Settings,
  VolumeUp,
  VolumeOff,
  Schedule,
  Person,
  LocationOn,
  Timeline
} from '@mui/icons-material';
import { format, formatDistanceToNow } from 'date-fns';
import { DashboardAlert, AlertLevel, AlertType, NotificationSettings } from '../../types/dashboard';
import { GeologicalData, SeismicEvent, VolcanicState } from '../../types/geological';

interface AlertManagerProps {
  alerts: DashboardAlert[];
  onAlertAcknowledge?: (alertId: string) => void;
  onAlertResolve?: (alertId: string) => void;
  onAlertDismiss?: (alertId: string) => void;
  onSettingsChange?: (settings: NotificationSettings) => void;
  notificationSettings?: NotificationSettings;
  realTimeData?: {
    geological: GeologicalData[];
    seismic: SeismicEvent[];
    volcanic: VolcanicState[];
  };
  maxAlerts?: number;
  autoRefresh?: boolean;
}

// Alert severity configuration
const ALERT_SEVERITY_CONFIG = {
  info: {
    icon: Info,
    color: 'info' as const,
    priority: 1,
    sound: 'notification.mp3',
    autoAcknowledge: true,
    autoAcknowledgeDelay: 300000 // 5 minutes
  },
  warning: {
    icon: Warning,
    color: 'warning' as const,
    priority: 2,
    sound: 'warning.mp3',
    autoAcknowledge: false,
    autoAcknowledgeDelay: 0
  },
  error: {
    icon: Error,
    color: 'error' as const,
    priority: 3,
    sound: 'error.mp3',
    autoAcknowledge: false,
    autoAcknowledgeDelay: 0
  },
  critical: {
    icon: Error,
    color: 'error' as const,
    priority: 4,
    sound: 'critical.mp3',
    autoAcknowledge: false,
    autoAcknowledgeDelay: 0
  }
};

// Alert type configuration
const ALERT_TYPE_CONFIG = {
  seismic_event: {
    label: 'Seismic Event',
    description: 'Earthquake or seismic activity detected',
    category: 'geological'
  },
  volcanic_activity: {
    label: 'Volcanic Activity',
    description: 'Volcanic eruption or increased activity',
    category: 'geological'
  },
  system_error: {
    label: 'System Error',
    description: 'System malfunction or error',
    category: 'system'
  },
  data_quality: {
    label: 'Data Quality',
    description: 'Data quality issues detected',
    category: 'data'
  },
  connection_lost: {
    label: 'Connection Lost',
    description: 'Network or sensor connection lost',
    category: 'system'
  },
  threshold_exceeded: {
    label: 'Threshold Exceeded',
    description: 'Monitoring threshold exceeded',
    category: 'monitoring'
  },
  model_update: {
    label: 'Model Update',
    description: 'AI model update or retrain completed',
    category: 'system'
  },
  maintenance: {
    label: 'Maintenance',
    description: 'Scheduled maintenance notification',
    category: 'system'
  },
  security: {
    label: 'Security',
    description: 'Security-related alert',
    category: 'security'
  },
  custom: {
    label: 'Custom',
    description: 'Custom alert type',
    category: 'custom'
  }
};

export const AlertManager: React.FC<AlertManagerProps> = ({
  alerts,
  onAlertAcknowledge,
  onAlertResolve,
  onAlertDismiss,
  onSettingsChange,
  notificationSettings,
  realTimeData,
  maxAlerts = 100,
  autoRefresh = true
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [filterLevel, setFilterLevel] = useState<AlertLevel | 'all'>('all');
  const [filterType, setFilterType] = useState<AlertType | 'all'>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'level' | 'type'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [selectedAlert, setSelectedAlert] = useState<DashboardAlert | null>(null);
  const [settingsDialog, setSettingsDialog] = useState(false);
  const [filterDialog, setFilterDialog] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(notificationSettings?.sound ?? true);
  const [notification, setNotification] = useState<{ message: string; severity: 'success' | 'error' | 'info' } | null>(null);

  // Audio context for alert sounds
  const audioContext = useMemo(() => {
    if (typeof window !== 'undefined' && 'AudioContext' in window) {
      return new AudioContext();
    }
    return null;
  }, []);

  // Filter and sort alerts
  const filteredAlerts = useMemo(() => {
    let filtered = alerts;

    // Filter by level
    if (filterLevel !== 'all') {
      filtered = filtered.filter(alert => alert.level === filterLevel);
    }

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(alert => alert.type === filterType);
    }

    // Filter by tab (active/acknowledged/resolved)
    switch (currentTab) {
      case 0: // Active
        filtered = filtered.filter(alert => !alert.acknowledged && !alert.resolved);
        break;
      case 1: // Acknowledged
        filtered = filtered.filter(alert => alert.acknowledged && !alert.resolved);
        break;
      case 2: // Resolved
        filtered = filtered.filter(alert => alert.resolved);
        break;
      case 3: // All
        break;
    }

    // Sort alerts
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'timestamp':
          comparison = a.timestamp.getTime() - b.timestamp.getTime();
          break;
        case 'level':
          comparison = ALERT_SEVERITY_CONFIG[a.level].priority - ALERT_SEVERITY_CONFIG[b.level].priority;
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
      }

      return sortOrder === 'desc' ? -comparison : comparison;
    });

    return filtered.slice(0, maxAlerts);
  }, [alerts, filterLevel, filterType, currentTab, sortBy, sortOrder, maxAlerts]);

  // Alert statistics
  const alertStats = useMemo(() => {
    const stats = {
      total: alerts.length,
      active: alerts.filter(a => !a.acknowledged && !a.resolved).length,
      acknowledged: alerts.filter(a => a.acknowledged && !a.resolved).length,
      resolved: alerts.filter(a => a.resolved).length,
      byLevel: {
        info: alerts.filter(a => a.level === 'info').length,
        warning: alerts.filter(a => a.level === 'warning').length,
        error: alerts.filter(a => a.level === 'error').length,
        critical: alerts.filter(a => a.level === 'critical').length
      },
      byType: {} as Record<AlertType, number>
    };

    // Count by type
    Object.keys(ALERT_TYPE_CONFIG).forEach(type => {
      stats.byType[type as AlertType] = alerts.filter(a => a.type === type).length;
    });

    return stats;
  }, [alerts]);

  // Play alert sound
  const playAlertSound = useCallback((level: AlertLevel) => {
    if (!soundEnabled || !audioContext || !notificationSettings?.sound) return;

    const config = ALERT_SEVERITY_CONFIG[level];
    if (config.sound) {
      // In a real implementation, you would load and play the actual sound file
      // For now, we'll use a simple beep
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      oscillator.frequency.setValueAtTime(level === 'critical' ? 800 : level === 'error' ? 600 : 400, audioContext.currentTime);
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      
      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.2);
    }
  }, [soundEnabled, audioContext, notificationSettings?.sound]);

  // Handle new alerts
  useEffect(() => {
    if (alerts.length === 0) return;

    const latestAlert = alerts[0];
    const isNewAlert = Date.now() - latestAlert.timestamp.getTime() < 5000; // 5 seconds

    if (isNewAlert && !latestAlert.acknowledged) {
      playAlertSound(latestAlert.level);
      
      // Show desktop notification if enabled
      if (notificationSettings?.desktop && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(latestAlert.title, {
          body: latestAlert.message,
          icon: '/favicon.ico',
          tag: latestAlert.id
        });
      }
    }
  }, [alerts, playAlertSound, notificationSettings?.desktop]);

  // Auto-acknowledge alerts
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      alerts.forEach(alert => {
        const config = ALERT_SEVERITY_CONFIG[alert.level];
        if (
          config.autoAcknowledge &&
          !alert.acknowledged &&
          Date.now() - alert.timestamp.getTime() > config.autoAcknowledgeDelay
        ) {
          onAlertAcknowledge?.(alert.id);
        }
      });
    }, 60000); // Check every minute

    return () => clearInterval(interval);
  }, [alerts, autoRefresh, onAlertAcknowledge]);

  // Handle alert actions
  const handleAcknowledge = useCallback((alertId: string) => {
    onAlertAcknowledge?.(alertId);
    setNotification({ message: 'Alert acknowledged', severity: 'success' });
  }, [onAlertAcknowledge]);

  const handleResolve = useCallback((alertId: string) => {
    onAlertResolve?.(alertId);
    setNotification({ message: 'Alert resolved', severity: 'success' });
  }, [onAlertResolve]);

  const handleDismiss = useCallback((alertId: string) => {
    onAlertDismiss?.(alertId);
    setNotification({ message: 'Alert dismissed', severity: 'info' });
  }, [onAlertDismiss]);

  // Bulk actions
  const handleBulkAcknowledge = useCallback(() => {
    const activeAlerts = filteredAlerts.filter(a => !a.acknowledged && !a.resolved);
    activeAlerts.forEach(alert => onAlertAcknowledge?.(alert.id));
    setNotification({ message: `${activeAlerts.length} alerts acknowledged`, severity: 'success' });
  }, [filteredAlerts, onAlertAcknowledge]);

  const handleBulkResolve = useCallback(() => {
    const unresolvedAlerts = filteredAlerts.filter(a => !a.resolved);
    unresolvedAlerts.forEach(alert => onAlertResolve?.(alert.id));
    setNotification({ message: `${unresolvedAlerts.length} alerts resolved`, severity: 'success' });
  }, [filteredAlerts, onAlertResolve]);

  // Render alert item
  const renderAlertItem = useCallback((alert: DashboardAlert) => {
    const config = ALERT_SEVERITY_CONFIG[alert.level];
    const typeConfig = ALERT_TYPE_CONFIG[alert.type];
    const IconComponent = config.icon;

    return (
      <ListItem
        key={alert.id}
        sx={{
          border: '1px solid',
          borderColor: `${config.color}.main`,
          borderRadius: 1,
          mb: 1,
          backgroundColor: alert.acknowledged ? 'action.hover' : 'background.paper',
          opacity: alert.resolved ? 0.6 : 1
        }}
      >
        <ListItemIcon>
          <IconComponent color={config.color} />
        </ListItemIcon>
        
        <ListItemText
          primary={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="subtitle2">{alert.title}</Typography>
              <Chip label={typeConfig.label} size="small" variant="outlined" />
              <Chip label={alert.level.toUpperCase()} size="small" color={config.color} />
            </Box>
          }
          secondary={
            <Box>
              <Typography variant="body2" color="textSecondary">
                {alert.message}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
                <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Schedule fontSize="small" />
                  {format(alert.timestamp, 'MMM dd, HH:mm:ss')}
                </Typography>
                <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <LocationOn fontSize="small" />
                  {alert.source}
                </Typography>
                {alert.acknowledgedBy && (
                  <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Person fontSize="small" />
                    Ack: {alert.acknowledgedBy}
                  </Typography>
                )}
              </Box>
            </Box>
          }
        />
        
        <ListItemSecondaryAction>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {!alert.acknowledged && (
              <Tooltip title="Acknowledge">
                <IconButton size="small" onClick={() => handleAcknowledge(alert.id)}>
                  <Check />
                </IconButton>
              </Tooltip>
            )}
            {!alert.resolved && (
              <Tooltip title="Resolve">
                <IconButton size="small" onClick={() => handleResolve(alert.id)}>
                  <CheckCircle />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title="Dismiss">
              <IconButton size="small" onClick={() => handleDismiss(alert.id)}>
                <Close />
              </IconButton>
            </Tooltip>
            <IconButton
              size="small"
              onClick={(e) => {
                setSelectedAlert(alert);
                setMenuAnchor(e.currentTarget);
              }}
            >
              <MoreVert />
            </IconButton>
          </Box>
        </ListItemSecondaryAction>
      </ListItem>
    );
  }, [handleAcknowledge, handleResolve, handleDismiss]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Badge badgeContent={alertStats.active} color="error">
              <Notifications />
            </Badge>
            <Typography variant="h6">Alert Manager</Typography>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title={soundEnabled ? 'Disable Sound' : 'Enable Sound'}>
              <IconButton onClick={() => setSoundEnabled(!soundEnabled)}>
                {soundEnabled ? <VolumeUp /> : <VolumeOff />}
              </IconButton>
            </Tooltip>
            <IconButton onClick={() => setFilterDialog(true)}>
              <FilterList />
            </IconButton>
            <IconButton onClick={() => setSettingsDialog(true)}>
              <Settings />
            </IconButton>
          </Box>
        </Box>

        {/* Statistics */}
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Chip label={`Total: ${alertStats.total}`} />
          <Chip label={`Active: ${alertStats.active}`} color="error" />
          <Chip label={`Critical: ${alertStats.byLevel.critical}`} color="error" />
          <Chip label={`Warnings: ${alertStats.byLevel.warning}`} color="warning" />
        </Box>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ mb: 2 }}>
        <Tabs value={currentTab} onChange={(_, value) => setCurrentTab(value)}>
          <Tab label={`Active (${alertStats.active})`} />
          <Tab label={`Acknowledged (${alertStats.acknowledged})`} />
          <Tab label={`Resolved (${alertStats.resolved})`} />
          <Tab label={`All (${alertStats.total})`} />
        </Tabs>
      </Paper>

      {/* Bulk Actions */}
      {filteredAlerts.length > 0 && (
        <Box sx={{ mb: 2, display: 'flex', gap: 1 }}>
          <Button
            size="small"
            onClick={handleBulkAcknowledge}
            disabled={filteredAlerts.filter(a => !a.acknowledged && !a.resolved).length === 0}
          >
            Acknowledge All
          </Button>
          <Button
            size="small"
            onClick={handleBulkResolve}
            disabled={filteredAlerts.filter(a => !a.resolved).length === 0}
          >
            Resolve All
          </Button>
        </Box>
      )}

      {/* Alert List */}
      <Paper sx={{ flex: 1, overflow: 'auto' }}>
        {filteredAlerts.length > 0 ? (
          <List sx={{ p: 2 }}>
            {filteredAlerts.map(renderAlertItem)}
          </List>
        ) : (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="body2" color="textSecondary">
              No alerts to display
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Alert Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={() => setMenuAnchor(null)}
      >
        <MenuItem onClick={() => {
          // View details
          setMenuAnchor(null);
        }}>
          View Details
        </MenuItem>
        <MenuItem onClick={() => {
          // Copy to clipboard
          if (selectedAlert) {
            navigator.clipboard.writeText(JSON.stringify(selectedAlert, null, 2));
            setNotification({ message: 'Alert copied to clipboard', severity: 'info' });
          }
          setMenuAnchor(null);
        }}>
          Copy Details
        </MenuItem>
      </Menu>

      {/* Filter Dialog */}
      <Dialog open={filterDialog} onClose={() => setFilterDialog(false)}>
        <DialogTitle>Filter Alerts</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2, minWidth: 300 }}>
            <FormControl fullWidth>
              <InputLabel>Alert Level</InputLabel>
              <Select
                value={filterLevel}
                onChange={(e) => setFilterLevel(e.target.value as AlertLevel | 'all')}
              >
                <MenuItem value="all">All Levels</MenuItem>
                <MenuItem value="info">Info</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth>
              <InputLabel>Alert Type</InputLabel>
              <Select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as AlertType | 'all')}
              >
                <MenuItem value="all">All Types</MenuItem>
                {Object.entries(ALERT_TYPE_CONFIG).map(([type, config]) => (
                  <MenuItem key={type} value={type}>
                    {config.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl fullWidth>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'timestamp' | 'level' | 'type')}
              >
                <MenuItem value="timestamp">Timestamp</MenuItem>
                <MenuItem value="level">Severity Level</MenuItem>
                <MenuItem value="type">Alert Type</MenuItem>
              </Select>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch
                  checked={sortOrder === 'desc'}
                  onChange={(e) => setSortOrder(e.target.checked ? 'desc' : 'asc')}
                />
              }
              label="Descending Order"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFilterDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Settings Dialog */}
      <Dialog open={settingsDialog} onClose={() => setSettingsDialog(false)}>
        <DialogTitle>Notification Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2, minWidth: 400 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={notificationSettings?.enabled ?? true}
                  onChange={(e) => onSettingsChange?.({
                    ...notificationSettings!,
                    enabled: e.target.checked
                  })}
                />
              }
              label="Enable Notifications"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={notificationSettings?.sound ?? true}
                  onChange={(e) => onSettingsChange?.({
                    ...notificationSettings!,
                    sound: e.target.checked
                  })}
                />
              }
              label="Sound Notifications"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={notificationSettings?.desktop ?? true}
                  onChange={(e) => onSettingsChange?.({
                    ...notificationSettings!,
                    desktop: e.target.checked
                  })}
                />
              }
              label="Desktop Notifications"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={notificationSettings?.email ?? false}
                  onChange={(e) => onSettingsChange?.({
                    ...notificationSettings!,
                    email: e.target.checked
                  })}
                />
              }
              label="Email Notifications"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Notification Snackbar */}
      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={3000}
        onClose={() => setNotification(null)}
      >
        {notification && (
          <MuiAlert severity={notification.severity} onClose={() => setNotification(null)}>
            {notification.message}
          </MuiAlert>
        )}
      </Snackbar>
    </Box>
  );
};

export default AlertManager;