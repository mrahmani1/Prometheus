import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  // TextField, FormControl, InputLabel, Select - unused
  Grid,
  Chip,
  Alert,
  Snackbar,
  Fab,
  // Tooltip - unused
} from '@mui/material';
import {
  Add,
  // Settings - unused
  MoreVert,
  Edit,
  Delete,
  Fullscreen,
  FullscreenExit,
  Refresh,
  Save,
  // Restore - unused
  Dashboard as DashboardIcon,
  // GridView, ViewModule - unused
} from '@mui/icons-material';
import { Responsive, WidthProvider, Layout } from 'react-grid-layout';
import { DashboardWidget, WidgetConfig, DashboardConfig, UserPreferences } from '../../types/dashboard';
import { GeologicalData, SeismicEvent, VolcanicState } from '../../types/geological';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

interface DashboardLayoutProps {
  widgets: DashboardWidget[];
  onWidgetAdd?: (widget: DashboardWidget) => void;
  onWidgetUpdate?: (widgetId: string, config: WidgetConfig) => void;
  onWidgetRemove?: (widgetId: string) => void;
  onLayoutChange?: (layout: Layout[]) => void;
  onDashboardSave?: (config: DashboardConfig) => void;
  userPreferences?: UserPreferences;
  data?: {
    geological: GeologicalData[];
    seismic: SeismicEvent[];
    volcanic: VolcanicState[];
  };
  realTimeUpdates?: boolean;
  readOnly?: boolean;
}

// Available widget types for geological monitoring
const WIDGET_TYPES = [
  {
    id: 'seismic-monitor',
    name: 'Seismic Monitor',
    description: 'Real-time seismic event monitoring',
    category: 'monitoring',
    defaultSize: { w: 6, h: 4 },
    icon: '📊'
  },
  {
    id: 'volcanic-status',
    name: 'Volcanic Status',
    description: 'Volcano alert levels and status',
    category: 'monitoring',
    defaultSize: { w: 4, h: 3 },
    icon: '🌋'
  },
  {
    id: 'geological-map',
    name: 'Geological Map',
    description: 'Interactive geological mapping',
    category: 'visualization',
    defaultSize: { w: 8, h: 6 },
    icon: '🗺️'
  },
  {
    id: 'data-chart',
    name: 'Data Chart',
    description: 'Scientific data visualization',
    category: 'visualization',
    defaultSize: { w: 6, h: 4 },
    icon: '📈'
  },
  {
    id: 'alert-panel',
    name: 'Alert Panel',
    description: 'System alerts and notifications',
    category: 'alerts',
    defaultSize: { w: 4, h: 2 },
    icon: '⚠️'
  },
  {
    id: 'data-grid',
    name: 'Data Grid',
    description: 'Tabular data display',
    category: 'data',
    defaultSize: { w: 8, h: 5 },
    icon: '📋'
  },
  {
    id: 'statistics',
    name: 'Statistics',
    description: 'Key performance indicators',
    category: 'analytics',
    defaultSize: { w: 3, h: 2 },
    icon: '📊'
  },
  {
    id: '3d-visualization',
    name: '3D Visualization',
    description: 'Three-dimensional geological models',
    category: 'visualization',
    defaultSize: { w: 8, h: 6 },
    icon: '🎯'
  }
];

// Default dashboard layouts for different screen sizes
const DEFAULT_LAYOUTS = {
  lg: [
    { i: 'seismic-monitor', x: 0, y: 0, w: 6, h: 4 },
    { i: 'volcanic-status', x: 6, y: 0, w: 4, h: 3 },
    { i: 'geological-map', x: 0, y: 4, w: 8, h: 6 },
    { i: 'alert-panel', x: 8, y: 0, w: 4, h: 2 }
  ],
  md: [
    { i: 'seismic-monitor', x: 0, y: 0, w: 6, h: 4 },
    { i: 'volcanic-status', x: 6, y: 0, w: 4, h: 3 },
    { i: 'geological-map', x: 0, y: 4, w: 10, h: 6 },
    { i: 'alert-panel', x: 0, y: 10, w: 10, h: 2 }
  ],
  sm: [
    { i: 'seismic-monitor', x: 0, y: 0, w: 6, h: 4 },
    { i: 'volcanic-status', x: 0, y: 4, w: 6, h: 3 },
    { i: 'geological-map', x: 0, y: 7, w: 6, h: 6 },
    { i: 'alert-panel', x: 0, y: 13, w: 6, h: 2 }
  ]
};

// Breakpoints for responsive layout
const BREAKPOINTS = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 };
const COLS = { lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 };

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  widgets,
  onWidgetAdd,
  onWidgetUpdate: _onWidgetUpdate,
  onWidgetRemove,
  onLayoutChange,
  onDashboardSave,
  userPreferences,
  data,
  realTimeUpdates = false,
  readOnly = false
}) => {
  const [layouts, setLayouts] = useState(DEFAULT_LAYOUTS);
  const [, setCurrentBreakpoint] = useState('lg');
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedWidget, setSelectedWidget] = useState<string | null>(null);
  const [widgetMenuAnchor, setWidgetMenuAnchor] = useState<HTMLElement | null>(null);
  const [addWidgetDialog, setAddWidgetDialog] = useState(false);
  const [, setSettingsDialog] = useState(false);
  const [fullscreenWidget, setFullscreenWidget] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ message: string; severity: 'success' | 'error' | 'warning' | 'info' } | null>(null);

  // Dashboard configuration state
  const [dashboardConfig] = useState<DashboardConfig>({
    id: 'default-dashboard',
    name: 'Geological Monitoring Dashboard',
    description: 'Real-time geological data monitoring and analysis',
    layout: layouts,
    widgets: widgets.map(w => w.id),
    theme: userPreferences?.theme || 'light',
    autoRefresh: realTimeUpdates,
    refreshInterval: 30000,
    permissions: {
      canEdit: !readOnly,
      canAdd: !readOnly,
      canDelete: !readOnly,
      canExport: true
    }
  });

  // Widget instances with data binding
  const widgetInstances = useMemo(() => {
    return widgets.map(widget => ({
      ...widget,
      data: getWidgetData(widget.type, data),
      lastUpdated: new Date()
    }));
  }, [widgets, data]);

  // Get appropriate data for widget type
  function getWidgetData(widgetType: string, allData?: any) {
    if (!allData) return null;

    switch (widgetType) {
      case 'seismic-monitor':
        return allData.seismic;
      case 'volcanic-status':
        return allData.volcanic;
      case 'geological-map':
        return {
          geological: allData.geological,
          seismic: allData.seismic,
          volcanic: allData.volcanic
        };
      case 'data-chart':
        return allData.geological;
      case 'data-grid':
        return allData.geological;
      default:
        return allData;
    }
  }

  // Handle layout changes
  const handleLayoutChange = useCallback((layout: Layout[], layouts: any) => {
    setLayouts(layouts);
    onLayoutChange?.(layout);
  }, [onLayoutChange]);

  // Handle breakpoint changes
  const handleBreakpointChange = useCallback((breakpoint: string) => {
    setCurrentBreakpoint(breakpoint);
  }, []);

  // Add new widget
  const handleAddWidget = useCallback((widgetType: string) => {
    const widgetTemplate = WIDGET_TYPES.find(t => t.id === widgetType);
    if (!widgetTemplate) return;

    const newWidget: DashboardWidget = {
      id: `${widgetType}-${Date.now()}`,
      type: widgetType,
      title: widgetTemplate.name,
      config: {
        refreshInterval: 30000,
        showHeader: true,
        showBorder: true,
        backgroundColor: 'transparent',
        ...widgetTemplate.defaultSize
      },
      position: {
        x: 0,
        y: 0,
        ...widgetTemplate.defaultSize
      },
      permissions: {
        canEdit: !readOnly,
        canDelete: !readOnly,
        canMove: !readOnly,
        canResize: !readOnly
      }
    };

    onWidgetAdd?.(newWidget);
    setAddWidgetDialog(false);
    setNotification({ message: 'Widget added successfully', severity: 'success' });
  }, [onWidgetAdd, readOnly]);

  // Remove widget
  const handleRemoveWidget = useCallback((widgetId: string) => {
    onWidgetRemove?.(widgetId);
    setWidgetMenuAnchor(null);
    setSelectedWidget(null);
    setNotification({ message: 'Widget removed', severity: 'info' });
  }, [onWidgetRemove]);

  // Edit widget
  const handleEditWidget = useCallback((widgetId: string) => {
    setSelectedWidget(widgetId);
    setSettingsDialog(true);
    setWidgetMenuAnchor(null);
  }, []);

  // Toggle fullscreen for widget
  const handleToggleFullscreen = useCallback((widgetId: string) => {
    setFullscreenWidget(fullscreenWidget === widgetId ? null : widgetId);
    setWidgetMenuAnchor(null);
  }, [fullscreenWidget]);

  // Save dashboard configuration
  const handleSaveDashboard = useCallback(() => {
    const config: DashboardConfig = {
      ...dashboardConfig,
      layout: layouts,
      widgets: widgets.map(w => w.id),
      lastModified: new Date()
    };

    onDashboardSave?.(config);
    setNotification({ message: 'Dashboard saved successfully', severity: 'success' });
  }, [dashboardConfig, layouts, widgets, onDashboardSave]);

  // Render widget content based on type
  const renderWidgetContent = useCallback((widget: DashboardWidget) => {
    const widgetData = getWidgetData(widget.type, data);

    switch (widget.type) {
      case 'seismic-monitor':
        return (
          <Box sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Seismic Activity</Typography>
            {widgetData && widgetData.length > 0 ? (
              <Box>
                <Typography variant="body2">
                  Recent Events: {widgetData.length}
                </Typography>
                <Typography variant="body2">
                  Latest: M{widgetData[0]?.magnitude?.toFixed(1)} at {widgetData[0]?.timestamp?.toLocaleTimeString()}
                </Typography>
              </Box>
            ) : (
              <Typography variant="body2" color="textSecondary">
                No recent seismic activity
              </Typography>
            )}
          </Box>
        );

      case 'volcanic-status':
        return (
          <Box sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Volcanic Status</Typography>
            {widgetData && widgetData.length > 0 ? (
              <Box>
                {widgetData.slice(0, 3).map((volcano: any, index: number) => (
                  <Chip
                    key={index}
                    label={`${volcano.volcanoId}: ${volcano.alertLevel}`}
                    color={volcano.alertLevel === 'red' ? 'error' : volcano.alertLevel === 'orange' ? 'warning' : 'success'}
                    size="small"
                    sx={{ mr: 1, mb: 1 }}
                  />
                ))}
              </Box>
            ) : (
              <Typography variant="body2" color="textSecondary">
                All volcanoes normal
              </Typography>
            )}
          </Box>
        );

      case 'alert-panel':
        return (
          <Box sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>System Alerts</Typography>
            <Alert severity="info" sx={{ mb: 1 }}>
              System operational
            </Alert>
            {realTimeUpdates && (
              <Alert severity="success">
                Real-time updates active
              </Alert>
            )}
          </Box>
        );

      case 'statistics':
        return (
          <Box sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Statistics</Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2">Geological</Typography>
                <Typography variant="h6">{data?.geological?.length || 0}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">Seismic</Typography>
                <Typography variant="h6">{data?.seismic?.length || 0}</Typography>
              </Grid>
            </Grid>
          </Box>
        );

      default:
        return (
          <Box sx={{ p: 2, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography variant="body2" color="textSecondary">
              {widget.type} widget content
            </Typography>
          </Box>
        );
    }
  }, [data, realTimeUpdates]);

  // Render individual widget
  const renderWidget = useCallback((widget: DashboardWidget) => {
    const isFullscreen = fullscreenWidget === widget.id;

    return (
      <Paper
        key={widget.id}
        elevation={isEditMode ? 3 : 1}
        sx={{
          height: '100%',
          position: 'relative',
          border: isEditMode ? '2px dashed #ccc' : 'none',
          backgroundColor: widget.config.backgroundColor || 'background.paper',
          ...(isFullscreen && {
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            zIndex: 9999
          })
        }}
      >
        {/* Widget Header */}
        {widget.config.showHeader !== false && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              p: 1,
              borderBottom: '1px solid',
              borderColor: 'divider',
              backgroundColor: 'action.hover'
            }}
          >
            <Typography variant="subtitle2" noWrap>
              {widget.title}
            </Typography>
            <Box>
              <IconButton
                size="small"
                onClick={() => {
                  // Refresh widget data
                  setNotification({ message: 'Widget refreshed', severity: 'info' });
                }}
              >
                <Refresh fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={(e) => {
                  setSelectedWidget(widget.id);
                  setWidgetMenuAnchor(e.currentTarget);
                }}
              >
                <MoreVert fontSize="small" />
              </IconButton>
            </Box>
          </Box>
        )}

        {/* Widget Content */}
        <Box sx={{ height: widget.config.showHeader !== false ? 'calc(100% - 48px)' : '100%' }}>
          {renderWidgetContent(widget)}
        </Box>
      </Paper>
    );
  }, [isEditMode, fullscreenWidget, renderWidgetContent]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Dashboard Toolbar */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <DashboardIcon />
            <Typography variant="h6">{dashboardConfig.name}</Typography>
            {realTimeUpdates && (
              <Chip label="Live" color="success" size="small" />
            )}
          </Box>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant={isEditMode ? 'contained' : 'outlined'}
              onClick={() => setIsEditMode(!isEditMode)}
              disabled={readOnly}
              startIcon={<Edit />}
            >
              {isEditMode ? 'Done' : 'Edit'}
            </Button>
            <Button
              variant="outlined"
              onClick={() => setAddWidgetDialog(true)}
              disabled={readOnly}
              startIcon={<Add />}
            >
              Add Widget
            </Button>
            <Button
              variant="outlined"
              onClick={handleSaveDashboard}
              disabled={readOnly}
              startIcon={<Save />}
            >
              Save
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Dashboard Grid */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <ResponsiveGridLayout
          className="layout"
          layouts={layouts}
          breakpoints={BREAKPOINTS}
          cols={COLS}
          onLayoutChange={handleLayoutChange}
          onBreakpointChange={handleBreakpointChange}
          isDraggable={isEditMode && !readOnly}
          isResizable={isEditMode && !readOnly}
          margin={[16, 16]}
          containerPadding={[16, 16]}
          rowHeight={60}
        >
          {widgetInstances.map(renderWidget)}
        </ResponsiveGridLayout>
      </Box>

      {/* Floating Action Button for Quick Add */}
      {!readOnly && (
        <Fab
          color="primary"
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={() => setAddWidgetDialog(true)}
        >
          <Add />
        </Fab>
      )}

      {/* Widget Context Menu */}
      <Menu
        anchorEl={widgetMenuAnchor}
        open={Boolean(widgetMenuAnchor)}
        onClose={() => setWidgetMenuAnchor(null)}
      >
        <MenuItem onClick={() => handleEditWidget(selectedWidget!)}>
          <Edit sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={() => handleToggleFullscreen(selectedWidget!)}>
          {fullscreenWidget === selectedWidget ? <FullscreenExit sx={{ mr: 1 }} /> : <Fullscreen sx={{ mr: 1 }} />}
          {fullscreenWidget === selectedWidget ? 'Exit Fullscreen' : 'Fullscreen'}
        </MenuItem>
        <MenuItem onClick={() => handleRemoveWidget(selectedWidget!)} sx={{ color: 'error.main' }}>
          <Delete sx={{ mr: 1 }} />
          Remove
        </MenuItem>
      </Menu>

      {/* Add Widget Dialog */}
      <Dialog
        open={addWidgetDialog}
        onClose={() => setAddWidgetDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Add Widget</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {WIDGET_TYPES.map(widgetType => (
              <Grid item xs={12} sm={6} md={4} key={widgetType.id}>
                <Paper
                  sx={{
                    p: 2,
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                  onClick={() => handleAddWidget(widgetType.id)}
                >
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ mb: 1 }}>
                      {widgetType.icon}
                    </Typography>
                    <Typography variant="subtitle1" gutterBottom>
                      {widgetType.name}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {widgetType.description}
                    </Typography>
                    <Chip
                      label={widgetType.category}
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddWidgetDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      {/* Notifications */}
      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={3000}
        onClose={() => setNotification(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
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

export default DashboardLayout;