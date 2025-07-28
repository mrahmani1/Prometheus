import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Collapse,
  Divider,
  Box,
  Typography,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Map as MapIcon,
  Science as ScienceIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  ExpandLess,
  ExpandMore,
  ChevronLeft,
  ChevronRight,
  Terrain,
  Warning,
  Analytics,
  AccountTree,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAppSelector, useAppDispatch } from '@/store';
import { toggleSidebarCollapsed } from '@/store/slices/uiSlice';
import { NavigationItem } from '@/types/ui';

const DRAWER_WIDTH = 280;
const COLLAPSED_WIDTH = 64;

// Navigation structure for geological workflows
const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    path: '/dashboard',
    icon: 'dashboard',
  },
  {
    id: 'monitoring',
    label: 'Real-time Monitoring',
    path: '/monitoring',
    icon: 'timeline',
    children: [
      {
        id: 'seismic',
        label: 'Seismic Events',
        path: '/monitoring/seismic',
        icon: 'terrain',
      },
      {
        id: 'deformation',
        label: 'Ground Deformation',
        path: '/monitoring/deformation',
        icon: 'map',
      },
      {
        id: 'thermal',
        label: 'Thermal Analysis',
        path: '/monitoring/thermal',
        icon: 'science',
      },
      {
        id: 'alerts',
        label: 'Volcanic Alerts',
        path: '/monitoring/alerts',
        icon: 'warning',
      },
    ],
  },
  {
    id: 'simulations',
    label: 'Geological Simulations',
    path: '/simulations',
    icon: 'science',
    children: [
      {
        id: 'helios',
        label: 'Helios AI Models',
        path: '/simulations/helios',
        icon: 'science',
      },
      {
        id: 'long-term',
        label: 'Long-term Evolution',
        path: '/simulations/long-term',
        icon: 'timeline',
      },
      {
        id: 'risk-assessment',
        label: 'Risk Assessment',
        path: '/simulations/risk',
        icon: 'assessment',
      },
    ],
  },
  {
    id: 'workflows',
    label: 'Analysis Workflows',
    path: '/workflows',
    icon: 'account-tree',
    children: [
      {
        id: 'workflow-builder',
        label: 'Workflow Builder',
        path: '/workflows/builder',
        icon: 'account-tree',
      },
      {
        id: 'workflow-templates',
        label: 'Templates',
        path: '/workflows/templates',
        icon: 'assessment',
      },
      {
        id: 'workflow-history',
        label: 'Execution History',
        path: '/workflows/history',
        icon: 'timeline',
      },
    ],
  },
  {
    id: 'visualization',
    label: 'Data Visualization',
    path: '/visualization',
    icon: 'analytics',
    children: [
      {
        id: 'maps',
        label: '2D/3D Maps',
        path: '/visualization/maps',
        icon: 'map',
      },
      {
        id: 'charts',
        label: 'Charts & Graphs',
        path: '/visualization/charts',
        icon: 'analytics',
      },
      {
        id: 'time-series',
        label: 'Time Series',
        path: '/visualization/time-series',
        icon: 'timeline',
      },
    ],
  },
  {
    id: 'data',
    label: 'Data Management',
    path: '/data',
    icon: 'assessment',
    children: [
      {
        id: 'import',
        label: 'Data Import',
        path: '/data/import',
        icon: 'assessment',
      },
      {
        id: 'export',
        label: 'Data Export',
        path: '/data/export',
        icon: 'assessment',
      },
      {
        id: 'quality',
        label: 'Quality Control',
        path: '/data/quality',
        icon: 'assessment',
      },
    ],
  },
];

const getIcon = (iconName: string) => {
  const iconMap: Record<string, React.ReactElement> = {
    dashboard: <DashboardIcon />,
    timeline: <TimelineIcon />,
    map: <MapIcon />,
    science: <ScienceIcon />,
    assessment: <AssessmentIcon />,
    settings: <SettingsIcon />,
    terrain: <Terrain />,
    warning: <Warning />,
    analytics: <Analytics />,
    'account-tree': <AccountTree />,
  };
  return iconMap[iconName] || <DashboardIcon />;
};

export const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useAppDispatch();
  
  const { sidebarCollapsed } = useAppSelector((state) => state.ui);
  const { user } = useAppSelector((state) => state.auth);
  
  const [expandedItems, setExpandedItems] = React.useState<string[]>([]);

  const handleToggleExpanded = (itemId: string) => {
    setExpandedItems(prev => 
      prev.includes(itemId) 
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    );
  };

  const handleNavigate = (path: string) => {
    navigate(path);
  };

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  const renderNavigationItem = (item: NavigationItem, level: number = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.id);
    const active = isActive(item.path);

    return (
      <React.Fragment key={item.id}>
        <ListItem disablePadding sx={{ display: 'block' }}>
          <ListItemButton
            onClick={() => {
              if (hasChildren) {
                handleToggleExpanded(item.id);
              } else {
                handleNavigate(item.path);
              }
            }}
            selected={active}
            sx={{
              minHeight: 48,
              justifyContent: sidebarCollapsed ? 'center' : 'initial',
              px: 2.5,
              pl: level > 0 ? 4 : 2.5,
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: sidebarCollapsed ? 0 : 3,
                justifyContent: 'center',
              }}
            >
              {item.icon && getIcon(item.icon)}
            </ListItemIcon>
            
            {!sidebarCollapsed && (
              <>
                <ListItemText 
                  primary={item.label}
                  sx={{ opacity: 1 }}
                />
                {hasChildren && (
                  isExpanded ? <ExpandLess /> : <ExpandMore />
                )}
              </>
            )}
          </ListItemButton>
        </ListItem>
        
        {hasChildren && !sidebarCollapsed && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children?.map(child => renderNavigationItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: sidebarCollapsed ? COLLAPSED_WIDTH : DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: sidebarCollapsed ? COLLAPSED_WIDTH : DRAWER_WIDTH,
          boxSizing: 'border-box',
          transition: 'width 0.3s ease',
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: sidebarCollapsed ? 'center' : 'space-between',
          padding: 2,
          minHeight: 64,
        }}
      >
        {!sidebarCollapsed && (
          <Typography variant="h6" noWrap component="div">
            Vulcan UI
          </Typography>
        )}
        
        <Tooltip title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
          <IconButton
            onClick={() => dispatch(toggleSidebarCollapsed())}
            size="small"
          >
            {sidebarCollapsed ? <ChevronRight /> : <ChevronLeft />}
          </IconButton>
        </Tooltip>
      </Box>
      
      <Divider />
      
      {/* User info */}
      {!sidebarCollapsed && user && (
        <Box sx={{ p: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Welcome back,
          </Typography>
          <Typography variant="subtitle2" noWrap>
            {user.firstName} {user.lastName}
          </Typography>
          <Typography variant="caption" color="text.secondary" noWrap>
            {user.institution}
          </Typography>
        </Box>
      )}
      
      <Divider />
      
      {/* Navigation items */}
      <List sx={{ flexGrow: 1, py: 1 }}>
        {navigationItems.map(item => renderNavigationItem(item))}
      </List>
      
      <Divider />
      
      {/* Settings */}
      <List>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => handleNavigate('/settings')}
            selected={isActive('/settings')}
            sx={{
              minHeight: 48,
              justifyContent: sidebarCollapsed ? 'center' : 'initial',
              px: 2.5,
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: sidebarCollapsed ? 0 : 3,
                justifyContent: 'center',
              }}
            >
              <SettingsIcon />
            </ListItemIcon>
            {!sidebarCollapsed && (
              <ListItemText primary="Settings" />
            )}
          </ListItemButton>
        </ListItem>
      </List>
    </Drawer>
  );
};