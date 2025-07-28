import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Menu,
  MenuItem,
  Avatar,
  Box,
  Tooltip,
  Chip,
  Button,
} from '@mui/material';
import {
  Notifications as NotificationsIcon,
  AccountCircle,
  Brightness4,
  Brightness7,
  Logout,
  Settings,
  Help,
  Refresh,
} from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '@/store';
import { setTheme } from '@/store/slices/uiSlice';
import { logoutUser } from '@/store/slices/authSlice';
import { useNavigate } from 'react-router-dom';

export const TopBar: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  
  const { layout, notifications } = useAppSelector((state) => state.ui);
  const { user } = useAppSelector((state) => state.auth);
  const { activeSimulation } = useAppSelector((state) => state.geologicalData);
  
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [notificationAnchorEl, setNotificationAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchorEl(event.currentTarget);
  };

  const handleNotificationMenuClose = () => {
    setNotificationAnchorEl(null);
  };

  const handleThemeToggle = () => {
    dispatch(setTheme(layout.theme === 'light' ? 'dark' : 'light'));
  };

  const handleLogout = () => {
    dispatch(logoutUser());
    handleProfileMenuClose();
    navigate('/login');
  };

  const handleSettings = () => {
    navigate('/settings');
    handleProfileMenuClose();
  };

  const unreadNotifications = notifications.filter(n => 
    n.type === 'error' || n.type === 'warning'
  ).length;

  const getStatusColor = () => {
    if (activeSimulation.isRunning) return 'warning';
    if (unreadNotifications > 0) return 'error';
    return 'success';
  };

  const getStatusText = () => {
    if (activeSimulation.isRunning) return 'Simulation Running';
    if (unreadNotifications > 0) return `${unreadNotifications} Alerts`;
    return 'System Normal';
  };

  return (
    <AppBar 
      position="static" 
      elevation={1}
      sx={{ 
        backgroundColor: 'background.paper',
        color: 'text.primary',
        borderBottom: 1,
        borderColor: 'divider',
      }}
    >
      <Toolbar>
        {/* Page title and breadcrumb */}
        <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
          <Typography variant="h6" component="div" sx={{ mr: 2 }}>
            Prometheus AI Platform
          </Typography>
          
          {/* System status indicator */}
          <Chip
            label={getStatusText()}
            color={getStatusColor()}
            size="small"
            variant="outlined"
            sx={{ mr: 2 }}
          />
          
          {/* Active simulation progress */}
          {activeSimulation.isRunning && (
            <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
              <Typography variant="body2" sx={{ mr: 1 }}>
                Progress: {activeSimulation.progress}%
              </Typography>
              <Button
                size="small"
                variant="outlined"
                onClick={() => navigate('/simulations')}
              >
                View Details
              </Button>
            </Box>
          )}
        </Box>

        {/* Action buttons */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Refresh button */}
          <Tooltip title="Refresh data">
            <IconButton
              size="large"
              color="inherit"
              onClick={() => window.location.reload()}
            >
              <Refresh />
            </IconButton>
          </Tooltip>

          {/* Theme toggle */}
          <Tooltip title={`Switch to ${layout.theme === 'light' ? 'dark' : 'light'} mode`}>
            <IconButton
              size="large"
              color="inherit"
              onClick={handleThemeToggle}
            >
              {layout.theme === 'light' ? <Brightness4 /> : <Brightness7 />}
            </IconButton>
          </Tooltip>

          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton
              size="large"
              color="inherit"
              onClick={handleNotificationMenuOpen}
            >
              <Badge badgeContent={unreadNotifications} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* User profile */}
          <Tooltip title="Account settings">
            <IconButton
              size="large"
              edge="end"
              onClick={handleProfileMenuOpen}
              color="inherit"
              sx={{ ml: 1 }}
            >
              {user?.firstName ? (
                <Avatar sx={{ width: 32, height: 32 }}>
                  {user.firstName[0]}{user.lastName[0]}
                </Avatar>
              ) : (
                <AccountCircle />
              )}
            </IconButton>
          </Tooltip>
        </Box>

        {/* Profile menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleProfileMenuClose}
          onClick={handleProfileMenuClose}
          PaperProps={{
            elevation: 0,
            sx: {
              overflow: 'visible',
              filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
              mt: 1.5,
              '& .MuiAvatar-root': {
                width: 32,
                height: 32,
                ml: -0.5,
                mr: 1,
              },
            },
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          {user && (
            <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider' }}>
              <Typography variant="subtitle2">
                {user.firstName} {user.lastName}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {user.email}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {user.role} • {user.institution}
              </Typography>
            </Box>
          )}
          
          <MenuItem onClick={handleSettings}>
            <Settings fontSize="small" sx={{ mr: 1 }} />
            Settings
          </MenuItem>
          
          <MenuItem onClick={() => navigate('/help')}>
            <Help fontSize="small" sx={{ mr: 1 }} />
            Help & Support
          </MenuItem>
          
          <MenuItem onClick={handleLogout}>
            <Logout fontSize="small" sx={{ mr: 1 }} />
            Logout
          </MenuItem>
        </Menu>

        {/* Notifications menu */}
        <Menu
          anchorEl={notificationAnchorEl}
          open={Boolean(notificationAnchorEl)}
          onClose={handleNotificationMenuClose}
          PaperProps={{
            elevation: 0,
            sx: {
              overflow: 'visible',
              filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
              mt: 1.5,
              maxWidth: 400,
              maxHeight: 400,
            },
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          {notifications.length === 0 ? (
            <MenuItem>
              <Typography variant="body2" color="text.secondary">
                No notifications
              </Typography>
            </MenuItem>
          ) : (
            notifications.slice(0, 5).map((notification) => (
              <MenuItem key={notification.id} onClick={handleNotificationMenuClose}>
                <Box>
                  <Typography variant="subtitle2">
                    {notification.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {notification.message}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {new Date(notification.timestamp).toLocaleTimeString()}
                  </Typography>
                </Box>
              </MenuItem>
            ))
          )}
          
          {notifications.length > 5 && (
            <MenuItem onClick={() => navigate('/notifications')}>
              <Typography variant="body2" color="primary">
                View all notifications
              </Typography>
            </MenuItem>
          )}
        </Menu>
      </Toolbar>
    </AppBar>
  );
};