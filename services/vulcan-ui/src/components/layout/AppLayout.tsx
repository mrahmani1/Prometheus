import React, { useEffect } from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { useAppSelector, useAppDispatch } from '@/store';
import { getCurrentUser } from '@/store/slices/authSlice';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { NotificationCenter } from './NotificationCenter';
import { LoadingOverlay } from '../common/LoadingOverlay';
import { ErrorBoundary } from '../common/ErrorBoundary';

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const { layout, loading } = useAppSelector((state) => state.ui);
  const { isAuthenticated, token } = useAppSelector((state) => state.auth);

  // Create theme based on user preferences
  const theme = createTheme({
    palette: {
      mode: layout.theme,
      primary: {
        main: layout.primaryColor,
      },
    },
    components: {
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: layout.theme === 'dark' ? '#1e1e1e' : '#f5f5f5',
          },
        },
      },
    },
  });

  // Load user data on mount if authenticated
  useEffect(() => {
    if (isAuthenticated && token) {
      dispatch(getCurrentUser());
    }
  }, [dispatch, isAuthenticated, token]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <Box sx={{ display: 'flex', height: '100vh' }}>
          {/* Sidebar */}
          <Sidebar />
          
          {/* Main content area */}
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Top navigation bar */}
            <TopBar />
            
            {/* Page content */}
            <Box
              sx={{
                flexGrow: 1,
                overflow: 'auto',
                padding: layout.compactMode ? 1 : 2,
                backgroundColor: theme.palette.background.default,
              }}
            >
              {children}
            </Box>
          </Box>
          
          {/* Notification center */}
          <NotificationCenter />
          
          {/* Loading overlay */}
          {loading.isLoading && (
            <LoadingOverlay
              message={loading.message}
              progress={loading.progress}
            />
          )}
        </Box>
      </ErrorBoundary>
    </ThemeProvider>
  );
};