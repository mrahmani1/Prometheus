/**
 * Main App component for Prometheus AI Platform
 * Implements modern React architecture with comprehensive error handling
 */

import React, { Suspense, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { HelmetProvider } from 'react-helmet-async';
import { ErrorBoundary } from 'react-error-boundary';

import { store } from './store';
import { useAppSelector, useAppDispatch } from './hooks/redux';
import { lightTheme, darkTheme, accessibilityTheme, highContrastTheme } from './theme';
import { updateViewport } from './store/slices/uiSlice';
import './i18n';

// Lazy-loaded components for code splitting
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const SeismicAnalysis = React.lazy(() => import('./pages/SeismicAnalysis'));
const VolcanicMonitoring = React.lazy(() => import('./pages/VolcanicMonitoring'));
const DeformationAnalysis = React.lazy(() => import('./pages/DeformationAnalysis'));
const ResourceAssessment = React.lazy(() => import('./pages/ResourceAssessment'));
const GeologicalModeling = React.lazy(() => import('./pages/GeologicalModeling'));
const Login = React.lazy(() => import('./pages/Login'));
const Settings = React.lazy(() => import('./pages/Settings'));

// Layout components
const MainLayout = React.lazy(() => import('./components/layout/MainLayout'));
const AuthLayout = React.lazy(() => import('./components/layout/AuthLayout'));

// Error fallback components
const ErrorFallback = React.lazy(() => import('./components/common/ErrorFallback'));
const LoadingSpinner = React.lazy(() => import('./components/common/LoadingSpinner'));

// Notification system
const NotificationSystem = React.lazy(() => import('./components/common/NotificationSystem'));

// WebSocket manager
const WebSocketManager = React.lazy(() => import('./components/common/WebSocketManager'));

/**
 * Theme selector hook
 */
const useTheme = () => {
  const { theme, accessibility } = useAppSelector(state => state.ui);
  
  let selectedTheme = theme === 'dark' ? darkTheme : lightTheme;
  
  // Auto theme detection
  if (theme === 'auto') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    selectedTheme = prefersDark ? darkTheme : lightTheme;
  }
  
  // Accessibility enhancements
  if (accessibility.highContrast) {
    selectedTheme = highContrastTheme;
  } else if (accessibility.screenReader || accessibility.reducedMotion) {
    selectedTheme = accessibilityTheme(selectedTheme);
  }
  
  return selectedTheme;
};

/**
 * Viewport manager component
 */
const ViewportManager: React.FC = () => {
  const dispatch = useAppDispatch();
  
  useEffect(() => {
    const handleResize = () => {
      dispatch(updateViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      }));
    };
    
    // Initial viewport size
    handleResize();
    
    // Listen for resize events
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [dispatch]);
  
  return null;
};

/**
 * Protected route component
 */
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAppSelector(state => state.auth);
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

/**
 * App content component (inside providers)
 */
const AppContent: React.FC = () => {
  const theme = useTheme();
  const { isAuthenticated } = useAppSelector(state => state.auth);
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <HelmetProvider>
          <Router>
            <ViewportManager />
            <ErrorBoundary
              FallbackComponent={ErrorFallback}
              onError={(error, errorInfo) => {
                console.error('Application error:', error, errorInfo);
                // Here you could send error to monitoring service
              }}
            >
              <Suspense fallback={<LoadingSpinner />}>
                <Routes>
                  {/* Authentication routes */}
                  <Route
                    path="/login"
                    element={
                      <AuthLayout>
                        <Login />
                      </AuthLayout>
                    }
                  />
                  
                  {/* Protected application routes */}
                  <Route
                    path="/"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <Dashboard />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/dashboard"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <Dashboard />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/seismic"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <SeismicAnalysis />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/volcanic"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <VolcanicMonitoring />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/deformation"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <DeformationAnalysis />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/resources"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <ResourceAssessment />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/geological"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <GeologicalModeling />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  <Route
                    path="/settings"
                    element={
                      <ProtectedRoute>
                        <MainLayout>
                          <Settings />
                        </MainLayout>
                      </ProtectedRoute>
                    }
                  />
                  
                  {/* Redirect unknown routes */}
                  <Route
                    path="*"
                    element={
                      isAuthenticated ? (
                        <Navigate to="/dashboard" replace />
                      ) : (
                        <Navigate to="/login" replace />
                      )
                    }
                  />
                </Routes>
                
                {/* Global components */}
                <NotificationSystem />
                <WebSocketManager />
              </Suspense>
            </ErrorBoundary>
          </Router>
        </HelmetProvider>
      </LocalizationProvider>
    </ThemeProvider>
  );
};

/**
 * Main App component
 */
const App: React.FC = () => {
  return (
    <Provider store={store}>
      <AppContent />
    </Provider>
  );
};

export default App;