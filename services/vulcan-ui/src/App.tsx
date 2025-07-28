import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Provider } from 'react-redux';
import { store } from '@/store';
import { AppLayout } from '@/components/layout/AppLayout';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { LoginPage } from '@/pages/auth/LoginPage';
import { DashboardPage } from '@/pages/DashboardPage';
import { MonitoringPage } from '@/pages/monitoring/MonitoringPage';
import { SimulationsPage } from '@/pages/simulations/SimulationsPage';
import { WorkflowsPage } from '@/pages/workflows/WorkflowsPage';
import { VisualizationPage } from '@/pages/visualization/VisualizationPage';
import { DataManagementPage } from '@/pages/data/DataManagementPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { NotFoundPage } from '@/pages/NotFoundPage';
import { UnauthorizedPage } from '@/pages/UnauthorizedPage';

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/unauthorized" element={<UnauthorizedPage />} />
          
          {/* Protected routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <Routes>
                    <Route index element={<Navigate to="/dashboard" replace />} />
                    <Route path="dashboard" element={<DashboardPage />} />
                    
                    {/* Monitoring routes */}
                    <Route path="monitoring/*" element={<MonitoringPage />} />
                    
                    {/* Simulation routes */}
                    <Route path="simulations/*" element={<SimulationsPage />} />
                    
                    {/* Workflow routes */}
                    <Route path="workflows/*" element={<WorkflowsPage />} />
                    
                    {/* Visualization routes */}
                    <Route path="visualization/*" element={<VisualizationPage />} />
                    
                    {/* Data management routes */}
                    <Route path="data/*" element={<DataManagementPage />} />
                    
                    {/* Settings */}
                    <Route path="settings" element={<SettingsPage />} />
                    
                    {/* 404 */}
                    <Route path="*" element={<NotFoundPage />} />
                  </Routes>
                </AppLayout>
              </ProtectedRoute>
            }
          />
          
          {/* Catch all - redirect to login */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Router>
    </Provider>
  );
};

export default App;