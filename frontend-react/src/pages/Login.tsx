/**
 * Login page component
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  Stack,
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { loginStart, loginSuccess, loginFailure } from '../store/slices/authSlice';

const Login: React.FC = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { loading, error } = useAppSelector(state => state.auth);
  
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    dispatch(loginStart());
    
    try {
      // Production authentication with backend API
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Authentication failed');
      }

      const authData = await response.json();
      
      // Validate response structure
      if (!authData.token || !authData.user) {
        throw new Error('Invalid authentication response');
      }

      // Store authentication tokens securely
      localStorage.setItem('auth_token', authData.token);
      if (authData.refreshToken) {
        localStorage.setItem('refresh_token', authData.refreshToken);
      }

      dispatch(loginSuccess({
        token: authData.token,
        refreshToken: authData.refreshToken,
        user: {
          id: authData.user.id,
          username: authData.user.username,
          email: authData.user.email,
          firstName: authData.user.firstName,
          lastName: authData.user.lastName,
          roles: authData.user.roles || [],
          permissions: authData.user.permissions || [],
          preferences: authData.user.preferences || {
            theme: 'light',
            language: 'en',
            timezone: 'UTC',
            units: {
              length: 'metric',
              temperature: 'celsius',
              pressure: 'pascal',
              mass: 'kilogram',
            },
            notifications: {
              email: true,
              push: true,
              sms: false,
              alertLevels: ['warning', 'error'],
            },
            dashboard: {
              layout: 'default',
              widgets: [],
              refreshInterval: 30000,
            },
          },
          lastLogin: authData.user.lastLogin || new Date().toISOString(),
        },
        expiresIn: authData.expiresIn || 3600,
      }));

      // Log successful authentication
      console.log('User authenticated successfully:', authData.user.username);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Authentication failed';
      console.error('Authentication error:', errorMessage);
      dispatch(loginFailure(errorMessage));
      
      // Clear any stored tokens on failure
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
    }
  };

  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setCredentials(prev => ({
      ...prev,
      [field]: e.target.value,
    }));
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        p: 3,
      }}
    >
      <Card sx={{ maxWidth: 400, width: '100%' }}>
        <CardContent>
          <Stack spacing={3}>
            <Typography variant="h4" component="h1" textAlign="center">
              {t('auth.login')}
            </Typography>
            
            <Typography variant="body2" textAlign="center" color="textSecondary">
              Prometheus AI Platform
            </Typography>
            
            {error && (
              <Alert severity="error">
                {error}
              </Alert>
            )}
            
            <form onSubmit={handleSubmit}>
              <Stack spacing={2}>
                <TextField
                  fullWidth
                  label={t('auth.username')}
                  value={credentials.username}
                  onChange={handleChange('username')}
                  required
                  autoComplete="username"
                />
                
                <TextField
                  fullWidth
                  type="password"
                  label={t('auth.password')}
                  value={credentials.password}
                  onChange={handleChange('password')}
                  required
                  autoComplete="current-password"
                />
                
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  size="large"
                  disabled={loading}
                >
                  {loading ? t('common.loading') : t('auth.login')}
                </Button>
              </Stack>
            </form>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Login;