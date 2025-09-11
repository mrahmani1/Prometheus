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
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock successful login
      dispatch(loginSuccess({
        token: 'mock-jwt-token',
        refreshToken: 'mock-refresh-token',
        user: {
          id: '1',
          username: credentials.username,
          email: 'user@example.com',
          firstName: 'John',
          lastName: 'Doe',
          roles: [],
          permissions: [],
          preferences: {
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
          lastLogin: new Date().toISOString(),
        },
        expiresIn: 3600,
      }));
    } catch (error) {
      dispatch(loginFailure('Invalid credentials'));
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