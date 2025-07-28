import React from 'react';
import { Box, Typography, Button, Paper, Alert } from '@mui/material';
import { Lock as LockIcon, Home as HomeIcon } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

export const UnauthorizedPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const requiredPermissions = location.state?.requiredPermissions || [];

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
      <Paper sx={{ p: 4, textAlign: 'center', maxWidth: 500 }}>
        <LockIcon sx={{ fontSize: '4rem', color: 'warning.main', mb: 2 }} />
        
        <Typography variant="h4" component="h1" gutterBottom>
          Access Denied
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          You don't have permission to access this resource.
        </Typography>

        {requiredPermissions.length > 0 && (
          <Alert severity="info" sx={{ mb: 3, textAlign: 'left' }}>
            <Typography variant="subtitle2" gutterBottom>
              Required permissions:
            </Typography>
            <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
              {requiredPermissions.map((permission: string, index: number) => (
                <li key={index}>
                  <Typography variant="body2">{permission}</Typography>
                </li>
              ))}
            </ul>
          </Alert>
        )}

        <Typography variant="body2" color="text.secondary" paragraph>
          Please contact your administrator if you believe you should have access to this resource.
        </Typography>

        <Button
          variant="contained"
          startIcon={<HomeIcon />}
          onClick={() => navigate('/dashboard')}
          sx={{ mt: 2 }}
        >
          Go to Dashboard
        </Button>
      </Paper>
    </Box>
  );
};