/**
 * Error boundary fallback component
 */

import React from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Alert,
  Stack,
} from '@mui/material';
import { ErrorOutline, Refresh } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetErrorBoundary,
}) => {
  const theme = useTheme();

  const handleReload = () => {
    window.location.reload();
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        padding: theme.spacing(3),
        backgroundColor: theme.palette.background.default,
      }}
    >
      <Card sx={{ maxWidth: 600, width: '100%' }}>
        <CardContent>
          <Stack spacing={3} alignItems="center">
            <ErrorOutline
              sx={{
                fontSize: 64,
                color: theme.palette.error.main,
              }}
            />
            
            <Typography variant="h4" component="h1" textAlign="center">
              Something went wrong
            </Typography>
            
            <Typography variant="body1" textAlign="center" color="textSecondary">
              We're sorry, but an unexpected error occurred. Our team has been notified
              and is working to fix the issue.
            </Typography>
            
            <Alert severity="error" sx={{ width: '100%' }}>
              <Typography variant="body2">
                <strong>Error:</strong> {error.message}
              </Typography>
            </Alert>
            
            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={resetErrorBoundary}
              >
                Try Again
              </Button>
              
              <Button
                variant="outlined"
                onClick={handleReload}
              >
                Reload Page
              </Button>
            </Stack>
            
            {process.env.NODE_ENV === 'development' && (
              <Box
                sx={{
                  width: '100%',
                  mt: 2,
                  p: 2,
                  backgroundColor: theme.palette.grey[100],
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  overflow: 'auto',
                  maxHeight: 200,
                }}
              >
                <Typography variant="caption" display="block" gutterBottom>
                  Stack Trace (Development Only):
                </Typography>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {error.stack}
                </pre>
              </Box>
            )}
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ErrorFallback;