import React from 'react';
import {
  Backdrop,
  CircularProgress,
  Box,
  Typography,
  LinearProgress,
} from '@mui/material';

interface LoadingOverlayProps {
  message?: string;
  progress?: number;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  message,
  progress,
}) => {
  return (
    <Backdrop
      sx={{
        color: '#fff',
        zIndex: (theme: any) => theme.zIndex.drawer + 1,
        flexDirection: 'column',
      }}
      open={true}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2,
          minWidth: 300,
        }}
      >
        {progress !== undefined ? (
          <Box sx={{ width: '100%' }}>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
              {progress}% Complete
            </Typography>
          </Box>
        ) : (
          <CircularProgress size={60} thickness={4} />
        )}
        
        {message && (
          <Typography variant="h6" component="div" sx={{ textAlign: 'center' }}>
            {message}
          </Typography>
        )}
      </Box>
    </Backdrop>
  );
};