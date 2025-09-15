import React from 'react';
import { Alert as MuiAlert } from '@mui/material';

export const Alert: React.FC<{ 
  children: React.ReactNode; 
  severity?: 'error' | 'warning' | 'info' | 'success';
  className?: string;
}> = ({ children, severity = 'info', className }) => (
  <MuiAlert severity={severity} className={className}>
    {children}
  </MuiAlert>
);

export const AlertDescription: React.FC<{ 
  children: React.ReactNode;
  className?: string;
}> = ({ children, className }) => (
  <div className={className}>{children}</div>
);