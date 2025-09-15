import React from 'react';
import { Card as MuiCard, CardHeader as MuiCardHeader, CardContent as MuiCardContent } from '@mui/material';
import { Typography } from '@mui/material';

export const Card = MuiCard;

export const CardHeader: React.FC<{ 
  children: React.ReactNode;
  className?: string;
}> = ({ children, className }) => (
  <MuiCardHeader title={children} className={className} />
);

export const CardTitle: React.FC<{ 
  children: React.ReactNode;
  className?: string;
}> = ({ children, className }) => (
  <Typography variant="h6" component="h2" className={className}>
    {children}
  </Typography>
);

export const CardContent = MuiCardContent;