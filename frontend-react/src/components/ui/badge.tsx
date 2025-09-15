import React from 'react';
import { Chip } from '@mui/material';

export const Badge: React.FC<{ 
  children: React.ReactNode;
  variant?: 'default' | 'secondary' | 'destructive' | 'outline';
  className?: string;
}> = ({ children, variant = 'default', className }) => {
  const getColor = () => {
    switch (variant) {
      case 'secondary': return 'secondary';
      case 'destructive': return 'error';
      case 'outline': return 'default';
      default: return 'primary';
    }
  };

  return (
    <Chip 
      label={children} 
      color={getColor() as any}
      size="small"
      className={className}
    />
  );
};