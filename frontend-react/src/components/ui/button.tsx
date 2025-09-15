import React from 'react';
import { Button as MuiButton } from '@mui/material';

export const Button: React.FC<{ 
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
}> = ({ children, onClick, variant = 'default', size = 'default', disabled }) => {
  const getMuiVariant = () => {
    switch (variant) {
      case 'outline': return 'outlined';
      case 'secondary': return 'text';
      case 'ghost': return 'text';
      case 'link': return 'text';
      default: return 'contained';
    }
  };

  const getMuiSize = () => {
    switch (size) {
      case 'sm': return 'small';
      case 'lg': return 'large';
      default: return 'medium';
    }
  };

  return (
    <MuiButton 
      variant={getMuiVariant()}
      size={getMuiSize()}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </MuiButton>
  );
};