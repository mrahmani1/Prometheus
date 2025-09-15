import React from 'react';
import { Tabs as MuiTabs, Tab, Box } from '@mui/material';

export const Tabs: React.FC<{ 
  children: React.ReactNode;
  value?: string;
  onValueChange?: (value: string) => void;
  defaultValue?: string;
  className?: string;
}> = ({ children, value, onValueChange, defaultValue, className }) => {
  const [tabValue, setTabValue] = React.useState(value || defaultValue || '0');

  const handleChange = (_event: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
    onValueChange?.(newValue);
  };

  return (
    <Box className={className}>
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && child.type === TabsList) {
          return React.cloneElement(child, { value: tabValue, onChange: handleChange });
        }
        if (React.isValidElement(child) && child.type === TabsContent) {
          return child;
        }
        return child;
      })}
    </Box>
  );
};

export const TabsList: React.FC<{ 
  children: React.ReactNode;
  value?: string;
  onChange?: (event: React.SyntheticEvent, newValue: string) => void;
  className?: string;
}> = ({ children, value, onChange, className }) => (
  <MuiTabs value={value} onChange={onChange} className={className}>
    {children}
  </MuiTabs>
);

export const TabsTrigger: React.FC<{ 
  children: React.ReactNode;
  value: string;
}> = ({ children, value }) => (
  <Tab label={children} value={value} />
);

export const TabsContent: React.FC<{ 
  children: React.ReactNode;
  value: string;
  className?: string;
}> = ({ children, value, className }) => (
  <Box role="tabpanel" id={`tabpanel-${value}`} className={className}>
    {children}
  </Box>
);