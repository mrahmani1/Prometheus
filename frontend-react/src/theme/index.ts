/**
 * Material-UI theme configuration for Prometheus AI Platform
 * Implements geological domain-specific design system
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';

// Geological color palette
const geologicalColors = {
  // Primary geological colors
  igneous: '#8B4513', // Saddle brown for igneous rocks
  sedimentary: '#D2B48C', // Tan for sedimentary rocks
  metamorphic: '#708090', // Slate gray for metamorphic rocks
  
  // Volcanic alert levels
  volcanic: {
    green: '#4CAF50',
    yellow: '#FFC107',
    orange: '#FF9800',
    red: '#F44336',
  },
  
  // Seismic magnitude colors
  seismic: {
    minor: '#81C784',
    light: '#FFB74D',
    moderate: '#FF8A65',
    strong: '#E57373',
    major: '#F06292',
    great: '#BA68C8',
  },
  
  // Deformation colors
  deformation: {
    uplift: '#FF5722',
    subsidence: '#2196F3',
    lateral: '#9C27B0',
  },
  
  // Resource colors
  resources: {
    copper: '#B87333',
    gold: '#FFD700',
    silver: '#C0C0C0',
    lithium: '#CC99FF',
    geothermal: '#FF6B35',
  },
};

// Base theme configuration
const baseTheme: ThemeOptions = {
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
      contrastText: '#ffffff',
    },
    error: {
      main: '#f44336',
      light: '#e57373',
      dark: '#d32f2f',
    },
    warning: {
      main: '#ff9800',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    info: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    success: {
      main: '#4caf50',
      light: '#81c784',
      dark: '#388e3c',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.6,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
    },
  },
  shape: {
    borderRadius: 8,
  },
  spacing: 8,
};

// Light theme
export const lightTheme = createTheme({
  ...baseTheme,
  palette: {
    ...baseTheme.palette,
    mode: 'light',
    background: {
      default: '#fafafa',
      paper: '#ffffff',
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.6)',
      disabled: 'rgba(0, 0, 0, 0.38)',
    },
    divider: 'rgba(0, 0, 0, 0.12)',
    // Custom geological colors
    geological: geologicalColors,
  } as any,
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          color: 'rgba(0, 0, 0, 0.87)',
          boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.12)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#f5f5f5',
          borderRight: '1px solid rgba(0, 0, 0, 0.12)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.15)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
});

// Dark theme
export const darkTheme = createTheme({
  ...baseTheme,
  palette: {
    ...baseTheme.palette,
    mode: 'dark',
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
      disabled: 'rgba(255, 255, 255, 0.5)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
    // Custom geological colors (adjusted for dark theme)
    geological: {
      ...geologicalColors,
      igneous: '#CD853F',
      sedimentary: '#F5DEB3',
      metamorphic: '#B0C4DE',
    },
  } as any,
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#1e1e1e',
          color: '#ffffff',
          boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#2d2d2d',
          borderRight: '1px solid rgba(255, 255, 255, 0.12)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#2d2d2d',
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)',
          '&:hover': {
            boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.4)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
});

// Geological visualization color schemes
export const geologicalColorSchemes = {
  // Seismic color schemes
  seismic: {
    magnitude: ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'],
    depth: ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b'],
    frequency: ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837'],
  },
  
  // Volcanic color schemes
  volcanic: {
    temperature: ['#000080', '#0000ff', '#00ffff', '#ffff00', '#ff8000', '#ff0000', '#800000'],
    gasComposition: ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'],
    alertLevel: ['#4caf50', '#ffc107', '#ff9800', '#f44336'],
  },
  
  // Deformation color schemes
  deformation: {
    displacement: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
    velocity: ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'],
    strain: ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'],
  },
  
  // Geological color schemes
  geological: {
    age: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
    lithology: ['#8b4513', '#d2b48c', '#708090', '#daa520', '#cd853f', '#f4a460', '#bc8f8f', '#d2691e', '#b8860b', '#a0522d'],
    structure: ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#800080', '#ffa500', '#008000', '#000080'],
  },
  
  // Resource color schemes
  resources: {
    prospectivity: ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081'],
    grade: ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000'],
    confidence: ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#980043', '#67001f'],
  },
};

// Accessibility enhancements
export const accessibilityTheme = (baseTheme: any) => createTheme({
  ...baseTheme,
  components: {
    ...baseTheme.components,
    MuiButton: {
      styleOverrides: {
        root: {
          ...baseTheme.components?.MuiButton?.styleOverrides?.root,
          minHeight: 44, // WCAG AA minimum touch target
          '&:focus': {
            outline: `2px solid ${baseTheme.palette.primary.main}`,
            outlineOffset: 2,
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          minHeight: 44,
          minWidth: 44,
          '&:focus': {
            outline: `2px solid ${baseTheme.palette.primary.main}`,
            outlineOffset: 2,
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiInputBase-root': {
            minHeight: 44,
          },
          '& .MuiInputBase-input:focus': {
            outline: `2px solid ${baseTheme.palette.primary.main}`,
            outlineOffset: -2,
          },
        },
      },
    },
  },
});

// High contrast theme
export const highContrastTheme = createTheme({
  ...baseTheme,
  palette: {
    ...baseTheme.palette,
    mode: 'dark',
    primary: {
      main: '#ffffff',
      contrastText: '#000000',
    },
    secondary: {
      main: '#ffff00',
      contrastText: '#000000',
    },
    background: {
      default: '#000000',
      paper: '#000000',
    },
    text: {
      primary: '#ffffff',
      secondary: '#ffffff',
    },
    divider: '#ffffff',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          border: '2px solid #ffffff',
          '&:hover': {
            backgroundColor: '#ffffff',
            color: '#000000',
          },
        },
      },
    },
  },
});

export default lightTheme;