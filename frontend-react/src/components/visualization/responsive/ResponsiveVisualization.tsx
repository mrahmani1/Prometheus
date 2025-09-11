import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, useTheme, useMediaQuery } from '@mui/material';
import { GeologicalData, SeismicEvent, VolcanicState } from '../../../types/geological';

interface ResponsiveVisualizationProps {
  data: (GeologicalData | SeismicEvent | VolcanicState)[];
  children: (dimensions: ViewportDimensions, isMobile: boolean, isTablet: boolean) => React.ReactNode;
  minHeight?: number;
  maxHeight?: number;
  aspectRatio?: number;
  breakpoints?: {
    mobile: number;
    tablet: number;
    desktop: number;
  };
}

interface ViewportDimensions {
  width: number;
  height: number;
  availableWidth: number;
  availableHeight: number;
}

// Default breakpoints for geological visualization
const DEFAULT_BREAKPOINTS = {
  mobile: 768,
  tablet: 1024,
  desktop: 1440
};

// Responsive scaling factors for different geological visualization elements
const SCALING_FACTORS = {
  mobile: {
    fontSize: 0.8,
    pointSize: 0.7,
    lineWidth: 0.8,
    margin: 0.5,
    controlSize: 1.2
  },
  tablet: {
    fontSize: 0.9,
    pointSize: 0.85,
    lineWidth: 0.9,
    margin: 0.75,
    controlSize: 1.1
  },
  desktop: {
    fontSize: 1.0,
    pointSize: 1.0,
    lineWidth: 1.0,
    margin: 1.0,
    controlSize: 1.0
  }
};

export const ResponsiveVisualization: React.FC<ResponsiveVisualizationProps> = ({
  data,
  children,
  minHeight = 300,
  maxHeight = 800,
  aspectRatio,
  breakpoints = DEFAULT_BREAKPOINTS
}) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [dimensions, setDimensions] = useState<ViewportDimensions>({
    width: 800,
    height: 600,
    availableWidth: 800,
    availableHeight: 600
  });

  // Media queries for responsive behavior
  const isMobile = useMediaQuery(`(max-width:${breakpoints.mobile}px)`);
  const isTablet = useMediaQuery(`(max-width:${breakpoints.tablet}px) and (min-width:${breakpoints.mobile + 1}px)`);
  const isDesktop = useMediaQuery(`(min-width:${breakpoints.desktop}px)`);

  // Calculate optimal dimensions based on container and device
  const calculateDimensions = useCallback(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const containerRect = container.getBoundingClientRect();
    const containerWidth = containerRect.width;
    
    // Calculate available space accounting for padding and margins
    const padding = isMobile ? 16 : isTablet ? 24 : 32;
    const availableWidth = containerWidth - (padding * 2);
    
    // Calculate height based on aspect ratio or constraints
    let calculatedHeight: number;
    
    if (aspectRatio) {
      calculatedHeight = availableWidth / aspectRatio;
    } else {
      // Use responsive height calculation
      if (isMobile) {
        calculatedHeight = Math.min(availableWidth * 0.75, window.innerHeight * 0.4);
      } else if (isTablet) {
        calculatedHeight = Math.min(availableWidth * 0.6, window.innerHeight * 0.5);
      } else {
        calculatedHeight = Math.min(availableWidth * 0.5, window.innerHeight * 0.6);
      }
    }
    
    // Apply min/max height constraints
    calculatedHeight = Math.max(minHeight, Math.min(maxHeight, calculatedHeight));
    
    setDimensions({
      width: availableWidth,
      height: calculatedHeight,
      availableWidth,
      availableHeight: calculatedHeight
    });
  }, [isMobile, isTablet, aspectRatio, minHeight, maxHeight]);

  // Resize observer for responsive updates
  useEffect(() => {
    const resizeObserver = new ResizeObserver(() => {
      calculateDimensions();
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    // Initial calculation
    calculateDimensions();

    return () => {
      resizeObserver.disconnect();
    };
  }, [calculateDimensions]);

  // Window resize handler for orientation changes
  useEffect(() => {
    const handleResize = () => {
      setTimeout(calculateDimensions, 100); // Debounce for orientation changes
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, [calculateDimensions]);

  // Get current device type for styling
  const getDeviceType = () => {
    if (isMobile) return 'mobile';
    if (isTablet) return 'tablet';
    return 'desktop';
  };

  // Get scaling factors for current device
  const getScalingFactors = () => {
    return SCALING_FACTORS[getDeviceType()];
  };

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: 'auto',
        position: 'relative',
        // Responsive padding
        p: isMobile ? 1 : isTablet ? 2 : 3,
        // Ensure proper touch handling on mobile
        touchAction: isMobile ? 'pan-x pan-y' : 'auto',
        // Optimize for performance on mobile
        willChange: isMobile ? 'transform' : 'auto',
        // Prevent text selection on mobile during interactions
        userSelect: isMobile ? 'none' : 'auto',
        // Responsive overflow handling
        overflow: 'hidden',
        // Smooth transitions for responsive changes
        transition: theme.transitions.create(['padding', 'height'], {
          duration: theme.transitions.duration.short
        })
      }}
    >
      {/* Render children with responsive context */}
      {children(
        {
          ...dimensions,
          // Add scaling factors to dimensions
          scalingFactors: getScalingFactors(),
          deviceType: getDeviceType()
        } as any,
        isMobile,
        isTablet
      )}
    </Box>
  );
};

// Hook for responsive geological visualization settings
export const useResponsiveGeologicalVisualization = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isTablet = useMediaQuery(theme.breakpoints.between('md', 'lg'));
  
  const getResponsiveSettings = useCallback(() => {
    const deviceType = isMobile ? 'mobile' : isTablet ? 'tablet' : 'desktop';
    const scalingFactors = SCALING_FACTORS[deviceType];
    
    return {
      // Chart settings
      chart: {
        fontSize: 12 * scalingFactors.fontSize,
        pointSize: 4 * scalingFactors.pointSize,
        lineWidth: 2 * scalingFactors.lineWidth,
        margin: {
          top: 20 * scalingFactors.margin,
          right: 30 * scalingFactors.margin,
          bottom: 40 * scalingFactors.margin,
          left: 50 * scalingFactors.margin
        }
      },
      
      // Map settings
      map: {
        zoom: isMobile ? 6 : isTablet ? 8 : 10,
        pitch: isMobile ? 0 : 45,
        bearing: 0,
        controls: {
          showCompass: !isMobile,
          showZoom: true,
          showFullscreen: !isMobile,
          showNavigation: !isMobile
        }
      },
      
      // 3D visualization settings
      visualization3D: {
        cameraDistance: isMobile ? 15 : isTablet ? 12 : 10,
        fieldOfView: isMobile ? 50 : 60,
        enableOrbitControls: true,
        enablePanControls: !isMobile,
        enableZoomControls: true,
        maxDistance: isMobile ? 50 : 100,
        minDistance: isMobile ? 5 : 1
      },
      
      // Grid settings
      grid: {
        pageSize: isMobile ? 10 : isTablet ? 25 : 50,
        density: isMobile ? 'compact' : 'standard',
        showToolbar: !isMobile,
        showFilters: !isMobile,
        columnWidth: isMobile ? 100 : isTablet ? 120 : 150
      },
      
      // Touch and interaction settings
      interaction: {
        enableTouch: isMobile,
        touchSensitivity: isMobile ? 1.5 : 1.0,
        hoverDelay: isMobile ? 0 : 300,
        clickDelay: isMobile ? 150 : 0,
        enableGestures: isMobile,
        enableKeyboard: !isMobile
      },
      
      // Performance settings
      performance: {
        enableVirtualization: true,
        maxDataPoints: isMobile ? 1000 : isTablet ? 5000 : 10000,
        renderQuality: isMobile ? 'medium' : 'high',
        enableWebGL: !isMobile,
        enableWorkers: !isMobile,
        debounceDelay: isMobile ? 300 : 100
      },
      
      // Layout settings
      layout: {
        showSidebar: !isMobile,
        sidebarWidth: isMobile ? '100%' : isTablet ? 300 : 350,
        showLegend: true,
        legendPosition: isMobile ? 'bottom' : 'right',
        showControls: true,
        controlsPosition: isMobile ? 'bottom' : 'top-right'
      },
      
      // Device info
      device: {
        type: deviceType,
        isMobile,
        isTablet,
        isDesktop: !isMobile && !isTablet,
        scalingFactors,
        supportsHover: !isMobile,
        supportsTouch: isMobile,
        hasHighDPI: window.devicePixelRatio > 1
      }
    };
  }, [isMobile, isTablet, theme]);
  
  return getResponsiveSettings();
};

// Utility component for responsive geological data visualization
export const ResponsiveGeologicalContainer: React.FC<{
  children: React.ReactNode;
  enableFullscreen?: boolean;
  enableExport?: boolean;
}> = ({ 
  children, 
  enableFullscreen = true, 
  enableExport = true 
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const settings = useResponsiveGeologicalVisualization();
  
  const toggleFullscreen = useCallback(() => {
    if (!enableFullscreen) return;
    
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, [enableFullscreen]);
  
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);
  
  return (
    <Box
      sx={{
        width: '100%',
        height: isFullscreen ? '100vh' : 'auto',
        position: isFullscreen ? 'fixed' : 'relative',
        top: isFullscreen ? 0 : 'auto',
        left: isFullscreen ? 0 : 'auto',
        zIndex: isFullscreen ? 9999 : 'auto',
        backgroundColor: isFullscreen ? 'background.default' : 'transparent',
        // Responsive container styles
        ...settings.device.isMobile && {
          '& .geological-visualization': {
            fontSize: settings.chart.fontSize,
            '& .point': {
              r: settings.chart.pointSize
            },
            '& .line': {
              strokeWidth: settings.chart.lineWidth
            }
          }
        }
      }}
    >
      {children}
    </Box>
  );
};

export default ResponsiveVisualization;