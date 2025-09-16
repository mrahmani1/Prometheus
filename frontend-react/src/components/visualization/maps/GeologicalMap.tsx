import React, { useCallback, useMemo } from 'react';
import { GeologicalData, VolcanicState, SeismicEvent } from '../../../types/geological';
import GoogleGeologicalMap from './GoogleGeologicalMap';

interface GeologicalMapProps {
  data: GeologicalData[];
  volcanicStates?: VolcanicState[];
  seismicEvents?: SeismicEvent[];
  onFeatureClick?: (feature: any) => void;
  onViewportChange?: (viewport: { center: google.maps.LatLngLiteral; zoom: number }) => void;
  initialViewport?: { center: google.maps.LatLngLiteral; zoom: number };
  showRealTimeData?: boolean;
  geologicalLayers?: string[];
  height?: number | string;
  width?: number | string;
}

// Legacy component now uses Google Maps implementation

export const GeologicalMap: React.FC<GeologicalMapProps> = ({
  data,
  volcanicStates = [],
  seismicEvents = [],
  onFeatureClick,
  onViewportChange,
  initialViewport,
  showRealTimeData = true,
  geologicalLayers = ['geological', 'seismic', 'volcanic'],
  height = '100%',
  width = '100%'
}) => {
  // Convert legacy viewport format to Google Maps format
  const googleMapsCenter = useMemo(() => {
    if (initialViewport?.center) {
      return initialViewport.center;
    }
    return { lat: 37.8, lng: -122.4 };
  }, [initialViewport]);

  const googleMapsZoom = useMemo(() => {
    return initialViewport?.zoom || 8;
  }, [initialViewport]);

  // Convert legacy viewport change handler
  const handleGoogleMapsViewportChange = useCallback((viewport: { center: google.maps.LatLngLiteral; zoom: number }) => {
    if (onViewportChange) {
      onViewportChange(viewport);
    }
  }, [onViewportChange]);

  return (
    <GoogleGeologicalMap
      data={data}
      volcanicStates={volcanicStates}
      seismicEvents={seismicEvents}
      onFeatureClick={onFeatureClick}
      onViewportChange={handleGoogleMapsViewportChange}
      initialCenter={googleMapsCenter}
      initialZoom={googleMapsZoom}
      showRealTimeData={showRealTimeData}
      geologicalLayers={geologicalLayers}
      enableAdvancedFeatures={true}
      height={height}
      width={width}
    />
  );
};

export default GeologicalMap;