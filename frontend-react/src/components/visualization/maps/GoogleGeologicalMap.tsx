/**
 * Google Geological Map Component
 * Advanced Google Maps integration for geological data visualization
 * Replaces Mapbox implementation with comprehensive Google Maps features
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Switch, 
  FormControlLabel, 
  Slider, 
  Chip, 
  IconButton,
  Tooltip,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  Layers as LayersIcon,
  Satellite as SatelliteIcon,
  Terrain as TerrainIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { GeologicalData, VolcanicState, SeismicEvent } from '../../../types/geological';
import GoogleMapsService, { MapOptions, GeologicalLayerOptions } from '../../../services/GoogleMapsService';

interface GoogleGeologicalMapProps {
  data: GeologicalData[];
  volcanicStates?: VolcanicState[];
  seismicEvents?: SeismicEvent[];
  onFeatureClick?: (feature: any) => void;
  onViewportChange?: (viewport: { center: google.maps.LatLngLiteral; zoom: number }) => void;
  initialCenter?: google.maps.LatLngLiteral;
  initialZoom?: number;
  showRealTimeData?: boolean;
  geologicalLayers?: string[];
  enableAdvancedFeatures?: boolean;
  height?: number | string;
  width?: number | string;
}

// Color schemes handled by GoogleMapsService

// Map type configurations
const MAP_TYPES = {
  hybrid: { id: 'hybrid', name: 'Hybrid', icon: <SatelliteIcon /> },
  satellite: { id: 'satellite', name: 'Satellite', icon: <SatelliteIcon /> },
  terrain: { id: 'terrain', name: 'Terrain', icon: <TerrainIcon /> },
  roadmap: { id: 'roadmap', name: 'Roadmap', icon: <LayersIcon /> },
};

export const GoogleGeologicalMap: React.FC<GoogleGeologicalMapProps> = ({
  data,
  volcanicStates = [],
  seismicEvents = [],
  onViewportChange,
  initialCenter = { lat: 37.8, lng: -122.4 },
  initialZoom = 8,
  showRealTimeData = true,
  enableAdvancedFeatures = true,
  height = '100%',
  width = '100%'
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const googleMapsService = useRef<GoogleMapsService | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Layer visibility state
  const [layerVisibility, setLayerVisibility] = useState({
    geological: true,
    seismic: true,
    volcanic: true,
    terrain: true,
    satellite: false,
    heatmap: false,
    clustering: true,
    hazardZones: true,
    realTimeUpdates: showRealTimeData
  });

  // Map configuration state
  const [mapConfig, setMapConfig] = useState({
    mapType: 'hybrid' as keyof typeof MAP_TYPES,
    showTraffic: false,
    showTransit: false,
    show3D: enableAdvancedFeatures,
    tilt: enableAdvancedFeatures ? 45 : 0
  });

  // Time range and filtering
  const [timeRange, setTimeRange] = useState([0, 100]);
  const [magnitudeFilter, setMagnitudeFilter] = useState([0, 10]);
  const [selectedFeature, setSelectedFeature] = useState<any>(null);
  const [dataStats, setDataStats] = useState({
    geologicalPoints: 0,
    seismicEvents: 0,
    volcanicSystems: 0,
    lastUpdate: new Date()
  });

  // Initialize Google Maps
  useEffect(() => {
    const initializeMap = async () => {
      try {
        setIsLoading(true);
        setError(null);

        if (!mapContainerRef.current) return;

        // Get API key from environment
        const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
        const mapId = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID;

        if (!apiKey) {
          throw new Error('Google Maps API key not found in environment variables');
        }

        // Initialize Google Maps Service
        googleMapsService.current = GoogleMapsService.getInstance({
          apiKey,
          mapId,
          libraries: ['maps', 'marker', 'geometry', 'visualization', 'places', 'drawing']
        });

        // Create map options
        const mapOptions: MapOptions = {
          center: initialCenter,
          zoom: initialZoom,
          mapTypeId: google.maps.MapTypeId.HYBRID,
          mapId: mapId,
          gestureHandling: 'auto',
          zoomControl: true,
          mapTypeControl: true,
          scaleControl: true,
          streetViewControl: true,
          rotateControl: enableAdvancedFeatures,
          fullscreenControl: true,
        };

        // Create the map
        const mapInstance = await googleMapsService.current.createMap(
          mapContainerRef.current,
          mapOptions
        );

        setMap(mapInstance);

        // Add event listeners
        mapInstance.addListener('center_changed', () => {
          const center = mapInstance.getCenter();
          const zoom = mapInstance.getZoom();
          if (center && zoom && onViewportChange) {
            onViewportChange({
              center: { lat: center.lat(), lng: center.lng() },
              zoom
            });
          }
        });

        mapInstance.addListener('zoom_changed', () => {
          const center = mapInstance.getCenter();
          const zoom = mapInstance.getZoom();
          if (center && zoom && onViewportChange) {
            onViewportChange({
              center: { lat: center.lat(), lng: center.lng() },
              zoom
            });
          }
        });

        setIsLoading(false);
      } catch (err) {
        console.error('Failed to initialize Google Maps:', err);
        setError(err instanceof Error ? err.message : 'Failed to initialize map');
        setIsLoading(false);
      }
    };

    initializeMap();

    // Cleanup on unmount
    return () => {
      if (googleMapsService.current) {
        googleMapsService.current.cleanup();
      }
    };
  }, [initialCenter, initialZoom, enableAdvancedFeatures, onViewportChange]);

  // Update geological data
  useEffect(() => {
    const updateGeologicalData = async () => {
      if (!googleMapsService.current || !layerVisibility.geological) return;

      try {
        const layerOptions: GeologicalLayerOptions = {
          showGeological: layerVisibility.geological,
          enableClustering: layerVisibility.clustering,
          enableHeatmap: layerVisibility.heatmap,
          enableRealTimeUpdates: layerVisibility.realTimeUpdates
        };

        await googleMapsService.current.addGeologicalData(data, layerOptions);
        
        setDataStats(prev => ({
          ...prev,
          geologicalPoints: data.length,
          lastUpdate: new Date()
        }));
      } catch (err) {
        console.error('Failed to update geological data:', err);
      }
    };

    updateGeologicalData();
  }, [data, layerVisibility.geological, layerVisibility.clustering, layerVisibility.heatmap]);

  // Update seismic events
  useEffect(() => {
    const updateSeismicEvents = async () => {
      if (!googleMapsService.current || !layerVisibility.seismic) return;

      try {
        // Filter events by magnitude
        const filteredEvents = seismicEvents.filter(
          event => event.magnitude >= magnitudeFilter[0] && event.magnitude <= magnitudeFilter[1]
        );

        const layerOptions: GeologicalLayerOptions = {
          showSeismic: layerVisibility.seismic,
          enableClustering: layerVisibility.clustering,
          enableHeatmap: layerVisibility.heatmap,
          enableRealTimeUpdates: layerVisibility.realTimeUpdates
        };

        await googleMapsService.current.addSeismicEvents(filteredEvents, layerOptions);
        
        setDataStats(prev => ({
          ...prev,
          seismicEvents: filteredEvents.length,
          lastUpdate: new Date()
        }));
      } catch (err) {
        console.error('Failed to update seismic events:', err);
      }
    };

    updateSeismicEvents();
  }, [seismicEvents, layerVisibility.seismic, layerVisibility.clustering, layerVisibility.heatmap, magnitudeFilter]);

  // Update volcanic states
  useEffect(() => {
    const updateVolcanicStates = async () => {
      if (!googleMapsService.current || !layerVisibility.volcanic) return;

      try {
        const layerOptions: GeologicalLayerOptions = {
          showVolcanic: layerVisibility.volcanic,
          enableClustering: layerVisibility.clustering,
          enableRealTimeUpdates: layerVisibility.realTimeUpdates
        };

        await googleMapsService.current.addVolcanicStates(volcanicStates, layerOptions);
        
        setDataStats(prev => ({
          ...prev,
          volcanicSystems: volcanicStates.length,
          lastUpdate: new Date()
        }));
      } catch (err) {
        console.error('Failed to update volcanic states:', err);
      }
    };

    updateVolcanicStates();
  }, [volcanicStates, layerVisibility.volcanic, layerVisibility.clustering]);

  // Enable real-time updates
  useEffect(() => {
    if (!googleMapsService.current || !layerVisibility.realTimeUpdates) return;

    const updateCallback = async () => {
      // Refresh all data layers
      if (layerVisibility.geological && data.length > 0) {
        await googleMapsService.current!.addGeologicalData(data, {
          showGeological: true,
          enableClustering: layerVisibility.clustering,
          enableHeatmap: layerVisibility.heatmap,
          enableRealTimeUpdates: true
        });
      }

      if (layerVisibility.seismic && seismicEvents.length > 0) {
        const filteredEvents = seismicEvents.filter(
          event => event.magnitude >= magnitudeFilter[0] && event.magnitude <= magnitudeFilter[1]
        );
        await googleMapsService.current!.addSeismicEvents(filteredEvents, {
          showSeismic: true,
          enableClustering: layerVisibility.clustering,
          enableHeatmap: layerVisibility.heatmap,
          enableRealTimeUpdates: true
        });
      }

      if (layerVisibility.volcanic && volcanicStates.length > 0) {
        await googleMapsService.current!.addVolcanicStates(volcanicStates, {
          showVolcanic: true,
          enableClustering: layerVisibility.clustering,
          enableRealTimeUpdates: true
        });
      }

      setDataStats(prev => ({ ...prev, lastUpdate: new Date() }));
    };

    googleMapsService.current.enableRealTimeUpdates(updateCallback, 30000); // 30 seconds

    return () => {
      if (googleMapsService.current) {
        googleMapsService.current.disableRealTimeUpdates();
      }
    };
  }, [layerVisibility.realTimeUpdates, data, seismicEvents, volcanicStates, layerVisibility, magnitudeFilter]);

  // Toggle layer visibility
  const toggleLayer = useCallback((layerName: string) => {
    setLayerVisibility(prev => {
      const newVisibility = {
        ...prev,
        [layerName]: !prev[layerName as keyof typeof prev]
      };

      // Clear layer if being hidden
      if (!newVisibility[layerName as keyof typeof newVisibility] && googleMapsService.current) {
        googleMapsService.current.clearLayer(layerName);
      }

      return newVisibility;
    });
  }, []);

  // Change map type
  const changeMapType = useCallback((mapType: keyof typeof MAP_TYPES) => {
    if (!map) return;

    const mapTypeIds = {
      hybrid: google.maps.MapTypeId.HYBRID,
      satellite: google.maps.MapTypeId.SATELLITE,
      terrain: google.maps.MapTypeId.TERRAIN,
      roadmap: google.maps.MapTypeId.ROADMAP,
    };

    map.setMapTypeId(mapTypeIds[mapType]);
    setMapConfig(prev => ({ ...prev, mapType }));
  }, [map]);

  // Refresh data
  const refreshData = useCallback(async () => {
    if (!googleMapsService.current) return;

    try {
      setIsLoading(true);
      
      // Clear all layers
      googleMapsService.current.clearAllLayers();
      
      // Re-add all visible layers
      if (layerVisibility.geological && data.length > 0) {
        await googleMapsService.current.addGeologicalData(data, {
          showGeological: true,
          enableClustering: layerVisibility.clustering,
          enableHeatmap: layerVisibility.heatmap
        });
      }

      if (layerVisibility.seismic && seismicEvents.length > 0) {
        const filteredEvents = seismicEvents.filter(
          event => event.magnitude >= magnitudeFilter[0] && event.magnitude <= magnitudeFilter[1]
        );
        await googleMapsService.current.addSeismicEvents(filteredEvents, {
          showSeismic: true,
          enableClustering: layerVisibility.clustering,
          enableHeatmap: layerVisibility.heatmap
        });
      }

      if (layerVisibility.volcanic && volcanicStates.length > 0) {
        await googleMapsService.current.addVolcanicStates(volcanicStates, {
          showVolcanic: true,
          enableClustering: layerVisibility.clustering
        });
      }

      setDataStats(prev => ({ ...prev, lastUpdate: new Date() }));
      setIsLoading(false);
    } catch (err) {
      console.error('Failed to refresh data:', err);
      setIsLoading(false);
    }
  }, [data, seismicEvents, volcanicStates, layerVisibility, magnitudeFilter]);

  // Render loading state
  if (isLoading) {
    return (
      <Box 
        sx={{ 
          width, 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          backgroundColor: '#f5f5f5'
        }}
      >
        <Typography variant="h6">Loading Google Maps...</Typography>
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <Box 
        sx={{ 
          width, 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          backgroundColor: '#ffebee',
          color: '#c62828'
        }}
      >
        <Typography variant="h6">Error: {error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width, height, position: 'relative' }}>
      {/* Map Container */}
      <div
        ref={mapContainerRef}
        style={{
          width: '100%',
          height: '100%',
          borderRadius: '8px',
          overflow: 'hidden'
        }}
      />

      {/* Layer Controls Panel */}
      <Paper
        sx={{
          position: 'absolute',
          top: 16,
          left: 16,
          p: 2,
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          minWidth: 280,
          maxHeight: '80vh',
          overflowY: 'auto'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <LayersIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Map Controls</Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Tooltip title="Refresh Data">
            <IconButton size="small" onClick={refreshData}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Map Type Selection */}
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Map Type</InputLabel>
          <Select
            value={mapConfig.mapType}
            onChange={(e) => changeMapType(e.target.value as keyof typeof MAP_TYPES)}
            size="small"
          >
            {Object.entries(MAP_TYPES).map(([key, type]) => (
              <MenuItem key={key} value={key}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {type.icon}
                  <Typography sx={{ ml: 1 }}>{type.name}</Typography>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Layer Toggles */}
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Data Layers
        </Typography>
        
        {Object.entries(layerVisibility).map(([layer, visible]) => (
          <FormControlLabel
            key={layer}
            control={
              <Switch
                checked={visible}
                onChange={() => toggleLayer(layer)}
                size="small"
                color="primary"
              />
            }
            label={
              <Typography variant="body2">
                {layer.charAt(0).toUpperCase() + layer.slice(1).replace(/([A-Z])/g, ' $1')}
              </Typography>
            }
            sx={{ display: 'block', mb: 0.5 }}
          />
        ))}

        {/* Magnitude Filter */}
        <Typography variant="subtitle2" sx={{ mt: 2, mb: 1, fontWeight: 'bold' }}>
          Magnitude Filter
        </Typography>
        <Slider
          value={magnitudeFilter}
          onChange={(_: Event, value: number | number[]) => setMagnitudeFilter(value as number[])}
          valueLabelDisplay="auto"
          min={0}
          max={10}
          step={0.1}
          marks={[
            { value: 0, label: '0' },
            { value: 5, label: '5' },
            { value: 10, label: '10' }
          ]}
          sx={{ mb: 2 }}
        />

        {/* Time Range Control */}
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Time Range
        </Typography>
        <Slider
          value={timeRange}
          onChange={(_: Event, value: number | number[]) => setTimeRange(value as number[])}
          valueLabelDisplay="auto"
          min={0}
          max={100}
          sx={{ mb: 2 }}
        />

        {/* Data Statistics */}
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Data Statistics
        </Typography>
        <Grid container spacing={1}>
          <Grid item xs={12}>
            <Chip 
              label={`${dataStats.geologicalPoints} Geological Points`} 
              size="small" 
              color="primary"
              sx={{ mb: 0.5, mr: 0.5 }} 
            />
          </Grid>
          <Grid item xs={12}>
            <Chip 
              label={`${dataStats.seismicEvents} Seismic Events`} 
              size="small" 
              color="secondary"
              sx={{ mb: 0.5, mr: 0.5 }} 
            />
          </Grid>
          <Grid item xs={12}>
            <Chip 
              label={`${dataStats.volcanicSystems} Volcanoes`} 
              size="small" 
              color="warning"
              sx={{ mb: 0.5 }} 
            />
          </Grid>
        </Grid>

        <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
          Last updated: {dataStats.lastUpdate.toLocaleTimeString()}
        </Typography>
      </Paper>

      {/* Feature Information Panel */}
      {selectedFeature && (
        <Paper
          sx={{
            position: 'absolute',
            bottom: 16,
            right: 16,
            p: 2,
            maxWidth: 350,
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)'
          }}
        >
          <Typography variant="h6" gutterBottom>
            Feature Information
          </Typography>
          <Typography variant="body2">
            Type: {selectedFeature.properties?.data_type || 'Unknown'}
          </Typography>
          <Typography variant="body2">
            Location: {selectedFeature.geometry?.coordinates?.join(', ') || 'N/A'}
          </Typography>
          {selectedFeature.properties?.magnitude && (
            <Typography variant="body2">
              Magnitude: {selectedFeature.properties.magnitude}
            </Typography>
          )}
          {selectedFeature.properties?.alert_level && (
            <Typography variant="body2">
              Alert Level: {selectedFeature.properties.alert_level}
            </Typography>
          )}
          <Button
            size="small"
            onClick={() => setSelectedFeature(null)}
            sx={{ mt: 1 }}
          >
            Close
          </Button>
        </Paper>
      )}

      {/* Map Attribution */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 8,
          left: 8,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          px: 1,
          py: 0.5,
          borderRadius: 1,
          fontSize: '0.75rem'
        }}
      >
        <Typography variant="caption">
          Powered by Google Maps | Geological data from multiple sources
        </Typography>
      </Box>
    </Box>
  );
};

export default GoogleGeologicalMap;