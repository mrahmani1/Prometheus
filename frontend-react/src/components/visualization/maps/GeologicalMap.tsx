import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import Map, { 
  Source, 
  Layer, 
  MapRef, 
  ViewState, 
  LayerProps 
} from 'react-map-gl';
import mapboxgl from 'mapbox-gl';
import { Box, Paper, Typography, Switch, FormControlLabel, Slider, Chip } from '@mui/material';
import { GeologicalData, VolcanicState, SeismicEvent } from '../../../types/geological';
// import proj4 from 'proj4'; // Unused
// import * as turf from '@turf/turf'; // Temporarily disabled due to type issues

// Mapbox access token - should be in environment variables
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'your-mapbox-token';

interface GeologicalMapProps {
  data: GeologicalData[];
  volcanicStates?: VolcanicState[];
  seismicEvents?: SeismicEvent[];
  onFeatureClick?: (feature: any) => void;
  onViewportChange?: (viewport: ViewState) => void;
  initialViewport?: Partial<ViewState>;
  showRealTimeData?: boolean;
  geologicalLayers?: string[];
}

// Advanced geological color schemes for different data types
const GEOLOGICAL_COLORS = {
  igneous: '#FF6B6B',
  sedimentary: '#4ECDC4',
  metamorphic: '#45B7D1',
  volcanicRock: '#FF8E53',
  fault: '#FF4757',
  seismic: {
    magnitude1: '#FFF3CD',
    magnitude2: '#FFE69C',
    magnitude3: '#FFD93D',
    magnitude4: '#FFC107',
    magnitude5: '#FF8F00',
    magnitude6: '#FF5722',
    magnitude7: '#D32F2F',
    magnitude8: '#B71C1C'
  },
  volcanic: {
    green: '#4CAF50',
    yellow: '#FF9800', 
    orange: '#FF5722',
    red: '#F44336'
  }
};

// WebGL custom layer for advanced geological rendering
class GeologicalWebGLLayer {
  id: string;
  type: 'custom';
  renderingMode: '3d';
  program: WebGLProgram | null = null;
  buffer: WebGLBuffer | null = null;
  
  constructor(id: string) {
    this.id = id;
    this.type = 'custom';
    this.renderingMode = '3d';
  }

  onAdd(_map: mapboxgl.Map, gl: WebGLRenderingContext) {
    // Vertex shader for geological data
    const vertexSource = `
      uniform mat4 u_matrix;
      uniform float u_time;
      attribute vec2 a_pos;
      attribute float a_elevation;
      attribute vec3 a_color;
      varying vec3 v_color;
      varying float v_elevation;
      
      void main() {
        v_color = a_color;
        v_elevation = a_elevation;
        
        // Apply geological deformation based on time
        vec2 deformed_pos = a_pos;
        deformed_pos.x += sin(a_pos.y * 0.001 + u_time * 0.001) * 0.0001;
        
        gl_Position = u_matrix * vec4(deformed_pos, a_elevation * 0.001, 1.0);
        gl_PointSize = 3.0 + a_elevation * 0.01;
      }
    `;

    // Fragment shader for geological visualization
    const fragmentSource = `
      precision mediump float;
      varying vec3 v_color;
      varying float v_elevation;
      uniform float u_opacity;
      
      void main() {
        // Apply elevation-based shading
        float elevation_factor = clamp(v_elevation / 5000.0, 0.0, 1.0);
        vec3 final_color = mix(v_color, vec3(1.0), elevation_factor * 0.3);
        
        gl_FragColor = vec4(final_color, u_opacity);
      }
    `;

    // Create and compile shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);

    // Create program
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);

    // Create buffer
    this.buffer = gl.createBuffer();
  }

  render(gl: WebGLRenderingContext, matrix: number[]) {
    if (!this.program) return;

    gl.useProgram(this.program);
    
    // Set uniforms
    const matrixLocation = gl.getUniformLocation(this.program, 'u_matrix');
    const timeLocation = gl.getUniformLocation(this.program, 'u_time');
    const opacityLocation = gl.getUniformLocation(this.program, 'u_opacity');
    
    gl.uniformMatrix4fv(matrixLocation, false, matrix);
    gl.uniform1f(timeLocation, Date.now());
    gl.uniform1f(opacityLocation, 0.8);

    // Render geological data points
    gl.drawArrays(gl.POINTS, 0, 1000); // Placeholder count
  }
}

// Coordinate transformation utilities
// const transformCoordinates = (coords: [number, number], fromProj: string, toProj: string = 'EPSG:4326') => { // Unused
//   try {
//     return proj4(fromProj, toProj, coords);
//   } catch (error) {
//     console.warn('Coordinate transformation failed:', error);
//     return coords;
//   }
// };

// Generate GeoJSON from geological data
const generateGeologicalGeoJSON = (data: GeologicalData[]): GeoJSON.FeatureCollection => {
  return {
    type: 'FeatureCollection',
    features: data.map((item, index) => ({
      type: 'Feature',
      id: item.id || index,
      geometry: {
        type: 'Point',
        coordinates: [item.location.longitude || 0, item.location.latitude || 0]
      },
      properties: {
        ...item,
        color: GEOLOGICAL_COLORS[item.dataType as keyof typeof GEOLOGICAL_COLORS] || '#888888'
      }
    }))
  };
};

// Generate seismic events GeoJSON
const generateSeismicGeoJSON = (events: SeismicEvent[]): GeoJSON.FeatureCollection => {
  return {
    type: 'FeatureCollection',
    features: events.map((event, index) => {
      const magnitude = Math.floor(event.magnitude);
      const colorKey = `magnitude${Math.min(magnitude, 8)}` as keyof typeof GEOLOGICAL_COLORS.seismic;
      
      return {
        type: 'Feature',
        id: event.eventId || index,
        geometry: {
          type: 'Point',
          coordinates: [event.location.x, event.location.y]
        },
        properties: {
          ...event,
          color: GEOLOGICAL_COLORS.seismic[colorKey],
          radius: Math.max(event.magnitude * 2, 3)
        }
      };
    })
  };
};

// Generate volcanic states GeoJSON
const generateVolcanicGeoJSON = (states: VolcanicState[]): GeoJSON.FeatureCollection => {
  return {
    type: 'FeatureCollection',
    features: states.map((state, index) => ({
      type: 'Feature',
      id: state.volcanoId || index,
      geometry: {
        type: 'Point',
        coordinates: [0, 0] // TODO: Add location property to VolcanicState interface
      },
      properties: {
        ...state,
        color: GEOLOGICAL_COLORS.volcanic[state.alertLevel] || GEOLOGICAL_COLORS.volcanic.green
      }
    }))
  };
};

export const GeologicalMap: React.FC<GeologicalMapProps> = ({
  data,
  volcanicStates = [],
  seismicEvents = [],
  onFeatureClick,
  onViewportChange,
  initialViewport,
  // showRealTimeData = true, // Unused
  // geologicalLayers = ['geological', 'seismic', 'volcanic'] // Unused
}) => {
  const mapRef = useRef<MapRef>(null);
  const [viewport, setViewport] = useState<ViewState>({
    longitude: -122.4,
    latitude: 37.8,
    zoom: 8,
    bearing: 0,
    pitch: 0,
    padding: { top: 0, bottom: 0, left: 0, right: 0 },
    ...initialViewport
  });

  const [layerVisibility, setLayerVisibility] = useState({
    geological: true,
    seismic: true,
    volcanic: true,
    terrain: true,
    satellite: false
  });

  const [timeRange, setTimeRange] = useState([0, 100]);
  const [selectedFeature, setSelectedFeature] = useState<any>(null);

  // Generate GeoJSON data
  const geologicalGeoJSON = useMemo(() => generateGeologicalGeoJSON(data), [data]);
  const seismicGeoJSON = useMemo(() => generateSeismicGeoJSON(seismicEvents), [seismicEvents]);
  const volcanicGeoJSON = useMemo(() => generateVolcanicGeoJSON(volcanicStates), [volcanicStates]);

  // Layer definitions with advanced styling
  const geologicalLayer: LayerProps = {
    id: 'geological-points',
    type: 'circle',
    paint: {
      'circle-radius': [
        'interpolate',
        ['linear'],
        ['zoom'],
        5, 3,
        15, 8
      ],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.8,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#ffffff'
    }
  };

  const seismicLayer: LayerProps = {
    id: 'seismic-events',
    type: 'circle',
    paint: {
      'circle-radius': [
        'interpolate',
        ['exponential', 2],
        ['get', 'magnitude'],
        1, 3,
        8, 30
      ],
      'circle-color': ['get', 'color'],
      'circle-opacity': [
        'interpolate',
        ['linear'],
        ['get', 'magnitude'],
        1, 0.3,
        8, 0.8
      ],
      'circle-stroke-width': 2,
      'circle-stroke-color': '#ffffff'
    }
  };

  const volcanicLayer: LayerProps = {
    id: 'volcanic-states',
    type: 'circle',
    paint: {
      'circle-radius': [
        'case',
        ['==', ['get', 'alert_level'], 'critical'], 15,
        ['==', ['get', 'alert_level'], 'warning'], 12,
        ['==', ['get', 'alert_level'], 'watch'], 10,
        ['==', ['get', 'alert_level'], 'advisory'], 8,
        6
      ],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.7,
      'circle-stroke-width': 2,
      'circle-stroke-color': '#ffffff'
    }
  };

  // Heat map layer for seismic activity
  const seismicHeatmapLayer: LayerProps = {
    id: 'seismic-heatmap',
    type: 'heatmap',
    paint: {
      'heatmap-weight': [
        'interpolate',
        ['linear'],
        ['get', 'magnitude'],
        1, 0,
        8, 1
      ],
      'heatmap-intensity': [
        'interpolate',
        ['linear'],
        ['zoom'],
        0, 1,
        15, 3
      ],
      'heatmap-color': [
        'interpolate',
        ['linear'],
        ['heatmap-density'],
        0, 'rgba(33,102,172,0)',
        0.2, 'rgb(103,169,207)',
        0.4, 'rgb(209,229,240)',
        0.6, 'rgb(253,219,199)',
        0.8, 'rgb(239,138,98)',
        1, 'rgb(178,24,43)'
      ],
      'heatmap-radius': [
        'interpolate',
        ['linear'],
        ['zoom'],
        0, 2,
        15, 20
      ]
    }
  };

  // Handle map click events
  const handleMapClick = useCallback((event: any) => {
    const features = mapRef.current?.queryRenderedFeatures([event.lngLat.lng, event.lngLat.lat]);
    if (features && features.length > 0) {
      const feature = features[0];
      setSelectedFeature(feature);
      onFeatureClick?.(feature);
    }
  }, [onFeatureClick]);

  // Handle viewport changes
  const handleViewportChange = useCallback((newViewport: ViewState) => {
    setViewport(newViewport);
    onViewportChange?.(newViewport);
  }, [onViewportChange]);

  // Toggle layer visibility
  const toggleLayer = useCallback((layerName: string) => {
    setLayerVisibility(prev => ({
      ...prev,
      [layerName]: !prev[layerName as keyof typeof prev]
    }));
  }, []);

  // Add custom WebGL layer for advanced rendering
  useEffect(() => {
    if (mapRef.current) {
      const map = mapRef.current.getMap();
      const customLayer = new GeologicalWebGLLayer('geological-webgl');
      
      if (!map.getLayer('geological-webgl')) {
        map.addLayer(customLayer as any);
      }
    }
  }, []);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <Map
        ref={mapRef}
        {...viewport}
        onMove={(evt) => handleViewportChange(evt.viewState)}
        onClick={handleMapClick as any}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        style={{ width: '100%', height: '100%' }}
        terrain={{ source: 'mapbox-dem', exaggeration: 1.5 }}
      >
        {/* Geological data layer */}
        {layerVisibility.geological && (
          <Source id="geological-data" type="geojson" data={geologicalGeoJSON}>
            <Layer {...geologicalLayer} />
          </Source>
        )}

        {/* Seismic events layers */}
        {layerVisibility.seismic && (
          <Source id="seismic-data" type="geojson" data={seismicGeoJSON}>
            <Layer {...seismicHeatmapLayer} />
            <Layer {...seismicLayer} />
          </Source>
        )}

        {/* Volcanic states layer */}
        {layerVisibility.volcanic && (
          <Source id="volcanic-data" type="geojson" data={volcanicGeoJSON}>
            <Layer {...volcanicLayer} />
          </Source>
        )}
      </Map>

      {/* Layer Controls */}
      <Paper
        sx={{
          position: 'absolute',
          top: 16,
          left: 16,
          p: 2,
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          minWidth: 200
        }}
      >
        <Typography variant="h6" gutterBottom>
          Map Layers
        </Typography>
        
        {Object.entries(layerVisibility).map(([layer, visible]) => (
          <FormControlLabel
            key={layer}
            control={
              <Switch
                checked={visible}
                onChange={() => toggleLayer(layer)}
                size="small"
              />
            }
            label={layer.charAt(0).toUpperCase() + layer.slice(1)}
            sx={{ display: 'block', mb: 1 }}
          />
        ))}

        {/* Time Range Control */}
        <Typography variant="body2" sx={{ mt: 2, mb: 1 }}>
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
        <Box sx={{ mt: 2 }}>
          <Chip label={`${data.length} Geological Points`} size="small" sx={{ mr: 1, mb: 1 }} />
          <Chip label={`${seismicEvents.length} Seismic Events`} size="small" sx={{ mr: 1, mb: 1 }} />
          <Chip label={`${volcanicStates.length} Volcanoes`} size="small" sx={{ mb: 1 }} />
        </Box>
      </Paper>

      {/* Feature Information Panel */}
      {selectedFeature && (
        <Paper
          sx={{
            position: 'absolute',
            bottom: 16,
            right: 16,
            p: 2,
            maxWidth: 300,
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
            Location: {selectedFeature.geometry.coordinates.join(', ')}
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
        </Paper>
      )}
    </Box>
  );
};

export default GeologicalMap;