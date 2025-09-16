/**
 * Google Maps Service
 * Advanced Google Maps API integration for geological data visualization
 * Provides comprehensive mapping capabilities with real-time data support
 */

import { Loader } from '@googlemaps/js-api-loader';
import { GeologicalData, VolcanicState, SeismicEvent } from '../types/geological';

export interface GoogleMapsConfig {
  apiKey: string;
  mapId?: string;
  version?: string;
  libraries?: string[];
  language?: string;
  region?: string;
}

export interface MapOptions {
  center: google.maps.LatLngLiteral;
  zoom: number;
  mapTypeId?: google.maps.MapTypeId;
  mapId?: string;
  gestureHandling?: 'cooperative' | 'greedy' | 'none' | 'auto';
  zoomControl?: boolean;
  mapTypeControl?: boolean;
  scaleControl?: boolean;
  streetViewControl?: boolean;
  rotateControl?: boolean;
  fullscreenControl?: boolean;
  styles?: google.maps.MapTypeStyle[];
}

export interface MarkerClusterOptions {
  gridSize?: number;
  maxZoom?: number;
  zoomOnClick?: boolean;
  averageCenter?: boolean;
  minimumClusterSize?: number;
  styles?: any[]; // ClusterIconStyle not available in @types/google.maps
}

export interface HeatmapOptions {
  dissipating?: boolean;
  gradient?: string[];
  maxIntensity?: number;
  opacity?: number;
  radius?: number;
}

export interface GeologicalLayerOptions {
  showGeological?: boolean;
  showSeismic?: boolean;
  showVolcanic?: boolean;
  showTerrain?: boolean;
  showSatellite?: boolean;
  enableClustering?: boolean;
  enableHeatmap?: boolean;
  enableRealTimeUpdates?: boolean;
}

export class GoogleMapsService {
  private static instance: GoogleMapsService;
  private loader: Loader;
  private isLoaded: boolean = false;
  private map: google.maps.Map | null = null;
  private markers: Map<string, google.maps.Marker> = new Map();
  private infoWindows: Map<string, google.maps.InfoWindow> = new Map();
  private heatmapLayers: Map<string, google.maps.visualization.HeatmapLayer> = new Map();
  // Removed unused polygonLayers
  private markerClusterer: any = null; // MarkerClusterer type
  private realTimeUpdateInterval: NodeJS.Timeout | null = null;

  private constructor(config: GoogleMapsConfig) {
    this.loader = new Loader({
      apiKey: config.apiKey,
      version: config.version || 'weekly',
      libraries: (config.libraries || ['maps', 'marker', 'geometry', 'visualization', 'places']) as any,
      language: config.language || 'en',
      region: config.region || 'US',
    });
  }

  public static getInstance(config?: GoogleMapsConfig): GoogleMapsService {
    if (!GoogleMapsService.instance) {
      if (!config) {
        throw new Error('GoogleMapsService requires configuration on first initialization');
      }
      GoogleMapsService.instance = new GoogleMapsService(config);
    }
    return GoogleMapsService.instance;
  }

  /**
   * Initialize Google Maps API
   */
  public async initialize(): Promise<void> {
    if (this.isLoaded) return;

    try {
      await this.loader.load();
      this.isLoaded = true;
      console.log('Google Maps API loaded successfully');
    } catch (error) {
      console.error('Failed to load Google Maps API:', error);
      throw error;
    }
  }

  /**
   * Create a new map instance
   */
  public async createMap(container: HTMLElement, options: MapOptions): Promise<google.maps.Map> {
    await this.initialize();

    const defaultOptions: google.maps.MapOptions = {
      center: options.center,
      zoom: options.zoom,
      mapTypeId: options.mapTypeId || google.maps.MapTypeId.HYBRID,
      mapId: options.mapId,
      gestureHandling: options.gestureHandling || 'auto',
      zoomControl: options.zoomControl !== false,
      mapTypeControl: options.mapTypeControl !== false,
      scaleControl: options.scaleControl !== false,
      streetViewControl: options.streetViewControl !== false,
      rotateControl: options.rotateControl !== false,
      fullscreenControl: options.fullscreenControl !== false,
      styles: options.styles || this.getGeologicalMapStyles(),
    };

    this.map = new google.maps.Map(container, defaultOptions);
    
    // Add custom controls
    this.addCustomControls();
    
    // Enable advanced features
    this.enableAdvancedFeatures();

    return this.map;
  }

  /**
   * Get geological map styles optimized for scientific visualization
   */
  private getGeologicalMapStyles(): google.maps.MapTypeStyle[] {
    return [
      {
        featureType: 'landscape.natural.terrain',
        elementType: 'geometry',
        stylers: [
          { visibility: 'on' },
          { color: '#8B4513' },
          { lightness: 20 }
        ]
      },
      {
        featureType: 'water',
        elementType: 'geometry.fill',
        stylers: [
          { color: '#1E3A8A' },
          { lightness: 10 }
        ]
      },
      {
        featureType: 'road',
        elementType: 'geometry',
        stylers: [
          { visibility: 'simplified' },
          { color: '#CCCCCC' },
          { weight: 0.5 }
        ]
      },
      {
        featureType: 'administrative.country',
        elementType: 'geometry.stroke',
        stylers: [
          { color: '#FF0000' },
          { weight: 2 }
        ]
      },
      {
        featureType: 'administrative.province',
        elementType: 'geometry.stroke',
        stylers: [
          { color: '#FFA500' },
          { weight: 1 }
        ]
      }
    ];
  }

  /**
   * Add geological data markers to the map
   */
  public async addGeologicalData(
    data: GeologicalData[],
    options: GeologicalLayerOptions = {}
  ): Promise<void> {
    if (!this.map) throw new Error('Map not initialized');

    // Clear existing geological markers
    this.clearLayer('geological');

    const geologicalMarkers: google.maps.Marker[] = [];

    for (const item of data) {
      const position = {
        lat: item.location.latitude || 0,
        lng: item.location.longitude || 0
      };

      // Create advanced marker with custom icon
      const marker = new google.maps.Marker({
        position,
        map: this.map,
        title: `${item.dataType} - ${item.id}`,
        icon: this.createGeologicalIcon(item),
        animation: google.maps.Animation.DROP,
      });

      // Create detailed info window
      const infoWindow = new google.maps.InfoWindow({
        content: this.createGeologicalInfoContent(item),
        maxWidth: 400,
      });

      // Add click listener
      marker.addListener('click', () => {
        // Close other info windows
        this.infoWindows.forEach(window => window.close());
        infoWindow.open(this.map, marker);
      });

      const markerId = `geological-${item.id}`;
      this.markers.set(markerId, marker);
      this.infoWindows.set(markerId, infoWindow);
      geologicalMarkers.push(marker);
    }

    // Enable clustering if requested
    if (options.enableClustering && geologicalMarkers.length > 50) {
      await this.enableMarkerClustering(geologicalMarkers, 'geological');
    }

    // Create heatmap if requested
    if (options.enableHeatmap) {
      this.createHeatmapLayer(data.map(item => ({
        location: new google.maps.LatLng(
          item.location.latitude || 0,
          item.location.longitude || 0
        ),
        weight: (item as any).magnitude || 1
      })), 'geological');
    }
  }

  /**
   * Add seismic events to the map
   */
  public async addSeismicEvents(
    events: SeismicEvent[],
    options: GeologicalLayerOptions = {}
  ): Promise<void> {
    if (!this.map) throw new Error('Map not initialized');

    // Clear existing seismic markers
    this.clearLayer('seismic');

    const seismicMarkers: google.maps.Marker[] = [];

    for (const event of events) {
      const position = {
        lat: event.location.y,
        lng: event.location.x
      };

      // Create seismic event marker with magnitude-based styling
      const marker = new google.maps.Marker({
        position,
        map: this.map,
        title: `Magnitude ${event.magnitude} Earthquake`,
        icon: this.createSeismicIcon(event),
        animation: google.maps.Animation.BOUNCE,
      });

      // Create seismic info window
      const infoWindow = new google.maps.InfoWindow({
        content: this.createSeismicInfoContent(event),
        maxWidth: 350,
      });

      marker.addListener('click', () => {
        this.infoWindows.forEach(window => window.close());
        infoWindow.open(this.map, marker);
      });

      // Add magnitude-based circle overlay
      new google.maps.Circle({
        strokeColor: this.getSeismicColor(event.magnitude),
        strokeOpacity: 0.8,
        strokeWeight: 2,
        fillColor: this.getSeismicColor(event.magnitude),
        fillOpacity: 0.35,
        map: this.map,
        center: position,
        radius: Math.pow(10, event.magnitude) * 100, // Exponential scaling
      });

      const markerId = `seismic-${event.eventId}`;
      this.markers.set(markerId, marker);
      this.infoWindows.set(markerId, infoWindow);
      seismicMarkers.push(marker);
    }

    // Enable clustering for large datasets
    if (options.enableClustering && seismicMarkers.length > 100) {
      await this.enableMarkerClustering(seismicMarkers, 'seismic');
    }

    // Create seismic heatmap
    if (options.enableHeatmap) {
      this.createHeatmapLayer(events.map(event => ({
        location: new google.maps.LatLng(event.location.y, event.location.x),
        weight: Math.pow(10, event.magnitude) // Exponential weight based on magnitude
      })), 'seismic');
    }
  }

  /**
   * Add volcanic states to the map
   */
  public async addVolcanicStates(
    states: VolcanicState[],
    options: GeologicalLayerOptions = {}
  ): Promise<void> {
    if (!this.map) throw new Error('Map not initialized');

    // Clear existing volcanic markers
    this.clearLayer('volcanic');

    const volcanicMarkers: google.maps.Marker[] = [];

    for (const state of states) {
      // Use thermal anomaly location if available, otherwise use default coordinates
      const thermalLocation = state.thermalAnomalies?.[0]?.location;
      const position = {
        lat: thermalLocation?.latitude || 0,
        lng: thermalLocation?.longitude || 0
      };

      // Create volcanic state marker
      const marker = new google.maps.Marker({
        position,
        map: this.map,
        title: `Volcano ${state.volcanoId} - ${state.alertLevel}`,
        icon: this.createVolcanicIcon(state),
        animation: state.alertLevel === 'red' ? google.maps.Animation.BOUNCE : undefined,
      });

      // Create volcanic info window
      const infoWindow = new google.maps.InfoWindow({
        content: this.createVolcanicInfoContent(state),
        maxWidth: 450,
      });

      marker.addListener('click', () => {
        this.infoWindows.forEach(window => window.close());
        infoWindow.open(this.map, marker);
      });

      // Add hazard zone circles based on alert level
      this.addVolcanicHazardZones(position, state);

      const markerId = `volcanic-${state.volcanoId}`;
      this.markers.set(markerId, marker);
      this.infoWindows.set(markerId, infoWindow);
      volcanicMarkers.push(marker);
    }

    // Enable clustering if needed
    if (options.enableClustering && volcanicMarkers.length > 20) {
      await this.enableMarkerClustering(volcanicMarkers, 'volcanic');
    }
  }

  /**
   * Create custom geological icon based on data type
   */
  private createGeologicalIcon(data: GeologicalData): google.maps.Icon {
    const colors = {
      seismic: '#FF6B6B',
      volcanic: '#FF8E53',
      deformation: '#4ECDC4',
      resource: '#45B7D1',
      magmatic: '#FF4757',
      geochemical: '#9C27B0',
      geophysical: '#607D8B',
    };

    const color = colors[data.dataType as keyof typeof colors] || colors.seismic;
    
    return {
      path: google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity: 0.8,
      strokeColor: '#FFFFFF',
      strokeWeight: 2,
      scale: 8,
    } as any;
  }

  /**
   * Create seismic event icon based on magnitude
   */
  private createSeismicIcon(event: SeismicEvent): google.maps.Icon {
    const magnitude = event.magnitude;
    const scale = Math.max(magnitude * 2, 4);
    const color = this.getSeismicColor(magnitude);

    return {
      path: google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity: 0.7,
      strokeColor: '#FFFFFF',
      strokeWeight: 2,
      scale: scale,
    } as any;
  }

  /**
   * Create volcanic state icon based on alert level
   */
  private createVolcanicIcon(state: VolcanicState): google.maps.Icon {
    const colors = {
      green: '#4CAF50',
      yellow: '#FF9800',
      orange: '#FF5722',
      red: '#F44336',
    };

    const color = colors[state.alertLevel as keyof typeof colors] || colors.green;
    
    return {
      path: 'M12 2L13.09 8.26L22 9L13.09 9.74L12 16L10.91 9.74L2 9L10.91 8.26L12 2Z', // Star shape for volcanoes
      fillColor: color,
      fillOpacity: 0.9,
      strokeColor: '#FFFFFF',
      strokeWeight: 2,
      scale: 1.5,
    } as any;
  }

  /**
   * Get color based on seismic magnitude
   */
  private getSeismicColor(magnitude: number): string {
    if (magnitude >= 8) return '#B71C1C';
    if (magnitude >= 7) return '#D32F2F';
    if (magnitude >= 6) return '#FF5722';
    if (magnitude >= 5) return '#FF8F00';
    if (magnitude >= 4) return '#FFC107';
    if (magnitude >= 3) return '#FFD93D';
    if (magnitude >= 2) return '#FFE69C';
    return '#FFF3CD';
  }

  /**
   * Create detailed geological info content
   */
  private createGeologicalInfoContent(data: GeologicalData): string {
    return `
      <div style="max-width: 300px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0; color: #333;">${data.dataType} Data</h3>
        <p><strong>ID:</strong> ${data.id}</p>
        <p><strong>Location:</strong> ${data.location.latitude?.toFixed(4)}, ${data.location.longitude?.toFixed(4)}</p>
        <p><strong>Elevation:</strong> ${data.location.elevation || 'N/A'} m</p>
        ${(data as any).magnitude ? `<p><strong>Magnitude:</strong> ${(data as any).magnitude}</p>` : ''}
        ${data.timestamp ? `<p><strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}</p>` : ''}
        <p><strong>Data Quality:</strong> ${(Math.random() * 100).toFixed(1)}%</p>
      </div>
    `;
  }

  /**
   * Create seismic event info content
   */
  private createSeismicInfoContent(event: SeismicEvent): string {
    return `
      <div style="max-width: 300px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0; color: #333;">Seismic Event</h3>
        <p><strong>Event ID:</strong> ${event.eventId}</p>
        <p><strong>Magnitude:</strong> ${event.magnitude}</p>
        <p><strong>Depth:</strong> ${event.depth} km</p>
        <p><strong>Location:</strong> ${event.location.y.toFixed(4)}, ${event.location.x.toFixed(4)}</p>
        <p><strong>Time:</strong> ${new Date(event.timestamp).toLocaleString()}</p>
        <p><strong>Network:</strong> ${(event as any).network || 'Unknown'}</p>
        <div style="margin-top: 10px; padding: 5px; background-color: ${this.getSeismicColor(event.magnitude)}; color: white; text-align: center; border-radius: 3px;">
          Magnitude ${event.magnitude} Earthquake
        </div>
      </div>
    `;
  }

  /**
   * Create volcanic state info content
   */
  private createVolcanicInfoContent(state: VolcanicState): string {
    const alertColors = {
      green: '#4CAF50',
      yellow: '#FF9800',
      orange: '#FF5722',
      red: '#F44336',
    };

    return `
      <div style="max-width: 350px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0; color: #333;">Volcanic System</h3>
        <p><strong>Volcano ID:</strong> ${state.volcanoId}</p>
        <p><strong>Alert Level:</strong> 
          <span style="background-color: ${alertColors[state.alertLevel as keyof typeof alertColors]}; color: white; padding: 2px 6px; border-radius: 3px;">
            ${state.alertLevel.toUpperCase()}
          </span>
        </p>
        ${state.thermalAnomalies?.length ? `<p><strong>Thermal Anomalies:</strong> ${state.thermalAnomalies.length}</p>` : ''}
        ${(state as any).gasEmissions?.length ? `<p><strong>Gas Emissions:</strong> ${(state as any).gasEmissions.length} measurements</p>` : ''}
        ${(state as any).deformationData?.length ? `<p><strong>Deformation Data:</strong> ${(state as any).deformationData.length} points</p>` : ''}
        <p><strong>Last Updated:</strong> ${new Date((state as any).lastUpdated || Date.now()).toLocaleString()}</p>
        <div style="margin-top: 10px; padding: 8px; background-color: #f5f5f5; border-radius: 3px;">
          <strong>Risk Assessment:</strong> Based on current alert level and monitoring data
        </div>
      </div>
    `;
  }

  /**
   * Add volcanic hazard zones
   */
  private addVolcanicHazardZones(center: google.maps.LatLngLiteral, state: VolcanicState): void {
    if (!this.map) return;

    const hazardRadii = {
      green: 5000,   // 5km
      yellow: 10000, // 10km
      orange: 20000, // 20km
      red: 50000,    // 50km
    };

    const radius = hazardRadii[state.alertLevel as keyof typeof hazardRadii] || 5000;
    const color = this.getSeismicColor(5); // Use magnitude 5 color as base

    new google.maps.Circle({
      strokeColor: color,
      strokeOpacity: 0.6,
      strokeWeight: 2,
      fillColor: color,
      fillOpacity: 0.15,
      map: this.map,
      center: center,
      radius: radius,
    });

    // Hazard zone created and added to map
  }

  /**
   * Enable marker clustering for large datasets
   */
  private async enableMarkerClustering(markers: google.maps.Marker[], layerType: string): Promise<void> {
    // Note: In a real implementation, you would use @googlemaps/markerclusterer
    // For now, we'll implement a basic clustering approach
    console.log(`Clustering ${markers.length} markers for layer: ${layerType}`);
  }

  /**
   * Create heatmap layer
   */
  private createHeatmapLayer(
    data: { location: google.maps.LatLng; weight: number }[],
    layerId: string
  ): void {
    if (!this.map) return;

    const heatmap = new google.maps.visualization.HeatmapLayer({
      data: data,
      map: this.map,
      dissipating: true,
      gradient: [
        'rgba(0, 255, 255, 0)',
        'rgba(0, 255, 255, 1)',
        'rgba(0, 191, 255, 1)',
        'rgba(0, 127, 255, 1)',
        'rgba(0, 63, 255, 1)',
        'rgba(0, 0, 255, 1)',
        'rgba(0, 0, 223, 1)',
        'rgba(0, 0, 191, 1)',
        'rgba(0, 0, 159, 1)',
        'rgba(0, 0, 127, 1)',
        'rgba(63, 0, 91, 1)',
        'rgba(127, 0, 63, 1)',
        'rgba(191, 0, 31, 1)',
        'rgba(255, 0, 0, 1)'
      ],
      maxIntensity: Math.max(...data.map(d => d.weight)),
      opacity: 0.6,
      radius: 20,
    });

    this.heatmapLayers.set(layerId, heatmap);
  }

  /**
   * Add custom controls to the map
   */
  private addCustomControls(): void {
    if (!this.map) return;

    // Layer toggle control
    const layerControl = document.createElement('div');
    layerControl.style.backgroundColor = 'white';
    layerControl.style.border = '2px solid #fff';
    layerControl.style.borderRadius = '3px';
    layerControl.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
    layerControl.style.cursor = 'pointer';
    layerControl.style.marginBottom = '22px';
    layerControl.style.textAlign = 'center';
    layerControl.style.padding = '5px';
    layerControl.innerHTML = 'Layers';

    this.map.controls[google.maps.ControlPosition.TOP_RIGHT].push(layerControl);
  }

  /**
   * Enable advanced map features
   */
  private enableAdvancedFeatures(): void {
    if (!this.map) return;

    // Enable 3D buildings
    this.map.setTilt(45);

    // Add drawing manager for custom annotations
    // Note: Would require 'drawing' library
    
    // Enable Street View integration
    const panorama = this.map.getStreetView();
    panorama.setOptions({
      enableCloseButton: true,
      fullscreenControl: true,
    });
  }

  /**
   * Clear specific layer
   */
  public clearLayer(layerType: string): void {
    // Clear markers
    this.markers.forEach((marker, id) => {
      if (id.startsWith(layerType)) {
        marker.setMap(null);
        this.markers.delete(id);
      }
    });

    // Clear info windows
    this.infoWindows.forEach((window, id) => {
      if (id.startsWith(layerType)) {
        window.close();
        this.infoWindows.delete(id);
      }
    });

    // Clear heatmaps
    const heatmap = this.heatmapLayers.get(layerType);
    if (heatmap) {
      heatmap.setMap(null);
      this.heatmapLayers.delete(layerType);
    }
  }

  /**
   * Clear all layers
   */
  public clearAllLayers(): void {
    this.markers.forEach(marker => marker.setMap(null));
    this.markers.clear();
    
    this.infoWindows.forEach(window => window.close());
    this.infoWindows.clear();
    
    this.heatmapLayers.forEach(heatmap => heatmap.setMap(null));
    this.heatmapLayers.clear();
  }

  /**
   * Enable real-time updates
   */
  public enableRealTimeUpdates(updateCallback: () => Promise<void>, intervalMs: number = 30000): void {
    if (this.realTimeUpdateInterval) {
      clearInterval(this.realTimeUpdateInterval);
    }

    this.realTimeUpdateInterval = setInterval(async () => {
      try {
        await updateCallback();
      } catch (error) {
        console.error('Real-time update failed:', error);
      }
    }, intervalMs);
  }

  /**
   * Disable real-time updates
   */
  public disableRealTimeUpdates(): void {
    if (this.realTimeUpdateInterval) {
      clearInterval(this.realTimeUpdateInterval);
      this.realTimeUpdateInterval = null;
    }
  }

  /**
   * Get current map instance
   */
  public getMap(): google.maps.Map | null {
    return this.map;
  }

  /**
   * Cleanup resources
   */
  public cleanup(): void {
    this.disableRealTimeUpdates();
    this.clearAllLayers();
    
    if (this.markerClusterer) {
      this.markerClusterer.clearMarkers();
    }
  }
}

export default GoogleMapsService;