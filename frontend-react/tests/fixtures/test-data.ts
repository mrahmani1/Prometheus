/**
 * Test data fixtures for geological and volcanic monitoring tests
 * Contains realistic geological data for comprehensive testing
 */

export interface SeismicEvent {
  id: string;
  timestamp: string;
  magnitude: number;
  depth: number;
  latitude: number;
  longitude: number;
  eventType: 'earthquake' | 'volcanic' | 'explosion' | 'landslide';
  location: string;
}

export interface VolcanoData {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation: number;
  alertLevel: 'green' | 'yellow' | 'orange' | 'red';
  eruptionProbability: number;
  lastEruption: string;
  volcanoType: string;
  deformationRate: number;
  thermalAnomalies: number;
  gasEmissions: {
    so2: number;
    co2: number;
    h2s: number;
  };
}

export interface GeologicalModel {
  id: string;
  name: string;
  region: string;
  modelType: '3d-geological' | 'magmatic-system' | 'stress-field' | 'mantle-convection';
  accuracy: number;
  lastUpdated: string;
  parameters: Record<string, any>;
}

export interface ResourceProspect {
  id: string;
  name: string;
  resourceType: 'mineral' | 'geothermal' | 'oil-gas';
  prospectivityScore: number;
  confidence: number;
  location: {
    latitude: number;
    longitude: number;
  };
  geologicalSetting: string;
}

/**
 * Mock seismic events for testing
 */
export const mockSeismicEvents: SeismicEvent[] = [
  {
    id: 'evt_001',
    timestamp: '2024-01-15T10:30:00Z',
    magnitude: 4.2,
    depth: 8.5,
    latitude: 44.4280,
    longitude: -110.5885,
    eventType: 'volcanic',
    location: 'Yellowstone Caldera'
  },
  {
    id: 'evt_002',
    timestamp: '2024-01-15T11:15:00Z',
    magnitude: 3.8,
    depth: 12.3,
    latitude: 44.4350,
    longitude: -110.5920,
    eventType: 'earthquake',
    location: 'Yellowstone Caldera'
  },
  {
    id: 'evt_003',
    timestamp: '2024-01-15T12:45:00Z',
    magnitude: 5.1,
    depth: 6.2,
    latitude: 19.4326,
    longitude: -155.2920,
    eventType: 'volcanic',
    location: 'Kilauea Volcano'
  },
  {
    id: 'evt_004',
    timestamp: '2024-01-15T14:20:00Z',
    magnitude: 2.9,
    depth: 15.7,
    latitude: 46.2000,
    longitude: -122.1800,
    eventType: 'earthquake',
    location: 'Mount St. Helens'
  }
];

/**
 * Mock volcano data for testing
 */
export const mockVolcanoes: VolcanoData[] = [
  {
    id: 'yellowstone',
    name: 'Yellowstone Caldera',
    latitude: 44.4280,
    longitude: -110.5885,
    elevation: 2805,
    alertLevel: 'yellow',
    eruptionProbability: 15.2,
    lastEruption: '70000 years ago',
    volcanoType: 'Supervolcano Caldera',
    deformationRate: 2.3,
    thermalAnomalies: 12,
    gasEmissions: {
      so2: 450,
      co2: 8900,
      h2s: 23
    }
  },
  {
    id: 'kilauea',
    name: 'Kilauea',
    latitude: 19.4326,
    longitude: -155.2920,
    elevation: 1247,
    alertLevel: 'orange',
    eruptionProbability: 78.5,
    lastEruption: '2023-09-10',
    volcanoType: 'Shield Volcano',
    deformationRate: 15.7,
    thermalAnomalies: 45,
    gasEmissions: {
      so2: 2300,
      co2: 15600,
      h2s: 89
    }
  },
  {
    id: 'mount-st-helens',
    name: 'Mount St. Helens',
    latitude: 46.2000,
    longitude: -122.1800,
    elevation: 2549,
    alertLevel: 'green',
    eruptionProbability: 5.8,
    lastEruption: '2008-07-10',
    volcanoType: 'Stratovolcano',
    deformationRate: 0.8,
    thermalAnomalies: 3,
    gasEmissions: {
      so2: 120,
      co2: 3400,
      h2s: 8
    }
  },
  {
    id: 'mount-rainier',
    name: 'Mount Rainier',
    latitude: 46.8523,
    longitude: -121.7603,
    elevation: 4392,
    alertLevel: 'green',
    eruptionProbability: 3.2,
    lastEruption: '1894-12-01',
    volcanoType: 'Stratovolcano',
    deformationRate: 0.3,
    thermalAnomalies: 1,
    gasEmissions: {
      so2: 45,
      co2: 1200,
      h2s: 2
    }
  }
];

/**
 * Mock geological models for testing
 */
export const mockGeologicalModels: GeologicalModel[] = [
  {
    id: 'model_001',
    name: 'Yellowstone Magma Chamber 3D',
    region: 'Yellowstone National Park',
    modelType: '3d-geological',
    accuracy: 92.5,
    lastUpdated: '2024-01-10T08:00:00Z',
    parameters: {
      depth: 8000,
      volume: 15000,
      temperature: 850,
      composition: 'rhyolitic'
    }
  },
  {
    id: 'model_002',
    name: 'Pacific Northwest Stress Field',
    region: 'Cascades Volcanic Arc',
    modelType: 'stress-field',
    accuracy: 87.3,
    lastUpdated: '2024-01-12T14:30:00Z',
    parameters: {
      principalStress: 45.2,
      orientation: 'NE-SW',
      magnitude: 'moderate'
    }
  },
  {
    id: 'model_003',
    name: 'Hawaiian Mantle Plume',
    region: 'Hawaiian Islands',
    modelType: 'mantle-convection',
    accuracy: 89.7,
    lastUpdated: '2024-01-08T16:45:00Z',
    parameters: {
      plumeVelocity: 12.5,
      temperature: 1350,
      depth: 2900
    }
  }
];

/**
 * Mock resource prospects for testing
 */
export const mockResourceProspects: ResourceProspect[] = [
  {
    id: 'prospect_001',
    name: 'Yellowstone Geothermal Field',
    resourceType: 'geothermal',
    prospectivityScore: 85.2,
    confidence: 78.5,
    location: {
      latitude: 44.4600,
      longitude: -110.8300
    },
    geologicalSetting: 'Volcanic Caldera'
  },
  {
    id: 'prospect_002',
    name: 'Nevada Copper Deposit',
    resourceType: 'mineral',
    prospectivityScore: 72.8,
    confidence: 65.3,
    location: {
      latitude: 39.5296,
      longitude: -116.9325
    },
    geologicalSetting: 'Porphyry System'
  },
  {
    id: 'prospect_003',
    name: 'Alaska Oil Prospect',
    resourceType: 'oil-gas',
    prospectivityScore: 68.4,
    confidence: 71.2,
    location: {
      latitude: 70.2500,
      longitude: -148.8000
    },
    geologicalSetting: 'Sedimentary Basin'
  }
];

/**
 * Mock WebSocket messages for real-time testing
 */
export const mockWebSocketMessages = {
  seismicEvent: {
    type: 'seismic_event',
    data: {
      id: 'evt_realtime_001',
      timestamp: new Date().toISOString(),
      magnitude: 3.5,
      depth: 9.2,
      latitude: 44.4300,
      longitude: -110.5900,
      eventType: 'volcanic'
    }
  },
  
  alertLevelChange: {
    type: 'alert_level_change',
    data: {
      volcanoId: 'yellowstone',
      oldLevel: 'yellow',
      newLevel: 'orange',
      timestamp: new Date().toISOString(),
      reason: 'Increased seismic activity'
    }
  },
  
  deformationUpdate: {
    type: 'deformation_update',
    data: {
      volcanoId: 'kilauea',
      timestamp: new Date().toISOString(),
      displacement: {
        x: 2.3,
        y: -1.8,
        z: 4.7
      },
      units: 'mm'
    }
  },
  
  gasEmissionUpdate: {
    type: 'gas_emission_update',
    data: {
      volcanoId: 'kilauea',
      timestamp: new Date().toISOString(),
      so2: 2450,
      co2: 16200,
      h2s: 92,
      units: 'ppm'
    }
  }
};

/**
 * Mock user credentials for testing different roles
 */
export const mockUsers = {
  researcher: {
    username: 'researcher@prometheus.ai',
    password: 'research123',
    role: 'researcher',
    permissions: ['view_data', 'analyze_data', 'export_data']
  },
  
  operator: {
    username: 'operator@prometheus.ai',
    password: 'operator123',
    role: 'operator',
    permissions: ['view_data', 'monitor_alerts', 'configure_thresholds']
  },
  
  industry: {
    username: 'industry@prometheus.ai',
    password: 'industry123',
    role: 'industry',
    permissions: ['view_data', 'resource_assessment', 'export_reports']
  },
  
  admin: {
    username: 'admin@prometheus.ai',
    password: 'admin123',
    role: 'admin',
    permissions: ['all']
  }
};

/**
 * Mock API responses for testing
 */
export const mockApiResponses = {
  volcanoes: {
    status: 200,
    data: mockVolcanoes
  },
  
  seismicEvents: {
    status: 200,
    data: mockSeismicEvents
  },
  
  geologicalModels: {
    status: 200,
    data: mockGeologicalModels
  },
  
  resourceProspects: {
    status: 200,
    data: mockResourceProspects
  },
  
  error: {
    status: 500,
    error: 'Internal Server Error',
    message: 'Failed to fetch geological data'
  },
  
  unauthorized: {
    status: 401,
    error: 'Unauthorized',
    message: 'Invalid authentication credentials'
  }
};

/**
 * Performance benchmarks for testing
 */
export const performanceBenchmarks = {
  pageLoad: {
    dashboard: 3000, // 3 seconds max
    visualization: 5000, // 5 seconds max for 3D content
    monitoring: 2000 // 2 seconds max
  },
  
  apiResponse: {
    fast: 200, // 200ms for cached data
    normal: 1000, // 1 second for database queries
    slow: 3000 // 3 seconds for complex calculations
  },
  
  rendering: {
    minFPS: 30, // Minimum 30 FPS for smooth interaction
    maxMemory: 1000, // Maximum 1GB memory usage
    maxCPU: 80 // Maximum 80% CPU usage
  }
};

/**
 * Accessibility test configurations
 */
export const accessibilityConfig = {
  rules: {
    'color-contrast': { enabled: true },
    'keyboard-navigation': { enabled: true },
    'aria-labels': { enabled: true },
    'focus-management': { enabled: true },
    'screen-reader': { enabled: true }
  },
  
  standards: ['WCAG2A', 'WCAG2AA'],
  
  tags: ['wcag2a', 'wcag2aa', 'best-practice']
};

/**
 * Visual regression test configurations
 */
export const visualRegressionConfig = {
  threshold: 0.2, // 20% pixel difference threshold
  
  viewports: [
    { width: 1920, height: 1080, name: 'desktop' },
    { width: 1366, height: 768, name: 'laptop' },
    { width: 768, height: 1024, name: 'tablet' },
    { width: 375, height: 667, name: 'mobile' }
  ],
  
  animations: 'disabled',
  
  maskElements: [
    '[data-testid="timestamp"]',
    '[data-testid="real-time-data"]',
    '[data-testid="current-time"]'
  ]
};