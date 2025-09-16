/**
 * Application constants and configuration
 */

// API Configuration - Production geological data APIs
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'https://api.prometheus-ai.platform/api/v1',
  DATABASE_API: import.meta.env.VITE_DATABASE_API_URL || 'https://api.prometheus-ai.platform/api/v1/database',
  
  // Real Geological Data Sources
  IRIS_API: import.meta.env.VITE_IRIS_API_URL || 'https://service.iris.edu',
  USGS_API: import.meta.env.VITE_USGS_API_URL || 'https://earthquake.usgs.gov',
  NASA_API: import.meta.env.VITE_NASA_API_URL || 'https://firms.modaps.eosdis.nasa.gov',
  COPERNICUS_API: import.meta.env.VITE_COPERNICUS_API_URL || 'https://scihub.copernicus.eu',
  ASF_HYP3_API: import.meta.env.VITE_ASF_HYP3_API_URL || 'https://hyp3-api.asf.alaska.edu',
  EARTHCHEM_API: import.meta.env.VITE_EARTHCHEM_API_URL || 'https://www.earthchem.org',
  
  // Academic API Sources
  SEMANTIC_SCHOLAR_API: import.meta.env.VITE_SEMANTIC_SCHOLAR_API_URL || 'https://api.semanticscholar.org/graph/v1',
  CONNECTED_PAPERS_API: import.meta.env.VITE_CONNECTED_PAPERS_API_URL || 'https://api.connectedpapers.com/v2',
  WILEY_API: import.meta.env.VITE_WILEY_API_URL || 'https://api.wiley.com/onlinelibrary/tdm/v1',
  GEOSCIENCE_WORLD_API: import.meta.env.VITE_GEOSCIENCE_WORLD_API_URL || 'https://api.geoscienceworld.org/v1',
  
  TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000'),
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
  RATE_LIMIT: {
    REQUESTS_PER_MINUTE: 60,
    BURST_LIMIT: 10,
  },
} as const;

// WebSocket Configuration - Production geological data streams
export const WEBSOCKET_CONFIG = {
  BASE_URL: import.meta.env.VITE_WS_URL || 'wss://api.prometheus-ai.platform',
  IRIS_WS: import.meta.env.VITE_IRIS_WS_URL || 'wss://api.prometheus-ai.platform/ws/iris',
  GEOLOGICAL_WS: import.meta.env.VITE_GEOLOGICAL_WS_URL || 'wss://api.prometheus-ai.platform/ws/geological',
  PROCESS_WS: import.meta.env.VITE_PROCESS_WS_URL || 'wss://api.prometheus-ai.platform/ws/process',
  MONITORING_WS: import.meta.env.VITE_MONITORING_WS_URL || 'wss://api.prometheus-ai.platform/ws/monitoring',
  RECONNECT_INTERVAL: parseInt(import.meta.env.VITE_WS_RECONNECT_INTERVAL || '5000'),
  MAX_RECONNECT_ATTEMPTS: 10,
  HEARTBEAT_INTERVAL: 30000,
  CONNECTION_TIMEOUT: 10000,
} as const;

// Database Configuration
export const DB_CONFIG = {
  POSTGRESQL: {
    HOST: import.meta.env.VITE_POSTGRESQL_HOST || 'postgresql.prometheus-ai.platform',
    PORT: parseInt(import.meta.env.VITE_POSTGRESQL_PORT || '5432'),
    DATABASE: import.meta.env.VITE_POSTGRESQL_DATABASE || 'prometheus_geological',
    SSL: import.meta.env.VITE_POSTGRESQL_SSL === 'true',
  },
  INFLUXDB: {
    HOST: import.meta.env.VITE_INFLUXDB_HOST || 'influxdb.prometheus-ai.platform',
    PORT: parseInt(import.meta.env.VITE_INFLUXDB_PORT || '8086'),
    DATABASE: import.meta.env.VITE_INFLUXDB_DATABASE || 'geological_timeseries',
    SSL: import.meta.env.VITE_INFLUXDB_SSL === 'true',
  },
  REDIS: {
    HOST: import.meta.env.VITE_REDIS_HOST || 'redis.prometheus-ai.platform',
    PORT: parseInt(import.meta.env.VITE_REDIS_PORT || '6379'),
    SSL: import.meta.env.VITE_REDIS_SSL === 'true',
  },
  NEO4J: {
    HOST: import.meta.env.VITE_NEO4J_HOST || 'neo4j.prometheus-ai.platform',
    PORT: parseInt(import.meta.env.VITE_NEO4J_PORT || '7687'),
    DATABASE: import.meta.env.VITE_NEO4J_DATABASE || 'causal_relationships',
    SSL: import.meta.env.VITE_NEO4J_SSL === 'true',
  },
  TIMESCALEDB: {
    HOST: import.meta.env.VITE_TIMESCALEDB_HOST || 'timescaledb.prometheus-ai.platform',
    PORT: parseInt(import.meta.env.VITE_TIMESCALEDB_PORT || '5432'),
    DATABASE: import.meta.env.VITE_TIMESCALEDB_DATABASE || 'geological_timeseries',
    SSL: import.meta.env.VITE_TIMESCALEDB_SSL === 'true',
  },
} as const;

// Feature Flags
export const FEATURE_FLAGS = {
  ENABLE_REAL_TIME_MONITORING: import.meta.env.VITE_ENABLE_REAL_TIME_MONITORING === 'true',
  ENABLE_ACADEMIC_INTEGRATION: import.meta.env.VITE_ENABLE_ACADEMIC_INTEGRATION === 'true',
  ENABLE_CAUSAL_ANALYSIS: import.meta.env.VITE_ENABLE_CAUSAL_ANALYSIS === 'true',
  ENABLE_MT_SAR_PROCESSING: import.meta.env.VITE_ENABLE_MT_SAR_PROCESSING === 'true',
  ENABLE_LITERATURE_VALIDATION: import.meta.env.VITE_ENABLE_LITERATURE_VALIDATION === 'true',
  ENABLE_DEBUG_MODE: import.meta.env.VITE_ENABLE_DEBUG_MODE === 'true',
} as const;

// Geological Constants
export const GEOLOGICAL_CONSTANTS = {
  // Earth parameters
  EARTH_RADIUS: 6371000, // meters
  EARTH_CIRCUMFERENCE: 40075000, // meters
  
  // Seismic parameters
  SEISMIC_WAVE_VELOCITIES: {
    P_WAVE_CRUSTAL: 6000, // m/s
    S_WAVE_CRUSTAL: 3500, // m/s
    P_WAVE_MANTLE: 8000, // m/s
    S_WAVE_MANTLE: 4500, // m/s
  },
  
  // Magnitude scales
  MAGNITUDE_SCALES: {
    RICHTER: 'ML',
    MOMENT: 'Mw',
    BODY_WAVE: 'mb',
    SURFACE_WAVE: 'Ms',
  },
  
  // Volcanic alert levels
  VOLCANIC_ALERT_LEVELS: {
    GREEN: 'Normal',
    YELLOW: 'Advisory',
    ORANGE: 'Watch',
    RED: 'Warning',
  },
  
  // Rock types
  ROCK_TYPES: {
    IGNEOUS: ['granite', 'basalt', 'andesite', 'rhyolite', 'gabbro', 'diorite'],
    SEDIMENTARY: ['sandstone', 'limestone', 'shale', 'conglomerate', 'mudstone'],
    METAMORPHIC: ['gneiss', 'schist', 'slate', 'marble', 'quartzite', 'amphibolite'],
  },
  
  // Mineral hardness (Mohs scale)
  MOHS_HARDNESS: {
    TALC: 1,
    GYPSUM: 2,
    CALCITE: 3,
    FLUORITE: 4,
    APATITE: 5,
    ORTHOCLASE: 6,
    QUARTZ: 7,
    TOPAZ: 8,
    CORUNDUM: 9,
    DIAMOND: 10,
  },
} as const;

// Visualization Constants
export const VISUALIZATION_CONSTANTS = {
  // Default camera settings
  DEFAULT_CAMERA: {
    FOV: 75,
    NEAR: 0.1,
    FAR: 10000,
    POSITION: [0, 0, 100],
  },
  
  // Color schemes
  COLOR_SCHEMES: {
    SEISMIC: 'viridis',
    VOLCANIC: 'hot',
    DEFORMATION: 'coolwarm',
    TOPOGRAPHIC: 'terrain',
    GEOLOGICAL: 'geological',
  },
  
  // Rendering quality levels
  QUALITY_LEVELS: {
    LOW: { shadows: false, antialiasing: false, postProcessing: false },
    MEDIUM: { shadows: true, antialiasing: false, postProcessing: false },
    HIGH: { shadows: true, antialiasing: true, postProcessing: false },
    ULTRA: { shadows: true, antialiasing: true, postProcessing: true },
  },
  
  // Performance thresholds
  PERFORMANCE_THRESHOLDS: {
    MIN_FPS: 30,
    TARGET_FPS: 60,
    MAX_TRIANGLES: 1000000,
    MAX_DRAW_CALLS: 1000,
  },
} as const;

// Data Processing Constants
export const DATA_PROCESSING = {
  // Batch sizes
  BATCH_SIZES: {
    SEISMIC_EVENTS: 1000,
    DEFORMATION_POINTS: 5000,
    VOLCANIC_MEASUREMENTS: 500,
    RESOURCE_SAMPLES: 2000,
  },
  
  // Cache settings
  CACHE_SETTINGS: {
    MAX_AGE: 3600000, // 1 hour
    MAX_SIZE: 100 * 1024 * 1024, // 100MB
    CLEANUP_INTERVAL: 300000, // 5 minutes
  },
  
  // Data quality thresholds
  QUALITY_THRESHOLDS: {
    COMPLETENESS: 0.8,
    ACCURACY: 0.9,
    CONSISTENCY: 0.85,
    TIMELINESS: 0.95,
  },
} as const;

// User Interface Constants
export const UI_CONSTANTS = {
  // Layout dimensions
  HEADER_HEIGHT: 64,
  SIDEBAR_WIDTH: 280,
  SIDEBAR_COLLAPSED_WIDTH: 64,
  FOOTER_HEIGHT: 48,
  
  // Breakpoints (matching Material-UI)
  BREAKPOINTS: {
    XS: 0,
    SM: 600,
    MD: 900,
    LG: 1200,
    XL: 1536,
  },
  
  // Animation durations
  ANIMATION_DURATION: {
    SHORT: 150,
    STANDARD: 300,
    LONG: 500,
  },
  
  // Z-index layers
  Z_INDEX: {
    DRAWER: 1200,
    APP_BAR: 1100,
    MODAL: 1300,
    SNACKBAR: 1400,
    TOOLTIP: 1500,
  },
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network connection error. Please check your internet connection.',
  AUTHENTICATION_ERROR: 'Authentication failed. Please log in again.',
  AUTHORIZATION_ERROR: 'You do not have permission to access this resource.',
  VALIDATION_ERROR: 'Please check your input and try again.',
  SERVER_ERROR: 'Server error occurred. Please try again later.',
  TIMEOUT_ERROR: 'Request timed out. Please try again.',
  UNKNOWN_ERROR: 'An unexpected error occurred. Please try again.',
  
  // Geological domain-specific errors
  GEOLOGICAL_DATA_ERROR: 'Invalid geological data format.',
  SEISMIC_ANALYSIS_ERROR: 'Seismic analysis failed. Please check input parameters.',
  VOLCANIC_MONITORING_ERROR: 'Volcanic monitoring data unavailable.',
  DEFORMATION_PROCESSING_ERROR: 'Deformation data processing failed.',
  RESOURCE_ASSESSMENT_ERROR: 'Resource assessment calculation error.',
} as const;

// Success Messages
export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: 'Successfully logged in.',
  LOGOUT_SUCCESS: 'Successfully logged out.',
  DATA_SAVED: 'Data saved successfully.',
  DATA_EXPORTED: 'Data exported successfully.',
  ANALYSIS_COMPLETE: 'Analysis completed successfully.',
  MODEL_TRAINED: 'Model training completed successfully.',
  FORECAST_GENERATED: 'Forecast generated successfully.',
} as const;

// File Format Constants
export const FILE_FORMATS = {
  SUPPORTED_IMAGES: ['.png', '.jpg', '.jpeg', '.svg', '.webp'],
  SUPPORTED_DATA: ['.csv', '.json', '.geojson', '.netcdf', '.hdf5'],
  SUPPORTED_MODELS: ['.pkl', '.joblib', '.onnx', '.pb'],
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
} as const;

// Coordinate Systems
export const COORDINATE_SYSTEMS = {
  WGS84: 'EPSG:4326',
  WEB_MERCATOR: 'EPSG:3857',
  UTM_ZONES: Array.from({ length: 60 }, (_, i) => `EPSG:${32601 + i}`), // UTM zones 1-60N
} as const;

// Units and Conversions
export const UNITS = {
  LENGTH: {
    METER: 'm',
    KILOMETER: 'km',
    FOOT: 'ft',
    MILE: 'mi',
  },
  TEMPERATURE: {
    CELSIUS: '°C',
    FAHRENHEIT: '°F',
    KELVIN: 'K',
  },
  PRESSURE: {
    PASCAL: 'Pa',
    BAR: 'bar',
    PSI: 'psi',
    ATM: 'atm',
  },
  TIME: {
    SECOND: 's',
    MINUTE: 'min',
    HOUR: 'h',
    DAY: 'd',
    YEAR: 'yr',
  },
} as const;

// Regular Expressions
export const REGEX_PATTERNS = {
  EMAIL: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PHONE: /^\+?[\d\s\-\(\)]+$/,
  COORDINATES: /^-?\d+\.?\d*,-?\d+\.?\d*$/,
  MAGNITUDE: /^\d+\.?\d*$/,
  DEPTH: /^\d+\.?\d*$/,
} as const;