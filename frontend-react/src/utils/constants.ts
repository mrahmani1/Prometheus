/**
 * Application constants and configuration
 */

// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
} as const;

// WebSocket Configuration
export const WEBSOCKET_CONFIG = {
  URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  RECONNECT_INTERVAL: 5000,
  MAX_RECONNECT_ATTEMPTS: 10,
  HEARTBEAT_INTERVAL: 30000,
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