import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Database-specific performance metrics
const dbQueryTime = new Trend('database_query_duration');
const dbConnectionErrors = new Rate('database_connection_errors');
const largeDatasetProcessing = new Trend('large_dataset_processing_time');
const spatialQueryPerformance = new Trend('spatial_query_duration');
const timeSeriesInsertRate = new Counter('timeseries_inserts_per_second');

export const options = {
  scenarios: {
    // Database stress test with large geological datasets
    large_dataset_stress: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '2m', target: 10 },
        { duration: '5m', target: 25 },
        { duration: '3m', target: 50 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'database_stress' },
    },
    // Time-series data ingestion test
    timeseries_ingestion: {
      executor: 'constant-arrival-rate',
      rate: 100, // 100 requests per second
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 20,
      maxVUs: 100,
      tags: { test_type: 'timeseries_load' },
    },
    // Geospatial query performance
    spatial_query_test: {
      executor: 'constant-vus',
      vus: 15,
      duration: '8m',
      tags: { test_type: 'spatial_queries' },
    },
  },
  thresholds: {
    database_query_duration: ['p(95)<3000'], // 95% of DB queries under 3s
    database_connection_errors: ['rate<0.01'], // DB connection errors under 1%
    large_dataset_processing_time: ['p(90)<15000'], // 90% large datasets under 15s
    spatial_query_duration: ['p(95)<2000'], // 95% spatial queries under 2s
    timeseries_inserts_per_second: ['rate>50'], // At least 50 inserts/second
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

// Large geological dataset for testing
const largeSeismicDataset = {
  events: Array(1000).fill(0).map((_, i) => ({
    id: `event_${i}`,
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    magnitude: 2.0 + Math.random() * 6.0,
    depth: Math.random() * 50,
    location: {
      lat: 19.4 + (Math.random() - 0.5) * 2,
      lon: -155.3 + (Math.random() - 0.5) * 2,
      elevation: Math.random() * 4000,
    },
    waveform_data: Array(2048).fill(0).map(() => Math.random() * 1000),
  })),
};

const geologicalSurveyData = {
  survey_id: 'performance_test_survey',
  region: {
    bounds: {
      north: 20.0,
      south: 18.0,
      east: -154.0,
      west: -157.0,
    },
  },
  measurements: Array(5000).fill(0).map((_, i) => ({
    id: `measurement_${i}`,
    location: {
      lat: 18.0 + Math.random() * 2.0,
      lon: -157.0 + Math.random() * 3.0,
      elevation: Math.random() * 4000,
    },
    rock_type: ['basalt', 'andesite', 'rhyolite', 'obsidian'][Math.floor(Math.random() * 4)],
    age_ma: Math.random() * 100,
    composition: {
      SiO2: 45 + Math.random() * 30,
      Al2O3: 10 + Math.random() * 10,
      FeO: 5 + Math.random() * 15,
      MgO: 2 + Math.random() * 20,
      CaO: 5 + Math.random() * 15,
    },
    structural_data: {
      strike: Math.random() * 360,
      dip: Math.random() * 90,
      fracture_density: Math.random() * 100,
    },
  })),
};

export default function () {
  const headers = {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json',
  };

  // Randomly select test type based on scenario
  const testType = __ENV.K6_SCENARIO || 'mixed';
  
  switch (testType) {
    case 'large_dataset_stress':
      testLargeDatasetProcessing(headers);
      break;
    case 'timeseries_load':
      testTimeSeriesIngestion(headers);
      break;
    case 'spatial_queries':
      testSpatialQueries(headers);
      break;
    default:
      // Mixed workload
      const testChoice = Math.random();
      if (testChoice < 0.4) {
        testLargeDatasetProcessing(headers);
      } else if (testChoice < 0.7) {
        testTimeSeriesIngestion(headers);
      } else {
        testSpatialQueries(headers);
      }
  }
  
  sleep(0.5);
}

function testLargeDatasetProcessing(headers) {
  const startTime = Date.now();
  
  // Test bulk seismic data insertion
  const response = http.post(`${BASE_URL}/api/v1/seismic/bulk-insert`, 
    JSON.stringify(largeSeismicDataset), 
    {
      headers,
      tags: { endpoint: 'bulk_seismic_insert' },
      timeout: '30s',
    }
  );
  
  const processingTime = Date.now() - startTime;
  largeDatasetProcessing.add(processingTime);
  
  const success = check(response, {
    'bulk insert successful': (r) => r.status === 200 || r.status === 202,
    'processing time acceptable': (r) => processingTime < 30000,
    'response has insert count': (r) => {
      if (r.status === 200) {
        const result = JSON.parse(r.body);
        return result.inserted_count !== undefined;
      }
      return true;
    },
  });
  
  dbConnectionErrors.add(!success);
}

function testTimeSeriesIngestion(headers) {
  const startTime = Date.now();
  
  // Generate real-time sensor data
  const sensorData = {
    sensor_id: `sensor_${Math.floor(Math.random() * 100)}`,
    timestamp: new Date().toISOString(),
    measurements: {
      seismic_amplitude: Math.random() * 1000,
      ground_velocity: Math.random() * 10,
      temperature: 20 + Math.random() * 80,
      gas_co2: Math.random() * 1000,
      gas_so2: Math.random() * 500,
      tilt_x: (Math.random() - 0.5) * 0.001,
      tilt_y: (Math.random() - 0.5) * 0.001,
    },
    location: {
      lat: 19.4 + (Math.random() - 0.5) * 0.1,
      lon: -155.3 + (Math.random() - 0.5) * 0.1,
      elevation: 1000 + Math.random() * 3000,
    },
  };
  
  const response = http.post(`${BASE_URL}/api/v1/sensors/data`, 
    JSON.stringify(sensorData), 
    {
      headers,
      tags: { endpoint: 'sensor_data_insert' },
    }
  );
  
  const queryTime = Date.now() - startTime;
  dbQueryTime.add(queryTime);
  
  const success = check(response, {
    'sensor data inserted': (r) => r.status === 200 || r.status === 201,
    'insert time fast': (r) => queryTime < 1000,
  });
  
  if (success) {
    timeSeriesInsertRate.add(1);
  }
  
  dbConnectionErrors.add(!success);
}

function testSpatialQueries(headers) {
  const startTime = Date.now();
  
  // Test complex spatial query
  const spatialQuery = {
    query_type: 'spatial_intersection',
    geometry: {
      type: 'Polygon',
      coordinates: [[
        [-155.5, 19.2],
        [-155.0, 19.2],
        [-155.0, 19.7],
        [-155.5, 19.7],
        [-155.5, 19.2],
      ]],
    },
    filters: {
      rock_age_range: { min: 0, max: 10 }, // Last 10 million years
      magnitude_range: { min: 3.0, max: 8.0 },
      depth_range: { min: 0, max: 30 },
    },
    include_geometry: true,
    limit: 1000,
  };
  
  const response = http.post(`${BASE_URL}/api/v1/geological/spatial-query`, 
    JSON.stringify(spatialQuery), 
    {
      headers,
      tags: { endpoint: 'spatial_query' },
      timeout: '10s',
    }
  );
  
  const queryTime = Date.now() - startTime;
  spatialQueryPerformance.add(queryTime);
  
  const success = check(response, {
    'spatial query successful': (r) => r.status === 200,
    'results returned': (r) => {
      if (r.status === 200) {
        const results = JSON.parse(r.body);
        return results.features !== undefined && Array.isArray(results.features);
      }
      return false;
    },
    'query time acceptable': (r) => queryTime < 5000,
  });
  
  dbConnectionErrors.add(!success);
}

// Test geological survey data bulk processing
function testGeologicalSurveyProcessing(headers) {
  const startTime = Date.now();
  
  const response = http.post(`${BASE_URL}/api/v1/geological/survey/process`, 
    JSON.stringify(geologicalSurveyData), 
    {
      headers,
      tags: { endpoint: 'survey_processing' },
      timeout: '45s',
    }
  );
  
  const processingTime = Date.now() - startTime;
  largeDatasetProcessing.add(processingTime);
  
  check(response, {
    'survey processing initiated': (r) => r.status === 200 || r.status === 202,
    'processing time reasonable': (r) => processingTime < 45000,
  });
}

export function teardown(data) {
  console.log('Database performance test completed');
  console.log(`Average query time: ${dbQueryTime.avg}ms`);
  console.log(`Time-series insert rate: ${timeSeriesInsertRate.count / (10 * 60)} inserts/second`);
}