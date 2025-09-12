import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics for geological API performance
const geologicalApiErrors = new Rate('geological_api_errors');
const seismicProcessingTime = new Trend('seismic_processing_duration');
const volcanicDataThroughput = new Counter('volcanic_data_processed');
const realTimeLatency = new Trend('real_time_data_latency');

// Test configuration for different load scenarios
export const options = {
  scenarios: {
    // Baseline load test
    baseline_load: {
      executor: 'constant-vus',
      vus: 10,
      duration: '5m',
      tags: { test_type: 'baseline' },
    },
    // Stress test for peak volcanic activity
    volcanic_stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'volcanic_stress' },
    },
    // Spike test for emergency scenarios
    emergency_spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 500 },
        { duration: '1m', target: 500 },
        { duration: '10s', target: 0 },
      ],
      tags: { test_type: 'emergency_spike' },
    },
    // Soak test for long-running monitoring
    monitoring_soak: {
      executor: 'constant-vus',
      vus: 20,
      duration: '30m',
      tags: { test_type: 'soak' },
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.05'], // Error rate under 5%
    geological_api_errors: ['rate<0.02'], // Geological API errors under 2%
    seismic_processing_duration: ['p(90)<5000'], // 90% seismic processing under 5s
    real_time_data_latency: ['p(95)<500'], // 95% real-time data under 500ms
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

// Test data for geological simulations
const testVolcanoes = [
  { id: 'kilauea', lat: 19.4069, lon: -155.2834 },
  { id: 'etna', lat: 37.7510, lon: 14.9934 },
  { id: 'fuji', lat: 35.3606, lon: 138.7274 },
];

const seismicTestData = {
  magnitude: 4.5,
  depth: 10.2,
  location: { lat: 19.4069, lon: -155.2834, elevation: 1247 },
  timestamp: new Date().toISOString(),
  waveform_data: Array(1000).fill(0).map(() => Math.random() * 100),
};

export default function () {
  const headers = {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json',
  };

  // Test 1: Volcanic monitoring endpoint
  testVolcanicMonitoring(headers);
  
  // Test 2: Seismic data processing
  testSeismicProcessing(headers);
  
  // Test 3: Real-time data streaming
  testRealTimeStreaming(headers);
  
  // Test 4: 3D geological modeling
  testGeologicalModeling(headers);
  
  // Test 5: Resource assessment
  testResourceAssessment(headers);
  
  sleep(1);
}

function testVolcanicMonitoring(headers) {
  const volcano = testVolcanoes[Math.floor(Math.random() * testVolcanoes.length)];
  
  const response = http.get(`${BASE_URL}/api/v1/volcanoes/${volcano.id}/status`, {
    headers,
    tags: { endpoint: 'volcanic_monitoring' },
  });
  
  const success = check(response, {
    'volcanic status retrieved': (r) => r.status === 200,
    'response has alert level': (r) => JSON.parse(r.body).alert_level !== undefined,
    'response time acceptable': (r) => r.timings.duration < 2000,
  });
  
  geologicalApiErrors.add(!success);
  
  if (success) {
    volcanicDataThroughput.add(1);
  }
}

function testSeismicProcessing(headers) {
  const startTime = Date.now();
  
  const response = http.post(`${BASE_URL}/api/v1/seismic/analyze`, 
    JSON.stringify(seismicTestData), 
    {
      headers,
      tags: { endpoint: 'seismic_processing' },
    }
  );
  
  const processingTime = Date.now() - startTime;
  seismicProcessingTime.add(processingTime);
  
  check(response, {
    'seismic analysis completed': (r) => r.status === 200,
    'event classification present': (r) => JSON.parse(r.body).event_type !== undefined,
    'processing time reasonable': (r) => processingTime < 10000,
  });
}

function testRealTimeStreaming(headers) {
  const startTime = Date.now();
  
  const response = http.get(`${BASE_URL}/api/v1/streams/volcanic-data/latest`, {
    headers,
    tags: { endpoint: 'real_time_streaming' },
  });
  
  const latency = Date.now() - startTime;
  realTimeLatency.add(latency);
  
  check(response, {
    'real-time data available': (r) => r.status === 200,
    'data is recent': (r) => {
      const data = JSON.parse(r.body);
      const dataAge = Date.now() - new Date(data.timestamp).getTime();
      return dataAge < 60000; // Data less than 1 minute old
    },
    'low latency': (r) => latency < 1000,
  });
}

function testGeologicalModeling(headers) {
  const modelRequest = {
    region: {
      bounds: {
        north: 20.0,
        south: 19.0,
        east: -155.0,
        west: -156.0,
      },
      depth_range: { min: 0, max: 50000 },
    },
    resolution: 'medium',
    include_uncertainty: true,
  };
  
  const response = http.post(`${BASE_URL}/api/v1/geological/model/3d`, 
    JSON.stringify(modelRequest), 
    {
      headers,
      tags: { endpoint: 'geological_modeling' },
      timeout: '30s',
    }
  );
  
  check(response, {
    '3D model generated': (r) => r.status === 200 || r.status === 202,
    'model has spatial data': (r) => {
      if (r.status === 200) {
        const model = JSON.parse(r.body);
        return model.spatial_grid !== undefined;
      }
      return true; // Async processing accepted
    },
  });
}

function testResourceAssessment(headers) {
  const assessmentRequest = {
    region: {
      center: { lat: 19.4069, lon: -155.2834 },
      radius_km: 50,
    },
    resource_types: ['geothermal', 'mineral'],
    confidence_threshold: 0.7,
  };
  
  const response = http.post(`${BASE_URL}/api/v1/resources/assess`, 
    JSON.stringify(assessmentRequest), 
    {
      headers,
      tags: { endpoint: 'resource_assessment' },
      timeout: '20s',
    }
  );
  
  check(response, {
    'assessment completed': (r) => r.status === 200 || r.status === 202,
    'prospectivity data present': (r) => {
      if (r.status === 200) {
        const assessment = JSON.parse(r.body);
        return assessment.prospectivity_map !== undefined;
      }
      return true;
    },
  });
}

// Teardown function for cleanup
export function teardown(data) {
  console.log('Performance test completed');
  console.log(`Total volcanic data processed: ${volcanicDataThroughput.count}`);
}