import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics for geological API performance
const geologicalApiErrors = new Rate('geological_api_errors');
const seismicProcessingTime = new Trend('seismic_processing_duration');
const volcanicDataThroughput = new Counter('volcanic_data_throughput');
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
        { duration: '10s', target: 300 },
        { duration: '1m', target: 300 },
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

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Test data for geological simulations
const testVolcanoData = {
  volcano_id: 'test-volcano-001',
  location: { lat: 40.7128, lon: -74.0060, elevation: 2500 },
  seismic_data: generateSeismicData(),
  deformation_data: generateDeformationData(),
  gas_emissions: generateGasData(),
};

function generateSeismicData() {
  return Array.from({ length: 100 }, (_, i) => ({
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    magnitude: Math.random() * 5 + 1,
    depth: Math.random() * 50 + 5,
    location: {
      lat: 40.7128 + (Math.random() - 0.5) * 0.1,
      lon: -74.0060 + (Math.random() - 0.5) * 0.1,
    },
  }));
}

function generateDeformationData() {
  return Array.from({ length: 50 }, (_, i) => ({
    timestamp: new Date(Date.now() - i * 300000).toISOString(),
    displacement_x: (Math.random() - 0.5) * 0.1,
    displacement_y: (Math.random() - 0.5) * 0.1,
    displacement_z: (Math.random() - 0.5) * 0.05,
  }));
}

function generateGasData() {
  return {
    co2: Math.random() * 1000 + 100,
    so2: Math.random() * 500 + 50,
    h2s: Math.random() * 100 + 10,
    temperature: Math.random() * 200 + 800,
  };
}

export default function () {
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
  };

  // Test 1: Health check endpoint
  testHealthCheck(params);
  
  // Test 2: Seismic data processing
  testSeismicProcessing(params);
  
  // Test 3: Volcanic monitoring dashboard
  testVolcanicMonitoring(params);
  
  // Test 4: Real-time data streaming
  testRealTimeStreaming(params);
  
  // Test 5: 3D geological modeling
  test3DGeologicalModeling(params);
  
  // Test 6: Resource assessment
  testResourceAssessment(params);
  
  sleep(1);
}

function testHealthCheck(params) {
  const response = http.get(`${BASE_URL}/health`, params);
  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  });
}

function testSeismicProcessing(params) {
  const startTime = Date.now();
  const response = http.post(
    `${BASE_URL}/api/v1/seismic/process`,
    JSON.stringify({ seismic_data: testVolcanoData.seismic_data }),
    params
  );
  
  const processingTime = Date.now() - startTime;
  seismicProcessingTime.add(processingTime);
  
  const success = check(response, {
    'seismic processing status is 200': (r) => r.status === 200,
    'seismic processing has results': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.events && Array.isArray(data.events);
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!success) {
    geologicalApiErrors.add(1);
  }
}

function testVolcanicMonitoring(params) {
  const response = http.post(
    `${BASE_URL}/api/v1/volcano/monitor`,
    JSON.stringify(testVolcanoData),
    params
  );
  
  check(response, {
    'volcanic monitoring status is 200': (r) => r.status === 200,
    'volcanic monitoring returns forecast': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.eruption_probability !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  volcanicDataThroughput.add(1);
}

function testRealTimeStreaming(params) {
  const startTime = Date.now();
  const response = http.get(`${BASE_URL}/api/v1/stream/latest`, params);
  const latency = Date.now() - startTime;
  
  realTimeLatency.add(latency);
  
  check(response, {
    'real-time stream status is 200': (r) => r.status === 200,
    'real-time stream latency < 500ms': (r) => r.timings.duration < 500,
  });
}

function test3DGeologicalModeling(params) {
  const modelingRequest = {
    region: {
      bounds: {
        north: 41.0,
        south: 40.0,
        east: -73.0,
        west: -75.0,
      },
      depth_range: { min: 0, max: 50000 },
    },
    resolution: 'medium',
    include_subsurface: true,
  };
  
  const response = http.post(
    `${BASE_URL}/api/v1/geology/model-3d`,
    JSON.stringify(modelingRequest),
    params
  );
  
  check(response, {
    '3D modeling status is 200 or 202': (r) => r.status === 200 || r.status === 202,
    '3D modeling returns job_id': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.job_id !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
}

function testResourceAssessment(params) {
  const assessmentRequest = {
    region: testVolcanoData.location,
    resource_types: ['copper', 'gold', 'lithium'],
    assessment_depth: 5000,
    confidence_level: 0.95,
  };
  
  const response = http.post(
    `${BASE_URL}/api/v1/resources/assess`,
    JSON.stringify(assessmentRequest),
    params
  );
  
  check(response, {
    'resource assessment status is 200': (r) => r.status === 200,
    'resource assessment returns prospects': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.prospects && Array.isArray(data.prospects);
      } catch (e) {
        return false;
      }
    },
  });
}

// Teardown function for cleanup
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Geological API error rate: ${geologicalApiErrors.rate}`);
  console.log(`Average seismic processing time: ${seismicProcessingTime.avg}ms`);
  console.log(`Volcanic data throughput: ${volcanicDataThroughput.count} requests`);
  console.log(`Average real-time latency: ${realTimeLatency.avg}ms`);
}