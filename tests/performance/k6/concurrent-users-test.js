import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { SharedArray } from 'k6/data';

// Concurrent user metrics
const concurrentUsers = new Gauge('concurrent_active_users');
const userSessionDuration = new Trend('user_session_duration');
const collaborationConflicts = new Rate('collaboration_conflicts');
const resourceContention = new Rate('resource_contention');
const systemThroughput = new Counter('system_throughput_requests');
const userWorkflowCompletion = new Rate('user_workflow_completion');

// Shared test data for concurrent access
const testVolcanoes = new SharedArray('volcanoes', function () {
  return [
    { id: 'kilauea', name: 'Kilauea', lat: 19.4069, lon: -155.2834 },
    { id: 'etna', name: 'Mount Etna', lat: 37.7510, lon: 14.9934 },
    { id: 'fuji', name: 'Mount Fuji', lat: 35.3606, lon: 138.7274 },
    { id: 'vesuvius', name: 'Mount Vesuvius', lat: 40.8214, lon: 14.4289 },
    { id: 'rainier', name: 'Mount Rainier', lat: 46.8529, lon: -121.7604 },
  ];
});

const userProfiles = new SharedArray('user_profiles', function () {
  return [
    { type: 'researcher', permissions: ['read', 'analyze', 'export'], concurrent_limit: 50 },
    { type: 'operator', permissions: ['read', 'monitor', 'alert'], concurrent_limit: 100 },
    { type: 'admin', permissions: ['read', 'write', 'delete', 'configure'], concurrent_limit: 10 },
    { type: 'student', permissions: ['read'], concurrent_limit: 200 },
    { type: 'emergency', permissions: ['read', 'monitor', 'alert', 'emergency'], concurrent_limit: 25 },
  ];
});

export const options = {
  scenarios: {
    // Normal operational load
    normal_operations: {
      executor: 'constant-vus',
      vus: 50,
      duration: '15m',
      tags: { scenario: 'normal_operations' },
    },
    // Peak research activity (conference season)
    peak_research: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '5m', target: 100 },
        { duration: '10m', target: 200 },
        { duration: '15m', target: 300 },
        { duration: '10m', target: 200 },
        { duration: '5m', target: 0 },
      ],
      tags: { scenario: 'peak_research' },
    },
    // Emergency response scenario
    emergency_response: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      stages: [
        { duration: '2m', target: 50 }, // Rapid escalation
        { duration: '5m', target: 100 }, // Peak emergency response
        { duration: '3m', target: 20 }, // Wind down
      ],
      preAllocatedVUs: 50,
      maxVUs: 150,
      tags: { scenario: 'emergency_response' },
    },
    // Collaborative research session
    collaborative_session: {
      executor: 'shared-iterations',
      vus: 25,
      iterations: 500,
      maxDuration: '20m',
      tags: { scenario: 'collaborative_session' },
    },
    // Long-running monitoring
    continuous_monitoring: {
      executor: 'constant-vus',
      vus: 30,
      duration: '30m',
      tags: { scenario: 'continuous_monitoring' },
    },
  },
  thresholds: {
    concurrent_active_users: ['value<=500'], // Max 500 concurrent users
    user_session_duration: ['p(95)<300000'], // 95% of sessions under 5 minutes
    collaboration_conflicts: ['rate<0.02'], // Collaboration conflicts under 2%
    resource_contention: ['rate<0.05'], // Resource contention under 5%
    system_throughput_requests: ['rate>100'], // At least 100 requests/second
    user_workflow_completion: ['rate>0.95'], // 95% workflow completion rate
    http_req_duration: ['p(95)<3000'], // 95% of requests under 3s
    http_req_failed: ['rate<0.02'], // Error rate under 2%
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

export default function () {
  const userProfile = userProfiles[Math.floor(Math.random() * userProfiles.length)];
  const volcano = testVolcanoes[Math.floor(Math.random() * testVolcanoes.length)];
  const sessionStart = Date.now();
  
  // Update concurrent users metric
  concurrentUsers.add(1);
  
  const headers = {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json',
    'X-User-Type': userProfile.type,
    'X-Session-ID': `session_${__VU}_${__ITER}`,
  };

  // Execute user workflow based on profile
  let workflowSuccess = false;
  
  try {
    switch (userProfile.type) {
      case 'researcher':
        workflowSuccess = executeResearcherWorkflow(headers, volcano);
        break;
      case 'operator':
        workflowSuccess = executeOperatorWorkflow(headers, volcano);
        break;
      case 'admin':
        workflowSuccess = executeAdminWorkflow(headers, volcano);
        break;
      case 'student':
        workflowSuccess = executeStudentWorkflow(headers, volcano);
        break;
      case 'emergency':
        workflowSuccess = executeEmergencyWorkflow(headers, volcano);
        break;
    }
  } catch (error) {
    console.error(`Workflow error for ${userProfile.type}: ${error}`);
    workflowSuccess = false;
  }
  
  userWorkflowCompletion.add(workflowSuccess ? 1 : 0);
  
  const sessionDuration = Date.now() - sessionStart;
  userSessionDuration.add(sessionDuration);
  
  // Simulate user think time
  sleep(Math.random() * 3 + 1);
}

function executeResearcherWorkflow(headers, volcano) {
  let success = true;
  
  // 1. Access volcano data
  const volcanoResponse = http.get(`${BASE_URL}/api/v1/volcanoes/${volcano.id}`, {
    headers,
    tags: { workflow: 'researcher', step: 'volcano_data' },
  });
  
  success &= check(volcanoResponse, {
    'volcano data retrieved': (r) => r.status === 200,
  });
  
  systemThroughput.add(1);
  
  // 2. Request historical seismic data
  const seismicRequest = {
    volcano_id: volcano.id,
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    magnitude_min: 2.0,
    include_waveforms: true,
  };
  
  const seismicResponse = http.post(`${BASE_URL}/api/v1/seismic/historical`, 
    JSON.stringify(seismicRequest), 
    {
      headers,
      tags: { workflow: 'researcher', step: 'seismic_data' },
    }
  );
  
  success &= check(seismicResponse, {
    'seismic data retrieved': (r) => r.status === 200 || r.status === 202,
  });
  
  systemThroughput.add(1);
  
  // 3. Generate 3D geological model
  const modelRequest = {
    region: {
      center: { lat: volcano.lat, lon: volcano.lon },
      radius_km: 25,
    },
    resolution: 'high',
    include_uncertainty: true,
  };
  
  const modelResponse = http.post(`${BASE_URL}/api/v1/geological/model/3d`, 
    JSON.stringify(modelRequest), 
    {
      headers,
      tags: { workflow: 'researcher', step: 'geological_model' },
      timeout: '30s',
    }
  );
  
  success &= check(modelResponse, {
    '3D model generated': (r) => r.status === 200 || r.status === 202,
  });
  
  systemThroughput.add(1);
  
  // 4. Test collaborative annotation (potential conflict point)
  const annotationData = {
    volcano_id: volcano.id,
    annotation_type: 'research_note',
    content: `Research observation by user ${__VU} at ${new Date().toISOString()}`,
    location: { lat: volcano.lat, lon: volcano.lon },
    visibility: 'team',
  };
  
  const annotationResponse = http.post(`${BASE_URL}/api/v1/annotations`, 
    JSON.stringify(annotationData), 
    {
      headers,
      tags: { workflow: 'researcher', step: 'annotation' },
    }
  );
  
  const annotationSuccess = check(annotationResponse, {
    'annotation created': (r) => r.status === 200 || r.status === 201,
    'no collaboration conflict': (r) => r.status !== 409,
  });
  
  if (annotationResponse.status === 409) {
    collaborationConflicts.add(1);
  }
  
  success &= annotationSuccess;
  systemThroughput.add(1);
  
  return success;
}

function executeOperatorWorkflow(headers, volcano) {
  let success = true;
  
  // 1. Check real-time volcano status
  const statusResponse = http.get(`${BASE_URL}/api/v1/volcanoes/${volcano.id}/status`, {
    headers,
    tags: { workflow: 'operator', step: 'status_check' },
  });
  
  success &= check(statusResponse, {
    'status retrieved': (r) => r.status === 200,
    'alert level present': (r) => {
      if (r.status === 200) {
        const data = JSON.parse(r.body);
        return data.alert_level !== undefined;
      }
      return false;
    },
  });
  
  systemThroughput.add(1);
  
  // 2. Access real-time monitoring streams
  const streamResponse = http.get(`${BASE_URL}/api/v1/streams/volcanic-data/latest`, {
    headers,
    tags: { workflow: 'operator', step: 'monitoring_stream' },
  });
  
  success &= check(streamResponse, {
    'monitoring data available': (r) => r.status === 200,
  });
  
  systemThroughput.add(1);
  
  // 3. Test alert system access (resource contention point)
  const alertResponse = http.get(`${BASE_URL}/api/v1/alerts/active`, {
    headers,
    tags: { workflow: 'operator', step: 'alert_access' },
  });
  
  const alertSuccess = check(alertResponse, {
    'alerts retrieved': (r) => r.status === 200,
    'no resource contention': (r) => r.status !== 503,
  });
  
  if (alertResponse.status === 503) {
    resourceContention.add(1);
  }
  
  success &= alertSuccess;
  systemThroughput.add(1);
  
  return success;
}

function executeAdminWorkflow(headers, volcano) {
  let success = true;
  
  // 1. System health check
  const healthResponse = http.get(`${BASE_URL}/api/v1/admin/system/health`, {
    headers,
    tags: { workflow: 'admin', step: 'system_health' },
  });
  
  success &= check(healthResponse, {
    'system health retrieved': (r) => r.status === 200,
  });
  
  systemThroughput.add(1);
  
  // 2. User management operations
  const usersResponse = http.get(`${BASE_URL}/api/v1/admin/users/active`, {
    headers,
    tags: { workflow: 'admin', step: 'user_management' },
  });
  
  success &= check(usersResponse, {
    'active users retrieved': (r) => r.status === 200,
  });
  
  systemThroughput.add(1);
  
  // 3. Configuration update (potential contention)
  const configUpdate = {
    setting: 'alert_threshold',
    value: Math.random() * 5 + 2, // Random threshold between 2-7
    volcano_id: volcano.id,
  };
  
  const configResponse = http.put(`${BASE_URL}/api/v1/admin/config/volcano`, 
    JSON.stringify(configUpdate), 
    {
      headers,
      tags: { workflow: 'admin', step: 'config_update' },
    }
  );
  
  const configSuccess = check(configResponse, {
    'config updated': (r) => r.status === 200,
    'no resource lock': (r) => r.status !== 423,
  });
  
  if (configResponse.status === 423) {
    resourceContention.add(1);
  }
  
  success &= configSuccess;
  systemThroughput.add(1);
  
  return success;
}

function executeStudentWorkflow(headers, volcano) {
  let success = true;
  
  // 1. Basic volcano information
  const volcanoResponse = http.get(`${BASE_URL}/api/v1/volcanoes/${volcano.id}/info`, {
    headers,
    tags: { workflow: 'student', step: 'volcano_info' },
  });
  
  success &= check(volcanoResponse, {
    'volcano info retrieved': (r) => r.status === 200,
  });
  
  systemThroughput.add(1);
  
  // 2. Educational content access
  const educationResponse = http.get(`${BASE_URL}/api/v1/education/volcano/${volcano.id}`, {
    headers,
    tags: { workflow: 'student', step: 'educational_content' },
  });
  
  success &= check(educationResponse, {
    'educational content available': (r) => r.status === 200,
  });
  
  systemThroughput.add(1);
  
  return success;
}

function executeEmergencyWorkflow(headers, volcano) {
  let success = true;
  
  // 1. Immediate status check
  const emergencyResponse = http.get(`${BASE_URL}/api/v1/emergency/volcano/${volcano.id}/status`, {
    headers,
    tags: { workflow: 'emergency', step: 'emergency_status' },
  });
  
  success &= check(emergencyResponse, {
    'emergency status retrieved': (r) => r.status === 200,
    'response time critical': (r) => r.timings.duration < 1000, // Under 1 second
  });
  
  systemThroughput.add(1);
  
  // 2. Alert broadcasting
  const alertBroadcast = {
    volcano_id: volcano.id,
    alert_level: 'watch',
    message: `Emergency test alert from user ${__VU}`,
    priority: 'high',
  };
  
  const broadcastResponse = http.post(`${BASE_URL}/api/v1/emergency/alerts/broadcast`, 
    JSON.stringify(alertBroadcast), 
    {
      headers,
      tags: { workflow: 'emergency', step: 'alert_broadcast' },
    }
  );
  
  success &= check(broadcastResponse, {
    'alert broadcast successful': (r) => r.status === 200 || r.status === 202,
  });
  
  systemThroughput.add(1);
  
  return success;
}

export function teardown(data) {
  console.log('Concurrent user test completed');
  console.log(`Peak concurrent users: ${concurrentUsers.value}`);
  console.log(`Average session duration: ${userSessionDuration.avg}ms`);
  console.log(`Collaboration conflict rate: ${(collaborationConflicts.rate * 100).toFixed(2)}%`);
  console.log(`Resource contention rate: ${(resourceContention.rate * 100).toFixed(2)}%`);
  console.log(`System throughput: ${systemThroughput.count / (30 * 60)} requests/second`);
  console.log(`Workflow completion rate: ${(userWorkflowCompletion.rate * 100).toFixed(2)}%`);
}