import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Network-specific performance metrics
const networkLatency = new Trend('network_latency');
const dataTransferRate = new Trend('data_transfer_rate');
const connectionTimeouts = new Rate('connection_timeouts');
const realTimeDataLoss = new Rate('real_time_data_loss');
const adaptiveQualityChanges = new Counter('adaptive_quality_changes');

export const options = {
  scenarios: {
    // Slow 3G network simulation
    slow_3g_network: {
      executor: 'constant-vus',
      vus: 5,
      duration: '10m',
      tags: { network_type: 'slow_3g' },
      env: { NETWORK_PROFILE: 'slow_3g' },
    },
    // Fast 3G network simulation
    fast_3g_network: {
      executor: 'constant-vus',
      vus: 10,
      duration: '8m',
      tags: { network_type: 'fast_3g' },
      env: { NETWORK_PROFILE: 'fast_3g' },
    },
    // Satellite connection simulation (high latency)
    satellite_connection: {
      executor: 'constant-vus',
      vus: 3,
      duration: '12m',
      tags: { network_type: 'satellite' },
      env: { NETWORK_PROFILE: 'satellite' },
    },
    // Intermittent connectivity
    intermittent_network: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 5 },
        { duration: '1m', target: 0 }, // Simulate disconnection
        { duration: '2m', target: 8 },
        { duration: '1m', target: 0 }, // Another disconnection
        { duration: '2m', target: 5 },
      ],
      tags: { network_type: 'intermittent' },
    },
  },
  thresholds: {
    network_latency: ['p(95)<5000'], // 95% of requests under 5s on slow networks
    data_transfer_rate: ['p(50)>10'], // Median transfer rate > 10 KB/s
    connection_timeouts: ['rate<0.1'], // Timeout rate under 10%
    real_time_data_loss: ['rate<0.05'], // Data loss under 5%
    http_req_duration: ['p(90)<10000'], // 90% of requests under 10s
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';

// Network profiles for different connection types
const networkProfiles = {
  slow_3g: {
    bandwidth: 50, // KB/s
    latency: 2000, // ms
    packetLoss: 0.02, // 2%
    timeout: 30000, // 30s
  },
  fast_3g: {
    bandwidth: 200, // KB/s
    latency: 500, // ms
    packetLoss: 0.005, // 0.5%
    timeout: 15000, // 15s
  },
  satellite: {
    bandwidth: 100, // KB/s
    latency: 800, // ms
    packetLoss: 0.01, // 1%
    timeout: 45000, // 45s
  },
};

export default function () {
  const networkProfile = __ENV.NETWORK_PROFILE || 'fast_3g';
  const profile = networkProfiles[networkProfile];
  
  const headers = {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json',
  };

  // Test different data types with network constraints
  testRealTimeDataStreaming(headers, profile);
  testLargeDatasetDownload(headers, profile);
  testAdaptiveQualityStreaming(headers, profile);
  testOfflineCapabilities(headers, profile);
  
  // Simulate network delay based on profile
  sleep(profile.latency / 1000);
}

function testRealTimeDataStreaming(headers, profile) {
  const startTime = Date.now();
  
  // Test WebSocket-like real-time data endpoint
  const response = http.get(`${BASE_URL}/api/v1/streams/seismic/real-time`, {
    headers,
    tags: { test_type: 'real_time_streaming' },
    timeout: `${profile.timeout}ms`,
  });
  
  const latency = Date.now() - startTime;
  networkLatency.add(latency);
  
  const success = check(response, {
    'real-time data received': (r) => r.status === 200,
    'data is recent': (r) => {
      if (r.status === 200) {
        try {
          const data = JSON.parse(r.body);
          const dataAge = Date.now() - new Date(data.timestamp).getTime();
          return dataAge < 30000; // Data less than 30 seconds old
        } catch (e) {
          return false;
        }
      }
      return false;
    },
    'acceptable latency for network': (r) => latency < profile.timeout,
  });
  
  if (!success) {
    realTimeDataLoss.add(1);
  }
  
  connectionTimeouts.add(response.status === 0);
}

function testLargeDatasetDownload(headers, profile) {
  const startTime = Date.now();
  
  // Request large geological dataset
  const datasetRequest = {
    region: {
      bounds: {
        north: 20.0,
        south: 18.0,
        east: -154.0,
        west: -157.0,
      },
    },
    data_types: ['seismic_events', 'geological_units', 'elevation_data'],
    resolution: 'high',
    format: 'geojson',
    compression: 'gzip',
  };
  
  const response = http.post(`${BASE_URL}/api/v1/geological/dataset/export`, 
    JSON.stringify(datasetRequest), 
    {
      headers,
      tags: { test_type: 'large_dataset_download' },
      timeout: `${profile.timeout}ms`,
    }
  );
  
  const downloadTime = Date.now() - startTime;
  
  if (response.status === 200 && response.body) {
    const dataSize = response.body.length; // bytes
    const transferRate = (dataSize / 1024) / (downloadTime / 1000); // KB/s
    dataTransferRate.add(transferRate);
    
    check(response, {
      'large dataset downloaded': (r) => r.status === 200,
      'transfer rate acceptable': (r) => transferRate > profile.bandwidth * 0.5, // At least 50% of expected bandwidth
      'data integrity maintained': (r) => {
        try {
          // Basic JSON validation for geological data
          const data = JSON.parse(r.body);
          return data.features !== undefined || data.type === 'FeatureCollection';
        } catch (e) {
          return false;
        }
      },
    });
  } else {
    connectionTimeouts.add(response.status === 0);
  }
}

function testAdaptiveQualityStreaming(headers, profile) {
  // Test adaptive quality based on network conditions
  const qualityLevels = ['low', 'medium', 'high'];
  let selectedQuality = 'high';
  
  // Adapt quality based on network profile
  if (profile.bandwidth < 100) {
    selectedQuality = 'low';
    adaptiveQualityChanges.add(1);
  } else if (profile.bandwidth < 200) {
    selectedQuality = 'medium';
    adaptiveQualityChanges.add(1);
  }
  
  const streamRequest = {
    stream_type: 'volcanic_monitoring',
    quality: selectedQuality,
    adaptive: true,
    buffer_size: profile.latency > 1000 ? 'large' : 'medium',
  };
  
  const response = http.post(`${BASE_URL}/api/v1/streams/adaptive`, 
    JSON.stringify(streamRequest), 
    {
      headers,
      tags: { test_type: 'adaptive_streaming' },
      timeout: `${profile.timeout}ms`,
    }
  );
  
  check(response, {
    'adaptive stream started': (r) => r.status === 200 || r.status === 202,
    'quality adapted to network': (r) => {
      if (r.status === 200) {
        try {
          const data = JSON.parse(r.body);
          return data.quality === selectedQuality;
        } catch (e) {
          return false;
        }
      }
      return true;
    },
  });
}

function testOfflineCapabilities(headers, profile) {
  // Test offline data caching and synchronization
  const offlineRequest = {
    action: 'cache_for_offline',
    data_types: ['recent_seismic_events', 'volcano_status', 'alert_levels'],
    region: 'hawaii',
    cache_duration: '24h',
  };
  
  const response = http.post(`${BASE_URL}/api/v1/offline/cache`, 
    JSON.stringify(offlineRequest), 
    {
      headers,
      tags: { test_type: 'offline_capabilities' },
      timeout: `${profile.timeout}ms`,
    }
  );
  
  check(response, {
    'offline cache prepared': (r) => r.status === 200 || r.status === 202,
    'cache size reasonable': (r) => {
      if (r.status === 200 && r.body) {
        try {
          const data = JSON.parse(r.body);
          return data.cache_size_mb < 50; // Cache under 50MB for mobile
        } catch (e) {
          return false;
        }
      }
      return true;
    },
  });
}

// Test network resilience with connection drops
function testNetworkResilience(headers, profile) {
  // Simulate connection drop by using very short timeout
  const response = http.get(`${BASE_URL}/api/v1/health`, {
    headers,
    tags: { test_type: 'network_resilience' },
    timeout: '100ms', // Very short timeout to simulate connection issues
  });
  
  // Test automatic retry mechanism
  if (response.status === 0) {
    sleep(2); // Wait for retry
    
    const retryResponse = http.get(`${BASE_URL}/api/v1/health`, {
      headers,
      tags: { test_type: 'network_retry' },
      timeout: `${profile.timeout}ms`,
    });
    
    check(retryResponse, {
      'connection recovered': (r) => r.status === 200,
      'service available after retry': (r) => {
        if (r.status === 200) {
          try {
            const data = JSON.parse(r.body);
            return data.status === 'healthy';
          } catch (e) {
            return false;
          }
        }
        return false;
      },
    });
  }
}

export function teardown(data) {
  console.log('Network throttling test completed');
  console.log(`Average network latency: ${networkLatency.avg}ms`);
  console.log(`Average data transfer rate: ${dataTransferRate.avg} KB/s`);
  console.log(`Connection timeout rate: ${(connectionTimeouts.rate * 100).toFixed(2)}%`);
  console.log(`Real-time data loss rate: ${(realTimeDataLoss.rate * 100).toFixed(2)}%`);
}