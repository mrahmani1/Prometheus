/**
 * Network Throttling Tests for Real-Time Geological Data
 * Tests application performance under various network conditions
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

class NetworkThrottlingTester {
  constructor(config = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:3000',
      outputDir: config.outputDir || './tests/performance/network-reports',
      testDuration: config.testDuration || 120000, // 2 minutes per test
      ...config,
    };

    // Network profiles for different geological monitoring scenarios
    this.networkProfiles = [
      {
        name: 'fast-3g',
        description: 'Fast 3G - Remote field station',
        downloadThroughput: 1.6 * 1024 * 1024 / 8, // 1.6 Mbps in bytes/s
        uploadThroughput: 750 * 1024 / 8, // 750 Kbps in bytes/s
        latency: 150, // ms
        scenario: 'Remote volcanic monitoring station with cellular connection',
      },
      {
        name: 'slow-3g',
        description: 'Slow 3G - Emergency response',
        downloadThroughput: 500 * 1024 / 8, // 500 Kbps in bytes/s
        uploadThroughput: 250 * 1024 / 8, // 250 Kbps in bytes/s
        latency: 300, // ms
        scenario: 'Emergency response team in remote area',
      },
      {
        name: 'satellite',
        description: 'Satellite connection - Arctic research',
        downloadThroughput: 2 * 1024 * 1024 / 8, // 2 Mbps in bytes/s
        uploadThroughput: 512 * 1024 / 8, // 512 Kbps in bytes/s
        latency: 600, // ms - high latency for satellite
        scenario: 'Arctic geological research station',
      },
      {
        name: 'edge',
        description: 'EDGE - Extreme remote location',
        downloadThroughput: 240 * 1024 / 8, // 240 Kbps in bytes/s
        uploadThroughput: 200 * 1024 / 8, // 200 Kbps in bytes/s
        latency: 400, // ms
        scenario: 'Extreme remote geological survey',
      },
      {
        name: 'offline-intermittent',
        description: 'Intermittent connectivity',
        downloadThroughput: 0, // Will be toggled during test
        uploadThroughput: 0,
        latency: 0,
        scenario: 'Unstable connection with periodic outages',
        intermittent: true,
      },
    ];

    // Test scenarios for geological applications
    this.testScenarios = [
      {
        name: 'real-time-seismic-monitoring',
        url: '/monitoring/seismic-realtime',
        description: 'Real-time seismic data streaming',
        criticalFeatures: [
          'data-stream-connection',
          'alert-notifications',
          'waveform-display',
        ],
        dataIntensive: true,
        realTime: true,
      },
      {
        name: 'volcanic-alert-dashboard',
        url: '/volcano/alerts',
        description: 'Volcanic alert monitoring dashboard',
        criticalFeatures: [
          'alert-status',
          'webcam-feeds',
          'sensor-readings',
        ],
        dataIntensive: true,
        realTime: true,
      },
      {
        name: '3d-geological-model-viewer',
        url: '/visualization/3d-model',
        description: '3D geological model visualization',
        criticalFeatures: [
          'model-loading',
          'texture-streaming',
          'interaction-responsiveness',
        ],
        dataIntensive: true,
        realTime: false,
      },
      {
        name: 'field-data-collection',
        url: '/field/data-entry',
        description: 'Field data collection interface',
        criticalFeatures: [
          'form-submission',
          'photo-upload',
          'gps-tracking',
        ],
        dataIntensive: false,
        realTime: false,
      },
      {
        name: 'emergency-response-portal',
        url: '/emergency/response',
        description: 'Emergency response coordination',
        criticalFeatures: [
          'communication-panel',
          'resource-tracking',
          'evacuation-maps',
        ],
        dataIntensive: true,
        realTime: true,
      },
    ];
  }

  async runNetworkThrottlingTests() {
    console.log('🌐 Starting Network Throttling Tests for Geological Applications...');
    
    await this.ensureOutputDirectory();
    const results = [];
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    for (const scenario of this.testScenarios) {
      console.log(`\n📊 Testing scenario: ${scenario.name}`);
      
      for (const profile of this.networkProfiles) {
        console.log(`  🔗 Network profile: ${profile.name} (${profile.description})`);
        
        try {
          const testResult = await this.runSingleNetworkTest(scenario, profile);
          results.push(testResult);
          
          console.log(`    ✓ Completed - Performance Score: ${testResult.performanceScore}/100`);
          console.log(`    📈 Data Transfer: ${(testResult.totalDataTransferred / 1024 / 1024).toFixed(2)} MB`);
          console.log(`    ⏱️  Average Latency: ${testResult.averageLatency}ms`);
          
        } catch (error) {
          console.error(`    ❌ Failed: ${error.message}`);
          results.push({
            scenario: scenario.name,
            networkProfile: profile.name,
            error: error.message,
            performanceScore: 0,
          });
        }
      }
    }

    // Generate comprehensive report
    const report = this.generateNetworkPerformanceReport(results, timestamp);
    const reportPath = path.join(
      this.config.outputDir,
      `network-throttling-report-${timestamp}.json`
    );
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

    // Generate HTML visualization
    await this.generateNetworkVisualization(results, timestamp);

    return report;
  }

  async runSingleNetworkTest(scenario, profile) {
    const browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });

    const page = await browser.newPage();
    
    try {
      // Set up network throttling
      const client = await page.target().createCDPSession();
      await client.send('Network.enable');
      
      if (profile.intermittent) {
        // Handle intermittent connectivity separately
        return await this.runIntermittentConnectivityTest(page, client, scenario, profile);
      } else {
        await client.send('Network.emulateNetworkConditions', {
          offline: false,
          downloadThroughput: profile.downloadThroughput,
          uploadThroughput: profile.uploadThroughput,
          latency: profile.latency,
        });
      }

      // Set up performance monitoring
      await client.send('Performance.enable');
      await client.send('Runtime.enable');

      // Track network requests
      const networkRequests = [];
      const networkErrors = [];
      
      client.on('Network.responseReceived', (params) => {
        networkRequests.push({
          url: params.response.url,
          status: params.response.status,
          mimeType: params.response.mimeType,
          timestamp: params.timestamp,
          fromCache: params.response.fromDiskCache || params.response.fromServiceWorker,
        });
      });

      client.on('Network.loadingFailed', (params) => {
        networkErrors.push({
          url: params.request.url,
          errorText: params.errorText,
          timestamp: params.timestamp,
        });
      });

      // Navigate to test page
      const navigationStart = Date.now();
      const fullUrl = `${this.config.baseUrl}${scenario.url}`;
      
      try {
        await page.goto(fullUrl, { 
          waitUntil: 'networkidle0', 
          timeout: 60000 
        });
      } catch (error) {
        if (error.name === 'TimeoutError') {
          console.warn(`    ⚠️  Navigation timeout for ${scenario.name} with ${profile.name}`);
        } else {
          throw error;
        }
      }

      const navigationEnd = Date.now();
      const navigationTime = navigationEnd - navigationStart;

      // Wait for initial load
      await page.waitForTimeout(5000);

      // Test critical features
      const featureResults = await this.testCriticalFeatures(page, scenario.criticalFeatures, profile);

      // Monitor real-time performance if applicable
      let realTimeMetrics = {};
      if (scenario.realTime) {
        realTimeMetrics = await this.monitorRealTimePerformance(page, client, profile);
      }

      // Test data-intensive operations if applicable
      let dataIntensiveMetrics = {};
      if (scenario.dataIntensive) {
        dataIntensiveMetrics = await this.testDataIntensiveOperations(page, client, profile);
      }

      // Collect final metrics
      const performanceMetrics = await client.send('Performance.getMetrics');
      const memoryUsage = await page.evaluate(() => {
        return performance.memory ? {
          usedJSHeapSize: performance.memory.usedJSHeapSize,
          totalJSHeapSize: performance.memory.totalJSHeapSize,
        } : {};
      });

      // Calculate performance score
      const performanceScore = this.calculatePerformanceScore({
        navigationTime,
        networkErrors: networkErrors.length,
        featureResults,
        realTimeMetrics,
        dataIntensiveMetrics,
        profile,
      });

      return {
        scenario: scenario.name,
        scenarioDescription: scenario.description,
        networkProfile: profile.name,
        networkDescription: profile.description,
        networkScenario: profile.scenario,
        navigationTime,
        performanceScore,
        networkRequests: networkRequests.length,
        networkErrors: networkErrors.length,
        totalDataTransferred: this.calculateDataTransferred(networkRequests),
        averageLatency: this.calculateAverageLatency(networkRequests),
        featureResults,
        realTimeMetrics,
        dataIntensiveMetrics,
        memoryUsage,
        recommendations: this.generateNetworkRecommendations({
          scenario,
          profile,
          performanceScore,
          networkErrors: networkErrors.length,
          navigationTime,
        }),
        timestamp: new Date().toISOString(),
      };

    } finally {
      await browser.close();
    }
  }

  async runIntermittentConnectivityTest(page, client, scenario, profile) {
    console.log('    🔄 Testing intermittent connectivity...');
    
    const results = {
      connectionDrops: 0,
      reconnectionTimes: [],
      dataLosses: 0,
      offlineFunctionality: {},
    };

    // Simulate connection drops
    const testDuration = 60000; // 1 minute
    const dropInterval = 15000; // Drop every 15 seconds
    const dropDuration = 5000; // 5 second outages
    
    const startTime = Date.now();
    
    // Navigate initially
    await page.goto(`${this.config.baseUrl}${scenario.url}`, { 
      waitUntil: 'networkidle0', 
      timeout: 30000 
    });

    while (Date.now() - startTime < testDuration) {
      // Normal connectivity period
      await client.send('Network.emulateNetworkConditions', {
        offline: false,
        downloadThroughput: 1 * 1024 * 1024 / 8, // 1 Mbps
        uploadThroughput: 500 * 1024 / 8, // 500 Kbps
        latency: 100,
      });

      await page.waitForTimeout(dropInterval - dropDuration);

      // Connection drop
      const dropStart = Date.now();
      await client.send('Network.emulateNetworkConditions', {
        offline: true,
        downloadThroughput: 0,
        uploadThroughput: 0,
        latency: 0,
      });

      results.connectionDrops++;

      // Test offline functionality
      const offlineTest = await this.testOfflineFunctionality(page, scenario);
      Object.assign(results.offlineFunctionality, offlineTest);

      await page.waitForTimeout(dropDuration);

      // Reconnection
      await client.send('Network.emulateNetworkConditions', {
        offline: false,
        downloadThroughput: 1 * 1024 * 1024 / 8,
        uploadThroughput: 500 * 1024 / 8,
        latency: 100,
      });

      const reconnectionTime = Date.now() - dropStart - dropDuration;
      results.reconnectionTimes.push(reconnectionTime);
    }

    // Calculate performance score for intermittent connectivity
    const avgReconnectionTime = results.reconnectionTimes.reduce((a, b) => a + b, 0) / results.reconnectionTimes.length;
    const performanceScore = Math.max(0, 100 - (results.connectionDrops * 10) - (avgReconnectionTime / 100));

    return {
      scenario: scenario.name,
      networkProfile: profile.name,
      performanceScore,
      intermittentResults: results,
      recommendations: this.generateIntermittentConnectivityRecommendations(results),
    };
  }

  async testCriticalFeatures(page, features, profile) {
    const results = {};

    for (const feature of features) {
      const startTime = Date.now();
      
      try {
        switch (feature) {
          case 'data-stream-connection':
            results[feature] = await this.testDataStreamConnection(page, profile);
            break;
          case 'alert-notifications':
            results[feature] = await this.testAlertNotifications(page, profile);
            break;
          case 'waveform-display':
            results[feature] = await this.testWaveformDisplay(page, profile);
            break;
          case 'alert-status':
            results[feature] = await this.testAlertStatus(page, profile);
            break;
          case 'webcam-feeds':
            results[feature] = await this.testWebcamFeeds(page, profile);
            break;
          case 'sensor-readings':
            results[feature] = await this.testSensorReadings(page, profile);
            break;
          case 'model-loading':
            results[feature] = await this.testModelLoading(page, profile);
            break;
          case 'texture-streaming':
            results[feature] = await this.testTextureStreaming(page, profile);
            break;
          case 'interaction-responsiveness':
            results[feature] = await this.testInteractionResponsiveness(page, profile);
            break;
          case 'form-submission':
            results[feature] = await this.testFormSubmission(page, profile);
            break;
          case 'photo-upload':
            results[feature] = await this.testPhotoUpload(page, profile);
            break;
          case 'gps-tracking':
            results[feature] = await this.testGPSTracking(page, profile);
            break;
          case 'communication-panel':
            results[feature] = await this.testCommunicationPanel(page, profile);
            break;
          case 'resource-tracking':
            results[feature] = await this.testResourceTracking(page, profile);
            break;
          case 'evacuation-maps':
            results[feature] = await this.testEvacuationMaps(page, profile);
            break;
          default:
            results[feature] = { success: false, error: 'Unknown feature' };
        }
        
        results[feature].duration = Date.now() - startTime;
        
      } catch (error) {
        results[feature] = {
          success: false,
          error: error.message,
          duration: Date.now() - startTime,
        };
      }
    }

    return results;
  }

  async testDataStreamConnection(page, profile) {
    // Test WebSocket connection for real-time data
    const result = await page.evaluate(() => {
      return new Promise((resolve) => {
        const ws = new WebSocket('ws://localhost:8080/seismic-stream');
        const startTime = Date.now();
        let messageCount = 0;
        
        ws.onopen = () => {
          setTimeout(() => {
            resolve({
              success: true,
              connectionTime: Date.now() - startTime,
              messagesReceived: messageCount,
            });
            ws.close();
          }, 10000); // Test for 10 seconds
        };
        
        ws.onmessage = () => {
          messageCount++;
        };
        
        ws.onerror = () => {
          resolve({
            success: false,
            error: 'WebSocket connection failed',
            connectionTime: Date.now() - startTime,
          });
        };
        
        setTimeout(() => {
          resolve({
            success: false,
            error: 'Connection timeout',
            connectionTime: Date.now() - startTime,
          });
        }, 15000);
      });
    });

    return result;
  }

  async testAlertNotifications(page, profile) {
    // Test alert notification system
    return await page.evaluate(() => {
      const alertElement = document.querySelector('[data-alert-system]');
      if (!alertElement) {
        return { success: false, error: 'Alert system not found' };
      }

      // Simulate alert trigger
      const event = new CustomEvent('geological-alert', {
        detail: { level: 'warning', message: 'Test alert' }
      });
      
      document.dispatchEvent(event);
      
      // Check if alert appears
      setTimeout(() => {
        const alertVisible = document.querySelector('.alert-notification');
        return {
          success: !!alertVisible,
          alertDisplayed: !!alertVisible,
        };
      }, 2000);
    });
  }

  async testWaveformDisplay(page, profile) {
    // Test seismic waveform rendering performance
    return await page.evaluate(() => {
      const canvas = document.querySelector('canvas[data-waveform]');
      if (!canvas) {
        return { success: false, error: 'Waveform canvas not found' };
      }

      const startTime = performance.now();
      
      // Simulate waveform data update
      if (window.updateWaveform) {
        const testData = Array.from({ length: 1000 }, (_, i) => Math.sin(i * 0.1) + Math.random() * 0.1);
        window.updateWaveform(testData);
      }
      
      const renderTime = performance.now() - startTime;
      
      return {
        success: true,
        renderTime,
        canvasSize: { width: canvas.width, height: canvas.height },
      };
    });
  }

  async testModelLoading(page, profile) {
    // Test 3D model loading performance under network constraints
    const startTime = Date.now();
    
    const result = await page.evaluate(() => {
      return new Promise((resolve) => {
        if (!window.loadGeologicalModel) {
          resolve({ success: false, error: '3D model loader not available' });
          return;
        }

        const modelUrl = '/models/test-geological-model.gltf';
        window.loadGeologicalModel(modelUrl)
          .then((model) => {
            resolve({
              success: true,
              modelSize: model.size || 'unknown',
              vertexCount: model.vertexCount || 'unknown',
            });
          })
          .catch((error) => {
            resolve({
              success: false,
              error: error.message,
            });
          });
      });
    });

    result.loadTime = Date.now() - startTime;
    return result;
  }

  async testPhotoUpload(page, profile) {
    // Test photo upload performance under network constraints
    return await page.evaluate(() => {
      const fileInput = document.querySelector('input[type="file"][data-photo-upload]');
      if (!fileInput) {
        return { success: false, error: 'Photo upload input not found' };
      }

      // Create a test blob (simulating a photo)
      const canvas = document.createElement('canvas');
      canvas.width = 1920;
      canvas.height = 1080;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ff0000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      return new Promise((resolve) => {
        canvas.toBlob((blob) => {
          const file = new File([blob], 'test-photo.jpg', { type: 'image/jpeg' });
          
          // Simulate file selection
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          fileInput.files = dataTransfer.files;
          
          const startTime = Date.now();
          fileInput.dispatchEvent(new Event('change'));
          
          // Monitor upload progress
          setTimeout(() => {
            resolve({
              success: true,
              fileSize: file.size,
              uploadTime: Date.now() - startTime,
            });
          }, 5000);
        }, 'image/jpeg', 0.8);
      });
    });
  }

  async monitorRealTimePerformance(page, client, profile) {
    console.log('    📡 Monitoring real-time performance...');
    
    const metrics = {
      dataUpdateLatency: [],
      frameDrops: 0,
      connectionStability: 100,
    };

    // Monitor for 30 seconds
    const monitorDuration = 30000;
    const startTime = Date.now();
    
    while (Date.now() - startTime < monitorDuration) {
      // Measure data update latency
      const latency = await page.evaluate(() => {
        if (window.getLastDataUpdateLatency) {
          return window.getLastDataUpdateLatency();
        }
        return null;
      });
      
      if (latency !== null) {
        metrics.dataUpdateLatency.push(latency);
      }

      // Check for frame drops
      const frameRate = await page.evaluate(() => {
        if (window.getCurrentFrameRate) {
          return window.getCurrentFrameRate();
        }
        return 60; // Assume 60fps if not available
      });
      
      if (frameRate < 30) {
        metrics.frameDrops++;
      }

      await page.waitForTimeout(1000); // Check every second
    }

    // Calculate averages
    if (metrics.dataUpdateLatency.length > 0) {
      metrics.averageLatency = metrics.dataUpdateLatency.reduce((a, b) => a + b, 0) / metrics.dataUpdateLatency.length;
      metrics.maxLatency = Math.max(...metrics.dataUpdateLatency);
    }

    return metrics;
  }

  async testDataIntensiveOperations(page, client, profile) {
    console.log('    📊 Testing data-intensive operations...');
    
    const operations = [
      'large-dataset-load',
      'real-time-streaming',
      'bulk-data-export',
    ];

    const results = {};

    for (const operation of operations) {
      const startTime = Date.now();
      
      try {
        switch (operation) {
          case 'large-dataset-load':
            results[operation] = await this.testLargeDatasetLoad(page, profile);
            break;
          case 'real-time-streaming':
            results[operation] = await this.testRealTimeStreaming(page, profile);
            break;
          case 'bulk-data-export':
            results[operation] = await this.testBulkDataExport(page, profile);
            break;
        }
        
        results[operation].duration = Date.now() - startTime;
        
      } catch (error) {
        results[operation] = {
          success: false,
          error: error.message,
          duration: Date.now() - startTime,
        };
      }
    }

    return results;
  }

  async testLargeDatasetLoad(page, profile) {
    return await page.evaluate(() => {
      if (!window.loadLargeGeologicalDataset) {
        return { success: false, error: 'Large dataset loader not available' };
      }

      const startTime = performance.now();
      
      return window.loadLargeGeologicalDataset('/api/seismic/large-dataset')
        .then((dataset) => {
          return {
            success: true,
            loadTime: performance.now() - startTime,
            recordCount: dataset.length || 'unknown',
            dataSize: JSON.stringify(dataset).length,
          };
        })
        .catch((error) => {
          return {
            success: false,
            error: error.message,
            loadTime: performance.now() - startTime,
          };
        });
    });
  }

  async testOfflineFunctionality(page, scenario) {
    // Test what functionality remains available offline
    return await page.evaluate(() => {
      const results = {
        cacheAvailable: 'caches' in window,
        serviceWorkerActive: 'serviceWorker' in navigator,
        localStorageWorking: false,
        offlineIndicator: false,
      };

      // Test local storage
      try {
        localStorage.setItem('offline-test', 'test-value');
        results.localStorageWorking = localStorage.getItem('offline-test') === 'test-value';
        localStorage.removeItem('offline-test');
      } catch (e) {
        results.localStorageWorking = false;
      }

      // Check for offline indicator
      results.offlineIndicator = !!document.querySelector('[data-offline-indicator]');

      return results;
    });
  }

  calculateDataTransferred(networkRequests) {
    // Estimate data transferred based on request types and sizes
    let totalBytes = 0;
    
    networkRequests.forEach(request => {
      // Estimate sizes based on content type
      if (request.mimeType) {
        if (request.mimeType.includes('image')) {
          totalBytes += 500000; // ~500KB average image
        } else if (request.mimeType.includes('video')) {
          totalBytes += 5000000; // ~5MB average video chunk
        } else if (request.mimeType.includes('json')) {
          totalBytes += 10000; // ~10KB average JSON
        } else if (request.mimeType.includes('javascript')) {
          totalBytes += 100000; // ~100KB average JS file
        } else {
          totalBytes += 5000; // ~5KB default
        }
      }
    });

    return totalBytes;
  }

  calculateAverageLatency(networkRequests) {
    if (networkRequests.length === 0) return 0;
    
    // This is a simplified calculation
    // In a real implementation, you'd track request/response timing
    return networkRequests.length * 50; // Estimate based on request count
  }

  calculatePerformanceScore(metrics) {
    let score = 100;

    // Navigation time penalty
    if (metrics.navigationTime > 10000) score -= 30;
    else if (metrics.navigationTime > 5000) score -= 15;
    else if (metrics.navigationTime > 3000) score -= 5;

    // Network errors penalty
    score -= metrics.networkErrors * 10;

    // Feature failures penalty
    const failedFeatures = Object.values(metrics.featureResults).filter(f => !f.success).length;
    score -= failedFeatures * 15;

    // Real-time performance penalty
    if (metrics.realTimeMetrics.averageLatency > 1000) score -= 20;
    else if (metrics.realTimeMetrics.averageLatency > 500) score -= 10;

    // Frame drops penalty
    if (metrics.realTimeMetrics.frameDrops > 10) score -= 15;

    return Math.max(0, Math.min(100, score));
  }

  generateNetworkRecommendations(params) {
    const recommendations = [];
    const { scenario, profile, performanceScore, networkErrors, navigationTime } = params;

    if (performanceScore < 50) {
      recommendations.push({
        type: 'critical',
        message: 'Poor performance under network constraints. Implement aggressive caching and data compression.',
        priority: 'high',
      });
    }

    if (networkErrors > 0) {
      recommendations.push({
        type: 'reliability',
        message: 'Network errors detected. Implement retry logic and offline fallbacks.',
        priority: 'high',
      });
    }

    if (navigationTime > 10000) {
      recommendations.push({
        type: 'optimization',
        message: 'Slow initial load. Consider lazy loading and progressive enhancement.',
        priority: 'medium',
      });
    }

    if (profile.latency > 300 && scenario.realTime) {
      recommendations.push({
        type: 'architecture',
        message: 'High latency affects real-time features. Consider edge computing or local processing.',
        priority: 'medium',
      });
    }

    if (scenario.dataIntensive && profile.downloadThroughput < 1000000) {
      recommendations.push({
        type: 'optimization',
        message: 'Data-intensive application on slow connection. Implement data pagination and compression.',
        priority: 'medium',
      });
    }

    return recommendations;
  }

  generateIntermittentConnectivityRecommendations(results) {
    const recommendations = [];

    if (results.connectionDrops > 3) {
      recommendations.push({
        type: 'resilience',
        message: 'Frequent connection drops. Implement robust offline functionality and data synchronization.',
        priority: 'high',
      });
    }

    if (!results.offlineFunctionality.serviceWorkerActive) {
      recommendations.push({
        type: 'offline',
        message: 'No service worker detected. Implement service worker for offline capabilities.',
        priority: 'high',
      });
    }

    const avgReconnectionTime = results.reconnectionTimes.reduce((a, b) => a + b, 0) / results.reconnectionTimes.length;
    if (avgReconnectionTime > 5000) {
      recommendations.push({
        type: 'performance',
        message: 'Slow reconnection times. Optimize connection recovery and data synchronization.',
        priority: 'medium',
      });
    }

    return recommendations;
  }

  generateNetworkPerformanceReport(results, timestamp) {
    const summary = {
      timestamp,
      totalTests: results.length,
      averagePerformanceScore: 0,
      networkProfilePerformance: {},
      scenarioPerformance: {},
      criticalIssues: [],
      recommendations: [],
    };

    // Calculate averages
    const validResults = results.filter(r => !r.error && r.performanceScore !== undefined);
    if (validResults.length > 0) {
      summary.averagePerformanceScore = validResults.reduce((sum, r) => sum + r.performanceScore, 0) / validResults.length;
    }

    // Group by network profile
    const profileGroups = {};
    validResults.forEach(result => {
      if (!profileGroups[result.networkProfile]) {
        profileGroups[result.networkProfile] = [];
      }
      profileGroups[result.networkProfile].push(result.performanceScore);
    });

    Object.entries(profileGroups).forEach(([profile, scores]) => {
      summary.networkProfilePerformance[profile] = {
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        testCount: scores.length,
      };
    });

    // Group by scenario
    const scenarioGroups = {};
    validResults.forEach(result => {
      if (!scenarioGroups[result.scenario]) {
        scenarioGroups[result.scenario] = [];
      }
      scenarioGroups[result.scenario].push(result.performanceScore);
    });

    Object.entries(scenarioGroups).forEach(([scenario, scores]) => {
      summary.scenarioPerformance[scenario] = {
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        testCount: scores.length,
      };
    });

    // Identify critical issues
    validResults.forEach(result => {
      if (result.performanceScore < 30) {
        summary.criticalIssues.push({
          scenario: result.scenario,
          networkProfile: result.networkProfile,
          performanceScore: result.performanceScore,
          issue: 'Very poor performance under network constraints',
        });
      }
    });

    // Aggregate recommendations
    const allRecommendations = validResults
      .filter(r => r.recommendations)
      .flatMap(r => r.recommendations);
    
    const recommendationCounts = {};
    allRecommendations.forEach(rec => {
      const key = rec.message;
      recommendationCounts[key] = (recommendationCounts[key] || 0) + 1;
    });

    summary.recommendations = Object.entries(recommendationCounts)
      .map(([message, count]) => ({ message, occurrences: count }))
      .sort((a, b) => b.occurrences - a.occurrences);

    return { summary, detailedResults: results };
  }

  async generateNetworkVisualization(results, timestamp) {
    // Generate HTML report with network performance charts
    const htmlReport = this.generateNetworkHTMLReport(results);
    const reportPath = path.join(
      this.config.outputDir,
      `network-performance-report-${timestamp}.html`
    );
    await fs.writeFile(reportPath, htmlReport);
    
    console.log(`📊 Network performance visualization saved to ${reportPath}`);
  }

  generateNetworkHTMLReport(results) {
    const validResults = results.filter(r => !r.error && r.performanceScore !== undefined);
    
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Network Throttling Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .test-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        .poor-performance { border-color: #ff4444; background-color: #fff5f5; }
        .good-performance { border-color: #44ff44; background-color: #f5fff5; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        .summary { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .recommendations { background-color: #fff8dc; padding: 15px; margin: 10px 0; }
        .network-profile { background-color: #e6f3ff; padding: 10px; margin: 5px 0; }
    </style>
</head>
<body>
    <h1>🌐 Network Throttling Performance Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> ${results.length}</p>
        <p><strong>Average Performance Score:</strong> ${validResults.length > 0 ? (validResults.reduce((sum, r) => sum + r.performanceScore, 0) / validResults.length).toFixed(1) : 'N/A'}</p>
        <p><strong>Tests with Errors:</strong> ${results.filter(r => r.error).length}</p>
    </div>

    <div class="chart-container">
        <canvas id="performance-by-network"></canvas>
    </div>

    <div class="chart-container">
        <canvas id="performance-by-scenario"></canvas>
    </div>

    ${validResults.map((result, index) => `
    <div class="test-section ${result.performanceScore < 50 ? 'poor-performance' : 'good-performance'}">
        <h3>${result.scenario} - ${result.networkProfile} (Score: ${result.performanceScore})</h3>
        
        <div class="network-profile">
            <strong>Network Scenario:</strong> ${result.networkScenario || 'Standard test'}
        </div>
        
        <p><strong>Navigation Time:</strong> ${result.navigationTime}ms</p>
        <p><strong>Network Requests:</strong> ${result.networkRequests}</p>
        <p><strong>Network Errors:</strong> ${result.networkErrors}</p>
        <p><strong>Data Transferred:</strong> ${(result.totalDataTransferred / 1024 / 1024).toFixed(2)} MB</p>
        
        ${result.recommendations && result.recommendations.length > 0 ? `
        <div class="recommendations">
            <h4>Recommendations:</h4>
            <ul>
                ${result.recommendations.map(rec => `<li><strong>${rec.type}:</strong> ${rec.message} (${rec.priority})</li>`).join('')}
            </ul>
        </div>
        ` : ''}
    </div>
    `).join('')}

    <script>
        // Performance by Network Profile Chart
        const networkData = {};
        ${JSON.stringify(validResults)}.forEach(result => {
            if (!networkData[result.networkProfile]) {
                networkData[result.networkProfile] = [];
            }
            networkData[result.networkProfile].push(result.performanceScore);
        });

        new Chart(document.getElementById('performance-by-network'), {
            type: 'bar',
            data: {
                labels: Object.keys(networkData),
                datasets: [{
                    label: 'Average Performance Score',
                    data: Object.values(networkData).map(scores => 
                        scores.reduce((a, b) => a + b, 0) / scores.length
                    ),
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                },
                plugins: {
                    title: { display: true, text: 'Performance Score by Network Profile' }
                }
            }
        });

        // Performance by Scenario Chart
        const scenarioData = {};
        ${JSON.stringify(validResults)}.forEach(result => {
            if (!scenarioData[result.scenario]) {
                scenarioData[result.scenario] = [];
            }
            scenarioData[result.scenario].push(result.performanceScore);
        });

        new Chart(document.getElementById('performance-by-scenario'), {
            type: 'bar',
            data: {
                labels: Object.keys(scenarioData),
                datasets: [{
                    label: 'Average Performance Score',
                    data: Object.values(scenarioData).map(scores => 
                        scores.reduce((a, b) => a + b, 0) / scores.length
                    ),
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100 },
                    x: { ticks: { maxRotation: 45 } }
                },
                plugins: {
                    title: { display: true, text: 'Performance Score by Test Scenario' }
                }
            }
        });
    </script>
</body>
</html>`;
  }

  async ensureOutputDirectory() {
    try {
      await fs.access(this.config.outputDir);
    } catch {
      await fs.mkdir(this.config.outputDir, { recursive: true });
    }
  }
}

// CLI interface
if (require.main === module) {
  const config = {
    baseUrl: process.env.BASE_URL || 'http://localhost:3000',
    outputDir: process.env.OUTPUT_DIR || './tests/performance/network-reports',
    testDuration: parseInt(process.env.TEST_DURATION) || 120000,
  };

  const tester = new NetworkThrottlingTester(config);
  
  tester.runNetworkThrottlingTests()
    .then(report => {
      console.log('\n🎯 Network Throttling Test Summary:');
      console.log(`Total tests: ${report.summary.totalTests}`);
      console.log(`Average performance score: ${report.summary.averagePerformanceScore.toFixed(1)}`);
      
      if (report.summary.criticalIssues.length > 0) {
        console.log('\n⚠️  Critical Issues:');
        report.summary.criticalIssues.forEach(issue => {
          console.log(`  ${issue.scenario} on ${issue.networkProfile}: Score ${issue.performanceScore}`);
        });
      }
      
      process.exit(report.summary.averagePerformanceScore < 50 ? 1 : 0);
    })
    .catch(error => {
      console.error('Network throttling tests failed:', error);
      process.exit(1);
    });
}

module.exports = NetworkThrottlingTester;