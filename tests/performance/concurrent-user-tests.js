/**
 * Concurrent User Testing for Geological Monitoring Platform
 * Simulates multiple users accessing geological data and visualizations simultaneously
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');
const cluster = require('cluster');
const os = require('os');

class ConcurrentUserTester {
  constructor(config = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:3000',
      maxConcurrentUsers: config.maxConcurrentUsers || 50,
      testDuration: config.testDuration || 300000, // 5 minutes
      rampUpTime: config.rampUpTime || 60000, // 1 minute ramp-up
      outputDir: config.outputDir || './tests/performance/concurrent-reports',
      ...config,
    };

    // User behavior patterns for geological monitoring
    this.userProfiles = [
      {
        name: 'seismic-analyst',
        weight: 0.3, // 30% of users
        description: 'Seismic data analyst monitoring earthquake activity',
        behaviors: [
          { action: 'view-seismic-dashboard', probability: 0.8, duration: 30000 },
          { action: 'analyze-waveforms', probability: 0.6, duration: 45000 },
          { action: 'generate-reports', probability: 0.4, duration: 60000 },
          { action: 'configure-alerts', probability: 0.3, duration: 20000 },
        ],
        sessionDuration: 1800000, // 30 minutes
        dataUsage: 'high',
      },
      {
        name: 'volcano-monitor',
        weight: 0.25, // 25% of users
        description: 'Volcanologist monitoring volcanic activity',
        behaviors: [
          { action: 'view-volcano-dashboard', probability: 0.9, duration: 25000 },
          { action: 'check-thermal-cameras', probability: 0.7, duration: 15000 },
          { action: 'analyze-deformation', probability: 0.5, duration: 40000 },
          { action: 'update-alert-levels', probability: 0.2, duration: 30000 },
        ],
        sessionDuration: 2400000, // 40 minutes
        dataUsage: 'very-high',
      },
      {
        name: 'field-researcher',
        weight: 0.2, // 20% of users
        description: 'Field researcher collecting and uploading data',
        behaviors: [
          { action: 'upload-field-data', probability: 0.8, duration: 120000 },
          { action: 'view-regional-maps', probability: 0.6, duration: 20000 },
          { action: 'sync-gps-data', probability: 0.7, duration: 30000 },
          { action: 'submit-observations', probability: 0.5, duration: 45000 },
        ],
        sessionDuration: 900000, // 15 minutes
        dataUsage: 'medium',
      },
      {
        name: 'emergency-responder',
        weight: 0.15, // 15% of users
        description: 'Emergency response coordinator',
        behaviors: [
          { action: 'view-alert-dashboard', probability: 0.95, duration: 10000 },
          { action: 'check-evacuation-zones', probability: 0.8, duration: 15000 },
          { action: 'coordinate-resources', probability: 0.6, duration: 25000 },
          { action: 'communicate-updates', probability: 0.7, duration: 20000 },
        ],
        sessionDuration: 3600000, // 60 minutes (long monitoring sessions)
        dataUsage: 'medium',
      },
      {
        name: 'public-viewer',
        weight: 0.1, // 10% of users
        description: 'General public viewing geological information',
        behaviors: [
          { action: 'view-public-dashboard', probability: 0.9, duration: 15000 },
          { action: 'check-recent-earthquakes', probability: 0.7, duration: 10000 },
          { action: 'view-educational-content', probability: 0.4, duration: 30000 },
        ],
        sessionDuration: 600000, // 10 minutes
        dataUsage: 'low',
      },
    ];

    // System stress scenarios
    this.stressScenarios = [
      {
        name: 'normal-operations',
        description: 'Normal daily operations',
        userMultiplier: 1.0,
        duration: 300000, // 5 minutes
      },
      {
        name: 'volcanic-emergency',
        description: 'Volcanic emergency with increased monitoring',
        userMultiplier: 3.0,
        duration: 600000, // 10 minutes
        profileAdjustments: {
          'volcano-monitor': 0.5, // 50% of users are volcano monitors
          'emergency-responder': 0.3,
          'seismic-analyst': 0.15,
          'public-viewer': 0.05,
        },
      },
      {
        name: 'major-earthquake',
        description: 'Major earthquake event response',
        userMultiplier: 4.0,
        duration: 900000, // 15 minutes
        profileAdjustments: {
          'seismic-analyst': 0.4,
          'emergency-responder': 0.35,
          'volcano-monitor': 0.1,
          'field-researcher': 0.1,
          'public-viewer': 0.05,
        },
      },
      {
        name: 'system-peak-load',
        description: 'Peak system load testing',
        userMultiplier: 5.0,
        duration: 180000, // 3 minutes
      },
    ];
  }

  async runConcurrentUserTests() {
    console.log('👥 Starting Concurrent User Testing for Geological Platform...');
    
    await this.ensureOutputDirectory();
    const results = [];
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    for (const scenario of this.stressScenarios) {
      console.log(`\n🎯 Testing scenario: ${scenario.name} - ${scenario.description}`);
      
      try {
        const testResult = await this.runStressScenario(scenario);
        results.push(testResult);
        
        console.log(`✓ Scenario completed`);
        console.log(`  Peak Users: ${testResult.peakConcurrentUsers}`);
        console.log(`  Success Rate: ${(testResult.successRate * 100).toFixed(1)}%`);
        console.log(`  Avg Response Time: ${testResult.averageResponseTime}ms`);
        
        // Cool-down period between scenarios
        console.log('  😴 Cooling down for 30 seconds...');
        await new Promise(resolve => setTimeout(resolve, 30000));
        
      } catch (error) {
        console.error(`❌ Scenario ${scenario.name} failed:`, error.message);
        results.push({
          scenario: scenario.name,
          error: error.message,
          success: false,
        });
      }
    }

    // Generate comprehensive report
    const report = this.generateConcurrentUserReport(results, timestamp);
    const reportPath = path.join(
      this.config.outputDir,
      `concurrent-user-report-${timestamp}.json`
    );
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

    // Generate visualization
    await this.generateConcurrentUserVisualization(results, timestamp);

    return report;
  }

  async runStressScenario(scenario) {
    const maxUsers = Math.floor(this.config.maxConcurrentUsers * scenario.userMultiplier);
    const rampUpInterval = this.config.rampUpTime / maxUsers;
    
    console.log(`  📈 Ramping up to ${maxUsers} concurrent users over ${this.config.rampUpTime / 1000}s`);
    
    const userSessions = [];
    const metrics = {
      startTime: Date.now(),
      endTime: null,
      peakConcurrentUsers: 0,
      totalSessions: 0,
      successfulSessions: 0,
      failedSessions: 0,
      responseTimes: [],
      errorTypes: {},
      resourceUsage: [],
      throughput: [],
    };

    // Start resource monitoring
    const resourceMonitor = this.startResourceMonitoring(metrics);

    try {
      // Ramp up users
      for (let i = 0; i < maxUsers; i++) {
        const userProfile = this.selectUserProfile(scenario.profileAdjustments);
        const userSession = this.startUserSession(userProfile, scenario, metrics);
        userSessions.push(userSession);
        
        // Update peak concurrent users
        const activeSessions = userSessions.filter(session => !session.completed);
        metrics.peakConcurrentUsers = Math.max(metrics.peakConcurrentUsers, activeSessions.length);
        
        // Wait for ramp-up interval
        await new Promise(resolve => setTimeout(resolve, rampUpInterval));
      }

      console.log(`  🏃 Running ${maxUsers} concurrent users for ${scenario.duration / 1000}s`);
      
      // Wait for scenario duration
      await new Promise(resolve => setTimeout(resolve, scenario.duration));
      
      console.log('  🛑 Stopping user sessions...');
      
      // Stop all user sessions
      await Promise.all(userSessions.map(session => this.stopUserSession(session)));
      
    } finally {
      // Stop resource monitoring
      clearInterval(resourceMonitor);
      metrics.endTime = Date.now();
    }

    // Calculate final metrics
    const finalMetrics = this.calculateScenarioMetrics(metrics, scenario);
    
    return {
      scenario: scenario.name,
      description: scenario.description,
      ...finalMetrics,
      timestamp: new Date().toISOString(),
    };
  }

  selectUserProfile(profileAdjustments = null) {
    let profiles = this.userProfiles;
    
    // Apply profile adjustments for specific scenarios
    if (profileAdjustments) {
      profiles = this.userProfiles.map(profile => ({
        ...profile,
        weight: profileAdjustments[profile.name] || profile.weight,
      }));
    }

    // Weighted random selection
    const totalWeight = profiles.reduce((sum, profile) => sum + profile.weight, 0);
    let random = Math.random() * totalWeight;
    
    for (const profile of profiles) {
      random -= profile.weight;
      if (random <= 0) {
        return profile;
      }
    }
    
    return profiles[0]; // Fallback
  }

  async startUserSession(userProfile, scenario, metrics) {
    const session = {
      id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      profile: userProfile.name,
      startTime: Date.now(),
      endTime: null,
      completed: false,
      success: false,
      actions: [],
      errors: [],
      browser: null,
      page: null,
    };

    metrics.totalSessions++;

    try {
      // Launch browser for this user session
      session.browser = await puppeteer.launch({
        headless: true,
        args: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
          '--disable-gpu',
        ],
      });

      session.page = await session.browser.newPage();
      
      // Set up performance monitoring for this session
      await session.page.setRequestInterception(true);
      
      session.page.on('request', (request) => {
        const startTime = Date.now();
        request.continue();
        
        // Track request timing
        request.response().then(() => {
          const responseTime = Date.now() - startTime;
          metrics.responseTimes.push(responseTime);
        }).catch(() => {
          // Request failed
          const errorType = 'request_failed';
          metrics.errorTypes[errorType] = (metrics.errorTypes[errorType] || 0) + 1;
        });
      });

      session.page.on('pageerror', (error) => {
        session.errors.push({
          type: 'page_error',
          message: error.message,
          timestamp: Date.now(),
        });
      });

      // Start user behavior simulation
      this.simulateUserBehavior(session, userProfile, scenario, metrics);
      
      session.success = true;
      
    } catch (error) {
      session.errors.push({
        type: 'session_start_error',
        message: error.message,
        timestamp: Date.now(),
      });
      
      metrics.failedSessions++;
      
      if (session.browser) {
        await session.browser.close();
      }
    }

    return session;
  }

  async simulateUserBehavior(session, userProfile, scenario, metrics) {
    const sessionEndTime = session.startTime + userProfile.sessionDuration;
    
    while (Date.now() < sessionEndTime && !session.completed) {
      try {
        // Select next behavior based on probabilities
        const behavior = this.selectBehavior(userProfile.behaviors);
        
        if (behavior) {
          const actionStart = Date.now();
          await this.performUserAction(session, behavior);
          const actionDuration = Date.now() - actionStart;
          
          session.actions.push({
            action: behavior.action,
            duration: actionDuration,
            timestamp: actionStart,
            success: true,
          });
          
          // Wait for behavior duration
          await new Promise(resolve => setTimeout(resolve, behavior.duration));
        }
        
        // Random think time between actions (1-5 seconds)
        const thinkTime = Math.random() * 4000 + 1000;
        await new Promise(resolve => setTimeout(resolve, thinkTime));
        
      } catch (error) {
        session.errors.push({
          type: 'behavior_error',
          message: error.message,
          timestamp: Date.now(),
        });
        
        // Continue with next behavior after error
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  }

  selectBehavior(behaviors) {
    // Select behavior based on probability
    const availableBehaviors = behaviors.filter(b => Math.random() < b.probability);
    
    if (availableBehaviors.length === 0) {
      return null;
    }
    
    return availableBehaviors[Math.floor(Math.random() * availableBehaviors.length)];
  }

  async performUserAction(session, behavior) {
    const { page } = session;
    
    switch (behavior.action) {
      case 'view-seismic-dashboard':
        await page.goto(`${this.config.baseUrl}/seismic/dashboard`, { waitUntil: 'networkidle2' });
        await this.simulateSeismicDashboardInteraction(page);
        break;
        
      case 'analyze-waveforms':
        await page.goto(`${this.config.baseUrl}/seismic/waveforms`, { waitUntil: 'networkidle2' });
        await this.simulateWaveformAnalysis(page);
        break;
        
      case 'view-volcano-dashboard':
        await page.goto(`${this.config.baseUrl}/volcano/dashboard`, { waitUntil: 'networkidle2' });
        await this.simulateVolcanoDashboardInteraction(page);
        break;
        
      case 'check-thermal-cameras':
        await page.goto(`${this.config.baseUrl}/volcano/thermal`, { waitUntil: 'networkidle2' });
        await this.simulateThermalCameraViewing(page);
        break;
        
      case 'upload-field-data':
        await page.goto(`${this.config.baseUrl}/field/upload`, { waitUntil: 'networkidle2' });
        await this.simulateFieldDataUpload(page);
        break;
        
      case 'view-alert-dashboard':
        await page.goto(`${this.config.baseUrl}/alerts/dashboard`, { waitUntil: 'networkidle2' });
        await this.simulateAlertDashboardInteraction(page);
        break;
        
      case 'view-public-dashboard':
        await page.goto(`${this.config.baseUrl}/public/dashboard`, { waitUntil: 'networkidle2' });
        await this.simulatePublicDashboardViewing(page);
        break;
        
      default:
        // Generic page interaction
        await page.goto(`${this.config.baseUrl}/`, { waitUntil: 'networkidle2' });
        await this.simulateGenericInteraction(page);
    }
  }

  async simulateSeismicDashboardInteraction(page) {
    // Simulate typical seismic analyst interactions
    await page.waitForTimeout(2000);
    
    // Click on different time ranges
    const timeRangeButtons = await page.$$('[data-time-range]');
    if (timeRangeButtons.length > 0) {
      const randomButton = timeRangeButtons[Math.floor(Math.random() * timeRangeButtons.length)];
      await randomButton.click();
      await page.waitForTimeout(3000);
    }
    
    // Interact with magnitude filters
    const magnitudeFilter = await page.$('[data-magnitude-filter]');
    if (magnitudeFilter) {
      await magnitudeFilter.click();
      await page.waitForTimeout(1000);
    }
    
    // Scroll through event list
    await page.evaluate(() => {
      const eventList = document.querySelector('[data-event-list]');
      if (eventList) {
        eventList.scrollTop = Math.random() * eventList.scrollHeight;
      }
    });
  }

  async simulateWaveformAnalysis(page) {
    // Simulate waveform analysis interactions
    await page.waitForTimeout(3000);
    
    // Select different stations
    const stationSelects = await page.$$('select[data-station]');
    if (stationSelects.length > 0) {
      for (const select of stationSelects.slice(0, 3)) {
        await select.click();
        await page.waitForTimeout(500);
      }
    }
    
    // Zoom and pan on waveforms
    const canvas = await page.$('canvas[data-waveform]');
    if (canvas) {
      const box = await canvas.boundingBox();
      if (box) {
        // Simulate mouse interactions on waveform
        await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.5);
        await page.mouse.down();
        await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.5);
        await page.mouse.up();
        await page.waitForTimeout(2000);
      }
    }
  }

  async simulateVolcanoDashboardInteraction(page) {
    // Simulate volcano monitoring interactions
    await page.waitForTimeout(2000);
    
    // Switch between different volcanoes
    const volcanoSelect = await page.$('select[data-volcano-select]');
    if (volcanoSelect) {
      await volcanoSelect.click();
      await page.waitForTimeout(1000);
      
      const options = await page.$$eval('select[data-volcano-select] option', options => 
        options.map(option => option.value)
      );
      
      if (options.length > 1) {
        const randomOption = options[Math.floor(Math.random() * options.length)];
        await page.select('select[data-volcano-select]', randomOption);
        await page.waitForTimeout(3000);
      }
    }
    
    // Check different monitoring parameters
    const parameterTabs = await page.$$('[data-parameter-tab]');
    for (const tab of parameterTabs.slice(0, 3)) {
      await tab.click();
      await page.waitForTimeout(2000);
    }
  }

  async simulateThermalCameraViewing(page) {
    // Simulate thermal camera interactions
    await page.waitForTimeout(2000);
    
    // Switch between camera views
    const cameraButtons = await page.$$('[data-camera-view]');
    for (const button of cameraButtons.slice(0, 2)) {
      await button.click();
      await page.waitForTimeout(4000); // Wait for video to load
    }
    
    // Adjust thermal settings
    const thermalControls = await page.$$('[data-thermal-control]');
    for (const control of thermalControls.slice(0, 2)) {
      await control.click();
      await page.waitForTimeout(1000);
    }
  }

  async simulateFieldDataUpload(page) {
    // Simulate field data upload
    await page.waitForTimeout(2000);
    
    // Fill out data form
    const locationInput = await page.$('input[data-location]');
    if (locationInput) {
      await locationInput.type(`${40.7128 + (Math.random() - 0.5) * 0.1}, ${-74.0060 + (Math.random() - 0.5) * 0.1}`);
    }
    
    const observationText = await page.$('textarea[data-observation]');
    if (observationText) {
      await observationText.type('Test geological observation from concurrent user test');
    }
    
    // Simulate file upload (without actual file)
    const fileInput = await page.$('input[type="file"]');
    if (fileInput) {
      // Just click to simulate file selection dialog
      await fileInput.click();
      await page.waitForTimeout(2000);
    }
  }

  async simulateAlertDashboardInteraction(page) {
    // Simulate emergency response interactions
    await page.waitForTimeout(1000);
    
    // Check alert status
    const alertItems = await page.$$('[data-alert-item]');
    for (const item of alertItems.slice(0, 3)) {
      await item.click();
      await page.waitForTimeout(1500);
    }
    
    // Refresh data frequently (emergency scenario)
    const refreshButton = await page.$('[data-refresh]');
    if (refreshButton) {
      await refreshButton.click();
      await page.waitForTimeout(3000);
    }
  }

  async simulatePublicDashboardViewing(page) {
    // Simulate public user interactions
    await page.waitForTimeout(2000);
    
    // Browse recent events
    const eventLinks = await page.$$('[data-recent-event]');
    if (eventLinks.length > 0) {
      const randomLink = eventLinks[Math.floor(Math.random() * eventLinks.length)];
      await randomLink.click();
      await page.waitForTimeout(3000);
    }
    
    // View educational content
    const infoButtons = await page.$$('[data-info-button]');
    for (const button of infoButtons.slice(0, 2)) {
      await button.click();
      await page.waitForTimeout(2000);
    }
  }

  async simulateGenericInteraction(page) {
    // Generic page interactions
    await page.waitForTimeout(1000);
    
    // Scroll page
    await page.evaluate(() => {
      window.scrollTo(0, Math.random() * document.body.scrollHeight);
    });
    
    await page.waitForTimeout(2000);
    
    // Click random interactive elements
    const clickableElements = await page.$$('button, a, [data-clickable]');
    if (clickableElements.length > 0) {
      const randomElement = clickableElements[Math.floor(Math.random() * clickableElements.length)];
      try {
        await randomElement.click();
        await page.waitForTimeout(1000);
      } catch (error) {
        // Element might not be clickable, continue
      }
    }
  }

  async stopUserSession(session) {
    session.completed = true;
    session.endTime = Date.now();
    
    if (session.browser) {
      try {
        await session.browser.close();
      } catch (error) {
        // Browser might already be closed
      }
    }
    
    return session;
  }

  startResourceMonitoring(metrics) {
    return setInterval(() => {
      const usage = process.memoryUsage();
      const cpuUsage = process.cpuUsage();
      
      metrics.resourceUsage.push({
        timestamp: Date.now(),
        memoryUsage: usage.heapUsed,
        memoryTotal: usage.heapTotal,
        cpuUser: cpuUsage.user,
        cpuSystem: cpuUsage.system,
      });
      
      // Calculate throughput (requests per second)
      const recentResponses = metrics.responseTimes.filter(
        time => Date.now() - time < 60000 // Last minute
      );
      metrics.throughput.push({
        timestamp: Date.now(),
        requestsPerSecond: recentResponses.length / 60,
      });
      
    }, 5000); // Every 5 seconds
  }

  calculateScenarioMetrics(metrics, scenario) {
    const totalDuration = metrics.endTime - metrics.startTime;
    const successRate = metrics.totalSessions > 0 ? metrics.successfulSessions / metrics.totalSessions : 0;
    
    const averageResponseTime = metrics.responseTimes.length > 0 
      ? metrics.responseTimes.reduce((sum, time) => sum + time, 0) / metrics.responseTimes.length 
      : 0;
    
    const p95ResponseTime = metrics.responseTimes.length > 0
      ? metrics.responseTimes.sort((a, b) => a - b)[Math.floor(metrics.responseTimes.length * 0.95)]
      : 0;
    
    const errorRate = metrics.totalSessions > 0 ? metrics.failedSessions / metrics.totalSessions : 0;
    
    const peakMemoryUsage = metrics.resourceUsage.length > 0
      ? Math.max(...metrics.resourceUsage.map(r => r.memoryUsage))
      : 0;
    
    const averageThroughput = metrics.throughput.length > 0
      ? metrics.throughput.reduce((sum, t) => sum + t.requestsPerSecond, 0) / metrics.throughput.length
      : 0;

    return {
      totalDuration,
      peakConcurrentUsers: metrics.peakConcurrentUsers,
      totalSessions: metrics.totalSessions,
      successfulSessions: metrics.successfulSessions,
      failedSessions: metrics.failedSessions,
      successRate,
      errorRate,
      averageResponseTime,
      p95ResponseTime,
      peakMemoryUsage,
      averageThroughput,
      errorTypes: metrics.errorTypes,
      resourceUsage: metrics.resourceUsage,
      throughput: metrics.throughput,
    };
  }

  generateConcurrentUserReport(results, timestamp) {
    const summary = {
      timestamp,
      totalScenarios: results.length,
      successfulScenarios: results.filter(r => r.success !== false).length,
      overallPerformance: 'good',
      criticalIssues: [],
      recommendations: [],
    };

    // Analyze overall performance
    const validResults = results.filter(r => r.success !== false);
    
    if (validResults.length > 0) {
      const avgSuccessRate = validResults.reduce((sum, r) => sum + r.successRate, 0) / validResults.length;
      const avgResponseTime = validResults.reduce((sum, r) => sum + r.averageResponseTime, 0) / validResults.length;
      
      if (avgSuccessRate < 0.9 || avgResponseTime > 5000) {
        summary.overallPerformance = 'critical';
      } else if (avgSuccessRate < 0.95 || avgResponseTime > 2000) {
        summary.overallPerformance = 'warning';
      }
      
      // Identify critical issues
      validResults.forEach(result => {
        if (result.successRate < 0.8) {
          summary.criticalIssues.push({
            scenario: result.scenario,
            issue: 'Low success rate',
            value: `${(result.successRate * 100).toFixed(1)}%`,
          });
        }
        
        if (result.averageResponseTime > 10000) {
          summary.criticalIssues.push({
            scenario: result.scenario,
            issue: 'High response time',
            value: `${result.averageResponseTime}ms`,
          });
        }
      });
    }

    // Generate recommendations
    summary.recommendations = this.generateConcurrentUserRecommendations(validResults);

    return { summary, detailedResults: results };
  }

  generateConcurrentUserRecommendations(results) {
    const recommendations = [];

    results.forEach(result => {
      if (result.errorRate > 0.1) {
        recommendations.push({
          scenario: result.scenario,
          type: 'reliability',
          message: 'High error rate detected. Review error handling and system capacity.',
          priority: 'high',
        });
      }

      if (result.averageResponseTime > 5000) {
        recommendations.push({
          scenario: result.scenario,
          type: 'performance',
          message: 'Slow response times under load. Consider caching and optimization.',
          priority: 'high',
        });
      }

      if (result.peakMemoryUsage > 1024 * 1024 * 1024) { // 1GB
        recommendations.push({
          scenario: result.scenario,
          type: 'resource',
          message: 'High memory usage detected. Review memory management and garbage collection.',
          priority: 'medium',
        });
      }
    });

    return recommendations;
  }

  async generateConcurrentUserVisualization(results, timestamp) {
    // Generate HTML report with concurrent user performance charts
    const htmlReport = this.generateConcurrentUserHTMLReport(results);
    const reportPath = path.join(
      this.config.outputDir,
      `concurrent-user-report-${timestamp}.html`
    );
    await fs.writeFile(reportPath, htmlReport);
    
    console.log(`📊 Concurrent user visualization saved to ${reportPath}`);
  }

  generateConcurrentUserHTMLReport(results) {
    const validResults = results.filter(r => r.success !== false);
    
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Concurrent User Testing Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .scenario-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        .critical { border-color: #ff4444; background-color: #fff5f5; }
        .warning { border-color: #ffaa00; background-color: #fffaf0; }
        .good { border-color: #44ff44; background-color: #f5fff5; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        .summary { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
        .metric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>👥 Concurrent User Testing Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Scenarios:</strong> ${results.length}</p>
        <p><strong>Successful Scenarios:</strong> ${validResults.length}</p>
        <p><strong>Peak Concurrent Users:</strong> ${validResults.length > 0 ? Math.max(...validResults.map(r => r.peakConcurrentUsers)) : 'N/A'}</p>
    </div>

    <div class="chart-container">
        <canvas id="success-rate-chart"></canvas>
    </div>

    <div class="chart-container">
        <canvas id="response-time-chart"></canvas>
    </div>

    ${validResults.map((result, index) => {
      const performanceClass = result.successRate < 0.8 ? 'critical' : 
                              result.successRate < 0.95 ? 'warning' : 'good';
      
      return `
    <div class="scenario-section ${performanceClass}">
        <h3>${result.scenario} - ${result.description}</h3>
        
        <div class="metrics">
            <div class="metric">
                <strong>Peak Users:</strong><br>
                ${result.peakConcurrentUsers}
            </div>
            <div class="metric">
                <strong>Success Rate:</strong><br>
                ${(result.successRate * 100).toFixed(1)}%
            </div>
            <div class="metric">
                <strong>Avg Response Time:</strong><br>
                ${result.averageResponseTime}ms
            </div>
            <div class="metric">
                <strong>P95 Response Time:</strong><br>
                ${result.p95ResponseTime}ms
            </div>
            <div class="metric">
                <strong>Error Rate:</strong><br>
                ${(result.errorRate * 100).toFixed(1)}%
            </div>
            <div class="metric">
                <strong>Peak Memory:</strong><br>
                ${(result.peakMemoryUsage / 1024 / 1024).toFixed(1)} MB
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="throughput-chart-${index}"></canvas>
        </div>
    </div>
      `;
    }).join('')}

    <script>
        // Success Rate Chart
        new Chart(document.getElementById('success-rate-chart'), {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(validResults.map(r => r.scenario))},
                datasets: [{
                    label: 'Success Rate (%)',
                    data: ${JSON.stringify(validResults.map(r => r.successRate * 100))},
                    backgroundColor: ${JSON.stringify(validResults.map(r => 
                      r.successRate < 0.8 ? 'rgba(255, 68, 68, 0.6)' :
                      r.successRate < 0.95 ? 'rgba(255, 170, 0, 0.6)' :
                      'rgba(68, 255, 68, 0.6)'
                    ))},
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 100 } },
                plugins: { title: { display: true, text: 'Success Rate by Scenario' } }
            }
        });

        // Response Time Chart
        new Chart(document.getElementById('response-time-chart'), {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(validResults.map(r => r.scenario))},
                datasets: [
                    {
                        label: 'Average Response Time (ms)',
                        data: ${JSON.stringify(validResults.map(r => r.averageResponseTime))},
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderWidth: 1
                    },
                    {
                        label: 'P95 Response Time (ms)',
                        data: ${JSON.stringify(validResults.map(r => r.p95ResponseTime))},
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } },
                plugins: { title: { display: true, text: 'Response Times by Scenario' } }
            }
        });

        // Individual throughput charts
        ${validResults.map((result, index) => `
        if (${JSON.stringify(result.throughput)}.length > 0) {
            new Chart(document.getElementById('throughput-chart-${index}'), {
                type: 'line',
                data: {
                    labels: ${JSON.stringify(result.throughput.map(t => new Date(t.timestamp).toLocaleTimeString()))},
                    datasets: [{
                        label: 'Requests/Second',
                        data: ${JSON.stringify(result.throughput.map(t => t.requestsPerSecond))},
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { title: { display: true, text: '${result.scenario} - Throughput Over Time' } }
                }
            });
        }
        `).join('')}
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
    maxConcurrentUsers: parseInt(process.env.MAX_USERS) || 50,
    testDuration: parseInt(process.env.TEST_DURATION) || 300000,
    outputDir: process.env.OUTPUT_DIR || './tests/performance/concurrent-reports',
  };

  const tester = new ConcurrentUserTester(config);
  
  tester.runConcurrentUserTests()
    .then(report => {
      console.log('\n🎯 Concurrent User Testing Summary:');
      console.log(`Total scenarios: ${report.summary.totalScenarios}`);
      console.log(`Successful scenarios: ${report.summary.successfulScenarios}`);
      console.log(`Overall performance: ${report.summary.overallPerformance}`);
      
      if (report.summary.criticalIssues.length > 0) {
        console.log('\n⚠️  Critical Issues:');
        report.summary.criticalIssues.forEach(issue => {
          console.log(`  ${issue.scenario}: ${issue.issue} (${issue.value})`);
        });
      }
      
      process.exit(report.summary.overallPerformance === 'critical' ? 1 : 0);
    })
    .catch(error => {
      console.error('Concurrent user testing failed:', error);
      process.exit(1);
    });
}

module.exports = ConcurrentUserTester;