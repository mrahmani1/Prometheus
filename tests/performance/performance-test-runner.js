#!/usr/bin/env node
/**
 * Performance Test Runner for Prometheus AI Platform
 * Orchestrates all performance testing components and generates comprehensive reports
 */

const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

// Import performance testing modules
const LighthouseCIRunner = require('./lighthouse-ci');
const DatabasePerformanceTester = require('./database-performance-tests');
const MemoryLeakDetector = require('./memory-leak-detection');
const NetworkThrottlingTester = require('./network-throttling-tests');
const ConcurrentUserTester = require('./concurrent-user-tests');
const ResourceMonitor = require('./resource-monitoring');
const PerformanceRegressionDetector = require('./regression-detection');

class PerformanceTestRunner {
  constructor(config = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:3000',
      outputDir: config.outputDir || './tests/performance/reports',
      runK6Tests: config.runK6Tests !== false,
      runLighthouseTests: config.runLighthouseTests !== false,
      runDatabaseTests: config.runDatabaseTests !== false,
      runMemoryTests: config.runMemoryTests !== false,
      runNetworkTests: config.runNetworkTests !== false,
      runConcurrentTests: config.runConcurrentTests !== false,
      runResourceMonitoring: config.runResourceMonitoring !== false,
      runRegressionDetection: config.runRegressionDetection !== false,
      testSuite: config.testSuite || 'full', // 'full', 'quick', 'smoke'
      ...config,
    };

    this.results = {
      summary: {},
      k6Results: null,
      lighthouseResults: null,
      databaseResults: null,
      memoryResults: null,
      networkResults: null,
      concurrentResults: null,
      resourceResults: null,
      regressionResults: null,
    };

    this.testSuites = {
      smoke: {
        description: 'Quick smoke tests for basic functionality',
        duration: 60000, // 1 minute
        tests: ['k6', 'lighthouse'],
      },
      quick: {
        description: 'Quick performance validation',
        duration: 300000, // 5 minutes
        tests: ['k6', 'lighthouse', 'memory', 'resource'],
      },
      full: {
        description: 'Comprehensive performance testing',
        duration: 1800000, // 30 minutes
        tests: ['k6', 'lighthouse', 'database', 'memory', 'network', 'concurrent', 'resource'],
      },
    };
  }

  async runPerformanceTests() {
    console.log('🚀 Starting Prometheus AI Platform Performance Tests...');
    console.log(`📋 Test Suite: ${this.config.testSuite} - ${this.testSuites[this.config.testSuite]?.description || 'Custom'}`);
    
    const startTime = Date.now();
    await this.ensureOutputDirectory();
    
    try {
      // Start resource monitoring if enabled
      let resourceMonitor = null;
      if (this.config.runResourceMonitoring && this.shouldRunTest('resource')) {
        resourceMonitor = new ResourceMonitor({
          outputDir: path.join(this.config.outputDir, 'resource-reports'),
          monitoringDuration: 0, // Continuous monitoring
        });
        await resourceMonitor.startMonitoring('performance-test-suite');
      }

      // Run performance tests based on configuration
      const testPromises = [];

      if (this.config.runK6Tests && this.shouldRunTest('k6')) {
        testPromises.push(this.runK6LoadTests());
      }

      if (this.config.runLighthouseTests && this.shouldRunTest('lighthouse')) {
        testPromises.push(this.runLighthouseTests());
      }

      if (this.config.runDatabaseTests && this.shouldRunTest('database')) {
        testPromises.push(this.runDatabaseTests());
      }

      if (this.config.runMemoryTests && this.shouldRunTest('memory')) {
        testPromises.push(this.runMemoryLeakTests());
      }

      if (this.config.runNetworkTests && this.shouldRunTest('network')) {
        testPromises.push(this.runNetworkThrottlingTests());
      }

      if (this.config.runConcurrentTests && this.shouldRunTest('concurrent')) {
        testPromises.push(this.runConcurrentUserTests());
      }

      // Run tests in parallel where possible
      await Promise.allSettled(testPromises);

      // Stop resource monitoring
      if (resourceMonitor) {
        this.results.resourceResults = await resourceMonitor.stopMonitoring();
      }

      // Run regression detection
      if (this.config.runRegressionDetection) {
        await this.runRegressionDetection();
      }

      // Generate comprehensive report
      const report = await this.generateComprehensiveReport();
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      console.log('\n🎯 Performance Testing Summary:');
      console.log(`⏱️  Total Duration: ${(duration / 1000 / 60).toFixed(1)} minutes`);
      console.log(`📊 Tests Completed: ${report.summary.completedTests}/${report.summary.totalTests}`);
      console.log(`🎯 Overall Score: ${report.summary.overallScore}/100`);
      console.log(`🚨 Critical Issues: ${report.summary.criticalIssues}`);
      
      if (report.summary.overallScore < 70) {
        console.log('⚠️  PERFORMANCE ISSUES DETECTED - Review required');
        return { success: false, report };
      } else if (report.summary.overallScore < 85) {
        console.log('⚠️  Performance could be improved');
        return { success: true, report, warnings: true };
      } else {
        console.log('✅ Performance tests passed successfully');
        return { success: true, report };
      }

    } catch (error) {
      console.error('❌ Performance testing failed:', error.message);
      return { success: false, error: error.message };
    }
  }

  shouldRunTest(testType) {
    const suite = this.testSuites[this.config.testSuite];
    return !suite || suite.tests.includes(testType);
  }

  async runK6LoadTests() {
    console.log('\n📈 Running K6 Load Tests...');
    
    try {
      const k6TestFile = path.join(__dirname, 'k6-load-tests.js');
      const result = await this.executeK6Test(k6TestFile);
      
      this.results.k6Results = result;
      console.log('✅ K6 load tests completed');
      
    } catch (error) {
      console.error('❌ K6 load tests failed:', error.message);
      this.results.k6Results = { error: error.message };
    }
  }

  async executeK6Test(testFile) {
    return new Promise((resolve, reject) => {
      const k6Process = spawn('k6', [
        'run',
        '--out', 'json=k6-results.json',
        '--env', `BASE_URL=${this.config.baseUrl}`,
        testFile
      ], {
        stdio: 'pipe',
        cwd: this.config.outputDir,
      });

      let output = '';
      let errorOutput = '';

      k6Process.stdout.on('data', (data) => {
        output += data.toString();
        process.stdout.write(data); // Show real-time output
      });

      k6Process.stderr.on('data', (data) => {
        errorOutput += data.toString();
        process.stderr.write(data);
      });

      k6Process.on('close', async (code) => {
        if (code === 0) {
          try {
            // Parse K6 results
            const resultsFile = path.join(this.config.outputDir, 'k6-results.json');
            const results = await this.parseK6Results(resultsFile);
            resolve(results);
          } catch (parseError) {
            reject(new Error(`Failed to parse K6 results: ${parseError.message}`));
          }
        } else {
          reject(new Error(`K6 test failed with code ${code}: ${errorOutput}`));
        }
      });

      k6Process.on('error', (error) => {
        reject(new Error(`Failed to start K6: ${error.message}`));
      });
    });
  }

  async parseK6Results(resultsFile) {
    try {
      const data = await fs.readFile(resultsFile, 'utf8');
      const lines = data.trim().split('\n');
      const metrics = {};
      
      lines.forEach(line => {
        try {
          const result = JSON.parse(line);
          if (result.type === 'Point' && result.data) {
            const metricName = result.metric;
            if (!metrics[metricName]) {
              metrics[metricName] = [];
            }
            metrics[metricName].push(result.data.value);
          }
        } catch (e) {
          // Skip invalid JSON lines
        }
      });

      // Calculate summary statistics
      const summary = {};
      Object.entries(metrics).forEach(([metric, values]) => {
        if (values.length > 0) {
          summary[metric] = {
            count: values.length,
            avg: values.reduce((sum, val) => sum + val, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            p95: this.calculatePercentile(values, 0.95),
          };
        }
      });

      return {
        summary,
        rawMetrics: metrics,
        testDuration: summary.iteration_duration?.count || 0,
        totalRequests: summary.http_reqs?.count || 0,
        errorRate: summary.http_req_failed ? summary.http_req_failed.avg : 0,
        avgResponseTime: summary.http_req_duration ? summary.http_req_duration.avg : 0,
        throughput: summary.http_reqs ? summary.http_reqs.count / (summary.iteration_duration?.avg || 1) : 0,
      };

    } catch (error) {
      throw new Error(`Failed to parse K6 results: ${error.message}`);
    }
  }

  async runLighthouseTests() {
    console.log('\n🔍 Running Lighthouse Performance Tests...');
    
    try {
      const lighthouse = new LighthouseCIRunner({
        baseUrl: this.config.baseUrl,
        outputDir: path.join(this.config.outputDir, 'lighthouse-reports'),
      });
      
      const result = await lighthouse.runGeologicalPerformanceAudit();
      this.results.lighthouseResults = result;
      console.log('✅ Lighthouse tests completed');
      
    } catch (error) {
      console.error('❌ Lighthouse tests failed:', error.message);
      this.results.lighthouseResults = { error: error.message };
    }
  }

  async runDatabaseTests() {
    console.log('\n🗄️  Running Database Performance Tests...');
    
    try {
      // Load database configuration
      const dbConfig = await this.loadDatabaseConfig();
      
      const dbTester = new DatabasePerformanceTester(dbConfig);
      const result = await dbTester.runAllTests();
      
      this.results.databaseResults = result;
      console.log('✅ Database tests completed');
      
    } catch (error) {
      console.error('❌ Database tests failed:', error.message);
      this.results.databaseResults = { error: error.message };
    }
  }

  async runMemoryLeakTests() {
    console.log('\n🧠 Running Memory Leak Detection Tests...');
    
    try {
      const memoryDetector = new MemoryLeakDetector({
        baseUrl: this.config.baseUrl,
        testDuration: this.getTestDuration('memory'),
        outputDir: path.join(this.config.outputDir, 'memory-reports'),
      });
      
      const result = await memoryDetector.runMemoryLeakDetection();
      this.results.memoryResults = result;
      console.log('✅ Memory leak tests completed');
      
    } catch (error) {
      console.error('❌ Memory leak tests failed:', error.message);
      this.results.memoryResults = { error: error.message };
    }
  }

  async runNetworkThrottlingTests() {
    console.log('\n🌐 Running Network Throttling Tests...');
    
    try {
      const networkTester = new NetworkThrottlingTester({
        baseUrl: this.config.baseUrl,
        testDuration: this.getTestDuration('network'),
        outputDir: path.join(this.config.outputDir, 'network-reports'),
      });
      
      const result = await networkTester.runNetworkThrottlingTests();
      this.results.networkResults = result;
      console.log('✅ Network throttling tests completed');
      
    } catch (error) {
      console.error('❌ Network throttling tests failed:', error.message);
      this.results.networkResults = { error: error.message };
    }
  }

  async runConcurrentUserTests() {
    console.log('\n👥 Running Concurrent User Tests...');
    
    try {
      const concurrentTester = new ConcurrentUserTester({
        baseUrl: this.config.baseUrl,
        maxConcurrentUsers: this.getMaxConcurrentUsers(),
        testDuration: this.getTestDuration('concurrent'),
        outputDir: path.join(this.config.outputDir, 'concurrent-reports'),
      });
      
      const result = await concurrentTester.runConcurrentUserTests();
      this.results.concurrentResults = result;
      console.log('✅ Concurrent user tests completed');
      
    } catch (error) {
      console.error('❌ Concurrent user tests failed:', error.message);
      this.results.concurrentResults = { error: error.message };
    }
  }

  async runRegressionDetection() {
    console.log('\n🔍 Running Performance Regression Detection...');
    
    try {
      const regressionDetector = new PerformanceRegressionDetector({
        baselineDir: path.join(this.config.outputDir, 'baselines'),
        outputDir: path.join(this.config.outputDir, 'regression-reports'),
      });
      
      const result = await regressionDetector.detectRegressions(this.results, 'comprehensive');
      this.results.regressionResults = result;
      console.log('✅ Regression detection completed');
      
    } catch (error) {
      console.error('❌ Regression detection failed:', error.message);
      this.results.regressionResults = { error: error.message };
    }
  }

  getTestDuration(testType) {
    const suite = this.testSuites[this.config.testSuite];
    if (!suite) return 300000; // Default 5 minutes
    
    const baseDuration = suite.duration;
    const multipliers = {
      memory: 0.5, // 50% of suite duration
      network: 0.3, // 30% of suite duration
      concurrent: 0.4, // 40% of suite duration
    };
    
    return Math.floor(baseDuration * (multipliers[testType] || 0.2));
  }

  getMaxConcurrentUsers() {
    const suite = this.testSuites[this.config.testSuite];
    const userCounts = {
      smoke: 10,
      quick: 25,
      full: 50,
    };
    
    return userCounts[this.config.testSuite] || 25;
  }

  async loadDatabaseConfig() {
    // Try to load database configuration from file
    const configPaths = [
      path.join(process.cwd(), 'tests/performance/db-config.json'),
      path.join(process.cwd(), 'config/database.json'),
      path.join(process.cwd(), '.env.json'),
    ];
    
    for (const configPath of configPaths) {
      try {
        const data = await fs.readFile(configPath, 'utf8');
        const config = JSON.parse(data);
        if (config.postgres || config.database) {
          return config;
        }
      } catch (error) {
        // Continue to next config file
      }
    }
    
    // Return default configuration
    return {
      postgres: {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: parseInt(process.env.POSTGRES_PORT) || 5432,
        user: process.env.POSTGRES_USER || 'prometheus',
        password: process.env.POSTGRES_PASSWORD || 'prometheus123',
        database: process.env.POSTGRES_DB || 'prometheus',
      },
      influxdb: {
        url: process.env.INFLUXDB_URL || 'http://localhost:8086',
        token: process.env.INFLUXDB_TOKEN || 'prometheus-token',
        org: process.env.INFLUXDB_ORG || 'prometheus',
        bucket: process.env.INFLUXDB_BUCKET || 'geological-data',
      },
      neo4j: {
        uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
        user: process.env.NEO4J_USER || 'neo4j',
        password: process.env.NEO4J_PASSWORD || 'prometheus123',
      },
      redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT) || 6379,
        password: process.env.REDIS_PASSWORD || null,
      },
    };
  }

  async generateComprehensiveReport() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Calculate overall performance score
    const scores = this.calculatePerformanceScores();
    const overallScore = this.calculateOverallScore(scores);
    
    // Identify critical issues
    const criticalIssues = this.identifyCriticalIssues();
    
    // Generate recommendations
    const recommendations = this.generateRecommendations();
    
    const report = {
      timestamp: new Date().toISOString(),
      testSuite: this.config.testSuite,
      configuration: this.config,
      summary: {
        overallScore,
        completedTests: this.countCompletedTests(),
        totalTests: this.countTotalTests(),
        criticalIssues: criticalIssues.length,
        hasRegressions: this.results.regressionResults?.hasRegression || false,
        testDuration: this.calculateTotalTestDuration(),
      },
      scores,
      criticalIssues,
      recommendations,
      results: this.results,
    };
    
    // Save comprehensive report
    const reportPath = path.join(
      this.config.outputDir,
      `comprehensive-performance-report-${timestamp}.json`
    );
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    // Generate HTML dashboard
    await this.generateHTMLDashboard(report, timestamp);
    
    console.log(`📊 Comprehensive report saved: ${reportPath}`);
    
    return report;
  }

  calculatePerformanceScores() {
    const scores = {};
    
    // K6 Load Test Score
    if (this.results.k6Results && !this.results.k6Results.error) {
      const k6 = this.results.k6Results;
      let k6Score = 100;
      
      if (k6.errorRate > 0.05) k6Score -= 30; // High error rate
      if (k6.avgResponseTime > 2000) k6Score -= 20; // Slow response
      if (k6.throughput < 100) k6Score -= 15; // Low throughput
      
      scores.k6 = Math.max(0, k6Score);
    }
    
    // Lighthouse Score
    if (this.results.lighthouseResults && !this.results.lighthouseResults.error) {
      const lighthouse = this.results.lighthouseResults;
      scores.lighthouse = lighthouse.averageScores?.performance || 0;
    }
    
    // Database Performance Score
    if (this.results.databaseResults && !this.results.databaseResults.error) {
      const db = this.results.databaseResults;
      let dbScore = 100;
      
      if (db.total_errors > 0) dbScore -= 25;
      if (Object.values(db.average_throughput_by_db || {}).some(t => t < 100)) dbScore -= 20;
      
      scores.database = Math.max(0, dbScore);
    }
    
    // Memory Leak Score
    if (this.results.memoryResults && !this.results.memoryResults.error) {
      const memory = this.results.memoryResults;
      const leaksDetected = memory.summary?.leaksDetected || 0;
      scores.memory = Math.max(0, 100 - (leaksDetected * 25));
    }
    
    // Network Performance Score
    if (this.results.networkResults && !this.results.networkResults.error) {
      const network = this.results.networkResults;
      scores.network = network.summary?.averagePerformanceScore || 0;
    }
    
    // Concurrent User Score
    if (this.results.concurrentResults && !this.results.concurrentResults.error) {
      const concurrent = this.results.concurrentResults;
      scores.concurrent = concurrent.summary?.overallPerformance === 'good' ? 85 :
                         concurrent.summary?.overallPerformance === 'warning' ? 65 : 35;
    }
    
    // Resource Usage Score
    if (this.results.resourceResults && !this.results.resourceResults.error) {
      const resource = this.results.resourceResults;
      scores.resource = resource.performanceAssessment?.score || 0;
    }
    
    return scores;
  }

  calculateOverallScore(scores) {
    const validScores = Object.values(scores).filter(score => typeof score === 'number');
    if (validScores.length === 0) return 0;
    
    // Weighted average with higher weight for critical components
    const weights = {
      k6: 0.25,
      lighthouse: 0.15,
      database: 0.20,
      memory: 0.15,
      network: 0.10,
      concurrent: 0.10,
      resource: 0.05,
    };
    
    let weightedSum = 0;
    let totalWeight = 0;
    
    Object.entries(scores).forEach(([test, score]) => {
      const weight = weights[test] || 0.1;
      weightedSum += score * weight;
      totalWeight += weight;
    });
    
    return totalWeight > 0 ? Math.round(weightedSum / totalWeight) : 0;
  }

  identifyCriticalIssues() {
    const issues = [];
    
    // Check for regressions
    if (this.results.regressionResults?.hasSignificantRegression) {
      issues.push({
        type: 'regression',
        severity: 'critical',
        message: 'Significant performance regression detected',
        details: this.results.regressionResults.regressions?.filter(r => r.severity === 'critical'),
      });
    }
    
    // Check K6 results
    if (this.results.k6Results && !this.results.k6Results.error) {
      const k6 = this.results.k6Results;
      if (k6.errorRate > 0.1) {
        issues.push({
          type: 'reliability',
          severity: 'critical',
          message: `High error rate: ${(k6.errorRate * 100).toFixed(1)}%`,
        });
      }
      if (k6.avgResponseTime > 5000) {
        issues.push({
          type: 'performance',
          severity: 'high',
          message: `Very slow response times: ${k6.avgResponseTime}ms`,
        });
      }
    }
    
    // Check memory leaks
    if (this.results.memoryResults?.summary?.leaksDetected > 0) {
      issues.push({
        type: 'memory',
        severity: 'high',
        message: `Memory leaks detected in ${this.results.memoryResults.summary.leaksDetected} tests`,
      });
    }
    
    // Check resource usage
    if (this.results.resourceResults) {
      const resource = this.results.resourceResults;
      if (resource.peakCPU > 95) {
        issues.push({
          type: 'resource',
          severity: 'critical',
          message: `Extreme CPU usage: ${resource.peakCPU.toFixed(1)}%`,
        });
      }
      if (resource.peakMemory > 95) {
        issues.push({
          type: 'resource',
          severity: 'critical',
          message: `Extreme memory usage: ${resource.peakMemory.toFixed(1)}%`,
        });
      }
    }
    
    return issues;
  }

  generateRecommendations() {
    const recommendations = [];
    
    // Performance optimization recommendations
    const scores = this.calculatePerformanceScores();
    
    if (scores.k6 < 70) {
      recommendations.push({
        category: 'api_performance',
        priority: 'high',
        message: 'API performance needs optimization',
        actions: [
          'Review database query performance',
          'Implement caching strategies',
          'Optimize geological data processing algorithms',
          'Consider API rate limiting and throttling',
        ],
      });
    }
    
    if (scores.lighthouse < 70) {
      recommendations.push({
        category: 'frontend_performance',
        priority: 'medium',
        message: 'Frontend performance could be improved',
        actions: [
          'Optimize JavaScript bundle sizes',
          'Implement lazy loading for 3D visualizations',
          'Compress and optimize geological imagery',
          'Use WebGL efficiently for geological models',
        ],
      });
    }
    
    if (scores.database < 70) {
      recommendations.push({
        category: 'database_performance',
        priority: 'high',
        message: 'Database performance optimization needed',
        actions: [
          'Add indexes for geological data queries',
          'Optimize time-series data storage in InfluxDB',
          'Review PostgreSQL configuration for large datasets',
          'Implement database connection pooling',
        ],
      });
    }
    
    if (this.results.memoryResults?.summary?.leaksDetected > 0) {
      recommendations.push({
        category: 'memory_management',
        priority: 'high',
        message: 'Memory leak issues detected',
        actions: [
          'Review WebGL resource cleanup in 3D visualizations',
          'Implement proper event listener cleanup',
          'Optimize geological data structure management',
          'Add memory monitoring to production systems',
        ],
      });
    }
    
    // Geological-specific recommendations
    if (scores.concurrent < 70) {
      recommendations.push({
        category: 'geological_scalability',
        priority: 'medium',
        message: 'System may not handle emergency scenarios well',
        actions: [
          'Implement auto-scaling for volcanic emergencies',
          'Optimize real-time seismic data processing',
          'Add load balancing for geological services',
          'Prepare for high-traffic during major events',
        ],
      });
    }
    
    return recommendations;
  }

  countCompletedTests() {
    let completed = 0;
    Object.values(this.results).forEach(result => {
      if (result && !result.error) completed++;
    });
    return completed;
  }

  countTotalTests() {
    return Object.keys(this.results).length - 1; // Exclude summary
  }

  calculateTotalTestDuration() {
    // Estimate based on test suite
    const suite = this.testSuites[this.config.testSuite];
    return suite ? suite.duration : 0;
  }

  async generateHTMLDashboard(report, timestamp) {
    const htmlContent = this.generateDashboardHTML(report);
    const dashboardPath = path.join(
      this.config.outputDir,
      `performance-dashboard-${timestamp}.html`
    );
    
    await fs.writeFile(dashboardPath, htmlContent);
    console.log(`📊 Performance dashboard saved: ${dashboardPath}`);
  }

  generateDashboardHTML(report) {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Prometheus AI Platform - Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .score { font-size: 3em; font-weight: bold; text-align: center; margin: 10px 0; }
        .score-excellent { color: #28a745; }
        .score-good { color: #17a2b8; }
        .score-warning { color: #ffc107; }
        .score-critical { color: #dc3545; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .issues-section { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .issue-item { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .issue-critical { background-color: #f8d7da; border-left: 5px solid #dc3545; }
        .issue-high { background-color: #fff3cd; border-left: 5px solid #ffc107; }
        .issue-medium { background-color: #d1ecf1; border-left: 5px solid #17a2b8; }
        .recommendations { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .recommendation-item { background-color: #e7f3ff; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 5px solid #007bff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌋 Prometheus AI Platform - Performance Dashboard</h1>
        <p>Comprehensive Performance Testing Results</p>
        <p><strong>Test Suite:</strong> ${report.testSuite} | <strong>Timestamp:</strong> ${new Date(report.timestamp).toLocaleString()}</p>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <h3>Overall Performance</h3>
            <div class="score score-${this.getScoreClass(report.summary.overallScore)}">${report.summary.overallScore}</div>
            <p>Performance Score</p>
        </div>
        <div class="summary-card">
            <h3>Test Completion</h3>
            <div class="score score-${report.summary.completedTests === report.summary.totalTests ? 'excellent' : 'warning'}">
                ${report.summary.completedTests}/${report.summary.totalTests}
            </div>
            <p>Tests Completed</p>
        </div>
        <div class="summary-card">
            <h3>Critical Issues</h3>
            <div class="score score-${report.summary.criticalIssues === 0 ? 'excellent' : 'critical'}">${report.summary.criticalIssues}</div>
            <p>Issues Found</p>
        </div>
        <div class="summary-card">
            <h3>Regressions</h3>
            <div class="score score-${report.summary.hasRegressions ? 'critical' : 'excellent'}">
                ${report.summary.hasRegressions ? 'YES' : 'NO'}
            </div>
            <p>Performance Regressions</p>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="performance-scores-chart" width="400" height="200"></canvas>
    </div>

    ${report.criticalIssues.length > 0 ? `
    <div class="issues-section">
        <h3>🚨 Critical Issues</h3>
        ${report.criticalIssues.map(issue => `
        <div class="issue-item issue-${issue.severity}">
            <h4>${issue.type.toUpperCase()}: ${issue.message}</h4>
            ${issue.details ? `<p>Details: ${JSON.stringify(issue.details)}</p>` : ''}
        </div>
        `).join('')}
    </div>
    ` : ''}

    <div class="recommendations">
        <h3>💡 Performance Recommendations</h3>
        ${report.recommendations.map(rec => `
        <div class="recommendation-item">
            <h4>${rec.category.replace(/_/g, ' ').toUpperCase()} (${rec.priority})</h4>
            <p>${rec.message}</p>
            <ul>
                ${rec.actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
        </div>
        `).join('')}
    </div>

    <script>
        // Performance Scores Chart
        const scoresData = ${JSON.stringify(report.scores)};
        const labels = Object.keys(scoresData);
        const scores = Object.values(scoresData);

        new Chart(document.getElementById('performance-scores-chart'), {
            type: 'radar',
            data: {
                labels: labels.map(label => label.toUpperCase()),
                datasets: [{
                    label: 'Performance Scores',
                    data: scores,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Scores by Test Category'
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>`;
  }

  getScoreClass(score) {
    if (score >= 85) return 'excellent';
    if (score >= 70) return 'good';
    if (score >= 50) return 'warning';
    return 'critical';
  }

  calculatePercentile(values, percentile) {
    if (!Array.isArray(values) || values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[Math.max(0, index)];
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
  const args = process.argv.slice(2);
  const config = {
    baseUrl: process.env.BASE_URL || 'http://localhost:3000',
    testSuite: process.env.TEST_SUITE || 'quick',
    outputDir: process.env.OUTPUT_DIR || './tests/performance/reports',
  };

  // Parse command line arguments
  args.forEach(arg => {
    if (arg.startsWith('--suite=')) {
      config.testSuite = arg.split('=')[1];
    } else if (arg.startsWith('--url=')) {
      config.baseUrl = arg.split('=')[1];
    } else if (arg.startsWith('--output=')) {
      config.outputDir = arg.split('=')[1];
    } else if (arg === '--no-k6') {
      config.runK6Tests = false;
    } else if (arg === '--no-lighthouse') {
      config.runLighthouseTests = false;
    } else if (arg === '--no-database') {
      config.runDatabaseTests = false;
    } else if (arg === '--no-memory') {
      config.runMemoryTests = false;
    } else if (arg === '--no-network') {
      config.runNetworkTests = false;
    } else if (arg === '--no-concurrent') {
      config.runConcurrentTests = false;
    } else if (arg === '--no-resource') {
      config.runResourceMonitoring = false;
    } else if (arg === '--no-regression') {
      config.runRegressionDetection = false;
    }
  });

  console.log('🚀 Starting Prometheus AI Platform Performance Tests');
  console.log(`Configuration: ${JSON.stringify(config, null, 2)}`);

  const runner = new PerformanceTestRunner(config);
  
  runner.runPerformanceTests()
    .then(result => {
      if (result.success) {
        console.log('\n✅ Performance testing completed successfully');
        if (result.warnings) {
          console.log('⚠️  Some performance issues detected - review recommended');
          process.exit(1);
        } else {
          process.exit(0);
        }
      } else {
        console.log('\n❌ Performance testing failed');
        console.error(result.error || 'Unknown error');
        process.exit(1);
      }
    })
    .catch(error => {
      console.error('\n💥 Performance testing crashed:', error.message);
      process.exit(1);
    });
}

module.exports = PerformanceTestRunner;