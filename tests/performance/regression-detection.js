/**
 * Performance Regression Detection System
 * Compares current performance metrics against historical baselines
 */

const fs = require('fs').promises;
const path = require('path');

class PerformanceRegressionDetector {
  constructor(config = {}) {
    this.config = {
      baselineDir: config.baselineDir || './tests/performance/baselines',
      reportsDir: config.reportsDir || './tests/performance/reports',
      outputDir: config.outputDir || './tests/performance/regression-reports',
      regressionThresholds: {
        responseTime: config.regressionThresholds?.responseTime || 0.2, // 20% increase
        throughput: config.regressionThresholds?.throughput || 0.15, // 15% decrease
        errorRate: config.regressionThresholds?.errorRate || 0.05, // 5% increase
        memoryUsage: config.regressionThresholds?.memoryUsage || 0.25, // 25% increase
        cpuUsage: config.regressionThresholds?.cpuUsage || 0.2, // 20% increase
        ...config.regressionThresholds,
      },
      ...config,
    };

    // Geological-specific performance metrics
    this.geologicalMetrics = [
      'seismic_processing_time',
      'volcanic_alert_latency',
      'geological_model_render_time',
      'real_time_data_latency',
      'database_query_time',
      'api_response_time',
      'memory_leak_rate',
      'concurrent_user_capacity',
    ];
  }

  async detectRegressions(currentResults, testType = 'general') {
    console.log('🔍 Starting Performance Regression Detection...');
    
    await this.ensureDirectories();
    
    try {
      // Load historical baselines
      const baselines = await this.loadBaselines(testType);
      
      if (!baselines || Object.keys(baselines).length === 0) {
        console.log('📊 No baselines found. Creating initial baseline...');
        await this.createBaseline(currentResults, testType);
        return {
          hasRegression: false,
          isBaseline: true,
          message: 'Initial baseline created',
        };
      }
      
      // Perform regression analysis
      const regressionAnalysis = await this.analyzeRegressions(currentResults, baselines, testType);
      
      // Generate regression report
      const report = await this.generateRegressionReport(regressionAnalysis, testType);
      
      // Update baselines if no significant regressions
      if (!regressionAnalysis.hasSignificantRegression) {
        await this.updateBaselines(currentResults, baselines, testType);
      }
      
      return report;
      
    } catch (error) {
      console.error('Regression detection failed:', error.message);
      return {
        hasRegression: false,
        error: error.message,
      };
    }
  }

  async loadBaselines(testType) {
    try {
      const baselineFile = path.join(this.config.baselineDir, `${testType}-baseline.json`);
      const data = await fs.readFile(baselineFile, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      return null;
    }
  }

  async createBaseline(results, testType) {
    const baseline = {
      testType,
      createdAt: new Date().toISOString(),
      version: '1.0.0',
      metrics: this.extractMetrics(results),
      statisticalData: this.calculateStatistics(results),
    };
    
    const baselineFile = path.join(this.config.baselineDir, `${testType}-baseline.json`);
    await fs.writeFile(baselineFile, JSON.stringify(baseline, null, 2));
    
    console.log(`✅ Baseline created: ${baselineFile}`);
    return baseline;
  }

  extractMetrics(results) {
    const metrics = {};
    
    // Extract different types of performance metrics based on test type
    if (results.k6Results) {
      metrics.k6 = this.extractK6Metrics(results.k6Results);
    }
    
    if (results.lighthouseResults) {
      metrics.lighthouse = this.extractLighthouseMetrics(results.lighthouseResults);
    }
    
    if (results.databaseResults) {
      metrics.database = this.extractDatabaseMetrics(results.databaseResults);
    }
    
    if (results.memoryResults) {
      metrics.memory = this.extractMemoryMetrics(results.memoryResults);
    }
    
    if (results.networkResults) {
      metrics.network = this.extractNetworkMetrics(results.networkResults);
    }
    
    if (results.concurrentResults) {
      metrics.concurrent = this.extractConcurrentMetrics(results.concurrentResults);
    }
    
    if (results.resourceResults) {
      metrics.resource = this.extractResourceMetrics(results.resourceResults);
    }
    
    return metrics;
  }

  extractK6Metrics(k6Results) {
    return {
      avgResponseTime: this.calculateAverage(k6Results.responseTimes || []),
      p95ResponseTime: this.calculatePercentile(k6Results.responseTimes || [], 0.95),
      throughput: k6Results.throughput || 0,
      errorRate: k6Results.errorRate || 0,
      geologicalApiLatency: k6Results.geologicalApiLatency || 0,
      seismicProcessingTime: k6Results.seismicProcessingTime || 0,
      volcanicDataThroughput: k6Results.volcanicDataThroughput || 0,
    };
  }

  extractLighthouseMetrics(lighthouseResults) {
    const metrics = {};
    
    if (Array.isArray(lighthouseResults)) {
      lighthouseResults.forEach(result => {
        if (result.scores) {
          metrics[result.page] = {
            performance: result.scores.performance,
            accessibility: result.scores.accessibility,
            firstContentfulPaint: result.metrics?.['first-contentful-paint'],
            largestContentfulPaint: result.metrics?.['largest-contentful-paint'],
            speedIndex: result.metrics?.['speed-index'],
            cumulativeLayoutShift: result.metrics?.['cumulative-layout-shift'],
            totalBlockingTime: result.metrics?.['total-blocking-time'],
          };
        }
      });
    }
    
    return metrics;
  }

  extractDatabaseMetrics(databaseResults) {
    return {
      postgresqlThroughput: databaseResults.postgresql?.throughput || 0,
      postgresqlLatency: databaseResults.postgresql?.averageLatency || 0,
      influxdbWriteRate: databaseResults.influxdb?.writeRate || 0,
      influxdbQueryLatency: databaseResults.influxdb?.queryLatency || 0,
      neo4jTransactionRate: databaseResults.neo4j?.transactionRate || 0,
      redisLatency: databaseResults.redis?.latency || 0,
      geologicalQueryPerformance: databaseResults.geologicalQueries?.averageTime || 0,
    };
  }

  extractMemoryMetrics(memoryResults) {
    const metrics = {};
    
    if (Array.isArray(memoryResults)) {
      memoryResults.forEach(result => {
        metrics[result.testName] = {
          memoryGrowthRate: result.memoryGrowthRate || 0,
          peakMemory: result.peakMemory || 0,
          leakDetected: result.leakDetected || false,
          gcFrequency: result.gcAnalysis?.gcFrequency || 0,
          memoryVolatility: result.memoryVolatility || 0,
        };
      });
    }
    
    return metrics;
  }

  extractNetworkMetrics(networkResults) {
    const metrics = {};
    
    if (Array.isArray(networkResults)) {
      networkResults.forEach(result => {
        const key = `${result.scenario}_${result.networkProfile}`;
        metrics[key] = {
          performanceScore: result.performanceScore || 0,
          navigationTime: result.navigationTime || 0,
          dataTransferred: result.totalDataTransferred || 0,
          averageLatency: result.averageLatency || 0,
          networkErrors: result.networkErrors || 0,
        };
      });
    }
    
    return metrics;
  }

  extractConcurrentMetrics(concurrentResults) {
    const metrics = {};
    
    if (Array.isArray(concurrentResults)) {
      concurrentResults.forEach(result => {
        metrics[result.scenario] = {
          peakConcurrentUsers: result.peakConcurrentUsers || 0,
          successRate: result.successRate || 0,
          averageResponseTime: result.averageResponseTime || 0,
          p95ResponseTime: result.p95ResponseTime || 0,
          errorRate: result.errorRate || 0,
          throughput: result.averageThroughput || 0,
        };
      });
    }
    
    return metrics;
  }

  extractResourceMetrics(resourceResults) {
    return {
      peakCPU: resourceResults.peakCPU || 0,
      avgCPU: resourceResults.avgCPU || 0,
      peakMemory: resourceResults.peakMemory || 0,
      avgMemory: resourceResults.avgMemory || 0,
      peakDiskIO: resourceResults.peakDiskIO || 0,
      totalAlerts: resourceResults.totalAlerts || 0,
      performanceScore: resourceResults.performanceAssessment?.score || 0,
    };
  }

  calculateStatistics(results) {
    const stats = {};
    
    // Calculate statistical measures for key metrics
    const metrics = this.extractMetrics(results);
    
    Object.entries(metrics).forEach(([category, categoryMetrics]) => {
      stats[category] = {};
      
      Object.entries(categoryMetrics).forEach(([metric, value]) => {
        if (typeof value === 'number') {
          stats[category][metric] = {
            mean: value,
            std: 0, // Will be calculated with more data points
            min: value,
            max: value,
            count: 1,
          };
        }
      });
    });
    
    return stats;
  }

  async analyzeRegressions(currentResults, baselines, testType) {
    const currentMetrics = this.extractMetrics(currentResults);
    const baselineMetrics = baselines.metrics;
    
    const regressions = [];
    const improvements = [];
    let hasSignificantRegression = false;
    
    // Compare each metric category
    for (const [category, categoryMetrics] of Object.entries(currentMetrics)) {
      if (!baselineMetrics[category]) continue;
      
      const categoryRegressions = await this.compareCategoryMetrics(
        categoryMetrics,
        baselineMetrics[category],
        category
      );
      
      regressions.push(...categoryRegressions.regressions);
      improvements.push(...categoryRegressions.improvements);
      
      if (categoryRegressions.hasSignificantRegression) {
        hasSignificantRegression = true;
      }
    }
    
    // Geological-specific regression analysis
    const geologicalRegressions = await this.analyzeGeologicalRegressions(
      currentMetrics,
      baselineMetrics
    );
    
    regressions.push(...geologicalRegressions);
    
    return {
      hasRegression: regressions.length > 0,
      hasSignificantRegression,
      regressions,
      improvements,
      currentMetrics,
      baselineMetrics,
      testType,
      timestamp: new Date().toISOString(),
    };
  }

  async compareCategoryMetrics(currentMetrics, baselineMetrics, category) {
    const regressions = [];
    const improvements = [];
    let hasSignificantRegression = false;
    
    for (const [metric, currentValue] of Object.entries(currentMetrics)) {
      if (typeof currentValue !== 'number') continue;
      
      const baselineValue = baselineMetrics[metric];
      if (typeof baselineValue !== 'number') continue;
      
      const change = this.calculatePercentageChange(baselineValue, currentValue);
      const isRegression = this.isRegression(metric, change);
      const isSignificant = this.isSignificantChange(metric, Math.abs(change));
      
      if (isRegression && isSignificant) {
        hasSignificantRegression = true;
        regressions.push({
          category,
          metric,
          baselineValue,
          currentValue,
          change,
          severity: this.calculateSeverity(metric, Math.abs(change)),
          impact: this.assessImpact(category, metric, change),
        });
      } else if (!isRegression && isSignificant) {
        improvements.push({
          category,
          metric,
          baselineValue,
          currentValue,
          change,
          improvement: Math.abs(change),
        });
      }
    }
    
    return {
      regressions,
      improvements,
      hasSignificantRegression,
    };
  }

  async analyzeGeologicalRegressions(currentMetrics, baselineMetrics) {
    const geologicalRegressions = [];
    
    // Seismic processing performance
    if (currentMetrics.k6?.seismicProcessingTime && baselineMetrics.k6?.seismicProcessingTime) {
      const change = this.calculatePercentageChange(
        baselineMetrics.k6.seismicProcessingTime,
        currentMetrics.k6.seismicProcessingTime
      );
      
      if (change > 0.3) { // 30% increase in processing time
        geologicalRegressions.push({
          category: 'geological',
          metric: 'seismic_processing_time',
          baselineValue: baselineMetrics.k6.seismicProcessingTime,
          currentValue: currentMetrics.k6.seismicProcessingTime,
          change,
          severity: 'high',
          impact: 'Real-time seismic monitoring may be affected',
        });
      }
    }
    
    // Volcanic alert system performance
    if (currentMetrics.k6?.geologicalApiLatency && baselineMetrics.k6?.geologicalApiLatency) {
      const change = this.calculatePercentageChange(
        baselineMetrics.k6.geologicalApiLatency,
        currentMetrics.k6.geologicalApiLatency
      );
      
      if (change > 0.5) { // 50% increase in API latency
        geologicalRegressions.push({
          category: 'geological',
          metric: 'volcanic_alert_latency',
          baselineValue: baselineMetrics.k6.geologicalApiLatency,
          currentValue: currentMetrics.k6.geologicalApiLatency,
          change,
          severity: 'critical',
          impact: 'Volcanic alert system response time degraded',
        });
      }
    }
    
    // 3D geological model rendering
    if (currentMetrics.lighthouse) {
      Object.entries(currentMetrics.lighthouse).forEach(([page, metrics]) => {
        if (page.includes('3d') && baselineMetrics.lighthouse?.[page]) {
          const currentPerf = metrics.performance;
          const baselinePerf = baselineMetrics.lighthouse[page].performance;
          
          if (currentPerf && baselinePerf) {
            const change = this.calculatePercentageChange(baselinePerf, currentPerf);
            
            if (change < -0.2) { // 20% decrease in performance score
              geologicalRegressions.push({
                category: 'geological',
                metric: 'geological_model_render_time',
                baselineValue: baselinePerf,
                currentValue: currentPerf,
                change: Math.abs(change),
                severity: 'medium',
                impact: '3D geological visualization performance degraded',
              });
            }
          }
        }
      });
    }
    
    // Database query performance for geological data
    if (currentMetrics.database?.geologicalQueryPerformance && 
        baselineMetrics.database?.geologicalQueryPerformance) {
      const change = this.calculatePercentageChange(
        baselineMetrics.database.geologicalQueryPerformance,
        currentMetrics.database.geologicalQueryPerformance
      );
      
      if (change > 0.4) { // 40% increase in query time
        geologicalRegressions.push({
          category: 'geological',
          metric: 'database_query_time',
          baselineValue: baselineMetrics.database.geologicalQueryPerformance,
          currentValue: currentMetrics.database.geologicalQueryPerformance,
          change,
          severity: 'high',
          impact: 'Geological data retrieval performance degraded',
        });
      }
    }
    
    return geologicalRegressions;
  }

  calculatePercentageChange(baseline, current) {
    if (baseline === 0) return current === 0 ? 0 : 1;
    return (current - baseline) / baseline;
  }

  isRegression(metric, change) {
    // Define which metrics should decrease vs increase for better performance
    const lowerIsBetter = [
      'responseTime', 'latency', 'errorRate', 'memoryUsage', 'cpuUsage',
      'navigationTime', 'processingTime', 'loadTime', 'renderTime'
    ];
    
    const higherIsBetter = [
      'throughput', 'performance', 'successRate', 'cacheHitRatio',
      'fps', 'score', 'capacity'
    ];
    
    const metricLower = metric.toLowerCase();
    
    if (lowerIsBetter.some(term => metricLower.includes(term))) {
      return change > 0; // Increase is bad
    }
    
    if (higherIsBetter.some(term => metricLower.includes(term))) {
      return change < 0; // Decrease is bad
    }
    
    // Default: assume lower is better
    return change > 0;
  }

  isSignificantChange(metric, changePercent) {
    const metricLower = metric.toLowerCase();
    
    // Use metric-specific thresholds
    if (metricLower.includes('responsetime') || metricLower.includes('latency')) {
      return changePercent > this.config.regressionThresholds.responseTime;
    }
    
    if (metricLower.includes('throughput')) {
      return changePercent > this.config.regressionThresholds.throughput;
    }
    
    if (metricLower.includes('error')) {
      return changePercent > this.config.regressionThresholds.errorRate;
    }
    
    if (metricLower.includes('memory')) {
      return changePercent > this.config.regressionThresholds.memoryUsage;
    }
    
    if (metricLower.includes('cpu')) {
      return changePercent > this.config.regressionThresholds.cpuUsage;
    }
    
    // Default threshold
    return changePercent > 0.1; // 10%
  }

  calculateSeverity(metric, changePercent) {
    if (changePercent > 0.5) return 'critical';
    if (changePercent > 0.3) return 'high';
    if (changePercent > 0.15) return 'medium';
    return 'low';
  }

  assessImpact(category, metric, change) {
    const impacts = {
      k6: {
        responseTime: 'API response times may affect user experience',
        throughput: 'System capacity may be reduced',
        errorRate: 'System reliability may be compromised',
      },
      lighthouse: {
        performance: 'User interface performance may be degraded',
        accessibility: 'Application accessibility may be reduced',
      },
      database: {
        throughput: 'Database performance may affect data processing',
        latency: 'Data retrieval times may be increased',
      },
      memory: {
        memoryGrowthRate: 'Memory leaks may cause system instability',
        peakMemory: 'High memory usage may cause performance issues',
      },
      concurrent: {
        successRate: 'System may not handle concurrent users effectively',
        responseTime: 'Performance under load may be degraded',
      },
    };
    
    return impacts[category]?.[metric] || 'Performance may be affected';
  }

  async generateRegressionReport(analysis, testType) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    const report = {
      testType,
      timestamp: analysis.timestamp,
      summary: {
        hasRegression: analysis.hasRegression,
        hasSignificantRegression: analysis.hasSignificantRegression,
        totalRegressions: analysis.regressions.length,
        totalImprovements: analysis.improvements.length,
        criticalRegressions: analysis.regressions.filter(r => r.severity === 'critical').length,
        highRegressions: analysis.regressions.filter(r => r.severity === 'high').length,
      },
      regressions: analysis.regressions,
      improvements: analysis.improvements,
      recommendations: this.generateRegressionRecommendations(analysis),
      detailedAnalysis: analysis,
    };
    
    // Save report
    const reportPath = path.join(
      this.config.outputDir,
      `regression-report-${testType}-${timestamp}.json`
    );
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    // Generate HTML visualization
    await this.generateRegressionVisualization(report, timestamp);
    
    console.log(`📊 Regression report saved: ${reportPath}`);
    
    if (analysis.hasSignificantRegression) {
      console.log('🚨 SIGNIFICANT PERFORMANCE REGRESSION DETECTED!');
      analysis.regressions
        .filter(r => r.severity === 'critical' || r.severity === 'high')
        .forEach(regression => {
          console.log(`  ⚠️  ${regression.category}.${regression.metric}: ${(regression.change * 100).toFixed(1)}% regression`);
        });
    } else if (analysis.hasRegression) {
      console.log('⚠️  Minor performance regressions detected');
    } else {
      console.log('✅ No performance regressions detected');
    }
    
    if (analysis.improvements.length > 0) {
      console.log(`🎉 ${analysis.improvements.length} performance improvements detected!`);
    }
    
    return report;
  }

  generateRegressionRecommendations(analysis) {
    const recommendations = [];
    
    // Critical regressions
    const criticalRegressions = analysis.regressions.filter(r => r.severity === 'critical');
    if (criticalRegressions.length > 0) {
      recommendations.push({
        priority: 'critical',
        type: 'immediate_action',
        message: `${criticalRegressions.length} critical performance regressions detected. Immediate investigation required.`,
        actions: [
          'Review recent code changes',
          'Check system resource usage',
          'Validate database performance',
          'Consider rollback if necessary',
        ],
      });
    }
    
    // High regressions
    const highRegressions = analysis.regressions.filter(r => r.severity === 'high');
    if (highRegressions.length > 0) {
      recommendations.push({
        priority: 'high',
        type: 'performance_optimization',
        message: `${highRegressions.length} high-impact performance regressions detected.`,
        actions: [
          'Profile application performance',
          'Optimize database queries',
          'Review caching strategies',
          'Monitor resource usage patterns',
        ],
      });
    }
    
    // Geological-specific recommendations
    const geologicalRegressions = analysis.regressions.filter(r => r.category === 'geological');
    if (geologicalRegressions.length > 0) {
      recommendations.push({
        priority: 'high',
        type: 'geological_systems',
        message: 'Geological monitoring systems performance degraded.',
        actions: [
          'Check seismic data processing pipelines',
          'Validate volcanic alert system response times',
          'Review 3D visualization rendering performance',
          'Monitor real-time data streaming latency',
        ],
      });
    }
    
    // Memory-related regressions
    const memoryRegressions = analysis.regressions.filter(r => 
      r.metric.toLowerCase().includes('memory') || r.category === 'memory'
    );
    if (memoryRegressions.length > 0) {
      recommendations.push({
        priority: 'medium',
        type: 'memory_optimization',
        message: 'Memory usage regressions detected.',
        actions: [
          'Check for memory leaks in long-running visualizations',
          'Review garbage collection patterns',
          'Optimize data structures for geological datasets',
          'Implement memory monitoring alerts',
        ],
      });
    }
    
    return recommendations;
  }

  async updateBaselines(currentResults, existingBaselines, testType) {
    // Update baselines with current results using exponential moving average
    const alpha = 0.3; // Weight for new data
    const updatedMetrics = this.extractMetrics(currentResults);
    const baselineMetrics = existingBaselines.metrics;
    
    // Merge metrics using weighted average
    const mergedMetrics = this.mergeMetrics(baselineMetrics, updatedMetrics, alpha);
    
    const updatedBaseline = {
      ...existingBaselines,
      lastUpdated: new Date().toISOString(),
      updateCount: (existingBaselines.updateCount || 0) + 1,
      metrics: mergedMetrics,
    };
    
    const baselineFile = path.join(this.config.baselineDir, `${testType}-baseline.json`);
    await fs.writeFile(baselineFile, JSON.stringify(updatedBaseline, null, 2));
    
    console.log('📈 Baseline updated with current results');
  }

  mergeMetrics(baselineMetrics, currentMetrics, alpha) {
    const merged = JSON.parse(JSON.stringify(baselineMetrics)); // Deep copy
    
    Object.entries(currentMetrics).forEach(([category, categoryMetrics]) => {
      if (!merged[category]) {
        merged[category] = categoryMetrics;
        return;
      }
      
      Object.entries(categoryMetrics).forEach(([metric, value]) => {
        if (typeof value === 'number' && typeof merged[category][metric] === 'number') {
          // Exponential moving average
          merged[category][metric] = (1 - alpha) * merged[category][metric] + alpha * value;
        } else {
          merged[category][metric] = value;
        }
      });
    });
    
    return merged;
  }

  async generateRegressionVisualization(report, timestamp) {
    const htmlReport = this.generateRegressionHTMLReport(report);
    const reportPath = path.join(
      this.config.outputDir,
      `regression-visualization-${report.testType}-${timestamp}.html`
    );
    await fs.writeFile(reportPath, htmlReport);
    
    console.log(`📊 Regression visualization saved to ${reportPath}`);
  }

  generateRegressionHTMLReport(report) {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Performance Regression Report - ${report.testType}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .regression-${report.summary.hasSignificantRegression ? 'critical' : report.summary.hasRegression ? 'warning' : 'good'} {
          border-left: 5px solid ${report.summary.hasSignificantRegression ? '#dc3545' : report.summary.hasRegression ? '#ffc107' : '#28a745'};
        }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        .regression-item { 
          background-color: #fff5f5; 
          border: 1px solid #ff6b6b; 
          padding: 10px; 
          margin: 10px 0; 
          border-radius: 5px; 
        }
        .improvement-item { 
          background-color: #f5fff5; 
          border: 1px solid #6bcf7f; 
          padding: 10px; 
          margin: 10px 0; 
          border-radius: 5px; 
        }
        .severity-critical { border-left: 5px solid #dc3545; }
        .severity-high { border-left: 5px solid #fd7e14; }
        .severity-medium { border-left: 5px solid #ffc107; }
        .severity-low { border-left: 5px solid #20c997; }
        .recommendations { background-color: #d1ecf1; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🔍 Performance Regression Report</h1>
    <h2>${report.testType}</h2>
    
    <div class="summary regression-${report.summary.hasSignificantRegression ? 'critical' : report.summary.hasRegression ? 'warning' : 'good'}">
        <h3>Summary</h3>
        <p><strong>Status:</strong> ${
          report.summary.hasSignificantRegression ? '🚨 SIGNIFICANT REGRESSION' :
          report.summary.hasRegression ? '⚠️ MINOR REGRESSION' : '✅ NO REGRESSION'
        }</p>
        <p><strong>Total Regressions:</strong> ${report.summary.totalRegressions}</p>
        <p><strong>Critical Regressions:</strong> ${report.summary.criticalRegressions}</p>
        <p><strong>High Impact Regressions:</strong> ${report.summary.highRegressions}</p>
        <p><strong>Improvements:</strong> ${report.summary.totalImprovements}</p>
        <p><strong>Timestamp:</strong> ${new Date(report.timestamp).toLocaleString()}</p>
    </div>

    ${report.regressions.length > 0 ? `
    <h3>🚨 Performance Regressions</h3>
    ${report.regressions.map(regression => `
    <div class="regression-item severity-${regression.severity}">
        <h4>${regression.category}.${regression.metric}</h4>
        <p><strong>Change:</strong> ${(regression.change * 100).toFixed(1)}% regression</p>
        <p><strong>Baseline:</strong> ${regression.baselineValue}</p>
        <p><strong>Current:</strong> ${regression.currentValue}</p>
        <p><strong>Severity:</strong> ${regression.severity.toUpperCase()}</p>
        <p><strong>Impact:</strong> ${regression.impact}</p>
    </div>
    `).join('')}
    ` : ''}

    ${report.improvements.length > 0 ? `
    <h3>🎉 Performance Improvements</h3>
    ${report.improvements.map(improvement => `
    <div class="improvement-item">
        <h4>${improvement.category}.${improvement.metric}</h4>
        <p><strong>Improvement:</strong> ${(improvement.improvement * 100).toFixed(1)}%</p>
        <p><strong>Baseline:</strong> ${improvement.baselineValue}</p>
        <p><strong>Current:</strong> ${improvement.currentValue}</p>
    </div>
    `).join('')}
    ` : ''}

    ${report.recommendations.length > 0 ? `
    <div class="recommendations">
        <h3>💡 Recommendations</h3>
        ${report.recommendations.map(rec => `
        <div style="margin: 15px 0;">
            <h4>${rec.type.replace(/_/g, ' ').toUpperCase()} (${rec.priority})</h4>
            <p>${rec.message}</p>
            ${rec.actions ? `
            <ul>
                ${rec.actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
            ` : ''}
        </div>
        `).join('')}
    </div>
    ` : ''}

    <div class="chart-container">
        <canvas id="regression-chart"></canvas>
    </div>

    <script>
        // Regression Chart
        const regressionData = ${JSON.stringify(report.regressions.map(r => ({
          label: `${r.category}.${r.metric}`,
          change: r.change * 100,
          severity: r.severity
        })))};

        if (regressionData.length > 0) {
            new Chart(document.getElementById('regression-chart'), {
                type: 'bar',
                data: {
                    labels: regressionData.map(r => r.label),
                    datasets: [{
                        label: 'Performance Change (%)',
                        data: regressionData.map(r => r.change),
                        backgroundColor: regressionData.map(r => {
                            switch(r.severity) {
                                case 'critical': return 'rgba(220, 53, 69, 0.8)';
                                case 'high': return 'rgba(253, 126, 20, 0.8)';
                                case 'medium': return 'rgba(255, 193, 7, 0.8)';
                                default: return 'rgba(32, 201, 151, 0.8)';
                            }
                        }),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            beginAtZero: true,
                            title: { display: true, text: 'Performance Change (%)' }
                        },
                        x: { 
                            ticks: { maxRotation: 45 }
                        }
                    },
                    plugins: {
                        title: { display: true, text: 'Performance Regressions by Metric' },
                        legend: { display: false }
                    }
                }
            });
        }
    </script>
</body>
</html>`;
  }

  calculateAverage(values) {
    if (!Array.isArray(values) || values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  calculatePercentile(values, percentile) {
    if (!Array.isArray(values) || values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[Math.max(0, index)];
  }

  async ensureDirectories() {
    const dirs = [this.config.baselineDir, this.config.outputDir];
    
    for (const dir of dirs) {
      try {
        await fs.access(dir);
      } catch {
        await fs.mkdir(dir, { recursive: true });
      }
    }
  }
}

// CLI interface
if (require.main === module) {
  const config = {
    baselineDir: process.env.BASELINE_DIR || './tests/performance/baselines',
    outputDir: process.env.OUTPUT_DIR || './tests/performance/regression-reports',
  };

  const detector = new PerformanceRegressionDetector(config);
  
  // Example usage
  const exampleResults = {
    k6Results: {
      responseTimes: [100, 150, 200, 120, 180],
      throughput: 1000,
      errorRate: 0.02,
    },
  };
  
  detector.detectRegressions(exampleResults, 'example')
    .then(report => {
      console.log('Regression detection completed');
      process.exit(report.hasSignificantRegression ? 1 : 0);
    })
    .catch(error => {
      console.error('Regression detection failed:', error);
      process.exit(1);
    });
}

module.exports = PerformanceRegressionDetector;