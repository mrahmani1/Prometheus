const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const fs = require('fs').promises;
const path = require('path');

class LighthouseCIRunner {
  constructor(config = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:3000',
      outputDir: config.outputDir || './tests/performance/lighthouse-reports',
      thresholds: {
        performance: config.thresholds?.performance || 80,
        accessibility: config.thresholds?.accessibility || 90,
        'best-practices': config.thresholds?.['best-practices'] || 80,
        seo: config.thresholds?.seo || 80,
        pwa: config.thresholds?.pwa || 70,
      },
      ...config,
    };
    
    this.geologicalPages = [
      {
        name: 'dashboard',
        url: '/dashboard',
        description: 'Main geological monitoring dashboard',
        customThresholds: {
          performance: 85, // Higher threshold for critical monitoring
          'largest-contentful-paint': 2500,
          'cumulative-layout-shift': 0.1,
        },
      },
      {
        name: 'volcano-monitoring',
        url: '/volcano/monitor',
        description: 'Real-time volcanic monitoring interface',
        customThresholds: {
          performance: 80,
          'largest-contentful-paint': 3000, // Allow more time for 3D rendering
          'total-blocking-time': 300,
        },
      },
      {
        name: '3d-visualization',
        url: '/visualization/3d',
        description: '3D geological visualization',
        customThresholds: {
          performance: 75, // Lower threshold due to WebGL complexity
          'largest-contentful-paint': 4000,
          'max-potential-fid': 130,
        },
      },
      {
        name: 'seismic-analysis',
        url: '/seismic/analysis',
        description: 'Seismic data analysis interface',
        customThresholds: {
          performance: 80,
          'speed-index': 3000,
        },
      },
      {
        name: 'resource-assessment',
        url: '/resources/assessment',
        description: 'Resource prospectivity assessment',
        customThresholds: {
          performance: 78,
          'interactive': 3500,
        },
      },
    ];
  }

  async runAudit(url, options = {}) {
    const chrome = await chromeLauncher.launch({
      chromeFlags: ['--headless', '--no-sandbox', '--disable-gpu'],
    });

    const lighthouseOptions = {
      logLevel: 'info',
      output: 'json',
      onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo', 'pwa'],
      port: chrome.port,
      ...options,
    };

    const config = {
      extends: 'lighthouse:default',
      settings: {
        // Geological application specific settings
        throttlingMethod: 'simulate',
        throttling: {
          rttMs: 40,
          throughputKbps: 10240, // Simulate good connection for scientific work
          cpuSlowdownMultiplier: 1,
        },
        // Extended timeout for complex geological calculations
        maxWaitForLoad: 45000,
        // Custom audits for geological applications
        skipAudits: [
          'unused-javascript', // Scientific libraries may have unused code
          'unused-css-rules', // Visualization libraries may have unused CSS
        ],
      },
    };

    try {
      const runnerResult = await lighthouse(url, lighthouseOptions, config);
      await chrome.kill();
      return runnerResult;
    } catch (error) {
      await chrome.kill();
      throw error;
    }
  }

  async runGeologicalPerformanceAudit() {
    console.log('Starting Lighthouse CI for Geological Applications...');
    
    await this.ensureOutputDirectory();
    const results = [];
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    for (const page of this.geologicalPages) {
      console.log(`\nAuditing ${page.name}: ${page.description}`);
      
      try {
        const fullUrl = `${this.config.baseUrl}${page.url}`;
        const result = await this.runAudit(fullUrl);
        
        const auditResult = this.processAuditResult(result, page);
        results.push(auditResult);
        
        // Save individual report
        const reportPath = path.join(
          this.config.outputDir,
          `${page.name}-${timestamp}.json`
        );
        await fs.writeFile(reportPath, JSON.stringify(result.lhr, null, 2));
        
        console.log(`✓ Audit completed for ${page.name}`);
        console.log(`  Performance: ${auditResult.scores.performance}`);
        console.log(`  Accessibility: ${auditResult.scores.accessibility}`);
        console.log(`  Report saved: ${reportPath}`);
        
      } catch (error) {
        console.error(`✗ Failed to audit ${page.name}:`, error.message);
        results.push({
          page: page.name,
          error: error.message,
          passed: false,
        });
      }
    }

    // Generate summary report
    const summaryReport = this.generateSummaryReport(results, timestamp);
    const summaryPath = path.join(
      this.config.outputDir,
      `summary-${timestamp}.json`
    );
    await fs.writeFile(summaryPath, JSON.stringify(summaryReport, null, 2));

    // Generate performance regression detection
    await this.detectPerformanceRegression(results);

    return summaryReport;
  }

  processAuditResult(result, page) {
    const lhr = result.lhr;
    const scores = {
      performance: Math.round(lhr.categories.performance.score * 100),
      accessibility: Math.round(lhr.categories.accessibility.score * 100),
      'best-practices': Math.round(lhr.categories['best-practices'].score * 100),
      seo: Math.round(lhr.categories.seo.score * 100),
      pwa: Math.round(lhr.categories.pwa.score * 100),
    };

    // Check against thresholds
    const thresholds = { ...this.config.thresholds, ...page.customThresholds };
    const passed = Object.entries(scores).every(([category, score]) => {
      const threshold = thresholds[category];
      return !threshold || score >= threshold;
    });

    // Extract key metrics for geological applications
    const metrics = {
      'first-contentful-paint': lhr.audits['first-contentful-paint']?.numericValue,
      'largest-contentful-paint': lhr.audits['largest-contentful-paint']?.numericValue,
      'speed-index': lhr.audits['speed-index']?.numericValue,
      'cumulative-layout-shift': lhr.audits['cumulative-layout-shift']?.numericValue,
      'total-blocking-time': lhr.audits['total-blocking-time']?.numericValue,
      'interactive': lhr.audits['interactive']?.numericValue,
    };

    // Geological-specific performance analysis
    const geologicalMetrics = this.analyzeGeologicalPerformance(lhr, page);

    return {
      page: page.name,
      url: page.url,
      description: page.description,
      scores,
      metrics,
      geologicalMetrics,
      passed,
      thresholds,
      timestamp: new Date().toISOString(),
    };
  }

  analyzeGeologicalPerformance(lhr, page) {
    const metrics = {};

    // WebGL performance for 3D visualizations
    if (page.name.includes('3d') || page.name.includes('visualization')) {
      metrics.webglSupport = this.checkWebGLSupport(lhr);
      metrics.gpuAcceleration = this.checkGPUAcceleration(lhr);
    }

    // Real-time data performance
    if (page.name.includes('monitoring') || page.name.includes('seismic')) {
      metrics.websocketPerformance = this.analyzeWebSocketPerformance(lhr);
      metrics.dataUpdateLatency = this.analyzeDataUpdateLatency(lhr);
    }

    // Memory usage for large datasets
    metrics.memoryUsage = this.analyzeMemoryUsage(lhr);
    metrics.longTasksCount = lhr.audits['long-tasks']?.details?.items?.length || 0;

    // Network performance for geological data
    metrics.networkEfficiency = this.analyzeNetworkEfficiency(lhr);
    metrics.cacheEffectiveness = this.analyzeCacheEffectiveness(lhr);

    return metrics;
  }

  checkWebGLSupport(lhr) {
    // Check for WebGL-related resources and errors
    const networkRequests = lhr.audits['network-requests']?.details?.items || [];
    const webglResources = networkRequests.filter(req => 
      req.url.includes('webgl') || req.url.includes('three.js') || req.url.includes('babylon')
    );
    return {
      resourcesLoaded: webglResources.length,
      totalSize: webglResources.reduce((sum, req) => sum + (req.transferSize || 0), 0),
    };
  }

  checkGPUAcceleration(lhr) {
    // Analyze rendering performance indicators
    const paintMetrics = {
      firstPaint: lhr.audits['first-contentful-paint']?.numericValue,
      largestPaint: lhr.audits['largest-contentful-paint']?.numericValue,
    };
    return paintMetrics;
  }

  analyzeWebSocketPerformance(lhr) {
    // Analyze network requests for WebSocket connections
    const networkRequests = lhr.audits['network-requests']?.details?.items || [];
    const wsRequests = networkRequests.filter(req => 
      req.protocol === 'ws' || req.protocol === 'wss'
    );
    return {
      connectionCount: wsRequests.length,
      connectionTime: wsRequests.reduce((sum, req) => sum + (req.networkRequestTime || 0), 0),
    };
  }

  analyzeDataUpdateLatency(lhr) {
    // Analyze JavaScript execution time for data updates
    const jsExecution = lhr.audits['bootup-time']?.details?.items || [];
    const dataUpdateScripts = jsExecution.filter(item => 
      item.url.includes('data') || item.url.includes('update') || item.url.includes('stream')
    );
    return {
      executionTime: dataUpdateScripts.reduce((sum, item) => sum + (item.total || 0), 0),
      scriptCount: dataUpdateScripts.length,
    };
  }

  analyzeMemoryUsage(lhr) {
    // Analyze memory-related metrics
    return {
      domSize: lhr.audits['dom-size']?.numericValue,
      unusedJavaScript: lhr.audits['unused-javascript']?.details?.overallSavingsBytes || 0,
      unusedCSS: lhr.audits['unused-css-rules']?.details?.overallSavingsBytes || 0,
    };
  }

  analyzeNetworkEfficiency(lhr) {
    const networkRequests = lhr.audits['network-requests']?.details?.items || [];
    const totalBytes = networkRequests.reduce((sum, req) => sum + (req.transferSize || 0), 0);
    const requestCount = networkRequests.length;
    
    return {
      totalBytes,
      requestCount,
      averageRequestSize: requestCount > 0 ? totalBytes / requestCount : 0,
      compressionRatio: this.calculateCompressionRatio(networkRequests),
    };
  }

  calculateCompressionRatio(requests) {
    const compressedRequests = requests.filter(req => req.resourceSize && req.transferSize);
    if (compressedRequests.length === 0) return 0;
    
    const totalOriginal = compressedRequests.reduce((sum, req) => sum + req.resourceSize, 0);
    const totalTransferred = compressedRequests.reduce((sum, req) => sum + req.transferSize, 0);
    
    return totalOriginal > 0 ? (totalOriginal - totalTransferred) / totalOriginal : 0;
  }

  analyzeCacheEffectiveness(lhr) {
    const cacheAudit = lhr.audits['uses-long-cache-ttl'];
    return {
      cacheable: cacheAudit?.details?.items?.length || 0,
      potentialSavings: cacheAudit?.details?.overallSavingsBytes || 0,
    };
  }

  generateSummaryReport(results, timestamp) {
    const summary = {
      timestamp,
      totalPages: results.length,
      passedPages: results.filter(r => r.passed).length,
      failedPages: results.filter(r => !r.passed).length,
      averageScores: {},
      performanceRegression: false,
      recommendations: [],
    };

    // Calculate average scores
    const categories = ['performance', 'accessibility', 'best-practices', 'seo', 'pwa'];
    categories.forEach(category => {
      const validResults = results.filter(r => r.scores && r.scores[category]);
      if (validResults.length > 0) {
        summary.averageScores[category] = Math.round(
          validResults.reduce((sum, r) => sum + r.scores[category], 0) / validResults.length
        );
      }
    });

    // Generate recommendations
    summary.recommendations = this.generateRecommendations(results);

    return summary;
  }

  generateRecommendations(results) {
    const recommendations = [];

    results.forEach(result => {
      if (!result.passed && result.scores) {
        if (result.scores.performance < 80) {
          recommendations.push({
            page: result.page,
            type: 'performance',
            message: 'Consider optimizing JavaScript execution and reducing render-blocking resources',
            priority: 'high',
          });
        }

        if (result.geologicalMetrics?.memoryUsage?.domSize > 1500) {
          recommendations.push({
            page: result.page,
            type: 'memory',
            message: 'Large DOM size detected. Consider virtualizing large geological datasets',
            priority: 'medium',
          });
        }

        if (result.geologicalMetrics?.longTasksCount > 5) {
          recommendations.push({
            page: result.page,
            type: 'responsiveness',
            message: 'Multiple long tasks detected. Consider breaking up geological calculations',
            priority: 'high',
          });
        }
      }
    });

    return recommendations;
  }

  async detectPerformanceRegression(currentResults) {
    try {
      const previousReportPath = await this.findPreviousReport();
      if (!previousReportPath) {
        console.log('No previous report found for regression detection');
        return;
      }

      const previousReport = JSON.parse(await fs.readFile(previousReportPath, 'utf8'));
      const regression = this.comparePerformanceResults(previousReport, currentResults);
      
      if (regression.hasRegression) {
        console.warn('⚠️  Performance regression detected!');
        regression.regressions.forEach(reg => {
          console.warn(`  ${reg.page}: ${reg.metric} decreased by ${reg.change}%`);
        });
      } else {
        console.log('✓ No performance regression detected');
      }

      return regression;
    } catch (error) {
      console.error('Error detecting performance regression:', error.message);
    }
  }

  async findPreviousReport() {
    try {
      const files = await fs.readdir(this.config.outputDir);
      const summaryFiles = files
        .filter(f => f.startsWith('summary-') && f.endsWith('.json'))
        .sort()
        .reverse();
      
      return summaryFiles.length > 1 ? 
        path.join(this.config.outputDir, summaryFiles[1]) : null;
    } catch (error) {
      return null;
    }
  }

  comparePerformanceResults(previous, current) {
    const regressions = [];
    const threshold = 5; // 5% regression threshold

    current.forEach(currentResult => {
      const previousResult = previous.find(p => p.page === currentResult.page);
      if (!previousResult || !currentResult.scores) return;

      Object.entries(currentResult.scores).forEach(([metric, currentScore]) => {
        const previousScore = previousResult.scores?.[metric];
        if (previousScore && currentScore < previousScore) {
          const change = ((previousScore - currentScore) / previousScore) * 100;
          if (change > threshold) {
            regressions.push({
              page: currentResult.page,
              metric,
              previousScore,
              currentScore,
              change: Math.round(change),
            });
          }
        }
      });
    });

    return {
      hasRegression: regressions.length > 0,
      regressions,
      threshold,
    };
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
    outputDir: process.env.OUTPUT_DIR || './tests/performance/lighthouse-reports',
  };

  const runner = new LighthouseCIRunner(config);
  
  runner.runGeologicalPerformanceAudit()
    .then(summary => {
      console.log('\n📊 Lighthouse CI Summary:');
      console.log(`Total pages audited: ${summary.totalPages}`);
      console.log(`Passed: ${summary.passedPages}, Failed: ${summary.failedPages}`);
      console.log('Average scores:', summary.averageScores);
      
      if (summary.recommendations.length > 0) {
        console.log('\n💡 Recommendations:');
        summary.recommendations.forEach(rec => {
          console.log(`  ${rec.page}: ${rec.message} (${rec.priority})`);
        });
      }
      
      process.exit(summary.failedPages > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Lighthouse CI failed:', error);
      process.exit(1);
    });
}

module.exports = LighthouseCIRunner;