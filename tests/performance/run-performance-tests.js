#!/usr/bin/env node

// Performance test runner for Prometheus AI Platform
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

class PerformanceTestRunner {
  constructor(options = {}) {
    this.options = {
      baseUrl: options.baseUrl || 'http://localhost:8000',
      apiToken: options.apiToken || 'test-token',
      outputDir: options.outputDir || './test-results',
      parallel: options.parallel || false,
      verbose: options.verbose || false,
      ...options,
    };

    this.testSuites = [
      {
        name: 'API Load Tests',
        script: 'k6/api-load-tests.js',
        type: 'k6',
        scenarios: ['baseline_load', 'volcanic_stress', 'emergency_spike', 'monitoring_soak'],
      },
      {
        name: 'Database Performance Tests',
        script: 'k6/database-performance.js',
        type: 'k6',
        scenarios: ['large_dataset_stress', 'timeseries_ingestion', 'spatial_query_test'],
      },
      {
        name: 'Network Throttling Tests',
        script: 'k6/network-throttling-tests.js',
        type: 'k6',
        scenarios: ['slow_3g_network', 'fast_3g_network', 'satellite_connection', 'intermittent_network'],
      },
      {
        name: 'Concurrent Users Test',
        script: 'k6/concurrent-users-test.js',
        type: 'k6',
        scenarios: ['normal_operations', 'peak_research', 'emergency_response', 'collaborative_session'],
      },
      {
        name: 'Frontend Performance Tests',
        script: 'lighthouse/lighthouse-ci.json',
        type: 'lighthouse',
        urls: [
          '/',
          '/dashboard',
          '/volcanic-monitoring',
          '/seismic-analysis',
          '/3d-visualization',
          '/resource-assessment',
        ],
      },
    ];

    this.results = [];
  }

  async runAllTests() {
    console.log('🚀 Starting Prometheus AI Platform Performance Tests');
    console.log(`Base URL: ${this.options.baseUrl}`);
    console.log(`Output Directory: ${this.options.outputDir}`);
    console.log('=' .repeat(60));

    // Ensure output directory exists
    this.ensureOutputDirectory();

    // Run pre-test checks
    await this.preTestChecks();

    // Run test suites
    if (this.options.parallel) {
      await this.runTestsInParallel();
    } else {
      await this.runTestsSequentially();
    }

    // Generate comprehensive report
    await this.generateReport();

    console.log('✅ Performance testing completed');
  }

  ensureOutputDirectory() {
    if (!fs.existsSync(this.options.outputDir)) {
      fs.mkdirSync(this.options.outputDir, { recursive: true });
    }

    // Create subdirectories for different test types
    ['k6', 'lighthouse', 'memory', 'regression'].forEach(dir => {
      const fullPath = path.join(this.options.outputDir, dir);
      if (!fs.existsSync(fullPath)) {
        fs.mkdirSync(fullPath, { recursive: true });
      }
    });
  }

  async preTestChecks() {
    console.log('🔍 Running pre-test checks...');

    // Check if services are running
    try {
      const response = await fetch(`${this.options.baseUrl}/api/v1/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      console.log('✅ API health check passed');
    } catch (error) {
      console.error('❌ API health check failed:', error.message);
      console.log('Please ensure the Prometheus AI Platform is running');
      process.exit(1);
    }

    // Check required tools
    this.checkRequiredTools();
  }

  checkRequiredTools() {
    const tools = [
      { name: 'k6', command: 'k6 version' },
      { name: 'lighthouse', command: 'lighthouse --version' },
    ];

    tools.forEach(tool => {
      try {
        execSync(tool.command, { stdio: 'ignore' });
        console.log(`✅ ${tool.name} is available`);
      } catch (error) {
        console.error(`❌ ${tool.name} is not installed or not in PATH`);
        console.log(`Please install ${tool.name} to run performance tests`);
        process.exit(1);
      }
    });
  }

  async runTestsSequentially() {
    console.log('🔄 Running tests sequentially...');

    for (const suite of this.testSuites) {
      console.log(`\n📊 Running ${suite.name}...`);
      const result = await this.runTestSuite(suite);
      this.results.push(result);
    }
  }

  async runTestsInParallel() {
    console.log('🔄 Running tests in parallel...');

    const promises = this.testSuites.map(suite => this.runTestSuite(suite));
    this.results = await Promise.all(promises);
  }

  async runTestSuite(suite) {
    const startTime = Date.now();
    const result = {
      name: suite.name,
      type: suite.type,
      startTime,
      endTime: null,
      duration: null,
      success: false,
      output: '',
      error: null,
      metrics: {},
    };

    try {
      if (suite.type === 'k6') {
        result.output = await this.runK6Test(suite);
        result.metrics = this.parseK6Output(result.output);
      } else if (suite.type === 'lighthouse') {
        result.output = await this.runLighthouseTest(suite);
        result.metrics = this.parseLighthouseOutput(result.output);
      }

      result.success = true;
    } catch (error) {
      result.error = error.message;
      console.error(`❌ ${suite.name} failed:`, error.message);
    }

    result.endTime = Date.now();
    result.duration = result.endTime - result.startTime;

    if (result.success) {
      console.log(`✅ ${suite.name} completed in ${(result.duration / 1000).toFixed(1)}s`);
    }

    return result;
  }

  async runK6Test(suite) {
    const scriptPath = path.join(__dirname, suite.script);
    const outputFile = path.join(this.options.outputDir, 'k6', `${suite.name.replace(/\s+/g, '-').toLowerCase()}.json`);

    const env = {
      API_BASE_URL: this.options.baseUrl,
      API_TOKEN: this.options.apiToken,
      ...process.env,
    };

    const command = [
      'k6',
      'run',
      '--out', `json=${outputFile}`,
      '--summary-export', outputFile.replace('.json', '-summary.json'),
      scriptPath,
    ];

    if (this.options.verbose) {
      console.log(`Running: ${command.join(' ')}`);
    }

    return new Promise((resolve, reject) => {
      const process = spawn(command[0], command.slice(1), {
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
        if (this.options.verbose) {
          console.log(data.toString());
        }
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
        if (this.options.verbose) {
          console.error(data.toString());
        }
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`k6 process exited with code ${code}: ${stderr}`));
        }
      });
    });
  }

  async runLighthouseTest(suite) {
    const configPath = path.join(__dirname, suite.script);
    const outputDir = path.join(this.options.outputDir, 'lighthouse');

    const command = [
      'lhci',
      'autorun',
      '--config', configPath,
      '--upload.target', 'filesystem',
      '--upload.outputDir', outputDir,
    ];

    if (this.options.verbose) {
      console.log(`Running: ${command.join(' ')}`);
    }

    return new Promise((resolve, reject) => {
      const process = spawn(command[0], command.slice(1), {
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
        if (this.options.verbose) {
          console.log(data.toString());
        }
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
        if (this.options.verbose) {
          console.error(data.toString());
        }
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`Lighthouse CI process exited with code ${code}: ${stderr}`));
        }
      });
    });
  }

  parseK6Output(output) {
    const metrics = {};

    // Parse k6 summary output
    const lines = output.split('\n');
    lines.forEach(line => {
      // Parse metric lines like "http_req_duration..........: avg=123.45ms min=50ms med=120ms max=500ms p(90)=200ms p(95)=250ms"
      const metricMatch = line.match(/(\w+)\.+:\s*(.+)/);
      if (metricMatch) {
        const [, metricName, metricValues] = metricMatch;
        metrics[metricName] = this.parseK6MetricValues(metricValues);
      }
    });

    return metrics;
  }

  parseK6MetricValues(valueString) {
    const values = {};
    
    // Parse values like "avg=123.45ms min=50ms med=120ms max=500ms p(90)=200ms p(95)=250ms"
    const valueMatches = valueString.matchAll(/(\w+(?:\(\d+\))?)\s*=\s*([^\s]+)/g);
    for (const match of valueMatches) {
      const [, key, value] = match;
      values[key] = this.parseK6Value(value);
    }

    return values;
  }

  parseK6Value(value) {
    // Remove units and convert to number
    const numericValue = parseFloat(value.replace(/[^\d.-]/g, ''));
    return isNaN(numericValue) ? value : numericValue;
  }

  parseLighthouseOutput(output) {
    const metrics = {};

    // Parse Lighthouse output for key metrics
    const lines = output.split('\n');
    lines.forEach(line => {
      // Look for performance scores and metrics
      if (line.includes('Performance:')) {
        const scoreMatch = line.match(/Performance:\s*(\d+)/);
        if (scoreMatch) {
          metrics.performance_score = parseInt(scoreMatch[1]);
        }
      }
      
      if (line.includes('First Contentful Paint:')) {
        const fcp = line.match(/(\d+\.?\d*)\s*ms/);
        if (fcp) {
          metrics.first_contentful_paint = parseFloat(fcp[1]);
        }
      }
      
      if (line.includes('Largest Contentful Paint:')) {
        const lcp = line.match(/(\d+\.?\d*)\s*ms/);
        if (lcp) {
          metrics.largest_contentful_paint = parseFloat(lcp[1]);
        }
      }
    });

    return metrics;
  }

  async generateReport() {
    console.log('\n📋 Generating performance report...');

    const report = {
      timestamp: new Date().toISOString(),
      configuration: {
        baseUrl: this.options.baseUrl,
        outputDir: this.options.outputDir,
        parallel: this.options.parallel,
      },
      summary: {
        totalTests: this.results.length,
        successfulTests: this.results.filter(r => r.success).length,
        failedTests: this.results.filter(r => !r.success).length,
        totalDuration: this.results.reduce((sum, r) => sum + (r.duration || 0), 0),
      },
      results: this.results,
      recommendations: this.generateRecommendations(),
    };

    // Save report
    const reportPath = path.join(this.options.outputDir, 'performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    // Generate HTML report
    await this.generateHtmlReport(report);

    console.log(`📄 Report saved to: ${reportPath}`);
    
    // Print summary
    this.printSummary(report);
  }

  generateRecommendations() {
    const recommendations = [];

    this.results.forEach(result => {
      if (!result.success) {
        recommendations.push({
          type: 'error',
          test: result.name,
          message: `Test failed: ${result.error}`,
          priority: 'high',
        });
        return;
      }

      // Analyze metrics for recommendations
      if (result.type === 'k6' && result.metrics) {
        // Check response times
        if (result.metrics.http_req_duration && result.metrics.http_req_duration.avg > 2000) {
          recommendations.push({
            type: 'performance',
            test: result.name,
            message: 'High average response time detected',
            value: `${result.metrics.http_req_duration.avg}ms`,
            threshold: '2000ms',
            priority: 'medium',
          });
        }

        // Check error rates
        if (result.metrics.http_req_failed && result.metrics.http_req_failed.rate > 0.05) {
          recommendations.push({
            type: 'reliability',
            test: result.name,
            message: 'High error rate detected',
            value: `${(result.metrics.http_req_failed.rate * 100).toFixed(2)}%`,
            threshold: '5%',
            priority: 'high',
          });
        }
      }

      if (result.type === 'lighthouse' && result.metrics) {
        // Check performance score
        if (result.metrics.performance_score < 80) {
          recommendations.push({
            type: 'frontend_performance',
            test: result.name,
            message: 'Low Lighthouse performance score',
            value: result.metrics.performance_score,
            threshold: '80',
            priority: 'medium',
          });
        }
      }
    });

    return recommendations;
  }

  async generateHtmlReport(report) {
    const htmlTemplate = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prometheus AI Platform - Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card h3 { margin: 0 0 10px 0; color: #333; }
        .summary-card .value { font-size: 2em; font-weight: bold; color: #007bff; }
        .test-results { margin-bottom: 30px; }
        .test-result { margin-bottom: 20px; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }
        .test-result.success { border-left-color: #28a745; background: #f8fff9; }
        .test-result.failure { border-left-color: #dc3545; background: #fff8f8; }
        .recommendations { margin-top: 30px; }
        .recommendation { padding: 15px; margin-bottom: 10px; border-radius: 8px; }
        .recommendation.high { background: #fff3cd; border-left: 4px solid #ffc107; }
        .recommendation.medium { background: #d1ecf1; border-left: 4px solid #17a2b8; }
        .recommendation.low { background: #d4edda; border-left: 4px solid #28a745; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; }
        .metric { background: #f8f9fa; padding: 10px; border-radius: 4px; text-align: center; }
        .metric .label { font-size: 0.8em; color: #666; }
        .metric .value { font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Prometheus AI Platform</h1>
            <h2>Performance Test Report</h2>
            <p>Generated on ${new Date(report.timestamp).toLocaleString()}</p>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">${report.summary.totalTests}</div>
            </div>
            <div class="summary-card">
                <h3>Successful</h3>
                <div class="value" style="color: #28a745;">${report.summary.successfulTests}</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="value" style="color: #dc3545;">${report.summary.failedTests}</div>
            </div>
            <div class="summary-card">
                <h3>Duration</h3>
                <div class="value">${(report.summary.totalDuration / 1000 / 60).toFixed(1)}m</div>
            </div>
        </div>

        <div class="test-results">
            <h3>Test Results</h3>
            ${report.results.map(result => `
                <div class="test-result ${result.success ? 'success' : 'failure'}">
                    <h4>${result.name} ${result.success ? '✅' : '❌'}</h4>
                    <p><strong>Duration:</strong> ${(result.duration / 1000).toFixed(1)}s</p>
                    ${result.error ? `<p><strong>Error:</strong> ${result.error}</p>` : ''}
                    ${result.metrics && Object.keys(result.metrics).length > 0 ? `
                        <div class="metrics">
                            ${Object.entries(result.metrics).map(([key, value]) => `
                                <div class="metric">
                                    <div class="label">${key.replace(/_/g, ' ')}</div>
                                    <div class="value">${typeof value === 'object' ? JSON.stringify(value) : value}</div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        </div>

        ${report.recommendations.length > 0 ? `
            <div class="recommendations">
                <h3>Recommendations</h3>
                ${report.recommendations.map(rec => `
                    <div class="recommendation ${rec.priority}">
                        <strong>${rec.test}:</strong> ${rec.message}
                        ${rec.value ? `<br><small>Current: ${rec.value} | Threshold: ${rec.threshold}</small>` : ''}
                    </div>
                `).join('')}
            </div>
        ` : ''}
    </div>
</body>
</html>
    `;

    const htmlPath = path.join(this.options.outputDir, 'performance-report.html');
    fs.writeFileSync(htmlPath, htmlTemplate);
    console.log(`📄 HTML report saved to: ${htmlPath}`);
  }

  printSummary(report) {
    console.log('\n' + '='.repeat(60));
    console.log('📊 PERFORMANCE TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total Tests: ${report.summary.totalTests}`);
    console.log(`Successful: ${report.summary.successfulTests} ✅`);
    console.log(`Failed: ${report.summary.failedTests} ${report.summary.failedTests > 0 ? '❌' : '✅'}`);
    console.log(`Total Duration: ${(report.summary.totalDuration / 1000 / 60).toFixed(1)} minutes`);
    
    if (report.recommendations.length > 0) {
      console.log(`\n⚠️  ${report.recommendations.length} recommendations generated`);
      console.log('See the full report for details.');
    } else {
      console.log('\n✅ No performance issues detected');
    }
    
    console.log('='.repeat(60));
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};

  // Parse command line arguments
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    if (key === 'parallel') {
      options.parallel = true;
      i--; // No value for this flag
    } else if (key === 'verbose') {
      options.verbose = true;
      i--; // No value for this flag
    } else {
      options[key] = value;
    }
  }

  const runner = new PerformanceTestRunner(options);
  runner.runAllTests().catch(error => {
    console.error('❌ Performance test runner failed:', error);
    process.exit(1);
  });
}

module.exports = PerformanceTestRunner;