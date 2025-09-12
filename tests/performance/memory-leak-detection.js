/**
 * Memory Leak Detection for Long-Running Geological Visualizations
 * Monitors memory usage patterns and detects potential leaks in 3D/4D visualizations
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

class MemoryLeakDetector {
  constructor(config = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:3000',
      testDuration: config.testDuration || 300000, // 5 minutes default
      samplingInterval: config.samplingInterval || 5000, // 5 seconds
      memoryThreshold: config.memoryThreshold || 100 * 1024 * 1024, // 100MB
      leakThreshold: config.leakThreshold || 0.1, // 10% growth threshold
      outputDir: config.outputDir || './tests/performance/memory-reports',
      ...config,
    };
    
    this.geologicalVisualizationTests = [
      {
        name: '3d-geological-model',
        url: '/visualization/3d-model',
        description: '3D geological model with WebGL rendering',
        interactions: [
          { action: 'rotate', interval: 2000 },
          { action: 'zoom', interval: 3000 },
          { action: 'layer-toggle', interval: 5000 },
        ],
        expectedMemoryPattern: 'stable-with-spikes',
      },
      {
        name: 'real-time-seismic',
        url: '/monitoring/seismic-realtime',
        description: 'Real-time seismic data visualization',
        interactions: [
          { action: 'data-stream', interval: 1000 },
          { action: 'filter-change', interval: 10000 },
          { action: 'time-window-adjust', interval: 15000 },
        ],
        expectedMemoryPattern: 'gradual-growth',
      },
      {
        name: '4d-temporal-analysis',
        url: '/analysis/4d-temporal',
        description: '4D temporal geological analysis',
        interactions: [
          { action: 'time-scrub', interval: 1500 },
          { action: 'animation-play', interval: 30000 },
          { action: 'data-layer-add', interval: 20000 },
        ],
        expectedMemoryPattern: 'cyclic-with-baseline-growth',
      },
      {
        name: 'volcanic-monitoring-dashboard',
        url: '/volcano/dashboard',
        description: 'Comprehensive volcanic monitoring dashboard',
        interactions: [
          { action: 'refresh-data', interval: 5000 },
          { action: 'switch-volcano', interval: 25000 },
          { action: 'alert-simulation', interval: 45000 },
        ],
        expectedMemoryPattern: 'stable-with-periodic-cleanup',
      },
    ];
  }

  async runMemoryLeakDetection() {
    console.log('🔍 Starting Memory Leak Detection for Geological Visualizations...');
    
    await this.ensureOutputDirectory();
    const results = [];
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    for (const test of this.geologicalVisualizationTests) {
      console.log(`\n📊 Testing: ${test.name} - ${test.description}`);
      
      try {
        const testResult = await this.runSingleMemoryTest(test);
        results.push(testResult);
        
        // Save individual test report
        const reportPath = path.join(
          this.config.outputDir,
          `memory-leak-${test.name}-${timestamp}.json`
        );
        await fs.writeFile(reportPath, JSON.stringify(testResult, null, 2));
        
        console.log(`✓ Memory test completed for ${test.name}`);
        console.log(`  Peak Memory: ${(testResult.peakMemory / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  Memory Growth: ${(testResult.memoryGrowthRate * 100).toFixed(2)}%`);
        console.log(`  Leak Detected: ${testResult.leakDetected ? '⚠️  YES' : '✅ NO'}`);
        
      } catch (error) {
        console.error(`❌ Failed to test ${test.name}:`, error.message);
        results.push({
          testName: test.name,
          error: error.message,
          leakDetected: null,
        });
      }
    }

    // Generate comprehensive report
    const summaryReport = this.generateMemoryLeakReport(results, timestamp);
    const summaryPath = path.join(
      this.config.outputDir,
      `memory-leak-summary-${timestamp}.json`
    );
    await fs.writeFile(summaryPath, JSON.stringify(summaryReport, null, 2));

    // Generate visualizations
    await this.generateMemoryVisualization(results, timestamp);

    return summaryReport;
  }

  async runSingleMemoryTest(test) {
    const browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--enable-webgl',
        '--enable-accelerated-2d-canvas',
      ],
    });

    const page = await browser.newPage();
    
    try {
      // Enable runtime domain for memory monitoring
      const client = await page.target().createCDPSession();
      await client.send('Runtime.enable');
      await client.send('Performance.enable');

      // Navigate to test page
      const fullUrl = `${this.config.baseUrl}${test.url}`;
      await page.goto(fullUrl, { waitUntil: 'networkidle0', timeout: 30000 });

      // Wait for initial load and WebGL initialization
      await page.waitForTimeout(5000);

      // Start memory monitoring
      const memoryData = [];
      const performanceData = [];
      const startTime = Date.now();
      const endTime = startTime + this.config.testDuration;

      // Set up interaction intervals
      const interactionTimers = this.setupInteractions(page, test.interactions);

      // Memory sampling loop
      while (Date.now() < endTime) {
        const timestamp = Date.now() - startTime;
        
        // Collect memory metrics
        const memoryMetrics = await this.collectMemoryMetrics(client, page);
        memoryData.push({ timestamp, ...memoryMetrics });

        // Collect performance metrics
        const performanceMetrics = await this.collectPerformanceMetrics(client);
        performanceData.push({ timestamp, ...performanceMetrics });

        // Check for memory warnings
        if (memoryMetrics.usedJSHeapSize > this.config.memoryThreshold) {
          console.warn(`⚠️  High memory usage detected: ${(memoryMetrics.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
        }

        await page.waitForTimeout(this.config.samplingInterval);
      }

      // Clean up interaction timers
      interactionTimers.forEach(timer => clearInterval(timer));

      // Analyze memory patterns
      const analysis = this.analyzeMemoryPattern(memoryData, performanceData, test);

      return {
        testName: test.name,
        description: test.description,
        url: test.url,
        duration: this.config.testDuration,
        samplingInterval: this.config.samplingInterval,
        memoryData,
        performanceData,
        analysis,
        ...analysis, // Flatten analysis results
      };

    } finally {
      await browser.close();
    }
  }

  setupInteractions(page, interactions) {
    const timers = [];

    interactions.forEach(interaction => {
      const timer = setInterval(async () => {
        try {
          await this.performInteraction(page, interaction.action);
        } catch (error) {
          console.warn(`Interaction ${interaction.action} failed:`, error.message);
        }
      }, interaction.interval);
      
      timers.push(timer);
    });

    return timers;
  }

  async performInteraction(page, action) {
    switch (action) {
      case 'rotate':
        // Simulate 3D model rotation
        await page.evaluate(() => {
          const canvas = document.querySelector('canvas');
          if (canvas) {
            const event = new MouseEvent('mousemove', {
              clientX: Math.random() * canvas.width,
              clientY: Math.random() * canvas.height,
              buttons: 1,
            });
            canvas.dispatchEvent(event);
          }
        });
        break;

      case 'zoom':
        // Simulate zoom interaction
        await page.evaluate(() => {
          const canvas = document.querySelector('canvas');
          if (canvas) {
            const event = new WheelEvent('wheel', {
              deltaY: (Math.random() - 0.5) * 100,
            });
            canvas.dispatchEvent(event);
          }
        });
        break;

      case 'layer-toggle':
        // Toggle geological layers
        await page.evaluate(() => {
          const layerButtons = document.querySelectorAll('[data-layer-toggle]');
          if (layerButtons.length > 0) {
            const randomButton = layerButtons[Math.floor(Math.random() * layerButtons.length)];
            randomButton.click();
          }
        });
        break;

      case 'data-stream':
        // Simulate real-time data updates
        await page.evaluate(() => {
          if (window.updateSeismicData) {
            window.updateSeismicData();
          }
        });
        break;

      case 'filter-change':
        // Change data filters
        await page.evaluate(() => {
          const filterSelects = document.querySelectorAll('select[data-filter]');
          if (filterSelects.length > 0) {
            const randomSelect = filterSelects[Math.floor(Math.random() * filterSelects.length)];
            const options = randomSelect.querySelectorAll('option');
            if (options.length > 1) {
              randomSelect.selectedIndex = Math.floor(Math.random() * options.length);
              randomSelect.dispatchEvent(new Event('change'));
            }
          }
        });
        break;

      case 'time-window-adjust':
        // Adjust time window for analysis
        await page.evaluate(() => {
          const timeSliders = document.querySelectorAll('input[type="range"][data-time]');
          if (timeSliders.length > 0) {
            const randomSlider = timeSliders[Math.floor(Math.random() * timeSliders.length)];
            randomSlider.value = Math.random() * (randomSlider.max - randomSlider.min) + parseFloat(randomSlider.min);
            randomSlider.dispatchEvent(new Event('input'));
          }
        });
        break;

      case 'time-scrub':
        // Scrub through temporal data
        await page.evaluate(() => {
          if (window.scrubToTime) {
            const randomTime = Math.random() * 86400000; // Random time in a day
            window.scrubToTime(randomTime);
          }
        });
        break;

      case 'animation-play':
        // Start/stop animations
        await page.evaluate(() => {
          const playButton = document.querySelector('[data-animation-play]');
          if (playButton) {
            playButton.click();
          }
        });
        break;

      case 'data-layer-add':
        // Add new data layers
        await page.evaluate(() => {
          if (window.addRandomDataLayer) {
            window.addRandomDataLayer();
          }
        });
        break;

      case 'refresh-data':
        // Refresh dashboard data
        await page.evaluate(() => {
          const refreshButton = document.querySelector('[data-refresh]');
          if (refreshButton) {
            refreshButton.click();
          }
        });
        break;

      case 'switch-volcano':
        // Switch between different volcanoes
        await page.evaluate(() => {
          const volcanoSelect = document.querySelector('select[data-volcano-select]');
          if (volcanoSelect && volcanoSelect.options.length > 1) {
            volcanoSelect.selectedIndex = Math.floor(Math.random() * volcanoSelect.options.length);
            volcanoSelect.dispatchEvent(new Event('change'));
          }
        });
        break;

      case 'alert-simulation':
        // Simulate alert conditions
        await page.evaluate(() => {
          if (window.simulateAlert) {
            window.simulateAlert();
          }
        });
        break;

      default:
        console.warn(`Unknown interaction: ${action}`);
    }
  }

  async collectMemoryMetrics(client, page) {
    // Get JavaScript heap usage
    const heapUsage = await client.send('Runtime.getHeapUsage');
    
    // Get performance memory info
    const memoryInfo = await page.evaluate(() => {
      if (performance.memory) {
        return {
          usedJSHeapSize: performance.memory.usedJSHeapSize,
          totalJSHeapSize: performance.memory.totalJSHeapSize,
          jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
        };
      }
      return null;
    });

    // Get WebGL memory info if available
    const webglMemory = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (canvas) {
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
          const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
          const memoryInfo = gl.getExtension('WEBGL_lose_context');
          return {
            renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown',
            vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'unknown',
            maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
            maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
          };
        }
      }
      return null;
    });

    return {
      heapUsed: heapUsage.usedSize,
      heapTotal: heapUsage.totalSize,
      usedJSHeapSize: memoryInfo?.usedJSHeapSize || 0,
      totalJSHeapSize: memoryInfo?.totalJSHeapSize || 0,
      jsHeapSizeLimit: memoryInfo?.jsHeapSizeLimit || 0,
      webglInfo: webglMemory,
    };
  }

  async collectPerformanceMetrics(client) {
    const metrics = await client.send('Performance.getMetrics');
    
    const metricsMap = {};
    metrics.metrics.forEach(metric => {
      metricsMap[metric.name] = metric.value;
    });

    return {
      timestamp: Date.now(),
      jsEventListeners: metricsMap.JSEventListeners || 0,
      documents: metricsMap.Documents || 0,
      domNodes: metricsMap.Nodes || 0,
      layoutCount: metricsMap.LayoutCount || 0,
      recalcStyleCount: metricsMap.RecalcStyleCount || 0,
      layoutDuration: metricsMap.LayoutDuration || 0,
      recalcStyleDuration: metricsMap.RecalcStyleDuration || 0,
    };
  }

  analyzeMemoryPattern(memoryData, performanceData, test) {
    if (memoryData.length < 2) {
      return { error: 'Insufficient data for analysis' };
    }

    const initialMemory = memoryData[0].usedJSHeapSize;
    const finalMemory = memoryData[memoryData.length - 1].usedJSHeapSize;
    const peakMemory = Math.max(...memoryData.map(d => d.usedJSHeapSize));
    const minMemory = Math.min(...memoryData.map(d => d.usedJSHeapSize));

    // Calculate memory growth rate
    const memoryGrowthRate = (finalMemory - initialMemory) / initialMemory;
    
    // Detect memory leaks
    const leakDetected = this.detectMemoryLeak(memoryData, test.expectedMemoryPattern);
    
    // Calculate memory volatility
    const memoryVolatility = this.calculateMemoryVolatility(memoryData);
    
    // Analyze garbage collection patterns
    const gcAnalysis = this.analyzeGarbageCollection(memoryData);
    
    // Performance impact analysis
    const performanceImpact = this.analyzePerformanceImpact(performanceData);

    return {
      initialMemory,
      finalMemory,
      peakMemory,
      minMemory,
      memoryGrowthRate,
      memoryVolatility,
      leakDetected,
      gcAnalysis,
      performanceImpact,
      recommendations: this.generateMemoryRecommendations({
        memoryGrowthRate,
        memoryVolatility,
        leakDetected,
        peakMemory,
        gcAnalysis,
      }),
    };
  }

  detectMemoryLeak(memoryData, expectedPattern) {
    const windowSize = Math.min(10, Math.floor(memoryData.length / 4));
    const windows = [];
    
    // Create sliding windows
    for (let i = 0; i <= memoryData.length - windowSize; i += windowSize) {
      const window = memoryData.slice(i, i + windowSize);
      const avgMemory = window.reduce((sum, d) => sum + d.usedJSHeapSize, 0) / window.length;
      windows.push(avgMemory);
    }

    if (windows.length < 3) return false;

    // Check for consistent upward trend
    let increasingWindows = 0;
    for (let i = 1; i < windows.length; i++) {
      if (windows[i] > windows[i - 1] * (1 + this.config.leakThreshold)) {
        increasingWindows++;
      }
    }

    // Pattern-specific leak detection
    switch (expectedPattern) {
      case 'stable-with-spikes':
        // Should return to baseline after spikes
        return increasingWindows > windows.length * 0.6;
      
      case 'gradual-growth':
        // Some growth is expected, but should stabilize
        return increasingWindows > windows.length * 0.8;
      
      case 'cyclic-with-baseline-growth':
        // Check if baseline is growing too much
        const baselines = this.extractBaselines(memoryData);
        return this.detectBaselineGrowth(baselines);
      
      case 'stable-with-periodic-cleanup':
        // Should have periodic drops
        return !this.detectPeriodicCleanup(memoryData);
      
      default:
        return increasingWindows > windows.length * 0.7;
    }
  }

  calculateMemoryVolatility(memoryData) {
    const values = memoryData.map(d => d.usedJSHeapSize);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance) / mean; // Coefficient of variation
  }

  analyzeGarbageCollection(memoryData) {
    const drops = [];
    const threshold = 0.05; // 5% drop threshold
    
    for (let i = 1; i < memoryData.length; i++) {
      const prev = memoryData[i - 1].usedJSHeapSize;
      const curr = memoryData[i].usedJSHeapSize;
      
      if (curr < prev * (1 - threshold)) {
        drops.push({
          timestamp: memoryData[i].timestamp,
          amount: prev - curr,
          percentage: (prev - curr) / prev,
        });
      }
    }

    return {
      gcEvents: drops.length,
      avgGcAmount: drops.length > 0 ? drops.reduce((sum, d) => sum + d.amount, 0) / drops.length : 0,
      avgGcPercentage: drops.length > 0 ? drops.reduce((sum, d) => sum + d.percentage, 0) / drops.length : 0,
      gcFrequency: drops.length / (memoryData.length * this.config.samplingInterval / 1000 / 60), // per minute
    };
  }

  analyzePerformanceImpact(performanceData) {
    if (performanceData.length < 2) return {};

    const domNodeGrowth = performanceData[performanceData.length - 1].domNodes - performanceData[0].domNodes;
    const layoutCountGrowth = performanceData[performanceData.length - 1].layoutCount - performanceData[0].layoutCount;
    const avgLayoutDuration = performanceData.reduce((sum, d) => sum + d.layoutDuration, 0) / performanceData.length;

    return {
      domNodeGrowth,
      layoutCountGrowth,
      avgLayoutDuration,
      performanceDegradation: domNodeGrowth > 1000 || avgLayoutDuration > 50,
    };
  }

  extractBaselines(memoryData) {
    // Simple baseline extraction - find local minima
    const baselines = [];
    for (let i = 1; i < memoryData.length - 1; i++) {
      if (memoryData[i].usedJSHeapSize < memoryData[i - 1].usedJSHeapSize &&
          memoryData[i].usedJSHeapSize < memoryData[i + 1].usedJSHeapSize) {
        baselines.push(memoryData[i].usedJSHeapSize);
      }
    }
    return baselines;
  }

  detectBaselineGrowth(baselines) {
    if (baselines.length < 3) return false;
    
    let growthCount = 0;
    for (let i = 1; i < baselines.length; i++) {
      if (baselines[i] > baselines[i - 1] * 1.1) { // 10% growth threshold
        growthCount++;
      }
    }
    
    return growthCount > baselines.length * 0.6;
  }

  detectPeriodicCleanup(memoryData) {
    const drops = [];
    const threshold = 0.1; // 10% drop threshold
    
    for (let i = 1; i < memoryData.length; i++) {
      const prev = memoryData[i - 1].usedJSHeapSize;
      const curr = memoryData[i].usedJSHeapSize;
      
      if (curr < prev * (1 - threshold)) {
        drops.push(memoryData[i].timestamp);
      }
    }
    
    // Check if drops occur regularly (at least every 2 minutes)
    const maxInterval = 2 * 60 * 1000; // 2 minutes
    let hasRegularCleanup = true;
    
    for (let i = 1; i < drops.length; i++) {
      if (drops[i] - drops[i - 1] > maxInterval) {
        hasRegularCleanup = false;
        break;
      }
    }
    
    return hasRegularCleanup && drops.length > 0;
  }

  generateMemoryRecommendations(analysis) {
    const recommendations = [];

    if (analysis.leakDetected) {
      recommendations.push({
        type: 'critical',
        message: 'Memory leak detected. Review event listeners, WebGL resources, and data structures for proper cleanup.',
        priority: 'high',
      });
    }

    if (analysis.memoryGrowthRate > 0.5) {
      recommendations.push({
        type: 'warning',
        message: 'High memory growth rate. Consider implementing object pooling and data pagination.',
        priority: 'medium',
      });
    }

    if (analysis.peakMemory > 200 * 1024 * 1024) { // 200MB
      recommendations.push({
        type: 'optimization',
        message: 'High peak memory usage. Consider level-of-detail (LOD) techniques for 3D models.',
        priority: 'medium',
      });
    }

    if (analysis.gcAnalysis.gcFrequency < 0.5) { // Less than 0.5 GC per minute
      recommendations.push({
        type: 'optimization',
        message: 'Infrequent garbage collection. Consider manual cleanup of large objects.',
        priority: 'low',
      });
    }

    if (analysis.memoryVolatility > 0.3) {
      recommendations.push({
        type: 'optimization',
        message: 'High memory volatility. Consider smoothing data updates and buffering.',
        priority: 'medium',
      });
    }

    return recommendations;
  }

  generateMemoryLeakReport(results, timestamp) {
    const summary = {
      timestamp,
      totalTests: results.length,
      leaksDetected: results.filter(r => r.leakDetected === true).length,
      testsWithErrors: results.filter(r => r.error).length,
      overallMemoryHealth: 'good',
      criticalIssues: [],
      recommendations: [],
    };

    // Analyze overall health
    const leakRate = summary.leaksDetected / summary.totalTests;
    if (leakRate > 0.5) {
      summary.overallMemoryHealth = 'critical';
    } else if (leakRate > 0.2) {
      summary.overallMemoryHealth = 'warning';
    }

    // Collect critical issues
    results.forEach(result => {
      if (result.leakDetected) {
        summary.criticalIssues.push({
          test: result.testName,
          issue: 'Memory leak detected',
          memoryGrowth: `${(result.memoryGrowthRate * 100).toFixed(1)}%`,
          peakMemory: `${(result.peakMemory / 1024 / 1024).toFixed(1)} MB`,
        });
      }
    });

    // Aggregate recommendations
    const allRecommendations = results
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

  async generateMemoryVisualization(results, timestamp) {
    // Generate HTML report with charts
    const htmlReport = this.generateHTMLReport(results);
    const reportPath = path.join(
      this.config.outputDir,
      `memory-leak-report-${timestamp}.html`
    );
    await fs.writeFile(reportPath, htmlReport);
    
    console.log(`📊 Memory leak visualization saved to ${reportPath}`);
  }

  generateHTMLReport(results) {
    const validResults = results.filter(r => !r.error && r.memoryData);
    
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Memory Leak Detection Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .test-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        .leak-detected { border-color: #ff4444; background-color: #fff5f5; }
        .no-leak { border-color: #44ff44; background-color: #f5fff5; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        .summary { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .recommendations { background-color: #fff8dc; padding: 15px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🔍 Memory Leak Detection Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> ${results.length}</p>
        <p><strong>Leaks Detected:</strong> ${results.filter(r => r.leakDetected).length}</p>
        <p><strong>Tests with Errors:</strong> ${results.filter(r => r.error).length}</p>
    </div>

    ${validResults.map((result, index) => `
    <div class="test-section ${result.leakDetected ? 'leak-detected' : 'no-leak'}">
        <h3>${result.testName} ${result.leakDetected ? '⚠️' : '✅'}</h3>
        <p><strong>Description:</strong> ${result.description}</p>
        <p><strong>Memory Growth:</strong> ${(result.memoryGrowthRate * 100).toFixed(2)}%</p>
        <p><strong>Peak Memory:</strong> ${(result.peakMemory / 1024 / 1024).toFixed(2)} MB</p>
        <p><strong>Leak Detected:</strong> ${result.leakDetected ? 'YES' : 'NO'}</p>
        
        <div class="chart-container">
            <canvas id="chart-${index}"></canvas>
        </div>
        
        ${result.recommendations && result.recommendations.length > 0 ? `
        <div class="recommendations">
            <h4>Recommendations:</h4>
            <ul>
                ${result.recommendations.map(rec => `<li><strong>${rec.type}:</strong> ${rec.message}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
    </div>
    `).join('')}

    <script>
        ${validResults.map((result, index) => `
        new Chart(document.getElementById('chart-${index}'), {
            type: 'line',
            data: {
                labels: ${JSON.stringify(result.memoryData.map(d => Math.round(d.timestamp / 1000)))},
                datasets: [{
                    label: 'Memory Usage (MB)',
                    data: ${JSON.stringify(result.memoryData.map(d => d.usedJSHeapSize / 1024 / 1024))},
                    borderColor: '${result.leakDetected ? '#ff4444' : '#44ff44'}',
                    backgroundColor: '${result.leakDetected ? 'rgba(255, 68, 68, 0.1)' : 'rgba(68, 255, 68, 0.1)'}',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Time (seconds)' } },
                    y: { title: { display: true, text: 'Memory Usage (MB)' } }
                },
                plugins: {
                    title: { display: true, text: '${result.testName} - Memory Usage Over Time' }
                }
            }
        });
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
    testDuration: parseInt(process.env.TEST_DURATION) || 300000, // 5 minutes
    samplingInterval: parseInt(process.env.SAMPLING_INTERVAL) || 5000, // 5 seconds
    outputDir: process.env.OUTPUT_DIR || './tests/performance/memory-reports',
  };

  const detector = new MemoryLeakDetector(config);
  
  detector.runMemoryLeakDetection()
    .then(report => {
      console.log('\n🎯 Memory Leak Detection Summary:');
      console.log(`Total tests: ${report.summary.totalTests}`);
      console.log(`Leaks detected: ${report.summary.leaksDetected}`);
      console.log(`Overall health: ${report.summary.overallMemoryHealth}`);
      
      if (report.summary.criticalIssues.length > 0) {
        console.log('\n⚠️  Critical Issues:');
        report.summary.criticalIssues.forEach(issue => {
          console.log(`  ${issue.test}: ${issue.issue} (${issue.memoryGrowth} growth, ${issue.peakMemory} peak)`);
        });
      }
      
      process.exit(report.summary.leaksDetected > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Memory leak detection failed:', error);
      process.exit(1);
    });
}

module.exports = MemoryLeakDetector;