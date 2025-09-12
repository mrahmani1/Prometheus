// Resource usage monitoring during peak loads for geological applications

class ResourceMonitor {
  constructor(options = {}) {
    this.options = {
      samplingInterval: options.samplingInterval || 2000, // 2 seconds
      maxSamples: options.maxSamples || 1800, // 1 hour of samples
      alertThresholds: {
        cpuUsage: options.cpuThreshold || 80, // 80%
        memoryUsage: options.memoryThreshold || 85, // 85%
        networkLatency: options.latencyThreshold || 2000, // 2 seconds
        diskUsage: options.diskThreshold || 90, // 90%
        ...options.alertThresholds,
      },
      ...options,
    };

    this.samples = [];
    this.alerts = [];
    this.isMonitoring = false;
    this.intervalId = null;
    this.startTime = null;
    this.performanceObserver = null;

    // Bind methods
    this.takeSample = this.takeSample.bind(this);
    this.handlePerformanceEntry = this.handlePerformanceEntry.bind(this);
  }

  start() {
    if (this.isMonitoring) {
      console.warn('Resource monitor is already running');
      return;
    }

    this.isMonitoring = true;
    this.startTime = Date.now();
    this.samples = [];
    this.alerts = [];

    // Set up performance observer for detailed metrics
    if (typeof PerformanceObserver !== 'undefined') {
      try {
        this.performanceObserver = new PerformanceObserver(this.handlePerformanceEntry);
        this.performanceObserver.observe({ 
          entryTypes: ['measure', 'navigation', 'resource', 'paint'] 
        });
      } catch (error) {
        console.warn('Performance Observer not available:', error.message);
      }
    }

    // Start resource sampling
    this.intervalId = setInterval(this.takeSample, this.options.samplingInterval);
    
    console.log('Resource monitor started');
    this.takeSample(); // Take initial sample
  }

  stop() {
    if (!this.isMonitoring) {
      return null;
    }

    this.isMonitoring = false;
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
      this.performanceObserver = null;
    }

    const report = this.generateReport();
    console.log('Resource monitor stopped');
    
    return report;
  }

  async takeSample() {
    if (!this.isMonitoring) return;

    const sample = {
      timestamp: Date.now(),
      cpu: await this.getCPUUsage(),
      memory: this.getMemoryUsage(),
      network: await this.getNetworkMetrics(),
      storage: await this.getStorageMetrics(),
      geological: await this.getGeologicalAppMetrics(),
      browser: this.getBrowserMetrics(),
    };

    this.samples.push(sample);

    // Keep only the most recent samples
    if (this.samples.length > this.options.maxSamples) {
      this.samples.shift();
    }

    // Check for resource alerts
    this.checkResourceAlerts(sample);
  }

  async getCPUUsage() {
    const cpuMetrics = {
      usage: 0,
      cores: navigator.hardwareConcurrency || 1,
      loadAverage: null,
    };

    // Browser environment - estimate CPU usage
    if (typeof performance !== 'undefined') {
      const start = performance.now();
      
      // Perform a small computational task to estimate CPU load
      let iterations = 0;
      const testStart = Date.now();
      while (Date.now() - testStart < 10) { // 10ms test
        iterations++;
      }
      
      const end = performance.now();
      const expectedIterations = 100000; // Baseline for idle CPU
      cpuMetrics.usage = Math.min(100, Math.max(0, 
        100 - (iterations / expectedIterations) * 100
      ));
    }

    // Node.js environment
    if (typeof process !== 'undefined' && process.cpuUsage) {
      const cpuUsage = process.cpuUsage();
      cpuMetrics.user = cpuUsage.user;
      cpuMetrics.system = cpuUsage.system;
      
      // Calculate usage percentage (approximation)
      if (this.lastCpuUsage) {
        const userDiff = cpuUsage.user - this.lastCpuUsage.user;
        const systemDiff = cpuUsage.system - this.lastCpuUsage.system;
        const totalDiff = userDiff + systemDiff;
        const timeDiff = this.options.samplingInterval * 1000; // microseconds
        
        cpuMetrics.usage = Math.min(100, (totalDiff / timeDiff) * 100);
      }
      
      this.lastCpuUsage = cpuUsage;
    }

    return cpuMetrics;
  }

  getMemoryUsage() {
    const memoryMetrics = {
      used: 0,
      total: 0,
      percentage: 0,
      heap: {},
    };

    // Browser environment
    if (typeof performance !== 'undefined' && performance.memory) {
      memoryMetrics.used = performance.memory.usedJSHeapSize;
      memoryMetrics.total = performance.memory.jsHeapSizeLimit;
      memoryMetrics.percentage = (memoryMetrics.used / memoryMetrics.total) * 100;
      
      memoryMetrics.heap = {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit,
      };
    }

    // Node.js environment
    if (typeof process !== 'undefined' && process.memoryUsage) {
      const nodeMemory = process.memoryUsage();
      memoryMetrics.used = nodeMemory.heapUsed;
      memoryMetrics.total = nodeMemory.heapTotal;
      memoryMetrics.percentage = (nodeMemory.heapUsed / nodeMemory.heapTotal) * 100;
      
      memoryMetrics.heap = {
        used: nodeMemory.heapUsed,
        total: nodeMemory.heapTotal,
        external: nodeMemory.external,
        rss: nodeMemory.rss,
      };
    }

    return memoryMetrics;
  }

  async getNetworkMetrics() {
    const networkMetrics = {
      latency: 0,
      throughput: 0,
      activeConnections: 0,
      errors: 0,
    };

    // Test network latency with a lightweight request
    try {
      const start = performance.now();
      const response = await fetch('/api/v1/health', { 
        method: 'HEAD',
        cache: 'no-cache',
      });
      const end = performance.now();
      
      networkMetrics.latency = end - start;
      networkMetrics.activeConnections = 1;
      
      if (!response.ok) {
        networkMetrics.errors = 1;
      }
    } catch (error) {
      networkMetrics.latency = 9999; // High latency for failed requests
      networkMetrics.errors = 1;
    }

    // Get connection info if available
    if (navigator.connection) {
      networkMetrics.connectionType = navigator.connection.effectiveType;
      networkMetrics.downlink = navigator.connection.downlink;
      networkMetrics.rtt = navigator.connection.rtt;
    }

    return networkMetrics;
  }

  async getStorageMetrics() {
    const storageMetrics = {
      used: 0,
      available: 0,
      percentage: 0,
      quota: 0,
    };

    // Browser storage API
    if (navigator.storage && navigator.storage.estimate) {
      try {
        const estimate = await navigator.storage.estimate();
        storageMetrics.used = estimate.usage || 0;
        storageMetrics.quota = estimate.quota || 0;
        storageMetrics.available = storageMetrics.quota - storageMetrics.used;
        storageMetrics.percentage = storageMetrics.quota > 0 
          ? (storageMetrics.used / storageMetrics.quota) * 100 
          : 0;
      } catch (error) {
        console.warn('Storage estimation not available:', error.message);
      }
    }

    return storageMetrics;
  }

  async getGeologicalAppMetrics() {
    const geologicalMetrics = {
      activeVisualizations: 0,
      dataStreams: 0,
      processingTasks: 0,
      webglContexts: 0,
      cacheSize: 0,
    };

    // Count active WebGL contexts (for 3D visualizations)
    if (typeof document !== 'undefined') {
      const canvases = document.querySelectorAll('canvas');
      canvases.forEach(canvas => {
        try {
          const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
          if (gl) {
            geologicalMetrics.webglContexts++;
            geologicalMetrics.activeVisualizations++;
          }
        } catch (error) {
          // Ignore context creation errors
        }
      });
    }

    // Check for active data streams (WebSocket connections)
    if (typeof window !== 'undefined' && window.geologicalDataStreams) {
      geologicalMetrics.dataStreams = window.geologicalDataStreams.length || 0;
    }

    // Check processing tasks
    if (typeof window !== 'undefined' && window.geologicalProcessingQueue) {
      geologicalMetrics.processingTasks = window.geologicalProcessingQueue.length || 0;
    }

    // Estimate cache size
    try {
      if (typeof localStorage !== 'undefined') {
        let cacheSize = 0;
        for (let key in localStorage) {
          if (key.startsWith('geological_') || key.startsWith('seismic_') || key.startsWith('volcanic_')) {
            cacheSize += localStorage[key].length;
          }
        }
        geologicalMetrics.cacheSize = cacheSize;
      }
    } catch (error) {
      // Ignore localStorage access errors
    }

    return geologicalMetrics;
  }

  getBrowserMetrics() {
    const browserMetrics = {
      userAgent: '',
      viewport: { width: 0, height: 0 },
      devicePixelRatio: 1,
      online: true,
      cookiesEnabled: false,
    };

    if (typeof navigator !== 'undefined') {
      browserMetrics.userAgent = navigator.userAgent;
      browserMetrics.online = navigator.onLine;
      browserMetrics.cookiesEnabled = navigator.cookieEnabled;
    }

    if (typeof window !== 'undefined') {
      browserMetrics.viewport = {
        width: window.innerWidth,
        height: window.innerHeight,
      };
      browserMetrics.devicePixelRatio = window.devicePixelRatio || 1;
    }

    return browserMetrics;
  }

  handlePerformanceEntry(list) {
    const entries = list.getEntries();
    entries.forEach(entry => {
      // Track resource loading performance
      if (entry.entryType === 'resource') {
        if (entry.name.includes('/api/v1/geological/') || 
            entry.name.includes('/api/v1/seismic/') ||
            entry.name.includes('/api/v1/volcanic/')) {
          
          const resourceMetric = {
            timestamp: Date.now(),
            type: 'geological_resource',
            name: entry.name,
            duration: entry.duration,
            transferSize: entry.transferSize,
            responseStart: entry.responseStart,
            responseEnd: entry.responseEnd,
          };
          
          // Add to recent performance entries
          if (!this.recentPerformanceEntries) {
            this.recentPerformanceEntries = [];
          }
          this.recentPerformanceEntries.push(resourceMetric);
          
          // Keep only recent entries (last 100)
          if (this.recentPerformanceEntries.length > 100) {
            this.recentPerformanceEntries.shift();
          }
        }
      }
    });
  }

  checkResourceAlerts(sample) {
    const alerts = [];

    // CPU usage alert
    if (sample.cpu.usage > this.options.alertThresholds.cpuUsage) {
      alerts.push({
        type: 'cpu_high',
        severity: 'warning',
        message: `High CPU usage: ${sample.cpu.usage.toFixed(1)}%`,
        value: sample.cpu.usage,
        threshold: this.options.alertThresholds.cpuUsage,
        timestamp: sample.timestamp,
      });
    }

    // Memory usage alert
    if (sample.memory.percentage > this.options.alertThresholds.memoryUsage) {
      alerts.push({
        type: 'memory_high',
        severity: 'warning',
        message: `High memory usage: ${sample.memory.percentage.toFixed(1)}%`,
        value: sample.memory.percentage,
        threshold: this.options.alertThresholds.memoryUsage,
        timestamp: sample.timestamp,
      });
    }

    // Network latency alert
    if (sample.network.latency > this.options.alertThresholds.networkLatency) {
      alerts.push({
        type: 'network_slow',
        severity: 'warning',
        message: `High network latency: ${sample.network.latency.toFixed(0)}ms`,
        value: sample.network.latency,
        threshold: this.options.alertThresholds.networkLatency,
        timestamp: sample.timestamp,
      });
    }

    // Storage usage alert
    if (sample.storage.percentage > this.options.alertThresholds.diskUsage) {
      alerts.push({
        type: 'storage_high',
        severity: 'warning',
        message: `High storage usage: ${sample.storage.percentage.toFixed(1)}%`,
        value: sample.storage.percentage,
        threshold: this.options.alertThresholds.diskUsage,
        timestamp: sample.timestamp,
      });
    }

    // Geological app specific alerts
    if (sample.geological.webglContexts > 5) {
      alerts.push({
        type: 'webgl_contexts_high',
        severity: 'info',
        message: `Many active WebGL contexts: ${sample.geological.webglContexts}`,
        value: sample.geological.webglContexts,
        threshold: 5,
        timestamp: sample.timestamp,
      });
    }

    // Add alerts to collection
    alerts.forEach(alert => {
      this.alerts.push(alert);
      console.warn(`Resource Alert [${alert.type}]:`, alert.message);
    });

    // Keep only recent alerts (last 100)
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(-100);
    }
  }

  generateReport() {
    if (this.samples.length === 0) {
      return { error: 'No samples collected' };
    }

    const totalTime = Date.now() - this.startTime;
    
    // Calculate statistics for each metric
    const stats = this.calculateStatistics();
    
    // Analyze performance trends
    const trends = this.analyzeTrends();
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(stats, trends);

    const report = {
      summary: {
        monitoringDuration: totalTime,
        samplesCollected: this.samples.length,
        alertsGenerated: this.alerts.length,
        samplingInterval: this.options.samplingInterval,
      },
      statistics: stats,
      trends: trends,
      alerts: this.alerts,
      recommendations: recommendations,
      performanceEntries: this.recentPerformanceEntries || [],
      rawData: this.samples,
    };

    return report;
  }

  calculateStatistics() {
    const stats = {
      cpu: this.calculateMetricStats(this.samples.map(s => s.cpu.usage)),
      memory: this.calculateMetricStats(this.samples.map(s => s.memory.percentage)),
      network: {
        latency: this.calculateMetricStats(this.samples.map(s => s.network.latency)),
        errors: this.samples.reduce((sum, s) => sum + s.network.errors, 0),
      },
      storage: this.calculateMetricStats(this.samples.map(s => s.storage.percentage)),
      geological: {
        avgVisualizations: this.calculateMetricStats(this.samples.map(s => s.geological.activeVisualizations)).avg,
        avgDataStreams: this.calculateMetricStats(this.samples.map(s => s.geological.dataStreams)).avg,
        avgWebGLContexts: this.calculateMetricStats(this.samples.map(s => s.geological.webglContexts)).avg,
        maxCacheSize: Math.max(...this.samples.map(s => s.geological.cacheSize)),
      },
    };

    return stats;
  }

  calculateMetricStats(values) {
    if (values.length === 0) return { min: 0, max: 0, avg: 0, p95: 0, p99: 0 };

    const sorted = values.slice().sort((a, b) => a - b);
    const sum = values.reduce((sum, val) => sum + val, 0);

    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      avg: sum / values.length,
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
    };
  }

  analyzeTrends() {
    const trends = {};

    // Analyze CPU trend
    trends.cpu = this.calculateTrend(this.samples.map(s => s.cpu.usage));
    trends.memory = this.calculateTrend(this.samples.map(s => s.memory.percentage));
    trends.networkLatency = this.calculateTrend(this.samples.map(s => s.network.latency));
    trends.geologicalLoad = this.calculateTrend(this.samples.map(s => 
      s.geological.activeVisualizations + s.geological.dataStreams
    ));

    return trends;
  }

  calculateTrend(values) {
    if (values.length < 2) return { direction: 'stable', slope: 0, confidence: 0 };

    // Simple linear regression
    const n = values.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * values[i], 0);
    const sumXX = x.reduce((sum, val) => sum + val * val, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Determine trend direction
    let direction = 'stable';
    if (Math.abs(slope) > 0.1) {
      direction = slope > 0 ? 'increasing' : 'decreasing';
    }

    // Calculate confidence (R-squared approximation)
    const yMean = sumY / n;
    const ssRes = values.reduce((sum, val, i) => {
      const predicted = slope * i + intercept;
      return sum + Math.pow(val - predicted, 2);
    }, 0);
    const ssTot = values.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
    const confidence = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;

    return { direction, slope, confidence: Math.max(0, confidence) };
  }

  generateRecommendations(stats, trends) {
    const recommendations = [];

    // CPU recommendations
    if (stats.cpu.avg > 70) {
      recommendations.push({
        type: 'cpu_optimization',
        severity: 'medium',
        message: 'High average CPU usage detected',
        suggestions: [
          'Optimize geological data processing algorithms',
          'Use Web Workers for heavy computational tasks',
          'Implement progressive loading for large datasets',
          'Consider reducing visualization quality during peak loads',
        ],
      });
    }

    // Memory recommendations
    if (stats.memory.avg > 75) {
      recommendations.push({
        type: 'memory_optimization',
        severity: 'medium',
        message: 'High memory usage detected',
        suggestions: [
          'Implement data streaming for large geological datasets',
          'Use object pooling for frequently created objects',
          'Clear unused visualization buffers',
          'Implement progressive mesh loading for 3D models',
        ],
      });
    }

    // Network recommendations
    if (stats.network.latency.avg > 1000) {
      recommendations.push({
        type: 'network_optimization',
        severity: 'medium',
        message: 'High network latency detected',
        suggestions: [
          'Implement data caching for frequently accessed geological data',
          'Use compression for large dataset transfers',
          'Implement adaptive quality based on network conditions',
          'Consider CDN for static geological resources',
        ],
      });
    }

    // Geological app specific recommendations
    if (stats.geological.avgWebGLContexts > 3) {
      recommendations.push({
        type: 'webgl_optimization',
        severity: 'low',
        message: 'Multiple WebGL contexts detected',
        suggestions: [
          'Consider sharing WebGL contexts between visualizations',
          'Implement context pooling for better resource management',
          'Use instanced rendering for similar geological features',
        ],
      });
    }

    // Trend-based recommendations
    if (trends.memory.direction === 'increasing' && trends.memory.confidence > 0.7) {
      recommendations.push({
        type: 'memory_leak_warning',
        severity: 'high',
        message: 'Memory usage is consistently increasing',
        suggestions: [
          'Investigate potential memory leaks in visualization components',
          'Review event listener cleanup in geological data handlers',
          'Check for unreleased WebGL resources',
        ],
      });
    }

    return recommendations;
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ResourceMonitor;
} else if (typeof window !== 'undefined') {
  window.ResourceMonitor = ResourceMonitor;
}

// Browser integration
if (typeof window !== 'undefined') {
  window.startResourceMonitoring = function(options = {}) {
    if (window.geologicalResourceMonitor) {
      console.warn('Resource monitoring already active');
      return;
    }

    window.geologicalResourceMonitor = new ResourceMonitor({
      samplingInterval: 5000, // 5 seconds for production monitoring
      ...options,
    });

    window.geologicalResourceMonitor.start();
    console.log('Geological resource monitoring started');
  };

  window.stopResourceMonitoring = function() {
    if (!window.geologicalResourceMonitor) {
      console.warn('No active resource monitoring');
      return null;
    }

    const report = window.geologicalResourceMonitor.stop();
    window.geologicalResourceMonitor = null;
    
    console.log('Resource monitoring report:', report);
    return report;
  };
}