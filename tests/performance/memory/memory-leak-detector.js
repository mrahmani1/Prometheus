// Memory leak detection for long-running geological visualizations
// This script monitors memory usage patterns and detects potential leaks

class MemoryLeakDetector {
  constructor(options = {}) {
    this.options = {
      samplingInterval: options.samplingInterval || 5000, // 5 seconds
      maxSamples: options.maxSamples || 1000,
      leakThreshold: options.leakThreshold || 10 * 1024 * 1024, // 10MB growth
      gcThreshold: options.gcThreshold || 50, // Max GC events per minute
      ...options,
    };
    
    this.samples = [];
    this.gcEvents = [];
    this.isMonitoring = false;
    this.intervalId = null;
    this.startTime = null;
    
    // Bind methods
    this.handleGC = this.handleGC.bind(this);
    this.takeSample = this.takeSample.bind(this);
  }

  start() {
    if (this.isMonitoring) {
      console.warn('Memory leak detector is already running');
      return;
    }

    this.isMonitoring = true;
    this.startTime = Date.now();
    this.samples = [];
    this.gcEvents = [];

    // Set up performance observer for GC events
    if (typeof PerformanceObserver !== 'undefined') {
      try {
        this.gcObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach(entry => {
            if (entry.entryType === 'measure' && entry.name.includes('gc')) {
              this.handleGC(entry);
            }
          });
        });
        this.gcObserver.observe({ entryTypes: ['measure'] });
      } catch (error) {
        console.warn('GC monitoring not available:', error.message);
      }
    }

    // Start memory sampling
    this.intervalId = setInterval(this.takeSample, this.options.samplingInterval);
    
    console.log('Memory leak detector started');
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

    if (this.gcObserver) {
      this.gcObserver.disconnect();
      this.gcObserver = null;
    }

    const report = this.generateReport();
    console.log('Memory leak detector stopped');
    
    return report;
  }

  takeSample() {
    if (!this.isMonitoring) return;

    const sample = {
      timestamp: Date.now(),
      memory: this.getMemoryUsage(),
      performance: this.getPerformanceMetrics(),
    };

    this.samples.push(sample);

    // Keep only the most recent samples
    if (this.samples.length > this.options.maxSamples) {
      this.samples.shift();
    }

    // Check for immediate issues
    this.checkForLeaks(sample);
  }

  getMemoryUsage() {
    const memory = {
      used: 0,
      total: 0,
      external: 0,
    };

    // Browser environment
    if (typeof performance !== 'undefined' && performance.memory) {
      memory.used = performance.memory.usedJSHeapSize;
      memory.total = performance.memory.totalJSHeapSize;
      memory.limit = performance.memory.jsHeapSizeLimit;
    }
    
    // Node.js environment
    if (typeof process !== 'undefined' && process.memoryUsage) {
      const nodeMemory = process.memoryUsage();
      memory.used = nodeMemory.heapUsed;
      memory.total = nodeMemory.heapTotal;
      memory.external = nodeMemory.external;
      memory.rss = nodeMemory.rss;
    }

    return memory;
  }

  getPerformanceMetrics() {
    const metrics = {
      timestamp: Date.now(),
    };

    // Get performance timing if available
    if (typeof performance !== 'undefined') {
      if (performance.now) {
        metrics.performanceNow = performance.now();
      }
      
      if (performance.getEntriesByType) {
        // Get navigation timing
        const navigation = performance.getEntriesByType('navigation')[0];
        if (navigation) {
          metrics.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart;
          metrics.loadComplete = navigation.loadEventEnd - navigation.loadEventStart;
        }

        // Get resource timing for geological data
        const resources = performance.getEntriesByType('resource');
        const geologicalResources = resources.filter(resource => 
          resource.name.includes('/api/v1/geological/') ||
          resource.name.includes('/api/v1/seismic/') ||
          resource.name.includes('/api/v1/volcanic/')
        );
        
        if (geologicalResources.length > 0) {
          metrics.avgResourceLoadTime = geologicalResources.reduce((sum, resource) => 
            sum + (resource.responseEnd - resource.requestStart), 0) / geologicalResources.length;
        }
      }
    }

    return metrics;
  }

  handleGC(entry) {
    const gcEvent = {
      timestamp: Date.now(),
      duration: entry.duration,
      type: entry.name,
      startTime: entry.startTime,
    };

    this.gcEvents.push(gcEvent);

    // Keep only recent GC events (last hour)
    const oneHourAgo = Date.now() - 60 * 60 * 1000;
    this.gcEvents = this.gcEvents.filter(event => event.timestamp > oneHourAgo);
  }

  checkForLeaks(currentSample) {
    if (this.samples.length < 10) return; // Need enough samples

    const recentSamples = this.samples.slice(-10);
    const oldestSample = recentSamples[0];
    const memoryGrowth = currentSample.memory.used - oldestSample.memory.used;
    const timeSpan = currentSample.timestamp - oldestSample.timestamp;

    // Check for sustained memory growth
    if (memoryGrowth > this.options.leakThreshold) {
      const growthRate = memoryGrowth / (timeSpan / 1000); // bytes per second
      
      console.warn(`Potential memory leak detected:`, {
        memoryGrowth: `${(memoryGrowth / 1024 / 1024).toFixed(2)} MB`,
        growthRate: `${(growthRate / 1024).toFixed(2)} KB/s`,
        timeSpan: `${(timeSpan / 1000).toFixed(1)} seconds`,
        currentUsage: `${(currentSample.memory.used / 1024 / 1024).toFixed(2)} MB`,
      });
    }

    // Check GC frequency
    const recentGCEvents = this.gcEvents.filter(event => 
      event.timestamp > Date.now() - 60000 // Last minute
    );
    
    if (recentGCEvents.length > this.options.gcThreshold) {
      console.warn(`High GC frequency detected: ${recentGCEvents.length} events in last minute`);
    }
  }

  generateReport() {
    if (this.samples.length === 0) {
      return { error: 'No samples collected' };
    }

    const firstSample = this.samples[0];
    const lastSample = this.samples[this.samples.length - 1];
    const totalTime = lastSample.timestamp - firstSample.timestamp;
    const totalMemoryGrowth = lastSample.memory.used - firstSample.memory.used;

    // Calculate memory statistics
    const memoryUsages = this.samples.map(s => s.memory.used);
    const maxMemory = Math.max(...memoryUsages);
    const minMemory = Math.min(...memoryUsages);
    const avgMemory = memoryUsages.reduce((sum, usage) => sum + usage, 0) / memoryUsages.length;

    // Detect memory leak patterns
    const leakDetection = this.analyzeLeakPatterns();

    // GC analysis
    const gcAnalysis = this.analyzeGCPatterns();

    const report = {
      summary: {
        monitoringDuration: totalTime,
        samplesCollected: this.samples.length,
        totalMemoryGrowth: totalMemoryGrowth,
        growthRate: totalMemoryGrowth / (totalTime / 1000), // bytes per second
      },
      memory: {
        initial: firstSample.memory.used,
        final: lastSample.memory.used,
        peak: maxMemory,
        minimum: minMemory,
        average: avgMemory,
        growth: totalMemoryGrowth,
        growthPercentage: ((totalMemoryGrowth / firstSample.memory.used) * 100),
      },
      leakDetection,
      gcAnalysis,
      recommendations: this.generateRecommendations(leakDetection, gcAnalysis),
      rawData: {
        samples: this.samples,
        gcEvents: this.gcEvents,
      },
    };

    return report;
  }

  analyzeLeakPatterns() {
    const analysis = {
      hasLeak: false,
      confidence: 0,
      patterns: [],
      growthTrend: 'stable',
    };

    if (this.samples.length < 20) {
      analysis.confidence = 0;
      return analysis;
    }

    // Analyze memory growth trend
    const memoryValues = this.samples.map(s => s.memory.used);
    const timeValues = this.samples.map(s => s.timestamp);

    // Simple linear regression to detect trend
    const n = memoryValues.length;
    const sumX = timeValues.reduce((sum, t) => sum + t, 0);
    const sumY = memoryValues.reduce((sum, m) => sum + m, 0);
    const sumXY = timeValues.reduce((sum, t, i) => sum + t * memoryValues[i], 0);
    const sumXX = timeValues.reduce((sum, t) => sum + t * t, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Determine growth trend
    if (slope > 1000) { // Growing more than 1KB per millisecond
      analysis.growthTrend = 'increasing';
      analysis.hasLeak = true;
      analysis.confidence = Math.min(0.9, slope / 10000);
    } else if (slope < -1000) {
      analysis.growthTrend = 'decreasing';
    } else {
      analysis.growthTrend = 'stable';
    }

    // Look for sawtooth patterns (memory grows then drops due to GC)
    const sawtoothPattern = this.detectSawtoothPattern(memoryValues);
    if (sawtoothPattern.detected) {
      analysis.patterns.push('sawtooth');
      analysis.hasLeak = sawtoothPattern.growingBaseline;
      analysis.confidence = Math.max(analysis.confidence, sawtoothPattern.confidence);
    }

    // Look for step increases (sudden memory jumps)
    const stepPattern = this.detectStepPattern(memoryValues);
    if (stepPattern.detected) {
      analysis.patterns.push('step_increases');
      analysis.hasLeak = true;
      analysis.confidence = Math.max(analysis.confidence, stepPattern.confidence);
    }

    return analysis;
  }

  detectSawtoothPattern(memoryValues) {
    const result = {
      detected: false,
      growingBaseline: false,
      confidence: 0,
    };

    // Find local maxima and minima
    const peaks = [];
    const valleys = [];

    for (let i = 1; i < memoryValues.length - 1; i++) {
      if (memoryValues[i] > memoryValues[i-1] && memoryValues[i] > memoryValues[i+1]) {
        peaks.push({ index: i, value: memoryValues[i] });
      }
      if (memoryValues[i] < memoryValues[i-1] && memoryValues[i] < memoryValues[i+1]) {
        valleys.push({ index: i, value: memoryValues[i] });
      }
    }

    if (peaks.length >= 3 && valleys.length >= 3) {
      result.detected = true;

      // Check if baseline is growing
      const firstValley = valleys[0].value;
      const lastValley = valleys[valleys.length - 1].value;
      const baselineGrowth = lastValley - firstValley;

      if (baselineGrowth > firstValley * 0.1) { // 10% growth in baseline
        result.growingBaseline = true;
        result.confidence = Math.min(0.8, baselineGrowth / (firstValley * 0.5));
      }
    }

    return result;
  }

  detectStepPattern(memoryValues) {
    const result = {
      detected: false,
      confidence: 0,
      steps: [],
    };

    const threshold = memoryValues[0] * 0.05; // 5% of initial memory

    for (let i = 1; i < memoryValues.length; i++) {
      const increase = memoryValues[i] - memoryValues[i-1];
      if (increase > threshold) {
        result.steps.push({
          index: i,
          increase: increase,
          percentage: (increase / memoryValues[i-1]) * 100,
        });
      }
    }

    if (result.steps.length >= 3) {
      result.detected = true;
      result.confidence = Math.min(0.9, result.steps.length / 10);
    }

    return result;
  }

  analyzeGCPatterns() {
    const analysis = {
      frequency: 0,
      averageDuration: 0,
      totalTime: 0,
      efficiency: 'good',
    };

    if (this.gcEvents.length === 0) {
      return analysis;
    }

    const totalMonitoringTime = Date.now() - this.startTime;
    analysis.frequency = (this.gcEvents.length / totalMonitoringTime) * 60000; // events per minute

    const durations = this.gcEvents.map(event => event.duration);
    analysis.averageDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length;
    analysis.totalTime = durations.reduce((sum, d) => sum + d, 0);

    // Determine GC efficiency
    if (analysis.frequency > 10) {
      analysis.efficiency = 'poor';
    } else if (analysis.frequency > 5) {
      analysis.efficiency = 'fair';
    } else {
      analysis.efficiency = 'good';
    }

    return analysis;
  }

  generateRecommendations(leakDetection, gcAnalysis) {
    const recommendations = [];

    if (leakDetection.hasLeak) {
      recommendations.push({
        type: 'memory_leak',
        severity: 'high',
        message: 'Potential memory leak detected. Consider reviewing event listeners, closures, and object references.',
        actions: [
          'Check for unremoved event listeners on geological visualization components',
          'Review WebGL context and buffer management',
          'Ensure proper cleanup of real-time data subscriptions',
          'Verify that large geological datasets are properly disposed',
        ],
      });
    }

    if (gcAnalysis.frequency > 10) {
      recommendations.push({
        type: 'gc_pressure',
        severity: 'medium',
        message: 'High garbage collection frequency detected. Consider optimizing object allocation.',
        actions: [
          'Reduce object creation in animation loops',
          'Use object pooling for frequently created geological data objects',
          'Optimize real-time data processing to reduce temporary objects',
          'Consider using TypedArrays for large numerical datasets',
        ],
      });
    }

    if (gcAnalysis.averageDuration > 50) {
      recommendations.push({
        type: 'gc_duration',
        severity: 'medium',
        message: 'Long garbage collection pauses detected. This may affect real-time visualization performance.',
        actions: [
          'Reduce heap size by optimizing data structures',
          'Implement incremental data loading for large geological datasets',
          'Use Web Workers for heavy computational tasks',
          'Consider streaming data processing instead of batch processing',
        ],
      });
    }

    if (recommendations.length === 0) {
      recommendations.push({
        type: 'healthy',
        severity: 'info',
        message: 'Memory usage appears healthy. Continue monitoring during extended use.',
        actions: [
          'Maintain current memory management practices',
          'Continue periodic monitoring during long visualization sessions',
        ],
      });
    }

    return recommendations;
  }
}

// Export for use in different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = MemoryLeakDetector;
} else if (typeof window !== 'undefined') {
  window.MemoryLeakDetector = MemoryLeakDetector;
}

// Usage example for geological visualization monitoring
if (typeof window !== 'undefined') {
  // Browser environment - attach to window for easy access
  window.startGeologicalMemoryMonitoring = function() {
    if (window.geologicalMemoryDetector) {
      console.warn('Memory monitoring already active');
      return;
    }

    window.geologicalMemoryDetector = new MemoryLeakDetector({
      samplingInterval: 3000, // 3 seconds for real-time monitoring
      leakThreshold: 5 * 1024 * 1024, // 5MB threshold for geological apps
      gcThreshold: 30, // Lower threshold for real-time apps
    });

    window.geologicalMemoryDetector.start();
    console.log('Geological memory monitoring started');
  };

  window.stopGeologicalMemoryMonitoring = function() {
    if (!window.geologicalMemoryDetector) {
      console.warn('No active memory monitoring');
      return null;
    }

    const report = window.geologicalMemoryDetector.stop();
    window.geologicalMemoryDetector = null;
    
    console.log('Memory monitoring report:', report);
    return report;
  };
}