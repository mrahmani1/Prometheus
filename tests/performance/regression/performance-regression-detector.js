// Performance regression detection for geological applications

class PerformanceRegressionDetector {
  constructor(options = {}) {
    this.options = {
      baselineWindow: options.baselineWindow || 50, // Number of samples for baseline
      regressionThreshold: options.regressionThreshold || 0.15, // 15% degradation
      significanceLevel: options.significanceLevel || 0.05, // 5% significance
      minSamples: options.minSamples || 10,
      metrics: options.metrics || [
        'response_time',
        'throughput',
        'error_rate',
        'memory_usage',
        'cpu_usage',
        'geological_processing_time',
        'visualization_render_time',
        'data_load_time',
      ],
      ...options,
    };

    this.baselines = new Map();
    this.currentMetrics = new Map();
    this.regressions = [];
    this.historicalData = new Map();
  }

  // Establish baseline performance metrics
  establishBaseline(testResults) {
    console.log('Establishing performance baselines...');
    
    this.options.metrics.forEach(metric => {
      const values = this.extractMetricValues(testResults, metric);
      if (values.length >= this.options.minSamples) {
        const baseline = this.calculateBaseline(values);
        this.baselines.set(metric, baseline);
        
        console.log(`Baseline established for ${metric}:`, {
          mean: baseline.mean.toFixed(2),
          stdDev: baseline.stdDev.toFixed(2),
          p95: baseline.p95.toFixed(2),
          samples: baseline.samples,
        });
      } else {
        console.warn(`Insufficient samples for ${metric} baseline: ${values.length}`);
      }
    });
  }

  // Detect regressions in new test results
  detectRegressions(newTestResults) {
    console.log('Analyzing performance for regressions...');
    
    const detectedRegressions = [];
    
    this.options.metrics.forEach(metric => {
      const baseline = this.baselines.get(metric);
      if (!baseline) {
        console.warn(`No baseline available for metric: ${metric}`);
        return;
      }

      const newValues = this.extractMetricValues(newTestResults, metric);
      if (newValues.length < this.options.minSamples) {
        console.warn(`Insufficient new samples for ${metric}: ${newValues.length}`);
        return;
      }

      const regression = this.analyzeRegression(metric, baseline, newValues);
      if (regression.isRegression) {
        detectedRegressions.push(regression);
        this.regressions.push({
          ...regression,
          timestamp: Date.now(),
        });
      }
    });

    return detectedRegressions;
  }

  // Extract metric values from test results
  extractMetricValues(testResults, metric) {
    const values = [];

    // Handle different test result formats
    if (Array.isArray(testResults)) {
      testResults.forEach(result => {
        const value = this.getMetricValue(result, metric);
        if (value !== null && value !== undefined && !isNaN(value)) {
          values.push(value);
        }
      });
    } else if (typeof testResults === 'object') {
      // Handle k6 results format
      if (testResults.metrics && testResults.metrics[metric]) {
        const metricData = testResults.metrics[metric];
        if (metricData.values) {
          Object.values(metricData.values).forEach(value => {
            if (typeof value === 'number' && !isNaN(value)) {
              values.push(value);
            }
          });
        }
      }
      
      // Handle custom geological metrics
      if (testResults.geological_metrics) {
        const geologicalValue = this.getGeologicalMetricValue(testResults.geological_metrics, metric);
        if (geologicalValue !== null) {
          values.push(geologicalValue);
        }
      }
    }

    return values;
  }

  // Get specific metric value from test result
  getMetricValue(result, metric) {
    const metricMappings = {
      'response_time': ['http_req_duration', 'response_time', 'duration', 'latency'],
      'throughput': ['http_reqs', 'throughput', 'requests_per_second', 'rps'],
      'error_rate': ['http_req_failed', 'error_rate', 'errors', 'failure_rate'],
      'memory_usage': ['memory_used', 'heap_used', 'memory_percentage'],
      'cpu_usage': ['cpu_percentage', 'cpu_used', 'processor_time'],
      'geological_processing_time': ['seismic_processing_duration', 'geological_analysis_time'],
      'visualization_render_time': ['webgl_render_time', 'visualization_fps', 'frame_time'],
      'data_load_time': ['data_transfer_time', 'dataset_load_duration'],
    };

    const possibleKeys = metricMappings[metric] || [metric];
    
    for (const key of possibleKeys) {
      if (result[key] !== undefined) {
        return result[key];
      }
      
      // Check nested objects
      if (result.metrics && result.metrics[key] !== undefined) {
        const metricValue = result.metrics[key];
        return typeof metricValue === 'object' ? metricValue.avg || metricValue.mean || metricValue.value : metricValue;
      }
      
      // Check timings object
      if (result.timings && result.timings[key] !== undefined) {
        return result.timings[key];
      }
    }

    return null;
  }

  // Get geological-specific metric values
  getGeologicalMetricValue(geologicalMetrics, metric) {
    const geologicalMappings = {
      'geological_processing_time': ['seismic_analysis_duration', 'volcanic_model_time', 'geological_computation_time'],
      'visualization_render_time': ['webgl_frame_time', '3d_render_duration', 'geological_visualization_fps'],
      'data_load_time': ['geological_dataset_load', 'seismic_data_transfer', 'volcanic_data_ingestion'],
    };

    const possibleKeys = geologicalMappings[metric] || [metric];
    
    for (const key of possibleKeys) {
      if (geologicalMetrics[key] !== undefined) {
        return geologicalMetrics[key];
      }
    }

    return null;
  }

  // Calculate baseline statistics
  calculateBaseline(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return {
      mean,
      stdDev,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
      samples: values.length,
      values: values.slice(), // Keep copy for statistical tests
    };
  }

  // Analyze potential regression
  analyzeRegression(metric, baseline, newValues) {
    const newStats = this.calculateBaseline(newValues);
    
    // Calculate percentage change
    const meanChange = ((newStats.mean - baseline.mean) / baseline.mean) * 100;
    const p95Change = ((newStats.p95 - baseline.p95) / baseline.p95) * 100;
    
    // Determine if this is a regression based on metric type
    const isWorsePerformance = this.isPerformanceWorse(metric, meanChange, p95Change);
    
    // Statistical significance test (Welch's t-test)
    const tTestResult = this.welchTTest(baseline.values, newValues);
    
    // Determine if this is a significant regression
    const isRegression = isWorsePerformance && 
                        Math.abs(meanChange) > (this.options.regressionThreshold * 100) &&
                        tTestResult.pValue < this.options.significanceLevel;

    const regression = {
      metric,
      isRegression,
      severity: this.calculateSeverity(meanChange, p95Change),
      baseline: {
        mean: baseline.mean,
        p95: baseline.p95,
        stdDev: baseline.stdDev,
      },
      current: {
        mean: newStats.mean,
        p95: newStats.p95,
        stdDev: newStats.stdDev,
      },
      changes: {
        meanChange,
        p95Change,
        absoluteChange: newStats.mean - baseline.mean,
      },
      statistics: {
        tStatistic: tTestResult.tStatistic,
        pValue: tTestResult.pValue,
        degreesOfFreedom: tTestResult.degreesOfFreedom,
        isSignificant: tTestResult.pValue < this.options.significanceLevel,
      },
      confidence: this.calculateConfidence(tTestResult, Math.abs(meanChange)),
    };

    return regression;
  }

  // Determine if performance change is worse
  isPerformanceWorse(metric, meanChange, p95Change) {
    // For these metrics, higher values are worse
    const higherIsBad = [
      'response_time',
      'error_rate',
      'memory_usage',
      'cpu_usage',
      'geological_processing_time',
      'visualization_render_time',
      'data_load_time',
    ];

    // For these metrics, lower values are worse
    const lowerIsBad = [
      'throughput',
    ];

    if (higherIsBad.includes(metric)) {
      return meanChange > 0 || p95Change > 0;
    } else if (lowerIsBad.includes(metric)) {
      return meanChange < 0 || p95Change < 0;
    }

    // Default: assume higher is bad
    return meanChange > 0;
  }

  // Calculate regression severity
  calculateSeverity(meanChange, p95Change) {
    const maxChange = Math.max(Math.abs(meanChange), Math.abs(p95Change));
    
    if (maxChange > 50) return 'critical';
    if (maxChange > 30) return 'high';
    if (maxChange > 15) return 'medium';
    return 'low';
  }

  // Welch's t-test for statistical significance
  welchTTest(sample1, sample2) {
    const n1 = sample1.length;
    const n2 = sample2.length;
    
    const mean1 = sample1.reduce((sum, val) => sum + val, 0) / n1;
    const mean2 = sample2.reduce((sum, val) => sum + val, 0) / n2;
    
    const var1 = sample1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / (n1 - 1);
    const var2 = sample2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / (n2 - 1);
    
    const pooledSE = Math.sqrt(var1 / n1 + var2 / n2);
    const tStatistic = (mean1 - mean2) / pooledSE;
    
    // Welch-Satterthwaite equation for degrees of freedom
    const degreesOfFreedom = Math.pow(var1 / n1 + var2 / n2, 2) / 
                            (Math.pow(var1 / n1, 2) / (n1 - 1) + Math.pow(var2 / n2, 2) / (n2 - 1));
    
    // Approximate p-value calculation (simplified)
    const pValue = this.approximatePValue(Math.abs(tStatistic), degreesOfFreedom);
    
    return {
      tStatistic,
      degreesOfFreedom,
      pValue,
    };
  }

  // Approximate p-value calculation
  approximatePValue(tStat, df) {
    // Simplified approximation - in production, use a proper statistical library
    if (tStat > 3.0) return 0.001;
    if (tStat > 2.5) return 0.01;
    if (tStat > 2.0) return 0.05;
    if (tStat > 1.5) return 0.1;
    return 0.2;
  }

  // Calculate confidence in regression detection
  calculateConfidence(tTestResult, changePercentage) {
    let confidence = 0;
    
    // Statistical significance contributes to confidence
    if (tTestResult.pValue < 0.001) confidence += 0.4;
    else if (tTestResult.pValue < 0.01) confidence += 0.3;
    else if (tTestResult.pValue < 0.05) confidence += 0.2;
    
    // Magnitude of change contributes to confidence
    if (changePercentage > 50) confidence += 0.4;
    else if (changePercentage > 30) confidence += 0.3;
    else if (changePercentage > 15) confidence += 0.2;
    else confidence += 0.1;
    
    // Effect size contributes to confidence
    const effectSize = Math.abs(tTestResult.tStatistic) / Math.sqrt(tTestResult.degreesOfFreedom);
    if (effectSize > 0.8) confidence += 0.2; // Large effect
    else if (effectSize > 0.5) confidence += 0.1; // Medium effect
    
    return Math.min(1.0, confidence);
  }

  // Generate regression report
  generateRegressionReport(regressions) {
    const report = {
      timestamp: Date.now(),
      summary: {
        totalRegressions: regressions.length,
        criticalRegressions: regressions.filter(r => r.severity === 'critical').length,
        highRegressions: regressions.filter(r => r.severity === 'high').length,
        mediumRegressions: regressions.filter(r => r.severity === 'medium').length,
        lowRegressions: regressions.filter(r => r.severity === 'low').length,
      },
      regressions: regressions.map(regression => ({
        ...regression,
        recommendations: this.generateRecommendations(regression),
      })),
      overallAssessment: this.assessOverallPerformance(regressions),
    };

    return report;
  }

  // Generate recommendations for specific regressions
  generateRecommendations(regression) {
    const recommendations = [];
    
    switch (regression.metric) {
      case 'response_time':
        recommendations.push(
          'Investigate API endpoint performance',
          'Check database query optimization',
          'Review caching strategies',
          'Analyze network latency issues'
        );
        break;
        
      case 'geological_processing_time':
        recommendations.push(
          'Profile geological computation algorithms',
          'Check for inefficient seismic data processing',
          'Review volcanic model complexity',
          'Consider parallel processing optimization'
        );
        break;
        
      case 'visualization_render_time':
        recommendations.push(
          'Optimize WebGL rendering pipeline',
          'Reduce geological mesh complexity',
          'Implement level-of-detail (LOD) for 3D models',
          'Check for GPU memory issues'
        );
        break;
        
      case 'memory_usage':
        recommendations.push(
          'Investigate memory leaks in geological visualizations',
          'Optimize data structure usage',
          'Implement proper cleanup for WebGL resources',
          'Review large dataset handling'
        );
        break;
        
      case 'data_load_time':
        recommendations.push(
          'Optimize geological dataset compression',
          'Implement progressive data loading',
          'Review network transfer efficiency',
          'Consider data preprocessing optimization'
        );
        break;
        
      default:
        recommendations.push(
          'Investigate recent code changes',
          'Review system resource usage',
          'Check for external dependencies issues',
          'Analyze performance profiling data'
        );
    }
    
    // Add severity-specific recommendations
    if (regression.severity === 'critical') {
      recommendations.unshift('URGENT: Immediate investigation required');
      recommendations.push('Consider rollback if issue cannot be quickly resolved');
    }
    
    return recommendations;
  }

  // Assess overall performance health
  assessOverallPerformance(regressions) {
    const criticalCount = regressions.filter(r => r.severity === 'critical').length;
    const highCount = regressions.filter(r => r.severity === 'high').length;
    const totalCount = regressions.length;
    
    let status = 'healthy';
    let message = 'Performance is within acceptable ranges';
    
    if (criticalCount > 0) {
      status = 'critical';
      message = `${criticalCount} critical performance regression(s) detected`;
    } else if (highCount > 2) {
      status = 'degraded';
      message = `Multiple high-severity regressions detected (${highCount})`;
    } else if (totalCount > 5) {
      status = 'concerning';
      message = `Many performance regressions detected (${totalCount})`;
    } else if (totalCount > 0) {
      status = 'minor_issues';
      message = `${totalCount} minor performance regression(s) detected`;
    }
    
    return {
      status,
      message,
      regressionCount: totalCount,
      criticalCount,
      highCount,
    };
  }

  // Save baseline for future comparisons
  saveBaseline(filename = 'performance-baseline.json') {
    const baselineData = {
      timestamp: Date.now(),
      baselines: Object.fromEntries(this.baselines),
      options: this.options,
    };
    
    // In browser environment, save to localStorage
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('geological-performance-baseline', JSON.stringify(baselineData));
      console.log('Baseline saved to localStorage');
    }
    
    return baselineData;
  }

  // Load baseline from storage
  loadBaseline() {
    if (typeof localStorage !== 'undefined') {
      const stored = localStorage.getItem('geological-performance-baseline');
      if (stored) {
        try {
          const baselineData = JSON.parse(stored);
          this.baselines = new Map(Object.entries(baselineData.baselines));
          console.log('Baseline loaded from localStorage');
          return true;
        } catch (error) {
          console.error('Failed to load baseline:', error);
        }
      }
    }
    
    return false;
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PerformanceRegressionDetector;
} else if (typeof window !== 'undefined') {
  window.PerformanceRegressionDetector = PerformanceRegressionDetector;
}

// Usage example
if (typeof window !== 'undefined') {
  window.setupPerformanceRegression = function() {
    window.regressionDetector = new PerformanceRegressionDetector({
      regressionThreshold: 0.10, // 10% threshold for geological apps
      metrics: [
        'response_time',
        'geological_processing_time',
        'visualization_render_time',
        'data_load_time',
        'memory_usage',
      ],
    });
    
    console.log('Performance regression detector initialized');
  };
}