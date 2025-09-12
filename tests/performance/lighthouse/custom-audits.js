// Custom Lighthouse audits for geological visualization performance

const Audit = require('lighthouse').Audit;

class GeologicalVisualizationPerformance extends Audit {
  static get meta() {
    return {
      id: 'geological-visualization-performance',
      title: 'Geological 3D Visualization Performance',
      failureTitle: 'Geological 3D visualizations are not optimized for performance',
      description: 'Measures the performance of WebGL-based geological visualizations including frame rates, memory usage, and rendering efficiency.',
      requiredArtifacts: ['traces', 'devtoolsLogs'],
    };
  }

  static audit(artifacts, context) {
    const trace = artifacts.traces[Audit.DEFAULT_PASS];
    const devtoolsLog = artifacts.devtoolsLogs[Audit.DEFAULT_PASS];
    
    // Analyze WebGL performance metrics
    const webglEvents = trace.traceEvents.filter(event => 
      event.name.includes('WebGL') || 
      event.name.includes('GPU') ||
      event.cat.includes('gpu')
    );
    
    // Calculate average frame time for 3D visualizations
    const frameEvents = trace.traceEvents.filter(event => 
      event.name === 'BeginFrame' || event.name === 'DrawFrame'
    );
    
    let totalFrameTime = 0;
    let frameCount = 0;
    
    for (let i = 1; i < frameEvents.length; i++) {
      const frameTime = frameEvents[i].ts - frameEvents[i-1].ts;
      if (frameTime > 0 && frameTime < 100000) { // Filter out outliers
        totalFrameTime += frameTime;
        frameCount++;
      }
    }
    
    const averageFrameTime = frameCount > 0 ? totalFrameTime / frameCount / 1000 : 0; // Convert to ms
    const fps = averageFrameTime > 0 ? 1000 / averageFrameTime : 0;
    
    // Check for geological data loading performance
    const geologicalDataRequests = devtoolsLog.filter(entry => 
      entry.url && (
        entry.url.includes('/api/v1/geological/') ||
        entry.url.includes('/api/v1/seismic/') ||
        entry.url.includes('/api/v1/volcanic/')
      )
    );
    
    const averageDataLoadTime = geologicalDataRequests.length > 0 
      ? geologicalDataRequests.reduce((sum, req) => sum + (req.responseReceivedTime - req.requestTime), 0) / geologicalDataRequests.length * 1000
      : 0;
    
    // Performance scoring
    let score = 1.0;
    
    if (fps < 30) score -= 0.3; // Poor frame rate
    if (fps < 15) score -= 0.3; // Very poor frame rate
    if (averageDataLoadTime > 2000) score -= 0.2; // Slow data loading
    if (averageDataLoadTime > 5000) score -= 0.2; // Very slow data loading
    
    score = Math.max(0, score);
    
    const details = {
      type: 'table',
      headings: [
        { key: 'metric', itemType: 'text', text: 'Metric' },
        { key: 'value', itemType: 'text', text: 'Value' },
        { key: 'threshold', itemType: 'text', text: 'Target' },
      ],
      items: [
        {
          metric: 'Average FPS',
          value: fps.toFixed(1),
          threshold: '≥30 FPS',
        },
        {
          metric: 'Average Frame Time',
          value: `${averageFrameTime.toFixed(1)}ms`,
          threshold: '≤33ms',
        },
        {
          metric: 'Geological Data Load Time',
          value: `${averageDataLoadTime.toFixed(0)}ms`,
          threshold: '≤2000ms',
        },
        {
          metric: 'WebGL Events',
          value: webglEvents.length.toString(),
          threshold: 'Optimized',
        },
      ],
    };
    
    return {
      score,
      numericValue: fps,
      numericUnit: 'unitless',
      displayValue: `${fps.toFixed(1)} FPS`,
      details,
    };
  }
}

class RealTimeDataStreamingPerformance extends Audit {
  static get meta() {
    return {
      id: 'real-time-streaming-performance',
      title: 'Real-time Data Streaming Performance',
      failureTitle: 'Real-time data streaming is not optimized',
      description: 'Measures the performance of WebSocket connections and real-time data processing for volcanic monitoring.',
      requiredArtifacts: ['devtoolsLogs', 'traces'],
    };
  }

  static audit(artifacts, context) {
    const devtoolsLog = artifacts.devtoolsLogs[Audit.DEFAULT_PASS];
    const trace = artifacts.traces[Audit.DEFAULT_PASS];
    
    // Find WebSocket connections
    const wsConnections = devtoolsLog.filter(entry => 
      entry.method === 'Network.webSocketCreated' ||
      entry.method === 'Network.webSocketFrameReceived' ||
      entry.method === 'Network.webSocketFrameSent'
    );
    
    // Analyze WebSocket message frequency and latency
    const wsMessages = wsConnections.filter(entry => 
      entry.method === 'Network.webSocketFrameReceived'
    );
    
    let totalLatency = 0;
    let messageCount = 0;
    let messageFrequency = 0;
    
    if (wsMessages.length > 1) {
      const timeSpan = wsMessages[wsMessages.length - 1].timestamp - wsMessages[0].timestamp;
      messageFrequency = wsMessages.length / timeSpan;
      
      // Estimate latency from message timing patterns
      for (let i = 1; i < wsMessages.length; i++) {
        const timeDiff = (wsMessages[i].timestamp - wsMessages[i-1].timestamp) * 1000;
        if (timeDiff < 5000) { // Filter out connection gaps
          totalLatency += timeDiff;
          messageCount++;
        }
      }
    }
    
    const averageLatency = messageCount > 0 ? totalLatency / messageCount : 0;
    
    // Check for real-time processing events
    const processingEvents = trace.traceEvents.filter(event => 
      event.name.includes('ProcessRealTimeData') ||
      event.name.includes('UpdateVisualization') ||
      event.cat.includes('realtime')
    );
    
    // Performance scoring
    let score = 1.0;
    
    if (averageLatency > 500) score -= 0.3; // High latency
    if (averageLatency > 1000) score -= 0.3; // Very high latency
    if (messageFrequency < 1) score -= 0.2; // Low message frequency
    if (wsConnections.length === 0) score -= 0.5; // No WebSocket connections
    
    score = Math.max(0, score);
    
    const details = {
      type: 'table',
      headings: [
        { key: 'metric', itemType: 'text', text: 'Metric' },
        { key: 'value', itemType: 'text', text: 'Value' },
        { key: 'threshold', itemType: 'text', text: 'Target' },
      ],
      items: [
        {
          metric: 'WebSocket Connections',
          value: wsConnections.length.toString(),
          threshold: '≥1',
        },
        {
          metric: 'Message Frequency',
          value: `${messageFrequency.toFixed(2)} msg/s`,
          threshold: '≥1 msg/s',
        },
        {
          metric: 'Average Latency',
          value: `${averageLatency.toFixed(0)}ms`,
          threshold: '≤500ms',
        },
        {
          metric: 'Processing Events',
          value: processingEvents.length.toString(),
          threshold: 'Active',
        },
      ],
    };
    
    return {
      score,
      numericValue: averageLatency,
      numericUnit: 'millisecond',
      displayValue: `${averageLatency.toFixed(0)}ms latency`,
      details,
    };
  }
}

class MemoryLeakDetection extends Audit {
  static get meta() {
    return {
      id: 'memory-leak-detection',
      title: 'Memory Leak Detection for Long-running Visualizations',
      failureTitle: 'Potential memory leaks detected in geological visualizations',
      description: 'Detects memory leaks in long-running 3D geological visualizations and real-time data processing.',
      requiredArtifacts: ['traces'],
    };
  }

  static audit(artifacts, context) {
    const trace = artifacts.traces[Audit.DEFAULT_PASS];
    
    // Find memory allocation events
    const memoryEvents = trace.traceEvents.filter(event => 
      event.name.includes('GC') ||
      event.name.includes('Memory') ||
      event.name === 'UpdateCounters' ||
      event.cat.includes('memory')
    );
    
    // Analyze memory usage trends
    const memoryUsageEvents = memoryEvents.filter(event => 
      event.args && event.args.data && event.args.data.jsHeapSizeUsed
    );
    
    let memoryTrend = 0;
    let maxMemoryUsage = 0;
    let memoryGrowthRate = 0;
    
    if (memoryUsageEvents.length > 10) {
      const startMemory = memoryUsageEvents[0].args.data.jsHeapSizeUsed;
      const endMemory = memoryUsageEvents[memoryUsageEvents.length - 1].args.data.jsHeapSizeUsed;
      const timeSpan = memoryUsageEvents[memoryUsageEvents.length - 1].ts - memoryUsageEvents[0].ts;
      
      memoryTrend = endMemory - startMemory;
      memoryGrowthRate = memoryTrend / (timeSpan / 1000000); // MB per second
      maxMemoryUsage = Math.max(...memoryUsageEvents.map(e => e.args.data.jsHeapSizeUsed));
    }
    
    // Count garbage collection events
    const gcEvents = memoryEvents.filter(event => 
      event.name.includes('GC') || event.name === 'MinorGC' || event.name === 'MajorGC'
    );
    
    const gcFrequency = gcEvents.length;
    
    // Performance scoring
    let score = 1.0;
    
    if (memoryGrowthRate > 1048576) score -= 0.4; // Growing > 1MB/s
    if (memoryGrowthRate > 5242880) score -= 0.4; // Growing > 5MB/s
    if (maxMemoryUsage > 536870912) score -= 0.2; // Max usage > 512MB
    if (gcFrequency > 50) score -= 0.2; // Frequent GC
    
    score = Math.max(0, score);
    
    const details = {
      type: 'table',
      headings: [
        { key: 'metric', itemType: 'text', text: 'Metric' },
        { key: 'value', itemType: 'text', text: 'Value' },
        { key: 'threshold', itemType: 'text', text: 'Target' },
      ],
      items: [
        {
          metric: 'Memory Growth Rate',
          value: `${(memoryGrowthRate / 1048576).toFixed(2)} MB/s`,
          threshold: '≤1 MB/s',
        },
        {
          metric: 'Max Memory Usage',
          value: `${(maxMemoryUsage / 1048576).toFixed(0)} MB`,
          threshold: '≤512 MB',
        },
        {
          metric: 'GC Events',
          value: gcFrequency.toString(),
          threshold: '≤50',
        },
        {
          metric: 'Memory Trend',
          value: `${(memoryTrend / 1048576).toFixed(1)} MB`,
          threshold: 'Stable',
        },
      ],
    };
    
    return {
      score,
      numericValue: memoryGrowthRate,
      numericUnit: 'byte',
      displayValue: `${(memoryGrowthRate / 1048576).toFixed(2)} MB/s growth`,
      details,
    };
  }
}

module.exports = {
  GeologicalVisualizationPerformance,
  RealTimeDataStreamingPerformance,
  MemoryLeakDetection,
};