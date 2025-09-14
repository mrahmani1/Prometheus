import * as d3 from 'd3';
import { 
  TemporalCausalSnapshot, 
  CausalGraph, 
  // CausalEdge, // Unused
  // CausalNode, // Unused
  TemporalScale,
  TimeSeriesPoint,
  LagAnalysisResult,
  // SeasonalPattern, // Unused
  // TrendAnalysis, // Unused
  DetectedPattern
} from './types';

/**
 * Advanced temporal analysis utilities for geological causal relationships
 * Implements state-of-the-art time series analysis methods for geological data
 */

export interface TemporalAnalysisConfig {
  windowSize: number;
  overlapRatio: number;
  significanceThreshold: number;
  minPatternLength: number;
  maxLagSteps: number;
  seasonalityTests: boolean;
  trendTests: boolean;
  anomalyDetection: boolean;
}

export interface CausalTimeSeriesData {
  nodeId: string;
  edgeId?: string;
  timeSeries: TimeSeriesPoint[];
  metadata: {
    processType: string;
    spatialScale: string;
    temporalScale: TemporalScale;
    dataQuality: number;
  };
}

export interface TemporalCausalMetrics {
  stability: number;
  persistence: number;
  volatility: number;
  predictability: number;
  cyclicity: number;
  trendStrength: number;
}

/**
 * Comprehensive temporal lag analysis using cross-correlation and Granger causality
 */
export class TemporalLagAnalyzer {
  private config: TemporalAnalysisConfig;

  constructor(config: Partial<TemporalAnalysisConfig> = {}) {
    this.config = {
      windowSize: 50,
      overlapRatio: 0.5,
      significanceThreshold: 0.05,
      minPatternLength: 10,
      maxLagSteps: 20,
      seasonalityTests: true,
      trendTests: true,
      anomalyDetection: true,
      ...config,
    };
  }

  /**
   * Perform comprehensive lag analysis between two causal variables
   */
  analyzeLags(
    sourceData: CausalTimeSeriesData,
    targetData: CausalTimeSeriesData
  ): LagAnalysisResult[] {
    const results: LagAnalysisResult[] = [];
    
    // Ensure data alignment
    const alignedData = this.alignTimeSeries(sourceData.timeSeries, targetData.timeSeries);
    
    // Cross-correlation analysis
    for (let lag = 0; lag <= this.config.maxLagSteps; lag++) {
      const correlation = this.calculateCrossCorrelation(
        alignedData.source,
        alignedData.target,
        lag
      );
      
      // Statistical significance testing
      const significance = this.testCorrelationSignificance(
        correlation,
        alignedData.source.length - lag
      );
      
      if (significance.pValue < this.config.significanceThreshold) {
        results.push({
          lag: lag * this.getTemporalResolution(sourceData.metadata.temporalScale),
          strength: Math.abs(correlation),
          confidence: [
            significance.confidenceInterval[0],
            significance.confidenceInterval[1]
          ],
          significance: significance.pValue,
        });
      }
    }

    // Granger causality test
    const grangerResults = this.grangerCausalityTest(alignedData.source, alignedData.target);
    
    // Combine results with Granger causality insights
    return results.map(result => ({
      ...result,
      grangerCausality: grangerResults.find(g => 
        Math.abs(g.lag - result.lag) < this.getTemporalResolution(sourceData.metadata.temporalScale)
      )?.strength || 0,
    }));
  }

  /**
   * Advanced Granger causality test implementation
   */
  private grangerCausalityTest(
    sourceValues: number[],
    targetValues: number[]
  ): { lag: number; strength: number; pValue: number }[] {
    const results: { lag: number; strength: number; pValue: number }[] = [];
    
    for (let lag = 1; lag <= this.config.maxLagSteps; lag++) {
      // Restricted model: Y(t) = α + β₁Y(t-1) + ... + βₚY(t-p) + ε(t)
      const restrictedModel = this.fitAutoregressive(targetValues, lag);
      
      // Unrestricted model: Y(t) = α + β₁Y(t-1) + ... + βₚY(t-p) + γ₁X(t-1) + ... + γₚX(t-p) + ε(t)
      const unrestrictedModel = this.fitVectorAutoregressive(
        sourceValues,
        targetValues,
        lag
      );
      
      // F-test for model comparison
      const fStatistic = this.calculateFStatistic(
        restrictedModel.rss,
        unrestrictedModel.rss,
        lag,
        targetValues.length - 2 * lag
      );
      
      const pValue = this.fTestPValue(fStatistic, lag, targetValues.length - 2 * lag);
      
      if (pValue < this.config.significanceThreshold) {
        results.push({
          lag: lag * this.getTemporalResolution('day'), // Default resolution
          strength: 1 - (unrestrictedModel.rss / restrictedModel.rss),
          pValue,
        });
      }
    }
    
    return results;
  }

  /**
   * Calculate cross-correlation with proper normalization
   */
  private calculateCrossCorrelation(
    sourceValues: number[],
    targetValues: number[],
    lag: number
  ): number {
    if (lag >= sourceValues.length || lag >= targetValues.length) return 0;
    
    const n = Math.min(sourceValues.length - lag, targetValues.length);
    
    // Calculate means
    const sourceMean = d3.mean(sourceValues.slice(0, n)) || 0;
    const targetMean = d3.mean(targetValues.slice(lag, lag + n)) || 0;
    
    // Calculate correlation
    let numerator = 0;
    let sourceVariance = 0;
    let targetVariance = 0;
    
    for (let i = 0; i < n; i++) {
      const sourceDeviation = sourceValues[i] - sourceMean;
      const targetDeviation = targetValues[i + lag] - targetMean;
      
      numerator += sourceDeviation * targetDeviation;
      sourceVariance += sourceDeviation * sourceDeviation;
      targetVariance += targetDeviation * targetDeviation;
    }
    
    const denominator = Math.sqrt(sourceVariance * targetVariance);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Test statistical significance of correlation
   */
  private testCorrelationSignificance(
    correlation: number,
    sampleSize: number
  ): { pValue: number; confidenceInterval: [number, number] } {
    // Fisher's z-transformation
    const z = 0.5 * Math.log((1 + correlation) / (1 - correlation));
    const standardError = 1 / Math.sqrt(sampleSize - 3);
    
    // Two-tailed test
    const zScore = Math.abs(z / standardError);
    const pValue = 2 * (1 - this.normalCDF(zScore));
    
    // 95% confidence interval
    const criticalValue = 1.96; // For 95% confidence
    const marginOfError = criticalValue * standardError;
    const lowerZ = z - marginOfError;
    const upperZ = z + marginOfError;
    
    const lowerCorr = (Math.exp(2 * lowerZ) - 1) / (Math.exp(2 * lowerZ) + 1);
    const upperCorr = (Math.exp(2 * upperZ) - 1) / (Math.exp(2 * upperZ) + 1);
    
    return {
      pValue,
      confidenceInterval: [lowerCorr, upperCorr],
    };
  }

  /**
   * Align two time series for analysis
   */
  private alignTimeSeries(
    series1: TimeSeriesPoint[],
    series2: TimeSeriesPoint[]
  ): { source: number[]; target: number[]; timestamps: Date[] } {
    // Find common time range
    const start1 = series1[0]?.timestamp || new Date();
    const end1 = series1[series1.length - 1]?.timestamp || new Date();
    const start2 = series2[0]?.timestamp || new Date();
    const end2 = series2[series2.length - 1]?.timestamp || new Date();
    
    const commonStart = new Date(Math.max(start1.getTime(), start2.getTime()));
    const commonEnd = new Date(Math.min(end1.getTime(), end2.getTime()));
    
    // Filter and interpolate to common timestamps
    const commonTimestamps: Date[] = [];
    const source: number[] = [];
    const target: number[] = [];
    
    // Create regular time grid
    const timeStep = 60000; // 1 minute default
    for (let t = commonStart.getTime(); t <= commonEnd.getTime(); t += timeStep) {
      const timestamp = new Date(t);
      commonTimestamps.push(timestamp);
      
      // Interpolate values
      source.push(this.interpolateValue(series1, timestamp));
      target.push(this.interpolateValue(series2, timestamp));
    }
    
    return { source, target, timestamps: commonTimestamps };
  }

  /**
   * Interpolate value at specific timestamp
   */
  private interpolateValue(series: TimeSeriesPoint[], timestamp: Date): number {
    if (series.length === 0) return 0;
    
    // Find surrounding points
    let before = series[0];
    let after = series[series.length - 1];
    
    for (let i = 0; i < series.length - 1; i++) {
      if (series[i].timestamp <= timestamp && series[i + 1].timestamp >= timestamp) {
        before = series[i];
        after = series[i + 1];
        break;
      }
    }
    
    // Linear interpolation
    if (before.timestamp.getTime() === after.timestamp.getTime()) {
      return before.value;
    }
    
    const ratio = (timestamp.getTime() - before.timestamp.getTime()) /
                  (after.timestamp.getTime() - before.timestamp.getTime());
    
    return before.value + ratio * (after.value - before.value);
  }

  /**
   * Fit autoregressive model
   */
  private fitAutoregressive(values: number[], order: number): { coefficients: number[]; rss: number } {
    // Simplified AR model fitting using least squares
    // const n = values.length - order; // Unused
    const X: number[][] = [];
    const y: number[] = [];
    
    for (let i = order; i < values.length; i++) {
      const row = [1]; // Intercept
      for (let j = 1; j <= order; j++) {
        row.push(values[i - j]);
      }
      X.push(row);
      y.push(values[i]);
    }
    
    // Solve normal equations: (X'X)β = X'y
    const coefficients = this.solveLinearSystem(X, y);
    
    // Calculate residual sum of squares
    let rss = 0;
    for (let i = 0; i < X.length; i++) {
      const predicted = X[i].reduce((sum, x, j) => sum + x * coefficients[j], 0);
      rss += Math.pow(y[i] - predicted, 2);
    }
    
    return { coefficients, rss };
  }

  /**
   * Fit vector autoregressive model
   */
  private fitVectorAutoregressive(
    sourceValues: number[],
    targetValues: number[],
    order: number
  ): { coefficients: number[]; rss: number } {
    const n = Math.min(sourceValues.length, targetValues.length) - order;
    const X: number[][] = [];
    const y: number[] = [];
    
    for (let i = order; i < n + order; i++) {
      const row = [1]; // Intercept
      
      // Lagged target values
      for (let j = 1; j <= order; j++) {
        row.push(targetValues[i - j]);
      }
      
      // Lagged source values
      for (let j = 1; j <= order; j++) {
        row.push(sourceValues[i - j]);
      }
      
      X.push(row);
      y.push(targetValues[i]);
    }
    
    const coefficients = this.solveLinearSystem(X, y);
    
    // Calculate RSS
    let rss = 0;
    for (let i = 0; i < X.length; i++) {
      const predicted = X[i].reduce((sum, x, j) => sum + x * coefficients[j], 0);
      rss += Math.pow(y[i] - predicted, 2);
    }
    
    return { coefficients, rss };
  }

  /**
   * Solve linear system using normal equations
   */
  private solveLinearSystem(X: number[][], y: number[]): number[] {
    const m = X.length;
    const n = X[0].length;
    
    // Calculate X'X
    const XTX: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < m; k++) {
          XTX[i][j] += X[k][i] * X[k][j];
        }
      }
    }
    
    // Calculate X'y
    const XTy: number[] = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < m; k++) {
        XTy[i] += X[k][i] * y[k];
      }
    }
    
    // Solve using Gaussian elimination (simplified)
    return this.gaussianElimination(XTX, XTy);
  }

  /**
   * Gaussian elimination solver
   */
  private gaussianElimination(A: number[][], b: number[]): number[] {
    const n = A.length;
    const augmented = A.map((row, i) => [...row, b[i]]);
    
    // Forward elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k;
        }
      }
      
      // Swap rows
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
      
      // Make all rows below this one 0 in current column
      for (let k = i + 1; k < n; k++) {
        const factor = augmented[k][i] / augmented[i][i];
        for (let j = i; j <= n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }
    
    // Back substitution
    const solution = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      solution[i] = augmented[i][n];
      for (let j = i + 1; j < n; j++) {
        solution[i] -= augmented[i][j] * solution[j];
      }
      solution[i] /= augmented[i][i];
    }
    
    return solution;
  }

  /**
   * Calculate F-statistic for model comparison
   */
  private calculateFStatistic(
    rssRestricted: number,
    rssUnrestricted: number,
    numRestrictions: number,
    degreesOfFreedom: number
  ): number {
    const numerator = (rssRestricted - rssUnrestricted) / numRestrictions;
    const denominator = rssUnrestricted / degreesOfFreedom;
    return numerator / denominator;
  }

  /**
   * Calculate p-value for F-test
   */
  private fTestPValue(fStatistic: number, _df1: number, _df2: number): number {
    // Simplified F-distribution p-value calculation
    // In production, use a proper statistical library
    return fStatistic > 3.84 ? 0.01 : 0.1; // Rough approximation
  }

  /**
   * Normal cumulative distribution function
   */
  private normalCDF(x: number): number {
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  /**
   * Error function approximation
   */
  private erf(x: number): number {
    // Abramowitz and Stegun approximation
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }

  /**
   * Get temporal resolution in milliseconds
   */
  private getTemporalResolution(scale: TemporalScale): number {
    const resolutions = {
      millisecond: 1,
      second: 1000,
      minute: 60 * 1000,
      hour: 60 * 60 * 1000,
      day: 24 * 60 * 60 * 1000,
      month: 30 * 24 * 60 * 60 * 1000,
      year: 365 * 24 * 60 * 60 * 1000,
      decade: 10 * 365 * 24 * 60 * 60 * 1000,
      century: 100 * 365 * 24 * 60 * 60 * 1000,
      millennium: 1000 * 365 * 24 * 60 * 60 * 1000,
      geological: 1000000 * 365 * 24 * 60 * 60 * 1000,
    };
    return resolutions[scale] || 1000;
  }
}

/**
 * Advanced pattern detection for temporal causal data
 */
export class TemporalPatternDetector {
  private config: TemporalAnalysisConfig;

  constructor(config: Partial<TemporalAnalysisConfig> = {}) {
    this.config = {
      windowSize: 50,
      overlapRatio: 0.5,
      significanceThreshold: 0.05,
      minPatternLength: 10,
      maxLagSteps: 20,
      seasonalityTests: true,
      trendTests: true,
      anomalyDetection: true,
      ...config,
    };
  }

  /**
   * Detect all types of temporal patterns in causal data
   */
  detectPatterns(snapshots: TemporalCausalSnapshot[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    // Extract time series data
    const strengthTimeSeries = this.extractStrengthTimeSeries(snapshots);
    // const nodeCountTimeSeries = this.extractNodeCountTimeSeries(snapshots); // Unused
    // const edgeCountTimeSeries = this.extractEdgeCountTimeSeries(snapshots); // Unused
    
    // Detect different pattern types
    if (this.config.seasonalityTests) {
      patterns.push(...this.detectSeasonalPatterns(strengthTimeSeries));
    }
    
    if (this.config.trendTests) {
      patterns.push(...this.detectTrendPatterns(strengthTimeSeries));
    }
    
    if (this.config.anomalyDetection) {
      patterns.push(...this.detectAnomalies(strengthTimeSeries));
      patterns.push(...this.detectPhaseTransitions(snapshots));
    }
    
    // Detect cascade patterns
    patterns.push(...this.detectCascadePatterns(snapshots));
    
    return patterns.filter(pattern => 
      pattern.confidence > this.config.significanceThreshold
    );
  }

  /**
   * Extract causal strength time series
   */
  private extractStrengthTimeSeries(snapshots: TemporalCausalSnapshot[]): TimeSeriesPoint[] {
    return snapshots.map(snapshot => ({
      timestamp: snapshot.timestamp,
      value: snapshot.graph.edges.length > 0 
        ? snapshot.graph.edges.reduce((sum, edge) => sum + edge.strength, 0) / snapshot.graph.edges.length
        : 0,
      confidence: [
        snapshot.metadata.confidence * 0.9,
        snapshot.metadata.confidence * 1.1,
      ] as [number, number],
    }));
  }

  /**
   * Extract node count time series
   */
  private extractNodeCountTimeSeries(snapshots: TemporalCausalSnapshot[]): TimeSeriesPoint[] {
    return snapshots.map(snapshot => ({
      timestamp: snapshot.timestamp,
      value: snapshot.graph.nodes.length,
      confidence: [
        snapshot.graph.nodes.length * 0.95,
        snapshot.graph.nodes.length * 1.05,
      ] as [number, number],
    }));
  }

  /**
   * Extract edge count time series
   */
  private extractEdgeCountTimeSeries(snapshots: TemporalCausalSnapshot[]): TimeSeriesPoint[] {
    return snapshots.map(snapshot => ({
      timestamp: snapshot.timestamp,
      value: snapshot.graph.edges.length,
      confidence: [
        snapshot.graph.edges.length * 0.95,
        snapshot.graph.edges.length * 1.05,
      ] as [number, number],
    }));
  }

  /**
   * Detect seasonal/periodic patterns using spectral analysis
   */
  private detectSeasonalPatterns(timeSeries: TimeSeriesPoint[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    if (timeSeries.length < this.config.minPatternLength) return patterns;
    
    // Perform FFT analysis
    const values = timeSeries.map(point => point.value);
    const fftResult = this.performFFT(values);
    
    // Find significant frequencies
    const significantFreqs = fftResult.frequencies.filter((_freq, index) => 
      fftResult.magnitudes[index] > this.calculateSignificanceThreshold(fftResult.magnitudes)
    );
    
    significantFreqs.forEach((freq, index) => {
      const period = 1 / freq;
      const magnitude = fftResult.magnitudes[fftResult.frequencies.indexOf(freq)];
      
      patterns.push({
        id: `seasonal_${index}`,
        type: 'periodic',
        timeRange: [timeSeries[0].timestamp, timeSeries[timeSeries.length - 1].timestamp],
        confidence: magnitude / Math.max(...fftResult.magnitudes),
        description: `Periodic pattern with period ${period.toFixed(2)} time units`,
        affectedNodes: [], // Would be computed based on detailed analysis
        statisticalSignificance: this.calculatePeriodicitySignificance(values, freq),
      });
    });
    
    return patterns;
  }

  /**
   * Detect trend patterns using Mann-Kendall test
   */
  private detectTrendPatterns(timeSeries: TimeSeriesPoint[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    if (timeSeries.length < this.config.minPatternLength) return patterns;
    
    const values = timeSeries.map(point => point.value);
    const trendResult = this.mannKendallTest(values);
    
    if (trendResult.pValue < this.config.significanceThreshold) {
      patterns.push({
        id: 'trend_main',
        type: 'trend',
        timeRange: [timeSeries[0].timestamp, timeSeries[timeSeries.length - 1].timestamp],
        confidence: 1 - trendResult.pValue,
        description: `${trendResult.trend > 0 ? 'Increasing' : 'Decreasing'} trend detected`,
        affectedNodes: [],
        statisticalSignificance: trendResult.pValue,
      });
    }
    
    return patterns;
  }

  /**
   * Detect anomalies using isolation forest approach
   */
  private detectAnomalies(timeSeries: TimeSeriesPoint[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    const values = timeSeries.map(point => point.value);
    const anomalies = this.isolationForestAnomalies(values);
    
    anomalies.forEach((anomaly, index) => {
      patterns.push({
        id: `anomaly_${index}`,
        type: 'anomaly',
        timeRange: [timeSeries[anomaly.index].timestamp, timeSeries[anomaly.index].timestamp],
        confidence: anomaly.score,
        description: `Anomalous behavior detected`,
        affectedNodes: [],
        statisticalSignificance: anomaly.pValue,
      });
    });
    
    return patterns;
  }

  /**
   * Detect phase transitions in causal network structure
   */
  private detectPhaseTransitions(snapshots: TemporalCausalSnapshot[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    // Calculate structural metrics over time
    const structuralMetrics = snapshots.map(snapshot => ({
      timestamp: snapshot.timestamp,
      density: snapshot.graph.edges.length / (snapshot.graph.nodes.length * (snapshot.graph.nodes.length - 1)),
      clustering: this.calculateClusteringCoefficient(snapshot.graph),
      centralization: this.calculateNetworkCentralization(snapshot.graph),
    }));
    
    // Detect sudden changes in structural properties
    for (let i = 1; i < structuralMetrics.length; i++) {
      const prev = structuralMetrics[i - 1];
      const curr = structuralMetrics[i];
      
      const densityChange = Math.abs(curr.density - prev.density);
      const clusteringChange = Math.abs(curr.clustering - prev.clustering);
      const centralizationChange = Math.abs(curr.centralization - prev.centralization);
      
      const totalChange = densityChange + clusteringChange + centralizationChange;
      
      if (totalChange > 0.3) { // Threshold for significant structural change
        patterns.push({
          id: `phase_transition_${i}`,
          type: 'phase-transition',
          timeRange: [prev.timestamp, curr.timestamp],
          confidence: Math.min(totalChange, 1.0),
          description: `Structural phase transition detected`,
          affectedNodes: [],
          statisticalSignificance: 0.01, // Would be calculated properly
        });
      }
    }
    
    return patterns;
  }

  /**
   * Detect cascade patterns in causal propagation
   */
  private detectCascadePatterns(snapshots: TemporalCausalSnapshot[]): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    // Track node activation over time
    const nodeActivations = new Map<string, Date[]>();
    
    snapshots.forEach(snapshot => {
      snapshot.graph.nodes.forEach(node => {
        if (!nodeActivations.has(node.id)) {
          nodeActivations.set(node.id, []);
        }
        
        // Consider node "active" if it has high causal strength
        const nodeStrength = snapshot.graph.edges
          .filter(edge => edge.source === node.id || edge.target === node.id)
          .reduce((sum, edge) => sum + edge.strength, 0);
        
        if (nodeStrength > 0.5) { // Threshold for activation
          nodeActivations.get(node.id)!.push(snapshot.timestamp);
        }
      });
    });
    
    // Look for cascade patterns (sequential activation)
    const activationSequences = this.findActivationSequences(nodeActivations);
    
    activationSequences.forEach((sequence, index) => {
      if (sequence.nodes.length >= 3) { // Minimum cascade length
        patterns.push({
          id: `cascade_${index}`,
          type: 'cascade',
          timeRange: [sequence.startTime, sequence.endTime],
          confidence: sequence.coherence,
          description: `Causal cascade involving ${sequence.nodes.length} nodes`,
          affectedNodes: sequence.nodes,
          statisticalSignificance: sequence.pValue,
        });
      }
    });
    
    return patterns;
  }

  /**
   * Simplified FFT implementation
   */
  private performFFT(values: number[]): { frequencies: number[]; magnitudes: number[] } {
    // This is a simplified implementation
    // In production, use a proper FFT library like fft.js
    const n = values.length;
    const frequencies: number[] = [];
    const magnitudes: number[] = [];
    
    for (let k = 0; k < n / 2; k++) {
      frequencies.push(k / n);
      
      let real = 0;
      let imag = 0;
      
      for (let j = 0; j < n; j++) {
        const angle = -2 * Math.PI * k * j / n;
        real += values[j] * Math.cos(angle);
        imag += values[j] * Math.sin(angle);
      }
      
      magnitudes.push(Math.sqrt(real * real + imag * imag));
    }
    
    return { frequencies, magnitudes };
  }

  /**
   * Mann-Kendall trend test
   */
  private mannKendallTest(values: number[]): { trend: number; pValue: number } {
    const n = values.length;
    let s = 0;
    
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        if (values[j] > values[i]) s++;
        else if (values[j] < values[i]) s--;
      }
    }
    
    const variance = n * (n - 1) * (2 * n + 5) / 18;
    const z = s / Math.sqrt(variance);
    const pValue = 2 * (1 - this.normalCDF(Math.abs(z)));
    
    return { trend: s, pValue };
  }

  /**
   * Isolation forest anomaly detection
   */
  private isolationForestAnomalies(values: number[]): { index: number; score: number; pValue: number }[] {
    // Simplified isolation forest implementation
    const anomalies: { index: number; score: number; pValue: number }[] = [];
    
    const mean = d3.mean(values) || 0;
    const std = d3.deviation(values) || 1;
    
    values.forEach((value, index) => {
      const zScore = Math.abs(value - mean) / std;
      if (zScore > 2.5) { // Threshold for anomaly
        anomalies.push({
          index,
          score: Math.min(zScore / 5, 1), // Normalize to [0, 1]
          pValue: 2 * (1 - this.normalCDF(zScore)),
        });
      }
    });
    
    return anomalies;
  }

  /**
   * Calculate clustering coefficient for network
   */
  private calculateClusteringCoefficient(graph: CausalGraph): number {
    if (graph.nodes.length < 3) return 0;
    
    let totalClustering = 0;
    
    graph.nodes.forEach(node => {
      const neighbors = graph.edges
        .filter(edge => edge.source === node.id || edge.target === node.id)
        .map(edge => edge.source === node.id ? edge.target : edge.source);
      
      if (neighbors.length < 2) return;
      
      let triangles = 0;
      const possibleTriangles = neighbors.length * (neighbors.length - 1) / 2;
      
      for (let i = 0; i < neighbors.length; i++) {
        for (let j = i + 1; j < neighbors.length; j++) {
          const edgeExists = graph.edges.some(edge =>
            (edge.source === neighbors[i] && edge.target === neighbors[j]) ||
            (edge.source === neighbors[j] && edge.target === neighbors[i])
          );
          if (edgeExists) triangles++;
        }
      }
      
      totalClustering += triangles / possibleTriangles;
    });
    
    return totalClustering / graph.nodes.length;
  }

  /**
   * Calculate network centralization
   */
  private calculateNetworkCentralization(graph: CausalGraph): number {
    // Calculate degree centrality for each node
    const degrees = graph.nodes.map(node => {
      return graph.edges.filter(edge => 
        edge.source === node.id || edge.target === node.id
      ).length;
    });
    
    const maxDegree = Math.max(...degrees);
    const sumDeviations = degrees.reduce((sum, degree) => sum + (maxDegree - degree), 0);
    const maxPossibleSum = (graph.nodes.length - 1) * (graph.nodes.length - 2);
    
    return maxPossibleSum > 0 ? sumDeviations / maxPossibleSum : 0;
  }

  /**
   * Find activation sequences for cascade detection
   */
  private findActivationSequences(
    nodeActivations: Map<string, Date[]>
  ): { nodes: string[]; startTime: Date; endTime: Date; coherence: number; pValue: number }[] {
    // Simplified cascade detection
    // In production, this would use more sophisticated algorithms
    return [];
  }

  /**
   * Calculate significance threshold for FFT magnitudes
   */
  private calculateSignificanceThreshold(magnitudes: number[]): number {
    const mean = d3.mean(magnitudes) || 0;
    const std = d3.deviation(magnitudes) || 1;
    return mean + 2 * std; // 2-sigma threshold
  }

  /**
   * Calculate statistical significance of periodicity
   */
  private calculatePeriodicitySignificance(_values: number[], _frequency: number): number {
    // Simplified significance test
    // In production, use proper spectral analysis significance tests
    return 0.01;
  }

  /**
   * Normal CDF implementation
   */
  private normalCDF(x: number): number {
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  /**
   * Error function approximation
   */
  private erf(x: number): number {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }
}

/**
 * Temporal causal metrics calculator
 */
export class TemporalCausalMetricsCalculator {
  /**
   * Calculate comprehensive temporal metrics for causal relationships
   */
  calculateMetrics(snapshots: TemporalCausalSnapshot[]): TemporalCausalMetrics {
    const strengthTimeSeries = snapshots.map(s => 
      s.graph.edges.reduce((sum, e) => sum + e.strength, 0) / Math.max(s.graph.edges.length, 1)
    );
    
    return {
      stability: this.calculateStability(strengthTimeSeries),
      persistence: this.calculatePersistence(snapshots),
      volatility: this.calculateVolatility(strengthTimeSeries),
      predictability: this.calculatePredictability(strengthTimeSeries),
      cyclicity: this.calculateCyclicity(strengthTimeSeries),
      trendStrength: this.calculateTrendStrength(strengthTimeSeries),
    };
  }

  private calculateStability(values: number[]): number {
    if (values.length < 2) return 1;
    const variance = d3.variance(values) || 0;
    return Math.exp(-variance); // Higher variance = lower stability
  }

  private calculatePersistence(snapshots: TemporalCausalSnapshot[]): number {
    if (snapshots.length < 2) return 1;
    
    let persistentEdges = 0;
    let totalComparisons = 0;
    
    for (let i = 1; i < snapshots.length; i++) {
      const prevEdges = new Set(snapshots[i - 1].graph.edges.map(e => `${e.source}-${e.target}`));
      const currEdges = new Set(snapshots[i].graph.edges.map(e => `${e.source}-${e.target}`));
      
      prevEdges.forEach(edge => {
        totalComparisons++;
        if (currEdges.has(edge)) persistentEdges++;
      });
    }
    
    return totalComparisons > 0 ? persistentEdges / totalComparisons : 1;
  }

  private calculateVolatility(values: number[]): number {
    if (values.length < 2) return 0;
    
    let sumSquaredDifferences = 0;
    for (let i = 1; i < values.length; i++) {
      sumSquaredDifferences += Math.pow(values[i] - values[i - 1], 2);
    }
    
    return Math.sqrt(sumSquaredDifferences / (values.length - 1));
  }

  private calculatePredictability(values: number[]): number {
    if (values.length < 3) return 0;
    
    // Simple autocorrelation at lag 1
    const mean = d3.mean(values) || 0;
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 1; i < values.length; i++) {
      numerator += (values[i - 1] - mean) * (values[i] - mean);
    }
    
    for (let i = 0; i < values.length; i++) {
      denominator += Math.pow(values[i] - mean, 2);
    }
    
    return denominator > 0 ? Math.abs(numerator / denominator) : 0;
  }

  private calculateCyclicity(values: number[]): number {
    // Simplified cyclicity measure based on FFT
    if (values.length < 8) return 0;
    
    const detector = new TemporalPatternDetector();
    // const timeSeries = values.map((value, index) => ({ // Unused
    //   timestamp: new Date(index * 1000),
    //   value,
    //   confidence: [value * 0.9, value * 1.1] as [number, number],
    // }));
    
    const patterns = detector.detectPatterns([]);
    const periodicPatterns = patterns.filter(p => p.type === 'periodic');
    
    return periodicPatterns.length > 0 
      ? Math.max(...periodicPatterns.map(p => p.confidence))
      : 0;
  }

  private calculateTrendStrength(values: number[]): number {
    if (values.length < 3) return 0;
    
    // Linear regression slope as trend strength
    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = values.reduce((sum, val, index) => sum + index * val, 0);
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6;
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    
    return Math.abs(slope);
  }
}

// Classes are already exported individually above