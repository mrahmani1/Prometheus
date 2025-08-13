/**
 * Performance Benchmark Runner for Prometheus AI Geodynamic Platform
 * Comprehensive performance testing and benchmarking suite
 */

const fs = require('fs');
const path = require('path');
const { performance, PerformanceObserver } = require('perf_hooks');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// Import test suite components
const { MockDigitalTwinEarth, MockHypothesisGenerator, MockStreamManager } = require('./test_suite.js');

/**
 * Benchmark Configuration
 */
const BENCHMARK_CONFIG = {
    warmupRuns: 3,
    benchmarkRuns: 10,
    memoryThreshold: 100 * 1024 * 1024, // 100MB
    timeThreshold: 5000, // 5 seconds
    concurrencyLevels: [1, 5, 10, 25, 50, 100],
    dataSizes: [100, 1000, 10000, 100000],
    scenarios: [
        'digital_twin_state_updates',
        'hypothesis_generation',
        'stream_processing',
        'data_integration',
        'prediction_generation',
        'anomaly_detection',
        'concurrent_operations',
        'memory_usage',
        'throughput_testing'
    ]
};

/**
 * Performance Metrics Collection
 */
class PerformanceMetrics {
    constructor() {
        this.metrics = new Map();
        this.memoryUsage = [];
        this.startTime = null;
        this.endTime = null;
        
        // Setup performance observer
        this.observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                this.recordMetric(entry.name, entry.duration);
            }
        });
        this.observer.observe({ entryTypes: ['measure'] });
    }

    recordMetric(name, value) {
        if (!this.metrics.has(name)) {
            this.metrics.set(name, []);
        }
        this.metrics.get(name).push(value);
    }

    recordMemoryUsage() {
        const memUsage = process.memoryUsage();
        this.memoryUsage.push({
            timestamp: Date.now(),
            rss: memUsage.rss,
            heapTotal: memUsage.heapTotal,
            heapUsed: memUsage.heapUsed,
            external: memUsage.external
        });
    }

    getStatistics(name) {
        const values = this.metrics.get(name) || [];
        if (values.length === 0) return null;

        const sorted = [...values].sort((a, b) => a - b);
        return {
            count: values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            mean: values.reduce((a, b) => a + b) / values.length,
            median: sorted[Math.floor(sorted.length / 2)],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            p99: sorted[Math.floor(sorted.length * 0.99)],
            stdDev: this.calculateStandardDeviation(values)
        };
    }

    calculateStandardDeviation(values) {
        const mean = values.reduce((a, b) => a + b) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    clear() {
        this.metrics.clear();
        this.memoryUsage = [];
    }
}

/**
 * Benchmark Runner Class
 */
class BenchmarkRunner {
    constructor() {
        this.metrics = new PerformanceMetrics();
        this.results = new Map();
        this.config = BENCHMARK_CONFIG;
        this.startTime = null;
        this.endTime = null;
    }

    /**
     * Run all benchmarks
     */
    async runAllBenchmarks() {
        console.log('🚀 Starting Prometheus AI Performance Benchmarks...\n');
        this.startTime = performance.now();

        // System Information
        await this.collectSystemInfo();
        
        // Core Component Benchmarks
        await this.runDigitalTwinBenchmarks();
        await this.runHypothesisGenerationBenchmarks();
        await this.runStreamProcessingBenchmarks();
        await this.runDataIntegrationBenchmarks();
        
        // Scalability Benchmarks
        await this.runScalabilityBenchmarks();
        await this.runConcurrencyBenchmarks();
        
        // Memory and Resource Benchmarks
        await this.runMemoryBenchmarks();
        await this.runThroughputBenchmarks();

        this.endTime = performance.now();
        await this.generateBenchmarkReport();
    }

    /**
     * Collect system information
     */
    async collectSystemInfo() {
        const os = require('os');
        const systemInfo = {
            platform: os.platform(),
            arch: os.arch(),
            nodeVersion: process.version,
            cpus: os.cpus().length,
            totalMemory: os.totalmem(),
            freeMemory: os.freemem(),
            loadAverage: os.loadavg()
        };

        this.results.set('system_info', systemInfo);
        console.log('💻 System Information:');
        console.log(`   Platform: ${systemInfo.platform} ${systemInfo.arch}`);
        console.log(`   Node.js: ${systemInfo.nodeVersion}`);
        console.log(`   CPUs: ${systemInfo.cpus}`);
        console.log(`   Memory: ${(systemInfo.totalMemory / 1024 / 1024 / 1024).toFixed(2)} GB total\n`);
    }

    /**
     * Digital Twin Benchmarks
     */
    async runDigitalTwinBenchmarks() {
        console.log('🌍 Running Digital Twin Benchmarks...');
        
        // Initialization Benchmark
        await this.benchmark('digital_twin_initialization', async () => {
            const twin = new MockDigitalTwinEarth();
            await twin.initialize();
            return twin.getCurrentState();
        });

        // State Update Benchmark
        await this.benchmark('digital_twin_state_updates', async () => {
            const twin = new MockDigitalTwinEarth();
            await twin.initialize();
            
            const updates = [];
            for (let i = 0; i < 100; i++) {
                updates.push(twin.getCurrentState());
            }
            return updates;
        });

        // Prediction Generation Benchmark
        await this.benchmark('digital_twin_predictions', async () => {
            const twin = new MockDigitalTwinEarth();
            await twin.initialize();
            
            const scenarios = await twin.generatePredictiveScenarios(10, 24);
            return scenarios;
        });

        // Batch Processing Benchmark
        await this.benchmark('digital_twin_batch_processing', async () => {
            const twins = [];
            const promises = [];
            
            for (let i = 0; i < 10; i++) {
                const twin = new MockDigitalTwinEarth();
                twins.push(twin);
                promises.push(twin.initialize());
            }
            
            await Promise.all(promises);
            
            const results = twins.map(twin => twin.getCurrentState());
            return results;
        });

        console.log('   ✅ Digital Twin benchmarks completed\n');
    }

    /**
     * Hypothesis Generation Benchmarks
     */
    async runHypothesisGenerationBenchmarks() {
        console.log('🧠 Running Hypothesis Generation Benchmarks...');
        
        // Single Hypothesis Generation
        await this.benchmark('hypothesis_single_generation', async () => {
            const generator = new MockHypothesisGenerator();
            const context = { temperature: 25, pressure: 1013, humidity: 60 };
            return await generator.generateHypotheses(context, ['climate']);
        });

        // Batch Hypothesis Generation
        await this.benchmark('hypothesis_batch_generation', async () => {
            const generator = new MockHypothesisGenerator();
            const contexts = Array.from({ length: 50 }, (_, i) => ({
                temperature: 20 + i * 0.5,
                pressure: 1000 + i,
                context_id: i
            }));
            
            const promises = contexts.map(context => 
                generator.generateHypotheses(context, ['climate', 'weather'])
            );
            return await Promise.all(promises);
        });

        // Domain-Focused Generation
        await this.benchmark('hypothesis_domain_focused', async () => {
            const generator = new MockHypothesisGenerator();
            const domains = [
                ['seismology', 'tectonics'],
                ['volcanism', 'geochemistry'],
                ['climate', 'atmosphere'],
                ['hydrology', 'oceanography'],
                ['ecology', 'biosphere']
            ];
            
            const results = [];
            for (const domain of domains) {
                const hypotheses = await generator.generateHypotheses({}, domain);
                results.push(hypotheses);
            }
            return results;
        });

        console.log('   ✅ Hypothesis Generation benchmarks completed\n');
    }

    /**
     * Stream Processing Benchmarks
     */
    async runStreamProcessingBenchmarks() {
        console.log('🌊 Running Stream Processing Benchmarks...');
        
        // Event Production Benchmark
        await this.benchmark('stream_event_production', async () => {
            const manager = new MockStreamManager();
            const events = Array.from({ length: 1000 }, (_, i) => ({
                event_id: `bench_${i}`,
                event_type: 'test',
                data: { value: Math.random() * 100 },
                timestamp: new Date()
            }));
            
            for (const event of events) {
                await manager.produceEvent(event, 'benchmark');
            }
            
            return manager.getEventCount('benchmark');
        });

        // Concurrent Event Processing
        await this.benchmark('stream_concurrent_processing', async () => {
            const manager = new MockStreamManager();
            const eventCount = 500;
            
            const promises = Array.from({ length: eventCount }, (_, i) =>
                manager.produceEvent({
                    event_id: `concurrent_${i}`,
                    data: { index: i }
                }, 'concurrent')
            );
            
            await Promise.all(promises);
            return manager.getEventCount('concurrent');
        });

        // Stream Consumption Benchmark
        await this.benchmark('stream_consumption', async () => {
            const manager = new MockStreamManager();
            const processedEvents = [];
            
            // Setup consumer
            await manager.consumeEvents('consumption_test', (event) => {
                processedEvents.push(event);
            });
            
            // Produce events
            const eventPromises = Array.from({ length: 100 }, (_, i) =>
                manager.produceEvent({
                    event_id: `consume_${i}`,
                    data: { value: i }
                }, 'consumption_test')
            );
            
            await Promise.all(eventPromises);
            
            // Wait for processing
            await new Promise(resolve => setTimeout(resolve, 50));
            
            return processedEvents.length;
        });

        console.log('   ✅ Stream Processing benchmarks completed\n');
    }

    /**
     * Data Integration Benchmarks
     */
    async runDataIntegrationBenchmarks() {
        console.log('🔗 Running Data Integration Benchmarks...');
        
        // Multi-Component Integration
        await this.benchmark('data_integration_multi_component', async () => {
            const twin = new MockDigitalTwinEarth();
            const generator = new MockHypothesisGenerator();
            const manager = new MockStreamManager();
            
            await twin.initialize();
            const state = twin.getCurrentState();
            
            const hypotheses = await generator.generateHypotheses(state);
            
            for (const hypothesis of hypotheses) {
                await manager.produceEvent({
                    event_id: hypothesis.hypothesis_id,
                    event_type: 'hypothesis',
                    data: hypothesis
                }, 'integration');
            }
            
            return {
                stateGenerated: !!state,
                hypothesesCount: hypotheses.length,
                eventsProduced: manager.getEventCount('integration')
            };
        });

        // Data Pipeline Performance
        await this.benchmark('data_pipeline_performance', async () => {
            const pipeline = this.createDataPipeline();
            const testData = Array.from({ length: 1000 }, (_, i) => ({
                id: i,
                value: Math.random() * 1000,
                timestamp: new Date(Date.now() + i * 1000),
                metadata: { source: `sensor_${i % 10}` }
            }));
            
            const results = [];
            for (const data of testData) {
                const result = await pipeline.process(data);
                results.push(result);
            }
            
            return results.length;
        });

        console.log('   ✅ Data Integration benchmarks completed\n');
    }

    /**
     * Scalability Benchmarks
     */
    async runScalabilityBenchmarks() {
        console.log('📈 Running Scalability Benchmarks...');
        
        for (const dataSize of this.config.dataSizes) {
            await this.benchmark(`scalability_data_size_${dataSize}`, async () => {
                const manager = new MockStreamManager();
                const events = Array.from({ length: dataSize }, (_, i) => ({
                    event_id: `scale_${dataSize}_${i}`,
                    data: { value: Math.random() }
                }));
                
                const startTime = performance.now();
                for (const event of events) {
                    await manager.produceEvent(event, 'scalability');
                }
                const endTime = performance.now();
                
                return {
                    dataSize,
                    processingTime: endTime - startTime,
                    throughput: dataSize / ((endTime - startTime) / 1000)
                };
            });
        }

        console.log('   ✅ Scalability benchmarks completed\n');
    }

    /**
     * Concurrency Benchmarks
     */
    async runConcurrencyBenchmarks() {
        console.log('⚡ Running Concurrency Benchmarks...');
        
        for (const concurrencyLevel of this.config.concurrencyLevels) {
            await this.benchmark(`concurrency_level_${concurrencyLevel}`, async () => {
                const generator = new MockHypothesisGenerator();
                
                const promises = Array.from({ length: concurrencyLevel }, (_, i) =>
                    generator.generateHypotheses({
                        context_id: i,
                        value: Math.random() * 100
                    })
                );
                
                const startTime = performance.now();
                const results = await Promise.all(promises);
                const endTime = performance.now();
                
                return {
                    concurrencyLevel,
                    totalResults: results.reduce((sum, r) => sum + r.length, 0),
                    processingTime: endTime - startTime,
                    averageTimePerTask: (endTime - startTime) / concurrencyLevel
                };
            });
        }

        console.log('   ✅ Concurrency benchmarks completed\n');
    }

    /**
     * Memory Benchmarks
     */
    async runMemoryBenchmarks() {
        console.log('💾 Running Memory Benchmarks...');
        
        // Memory Usage Pattern
        await this.benchmark('memory_usage_pattern', async () => {
            const initialMemory = process.memoryUsage();
            const objects = [];
            
            // Create memory load
            for (let i = 0; i < 10000; i++) {
                objects.push({
                    id: i,
                    data: new Array(1000).fill(Math.random()),
                    metadata: { created: Date.now() }
                });
                
                if (i % 1000 === 0) {
                    this.metrics.recordMemoryUsage();
                }
            }
            
            const peakMemory = process.memoryUsage();
            
            // Cleanup
            objects.length = 0;
            if (global.gc) global.gc();
            
            const finalMemory = process.memoryUsage();
            
            return {
                initialHeap: initialMemory.heapUsed,
                peakHeap: peakMemory.heapUsed,
                finalHeap: finalMemory.heapUsed,
                memoryIncrease: peakMemory.heapUsed - initialMemory.heapUsed,
                memoryRecovered: peakMemory.heapUsed - finalMemory.heapUsed
            };
        });

        // Memory Leak Detection
        await this.benchmark('memory_leak_detection', async () => {
            const iterations = 100;
            const memoryReadings = [];
            
            for (let i = 0; i < iterations; i++) {
                // Simulate operations that might leak
                const twin = new MockDigitalTwinEarth();
                await twin.initialize();
                const state = twin.getCurrentState();
                
                if (i % 10 === 0) {
                    const memory = process.memoryUsage();
                    memoryReadings.push({
                        iteration: i,
                        heapUsed: memory.heapUsed,
                        heapTotal: memory.heapTotal
                    });
                }
            }
            
            // Calculate memory growth trend
            const firstReading = memoryReadings[0];
            const lastReading = memoryReadings[memoryReadings.length - 1];
            const memoryGrowth = lastReading.heapUsed - firstReading.heapUsed;
            
            return {
                iterations,
                memoryGrowth,
                readings: memoryReadings,
                possibleLeak: memoryGrowth > 10 * 1024 * 1024 // > 10MB growth
            };
        });

        console.log('   ✅ Memory benchmarks completed\n');
    }

    /**
     * Throughput Benchmarks
     */
    async runThroughputBenchmarks() {
        console.log('🚄 Running Throughput Benchmarks...');
        
        // Maximum Throughput Test
        await this.benchmark('maximum_throughput', async () => {
            const manager = new MockStreamManager();
            const testDuration = 5000; // 5 seconds
            const startTime = Date.now();
            let eventCount = 0;
            
            while (Date.now() - startTime < testDuration) {
                await manager.produceEvent({
                    event_id: `throughput_${eventCount}`,
                    data: { value: eventCount }
                }, 'throughput');
                eventCount++;
            }
            
            const actualDuration = Date.now() - startTime;
            const throughput = (eventCount / actualDuration) * 1000; // events per second
            
            return {
                eventCount,
                duration: actualDuration,
                throughput: Math.round(throughput)
            };
        });

        // Sustained Throughput Test
        await this.benchmark('sustained_throughput', async () => {
            const manager = new MockStreamManager();
            const testDuration = 10000; // 10 seconds
            const measurementInterval = 1000; // 1 second
            const measurements = [];
            
            const startTime = Date.now();
            let totalEvents = 0;
            let lastMeasurement = startTime;
            let intervalEvents = 0;
            
            while (Date.now() - startTime < testDuration) {
                await manager.produceEvent({
                    event_id: `sustained_${totalEvents}`,
                    data: { value: totalEvents }
                }, 'sustained');
                
                totalEvents++;
                intervalEvents++;
                
                if (Date.now() - lastMeasurement >= measurementInterval) {
                    const currentTime = Date.now();
                    const intervalDuration = currentTime - lastMeasurement;
                    const intervalThroughput = (intervalEvents / intervalDuration) * 1000;
                    
                    measurements.push({
                        timestamp: currentTime - startTime,
                        throughput: Math.round(intervalThroughput),
                        events: intervalEvents
                    });
                    
                    lastMeasurement = currentTime;
                    intervalEvents = 0;
                }
            }
            
            const averageThroughput = measurements.reduce((sum, m) => sum + m.throughput, 0) / measurements.length;
            
            return {
                totalEvents,
                measurements,
                averageThroughput: Math.round(averageThroughput),
                duration: Date.now() - startTime
            };
        });

        console.log('   ✅ Throughput benchmarks completed\n');
    }

    /**
     * Benchmark execution wrapper
     */
    async benchmark(name, benchmarkFunction) {
        console.log(`   🏃 Running ${name}...`);
        
        // Warmup runs
        for (let i = 0; i < this.config.warmupRuns; i++) {
            try {
                await benchmarkFunction();
            } catch (error) {
                console.warn(`   ⚠️  Warmup run ${i + 1} failed: ${error.message}`);
            }
        }
        
        // Actual benchmark runs
        const results = [];
        for (let i = 0; i < this.config.benchmarkRuns; i++) {
            try {
                performance.mark(`${name}_start`);
                this.metrics.recordMemoryUsage();
                
                const result = await benchmarkFunction();
                
                performance.mark(`${name}_end`);
                performance.measure(name, `${name}_start`, `${name}_end`);
                this.metrics.recordMemoryUsage();
                
                const duration = performance.now() - performance.now();
                results.push({
                    run: i + 1,
                    result,
                    duration,
                    success: true
                });
                
            } catch (error) {
                results.push({
                    run: i + 1,
                    result: null,
                    duration: 0,
                    success: false,
                    error: error.message
                });
                console.warn(`   ⚠️  Run ${i + 1} failed: ${error.message}`);
            }
        }
        
        const stats = this.metrics.getStatistics(name);
        const successfulRuns = results.filter(r => r.success).length;
        
        this.results.set(name, {
            name,
            stats,
            results,
            successRate: (successfulRuns / this.config.benchmarkRuns) * 100
        });
        
        if (stats) {
            console.log(`   ✅ ${name}: ${stats.mean.toFixed(2)}ms avg (${successfulRuns}/${this.config.benchmarkRuns} successful)`);
        }
    }

    /**
     * Create mock data pipeline
     */
    createDataPipeline() {
        return {
            process: async (data) => {
                // Validation stage
                if (!data.id || !data.value) {
                    throw new Error('Invalid data');
                }
                
                // Transformation stage
                const transformed = {
                    ...data,
                    normalizedValue: data.value / 1000,
                    processed: true,
                    processingTime: Date.now()
                };
                
                // Enrichment stage
                const enriched = {
                    ...transformed,
                    category: data.value > 500 ? 'high' : 'low',
                    enriched: true
                };
                
                return enriched;
            }
        };
    }

    /**
     * Generate comprehensive benchmark report
     */
    async generateBenchmarkReport() {
        const totalDuration = (this.endTime - this.startTime) / 1000;
        
        console.log('\n📊 BENCHMARK RESULTS SUMMARY');
        console.log('==============================');
        console.log(`Total Benchmark Duration: ${totalDuration.toFixed(2)}s`);
        console.log(`Total Benchmarks: ${this.results.size}`);
        
        // Performance Overview
        console.log('\n⚡ PERFORMANCE OVERVIEW:');
        const performanceResults = [];
        
        for (const [name, result] of this.results) {
            if (result.stats && name !== 'system_info') {
                performanceResults.push({
                    name,
                    mean: result.stats.mean,
                    p95: result.stats.p95,
                    successRate: result.successRate
                });
            }
        }
        
        // Sort by mean duration
        performanceResults.sort((a, b) => a.mean - b.mean);
        
        performanceResults.slice(0, 10).forEach(result => {
            console.log(`   ${result.name}: ${result.mean.toFixed(2)}ms (p95: ${result.p95.toFixed(2)}ms, ${result.successRate.toFixed(1)}% success)`);
        });
        
        // Memory Usage Summary
        console.log('\n💾 MEMORY USAGE SUMMARY:');
        if (this.metrics.memoryUsage.length > 0) {
            const memoryStats = this.analyzeMemoryUsage();
            console.log(`   Peak Memory Usage: ${(memoryStats.peakHeapUsed / 1024 / 1024).toFixed(2)} MB`);
            console.log(`   Average Memory Usage: ${(memoryStats.avgHeapUsed / 1024 / 1024).toFixed(2)} MB`);
            console.log(`   Memory Growth Rate: ${memoryStats.growthRate.toFixed(2)} MB/min`);
        }
        
        // Save detailed report
        await this.saveDetailedReport(totalDuration);
    }

    /**
     * Analyze memory usage patterns
     */
    analyzeMemoryUsage() {
        if (this.metrics.memoryUsage.length === 0) return null;
        
        const heapUsages = this.metrics.memoryUsage.map(m => m.heapUsed);
        const peakHeapUsed = Math.max(...heapUsages);
        const avgHeapUsed = heapUsages.reduce((a, b) => a + b) / heapUsages.length;
        
        // Calculate growth rate (MB per minute)
        const firstReading = this.metrics.memoryUsage[0];
        const lastReading = this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1];
        const timeDiff = (lastReading.timestamp - firstReading.timestamp) / 1000 / 60; // minutes
        const memoryDiff = (lastReading.heapUsed - firstReading.heapUsed) / 1024 / 1024; // MB
        const growthRate = timeDiff > 0 ? memoryDiff / timeDiff : 0;
        
        return {
            peakHeapUsed,
            avgHeapUsed,
            growthRate,
            readings: this.metrics.memoryUsage.length
        };
    }

    /**
     * Save detailed benchmark report
     */
    async saveDetailedReport(totalDuration) {
        const report = {
            meta: {
                timestamp: new Date().toISOString(),
                totalDuration,
                nodeVersion: process.version,
                benchmarkConfig: this.config
            },
            systemInfo: this.results.get('system_info'),
            benchmarks: {},
            memoryAnalysis: this.analyzeMemoryUsage(),
            summary: {
                totalBenchmarks: this.results.size - 1, // Exclude system_info
                averagePerformance: this.calculateAveragePerformance(),
                recommendations: this.generateRecommendations()
            }
        };
        
        // Add all benchmark results
        for (const [name, result] of this.results) {
            if (name !== 'system_info') {
                report.benchmarks[name] = result;
            }
        }
        
        // Save report
        const reportPath = path.join(__dirname, 'benchmark_report.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        
        // Save CSV summary
        await this.saveCsvReport(report);
        
        console.log(`\n📄 Detailed benchmark report saved to: ${reportPath}`);
        console.log(`📊 CSV summary saved to: ${path.join(__dirname, 'benchmark_summary.csv')}`);
    }

    /**
     * Calculate average performance metrics
     */
    calculateAveragePerformance() {
        const performanceMetrics = [];
        
        for (const [name, result] of this.results) {
            if (result.stats && name !== 'system_info') {
                performanceMetrics.push({
                    name,
                    mean: result.stats.mean,
                    p95: result.stats.p95,
                    successRate: result.successRate
                });
            }
        }
        
        if (performanceMetrics.length === 0) return null;
        
        return {
            averageMean: performanceMetrics.reduce((sum, m) => sum + m.mean, 0) / performanceMetrics.length,
            averageP95: performanceMetrics.reduce((sum, m) => sum + m.p95, 0) / performanceMetrics.length,
            overallSuccessRate: performanceMetrics.reduce((sum, m) => sum + m.successRate, 0) / performanceMetrics.length
        };
    }

    /**
     * Generate performance recommendations
     */
    generateRecommendations() {
        const recommendations = [];
        const memoryAnalysis = this.analyzeMemoryUsage();
        
        // Memory recommendations
        if (memoryAnalysis && memoryAnalysis.growthRate > 10) {
            recommendations.push({
                type: 'memory',
                severity: 'high',
                message: 'High memory growth rate detected. Consider implementing memory pooling or more aggressive garbage collection.'
            });
        }
        
        // Performance recommendations
        const performanceMetrics = [];
        for (const [name, result] of this.results) {
            if (result.stats && result.stats.mean > 1000) { // > 1 second
                recommendations.push({
                    type: 'performance',
                    severity: 'medium',
                    message: `${name} has high latency (${result.stats.mean.toFixed(2)}ms). Consider optimization.`
                });
            }
            
            if (result.successRate < 90) {
                recommendations.push({
                    type: 'reliability',
                    severity: 'high',
                    message: `${name} has low success rate (${result.successRate.toFixed(1)}%). Investigate error causes.`
                });
            }
        }
        
        // Scalability recommendations
        const scalabilityResults = Array.from(this.results.entries())
            .filter(([name]) => name.includes('scalability'))
            .map(([name, result]) => result);
            
        if (scalabilityResults.length > 0) {
            // Analyze scalability trends
            recommendations.push({
                type: 'scalability',
                severity: 'info',
                message: 'Review scalability test results to identify bottlenecks at higher data volumes.'
            });
        }
        
        return recommendations;
    }

    /**
     * Save CSV summary report
     */
    async saveCsvReport(report) {
        const csvRows = ['Benchmark,Mean(ms),P95(ms),Success Rate(%),Min(ms),Max(ms)'];
        
        for (const [name, result] of Object.entries(report.benchmarks)) {
            if (result.stats) {
                csvRows.push([
                    name,
                    result.stats.mean.toFixed(2),
                    result.stats.p95.toFixed(2),
                    result.successRate.toFixed(1),
                    result.stats.min.toFixed(2),
                    result.stats.max.toFixed(2)
                ].join(','));
            }
        }
        
        const csvPath = path.join(__dirname, 'benchmark_summary.csv');
        fs.writeFileSync(csvPath, csvRows.join('\n'));
    }
}

// Worker thread support for concurrent benchmarks
if (!isMainThread) {
    // Worker thread code
    const { benchmarkName, benchmarkData } = workerData;
    
    async function runWorkerBenchmark() {
        try {
            // Execute benchmark in worker thread
            const startTime = performance.now();
            
            // Simulate benchmark work
            await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
            
            const endTime = performance.now();
            
            parentPort.postMessage({
                success: true,
                duration: endTime - startTime,
                result: benchmarkData
            });
        } catch (error) {
            parentPort.postMessage({
                success: false,
                error: error.message
            });
        }
    }
    
    runWorkerBenchmark();
}

// Export for use in other modules
module.exports = {
    BenchmarkRunner,
    PerformanceMetrics,
    BENCHMARK_CONFIG
};

// Run benchmarks if this file is executed directly
if (require.main === module) {
    const runner = new BenchmarkRunner();
    runner.runAllBenchmarks().catch(error => {
        console.error('Benchmark suite failed:', error);
        process.exit(1);
    });
}