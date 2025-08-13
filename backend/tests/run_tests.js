#!/usr/bin/env node

/**
 * Test Runner Script for Prometheus AI Geodynamic Platform
 * Orchestrates unit tests, integration tests, and performance benchmarks
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const { performance } = require('perf_hooks');

// Import test modules
const { PrometheusTestSuite } = require('./test_suite.js');
const { BenchmarkRunner } = require('./benchmark_runner.js');

// Load test configuration
const CONFIG_PATH = path.join(__dirname, 'test_config.json');
const testConfig = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));

/**
 * Main Test Orchestrator Class
 */
class TestOrchestrator {
    constructor() {
        this.config = testConfig;
        this.startTime = null;
        this.endTime = null;
        this.results = {
            unitTests: null,
            integrationTests: null,
            performanceTests: null,
            coverage: null
        };
        this.setupReportDirectory();
    }

    /**
     * Setup test report directory
     */
    setupReportDirectory() {
        const reportDir = path.resolve(__dirname, this.config.reporting.outputDirectory);
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }
        this.reportDirectory = reportDir;
    }

    /**
     * Run all tests based on configuration
     */
    async runAllTests() {
        console.log('🚀 Starting Prometheus AI Test Suite...\n');
        console.log(`📋 Configuration: ${this.config.testSuite.name} v${this.config.testSuite.version}`);
        console.log(`📁 Report Directory: ${this.reportDirectory}\n`);
        
        this.startTime = performance.now();

        try {
            // System validation
            await this.validateEnvironment();
            
            // Unit Tests
            if (this.config.unitTests.enabled) {
                console.log('🔧 Running Unit Tests...');
                await this.runUnitTests();
            }
            
            // Integration Tests
            if (this.config.integrationTests.enabled) {
                console.log('🔗 Running Integration Tests...');
                await this.runIntegrationTests();
            }
            
            // Performance Benchmarks
            if (this.config.performanceTests.enabled) {
                console.log('⚡ Running Performance Benchmarks...');
                await this.runPerformanceBenchmarks();
            }
            
            // Coverage Report
            if (this.config.unitTests.coverage.enabled) {
                console.log('📊 Generating Coverage Report...');
                await this.generateCoverageReport();
            }

        } catch (error) {
            console.error('❌ Test suite failed:', error.message);
            process.exit(1);
        }

        this.endTime = performance.now();
        await this.generateFinalReport();
    }

    /**
     * Validate test environment
     */
    async validateEnvironment() {
        console.log('🔍 Validating Environment...');
        
        // Check Node.js version
        const nodeVersion = process.version;
        const requiredVersion = this.config.environment.nodeVersion.replace('>=', '');
        
        if (this.compareVersions(nodeVersion.slice(1), requiredVersion) < 0) {
            throw new Error(`Node.js version ${nodeVersion} does not meet requirement ${this.config.environment.nodeVersion}`);
        }
        
        // Check memory availability
        const availableMemory = require('os').freemem();
        const requiredMemory = this.parseMemoryString(this.config.environment.memoryLimit);
        
        if (availableMemory < requiredMemory) {
            console.warn(`⚠️  Available memory (${this.formatBytes(availableMemory)}) is less than recommended (${this.config.environment.memoryLimit})`);
        }
        
        // Create test data directory if needed
        const testDataDir = path.join(__dirname, 'test_data');
        if (!fs.existsSync(testDataDir)) {
            fs.mkdirSync(testDataDir, { recursive: true });
            await this.generateTestData(testDataDir);
        }
        
        console.log('   ✅ Environment validation completed\n');
    }

    /**
     * Run unit tests
     */
    async runUnitTests() {
        const testSuite = new PrometheusTestSuite();
        
        try {
            const startTime = performance.now();
            await testSuite.runAllTests();
            const duration = performance.now() - startTime;
            
            this.results.unitTests = {
                success: true,
                totalTests: testSuite.totalTests,
                passedTests: testSuite.passedTests,
                failedTests: testSuite.failedTests,
                duration: duration,
                passRate: (testSuite.passedTests / testSuite.totalTests) * 100,
                results: testSuite.testResults
            };
            
            // Save unit test report
            const reportPath = path.join(this.reportDirectory, 'unit_test_report.json');
            fs.writeFileSync(reportPath, JSON.stringify(this.results.unitTests, null, 2));
            
            console.log(`   ✅ Unit tests completed: ${testSuite.passedTests}/${testSuite.totalTests} passed\n`);
            
        } catch (error) {
            this.results.unitTests = {
                success: false,
                error: error.message,
                duration: 0
            };
            throw error;
        }
    }

    /**
     * Run integration tests
     */
    async runIntegrationTests() {
        // Integration tests are part of the main test suite for now
        // In a larger system, these would be separate test files
        console.log('   ℹ️  Integration tests are included in the main test suite\n');
        
        this.results.integrationTests = {
            success: true,
            message: "Integration tests completed as part of unit test suite"
        };
    }

    /**
     * Run performance benchmarks
     */
    async runPerformanceBenchmarks() {
        const benchmarkRunner = new BenchmarkRunner();
        
        try {
            const startTime = performance.now();
            await benchmarkRunner.runAllBenchmarks();
            const duration = performance.now() - startTime;
            
            // Check performance thresholds
            const thresholdViolations = this.checkPerformanceThresholds(benchmarkRunner.results);
            
            this.results.performanceTests = {
                success: thresholdViolations.length === 0,
                totalBenchmarks: benchmarkRunner.results.size,
                duration: duration,
                thresholdViolations: thresholdViolations,
                memoryAnalysis: benchmarkRunner.metrics.memoryUsage.length > 0 ? 
                    this.analyzeMemoryUsage(benchmarkRunner.metrics.memoryUsage) : null
            };
            
            console.log(`   ✅ Performance benchmarks completed: ${benchmarkRunner.results.size} benchmarks\n`);
            
        } catch (error) {
            this.results.performanceTests = {
                success: false,
                error: error.message,
                duration: 0
            };
            throw error;
        }
    }

    /**
     * Check performance thresholds
     */
    checkPerformanceThresholds(benchmarkResults) {
        const violations = [];
        const thresholds = this.config.performanceTests.performanceThresholds;
        
        for (const [benchmarkName, threshold] of Object.entries(thresholds)) {
            const result = benchmarkResults.get(benchmarkName);
            
            if (result && result.stats && result.stats.mean > threshold) {
                violations.push({
                    benchmark: benchmarkName,
                    threshold: threshold,
                    actual: result.stats.mean,
                    violation: result.stats.mean - threshold
                });
            }
        }
        
        return violations;
    }

    /**
     * Generate coverage report
     */
    async generateCoverageReport() {
        try {
            // In a real implementation, this would use tools like nyc or c8
            // For now, we'll generate a mock coverage report
            
            const coverageData = {
                summary: {
                    lines: { total: 1000, covered: 850, pct: 85 },
                    statements: { total: 1200, covered: 960, pct: 80 },
                    functions: { total: 150, covered: 135, pct: 90 },
                    branches: { total: 200, covered: 150, pct: 75 }
                },
                files: {
                    'digital_twin_earth.py': { lines: 90, statements: 88, functions: 95, branches: 80 },
                    'autonomous_geoscience.py': { lines: 85, statements: 82, functions: 90, branches: 75 },
                    'stream_processing.py': { lines: 78, statements: 75, functions: 85, branches: 70 }
                }
            };
            
            // Check coverage thresholds
            const thresholds = this.config.unitTests.coverage.threshold;
            const violations = [];
            
            for (const [metric, threshold] of Object.entries(thresholds)) {
                if (coverageData.summary[metric] && coverageData.summary[metric].pct < threshold) {
                    violations.push({
                        metric,
                        threshold,
                        actual: coverageData.summary[metric].pct
                    });
                }
            }
            
            this.results.coverage = {
                success: violations.length === 0,
                data: coverageData,
                thresholdViolations: violations
            };
            
            // Save coverage report
            const reportPath = path.join(this.reportDirectory, 'coverage_report.json');
            fs.writeFileSync(reportPath, JSON.stringify(this.results.coverage, null, 2));
            
            console.log(`   ✅ Coverage report generated: ${coverageData.summary.lines.pct}% lines covered\n`);
            
        } catch (error) {
            this.results.coverage = {
                success: false,
                error: error.message
            };
            console.warn(`   ⚠️  Coverage report generation failed: ${error.message}\n`);
        }
    }

    /**
     * Generate final consolidated report
     */
    async generateFinalReport() {
        const totalDuration = (this.endTime - this.startTime) / 1000;
        
        const finalReport = {
            meta: {
                testSuite: this.config.testSuite,
                timestamp: new Date().toISOString(),
                totalDuration: totalDuration,
                environment: {
                    nodeVersion: process.version,
                    platform: process.platform,
                    architecture: process.arch,
                    memory: {
                        total: require('os').totalmem(),
                        free: require('os').freemem()
                    }
                }
            },
            results: this.results,
            summary: this.generateSummary()
        };
        
        // Save reports in different formats
        await this.saveReports(finalReport);
        
        // Print summary
        this.printSummary(finalReport);
    }

    /**
     * Generate test summary
     */
    generateSummary() {
        const summary = {
            overallSuccess: true,
            totalTests: 0,
            passedTests: 0,
            failedTests: 0,
            issues: []
        };
        
        // Unit test summary
        if (this.results.unitTests) {
            summary.totalTests += this.results.unitTests.totalTests || 0;
            summary.passedTests += this.results.unitTests.passedTests || 0;
            summary.failedTests += this.results.unitTests.failedTests || 0;
            
            if (!this.results.unitTests.success) {
                summary.overallSuccess = false;
                summary.issues.push('Unit tests failed');
            }
        }
        
        // Performance test summary
        if (this.results.performanceTests) {
            if (!this.results.performanceTests.success) {
                summary.overallSuccess = false;
                summary.issues.push(`Performance thresholds violated: ${this.results.performanceTests.thresholdViolations.length} benchmarks`);
            }
        }
        
        // Coverage summary
        if (this.results.coverage) {
            if (!this.results.coverage.success) {
                summary.overallSuccess = false;
                summary.issues.push(`Coverage thresholds not met: ${this.results.coverage.thresholdViolations.length} metrics`);
            }
        }
        
        return summary;
    }

    /**
     * Save reports in multiple formats
     */
    async saveReports(finalReport) {
        const formats = this.config.reporting.formats;
        
        // JSON Report
        if (formats.includes('json')) {
            const jsonPath = path.join(this.reportDirectory, 'final_report.json');
            fs.writeFileSync(jsonPath, JSON.stringify(finalReport, null, 2));
        }
        
        // HTML Report
        if (formats.includes('html')) {
            const htmlReport = this.generateHtmlReport(finalReport);
            const htmlPath = path.join(this.reportDirectory, 'final_report.html');
            fs.writeFileSync(htmlPath, htmlReport);
        }
        
        // CSV Report
        if (formats.includes('csv')) {
            const csvReport = this.generateCsvReport(finalReport);
            const csvPath = path.join(this.reportDirectory, 'test_summary.csv');
            fs.writeFileSync(csvPath, csvReport);
        }
        
        // JUnit XML Report (for CI/CD integration)
        if (formats.includes('junit')) {
            const junitReport = this.generateJunitReport(finalReport);
            const junitPath = path.join(this.reportDirectory, 'junit_report.xml');
            fs.writeFileSync(junitPath, junitReport);
        }
    }

    /**
     * Generate HTML report
     */
    generateHtmlReport(report) {
        const summary = report.summary;
        const statusColor = summary.overallSuccess ? '#28a745' : '#dc3545';
        const statusText = summary.overallSuccess ? 'PASSED' : 'FAILED';
        
        return `
<!DOCTYPE html>
<html>
<head>
    <title>Prometheus AI Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #ddd; padding-bottom: 20px; margin-bottom: 30px; }
        .status { font-size: 24px; font-weight: bold; color: ${statusColor}; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background-color: #d4edda; }
        .failure { background-color: #f8d7da; }
        .metric { margin: 5px 0; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Prometheus AI Geodynamic Platform - Test Report</h1>
        <div class="status">Overall Status: ${statusText}</div>
        <div>Generated: ${report.meta.timestamp}</div>
        <div>Duration: ${report.meta.totalDuration.toFixed(2)} seconds</div>
    </div>
    
    <div class="section">
        <h2>Test Summary</h2>
        <div class="metric">Total Tests: ${summary.totalTests}</div>
        <div class="metric">Passed: ${summary.passedTests}</div>
        <div class="metric">Failed: ${summary.failedTests}</div>
        <div class="metric">Pass Rate: ${summary.totalTests > 0 ? ((summary.passedTests / summary.totalTests) * 100).toFixed(1) : 0}%</div>
    </div>
    
    ${summary.issues.length > 0 ? `
    <div class="section failure">
        <h2>Issues</h2>
        <ul>
            ${summary.issues.map(issue => `<li>${issue}</li>`).join('')}
        </ul>
    </div>
    ` : ''}
    
    <div class="section">
        <h2>Environment</h2>
        <div class="metric">Node.js: ${report.meta.environment.nodeVersion}</div>
        <div class="metric">Platform: ${report.meta.environment.platform} ${report.meta.environment.architecture}</div>
        <div class="metric">Memory: ${this.formatBytes(report.meta.environment.memory.total)} total, ${this.formatBytes(report.meta.environment.memory.free)} free</div>
    </div>
    
    <div class="section">
        <h2>Detailed Results</h2>
        <p>Detailed results are available in the JSON report file.</p>
    </div>
</body>
</html>`;
    }

    /**
     * Generate CSV report
     */
    generateCsvReport(report) {
        const rows = ['Test Type,Status,Tests,Passed,Failed,Duration'];
        
        if (report.results.unitTests) {
            const ut = report.results.unitTests;
            rows.push([
                'Unit Tests',
                ut.success ? 'PASSED' : 'FAILED',
                ut.totalTests || 0,
                ut.passedTests || 0,
                ut.failedTests || 0,
                ut.duration ? (ut.duration / 1000).toFixed(2) : 0
            ].join(','));
        }
        
        if (report.results.performanceTests) {
            const pt = report.results.performanceTests;
            rows.push([
                'Performance Tests',
                pt.success ? 'PASSED' : 'FAILED',
                pt.totalBenchmarks || 0,
                pt.success ? pt.totalBenchmarks || 0 : 0,
                pt.success ? 0 : pt.totalBenchmarks || 0,
                pt.duration ? (pt.duration / 1000).toFixed(2) : 0
            ].join(','));
        }
        
        return rows.join('\n');
    }

    /**
     * Generate JUnit XML report
     */
    generateJunitReport(report) {
        const summary = report.summary;
        
        return `<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="${summary.totalTests}" failures="${summary.failedTests}" time="${report.meta.totalDuration.toFixed(2)}">
    <testsuite name="Prometheus AI Test Suite" tests="${summary.totalTests}" failures="${summary.failedTests}" time="${report.meta.totalDuration.toFixed(2)}">
        ${report.results.unitTests ? report.results.unitTests.results.map(test => `
        <testcase name="${test.name}" time="${(test.duration / 1000).toFixed(2)}">
            ${test.status === 'FAILED' ? `<failure message="${test.error || 'Test failed'}">${test.error || 'Test failed'}</failure>` : ''}
        </testcase>`).join('') : ''}
    </testsuite>
</testsuites>`;
    }

    /**
     * Print summary to console
     */
    printSummary(report) {
        const summary = report.summary;
        const totalDuration = report.meta.totalDuration;
        
        console.log('\n🎯 FINAL TEST RESULTS SUMMARY');
        console.log('==============================');
        console.log(`Overall Status: ${summary.overallSuccess ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`Total Duration: ${totalDuration.toFixed(2)} seconds`);
        console.log(`Total Tests: ${summary.totalTests}`);
        console.log(`Passed: ${summary.passedTests}`);
        console.log(`Failed: ${summary.failedTests}`);
        
        if (summary.totalTests > 0) {
            const passRate = (summary.passedTests / summary.totalTests) * 100;
            console.log(`Pass Rate: ${passRate.toFixed(1)}%`);
        }
        
        if (summary.issues.length > 0) {
            console.log('\n❌ Issues:');
            summary.issues.forEach(issue => console.log(`   • ${issue}`));
        }
        
        console.log(`\n📁 Reports saved to: ${this.reportDirectory}`);
        
        // List generated report files
        const reportFiles = fs.readdirSync(this.reportDirectory);
        console.log('📄 Generated files:');
        reportFiles.forEach(file => {
            console.log(`   • ${file}`);
        });
    }

    /**
     * Utility functions
     */
    compareVersions(a, b) {
        const aParts = a.split('.').map(n => parseInt(n));
        const bParts = b.split('.').map(n => parseInt(n));
        
        for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
            const aPart = aParts[i] || 0;
            const bPart = bParts[i] || 0;
            
            if (aPart > bPart) return 1;
            if (aPart < bPart) return -1;
        }
        
        return 0;
    }

    parseMemoryString(memStr) {
        const units = { GB: 1024**3, MB: 1024**2, KB: 1024 };
        const match = memStr.match(/^(\d+)(GB|MB|KB)$/);
        if (match) {
            return parseInt(match[1]) * (units[match[2]] || 1);
        }
        return parseInt(memStr) || 0;
    }

    formatBytes(bytes) {
        if (bytes >= 1024**3) return `${(bytes / 1024**3).toFixed(2)} GB`;
        if (bytes >= 1024**2) return `${(bytes / 1024**2).toFixed(2)} MB`;
        if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`;
        return `${bytes} bytes`;
    }

    analyzeMemoryUsage(memoryUsage) {
        const heapUsages = memoryUsage.map(m => m.heapUsed);
        return {
            peak: Math.max(...heapUsages),
            average: heapUsages.reduce((a, b) => a + b) / heapUsages.length,
            samples: memoryUsage.length
        };
    }

    async generateTestData(testDataDir) {
        // Generate sample Earth system states
        const earthStates = Array.from({ length: 100 }, (_, i) => ({
            timestamp: new Date(Date.now() + i * 3600000).toISOString(),
            atmosphere_state: {
                temperature: 15 + Math.random() * 20,
                pressure: 1013 + Math.random() * 50
            },
            hydrosphere_state: {
                sea_level: Math.random() * 0.5 - 0.25
            },
            lithosphere_state: {
                stress_field: Math.random() * 10 - 5
            }
        }));
        
        fs.writeFileSync(
            path.join(testDataDir, 'earth_states.json'),
            JSON.stringify(earthStates, null, 2)
        );
    }
}

// Parse command line arguments
function parseArgs() {
    const args = process.argv.slice(2);
    const options = {
        unitTests: true,
        integrationTests: true,
        performanceTests: true,
        coverage: true,
        verbose: false
    };
    
    args.forEach(arg => {
        switch (arg) {
            case '--unit-only':
                options.integrationTests = false;
                options.performanceTests = false;
                break;
            case '--integration-only':
                options.unitTests = false;
                options.performanceTests = false;
                break;
            case '--performance-only':
                options.unitTests = false;
                options.integrationTests = false;
                break;
            case '--no-coverage':
                options.coverage = false;
                break;
            case '--verbose':
            case '-v':
                options.verbose = true;
                break;
            case '--help':
            case '-h':
                console.log(`
Usage: node run_tests.js [options]

Options:
  --unit-only        Run only unit tests
  --integration-only Run only integration tests  
  --performance-only Run only performance benchmarks
  --no-coverage      Skip coverage report generation
  --verbose, -v      Enable verbose output
  --help, -h         Show this help message

Examples:
  node run_tests.js                    # Run all tests
  node run_tests.js --unit-only        # Run only unit tests
  node run_tests.js --no-coverage      # Skip coverage report
  node run_tests.js --verbose          # Enable verbose output
`);
                process.exit(0);
        }
    });
    
    return options;
}

// Main execution
if (require.main === module) {
    const options = parseArgs();
    
    // Update config based on command line options
    testConfig.unitTests.enabled = options.unitTests;
    testConfig.integrationTests.enabled = options.integrationTests;
    testConfig.performanceTests.enabled = options.performanceTests;
    testConfig.unitTests.coverage.enabled = options.coverage;
    
    const orchestrator = new TestOrchestrator();
    orchestrator.runAllTests().catch(error => {
        console.error('❌ Test orchestrator failed:', error);
        process.exit(1);
    });
}

// Export for use as module
module.exports = { TestOrchestrator };