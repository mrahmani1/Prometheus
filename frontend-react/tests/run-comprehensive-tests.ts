#!/usr/bin/env node

/**
 * Comprehensive test runner for Prometheus AI Platform
 * Executes all test suites with proper sequencing and reporting
 */

import { execSync } from 'child_process';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';

interface TestSuite {
  name: string;
  file: string;
  description: string;
  timeout: number;
  critical: boolean;
}

interface TestResult {
  suite: string;
  passed: boolean;
  duration: number;
  error?: string;
}

class ComprehensiveTestRunner {
  private testSuites: TestSuite[] = [
    {
      name: 'User Workflows',
      file: 'comprehensive-user-workflows.spec.ts',
      description: 'Complete user journey testing across all roles',
      timeout: 300000, // 5 minutes
      critical: true
    },
    {
      name: 'Real-time Data Streaming',
      file: 'real-time-data-streaming.spec.ts',
      description: 'WebSocket connections and real-time data validation',
      timeout: 180000, // 3 minutes
      critical: true
    },
    {
      name: 'Geological Calculations',
      file: 'geological-calculation-accuracy.spec.ts',
      description: 'Scientific accuracy of geological computations',
      timeout: 240000, // 4 minutes
      critical: true
    },
    {
      name: 'Visual Regression',
      file: 'visual-regression.spec.ts',
      description: 'UI consistency and visual validation',
      timeout: 360000, // 6 minutes
      critical: false
    },
    {
      name: 'Mobile Responsive',
      file: 'mobile-responsive.spec.ts',
      description: 'Mobile and tablet device compatibility',
      timeout: 240000, // 4 minutes
      critical: false
    },
    {
      name: 'Accessibility',
      file: 'accessibility.spec.ts',
      description: 'WCAG compliance and accessibility features',
      timeout: 180000, // 3 minutes
      critical: false
    },
    {
      name: 'Performance',
      file: 'performance.spec.ts',
      description: 'Load times and rendering performance',
      timeout: 300000, // 5 minutes
      critical: false
    }
  ];

  private results: TestResult[] = [];
  private startTime: number = 0;

  constructor() {
    this.setupTestEnvironment();
  }

  private setupTestEnvironment(): void {
    console.log('🌋 Setting up Prometheus AI Platform test environment...');
    
    // Create test results directory
    const testResultsDir = join(process.cwd(), 'test-results');
    if (!existsSync(testResultsDir)) {
      mkdirSync(testResultsDir, { recursive: true });
    }

    // Create subdirectories
    const subdirs = ['screenshots', 'videos', 'traces', 'reports', 'coverage'];
    subdirs.forEach(dir => {
      const dirPath = join(testResultsDir, dir);
      if (!existsSync(dirPath)) {
        mkdirSync(dirPath, { recursive: true });
      }
    });

    console.log('📁 Test directories created');
  }

  private async runTestSuite(suite: TestSuite): Promise<TestResult> {
    console.log(`\n🧪 Running ${suite.name} tests...`);
    console.log(`📝 ${suite.description}`);
    
    const startTime = Date.now();
    
    try {
      const command = `npx playwright test tests/e2e/${suite.file} --timeout=${suite.timeout}`;
      
      console.log(`⚡ Executing: ${command}`);
      
      execSync(command, {
        stdio: 'inherit',
        timeout: suite.timeout + 30000 // Add 30s buffer
      });
      
      const duration = Date.now() - startTime;
      console.log(`✅ ${suite.name} completed in ${(duration / 1000).toFixed(2)}s`);
      
      return {
        suite: suite.name,
        passed: true,
        duration
      };
      
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error(`❌ ${suite.name} failed after ${(duration / 1000).toFixed(2)}s`);
      console.error(`Error: ${error}`);
      
      return {
        suite: suite.name,
        passed: false,
        duration,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  private generateReport(): void {
    console.log('\n📊 Generating comprehensive test report...');
    
    const totalDuration = Date.now() - this.startTime;
    const passedTests = this.results.filter(r => r.passed).length;
    const failedTests = this.results.filter(r => !r.passed).length;
    const criticalFailures = this.results.filter(r => !r.passed && 
      this.testSuites.find(s => s.name === r.suite)?.critical).length;
    
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalSuites: this.testSuites.length,
        passed: passedTests,
        failed: failedTests,
        criticalFailures,
        totalDuration: Math.round(totalDuration / 1000),
        successRate: Math.round((passedTests / this.testSuites.length) * 100)
      },
      results: this.results.map(result => ({
        ...result,
        critical: this.testSuites.find(s => s.name === result.suite)?.critical || false,
        durationSeconds: Math.round(result.duration / 1000)
      })),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        timestamp: new Date().toISOString()
      },
      recommendations: this.generateRecommendations()
    };
    
    // Write report to file
    const reportPath = join(process.cwd(), 'test-results', 'reports', 'comprehensive-test-report.json');
    writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    // Display summary
    console.log('\n📋 Test Execution Summary:');
    console.log(`   Total test suites: ${this.testSuites.length}`);
    console.log(`   Passed: ${passedTests} ✅`);
    console.log(`   Failed: ${failedTests} ❌`);
    console.log(`   Critical failures: ${criticalFailures} 🚨`);
    console.log(`   Success rate: ${report.summary.successRate}%`);
    console.log(`   Total duration: ${Math.round(totalDuration / 1000)}s`);
    
    if (failedTests > 0) {
      console.log('\n❌ Failed test suites:');
      this.results.filter(r => !r.passed).forEach(result => {
        const suite = this.testSuites.find(s => s.name === result.suite);
        const criticalFlag = suite?.critical ? ' 🚨 CRITICAL' : '';
        console.log(`   - ${result.suite}${criticalFlag}`);
        if (result.error) {
          console.log(`     Error: ${result.error}`);
        }
      });
    }
    
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
  }

  private generateRecommendations(): string[] {
    const recommendations: string[] = [];
    
    const failedCritical = this.results.filter(r => !r.passed && 
      this.testSuites.find(s => s.name === r.suite)?.critical);
    
    if (failedCritical.length > 0) {
      recommendations.push('🚨 Critical test failures detected - investigate immediately before deployment');
    }
    
    const slowTests = this.results.filter(r => r.duration > 180000); // > 3 minutes
    if (slowTests.length > 0) {
      recommendations.push('⏱️ Some tests are running slowly - consider optimization or parallel execution');
    }
    
    const failedVisual = this.results.find(r => r.suite === 'Visual Regression' && !r.passed);
    if (failedVisual) {
      recommendations.push('🎨 Visual regression failures may indicate UI changes - review screenshots');
    }
    
    const failedPerformance = this.results.find(r => r.suite === 'Performance' && !r.passed);
    if (failedPerformance) {
      recommendations.push('⚡ Performance test failures - check load times and memory usage');
    }
    
    const failedAccessibility = this.results.find(r => r.suite === 'Accessibility' && !r.passed);
    if (failedAccessibility) {
      recommendations.push('♿ Accessibility failures - ensure WCAG compliance before release');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('✅ All tests passed - system is ready for deployment');
    }
    
    return recommendations;
  }

  public async runAllTests(): Promise<void> {
    console.log('🚀 Starting comprehensive Prometheus AI Platform test execution...');
    console.log(`📋 Running ${this.testSuites.length} test suites\n`);
    
    this.startTime = Date.now();
    
    // Run critical tests first
    const criticalSuites = this.testSuites.filter(s => s.critical);
    const nonCriticalSuites = this.testSuites.filter(s => !s.critical);
    
    console.log('🚨 Running critical test suites first...');
    
    for (const suite of criticalSuites) {
      const result = await this.runTestSuite(suite);
      this.results.push(result);
      
      // Stop if critical test fails
      if (!result.passed) {
        console.log(`\n🛑 Critical test suite "${suite.name}" failed. Stopping execution.`);
        console.log('🔧 Please fix critical issues before running remaining tests.');
        break;
      }
    }
    
    // Only run non-critical tests if all critical tests passed
    const criticalPassed = this.results.every(r => r.passed);
    
    if (criticalPassed && nonCriticalSuites.length > 0) {
      console.log('\n✅ All critical tests passed. Running additional test suites...');
      
      for (const suite of nonCriticalSuites) {
        const result = await this.runTestSuite(suite);
        this.results.push(result);
      }
    } else if (!criticalPassed) {
      console.log('\n⚠️ Skipping non-critical tests due to critical failures.');
    }
    
    // Generate final report
    this.generateReport();
    
    // Exit with appropriate code
    const hasFailures = this.results.some(r => !r.passed);
    const hasCriticalFailures = this.results.some(r => !r.passed && 
      this.testSuites.find(s => s.name === r.suite)?.critical);
    
    if (hasCriticalFailures) {
      console.log('\n🚨 Critical test failures detected - deployment not recommended');
      process.exit(1);
    } else if (hasFailures) {
      console.log('\n⚠️ Some tests failed but no critical issues - review before deployment');
      process.exit(1);
    } else {
      console.log('\n🎉 All tests passed successfully - system ready for deployment!');
      process.exit(0);
    }
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const runner = new ComprehensiveTestRunner();
  runner.runAllTests().catch(error => {
    console.error('💥 Test runner failed:', error);
    process.exit(1);
  });
}

export default ComprehensiveTestRunner;