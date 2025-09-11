import { FullConfig } from '@playwright/test';

/**
 * Global teardown for Playwright tests
 * Cleans up test environment and generates final reports
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting Prometheus AI Platform test cleanup...');
  
  try {
    // Generate test summary report
    console.log('📊 Generating test summary report...');
    
    const fs = require('fs');
    const path = require('path');
    
    const testResultsDir = path.join(process.cwd(), 'test-results');
    const reportsDir = path.join(testResultsDir, 'reports');
    
    // Create test summary
    const testSummary = {
      timestamp: new Date().toISOString(),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch
      },
      configuration: {
        baseURL: config.projects[0]?.use?.baseURL || 'http://localhost:5173',
        browsers: config.projects.map(p => p.name),
        workers: config.workers,
        retries: config.retries
      },
      testCategories: [
        'comprehensive-user-workflows',
        'real-time-data-streaming',
        'geological-calculation-accuracy',
        'visual-regression',
        'mobile-responsive',
        'accessibility',
        'performance'
      ],
      cleanup: {
        screenshotsGenerated: 0,
        tracesGenerated: 0,
        reportsGenerated: 0
      }
    };
    
    // Count generated artifacts
    if (fs.existsSync(path.join(testResultsDir, 'screenshots'))) {
      const screenshots = fs.readdirSync(path.join(testResultsDir, 'screenshots'));
      testSummary.cleanup.screenshotsGenerated = screenshots.length;
    }
    
    if (fs.existsSync(path.join(testResultsDir, 'traces'))) {
      const traces = fs.readdirSync(path.join(testResultsDir, 'traces'));
      testSummary.cleanup.tracesGenerated = traces.length;
    }
    
    if (fs.existsSync(reportsDir)) {
      const reports = fs.readdirSync(reportsDir);
      testSummary.cleanup.reportsGenerated = reports.length;
    }
    
    // Write test summary
    const summaryPath = path.join(reportsDir, 'test-summary.json');
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }
    fs.writeFileSync(summaryPath, JSON.stringify(testSummary, null, 2));
    
    console.log('📋 Test summary written to:', summaryPath);
    
    // Process existing test results
    const resultsFile = path.join(testResultsDir, 'results.json');
    if (fs.existsSync(resultsFile)) {
      const results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
      
      console.log('📊 Test Execution Summary:');
      console.log(`   Total tests: ${results.stats?.total || 0}`);
      console.log(`   Passed: ${results.stats?.passed || 0}`);
      console.log(`   Failed: ${results.stats?.failed || 0}`);
      console.log(`   Skipped: ${results.stats?.skipped || 0}`);
      
      // Log failed tests
      if (results.stats?.failed > 0) {
        console.log('❌ Failed tests:');
        results.suites?.forEach((suite: any) => {
          suite.specs?.forEach((spec: any) => {
            spec.tests?.forEach((test: any) => {
              if (test.results?.some((result: any) => result.status === 'failed')) {
                console.log(`   - ${suite.title}: ${spec.title}`);
              }
            });
          });
        });
      }
    }
    
    // Clean up temporary files
    console.log('🗑️ Cleaning up temporary files...');
    
    const tempDirs = [
      path.join(testResultsDir, 'temp'),
      path.join(testResultsDir, 'cache')
    ];
    
    for (const tempDir of tempDirs) {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
        console.log(`🗑️ Cleaned up: ${tempDir}`);
      }
    }
    
    // Archive old test results (keep last 5 runs)
    console.log('📦 Archiving old test results...');
    
    const archiveDir = path.join(testResultsDir, 'archive');
    if (!fs.existsSync(archiveDir)) {
      fs.mkdirSync(archiveDir, { recursive: true });
    }
    
    // Clean up old screenshots (keep only last 50)
    const screenshotsDir = path.join(testResultsDir, 'screenshots');
    if (fs.existsSync(screenshotsDir)) {
      const screenshots = fs.readdirSync(screenshotsDir)
        .filter(file => file.endsWith('.png'))
        .sort((a, b) => {
          const statA = fs.statSync(path.join(screenshotsDir, a));
          const statB = fs.statSync(path.join(screenshotsDir, b));
          return statB.mtime.getTime() - statA.mtime.getTime();
        });
      
      // Remove old screenshots
      screenshots.slice(50).forEach(file => {
        fs.unlinkSync(path.join(screenshotsDir, file));
      });
      
      console.log(`🖼️ Cleaned up ${Math.max(0, screenshots.length - 50)} old screenshots`);
    }
    
    // Clean up old videos (keep only last 10)
    const videosDir = path.join(testResultsDir, 'videos');
    if (fs.existsSync(videosDir)) {
      const videos = fs.readdirSync(videosDir)
        .filter(file => file.endsWith('.webm'))
        .sort((a, b) => {
          const statA = fs.statSync(path.join(videosDir, a));
          const statB = fs.statSync(path.join(videosDir, b));
          return statB.mtime.getTime() - statA.mtime.getTime();
        });
      
      // Remove old videos
      videos.slice(10).forEach(file => {
        fs.unlinkSync(path.join(videosDir, file));
      });
      
      console.log(`🎥 Cleaned up ${Math.max(0, videos.length - 10)} old videos`);
    }
    
    // Generate accessibility report summary
    console.log('♿ Processing accessibility results...');
    
    const axeResultsDir = path.join(testResultsDir, 'axe-results');
    if (fs.existsSync(axeResultsDir)) {
      const axeFiles = fs.readdirSync(axeResultsDir).filter(file => file.endsWith('.json'));
      
      if (axeFiles.length > 0) {
        console.log('♿ Accessibility Summary:');
        let totalViolations = 0;
        let totalPasses = 0;
        
        axeFiles.forEach(file => {
          const results = JSON.parse(fs.readFileSync(path.join(axeResultsDir, file), 'utf8'));
          totalViolations += results.violations?.length || 0;
          totalPasses += results.passes?.length || 0;
        });
        
        console.log(`   Total violations: ${totalViolations}`);
        console.log(`   Total passes: ${totalPasses}`);
        
        if (totalViolations > 0) {
          console.log('⚠️  Accessibility violations found. Check axe-results for details.');
        } else {
          console.log('✅ No accessibility violations found');
        }
        
        // Write accessibility summary
        const accessibilityReport = {
          timestamp: new Date().toISOString(),
          totalViolations,
          totalPasses,
          wcagCompliance: totalViolations === 0 ? 'PASSED' : 'FAILED',
          filesProcessed: axeFiles.length
        };
        
        const accessibilityReportPath = path.join(reportsDir, 'accessibility-summary.json');
        fs.writeFileSync(accessibilityReportPath, JSON.stringify(accessibilityReport, null, 2));
      }
    }
    
    // Generate performance report summary
    console.log('⚡ Processing performance results...');
    
    const lighthouseResultsDir = path.join(testResultsDir, 'lighthouse-results');
    if (fs.existsSync(lighthouseResultsDir)) {
      const lighthouseFiles = fs.readdirSync(lighthouseResultsDir).filter(file => file.endsWith('.json'));
      
      if (lighthouseFiles.length > 0) {
        console.log('⚡ Performance Summary:');
        let totalScore = 0;
        const scores: any[] = [];
        
        lighthouseFiles.forEach(file => {
          const results = JSON.parse(fs.readFileSync(path.join(lighthouseResultsDir, file), 'utf8'));
          const performanceScore = results.lhr?.categories?.performance?.score || 0;
          totalScore += performanceScore;
          scores.push({ file, score: Math.round(performanceScore * 100) });
          console.log(`   ${file}: ${Math.round(performanceScore * 100)}/100`);
        });
        
        const avgScore = totalScore / lighthouseFiles.length;
        console.log(`   Average performance score: ${Math.round(avgScore * 100)}/100`);
        
        if (avgScore < 0.8) {
          console.log('⚠️  Performance below threshold. Check lighthouse-results for details.');
        } else {
          console.log('✅ Performance meets requirements');
        }
        
        // Write performance summary
        const performanceReport = {
          timestamp: new Date().toISOString(),
          averageScore: Math.round(avgScore * 100),
          scores,
          threshold: 80,
          passed: avgScore >= 0.8,
          filesProcessed: lighthouseFiles.length
        };
        
        const performanceReportPath = path.join(reportsDir, 'performance-summary.json');
        fs.writeFileSync(performanceReportPath, JSON.stringify(performanceReport, null, 2));
      }
    }
    
    // Log final statistics
    console.log('📈 Final test execution statistics:');
    console.log(`   Screenshots generated: ${testSummary.cleanup.screenshotsGenerated}`);
    console.log(`   Traces generated: ${testSummary.cleanup.tracesGenerated}`);
    console.log(`   Reports generated: ${testSummary.cleanup.reportsGenerated}`);
    
    // Display important file locations
    console.log('📁 Important file locations:');
    console.log(`   Test results: ${testResultsDir}`);
    console.log(`   Reports: ${reportsDir}`);
    console.log(`   Screenshots: ${path.join(testResultsDir, 'screenshots')}`);
    console.log(`   Videos: ${path.join(testResultsDir, 'videos')}`);
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw error in teardown to avoid masking test failures
  }
  
  console.log('🏁 Prometheus AI Platform testing session complete!');
}

export default globalTeardown;