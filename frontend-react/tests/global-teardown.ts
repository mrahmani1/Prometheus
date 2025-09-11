import { FullConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global teardown for Prometheus AI Platform tests...');
  
  try {
    // Clean up test artifacts
    const testResultsDir = path.join(process.cwd(), 'test-results');
    const screenshotsDir = path.join(process.cwd(), 'test-results', 'screenshots');
    const videosDir = path.join(process.cwd(), 'test-results', 'videos');
    
    // Generate test summary
    const resultsFile = path.join(testResultsDir, 'results.json');
    if (fs.existsSync(resultsFile)) {
      const results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
      
      console.log('📊 Test Summary:');
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
    
    // Clean up old screenshots (keep only last 10 runs)
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
    }
    
    // Clean up old videos (keep only last 5 runs)
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
    }
    
    // Generate accessibility report if axe results exist
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
        }
      }
    }
    
    // Generate performance report if lighthouse results exist
    const lighthouseResultsDir = path.join(testResultsDir, 'lighthouse-results');
    if (fs.existsSync(lighthouseResultsDir)) {
      const lighthouseFiles = fs.readdirSync(lighthouseResultsDir).filter(file => file.endsWith('.json'));
      
      if (lighthouseFiles.length > 0) {
        console.log('⚡ Performance Summary:');
        let totalScore = 0;
        
        lighthouseFiles.forEach(file => {
          const results = JSON.parse(fs.readFileSync(path.join(lighthouseResultsDir, file), 'utf8'));
          const performanceScore = results.lhr?.categories?.performance?.score || 0;
          totalScore += performanceScore;
          console.log(`   ${file}: ${Math.round(performanceScore * 100)}/100`);
        });
        
        const avgScore = totalScore / lighthouseFiles.length;
        console.log(`   Average performance score: ${Math.round(avgScore * 100)}/100`);
        
        if (avgScore < 0.8) {
          console.log('⚠️  Performance below threshold. Check lighthouse-results for details.');
        }
      }
    }
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw error in teardown to avoid masking test failures
  }
}

export default globalTeardown;