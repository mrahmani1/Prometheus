import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

test.describe('Performance Tests', () => {
  test('should meet Core Web Vitals thresholds', async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/dashboard');
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    // Measure Core Web Vitals
    const webVitals = await page.evaluate(() => {
      return new Promise((resolve) => {
        const vitals: any = {};
        
        // Largest Contentful Paint (LCP)
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          vitals.lcp = lastEntry.startTime;
        }).observe({ entryTypes: ['largest-contentful-paint'] });
        
        // First Input Delay (FID) - simulated
        vitals.fid = 0; // Will be measured through interaction
        
        // Cumulative Layout Shift (CLS)
        let clsValue = 0;
        new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              clsValue += (entry as any).value;
            }
          }
          vitals.cls = clsValue;
        }).observe({ entryTypes: ['layout-shift'] });
        
        // First Contentful Paint (FCP)
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          vitals.fcp = entries[0].startTime;
        }).observe({ entryTypes: ['paint'] });
        
        // Time to Interactive (TTI) - approximated
        vitals.tti = performance.now();
        
        setTimeout(() => resolve(vitals), 3000);
      });
    });

    // Assert Core Web Vitals thresholds
    expect(webVitals.lcp).toBeLessThan(2500); // LCP should be < 2.5s
    expect(webVitals.fcp).toBeLessThan(1800); // FCP should be < 1.8s
    expect(webVitals.cls).toBeLessThan(0.1);  // CLS should be < 0.1
    expect(webVitals.tti).toBeLessThan(3800); // TTI should be < 3.8s

    console.log('Core Web Vitals:', webVitals);
  });

  test('should load dashboard within performance budget', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');
    
    const loadTime = Date.now() - startTime;
    
    // Dashboard should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
    
    console.log(`Dashboard load time: ${loadTime}ms`);
  });

  test('should handle large datasets efficiently', async ({ page }) => {
    // Inject large test dataset
    await page.goto('/dashboard');
    
    await page.evaluate(() => {
      // Create large geological dataset
      const largeDataset = {
        geological: Array.from({ length: 10000 }, (_, i) => ({
          id: `geo-${i}`,
          timestamp: new Date(Date.now() - i * 1000).toISOString(),
          data_type: ['igneous', 'sedimentary', 'metamorphic', 'volcanic'][i % 4],
          latitude: -90 + (i % 180),
          longitude: -180 + (i % 360),
          elevation: Math.random() * 5000,
          quality_score: Math.random()
        })),
        seismic: Array.from({ length: 5000 }, (_, i) => ({
          eventId: `seismic-${i}`,
          timestamp: new Date(Date.now() - i * 2000).toISOString(),
          location: { x: -180 + (i % 360), y: -90 + (i % 180), z: Math.random() * 100 },
          magnitude: 1 + Math.random() * 8,
          depth: Math.random() * 700,
          eventType: 'earthquake',
          confidence: Math.random()
        })),
        volcanic: Array.from({ length: 1000 }, (_, i) => ({
          volcanoId: `volcano-${i}`,
          timestamp: new Date(Date.now() - i * 5000).toISOString(),
          alertLevel: ['green', 'yellow', 'orange', 'red'][i % 4],
          magmaChamberPressure: Math.random() * 200,
          deformationRate: Math.random() * 50,
          gasEmissions: {
            co2: 300 + Math.random() * 200,
            so2: Math.random() * 100,
            h2s: Math.random() * 50,
            h2o: 800 + Math.random() * 400,
            temperature: 60 + Math.random() * 40,
            flux: 100 + Math.random() * 200
          },
          thermalAnomalies: [],
          eruptionProbability: Math.random()
        }))
      };
      
      localStorage.setItem('prometheus-large-dataset', JSON.stringify(largeDataset));
    });

    // Measure rendering performance with large dataset
    const startTime = Date.now();
    
    // Trigger data load
    await page.reload();
    await page.waitForSelector('[data-testid="dashboard-title"]');
    
    // Wait for data to be processed
    await page.waitForFunction(() => {
      const widgets = document.querySelectorAll('[data-testid^="widget-"]');
      return Array.from(widgets).some(widget => 
        widget.textContent && widget.textContent.includes('10000')
      );
    }, { timeout: 10000 });
    
    const renderTime = Date.now() - startTime;
    
    // Should handle large dataset within 5 seconds
    expect(renderTime).toBeLessThan(5000);
    
    console.log(`Large dataset render time: ${renderTime}ms`);
  });

  test('should maintain smooth interactions under load', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    // Measure interaction responsiveness
    const interactions = [];
    
    for (let i = 0; i < 10; i++) {
      const startTime = Date.now();
      
      // Perform interaction (button click)
      const addButton = page.locator('[data-testid="add-widget-button"]');
      await addButton.click();
      
      // Wait for response
      const dialog = page.locator('[data-testid="add-widget-dialog"]');
      await dialog.waitFor({ state: 'visible' });
      
      const responseTime = Date.now() - startTime;
      interactions.push(responseTime);
      
      // Close dialog
      await page.keyboard.press('Escape');
      await dialog.waitFor({ state: 'hidden' });
      
      // Small delay between interactions
      await page.waitForTimeout(100);
    }
    
    // Calculate average response time
    const avgResponseTime = interactions.reduce((a, b) => a + b, 0) / interactions.length;
    
    // Interactions should respond within 100ms on average
    expect(avgResponseTime).toBeLessThan(100);
    
    console.log(`Average interaction response time: ${avgResponseTime}ms`);
  });

  test('should efficiently handle memory usage', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    // Measure initial memory usage
    const initialMemory = await page.evaluate(() => {
      return (performance as any).memory ? {
        usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
        totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
        jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
      } : null;
    });

    if (initialMemory) {
      // Perform memory-intensive operations
      for (let i = 0; i < 5; i++) {
        // Add widgets
        const addButton = page.locator('[data-testid="add-widget-button"]');
        await addButton.click();
        
        const dialog = page.locator('[data-testid="add-widget-dialog"]');
        await dialog.waitFor({ state: 'visible' });
        
        const widgetOption = dialog.locator('[data-testid="widget-option-data-chart"]');
        await widgetOption.click();
        
        await dialog.waitFor({ state: 'hidden' });
        await page.waitForTimeout(500);
      }

      // Remove widgets
      const widgets = page.locator('[data-testid^="widget-"]');
      const widgetCount = await widgets.count();
      
      for (let i = 0; i < widgetCount; i++) {
        const widget = widgets.nth(0); // Always remove first widget
        const removeButton = widget.locator('[data-testid="widget-remove-button"]');
        
        if (await removeButton.isVisible()) {
          await removeButton.click();
          await page.waitForTimeout(200);
        }
      }

      // Force garbage collection if available
      await page.evaluate(() => {
        if ((window as any).gc) {
          (window as any).gc();
        }
      });

      // Measure final memory usage
      const finalMemory = await page.evaluate(() => {
        return {
          usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
          totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
          jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
        };
      });

      // Memory usage should not increase significantly
      const memoryIncrease = finalMemory.usedJSHeapSize - initialMemory.usedJSHeapSize;
      const memoryIncreasePercent = (memoryIncrease / initialMemory.usedJSHeapSize) * 100;
      
      // Memory increase should be less than 50%
      expect(memoryIncreasePercent).toBeLessThan(50);
      
      console.log(`Memory increase: ${memoryIncreasePercent.toFixed(2)}%`);
    }
  });

  test('should optimize network requests', async ({ page }) => {
    const requests: any[] = [];
    
    // Monitor network requests
    page.on('request', request => {
      requests.push({
        url: request.url(),
        method: request.method(),
        resourceType: request.resourceType(),
        timestamp: Date.now()
      });
    });

    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');
    await page.waitForLoadState('networkidle');

    // Analyze requests
    const apiRequests = requests.filter(req => req.url.includes('/api/'));
    const staticRequests = requests.filter(req => 
      ['stylesheet', 'script', 'image', 'font'].includes(req.resourceType)
    );

    // Should not make excessive API requests
    expect(apiRequests.length).toBeLessThan(20);
    
    // Should not load excessive static resources
    expect(staticRequests.length).toBeLessThan(50);
    
    // Check for duplicate requests
    const uniqueUrls = new Set(requests.map(req => req.url));
    const duplicateRequests = requests.length - uniqueUrls.size;
    
    // Should minimize duplicate requests
    expect(duplicateRequests).toBeLessThan(5);
    
    console.log(`Total requests: ${requests.length}, API requests: ${apiRequests.length}, Duplicates: ${duplicateRequests}`);
  });

  test('should handle concurrent users simulation', async ({ browser }) => {
    const contexts = [];
    const pages = [];
    
    // Simulate 5 concurrent users
    for (let i = 0; i < 5; i++) {
      const context = await browser.newContext();
      const page = await context.newPage();
      contexts.push(context);
      pages.push(page);
    }

    const startTime = Date.now();
    
    // All users navigate to dashboard simultaneously
    const navigationPromises = pages.map(page => 
      page.goto('/dashboard').then(() => 
        page.waitForSelector('[data-testid="dashboard-title"]')
      )
    );
    
    await Promise.all(navigationPromises);
    
    const loadTime = Date.now() - startTime;
    
    // Should handle concurrent users within reasonable time
    expect(loadTime).toBeLessThan(10000);
    
    // All pages should have loaded successfully
    for (const page of pages) {
      const title = page.locator('[data-testid="dashboard-title"]');
      await expect(title).toBeVisible();
    }
    
    // Cleanup
    for (const context of contexts) {
      await context.close();
    }
    
    console.log(`Concurrent user load time: ${loadTime}ms`);
  });

  test('should generate performance report', async ({ page }) => {
    // This test generates a comprehensive performance report
    const performanceMetrics: any = {};
    
    // Measure page load performance
    const startTime = Date.now();
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');
    performanceMetrics.pageLoadTime = Date.now() - startTime;
    
    // Measure JavaScript execution time
    const jsMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        domComplete: navigation.domComplete - navigation.domLoading,
        loadComplete: navigation.loadEventEnd - navigation.loadEventStart
      };
    });
    
    performanceMetrics.jsMetrics = jsMetrics;
    
    // Measure resource loading
    const resourceMetrics = await page.evaluate(() => {
      const resources = performance.getEntriesByType('resource');
      return {
        totalResources: resources.length,
        totalSize: resources.reduce((sum, resource) => sum + (resource.transferSize || 0), 0),
        slowestResource: Math.max(...resources.map(r => r.duration))
      };
    });
    
    performanceMetrics.resourceMetrics = resourceMetrics;
    
    // Save performance report
    const resultsDir = path.join(process.cwd(), 'test-results', 'performance-results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    
    fs.writeFileSync(
      path.join(resultsDir, `performance-report-${Date.now()}.json`),
      JSON.stringify(performanceMetrics, null, 2)
    );
    
    // Assert performance thresholds
    expect(performanceMetrics.pageLoadTime).toBeLessThan(3000);
    expect(performanceMetrics.jsMetrics.domContentLoaded).toBeLessThan(1000);
    expect(performanceMetrics.resourceMetrics.slowestResource).toBeLessThan(2000);
    
    console.log('Performance Report Generated:', performanceMetrics);
  });
});