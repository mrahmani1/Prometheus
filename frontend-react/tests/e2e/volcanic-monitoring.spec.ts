import { test, expect } from '@playwright/test';
import { VolcanicMonitoringPage } from '../page-objects/VolcanicMonitoringPage';

test.describe('Volcanic Monitoring Interface', () => {
  let volcanicPage: VolcanicMonitoringPage;

  test.beforeEach(async ({ page }) => {
    volcanicPage = new VolcanicMonitoringPage(page);
    await volcanicPage.goto();
  });

  test('should load volcanic monitoring dashboard', async () => {
    await test.step('Verify page loads correctly', async () => {
      await expect(volcanicPage.pageTitle).toContainText('Volcanic Monitoring');
      await expect(volcanicPage.volcanoList).toBeVisible();
      await expect(volcanicPage.statisticsPanel).toBeVisible();
    });

    await test.step('Verify volcano data loads', async () => {
      await volcanicPage.waitForDataLoad();
      
      const volcanoCards = volcanicPage.volcanoList.locator('[data-testid="volcano-card"]');
      const cardCount = await volcanoCards.count();
      expect(cardCount).toBeGreaterThan(0);
    });
  });

  test('should display volcano information correctly', async () => {
    await test.step('Verify volcano card information', async () => {
      const volcanoCards = volcanicPage.volcanoList.locator('[data-testid="volcano-card"]');
      const firstCard = volcanoCards.first();
      
      // Should have alert level
      const alertChip = firstCard.locator('[data-testid="alert-level-chip"]');
      await expect(alertChip).toBeVisible();
      
      // Should have eruption probability
      const probability = firstCard.locator('[data-testid="eruption-probability"]');
      await expect(probability).toBeVisible();
      
      // Should have deformation rate
      const deformation = firstCard.locator('[data-testid="deformation-rate"]');
      await expect(deformation).toBeVisible();
      
      // Should have timestamp
      const timestamp = firstCard.locator('[data-testid="last-updated"]');
      await expect(timestamp).toBeVisible();
    });
  });

  test('should allow volcano selection and show details', async () => {
    await test.step('Select a volcano', async () => {
      const volcanoCards = volcanicPage.volcanoList.locator('[data-testid="volcano-card"]');
      const firstCard = volcanoCards.first();
      const volcanoId = await firstCard.getAttribute('data-volcano-id');
      
      if (volcanoId) {
        await volcanicPage.selectVolcano(volcanoId);
        
        // Verify detail view shows selected volcano
        await expect(volcanicPage.detailView).toContainText(volcanoId);
      }
    });

    await test.step('Verify detailed information is displayed', async () => {
      // Should show gas composition
      const gasComposition = volcanicPage.detailView.locator('[data-testid="gas-composition"]');
      await expect(gasComposition).toBeVisible();
      
      // Should show thermal anomalies
      const thermalAnomalies = volcanicPage.detailView.locator('[data-testid="thermal-anomalies"]');
      await expect(thermalAnomalies).toBeVisible();
      
      // Should show charts
      const charts = volcanicPage.detailView.locator('[data-testid="volcano-charts"]');
      await expect(charts).toBeVisible();
    });
  });

  test('should filter volcanoes by alert level', async () => {
    const alertLevels = ['green', 'yellow', 'orange', 'red'] as const;
    
    for (const level of alertLevels) {
      await test.step(`Filter by ${level} alert level`, async () => {
        await volcanicPage.filterByAlertLevel(level);
        
        // Verify only volcanoes with selected alert level are visible
        const visibleCards = volcanicPage.volcanoList.locator('[data-testid="volcano-card"]:visible');
        const cardCount = await visibleCards.count();
        
        if (cardCount > 0) {
          // Check first visible card has correct alert level
          const firstCard = visibleCards.first();
          const alertChip = firstCard.locator('[data-testid="alert-level-chip"]');
          await expect(alertChip).toContainText(level);
        }
      });
    }
  });

  test('should switch between different view modes', async () => {
    await test.step('Switch to map view', async () => {
      await volcanicPage.switchToMapView();
      await expect(volcanicPage.mapView).toBeVisible();
    });

    await test.step('Verify map functionality', async () => {
      await volcanicPage.verifyMapInteraction();
    });

    await test.step('Switch back to detail view', async () => {
      await volcanicPage.switchToDetailView();
      await expect(volcanicPage.detailView).toBeVisible();
    });
  });

  test('should handle real-time mode', async () => {
    await test.step('Enable real-time mode', async () => {
      await volcanicPage.enableRealTimeMode();
    });

    await test.step('Verify real-time updates', async () => {
      // Wait for potential updates
      await volcanicPage.page.waitForTimeout(2000);
      
      // Data should still be visible and valid
      await volcanicPage.waitForDataLoad();
    });

    await test.step('Disable real-time mode', async () => {
      await volcanicPage.disableRealTimeMode();
    });
  });

  test('should display accurate statistics', async () => {
    await test.step('Verify statistics panel', async () => {
      await volcanicPage.verifyStatistics({
        totalVolcanoes: 1, // Based on test data
        activeAlerts: 1,
        criticalAlerts: 0,
        averageRisk: 15 // Based on test data
      });
    });
  });

  test('should show alerts correctly', async () => {
    await test.step('Verify alerts panel', async () => {
      await volcanicPage.verifyAlerts();
    });

    await test.step('Test alert interactions', async () => {
      const alertItems = volcanicPage.alertsPanel.locator('[data-testid="alert-item"]');
      const alertCount = await alertItems.count();
      
      if (alertCount > 0) {
        const firstAlert = alertItems.first();
        await firstAlert.click();
        
        // Should show alert details or navigate to volcano
        // Implementation depends on specific behavior
      }
    });
  });

  test('should export volcanic data', async () => {
    const formats = ['csv', 'json', 'pdf'] as const;
    
    for (const format of formats) {
      await test.step(`Export as ${format}`, async () => {
        const download = await volcanicPage.exportData(format);
        expect(download.suggestedFilename()).toMatch(new RegExp(`\\.${format}$`));
      });
    }
  });

  test('should be accessible', async () => {
    await test.step('Verify accessibility features', async () => {
      await volcanicPage.verifyAccessibility();
    });

    await test.step('Test screen reader compatibility', async () => {
      // Verify ARIA labels are present
      const volcanoCards = volcanicPage.volcanoList.locator('[data-testid="volcano-card"]');
      const firstCard = volcanoCards.first();
      
      const ariaLabel = await firstCard.getAttribute('aria-label');
      expect(ariaLabel).toBeTruthy();
      expect(ariaLabel).toContain('volcano');
    });
  });

  test('should be responsive', async () => {
    await test.step('Test responsive design', async () => {
      await volcanicPage.verifyResponsiveDesign();
    });
  });

  test('should handle data refresh', async () => {
    await test.step('Refresh data', async () => {
      await volcanicPage.refreshData();
    });

    await test.step('Verify data is still valid after refresh', async () => {
      await volcanicPage.waitForDataLoad();
      
      const volcanoCards = volcanicPage.volcanoList.locator('[data-testid="volcano-card"]');
      const cardCount = await volcanoCards.count();
      expect(cardCount).toBeGreaterThan(0);
    });
  });

  test('should handle error states gracefully', async () => {
    await test.step('Simulate network error', async () => {
      await volcanicPage.page.route('**/api/volcanic/**', route => route.abort());
      
      await volcanicPage.refreshData();
      
      // Should show error message
      const errorMessage = volcanicPage.page.locator('[data-testid="error-message"]');
      await expect(errorMessage).toBeVisible();
    });

    await test.step('Test recovery from error', async () => {
      await volcanicPage.page.unroute('**/api/volcanic/**');
      
      await volcanicPage.refreshData();
      await volcanicPage.waitForDataLoad();
      
      // Error should be cleared
      const errorMessage = volcanicPage.page.locator('[data-testid="error-message"]');
      await expect(errorMessage).not.toBeVisible();
    });
  });
});

test.describe('Volcanic Monitoring Visual Regression', () => {
  test('should match visual snapshots', async ({ page }) => {
    const volcanicPage = new VolcanicMonitoringPage(page);
    await volcanicPage.goto();
    await volcanicPage.waitForDataLoad();
    
    // Full page screenshot
    await expect(page).toHaveScreenshot('volcanic-monitoring-full.png', {
      fullPage: true,
      animations: 'disabled'
    });
    
    // Volcano list screenshot
    await expect(volcanicPage.volcanoList).toHaveScreenshot('volcano-list.png', {
      animations: 'disabled'
    });
    
    // Statistics panel screenshot
    await expect(volcanicPage.statisticsPanel).toHaveScreenshot('volcanic-statistics.png', {
      animations: 'disabled'
    });
  });

  test('should match map view snapshots', async ({ page }) => {
    const volcanicPage = new VolcanicMonitoringPage(page);
    await volcanicPage.goto();
    await volcanicPage.switchToMapView();
    
    await expect(volcanicPage.mapView).toHaveScreenshot('volcanic-map-view.png', {
      animations: 'disabled'
    });
  });

  test('should match alert level filter snapshots', async ({ page }) => {
    const volcanicPage = new VolcanicMonitoringPage(page);
    await volcanicPage.goto();
    await volcanicPage.waitForDataLoad();
    
    // Test each alert level filter
    const alertLevels = ['green', 'yellow', 'orange', 'red'] as const;
    
    for (const level of alertLevels) {
      await volcanicPage.filterByAlertLevel(level);
      await expect(volcanicPage.volcanoList).toHaveScreenshot(`volcanic-filter-${level}.png`, {
        animations: 'disabled'
      });
    }
  });
});