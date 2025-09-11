import { test, expect } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';

test.describe('Dashboard Functionality', () => {
  let dashboardPage: DashboardPage;

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();
  });

  test('should load dashboard with default widgets', async () => {
    await test.step('Verify dashboard loads', async () => {
      await expect(dashboardPage.dashboardTitle).toContainText('Geological Monitoring Dashboard');
      await expect(dashboardPage.widgetContainer).toBeVisible();
    });

    await test.step('Verify default widgets are present', async () => {
      const widgetCount = await dashboardPage.getWidgetCount();
      expect(widgetCount).toBeGreaterThan(0);
    });

    await test.step('Verify data loads in widgets', async () => {
      await dashboardPage.waitForDataLoad();
    });
  });

  test('should allow adding and removing widgets', async () => {
    await test.step('Enable edit mode', async () => {
      await dashboardPage.enableEditMode();
    });

    await test.step('Add a new widget', async () => {
      const initialCount = await dashboardPage.getWidgetCount();
      const newWidget = await dashboardPage.addWidget('statistics');
      
      const finalCount = await dashboardPage.getWidgetCount();
      expect(finalCount).toBe(initialCount + 1);
      
      await expect(newWidget).toBeVisible();
    });

    await test.step('Remove the widget', async () => {
      const widgets = dashboardPage.widgetContainer.locator('[data-widget-type="statistics"]');
      const widget = widgets.last();
      const widgetId = await widget.getAttribute('data-widget-id');
      
      if (widgetId) {
        const initialCount = await dashboardPage.getWidgetCount();
        await dashboardPage.removeWidget(widgetId);
        
        const finalCount = await dashboardPage.getWidgetCount();
        expect(finalCount).toBe(initialCount - 1);
      }
    });
  });

  test('should support drag and drop widget arrangement', async () => {
    await test.step('Enable edit mode', async () => {
      await dashboardPage.enableEditMode();
    });

    await test.step('Move a widget', async () => {
      const widgets = dashboardPage.widgetContainer.locator('[data-testid^="widget-"]');
      const widgetCount = await widgets.count();
      
      if (widgetCount > 0) {
        const firstWidget = widgets.first();
        const widgetId = await firstWidget.getAttribute('data-widget-id');
        
        if (widgetId) {
          await dashboardPage.moveWidget(widgetId, { x: 200, y: 200 });
        }
      }
    });
  });

  test('should save and restore dashboard configuration', async () => {
    await test.step('Enable edit mode and make changes', async () => {
      await dashboardPage.enableEditMode();
      await dashboardPage.addWidget('alert-panel');
    });

    await test.step('Save dashboard', async () => {
      await dashboardPage.saveDashboard();
    });

    await test.step('Refresh and verify changes persist', async () => {
      await dashboardPage.page.reload();
      await dashboardPage.waitForLoad();
      
      // Verify the added widget is still there
      const alertPanels = dashboardPage.widgetContainer.locator('[data-widget-type="alert-panel"]');
      expect(await alertPanels.count()).toBeGreaterThan(0);
    });
  });

  test('should display real-time data updates', async () => {
    await test.step('Verify real-time mode can be enabled', async () => {
      const realTimeToggle = dashboardPage.page.locator('[data-testid="real-time-toggle"]');
      if (await realTimeToggle.isVisible()) {
        await realTimeToggle.check();
        await dashboardPage.verifyRealTimeMode(true);
      }
    });

    await test.step('Verify data refreshes', async () => {
      await dashboardPage.refreshDashboard();
      await dashboardPage.waitForDataLoad();
    });
  });

  test('should be accessible', async () => {
    await test.step('Verify accessibility features', async () => {
      await dashboardPage.verifyAccessibility();
    });

    await test.step('Test keyboard navigation', async () => {
      // Test tab navigation through interactive elements
      await dashboardPage.page.keyboard.press('Tab');
      await dashboardPage.page.keyboard.press('Tab');
      await dashboardPage.page.keyboard.press('Tab');
      
      // Verify focus is visible
      const focusedElement = await dashboardPage.page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    });
  });

  test('should be responsive across different screen sizes', async () => {
    const viewports = [
      { width: 375, height: 667, name: 'Mobile' },
      { width: 768, height: 1024, name: 'Tablet' },
      { width: 1920, height: 1080, name: 'Desktop' }
    ];

    for (const viewport of viewports) {
      await test.step(`Test ${viewport.name} layout`, async () => {
        await dashboardPage.verifyResponsiveLayout(viewport);
      });
    }
  });

  test('should handle errors gracefully', async () => {
    await test.step('Test network error handling', async () => {
      // Simulate network failure
      await dashboardPage.page.route('**/api/**', route => route.abort());
      
      await dashboardPage.refreshDashboard();
      
      // Should show error message
      const errorMessage = dashboardPage.page.locator('[data-testid="error-message"]');
      await expect(errorMessage).toBeVisible();
    });

    await test.step('Test recovery from errors', async () => {
      // Restore network
      await dashboardPage.page.unroute('**/api/**');
      
      await dashboardPage.refreshDashboard();
      await dashboardPage.waitForDataLoad();
      
      // Error message should disappear
      const errorMessage = dashboardPage.page.locator('[data-testid="error-message"]');
      await expect(errorMessage).not.toBeVisible();
    });
  });

  test('should export dashboard data', async () => {
    const formats = ['pdf', 'png', 'json'] as const;
    
    for (const format of formats) {
      await test.step(`Export as ${format}`, async () => {
        const download = await dashboardPage.exportDashboard(format);
        expect(download.suggestedFilename()).toMatch(new RegExp(`\\.${format}$`));
      });
    }
  });

  test('should maintain performance standards', async () => {
    await test.step('Verify performance metrics', async () => {
      await dashboardPage.verifyPerformance();
    });

    await test.step('Test with large datasets', async () => {
      // Add multiple widgets to test performance
      await dashboardPage.enableEditMode();
      
      for (let i = 0; i < 5; i++) {
        await dashboardPage.addWidget('data-chart');
      }
      
      await dashboardPage.waitForDataLoad();
      
      // Verify page remains responsive
      const startTime = Date.now();
      await dashboardPage.refreshDashboard();
      const refreshTime = Date.now() - startTime;
      
      expect(refreshTime).toBeLessThan(5000); // Should refresh within 5 seconds
    });
  });
});

test.describe('Dashboard Visual Regression', () => {
  test('should match visual snapshots', async ({ page }) => {
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();
    await dashboardPage.waitForDataLoad();
    
    // Take full page screenshot
    await expect(page).toHaveScreenshot('dashboard-full-page.png', {
      fullPage: true,
      animations: 'disabled'
    });
    
    // Take widget container screenshot
    await expect(dashboardPage.widgetContainer).toHaveScreenshot('dashboard-widgets.png', {
      animations: 'disabled'
    });
  });

  test('should match mobile layout snapshots', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    
    const dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();
    await dashboardPage.waitForDataLoad();
    
    await expect(page).toHaveScreenshot('dashboard-mobile.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });
});