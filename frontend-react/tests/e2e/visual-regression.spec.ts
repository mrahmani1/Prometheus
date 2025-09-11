import { test, expect } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';
import { VisualizationPage } from '../page-objects/VisualizationPage';
import { VolcanicMonitoringPage } from '../page-objects/VolcanicMonitoringPage';
import { MockApiService } from '../fixtures/mock-services';
import { visualRegressionConfig } from '../fixtures/test-data';

test.describe('Visual Regression Tests', () => {
  let dashboardPage: DashboardPage;
  let visualizationPage: VisualizationPage;
  let monitoringPage: VolcanicMonitoringPage;
  let mockApi: MockApiService;

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page);
    visualizationPage = new VisualizationPage(page);
    monitoringPage = new VolcanicMonitoringPage(page);
    mockApi = new MockApiService(page);

    await mockApi.setupSuccessfulMocks();

    // Ensure user is authenticated
    await page.goto('/dashboard');
    await dashboardPage.ensureAuthenticated();

    // Disable animations for consistent screenshots
    await page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `
    });
  });

  // Test dashboard visual regression across different viewports
  for (const viewport of visualRegressionConfig.viewports) {
    test(`Dashboard visual regression - ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      
      await dashboardPage.goto();
      await dashboardPage.waitForDashboardLoad();

      // Wait for all data to load
      await dashboardPage.waitForDataLoad();

      // Mask dynamic elements
      await page.locator('[data-testid="timestamp"]').evaluateAll(elements => {
        elements.forEach(el => el.style.visibility = 'hidden');
      });
      
      await page.locator('[data-testid="real-time-data"]').evaluateAll(elements => {
        elements.forEach(el => el.style.visibility = 'hidden');
      });

      // Take full page screenshot
      await expect(page).toHaveScreenshot(`dashboard-${viewport.name}.png`, {
        fullPage: true,
        threshold: visualRegressionConfig.threshold,
        animations: 'disabled'
      });

      // Take screenshot of specific components
      await expect(page.locator('[data-testid="volcanic-alerts"]')).toHaveScreenshot(
        `volcanic-alerts-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );

      await expect(page.locator('[data-testid="seismic-activity"]')).toHaveScreenshot(
        `seismic-activity-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );

      await expect(page.locator('[data-testid="statistics-panel"]')).toHaveScreenshot(
        `statistics-panel-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );
    });
  }

  // Test 3D visualization visual regression
  for (const viewport of visualRegressionConfig.viewports) {
    test(`3D Visualization visual regression - ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      
      await visualizationPage.goto();
      await visualizationPage.waitFor3DViewerReady();

      // Wait for 3D models to load completely
      await page.waitForTimeout(3000);

      // Reset camera to default position for consistent screenshots
      const resetCamera = page.locator('[data-testid="reset-camera"]');
      if (await resetCamera.isVisible()) {
        await resetCamera.click();
        await page.waitForTimeout(1000);
      }

      // Take screenshot of 3D viewer
      await expect(page.locator('[data-testid="3d-viewer"]')).toHaveScreenshot(
        `3d-viewer-${viewport.name}.png`,
        { 
          threshold: visualRegressionConfig.threshold,
          animations: 'disabled'
        }
      );

      // Test different geological layers
      const layerTypes = ['geological', 'seismic', 'volcanic', 'structural'];
      
      for (const layerType of layerTypes) {
        const layerToggle = page.locator(`[data-testid="${layerType}-layer-toggle"]`);
        if (await layerToggle.isVisible()) {
          await layerToggle.click();
          await page.waitForTimeout(1000);

          await expect(page.locator('[data-testid="3d-viewer"]')).toHaveScreenshot(
            `3d-viewer-${layerType}-layer-${viewport.name}.png`,
            { threshold: visualRegressionConfig.threshold }
          );
        }
      }

      // Test layer panel
      await expect(page.locator('[data-testid="layer-panel"]')).toHaveScreenshot(
        `layer-panel-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );

      // Test time controls
      const timeControls = page.locator('[data-testid="time-controls"]');
      if (await timeControls.isVisible()) {
        await expect(timeControls).toHaveScreenshot(
          `time-controls-${viewport.name}.png`,
          { threshold: visualRegressionConfig.threshold }
        );
      }
    });
  }

  // Test volcanic monitoring visual regression
  for (const viewport of visualRegressionConfig.viewports) {
    test(`Volcanic Monitoring visual regression - ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      
      await monitoringPage.goto();
      await monitoringPage.waitForMonitoringLoad();

      // Select a specific volcano for consistent testing
      await monitoringPage.selectVolcano('yellowstone');
      await page.waitForTimeout(2000);

      // Mask dynamic timestamps
      await page.locator('[data-testid="last-update"]').evaluateAll(elements => {
        elements.forEach(el => el.style.visibility = 'hidden');
      });

      // Take full page screenshot
      await expect(page).toHaveScreenshot(`volcanic-monitoring-${viewport.name}.png`, {
        fullPage: true,
        threshold: visualRegressionConfig.threshold
      });

      // Test individual monitoring components
      await expect(page.locator('[data-testid="seismic-waveforms"]')).toHaveScreenshot(
        `seismic-waveforms-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );

      await expect(page.locator('[data-testid="deformation-map"]')).toHaveScreenshot(
        `deformation-map-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );

      await expect(page.locator('[data-testid="gas-composition-chart"]')).toHaveScreenshot(
        `gas-composition-chart-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );

      const thermalImagery = page.locator('[data-testid="thermal-imagery"]');
      if (await thermalImagery.isVisible()) {
        await expect(thermalImagery).toHaveScreenshot(
          `thermal-imagery-${viewport.name}.png`,
          { threshold: visualRegressionConfig.threshold }
        );
      }

      // Test alert panels
      await expect(page.locator('[data-testid="alerts-panel"]')).toHaveScreenshot(
        `alerts-panel-${viewport.name}.png`,
        { threshold: visualRegressionConfig.threshold }
      );
    });
  }

  test('Chart and graph visual regression', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Test various chart types
    const chartTypes = [
      'seismic-timeline-chart',
      'deformation-trend-chart',
      'gas-emission-chart',
      'temperature-chart',
      'eruption-probability-chart'
    ];

    for (const chartType of chartTypes) {
      const chart = page.locator(`[data-testid="${chartType}"]`);
      if (await chart.isVisible()) {
        // Wait for chart to render
        await page.waitForTimeout(1000);

        await expect(chart).toHaveScreenshot(
          `${chartType}.png`,
          { threshold: visualRegressionConfig.threshold }
        );
      }
    }

    // Test chart interactions
    const interactiveChart = page.locator('[data-testid="seismic-timeline-chart"]');
    if (await interactiveChart.isVisible()) {
      // Hover over chart to show tooltip
      await interactiveChart.hover();
      await page.waitForTimeout(500);

      const tooltip = page.locator('[data-testid="chart-tooltip"]');
      if (await tooltip.isVisible()) {
        await expect(interactiveChart).toHaveScreenshot(
          'chart-with-tooltip.png',
          { threshold: visualRegressionConfig.threshold }
        );
      }
    }
  });

  test('Map visualization visual regression', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.switchToMapView();
    await page.waitForTimeout(2000);

    // Test base map
    await expect(page.locator('[data-testid="volcano-map"]')).toHaveScreenshot(
      'volcano-map-base.png',
      { threshold: visualRegressionConfig.threshold }
    );

    // Test with different map layers
    const mapLayers = ['satellite', 'terrain', 'geological'];
    
    for (const layer of mapLayers) {
      const layerButton = page.locator(`[data-testid="map-layer-${layer}"]`);
      if (await layerButton.isVisible()) {
        await layerButton.click();
        await page.waitForTimeout(1000);

        await expect(page.locator('[data-testid="volcano-map"]')).toHaveScreenshot(
          `volcano-map-${layer}.png`,
          { threshold: visualRegressionConfig.threshold }
        );
      }
    }

    // Test map with volcano markers
    const volcanoMarkers = page.locator('[data-testid="volcano-marker"]');
    const markerCount = await volcanoMarkers.count();
    
    if (markerCount > 0) {
      await expect(page.locator('[data-testid="volcano-map"]')).toHaveScreenshot(
        'volcano-map-with-markers.png',
        { threshold: visualRegressionConfig.threshold }
      );

      // Click on a marker to show popup
      await volcanoMarkers.first().click();
      await page.waitForTimeout(500);

      const popup = page.locator('[data-testid="volcano-popup"]');
      if (await popup.isVisible()) {
        await expect(page.locator('[data-testid="volcano-map"]')).toHaveScreenshot(
          'volcano-map-with-popup.png',
          { threshold: visualRegressionConfig.threshold }
        );
      }
    }
  });

  test('Alert and notification visual regression', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Test different alert levels
    const alertLevels = ['green', 'yellow', 'orange', 'red'];
    
    for (const level of alertLevels) {
      // Simulate alert of specific level
      await page.evaluate((alertLevel) => {
        const alertElement = document.querySelector('[data-testid="alert-level"]');
        if (alertElement) {
          alertElement.className = `alert-level alert-${alertLevel}`;
          alertElement.textContent = alertLevel.toUpperCase();
        }
      }, level);

      await page.waitForTimeout(500);

      await expect(page.locator('[data-testid="alert-level"]')).toHaveScreenshot(
        `alert-level-${level}.png`,
        { threshold: visualRegressionConfig.threshold }
      );
    }

    // Test notification banner
    const notificationBanner = page.locator('[data-testid="notification-banner"]');
    if (await notificationBanner.isVisible()) {
      await expect(notificationBanner).toHaveScreenshot(
        'notification-banner.png',
        { threshold: visualRegressionConfig.threshold }
      );
    }

    // Test alert history panel
    const alertHistory = page.locator('[data-testid="alert-history"]');
    if (await alertHistory.isVisible()) {
      await expect(alertHistory).toHaveScreenshot(
        'alert-history.png',
        { threshold: visualRegressionConfig.threshold }
      );
    }
  });

  test('Loading states visual regression', async ({ page }) => {
    // Test loading spinners and skeletons
    await page.goto('/dashboard');

    // Capture loading state before data loads
    const loadingSpinner = page.locator('[data-testid="loading-spinner"]');
    if (await loadingSpinner.isVisible()) {
      await expect(loadingSpinner).toHaveScreenshot(
        'loading-spinner.png',
        { threshold: visualRegressionConfig.threshold }
      );
    }

    // Test skeleton loading states
    const skeletonElements = page.locator('[data-testid="skeleton-loader"]');
    const skeletonCount = await skeletonElements.count();
    
    if (skeletonCount > 0) {
      await expect(page).toHaveScreenshot(
        'skeleton-loading-state.png',
        { 
          fullPage: true,
          threshold: visualRegressionConfig.threshold 
        }
      );
    }

    // Wait for full load and compare
    await dashboardPage.waitForDashboardLoad();
    
    await expect(page).toHaveScreenshot(
      'fully-loaded-state.png',
      { 
        fullPage: true,
        threshold: visualRegressionConfig.threshold 
      }
    );
  });

  test('Error states visual regression', async ({ page }) => {
    // Set up error mocks
    await mockApi.setupErrorMocks();

    await page.goto('/dashboard');

    // Wait for error state to appear
    const errorMessage = page.locator('[data-testid="error-message"]');
    await expect(errorMessage).toBeVisible();

    // Test error message display
    await expect(errorMessage).toHaveScreenshot(
      'error-message.png',
      { threshold: visualRegressionConfig.threshold }
    );

    // Test error boundary
    const errorBoundary = page.locator('[data-testid="error-boundary"]');
    if (await errorBoundary.isVisible()) {
      await expect(errorBoundary).toHaveScreenshot(
        'error-boundary.png',
        { threshold: visualRegressionConfig.threshold }
      );
    }

    // Test retry button
    const retryButton = page.locator('[data-testid="retry-button"]');
    if (await retryButton.isVisible()) {
      await expect(retryButton).toHaveScreenshot(
        'retry-button.png',
        { threshold: visualRegressionConfig.threshold }
      );
    }

    // Test network error state
    const networkError = page.locator('[data-testid="network-error"]');
    if (await networkError.isVisible()) {
      await expect(networkError).toHaveScreenshot(
        'network-error.png',
        { threshold: visualRegressionConfig.threshold }
      );
    }
  });

  test('Dark mode visual regression', async ({ page }) => {
    // Enable dark mode
    await page.goto('/dashboard');
    await dashboardPage.ensureAuthenticated();

    const darkModeToggle = page.locator('[data-testid="dark-mode-toggle"]');
    if (await darkModeToggle.isVisible()) {
      await darkModeToggle.click();
      await page.waitForTimeout(1000);

      // Test dashboard in dark mode
      await dashboardPage.waitForDashboardLoad();
      
      await expect(page).toHaveScreenshot(
        'dashboard-dark-mode.png',
        { 
          fullPage: true,
          threshold: visualRegressionConfig.threshold 
        }
      );

      // Test 3D visualization in dark mode
      await visualizationPage.goto();
      await visualizationPage.waitFor3DViewerReady();
      
      await expect(page.locator('[data-testid="3d-viewer"]')).toHaveScreenshot(
        '3d-viewer-dark-mode.png',
        { threshold: visualRegressionConfig.threshold }
      );

      // Test monitoring page in dark mode
      await monitoringPage.goto();
      await monitoringPage.waitForMonitoringLoad();
      
      await expect(page).toHaveScreenshot(
        'monitoring-dark-mode.png',
        { 
          fullPage: true,
          threshold: visualRegressionConfig.threshold 
        }
      );
    }
  });

  test('Print styles visual regression', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Emulate print media
    await page.emulateMedia({ media: 'print' });
    
    await expect(page).toHaveScreenshot(
      'dashboard-print-view.png',
      { 
        fullPage: true,
        threshold: visualRegressionConfig.threshold 
      }
    );

    // Test monitoring page print view
    await monitoringPage.goto();
    await monitoringPage.waitForMonitoringLoad();
    
    await expect(page).toHaveScreenshot(
      'monitoring-print-view.png',
      { 
        fullPage: true,
        threshold: visualRegressionConfig.threshold 
      }
    );
  });
});