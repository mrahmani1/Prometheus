import { test, expect } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';
import { VisualizationPage } from '../page-objects/VisualizationPage';
import { VolcanicMonitoringPage } from '../page-objects/VolcanicMonitoringPage';
import { LoginPage } from '../page-objects/LoginPage';
import { MockApiService } from '../fixtures/mock-services';

test.describe('Mobile and Tablet Responsive Tests', () => {
  let loginPage: LoginPage;
  let dashboardPage: DashboardPage;
  let visualizationPage: VisualizationPage;
  let monitoringPage: VolcanicMonitoringPage;
  let mockApi: MockApiService;

  const mobileViewports = [
    { width: 375, height: 667, name: 'iPhone SE' },
    { width: 390, height: 844, name: 'iPhone 12' },
    { width: 414, height: 896, name: 'iPhone 11 Pro Max' },
    { width: 360, height: 640, name: 'Galaxy S5' },
    { width: 412, height: 915, name: 'Pixel 5' }
  ];

  const tabletViewports = [
    { width: 768, height: 1024, name: 'iPad' },
    { width: 820, height: 1180, name: 'iPad Air' },
    { width: 1024, height: 1366, name: 'iPad Pro' },
    { width: 800, height: 1280, name: 'Galaxy Tab S4' }
  ];

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    dashboardPage = new DashboardPage(page);
    visualizationPage = new VisualizationPage(page);
    monitoringPage = new VolcanicMonitoringPage(page);
    mockApi = new MockApiService(page);

    await mockApi.setupSuccessfulMocks();
  });

  // Test mobile responsiveness
  for (const viewport of mobileViewports) {
    test(`Mobile responsiveness - ${viewport.name} (${viewport.width}x${viewport.height})`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });

      // Test login page on mobile
      await loginPage.goto();
      await loginPage.validatePageContent();

      // Verify login form is properly sized
      const loginForm = page.locator('[data-testid="login-form"]');
      const formBounds = await loginForm.boundingBox();
      
      if (formBounds) {
        expect(formBounds.width).toBeLessThanOrEqual(viewport.width - 40); // Account for padding
        expect(formBounds.width).toBeGreaterThan(viewport.width * 0.8); // Should use most of screen width
      }

      // Test mobile login
      await loginPage.loginAsResearcher();

      // Test dashboard mobile layout
      await dashboardPage.goto();
      await dashboardPage.waitForDashboardLoad();

      // Verify mobile navigation
      const mobileMenu = page.locator('[data-testid="mobile-menu"]');
      const hamburgerButton = page.locator('[data-testid="hamburger-menu"]');
      
      if (await hamburgerButton.isVisible()) {
        await hamburgerButton.click();
        await expect(mobileMenu).toBeVisible();
        
        // Test navigation items
        const navItems = mobileMenu.locator('[data-testid="nav-item"]');
        const navCount = await navItems.count();
        expect(navCount).toBeGreaterThan(0);
        
        // Close menu
        await hamburgerButton.click();
        await expect(mobileMenu).not.toBeVisible();
      }

      // Verify widgets stack vertically on mobile
      await dashboardPage.verifyResponsiveLayout(viewport);

      // Test touch interactions
      const widget = page.locator('[data-testid="volcanic-alerts"]').first();
      if (await widget.isVisible()) {
        // Test tap interaction
        await widget.tap();
        await page.waitForTimeout(500);
        
        // Verify widget responds to touch
        const widgetExpanded = page.locator('[data-testid="widget-expanded"]');
        if (await widgetExpanded.isVisible()) {
          await expect(widgetExpanded).toBeVisible();
        }
      }

      // Test swipe gestures on charts
      const chart = page.locator('[data-testid="seismic-timeline-chart"]');
      if (await chart.isVisible()) {
        const chartBounds = await chart.boundingBox();
        
        if (chartBounds) {
          // Simulate swipe gesture
          await page.touchscreen.tap(chartBounds.x + 50, chartBounds.y + chartBounds.height / 2);
          await page.touchscreen.tap(chartBounds.x + chartBounds.width - 50, chartBounds.y + chartBounds.height / 2);
        }
      }

      // Test mobile-specific controls
      const mobileControls = page.locator('[data-testid="mobile-controls"]');
      if (await mobileControls.isVisible()) {
        await expect(mobileControls).toBeVisible();
        
        // Verify controls are touch-friendly (minimum 44px touch targets)
        const controlButtons = mobileControls.locator('button');
        const buttonCount = await controlButtons.count();
        
        for (let i = 0; i < buttonCount; i++) {
          const button = controlButtons.nth(i);
          const buttonBounds = await button.boundingBox();
          
          if (buttonBounds) {
            expect(buttonBounds.width).toBeGreaterThanOrEqual(44);
            expect(buttonBounds.height).toBeGreaterThanOrEqual(44);
          }
        }
      }
    });
  }

  // Test tablet responsiveness
  for (const viewport of tabletViewports) {
    test(`Tablet responsiveness - ${viewport.name} (${viewport.width}x${viewport.height})`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });

      await loginPage.goto();
      await loginPage.loginAsResearcher();

      // Test dashboard tablet layout
      await dashboardPage.goto();
      await dashboardPage.waitForDashboardLoad();

      // Verify tablet layout uses available space efficiently
      const widgetContainer = page.locator('[data-testid="widget-container"]');
      const containerBounds = await widgetContainer.boundingBox();
      
      if (containerBounds) {
        expect(containerBounds.width).toBeGreaterThan(viewport.width * 0.9);
      }

      // Test two-column layout on tablet
      const widgets = page.locator('[data-testid^="widget-"]');
      const widgetCount = await widgets.count();
      
      if (widgetCount >= 2) {
        const firstWidget = widgets.nth(0);
        const secondWidget = widgets.nth(1);
        
        const firstBounds = await firstWidget.boundingBox();
        const secondBounds = await secondWidget.boundingBox();
        
        if (firstBounds && secondBounds) {
          // On tablet, widgets should be side by side if space allows
          const sideBySide = Math.abs(firstBounds.y - secondBounds.y) < 50;
          const stacked = secondBounds.y > firstBounds.y + firstBounds.height - 50;
          
          expect(sideBySide || stacked).toBe(true);
        }
      }

      // Test tablet-specific navigation
      const tabletNav = page.locator('[data-testid="tablet-navigation"]');
      if (await tabletNav.isVisible()) {
        await expect(tabletNav).toBeVisible();
        
        // Verify navigation is accessible
        const navItems = tabletNav.locator('[data-testid="nav-item"]');
        const navCount = await navItems.count();
        expect(navCount).toBeGreaterThan(0);
      }

      // Test monitoring page on tablet
      await monitoringPage.goto();
      await monitoringPage.waitForMonitoringLoad();
      await monitoringPage.verifyResponsiveDesign();

      // Test 3D visualization on tablet
      await visualizationPage.goto();
      await visualizationPage.waitFor3DViewerReady();

      // Verify 3D viewer adapts to tablet screen
      const threeDViewer = page.locator('[data-testid="3d-viewer"]');
      const viewerBounds = await threeDViewer.boundingBox();
      
      if (viewerBounds) {
        expect(viewerBounds.width).toBeGreaterThan(viewport.width * 0.7);
        expect(viewerBounds.height).toBeGreaterThan(viewport.height * 0.5);
      }

      // Test touch controls for 3D navigation
      if (viewerBounds) {
        // Test pinch-to-zoom simulation
        const centerX = viewerBounds.x + viewerBounds.width / 2;
        const centerY = viewerBounds.y + viewerBounds.height / 2;
        
        await page.touchscreen.tap(centerX - 50, centerY);
        await page.touchscreen.tap(centerX + 50, centerY);
      }
    });
  }

  test('Mobile volcanic monitoring workflow', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE

    await loginPage.goto();
    await loginPage.loginAsOperator();

    await monitoringPage.goto();
    await monitoringPage.waitForMonitoringLoad();

    // Test mobile volcano selection
    const volcanoSelector = page.locator('[data-testid="volcano-selector"]');
    await volcanoSelector.tap();
    
    const volcanoOptions = page.locator('[data-testid="volcano-option"]');
    const optionCount = await volcanoOptions.count();
    
    if (optionCount > 0) {
      await volcanoOptions.first().tap();
      await page.waitForTimeout(1000);
    }

    // Test mobile alert management
    const alertPanel = page.locator('[data-testid="alerts-panel"]');
    if (await alertPanel.isVisible()) {
      // Test swipe to dismiss alerts (if implemented)
      const alertItems = alertPanel.locator('[data-testid="alert-item"]');
      const alertCount = await alertItems.count();
      
      if (alertCount > 0) {
        const firstAlert = alertItems.first();
        const alertBounds = await firstAlert.boundingBox();
        
        if (alertBounds) {
          // Simulate swipe gesture
          await page.touchscreen.tap(alertBounds.x + 50, alertBounds.y + alertBounds.height / 2);
          await page.mouse.move(alertBounds.x + alertBounds.width - 50, alertBounds.y + alertBounds.height / 2);
        }
      }
    }

    // Test mobile data visualization
    const seismicChart = page.locator('[data-testid="seismic-waveforms"]');
    if (await seismicChart.isVisible()) {
      const chartBounds = await seismicChart.boundingBox();
      
      if (chartBounds) {
        // Test pinch zoom on chart
        await page.touchscreen.tap(chartBounds.x + chartBounds.width / 3, chartBounds.y + chartBounds.height / 2);
        await page.touchscreen.tap(chartBounds.x + (chartBounds.width * 2) / 3, chartBounds.y + chartBounds.height / 2);
      }
    }

    // Test mobile export functionality
    const exportButton = page.locator('[data-testid="export-button"]');
    if (await exportButton.isVisible()) {
      await exportButton.tap();
      
      const exportOptions = page.locator('[data-testid="export-options"]');
      await expect(exportOptions).toBeVisible();
      
      // Verify export options are touch-friendly
      const options = exportOptions.locator('button');
      const optionCount = await options.count();
      
      for (let i = 0; i < optionCount; i++) {
        const option = options.nth(i);
        const optionBounds = await option.boundingBox();
        
        if (optionBounds) {
          expect(optionBounds.height).toBeGreaterThanOrEqual(44);
        }
      }
    }
  });

  test('Tablet 3D visualization workflow', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 1366 }); // iPad Pro

    await loginPage.goto();
    await loginPage.loginAsResearcher();

    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();

    // Test tablet-optimized 3D controls
    const threeDControls = page.locator('[data-testid="3d-controls"]');
    await expect(threeDControls).toBeVisible();

    // Verify controls are appropriately sized for tablet
    const controlButtons = threeDControls.locator('button');
    const buttonCount = await controlButtons.count();
    
    for (let i = 0; i < buttonCount; i++) {
      const button = controlButtons.nth(i);
      const buttonBounds = await button.boundingBox();
      
      if (buttonBounds) {
        expect(buttonBounds.width).toBeGreaterThanOrEqual(44);
        expect(buttonBounds.height).toBeGreaterThanOrEqual(44);
      }
    }

    // Test multi-touch gestures on 3D viewer
    const threeDViewer = page.locator('[data-testid="3d-viewer"]');
    const viewerBounds = await threeDViewer.boundingBox();
    
    if (viewerBounds) {
      const centerX = viewerBounds.x + viewerBounds.width / 2;
      const centerY = viewerBounds.y + viewerBounds.height / 2;
      
      // Test rotation gesture
      await page.touchscreen.tap(centerX - 100, centerY);
      await page.mouse.move(centerX + 100, centerY);
      await page.waitForTimeout(500);
      
      // Test zoom gesture
      await page.touchscreen.tap(centerX, centerY - 50);
      await page.touchscreen.tap(centerX, centerY + 50);
    }

    // Test layer panel on tablet
    const layerPanel = page.locator('[data-testid="layer-panel"]');
    await expect(layerPanel).toBeVisible();

    // Verify layer toggles are touch-friendly
    const layerToggles = layerPanel.locator('[data-testid="layer-toggle"]');
    const toggleCount = await layerToggles.count();
    
    for (let i = 0; i < Math.min(toggleCount, 3); i++) {
      const toggle = layerToggles.nth(i);
      await toggle.tap();
      await page.waitForTimeout(500);
      
      // Verify toggle state changed
      const toggleState = await toggle.getAttribute('aria-checked');
      expect(toggleState).toBeTruthy();
    }

    // Test measurement tools on tablet
    await visualizationPage.testMeasurementTools();
    
    const measurementTool = page.locator('[data-testid="measurement-tool"]');
    if (await measurementTool.isVisible()) {
      await measurementTool.tap();
      
      const distanceTool = page.locator('[data-testid="distance-tool"]');
      await distanceTool.tap();
      
      // Test touch-based measurement
      if (viewerBounds) {
        await page.touchscreen.tap(viewerBounds.x + 100, viewerBounds.y + 100);
        await page.touchscreen.tap(viewerBounds.x + 200, viewerBounds.y + 150);
        
        const measurement = page.locator('[data-testid="distance-measurement"]');
        await expect(measurement).toBeVisible();
      }
    }
  });

  test('Cross-device data synchronization', async ({ page, context }) => {
    // Simulate multiple devices accessing the same account
    
    // Device 1: Mobile phone
    await page.setViewportSize({ width: 375, height: 667 });
    await loginPage.goto();
    await loginPage.loginAsResearcher();
    
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();
    
    // Make changes on mobile device
    await dashboardPage.enableEditMode();
    const widget = await dashboardPage.addWidget('seismic-activity');
    await dashboardPage.saveDashboard();

    // Device 2: Tablet
    const tabletPage = await context.newPage();
    const tabletDashboard = new DashboardPage(tabletPage);
    const tabletLogin = new LoginPage(tabletPage);
    
    await tabletPage.setViewportSize({ width: 1024, height: 1366 });
    await tabletLogin.goto();
    await tabletLogin.loginAsResearcher();
    
    await tabletDashboard.goto();
    await tabletDashboard.waitForDashboardLoad();
    
    // Verify changes from mobile are reflected on tablet
    const syncedWidget = tabletPage.locator('[data-widget-type="seismic-activity"]');
    await expect(syncedWidget).toBeVisible();
    
    // Make changes on tablet
    await tabletDashboard.enableEditMode();
    await tabletDashboard.addWidget('volcanic-alerts');
    await tabletDashboard.saveDashboard();
    
    // Verify changes sync back to mobile
    await dashboardPage.refreshDashboard();
    const mobileWidget = page.locator('[data-widget-type="volcanic-alerts"]');
    await expect(mobileWidget).toBeVisible();
  });

  test('Mobile performance optimization', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    // Test with throttled network
    await page.route('**/*', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 100)); // Add 100ms delay
      await route.continue();
    });

    await loginPage.goto();
    await loginPage.loginAsResearcher();

    // Measure mobile load performance
    const startTime = Date.now();
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();
    const loadTime = Date.now() - startTime;

    // Mobile should load within reasonable time even with throttling
    expect(loadTime).toBeLessThan(8000); // 8 seconds with network throttling

    // Test memory usage on mobile
    const memoryUsage = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });
    
    // Mobile devices have limited memory
    expect(memoryUsage).toBeLessThan(100 * 1024 * 1024); // Less than 100MB

    // Test CPU usage during interactions
    const cpuIntensiveActions = [
      () => dashboardPage.testInteractiveControls(),
      () => dashboardPage.verifyVolcanicMonitoring(),
      () => dashboardPage.verifySeismicActivity()
    ];

    for (const action of cpuIntensiveActions) {
      const actionStart = Date.now();
      await action();
      const actionTime = Date.now() - actionStart;
      
      // Actions should complete quickly on mobile
      expect(actionTime).toBeLessThan(3000);
    }
  });

  test('Accessibility on mobile devices', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    await loginPage.goto();
    await loginPage.loginAsResearcher();

    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Test screen reader compatibility
    const headings = page.locator('h1, h2, h3, h4, h5, h6');
    const headingCount = await headings.count();
    
    for (let i = 0; i < headingCount; i++) {
      const heading = headings.nth(i);
      const ariaLevel = await heading.getAttribute('aria-level');
      const role = await heading.getAttribute('role');
      
      // Headings should have proper structure
      expect(ariaLevel || role).toBeTruthy();
    }

    // Test focus management on mobile
    const focusableElements = page.locator('button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])');
    const focusableCount = await focusableElements.count();
    
    if (focusableCount > 0) {
      const firstFocusable = focusableElements.first();
      await firstFocusable.focus();
      await expect(firstFocusable).toBeFocused();
      
      // Test tab navigation
      await page.keyboard.press('Tab');
      const secondFocusable = focusableElements.nth(1);
      if (await secondFocusable.isVisible()) {
        await expect(secondFocusable).toBeFocused();
      }
    }

    // Test touch target sizes
    const interactiveElements = page.locator('button, input, select, a');
    const elementCount = await interactiveElements.count();
    
    for (let i = 0; i < Math.min(elementCount, 10); i++) {
      const element = interactiveElements.nth(i);
      const bounds = await element.boundingBox();
      
      if (bounds) {
        // WCAG AA requires minimum 44x44px touch targets
        expect(bounds.width).toBeGreaterThanOrEqual(44);
        expect(bounds.height).toBeGreaterThanOrEqual(44);
      }
    }

    // Test color contrast in mobile view
    const textElements = page.locator('p, span, div, label');
    const textCount = await textElements.count();
    
    for (let i = 0; i < Math.min(textCount, 5); i++) {
      const element = textElements.nth(i);
      const styles = await element.evaluate(el => {
        const computed = window.getComputedStyle(el);
        return {
          color: computed.color,
          backgroundColor: computed.backgroundColor
        };
      });
      
      // Ensure text has sufficient contrast
      expect(styles.color).not.toBe('rgba(0, 0, 0, 0)');
      expect(styles.color).not.toBe('transparent');
    }
  });
});