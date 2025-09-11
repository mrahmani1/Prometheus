import { test, expect } from '@playwright/test';
import { LoginPage } from '../page-objects/LoginPage';
import { DashboardPage } from '../page-objects/DashboardPage';
import { VolcanicMonitoringPage } from '../page-objects/VolcanicMonitoringPage';
import { VisualizationPage } from '../page-objects/VisualizationPage';
import { MockApiService, MockWebSocketService } from '../fixtures/mock-services';
import { mockUsers, performanceBenchmarks } from '../fixtures/test-data';

test.describe('Comprehensive User Workflows', () => {
  let loginPage: LoginPage;
  let dashboardPage: DashboardPage;
  let monitoringPage: VolcanicMonitoringPage;
  let visualizationPage: VisualizationPage;
  let mockApi: MockApiService;
  let mockWebSocket: MockWebSocketService;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    dashboardPage = new DashboardPage(page);
    monitoringPage = new VolcanicMonitoringPage(page);
    visualizationPage = new VisualizationPage(page);
    mockApi = new MockApiService(page);
    mockWebSocket = new MockWebSocketService(page);

    // Set up mocks
    await mockApi.setupSuccessfulMocks();
    await mockWebSocket.setupWebSocketMocks();
  });

  test('Complete researcher workflow: Login → Dashboard → Analysis → Visualization', async ({ page }) => {
    // Step 1: Login as researcher
    await loginPage.goto();
    await loginPage.loginAsResearcher();
    
    // Verify successful login
    await expect(page.locator('[data-testid="user-profile"]')).toBeVisible();

    // Step 2: Navigate to dashboard
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();
    
    // Verify dashboard content
    await dashboardPage.validatePageContent();
    
    // Check performance
    const dashboardMetrics = await dashboardPage.getPerformanceMetrics();
    expect(dashboardMetrics.loadTime).toBeLessThan(performanceBenchmarks.pageLoad.dashboard);

    // Step 3: Analyze volcanic data
    await dashboardPage.verifyVolcanicMonitoring();
    await dashboardPage.verifySeismicActivity();
    await dashboardPage.verifyRealTimeData();
    
    // Test interactive controls
    await dashboardPage.testInteractiveControls();

    // Step 4: Navigate to detailed monitoring
    await monitoringPage.goto();
    await monitoringPage.waitForMonitoringLoad();
    
    // Select specific volcano for analysis
    await monitoringPage.selectVolcano('yellowstone');
    
    // Verify detailed monitoring data
    await monitoringPage.verifySeismicData();
    await monitoringPage.verifyDeformationData();
    await monitoringPage.verifyGasEmissions();
    await monitoringPage.verifyThermalData();

    // Step 5: Advanced visualization
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();
    
    // Test 3D geological visualization
    await visualizationPage.test3DNavigation();
    await visualizationPage.testGeologicalLayers();
    await visualizationPage.test4DVisualization();
    
    // Verify scientific accuracy
    await visualizationPage.testScientificVisualization();

    // Step 6: Export analysis results
    await visualizationPage.testExportFunctionality();
    await monitoringPage.exportData('pdf');
    await dashboardPage.exportDashboard('png');

    // Verify overall workflow performance
    const totalWorkflowTime = await page.evaluate(() => performance.now());
    expect(totalWorkflowTime).toBeLessThan(30000); // Complete workflow under 30 seconds
  });

  test('Volcano observatory operator workflow: Real-time monitoring and alerting', async ({ page }) => {
    // Login as operator
    await loginPage.goto();
    await loginPage.loginAsOperator();

    // Navigate to monitoring dashboard
    await monitoringPage.goto();
    await monitoringPage.waitForMonitoringLoad();

    // Enable real-time mode
    await monitoringPage.enableRealTimeMode();

    // Simulate real-time events
    await mockWebSocket.simulateConnection();
    await mockWebSocket.simulateSeismicEvent();
    
    // Verify real-time data updates
    await expect(page.locator('[data-testid="real-time-indicator"]')).toBeVisible();
    await expect(page.locator('[data-testid="recent-seismic-event"]').first()).toBeVisible();

    // Test alert level changes
    await mockWebSocket.simulateAlertLevelChange();
    
    // Verify alert notifications
    await expect(page.locator('[data-testid="alert-notification"]')).toBeVisible();

    // Configure alert thresholds
    await page.locator('[data-testid="alert-settings"]').click();
    await page.locator('[data-testid="magnitude-threshold"]').fill('4.0');
    await page.locator('[data-testid="save-thresholds"]').click();

    // Verify threshold configuration saved
    await expect(page.locator('[data-testid="save-confirmation"]')).toBeVisible();

    // Test multiple volcano monitoring
    const volcanoes = ['yellowstone', 'kilauea', 'mount-st-helens'];
    
    for (const volcano of volcanoes) {
      await monitoringPage.selectVolcano(volcano);
      await monitoringPage.verifySeismicData();
      
      // Verify volcano-specific data loads correctly
      await expect(page.locator(`[data-volcano-id="${volcano}"]`)).toHaveClass(/selected/);
    }

    // Test alert filtering and management
    await monitoringPage.filterByAlertLevel('orange');
    await monitoringPage.verifyAlerts();

    // Export monitoring report
    await monitoringPage.exportData('csv');
  });

  test('Industry user workflow: Resource assessment and economic analysis', async ({ page }) => {
    // Login as industry user
    await loginPage.goto();
    await loginPage.loginAsIndustryUser();

    // Navigate to dashboard
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Focus on resource assessment widgets
    await dashboardPage.verifyGeologicalAnalysis();

    // Navigate to detailed visualization for resource exploration
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();

    // Test geological layer analysis for resource prospectivity
    await visualizationPage.testGeologicalLayers();

    // Use measurement tools for resource assessment
    await visualizationPage.testMeasurementTools();

    // Add annotations for potential resource locations
    await visualizationPage.testAnnotationTools();

    // Test cross-section analysis
    const crossSectionTool = page.locator('[data-testid="cross-section-tool"]');
    await crossSectionTool.click();

    // Draw cross-section for geological analysis
    const viewer = page.locator('[data-testid="3d-viewer"]');
    const viewerBounds = await viewer.boundingBox();
    
    if (viewerBounds) {
      await page.mouse.move(viewerBounds.x + 100, viewerBounds.y + 100);
      await page.mouse.down();
      await page.mouse.move(viewerBounds.x + 300, viewerBounds.y + 100);
      await page.mouse.up();
    }

    // Verify cross-section generated
    await expect(page.locator('[data-testid="cross-section-view"]')).toBeVisible();

    // Export geological models and analysis
    await visualizationPage.testExportFunctionality();

    // Generate comprehensive resource assessment report
    await page.locator('[data-testid="generate-report"]').click();
    await page.locator('[data-testid="report-type-resource-assessment"]').click();
    await page.locator('[data-testid="generate-report-confirm"]').click();

    // Wait for report generation
    await expect(page.locator('[data-testid="report-ready"]')).toBeVisible({ timeout: 30000 });
  });

  test('Multi-user collaboration workflow', async ({ page, context }) => {
    // Test collaborative features with multiple users
    
    // User 1: Researcher
    await loginPage.goto();
    await loginPage.loginAsResearcher();
    
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();

    // Add annotation as researcher
    await visualizationPage.testAnnotationTools();

    // Open new tab for second user
    const page2 = await context.newPage();
    const loginPage2 = new LoginPage(page2);
    const visualizationPage2 = new VisualizationPage(page2);

    // User 2: Operator
    await loginPage2.goto();
    await loginPage2.loginAsOperator();
    
    await visualizationPage2.goto();
    await visualizationPage2.waitFor3DViewerReady();

    // Verify annotations from user 1 are visible to user 2
    await expect(page2.locator('[data-testid="annotation-marker"]')).toBeVisible();

    // User 2 adds their own annotation
    await visualizationPage2.testAnnotationTools();

    // Verify real-time collaboration
    await expect(page.locator('[data-testid="annotation-marker"]')).toHaveCount(2);

    // Test concurrent data analysis
    await Promise.all([
      visualizationPage.test3DNavigation(),
      visualizationPage2.testGeologicalLayers()
    ]);

    // Verify no conflicts in concurrent operations
    await visualizationPage.checkRenderingPerformance();
    await visualizationPage2.checkRenderingPerformance();
  });

  test('Error handling and recovery workflow', async ({ page }) => {
    // Test system behavior under error conditions
    
    await loginPage.goto();
    await loginPage.loginAsResearcher();

    // Set up error mocks
    await mockApi.setupErrorMocks();

    await dashboardPage.goto();
    
    // Verify error handling
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();

    // Test retry functionality
    await mockApi.setupSuccessfulMocks(); // Restore successful responses
    await page.locator('[data-testid="retry-button"]').click();

    // Verify recovery
    await dashboardPage.waitForDashboardLoad();
    await dashboardPage.validatePageContent();

    // Test WebSocket disconnection and reconnection
    await mockWebSocket.simulateDisconnection();
    await expect(page.locator('[data-testid="connection-lost"]')).toBeVisible();

    await mockWebSocket.simulateConnection();
    await expect(page.locator('[data-testid="connection-restored"]')).toBeVisible();

    // Test partial data loading scenarios
    await mockApi.setupSlowMocks(2000);
    
    await monitoringPage.goto();
    
    // Verify loading states
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible();
    
    // Wait for data to load
    await monitoringPage.waitForMonitoringLoad();
    await monitoringPage.validatePageContent();
  });

  test('Performance optimization workflow', async ({ page }) => {
    // Test system performance under various conditions
    
    await loginPage.goto();
    await loginPage.loginAsResearcher();

    // Measure dashboard load performance
    const startTime = Date.now();
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();
    const dashboardLoadTime = Date.now() - startTime;

    expect(dashboardLoadTime).toBeLessThan(performanceBenchmarks.pageLoad.dashboard);

    // Test widget performance
    await dashboardPage.verifyPerformance();

    // Measure 3D visualization performance
    const vizStartTime = Date.now();
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();
    const vizLoadTime = Date.now() - vizStartTime;

    expect(vizLoadTime).toBeLessThan(performanceBenchmarks.pageLoad.visualization);

    // Test rendering performance
    await visualizationPage.checkRenderingPerformance();

    // Test memory usage during extended use
    for (let i = 0; i < 5; i++) {
      await visualizationPage.test3DNavigation();
      await visualizationPage.testGeologicalLayers();
      
      // Check memory hasn't increased significantly
      const memoryUsage = await page.evaluate(() => {
        return (performance as any).memory?.usedJSHeapSize || 0;
      });
      
      expect(memoryUsage).toBeLessThan(performanceBenchmarks.rendering.maxMemory * 1024 * 1024);
    }

    // Test concurrent operations performance
    await Promise.all([
      dashboardPage.testInteractiveControls(),
      visualizationPage.test4DVisualization(),
      monitoringPage.verifySeismicData()
    ]);
  });
});