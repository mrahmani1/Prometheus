import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

/**
 * Enhanced Dashboard page object with comprehensive geological monitoring capabilities
 */
export class DashboardPage extends BasePage {
  // Main dashboard elements
  private readonly dashboardTitle: Locator;
  private readonly dashboardGrid: Locator;
  private readonly widgetContainer: Locator;
  private readonly addWidgetButton: Locator;
  private readonly editModeToggle: Locator;
  private readonly saveButton: Locator;
  private readonly refreshButton: Locator;
  
  // Volcanic monitoring widgets
  private readonly volcanicAlerts: Locator;
  private readonly eruptionForecast: Locator;
  private readonly calderaMonitoring: Locator;
  private readonly supereruptionRisk: Locator;
  
  // Seismic monitoring widgets
  private readonly seismicActivity: Locator;
  private readonly seismicEvents: Locator;
  private readonly earthquakeMap: Locator;
  private readonly waveformDisplay: Locator;
  
  // Real-time data widgets
  private readonly realTimeData: Locator;
  private readonly dataStreamStatus: Locator;
  private readonly sensorNetwork: Locator;
  private readonly dataQualityIndicator: Locator;
  private readonly realTimeIndicator: Locator;
  
  // Geological analysis widgets
  private readonly geologicalModels: Locator;
  private readonly resourceAssessment: Locator;
  private readonly stressFieldMap: Locator;
  private readonly mantleConvection: Locator;
  
  // Interactive controls
  private readonly timeRangeSelector: Locator;
  private readonly regionSelector: Locator;
  private readonly layerToggle: Locator;
  private readonly alertSettings: Locator;
  
  // Performance indicators
  private readonly performanceMetrics: Locator;
  private readonly systemHealth: Locator;
  private readonly alertPanel: Locator;
  private readonly statisticsPanel: Locator;

  constructor(page: Page) {
    super(page);
    
    this.dashboardTitle = page.locator('[data-testid="dashboard-title"]');
    this.dashboardGrid = page.locator('[data-testid="dashboard-grid"]');
    this.widgetContainer = page.locator('[data-testid="widget-container"]');
    this.addWidgetButton = page.locator('[data-testid="add-widget-button"]');
    this.editModeToggle = page.locator('[data-testid="edit-mode-toggle"]');
    this.saveButton = page.locator('[data-testid="save-button"]');
    this.refreshButton = page.locator('[data-testid="refresh-button"]');
    
    // Volcanic monitoring
    this.volcanicAlerts = page.locator('[data-testid="volcanic-alerts"]');
    this.eruptionForecast = page.locator('[data-testid="eruption-forecast"]');
    this.calderaMonitoring = page.locator('[data-testid="caldera-monitoring"]');
    this.supereruptionRisk = page.locator('[data-testid="supereruption-risk"]');
    
    // Seismic monitoring
    this.seismicActivity = page.locator('[data-testid="seismic-activity"]');
    this.seismicEvents = page.locator('[data-testid="seismic-events"]');
    this.earthquakeMap = page.locator('[data-testid="earthquake-map"]');
    this.waveformDisplay = page.locator('[data-testid="waveform-display"]');
    
    // Real-time data
    this.realTimeData = page.locator('[data-testid="real-time-data"]');
    this.dataStreamStatus = page.locator('[data-testid="data-stream-status"]');
    this.sensorNetwork = page.locator('[data-testid="sensor-network"]');
    this.dataQualityIndicator = page.locator('[data-testid="data-quality-indicator"]');
    this.realTimeIndicator = page.locator('[data-testid="real-time-indicator"]');
    
    // Geological analysis
    this.geologicalModels = page.locator('[data-testid="geological-models"]');
    this.resourceAssessment = page.locator('[data-testid="resource-assessment"]');
    this.stressFieldMap = page.locator('[data-testid="stress-field-map"]');
    this.mantleConvection = page.locator('[data-testid="mantle-convection"]');
    
    // Controls
    this.timeRangeSelector = page.locator('[data-testid="time-range-selector"]');
    this.regionSelector = page.locator('[data-testid="region-selector"]');
    this.layerToggle = page.locator('[data-testid="layer-toggle"]');
    this.alertSettings = page.locator('[data-testid="alert-settings"]');
    
    // Performance
    this.performanceMetrics = page.locator('[data-testid="performance-metrics"]');
    this.systemHealth = page.locator('[data-testid="system-health"]');
    this.alertPanel = page.locator('[data-testid="alert-panel"]');
    this.statisticsPanel = page.locator('[data-testid="statistics-panel"]');
  }

  async goto(): Promise<void> {
    await super.goto('/dashboard');
  }

  /**
   * Wait for dashboard to fully load with all geological data
   */
  async waitForDashboardLoad(): Promise<void> {
    await this.waitForPageReady();
    await this.waitForGeologicalDataReady();
    
    // Wait for all critical widgets to load
    await expect(this.dashboardTitle).toBeVisible();
    await expect(this.widgetContainer).toBeVisible();
    
    // Verify WebSocket connections are established
    await this.waitForWebSocketConnection();
  }

  /**
   * Enable dashboard edit mode
   */
  async enableEditMode(): Promise<void> {
    await this.editModeToggle.click();
    await expect(this.editModeToggle).toHaveAttribute('aria-pressed', 'true');
  }

  /**
   * Disable dashboard edit mode
   */
  async disableEditMode(): Promise<void> {
    await this.editModeToggle.click();
    await expect(this.editModeToggle).toHaveAttribute('aria-pressed', 'false');
  }

  /**
   * Add a new widget to the dashboard
   */
  async addWidget(widgetType: string): Promise<Locator> {
    await this.addWidgetButton.click();
    
    // Wait for add widget dialog
    const dialog = this.page.locator('[data-testid="add-widget-dialog"]');
    await expect(dialog).toBeVisible();
    
    // Select widget type
    const widgetOption = dialog.locator(`[data-testid="widget-option-${widgetType}"]`);
    await widgetOption.click();
    
    // Wait for widget to be added
    await expect(dialog).not.toBeVisible();
    
    // Verify widget was added
    const newWidget = this.widgetContainer.locator(`[data-widget-type="${widgetType}"]`);
    await expect(newWidget).toBeVisible();
    
    return newWidget;
  }

  /**
   * Remove a widget from the dashboard
   */
  async removeWidget(widgetId: string): Promise<void> {
    const widget = this.widgetContainer.locator(`[data-widget-id="${widgetId}"]`);
    const removeButton = widget.locator('[data-testid="widget-remove-button"]');
    
    await removeButton.click();
    
    // Confirm removal if dialog appears
    const confirmDialog = this.page.locator('[data-testid="confirm-dialog"]');
    if (await confirmDialog.isVisible()) {
      await confirmDialog.locator('[data-testid="confirm-button"]').click();
    }
    
    // Verify widget was removed
    await expect(widget).not.toBeVisible();
  }

  /**
   * Verify all volcanic monitoring widgets are functional
   */
  async verifyVolcanicMonitoring(): Promise<void> {
    await expect(this.volcanicAlerts).toBeVisible();
    await expect(this.eruptionForecast).toBeVisible();
    await expect(this.calderaMonitoring).toBeVisible();
    
    // Verify alert levels are displayed
    const alertLevels = this.page.locator('[data-testid="alert-level"]');
    await expect(alertLevels.first()).toBeVisible();
    
    // Verify eruption probability is shown
    const eruptionProb = this.page.locator('[data-testid="eruption-probability"]');
    await expect(eruptionProb).toBeVisible();
    await expect(eruptionProb).toHaveText(/\d+(\.\d+)?%/);
  }

  /**
   * Verify seismic activity monitoring
   */
  async verifySeismicActivity(): Promise<void> {
    await expect(this.seismicActivity).toBeVisible();
    await expect(this.seismicEvents).toBeVisible();
    await expect(this.earthquakeMap).toBeVisible();
    
    // Verify recent events are displayed
    const recentEvents = this.page.locator('[data-testid="recent-seismic-event"]');
    const eventCount = await recentEvents.count();
    expect(eventCount).toBeGreaterThan(0);
    
    // Verify magnitude and depth information
    const magnitudes = this.page.locator('[data-testid="event-magnitude"]');
    await expect(magnitudes.first()).toHaveText(/M\d+\.\d+/);
  }

  /**
   * Verify real-time data streams
   */
  async verifyRealTimeData(): Promise<void> {
    await expect(this.realTimeData).toBeVisible();
    await expect(this.dataStreamStatus).toBeVisible();
    
    // Verify data stream indicators show active status
    const streamIndicators = this.page.locator('[data-testid="stream-indicator"]');
    const activeStreams = streamIndicators.locator('.status-active');
    const activeCount = await activeStreams.count();
    expect(activeCount).toBeGreaterThan(0);
    
    // Verify data quality is acceptable
    await expect(this.dataQualityIndicator).toHaveClass(/quality-good|quality-excellent/);
  }

  /**
   * Verify geological analysis widgets
   */
  async verifyGeologicalAnalysis(): Promise<void> {
    await expect(this.geologicalModels).toBeVisible();
    await expect(this.resourceAssessment).toBeVisible();
    
    // Verify 3D model is loaded
    const model3D = this.page.locator('[data-testid="3d-geological-model"]');
    await expect(model3D).toBeVisible();
    
    // Verify resource prospectivity data
    const prospectivity = this.page.locator('[data-testid="prospectivity-score"]');
    await expect(prospectivity).toBeVisible();
  }

  /**
   * Test interactive dashboard controls
   */
  async testInteractiveControls(): Promise<void> {
    // Test time range selector
    await this.timeRangeSelector.click();
    await this.page.locator('[data-testid="time-range-24h"]').click();
    await this.waitForGeologicalDataReady();
    
    // Test region selector
    await this.regionSelector.click();
    await this.page.locator('[data-testid="region-yellowstone"]').click();
    await this.waitForGeologicalDataReady();
    
    // Test layer toggle
    await this.layerToggle.click();
    const layerOptions = this.page.locator('[data-testid="layer-option"]');
    await layerOptions.first().click();
  }

  /**
   * Check system performance metrics
   */
  async checkPerformanceMetrics(): Promise<void> {
    await expect(this.performanceMetrics).toBeVisible();
    await expect(this.systemHealth).toBeVisible();
    
    // Verify system health is good
    const healthStatus = this.page.locator('[data-testid="health-status"]');
    await expect(healthStatus).toHaveClass(/health-good|health-excellent/);
    
    // Verify response times are acceptable
    const responseTime = this.page.locator('[data-testid="response-time"]');
    const responseTimeText = await responseTime.textContent();
    const responseTimeMs = parseInt(responseTimeText?.match(/\d+/)?.[0] || '0');
    expect(responseTimeMs).toBeLessThan(1000); // Less than 1 second
  }

  /**
   * Test drag and drop widget functionality
   */
  async testWidgetDragAndDrop(): Promise<void> {
    const sourceWidget = this.volcanicAlerts;
    const targetPosition = this.seismicActivity;
    
    // Get initial positions
    const sourceBounds = await sourceWidget.boundingBox();
    const targetBounds = await targetPosition.boundingBox();
    
    if (sourceBounds && targetBounds) {
      // Perform drag and drop
      await sourceWidget.dragTo(targetPosition);
      
      // Verify widget positions have changed
      await this.page.waitForTimeout(1000); // Allow animation to complete
      const newSourceBounds = await sourceWidget.boundingBox();
      expect(newSourceBounds?.x).not.toBe(sourceBounds.x);
    }
  }

  /**
   * Save dashboard configuration
   */
  async saveDashboard(): Promise<void> {
    await this.saveButton.click();
    
    // Wait for save confirmation
    const notification = this.page.locator('[data-testid="notification"]');
    await expect(notification).toContainText('saved');
  }

  /**
   * Refresh dashboard data
   */
  async refreshDashboard(): Promise<void> {
    await this.refreshButton.click();
    
    // Wait for refresh to complete
    await this.page.waitForLoadState('networkidle');
    await this.waitForGeologicalDataReady();
  }

  /**
   * Get current widget count
   */
  async getWidgetCount(): Promise<number> {
    const widgets = this.widgetContainer.locator('[data-testid^="widget-"]');
    return await widgets.count();
  }

  /**
   * Get current alert count
   */
  async getAlertCount(): Promise<number> {
    const alerts = this.alertPanel.locator('[data-testid="alert-item"]');
    return await alerts.count();
  }

  /**
   * Verify real-time mode status
   */
  async verifyRealTimeMode(enabled: boolean): Promise<void> {
    if (enabled) {
      await expect(this.realTimeIndicator).toBeVisible();
      await expect(this.realTimeIndicator).toContainText('Live');
    } else {
      await expect(this.realTimeIndicator).not.toBeVisible();
    }
  }

  /**
   * Wait for all data to load
   */
  async waitForDataLoad(): Promise<void> {
    // Wait for loading indicators to disappear
    const loadingIndicators = this.page.locator('[data-testid="loading"]');
    await expect(loadingIndicators).toHaveCount(0);
    
    // Wait for geological data to be ready
    await this.waitForGeologicalDataReady();
  }

  /**
   * Verify responsive layout at different viewport sizes
   */
  async verifyResponsiveLayout(viewport: { width: number; height: number }): Promise<void> {
    await this.page.setViewportSize(viewport);
    
    // Verify layout adapts to viewport
    const container = this.widgetContainer;
    const containerBox = await container.boundingBox();
    
    expect(containerBox?.width).toBeLessThanOrEqual(viewport.width);
    
    // Check if widgets stack on mobile
    if (viewport.width < 768) {
      const widgets = container.locator('[data-testid^="widget-"]');
      const widgetCount = await widgets.count();
      
      if (widgetCount > 1) {
        const firstWidget = widgets.nth(0);
        const secondWidget = widgets.nth(1);
        
        const firstBox = await firstWidget.boundingBox();
        const secondBox = await secondWidget.boundingBox();
        
        // Widgets should stack vertically on mobile
        expect(secondBox?.y).toBeGreaterThan((firstBox?.y || 0) + (firstBox?.height || 0));
      }
    }
  }

  /**
   * Export dashboard in specified format
   */
  async exportDashboard(format: 'pdf' | 'png' | 'json' | 'csv'): Promise<any> {
    // Open export menu
    const exportButton = this.page.locator('[data-testid="export-button"]');
    await exportButton.click();
    
    // Select format
    const formatOption = this.page.locator(`[data-testid="export-format-${format}"]`);
    await formatOption.click();
    
    // Wait for download
    const downloadPromise = this.page.waitForEvent('download');
    const confirmButton = this.page.locator('[data-testid="export-confirm"]');
    await confirmButton.click();
    
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toContain(format);
    
    return download;
  }

  /**
   * Validate page content and geological accuracy
   */
  async validatePageContent(): Promise<void> {
    // Verify dashboard title and structure
    await expect(this.dashboardTitle).toBeVisible();
    await expect(this.dashboardTitle).toContainText('Prometheus AI Dashboard');
    
    // Verify coordinate system is properly initialized
    await this.verifyCoordinateSystem();
    
    // Verify all major widget categories are present
    await this.verifyVolcanicMonitoring();
    await this.verifySeismicActivity();
    await this.verifyRealTimeData();
    await this.verifyGeologicalAnalysis();
    
    // Verify performance is acceptable
    await this.checkPerformanceMetrics();
    
    // Check for any errors
    await this.checkForErrors();
  }

  /**
   * Verify dashboard performance
   */
  async verifyPerformance(): Promise<void> {
    // Measure initial load time
    const startTime = Date.now();
    await this.waitForDashboardLoad();
    const loadTime = Date.now() - startTime;
    
    // Should load within 5 seconds for geological data
    expect(loadTime).toBeLessThan(5000);
    
    // Check for memory leaks by monitoring widget count
    const initialWidgetCount = await this.getWidgetCount();
    
    // Add and remove widgets multiple times
    for (let i = 0; i < 3; i++) {
      await this.addWidget('statistics');
      await this.page.waitForTimeout(100);
      const widgets = this.widgetContainer.locator('[data-widget-type="statistics"]');
      const lastWidget = widgets.last();
      const widgetId = await lastWidget.getAttribute('data-widget-id');
      if (widgetId) {
        await this.removeWidget(widgetId);
      }
    }
    
    // Widget count should return to initial
    const finalWidgetCount = await this.getWidgetCount();
    expect(finalWidgetCount).toBe(initialWidgetCount);
  }
}