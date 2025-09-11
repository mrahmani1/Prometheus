import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

/**
 * Enhanced Volcanic Monitoring page object with comprehensive real-time monitoring capabilities
 */
export class VolcanicMonitoringPage extends BasePage {
  // Main monitoring elements
  private readonly pageTitle: Locator;
  private readonly volcanoList: Locator;
  private readonly volcanoSelector: Locator;
  private readonly alertLevel: Locator;
  private readonly alertHistory: Locator;
  private readonly alertLevelFilter: Locator;
  
  // Views and navigation
  private readonly mapView: Locator;
  private readonly detailView: Locator;
  private readonly realTimeToggle: Locator;
  private readonly refreshButton: Locator;
  private readonly exportButton: Locator;
  
  // Real-time data streams
  private readonly seismicData: Locator;
  private readonly deformationData: Locator;
  private readonly gasEmissions: Locator;
  private readonly thermalData: Locator;
  private readonly acousticData: Locator;
  
  // Advanced monitoring features
  private readonly eruptionForecast: Locator;
  private readonly calderaAnalysis: Locator;
  private readonly supereruptionRisk: Locator;
  private readonly magmaChamberModel: Locator;
  
  // Interactive controls
  private readonly timeRangeControl: Locator;
  private readonly dataSourceToggle: Locator;
  private readonly alertThresholds: Locator;
  private readonly notificationSettings: Locator;
  
  // Visualization components
  private readonly seismicWaveforms: Locator;
  private readonly deformationMap: Locator;
  private readonly gasCompositionChart: Locator;
  private readonly thermalImagery: Locator;
  private readonly volcanoMap: Locator;
  
  // Analysis tools
  private readonly eventDetection: Locator;
  private readonly trendAnalysis: Locator;
  private readonly correlationAnalysis: Locator;
  private readonly riskAssessment: Locator;
  
  // Panels
  private readonly statisticsPanel: Locator;
  private readonly alertsPanel: Locator;

  constructor(page: Page) {
    super(page);
    
    this.pageTitle = page.locator('[data-testid="volcanic-monitoring-title"]');
    this.volcanoList = page.locator('[data-testid="volcano-list"]');
    this.volcanoSelector = page.locator('[data-testid="volcano-selector"]');
    this.alertLevel = page.locator('[data-testid="alert-level"]');
    this.alertHistory = page.locator('[data-testid="alert-history"]');
    this.alertLevelFilter = page.locator('[data-testid="alert-level-filter"]');
    
    // Views
    this.mapView = page.locator('[data-testid="map-view"]');
    this.detailView = page.locator('[data-testid="detail-view"]');
    this.realTimeToggle = page.locator('[data-testid="real-time-toggle"]');
    this.refreshButton = page.locator('[data-testid="refresh-button"]');
    this.exportButton = page.locator('[data-testid="export-button"]');
    
    // Real-time data
    this.seismicData = page.locator('[data-testid="seismic-data"]');
    this.deformationData = page.locator('[data-testid="deformation-data"]');
    this.gasEmissions = page.locator('[data-testid="gas-emissions"]');
    this.thermalData = page.locator('[data-testid="thermal-data"]');
    this.acousticData = page.locator('[data-testid="acoustic-data"]');
    
    // Advanced features
    this.eruptionForecast = page.locator('[data-testid="eruption-forecast"]');
    this.calderaAnalysis = page.locator('[data-testid="caldera-analysis"]');
    this.supereruptionRisk = page.locator('[data-testid="supereruption-risk"]');
    this.magmaChamberModel = page.locator('[data-testid="magma-chamber-model"]');
    
    // Controls
    this.timeRangeControl = page.locator('[data-testid="time-range-control"]');
    this.dataSourceToggle = page.locator('[data-testid="data-source-toggle"]');
    this.alertThresholds = page.locator('[data-testid="alert-thresholds"]');
    this.notificationSettings = page.locator('[data-testid="notification-settings"]');
    
    // Visualizations
    this.seismicWaveforms = page.locator('[data-testid="seismic-waveforms"]');
    this.deformationMap = page.locator('[data-testid="deformation-map"]');
    this.gasCompositionChart = page.locator('[data-testid="gas-composition-chart"]');
    this.thermalImagery = page.locator('[data-testid="thermal-imagery"]');
    this.volcanoMap = page.locator('[data-testid="volcano-map"]');
    
    // Analysis tools
    this.eventDetection = page.locator('[data-testid="event-detection"]');
    this.trendAnalysis = page.locator('[data-testid="trend-analysis"]');
    this.correlationAnalysis = page.locator('[data-testid="correlation-analysis"]');
    this.riskAssessment = page.locator('[data-testid="risk-assessment"]');
    
    // Panels
    this.statisticsPanel = page.locator('[data-testid="statistics-panel"]');
    this.alertsPanel = page.locator('[data-testid="alerts-panel"]');
  }

  async goto(): Promise<void> {
    await super.goto('/volcanic-monitoring');
  }

  /**
   * Wait for monitoring page to load with real-time data
   */
  async waitForMonitoringLoad(): Promise<void> {
    await this.waitForPageReady();
    await this.waitForGeologicalDataReady();
    
    // Wait for real-time data streams to connect
    await this.waitForWebSocketConnection();
    
    // Verify main components are loaded
    await expect(this.pageTitle).toBeVisible();
    await expect(this.volcanoList).toBeVisible();
    await this.waitForDataLoad();
  }

  /**
   * Wait for volcano data to load
   */
  async waitForDataLoad(): Promise<void> {
    // Wait for volcano data to load
    await this.page.waitForFunction(() => {
      const volcanoCards = document.querySelectorAll('[data-testid="volcano-card"]');
      return volcanoCards.length > 0;
    }, { timeout: 10000 });
    
    await this.waitForGeologicalDataReady();
  }

  /**
   * Select a specific volcano for monitoring
   */
  async selectVolcano(volcanoId: string): Promise<void> {
    const volcanoCard = this.volcanoList.locator(`[data-volcano-id="${volcanoId}"]`);
    await volcanoCard.click();
    
    // Verify selection
    await expect(volcanoCard).toHaveClass(/selected/);
    
    // Wait for detail view to update
    await expect(this.detailView).toContainText(volcanoId);
    
    // Wait for volcano-specific data to load
    await this.waitForGeologicalDataReady();
  }

  /**
   * Filter volcanoes by alert level
   */
  async filterByAlertLevel(alertLevel: 'green' | 'yellow' | 'orange' | 'red'): Promise<void> {
    await this.alertLevelFilter.click();
    
    const option = this.page.locator(`[data-testid="alert-level-${alertLevel}"]`);
    await option.click();
    
    // Verify filter applied
    const visibleVolcanoes = this.volcanoList.locator('[data-testid="volcano-card"]:visible');
    const count = await visibleVolcanoes.count();
    
    // All visible volcanoes should have the selected alert level
    for (let i = 0; i < count; i++) {
      const volcano = visibleVolcanoes.nth(i);
      const alertChip = volcano.locator('[data-testid="alert-level-chip"]');
      await expect(alertChip).toContainText(alertLevel);
    }
  }

  /**
   * Switch to map view
   */
  async switchToMapView(): Promise<void> {
    const mapViewButton = this.page.locator('[data-testid="map-view-button"]');
    await mapViewButton.click();
    
    await expect(this.mapView).toBeVisible();
    
    // Wait for map to load
    await this.page.waitForFunction(() => {
      const mapContainer = document.querySelector('[data-testid="map-view"]');
      return mapContainer && mapContainer.children.length > 0;
    });
  }

  /**
   * Switch to detail view
   */
  async switchToDetailView(): Promise<void> {
    const detailViewButton = this.page.locator('[data-testid="detail-view-button"]');
    await detailViewButton.click();
    
    await expect(this.detailView).toBeVisible();
  }

  /**
   * Enable real-time monitoring mode
   */
  async enableRealTimeMode(): Promise<void> {
    await this.realTimeToggle.check();
    await expect(this.realTimeToggle).toBeChecked();
    
    // Verify real-time indicator appears
    const realTimeIndicator = this.page.locator('[data-testid="real-time-indicator"]');
    await expect(realTimeIndicator).toBeVisible();
  }

  /**
   * Disable real-time monitoring mode
   */
  async disableRealTimeMode(): Promise<void> {
    await this.realTimeToggle.uncheck();
    await expect(this.realTimeToggle).not.toBeChecked();
    
    // Verify real-time indicator disappears
    const realTimeIndicator = this.page.locator('[data-testid="real-time-indicator"]');
    await expect(realTimeIndicator).not.toBeVisible();
  }

  /**
   * Refresh monitoring data
   */
  async refreshData(): Promise<void> {
    await this.refreshButton.click();
    
    // Wait for refresh to complete
    const loadingIndicator = this.page.locator('[data-testid="loading-indicator"]');
    await expect(loadingIndicator).toBeVisible();
    await expect(loadingIndicator).not.toBeVisible();
    
    await this.waitForGeologicalDataReady();
  }

  /**
   * Verify volcano data accuracy
   */
  async verifyVolcanoData(volcanoId: string, expectedData: {
    alertLevel?: string;
    eruptionProbability?: number;
    deformationRate?: number;
    thermalAnomalies?: number;
  }): Promise<void> {
    const volcanoCard = this.volcanoList.locator(`[data-volcano-id="${volcanoId}"]`);
    
    if (expectedData.alertLevel) {
      const alertChip = volcanoCard.locator('[data-testid="alert-level-chip"]');
      await expect(alertChip).toContainText(expectedData.alertLevel);
    }
    
    if (expectedData.eruptionProbability !== undefined) {
      const probabilityText = volcanoCard.locator('[data-testid="eruption-probability"]');
      await expect(probabilityText).toContainText(`${expectedData.eruptionProbability}%`);
    }
    
    if (expectedData.deformationRate !== undefined) {
      const deformationText = volcanoCard.locator('[data-testid="deformation-rate"]');
      await expect(deformationText).toContainText(`${expectedData.deformationRate}`);
    }
    
    if (expectedData.thermalAnomalies !== undefined) {
      const anomaliesText = volcanoCard.locator('[data-testid="thermal-anomalies"]');
      await expect(anomaliesText).toContainText(`${expectedData.thermalAnomalies}`);
    }
  }

  /**
   * Verify seismic data display and real-time updates
   */
  async verifySeismicData(): Promise<void> {
    await expect(this.seismicData).toBeVisible();
    await expect(this.seismicWaveforms).toBeVisible();
    
    // Verify recent seismic events
    const recentEvents = this.page.locator('[data-testid="recent-seismic-event"]');
    const eventCount = await recentEvents.count();
    expect(eventCount).toBeGreaterThan(0);
    
    // Verify magnitude and time information
    const firstEvent = recentEvents.first();
    await expect(firstEvent.locator('[data-testid="event-magnitude"]')).toHaveText(/M\d+\.\d+/);
    await expect(firstEvent.locator('[data-testid="event-time"]')).toBeVisible();
    
    // Test waveform interaction
    const waveformCanvas = this.seismicWaveforms.locator('canvas');
    await expect(waveformCanvas).toBeVisible();
    
    // Click on waveform to test interactivity
    await waveformCanvas.click();
    const waveformDetails = this.page.locator('[data-testid="waveform-details"]');
    await expect(waveformDetails).toBeVisible();
  }

  /**
   * Verify deformation data and InSAR processing
   */
  async verifyDeformationData(): Promise<void> {
    await expect(this.deformationData).toBeVisible();
    await expect(this.deformationMap).toBeVisible();
    
    // Verify deformation measurements
    const deformationValues = this.page.locator('[data-testid="deformation-value"]');
    const valueCount = await deformationValues.count();
    expect(valueCount).toBeGreaterThan(0);
    
    // Verify units are displayed (mm/year or cm/year)
    const firstValue = deformationValues.first();
    const valueText = await firstValue.textContent();
    expect(valueText).toMatch(/\d+(\.\d+)?\s*(mm|cm)\/year/);
    
    // Test deformation map interaction
    const mapContainer = this.deformationMap;
    await mapContainer.click();
    
    // Verify map controls are available
    const mapControls = this.page.locator('[data-testid="map-controls"]');
    await expect(mapControls).toBeVisible();
  }

  /**
   * Verify gas emission monitoring
   */
  async verifyGasEmissions(): Promise<void> {
    await expect(this.gasEmissions).toBeVisible();
    await expect(this.gasCompositionChart).toBeVisible();
    
    // Verify gas composition data
    const gasTypes = this.page.locator('[data-testid="gas-type"]');
    const gasTypeCount = await gasTypes.count();
    expect(gasTypeCount).toBeGreaterThan(0);
    
    // Verify common volcanic gases are monitored
    const so2Data = this.page.locator('[data-testid="so2-concentration"]');
    const co2Data = this.page.locator('[data-testid="co2-concentration"]');
    const h2sData = this.page.locator('[data-testid="h2s-concentration"]');
    
    await expect(so2Data).toBeVisible();
    await expect(co2Data).toBeVisible();
    await expect(h2sData).toBeVisible();
    
    // Verify concentration units (ppm, ppb, etc.)
    const so2Text = await so2Data.textContent();
    expect(so2Text).toMatch(/\d+(\.\d+)?\s*(ppm|ppb|mg\/m³)/);
  }

  /**
   * Verify thermal data and satellite imagery
   */
  async verifyThermalData(): Promise<void> {
    await expect(this.thermalData).toBeVisible();
    await expect(this.thermalImagery).toBeVisible();
    
    // Verify thermal anomaly detection
    const thermalAnomalies = this.page.locator('[data-testid="thermal-anomaly"]');
    const anomalyCount = await thermalAnomalies.count();
    
    if (anomalyCount > 0) {
      const firstAnomaly = thermalAnomalies.first();
      
      // Verify temperature values
      const temperature = firstAnomaly.locator('[data-testid="anomaly-temperature"]');
      await expect(temperature).toBeVisible();
      
      const tempText = await temperature.textContent();
      expect(tempText).toMatch(/\d+(\.\d+)?°[CF]/);
    }
    
    // Test thermal imagery interaction
    const thermalImage = this.thermalImagery.locator('img, canvas');
    await expect(thermalImage).toBeVisible();
  }

  /**
   * Verify statistics panel
   */
  async verifyStatistics(expectedStats: {
    totalVolcanoes?: number;
    activeAlerts?: number;
    criticalAlerts?: number;
    averageRisk?: number;
  }): Promise<void> {
    if (expectedStats.totalVolcanoes !== undefined) {
      const totalText = this.statisticsPanel.locator('[data-testid="total-volcanoes"]');
      await expect(totalText).toContainText(`${expectedStats.totalVolcanoes}`);
    }
    
    if (expectedStats.activeAlerts !== undefined) {
      const activeText = this.statisticsPanel.locator('[data-testid="active-alerts"]');
      await expect(activeText).toContainText(`${expectedStats.activeAlerts}`);
    }
    
    if (expectedStats.criticalAlerts !== undefined) {
      const criticalText = this.statisticsPanel.locator('[data-testid="critical-alerts"]');
      await expect(criticalText).toContainText(`${expectedStats.criticalAlerts}`);
    }
    
    if (expectedStats.averageRisk !== undefined) {
      const riskText = this.statisticsPanel.locator('[data-testid="average-risk"]');
      await expect(riskText).toContainText(`${expectedStats.averageRisk}%`);
    }
  }

  /**
   * Verify alerts panel
   */
  async verifyAlerts(): Promise<void> {
    const alertItems = this.alertsPanel.locator('[data-testid="alert-item"]');
    const alertCount = await alertItems.count();
    
    // Verify each alert has required information
    for (let i = 0; i < alertCount; i++) {
      const alert = alertItems.nth(i);
      
      // Should have alert level
      const alertLevel = alert.locator('[data-testid="alert-level"]');
      await expect(alertLevel).toBeVisible();
      
      // Should have timestamp
      const timestamp = alert.locator('[data-testid="alert-timestamp"]');
      await expect(timestamp).toBeVisible();
      
      // Should have volcano ID
      const volcanoId = alert.locator('[data-testid="alert-volcano-id"]');
      await expect(volcanoId).toBeVisible();
    }
  }

  /**
   * Export monitoring data
   */
  async exportData(format: 'csv' | 'json' | 'pdf'): Promise<any> {
    await this.exportButton.click();
    
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
   * Verify map interaction functionality
   */
  async verifyMapInteraction(): Promise<void> {
    await this.switchToMapView();
    
    // Verify map controls are present
    const zoomIn = this.mapView.locator('[data-testid="zoom-in"]');
    const zoomOut = this.mapView.locator('[data-testid="zoom-out"]');
    
    await expect(zoomIn).toBeVisible();
    await expect(zoomOut).toBeVisible();
    
    // Test zoom functionality
    await zoomIn.click();
    await this.page.waitForTimeout(500);
    
    await zoomOut.click();
    await this.page.waitForTimeout(500);
    
    // Verify volcano markers are clickable
    const volcanoMarkers = this.mapView.locator('[data-testid="volcano-marker"]');
    const markerCount = await volcanoMarkers.count();
    
    if (markerCount > 0) {
      const firstMarker = volcanoMarkers.first();
      await firstMarker.click();
      
      // Should show volcano popup
      const popup = this.mapView.locator('[data-testid="volcano-popup"]');
      await expect(popup).toBeVisible();
    }
  }

  /**
   * Validate page content and monitoring accuracy
   */
  async validatePageContent(): Promise<void> {
    // Wait for monitoring page to load
    await this.waitForMonitoringLoad();
    
    // Verify coordinate system is properly initialized
    await this.verifyCoordinateSystem();
    
    // Test all monitoring components
    await this.verifySeismicData();
    await this.verifyDeformationData();
    await this.verifyGasEmissions();
    await this.verifyThermalData();
    
    // Verify panels
    await this.verifyAlerts();
    
    // Test map interaction
    await this.verifyMapInteraction();
    
    // Check for any errors
    await this.checkForErrors();
  }

  /**
   * Verify responsive design across different screen sizes
   */
  async verifyResponsiveDesign(): Promise<void> {
    // Test mobile layout
    await this.page.setViewportSize({ width: 375, height: 667 });
    
    // Volcano list should stack vertically
    const volcanoCards = this.volcanoList.locator('[data-testid="volcano-card"]');
    const cardCount = await volcanoCards.count();
    
    if (cardCount > 1) {
      const firstCard = volcanoCards.nth(0);
      const secondCard = volcanoCards.nth(1);
      
      const firstBox = await firstCard.boundingBox();
      const secondBox = await secondCard.boundingBox();
      
      // Cards should stack vertically on mobile
      expect(secondBox?.y).toBeGreaterThan((firstBox?.y || 0) + (firstBox?.height || 0));
    }
    
    // Test tablet layout
    await this.page.setViewportSize({ width: 768, height: 1024 });
    
    // Statistics should be visible
    await expect(this.statisticsPanel).toBeVisible();
    
    // Test desktop layout
    await this.page.setViewportSize({ width: 1920, height: 1080 });
    
    // All panels should be visible
    await expect(this.volcanoList).toBeVisible();
    await expect(this.statisticsPanel).toBeVisible();
    await expect(this.detailView).toBeVisible();
  }
}