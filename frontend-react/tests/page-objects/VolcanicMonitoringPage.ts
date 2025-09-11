import { Page, Locator, expect } from '@playwright/test';

export class VolcanicMonitoringPage {
  readonly page: Page;
  readonly pageTitle: Locator;
  readonly volcanoList: Locator;
  readonly alertLevelFilter: Locator;
  readonly mapView: Locator;
  readonly detailView: Locator;
  readonly realTimeToggle: Locator;
  readonly refreshButton: Locator;
  readonly exportButton: Locator;
  readonly statisticsPanel: Locator;
  readonly alertsPanel: Locator;

  constructor(page: Page) {
    this.page = page;
    this.pageTitle = page.locator('[data-testid="volcanic-monitoring-title"]');
    this.volcanoList = page.locator('[data-testid="volcano-list"]');
    this.alertLevelFilter = page.locator('[data-testid="alert-level-filter"]');
    this.mapView = page.locator('[data-testid="map-view"]');
    this.detailView = page.locator('[data-testid="detail-view"]');
    this.realTimeToggle = page.locator('[data-testid="real-time-toggle"]');
    this.refreshButton = page.locator('[data-testid="refresh-button"]');
    this.exportButton = page.locator('[data-testid="export-button"]');
    this.statisticsPanel = page.locator('[data-testid="statistics-panel"]');
    this.alertsPanel = page.locator('[data-testid="alerts-panel"]');
  }

  async goto() {
    await this.page.goto('/volcanic-monitoring');
    await this.waitForLoad();
  }

  async waitForLoad() {
    await expect(this.pageTitle).toBeVisible();
    await expect(this.volcanoList).toBeVisible();
    await this.waitForDataLoad();
  }

  async waitForDataLoad() {
    // Wait for volcano data to load
    await this.page.waitForFunction(() => {
      const volcanoCards = document.querySelectorAll('[data-testid="volcano-card"]');
      return volcanoCards.length > 0;
    }, { timeout: 10000 });
  }

  async selectVolcano(volcanoId: string) {
    const volcanoCard = this.volcanoList.locator(`[data-volcano-id="${volcanoId}"]`);
    await volcanoCard.click();
    
    // Verify selection
    await expect(volcanoCard).toHaveClass(/selected/);
    
    // Wait for detail view to update
    await expect(this.detailView).toContainText(volcanoId);
  }

  async filterByAlertLevel(alertLevel: 'green' | 'yellow' | 'orange' | 'red') {
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

  async switchToMapView() {
    const mapViewButton = this.page.locator('[data-testid="map-view-button"]');
    await mapViewButton.click();
    
    await expect(this.mapView).toBeVisible();
    
    // Wait for map to load
    await this.page.waitForFunction(() => {
      const mapContainer = document.querySelector('[data-testid="map-view"]');
      return mapContainer && mapContainer.children.length > 0;
    });
  }

  async switchToDetailView() {
    const detailViewButton = this.page.locator('[data-testid="detail-view-button"]');
    await detailViewButton.click();
    
    await expect(this.detailView).toBeVisible();
  }

  async enableRealTimeMode() {
    await this.realTimeToggle.check();
    await expect(this.realTimeToggle).toBeChecked();
    
    // Verify real-time indicator appears
    const realTimeIndicator = this.page.locator('[data-testid="real-time-indicator"]');
    await expect(realTimeIndicator).toBeVisible();
  }

  async disableRealTimeMode() {
    await this.realTimeToggle.uncheck();
    await expect(this.realTimeToggle).not.toBeChecked();
    
    // Verify real-time indicator disappears
    const realTimeIndicator = this.page.locator('[data-testid="real-time-indicator"]');
    await expect(realTimeIndicator).not.toBeVisible();
  }

  async refreshData() {
    await this.refreshButton.click();
    
    // Wait for refresh to complete
    const loadingIndicator = this.page.locator('[data-testid="loading-indicator"]');
    await expect(loadingIndicator).toBeVisible();
    await expect(loadingIndicator).not.toBeVisible();
  }

  async verifyVolcanoData(volcanoId: string, expectedData: {
    alertLevel?: string;
    eruptionProbability?: number;
    deformationRate?: number;
    thermalAnomalies?: number;
  }) {
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

  async verifyStatistics(expectedStats: {
    totalVolcanoes?: number;
    activeAlerts?: number;
    criticalAlerts?: number;
    averageRisk?: number;
  }) {
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

  async verifyAlerts() {
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

  async exportData(format: 'csv' | 'json' | 'pdf') {
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

  async verifyMapInteraction() {
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

  async verifyAccessibility() {
    // Check ARIA labels and roles
    await expect(this.pageTitle).toHaveAttribute('role', 'heading');
    await expect(this.volcanoList).toHaveAttribute('role', 'list');
    
    // Check keyboard navigation
    const volcanoCards = this.volcanoList.locator('[data-testid="volcano-card"]');
    const cardCount = await volcanoCards.count();
    
    if (cardCount > 0) {
      const firstCard = volcanoCards.first();
      await firstCard.focus();
      await expect(firstCard).toBeFocused();
      
      // Test keyboard selection
      await this.page.keyboard.press('Enter');
      await expect(firstCard).toHaveClass(/selected/);
    }
    
    // Check color contrast for alert levels
    const alertChips = this.page.locator('[data-testid="alert-level-chip"]');
    const chipCount = await alertChips.count();
    
    for (let i = 0; i < chipCount; i++) {
      const chip = alertChips.nth(i);
      const backgroundColor = await chip.evaluate(el => 
        window.getComputedStyle(el).backgroundColor
      );
      const color = await chip.evaluate(el => 
        window.getComputedStyle(el).color
      );
      
      // Verify colors are defined (not transparent)
      expect(backgroundColor).not.toBe('rgba(0, 0, 0, 0)');
      expect(color).not.toBe('rgba(0, 0, 0, 0)');
    }
  }

  async verifyResponsiveDesign() {
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