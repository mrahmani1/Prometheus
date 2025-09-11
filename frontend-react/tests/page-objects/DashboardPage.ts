import { Page, Locator, expect } from '@playwright/test';

export class DashboardPage {
  readonly page: Page;
  readonly dashboardTitle: Locator;
  readonly addWidgetButton: Locator;
  readonly editModeToggle: Locator;
  readonly saveButton: Locator;
  readonly refreshButton: Locator;
  readonly widgetContainer: Locator;
  readonly alertPanel: Locator;
  readonly statisticsPanel: Locator;
  readonly realTimeIndicator: Locator;

  constructor(page: Page) {
    this.page = page;
    this.dashboardTitle = page.locator('[data-testid="dashboard-title"]');
    this.addWidgetButton = page.locator('[data-testid="add-widget-button"]');
    this.editModeToggle = page.locator('[data-testid="edit-mode-toggle"]');
    this.saveButton = page.locator('[data-testid="save-button"]');
    this.refreshButton = page.locator('[data-testid="refresh-button"]');
    this.widgetContainer = page.locator('[data-testid="widget-container"]');
    this.alertPanel = page.locator('[data-testid="alert-panel"]');
    this.statisticsPanel = page.locator('[data-testid="statistics-panel"]');
    this.realTimeIndicator = page.locator('[data-testid="real-time-indicator"]');
  }

  async goto() {
    await this.page.goto('/dashboard');
    await this.waitForLoad();
  }

  async waitForLoad() {
    await expect(this.dashboardTitle).toBeVisible();
    await expect(this.widgetContainer).toBeVisible();
  }

  async enableEditMode() {
    await this.editModeToggle.click();
    await expect(this.editModeToggle).toHaveAttribute('aria-pressed', 'true');
  }

  async disableEditMode() {
    await this.editModeToggle.click();
    await expect(this.editModeToggle).toHaveAttribute('aria-pressed', 'false');
  }

  async addWidget(widgetType: string) {
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

  async removeWidget(widgetId: string) {
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

  async moveWidget(widgetId: string, targetPosition: { x: number; y: number }) {
    const widget = this.widgetContainer.locator(`[data-widget-id="${widgetId}"]`);
    const widgetHeader = widget.locator('[data-testid="widget-header"]');
    
    // Get current position
    const currentBox = await widget.boundingBox();
    if (!currentBox) throw new Error('Widget not found');
    
    // Drag to new position
    await widgetHeader.dragTo(this.widgetContainer, {
      targetPosition
    });
    
    // Verify widget moved
    const newBox = await widget.boundingBox();
    expect(newBox?.x).not.toBe(currentBox.x);
  }

  async resizeWidget(widgetId: string, newSize: { width: number; height: number }) {
    const widget = this.widgetContainer.locator(`[data-widget-id="${widgetId}"]`);
    const resizeHandle = widget.locator('[data-testid="resize-handle"]');
    
    // Get current size
    const currentBox = await widget.boundingBox();
    if (!currentBox) throw new Error('Widget not found');
    
    // Drag resize handle
    await resizeHandle.dragTo(widget, {
      targetPosition: {
        x: currentBox.x + newSize.width,
        y: currentBox.y + newSize.height
      }
    });
    
    // Verify widget resized
    const newBox = await widget.boundingBox();
    expect(newBox?.width).toBeCloseTo(newSize.width, 10);
    expect(newBox?.height).toBeCloseTo(newSize.height, 10);
  }

  async saveDashboard() {
    await this.saveButton.click();
    
    // Wait for save confirmation
    const notification = this.page.locator('[data-testid="notification"]');
    await expect(notification).toContainText('saved');
  }

  async refreshDashboard() {
    await this.refreshButton.click();
    
    // Wait for refresh to complete
    await this.page.waitForLoadState('networkidle');
  }

  async getWidgetCount() {
    const widgets = this.widgetContainer.locator('[data-testid^="widget-"]');
    return await widgets.count();
  }

  async getAlertCount() {
    const alerts = this.alertPanel.locator('[data-testid="alert-item"]');
    return await alerts.count();
  }

  async verifyRealTimeMode(enabled: boolean) {
    if (enabled) {
      await expect(this.realTimeIndicator).toBeVisible();
      await expect(this.realTimeIndicator).toContainText('Live');
    } else {
      await expect(this.realTimeIndicator).not.toBeVisible();
    }
  }

  async waitForDataLoad() {
    // Wait for loading indicators to disappear
    const loadingIndicators = this.page.locator('[data-testid="loading"]');
    await expect(loadingIndicators).toHaveCount(0);
    
    // Wait for data to be displayed
    await this.page.waitForFunction(() => {
      const widgets = document.querySelectorAll('[data-testid^="widget-"]');
      return Array.from(widgets).some(widget => 
        widget.textContent && widget.textContent.trim().length > 0
      );
    });
  }

  async verifyAccessibility() {
    // Check for proper ARIA labels
    await expect(this.dashboardTitle).toHaveAttribute('role', 'heading');
    await expect(this.addWidgetButton).toHaveAttribute('aria-label');
    await expect(this.editModeToggle).toHaveAttribute('aria-pressed');
    
    // Check keyboard navigation
    await this.addWidgetButton.focus();
    await expect(this.addWidgetButton).toBeFocused();
    
    await this.page.keyboard.press('Tab');
    await expect(this.editModeToggle).toBeFocused();
  }

  async verifyResponsiveLayout(viewport: { width: number; height: number }) {
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

  async exportDashboard(format: 'pdf' | 'png' | 'json') {
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

  async verifyPerformance() {
    // Measure initial load time
    const startTime = Date.now();
    await this.waitForLoad();
    const loadTime = Date.now() - startTime;
    
    // Should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
    
    // Check for memory leaks by monitoring widget count
    const initialWidgetCount = await this.getWidgetCount();
    
    // Add and remove widgets multiple times
    for (let i = 0; i < 5; i++) {
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