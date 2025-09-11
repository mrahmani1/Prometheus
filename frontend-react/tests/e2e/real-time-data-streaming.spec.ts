import { test, expect } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';
import { VolcanicMonitoringPage } from '../page-objects/VolcanicMonitoringPage';
import { MockApiService, MockWebSocketService } from '../fixtures/mock-services';
import { mockWebSocketMessages, performanceBenchmarks } from '../fixtures/test-data';

test.describe('Real-Time Data Streaming Tests', () => {
  let dashboardPage: DashboardPage;
  let monitoringPage: VolcanicMonitoringPage;
  let mockApi: MockApiService;
  let mockWebSocket: MockWebSocketService;

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page);
    monitoringPage = new VolcanicMonitoringPage(page);
    mockApi = new MockApiService(page);
    mockWebSocket = new MockWebSocketService(page);

    await mockApi.setupSuccessfulMocks();
    await mockWebSocket.setupWebSocketMocks();

    // Ensure user is authenticated
    await page.goto('/dashboard');
    await dashboardPage.ensureAuthenticated();
  });

  test('WebSocket connection establishment and validation', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Verify WebSocket connection is established
    await expect(page.locator('[data-testid="websocket-connected"]')).toBeVisible();
    
    // Check connection status indicator
    const connectionStatus = page.locator('[data-testid="connection-status"]');
    await expect(connectionStatus).toHaveClass(/connected/);
    await expect(connectionStatus).toContainText('Connected');

    // Verify connection metadata
    const connectionInfo = await page.evaluate(() => {
      return (window as any).websocketInfo || {};
    });
    
    expect(connectionInfo.readyState).toBe(1); // WebSocket.OPEN
    expect(connectionInfo.url).toContain('ws://');
  });

  test('Real-time seismic event streaming', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.waitForMonitoringLoad();
    await monitoringPage.enableRealTimeMode();

    // Get initial event count
    const initialEvents = await page.locator('[data-testid="recent-seismic-event"]').count();

    // Simulate new seismic event
    await mockWebSocket.simulateSeismicEvent();

    // Verify new event appears in real-time
    await expect(page.locator('[data-testid="recent-seismic-event"]')).toHaveCount(initialEvents + 1);

    // Verify event data accuracy
    const latestEvent = page.locator('[data-testid="recent-seismic-event"]').first();
    await expect(latestEvent.locator('[data-testid="event-magnitude"]')).toBeVisible();
    await expect(latestEvent.locator('[data-testid="event-time"]')).toBeVisible();
    await expect(latestEvent.locator('[data-testid="event-location"]')).toBeVisible();

    // Verify event appears on map in real-time
    const eventMarker = page.locator('[data-testid="seismic-marker"]').first();
    await expect(eventMarker).toBeVisible();

    // Test multiple rapid events
    for (let i = 0; i < 5; i++) {
      await mockWebSocket.simulateSeismicEvent();
      await page.waitForTimeout(200);
    }

    // Verify all events are processed
    const finalEventCount = await page.locator('[data-testid="recent-seismic-event"]').count();
    expect(finalEventCount).toBeGreaterThan(initialEvents + 3);
  });

  test('Real-time volcanic alert level changes', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Get initial alert level
    const initialAlertLevel = await page.locator('[data-testid="alert-level"]').first().textContent();

    // Simulate alert level change
    await mockWebSocket.simulateAlertLevelChange();

    // Verify alert level updates in real-time
    await expect(page.locator('[data-testid="alert-notification"]')).toBeVisible();
    
    // Verify new alert level is displayed
    const updatedAlertLevel = await page.locator('[data-testid="alert-level"]').first().textContent();
    expect(updatedAlertLevel).not.toBe(initialAlertLevel);

    // Verify alert appears in alert panel
    const alertPanel = page.locator('[data-testid="alerts-panel"]');
    const newAlert = alertPanel.locator('[data-testid="alert-item"]').first();
    await expect(newAlert).toBeVisible();
    await expect(newAlert).toContainText('orange'); // From mock data

    // Verify alert timestamp is recent
    const alertTimestamp = await newAlert.locator('[data-testid="alert-timestamp"]').textContent();
    const alertTime = new Date(alertTimestamp || '').getTime();
    const now = Date.now();
    expect(now - alertTime).toBeLessThan(60000); // Within last minute

    // Test alert sound notification (if enabled)
    const audioElement = page.locator('audio[data-testid="alert-sound"]');
    if (await audioElement.isVisible()) {
      const isPlaying = await audioElement.evaluate((audio: HTMLAudioElement) => !audio.paused);
      expect(isPlaying).toBe(true);
    }
  });

  test('Real-time deformation data updates', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.selectVolcano('yellowstone');
    await monitoringPage.enableRealTimeMode();

    // Get initial deformation values
    const initialDeformation = await page.locator('[data-testid="deformation-value"]').first().textContent();

    // Simulate deformation update
    await mockWebSocket.simulateDeformationUpdate();

    // Verify deformation data updates
    await page.waitForTimeout(1000); // Allow for data processing
    const updatedDeformation = await page.locator('[data-testid="deformation-value"]').first().textContent();
    expect(updatedDeformation).not.toBe(initialDeformation);

    // Verify deformation map updates
    const deformationMap = page.locator('[data-testid="deformation-map"]');
    await expect(deformationMap.locator('[data-testid="deformation-vector"]')).toBeVisible();

    // Verify units are correct
    const deformationUnit = await page.locator('[data-testid="deformation-unit"]').textContent();
    expect(deformationUnit).toMatch(/mm|cm/);

    // Test deformation trend visualization
    const trendChart = page.locator('[data-testid="deformation-trend-chart"]');
    await expect(trendChart).toBeVisible();
    
    // Verify chart updates with new data point
    const dataPoints = await trendChart.locator('[data-testid="data-point"]').count();
    expect(dataPoints).toBeGreaterThan(0);
  });

  test('Real-time gas emission monitoring', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.selectVolcano('kilauea');
    await monitoringPage.enableRealTimeMode();

    // Monitor initial gas concentrations
    const initialSO2 = await page.locator('[data-testid="so2-concentration"]').textContent();
    const initialCO2 = await page.locator('[data-testid="co2-concentration"]').textContent();

    // Simulate gas emission update
    await mockWebSocket.simulateGasEmissionUpdate();

    // Verify gas concentrations update
    await page.waitForTimeout(1000);
    const updatedSO2 = await page.locator('[data-testid="so2-concentration"]').textContent();
    const updatedCO2 = await page.locator('[data-testid="co2-concentration"]').textContent();

    expect(updatedSO2).not.toBe(initialSO2);
    expect(updatedCO2).not.toBe(initialCO2);

    // Verify gas composition chart updates
    const gasChart = page.locator('[data-testid="gas-composition-chart"]');
    await expect(gasChart).toBeVisible();

    // Verify concentration values are within realistic ranges
    const so2Value = parseFloat(updatedSO2?.match(/\d+(\.\d+)?/)?.[0] || '0');
    const co2Value = parseFloat(updatedCO2?.match(/\d+(\.\d+)?/)?.[0] || '0');

    expect(so2Value).toBeGreaterThan(0);
    expect(so2Value).toBeLessThan(10000); // Reasonable upper limit for SO2 ppm
    expect(co2Value).toBeGreaterThan(0);
    expect(co2Value).toBeLessThan(50000); // Reasonable upper limit for CO2 ppm
  });

  test('WebSocket connection resilience and reconnection', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Verify initial connection
    await expect(page.locator('[data-testid="websocket-connected"]')).toBeVisible();

    // Simulate connection loss
    await mockWebSocket.simulateDisconnection();

    // Verify disconnection is detected
    await expect(page.locator('[data-testid="connection-lost"]')).toBeVisible();
    await expect(page.locator('[data-testid="reconnecting"]')).toBeVisible();

    // Simulate reconnection
    await page.waitForTimeout(2000); // Simulate reconnection delay
    await mockWebSocket.simulateConnection();

    // Verify reconnection
    await expect(page.locator('[data-testid="connection-restored"]')).toBeVisible();
    await expect(page.locator('[data-testid="websocket-connected"]')).toBeVisible();

    // Verify data streaming resumes after reconnection
    await mockWebSocket.simulateSeismicEvent();
    await expect(page.locator('[data-testid="recent-seismic-event"]').first()).toBeVisible();
  });

  test('Real-time data buffering and synchronization', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.enableRealTimeMode();

    // Simulate rapid data updates
    const updatePromises = [];
    for (let i = 0; i < 10; i++) {
      updatePromises.push(mockWebSocket.simulateSeismicEvent());
      updatePromises.push(mockWebSocket.simulateDeformationUpdate());
      updatePromises.push(mockWebSocket.simulateGasEmissionUpdate());
    }

    await Promise.all(updatePromises);

    // Verify data is properly buffered and displayed
    await page.waitForTimeout(2000); // Allow for data processing

    const seismicEvents = await page.locator('[data-testid="recent-seismic-event"]').count();
    expect(seismicEvents).toBeGreaterThan(5); // Should have processed multiple events

    // Verify no data loss or corruption
    const eventMagnitudes = await page.locator('[data-testid="event-magnitude"]').allTextContents();
    for (const magnitude of eventMagnitudes) {
      expect(magnitude).toMatch(/M\d+\.\d+/);
    }

    // Verify timestamps are in correct order (most recent first)
    const timestamps = await page.locator('[data-testid="event-time"]').allTextContents();
    for (let i = 1; i < timestamps.length; i++) {
      const current = new Date(timestamps[i-1]).getTime();
      const next = new Date(timestamps[i]).getTime();
      expect(current).toBeGreaterThanOrEqual(next);
    }
  });

  test('Real-time performance under load', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.waitForDashboardLoad();

    // Measure initial performance
    const initialMetrics = await dashboardPage.getPerformanceMetrics();

    // Simulate high-frequency data updates
    const startTime = Date.now();
    
    for (let i = 0; i < 50; i++) {
      await mockWebSocket.simulateSeismicEvent();
      await mockWebSocket.simulateDeformationUpdate();
      
      if (i % 10 === 0) {
        await mockWebSocket.simulateAlertLevelChange();
      }
      
      // Small delay to simulate realistic timing
      await page.waitForTimeout(50);
    }

    const endTime = Date.now();
    const processingTime = endTime - startTime;

    // Verify performance remains acceptable
    expect(processingTime).toBeLessThan(10000); // Should process 50 updates in under 10 seconds

    // Measure final performance
    const finalMetrics = await dashboardPage.getPerformanceMetrics();

    // Verify performance hasn't degraded significantly
    const performanceDegradation = finalMetrics.loadTime - initialMetrics.loadTime;
    expect(performanceDegradation).toBeLessThan(1000); // Less than 1 second degradation

    // Verify memory usage is stable
    const memoryUsage = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });
    
    expect(memoryUsage).toBeLessThan(performanceBenchmarks.rendering.maxMemory * 1024 * 1024);
  });

  test('Real-time data validation and error handling', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.enableRealTimeMode();

    // Simulate invalid data
    await page.evaluate(() => {
      const event = new CustomEvent('websocket-message', {
        detail: {
          type: 'seismic_event',
          data: {
            id: 'invalid_event',
            magnitude: 'invalid', // Invalid magnitude
            depth: -100, // Invalid depth
            timestamp: 'invalid_date'
          }
        }
      });
      document.dispatchEvent(event);
    });

    // Verify error handling
    await expect(page.locator('[data-testid="data-validation-error"]')).toBeVisible();

    // Verify system continues to function with valid data
    await mockWebSocket.simulateSeismicEvent();
    await expect(page.locator('[data-testid="recent-seismic-event"]').first()).toBeVisible();

    // Test WebSocket error handling
    await mockWebSocket.simulateError();
    await expect(page.locator('[data-testid="websocket-error"]')).toBeVisible();

    // Verify automatic recovery attempt
    await expect(page.locator('[data-testid="reconnecting"]')).toBeVisible();
  });

  test('Real-time data filtering and aggregation', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.enableRealTimeMode();

    // Set up data filters
    await page.locator('[data-testid="magnitude-filter"]').fill('3.0');
    await page.locator('[data-testid="apply-filter"]').click();

    // Simulate events with different magnitudes
    await page.evaluate(() => {
      // Event below threshold
      const lowMagEvent = new CustomEvent('websocket-message', {
        detail: {
          type: 'seismic_event',
          data: {
            id: 'low_mag_event',
            magnitude: 2.5,
            depth: 10,
            timestamp: new Date().toISOString()
          }
        }
      });
      document.dispatchEvent(lowMagEvent);

      // Event above threshold
      const highMagEvent = new CustomEvent('websocket-message', {
        detail: {
          type: 'seismic_event',
          data: {
            id: 'high_mag_event',
            magnitude: 4.2,
            depth: 8,
            timestamp: new Date().toISOString()
          }
        }
      });
      document.dispatchEvent(highMagEvent);
    });

    // Verify filtering works
    await page.waitForTimeout(1000);
    
    const visibleEvents = await page.locator('[data-testid="recent-seismic-event"]').count();
    const eventMagnitudes = await page.locator('[data-testid="event-magnitude"]').allTextContents();
    
    // All visible events should be above threshold
    for (const magnitudeText of eventMagnitudes) {
      const magnitude = parseFloat(magnitudeText.replace('M', ''));
      expect(magnitude).toBeGreaterThanOrEqual(3.0);
    }

    // Test real-time aggregation
    const aggregationPanel = page.locator('[data-testid="real-time-statistics"]');
    await expect(aggregationPanel).toBeVisible();
    
    const eventCount = await aggregationPanel.locator('[data-testid="event-count"]').textContent();
    const avgMagnitude = await aggregationPanel.locator('[data-testid="avg-magnitude"]').textContent();
    
    expect(eventCount).toMatch(/\d+/);
    expect(avgMagnitude).toMatch(/\d+\.\d+/);
  });
});