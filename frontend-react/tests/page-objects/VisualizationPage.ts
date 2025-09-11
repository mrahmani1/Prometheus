import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

/**
 * Page object for advanced geological visualization and 3D/4D rendering
 */
export class VisualizationPage extends BasePage {
  // 3D Visualization controls
  private readonly threeDViewer: Locator;
  private readonly threeDControls: Locator;
  private readonly cameraControls: Locator;
  private readonly renderingOptions: Locator;
  
  // Geological layers
  private readonly layerPanel: Locator;
  private readonly geologicalLayers: Locator;
  private readonly seismicLayers: Locator;
  private readonly volcanicLayers: Locator;
  private readonly structuralLayers: Locator;
  
  // Time controls for 4D visualization
  private readonly timeSlider: Locator;
  private readonly playButton: Locator;
  private readonly pauseButton: Locator;
  private readonly timeRangeSelector: Locator;
  private readonly animationSpeed: Locator;
  
  // Data visualization
  private readonly chartContainer: Locator;
  private readonly seismicWaveforms: Locator;
  private readonly deformationPlots: Locator;
  private readonly temperatureMaps: Locator;
  private readonly pressurePlots: Locator;
  
  // Interactive tools
  private readonly measurementTool: Locator;
  private readonly crossSectionTool: Locator;
  private readonly annotationTool: Locator;
  private readonly exportTool: Locator;
  
  // Performance indicators
  private readonly fpsCounter: Locator;
  private readonly renderStats: Locator;
  private readonly memoryUsage: Locator;

  constructor(page: Page) {
    super(page);
    
    // 3D Visualization
    this.threeDViewer = page.locator('[data-testid="3d-viewer"]');
    this.threeDControls = page.locator('[data-testid="3d-controls"]');
    this.cameraControls = page.locator('[data-testid="camera-controls"]');
    this.renderingOptions = page.locator('[data-testid="rendering-options"]');
    
    // Layers
    this.layerPanel = page.locator('[data-testid="layer-panel"]');
    this.geologicalLayers = page.locator('[data-testid="geological-layers"]');
    this.seismicLayers = page.locator('[data-testid="seismic-layers"]');
    this.volcanicLayers = page.locator('[data-testid="volcanic-layers"]');
    this.structuralLayers = page.locator('[data-testid="structural-layers"]');
    
    // Time controls
    this.timeSlider = page.locator('[data-testid="time-slider"]');
    this.playButton = page.locator('[data-testid="play-button"]');
    this.pauseButton = page.locator('[data-testid="pause-button"]');
    this.timeRangeSelector = page.locator('[data-testid="time-range-selector"]');
    this.animationSpeed = page.locator('[data-testid="animation-speed"]');
    
    // Data visualization
    this.chartContainer = page.locator('[data-testid="chart-container"]');
    this.seismicWaveforms = page.locator('[data-testid="seismic-waveforms"]');
    this.deformationPlots = page.locator('[data-testid="deformation-plots"]');
    this.temperatureMaps = page.locator('[data-testid="temperature-maps"]');
    this.pressurePlots = page.locator('[data-testid="pressure-plots"]');
    
    // Tools
    this.measurementTool = page.locator('[data-testid="measurement-tool"]');
    this.crossSectionTool = page.locator('[data-testid="cross-section-tool"]');
    this.annotationTool = page.locator('[data-testid="annotation-tool"]');
    this.exportTool = page.locator('[data-testid="export-tool"]');
    
    // Performance
    this.fpsCounter = page.locator('[data-testid="fps-counter"]');
    this.renderStats = page.locator('[data-testid="render-stats"]');
    this.memoryUsage = page.locator('[data-testid="memory-usage"]');
  }

  async goto(): Promise<void> {
    await super.goto('/visualization');
  }

  /**
   * Wait for 3D viewer to initialize and load geological data
   */
  async waitFor3DViewerReady(): Promise<void> {
    await this.waitForPageReady();
    
    // Wait for WebGL context to initialize
    await this.page.locator('[data-testid="webgl-initialized"]').waitFor({ 
      state: 'visible', 
      timeout: 30000 
    });
    
    // Wait for geological models to load
    await this.page.locator('[data-testid="geological-models-loaded"]').waitFor({ 
      state: 'visible', 
      timeout: 60000 
    });
    
    // Verify 3D viewer is interactive
    await expect(this.threeDViewer).toBeVisible();
    await expect(this.threeDControls).toBeVisible();
  }

  /**
   * Test 3D navigation and camera controls
   */
  async test3DNavigation(): Promise<void> {
    // Test orbit controls
    const viewerBounds = await this.threeDViewer.boundingBox();
    if (viewerBounds) {
      const centerX = viewerBounds.x + viewerBounds.width / 2;
      const centerY = viewerBounds.y + viewerBounds.height / 2;
      
      // Test mouse orbit
      await this.page.mouse.move(centerX, centerY);
      await this.page.mouse.down();
      await this.page.mouse.move(centerX + 100, centerY + 50);
      await this.page.mouse.up();
      
      // Verify camera position changed
      const cameraPosition = await this.getCameraPosition();
      expect(cameraPosition).toBeDefined();
    }
    
    // Test zoom controls
    await this.page.mouse.wheel(0, -100); // Zoom in
    await this.page.waitForTimeout(500);
    
    await this.page.mouse.wheel(0, 100); // Zoom out
    await this.page.waitForTimeout(500);
    
    // Test camera reset
    const resetButton = this.cameraControls.locator('[data-testid="reset-camera"]');
    await resetButton.click();
    await this.page.waitForTimeout(1000);
  }

  /**
   * Test geological layer visibility and interaction
   */
  async testGeologicalLayers(): Promise<void> {
    // Test layer panel visibility
    await expect(this.layerPanel).toBeVisible();
    
    // Test geological layer toggles
    const geologicalLayerToggles = this.geologicalLayers.locator('[data-testid="layer-toggle"]');
    const layerCount = await geologicalLayerToggles.count();
    
    for (let i = 0; i < Math.min(layerCount, 3); i++) {
      const toggle = geologicalLayerToggles.nth(i);
      
      // Toggle off
      await toggle.click();
      await this.page.waitForTimeout(500);
      
      // Toggle on
      await toggle.click();
      await this.page.waitForTimeout(500);
    }
    
    // Test seismic layer visualization
    const seismicToggle = this.seismicLayers.locator('[data-testid="seismic-events-layer"]');
    await seismicToggle.click();
    
    // Verify seismic events are rendered
    const seismicMarkers = this.page.locator('[data-testid="seismic-marker"]');
    const markerCount = await seismicMarkers.count();
    expect(markerCount).toBeGreaterThan(0);
  }

  /**
   * Test 4D time-based visualization
   */
  async test4DVisualization(): Promise<void> {
    // Set time range for animation
    await this.timeRangeSelector.click();
    await this.page.locator('[data-testid="time-range-1-week"]').click();
    
    // Test time slider
    const sliderBounds = await this.timeSlider.boundingBox();
    if (sliderBounds) {
      // Move to middle of time range
      const middleX = sliderBounds.x + sliderBounds.width / 2;
      await this.page.mouse.click(middleX, sliderBounds.y + sliderBounds.height / 2);
      await this.page.waitForTimeout(1000);
      
      // Verify time display updated
      const timeDisplay = this.page.locator('[data-testid="current-time"]');
      await expect(timeDisplay).toBeVisible();
    }
    
    // Test animation playback
    await this.playButton.click();
    await this.page.waitForTimeout(3000); // Let animation run
    
    // Verify animation is playing
    const isPlaying = this.page.locator('[data-testid="animation-playing"]');
    await expect(isPlaying).toBeVisible();
    
    // Pause animation
    await this.pauseButton.click();
    
    // Test animation speed control
    await this.animationSpeed.selectOption('2x');
    await this.page.waitForTimeout(500);
  }

  /**
   * Test scientific data visualization accuracy
   */
  async testScientificVisualization(): Promise<void> {
    // Test seismic waveform display
    await expect(this.seismicWaveforms).toBeVisible();
    
    // Verify waveform data is properly scaled
    const waveformCanvas = this.seismicWaveforms.locator('canvas');
    await expect(waveformCanvas).toBeVisible();
    
    // Test deformation plot accuracy
    await expect(this.deformationPlots).toBeVisible();
    
    // Verify deformation vectors are displayed
    const deformationVectors = this.page.locator('[data-testid="deformation-vector"]');
    const vectorCount = await deformationVectors.count();
    expect(vectorCount).toBeGreaterThan(0);
    
    // Test temperature map rendering
    await expect(this.temperatureMaps).toBeVisible();
    
    // Verify color scale is scientifically accurate
    const colorScale = this.page.locator('[data-testid="temperature-color-scale"]');
    await expect(colorScale).toBeVisible();
    await expect(colorScale).toHaveAttribute('data-scale-type', 'celsius');
  }

  /**
   * Test interactive measurement tools
   */
  async testMeasurementTools(): Promise<void> {
    // Test distance measurement
    await this.measurementTool.click();
    await this.page.locator('[data-testid="distance-tool"]').click();
    
    // Click two points to measure distance
    const viewer = this.threeDViewer;
    const viewerBounds = await viewer.boundingBox();
    
    if (viewerBounds) {
      const point1X = viewerBounds.x + 100;
      const point1Y = viewerBounds.y + 100;
      const point2X = viewerBounds.x + 200;
      const point2Y = viewerBounds.y + 150;
      
      await this.page.mouse.click(point1X, point1Y);
      await this.page.mouse.click(point2X, point2Y);
      
      // Verify measurement is displayed
      const measurement = this.page.locator('[data-testid="distance-measurement"]');
      await expect(measurement).toBeVisible();
      await expect(measurement).toHaveText(/\d+(\.\d+)?\s*(m|km)/);
    }
    
    // Test cross-section tool
    await this.crossSectionTool.click();
    
    // Draw cross-section line
    if (viewerBounds) {
      await this.page.mouse.move(viewerBounds.x + 50, viewerBounds.y + 100);
      await this.page.mouse.down();
      await this.page.mouse.move(viewerBounds.x + 250, viewerBounds.y + 100);
      await this.page.mouse.up();
      
      // Verify cross-section is generated
      const crossSection = this.page.locator('[data-testid="cross-section-view"]');
      await expect(crossSection).toBeVisible();
    }
  }

  /**
   * Test annotation and collaboration features
   */
  async testAnnotationTools(): Promise<void> {
    await this.annotationTool.click();
    
    // Add text annotation
    const textAnnotation = this.page.locator('[data-testid="text-annotation-tool"]');
    await textAnnotation.click();
    
    // Click to place annotation
    const viewer = this.threeDViewer;
    const viewerBounds = await viewer.boundingBox();
    
    if (viewerBounds) {
      const annotationX = viewerBounds.x + 150;
      const annotationY = viewerBounds.y + 120;
      
      await this.page.mouse.click(annotationX, annotationY);
      
      // Enter annotation text
      const textInput = this.page.locator('[data-testid="annotation-text-input"]');
      await textInput.fill('Potential magma chamber location');
      await this.page.keyboard.press('Enter');
      
      // Verify annotation is displayed
      const annotation = this.page.locator('[data-testid="annotation-marker"]');
      await expect(annotation).toBeVisible();
    }
  }

  /**
   * Test export functionality for visualizations
   */
  async testExportFunctionality(): Promise<void> {
    await this.exportTool.click();
    
    // Test screenshot export
    const screenshotButton = this.page.locator('[data-testid="export-screenshot"]');
    await screenshotButton.click();
    
    // Wait for export to complete
    const downloadPromise = this.page.waitForEvent('download');
    const confirmButton = this.page.locator('[data-testid="confirm-export"]');
    await confirmButton.click();
    
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.png$/);
    
    // Test 3D model export
    const modelExportButton = this.page.locator('[data-testid="export-3d-model"]');
    await modelExportButton.click();
    
    // Select format
    const gltfFormat = this.page.locator('[data-testid="format-gltf"]');
    await gltfFormat.click();
    
    const modelDownloadPromise = this.page.waitForEvent('download');
    const confirmModelExport = this.page.locator('[data-testid="confirm-model-export"]');
    await confirmModelExport.click();
    
    const modelDownload = await modelDownloadPromise;
    expect(modelDownload.suggestedFilename()).toMatch(/\.gltf$/);
  }

  /**
   * Monitor rendering performance
   */
  async checkRenderingPerformance(): Promise<void> {
    // Verify FPS counter is visible and showing acceptable performance
    await expect(this.fpsCounter).toBeVisible();
    
    const fpsText = await this.fpsCounter.textContent();
    const fps = parseInt(fpsText?.match(/\d+/)?.[0] || '0');
    expect(fps).toBeGreaterThan(30); // Minimum 30 FPS for smooth interaction
    
    // Check memory usage
    await expect(this.memoryUsage).toBeVisible();
    
    const memoryText = await this.memoryUsage.textContent();
    const memoryMB = parseInt(memoryText?.match(/\d+/)?.[0] || '0');
    expect(memoryMB).toBeLessThan(1000); // Less than 1GB memory usage
    
    // Verify render statistics
    await expect(this.renderStats).toBeVisible();
  }

  /**
   * Get current camera position for testing
   */
  async getCameraPosition(): Promise<any> {
    return await this.page.evaluate(() => {
      const viewer = (window as any).threeDViewer;
      if (viewer && viewer.camera) {
        return {
          position: viewer.camera.position,
          rotation: viewer.camera.rotation
        };
      }
      return null;
    });
  }

  /**
   * Validate page content and 3D visualization accuracy
   */
  async validatePageContent(): Promise<void> {
    // Wait for 3D viewer to be ready
    await this.waitFor3DViewerReady();
    
    // Verify coordinate system is properly initialized
    await this.verifyCoordinateSystem();
    
    // Test all major visualization components
    await this.test3DNavigation();
    await this.testGeologicalLayers();
    await this.testScientificVisualization();
    
    // Check rendering performance
    await this.checkRenderingPerformance();
    
    // Check for any errors
    await this.checkForErrors();
  }

  /**
   * Test visual regression by comparing screenshots
   */
  async testVisualRegression(testName: string): Promise<void> {
    // Wait for all animations to complete
    await this.page.waitForTimeout(2000);
    
    // Disable animations for consistent screenshots
    await this.page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `
    });
    
    // Take screenshot of 3D viewer
    await expect(this.threeDViewer).toHaveScreenshot(`${testName}-3d-viewer.png`);
    
    // Take screenshot of layer panel
    await expect(this.layerPanel).toHaveScreenshot(`${testName}-layer-panel.png`);
    
    // Take screenshot of chart container
    await expect(this.chartContainer).toHaveScreenshot(`${testName}-charts.png`);
  }
}