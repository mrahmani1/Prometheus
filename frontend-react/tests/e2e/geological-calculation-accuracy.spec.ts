import { test, expect } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';
import { VisualizationPage } from '../page-objects/VisualizationPage';
import { VolcanicMonitoringPage } from '../page-objects/VolcanicMonitoringPage';
import { MockApiService, MockGeologicalService } from '../fixtures/mock-services';
import { mockGeologicalModels, mockVolcanoes } from '../fixtures/test-data';

test.describe('Geological Calculation Accuracy Tests', () => {
  let dashboardPage: DashboardPage;
  let visualizationPage: VisualizationPage;
  let monitoringPage: VolcanicMonitoringPage;
  let mockApi: MockApiService;
  let mockGeological: MockGeologicalService;

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page);
    visualizationPage = new VisualizationPage(page);
    monitoringPage = new VolcanicMonitoringPage(page);
    mockApi = new MockApiService(page);
    mockGeological = new MockGeologicalService(page);

    await mockApi.setupSuccessfulMocks();
    await mockGeological.setupGeologicalMocks();

    // Ensure user is authenticated
    await page.goto('/dashboard');
    await dashboardPage.ensureAuthenticated();
  });

  test('Eruption probability calculation accuracy', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.selectVolcano('yellowstone');

    // Verify eruption probability calculation
    const eruptionProb = page.locator('[data-testid="eruption-probability"]');
    await expect(eruptionProb).toBeVisible();

    const probabilityText = await eruptionProb.textContent();
    const probability = parseFloat(probabilityText?.match(/\d+(\.\d+)?/)?.[0] || '0');

    // Verify probability is within valid range (0-100%)
    expect(probability).toBeGreaterThanOrEqual(0);
    expect(probability).toBeLessThanOrEqual(100);

    // Verify confidence interval is displayed
    const confidenceInterval = page.locator('[data-testid="confidence-interval"]');
    await expect(confidenceInterval).toBeVisible();

    const confidenceText = await confidenceInterval.textContent();
    expect(confidenceText).toMatch(/\d+(\.\d+)?%\s*-\s*\d+(\.\d+)?%/);

    // Test probability calculation factors
    const factorsPanel = page.locator('[data-testid="probability-factors"]');
    await expect(factorsPanel).toBeVisible();

    const factors = await factorsPanel.locator('[data-testid="factor-item"]').count();
    expect(factors).toBeGreaterThan(0);

    // Verify each factor has weight and value
    for (let i = 0; i < factors; i++) {
      const factor = factorsPanel.locator('[data-testid="factor-item"]').nth(i);
      const weight = factor.locator('[data-testid="factor-weight"]');
      const value = factor.locator('[data-testid="factor-value"]');

      await expect(weight).toBeVisible();
      await expect(value).toBeVisible();

      const weightText = await weight.textContent();
      const valueText = await value.textContent();

      const weightNum = parseFloat(weightText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      const valueNum = parseFloat(valueText?.match(/\d+(\.\d+)?/)?.[0] || '0');

      expect(weightNum).toBeGreaterThan(0);
      expect(weightNum).toBeLessThanOrEqual(1);
      expect(valueNum).toBeGreaterThanOrEqual(0);
      expect(valueNum).toBeLessThanOrEqual(1);
    }

    // Verify time horizon is specified
    const timeHorizon = page.locator('[data-testid="forecast-horizon"]');
    await expect(timeHorizon).toBeVisible();
    await expect(timeHorizon).toContainText(/\d+\s*(day|week|month|year)s?/);
  });

  test('Seismic magnitude and depth calculations', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.verifySeismicData();

    // Verify seismic event calculations
    const seismicEvents = page.locator('[data-testid="recent-seismic-event"]');
    const eventCount = await seismicEvents.count();
    expect(eventCount).toBeGreaterThan(0);

    for (let i = 0; i < Math.min(eventCount, 5); i++) {
      const event = seismicEvents.nth(i);
      
      // Verify magnitude calculation
      const magnitude = event.locator('[data-testid="event-magnitude"]');
      await expect(magnitude).toBeVisible();
      
      const magText = await magnitude.textContent();
      const magValue = parseFloat(magText?.replace('M', '') || '0');
      
      // Magnitude should be in realistic range
      expect(magValue).toBeGreaterThan(0);
      expect(magValue).toBeLessThan(10); // Richter scale maximum

      // Verify depth calculation
      const depth = event.locator('[data-testid="event-depth"]');
      await expect(depth).toBeVisible();
      
      const depthText = await depth.textContent();
      const depthValue = parseFloat(depthText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      // Depth should be positive and realistic
      expect(depthValue).toBeGreaterThan(0);
      expect(depthValue).toBeLessThan(700); // Maximum earthquake depth

      // Verify location accuracy
      const location = event.locator('[data-testid="event-location"]');
      await expect(location).toBeVisible();
      
      const coordinates = event.locator('[data-testid="event-coordinates"]');
      if (await coordinates.isVisible()) {
        const coordText = await coordinates.textContent();
        const latMatch = coordText?.match(/lat:\s*(-?\d+(\.\d+)?)/);
        const lonMatch = coordText?.match(/lon:\s*(-?\d+(\.\d+)?)/);
        
        if (latMatch && lonMatch) {
          const lat = parseFloat(latMatch[1]);
          const lon = parseFloat(lonMatch[1]);
          
          expect(lat).toBeGreaterThanOrEqual(-90);
          expect(lat).toBeLessThanOrEqual(90);
          expect(lon).toBeGreaterThanOrEqual(-180);
          expect(lon).toBeLessThanOrEqual(180);
        }
      }
    }

    // Test seismic energy calculation
    const energyCalculation = page.locator('[data-testid="seismic-energy"]');
    if (await energyCalculation.isVisible()) {
      const energyText = await energyCalculation.textContent();
      const energyValue = parseFloat(energyText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      // Energy should be positive
      expect(energyValue).toBeGreaterThan(0);
    }
  });

  test('Deformation rate calculations and InSAR processing', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.selectVolcano('yellowstone');
    await monitoringPage.verifyDeformationData();

    // Verify deformation rate calculations
    const deformationValues = page.locator('[data-testid="deformation-value"]');
    const valueCount = await deformationValues.count();
    expect(valueCount).toBeGreaterThan(0);

    for (let i = 0; i < Math.min(valueCount, 3); i++) {
      const deformationValue = deformationValues.nth(i);
      const valueText = await deformationValue.textContent();
      
      // Verify units are correct (mm/year or cm/year)
      expect(valueText).toMatch(/\d+(\.\d+)?\s*(mm|cm)\/year/);
      
      const numericValue = parseFloat(valueText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      // Deformation rates should be realistic
      if (valueText?.includes('mm/year')) {
        expect(numericValue).toBeGreaterThanOrEqual(0);
        expect(numericValue).toBeLessThan(1000); // Max 1000 mm/year
      } else if (valueText?.includes('cm/year')) {
        expect(numericValue).toBeGreaterThanOrEqual(0);
        expect(numericValue).toBeLessThan(100); // Max 100 cm/year
      }
    }

    // Verify 3D displacement vectors
    const displacementVectors = page.locator('[data-testid="displacement-vector"]');
    const vectorCount = await displacementVectors.count();
    
    if (vectorCount > 0) {
      for (let i = 0; i < Math.min(vectorCount, 3); i++) {
        const vector = displacementVectors.nth(i);
        
        // Verify X, Y, Z components
        const xComponent = vector.locator('[data-testid="displacement-x"]');
        const yComponent = vector.locator('[data-testid="displacement-y"]');
        const zComponent = vector.locator('[data-testid="displacement-z"]');
        
        if (await xComponent.isVisible()) {
          const xText = await xComponent.textContent();
          const xValue = parseFloat(xText?.match(/-?\d+(\.\d+)?/)?.[0] || '0');
          expect(Math.abs(xValue)).toBeLessThan(1000); // Reasonable displacement limit
        }
        
        if (await yComponent.isVisible()) {
          const yText = await yComponent.textContent();
          const yValue = parseFloat(yText?.match(/-?\d+(\.\d+)?/)?.[0] || '0');
          expect(Math.abs(yValue)).toBeLessThan(1000);
        }
        
        if (await zComponent.isVisible()) {
          const zText = await zComponent.textContent();
          const zValue = parseFloat(zText?.match(/-?\d+(\.\d+)?/)?.[0] || '0');
          expect(Math.abs(zValue)).toBeLessThan(1000);
        }
      }
    }

    // Test InSAR coherence values
    const coherenceValues = page.locator('[data-testid="insar-coherence"]');
    if (await coherenceValues.first().isVisible()) {
      const coherenceCount = await coherenceValues.count();
      
      for (let i = 0; i < Math.min(coherenceCount, 3); i++) {
        const coherence = coherenceValues.nth(i);
        const coherenceText = await coherence.textContent();
        const coherenceValue = parseFloat(coherenceText?.match(/\d+(\.\d+)?/)?.[0] || '0');
        
        // Coherence should be between 0 and 1
        expect(coherenceValue).toBeGreaterThanOrEqual(0);
        expect(coherenceValue).toBeLessThanOrEqual(1);
      }
    }
  });

  test('Gas concentration calculations and chemical analysis', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.selectVolcano('kilauea');
    await monitoringPage.verifyGasEmissions();

    // Verify SO2 concentration calculations
    const so2Data = page.locator('[data-testid="so2-concentration"]');
    await expect(so2Data).toBeVisible();
    
    const so2Text = await so2Data.textContent();
    const so2Value = parseFloat(so2Text?.match(/\d+(\.\d+)?/)?.[0] || '0');
    
    // SO2 concentrations should be realistic for volcanic emissions
    expect(so2Value).toBeGreaterThan(0);
    expect(so2Value).toBeLessThan(50000); // Max reasonable SO2 ppm
    
    // Verify units are correct
    expect(so2Text).toMatch(/\d+(\.\d+)?\s*(ppm|ppb|mg\/m³)/);

    // Verify CO2 concentration calculations
    const co2Data = page.locator('[data-testid="co2-concentration"]');
    await expect(co2Data).toBeVisible();
    
    const co2Text = await co2Data.textContent();
    const co2Value = parseFloat(co2Text?.match(/\d+(\.\d+)?/)?.[0] || '0');
    
    expect(co2Value).toBeGreaterThan(0);
    expect(co2Value).toBeLessThan(100000); // Max reasonable CO2 ppm

    // Verify H2S concentration calculations
    const h2sData = page.locator('[data-testid="h2s-concentration"]');
    await expect(h2sData).toBeVisible();
    
    const h2sText = await h2sData.textContent();
    const h2sValue = parseFloat(h2sText?.match(/\d+(\.\d+)?/)?.[0] || '0');
    
    expect(h2sValue).toBeGreaterThan(0);
    expect(h2sValue).toBeLessThan(10000); // Max reasonable H2S ppm

    // Test gas ratio calculations
    const gasRatios = page.locator('[data-testid="gas-ratio"]');
    if (await gasRatios.first().isVisible()) {
      const ratioCount = await gasRatios.count();
      
      for (let i = 0; i < ratioCount; i++) {
        const ratio = gasRatios.nth(i);
        const ratioText = await ratio.textContent();
        const ratioValue = parseFloat(ratioText?.match(/\d+(\.\d+)?/)?.[0] || '0');
        
        // Gas ratios should be positive
        expect(ratioValue).toBeGreaterThan(0);
      }
    }

    // Verify flux calculations
    const fluxCalculations = page.locator('[data-testid="gas-flux"]');
    if (await fluxCalculations.first().isVisible()) {
      const fluxText = await fluxCalculations.first().textContent();
      const fluxValue = parseFloat(fluxText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      // Flux should be positive and in reasonable range
      expect(fluxValue).toBeGreaterThan(0);
      expect(fluxValue).toBeLessThan(100000); // Max reasonable flux tons/day
      
      // Verify flux units
      expect(fluxText).toMatch(/\d+(\.\d+)?\s*(tons?|kg)\/day/);
    }
  });

  test('3D geological model accuracy and validation', async ({ page }) => {
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();

    // Verify 3D model loading and accuracy
    const geologicalModel = page.locator('[data-testid="3d-geological-model"]');
    await expect(geologicalModel).toBeVisible();

    // Test model validation metrics
    const validationPanel = page.locator('[data-testid="model-validation"]');
    if (await validationPanel.isVisible()) {
      const accuracyMetric = validationPanel.locator('[data-testid="model-accuracy"]');
      await expect(accuracyMetric).toBeVisible();
      
      const accuracyText = await accuracyMetric.textContent();
      const accuracy = parseFloat(accuracyText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      // Model accuracy should be reasonable
      expect(accuracy).toBeGreaterThan(50); // At least 50% accuracy
      expect(accuracy).toBeLessThanOrEqual(100);
    }

    // Test geological layer accuracy
    await visualizationPage.testGeologicalLayers();
    
    const geologicalLayers = page.locator('[data-testid="geological-layer"]');
    const layerCount = await geologicalLayers.count();
    expect(layerCount).toBeGreaterThan(0);

    // Verify each layer has proper geological properties
    for (let i = 0; i < Math.min(layerCount, 3); i++) {
      const layer = geologicalLayers.nth(i);
      
      // Check layer properties
      const layerName = layer.locator('[data-testid="layer-name"]');
      const layerAge = layer.locator('[data-testid="layer-age"]');
      const layerLithology = layer.locator('[data-testid="layer-lithology"]');
      
      await expect(layerName).toBeVisible();
      
      if (await layerAge.isVisible()) {
        const ageText = await layerAge.textContent();
        expect(ageText).toMatch(/\d+(\.\d+)?\s*(Ma|ka|years?)/); // Geological age format
      }
      
      if (await layerLithology.isVisible()) {
        const lithologyText = await layerLithology.textContent();
        expect(lithologyText).toBeTruthy();
        expect(lithologyText?.length).toBeGreaterThan(0);
      }
    }

    // Test cross-section accuracy
    await visualizationPage.testMeasurementTools();
    
    const crossSection = page.locator('[data-testid="cross-section-view"]');
    if (await crossSection.isVisible()) {
      // Verify cross-section shows geological structure
      const geologicalUnits = crossSection.locator('[data-testid="geological-unit"]');
      const unitCount = await geologicalUnits.count();
      expect(unitCount).toBeGreaterThan(0);
    }
  });

  test('Stress field and structural geology calculations', async ({ page }) => {
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();

    // Navigate to stress field visualization
    const stressFieldLayer = page.locator('[data-testid="stress-field-layer"]');
    if (await stressFieldLayer.isVisible()) {
      await stressFieldLayer.click();
    }

    // Verify stress field calculations
    const stressField = page.locator('[data-testid="stress-field-map"]');
    if (await stressField.isVisible()) {
      // Test principal stress calculations
      const principalStresses = page.locator('[data-testid="principal-stress"]');
      const stressCount = await principalStresses.count();
      
      if (stressCount > 0) {
        for (let i = 0; i < Math.min(stressCount, 3); i++) {
          const stress = principalStresses.nth(i);
          const stressText = await stress.textContent();
          const stressValue = parseFloat(stressText?.match(/\d+(\.\d+)?/)?.[0] || '0');
          
          // Stress values should be positive and realistic
          expect(stressValue).toBeGreaterThan(0);
          expect(stressValue).toBeLessThan(1000); // Max reasonable stress in MPa
        }
      }

      // Verify stress orientation
      const stressOrientation = page.locator('[data-testid="stress-orientation"]');
      if (await stressOrientation.isVisible()) {
        const orientationText = await stressOrientation.textContent();
        expect(orientationText).toMatch(/(N|S|E|W|NE|NW|SE|SW)/); // Cardinal directions
      }

      // Test fault plane solutions
      const faultPlanes = page.locator('[data-testid="fault-plane"]');
      const faultCount = await faultPlanes.count();
      
      if (faultCount > 0) {
        for (let i = 0; i < Math.min(faultCount, 3); i++) {
          const fault = faultPlanes.nth(i);
          
          // Verify strike, dip, and rake values
          const strike = fault.locator('[data-testid="fault-strike"]');
          const dip = fault.locator('[data-testid="fault-dip"]');
          const rake = fault.locator('[data-testid="fault-rake"]');
          
          if (await strike.isVisible()) {
            const strikeText = await strike.textContent();
            const strikeValue = parseFloat(strikeText?.match(/\d+(\.\d+)?/)?.[0] || '0');
            expect(strikeValue).toBeGreaterThanOrEqual(0);
            expect(strikeValue).toBeLessThan(360);
          }
          
          if (await dip.isVisible()) {
            const dipText = await dip.textContent();
            const dipValue = parseFloat(dipText?.match(/\d+(\.\d+)?/)?.[0] || '0');
            expect(dipValue).toBeGreaterThanOrEqual(0);
            expect(dipValue).toBeLessThanOrEqual(90);
          }
          
          if (await rake.isVisible()) {
            const rakeText = await rake.textContent();
            const rakeValue = parseFloat(rakeText?.match(/-?\d+(\.\d+)?/)?.[0] || '0');
            expect(rakeValue).toBeGreaterThanOrEqual(-180);
            expect(rakeValue).toBeLessThanOrEqual(180);
          }
        }
      }
    }
  });

  test('Thermal anomaly detection and temperature calculations', async ({ page }) => {
    await monitoringPage.goto();
    await monitoringPage.selectVolcano('kilauea');
    await monitoringPage.verifyThermalData();

    // Verify thermal anomaly detection
    const thermalAnomalies = page.locator('[data-testid="thermal-anomaly"]');
    const anomalyCount = await thermalAnomalies.count();
    
    if (anomalyCount > 0) {
      for (let i = 0; i < Math.min(anomalyCount, 3); i++) {
        const anomaly = thermalAnomalies.nth(i);
        
        // Verify temperature calculations
        const temperature = anomaly.locator('[data-testid="anomaly-temperature"]');
        await expect(temperature).toBeVisible();
        
        const tempText = await temperature.textContent();
        const tempValue = parseFloat(tempText?.match(/\d+(\.\d+)?/)?.[0] || '0');
        
        // Temperature should be realistic for volcanic thermal anomalies
        if (tempText?.includes('°C')) {
          expect(tempValue).toBeGreaterThan(-50); // Minimum reasonable temperature
          expect(tempValue).toBeLessThan(2000); // Maximum volcanic temperature
        } else if (tempText?.includes('°F')) {
          expect(tempValue).toBeGreaterThan(-58); // -50°C in Fahrenheit
          expect(tempValue).toBeLessThan(3632); // 2000°C in Fahrenheit
        }
        
        // Verify anomaly area calculation
        const anomalyArea = anomaly.locator('[data-testid="anomaly-area"]');
        if (await anomalyArea.isVisible()) {
          const areaText = await anomalyArea.textContent();
          const areaValue = parseFloat(areaText?.match(/\d+(\.\d+)?/)?.[0] || '0');
          
          expect(areaValue).toBeGreaterThan(0);
          expect(areaText).toMatch(/\d+(\.\d+)?\s*(m²|km²|hectares?)/);
        }
        
        // Verify thermal flux calculation
        const thermalFlux = anomaly.locator('[data-testid="thermal-flux"]');
        if (await thermalFlux.isVisible()) {
          const fluxText = await thermalFlux.textContent();
          const fluxValue = parseFloat(fluxText?.match(/\d+(\.\d+)?/)?.[0] || '0');
          
          expect(fluxValue).toBeGreaterThan(0);
          expect(fluxText).toMatch(/\d+(\.\d+)?\s*(W\/m²|MW)/);
        }
      }
    }

    // Test background temperature subtraction
    const backgroundTemp = page.locator('[data-testid="background-temperature"]');
    if (await backgroundTemp.isVisible()) {
      const bgTempText = await backgroundTemp.textContent();
      const bgTempValue = parseFloat(bgTempText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      // Background temperature should be reasonable
      expect(bgTempValue).toBeGreaterThan(-50);
      expect(bgTempValue).toBeLessThan(50); // Typical ambient temperatures
    }
  });

  test('Coordinate system transformations and spatial accuracy', async ({ page }) => {
    await visualizationPage.goto();
    await visualizationPage.waitFor3DViewerReady();

    // Verify coordinate system is properly initialized
    await visualizationPage.verifyCoordinateSystem();

    const coordSystem = page.locator('[data-testid="coordinate-system-indicator"]');
    await expect(coordSystem).toBeVisible();
    
    const coordText = await coordSystem.getAttribute('data-projection');
    expect(coordText).toMatch(/EPSG:\d+|WGS84|UTM/);

    // Test coordinate transformations
    const coordinateDisplay = page.locator('[data-testid="coordinate-display"]');
    if (await coordinateDisplay.isVisible()) {
      const coordDisplayText = await coordinateDisplay.textContent();
      
      // Should show valid coordinates
      const latMatch = coordDisplayText?.match(/lat:\s*(-?\d+(\.\d+)?)/);
      const lonMatch = coordDisplayText?.match(/lon:\s*(-?\d+(\.\d+)?)/);
      
      if (latMatch && lonMatch) {
        const lat = parseFloat(latMatch[1]);
        const lon = parseFloat(lonMatch[1]);
        
        expect(lat).toBeGreaterThanOrEqual(-90);
        expect(lat).toBeLessThanOrEqual(90);
        expect(lon).toBeGreaterThanOrEqual(-180);
        expect(lon).toBeLessThanOrEqual(180);
      }
    }

    // Test elevation accuracy
    const elevationDisplay = page.locator('[data-testid="elevation-display"]');
    if (await elevationDisplay.isVisible()) {
      const elevationText = await elevationDisplay.textContent();
      const elevation = parseFloat(elevationText?.match(/-?\d+(\.\d+)?/)?.[0] || '0');
      
      // Elevation should be realistic (below Everest, above Dead Sea)
      expect(elevation).toBeGreaterThan(-500);
      expect(elevation).toBeLessThan(9000);
    }

    // Test distance measurements accuracy
    await visualizationPage.testMeasurementTools();
    
    const distanceMeasurement = page.locator('[data-testid="distance-measurement"]');
    if (await distanceMeasurement.isVisible()) {
      const distanceText = await distanceMeasurement.textContent();
      const distance = parseFloat(distanceText?.match(/\d+(\.\d+)?/)?.[0] || '0');
      
      expect(distance).toBeGreaterThan(0);
      expect(distanceText).toMatch(/\d+(\.\d+)?\s*(m|km)/);
    }
  });
});