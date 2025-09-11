import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global setup for Prometheus AI Platform tests...');
  
  // Launch browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for the application to be ready
    console.log('⏳ Waiting for application to be ready...');
    await page.goto(config.projects[0].use.baseURL || 'http://localhost:5173');
    
    // Wait for the main app to load
    await page.waitForSelector('[data-testid="app-loaded"]', { timeout: 30000 });
    
    // Setup test data if needed
    console.log('📊 Setting up test data...');
    
    // Create mock geological data
    await page.evaluate(() => {
      // Store test data in localStorage for tests
      const testData = {
        geological: [
          {
            id: 'test-geo-1',
            timestamp: new Date().toISOString(),
            data_type: 'volcanic',
            latitude: 37.7749,
            longitude: -122.4194,
            elevation: 100,
            quality_score: 0.95
          }
        ],
        seismic: [
          {
            eventId: 'test-seismic-1',
            timestamp: new Date().toISOString(),
            location: { x: -122.4194, y: 37.7749, z: 10 },
            magnitude: 4.5,
            depth: 15,
            eventType: 'earthquake',
            confidence: 0.9
          }
        ],
        volcanic: [
          {
            volcanoId: 'test-volcano-1',
            timestamp: new Date().toISOString(),
            alertLevel: 'yellow',
            magmaChamberPressure: 75,
            deformationRate: 2.5,
            gasEmissions: {
              co2: 400,
              so2: 50,
              h2s: 10,
              h2o: 1000,
              temperature: 85,
              flux: 150
            },
            thermalAnomalies: [],
            eruptionProbability: 0.15
          }
        ]
      };
      
      localStorage.setItem('prometheus-test-data', JSON.stringify(testData));
    });
    
    // Setup user preferences for testing
    await page.evaluate(() => {
      const testPreferences = {
        theme: 'light',
        language: 'en',
        timezone: 'UTC',
        dateFormat: 'MM/dd/yyyy',
        numberFormat: 'en-US',
        autoSave: true,
        notifications: {
          enabled: true,
          sound: false, // Disable sound for tests
          desktop: false,
          email: false,
          alertLevels: ['warning', 'error', 'critical']
        },
        accessibility: {
          highContrast: false,
          largeText: false,
          reducedMotion: true, // Enable for stable tests
          screenReader: false,
          keyboardNavigation: true
        },
        performance: {
          enableVirtualization: true,
          maxDataPoints: 1000, // Reduced for tests
          renderQuality: 'medium',
          enableWebGL: false, // Disable for stability
          enableWorkers: false,
          cacheSize: 50
        }
      };
      
      localStorage.setItem('prometheus-user-preferences', JSON.stringify(testPreferences));
    });
    
    console.log('✅ Global setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;