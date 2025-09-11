import { chromium, FullConfig } from '@playwright/test';

/**
 * Global setup for Playwright tests
 * Initializes test environment and prepares geological test data
 */
async function globalSetup(config: FullConfig) {
  console.log('🌋 Starting Prometheus AI Platform test setup...');
  
  // Launch browser for setup operations
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Verify test server is running
    console.log('📡 Checking test server availability...');
    await page.goto(config.projects[0].use.baseURL || 'http://localhost:5173', { timeout: 30000 });
    
    // Initialize test database with geological data
    console.log('🗄️ Initializing test database...');
    await page.evaluate(() => {
      // Clear any existing test data
      localStorage.clear();
      sessionStorage.clear();
      
      // Set up test environment flags
      localStorage.setItem('test-mode', 'true');
      localStorage.setItem('mock-api', 'true');
      localStorage.setItem('geological-validation', 'enabled');
    });
    
    // Pre-authenticate test users
    console.log('👤 Setting up test user authentication...');
    const testUsers = [
      { username: 'researcher@prometheus.ai', role: 'researcher' },
      { username: 'operator@prometheus.ai', role: 'operator' },
      { username: 'industry@prometheus.ai', role: 'industry' }
    ];
    
    for (const user of testUsers) {
      await page.evaluate((userData) => {
        const token = `test-token-${userData.role}-${Date.now()}`;
        localStorage.setItem(`auth-token-${userData.username}`, token);
        localStorage.setItem(`user-role-${userData.username}`, userData.role);
      }, user);
    }
    
    // Initialize geological coordinate systems
    console.log('🌍 Setting up coordinate systems...');
    await page.evaluate(() => {
      const coordSystems = {
        'WGS84': { epsg: 4326, name: 'World Geodetic System 1984' },
        'UTM-11N': { epsg: 32611, name: 'UTM Zone 11N' },
        'UTM-12N': { epsg: 32612, name: 'UTM Zone 12N' }
      };
      localStorage.setItem('coordinate-systems', JSON.stringify(coordSystems));
      localStorage.setItem('default-projection', 'WGS84');
    });
    
    // Set up geological validation parameters
    console.log('🔬 Configuring geological validation...');
    await page.evaluate(() => {
      const validationConfig = {
        magnitudeRange: { min: 0, max: 10 },
        depthRange: { min: 0, max: 700 },
        temperatureRange: { min: -50, max: 2000 },
        deformationRange: { min: -1000, max: 1000 },
        gasConcentrationLimits: {
          so2: { min: 0, max: 50000 },
          co2: { min: 0, max: 100000 },
          h2s: { min: 0, max: 10000 }
        }
      };
      localStorage.setItem('geological-validation-config', JSON.stringify(validationConfig));
    });
    
    // Create comprehensive test data
    console.log('📊 Setting up comprehensive test data...');
    await page.evaluate(() => {
      const testData = {
        geological: [
          {
            id: 'yellowstone-caldera',
            timestamp: new Date().toISOString(),
            data_type: 'volcanic',
            latitude: 44.4280,
            longitude: -110.5885,
            elevation: 2805,
            quality_score: 0.95,
            volcanoType: 'supervolcano'
          },
          {
            id: 'kilauea-volcano',
            timestamp: new Date().toISOString(),
            data_type: 'volcanic',
            latitude: 19.4326,
            longitude: -155.2920,
            elevation: 1247,
            quality_score: 0.92,
            volcanoType: 'shield'
          }
        ],
        seismic: [
          {
            eventId: 'evt-001',
            timestamp: new Date().toISOString(),
            location: { x: -110.5885, y: 44.4280, z: 8.5 },
            magnitude: 4.2,
            depth: 8.5,
            eventType: 'volcanic',
            confidence: 0.9,
            waveformData: new Array(1000).fill(0).map((_, i) => Math.sin(i * 0.1) * Math.random())
          },
          {
            eventId: 'evt-002',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            location: { x: -155.2920, y: 19.4326, z: 6.2 },
            magnitude: 5.1,
            depth: 6.2,
            eventType: 'volcanic',
            confidence: 0.85
          }
        ],
        volcanic: [
          {
            volcanoId: 'yellowstone',
            name: 'Yellowstone Caldera',
            timestamp: new Date().toISOString(),
            alertLevel: 'yellow',
            magmaChamberPressure: 75,
            deformationRate: 2.3,
            gasEmissions: {
              co2: 8900,
              so2: 450,
              h2s: 23,
              h2o: 12000,
              temperature: 85,
              flux: 150
            },
            thermalAnomalies: [
              { id: 'ta-001', temperature: 95, area: 1200, location: { x: -110.58, y: 44.43 } }
            ],
            eruptionProbability: 15.2,
            calderaType: 'resurgent',
            supereruptionRisk: 'low'
          },
          {
            volcanoId: 'kilauea',
            name: 'Kilauea',
            timestamp: new Date().toISOString(),
            alertLevel: 'orange',
            magmaChamberPressure: 120,
            deformationRate: 15.7,
            gasEmissions: {
              co2: 15600,
              so2: 2300,
              h2s: 89,
              h2o: 25000,
              temperature: 1100,
              flux: 850
            },
            thermalAnomalies: [
              { id: 'ta-002', temperature: 1150, area: 5600, location: { x: -155.29, y: 19.43 } }
            ],
            eruptionProbability: 78.5,
            calderaType: 'pit',
            supereruptionRisk: 'none'
          }
        ],
        geologicalModels: [
          {
            id: 'yellowstone-3d-model',
            name: 'Yellowstone Magma Chamber 3D',
            type: '3d-geological',
            accuracy: 92.5,
            lastUpdated: new Date().toISOString(),
            parameters: {
              depth: 8000,
              volume: 15000,
              temperature: 850,
              composition: 'rhyolitic'
            }
          }
        ]
      };
      
      localStorage.setItem('prometheus-test-data', JSON.stringify(testData));
    });
    
    // Initialize performance monitoring
    console.log('📊 Setting up performance monitoring...');
    await page.evaluate(() => {
      const performanceConfig = {
        enableMetrics: true,
        trackMemoryUsage: true,
        trackRenderingPerformance: true,
        maxLoadTime: 5000,
        maxMemoryUsage: 1000 * 1024 * 1024, // 1GB
        targetFPS: 30
      };
      localStorage.setItem('performance-config', JSON.stringify(performanceConfig));
    });
    
    // Set up WebGL context for 3D visualization tests
    console.log('🎮 Initializing WebGL context...');
    const webglSupported = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return !!gl;
    });
    
    if (!webglSupported) {
      console.warn('⚠️ WebGL not supported - 3D visualization tests may fail');
    } else {
      console.log('✅ WebGL support confirmed');
    }
    
    // Initialize accessibility testing configuration
    console.log('♿ Setting up accessibility testing...');
    await page.evaluate(() => {
      const a11yConfig = {
        enableAxeCore: true,
        wcagLevel: 'AA',
        checkColorContrast: true,
        checkKeyboardNavigation: true,
        checkScreenReaderSupport: true,
        minTouchTargetSize: 44
      };
      localStorage.setItem('accessibility-config', JSON.stringify(a11yConfig));
    });
    
    // Set up visual regression testing baseline
    console.log('📸 Preparing visual regression baselines...');
    await page.evaluate(() => {
      const visualConfig = {
        threshold: 0.2,
        maskDynamicContent: true,
        disableAnimations: true,
        standardizeTimestamps: true,
        viewports: [
          { width: 1920, height: 1080, name: 'desktop' },
          { width: 1366, height: 768, name: 'laptop' },
          { width: 768, height: 1024, name: 'tablet' },
          { width: 375, height: 667, name: 'mobile' }
        ]
      };
      localStorage.setItem('visual-regression-config', JSON.stringify(visualConfig));
    });
    
    // Setup user preferences optimized for testing
    console.log('⚙️ Configuring user preferences...');
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
          enableWebGL: true,
          enableWorkers: false,
          cacheSize: 50
        },
        geological: {
          defaultProjection: 'WGS84',
          enableValidation: true,
          showUncertainty: true,
          realTimeUpdates: true
        }
      };
      
      localStorage.setItem('prometheus-user-preferences', JSON.stringify(testPreferences));
    });
    
    // Verify all systems are ready
    console.log('🔍 Verifying test environment...');
    const systemCheck = await page.evaluate(() => {
      const checks = {
        testMode: !!localStorage.getItem('test-mode'),
        testData: !!localStorage.getItem('prometheus-test-data'),
        coordinates: !!localStorage.getItem('coordinate-systems'),
        validation: !!localStorage.getItem('geological-validation-config'),
        performance: !!localStorage.getItem('performance-config'),
        accessibility: !!localStorage.getItem('accessibility-config'),
        preferences: !!localStorage.getItem('prometheus-user-preferences')
      };
      
      return {
        allPassed: Object.values(checks).every(check => check),
        details: checks
      };
    });
    
    if (!systemCheck.allPassed) {
      throw new Error(`System check failed: ${JSON.stringify(systemCheck.details)}`);
    }
    
    console.log('✅ All systems ready for testing');
    
    // Create test results directory structure
    const fs = require('fs');
    const path = require('path');
    
    const testResultsDir = path.join(process.cwd(), 'test-results');
    const screenshotsDir = path.join(testResultsDir, 'screenshots');
    const reportsDir = path.join(testResultsDir, 'reports');
    const tracesDir = path.join(testResultsDir, 'traces');
    const videosDir = path.join(testResultsDir, 'videos');
    
    const directories = [testResultsDir, screenshotsDir, reportsDir, tracesDir, videosDir];
    
    for (const dir of directories) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
    
    console.log('📁 Test directories created');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await context.close();
    await browser.close();
  }
  
  console.log('🚀 Prometheus AI Platform test environment ready!');
}

export default globalSetup;