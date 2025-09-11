/**
 * Mock services for testing API interactions and WebSocket connections
 */

import { Page } from '@playwright/test';
import { mockApiResponses, mockWebSocketMessages, mockVolcanoes, mockSeismicEvents } from './test-data';

/**
 * Mock API service for intercepting HTTP requests
 */
export class MockApiService {
  private page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * Set up API mocks for successful responses
   */
  async setupSuccessfulMocks(): Promise<void> {
    // Mock volcano data endpoint
    await this.page.route('**/api/v1/volcanoes', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockApiResponses.volcanoes)
      });
    });

    // Mock seismic events endpoint
    await this.page.route('**/api/v1/seismic-events', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockApiResponses.seismicEvents)
      });
    });

    // Mock geological models endpoint
    await this.page.route('**/api/v1/geological-models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockApiResponses.geologicalModels)
      });
    });

    // Mock resource prospects endpoint
    await this.page.route('**/api/v1/resource-prospects', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockApiResponses.resourceProspects)
      });
    });

    // Mock authentication endpoint
    await this.page.route('**/api/v1/auth/login', async (route) => {
      const request = route.request();
      const postData = request.postDataJSON();
      
      if (postData.username && postData.password) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            token: 'mock-jwt-token',
            user: {
              id: '1',
              username: postData.username,
              role: 'researcher'
            }
          })
        });
      } else {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify(mockApiResponses.unauthorized)
        });
      }
    });

    // Mock dashboard data endpoint
    await this.page.route('**/api/v1/dashboard/data', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          volcanoes: mockVolcanoes,
          recentEvents: mockSeismicEvents.slice(0, 5),
          alerts: [
            {
              id: 'alert_001',
              volcanoId: 'yellowstone',
              level: 'yellow',
              message: 'Increased seismic activity detected',
              timestamp: new Date().toISOString()
            }
          ],
          statistics: {
            totalVolcanoes: mockVolcanoes.length,
            activeAlerts: 3,
            criticalAlerts: 1,
            averageRisk: 25.7
          }
        })
      });
    });

    // Mock 3D model data endpoint
    await this.page.route('**/api/v1/visualization/3d-models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: [
            {
              id: 'yellowstone-3d',
              name: 'Yellowstone Caldera 3D Model',
              url: '/models/yellowstone.gltf',
              type: 'geological'
            }
          ]
        })
      });
    });
  }

  /**
   * Set up API mocks for error responses
   */
  async setupErrorMocks(): Promise<void> {
    await this.page.route('**/api/v1/**', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify(mockApiResponses.error)
      });
    });
  }

  /**
   * Set up API mocks for slow responses (testing loading states)
   */
  async setupSlowMocks(delay: number = 3000): Promise<void> {
    await this.page.route('**/api/v1/**', async (route) => {
      await new Promise(resolve => setTimeout(resolve, delay));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockApiResponses.volcanoes)
      });
    });
  }

  /**
   * Mock specific volcano data
   */
  async mockVolcanoData(volcanoId: string, data: any): Promise<void> {
    await this.page.route(`**/api/v1/volcanoes/${volcanoId}`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(data)
      });
    });
  }

  /**
   * Mock real-time data updates
   */
  async mockRealTimeUpdates(): Promise<void> {
    await this.page.route('**/api/v1/real-time/**', async (route) => {
      const url = route.request().url();
      
      if (url.includes('seismic')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            events: mockSeismicEvents.slice(-3),
            lastUpdate: new Date().toISOString()
          })
        });
      } else if (url.includes('deformation')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            measurements: [
              {
                timestamp: new Date().toISOString(),
                displacement: { x: 2.1, y: -1.5, z: 3.8 },
                units: 'mm'
              }
            ]
          })
        });
      }
    });
  }
}

/**
 * Mock WebSocket service for testing real-time features
 */
export class MockWebSocketService {
  private page: Page;
  private messageQueue: any[] = [];

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * Set up WebSocket mocks
   */
  async setupWebSocketMocks(): Promise<void> {
    await this.page.addInitScript(() => {
      // Mock WebSocket constructor
      (window as any).MockWebSocket = class {
        readyState = 1; // OPEN
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;

        constructor(url: string) {
          setTimeout(() => {
            if (this.onopen) {
              this.onopen(new Event('open'));
            }
          }, 100);
        }

        send(data: string) {
          // Mock sending data
          console.log('WebSocket send:', data);
        }

        close() {
          this.readyState = 3; // CLOSED
          if (this.onclose) {
            this.onclose(new CloseEvent('close'));
          }
        }

        // Method to simulate receiving messages
        simulateMessage(data: any) {
          if (this.onmessage) {
            this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
          }
        }
      };

      // Replace native WebSocket with mock
      (window as any).WebSocket = (window as any).MockWebSocket;
    });
  }

  /**
   * Simulate WebSocket connection
   */
  async simulateConnection(): Promise<void> {
    await this.page.evaluate(() => {
      const event = new CustomEvent('websocket-connected');
      document.dispatchEvent(event);
    });
  }

  /**
   * Simulate receiving a seismic event
   */
  async simulateSeismicEvent(): Promise<void> {
    await this.page.evaluate((message) => {
      const event = new CustomEvent('websocket-message', { 
        detail: message 
      });
      document.dispatchEvent(event);
    }, mockWebSocketMessages.seismicEvent);
  }

  /**
   * Simulate alert level change
   */
  async simulateAlertLevelChange(): Promise<void> {
    await this.page.evaluate((message) => {
      const event = new CustomEvent('websocket-message', { 
        detail: message 
      });
      document.dispatchEvent(event);
    }, mockWebSocketMessages.alertLevelChange);
  }

  /**
   * Simulate deformation update
   */
  async simulateDeformationUpdate(): Promise<void> {
    await this.page.evaluate((message) => {
      const event = new CustomEvent('websocket-message', { 
        detail: message 
      });
      document.dispatchEvent(event);
    }, mockWebSocketMessages.deformationUpdate);
  }

  /**
   * Simulate gas emission update
   */
  async simulateGasEmissionUpdate(): Promise<void> {
    await this.page.evaluate((message) => {
      const event = new CustomEvent('websocket-message', { 
        detail: message 
      });
      document.dispatchEvent(event);
    }, mockWebSocketMessages.gasEmissionUpdate);
  }

  /**
   * Simulate WebSocket disconnection
   */
  async simulateDisconnection(): Promise<void> {
    await this.page.evaluate(() => {
      const event = new CustomEvent('websocket-disconnected');
      document.dispatchEvent(event);
    });
  }

  /**
   * Simulate WebSocket error
   */
  async simulateError(): Promise<void> {
    await this.page.evaluate(() => {
      const event = new CustomEvent('websocket-error', {
        detail: { error: 'Connection failed' }
      });
      document.dispatchEvent(event);
    });
  }
}

/**
 * Mock geological calculation service
 */
export class MockGeologicalService {
  private page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * Mock geological calculation results
   */
  async setupGeologicalMocks(): Promise<void> {
    // Mock eruption probability calculation
    await this.page.route('**/api/v1/calculations/eruption-probability', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          probability: 15.7,
          confidence: 78.3,
          timeHorizon: '30 days',
          factors: [
            { name: 'Seismic Activity', weight: 0.35, value: 0.6 },
            { name: 'Deformation', weight: 0.25, value: 0.4 },
            { name: 'Gas Emissions', weight: 0.20, value: 0.7 },
            { name: 'Thermal Anomalies', weight: 0.20, value: 0.3 }
          ]
        })
      });
    });

    // Mock stress field calculation
    await this.page.route('**/api/v1/calculations/stress-field', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          principalStresses: [45.2, 32.1, 18.7],
          orientation: 'NE-SW',
          magnitude: 'moderate',
          confidence: 85.4
        })
      });
    });

    // Mock 3D geological model validation
    await this.page.route('**/api/v1/validation/geological-model', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          accuracy: 92.5,
          validationMetrics: {
            spatialAccuracy: 94.2,
            temporalAccuracy: 89.7,
            physicsConsistency: 93.8
          },
          errors: [],
          warnings: [
            'Limited data coverage in northern region'
          ]
        })
      });
    });
  }

  /**
   * Mock resource assessment calculations
   */
  async mockResourceAssessment(): Promise<void> {
    await this.page.route('**/api/v1/resource-assessment/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          prospectivityScore: 78.5,
          confidence: 72.3,
          resourceType: 'geothermal',
          estimatedCapacity: '150 MW',
          economicViability: 'high',
          riskFactors: [
            'Regulatory approval required',
            'Environmental impact assessment needed'
          ]
        })
      });
    });
  }
}

/**
 * Performance monitoring mock service
 */
export class MockPerformanceService {
  private page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * Mock performance metrics
   */
  async setupPerformanceMocks(): Promise<void> {
    await this.page.addInitScript(() => {
      // Mock performance API
      (window as any).mockPerformance = {
        now: () => Date.now(),
        mark: (name: string) => console.log(`Performance mark: ${name}`),
        measure: (name: string, start: string, end: string) => ({
          name,
          duration: Math.random() * 100 + 50 // Random duration between 50-150ms
        }),
        getEntriesByType: (type: string) => {
          if (type === 'navigation') {
            return [{
              loadEventEnd: 2500,
              loadEventStart: 2000,
              domContentLoadedEventEnd: 1800,
              domContentLoadedEventStart: 1500
            }];
          }
          return [];
        },
        getEntriesByName: (name: string) => {
          if (name === 'first-paint') {
            return [{ startTime: 800 }];
          }
          if (name === 'first-contentful-paint') {
            return [{ startTime: 1200 }];
          }
          return [];
        }
      };
    });
  }

  /**
   * Simulate performance metrics collection
   */
  async collectMetrics(): Promise<any> {
    return await this.page.evaluate(() => {
      return {
        loadTime: 2500,
        domContentLoaded: 1800,
        firstPaint: 800,
        firstContentfulPaint: 1200,
        memoryUsage: Math.random() * 500 + 200, // 200-700 MB
        fps: Math.floor(Math.random() * 30) + 30 // 30-60 FPS
      };
    });
  }
}