import { Page, Locator, expect } from '@playwright/test';

/**
 * Base page object class with common functionality for all pages
 * Provides geological domain-aware interactions and validations
 */
export abstract class BasePage {
  protected page: Page;
  protected baseUrl: string;

  // Common selectors
  protected readonly loadingSpinner: Locator;
  protected readonly errorBoundary: Locator;
  protected readonly navigationMenu: Locator;
  protected readonly userMenu: Locator;
  protected readonly notificationCenter: Locator;

  constructor(page: Page, baseUrl: string = 'http://localhost:5173') {
    this.page = page;
    this.baseUrl = baseUrl;
    
    // Initialize common locators
    this.loadingSpinner = page.locator('[data-testid="loading-spinner"]');
    this.errorBoundary = page.locator('[data-testid="error-boundary"]');
    this.navigationMenu = page.locator('[data-testid="navigation-menu"]');
    this.userMenu = page.locator('[data-testid="user-menu"]');
    this.notificationCenter = page.locator('[data-testid="notification-center"]');
  }

  /**
   * Navigate to the page and wait for it to be ready
   */
  async goto(path: string = ''): Promise<void> {
    await this.page.goto(`${this.baseUrl}${path}`);
    await this.waitForPageReady();
  }

  /**
   * Wait for page to be fully loaded and ready for interaction
   */
  async waitForPageReady(): Promise<void> {
    // Wait for loading spinner to disappear
    await this.loadingSpinner.waitFor({ state: 'hidden', timeout: 30000 });
    
    // Wait for network to be idle
    await this.page.waitForLoadState('networkidle');
    
    // Ensure no error boundary is visible
    await expect(this.errorBoundary).not.toBeVisible();
  }

  /**
   * Wait for geological data to be loaded and validated
   */
  async waitForGeologicalDataReady(): Promise<void> {
    // Wait for data validation indicators
    await this.page.locator('[data-testid="data-validation-complete"]').waitFor({ 
      state: 'visible', 
      timeout: 60000 
    });
    
    // Ensure data quality indicators are green
    const qualityIndicators = this.page.locator('[data-testid="data-quality-indicator"]');
    await expect(qualityIndicators).toHaveClass(/quality-good|quality-excellent/);
  }

  /**
   * Handle authentication if required
   */
  async ensureAuthenticated(): Promise<void> {
    const loginButton = this.page.locator('[data-testid="login-button"]');
    
    if (await loginButton.isVisible()) {
      await this.login();
    }
  }

  /**
   * Perform login with test credentials
   */
  async login(username: string = 'test@prometheus.ai', password: string = 'testpass123'): Promise<void> {
    await this.page.locator('[data-testid="username-input"]').fill(username);
    await this.page.locator('[data-testid="password-input"]').fill(password);
    await this.page.locator('[data-testid="login-submit"]').click();
    
    // Wait for successful login
    await this.page.locator('[data-testid="user-profile"]').waitFor({ state: 'visible' });
  }

  /**
   * Check for and handle any error states
   */
  async checkForErrors(): Promise<void> {
    const errorMessages = this.page.locator('[data-testid="error-message"]');
    const errorCount = await errorMessages.count();
    
    if (errorCount > 0) {
      const errors = await errorMessages.allTextContents();
      throw new Error(`Page errors detected: ${errors.join(', ')}`);
    }
  }

  /**
   * Wait for WebSocket connection to be established
   */
  async waitForWebSocketConnection(): Promise<void> {
    await this.page.locator('[data-testid="websocket-connected"]').waitFor({ 
      state: 'visible', 
      timeout: 15000 
    });
  }

  /**
   * Verify geological coordinate system is properly initialized
   */
  async verifyCoordinateSystem(): Promise<void> {
    const coordSystem = this.page.locator('[data-testid="coordinate-system-indicator"]');
    await expect(coordSystem).toBeVisible();
    await expect(coordSystem).toHaveAttribute('data-projection', /EPSG:\d+|WGS84|UTM/);
  }

  /**
   * Take a screenshot with geological context
   */
  async takeGeologicalScreenshot(name: string): Promise<void> {
    // Wait for all geological visualizations to be ready
    await this.page.locator('[data-testid="geological-viz-ready"]').waitFor({ 
      state: 'visible', 
      timeout: 30000 
    });
    
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
    
    await this.page.screenshot({ 
      path: `test-results/screenshots/${name}.png`,
      fullPage: true 
    });
  }

  /**
   * Verify accessibility compliance
   */
  async verifyAccessibility(): Promise<void> {
    // Check for proper ARIA labels on geological controls
    const geologicalControls = this.page.locator('[data-geological-control]');
    const controlCount = await geologicalControls.count();
    
    for (let i = 0; i < controlCount; i++) {
      const control = geologicalControls.nth(i);
      await expect(control).toHaveAttribute('aria-label');
    }
  }

  /**
   * Get current page performance metrics
   */
  async getPerformanceMetrics(): Promise<any> {
    return await this.page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        loadTime: navigation.loadEventEnd - navigation.loadEventStart,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
      };
    });
  }

  /**
   * Abstract method for page-specific validation
   */
  abstract validatePageContent(): Promise<void>;
}