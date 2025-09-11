import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'fs';
import path from 'path';

test.describe('Accessibility Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Ensure test data is available
    await page.goto('/');
    await page.waitForSelector('[data-testid="app-loaded"]');
  });

  test('should pass axe accessibility tests on dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();

    // Save results for reporting
    const resultsDir = path.join(process.cwd(), 'test-results', 'axe-results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    
    fs.writeFileSync(
      path.join(resultsDir, 'dashboard-axe-results.json'),
      JSON.stringify(accessibilityScanResults, null, 2)
    );

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should pass axe accessibility tests on volcanic monitoring', async ({ page }) => {
    await page.goto('/volcanic-monitoring');
    await page.waitForSelector('[data-testid="volcanic-monitoring-title"]');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();

    // Save results
    const resultsDir = path.join(process.cwd(), 'test-results', 'axe-results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    
    fs.writeFileSync(
      path.join(resultsDir, 'volcanic-monitoring-axe-results.json'),
      JSON.stringify(accessibilityScanResults, null, 2)
    );

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should support keyboard navigation', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    await test.step('Test tab navigation', async () => {
      // Start from the beginning
      await page.keyboard.press('Tab');
      
      let focusedElement = await page.locator(':focus');
      await expect(focusedElement).toBeVisible();
      
      // Continue tabbing through interactive elements
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press('Tab');
        focusedElement = await page.locator(':focus');
        
        // Verify focus is always visible
        await expect(focusedElement).toBeVisible();
        
        // Verify focus indicator is present
        const focusOutline = await focusedElement.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return styles.outline !== 'none' || styles.boxShadow !== 'none';
        });
        expect(focusOutline).toBe(true);
      }
    });

    await test.step('Test Enter key activation', async () => {
      // Focus on a button and activate with Enter
      const addButton = page.locator('[data-testid="add-widget-button"]');
      await addButton.focus();
      await page.keyboard.press('Enter');
      
      // Should open add widget dialog
      const dialog = page.locator('[data-testid="add-widget-dialog"]');
      await expect(dialog).toBeVisible();
      
      // Close dialog with Escape
      await page.keyboard.press('Escape');
      await expect(dialog).not.toBeVisible();
    });

    await test.step('Test arrow key navigation in lists', async () => {
      // Navigate to a list component
      const volcanoList = page.locator('[data-testid="volcano-list"]');
      if (await volcanoList.isVisible()) {
        const firstItem = volcanoList.locator('[data-testid="volcano-card"]').first();
        await firstItem.focus();
        
        // Use arrow keys to navigate
        await page.keyboard.press('ArrowDown');
        const secondItem = page.locator(':focus');
        
        // Verify focus moved to next item
        expect(await secondItem.getAttribute('data-testid')).toBe('volcano-card');
      }
    });
  });

  test('should support screen readers', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    await test.step('Verify ARIA labels and roles', async () => {
      // Check main heading
      const title = page.locator('[data-testid="dashboard-title"]');
      await expect(title).toHaveAttribute('role', 'heading');
      
      // Check navigation elements
      const nav = page.locator('nav');
      if (await nav.count() > 0) {
        await expect(nav.first()).toHaveAttribute('role', 'navigation');
      }
      
      // Check buttons have accessible names
      const buttons = page.locator('button');
      const buttonCount = await buttons.count();
      
      for (let i = 0; i < buttonCount; i++) {
        const button = buttons.nth(i);
        const ariaLabel = await button.getAttribute('aria-label');
        const textContent = await button.textContent();
        
        // Button should have either aria-label or text content
        expect(ariaLabel || textContent?.trim()).toBeTruthy();
      }
    });

    await test.step('Verify form accessibility', async () => {
      // Check for form labels
      const inputs = page.locator('input, select, textarea');
      const inputCount = await inputs.count();
      
      for (let i = 0; i < inputCount; i++) {
        const input = inputs.nth(i);
        const id = await input.getAttribute('id');
        const ariaLabel = await input.getAttribute('aria-label');
        const ariaLabelledBy = await input.getAttribute('aria-labelledby');
        
        if (id) {
          // Check for associated label
          const label = page.locator(`label[for="${id}"]`);
          const hasLabel = await label.count() > 0;
          
          // Input should have label, aria-label, or aria-labelledby
          expect(hasLabel || ariaLabel || ariaLabelledBy).toBeTruthy();
        }
      }
    });

    await test.step('Verify live regions for dynamic content', async () => {
      // Check for aria-live regions
      const liveRegions = page.locator('[aria-live]');
      const liveRegionCount = await liveRegions.count();
      
      if (liveRegionCount > 0) {
        for (let i = 0; i < liveRegionCount; i++) {
          const region = liveRegions.nth(i);
          const ariaLive = await region.getAttribute('aria-live');
          
          // Should be 'polite' or 'assertive'
          expect(['polite', 'assertive']).toContain(ariaLive);
        }
      }
    });
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    await test.step('Check text color contrast', async () => {
      const textElements = page.locator('p, span, div, h1, h2, h3, h4, h5, h6, button, a');
      const elementCount = Math.min(await textElements.count(), 20); // Limit for performance
      
      for (let i = 0; i < elementCount; i++) {
        const element = textElements.nth(i);
        const textContent = await element.textContent();
        
        if (textContent && textContent.trim().length > 0) {
          const styles = await element.evaluate(el => {
            const computed = window.getComputedStyle(el);
            return {
              color: computed.color,
              backgroundColor: computed.backgroundColor,
              fontSize: computed.fontSize
            };
          });
          
          // Parse RGB values
          const colorMatch = styles.color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
          const bgColorMatch = styles.backgroundColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
          
          if (colorMatch && bgColorMatch) {
            const textColor = {
              r: parseInt(colorMatch[1]),
              g: parseInt(colorMatch[2]),
              b: parseInt(colorMatch[3])
            };
            
            const bgColor = {
              r: parseInt(bgColorMatch[1]),
              g: parseInt(bgColorMatch[2]),
              b: parseInt(bgColorMatch[3])
            };
            
            // Calculate contrast ratio
            const contrast = calculateContrastRatio(textColor, bgColor);
            const fontSize = parseFloat(styles.fontSize);
            
            // WCAG AA requirements
            const minContrast = fontSize >= 18 || (fontSize >= 14 && await element.evaluate(el => 
              window.getComputedStyle(el).fontWeight === 'bold'
            )) ? 3 : 4.5;
            
            expect(contrast).toBeGreaterThanOrEqual(minContrast);
          }
        }
      }
    });
  });

  test('should support high contrast mode', async ({ page }) => {
    // Simulate high contrast mode
    await page.emulateMedia({ colorScheme: 'dark', forcedColors: 'active' });
    
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    await test.step('Verify elements are visible in high contrast', async () => {
      // Check that important elements are still visible
      const importantElements = [
        '[data-testid="dashboard-title"]',
        '[data-testid="add-widget-button"]',
        '[data-testid="widget-container"]'
      ];
      
      for (const selector of importantElements) {
        const element = page.locator(selector);
        await expect(element).toBeVisible();
        
        // Verify element has some visual distinction
        const styles = await element.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return {
            color: computed.color,
            backgroundColor: computed.backgroundColor,
            border: computed.border
          };
        });
        
        // Should have color, background, or border defined
        const hasVisualStyling = styles.color !== 'rgba(0, 0, 0, 0)' ||
                                styles.backgroundColor !== 'rgba(0, 0, 0, 0)' ||
                                styles.border !== 'none';
        
        expect(hasVisualStyling).toBe(true);
      }
    });
  });

  test('should support reduced motion preferences', async ({ page }) => {
    // Simulate reduced motion preference
    await page.emulateMedia({ reducedMotion: 'reduce' });
    
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    await test.step('Verify animations are disabled', async () => {
      // Check that CSS animations respect reduced motion
      const animatedElements = page.locator('[class*="animate"], [class*="transition"]');
      const elementCount = await animatedElements.count();
      
      for (let i = 0; i < Math.min(elementCount, 10); i++) {
        const element = animatedElements.nth(i);
        const animationDuration = await element.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return computed.animationDuration;
        });
        
        // Animation duration should be 0 or very short
        expect(animationDuration === '0s' || animationDuration === 'none').toBe(true);
      }
    });
  });

  test('should be usable with zoom up to 200%', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="dashboard-title"]');

    // Test different zoom levels
    const zoomLevels = [1.5, 2.0];
    
    for (const zoom of zoomLevels) {
      await test.step(`Test at ${zoom * 100}% zoom`, async () => {
        // Set zoom level
        await page.evaluate((zoomLevel) => {
          document.body.style.zoom = zoomLevel.toString();
        }, zoom);
        
        // Verify important elements are still accessible
        const title = page.locator('[data-testid="dashboard-title"]');
        await expect(title).toBeVisible();
        
        const addButton = page.locator('[data-testid="add-widget-button"]');
        await expect(addButton).toBeVisible();
        
        // Verify no horizontal scrolling is required for main content
        const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
        const viewportWidth = await page.evaluate(() => window.innerWidth);
        
        expect(bodyWidth).toBeLessThanOrEqual(viewportWidth * 1.1); // Allow small tolerance
      });
    }
    
    // Reset zoom
    await page.evaluate(() => {
      document.body.style.zoom = '1';
    });
  });
});

// Helper function to calculate color contrast ratio
function calculateContrastRatio(color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}): number {
  const luminance1 = getRelativeLuminance(color1);
  const luminance2 = getRelativeLuminance(color2);
  
  const lighter = Math.max(luminance1, luminance2);
  const darker = Math.min(luminance1, luminance2);
  
  return (lighter + 0.05) / (darker + 0.05);
}

function getRelativeLuminance(color: {r: number, g: number, b: number}): number {
  const rsRGB = color.r / 255;
  const gsRGB = color.g / 255;
  const bsRGB = color.b / 255;
  
  const r = rsRGB <= 0.03928 ? rsRGB / 12.92 : Math.pow((rsRGB + 0.055) / 1.055, 2.4);
  const g = gsRGB <= 0.03928 ? gsRGB / 12.92 : Math.pow((gsRGB + 0.055) / 1.055, 2.4);
  const b = bsRGB <= 0.03928 ? bsRGB / 12.92 : Math.pow((bsRGB + 0.055) / 1.055, 2.4);
  
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}