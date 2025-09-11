import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

/**
 * Page object for authentication and login functionality
 */
export class LoginPage extends BasePage {
  // Login form elements
  private readonly usernameInput: Locator;
  private readonly passwordInput: Locator;
  private readonly loginButton: Locator;
  private readonly forgotPasswordLink: Locator;
  private readonly registerLink: Locator;
  private readonly mfaCodeInput: Locator;
  private readonly mfaSubmitButton: Locator;
  
  // Error and success messages
  private readonly errorMessage: Locator;
  private readonly successMessage: Locator;
  
  // Security features
  private readonly captchaContainer: Locator;
  private readonly securityNotice: Locator;

  constructor(page: Page) {
    super(page);
    
    this.usernameInput = page.locator('[data-testid="username-input"]');
    this.passwordInput = page.locator('[data-testid="password-input"]');
    this.loginButton = page.locator('[data-testid="login-submit"]');
    this.forgotPasswordLink = page.locator('[data-testid="forgot-password-link"]');
    this.registerLink = page.locator('[data-testid="register-link"]');
    this.mfaCodeInput = page.locator('[data-testid="mfa-code-input"]');
    this.mfaSubmitButton = page.locator('[data-testid="mfa-submit"]');
    
    this.errorMessage = page.locator('[data-testid="login-error"]');
    this.successMessage = page.locator('[data-testid="login-success"]');
    
    this.captchaContainer = page.locator('[data-testid="captcha-container"]');
    this.securityNotice = page.locator('[data-testid="security-notice"]');
  }

  async goto(): Promise<void> {
    await super.goto('/login');
  }

  /**
   * Perform standard login
   */
  async login(username: string, password: string): Promise<void> {
    await this.usernameInput.fill(username);
    await this.passwordInput.fill(password);
    await this.loginButton.click();
    
    // Wait for either success or MFA prompt
    await Promise.race([
      this.page.locator('[data-testid="user-profile"]').waitFor({ state: 'visible' }),
      this.mfaCodeInput.waitFor({ state: 'visible' })
    ]);
  }

  /**
   * Complete MFA authentication
   */
  async completeMFA(code: string): Promise<void> {
    await this.mfaCodeInput.fill(code);
    await this.mfaSubmitButton.click();
    
    // Wait for successful authentication
    await this.page.locator('[data-testid="user-profile"]').waitFor({ state: 'visible' });
  }

  /**
   * Login with geological researcher role
   */
  async loginAsResearcher(): Promise<void> {
    await this.login('researcher@prometheus.ai', 'research123');
  }

  /**
   * Login with volcano observatory operator role
   */
  async loginAsOperator(): Promise<void> {
    await this.login('operator@prometheus.ai', 'operator123');
  }

  /**
   * Login with industry user role
   */
  async loginAsIndustryUser(): Promise<void> {
    await this.login('industry@prometheus.ai', 'industry123');
  }

  /**
   * Attempt login with invalid credentials
   */
  async loginWithInvalidCredentials(): Promise<void> {
    await this.usernameInput.fill('invalid@example.com');
    await this.passwordInput.fill('wrongpassword');
    await this.loginButton.click();
    
    // Wait for error message
    await this.errorMessage.waitFor({ state: 'visible' });
  }

  /**
   * Validate login page content and security features
   */
  async validatePageContent(): Promise<void> {
    // Verify form elements are present
    await expect(this.usernameInput).toBeVisible();
    await expect(this.passwordInput).toBeVisible();
    await expect(this.loginButton).toBeVisible();
    
    // Verify security notice is displayed
    await expect(this.securityNotice).toBeVisible();
    
    // Verify password field is properly masked
    await expect(this.passwordInput).toHaveAttribute('type', 'password');
    
    // Verify form has proper CSRF protection
    const csrfToken = this.page.locator('input[name="_token"]');
    await expect(csrfToken).toBeAttached();
  }

  /**
   * Verify error message content
   */
  async verifyErrorMessage(expectedMessage: string): Promise<void> {
    await expect(this.errorMessage).toBeVisible();
    await expect(this.errorMessage).toContainText(expectedMessage);
  }

  /**
   * Check if user is already logged in
   */
  async isLoggedIn(): Promise<boolean> {
    try {
      await this.page.locator('[data-testid="user-profile"]').waitFor({ 
        state: 'visible', 
        timeout: 2000 
      });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    await this.userMenu.click();
    await this.page.locator('[data-testid="logout-button"]').click();
    
    // Wait for redirect to login page
    await this.page.waitForURL('**/login');
  }
}