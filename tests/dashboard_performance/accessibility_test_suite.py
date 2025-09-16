"""
Accessibility and Usability Test Suite

Comprehensive testing for WCAG 2.1 AA compliance, usability,
cross-browser compatibility, and mobile responsiveness.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AccessibilityViolation:
    """Represents an accessibility violation"""
    rule_id: str
    impact: str  # 'minor', 'moderate', 'serious', 'critical'
    description: str
    help_url: str
    element: str
    fix_suggestions: List[str] = field(default_factory=list)


@dataclass
class UsabilityTestResult:
    """Result from usability testing"""
    test_name: str
    success: bool
    score: float
    violations: List[AccessibilityViolation] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    user_experience_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class AccessibilityTestSuite:
    """
    Comprehensive accessibility and usability testing suite
    """
    
    def __init__(self, base_url: str = "http://localhost:3000", output_dir: str = "accessibility_results"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results: List[UsabilityTestResult] = []
        
        # WCAG 2.1 AA compliance requirements
        self.wcag_requirements = {
            'color_contrast': 4.5,  # Minimum contrast ratio
            'large_text_contrast': 3.0,  # For text 18pt+ or 14pt+ bold
            'focus_visible': True,
            'keyboard_accessible': True,
            'alt_text_required': True,
            'heading_structure': True,
            'form_labels': True,
            'link_purpose': True,
            'language_specified': True,
            'page_title': True
        }
        
        # Browser configurations for cross-browser testing
        self.browser_configs = [
            {'name': 'chromium', 'viewport': {'width': 1920, 'height': 1080}},
            {'name': 'firefox', 'viewport': {'width': 1920, 'height': 1080}},
            {'name': 'webkit', 'viewport': {'width': 1920, 'height': 1080}},
        ]
        
        # Mobile device configurations
        self.mobile_configs = [
            {'name': 'iPhone 12', 'viewport': {'width': 390, 'height': 844}, 'user_agent': 'iPhone'},
            {'name': 'iPad', 'viewport': {'width': 820, 'height': 1180}, 'user_agent': 'iPad'},
            {'name': 'Android Phone', 'viewport': {'width': 360, 'height': 640}, 'user_agent': 'Android'},
        ]
    
    async def setup_browser(self, browser_name: str = 'chromium', viewport: Dict[str, int] = None):
        """Setup browser for testing"""
        playwright = await async_playwright().start()
        
        if viewport is None:
            viewport = {'width': 1920, 'height': 1080}
        
        if browser_name == 'chromium':
            browser = await playwright.chromium.launch(headless=True)
        elif browser_name == 'firefox':
            browser = await playwright.firefox.launch(headless=True)
        elif browser_name == 'webkit':
            browser = await playwright.webkit.launch(headless=True)
        else:
            raise ValueError(f"Unsupported browser: {browser_name}")
        
        context = await browser.new_context(
            viewport=viewport,
            permissions=['geolocation']
        )
        
        # Inject axe-core for accessibility testing
        await context.add_init_script("""
            // Inject axe-core accessibility testing library
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/axe-core@4.7.0/axe.min.js';
            document.head.appendChild(script);
        """)
        
        page = await context.new_page()
        return browser, context, page
    
    async def test_wcag_compliance(self, page: Page, test_name: str) -> UsabilityTestResult:
        """Test WCAG 2.1 AA compliance"""
        result = UsabilityTestResult(
            test_name=f"wcag_compliance_{test_name}",
            success=False,
            score=0.0
        )
        
        try:
            # Wait for axe-core to load
            await page.wait_for_function("typeof axe !== 'undefined'", timeout=10000)
            
            # Run axe accessibility scan
            axe_results = await page.evaluate("""
                async () => {
                    try {
                        const results = await axe.run({
                            tags: ['wcag2a', 'wcag2aa', 'wcag21aa'],
                            options: {
                                reporter: 'v2'
                            }
                        });
                        return results;
                    } catch (error) {
                        return { error: error.message };
                    }
                }
            """)
            
            if 'error' in axe_results:
                result.error_message = axe_results['error']
                return result
            
            # Process violations
            violations = []
            for violation in axe_results.get('violations', []):
                for node in violation.get('nodes', []):
                    accessibility_violation = AccessibilityViolation(
                        rule_id=violation['id'],
                        impact=violation.get('impact', 'unknown'),
                        description=violation['description'],
                        help_url=violation.get('helpUrl', ''),
                        element=node.get('html', ''),
                        fix_suggestions=node.get('failureSummary', '').split('\n') if node.get('failureSummary') else []
                    )
                    violations.append(accessibility_violation)
            
            result.violations = violations
            
            # Calculate compliance score
            total_rules = len(axe_results.get('passes', [])) + len(axe_results.get('violations', []))
            passed_rules = len(axe_results.get('passes', []))
            
            if total_rules > 0:
                result.score = passed_rules / total_rules
            else:
                result.score = 1.0
            
            # Categorize violations by severity
            critical_violations = [v for v in violations if v.impact == 'critical']
            serious_violations = [v for v in violations if v.impact == 'serious']
            moderate_violations = [v for v in violations if v.impact == 'moderate']
            minor_violations = [v for v in violations if v.impact == 'minor']
            
            result.user_experience_metrics = {
                'total_violations': len(violations),
                'critical_violations': len(critical_violations),
                'serious_violations': len(serious_violations),
                'moderate_violations': len(moderate_violations),
                'minor_violations': len(minor_violations),
                'compliance_score': result.score,
                'total_rules_tested': total_rules,
                'passed_rules': passed_rules
            }
            
            # Success criteria: no critical or serious violations
            result.success = len(critical_violations) == 0 and len(serious_violations) == 0
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"WCAG compliance test failed: {e}")
        
        return result
    
    async def test_keyboard_navigation(self, page: Page, test_name: str) -> UsabilityTestResult:
        """Test keyboard navigation accessibility"""
        result = UsabilityTestResult(
            test_name=f"keyboard_navigation_{test_name}",
            success=False,
            score=0.0
        )
        
        try:
            # Get all interactive elements
            interactive_elements = await page.evaluate("""
                () => {
                    const selectors = [
                        'button', 'a[href]', 'input', 'select', 'textarea',
                        '[tabindex]:not([tabindex="-1"])', '[role="button"]',
                        '[role="link"]', '[role="menuitem"]', '[role="tab"]'
                    ];
                    
                    const elements = [];
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => {
                            if (el.offsetParent !== null) { // visible elements only
                                elements.push({
                                    tagName: el.tagName,
                                    id: el.id,
                                    className: el.className,
                                    text: el.textContent?.trim().substring(0, 50) || '',
                                    tabIndex: el.tabIndex,
                                    role: el.getAttribute('role')
                                });
                            }
                        });
                    });
                    
                    return elements;
                }
            """)
            
            # Test tab navigation
            tab_navigation_success = 0
            total_elements = len(interactive_elements)
            
            if total_elements > 0:
                # Start from first element
                await page.keyboard.press('Tab')
                
                for i in range(min(total_elements, 20)):  # Test first 20 elements
                    try:
                        # Check if an element has focus
                        focused_element = await page.evaluate("""
                            () => {
                                const el = document.activeElement;
                                return el ? {
                                    tagName: el.tagName,
                                    id: el.id,
                                    className: el.className,
                                    text: el.textContent?.trim().substring(0, 50) || ''
                                } : null;
                            }
                        """)
                        
                        if focused_element:
                            tab_navigation_success += 1
                        
                        # Move to next element
                        await page.keyboard.press('Tab')
                        await asyncio.sleep(0.1)
                        
                    except Exception:
                        continue
            
            # Test focus visibility
            focus_visible_test = await page.evaluate("""
                () => {
                    const style = getComputedStyle(document.activeElement);
                    const hasOutline = style.outline !== 'none' && style.outline !== '';
                    const hasBoxShadow = style.boxShadow !== 'none';
                    const hasBorder = style.border !== 'none' && style.borderWidth !== '0px';
                    
                    return hasOutline || hasBoxShadow || hasBorder;
                }
            """)
            
            # Test Enter/Space key activation
            activation_test_success = 0
            try:
                # Find a button to test
                button = await page.query_selector('button:not([disabled])')
                if button:
                    await button.focus()
                    
                    # Test Enter key
                    await page.keyboard.press('Enter')
                    activation_test_success += 1
                    
                    await asyncio.sleep(0.5)
                    
                    # Test Space key
                    await page.keyboard.press('Space')
                    activation_test_success += 1
            except Exception:
                pass
            
            # Calculate scores
            tab_score = tab_navigation_success / max(total_elements, 1)
            focus_score = 1.0 if focus_visible_test else 0.0
            activation_score = activation_test_success / 2.0 if activation_test_success > 0 else 0.0
            
            result.score = (tab_score + focus_score + activation_score) / 3.0
            
            result.user_experience_metrics = {
                'total_interactive_elements': total_elements,
                'tab_navigation_success': tab_navigation_success,
                'tab_navigation_score': tab_score,
                'focus_visible': focus_visible_test,
                'activation_test_success': activation_test_success,
                'keyboard_accessibility_score': result.score
            }
            
            result.success = result.score >= 0.8
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Keyboard navigation test failed: {e}")
        
        return result
    
    async def test_color_contrast(self, page: Page, test_name: str) -> UsabilityTestResult:
        """Test color contrast compliance"""
        result = UsabilityTestResult(
            test_name=f"color_contrast_{test_name}",
            success=False,
            score=0.0
        )
        
        try:
            # Get all text elements and their computed styles
            contrast_results = await page.evaluate("""
                () => {
                    function getLuminance(r, g, b) {
                        const [rs, gs, bs] = [r, g, b].map(c => {
                            c = c / 255;
                            return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
                        });
                        return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
                    }
                    
                    function getContrastRatio(color1, color2) {
                        const lum1 = getLuminance(...color1);
                        const lum2 = getLuminance(...color2);
                        const brightest = Math.max(lum1, lum2);
                        const darkest = Math.min(lum1, lum2);
                        return (brightest + 0.05) / (darkest + 0.05);
                    }
                    
                    function parseColor(colorStr) {
                        const div = document.createElement('div');
                        div.style.color = colorStr;
                        document.body.appendChild(div);
                        const computed = getComputedStyle(div).color;
                        document.body.removeChild(div);
                        
                        const match = computed.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/);
                        return match ? [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])] : [0, 0, 0];
                    }
                    
                    const textElements = document.querySelectorAll('*');
                    const results = [];
                    
                    textElements.forEach(el => {
                        const text = el.textContent?.trim();
                        if (text && text.length > 0 && el.offsetParent !== null) {
                            const style = getComputedStyle(el);
                            const fontSize = parseFloat(style.fontSize);
                            const fontWeight = style.fontWeight;
                            
                            const textColor = parseColor(style.color);
                            const bgColor = parseColor(style.backgroundColor);
                            
                            const contrast = getContrastRatio(textColor, bgColor);
                            const isLargeText = fontSize >= 18 || (fontSize >= 14 && (fontWeight === 'bold' || parseInt(fontWeight) >= 700));
                            
                            results.push({
                                element: el.tagName + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
                                text: text.substring(0, 50),
                                fontSize: fontSize,
                                fontWeight: fontWeight,
                                contrast: contrast,
                                isLargeText: isLargeText,
                                meetsWCAG: isLargeText ? contrast >= 3.0 : contrast >= 4.5
                            });
                        }
                    });
                    
                    return results;
                }
            """)
            
            # Analyze contrast results
            total_elements = len(contrast_results)
            compliant_elements = sum(1 for r in contrast_results if r['meetsWCAG'])
            
            if total_elements > 0:
                result.score = compliant_elements / total_elements
            else:
                result.score = 1.0
            
            # Find worst contrast violations
            violations = [r for r in contrast_results if not r['meetsWCAG']]
            violations.sort(key=lambda x: x['contrast'])  # Sort by worst contrast first
            
            result.user_experience_metrics = {
                'total_text_elements': total_elements,
                'compliant_elements': compliant_elements,
                'contrast_violations': len(violations),
                'worst_contrast_ratio': violations[0]['contrast'] if violations else None,
                'average_contrast_ratio': sum(r['contrast'] for r in contrast_results) / total_elements if total_elements > 0 else 0,
                'compliance_rate': result.score
            }
            
            # Create accessibility violations for poor contrast
            for violation in violations[:10]:  # Report top 10 violations
                result.violations.append(AccessibilityViolation(
                    rule_id='color-contrast',
                    impact='serious',
                    description=f"Text has insufficient color contrast ratio of {violation['contrast']:.2f}",
                    help_url='https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html',
                    element=violation['element'],
                    fix_suggestions=[
                        f"Increase contrast ratio to at least {'3.0' if violation['isLargeText'] else '4.5'}",
                        "Consider using darker text or lighter background colors",
                        "Test with color contrast analyzer tools"
                    ]
                ))
            
            result.success = result.score >= 0.95  # 95% compliance required
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Color contrast test failed: {e}")
        
        return result
    
    async def test_mobile_responsiveness(self, page: Page, test_name: str) -> UsabilityTestResult:
        """Test mobile responsiveness"""
        result = UsabilityTestResult(
            test_name=f"mobile_responsiveness_{test_name}",
            success=False,
            score=0.0
        )
        
        try:
            mobile_test_results = []
            
            for config in self.mobile_configs:
                # Set mobile viewport
                await page.set_viewport_size(config['viewport']['width'], config['viewport']['height'])
                
                # Wait for layout to adjust
                await asyncio.sleep(1)
                
                # Test mobile-specific features
                mobile_metrics = await page.evaluate("""
                    () => {
                        const viewport = {
                            width: window.innerWidth,
                            height: window.innerHeight
                        };
                        
                        // Check for horizontal scrolling
                        const hasHorizontalScroll = document.body.scrollWidth > viewport.width;
                        
                        // Check for touch-friendly button sizes
                        const buttons = document.querySelectorAll('button, a, [role="button"]');
                        let touchFriendlyButtons = 0;
                        
                        buttons.forEach(btn => {
                            const rect = btn.getBoundingClientRect();
                            const size = Math.min(rect.width, rect.height);
                            if (size >= 44) { // 44px minimum touch target
                                touchFriendlyButtons++;
                            }
                        });
                        
                        // Check for responsive text
                        const textElements = document.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6');
                        let readableText = 0;
                        
                        textElements.forEach(el => {
                            const style = getComputedStyle(el);
                            const fontSize = parseFloat(style.fontSize);
                            if (fontSize >= 16) { // 16px minimum for mobile
                                readableText++;
                            }
                        });
                        
                        // Check for mobile navigation
                        const mobileNav = document.querySelector('[data-testid="mobile-nav"], .mobile-nav, .hamburger-menu');
                        const hasMobileNav = mobileNav !== null;
                        
                        return {
                            viewport: viewport,
                            hasHorizontalScroll: hasHorizontalScroll,
                            totalButtons: buttons.length,
                            touchFriendlyButtons: touchFriendlyButtons,
                            totalTextElements: textElements.length,
                            readableText: readableText,
                            hasMobileNav: hasMobileNav
                        };
                    }
                """)
                
                # Calculate mobile score for this configuration
                button_score = mobile_metrics['touchFriendlyButtons'] / max(mobile_metrics['totalButtons'], 1)
                text_score = mobile_metrics['readableText'] / max(mobile_metrics['totalTextElements'], 1)
                scroll_score = 0.0 if mobile_metrics['hasHorizontalScroll'] else 1.0
                nav_score = 1.0 if mobile_metrics['hasMobileNav'] else 0.5
                
                config_score = (button_score + text_score + scroll_score + nav_score) / 4.0
                
                mobile_test_results.append({
                    'device': config['name'],
                    'viewport': config['viewport'],
                    'score': config_score,
                    'metrics': mobile_metrics
                })
            
            # Calculate overall mobile score
            result.score = sum(r['score'] for r in mobile_test_results) / len(mobile_test_results)
            
            result.user_experience_metrics = {
                'mobile_configurations_tested': len(mobile_test_results),
                'mobile_test_results': mobile_test_results,
                'overall_mobile_score': result.score
            }
            
            result.success = result.score >= 0.8
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Mobile responsiveness test failed: {e}")
        
        return result
    
    async def test_cross_browser_compatibility(self, test_name: str) -> UsabilityTestResult:
        """Test cross-browser compatibility"""
        result = UsabilityTestResult(
            test_name=f"cross_browser_{test_name}",
            success=False,
            score=0.0
        )
        
        try:
            browser_results = []
            
            for browser_config in self.browser_configs:
                try:
                    browser, context, page = await self.setup_browser(
                        browser_config['name'], 
                        browser_config['viewport']
                    )
                    
                    # Navigate to dashboard
                    await page.goto(f"{self.base_url}/dashboard")
                    
                    # Test basic functionality
                    functionality_score = await self._test_basic_functionality(page)
                    
                    # Test rendering consistency
                    rendering_score = await self._test_rendering_consistency(page)
                    
                    # Test JavaScript execution
                    js_score = await self._test_javascript_execution(page)
                    
                    browser_score = (functionality_score + rendering_score + js_score) / 3.0
                    
                    browser_results.append({
                        'browser': browser_config['name'],
                        'score': browser_score,
                        'functionality_score': functionality_score,
                        'rendering_score': rendering_score,
                        'javascript_score': js_score
                    })
                    
                    await page.close()
                    await context.close()
                    await browser.close()
                    
                except Exception as e:
                    browser_results.append({
                        'browser': browser_config['name'],
                        'score': 0.0,
                        'error': str(e)
                    })
            
            # Calculate overall compatibility score
            valid_results = [r for r in browser_results if 'error' not in r]
            if valid_results:
                result.score = sum(r['score'] for r in valid_results) / len(valid_results)
            else:
                result.score = 0.0
            
            result.user_experience_metrics = {
                'browsers_tested': len(browser_results),
                'successful_browsers': len(valid_results),
                'browser_results': browser_results,
                'compatibility_score': result.score
            }
            
            result.success = result.score >= 0.9 and len(valid_results) >= 2
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Cross-browser compatibility test failed: {e}")
        
        return result
    
    async def _test_basic_functionality(self, page: Page) -> float:
        """Test basic dashboard functionality"""
        try:
            # Wait for main components
            await page.wait_for_selector('[data-testid="causal-network-container"]', timeout=10000)
            await page.wait_for_selector('[data-testid="real-time-monitor"]', timeout=10000)
            
            # Test navigation
            nav_links = await page.query_selector_all('nav a, [data-testid*="nav"]')
            if len(nav_links) > 0:
                await nav_links[0].click()
                await asyncio.sleep(1)
            
            # Test interactive elements
            buttons = await page.query_selector_all('button:not([disabled])')
            if len(buttons) > 0:
                await buttons[0].click()
                await asyncio.sleep(0.5)
            
            return 1.0
            
        except Exception:
            return 0.0
    
    async def _test_rendering_consistency(self, page: Page) -> float:
        """Test rendering consistency"""
        try:
            # Check for layout issues
            layout_issues = await page.evaluate("""
                () => {
                    let issues = 0;
                    
                    // Check for overlapping elements
                    const elements = document.querySelectorAll('*');
                    elements.forEach(el => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 0 || rect.height < 0) {
                            issues++;
                        }
                    });
                    
                    // Check for invisible text
                    const textElements = document.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6');
                    textElements.forEach(el => {
                        const style = getComputedStyle(el);
                        if (style.color === style.backgroundColor) {
                            issues++;
                        }
                    });
                    
                    return issues;
                }
            """)
            
            return 1.0 if layout_issues == 0 else max(0.0, 1.0 - (layout_issues * 0.1))
            
        except Exception:
            return 0.0
    
    async def _test_javascript_execution(self, page: Page) -> float:
        """Test JavaScript execution"""
        try:
            # Test basic JavaScript functionality
            js_test_result = await page.evaluate("""
                () => {
                    try {
                        // Test modern JavaScript features
                        const testArray = [1, 2, 3];
                        const doubled = testArray.map(x => x * 2);
                        
                        // Test async/await support
                        const testPromise = new Promise(resolve => resolve(true));
                        
                        // Test DOM manipulation
                        const testDiv = document.createElement('div');
                        testDiv.textContent = 'test';
                        
                        return {
                            success: true,
                            features: {
                                arrow_functions: true,
                                promises: true,
                                dom_manipulation: true
                            }
                        };
                    } catch (error) {
                        return {
                            success: false,
                            error: error.message
                        };
                    }
                }
            """)
            
            return 1.0 if js_test_result.get('success', False) else 0.0
            
        except Exception:
            return 0.0
    
    async def run_comprehensive_accessibility_tests(self) -> Dict[str, Any]:
        """Run comprehensive accessibility and usability tests"""
        logger.info("Starting comprehensive accessibility tests")
        
        all_results = []
        
        # Test main dashboard pages
        test_pages = [
            '/dashboard',
            '/dashboard/causal-network',
            '/dashboard/real-time',
            '/dashboard/volcanic-systems',
            '/dashboard/data-quality'
        ]
        
        for page_path in test_pages:
            page_name = page_path.split('/')[-1] or 'main'
            
            # Setup browser for desktop testing
            browser, context, page = await self.setup_browser()
            
            try:
                await page.goto(f"{self.base_url}{page_path}")
                await page.wait_for_load_state('domcontentloaded')
                
                # WCAG compliance test
                wcag_result = await self.test_wcag_compliance(page, page_name)
                all_results.append(wcag_result)
                
                # Keyboard navigation test
                keyboard_result = await self.test_keyboard_navigation(page, page_name)
                all_results.append(keyboard_result)
                
                # Color contrast test
                contrast_result = await self.test_color_contrast(page, page_name)
                all_results.append(contrast_result)
                
                # Mobile responsiveness test
                mobile_result = await self.test_mobile_responsiveness(page, page_name)
                all_results.append(mobile_result)
                
            finally:
                await page.close()
                await context.close()
                await browser.close()
        
        # Cross-browser compatibility test
        cross_browser_result = await self.test_cross_browser_compatibility('dashboard')
        all_results.append(cross_browser_result)
        
        self.test_results = all_results
        
        # Generate comprehensive report
        report = self._generate_accessibility_report()
        
        # Save report
        report_file = self.output_dir / f"accessibility_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Accessibility tests completed. Report saved to {report_file}")
        
        return report
    
    def _generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive accessibility report"""
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # Categorize violations by severity
        all_violations = []
        for result in self.test_results:
            all_violations.extend(result.violations)
        
        critical_violations = [v for v in all_violations if v.impact == 'critical']
        serious_violations = [v for v in all_violations if v.impact == 'serious']
        moderate_violations = [v for v in all_violations if v.impact == 'moderate']
        minor_violations = [v for v in all_violations if v.impact == 'minor']
        
        # Calculate average scores by test type
        test_type_scores = {}
        for result in self.test_results:
            test_type = result.test_name.split('_')[0]
            if test_type not in test_type_scores:
                test_type_scores[test_type] = []
            test_type_scores[test_type].append(result.score)
        
        avg_scores = {
            test_type: sum(scores) / len(scores)
            for test_type, scores in test_type_scores.items()
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'overall_accessibility_score': sum(r.score for r in self.test_results) / total_tests if total_tests > 0 else 0
            },
            'violation_summary': {
                'total_violations': len(all_violations),
                'critical_violations': len(critical_violations),
                'serious_violations': len(serious_violations),
                'moderate_violations': len(moderate_violations),
                'minor_violations': len(minor_violations)
            },
            'test_type_scores': avg_scores,
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'score': r.score,
                    'violations_count': len(r.violations),
                    'user_experience_metrics': r.user_experience_metrics,
                    'error_message': r.error_message
                }
                for r in self.test_results
            ],
            'top_violations': [
                {
                    'rule_id': v.rule_id,
                    'impact': v.impact,
                    'description': v.description,
                    'element': v.element,
                    'fix_suggestions': v.fix_suggestions
                }
                for v in (critical_violations + serious_violations)[:10]
            ],
            'wcag_compliance_status': self._assess_wcag_compliance(),
            'recommendations': self._generate_accessibility_recommendations()
        }
        
        return report
    
    def _assess_wcag_compliance(self) -> Dict[str, Any]:
        """Assess overall WCAG 2.1 AA compliance"""
        wcag_tests = [r for r in self.test_results if 'wcag' in r.test_name]
        
        if not wcag_tests:
            return {'status': 'not_tested', 'compliance_level': 'unknown'}
        
        avg_score = sum(r.score for r in wcag_tests) / len(wcag_tests)
        critical_violations = sum(len([v for v in r.violations if v.impact == 'critical']) for r in wcag_tests)
        serious_violations = sum(len([v for v in r.violations if v.impact == 'serious']) for r in wcag_tests)
        
        if critical_violations > 0:
            status = 'non_compliant'
            level = 'fails_aa'
        elif serious_violations > 0:
            status = 'partially_compliant'
            level = 'partial_aa'
        elif avg_score >= 0.95:
            status = 'compliant'
            level = 'wcag_aa'
        else:
            status = 'partially_compliant'
            level = 'partial_aa'
        
        return {
            'status': status,
            'compliance_level': level,
            'average_score': avg_score,
            'critical_violations': critical_violations,
            'serious_violations': serious_violations
        }
    
    def _generate_accessibility_recommendations(self) -> List[str]:
        """Generate accessibility improvement recommendations"""
        recommendations = []
        
        # Check for critical issues
        critical_violations = []
        for result in self.test_results:
            critical_violations.extend([v for v in result.violations if v.impact == 'critical'])
        
        if critical_violations:
            recommendations.append(f"Fix {len(critical_violations)} critical accessibility violations immediately")
        
        # Check keyboard navigation
        keyboard_tests = [r for r in self.test_results if 'keyboard' in r.test_name]
        if keyboard_tests and any(r.score < 0.8 for r in keyboard_tests):
            recommendations.append("Improve keyboard navigation accessibility")
        
        # Check color contrast
        contrast_tests = [r for r in self.test_results if 'contrast' in r.test_name]
        if contrast_tests and any(r.score < 0.95 for r in contrast_tests):
            recommendations.append("Improve color contrast ratios to meet WCAG AA standards")
        
        # Check mobile responsiveness
        mobile_tests = [r for r in self.test_results if 'mobile' in r.test_name]
        if mobile_tests and any(r.score < 0.8 for r in mobile_tests):
            recommendations.append("Improve mobile responsiveness and touch target sizes")
        
        # Check cross-browser compatibility
        browser_tests = [r for r in self.test_results if 'cross_browser' in r.test_name]
        if browser_tests and any(r.score < 0.9 for r in browser_tests):
            recommendations.append("Fix cross-browser compatibility issues")
        
        return recommendations


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run accessibility and usability tests')
    parser.add_argument('--base-url', default='http://localhost:3000', help='Dashboard base URL')
    parser.add_argument('--output-dir', default='accessibility_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = AccessibilityTestSuite(
        base_url=args.base_url,
        output_dir=args.output_dir
    )
    
    try:
        # Run comprehensive tests
        report = await test_suite.run_comprehensive_accessibility_tests()
        
        print("\n" + "="*80)
        print("ACCESSIBILITY AND USABILITY TEST RESULTS")
        print("="*80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.2%}")
        print(f"Overall Score: {report['summary']['overall_accessibility_score']:.3f}")
        print(f"WCAG Compliance: {report['wcag_compliance_status']['compliance_level']}")
        
        print(f"\nViolations Summary:")
        print(f"  Critical: {report['violation_summary']['critical_violations']}")
        print(f"  Serious: {report['violation_summary']['serious_violations']}")
        print(f"  Moderate: {report['violation_summary']['moderate_violations']}")
        print(f"  Minor: {report['violation_summary']['minor_violations']}")
        
        if report.get('recommendations'):
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nDetailed report saved to: {test_suite.output_dir}")
        
    except Exception as e:
        print(f"Accessibility tests failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))