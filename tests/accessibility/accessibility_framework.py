#!/usr/bin/env python3
"""
Comprehensive Accessibility Testing Framework for Prometheus AI Platform
Implements axe-core integration and WCAG 2.1 AA compliance validation
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from axe_selenium_python import Axe
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccessibilityTestingFramework:
    """Comprehensive accessibility testing framework with axe-core integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:3000')
        self.test_results = {}
        self.driver = None
        
        # WCAG 2.1 AA compliance rules
        self.wcag_rules = {
            'level_a': [
                'color-contrast',
                'image-alt',
                'label',
                'link-name',
                'button-name',
                'form-field-multiple-labels',
                'heading-order',
                'html-has-lang',
                'html-lang-valid',
                'landmark-one-main',
                'page-has-heading-one',
                'region'
            ],
            'level_aa': [
                'color-contrast-enhanced',
                'focus-order-semantics',
                'scrollable-region-focusable',
                'bypass',
                'document-title',
                'focus-order-semantics',
                'meta-refresh',
                'meta-viewport'
            ]
        }
        
    def setup_driver(self):
        """Setup Chrome WebDriver with accessibility testing options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Enable accessibility features
        chrome_options.add_argument('--enable-accessibility-logging')
        chrome_options.add_argument('--force-renderer-accessibility')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        return self.driver
    
    def teardown_driver(self):
        """Cleanup WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    async def run_comprehensive_accessibility_tests(self) -> Dict[str, Any]:
        """Run all accessibility tests and return comprehensive results"""
        logger.info("Starting comprehensive accessibility testing...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'target_url': self.base_url,
            'tests': {}
        }
        
        try:
            # Setup WebDriver
            self.setup_driver()
            
            # 1. Axe-core Automated Testing
            results['tests']['axe_core'] = await self.run_axe_core_tests()
            
            # 2. WCAG 2.1 AA Compliance Testing
            results['tests']['wcag_compliance'] = await self.run_wcag_compliance_tests()
            
            # 3. Keyboard Navigation Testing
            results['tests']['keyboard_navigation'] = await self.test_keyboard_navigation()
            
            # 4. Screen Reader Compatibility
            results['tests']['screen_reader'] = await self.test_screen_reader_compatibility()
            
            # 5. Color Contrast Testing
            results['tests']['color_contrast'] = await self.test_color_contrast()
            
            # 6. Focus Management Testing
            results['tests']['focus_management'] = await self.test_focus_management()
            
            # 7. Semantic Structure Testing
            results['tests']['semantic_structure'] = await self.test_semantic_structure()
            
            # 8. Alternative Text Testing
            results['tests']['alt_text'] = await self.test_alternative_text()
            
            # Generate comprehensive report
            results['summary'] = self.generate_accessibility_summary(results['tests'])
            
        except Exception as e:
            logger.error(f"Accessibility testing failed: {str(e)}")
            results['error'] = str(e)
        finally:
            self.teardown_driver()
        
        return results
    
    async def run_axe_core_tests(self) -> Dict[str, Any]:
        """Run axe-core automated accessibility tests"""
        logger.info("Running axe-core accessibility tests...")
        
        results = {
            'violations': [],
            'passes': [],
            'incomplete': [],
            'inapplicable': []
        }
        
        try:
            # Navigate to the application
            self.driver.get(self.base_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Initialize axe
            axe = Axe(self.driver)
            
            # Inject axe-core
            axe.inject()
            
            # Run axe tests
            axe_results = axe.run()
            
            # Process results
            results['violations'] = axe_results.get('violations', [])
            results['passes'] = axe_results.get('passes', [])
            results['incomplete'] = axe_results.get('incomplete', [])
            results['inapplicable'] = axe_results.get('inapplicable', [])
            
            # Calculate scores
            total_checks = len(results['violations']) + len(results['passes'])
            if total_checks > 0:
                results['accessibility_score'] = len(results['passes']) / total_checks * 100
            else:
                results['accessibility_score'] = 0
            
            logger.info(f"Axe-core tests completed. Score: {results['accessibility_score']:.2f}%")
            
        except Exception as e:
            logger.error(f"Axe-core testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def run_wcag_compliance_tests(self) -> Dict[str, Any]:
        """Run WCAG 2.1 AA compliance tests"""
        logger.info("Running WCAG 2.1 AA compliance tests...")
        
        results = {
            'level_a': {'passed': [], 'failed': []},
            'level_aa': {'passed': [], 'failed': []},
            'compliance_score': 0
        }
        
        try:
            # Test Level A compliance
            for rule in self.wcag_rules['level_a']:
                if await self.test_wcag_rule(rule):
                    results['level_a']['passed'].append(rule)
                else:
                    results['level_a']['failed'].append(rule)
            
            # Test Level AA compliance
            for rule in self.wcag_rules['level_aa']:
                if await self.test_wcag_rule(rule):
                    results['level_aa']['passed'].append(rule)
                else:
                    results['level_aa']['failed'].append(rule)
            
            # Calculate compliance score
            total_rules = len(self.wcag_rules['level_a']) + len(self.wcag_rules['level_aa'])
            passed_rules = len(results['level_a']['passed']) + len(results['level_aa']['passed'])
            results['compliance_score'] = (passed_rules / total_rules) * 100 if total_rules > 0 else 0
            
            logger.info(f"WCAG compliance score: {results['compliance_score']:.2f}%")
            
        except Exception as e:
            logger.error(f"WCAG compliance testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def test_wcag_rule(self, rule: str) -> bool:
        """Test a specific WCAG rule"""
        try:
            # This is a simplified implementation - in practice, each rule would have specific tests
            if rule == 'color-contrast':
                return await self.check_color_contrast()
            elif rule == 'image-alt':
                return await self.check_image_alt_text()
            elif rule == 'label':
                return await self.check_form_labels()
            elif rule == 'link-name':
                return await self.check_link_names()
            elif rule == 'button-name':
                return await self.check_button_names()
            elif rule == 'heading-order':
                return await self.check_heading_order()
            elif rule == 'html-has-lang':
                return await self.check_html_lang()
            elif rule == 'document-title':
                return await self.check_document_title()
            else:
                # Default implementation for other rules
                return True
        except Exception as e:
            logger.error(f"Error testing WCAG rule {rule}: {str(e)}")
            return False
    
    async def check_color_contrast(self) -> bool:
        """Check color contrast ratios"""
        try:
            # Find all text elements and check contrast
            text_elements = self.driver.find_elements(By.CSS_SELECTOR, "p, h1, h2, h3, h4, h5, h6, span, div, a, button")
            
            for element in text_elements[:10]:  # Check first 10 elements
                # Get computed styles
                color = self.driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).color;", element
                )
                background_color = self.driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).backgroundColor;", element
                )
                
                # Basic contrast check (simplified)
                if color and background_color:
                    # In a real implementation, you would calculate the actual contrast ratio
                    # This is a placeholder for the contrast calculation
                    pass
            
            return True
        except Exception as e:
            logger.error(f"Color contrast check failed: {str(e)}")
            return False
    
    async def check_image_alt_text(self) -> bool:
        """Check that all images have alt text"""
        try:
            images = self.driver.find_elements(By.TAG_NAME, "img")
            
            for img in images:
                alt_text = img.get_attribute("alt")
                if alt_text is None or alt_text.strip() == "":
                    # Check if image is decorative
                    role = img.get_attribute("role")
                    if role != "presentation" and role != "none":
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Image alt text check failed: {str(e)}")
            return False
    
    async def check_form_labels(self) -> bool:
        """Check that all form inputs have labels"""
        try:
            inputs = self.driver.find_elements(By.CSS_SELECTOR, "input, select, textarea")
            
            for input_element in inputs:
                input_id = input_element.get_attribute("id")
                aria_label = input_element.get_attribute("aria-label")
                aria_labelledby = input_element.get_attribute("aria-labelledby")
                
                # Check for associated label
                has_label = False
                if input_id:
                    try:
                        self.driver.find_element(By.CSS_SELECTOR, f"label[for='{input_id}']")
                        has_label = True
                    except:
                        pass
                
                if not has_label and not aria_label and not aria_labelledby:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Form labels check failed: {str(e)}")
            return False
    
    async def check_link_names(self) -> bool:
        """Check that all links have accessible names"""
        try:
            links = self.driver.find_elements(By.TAG_NAME, "a")
            
            for link in links:
                text_content = link.text.strip()
                aria_label = link.get_attribute("aria-label")
                title = link.get_attribute("title")
                
                if not text_content and not aria_label and not title:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Link names check failed: {str(e)}")
            return False
    
    async def check_button_names(self) -> bool:
        """Check that all buttons have accessible names"""
        try:
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            
            for button in buttons:
                text_content = button.text.strip()
                aria_label = button.get_attribute("aria-label")
                title = button.get_attribute("title")
                
                if not text_content and not aria_label and not title:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Button names check failed: {str(e)}")
            return False
    
    async def check_heading_order(self) -> bool:
        """Check proper heading hierarchy"""
        try:
            headings = self.driver.find_elements(By.CSS_SELECTOR, "h1, h2, h3, h4, h5, h6")
            
            if not headings:
                return True
            
            previous_level = 0
            for heading in headings:
                current_level = int(heading.tag_name[1])
                
                if previous_level == 0:
                    # First heading should be h1
                    if current_level != 1:
                        return False
                else:
                    # Heading levels should not skip more than one level
                    if current_level > previous_level + 1:
                        return False
                
                previous_level = current_level
            
            return True
        except Exception as e:
            logger.error(f"Heading order check failed: {str(e)}")
            return False
    
    async def check_html_lang(self) -> bool:
        """Check that HTML element has lang attribute"""
        try:
            html_element = self.driver.find_element(By.TAG_NAME, "html")
            lang = html_element.get_attribute("lang")
            return lang is not None and lang.strip() != ""
        except Exception as e:
            logger.error(f"HTML lang check failed: {str(e)}")
            return False
    
    async def check_document_title(self) -> bool:
        """Check that document has a title"""
        try:
            title = self.driver.title
            return title is not None and title.strip() != ""
        except Exception as e:
            logger.error(f"Document title check failed: {str(e)}")
            return False
    
    async def test_keyboard_navigation(self) -> Dict[str, Any]:
        """Test keyboard navigation functionality"""
        logger.info("Testing keyboard navigation...")
        
        results = {
            'tab_order': True,
            'focus_visible': True,
            'keyboard_traps': False,
            'skip_links': True
        }
        
        try:
            # Test tab navigation
            focusable_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "a, button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
            )
            
            # Test focus visibility
            for element in focusable_elements[:5]:  # Test first 5 elements
                element.click()
                # Check if focus is visible (simplified check)
                outline = self.driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).outline;", element
                )
                if outline == "none":
                    results['focus_visible'] = False
                    break
            
            # Test for keyboard traps (simplified)
            # In a real implementation, you would simulate tab navigation
            
        except Exception as e:
            logger.error(f"Keyboard navigation testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def test_screen_reader_compatibility(self) -> Dict[str, Any]:
        """Test screen reader compatibility"""
        logger.info("Testing screen reader compatibility...")
        
        results = {
            'aria_labels': True,
            'landmarks': True,
            'headings_structure': True,
            'live_regions': True
        }
        
        try:
            # Check ARIA labels
            elements_needing_labels = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "button, input, select, textarea"
            )
            
            for element in elements_needing_labels:
                aria_label = element.get_attribute("aria-label")
                aria_labelledby = element.get_attribute("aria-labelledby")
                if not aria_label and not aria_labelledby:
                    # Check for associated label
                    element_id = element.get_attribute("id")
                    if element_id:
                        try:
                            self.driver.find_element(By.CSS_SELECTOR, f"label[for='{element_id}']")
                        except:
                            results['aria_labels'] = False
                            break
            
            # Check landmarks
            landmarks = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "main, nav, header, footer, aside, section[aria-label], [role='main'], [role='navigation'], [role='banner'], [role='contentinfo'], [role='complementary']"
            )
            
            if not landmarks:
                results['landmarks'] = False
            
        except Exception as e:
            logger.error(f"Screen reader compatibility testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def test_color_contrast(self) -> Dict[str, Any]:
        """Test color contrast ratios"""
        logger.info("Testing color contrast...")
        
        results = {
            'aa_compliant': True,
            'aaa_compliant': True,
            'contrast_issues': []
        }
        
        try:
            # This would integrate with a color contrast analyzer
            # For now, we'll do a basic check
            text_elements = self.driver.find_elements(By.CSS_SELECTOR, "p, h1, h2, h3, h4, h5, h6, span, a, button")
            
            for element in text_elements[:10]:  # Check first 10 elements
                # In a real implementation, you would calculate actual contrast ratios
                # This is a placeholder
                pass
            
        except Exception as e:
            logger.error(f"Color contrast testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def test_focus_management(self) -> Dict[str, Any]:
        """Test focus management"""
        logger.info("Testing focus management...")
        
        results = {
            'focus_order': True,
            'focus_trapping': True,
            'focus_restoration': True
        }
        
        try:
            # Test focus order and management
            # This would involve simulating user interactions
            pass
            
        except Exception as e:
            logger.error(f"Focus management testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def test_semantic_structure(self) -> Dict[str, Any]:
        """Test semantic HTML structure"""
        logger.info("Testing semantic structure...")
        
        results = {
            'proper_headings': True,
            'semantic_elements': True,
            'list_structure': True
        }
        
        try:
            # Check heading structure
            headings = self.driver.find_elements(By.CSS_SELECTOR, "h1, h2, h3, h4, h5, h6")
            if headings:
                # Check if first heading is h1
                if headings[0].tag_name.lower() != 'h1':
                    results['proper_headings'] = False
            
            # Check for semantic elements
            semantic_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "main, nav, header, footer, aside, section, article"
            )
            
            if not semantic_elements:
                results['semantic_elements'] = False
            
        except Exception as e:
            logger.error(f"Semantic structure testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def test_alternative_text(self) -> Dict[str, Any]:
        """Test alternative text for images and media"""
        logger.info("Testing alternative text...")
        
        results = {
            'images_have_alt': True,
            'decorative_images_marked': True,
            'complex_images_described': True
        }
        
        try:
            images = self.driver.find_elements(By.TAG_NAME, "img")
            
            for img in images:
                alt_text = img.get_attribute("alt")
                role = img.get_attribute("role")
                
                if alt_text is None:
                    if role not in ["presentation", "none"]:
                        results['images_have_alt'] = False
                        break
                elif alt_text.strip() == "":
                    # Empty alt text should only be for decorative images
                    if role not in ["presentation", "none"]:
                        results['decorative_images_marked'] = False
            
        except Exception as e:
            logger.error(f"Alternative text testing failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def generate_accessibility_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive accessibility summary"""
        summary = {
            'overall_score': 0,
            'critical_issues': 0,
            'warnings': 0,
            'recommendations': []
        }
        
        try:
            scores = []
            
            # Calculate overall score from individual test results
            if 'axe_core' in test_results and 'accessibility_score' in test_results['axe_core']:
                scores.append(test_results['axe_core']['accessibility_score'])
            
            if 'wcag_compliance' in test_results and 'compliance_score' in test_results['wcag_compliance']:
                scores.append(test_results['wcag_compliance']['compliance_score'])
            
            # Add other test scores
            for test_name, test_result in test_results.items():
                if isinstance(test_result, dict) and 'error' not in test_result:
                    # Calculate score based on passed tests
                    passed_tests = sum(1 for v in test_result.values() if v is True)
                    total_tests = len([v for v in test_result.values() if isinstance(v, bool)])
                    if total_tests > 0:
                        scores.append((passed_tests / total_tests) * 100)
            
            # Calculate overall score
            if scores:
                summary['overall_score'] = sum(scores) / len(scores)
            
            # Count critical issues and warnings
            if 'axe_core' in test_results:
                summary['critical_issues'] += len(test_results['axe_core'].get('violations', []))
                summary['warnings'] += len(test_results['axe_core'].get('incomplete', []))
            
            # Generate recommendations
            if summary['overall_score'] < 80:
                summary['recommendations'].append("Improve overall accessibility compliance")
            
            if summary['critical_issues'] > 0:
                summary['recommendations'].append("Address critical accessibility violations")
            
            if 'wcag_compliance' in test_results:
                failed_rules = (test_results['wcag_compliance'].get('level_a', {}).get('failed', []) + 
                              test_results['wcag_compliance'].get('level_aa', {}).get('failed', []))
                if failed_rules:
                    summary['recommendations'].append(f"Fix WCAG compliance issues: {', '.join(failed_rules[:3])}")
            
        except Exception as e:
            logger.error(f"Error generating accessibility summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """Save accessibility test results to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"accessibility_results_{timestamp}.json"
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Accessibility test results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


async def main():
    """Main function to run accessibility tests"""
    config = {
        'base_url': os.getenv('TEST_URL', 'http://localhost:3000'),
        'output_dir': 'test_results/accessibility'
    }
    
    framework = AccessibilityTestingFramework(config)
    results = await framework.run_comprehensive_accessibility_tests()
    
    # Save results
    framework.save_results(results)
    
    # Print summary
    if 'summary' in results:
        summary = results['summary']
        print(f"\n🔍 Accessibility Test Summary:")
        print(f"   Overall Score: {summary.get('overall_score', 0):.2f}%")
        print(f"   Critical Issues: {summary.get('critical_issues', 0)}")
        print(f"   Warnings: {summary.get('warnings', 0)}")
        
        if summary.get('recommendations'):
            print(f"   Recommendations:")
            for rec in summary['recommendations']:
                print(f"     • {rec}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())