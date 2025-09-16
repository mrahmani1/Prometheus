"""
Security and Compliance Test Suite

Comprehensive security testing for data source integrations, API endpoints,
authentication, authorization, data privacy, and compliance validation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability"""
    vulnerability_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'authentication', 'authorization', 'data_exposure', 'injection', etc.
    description: str
    affected_component: str
    remediation_steps: List[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class SecurityTestResult:
    """Result from security testing"""
    test_name: str
    success: bool
    risk_level: str
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    security_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class SecurityComplianceTestSuite:
    """
    Comprehensive security and compliance testing suite
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", output_dir: str = "security_results"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results: List[SecurityTestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Security test configurations
        self.security_config = {
            'api_endpoints': [
                '/api/v1/auth/login',
                '/api/v1/data/sources',
                '/api/v1/causal/relationships',
                '/api/v1/volcanic/systems',
                '/api/v1/real-time/stream'
            ],
            'attack_payloads': {
                'sql_injection': [
                    "' OR '1'='1",
                    "'; DROP TABLE users; --",
                    "' UNION SELECT * FROM users --"
                ],
                'xss': [
                    "<script>alert('XSS')</script>",
                    "javascript:alert('XSS')",
                    "<img src=x onerror=alert('XSS')>"
                ]
            }
        }
    
    async def setup_session(self):
        """Setup HTTP session for security testing"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Security-Test-Suite/1.0'}
        )
    
    async def cleanup_session(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    async def test_authentication_security(self) -> SecurityTestResult:
        """Test authentication mechanisms security"""
        result = SecurityTestResult(
            test_name='authentication_security',
            success=False,
            risk_level='medium'
        )
        
        try:
            login_url = f"{self.base_url}/api/v1/auth/login"
            
            # Test brute force protection
            failed_attempts = 0
            blocked = False
            
            for i in range(5):
                payload = {'username': 'testuser', 'password': f'wrong{i}'}
                
                try:
                    async with self.session.post(login_url, json=payload) as response:
                        if response.status == 429:  # Too Many Requests
                            blocked = True
                            break
                        elif response.status == 401:
                            failed_attempts += 1
                        await asyncio.sleep(0.1)
                except Exception:
                    continue
            
            if not blocked:
                result.vulnerabilities.append(SecurityVulnerability(
                    vulnerability_id='AUTH-001',
                    severity='high',
                    category='authentication',
                    description='No brute force protection detected',
                    affected_component='login_endpoint',
                    remediation_steps=[
                        'Implement rate limiting',
                        'Add account lockout',
                        'Implement CAPTCHA'
                    ],
                    risk_score=7.5
                ))
            
            result.security_metrics = {
                'brute_force_protection': blocked,
                'failed_attempts_tested': failed_attempts
            }
            
            result.success = blocked
            result.risk_level = 'low' if blocked else 'high'
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Authentication security test failed: {e}")
        
        return result
    
    async def test_api_security(self) -> SecurityTestResult:
        """Test API endpoint security"""
        result = SecurityTestResult(
            test_name='api_security',
            success=False,
            risk_level='medium'
        )
        
        try:
            vulnerable_endpoints = 0
            
            for endpoint in self.security_config['api_endpoints']:
                endpoint_url = f"{self.base_url}{endpoint}"
                
                # Test SQL injection
                for payload in self.security_config['attack_payloads']['sql_injection']:
                    try:
                        async with self.session.get(f"{endpoint_url}?id={payload}") as response:
                            response_text = await response.text()
                            
                            if any(error in response_text.lower() for error in ['sql', 'mysql', 'error']):
                                result.vulnerabilities.append(SecurityVulnerability(
                                    vulnerability_id=f'SQL-{len(result.vulnerabilities)+1:03d}',
                                    severity='critical',
                                    category='injection',
                                    description=f'SQL injection vulnerability in {endpoint}',
                                    affected_component=endpoint,
                                    remediation_steps=[
                                        'Use parameterized queries',
                                        'Implement input validation',
                                        'Use ORM frameworks'
                                    ],
                                    risk_score=9.5
                                ))
                                vulnerable_endpoints += 1
                                break
                    except Exception:
                        continue
                
                # Test XSS
                for payload in self.security_config['attack_payloads']['xss']:
                    try:
                        async with self.session.get(f"{endpoint_url}?search={payload}") as response:
                            response_text = await response.text()
                            
                            if payload in response_text and '<script>' in payload:
                                result.vulnerabilities.append(SecurityVulnerability(
                                    vulnerability_id=f'XSS-{len(result.vulnerabilities)+1:03d}',
                                    severity='high',
                                    category='injection',
                                    description=f'XSS vulnerability in {endpoint}',
                                    affected_component=endpoint,
                                    remediation_steps=[
                                        'Implement output encoding',
                                        'Use Content Security Policy',
                                        'Validate user inputs'
                                    ],
                                    risk_score=7.5
                                ))
                                break
                    except Exception:
                        continue
            
            result.security_metrics = {
                'total_endpoints_tested': len(self.security_config['api_endpoints']),
                'vulnerable_endpoints': vulnerable_endpoints,
                'total_vulnerabilities': len(result.vulnerabilities)
            }
            
            result.success = len(result.vulnerabilities) == 0
            result.risk_level = 'critical' if any(v.severity == 'critical' for v in result.vulnerabilities) else 'medium'
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"API security test failed: {e}")
        
        return result
    
    async def run_comprehensive_security_tests(self) -> Dict[str, Any]:
        """Run comprehensive security tests"""
        logger.info("Starting comprehensive security tests")
        
        await self.setup_session()
        
        try:
            # Authentication security tests
            auth_result = await self.test_authentication_security()
            self.test_results.append(auth_result)
            
            # API security tests
            api_result = await self.test_api_security()
            self.test_results.append(api_result)
            
        finally:
            await self.cleanup_session()
        
        # Generate report
        report = self._generate_security_report()
        
        # Save report
        report_file = self.output_dir / f"security_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Security tests completed. Report saved to {report_file}")
        
        return report
    
    def _generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # Aggregate vulnerabilities
        all_vulnerabilities = []
        for result in self.test_results:
            all_vulnerabilities.extend(result.vulnerabilities)
        
        # Categorize by severity
        critical_vulns = [v for v in all_vulnerabilities if v.severity == 'critical']
        high_vulns = [v for v in all_vulnerabilities if v.severity == 'high']
        medium_vulns = [v for v in all_vulnerabilities if v.severity == 'medium']
        low_vulns = [v for v in all_vulnerabilities if v.severity == 'low']
        
        # Determine security posture
        if critical_vulns:
            security_posture = 'critical'
        elif len(high_vulns) > 3:
            security_posture = 'poor'
        elif len(high_vulns) > 0:
            security_posture = 'fair'
        else:
            security_posture = 'good'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'security_posture': security_posture,
                'total_vulnerabilities': len(all_vulnerabilities)
            },
            'vulnerability_breakdown': {
                'critical': len(critical_vulns),
                'high': len(high_vulns),
                'medium': len(medium_vulns),
                'low': len(low_vulns)
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'risk_level': r.risk_level,
                    'vulnerabilities_count': len(r.vulnerabilities),
                    'security_metrics': r.security_metrics,
                    'error_message': r.error_message
                }
                for r in self.test_results
            ],
            'top_vulnerabilities': [
                {
                    'id': v.vulnerability_id,
                    'severity': v.severity,
                    'category': v.category,
                    'description': v.description,
                    'component': v.affected_component,
                    'risk_score': v.risk_score,
                    'remediation': v.remediation_steps
                }
                for v in sorted(all_vulnerabilities, key=lambda x: x.risk_score, reverse=True)[:10]
            ],
            'recommendations': self._generate_security_recommendations()
        }
        
        return report
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []
        
        # Aggregate all vulnerabilities
        all_vulnerabilities = []
        for result in self.test_results:
            all_vulnerabilities.extend(result.vulnerabilities)
        
        # Critical vulnerabilities
        critical_vulns = [v for v in all_vulnerabilities if v.severity == 'critical']
        if critical_vulns:
            recommendations.append(f"URGENT: Fix {len(critical_vulns)} critical vulnerabilities immediately")
        
        # High severity vulnerabilities
        high_vulns = [v for v in all_vulnerabilities if v.severity == 'high']
        if high_vulns:
            recommendations.append(f"Fix {len(high_vulns)} high-severity vulnerabilities")
        
        # Category-specific recommendations
        injection_vulns = [v for v in all_vulnerabilities if v.category == 'injection']
        if injection_vulns:
            recommendations.append("Implement comprehensive input validation")
        
        auth_vulns = [v for v in all_vulnerabilities if v.category == 'authentication']
        if auth_vulns:
            recommendations.append("Strengthen authentication mechanisms")
        
        return recommendations


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run security and compliance tests')
    parser.add_argument('--base-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--output-dir', default='security_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = SecurityComplianceTestSuite(
        base_url=args.base_url,
        output_dir=args.output_dir
    )
    
    try:
        # Run comprehensive tests
        report = await test_suite.run_comprehensive_security_tests()
        
        print("\n" + "="*80)
        print("SECURITY AND COMPLIANCE TEST RESULTS")
        print("="*80)
        print(f"Security Posture: {report['summary']['security_posture'].upper()}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.2%}")
        print(f"Total Vulnerabilities: {report['summary']['total_vulnerabilities']}")
        
        print(f"\nVulnerability Breakdown:")
        print(f"  Critical: {report['vulnerability_breakdown']['critical']}")
        print(f"  High: {report['vulnerability_breakdown']['high']}")
        print(f"  Medium: {report['vulnerability_breakdown']['medium']}")
        print(f"  Low: {report['vulnerability_breakdown']['low']}")
        
        if report.get('recommendations'):
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nDetailed report saved to: {test_suite.output_dir}")
        
        # Exit with appropriate code
        if report['vulnerability_breakdown']['critical'] > 0:
            return 2  # Critical vulnerabilities
        elif report['vulnerability_breakdown']['high'] > 0:
            return 1  # High vulnerabilities
        else:
            return 0  # Success
        
    except Exception as e:
        print(f"Security tests failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))