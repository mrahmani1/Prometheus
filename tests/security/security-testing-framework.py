#!/usr/bin/env python3
"""
Comprehensive Security Testing Framework for Prometheus AI Platform
Implements OWASP ZAP integration, penetration testing, and security validation
"""

import asyncio
import json
import logging
import os
import ssl
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from zapv2 import ZAPv2
import jwt
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import socket
import nmap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTestingFramework:
    """Comprehensive security testing framework with OWASP ZAP integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:3000')
        self.zap_proxy = config.get('zap_proxy', 'http://127.0.0.1:8080')
        self.api_key = config.get('api_key', '')
        self.zap = None
        self.test_results = {}
        
    def initialize_zap(self):
        """Initialize OWASP ZAP connection"""
        try:
            self.zap = ZAPv2(proxies={'http': self.zap_proxy, 'https': self.zap_proxy})
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ZAP: {str(e)}")
            return False

    async def ensure_zap_running(self):
        """Ensure OWASP ZAP is running"""
        logger.info("Starting OWASP ZAP...")
        
        # Try to connect to ZAP
        try:
            self.zap.core.version
        except Exception:
            # Start ZAP if not running
            try:
                subprocess.Popen([
                    'zap.sh',
                    '-daemon',
                    '-port', '8080',
                    '-config', 'api.addrs.addr.name=.*',
                    '-config', 'api.addrs.addr.regex=true'
                ])
                
                # Wait for ZAP to start
                for _ in range(30):
                    try:
                        self.zap.core.version
                        break
                    except Exception:
                        await asyncio.sleep(2)
                        
            except Exception as e:
                logger.error(f"Failed to start ZAP: {str(e)}")

    async def run_comprehensive_security_tests(self) -> Dict[str, Any]:
        """Run all security tests and return comprehensive results"""
        logger.info("Running comprehensive security tests...")
        
        security_results = {
            'timestamp': datetime.now().isoformat(),
            'target_url': self.base_url,
            'tests': {}
        }
        
        try:
            # Initialize ZAP
            if not self.initialize_zap():
                return {'error': 'Failed to initialize OWASP ZAP'}
            
            # Ensure ZAP is running
            await self.ensure_zap_running()
            
            # 1. Authentication and Authorization Testing
            security_results['tests']['auth_testing'] = await self.test_authentication()
            
            # 2. Input Validation Testing
            security_results['tests']['input_validation'] = await self.test_input_validation()
            
            # 3. SSL/TLS Configuration Testing
            security_results['tests']['ssl_tls'] = await self.run_ssl_tls_tests()
            
            # 4. Network Security Testing
            security_results['tests']['network_security'] = await self.run_network_security_tests()
            
            # 5. Privacy and GDPR Compliance Testing
            security_results['tests']['privacy_compliance'] = await self.run_privacy_compliance_tests()
            
            # 6. OWASP ZAP Vulnerability Scanning
            security_results['tests']['zap_scan'] = await self.run_zap_vulnerability_scan()
            
            # Generate security summary
            security_results['summary'] = self.generate_security_summary(security_results['tests'])
            
        except Exception as e:
            logger.error(f"Security testing failed: {str(e)}")
            security_results['error'] = str(e)
        
        return security_results

    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication and authorization mechanisms"""
        logger.info("Testing authentication mechanisms...")
        
        results = {
            'brute_force_protection': False,
            'jwt_validation': False,
            'session_management': False,
            'status': 'completed'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test brute force protection
                login_url = f"{self.base_url}/api/v1/auth/login"
                
                # Attempt multiple failed logins
                for i in range(5):
                    async with session.post(login_url, json={
                        'username': 'testuser',
                        'password': f'wrongpassword{i}'
                    }, timeout=10) as response:
                        if response.status == 429:  # Rate limited
                            results['brute_force_protection'] = True
                            break
                
                # Test JWT validation
                async with session.post(login_url, json={
                    'username': 'testuser',
                    'password': 'correctpassword'
                }, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'token' in data:
                            results['jwt_validation'] = True
                
                # Test session management
                async with session.get(f"{self.base_url}/api/v1/user/profile", timeout=10) as response:
                    if response.status in [401, 403]:  # Unauthorized without token
                        results['session_management'] = True
                        
        except Exception as e:
            logger.error(f"Authentication testing failed: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'failed'
        
        return results

    async def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization"""
        logger.info("Testing input validation...")
        
        results = {
            'xss_protection': False,
            'sql_injection_protection': False,
            'file_upload_security': False,
            'status': 'completed'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test XSS protection
                xss_payloads = [
                    '<script>alert(1)</script>',
                    'javascript:alert(1)',
                    '<img src=x onerror=alert(1)>'
                ]
                
                for payload in xss_payloads:
                    async with session.post(f"{self.base_url}/api/v1/search", json={
                        'query': payload
                    }, timeout=10) as response:
                        response_text = await response.text()
                        if payload not in response_text:
                            results['xss_protection'] = True
                            break
                
                # Test SQL injection protection
                sql_payloads = [
                    "'; DROP TABLE users; --",
                    "' OR '1'='1",
                    "1' UNION SELECT * FROM users --"
                ]
                
                for payload in sql_payloads:
                    async with session.get(f"{self.base_url}/api/v1/data?id={payload}", timeout=10) as response:
                        if response.status != 500:  # Should not cause server error
                            results['sql_injection_protection'] = True
                
                # Test file upload security
                malicious_files = [
                    {'filename': 'test.php', 'content': '<?php system($_GET["cmd"]); ?>'},
                    {'filename': 'test.jsp', 'content': '<% Runtime.getRuntime().exec(request.getParameter("cmd")); %>'},
                    {'filename': '../../etc/passwd', 'content': 'root:x:0:0:root:/root:/bin/bash'}
                ]
                
                for file_data in malicious_files:
                    data = aiohttp.FormData()
                    data.add_field('file', file_data['content'], filename=file_data['filename'])
                    
                    async with session.post(f"{self.base_url}/api/v1/upload", data=data, timeout=10) as response:
                        if response.status != 200:  # Should reject malicious files
                            results['file_upload_security'] = True
                            break
                        
        except Exception as e:
            logger.error(f"Input validation testing failed: {str(e)}")
            results['error'] = str(e)
            results['status'] = 'failed'
        
        return results

    async def run_ssl_tls_tests(self) -> Dict[str, Any]:
        """Run SSL/TLS configuration tests"""
        logger.info("Running SSL/TLS configuration tests...")
        
        if not self.base_url.startswith('https://'):
            return {
                'status': 'skipped',
                'reason': 'HTTPS not configured'
            }
        
        ssl_results = {
            'certificate_validation': await self.test_certificate_validation(),
            'protocol_version_check': await self.test_ssl_protocols(),
            'cipher_suite_analysis': await self.test_cipher_suites(),
            'hsts_header_check': await self.test_hsts_headers()
        }
        
        return ssl_results

    async def test_certificate_validation(self) -> Dict[str, Any]:
        """Test SSL certificate validation"""
        logger.info("Testing SSL certificate...")
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 443
        
        try:
            # Get certificate
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
            
            # Parse certificate
            cert = x509.load_der_x509_certificate(cert_der, default_backend())
            
            # Check certificate validity
            now = datetime.now()
            is_valid = cert.not_valid_before <= now < cert.not_valid_after
            
            # Check subject alternative names
            try:
                san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                san_names = [name.value for name in san_ext.value if name.name in san_ext.value]
            except x509.ExtensionNotFound:
                san_names = []
            
            return {
                'status': 'completed',
                'valid': is_valid,
                'not_before': cert.not_valid_before.isoformat(),
                'not_after': cert.not_valid_after.isoformat(),
                'subject': str(cert.subject),
                'issuer': str(cert.issuer),
                'san_names': san_names
            }
            
        except Exception as e:
            logger.error(f"Certificate validation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def test_ssl_protocols(self) -> Dict[str, Any]:
        """Test SSL/TLS protocol versions"""
        logger.info("Testing SSL/TLS protocols...")
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 443
        
        protocols_to_test = [
            ('SSLv2', ssl.PROTOCOL_SSLv23),
            ('SSLv3', ssl.PROTOCOL_SSLv23),
            ('TLSv1.0', ssl.PROTOCOL_TLSv1),
            ('TLSv1.1', ssl.PROTOCOL_TLSv1_1),
            ('TLSv1.2', ssl.PROTOCOL_TLSv1_2)
        ]
        
        supported_protocols = []
        
        for protocol_name, protocol_version in protocols_to_test:
            try:
                context = ssl.SSLContext(protocol_version)
                with socket.create_connection((hostname, port), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        supported_protocols.append({
                            'protocol': protocol_name,
                            'supported': True,
                            'version': ssock.version()
                        })
            except Exception:
                supported_protocols.append({
                    'protocol': protocol_name,
                    'supported': False
                })
        
        return {
            'status': 'completed',
            'supported_protocols': supported_protocols
        }

    async def test_cipher_suites(self) -> Dict[str, Any]:
        """Test SSL cipher suites"""
        logger.info("Testing SSL cipher suites...")
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 443
        
        try:
            # Use nmap to test SSL ciphers
            nm = nmap.PortScanner()
            result = nm.scan(hostname, str(port), arguments='--script ssl-enum-ciphers')
            
            cipher_info = result['scan'][hostname]['tcp'][port].get('script', {})
            
            return {
                'status': 'completed',
                'cipher_info': cipher_info
            }
            
        except Exception as e:
            logger.error(f"Cipher suite testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def test_hsts_headers(self) -> Dict[str, Any]:
        """Test HSTS headers"""
        logger.info("Testing HSTS headers...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, timeout=10) as response:
                    hsts_header = response.headers.get('Strict-Transport-Security')
                    
                    return {
                        'status': 'completed',
                        'hsts_present': hsts_header is not None,
                        'hsts_header': hsts_header
                    }
                    
        except Exception as e:
            logger.error(f"HSTS header testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def run_network_security_tests(self) -> Dict[str, Any]:
        """Run network security tests"""
        logger.info("Running network security tests...")
        
        network_results = {
            'port_scan': await self.perform_port_scan(),
            'service_enumeration': await self.enumerate_services(),
            'firewall_testing': await self.test_firewall_rules()
        }
        
        return network_results

    async def perform_port_scan(self) -> Dict[str, Any]:
        """Perform port scan"""
        logger.info("Performing port scan...")
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        
        try:
            nm = nmap.PortScanner()
            result = nm.scan(hostname, '1-1000', arguments='-sS')
            
            open_ports = []
            for host in result['scan']:
                for port in result['scan'][host]['tcp']:
                    if result['scan'][host]['tcp'][port]['state'] == 'open':
                        open_ports.append({
                            'port': port,
                            'service': result['scan'][host]['tcp'][port].get('name', 'unknown'),
                            'version': result['scan'][host]['tcp'][port].get('version', 'unknown')
                        })
            
            return {
                'status': 'completed',
                'open_ports': open_ports,
                'open_ports_count': len(open_ports)
            }
            
        except Exception as e:
            logger.error(f"Port scan failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def enumerate_services(self) -> Dict[str, Any]:
        """Enumerate running services"""
        logger.info("Enumerating services...")
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        
        try:
            nm = nmap.PortScanner()
            result = nm.scan(hostname, arguments='-sV')
            
            services = []
            for host in result['scan']:
                for port in result['scan'][host]['tcp']:
                    port_info = result['scan'][host]['tcp'][port]
                    if port_info['state'] == 'open':
                        services.append({
                            'port': port,
                            'service': port_info.get('name', 'unknown'),
                            'product': port_info.get('product', 'unknown'),
                            'version': port_info.get('version', 'unknown')
                        })
            
            return {
                'status': 'completed',
                'services': services,
                'services_found': len(services)
            }
            
        except Exception as e:
            logger.error(f"Service enumeration failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def test_firewall_rules(self) -> Dict[str, Any]:
        """Test firewall rules"""
        logger.info("Testing firewall rules...")
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        
        # Test access to common administrative ports
        admin_ports = [22, 3389, 5432, 27017, 6379, 9200]
        accessible_ports = []
        
        for port in admin_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((hostname, port))
                sock.close()
                
                if result == 0:
                    accessible_ports.append(port)
                    
            except Exception:
                pass
        
        return {
            'status': 'completed',
            'accessible_ports': accessible_ports,
            'ports_found': len(accessible_ports)
        }

    async def run_privacy_compliance_tests(self) -> Dict[str, Any]:
        """Run privacy and GDPR compliance tests"""
        logger.info("Running privacy and GDPR compliance tests...")
        
        privacy_results = {
            'cookie_compliance': await self.test_cookie_compliance(),
            'privacy_policy': await self.check_privacy_policy(),
            'data_retention_policy': await self.test_data_retention(),
            'data_subject_rights': await self.test_data_subject_rights(),
            'consent_mechanisms': await self.test_consent_mechanisms(),
            'data_collection_audit': await self.audit_data_collection()
        }
        
        return privacy_results

    async def test_cookie_compliance(self) -> Dict[str, Any]:
        """Test cookie compliance"""
        logger.info("Testing cookie compliance...")
        
        cookie_issues = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    cookies = response.cookies
                    
                    for cookie in cookies:
                        # Check for secure flag on HTTPS
                        if self.base_url.startswith('https://') and not cookie.get('secure'):
                            cookie_issues.append({
                                'cookie': cookie,
                                'issue': 'Missing secure flag',
                                'risk': 'medium'
                            })
                        
                        # Check for HttpOnly flag
                        if not cookie.get('httponly'):
                            cookie_issues.append({
                                'cookie': cookie,
                                'issue': 'Missing HttpOnly flag',
                                'risk': 'medium'
                            })
                        
                        # Check for SameSite attribute
                        if not cookie.get('samesite'):
                            cookie_issues.append({
                                'cookie': cookie,
                                'issue': 'Missing SameSite attribute',
                                'risk': 'low'
                            })
            
            return {
                'status': 'completed',
                'cookie_issues': cookie_issues,
                'issues_count': len(cookie_issues)
            }
            
        except Exception as e:
            logger.error(f"Cookie compliance testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def check_privacy_policy(self) -> Dict[str, Any]:
        """Check privacy policy availability"""
        logger.info("Checking privacy policy...")
        
        privacy_urls = [
            '/privacy',
            '/privacy-policy',
            '/legal/privacy',
            '/api/v1/privacy'
        ]
        
        policy_found = False
        policy_url = None
        
        try:
            async with aiohttp.ClientSession() as session:
                for url in privacy_urls:
                    async with session.get(f"{self.base_url}{url}") as response:
                        if response.status == 200:
                            policy_found = True
                            policy_url = url
                            break
            
            return {
                'status': 'completed',
                'policy_found': policy_found,
                'policy_url': policy_url
            }
            
        except Exception as e:
            logger.error(f"Privacy policy check failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def test_data_retention(self) -> Dict[str, Any]:
        """Test data retention policies"""
        logger.info("Testing data retention policies...")
        
        retention_issues = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check if retention policy endpoint exists
                async with session.get(f"{self.base_url}/api/v1/privacy/retention") as response:
                    if response.status == 404:
                        retention_issues.append({
                            'issue': 'No data retention policy endpoint defined',
                            'risk': 'medium'
                        })
                    else:
                        policy_data = await response.json()
                        if not policy_data.get('retention_periods'):
                            retention_issues.append({
                                'issue': 'No defined retention periods',
                                'risk': 'medium'
                            })
            
            return {
                'status': 'completed',
                'retention_issues': retention_issues,
                'issues_count': len(retention_issues)
            }
            
        except Exception as e:
            logger.error(f"Data retention testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def test_data_subject_rights(self) -> Dict[str, Any]:
        """Test data subject rights implementation"""
        logger.info("Testing data subject rights...")
        
        rights_endpoints = [
            '/api/v1/privacy/access',      # Right to access
            '/api/v1/privacy/rectification', # Right to rectification
            '/api/v1/privacy/erasure',     # Right to erasure
            '/api/v1/privacy/portability', # Right to data portability
            '/api/v1/privacy/object'       # Right to object
        ]
        
        missing_rights = []
        implemented_rights = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in rights_endpoints:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        if response.status != 404:
                            implemented_rights.append(endpoint)
                        else:
                            missing_rights.append(endpoint)
            
            return {
                'status': 'completed',
                'implemented_rights': implemented_rights,
                'missing_rights': missing_rights,
                'rights_count': len(implemented_rights)
            }
            
        except Exception as e:
            logger.error(f"Data subject rights testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def test_consent_mechanisms(self) -> Dict[str, Any]:
        """Test consent mechanisms"""
        logger.info("Testing consent mechanisms...")
        
        consent_issues = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check if consent is required for data processing
                async with session.get(f"{self.base_url}/api/v1/privacy/consent") as response:
                    if response.status == 404:
                        consent_issues.append({
                            'issue': 'No consent mechanism found',
                            'risk': 'critical'
                        })
                
                # Test data processing without consent
                user_data = {
                    'email': 'test@example.com',
                    'name': 'Test User'
                }
                
                async with session.post(f"{self.base_url}/api/v1/register", json=user_data, timeout=10) as response:
                    if response.status == 200:
                        consent_issues.append({
                            'issue': 'Data processing allowed without consent',
                            'risk': 'critical'
                        })
            
            return {
                'status': 'completed',
                'consent_issues': consent_issues,
                'issues_count': len(consent_issues)
            }
            
        except Exception as e:
            logger.error(f"Consent mechanisms testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def audit_data_collection(self) -> Dict[str, Any]:
        """Audit data collection practices"""
        logger.info("Auditing data collection...")
        
        data_collection_points = []
        
        test_endpoints = [
            '/api/v1/register',
            '/api/v1/profile',
            '/api/v1/feedback',
            '/api/v1/analytics'
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in test_endpoints:
                    # Test what data is collected
                    test_data = {
                        'email': 'test@example.com',
                        'name': 'Test User',
                        'location': 'Test Location',
                        'phone': '+1234567890',
                        'ip_address': '192.168.1.1',
                        'user_agent': 'Test Agent'
                    }
                    
                    async with session.post(f"{self.base_url}{endpoint}", json=test_data, timeout=10) as response:
                        if response.status in [200, 201]:
                            data_collection_points.append({
                                'endpoint': endpoint,
                                'collects_personal_data': True,
                                'data_types': list(test_data.keys())
                            })
            
            return {
                'status': 'completed',
                'data_collection_points': data_collection_points,
                'collection_points_count': len(data_collection_points)
            }
            
        except Exception as e:
            logger.error(f"Data collection audit failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def run_zap_vulnerability_scan(self) -> Dict[str, Any]:
        """Run OWASP ZAP vulnerability scan"""
        logger.info("Running OWASP ZAP vulnerability scan...")
        
        try:
            if not self.zap:
                return {'error': 'ZAP not initialized'}
            
            # Spider the application
            logger.info("Spidering application...")
            scan_id = self.zap.spider.scan(self.base_url)
            
            # Wait for spider to complete
            while int(self.zap.spider.status(scan_id)) < 100:
                await asyncio.sleep(2)
            
            # Active scan
            logger.info("Running active scan...")
            scan_id = self.zap.ascan.scan(self.base_url)
            
            # Wait for active scan to complete
            while int(self.zap.ascan.status(scan_id)) < 100:
                await asyncio.sleep(5)
            
            # Get results
            alerts = self.zap.core.alerts()
            
            # Categorize alerts by risk
            high_risk = [alert for alert in alerts if alert['risk'] == 'High']
            medium_risk = [alert for alert in alerts if alert['risk'] == 'Medium']
            low_risk = [alert for alert in alerts if alert['risk'] == 'Low']
            
            return {
                'status': 'completed',
                'total_alerts': len(alerts),
                'high_risk': len(high_risk),
                'medium_risk': len(medium_risk),
                'low_risk': len(low_risk),
                'alerts': alerts
            }
            
        except Exception as e:
            logger.error(f"ZAP vulnerability scan failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def generate_security_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate security summary"""
        summary = {
            'security_score': 0,
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0,
            'total_issues': 0,
            'recommendations': []
        }
        
        # Count issues from all tests
        for test_category, results in test_results.items():
            if isinstance(results, dict) and 'error' not in results:
                # Count issues based on test results
                if 'issues' in results:
                    for issue in results['issues']:
                        risk = issue.get('risk', '').lower()
                        if risk == 'critical':
                            summary['critical_issues'] += 1
                        elif risk == 'high':
                            summary['high_issues'] += 1
                        elif risk == 'medium':
                            summary['medium_issues'] += 1
                        else:
                            summary['low_issues'] += 1
        
        # Add ZAP results
        if 'zap_scan' in test_results and 'alerts' in test_results['zap_scan']:
            zap_results = test_results['zap_scan']
            summary['high_issues'] += zap_results.get('high_risk', 0)
            summary['medium_issues'] += zap_results.get('medium_risk', 0)
            summary['low_issues'] += zap_results.get('low_risk', 0)
        
        summary['total_issues'] = (summary['critical_issues'] + summary['high_issues'] + 
                                 summary['medium_issues'] + summary['low_issues'])
        
        # Calculate security score
        security_score = max(0, 100 - (summary['critical_issues'] * 25 + 
                                     summary['high_issues'] * 10 + 
                                     summary['medium_issues'] * 5 + 
                                     summary['low_issues'] * 1))
        summary['security_score'] = security_score
        
        # Generate recommendations
        if summary['critical_issues'] > 0:
            summary['recommendations'].append("Address critical security vulnerabilities immediately")
        
        if summary['high_issues'] > 0:
            summary['recommendations'].append("Fix high-risk security issues")
        
        # Add specific recommendations based on test results
        if 'auth_testing' in test_results:
            auth_results = test_results['auth_testing']
            if not auth_results.get('brute_force_protection'):
                summary['recommendations'].append("Implement brute force protection for login endpoints")
        
        if 'ssl_tls' in test_results:
            ssl_results = test_results['ssl_tls']
            if 'hsts_header_check' in ssl_results and not ssl_results['hsts_header_check'].get('hsts_present'):
                summary['recommendations'].append("Add HSTS headers for enhanced SSL/TLS security")
        
        if 'input_validation' in test_results:
            input_results = test_results['input_validation']
            if not input_results.get('xss_protection'):
                summary['recommendations'].append("Implement proper input sanitization to prevent XSS attacks")
        
        return summary

    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """Save security test results"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"security_test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Security test results saved to: {output_file}")


async def main():
    """Main function to run security tests"""
    config = {
        'base_url': os.getenv('TEST_BASE_URL', 'http://localhost:3000'),
        'zap_proxy': os.getenv('ZAP_PROXY', 'http://127.0.0.1:8080'),
        'api_key': os.getenv('API_KEY', '')
    }
    
    framework = SecurityTestingFramework(config)
    results = await framework.run_comprehensive_security_tests()
    
    # Save results
    framework.save_results(results)
    
    # Print summary
    if 'summary' in results:
        summary = results['summary']
        print(f"Security Score: {summary.get('security_score', 0)}")
        print(f"Risk Level: {summary.get('risk_level', 'unknown')}")
        
        if summary.get('recommendations'):
            print("Recommendations:")
            for recommendation in summary['recommendations']:
                print(f"  - {recommendation}")


if __name__ == "__main__":
    asyncio.run(main())