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
        self.base_url = config.get('base_url', 'http://localhost:8000')
        self.zap_proxy = config.get('zap_proxy', 'http://127.0.0.1:8080')
        self.api_key = config.get('api_key', '')
        self.test_results = {}
        
        # Initialize ZAP client
        self.zap = ZAPv2(proxies={'http': self.zap_proxy, 'https': self.zap_proxy})
        
        # Test credentials for authentication testing
        self.test_credentials = {
            'valid_user': {'username': 'test_user', 'password': 'Test123!@#'},
            'admin_user': {'username': 'admin', 'password': 'Admin123!@#'},
            'invalid_user': {'username': 'invalid', 'password': 'wrong'},
            'weak_password': {'username': 'weak_user', 'password': '123'}
        }
        
    async def run_comprehensive_security_tests(self) -> Dict[str, Any]:
        """Run all security tests and return comprehensive results"""
        logger.info("Starting comprehensive security testing...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'target_url': self.base_url,
            'tests': {}
        }
        
        try:
            # 1. OWASP ZAP Security Scanning
            results['tests']['owasp_zap'] = await self.run_owasp_zap_scan()
            
            # 2. API Penetration Testing
            results['tests']['api_penetration'] = await self.run_api_penetration_tests()
            
            # 3. Authentication and Authorization Testing
            results['tests']['auth_testing'] = await self.run_authentication_tests()
            
            # 4. Input Validation and XSS Testing
            results['tests']['input_validation'] = await self.run_input_validation_tests()
            
            # 5. SSL/TLS Configuration Testing
            results['tests']['ssl_tls'] = await self.run_ssl_tls_tests()
            
            # 6. Data Privacy and GDPR Compliance
            results['tests']['privacy_gdpr'] = await self.run_privacy_compliance_tests()
            
            # 7. Network Security Testing
            results['tests']['network_security'] = await self.run_network_security_tests()
            
            # Generate security report
            results['summary'] = self.generate_security_summary(results['tests'])
            
        except Exception as e:
            logger.error(f"Security testing failed: {str(e)}")
            results['error'] = str(e)
            
        return results    
 
    async def run_owasp_zap_scan(self) -> Dict[str, Any]:
        """Run OWASP ZAP security scanning"""
        logger.info("Running OWASP ZAP security scan...")
        
        try:
            # Start ZAP if not running
            await self.ensure_zap_running()
            
            # Spider the application
            logger.info("Starting ZAP spider...")
            scan_id = self.zap.spider.scan(self.base_url)
            
            # Wait for spider to complete
            while int(self.zap.spider.status(scan_id)) < 100:
                await asyncio.sleep(2)
                
            logger.info("Spider completed, starting active scan...")
            
            # Run active scan
            scan_id = self.zap.ascan.scan(self.base_url)
            
            # Wait for active scan to complete
            while int(self.zap.ascan.status(scan_id)) < 100:
                await asyncio.sleep(5)
                
            # Get results
            alerts = self.zap.core.alerts()
            
            # Categorize alerts by risk level
            categorized_alerts = {
                'high': [],
                'medium': [],
                'low': [],
                'informational': []
            }
            
            for alert in alerts:
                risk_level = alert.get('risk', 'informational').lower()
                categorized_alerts[risk_level].append({
                    'name': alert.get('name'),
                    'description': alert.get('description'),
                    'url': alert.get('url'),
                    'param': alert.get('param'),
                    'solution': alert.get('solution')
                })
            
            return {
                'status': 'completed',
                'total_alerts': len(alerts),
                'alerts_by_risk': categorized_alerts,
                'scan_duration': 'completed'
            }
            
        except Exception as e:
            logger.error(f"OWASP ZAP scan failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def run_api_penetration_tests(self) -> Dict[str, Any]:
        """Run comprehensive API penetration testing"""
        logger.info("Running API penetration tests...")
        
        test_results = {
            'sql_injection': await self.test_sql_injection(),
            'nosql_injection': await self.test_nosql_injection(),
            'command_injection': await self.test_command_injection(),
            'path_traversal': await self.test_path_traversal(),
            'xxe_injection': await self.test_xxe_injection(),
            'deserialization': await self.test_deserialization_attacks(),
            'rate_limiting': await self.test_rate_limiting(),
            'cors_misconfiguration': await self.test_cors_configuration()
        }
        
        return test_results