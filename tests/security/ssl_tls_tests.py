#!/usr/bin/env python3
"""
SSL/TLS Configuration Testing Module
Implements comprehensive SSL/TLS security testing
"""

import asyncio
import aiohttp
import ssl
import socket
import logging
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Any
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import nmap

logger = logging.getLogger(__name__)

class SSLTLSTester:
    """SSL/TLS configuration testing implementation"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
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
            'cipher_suite_analysis': await self.test_cipher_suites(),
            'protocol_version_check': await self.test_ssl_protocols(),
            'hsts_header_check': await self.test_hsts_headers()
        }
        
        return ssl_results
    
    async def test_certificate_validation(self) -> Dict[str, Any]:
        """Test SSL certificate validation"""
        logger.info("Testing SSL certificate...")
        
        try:
            parsed_url = urlparse(self.base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            # Get certificate
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
                    cert = x509.load_der_x509_certificate(cert_der, default_backend())
                    
                    # Check certificate validity
                    now = datetime.now()
                    is_valid = cert.not_valid_before <= now <= cert.not_valid_after
                    
                    # Check subject alternative names
                    try:
                        san_ext = cert.extensions.get_extension_for_oid(
                            x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                        )
                        san_names = [name.value for name in san_ext.value]
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
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_cipher_suites(self) -> Dict[str, Any]:
        """Test SSL cipher suites"""
        logger.info("Testing SSL cipher suites...")
        
        try:
            # Use nmap to test SSL ciphers
            parsed_url = urlparse(self.base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            nm = nmap.PortScanner()
            result = nm.scan(hostname, str(port), arguments='--script ssl-enum-ciphers')
            
            cipher_info = result['scan'].get(hostname, {}).get('tcp', {}).get(port, {})
            
            return {
                'status': 'completed',
                'cipher_info': cipher_info
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_ssl_protocols(self) -> Dict[str, Any]:
        """Test SSL/TLS protocol versions"""
        logger.info("Testing SSL/TLS protocols...")
        
        protocols_to_test = [
            ('SSLv2', ssl.PROTOCOL_SSLv23),
            ('SSLv3', ssl.PROTOCOL_SSLv23),
            ('TLSv1.0', ssl.PROTOCOL_TLSv1),
            ('TLSv1.1', ssl.PROTOCOL_TLSv1_1),
            ('TLSv1.2', ssl.PROTOCOL_TLSv1_2)
        ]
        
        supported_protocols = []
        
        parsed_url = urlparse(self.base_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 443
        
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
                        'hsts_header': hsts_header,
                        'recommendation': 'Add HSTS header' if not hsts_header else 'HSTS configured'
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }