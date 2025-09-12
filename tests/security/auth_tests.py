#!/usr/bin/env python3
"""
Authentication and Authorization Testing Module
Implements comprehensive auth security testing
"""

import asyncio
import aiohttp
import jwt
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class AuthenticationTester:
    """Authentication and authorization testing implementation"""
    
    def __init__(self, base_url: str, test_credentials: Dict[str, Dict[str, str]]):
        self.base_url = base_url
        self.test_credentials = test_credentials
    
    async def run_authentication_tests(self) -> Dict[str, Any]:
        """Run comprehensive authentication and authorization tests"""
        logger.info("Running authentication and authorization tests...")
        
        auth_results = {
            'jwt_security': await self.test_jwt_security(),
            'session_management': await self.test_session_management(),
            'password_policy': await self.test_password_policy(),
            'brute_force_protection': await self.test_brute_force_protection(),
            'privilege_escalation': await self.test_privilege_escalation(),
            'token_validation': await self.test_token_validation()
        }
        
        return auth_results
    
    async def test_jwt_security(self) -> Dict[str, Any]:
        """Test JWT token security"""
        logger.info("Testing JWT security...")
        
        issues = []
        
        # Test weak JWT secrets
        weak_secrets = ['secret', '123456', 'jwt_secret', '']
        test_payload = {'user_id': 1, 'role': 'admin'}
        
        for secret in weak_secrets:
            try:
                token = jwt.encode(test_payload, secret, algorithm='HS256')
                
                # Try to use the token
                headers = {'Authorization': f'Bearer {token}'}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/v1/protected",
                                         headers=headers, timeout=10) as response:
                        if response.status == 200:
                            issues.append({
                                'issue': 'Weak JWT secret',
                                'secret': secret,
                                'risk': 'critical'
                            })
            except Exception:
                pass
        
        # Test algorithm confusion
        try:
            # Create token with 'none' algorithm
            token = jwt.encode(test_payload, '', algorithm='none')
            headers = {'Authorization': f'Bearer {token}'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v1/protected",
                                     headers=headers, timeout=10) as response:
                    if response.status == 200:
                        issues.append({
                            'issue': 'Algorithm confusion vulnerability',
                            'risk': 'critical'
                        })
        except Exception:
            pass
        
        return {
            'status': 'completed',
            'issues_found': len(issues),
            'issues': issues
        }
    
    async def test_session_management(self) -> Dict[str, Any]:
        """Test session management security"""
        logger.info("Testing session management...")
        
        issues = []
        
        async with aiohttp.ClientSession() as session:
            # Test session fixation
            try:
                # Get initial session
                async with session.get(f"{self.base_url}/api/v1/login") as response:
                    initial_cookies = response.cookies
                
                # Login with credentials
                login_data = self.test_credentials['valid_user']
                async with session.post(f"{self.base_url}/api/v1/login",
                                      json=login_data) as response:
                    post_login_cookies = response.cookies
                
                # Check if session ID changed after login
                if initial_cookies == post_login_cookies:
                    issues.append({
                        'issue': 'Session fixation vulnerability',
                        'risk': 'high'
                    })
                    
            except Exception:
                pass
        
        return {
            'status': 'completed',
            'issues_found': len(issues),
            'issues': issues
        }
    
    async def test_password_policy(self) -> Dict[str, Any]:
        """Test password policy enforcement"""
        logger.info("Testing password policy...")
        
        weak_passwords = [
            '123456',
            'password',
            'admin',
            'test',
            '12345678',
            'qwerty'
        ]
        
        policy_issues = []
        
        async with aiohttp.ClientSession() as session:
            for weak_password in weak_passwords:
                try:
                    user_data = {
                        'username': f'test_{weak_password}',
                        'password': weak_password,
                        'email': f'test_{weak_password}@example.com'
                    }
                    
                    async with session.post(f"{self.base_url}/api/v1/register",
                                          json=user_data, timeout=10) as response:
                        if response.status == 201:  # User created successfully
                            policy_issues.append({
                                'weak_password': weak_password,
                                'accepted': True
                            })
                            
                except Exception:
                    pass
        
        return {
            'status': 'completed',
            'weak_passwords_accepted': len(policy_issues),
            'issues': policy_issues
        }
    
    async def test_brute_force_protection(self) -> Dict[str, Any]:
        """Test brute force protection"""
        logger.info("Testing brute force protection...")
        
        login_attempts = 0
        locked_out = False
        
        async with aiohttp.ClientSession() as session:
            # Attempt multiple failed logins
            for i in range(20):
                try:
                    login_data = {
                        'username': 'admin',
                        'password': f'wrong_password_{i}'
                    }
                    
                    async with session.post(f"{self.base_url}/api/v1/login",
                                          json=login_data, timeout=10) as response:
                        login_attempts += 1
                        
                        if response.status == 429:  # Too many requests
                            locked_out = True
                            break
                            
                except Exception:
                    break
        
        return {
            'status': 'completed',
            'login_attempts': login_attempts,
            'brute_force_protection': locked_out,
            'recommendation': 'Implement account lockout' if not locked_out else 'Protection working'
        }
    
    async def test_privilege_escalation(self) -> Dict[str, Any]:
        """Test for privilege escalation vulnerabilities"""
        logger.info("Testing privilege escalation...")
        
        escalation_attempts = []
        
        # Test parameter manipulation
        test_cases = [
            {'user_id': 1, 'role': 'admin'},
            {'is_admin': True},
            {'permissions': ['admin', 'superuser']},
            {'access_level': 'administrator'}
        ]
        
        async with aiohttp.ClientSession() as session:
            for test_case in test_cases:
                try:
                    async with session.post(f"{self.base_url}/api/v1/update-profile",
                                          json=test_case, timeout=10) as response:
                        if response.status == 200:
                            escalation_attempts.append({
                                'method': 'parameter_manipulation',
                                'payload': test_case,
                                'successful': True
                            })
                except Exception:
                    pass
        
        return {
            'status': 'completed',
            'escalation_attempts': len(escalation_attempts),
            'vulnerabilities': escalation_attempts
        }
    
    async def test_token_validation(self) -> Dict[str, Any]:
        """Test token validation security"""
        logger.info("Testing token validation...")
        
        validation_issues = []
        
        # Test expired tokens, malformed tokens, etc.
        test_tokens = [
            'invalid_token',
            'Bearer invalid',
            '',
            'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature'
        ]
        
        async with aiohttp.ClientSession() as session:
            for token in test_tokens:
                try:
                    headers = {'Authorization': f'Bearer {token}'}
                    async with session.get(f"{self.base_url}/api/v1/protected",
                                         headers=headers, timeout=10) as response:
                        if response.status == 200:
                            validation_issues.append({
                                'token': token,
                                'accepted': True,
                                'risk': 'high'
                            })
                except Exception:
                    pass
        
        return {
            'status': 'completed',
            'validation_issues': len(validation_issues),
            'issues': validation_issues
        }