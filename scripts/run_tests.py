#!/usr/bin/env python3
"""
Comprehensive test runner for the Prometheus AI platform.
"""

from pathlib import Path
import os
import subprocess
import sys

def run_tests():
    """Run all tests with comprehensive coverage."""
    
    print("Running comprehensive test suite...")
    
    # Unit tests
    print("\nRunning unit tests...")
    result = subprocess.run([
        "pytest", 
        "services/", 
        "-v", 
        "--cov=services",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "-m", "not slow and not integration"
    ])
    
    if result.returncode != 0:
        print("Unit tests failed")
        return False
    
    # Integration tests
    print("\nRunning integration tests...")
    result = subprocess.run([
        "pytest", 
        "services/", 
        "-v",
        "-m", "integration",
        "--timeout=300"
    ])
    
    if result.returncode != 0:
        print("Some integration tests failed")
    
    # Security tests
    print("\nRunning security tests...")
    result = subprocess.run([
        "bandit", 
        "-r", "services",
        "-f", "json",
        "-o", "security-report.json"
    ])
    
    print("Test suite completed")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
