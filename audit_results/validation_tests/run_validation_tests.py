#!/usr/bin/env python3
"""
Validation Test Runner

Runs all generated validation tests to verify data source replacements.
"""

import pytest
import sys
from pathlib import Path

def main():
    """Run all validation tests"""
    test_dir = Path(__file__).parent
    
    # Run pytest with coverage
    exit_code = pytest.main([
        str(test_dir),
        "-v",
        "--tb=short",
        "--cov=backend",
        "--cov-report=html:coverage_html",
        "--cov-report=term-missing"
    ])
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
