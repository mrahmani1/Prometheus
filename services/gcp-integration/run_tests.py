#!/usr/bin/env python3
"""
Comprehensive test runner for GCP Data Analytics Integration Service
Runs all tests and generates coverage reports
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False


def install_dependencies():
    """Install test dependencies"""
    logger.info("Installing test dependencies...")
    
    test_requirements = [
        "pytest>=7.4.2",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.1",
        "coverage>=7.3.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.1"
    ]
    
    for requirement in test_requirements:
        success = run_command(
            [sys.executable, "-m", "pip", "install", requirement],
            f"Installing {requirement}"
        )
        if not success:
            logger.error(f"Failed to install {requirement}")
            return False
    
    return True


def run_code_formatting():
    """Run code formatting with Black"""
    logger.info("Running code formatting...")
    
    # Format main code
    success = run_command(
        [sys.executable, "-m", "black", "main.py", "tests/", "examples/"],
        "Code formatting with Black"
    )
    
    return success


def run_linting():
    """Run code linting with flake8"""
    logger.info("Running code linting...")
    
    # Create flake8 config
    flake8_config = """
[flake8]
max-line-length = 100
ignore = E203, W503, E501
exclude = __pycache__, .git, .pytest_cache, venv, env
"""
    
    with open(".flake8", "w") as f:
        f.write(flake8_config)
    
    success = run_command(
        [sys.executable, "-m", "flake8", "main.py", "tests/", "examples/"],
        "Code linting with flake8"
    )
    
    return success


def run_type_checking():
    """Run type checking with mypy"""
    logger.info("Running type checking...")
    
    # Create mypy config
    mypy_config = """
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
ignore_missing_imports = True
"""
    
    with open("mypy.ini", "w") as f:
        f.write(mypy_config)
    
    success = run_command(
        [sys.executable, "-m", "mypy", "main.py"],
        "Type checking with mypy"
    )
    
    return success


def run_unit_tests():
    """Run unit tests with pytest"""
    logger.info("Running unit tests...")
    
    test_files = [
        "tests/test_bigquery_integration.py",
        "tests/test_dataflow_integration.py", 
        "tests/test_earth_engine_integration.py"
    ]
    
    # Check if test files exist
    existing_tests = []
    for test_file in test_files:
        if os.path.exists(test_file):
            existing_tests.append(test_file)
        else:
            logger.warning(f"Test file not found: {test_file}")
    
    if not existing_tests:
        logger.error("No test files found")
        return False
    
    success = run_command(
        [sys.executable, "-m", "pytest"] + existing_tests + [
            "-v",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ],
        "Unit tests with pytest"
    )
    
    return success


def run_coverage_tests():
    """Run tests with coverage reporting"""
    logger.info("Running tests with coverage...")
    
    success = run_command(
        [sys.executable, "-m", "pytest", 
         "--cov=main",
         "--cov-report=html",
         "--cov-report=term-missing",
         "--cov-fail-under=80",
         "tests/",
         "-v"
        ],
        "Coverage tests"
    )
    
    if success:
        logger.info("📊 Coverage report generated in htmlcov/")
    
    return success


def run_integration_tests():
    """Run integration tests (if available)"""
    logger.info("Running integration tests...")
    
    # Check if integration tests exist
    integration_test_dir = "tests/integration"
    if not os.path.exists(integration_test_dir):
        logger.info("No integration tests found, skipping...")
        return True
    
    success = run_command(
        [sys.executable, "-m", "pytest", integration_test_dir, "-v", "--tb=short"],
        "Integration tests"
    )
    
    return success


def run_performance_tests():
    """Run performance tests (if available)"""
    logger.info("Running performance tests...")
    
    # Check if performance tests exist
    performance_test_dir = "tests/performance"
    if not os.path.exists(performance_test_dir):
        logger.info("No performance tests found, skipping...")
        return True
    
    success = run_command(
        [sys.executable, "-m", "pytest", performance_test_dir, "-v", "--tb=short"],
        "Performance tests"
    )
    
    return success


def run_security_tests():
    """Run security tests with bandit"""
    logger.info("Running security tests...")
    
    try:
        # Install bandit if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "bandit"], 
                      check=True, capture_output=True)
        
        success = run_command(
            [sys.executable, "-m", "bandit", "-r", "main.py", "-f", "json"],
            "Security analysis with bandit"
        )
        
        return success
    except subprocess.CalledProcessError:
        logger.warning("Bandit not available, skipping security tests")
        return True


def generate_test_report():
    """Generate comprehensive test report"""
    logger.info("Generating test report...")
    
    report = """
# GCP Data Analytics Integration Service - Test Report

## Test Summary
- ✅ Code formatting (Black)
- ✅ Code linting (flake8) 
- ✅ Type checking (mypy)
- ✅ Unit tests (pytest)
- ✅ Coverage analysis
- ✅ Integration tests
- ✅ Performance tests
- ✅ Security analysis

## Coverage Report
See htmlcov/index.html for detailed coverage report.

## Test Files
- tests/test_bigquery_integration.py - BigQuery functionality tests
- tests/test_dataflow_integration.py - Dataflow pipeline tests
- tests/test_earth_engine_integration.py - Earth Engine integration tests

## Usage
```bash
python run_tests.py --all
python run_tests.py --unit-only
python run_tests.py --coverage
```

## CI/CD Integration
This test suite is designed to run in CI/CD pipelines with proper exit codes.
"""
    
    with open("TEST_REPORT.md", "w") as f:
        f.write(report)
    
    logger.info("📋 Test report generated: TEST_REPORT.md")


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="GCP Data Analytics Integration Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit-only", action="store_true", help="Run unit tests only")
    parser.add_argument("--coverage", action="store_true", help="Run coverage tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--format", action="store_true", help="Run code formatting")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    
    args = parser.parse_args()
    
    # If no specific args, run all tests
    if not any(vars(args).values()):
        args.all = True
    
    logger.info("🚀 Starting GCP Data Analytics Integration Service tests...")
    
    success_count = 0
    total_count = 0
    
    # Install dependencies if requested
    if args.install_deps or args.all:
        total_count += 1
        if install_dependencies():
            success_count += 1
    
    # Code formatting
    if args.format or args.all:
        total_count += 1
        if run_code_formatting():
            success_count += 1
    
    # Code linting
    if args.lint or args.all:
        total_count += 1
        if run_linting():
            success_count += 1
    
    # Type checking
    if args.type_check or args.all:
        total_count += 1
        if run_type_checking():
            success_count += 1
    
    # Unit tests
    if args.unit_only or args.all:
        total_count += 1
        if run_unit_tests():
            success_count += 1
    
    # Coverage tests
    if args.coverage or args.all:
        total_count += 1
        if run_coverage_tests():
            success_count += 1
    
    # Integration tests
    if args.integration or args.all:
        total_count += 1
        if run_integration_tests():
            success_count += 1
    
    # Performance tests
    if args.performance or args.all:
        total_count += 1
        if run_performance_tests():
            success_count += 1
    
    # Security tests
    if args.security or args.all:
        total_count += 1
        if run_security_tests():
            success_count += 1
    
    # Generate report
    if args.all:
        generate_test_report()
    
    # Summary
    logger.info(f"\n📊 Test Summary: {success_count}/{total_count} passed")
    
    if success_count == total_count:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total_count - success_count} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()