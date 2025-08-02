#!/usr/bin/env python3
"""
Test the structure and syntax of TimescaleDB service files
"""

import ast
import os
import sys

import torch.nn.functional as F


def test_python_syntax(filepath):
    """Test Python file syntax"""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Parse the AST to check syntax
        ast.parse(content)
        print(f"✓ {filepath} has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath} has syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        "main.py",
        "requirements.txt",
        "Dockerfile",
        "healthcheck.py",
        "README.md",
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True


def test_docker_file():
    """Test Dockerfile structure"""
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()

        required_instructions = ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE", "CMD"]
        missing_instructions = []

        for instruction in required_instructions:
            if instruction not in content:
                missing_instructions.append(instruction)

        if missing_instructions:
            print(f"✗ Dockerfile missing instructions: {missing_instructions}")
            return False
        else:
            print("✓ Dockerfile has all required instructions")
            return True
    except Exception as e:
        print(f"✗ Error checking Dockerfile: {e}")
        return False


def test_requirements_file():
    """Test requirements.txt structure"""
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()

        required_packages = ["fastapi", "uvicorn", "asyncpg", "structlog", "pydantic"]
        missing_packages = []

        for package in required_packages:
            if package not in content:
                missing_packages.append(package)

        if missing_packages:
            print(f"✗ requirements.txt missing packages: {missing_packages}")
            return False
        else:
            print("✓ requirements.txt has all required packages")
            return True
    except Exception as e:
        print(f"✗ Error checking requirements.txt: {e}")
        return False


def test_sql_file_exists():
    """Test that SQL initialization file exists"""
    sql_file = "../../database/timescale/init.sql"
    if os.path.exists(sql_file):
        print("✓ TimescaleDB initialization SQL file exists")
        return True
    else:
        print(f"✗ SQL file not found: {sql_file}")
        return False


def main():
    """Run all structure tests"""
    print("Testing TimescaleDB service structure...")
    print("=" * 50)

    tests = [
        lambda: test_file_structure(),
        lambda: test_python_syntax("main.py"),
        lambda: test_python_syntax("healthcheck.py"),
        lambda: test_docker_file(),
        lambda: test_requirements_file(),
        lambda: test_sql_file_exists(),
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Structure tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All structure tests passed!")
        return 0
    else:
        print("✗ Some structure tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
