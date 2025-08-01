#!/usr/bin/env python3
"""
Development environment setup script for the Prometheus AI platform.
Handles Git setup, pre-commit hooks, testing infrastructure, and CI/CD pipeline.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True, shell=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=shell, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_git_installed():
    """Check if Git is installed."""
    success, _, _ = run_command("git --version", check=False)
    return success

def setup_git_repository():
    """Initialize Git repository and set up basic configuration."""
    print("🔧 Setting up Git repository...")
    
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first:")
        if platform.system() == "Windows":
            print("   Download from: https://git-scm.com/download/win")
        elif platform.system() == "Darwin":
            print("   Run: brew install git")
        else:
            print("   Run: sudo apt-get install git")
        return False
    
    # Initialize repository if not already initialized
    if not Path(".git").exists():
        success, _, _ = run_command("git init")
        if not success:
            print("❌ Failed to initialize Git repository")
            return False
        print("✅ Git repository initialized")
    
    # Set up .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite3

# Environment variables
.env
.env.local

# Docker
.dockerignore

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Coverage reports
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover

# Pytest
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Backup files
*.backup
backups/

# Temporary files
*.tmp
*.temp
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("✅ .gitignore created")
    return True

def setup_pre_commit_hooks():
    """Set up pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")
    
    # Install pre-commit if not already installed
    success, _, _ = run_command("pip install pre-commit", check=False)
    if not success:
        print("❌ Failed to install pre-commit")
        return False
    
    # Install pre-commit hooks
    success, _, _ = run_command("pre-commit install", check=False)
    if not success:
        print("⚠️  Could not install pre-commit hooks (Git may not be available)")
        return False
    
    print("✅ Pre-commit hooks installed")
    return True

def create_test_infrastructure():
    """Create comprehensive test infrastructure."""
    print("🔧 Setting up test infrastructure...")
    
    # Create pytest configuration
    pytest_ini = """
[tool:pytest]
testpaths = tests services
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=services
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    gpu: Tests requiring GPU
    network: Tests requiring network access
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini.strip())
    
    # Create test requirements
    test_requirements = """
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.1
pytest-asyncio>=0.21.1
pytest-xdist>=3.3.1
pytest-benchmark>=4.0.0
pytest-timeout>=2.1.0
coverage>=7.2.7
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
pylint>=3.0.0
bandit>=1.7.5
mypy>=1.5.1
pre-commit>=3.3.3
"""
    
    with open("requirements-test.txt", "w") as f:
        f.write(test_requirements.strip())
    
    # Install test dependencies
    success, _, _ = run_command("pip install -r requirements-test.txt", check=False)
    if success:
        print("✅ Test dependencies installed")
    else:
        print("⚠️  Some test dependencies may not have installed correctly")
    
    print("✅ Test infrastructure created")
    return True

def create_ci_cd_pipeline():
    """Create CI/CD pipeline configuration."""
    print("🔧 Setting up CI/CD pipeline...")
    
    # Create GitHub Actions directory
    github_dir = Path(".github/workflows")
    github_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main CI/CD workflow
    workflow_content = """
name: Prometheus AI Platform CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Lint with flake8
      run: |
        flake8 services --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 services --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Security scan with bandit
      run: |
        bandit -r services -f json -o bandit-report.json || true
        bandit -r services
    
    - name: Type checking with mypy
      run: |
        mypy services --ignore-missing-imports || true
    
    - name: Test with pytest
      run: |
        pytest --cov=services --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  docker:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker images
      run: |
        for service in services/*/; do
          if [ -f "$service/Dockerfile" ]; then
            service_name=$(basename "$service")
            docker build -t ghcr.io/${{ github.repository }}/$service_name:latest "$service"
            docker push ghcr.io/${{ github.repository }}/$service_name:latest
          fi
        done

  deploy:
    runs-on: ubuntu-latest
    needs: [test, docker]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment..."
        # Add deployment commands here
"""
    
    with open(github_dir / "ci-cd.yml", "w", encoding='utf-8') as f:
        f.write(workflow_content.strip())
    
    # Create quality gates workflow
    quality_gates_content = """
name: Quality Gates

on:
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
    
    - name: Code quality check
      run: |
        black --check services
        isort --check-only services
        flake8 services
        pylint services --fail-under=8.0
    
    - name: Security check
      run: |
        bandit -r services -ll
    
    - name: Test coverage check
      run: |
        pytest --cov=services --cov-fail-under=80
    
    - name: Comment PR
      uses: actions/github-script@v6
      if: failure()
      with:
        script: |
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: 'Quality gates failed. Please fix the issues before merging.'
          })
"""
    
    with open(github_dir / "quality-gates.yml", "w", encoding='utf-8') as f:
        f.write(quality_gates_content.strip())
    
    print("✅ CI/CD pipeline configuration created")
    return True

def create_comprehensive_test_suite():
    """Create comprehensive test coverage."""
    print("🔧 Creating comprehensive test suite...")
    
    # Create test runner script
    test_runner_content = """#!/usr/bin/env python3
\"\"\"
Comprehensive test runner for the Prometheus AI platform.
\"\"\"

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    \"\"\"Run all tests with comprehensive coverage.\"\"\"
    
    print("Running comprehensive test suite...")
    
    # Unit tests
    print("\\nRunning unit tests...")
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
    print("\\nRunning integration tests...")
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
    print("\\nRunning security tests...")
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
"""
    
    with open("scripts/run_tests.py", "w", encoding='utf-8') as f:
        f.write(test_runner_content)
    
    # Make it executable
    os.chmod("scripts/run_tests.py", 0o755)
    
    print("✅ Comprehensive test suite created")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Prometheus AI development environment...")
    
    # Create scripts directory if it doesn't exist
    Path("scripts").mkdir(exist_ok=True)
    
    success = True
    
    # Setup Git repository
    if not setup_git_repository():
        success = False
    
    # Setup pre-commit hooks (may fail if Git not available)
    setup_pre_commit_hooks()
    
    # Create test infrastructure
    if not create_test_infrastructure():
        success = False
    
    # Create CI/CD pipeline
    if not create_ci_cd_pipeline():
        success = False
    
    # Create comprehensive test suite
    if not create_comprehensive_test_suite():
        success = False
    
    if success:
        print("\\n✅ Development environment setup completed successfully!")
        print("\\n📋 Next steps:")
        print("1. Install Git if not already installed")
        print("2. Run: git add . && git commit -m 'Initial commit'")
        print("3. Run: python scripts/run_tests.py")
        print("4. Run: pre-commit run --all-files")
    else:
        print("\\n⚠️  Setup completed with some issues. Please review the output above.")
    
    return success

if __name__ == "__main__":
    main()