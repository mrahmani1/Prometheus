#!/usr/bin/env python3
"""
Comprehensive quality checker for the Prometheus AI platform.
Combines syntax checking, testing, security scanning, and code quality metrics.
"""

from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

class QualityChecker:
    """Comprehensive code quality checker."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "syntax_check": {},
            "code_quality": {},
            "test_coverage": {},
            "security_scan": {},
            "overall_score": 0
        }
    
    def check_syntax_errors(self):
        """Check for Python syntax errors."""
        print("🔍 Checking for syntax errors...")
        
        syntax_errors = []
        total_files = 0
        
        for root, dirs, files in os.walk("services"):
            for file in files:
                if file.endswith('.py'):
                    total_files += 1
                    file_path = Path(root) / file
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        compile(content, str(file_path), 'exec')
                        
                    except SyntaxError as e:
                        syntax_errors.append({
                            "file": str(file_path),
                            "line": e.lineno,
                            "error": str(e)
                        })
                    except Exception as e:
                        syntax_errors.append({
                            "file": str(file_path),
                            "line": 0,
                            "error": f"Compilation error: {str(e)}"
                        })
        
        self.results["syntax_check"] = {
            "total_files": total_files,
            "files_with_errors": len(syntax_errors),
            "errors": syntax_errors,
            "success_rate": ((total_files - len(syntax_errors)) / total_files * 100) if total_files > 0 else 0
        }
        
        if syntax_errors:
            print(f"❌ Found {len(syntax_errors)} files with syntax errors")
            for error in syntax_errors[:5]:  # Show first 5
                print(f"   {error['file']}:{error['line']} - {error['error']}")
            if len(syntax_errors) > 5:
                print(f"   ... and {len(syntax_errors) - 5} more")
        else:
            print("✅ No syntax errors found")
    
    def run_code_quality_checks(self):
        """Run code quality checks with flake8, black, isort."""
        print("📊 Running code quality checks...")
        
        quality_results = {}
        
        # Check Black formatting
        try:
            result = subprocess.run(
                ["black", "--check", "services"], 
                capture_output=True, text=True, check=False
            )
            quality_results["black"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            quality_results["black"] = {"passed": False, "error": "Black not installed"}
        
        # Check isort import sorting
        try:
            result = subprocess.run(
                ["isort", "--check-only", "services"], 
                capture_output=True, text=True, check=False
            )
            quality_results["isort"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            quality_results["isort"] = {"passed": False, "error": "isort not installed"}
        
        # Check flake8 linting
        try:
            result = subprocess.run(
                ["flake8", "services", "--max-line-length=88", "--extend-ignore=E203,W503"], 
                capture_output=True, text=True, check=False
            )
            quality_results["flake8"] = {
                "passed": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
        except FileNotFoundError:
            quality_results["flake8"] = {"passed": False, "error": "flake8 not installed"}
        
        self.results["code_quality"] = quality_results
        
        passed_checks = sum(1 for check in quality_results.values() if check.get("passed", False))
        total_checks = len(quality_results)
        
        print(f"✅ Code quality: {passed_checks}/{total_checks} checks passed")
    
    def run_test_coverage(self):
        """Run tests and check coverage."""
        print("🧪 Running tests and checking coverage...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                "pytest", 
                "services/", 
                "--cov=services",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "-v"
            ], capture_output=True, text=True, check=False)
            
            test_passed = result.returncode == 0
            
            # Try to read coverage report
            coverage_percentage = 0
            try:
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                    coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0)
            except Exception:
                pass
            
            self.results["test_coverage"] = {
                "tests_passed": test_passed,
                "coverage_percentage": coverage_percentage,
                "output": result.stdout + result.stderr
            }
            
            print(f"✅ Tests: {'PASSED' if test_passed else 'FAILED'}")
            print(f"✅ Coverage: {coverage_percentage:.1f}%")
            
        except FileNotFoundError:
            self.results["test_coverage"] = {
                "tests_passed": False,
                "coverage_percentage": 0,
                "error": "pytest not installed"
            }
            print("❌ pytest not available")
    
    def run_security_scan(self):
        """Run security scan."""
        print("🔒 Running security scan...")
        
        try:
            # Run bandit security scan
            result = subprocess.run([
                "bandit", 
                "-r", "services",
                "-f", "json"
            ], capture_output=True, text=True, check=False)
            
            security_issues = 0
            if result.returncode != 0:
                try:
                    bandit_output = json.loads(result.stdout)
                    security_issues = len(bandit_output.get("results", []))
                except Exception:
                    pass
            
            self.results["security_scan"] = {
                "issues_found": security_issues,
                "clean": security_issues == 0,
                "output": result.stdout + result.stderr
            }
            
            if security_issues == 0:
                print("✅ No security issues found")
            else:
                print(f"⚠️  Found {security_issues} security issues")
                
        except FileNotFoundError:
            self.results["security_scan"] = {
                "issues_found": 0,
                "clean": False,
                "error": "bandit not installed"
            }
            print("❌ bandit not available")
    
    def calculate_overall_score(self):
        """Calculate overall quality score."""
        score = 0
        max_score = 100
        
        # Syntax check (25 points)
        syntax_score = self.results["syntax_check"].get("success_rate", 0) * 0.25
        score += syntax_score
        
        # Code quality (25 points)
        quality_checks = self.results["code_quality"]
        passed_quality = sum(1 for check in quality_checks.values() if check.get("passed", False))
        total_quality = len(quality_checks) if quality_checks else 1
        quality_score = (passed_quality / total_quality) * 25
        score += quality_score
        
        # Test coverage (25 points)
        coverage_percentage = self.results["test_coverage"].get("coverage_percentage", 0)
        coverage_score = min(coverage_percentage * 0.25, 25)
        score += coverage_score
        
        # Security (25 points)
        security_clean = self.results["security_scan"].get("clean", False)
        security_score = 25 if security_clean else max(0, 25 - self.results["security_scan"].get("issues_found", 0) * 2)
        score += security_score
        
        self.results["overall_score"] = round(score, 1)
        
        return self.results["overall_score"]
    
    def generate_report(self):
        """Generate comprehensive quality report."""
        print("\n📋 Generating quality report...")
        
        # Create reports directory
        Path("quality-reports").mkdir(exist_ok=True)
        
        # Calculate overall score
        overall_score = self.calculate_overall_score()
        
        # Save detailed JSON report
        with open("quality-reports/quality-report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Generate human-readable report
        report_content = f"""
# Code Quality Report
Generated: {self.results['timestamp']}

## Overall Score: {overall_score}/100

### Syntax Check
- **Files Checked**: {self.results['syntax_check'].get('total_files', 0)}
- **Files with Errors**: {self.results['syntax_check'].get('files_with_errors', 0)}
- **Success Rate**: {self.results['syntax_check'].get('success_rate', 0):.1f}%

### Code Quality
"""
        
        for tool, result in self.results["code_quality"].items():
            status = "PASSED" if result.get("passed", False) else "FAILED"
            report_content += f"- **{tool.upper()}**: {status}\n"
        
        report_content += f"""
### Test Coverage
- **Tests Passed**: {'YES' if self.results['test_coverage'].get('tests_passed', False) else 'NO'}
- **Coverage**: {self.results['test_coverage'].get('coverage_percentage', 0):.1f}%

### Security Scan
- **Security Issues**: {self.results['security_scan'].get('issues_found', 0)}
- **Status**: {'CLEAN' if self.results['security_scan'].get('clean', False) else 'ISSUES FOUND'}

## Recommendations
"""
        
        if overall_score < 70:
            report_content += """
### Critical Actions Required
1. Fix all syntax errors immediately
2. Improve test coverage to at least 80%
3. Address all security issues
4. Run code formatting tools (black, isort)
"""
        elif overall_score < 85:
            report_content += """
### Improvements Recommended
1. Increase test coverage
2. Address remaining code quality issues
3. Regular security scans
"""
        else:
            report_content += """
### Excellent Quality!
1. Maintain current standards
2. Regular quality checks
3. Continue security monitoring
"""
        
        with open("quality-reports/quality-report.md", "w", encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Quality report saved to quality-reports/")
        print(f"   Overall Score: {overall_score}/100")
        
        return overall_score
    
    def run_full_check(self):
        """Run comprehensive quality check."""
        print("🚀 Starting comprehensive quality check...")
        print("=" * 50)
        
        # Run all checks
        self.check_syntax_errors()
        self.run_code_quality_checks()
        self.run_test_coverage()
        self.run_security_scan()
        
        # Generate report
        overall_score = self.generate_report()
        
        print("\n" + "=" * 50)
        print(f"🎯 Quality Check Complete!")
        print(f"   Overall Score: {overall_score}/100")
        
        if overall_score >= 85:
            print("   Grade: A - Excellent!")
        elif overall_score >= 70:
            print("   Grade: B - Good")
        elif overall_score >= 55:
            print("   Grade: C - Needs Improvement")
        else:
            print("   Grade: F - Critical Issues")
        
        return overall_score >= 70

def main():
    """Main function."""
    checker = QualityChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        # CI mode - fail if quality score is too low
        success = checker.run_full_check()
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        checker.run_full_check()

if __name__ == "__main__":
    main()