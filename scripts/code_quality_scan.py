#!/usr/bin/env python3
"""
Code Quality and Security Scanning Script for Prometheus AI Platform
Performs automated code quality checks, security scans, and compliance validation
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    file_path: str
    line_number: int
    severity: str  # error, warning, info
    category: str  # style, security, complexity, etc.
    message: str
    rule_id: str
    tool: str

@dataclass
class SecurityVulnerability:
    file_path: str
    line_number: int
    severity: str  # high, medium, low
    vulnerability_type: str
    description: str
    cwe_id: Optional[str] = None
    tool: str = "bandit"

@dataclass
class QualityReport:
    timestamp: str
    summary: Dict[str, int]
    quality_score: float
    security_score: float
    issues: List[QualityIssue]
    vulnerabilities: List[SecurityVulnerability]
    metrics: Dict[str, float]

class CodeQualityScanner:
    def __init__(self):
        # Handle cases where __file__ is not defined (e.g., when run via exec)
        try:
            self.project_root = Path(__file__).parent.parent
        except NameError:
            # Fallback to current working directory
            self.project_root = Path.cwd()
        
        self.services_dir = self.project_root / "services"
        self.quality_issues: List[QualityIssue] = []
        self.security_vulnerabilities: List[SecurityVulnerability] = []
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_quality_score": 8.0,
            "max_complexity": 10,
            "max_line_length": 120,
            "min_test_coverage": 80.0,
            "max_security_issues": 0
        }
    
    def run_pylint_scan(self, python_files: List[str]) -> List[QualityIssue]:
        """Run pylint code quality analysis"""
        logger.info("Running pylint code quality scan")
        issues = []
        
        if not python_files:
            return issues
        
        try:
            # Run pylint with JSON output
            cmd = [
                "python", "-m", "pylint",
                "--output-format=json",
                "--disable=C0114,C0115,C0116",  # Disable docstring warnings for now
                "--max-line-length=120"
            ] + python_files
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                try:
                    pylint_results = json.loads(result.stdout)
                    for item in pylint_results:
                        issues.append(QualityIssue(
                            file_path=item.get('path', ''),
                            line_number=item.get('line', 0),
                            severity=self.map_pylint_severity(item.get('type', '')),
                            category=item.get('category', 'style'),
                            message=item.get('message', ''),
                            rule_id=item.get('message-id', ''),
                            tool="pylint"
                        ))
                except json.JSONDecodeError:
                    logger.warning("Could not parse pylint JSON output")
        
        except FileNotFoundError:
            logger.warning("pylint not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pylint"])
            return self.run_pylint_scan(python_files)
        except Exception as e:
            logger.error(f"Error running pylint: {e}")
        
        return issues
    
    def run_flake8_scan(self, python_files: List[str]) -> List[QualityIssue]:
        """Run flake8 style checking"""
        logger.info("Running flake8 style scan")
        issues = []
        
        if not python_files:
            return issues
        
        try:
            cmd = [
                "python", "-m", "flake8",
                "--max-line-length=120",
                "--extend-ignore=E203,W503",
                "--format=json"
            ] + python_files
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse flake8 output (not JSON by default, need to parse text)
            for line in result.stdout.split('\n'):
                if line.strip():
                    match = re.match(r'([^:]+):(\d+):(\d+): (\w+) (.+)', line)
                    if match:
                        file_path, line_num, col, code, message = match.groups()
                        issues.append(QualityIssue(
                            file_path=file_path,
                            line_number=int(line_num),
                            severity=self.map_flake8_severity(code),
                            category="style",
                            message=message,
                            rule_id=code,
                            tool="flake8"
                        ))
        
        except FileNotFoundError:
            logger.warning("flake8 not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "flake8"])
            return self.run_flake8_scan(python_files)
        except Exception as e:
            logger.error(f"Error running flake8: {e}")
        
        return issues
    
    def run_bandit_security_scan(self, python_files: List[str]) -> List[SecurityVulnerability]:
        """Run bandit security vulnerability scan"""
        logger.info("Running bandit security scan")
        vulnerabilities = []
        
        if not python_files:
            return vulnerabilities
        
        try:
            cmd = [
                "python", "-m", "bandit",
                "-f", "json",
                "-r"
            ] + [str(self.services_dir)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                try:
                    bandit_results = json.loads(result.stdout)
                    for item in bandit_results.get('results', []):
                        vulnerabilities.append(SecurityVulnerability(
                            file_path=item.get('filename', ''),
                            line_number=item.get('line_number', 0),
                            severity=item.get('issue_severity', 'low').lower(),
                            vulnerability_type=item.get('test_name', ''),
                            description=item.get('issue_text', ''),
                            cwe_id=item.get('issue_cwe', {}).get('id'),
                            tool="bandit"
                        ))
                except json.JSONDecodeError:
                    logger.warning("Could not parse bandit JSON output")
        
        except FileNotFoundError:
            logger.warning("bandit not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "bandit"])
            return self.run_bandit_security_scan(python_files)
        except Exception as e:
            logger.error(f"Error running bandit: {e}")
        
        return vulnerabilities
    
    def run_eslint_scan(self, js_files: List[str]) -> List[QualityIssue]:
        """Run ESLint for JavaScript/TypeScript files"""
        logger.info("Running ESLint scan")
        issues = []
        
        if not js_files:
            return issues
        
        try:
            cmd = [
                "npx", "eslint",
                "--format", "json"
            ] + js_files
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                try:
                    eslint_results = json.loads(result.stdout)
                    for file_result in eslint_results:
                        for message in file_result.get('messages', []):
                            issues.append(QualityIssue(
                                file_path=file_result.get('filePath', ''),
                                line_number=message.get('line', 0),
                                severity=self.map_eslint_severity(message.get('severity', 1)),
                                category="style",
                                message=message.get('message', ''),
                                rule_id=message.get('ruleId', ''),
                                tool="eslint"
                            ))
                except json.JSONDecodeError:
                    logger.warning("Could not parse ESLint JSON output")
        
        except FileNotFoundError:
            logger.warning("ESLint not found, skipping JavaScript analysis")
        except Exception as e:
            logger.error(f"Error running ESLint: {e}")
        
        return issues
    
    def analyze_code_complexity(self, python_files: List[str]) -> Dict[str, float]:
        """Analyze code complexity metrics"""
        logger.info("Analyzing code complexity")
        complexity_metrics = {
            "average_complexity": 0.0,
            "max_complexity": 0.0,
            "total_functions": 0,
            "complex_functions": 0
        }
        
        total_complexity = 0
        function_count = 0
        max_complexity = 0
        complex_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to analyze complexity
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_count += 1
                        complexity = self.calculate_cyclomatic_complexity(node)
                        total_complexity += complexity
                        max_complexity = max(max_complexity, complexity)
                        
                        if complexity > self.quality_thresholds["max_complexity"]:
                            complex_functions += 1
                            
            except Exception as e:
                logger.warning(f"Could not analyze complexity for {file_path}: {e}")
        
        if function_count > 0:
            complexity_metrics["average_complexity"] = total_complexity / function_count
            complexity_metrics["max_complexity"] = max_complexity
            complexity_metrics["total_functions"] = function_count
            complexity_metrics["complex_functions"] = complex_functions
        
        return complexity_metrics
    
    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def discover_source_files(self) -> Tuple[List[str], List[str]]:
        """Discover Python and JavaScript source files"""
        python_files = []
        js_files = []
        
        # Scan services directory
        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir():
                # Python files
                for py_file in service_dir.rglob("*.py"):
                    if not any(exclude in str(py_file) for exclude in ["__pycache__", ".venv", "venv"]):
                        python_files.append(str(py_file))
                
                # JavaScript/TypeScript files
                for js_file in service_dir.rglob("*.js"):
                    if not any(exclude in str(js_file) for exclude in ["node_modules", "dist", "build"]):
                        js_files.append(str(js_file))
                
                for ts_file in service_dir.rglob("*.ts"):
                    if not any(exclude in str(ts_file) for exclude in ["node_modules", "dist", "build"]):
                        js_files.append(str(ts_file))
        
        return python_files, js_files
    
    def map_pylint_severity(self, pylint_type: str) -> str:
        """Map pylint message types to severity levels"""
        mapping = {
            'error': 'error',
            'warning': 'warning',
            'refactor': 'info',
            'convention': 'info',
            'info': 'info'
        }
        return mapping.get(pylint_type.lower(), 'info')
    
    def map_flake8_severity(self, code: str) -> str:
        """Map flake8 error codes to severity levels"""
        if code.startswith('E'):
            return 'error'
        elif code.startswith('W'):
            return 'warning'
        else:
            return 'info'
    
    def map_eslint_severity(self, severity: int) -> str:
        """Map ESLint severity levels"""
        if severity == 2:
            return 'error'
        elif severity == 1:
            return 'warning'
        else:
            return 'info'
    
    def calculate_quality_score(self, issues: List[QualityIssue], metrics: Dict[str, float]) -> float:
        """Calculate overall quality score (0-10)"""
        base_score = 10.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == 'error':
                base_score -= 0.5
            elif issue.severity == 'warning':
                base_score -= 0.2
            else:
                base_score -= 0.1
        
        # Deduct points for high complexity
        if metrics.get('average_complexity', 0) > self.quality_thresholds["max_complexity"]:
            base_score -= 1.0
        
        # Bonus for low complexity
        if metrics.get('average_complexity', 0) < 5.0:
            base_score += 0.5
        
        return max(0.0, min(10.0, base_score))
    
    def calculate_security_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate security score (0-10)"""
        base_score = 10.0
        
        for vuln in vulnerabilities:
            if vuln.severity == 'high':
                base_score -= 3.0
            elif vuln.severity == 'medium':
                base_score -= 1.0
            else:
                base_score -= 0.3
        
        return max(0.0, min(10.0, base_score))
    
    def run_comprehensive_scan(self) -> QualityReport:
        """Run comprehensive code quality and security scan"""
        logger.info("Starting comprehensive code quality and security scan")
        
        # Discover source files
        python_files, js_files = self.discover_source_files()
        logger.info(f"Found {len(python_files)} Python files and {len(js_files)} JS/TS files")
        
        # Run quality scans
        all_issues = []
        all_issues.extend(self.run_pylint_scan(python_files))
        all_issues.extend(self.run_flake8_scan(python_files))
        all_issues.extend(self.run_eslint_scan(js_files))
        
        # Run security scans
        vulnerabilities = self.run_bandit_security_scan(python_files)
        
        # Analyze complexity
        complexity_metrics = self.analyze_code_complexity(python_files)
        
        # Calculate scores
        quality_score = self.calculate_quality_score(all_issues, complexity_metrics)
        security_score = self.calculate_security_score(vulnerabilities)
        
        # Generate summary
        summary = {
            "total_issues": len(all_issues),
            "errors": len([i for i in all_issues if i.severity == 'error']),
            "warnings": len([i for i in all_issues if i.severity == 'warning']),
            "info": len([i for i in all_issues if i.severity == 'info']),
            "security_vulnerabilities": len(vulnerabilities),
            "high_severity_vulns": len([v for v in vulnerabilities if v.severity == 'high']),
            "files_scanned": len(python_files) + len(js_files)
        }
        
        return QualityReport(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            quality_score=quality_score,
            security_score=security_score,
            issues=all_issues,
            vulnerabilities=vulnerabilities,
            metrics=complexity_metrics
        )
    
    def save_quality_report(self, report: QualityReport):
        """Save quality report to file"""
        reports_dir = self.project_root / "quality-reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"quality_report_{timestamp}.json"
        
        # Convert dataclasses to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "summary": report.summary,
            "quality_score": report.quality_score,
            "security_score": report.security_score,
            "issues": [asdict(issue) for issue in report.issues],
            "vulnerabilities": [asdict(vuln) for vuln in report.vulnerabilities],
            "metrics": report.metrics
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Quality report saved to {report_file}")
        
        # Also save latest report
        latest_report = reports_dir / "latest_quality_report.json"
        with open(latest_report, 'w') as f:
            json.dump(report_dict, f, indent=2)

def main():
    """Main execution function"""
    scanner = CodeQualityScanner()
    
    try:
        # Run comprehensive scan
        report = scanner.run_comprehensive_scan()
        
        # Save report
        scanner.save_quality_report(report)
        
        # Print summary
        logger.info(f"Code quality scan completed:")
        logger.info(f"  Files scanned: {report.summary['files_scanned']}")
        logger.info(f"  Total issues: {report.summary['total_issues']}")
        logger.info(f"  Errors: {report.summary['errors']}")
        logger.info(f"  Warnings: {report.summary['warnings']}")
        logger.info(f"  Security vulnerabilities: {report.summary['security_vulnerabilities']}")
        logger.info(f"  Quality score: {report.quality_score:.1f}/10")
        logger.info(f"  Security score: {report.security_score:.1f}/10")
        
        # Check quality gates
        quality_gate_passed = True
        
        if report.quality_score < scanner.quality_thresholds["min_quality_score"]:
            logger.error(f"QUALITY GATE FAILED: Quality score {report.quality_score:.1f} below threshold {scanner.quality_thresholds['min_quality_score']}")
            quality_gate_passed = False
        
        if report.summary['high_severity_vulns'] > scanner.quality_thresholds["max_security_issues"]:
            logger.error(f"SECURITY GATE FAILED: {report.summary['high_severity_vulns']} high-severity vulnerabilities found")
            quality_gate_passed = False
        
        if report.metrics.get('max_complexity', 0) > scanner.quality_thresholds["max_complexity"]:
            logger.error(f"COMPLEXITY GATE FAILED: Maximum complexity {report.metrics['max_complexity']} exceeds threshold {scanner.quality_thresholds['max_complexity']}")
            quality_gate_passed = False
        
        # Exit with appropriate code
        if not quality_gate_passed:
            logger.error("Code quality gates failed!")
            sys.exit(1)
        else:
            logger.info("All quality gates passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Code quality scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()