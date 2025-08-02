#!/usr/bin/env python3
"""
Comprehensive security scanner for the Prometheus AI platform.
Uses Bandit and other security tools to identify vulnerabilities.
"""

from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

class SecurityScanner:
    """Comprehensive security scanning system."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "bandit": {},
            "safety": {},
            "semgrep": {},
            "summary": {}
        }
    
    def run_bandit_scan(self):
        """Run Bandit security scan."""
        print("Running Bandit security scan...")
        
        try:
            # Run bandit with JSON output
            result = subprocess.run([
                "bandit", 
                "-r", "services",
                "-f", "json",
                "-o", "security-reports/bandit-report.json",
                "--skip", "B101,B601"  # Skip assert and shell injection for demos
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("✅ Bandit scan completed - No issues found")
                self.results["bandit"]["status"] = "clean"
            else:
                print(f"⚠️  Bandit found security issues (exit code: {result.returncode})")
                self.results["bandit"]["status"] = "issues_found"
                
                # Try to load the JSON report
                try:
                    with open("security-reports/bandit-report.json", "r") as f:
                        bandit_data = json.load(f)
                        self.results["bandit"]["issues"] = len(bandit_data.get("results", []))
                        self.results["bandit"]["severity_breakdown"] = self._analyze_bandit_results(bandit_data)
                except Exception as e:
                    print(f"Could not parse Bandit results: {e}")
            
        except FileNotFoundError:
            print("❌ Bandit not installed. Installing...")
            subprocess.run(["pip", "install", "bandit"], check=True)
            return self.run_bandit_scan()
        except Exception as e:
            print(f"❌ Error running Bandit: {e}")
            self.results["bandit"]["status"] = "error"
            self.results["bandit"]["error"] = str(e)
    
    def run_safety_scan(self):
        """Run Safety scan for known vulnerabilities in dependencies."""
        print("Running Safety scan for dependency vulnerabilities...")
        
        try:
            result = subprocess.run([
                "safety", "check",
                "--json",
                "--output", "security-reports/safety-report.json"
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("✅ Safety scan completed - No known vulnerabilities")
                self.results["safety"]["status"] = "clean"
            else:
                print("⚠️  Safety found known vulnerabilities")
                self.results["safety"]["status"] = "vulnerabilities_found"
                
                try:
                    with open("security-reports/safety-report.json", "r") as f:
                        safety_data = json.load(f)
                        self.results["safety"]["vulnerabilities"] = len(safety_data)
                except Exception as e:
                    print(f"Could not parse Safety results: {e}")
                    
        except FileNotFoundError:
            print("❌ Safety not installed. Installing...")
            subprocess.run(["pip", "install", "safety"], check=True)
            return self.run_safety_scan()
        except Exception as e:
            print(f"❌ Error running Safety: {e}")
            self.results["safety"]["status"] = "error"
            self.results["safety"]["error"] = str(e)
    
    def run_semgrep_scan(self):
        """Run Semgrep static analysis."""
        print("Running Semgrep static analysis...")
        
        try:
            result = subprocess.run([
                "semgrep",
                "--config=auto",
                "--json",
                "--output=security-reports/semgrep-report.json",
                "services/"
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("✅ Semgrep scan completed")
                self.results["semgrep"]["status"] = "completed"
                
                try:
                    with open("security-reports/semgrep-report.json", "r") as f:
                        semgrep_data = json.load(f)
                        self.results["semgrep"]["findings"] = len(semgrep_data.get("results", []))
                except Exception as e:
                    print(f"Could not parse Semgrep results: {e}")
                    
        except FileNotFoundError:
            print("⚠️  Semgrep not installed. Skipping...")
            self.results["semgrep"]["status"] = "not_available"
        except Exception as e:
            print(f"❌ Error running Semgrep: {e}")
            self.results["semgrep"]["status"] = "error"
            self.results["semgrep"]["error"] = str(e)
    
    def check_secrets(self):
        """Check for hardcoded secrets and credentials."""
        print("Checking for hardcoded secrets...")
        
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'aws_access_key_id\s*=\s*["\'][^"\']+["\']',
            r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']'
        ]
        
        import re

        secrets_found = []
        
        for root, dirs, files in os.walk("services"):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for pattern in secret_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                secrets_found.append({
                                    "file": str(file_path),
                                    "pattern": pattern,
                                    "line": content[:match.start()].count('\n') + 1
                                })
                    except Exception:
                        continue
        
        if secrets_found:
            print(f"⚠️  Found {len(secrets_found)} potential hardcoded secrets")
            self.results["secrets"] = {
                "status": "found",
                "count": len(secrets_found),
                "details": secrets_found
            }
        else:
            print("✅ No hardcoded secrets detected")
            self.results["secrets"] = {"status": "clean"}
    
    def _analyze_bandit_results(self, bandit_data):
        """Analyze Bandit results by severity."""
        severity_count = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for result in bandit_data.get("results", []):
            severity = result.get("issue_severity", "UNKNOWN")
            if severity in severity_count:
                severity_count[severity] += 1
        
        return severity_count
    
    def generate_report(self):
        """Generate comprehensive security report."""
        print("\nGenerating security report...")
        
        # Create security reports directory
        Path("security-reports").mkdir(exist_ok=True)
        
        # Calculate overall security score
        score = 100
        issues = 0
        
        if self.results["bandit"].get("status") == "issues_found":
            bandit_issues = self.results["bandit"].get("issues", 0)
            score -= min(bandit_issues * 5, 30)
            issues += bandit_issues
        
        if self.results["safety"].get("status") == "vulnerabilities_found":
            safety_vulns = self.results["safety"].get("vulnerabilities", 0)
            score -= min(safety_vulns * 10, 40)
            issues += safety_vulns
        
        if self.results["secrets"].get("status") == "found":
            secret_count = self.results["secrets"].get("count", 0)
            score -= min(secret_count * 15, 50)
            issues += secret_count
        
        self.results["summary"] = {
            "security_score": max(score, 0),
            "total_issues": issues,
            "risk_level": self._get_risk_level(score)
        }
        
        # Save detailed report
        with open("security-reports/security-summary.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Generate human-readable report
        self._generate_human_report()
        
        print(f"✅ Security report generated")
        print(f"   Security Score: {self.results['summary']['security_score']}/100")
        print(f"   Risk Level: {self.results['summary']['risk_level']}")
        print(f"   Total Issues: {self.results['summary']['total_issues']}")
    
    def _get_risk_level(self, score):
        """Determine risk level based on score."""
        if score >= 90:
            return "LOW"
        elif score >= 70:
            return "MEDIUM"
        elif score >= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_human_report(self):
        """Generate human-readable security report."""
        report_content = f"""
# Security Scan Report
Generated: {self.results['timestamp']}

## Summary
- **Security Score**: {self.results['summary']['security_score']}/100
- **Risk Level**: {self.results['summary']['risk_level']}
- **Total Issues**: {self.results['summary']['total_issues']}

## Bandit Static Analysis
- **Status**: {self.results['bandit'].get('status', 'Not run')}
- **Issues Found**: {self.results['bandit'].get('issues', 0)}

## Dependency Vulnerabilities (Safety)
- **Status**: {self.results['safety'].get('status', 'Not run')}
- **Vulnerabilities**: {self.results['safety'].get('vulnerabilities', 0)}

## Secret Detection
- **Status**: {self.results['secrets'].get('status', 'Not run')}
- **Potential Secrets**: {self.results['secrets'].get('count', 0)}

## Recommendations
"""
        
        if self.results['summary']['security_score'] < 70:
            report_content += """
### High Priority Actions
1. Review and fix all HIGH severity Bandit issues
2. Update dependencies with known vulnerabilities
3. Remove any hardcoded secrets and use environment variables
4. Implement proper secret management (e.g., Azure Key Vault, AWS Secrets Manager)
"""
        
        if self.results['summary']['security_score'] < 90:
            report_content += """
### Medium Priority Actions
1. Review and fix MEDIUM severity Bandit issues
2. Implement additional input validation
3. Add security headers to web services
4. Regular security dependency updates
"""
        
        report_content += """
### Best Practices
1. Regular security scans in CI/CD pipeline
2. Dependency vulnerability monitoring
3. Code review for security issues
4. Security training for development team
"""
        
        with open("security-reports/security-report.md", "w") as f:
            f.write(report_content)
    
    def run_full_scan(self):
        """Run comprehensive security scan."""
        print("🔒 Starting comprehensive security scan...")
        
        # Create reports directory
        Path("security-reports").mkdir(exist_ok=True)
        
        # Run all scans
        self.run_bandit_scan()
        self.run_safety_scan()
        self.run_semgrep_scan()
        self.check_secrets()
        
        # Generate report
        self.generate_report()
        
        print("\n🔒 Security scan completed!")
        return self.results['summary']['security_score'] >= 70

def main():
    """Main function."""
    scanner = SecurityScanner()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        # CI mode - fail if security score is too low
        success = scanner.run_full_scan()
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        scanner.run_full_scan()
        
        print("\nSecurity scan results saved to security-reports/")
        print("Review the detailed report at security-reports/security-report.md")

if __name__ == "__main__":
    main()