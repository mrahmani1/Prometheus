#!/usr/bin/env python3
"""
Quality Gates Orchestrator

This script orchestrates all quality assurance tools and generates
a comprehensive quality report for the Prometheus AI Platform.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures

# Import quality assurance modules
try:
    from automated_quality_gates import ProductionQualityGates
    from static_analysis_tools import GeologicalStaticAnalyzer
    from scientific_accuracy_validator import ScientificAccuracyValidator
    from performance_benchmarking import GeologicalPerformanceBenchmarker
    from compliance_framework import ScientificComplianceFramework
except ImportError as e:
    print(f"Warning: Could not import quality assurance module: {e}")
    print("Ensure all quality assurance modules are in the same directory")


@dataclass
class QualityGatesSummary:
    """Summary of all quality gates execution"""
    timestamp: datetime
    project_path: str
    overall_status: str
    overall_score: float
    execution_time_seconds: float
    
    # Individual tool results
    automated_gates_passed: bool
    static_analysis_passed: bool
    scientific_validation_passed: bool
    performance_benchmarks_passed: bool
    compliance_assessment_passed: bool
    
    # Detailed scores
    code_quality_score: float
    scientific_accuracy_score: float
    performance_score: float
    compliance_score: float
    
    # Issue counts
    critical_issues: int
    major_issues: int
    minor_issues: int
    
    # Recommendations
    top_recommendations: List[str]
    
    # Certification status
    production_ready: bool
    certification_status: str


class QualityGatesOrchestrator:
    """
    Orchestrates all quality assurance tools and generates comprehensive reports
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        default_config = {
            "parallel_execution": True,
            "timeout_minutes": 60,
            "quality_thresholds": {
                "overall_minimum": 0.85,
                "code_quality_minimum": 0.80,
                "scientific_accuracy_minimum": 0.85,
                "performance_minimum": 0.75,
                "compliance_minimum": 0.90
            },
            "tools_enabled": {
                "automated_gates": True,
                "static_analysis": True,
                "scientific_validation": True,
                "performance_benchmarking": True,
                "compliance_assessment": True
            },
            "report_formats": ["json", "html", "pdf"],
            "notification_settings": {
                "email_on_failure": False,
                "slack_webhook": None,
                "teams_webhook": None
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def run_all_quality_gates(self, project_path: str) -> QualityGatesSummary:
        """Run all quality gates and generate comprehensive report"""
        print("🚀 Starting Comprehensive Quality Gates Assessment...")
        print(f"📁 Project Path: {project_path}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Create quality assurance directory
        qa_dir = os.path.join(project_path, ".kiro", "quality-assurance")
        os.makedirs(qa_dir, exist_ok=True)
        
        # Run quality gates
        if self.config["parallel_execution"]:
            self._run_parallel_quality_gates(project_path)
        else:
            self._run_sequential_quality_gates(project_path)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate comprehensive summary
        summary = self._generate_quality_summary(project_path, execution_time)
        
        # Save results
        self._save_comprehensive_report(summary)
        
        # Generate additional report formats
        self._generate_additional_reports(summary)
        
        # Print final summary
        self._print_final_summary(summary)
        
        # Send notifications if configured
        self._send_notifications(summary)
        
        return summary
    
    def _run_parallel_quality_gates(self, project_path: str):
        """Run quality gates in parallel for faster execution"""
        print("⚡ Running quality gates in parallel...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            # Submit all quality gate tasks
            if self.config["tools_enabled"]["automated_gates"]:
                futures["automated_gates"] = executor.submit(
                    self._run_automated_gates, project_path
                )
            
            if self.config["tools_enabled"]["static_analysis"]:
                futures["static_analysis"] = executor.submit(
                    self._run_static_analysis, project_path
                )
            
            if self.config["tools_enabled"]["scientific_validation"]:
                futures["scientific_validation"] = executor.submit(
                    self._run_scientific_validation, project_path
                )
            
            if self.config["tools_enabled"]["performance_benchmarking"]:
                futures["performance_benchmarking"] = executor.submit(
                    self._run_performance_benchmarking, project_path
                )
            
            if self.config["tools_enabled"]["compliance_assessment"]:
                futures["compliance_assessment"] = executor.submit(
                    self._run_compliance_assessment, project_path
                )
            
            # Collect results with timeout
            timeout_seconds = self.config["timeout_minutes"] * 60
            
            for tool_name, future in futures.items():
                try:
                    result = future.result(timeout=timeout_seconds)
                    self.results[tool_name] = result
                    print(f"  ✅ {tool_name} completed successfully")
                except concurrent.futures.TimeoutError:
                    print(f"  ⏰ {tool_name} timed out after {self.config['timeout_minutes']} minutes")
                    self.results[tool_name] = {"error": "timeout", "passed": False}
                except Exception as e:
                    print(f"  ❌ {tool_name} failed: {str(e)}")
                    self.results[tool_name] = {"error": str(e), "passed": False}
    
    def _run_sequential_quality_gates(self, project_path: str):
        """Run quality gates sequentially"""
        print("🔄 Running quality gates sequentially...")
        
        if self.config["tools_enabled"]["automated_gates"]:
            print("  🔍 Running automated quality gates...")
            self.results["automated_gates"] = self._run_automated_gates(project_path)
        
        if self.config["tools_enabled"]["static_analysis"]:
            print("  🔧 Running static analysis...")
            self.results["static_analysis"] = self._run_static_analysis(project_path)
        
        if self.config["tools_enabled"]["scientific_validation"]:
            print("  🔬 Running scientific validation...")
            self.results["scientific_validation"] = self._run_scientific_validation(project_path)
        
        if self.config["tools_enabled"]["performance_benchmarking"]:
            print("  ⚡ Running performance benchmarking...")
            self.results["performance_benchmarking"] = self._run_performance_benchmarking(project_path)
        
        if self.config["tools_enabled"]["compliance_assessment"]:
            print("  📋 Running compliance assessment...")
            self.results["compliance_assessment"] = self._run_compliance_assessment(project_path)
    
    def _run_automated_gates(self, project_path: str) -> Dict[str, Any]:
        """Run automated quality gates"""
        try:
            quality_gates = ProductionQualityGates()
            results = quality_gates.run_all_quality_gates(project_path)
            
            return {
                "passed": all(r.passed for r in results),
                "overall_score": sum(r.score for r in results) / len(results) if results else 0,
                "results": [asdict(r) for r in results],
                "total_violations": sum(len(r.violations) for r in results)
            }
        except Exception as e:
            return {"error": str(e), "passed": False, "overall_score": 0}
    
    def _run_static_analysis(self, project_path: str) -> Dict[str, Any]:
        """Run static analysis"""
        try:
            analyzer = GeologicalStaticAnalyzer()
            report = analyzer.analyze_project(project_path)
            
            return {
                "passed": report.critical_issues == 0 and report.overall_score >= 0.8,
                "overall_score": report.overall_score,
                "critical_issues": report.critical_issues,
                "major_issues": report.major_issues,
                "minor_issues": report.minor_issues,
                "geological_algorithms_analyzed": report.geological_algorithms_analyzed
            }
        except Exception as e:
            return {"error": str(e), "passed": False, "overall_score": 0}
    
    def _run_scientific_validation(self, project_path: str) -> Dict[str, Any]:
        """Run scientific accuracy validation"""
        try:
            validator = ScientificAccuracyValidator()
            report = validator.validate_project(project_path)
            
            return {
                "passed": (report.overall_accuracy >= 0.8 and 
                          report.physics_compliance_rate >= 0.9 and
                          len(report.failed_validations) == 0),
                "overall_accuracy": report.overall_accuracy,
                "physics_compliance_rate": report.physics_compliance_rate,
                "algorithms_tested": report.total_algorithms_tested,
                "benchmarks_used": report.total_benchmarks_used,
                "failed_validations": len(report.failed_validations)
            }
        except Exception as e:
            return {"error": str(e), "passed": False, "overall_accuracy": 0}
    
    def _run_performance_benchmarking(self, project_path: str) -> Dict[str, Any]:
        """Run performance benchmarking"""
        try:
            benchmarker = GeologicalPerformanceBenchmarker()
            report = benchmarker.benchmark_project(project_path)
            
            performance_acceptable = (
                report.performance_summary["average_processing_time_seconds"] < 300 and
                report.performance_summary["average_memory_usage_mb"] < 8000 and
                len(report.bottlenecks_identified) == 0
            )
            
            return {
                "passed": performance_acceptable,
                "performance_summary": report.performance_summary,
                "algorithms_tested": report.total_algorithms_tested,
                "bottlenecks_count": len(report.bottlenecks_identified),
                "load_test_results": len(report.load_test_results),
                "scalability_results": len(report.scalability_results)
            }
        except Exception as e:
            return {"error": str(e), "passed": False, "performance_summary": {}}
    
    def _run_compliance_assessment(self, project_path: str) -> Dict[str, Any]:
        """Run compliance assessment"""
        try:
            compliance_framework = ScientificComplianceFramework()
            report = compliance_framework.assess_compliance(project_path)
            
            compliance_acceptable = (
                report.overall_compliance_score >= 0.85 and
                len(report.critical_violations) == 0 and
                "NOT_CERTIFIED" not in report.certification_status
            )
            
            return {
                "passed": compliance_acceptable,
                "overall_compliance_score": report.overall_compliance_score,
                "certification_status": report.certification_status,
                "critical_violations": len(report.critical_violations),
                "regulatory_compliance": report.regulatory_compliance,
                "scientific_standards_compliance": report.scientific_standards_compliance
            }
        except Exception as e:
            return {"error": str(e), "passed": False, "overall_compliance_score": 0}
    
    def _generate_quality_summary(self, project_path: str, execution_time: float) -> QualityGatesSummary:
        """Generate comprehensive quality summary"""
        
        # Extract individual tool results
        automated_gates = self.results.get("automated_gates", {})
        static_analysis = self.results.get("static_analysis", {})
        scientific_validation = self.results.get("scientific_validation", {})
        performance_benchmarking = self.results.get("performance_benchmarking", {})
        compliance_assessment = self.results.get("compliance_assessment", {})
        
        # Calculate individual scores
        code_quality_score = (
            automated_gates.get("overall_score", 0) * 0.6 +
            static_analysis.get("overall_score", 0) * 0.4
        )
        
        scientific_accuracy_score = scientific_validation.get("overall_accuracy", 0)
        
        performance_score = self._calculate_performance_score(performance_benchmarking)
        
        compliance_score = compliance_assessment.get("overall_compliance_score", 0)
        
        # Calculate overall score
        overall_score = (
            code_quality_score * 0.25 +
            scientific_accuracy_score * 0.30 +
            performance_score * 0.20 +
            compliance_score * 0.25
        )
        
        # Determine overall status
        thresholds = self.config["quality_thresholds"]
        
        overall_status = "PASSED"
        if overall_score < thresholds["overall_minimum"]:
            overall_status = "FAILED"
        elif code_quality_score < thresholds["code_quality_minimum"]:
            overall_status = "FAILED"
        elif scientific_accuracy_score < thresholds["scientific_accuracy_minimum"]:
            overall_status = "FAILED"
        elif performance_score < thresholds["performance_minimum"]:
            overall_status = "FAILED"
        elif compliance_score < thresholds["compliance_minimum"]:
            overall_status = "FAILED"
        
        # Count issues
        critical_issues = (
            automated_gates.get("total_violations", 0) +
            static_analysis.get("critical_issues", 0) +
            compliance_assessment.get("critical_violations", 0)
        )
        
        major_issues = (
            static_analysis.get("major_issues", 0) +
            scientific_validation.get("failed_validations", 0) +
            performance_benchmarking.get("bottlenecks_count", 0)
        )
        
        minor_issues = static_analysis.get("minor_issues", 0)
        
        # Generate top recommendations
        top_recommendations = self._generate_top_recommendations()
        
        # Determine production readiness
        production_ready = (
            overall_status == "PASSED" and
            critical_issues == 0 and
            overall_score >= 0.90
        )
        
        # Determine certification status
        if production_ready:
            certification_status = "PRODUCTION_READY"
        elif overall_status == "PASSED":
            certification_status = "CONDITIONALLY_APPROVED"
        else:
            certification_status = "NOT_APPROVED"
        
        return QualityGatesSummary(
            timestamp=datetime.now(),
            project_path=project_path,
            overall_status=overall_status,
            overall_score=overall_score,
            execution_time_seconds=execution_time,
            
            automated_gates_passed=automated_gates.get("passed", False),
            static_analysis_passed=static_analysis.get("passed", False),
            scientific_validation_passed=scientific_validation.get("passed", False),
            performance_benchmarks_passed=performance_benchmarking.get("passed", False),
            compliance_assessment_passed=compliance_assessment.get("passed", False),
            
            code_quality_score=code_quality_score,
            scientific_accuracy_score=scientific_accuracy_score,
            performance_score=performance_score,
            compliance_score=compliance_score,
            
            critical_issues=critical_issues,
            major_issues=major_issues,
            minor_issues=minor_issues,
            
            top_recommendations=top_recommendations,
            
            production_ready=production_ready,
            certification_status=certification_status
        )
    
    def _calculate_performance_score(self, performance_results: Dict[str, Any]) -> float:
        """Calculate normalized performance score"""
        if "error" in performance_results:
            return 0.0
        
        summary = performance_results.get("performance_summary", {})
        
        # Normalize metrics to 0-1 scale
        processing_time = summary.get("average_processing_time_seconds", 300)
        memory_usage = summary.get("average_memory_usage_mb", 8000)
        throughput = summary.get("average_throughput_samples_per_second", 0)
        
        # Calculate component scores (higher is better)
        time_score = max(0, min(1, (300 - processing_time) / 300))
        memory_score = max(0, min(1, (8000 - memory_usage) / 8000))
        throughput_score = min(1, throughput / 1000)  # Normalize to 1000 samples/sec
        
        # Weighted average
        return (time_score * 0.4 + memory_score * 0.3 + throughput_score * 0.3)
    
    def _generate_top_recommendations(self) -> List[str]:
        """Generate top recommendations based on all results"""
        recommendations = []
        
        # Automated gates recommendations
        automated_gates = self.results.get("automated_gates", {})
        if not automated_gates.get("passed", True):
            recommendations.append("Address automated quality gate violations - remove placeholder code and implement complete functionality")
        
        # Static analysis recommendations
        static_analysis = self.results.get("static_analysis", {})
        if static_analysis.get("critical_issues", 0) > 0:
            recommendations.append("Fix critical static analysis issues - security vulnerabilities and code quality problems")
        
        # Scientific validation recommendations
        scientific_validation = self.results.get("scientific_validation", {})
        if scientific_validation.get("overall_accuracy", 1) < 0.8:
            recommendations.append("Improve scientific accuracy - validate algorithms against geological benchmarks")
        
        # Performance recommendations
        performance_benchmarking = self.results.get("performance_benchmarking", {})
        if performance_benchmarking.get("bottlenecks_count", 0) > 0:
            recommendations.append("Optimize performance bottlenecks - improve algorithm efficiency and resource usage")
        
        # Compliance recommendations
        compliance_assessment = self.results.get("compliance_assessment", {})
        if compliance_assessment.get("critical_violations", 0) > 0:
            recommendations.append("Address compliance violations - ensure regulatory and scientific standards compliance")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("All quality gates passed - consider additional optimization and monitoring")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _save_comprehensive_report(self, summary: QualityGatesSummary):
        """Save comprehensive quality report"""
        report_path = ".kiro/quality-assurance/comprehensive-quality-report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Combine summary with detailed results
        comprehensive_report = {
            "summary": asdict(summary),
            "detailed_results": self.results,
            "configuration": self.config,
            "metadata": {
                "report_version": "1.0",
                "tools_used": list(self.results.keys()),
                "execution_environment": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "working_directory": os.getcwd()
                }
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"📊 Comprehensive quality report saved: {report_path}")
    
    def _generate_additional_reports(self, summary: QualityGatesSummary):
        """Generate additional report formats"""
        if "html" in self.config["report_formats"]:
            self._generate_html_report(summary)
        
        if "pdf" in self.config["report_formats"]:
            self._generate_pdf_report(summary)
    
    def _generate_html_report(self, summary: QualityGatesSummary):
        """Generate HTML quality report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Prometheus AI Platform - Quality Gates Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-passed {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Prometheus AI Platform - Quality Gates Report</h1>
        <p><strong>Generated:</strong> {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Project:</strong> {summary.project_path}</p>
        <p><strong>Status:</strong> <span class="{'status-passed' if summary.overall_status == 'PASSED' else 'status-failed'}">{summary.overall_status}</span></p>
        <p><strong>Overall Score:</strong> <span class="score">{summary.overall_score:.2f}</span></p>
    </div>
    
    <div class="section">
        <h2>Quality Metrics</h2>
        <div class="metric">
            <strong>Code Quality:</strong> {summary.code_quality_score:.2f}
        </div>
        <div class="metric">
            <strong>Scientific Accuracy:</strong> {summary.scientific_accuracy_score:.2f}
        </div>
        <div class="metric">
            <strong>Performance:</strong> {summary.performance_score:.2f}
        </div>
        <div class="metric">
            <strong>Compliance:</strong> {summary.compliance_score:.2f}
        </div>
    </div>
    
    <div class="section">
        <h2>Tool Results</h2>
        <ul>
            <li>Automated Gates: {'✅ PASSED' if summary.automated_gates_passed else '❌ FAILED'}</li>
            <li>Static Analysis: {'✅ PASSED' if summary.static_analysis_passed else '❌ FAILED'}</li>
            <li>Scientific Validation: {'✅ PASSED' if summary.scientific_validation_passed else '❌ FAILED'}</li>
            <li>Performance Benchmarks: {'✅ PASSED' if summary.performance_benchmarks_passed else '❌ FAILED'}</li>
            <li>Compliance Assessment: {'✅ PASSED' if summary.compliance_assessment_passed else '❌ FAILED'}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Issues Summary</h2>
        <p><strong>Critical Issues:</strong> {summary.critical_issues}</p>
        <p><strong>Major Issues:</strong> {summary.major_issues}</p>
        <p><strong>Minor Issues:</strong> {summary.minor_issues}</p>
    </div>
    
    <div class="recommendations">
        <h2>Top Recommendations</h2>
        <ul>
        {''.join(f'<li>{rec}</li>' for rec in summary.top_recommendations)}
        </ul>
    </div>
    
    <div class="section">
        <h2>Certification Status</h2>
        <p><strong>Production Ready:</strong> {'Yes' if summary.production_ready else 'No'}</p>
        <p><strong>Certification:</strong> {summary.certification_status}</p>
    </div>
</body>
</html>
        """
        
        html_path = ".kiro/quality-assurance/quality-report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML report generated: {html_path}")
    
    def _generate_pdf_report(self, summary: QualityGatesSummary):
        """Generate PDF quality report"""
        try:
            # This would require a PDF generation library like reportlab
            # For now, just create a placeholder
            pdf_path = ".kiro/quality-assurance/quality-report.pdf"
            with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
                f.write(f"Prometheus AI Platform - Quality Gates Report\n")
                f.write(f"Generated: {summary.timestamp}\n")
                f.write(f"Overall Status: {summary.overall_status}\n")
                f.write(f"Overall Score: {summary.overall_score:.2f}\n")
                f.write(f"Production Ready: {summary.production_ready}\n")
            
            print(f"📄 PDF report placeholder generated: {pdf_path.replace('.pdf', '.txt')}")
            
        except Exception as e:
            print(f"⚠️  PDF generation failed: {str(e)}")
    
    def _print_final_summary(self, summary: QualityGatesSummary):
        """Print final quality gates summary"""
        print(f"\n" + "="*80)
        print(f"🎯 FINAL QUALITY GATES SUMMARY")
        print(f"="*80)
        
        status_emoji = "✅" if summary.overall_status == "PASSED" else "❌"
        print(f"{status_emoji} Overall Status: {summary.overall_status}")
        print(f"📊 Overall Score: {summary.overall_score:.2f}")
        print(f"⏱️  Execution Time: {summary.execution_time_seconds:.1f} seconds")
        
        print(f"\n📈 Quality Metrics:")
        print(f"   Code Quality: {summary.code_quality_score:.2f}")
        print(f"   Scientific Accuracy: {summary.scientific_accuracy_score:.2f}")
        print(f"   Performance: {summary.performance_score:.2f}")
        print(f"   Compliance: {summary.compliance_score:.2f}")
        
        print(f"\n🔧 Tool Results:")
        tools = [
            ("Automated Gates", summary.automated_gates_passed),
            ("Static Analysis", summary.static_analysis_passed),
            ("Scientific Validation", summary.scientific_validation_passed),
            ("Performance Benchmarks", summary.performance_benchmarks_passed),
            ("Compliance Assessment", summary.compliance_assessment_passed)
        ]
        
        for tool_name, passed in tools:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {tool_name}: {status}")
        
        print(f"\n🚨 Issues:")
        print(f"   Critical: {summary.critical_issues}")
        print(f"   Major: {summary.major_issues}")
        print(f"   Minor: {summary.minor_issues}")
        
        print(f"\n🏆 Certification:")
        print(f"   Production Ready: {'Yes' if summary.production_ready else 'No'}")
        print(f"   Status: {summary.certification_status}")
        
        if summary.top_recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(summary.top_recommendations, 1):
                print(f"   {i}. {rec}")
        
        print(f"\n" + "="*80)
    
    def _send_notifications(self, summary: QualityGatesSummary):
        """Send notifications based on configuration"""
        if summary.overall_status == "FAILED" and self.config["notification_settings"]["email_on_failure"]:
            print("📧 Sending failure notification email...")
            # Email notification implementation would go here
        
        if self.config["notification_settings"]["slack_webhook"]:
            print("📱 Sending Slack notification...")
            # Slack notification implementation would go here
        
        if self.config["notification_settings"]["teams_webhook"]:
            print("💬 Sending Teams notification...")
            # Teams notification implementation would go here


def main():
    """Main entry point for quality gates orchestrator"""
    if len(sys.argv) < 2:
        print("Usage: python run-quality-gates.py <project_path> [config_path]")
        print("\nThis script runs all quality assurance tools and generates comprehensive reports.")
        print("It includes:")
        print("  • Automated quality gates")
        print("  • Static analysis")
        print("  • Scientific accuracy validation")
        print("  • Performance benchmarking")
        print("  • Compliance assessment")
        sys.exit(1)
    
    project_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(project_path):
        print(f"Error: Project path '{project_path}' does not exist")
        sys.exit(1)
    
    # Initialize and run quality gates orchestrator
    orchestrator = QualityGatesOrchestrator(config_path)
    summary = orchestrator.run_all_quality_gates(project_path)
    
    # Exit with appropriate code
    sys.exit(0 if summary.overall_status == "PASSED" else 1)


if __name__ == "__main__":
    main()