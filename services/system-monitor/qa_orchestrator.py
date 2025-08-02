#!/usr/bin/env python3
"""
QA Orchestrator for Prometheus AI Platform
Coordinates automated QA processes and provides centralized monitoring
"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    component: str
    status: str  # passed, failed, warning
    score: float
    duration: float
    details: Dict[str, Any]
    timestamp: str
    issues_count: int = 0
    critical_issues: int = 0


@dataclass
class QAOrchestrationReport:
    timestamp: str
    overall_status: str
    overall_score: float
    total_duration: float
    component_results: List[QAResult]
    summary: Dict[str, Any]
    recommendations: List[str]


class QAOrchestrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.reports_dir = self.project_root / "qa-reports"
        self.reports_dir.mkdir(exist_ok=True)

        # QA component configurations
        self.qa_components = {
            "automated_tests": {
                "script": "run_automated_tests.py",
                "timeout": 1800,  # 30 minutes
                "weight": 0.4,
                "critical": True,
            },
            "model_validation": {
                "script": "validate_models.py",
                "timeout": 3600,  # 60 minutes
                "weight": 0.3,
                "critical": True,
            },
            "code_quality": {
                "script": "code_quality_scan.py",
                "timeout": 600,  # 10 minutes
                "weight": 0.2,
                "critical": False,
            },
            "documentation": {
                "script": "generate_validate_docs.py",
                "timeout": 900,  # 15 minutes
                "weight": 0.1,
                "critical": False,
            },
        }

        # Quality gates
        self.quality_gates = {
            "min_overall_score": 7.5,
            "max_critical_failures": 0,
            "min_test_success_rate": 0.95,
            "min_model_accuracy": 0.80,
            "min_code_quality": 7.0,
            "min_documentation_coverage": 0.85,
        }

    async def run_qa_component(
        self, component_name: str, config: Dict[str, Any]
    ) -> QAResult:
        """Run a single QA component"""
        logger.info(f"Running QA component: {component_name}")
        start_time = datetime.now()

        script_path = self.scripts_dir / config["script"]

        try:
            # Run the QA script
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=config["timeout"]
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Parse results based on component type
            details = await self.parse_component_results(
                component_name, stdout.decode(), stderr.decode()
            )

            # Determine status and score
            status = "passed" if process.returncode == 0 else "failed"
            score = self.calculate_component_score(component_name, details, status)

            # Count issues
            issues_count = details.get("total_issues", 0)
            critical_issues = details.get("critical_issues", 0)

            return QAResult(
                component=component_name,
                status=status,
                score=score,
                duration=duration,
                details=details,
                timestamp=datetime.now().isoformat(),
                issues_count=issues_count,
                critical_issues=critical_issues,
            )

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"QA component {component_name} timed out after {duration:.2f}s"
            )
            return QAResult(
                component=component_name,
                status="failed",
                score=0.0,
                duration=duration,
                details={"error": "Timeout", "timeout_seconds": config["timeout"]},
                timestamp=datetime.now().isoformat(),
                issues_count=1,
                critical_issues=1,
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running QA component {component_name}: {e}")
            return QAResult(
                component=component_name,
                status="failed",
                score=0.0,
                duration=duration,
                details={"error": str(e)},
                timestamp=datetime.now().isoformat(),
                issues_count=1,
                critical_issues=1,
            )

    async def parse_component_results(
        self, component_name: str, stdout: str, stderr: str
    ) -> Dict[str, Any]:
        """Parse results from QA component output"""
        details = {"stdout": stdout, "stderr": stderr}

        try:
            if component_name == "automated_tests":
                # Look for test report file
                latest_report = (
                    self.project_root / "test-reports" / "latest_test_report.json"
                )
                if latest_report.exists():
                    with open(latest_report, "r") as f:
                        test_data = json.load(f)
                        details.update(
                            {
                                "total_tests": test_data["summary"]["total_tests"],
                                "passed_tests": test_data["summary"]["passed"],
                                "failed_tests": test_data["summary"]["failed"],
                                "success_rate": test_data["summary"]["success_rate"],
                                "coverage": test_data["summary"].get(
                                    "average_coverage"
                                ),
                                "total_issues": test_data["summary"]["failed"],
                                "critical_issues": test_data["summary"]["failed"],
                            }
                        )

            elif component_name == "model_validation":
                # Look for validation report file
                latest_report = (
                    self.project_root
                    / "validation-reports"
                    / "latest_validation_report.json"
                )
                if latest_report.exists():
                    with open(latest_report, "r") as f:
                        validation_data = json.load(f)
                        details.update(
                            {
                                "total_validations": validation_data["summary"][
                                    "total_validations"
                                ],
                                "passed_validations": validation_data["summary"][
                                    "passed"
                                ],
                                "failed_validations": validation_data["summary"][
                                    "failed"
                                ],
                                "average_accuracy": validation_data["summary"][
                                    "average_accuracy"
                                ],
                                "physics_compliance": validation_data["summary"][
                                    "average_physics_compliance"
                                ],
                                "total_issues": validation_data["summary"]["failed"],
                                "critical_issues": validation_data["summary"]["failed"],
                            }
                        )

            elif component_name == "code_quality":
                # Look for quality report file
                latest_report = (
                    self.project_root / "quality-reports" / "latest_quality_report.json"
                )
                if latest_report.exists():
                    with open(latest_report, "r") as f:
                        quality_data = json.load(f)
                        details.update(
                            {
                                "quality_score": quality_data["quality_score"],
                                "security_score": quality_data["security_score"],
                                "total_issues": quality_data["summary"]["total_issues"],
                                "errors": quality_data["summary"]["errors"],
                                "warnings": quality_data["summary"]["warnings"],
                                "security_vulnerabilities": quality_data["summary"][
                                    "security_vulnerabilities"
                                ],
                                "critical_issues": quality_data["summary"][
                                    "high_severity_vulns"
                                ],
                            }
                        )

            elif component_name == "documentation":
                # Look for documentation report file
                latest_report = (
                    self.project_root
                    / "documentation-reports"
                    / "latest_documentation_report.json"
                )
                if latest_report.exists():
                    with open(latest_report, "r") as f:
                        doc_data = json.load(f)
                        details.update(
                            {
                                "coverage_score": doc_data["coverage_score"],
                                "api_completeness": doc_data["api_completeness"],
                                "total_issues": doc_data["summary"]["total_issues"],
                                "missing_docstrings": doc_data["summary"][
                                    "missing_docstrings"
                                ],
                                "files_generated": doc_data["summary"][
                                    "files_generated"
                                ],
                                "critical_issues": 0,  # Documentation issues are not critical
                            }
                        )

        except Exception as e:
            logger.warning(
                f"Could not parse detailed results for {component_name}: {e}"
            )

        return details

    def calculate_component_score(
        self, component_name: str, details: Dict[str, Any], status: str
    ) -> float:
        """Calculate score for a QA component"""
        if status == "failed":
            base_score = 0.0
        else:
            base_score = 10.0

        try:
            if component_name == "automated_tests":
                success_rate = details.get("success_rate", 0.0)
                coverage = details.get("coverage", 0.0) or 0.0
                score = (success_rate * 8.0) + (coverage / 100.0 * 2.0)
                return min(10.0, max(0.0, score))

            elif component_name == "model_validation":
                accuracy = details.get("average_accuracy", 0.0)
                physics_compliance = details.get("physics_compliance", 0.0)
                score = (accuracy * 6.0) + (physics_compliance * 4.0)
                return min(10.0, max(0.0, score))

            elif component_name == "code_quality":
                quality_score = details.get("quality_score", 0.0)
                security_score = details.get("security_score", 0.0)
                score = (quality_score * 0.7) + (security_score * 0.3)
                return min(10.0, max(0.0, score))

            elif component_name == "documentation":
                coverage = details.get("coverage_score", 0.0)
                api_completeness = details.get("api_completeness", 0.0)
                score = (coverage * 6.0) + (api_completeness * 4.0)
                return min(10.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"Could not calculate score for {component_name}: {e}")

        return base_score

    def calculate_overall_score(self, results: List[QAResult]) -> float:
        """Calculate weighted overall QA score"""
        total_weighted_score = 0.0
        total_weight = 0.0

        for result in results:
            component_config = self.qa_components.get(result.component, {})
            weight = component_config.get("weight", 0.25)

            total_weighted_score += result.score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def determine_overall_status(
        self, results: List[QAResult], overall_score: float
    ) -> str:
        """Determine overall QA status"""
        # Check for critical failures
        critical_failures = sum(
            1
            for r in results
            if r.status == "failed"
            and self.qa_components.get(r.component, {}).get("critical", False)
        )

        if critical_failures > self.quality_gates["max_critical_failures"]:
            return "failed"

        if overall_score < self.quality_gates["min_overall_score"]:
            return "failed"

        # Check individual component thresholds
        for result in results:
            if result.component == "automated_tests":
                success_rate = result.details.get("success_rate", 0.0)
                if success_rate < self.quality_gates["min_test_success_rate"]:
                    return "failed"

            elif result.component == "model_validation":
                accuracy = result.details.get("average_accuracy", 0.0)
                if accuracy < self.quality_gates["min_model_accuracy"]:
                    return "failed"

            elif result.component == "code_quality":
                if result.score < self.quality_gates["min_code_quality"]:
                    return "warning"

            elif result.component == "documentation":
                coverage = result.details.get("coverage_score", 0.0)
                if coverage < self.quality_gates["min_documentation_coverage"]:
                    return "warning"

        # Check if any component has warnings
        if any(r.status == "warning" for r in results):
            return "warning"

        return "passed"

    def generate_recommendations(self, results: List[QAResult]) -> List[str]:
        """Generate actionable recommendations based on QA results"""
        recommendations = []

        for result in results:
            if result.status == "failed":
                if result.component == "automated_tests":
                    failed_tests = result.details.get("failed_tests", 0)
                    if failed_tests > 0:
                        recommendations.append(
                            f"Fix {failed_tests} failing tests in the test suite"
                        )

                    coverage = result.details.get("coverage", 0.0) or 0.0
                    if coverage < 80.0:
                        recommendations.append(
                            f"Increase test coverage from {coverage:.1f}% to at least 80%"
                        )

                elif result.component == "model_validation":
                    accuracy = result.details.get("average_accuracy", 0.0)
                    if accuracy < 0.80:
                        recommendations.append(
                            f"Improve model accuracy from {accuracy:.3f} to at least 0.80"
                        )

                    physics_compliance = result.details.get("physics_compliance", 0.0)
                    if physics_compliance < 0.70:
                        recommendations.append(
                            f"Improve physics constraint compliance from {physics_compliance:.3f}"
                        )

                elif result.component == "code_quality":
                    critical_issues = result.details.get("critical_issues", 0)
                    if critical_issues > 0:
                        recommendations.append(
                            f"Fix {critical_issues} high-severity security vulnerabilities"
                        )

                    if result.score < 7.0:
                        recommendations.append(
                            "Address code quality issues to improve maintainability"
                        )

                elif result.component == "documentation":
                    coverage = result.details.get("coverage_score", 0.0)
                    if coverage < 0.85:
                        recommendations.append(
                            f"Improve documentation coverage from {coverage:.1%} to at least 85%"
                        )

            elif result.status == "warning":
                recommendations.append(
                    f"Review and address warnings in {result.component}"
                )

        # Add general recommendations
        if not recommendations:
            recommendations.append(
                "All QA components passed! Consider optimizing performance and adding more comprehensive tests."
            )

        return recommendations

    async def run_comprehensive_qa(self) -> QAOrchestrationReport:
        """Run comprehensive QA orchestration"""
        logger.info("Starting comprehensive QA orchestration")
        start_time = datetime.now()

        # Run all QA components in parallel
        tasks = [
            self.run_qa_component(component_name, config)
            for component_name, config in self.qa_components.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        qa_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_name = list(self.qa_components.keys())[i]
                qa_results.append(
                    QAResult(
                        component=component_name,
                        status="failed",
                        score=0.0,
                        duration=0.0,
                        details={"error": str(result)},
                        timestamp=datetime.now().isoformat(),
                        issues_count=1,
                        critical_issues=1,
                    )
                )
            else:
                qa_results.append(result)

        total_duration = (datetime.now() - start_time).total_seconds()

        # Calculate overall metrics
        overall_score = self.calculate_overall_score(qa_results)
        overall_status = self.determine_overall_status(qa_results, overall_score)
        recommendations = self.generate_recommendations(qa_results)

        # Generate summary
        summary = {
            "total_components": len(qa_results),
            "passed_components": len([r for r in qa_results if r.status == "passed"]),
            "failed_components": len([r for r in qa_results if r.status == "failed"]),
            "warning_components": len([r for r in qa_results if r.status == "warning"]),
            "total_issues": sum(r.issues_count for r in qa_results),
            "critical_issues": sum(r.critical_issues for r in qa_results),
            "average_component_score": (
                sum(r.score for r in qa_results) / len(qa_results)
                if qa_results
                else 0.0
            ),
        }

        return QAOrchestrationReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            overall_score=overall_score,
            total_duration=total_duration,
            component_results=qa_results,
            summary=summary,
            recommendations=recommendations,
        )

    def save_qa_report(self, report: QAOrchestrationReport):
        """Save QA orchestration report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"qa_orchestration_{timestamp}.json"

        # Convert dataclasses to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "overall_status": report.overall_status,
            "overall_score": report.overall_score,
            "total_duration": report.total_duration,
            "component_results": [
                asdict(result) for result in report.component_results
            ],
            "summary": report.summary,
            "recommendations": report.recommendations,
        }

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"QA orchestration report saved to {report_file}")

        # Also save latest report
        latest_report = self.reports_dir / "latest_qa_report.json"
        with open(latest_report, "w") as f:
            json.dump(report_dict, f, indent=2)


async def main():
    """Main execution function"""
    orchestrator = QAOrchestrator()

    try:
        # Run comprehensive QA
        report = await orchestrator.run_comprehensive_qa()

        # Save report
        orchestrator.save_qa_report(report)

        # Print summary
        logger.info("QA orchestration completed:")
        logger.info(f"  Overall status: {report.overall_status}")
        logger.info(f"  Overall score: {report.overall_score:.1f}/10")
        logger.info(f"  Total duration: {report.total_duration:.1f}s")
        logger.info(
            f"  Components passed: {report.summary['passed_components']}/{report.summary['total_components']}"
        )
        logger.info(f"  Total issues: {report.summary['total_issues']}")
        logger.info(f"  Critical issues: {report.summary['critical_issues']}")

        # Print component results
        for result in report.component_results:
            status_emoji = (
                "✅"
                if result.status == "passed"
                else "❌" if result.status == "failed" else "⚠️"
            )
            logger.info(
                f"  {status_emoji} {result.component}: {result.score:.1f}/10 ({result.duration:.1f}s)"
            )

        # Print recommendations
        if report.recommendations:
            logger.info("Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                logger.info(f"  {i}. {rec}")

        # Exit with appropriate code
        if report.overall_status == "failed":
            logger.error("QA orchestration failed!")
            sys.exit(1)
        elif report.overall_status == "warning":
            logger.warning("QA orchestration completed with warnings")
            sys.exit(0)
        else:
            logger.info("QA orchestration passed!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"QA orchestration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
