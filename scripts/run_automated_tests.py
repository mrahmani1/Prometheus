#!/usr/bin/env python3
"""
Automated Test Execution Script for Prometheus AI Platform
Runs comprehensive test suites across all services with intelligent test selection
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    service: str
    test_type: str
    status: str  # passed, failed, skipped
    duration: float
    coverage: Optional[float] = None
    details: Optional[str] = None


@dataclass
class TestSuite:
    name: str
    path: str
    command: str
    timeout: int
    dependencies: List[str]
    test_types: List[str]


class AutomatedTestRunner:
    def __init__(self):
        # Handle cases where __file__ is not defined (e.g., when run via exec)
        try:
            self.project_root = Path(__file__).parent.parent
        except NameError:
            # Fallback to current working directory
            self.project_root = Path.cwd()

        self.services_dir = self.project_root / "services"
        self.test_results: List[TestResult] = []
        self.changed_files: List[str] = []

    def detect_changed_files(self) -> List[str]:
        """Detect changed files using git diff"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                return (
                    result.stdout.strip().split("\n") if result.stdout.strip() else []
                )
        except Exception as e:
            logger.warning(f"Could not detect changed files: {e}")
        return []

    def get_affected_services(self, changed_files: List[str]) -> List[str]:
        """Determine which services are affected by file changes"""
        affected_services = set()

        for file_path in changed_files:
            if file_path.startswith("services/"):
                service_name = file_path.split("/")[1]
                affected_services.add(service_name)
            elif file_path.startswith("kubernetes/") or file_path.startswith(
                "database/"
            ):
                # Infrastructure changes affect all services
                affected_services.update(self.get_all_services())

        return list(affected_services)

    def get_all_services(self) -> List[str]:
        """Get list of all services"""
        services = []
        if self.services_dir.exists():
            for item in self.services_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    services.append(item.name)
        return services

    def load_test_suites(self) -> List[TestSuite]:
        """Load test suite configurations"""
        test_suites = []

        # Python services test suites
        python_services = [
            "helios-core",
            "data-ingestion",
            "volcano-watchtower",
            "nlp-service",
        ]
        for service in python_services:
            service_path = self.services_dir / service
            if service_path.exists():
                test_suites.append(
                    TestSuite(
                        name=f"{service}-unit-tests",
                        path=str(service_path),
                        command="pytest tests/ -v --cov=. --cov-report=json --cov-report=term",
                        timeout=600,
                        dependencies=[],
                        test_types=["unit", "integration"],
                    )
                )

        # Node.js services test suites
        nodejs_services = ["api-gateway", "vulcan-ui"]
        for service in nodejs_services:
            service_path = self.services_dir / service
            if service_path.exists():
                test_suites.append(
                    TestSuite(
                        name=f"{service}-unit-tests",
                        path=str(service_path),
                        command="npm test -- --coverage --watchAll=false",
                        timeout=300,
                        dependencies=[],
                        test_types=["unit", "integration"],
                    )
                )

        # Integration tests
        test_suites.append(
            TestSuite(
                name="integration-tests",
                path=str(self.project_root / "tests" / "integration"),
                command="pytest . -v --tb=short",
                timeout=1200,
                dependencies=python_services + nodejs_services,
                test_types=["integration", "e2e"],
            )
        )

        # Performance tests
        test_suites.append(
            TestSuite(
                name="performance-tests",
                path=str(self.project_root / "tests" / "performance"),
                command="pytest . -v --benchmark-only",
                timeout=1800,
                dependencies=["helios-core", "volcano-watchtower"],
                test_types=["performance", "load"],
            )
        )

        return test_suites

    async def run_test_suite(self, test_suite: TestSuite) -> TestResult:
        """Run a single test suite"""
        logger.info(f"Running test suite: {test_suite.name}")
        start_time = datetime.now()

        try:
            # Change to test directory
            process = await asyncio.create_subprocess_shell(
                test_suite.command,
                cwd=test_suite.path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=test_suite.timeout
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Parse test results
            status = "passed" if process.returncode == 0 else "failed"
            coverage = self.extract_coverage(stdout.decode())

            result = TestResult(
                service=test_suite.name,
                test_type=",".join(test_suite.test_types),
                status=status,
                duration=duration,
                coverage=coverage,
                details=stderr.decode() if process.returncode != 0 else None,
            )

            logger.info(f"Test suite {test_suite.name} {status} in {duration:.2f}s")
            return result

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Test suite {test_suite.name} timed out after {duration:.2f}s"
            )
            return TestResult(
                service=test_suite.name,
                test_type=",".join(test_suite.test_types),
                status="failed",
                duration=duration,
                details="Test suite timed out",
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error running test suite {test_suite.name}: {e}")
            return TestResult(
                service=test_suite.name,
                test_type=",".join(test_suite.test_types),
                status="failed",
                duration=duration,
                details=str(e),
            )

    def extract_coverage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from test output"""
        try:
            # Look for coverage patterns in output
            lines = output.split("\n")
            for line in lines:
                if "TOTAL" in line and "%" in line:
                    # Extract percentage from pytest-cov output
                    parts = line.split()
                    for part in parts:
                        if part.endswith("%"):
                            return float(part.rstrip("%"))
                elif "Coverage:" in line:
                    # Extract from other coverage formats
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Coverage:" and i + 1 < len(parts):
                            return float(parts[i + 1].rstrip("%"))
        except Exception as e:
            logger.warning(f"Could not extract coverage: {e}")
        return None

    async def run_smart_test_selection(self) -> List[TestResult]:
        """Run tests with smart selection based on changed files"""
        self.changed_files = self.detect_changed_files()
        affected_services = self.get_affected_services(self.changed_files)

        logger.info(f"Changed files: {len(self.changed_files)}")
        logger.info(f"Affected services: {affected_services}")

        test_suites = self.load_test_suites()

        # Filter test suites based on affected services
        if (
            affected_services and len(self.changed_files) < 50
        ):  # Smart selection for small changes
            filtered_suites = []
            for suite in test_suites:
                if any(service in suite.name for service in affected_services):
                    filtered_suites.append(suite)
                elif suite.name == "integration-tests":
                    # Always run integration tests if core services are affected
                    core_services = ["helios-core", "api-gateway", "volcano-watchtower"]
                    if any(service in affected_services for service in core_services):
                        filtered_suites.append(suite)
            test_suites = filtered_suites

        # Run test suites in parallel
        tasks = [self.run_test_suite(suite) for suite in test_suites]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        test_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_results.append(
                    TestResult(
                        service=test_suites[i].name,
                        test_type="error",
                        status="failed",
                        duration=0,
                        details=str(result),
                    )
                )
            else:
                test_results.append(result)

        return test_results

    def generate_test_report(self, results: List[TestResult]) -> Dict:
        """Generate comprehensive test report"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == "passed"])
        failed_tests = len([r for r in results if r.status == "failed"])
        total_duration = sum(r.duration for r in results)

        # Calculate average coverage
        coverage_results = [r.coverage for r in results if r.coverage is not None]
        avg_coverage = (
            sum(coverage_results) / len(coverage_results) if coverage_results else None
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "average_coverage": avg_coverage,
            },
            "results": [
                {
                    "service": r.service,
                    "test_type": r.test_type,
                    "status": r.status,
                    "duration": r.duration,
                    "coverage": r.coverage,
                    "details": r.details,
                }
                for r in results
            ],
            "changed_files": self.changed_files,
        }

        return report

    def save_test_report(self, report: Dict):
        """Save test report to file"""
        reports_dir = self.project_root / "test-reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"test_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Test report saved to {report_file}")

        # Also save latest report
        latest_report = reports_dir / "latest_test_report.json"
        with open(latest_report, "w") as f:
            json.dump(report, f, indent=2)


async def main():
    """Main execution function"""
    runner = AutomatedTestRunner()

    try:
        # Run smart test selection
        results = await runner.run_smart_test_selection()

        # Generate and save report
        report = runner.generate_test_report(results)
        runner.save_test_report(report)

        # Print summary
        summary = report["summary"]
        logger.info(f"Test execution completed:")
        logger.info(f"  Total tests: {summary['total_tests']}")
        logger.info(f"  Passed: {summary['passed']}")
        logger.info(f"  Failed: {summary['failed']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2%}")
        logger.info(f"  Total duration: {summary['total_duration']:.2f}s")
        if summary["average_coverage"]:
            logger.info(f"  Average coverage: {summary['average_coverage']:.1f}%")

        # Exit with appropriate code
        if summary["failed"] > 0:
            logger.error("Some tests failed!")
            sys.exit(1)
        else:
            logger.info("All tests passed!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
