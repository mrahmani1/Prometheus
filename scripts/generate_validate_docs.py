#!/usr/bin/env python3
"""
Automated Documentation Generation and Validation Script for Prometheus AI Platform
Generates API documentation, validates completeness, and ensures consistency
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast
import inspect
import json
import logging
import os
import re
import subprocess
import sys

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentationIssue:
    file_path: str
    line_number: int
    severity: str  # error, warning, info
    category: str  # missing_docstring, incomplete_api_doc, etc.
    message: str
    suggestion: Optional[str] = None


@dataclass
class APIEndpoint:
    path: str
    method: str
    function_name: str
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    parameters: List[Dict] = None
    responses: List[Dict] = None
    has_documentation: bool = False


@dataclass
class DocumentationReport:
    timestamp: str
    summary: Dict[str, int]
    coverage_score: float
    api_completeness: float
    issues: List[DocumentationIssue]
    api_endpoints: List[APIEndpoint]
    generated_files: List[str]


class DocumentationGenerator:
    def __init__(self):
        # Handle cases where __file__ is not defined (e.g., when run via exec)
        try:
            self.project_root = Path(__file__).parent.parent
        except NameError:
            # Fallback to current working directory
            self.project_root = Path.cwd()

        self.services_dir = self.project_root / "services"
        self.docs_dir = self.project_root / "docs"
        self.docs_dir.mkdir(exist_ok=True)

        self.documentation_issues: List[DocumentationIssue] = []
        self.api_endpoints: List[APIEndpoint] = []
        self.generated_files: List[str] = []

        # Documentation requirements
        self.doc_requirements = {
            "min_coverage": 0.90,
            "required_sections": ["overview", "api", "installation", "usage"],
            "api_doc_completeness": 0.95,
        }

    def discover_python_modules(self) -> List[Path]:
        """Discover Python modules for documentation"""
        python_files = []

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir():
                for py_file in service_dir.rglob("*.py"):
                    if not any(
                        exclude in str(py_file)
                        for exclude in ["__pycache__", "test", ".venv"]
                    ):
                        python_files.append(py_file)

        return python_files

    def analyze_python_docstrings(
        self, python_files: List[Path]
    ) -> List[DocumentationIssue]:
        """Analyze Python files for docstring completeness"""
        logger.info("Analyzing Python docstrings")
        issues = []

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not ast.get_docstring(node):
                            issues.append(
                                DocumentationIssue(
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    severity="warning",
                                    category="missing_docstring",
                                    message=f"Class '{node.name}' missing docstring",
                                    suggestion=f"Add docstring describing the purpose of class {node.name}",
                                )
                            )

                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_"):  # Skip private functions
                            docstring = ast.get_docstring(node)
                            if not docstring:
                                issues.append(
                                    DocumentationIssue(
                                        file_path=str(file_path),
                                        line_number=node.lineno,
                                        severity="warning",
                                        category="missing_docstring",
                                        message=f"Function '{node.name}' missing docstring",
                                        suggestion=f"Add docstring with parameters, returns, and description for {node.name}",
                                    )
                                )
                            else:
                                # Check docstring completeness
                                if (
                                    len(node.args.args) > 1
                                    and "Args:" not in docstring
                                    and "Parameters:" not in docstring
                                ):
                                    issues.append(
                                        DocumentationIssue(
                                            file_path=str(file_path),
                                            line_number=node.lineno,
                                            severity="info",
                                            category="incomplete_docstring",
                                            message=f"Function '{node.name}' docstring missing parameter documentation",
                                            suggestion="Add Args: or Parameters: section to docstring",
                                        )
                                    )

                                if (
                                    "return" in content[node.lineno : node.end_lineno]
                                    and "Returns:" not in docstring
                                ):
                                    issues.append(
                                        DocumentationIssue(
                                            file_path=str(file_path),
                                            line_number=node.lineno,
                                            severity="info",
                                            category="incomplete_docstring",
                                            message=f"Function '{node.name}' docstring missing return documentation",
                                            suggestion="Add Returns: section to docstring",
                                        )
                                    )

            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")

        return issues

    def discover_api_endpoints(self) -> List[APIEndpoint]:
        """Discover API endpoints in FastAPI and Express services"""
        logger.info("Discovering API endpoints")
        endpoints = []

        # Discover FastAPI endpoints (Python services)
        endpoints.extend(self.discover_fastapi_endpoints())

        # Discover Express endpoints (Node.js services)
        endpoints.extend(self.discover_express_endpoints())

        return endpoints

    def discover_fastapi_endpoints(self) -> List[APIEndpoint]:
        """Discover FastAPI endpoints"""
        endpoints = []

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir():
                for py_file in service_dir.rglob("*.py"):
                    try:
                        with open(py_file, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Look for FastAPI route decorators
                        route_patterns = [
                            r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                            r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                        ]

                        for pattern in route_patterns:
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                method = match.group(1).upper()
                                path = match.group(2)

                                # Find the function definition
                                lines = content[: match.start()].split("\n")
                                line_number = len(lines)

                                # Look for function definition after the decorator
                                func_match = re.search(
                                    r"def\s+(\w+)\s*\(", content[match.end() :]
                                )
                                if func_match:
                                    function_name = func_match.group(1)

                                    # Extract docstring if present
                                    func_start = match.end() + func_match.start()
                                    docstring = self.extract_function_docstring(
                                        content, func_start
                                    )

                                    endpoints.append(
                                        APIEndpoint(
                                            path=path,
                                            method=method,
                                            function_name=function_name,
                                            file_path=str(py_file),
                                            line_number=line_number,
                                            docstring=docstring,
                                            has_documentation=bool(docstring),
                                        )
                                    )

                    except Exception as e:
                        logger.warning(
                            f"Could not analyze {py_file} for FastAPI endpoints: {e}"
                        )

        return endpoints

    def discover_express_endpoints(self) -> List[APIEndpoint]:
        """Discover Express.js endpoints"""
        endpoints = []

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir():
                for js_file in service_dir.rglob("*.js"):
                    try:
                        with open(js_file, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Look for Express route definitions
                        route_patterns = [
                            r'app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                            r'router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                        ]

                        for pattern in route_patterns:
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                method = match.group(1).upper()
                                path = match.group(2)

                                lines = content[: match.start()].split("\n")
                                line_number = len(lines)

                                # Extract function name or create one
                                function_name = f"{method.lower()}_{path.replace('/', '_').replace(':', '')}"

                                # Look for JSDoc comments
                                jsdoc = self.extract_jsdoc_comment(
                                    content, match.start()
                                )

                                endpoints.append(
                                    APIEndpoint(
                                        path=path,
                                        method=method,
                                        function_name=function_name,
                                        file_path=str(js_file),
                                        line_number=line_number,
                                        docstring=jsdoc,
                                        has_documentation=bool(jsdoc),
                                    )
                                )

                    except Exception as e:
                        logger.warning(
                            f"Could not analyze {js_file} for Express endpoints: {e}"
                        )

        return endpoints

    def extract_function_docstring(
        self, content: str, func_start: int
    ) -> Optional[str]:
        """Extract docstring from Python function"""
        try:
            # Find the function definition and extract docstring
            func_content = content[func_start:]
            lines = func_content.split("\n")

            # Look for docstring in the first few lines after function definition
            for i, line in enumerate(lines[1:6]):  # Check first 5 lines
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Found docstring start
                    quote_type = '"""' if stripped.startswith('"""') else "'''"

                    if stripped.endswith(quote_type) and len(stripped) > 6:
                        # Single line docstring
                        return stripped[3:-3].strip()
                    else:
                        # Multi-line docstring
                        docstring_lines = [stripped[3:]]
                        for j, next_line in enumerate(lines[i + 2 :], i + 2):
                            if quote_type in next_line:
                                docstring_lines.append(
                                    next_line[: next_line.index(quote_type)]
                                )
                                break
                            docstring_lines.append(next_line.strip())
                        return "\n".join(docstring_lines).strip()
        except Exception:
            pass
        return None

    def extract_jsdoc_comment(self, content: str, position: int) -> Optional[str]:
        """Extract JSDoc comment before JavaScript function"""
        try:
            # Look backwards for JSDoc comment
            before_content = content[:position]
            lines = before_content.split("\n")

            # Look for /** comment block
            for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
                line = lines[i].strip()
                if line.startswith("/**"):
                    # Found JSDoc start, collect until */
                    jsdoc_lines = []
                    for j in range(i, len(lines)):
                        jsdoc_line = lines[j].strip()
                        if jsdoc_line.startswith("*"):
                            jsdoc_lines.append(jsdoc_line[1:].strip())
                        if jsdoc_line.endswith("*/"):
                            break
                    return "\n".join(jsdoc_lines).strip()
        except Exception:
            pass
        return None

    def generate_api_documentation(self, endpoints: List[APIEndpoint]) -> str:
        """Generate comprehensive API documentation"""
        logger.info("Generating API documentation")

        # Group endpoints by service
        services = {}
        for endpoint in endpoints:
            service_name = Path(endpoint.file_path).parts[
                -3
            ]  # Extract service name from path
            if service_name not in services:
                services[service_name] = []
            services[service_name].append(endpoint)

        # Generate markdown documentation
        doc_content = "# Prometheus AI Platform API Documentation\n\n"
        doc_content += (
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        doc_content += "## Overview\n\n"
        doc_content += "This document provides comprehensive API documentation for all services in the Prometheus AI Platform.\n\n"

        for service_name, service_endpoints in services.items():
            doc_content += f"## {service_name.title()} Service\n\n"

            # Group by HTTP method
            methods = {}
            for endpoint in service_endpoints:
                if endpoint.method not in methods:
                    methods[endpoint.method] = []
                methods[endpoint.method].append(endpoint)

            for method, method_endpoints in methods.items():
                doc_content += f"### {method} Endpoints\n\n"

                for endpoint in method_endpoints:
                    doc_content += f"#### `{method} {endpoint.path}`\n\n"

                    if endpoint.docstring:
                        doc_content += f"**Description:** {endpoint.docstring}\n\n"
                    else:
                        doc_content += "**Description:** *No description available*\n\n"

                    doc_content += f"**Function:** `{endpoint.function_name}`\n\n"
                    doc_content += f"**File:** `{endpoint.file_path}`\n\n"

                    # Add placeholder sections for parameters and responses
                    doc_content += "**Parameters:**\n"
                    doc_content += "- *To be documented*\n\n"

                    doc_content += "**Responses:**\n"
                    doc_content += "- `200`: Success\n"
                    doc_content += "- `400`: Bad Request\n"
                    doc_content += "- `500`: Internal Server Error\n\n"

                    doc_content += "---\n\n"

        # Save API documentation
        api_doc_file = self.docs_dir / "api_documentation.md"
        with open(api_doc_file, "w", encoding="utf-8") as f:
            f.write(doc_content)

        self.generated_files.append(str(api_doc_file))
        logger.info(f"API documentation generated: {api_doc_file}")

        return str(api_doc_file)

    def generate_service_documentation(self) -> List[str]:
        """Generate individual service documentation"""
        logger.info("Generating service documentation")
        generated_docs = []

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir():
                service_name = service_dir.name

                # Check if service has a README
                readme_file = service_dir / "README.md"
                if not readme_file.exists():
                    # Generate basic README
                    readme_content = f"# {service_name.title()} Service\n\n"
                    readme_content += f"## Overview\n\n"
                    readme_content += f"The {service_name} service is part of the Prometheus AI Platform.\n\n"
                    readme_content += f"## Installation\n\n"
                    readme_content += f"```bash\n"
                    readme_content += f"cd services/{service_name}\n"

                    # Check if it's a Python or Node.js service
                    if (service_dir / "requirements.txt").exists():
                        readme_content += f"pip install -r requirements.txt\n"
                        readme_content += f"python main.py\n"
                    elif (service_dir / "package.json").exists():
                        readme_content += f"npm install\n"
                        readme_content += f"npm start\n"

                    readme_content += f"```\n\n"
                    readme_content += f"## API Endpoints\n\n"
                    readme_content += f"See the main API documentation for detailed endpoint information.\n\n"
                    readme_content += f"## Configuration\n\n"
                    readme_content += f"Configuration options and environment variables will be documented here.\n\n"

                    with open(readme_file, "w", encoding="utf-8") as f:
                        f.write(readme_content)

                    generated_docs.append(str(readme_file))
                    self.generated_files.append(str(readme_file))
                    logger.info(f"Generated README for {service_name}")

        return generated_docs

    def generate_openapi_spec(self, endpoints: List[APIEndpoint]) -> str:
        """Generate OpenAPI specification"""
        logger.info("Generating OpenAPI specification")

        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Prometheus AI Platform API",
                "description": "Comprehensive API for the Prometheus AI geodynamic platform",
                "version": "1.0.0",
                "contact": {
                    "name": "Prometheus AI Team",
                    "email": "api@prometheus-ai.com",
                },
            },
            "servers": [
                {"url": "http://localhost:8000", "description": "Development server"},
                {
                    "url": "https://api.prometheus-ai.com",
                    "description": "Production server",
                },
            ],
            "paths": {},
        }

        # Group endpoints by path
        paths = {}
        for endpoint in endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            paths[endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.function_name.replace("_", " ").title(),
                "description": endpoint.docstring or "No description available",
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    },
                    "400": {"description": "Bad request"},
                    "500": {"description": "Internal server error"},
                },
            }

        openapi_spec["paths"] = paths

        # Save OpenAPI spec
        openapi_file = self.docs_dir / "openapi.json"
        with open(openapi_file, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2)

        self.generated_files.append(str(openapi_file))
        logger.info(f"OpenAPI specification generated: {openapi_file}")

        return str(openapi_file)

    def validate_documentation_completeness(
        self, endpoints: List[APIEndpoint]
    ) -> List[DocumentationIssue]:
        """Validate API documentation completeness"""
        logger.info("Validating documentation completeness")
        issues = []

        for endpoint in endpoints:
            if not endpoint.has_documentation:
                issues.append(
                    DocumentationIssue(
                        file_path=endpoint.file_path,
                        line_number=endpoint.line_number,
                        severity="warning",
                        category="missing_api_documentation",
                        message=f"API endpoint {endpoint.method} {endpoint.path} lacks documentation",
                        suggestion="Add docstring or JSDoc comment describing the endpoint",
                    )
                )

            elif endpoint.docstring and len(endpoint.docstring.strip()) < 20:
                issues.append(
                    DocumentationIssue(
                        file_path=endpoint.file_path,
                        line_number=endpoint.line_number,
                        severity="info",
                        category="incomplete_api_documentation",
                        message=f"API endpoint {endpoint.method} {endpoint.path} has minimal documentation",
                        suggestion="Expand documentation with parameters, responses, and examples",
                    )
                )

        return issues

    def calculate_coverage_score(
        self, python_files: List[Path], issues: List[DocumentationIssue]
    ) -> float:
        """Calculate documentation coverage score"""
        total_functions = 0
        documented_functions = 0

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        if not (hasattr(node, "name") and node.name.startswith("_")):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1

            except Exception:
                continue

        return documented_functions / total_functions if total_functions > 0 else 0.0

    def calculate_api_completeness(self, endpoints: List[APIEndpoint]) -> float:
        """Calculate API documentation completeness"""
        if not endpoints:
            return 1.0

        documented_endpoints = len([e for e in endpoints if e.has_documentation])
        return documented_endpoints / len(endpoints)

    def run_comprehensive_documentation_generation(self) -> DocumentationReport:
        """Run comprehensive documentation generation and validation"""
        logger.info("Starting comprehensive documentation generation")

        # Discover source files
        python_files = self.discover_python_modules()

        # Analyze docstrings
        docstring_issues = self.analyze_python_docstrings(python_files)

        # Discover API endpoints
        api_endpoints = self.discover_api_endpoints()

        # Validate API documentation
        api_issues = self.validate_documentation_completeness(api_endpoints)

        # Generate documentation
        self.generate_api_documentation(api_endpoints)
        self.generate_service_documentation()
        self.generate_openapi_spec(api_endpoints)

        # Calculate metrics
        coverage_score = self.calculate_coverage_score(python_files, docstring_issues)
        api_completeness = self.calculate_api_completeness(api_endpoints)

        # Combine all issues
        all_issues = docstring_issues + api_issues

        # Generate summary
        summary = {
            "total_issues": len(all_issues),
            "missing_docstrings": len(
                [i for i in all_issues if i.category == "missing_docstring"]
            ),
            "incomplete_docstrings": len(
                [i for i in all_issues if i.category == "incomplete_docstring"]
            ),
            "missing_api_docs": len(
                [i for i in all_issues if i.category == "missing_api_documentation"]
            ),
            "api_endpoints_total": len(api_endpoints),
            "api_endpoints_documented": len(
                [e for e in api_endpoints if e.has_documentation]
            ),
            "files_generated": len(self.generated_files),
        }

        return DocumentationReport(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            coverage_score=coverage_score,
            api_completeness=api_completeness,
            issues=all_issues,
            api_endpoints=api_endpoints,
            generated_files=self.generated_files,
        )

    def save_documentation_report(self, report: DocumentationReport):
        """Save documentation report to file"""
        reports_dir = self.project_root / "documentation-reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"documentation_report_{timestamp}.json"

        # Convert dataclasses to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "summary": report.summary,
            "coverage_score": report.coverage_score,
            "api_completeness": report.api_completeness,
            "issues": [asdict(issue) for issue in report.issues],
            "api_endpoints": [asdict(endpoint) for endpoint in report.api_endpoints],
            "generated_files": report.generated_files,
        }

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Documentation report saved to {report_file}")

        # Also save latest report
        latest_report = reports_dir / "latest_documentation_report.json"
        with open(latest_report, "w") as f:
            json.dump(report_dict, f, indent=2)


def main():
    """Main execution function"""
    generator = DocumentationGenerator()

    try:
        # Run comprehensive documentation generation
        report = generator.run_comprehensive_documentation_generation()

        # Save report
        generator.save_documentation_report(report)

        # Print summary
        logger.info(f"Documentation generation completed:")
        logger.info(f"  Files generated: {report.summary['files_generated']}")
        logger.info(f"  Documentation coverage: {report.coverage_score:.1%}")
        logger.info(f"  API completeness: {report.api_completeness:.1%}")
        logger.info(f"  Total issues: {report.summary['total_issues']}")
        logger.info(
            f"  API endpoints documented: {report.summary['api_endpoints_documented']}/{report.summary['api_endpoints_total']}"
        )

        # Check documentation gates
        documentation_gate_passed = True

        if report.coverage_score < generator.doc_requirements["min_coverage"]:
            logger.error(
                f"DOCUMENTATION GATE FAILED: Coverage {report.coverage_score:.1%} below threshold {generator.doc_requirements['min_coverage']:.1%}"
            )
            documentation_gate_passed = False

        if report.api_completeness < generator.doc_requirements["api_doc_completeness"]:
            logger.error(
                f"API DOCUMENTATION GATE FAILED: API completeness {report.api_completeness:.1%} below threshold {generator.doc_requirements['api_doc_completeness']:.1%}"
            )
            documentation_gate_passed = False

        # Exit with appropriate code
        if not documentation_gate_passed:
            logger.error("Documentation gates failed!")
            sys.exit(1)
        else:
            logger.info("All documentation gates passed!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
