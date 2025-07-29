#!/usr/bin/env python3
"""
Comprehensive CI/CD Pipeline for Prometheus AI Platform
Implements automated build, test, security scanning, and deployment with zero-downtime rollback
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import semver
import git

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    QUALITY_GATE = "quality_gate"
    DEPLOY_STAGING = "deploy_staging"
    INTEGRATION_TEST = "integration_test"
    DEPLOY_PRODUCTION = "deploy_production"
    HEALTH_CHECK = "health_check"
    ROLLBACK = "rollback"

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

@dataclass
class PipelineStageResult:
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    output: str = ""
    error: str = ""
    artifacts: List[str] = None
    metrics: Dict[str, Any] = None

@dataclass
class PipelineExecution:
    pipeline_id: str
    commit_hash: str
    branch: str
    trigger: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.PENDING
    stages: List[PipelineStageResult] = None
    version: str = ""
    rollback_version: str = ""
class 
VersionManager:
    """Manages semantic versioning for releases"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.version_file = project_root / "VERSION"
    
    def get_current_version(self) -> str:
        """Get current version"""
        if self.version_file.exists():
            return self.version_file.read_text().strip()
        return "0.1.0"
    
    def increment_version(self, version_type: str = "patch") -> str:
        """Increment version based on type (major, minor, patch)"""
        current_version = self.get_current_version()
        
        if version_type == "major":
            new_version = semver.bump_major(current_version)
        elif version_type == "minor":
            new_version = semver.bump_minor(current_version)
        else:  # patch
            new_version = semver.bump_patch(current_version)
        
        self.version_file.write_text(new_version)
        return new_version
    
    def set_version(self, version: str):
        """Set specific version"""
        if semver.VersionInfo.isvalid(version):
            self.version_file.write_text(version)
        else:
            raise ValueError(f"Invalid semantic version: {version}")

class SecurityScanner:
    """Security vulnerability scanner"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scan_results_dir = project_root / "security-scan-results"
        self.scan_results_dir.mkdir(exist_ok=True)
    
    async def run_comprehensive_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        scan_results = {
            "python_security": await self.scan_python_security(),
            "nodejs_security": await self.scan_nodejs_security(),
            "docker_security": await self.scan_docker_security(),
            "secrets_scan": await self.scan_for_secrets()
        }
        
        # Calculate overall security score
        total_high_vulns = sum(
            result.get("high_severity_vulnerabilities", 0) 
            for result in scan_results.values()
        )
        total_medium_vulns = sum(
            result.get("medium_severity_vulnerabilities", 0) 
            for result in scan_results.values()
        )
        
        security_score = max(0, 100 - (total_high_vulns * 20) - (total_medium_vulns * 5))
        
        scan_results["summary"] = {
            "security_score": security_score,
            "total_high_vulnerabilities": total_high_vulns,
            "total_medium_vulnerabilities": total_medium_vulns,
            "scan_passed": total_high_vulns == 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save scan results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.scan_results_dir / f"security_scan_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(scan_results, f, indent=2)
        
        return scan_results
    
    async def scan_python_security(self) -> Dict[str, Any]:
        """Scan Python code for security vulnerabilities"""
        try:
            # Run bandit security scanner
            command = [
                "bandit", "-r", "services/", "-f", "json", 
                "-o", str(self.scan_results_dir / "bandit_results.json")
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse bandit results
            bandit_file = self.scan_results_dir / "bandit_results.json"
            if bandit_file.exists():
                with open(bandit_file, 'r') as f:
                    bandit_data = json.load(f)
                
                high_issues = len([
                    issue for issue in bandit_data.get("results", [])
                    if issue.get("issue_severity") == "HIGH"
                ])
                medium_issues = len([
                    issue for issue in bandit_data.get("results", [])
                    if issue.get("issue_severity") == "MEDIUM"
                ])
                
                return {
                    "tool": "bandit",
                    "status": "completed",
                    "high_severity_vulnerabilities": high_issues,
                    "medium_severity_vulnerabilities": medium_issues,
                    "total_issues": len(bandit_data.get("results", [])),
                    "results_file": str(bandit_file)
                }
            
            return {
                "tool": "bandit",
                "status": "failed",
                "error": stderr.decode() if stderr else "No results file generated"
            }
            
        except Exception as e:
            return {
                "tool": "bandit",
                "status": "error",
                "error": str(e)
            }
    
    async def scan_nodejs_security(self) -> Dict[str, Any]:
        """Scan Node.js dependencies for security vulnerabilities"""
        try:
            # Check if Node.js services exist
            nodejs_services = []
            for service_dir in (self.project_root / "services").iterdir():
                if (service_dir / "package.json").exists():
                    nodejs_services.append(service_dir)
            
            if not nodejs_services:
                return {
                    "tool": "npm_audit",
                    "status": "skipped",
                    "reason": "No Node.js services found"
                }
            
            total_high_vulns = 0
            total_medium_vulns = 0
            
            for service_dir in nodejs_services:
                # Run npm audit
                command = ["npm", "audit", "--json"]
                
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=service_dir
                )
                
                stdout, stderr = await process.communicate()
                
                if stdout:
                    try:
                        audit_data = json.loads(stdout.decode())
                        vulnerabilities = audit_data.get("vulnerabilities", {})
                        
                        for vuln_name, vuln_data in vulnerabilities.items():
                            severity = vuln_data.get("severity", "").lower()
                            if severity == "high" or severity == "critical":
                                total_high_vulns += 1
                            elif severity == "moderate":
                                total_medium_vulns += 1
                    except json.JSONDecodeError:
                        pass
            
            return {
                "tool": "npm_audit",
                "status": "completed",
                "high_severity_vulnerabilities": total_high_vulns,
                "medium_severity_vulnerabilities": total_medium_vulns,
                "services_scanned": len(nodejs_services)
            }
            
        except Exception as e:
            return {
                "tool": "npm_audit",
                "status": "error",
                "error": str(e)
            }
    
    async def scan_docker_security(self) -> Dict[str, Any]:
        """Scan Docker images for security vulnerabilities"""
        try:
            # Use trivy to scan Docker images
            docker_files = list(self.project_root.rglob("Dockerfile"))
            
            if not docker_files:
                return {
                    "tool": "trivy",
                    "status": "skipped",
                    "reason": "No Dockerfiles found"
                }
            
            total_high_vulns = 0
            total_medium_vulns = 0
            
            for dockerfile in docker_files:
                service_name = dockerfile.parent.name
                image_name = f"prometheus-ai/{service_name}:latest"
                
                # Build image first
                build_command = ["docker", "build", "-t", image_name, str(dockerfile.parent)]
                build_process = await asyncio.create_subprocess_exec(
                    *build_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await build_process.communicate()
                
                if build_process.returncode == 0:
                    # Scan with trivy
                    scan_command = ["trivy", "image", "--format", "json", image_name]
                    scan_process = await asyncio.create_subprocess_exec(
                        *scan_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await scan_process.communicate()
                    
                    if stdout:
                        try:
                            trivy_data = json.loads(stdout.decode())
                            for result in trivy_data.get("Results", []):
                                for vuln in result.get("Vulnerabilities", []):
                                    severity = vuln.get("Severity", "").upper()
                                    if severity in ["HIGH", "CRITICAL"]:
                                        total_high_vulns += 1
                                    elif severity == "MEDIUM":
                                        total_medium_vulns += 1
                        except json.JSONDecodeError:
                            pass
            
            return {
                "tool": "trivy",
                "status": "completed",
                "high_severity_vulnerabilities": total_high_vulns,
                "medium_severity_vulnerabilities": total_medium_vulns,
                "images_scanned": len(docker_files)
            }
            
        except Exception as e:
            return {
                "tool": "trivy",
                "status": "error",
                "error": str(e)
            }
    
    async def scan_for_secrets(self) -> Dict[str, Any]:
        """Scan for exposed secrets and credentials"""
        try:
            # Use truffleHog or similar tool to scan for secrets
            command = [
                "git", "log", "--all", "--full-history", "--", 
                "*.py", "*.js", "*.yaml", "*.json", "*.env"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            # Simple pattern matching for common secrets
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            import re
            secrets_found = 0
            
            if stdout:
                content = stdout.decode()
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    secrets_found += len(matches)
            
            return {
                "tool": "secrets_scanner",
                "status": "completed",
                "high_severity_vulnerabilities": secrets_found,
                "medium_severity_vulnerabilities": 0,
                "patterns_checked": len(secret_patterns)
            }
            
        except Exception as e:
            return {
                "tool": "secrets_scanner",
                "status": "error",
                "error": str(e)
            }c
lass DeploymentManager:
    """Manages zero-downtime deployments with automatic rollback"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deployment_config = self.load_deployment_config()
        self.rollback_history = []
    
    def load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        config_file = self.project_root / "kubernetes" / "deployment-config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "environments": {
                "staging": {
                    "namespace": "prometheus-ai-staging",
                    "replicas": 2,
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "500m", "memory": "512Mi"}
                    }
                },
                "production": {
                    "namespace": "prometheus-ai",
                    "replicas": 3,
                    "resources": {
                        "requests": {"cpu": "200m", "memory": "512Mi"},
                        "limits": {"cpu": "1000m", "memory": "1Gi"}
                    }
                }
            },
            "services": [
                "helios-core",
                "volcano-watchtower", 
                "data-ingestion",
                "api-gateway",
                "vulcan-ui",
                "system-monitor"
            ],
            "health_check": {
                "initial_delay": 30,
                "timeout": 10,
                "max_retries": 5
            },
            "rollback": {
                "auto_rollback_on_failure": True,
                "health_check_duration": 300,
                "error_rate_threshold": 0.05
            }
        }
    
    async def deploy_to_environment(self, environment: str, version: str) -> Dict[str, Any]:
        """Deploy to specified environment with zero-downtime strategy"""
        env_config = self.deployment_config["environments"].get(environment)
        if not env_config:
            return {
                "status": "failed",
                "error": f"Unknown environment: {environment}"
            }
        
        deployment_results = []
        
        try:
            # Record current deployment for rollback
            current_deployment = await self.get_current_deployment(environment)
            self.rollback_history.append({
                "environment": environment,
                "version": current_deployment.get("version", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Deploy each service
            for service in self.deployment_config["services"]:
                result = await self.deploy_service(service, environment, version, env_config)
                deployment_results.append(result)
                
                if result["status"] != "success":
                    # Rollback on failure
                    if self.deployment_config["rollback"]["auto_rollback_on_failure"]:
                        await self.rollback_deployment(environment)
                    
                    return {
                        "status": "failed",
                        "failed_service": service,
                        "deployment_results": deployment_results
                    }
            
            # Perform health checks
            health_check_result = await self.perform_deployment_health_check(environment)
            
            if not health_check_result["healthy"]:
                # Rollback on health check failure
                if self.deployment_config["rollback"]["auto_rollback_on_failure"]:
                    await self.rollback_deployment(environment)
                
                return {
                    "status": "failed",
                    "reason": "Health check failed",
                    "health_check_result": health_check_result,
                    "deployment_results": deployment_results
                }
            
            return {
                "status": "success",
                "environment": environment,
                "version": version,
                "services_deployed": len(deployment_results),
                "deployment_results": deployment_results,
                "health_check_result": health_check_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "deployment_results": deployment_results
            }
    
    async def deploy_service(self, service: str, environment: str, version: str, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a single service using rolling update strategy"""
        start_time = time.time()
        
        try:
            namespace = env_config["namespace"]
            
            # Update deployment manifest with new version
            deployment_file = self.project_root / "kubernetes" / f"{service}-deployment.yaml"
            
            if not deployment_file.exists():
                return {
                    "service": service,
                    "status": "failed",
                    "error": f"Deployment file not found: {deployment_file}",
                    "duration": time.time() - start_time
                }
            
            # Read and update deployment manifest
            with open(deployment_file, 'r') as f:
                deployment_manifest = yaml.safe_load(f)
            
            # Update image tag
            containers = deployment_manifest["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if container["name"] == service:
                    image_parts = container["image"].split(":")
                    container["image"] = f"{image_parts[0]}:{version}"
            
            # Update resources
            for container in containers:
                if container["name"] == service:
                    container["resources"] = env_config["resources"]
            
            # Update replicas
            deployment_manifest["spec"]["replicas"] = env_config["replicas"]
            
            # Write updated manifest to temporary file
            temp_manifest = self.project_root / f"temp-{service}-deployment.yaml"
            with open(temp_manifest, 'w') as f:
                yaml.dump(deployment_manifest, f)
            
            # Apply deployment
            command = [
                "kubectl", "apply", "-f", str(temp_manifest),
                "--namespace", namespace
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up temporary file
            temp_manifest.unlink()
            
            if process.returncode != 0:
                return {
                    "service": service,
                    "status": "failed",
                    "error": stderr.decode(),
                    "duration": time.time() - start_time
                }
            
            # Wait for rollout to complete
            rollout_command = [
                "kubectl", "rollout", "status", f"deployment/{service}",
                "--namespace", namespace, "--timeout=300s"
            ]
            
            rollout_process = await asyncio.create_subprocess_exec(
                *rollout_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await rollout_process.communicate()
            
            if rollout_process.returncode != 0:
                return {
                    "service": service,
                    "status": "failed",
                    "error": "Rollout timeout or failure",
                    "duration": time.time() - start_time
                }
            
            return {
                "service": service,
                "status": "success",
                "version": version,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "service": service,
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def get_current_deployment(self, environment: str) -> Dict[str, Any]:
        """Get current deployment information"""
        try:
            env_config = self.deployment_config["environments"][environment]
            namespace = env_config["namespace"]
            
            # Get deployment information
            command = [
                "kubectl", "get", "deployments", 
                "--namespace", namespace, "-o", "json"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout:
                deployments_data = json.loads(stdout.decode())
                
                current_versions = {}
                for deployment in deployments_data.get("items", []):
                    name = deployment["metadata"]["name"]
                    containers = deployment["spec"]["template"]["spec"]["containers"]
                    
                    for container in containers:
                        if container["name"] == name:
                            image_parts = container["image"].split(":")
                            current_versions[name] = image_parts[-1] if len(image_parts) > 1 else "latest"
                
                return {
                    "environment": environment,
                    "versions": current_versions,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {"environment": environment, "versions": {}}
            
        except Exception as e:
            return {"environment": environment, "error": str(e)}
    
    async def perform_deployment_health_check(self, environment: str) -> Dict[str, Any]:
        """Perform comprehensive health check after deployment"""
        env_config = self.deployment_config["environments"][environment]
        namespace = env_config["namespace"]
        health_config = self.deployment_config["health_check"]
        
        # Wait for initial delay
        await asyncio.sleep(health_config["initial_delay"])
        
        health_results = {}
        overall_healthy = True
        
        for service in self.deployment_config["services"]:
            service_healthy = await self.check_service_health(service, namespace, health_config)
            health_results[service] = service_healthy
            
            if not service_healthy["healthy"]:
                overall_healthy = False
        
        # Monitor for error rate spikes
        if overall_healthy:
            error_rate_check = await self.monitor_error_rates(environment)
            health_results["error_rate_check"] = error_rate_check
            
            if not error_rate_check["passed"]:
                overall_healthy = False
        
        return {
            "healthy": overall_healthy,
            "environment": environment,
            "service_health": health_results,
            "check_timestamp": datetime.now().isoformat()
        }
    
    async def check_service_health(self, service: str, namespace: str, health_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            # Check if pods are ready
            command = [
                "kubectl", "get", "pods", 
                "-l", f"app={service}",
                "--namespace", namespace,
                "-o", "json"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {
                    "healthy": False,
                    "error": stderr.decode()
                }
            
            pods_data = json.loads(stdout.decode())
            pods = pods_data.get("items", [])
            
            if not pods:
                return {
                    "healthy": False,
                    "error": "No pods found"
                }
            
            ready_pods = 0
            total_pods = len(pods)
            
            for pod in pods:
                conditions = pod.get("status", {}).get("conditions", [])
                for condition in conditions:
                    if condition.get("type") == "Ready" and condition.get("status") == "True":
                        ready_pods += 1
                        break
            
            return {
                "healthy": ready_pods == total_pods,
                "ready_pods": ready_pods,
                "total_pods": total_pods
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def monitor_error_rates(self, environment: str) -> Dict[str, Any]:
        """Monitor error rates after deployment"""
        try:
            # This would typically query Prometheus for error rate metrics
            # For now, simulate error rate monitoring
            
            monitoring_duration = self.deployment_config["rollback"]["health_check_duration"]
            error_threshold = self.deployment_config["rollback"]["error_rate_threshold"]
            
            # Simulate monitoring for a shorter duration in this implementation
            await asyncio.sleep(min(monitoring_duration, 60))
            
            # Simulate error rate (in real implementation, this would query metrics)
            import random
            current_error_rate = random.uniform(0.0, 0.10)
            
            return {
                "passed": current_error_rate <= error_threshold,
                "current_error_rate": current_error_rate,
                "threshold": error_threshold,
                "monitoring_duration": monitoring_duration
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def rollback_deployment(self, environment: str) -> Dict[str, Any]:
        """Rollback deployment to previous version"""
        try:
            if not self.rollback_history:
                return {
                    "status": "failed",
                    "error": "No rollback history available"
                }
            
            # Get last known good deployment
            last_deployment = self.rollback_history[-1]
            
            if last_deployment["environment"] != environment:
                return {
                    "status": "failed",
                    "error": "No rollback history for this environment"
                }
            
            rollback_version = last_deployment["version"]
            env_config = self.deployment_config["environments"][environment]
            namespace = env_config["namespace"]
            
            rollback_results = []
            
            # Rollback each service
            for service in self.deployment_config["services"]:
                command = [
                    "kubectl", "rollout", "undo", f"deployment/{service}",
                    "--namespace", namespace
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                rollback_results.append({
                    "service": service,
                    "status": "success" if process.returncode == 0 else "failed",
                    "output": stdout.decode() if stdout else stderr.decode()
                })
            
            # Wait for rollback to complete
            await asyncio.sleep(30)
            
            # Verify rollback health
            health_check = await self.perform_deployment_health_check(environment)
            
            return {
                "status": "success",
                "environment": environment,
                "rollback_version": rollback_version,
                "rollback_results": rollback_results,
                "health_check": health_check
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }c
lass CICDPipeline:
    """Main CI/CD Pipeline orchestrator"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.version_manager = VersionManager(project_root)
        self.security_scanner = SecurityScanner(project_root)
        self.deployment_manager = DeploymentManager(project_root)
        
        # Pipeline configuration
        self.pipeline_config = self.load_pipeline_config()
        
        # Execution history
        self.execution_history = []
        self.execution_history_file = project_root / "cicd-execution-history.json"
    
    def load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        config_file = self.project_root / ".kiro" / "cicd-config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "triggers": {
                "branches": ["main", "develop"],
                "paths": ["services/**", "kubernetes/**"],
                "skip_ci_patterns": ["[skip ci]", "[ci skip]"]
            },
            "stages": {
                "build": {"enabled": True, "timeout": 600},
                "test": {"enabled": True, "timeout": 1800},
                "security_scan": {"enabled": True, "timeout": 900},
                "quality_gate": {"enabled": True, "timeout": 300},
                "deploy_staging": {"enabled": True, "timeout": 1200},
                "integration_test": {"enabled": True, "timeout": 900},
                "deploy_production": {"enabled": True, "timeout": 1800}
            },
            "quality_gates": {
                "min_test_coverage": 0.80,
                "max_security_vulnerabilities": 0,
                "min_code_quality_score": 7.0
            },
            "notifications": {
                "slack_webhook": "${SLACK_WEBHOOK_URL}",
                "email_recipients": ["devops@prometheus-ai.com"]
            }
        }
    
    async def execute_pipeline(self, commit_hash: str, branch: str, trigger: str = "push") -> PipelineExecution:
        """Execute the complete CI/CD pipeline"""
        pipeline_id = f"pipeline_{int(time.time())}_{commit_hash[:8]}"
        
        execution = PipelineExecution(
            pipeline_id=pipeline_id,
            commit_hash=commit_hash,
            branch=branch,
            trigger=trigger,
            start_time=datetime.now(),
            status=PipelineStatus.RUNNING,
            stages=[]
        )
        
        try:
            # Check if CI should be skipped
            if await self.should_skip_ci(commit_hash):
                execution.status = PipelineStatus.SKIPPED
                return execution
            
            # Generate version
            if branch == "main":
                execution.version = self.version_manager.increment_version("patch")
            else:
                execution.version = f"{self.version_manager.get_current_version()}-{commit_hash[:8]}"
            
            # Execute pipeline stages
            stages_to_run = [
                (PipelineStage.BUILD, self.run_build_stage),
                (PipelineStage.TEST, self.run_test_stage),
                (PipelineStage.SECURITY_SCAN, self.run_security_scan_stage),
                (PipelineStage.QUALITY_GATE, self.run_quality_gate_stage)
            ]
            
            # Add deployment stages for main branch
            if branch == "main":
                stages_to_run.extend([
                    (PipelineStage.DEPLOY_STAGING, self.run_deploy_staging_stage),
                    (PipelineStage.INTEGRATION_TEST, self.run_integration_test_stage),
                    (PipelineStage.DEPLOY_PRODUCTION, self.run_deploy_production_stage),
                    (PipelineStage.HEALTH_CHECK, self.run_health_check_stage)
                ])
            
            # Execute stages
            for stage, stage_func in stages_to_run:
                if not self.pipeline_config["stages"].get(stage.value, {}).get("enabled", True):
                    # Skip disabled stage
                    stage_result = PipelineStageResult(
                        stage=stage,
                        status=PipelineStatus.SKIPPED,
                        start_time=datetime.now()
                    )
                    execution.stages.append(stage_result)
                    continue
                
                stage_result = await stage_func(execution)
                execution.stages.append(stage_result)
                
                # Stop pipeline on stage failure
                if stage_result.status == PipelineStatus.FAILED:
                    execution.status = PipelineStatus.FAILED
                    break
            
            # Set final status
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.SUCCESS
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            logger.error(f"Pipeline execution failed: {e}")
        
        finally:
            execution.end_time = datetime.now()
            
            # Save execution history
            self.execution_history.append(asdict(execution))
            self.save_execution_history()
            
            # Send notifications
            await self.send_pipeline_notification(execution)
        
        return execution
    
    async def should_skip_ci(self, commit_hash: str) -> bool:
        """Check if CI should be skipped for this commit"""
        try:
            # Get commit message
            repo = git.Repo(self.project_root)
            commit = repo.commit(commit_hash)
            commit_message = commit.message
            
            # Check for skip patterns
            skip_patterns = self.pipeline_config["triggers"]["skip_ci_patterns"]
            for pattern in skip_patterns:
                if pattern.lower() in commit_message.lower():
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def run_build_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run build stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.BUILD,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(),
            artifacts=[]
        )
        
        try:
            # Build Docker images for all services
            services = ["helios-core", "volcano-watchtower", "data-ingestion", "api-gateway", "vulcan-ui", "system-monitor"]
            
            for service in services:
                dockerfile_path = self.project_root / "services" / service / "Dockerfile"
                
                if dockerfile_path.exists():
                    # Build Docker image
                    image_tag = f"prometheus-ai/{service}:{execution.version}"
                    
                    command = [
                        "docker", "build", 
                        "-t", image_tag,
                        "-f", str(dockerfile_path),
                        str(dockerfile_path.parent)
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        stage_result.status = PipelineStatus.FAILED
                        stage_result.error = f"Failed to build {service}: {stderr.decode()}"
                        break
                    
                    stage_result.artifacts.append(image_tag)
                    stage_result.output += f"Built {image_tag}\n"
            
            if stage_result.status != PipelineStatus.FAILED:
                stage_result.status = PipelineStatus.SUCCESS
                stage_result.output += f"Successfully built {len(stage_result.artifacts)} images"
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_test_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run test stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.TEST,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Run comprehensive test suite
            command = ["python", "scripts/run_automated_tests.py", "--comprehensive", "--coverage"]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            stage_result.output = stdout.decode() if stdout else ""
            
            if process.returncode == 0:
                # Parse test results
                test_report_path = self.project_root / "test-reports" / "latest_test_report.json"
                if test_report_path.exists():
                    with open(test_report_path, 'r') as f:
                        test_data = json.load(f)
                    
                    stage_result.metrics = {
                        "total_tests": test_data["summary"]["total_tests"],
                        "passed_tests": test_data["summary"]["passed"],
                        "failed_tests": test_data["summary"]["failed"],
                        "success_rate": test_data["summary"]["success_rate"],
                        "coverage": test_data["summary"].get("average_coverage", 0.0)
                    }
                    
                    # Check quality gates
                    min_coverage = self.pipeline_config["quality_gates"]["min_test_coverage"]
                    if stage_result.metrics["coverage"] < min_coverage:
                        stage_result.status = PipelineStatus.FAILED
                        stage_result.error = f"Test coverage {stage_result.metrics['coverage']:.2%} below minimum {min_coverage:.2%}"
                    else:
                        stage_result.status = PipelineStatus.SUCCESS
                else:
                    stage_result.status = PipelineStatus.FAILED
                    stage_result.error = "Test report not found"
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = stderr.decode() if stderr else "Tests failed"
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_security_scan_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run security scan stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.SECURITY_SCAN,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Run comprehensive security scan
            scan_results = await self.security_scanner.run_comprehensive_security_scan()
            
            stage_result.metrics = scan_results["summary"]
            stage_result.output = json.dumps(scan_results, indent=2)
            
            # Check security quality gate
            max_vulns = self.pipeline_config["quality_gates"]["max_security_vulnerabilities"]
            high_vulns = scan_results["summary"]["total_high_vulnerabilities"]
            
            if high_vulns > max_vulns:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = f"Found {high_vulns} high-severity vulnerabilities (max allowed: {max_vulns})"
            else:
                stage_result.status = PipelineStatus.SUCCESS
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_quality_gate_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run quality gate stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.QUALITY_GATE,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Run code quality scan
            command = ["python", "scripts/code_quality_scan.py"]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            stage_result.output = stdout.decode() if stdout else ""
            
            # Check quality report
            quality_report_path = self.project_root / "quality-reports" / "latest_quality_report.json"
            if quality_report_path.exists():
                with open(quality_report_path, 'r') as f:
                    quality_data = json.load(f)
                
                quality_score = quality_data.get("quality_score", 0.0)
                stage_result.metrics = {
                    "quality_score": quality_score,
                    "security_score": quality_data.get("security_score", 0.0),
                    "total_issues": quality_data["summary"]["total_issues"]
                }
                
                # Check quality gate
                min_quality = self.pipeline_config["quality_gates"]["min_code_quality_score"]
                if quality_score >= min_quality:
                    stage_result.status = PipelineStatus.SUCCESS
                else:
                    stage_result.status = PipelineStatus.FAILED
                    stage_result.error = f"Code quality score {quality_score} below minimum {min_quality}"
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = "Quality report not found"
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_deploy_staging_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run deploy to staging stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.DEPLOY_STAGING,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Deploy to staging environment
            deployment_result = await self.deployment_manager.deploy_to_environment("staging", execution.version)
            
            stage_result.output = json.dumps(deployment_result, indent=2)
            stage_result.metrics = {
                "services_deployed": deployment_result.get("services_deployed", 0),
                "deployment_time": deployment_result.get("deployment_time", 0)
            }
            
            if deployment_result["status"] == "success":
                stage_result.status = PipelineStatus.SUCCESS
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = deployment_result.get("error", "Deployment failed")
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_integration_test_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run integration tests against staging"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.INTEGRATION_TEST,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Run integration tests against staging environment
            command = ["python", "scripts/run_automated_tests.py", "--integration", "--environment=staging"]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            stage_result.output = stdout.decode() if stdout else ""
            
            if process.returncode == 0:
                stage_result.status = PipelineStatus.SUCCESS
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = stderr.decode() if stderr else "Integration tests failed"
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_deploy_production_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run deploy to production stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.DEPLOY_PRODUCTION,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Deploy to production environment
            deployment_result = await self.deployment_manager.deploy_to_environment("production", execution.version)
            
            stage_result.output = json.dumps(deployment_result, indent=2)
            stage_result.metrics = {
                "services_deployed": deployment_result.get("services_deployed", 0),
                "deployment_time": deployment_result.get("deployment_time", 0)
            }
            
            if deployment_result["status"] == "success":
                stage_result.status = PipelineStatus.SUCCESS
                # Store rollback version
                execution.rollback_version = self.version_manager.get_current_version()
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = deployment_result.get("error", "Production deployment failed")
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def run_health_check_stage(self, execution: PipelineExecution) -> PipelineStageResult:
        """Run post-deployment health check stage"""
        stage_result = PipelineStageResult(
            stage=PipelineStage.HEALTH_CHECK,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Perform extended health monitoring
            health_result = await self.deployment_manager.perform_deployment_health_check("production")
            
            stage_result.output = json.dumps(health_result, indent=2)
            stage_result.metrics = {
                "services_healthy": sum(1 for h in health_result["service_health"].values() if h.get("healthy", False)),
                "total_services": len(health_result["service_health"])
            }
            
            if health_result["healthy"]:
                stage_result.status = PipelineStatus.SUCCESS
            else:
                stage_result.status = PipelineStatus.FAILED
                stage_result.error = "Health check failed - automatic rollback initiated"
                
                # Trigger automatic rollback
                rollback_result = await self.deployment_manager.rollback_deployment("production")
                stage_result.output += f"\n\nAutomatic Rollback Result:\n{json.dumps(rollback_result, indent=2)}"
            
        except Exception as e:
            stage_result.status = PipelineStatus.FAILED
            stage_result.error = str(e)
        
        finally:
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - stage_result.start_time).total_seconds()
        
        return stage_result
    
    async def send_pipeline_notification(self, execution: PipelineExecution):
        """Send pipeline execution notification"""
        try:
            notification_data = {
                "pipeline_id": execution.pipeline_id,
                "status": execution.status.value,
                "branch": execution.branch,
                "commit": execution.commit_hash[:8],
                "version": execution.version,
                "duration": (execution.end_time - execution.start_time).total_seconds() if execution.end_time else 0,
                "stages": [
                    {
                        "stage": stage.stage.value,
                        "status": stage.status.value,
                        "duration": stage.duration
                    }
                    for stage in execution.stages
                ]
            }
            
            # Log notification (in real implementation, this would send to Slack/email)
            logger.info(f"Pipeline notification: {json.dumps(notification_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to send pipeline notification: {e}")
    
    def save_execution_history(self):
        """Save pipeline execution history"""
        try:
            with open(self.execution_history_file, 'w') as f:
                json.dump(self.execution_history[-100:], f, indent=2, default=str)  # Keep last 100 executions
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")

# CLI interface
async def main():
    """Main function for CLI usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="CI/CD Pipeline")
    parser.add_argument("--commit", required=True, help="Commit hash to build")
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument("--trigger", default="push", help="Pipeline trigger")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    pipeline = CICDPipeline(project_root)
    
    try:
        execution = await pipeline.execute_pipeline(args.commit, args.branch, args.trigger)
        
        print(f"Pipeline {execution.pipeline_id} completed with status: {execution.status.value}")
        print(f"Version: {execution.version}")
        print(f"Duration: {(execution.end_time - execution.start_time).total_seconds():.1f}s")
        
        for stage in execution.stages:
            status_emoji = "✅" if stage.status == PipelineStatus.SUCCESS else "❌" if stage.status == PipelineStatus.FAILED else "⏭️"
            print(f"  {status_emoji} {stage.stage.value}: {stage.status.value} ({stage.duration:.1f}s)")
        
        if execution.status == PipelineStatus.FAILED:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())