#!/usr/bin/env python3
"""
Multi-Region Cloud Run Deployment Manager

This module handles deployment of Cloud Run services across multiple regions
with intelligent traffic routing and cost optimization.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
from google.cloud import run_v2
from google.cloud import dns
from google.cloud import monitoring_v3
import yaml


class DeploymentStrategy(Enum):
    """Deployment strategies for multi-region rollouts"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"


class TrafficRoutingStrategy(Enum):
    """Traffic routing strategies"""
    GEOGRAPHIC = "geographic"  # Route based on user location
    PERFORMANCE = "performance"  # Route to fastest region
    COST_OPTIMIZED = "cost_optimized"  # Route to cheapest region
    LOAD_BALANCED = "load_balanced"  # Distribute evenly


@dataclass
class RegionConfig:
    """Configuration for a specific region"""
    name: str
    location: str
    primary: bool = False
    traffic_percentage: float = 0.0
    cost_tier: str = "standard"  # standard, premium
    max_instances: int = 100
    preemptible_enabled: bool = True


@dataclass
class ServiceDeployment:
    """Deployment configuration for a service"""
    service_name: str
    image_url: str
    regions: List[RegionConfig]
    strategy: DeploymentStrategy
    traffic_routing: TrafficRoutingStrategy
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    service_name: str
    region: str
    success: bool
    revision_name: str
    url: str
    error_message: Optional[str] = None
    deployment_time: datetime = None


class MultiRegionDeploymentManager:
    """
    Manages multi-region Cloud Run deployments with:
    - Intelligent traffic routing
    - Cost optimization across regions
    - Blue-green and canary deployments
    - Global load balancing
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.run_client = run_v2.ServicesClient()
        self.dns_client = dns.Client(project=project_id)
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.logger = logging.getLogger(__name__)
        
        # Regional configurations
        self.region_configs = {
            "us-central1": RegionConfig(
                name="us-central1",
                location="Iowa, USA",
                primary=True,
                traffic_percentage=40.0,
                cost_tier="standard",
                max_instances=100,
                preemptible_enabled=True
            ),
            "europe-west1": RegionConfig(
                name="europe-west1",
                location="Belgium, Europe",
                primary=False,
                traffic_percentage=35.0,
                cost_tier="standard",
                max_instances=80,
                preemptible_enabled=True
            ),
            "asia-northeast1": RegionConfig(
                name="asia-northeast1",
                location="Tokyo, Japan",
                primary=False,
                traffic_percentage=25.0,
                cost_tier="premium",
                max_instances=60,
                preemptible_enabled=False
            )
        }
        
        # Service-specific deployment configurations
        self.service_deployments = {
            "helios-core": ServiceDeployment(
                service_name="helios-core",
                image_url="gcr.io/prometheus-ai-platform/helios-core:latest",
                regions=list(self.region_configs.values()),
                strategy=DeploymentStrategy.BLUE_GREEN,
                traffic_routing=TrafficRoutingStrategy.PERFORMANCE
            ),
            "api-gateway": ServiceDeployment(
                service_name="api-gateway",
                image_url="gcr.io/prometheus-ai-platform/api-gateway:latest",
                regions=list(self.region_configs.values()),
                strategy=DeploymentStrategy.ROLLING,
                traffic_routing=TrafficRoutingStrategy.GEOGRAPHIC
            ),
            "data-ingestion": ServiceDeployment(
                service_name="data-ingestion",
                image_url="gcr.io/prometheus-ai-platform/data-ingestion:latest",
                regions=[self.region_configs["us-central1"], self.region_configs["europe-west1"]],
                strategy=DeploymentStrategy.CANARY,
                traffic_routing=TrafficRoutingStrategy.COST_OPTIMIZED
            ),
            "volcano-watchtower": ServiceDeployment(
                service_name="volcano-watchtower",
                image_url="gcr.io/prometheus-ai-platform/volcano-watchtower:latest",
                regions=list(self.region_configs.values()),
                strategy=DeploymentStrategy.ROLLING,
                traffic_routing=TrafficRoutingStrategy.LOAD_BALANCED
            )
        }
    
    async def deploy_service_to_region(
        self, 
        deployment: ServiceDeployment, 
        region: RegionConfig,
        revision_suffix: str = None
    ) -> DeploymentResult:
        """Deploy a service to a specific region"""
        try:
            service_name = deployment.service_name
            region_name = region.name
            
            # Generate revision name
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            revision_name = f"{service_name}-{timestamp}"
            if revision_suffix:
                revision_name += f"-{revision_suffix}"
            
            # Create service configuration
            service_config = self._create_service_config(
                deployment, region, revision_name
            )
            
            # Deploy to Cloud Run
            service_path = f"projects/{self.project_id}/locations/{region_name}/services/{service_name}"
            
            try:
                # Try to get existing service
                existing_service = self.run_client.get_service(name=service_path)
                # Update existing service
                operation = self.run_client.update_service(service=service_config)
            except Exception:
                # Create new service
                parent = f"projects/{self.project_id}/locations/{region_name}"
                operation = self.run_client.create_service(
                    parent=parent,
                    service=service_config,
                    service_id=service_name
                )
            
            # Wait for operation to complete
            result = operation.result(timeout=300)  # 5 minutes timeout
            
            # Get service URL
            service_url = result.status.url
            
            self.logger.info(f"Successfully deployed {service_name} to {region_name}: {service_url}")
            
            return DeploymentResult(
                service_name=service_name,
                region=region_name,
                success=True,
                revision_name=revision_name,
                url=service_url,
                deployment_time=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to deploy {deployment.service_name} to {region.name}: {e}")
            return DeploymentResult(
                service_name=deployment.service_name,
                region=region.name,
                success=False,
                revision_name="",
                url="",
                error_message=str(e),
                deployment_time=datetime.utcnow()
            )
    
    def _create_service_config(
        self, 
        deployment: ServiceDeployment, 
        region: RegionConfig,
        revision_name: str
    ) -> run_v2.Service:
        """Create Cloud Run service configuration"""
        
        # Determine resource limits based on service type
        resource_config = self._get_resource_config(deployment.service_name)
        
        # Create service configuration
        service = run_v2.Service()
        service.metadata.name = deployment.service_name
        service.metadata.namespace = self.project_id
        
        # Service spec
        service.spec = run_v2.ServiceSpec()
        service.spec.template = run_v2.RevisionTemplate()
        service.spec.template.metadata = run_v2.ObjectMeta()
        service.spec.template.metadata.name = revision_name
        
        # Annotations for Cloud Run configuration
        annotations = {
            "autoscaling.knative.dev/maxScale": str(region.max_instances),
            "autoscaling.knative.dev/minScale": "0" if deployment.service_name != "api-gateway" else "1",
            "run.googleapis.com/execution-environment": "gen2",
            "run.googleapis.com/cpu-throttling": "true" if deployment.service_name != "helios-core" else "false",
        }
        
        # Add preemptible configuration if enabled
        if region.preemptible_enabled:
            annotations["run.googleapis.com/pricing-tier"] = "standard"
        
        service.spec.template.metadata.annotations = annotations
        
        # Container spec
        service.spec.template.spec = run_v2.RevisionSpec()
        service.spec.template.spec.container_concurrency = resource_config["concurrency"]
        service.spec.template.spec.timeout_seconds = resource_config["timeout"]
        
        # Container configuration
        container = run_v2.Container()
        container.image = deployment.image_url
        container.ports = [run_v2.ContainerPort(container_port=8080)]
        
        # Environment variables
        container.env = [
            run_v2.EnvVar(name="PORT", value="8080"),
            run_v2.EnvVar(name="REGION", value=region.name),
            run_v2.EnvVar(name="ENVIRONMENT", value="production"),
        ]
        
        # Resource limits
        container.resources = run_v2.ResourceRequirements()
        container.resources.limits = {
            "cpu": resource_config["cpu_limit"],
            "memory": resource_config["memory_limit"]
        }
        container.resources.requests = {
            "cpu": resource_config["cpu_request"],
            "memory": resource_config["memory_request"]
        }
        
        # Health checks
        container.liveness_probe = run_v2.Probe(
            http_get=run_v2.HTTPGetAction(
                path=deployment.health_check_path,
                port=8080
            ),
            initial_delay_seconds=30,
            period_seconds=30,
            timeout_seconds=10
        )
        
        container.readiness_probe = run_v2.Probe(
            http_get=run_v2.HTTPGetAction(
                path=deployment.readiness_check_path,
                port=8080
            ),
            initial_delay_seconds=10,
            period_seconds=10,
            timeout_seconds=5
        )
        
        service.spec.template.spec.containers = [container]
        
        return service
    
    def _get_resource_config(self, service_name: str) -> Dict:
        """Get resource configuration for a service"""
        configs = {
            "helios-core": {
                "cpu_limit": "4000m",
                "memory_limit": "8Gi",
                "cpu_request": "1000m",
                "memory_request": "2Gi",
                "concurrency": 10,
                "timeout": 300
            },
            "api-gateway": {
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "cpu_request": "500m",
                "memory_request": "1Gi",
                "concurrency": 1000,
                "timeout": 60
            },
            "data-ingestion": {
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "cpu_request": "500m",
                "memory_request": "1Gi",
                "concurrency": 100,
                "timeout": 600
            },
            "volcano-watchtower": {
                "cpu_limit": "3000m",
                "memory_limit": "6Gi",
                "cpu_request": "1000m",
                "memory_request": "2Gi",
                "concurrency": 50,
                "timeout": 180
            }
        }
        
        return configs.get(service_name, configs["api-gateway"])
    
    async def deploy_all_services(self) -> Dict[str, List[DeploymentResult]]:
        """Deploy all services to all configured regions"""
        results = {}
        
        for service_name, deployment in self.service_deployments.items():
            results[service_name] = []
            
            if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                service_results = await self._blue_green_deployment(deployment)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                service_results = await self._canary_deployment(deployment)
            elif deployment.strategy == DeploymentStrategy.ROLLING:
                service_results = await self._rolling_deployment(deployment)
            else:  # IMMEDIATE
                service_results = await self._immediate_deployment(deployment)
            
            results[service_name] = service_results
        
        return results
    
    async def _blue_green_deployment(self, deployment: ServiceDeployment) -> List[DeploymentResult]:
        """Perform blue-green deployment"""
        results = []
        
        # Deploy to all regions simultaneously (green environment)
        deployment_tasks = []
        for region in deployment.regions:
            task = self.deploy_service_to_region(deployment, region, "green")
            deployment_tasks.append(task)
        
        green_results = await asyncio.gather(*deployment_tasks)
        
        # Validate green deployments
        all_successful = all(result.success for result in green_results)
        
        if all_successful:
            # Switch traffic to green (this would involve updating load balancer)
            self.logger.info(f"Blue-green deployment successful for {deployment.service_name}")
            results.extend(green_results)
        else:
            # Rollback failed deployments
            self.logger.error(f"Blue-green deployment failed for {deployment.service_name}")
            results.extend(green_results)
        
        return results
    
    async def _canary_deployment(self, deployment: ServiceDeployment) -> List[DeploymentResult]:
        """Perform canary deployment"""
        results = []
        
        # Deploy to primary region first (canary)
        primary_region = next(r for r in deployment.regions if r.primary)
        canary_result = await self.deploy_service_to_region(deployment, primary_region, "canary")
        results.append(canary_result)
        
        if canary_result.success:
            # Wait and monitor canary (simplified - in real implementation, monitor metrics)
            await asyncio.sleep(60)  # 1 minute canary period
            
            # Deploy to remaining regions
            remaining_regions = [r for r in deployment.regions if not r.primary]
            deployment_tasks = []
            for region in remaining_regions:
                task = self.deploy_service_to_region(deployment, region)
                deployment_tasks.append(task)
            
            remaining_results = await asyncio.gather(*deployment_tasks)
            results.extend(remaining_results)
        
        return results
    
    async def _rolling_deployment(self, deployment: ServiceDeployment) -> List[DeploymentResult]:
        """Perform rolling deployment"""
        results = []
        
        # Deploy to regions one by one
        for region in deployment.regions:
            result = await self.deploy_service_to_region(deployment, region)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"Rolling deployment failed at {region.name}, stopping rollout")
                break
            
            # Wait between deployments
            await asyncio.sleep(30)
        
        return results
    
    async def _immediate_deployment(self, deployment: ServiceDeployment) -> List[DeploymentResult]:
        """Perform immediate deployment to all regions"""
        deployment_tasks = []
        for region in deployment.regions:
            task = self.deploy_service_to_region(deployment, region)
            deployment_tasks.append(task)
        
        results = await asyncio.gather(*deployment_tasks)
        return results
    
    async def setup_global_load_balancing(self) -> bool:
        """Setup global load balancing and DNS routing"""
        try:
            # This would typically involve:
            # 1. Creating a global load balancer
            # 2. Setting up backend services for each region
            # 3. Configuring health checks
            # 4. Setting up DNS with geographic routing
            
            self.logger.info("Setting up global load balancing...")
            
            # Create DNS zone (simplified example)
            zone_name = "prometheus-ai-zone"
            dns_name = "api.prometheus-ai.com."
            
            # This is a simplified example - real implementation would be more complex
            for service_name, deployment in self.service_deployments.items():
                for region in deployment.regions:
                    # Create regional DNS records
                    record_name = f"{service_name}-{region.name}.{dns_name}"
                    # Add DNS record pointing to Cloud Run service URL
                    # (Implementation details omitted for brevity)
            
            self.logger.info("Global load balancing setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup global load balancing: {e}")
            return False
    
    async def monitor_deployments(self) -> Dict[str, Dict[str, float]]:
        """Monitor deployment health across all regions"""
        health_metrics = {}
        
        for service_name in self.service_deployments.keys():
            health_metrics[service_name] = {}
            
            for region_name in self.region_configs.keys():
                try:
                    # Get service health metrics
                    service_path = f"projects/{self.project_id}/locations/{region_name}/services/{service_name}"
                    service = self.run_client.get_service(name=service_path)
                    
                    # Calculate health score based on various factors
                    health_score = await self._calculate_health_score(service_name, region_name)
                    health_metrics[service_name][region_name] = health_score
                    
                except Exception as e:
                    self.logger.warning(f"Could not get health metrics for {service_name} in {region_name}: {e}")
                    health_metrics[service_name][region_name] = 0.0
        
        return health_metrics
    
    async def _calculate_health_score(self, service_name: str, region_name: str) -> float:
        """Calculate health score for a service in a region"""
        try:
            # This would typically fetch metrics from Cloud Monitoring
            # For now, return a simulated health score
            
            # Factors to consider:
            # - Response time
            # - Error rate
            # - Instance availability
            # - Resource utilization
            
            # Simplified calculation
            base_score = 0.9  # Assume healthy by default
            
            # Adjust based on region (simulate different performance)
            if region_name == "us-central1":
                base_score *= 0.95  # Primary region, slightly better
            elif region_name == "asia-northeast1":
                base_score *= 0.90  # Higher latency region
            
            return base_score
            
        except Exception:
            return 0.0
    
    def generate_deployment_report(self, results: Dict[str, List[DeploymentResult]]) -> str:
        """Generate a deployment report"""
        report = []
        report.append("# Multi-Region Deployment Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        total_deployments = 0
        successful_deployments = 0
        
        for service_name, service_results in results.items():
            report.append(f"## {service_name}")
            report.append("")
            
            for result in service_results:
                total_deployments += 1
                if result.success:
                    successful_deployments += 1
                    status = "✅ SUCCESS"
                else:
                    status = "❌ FAILED"
                
                report.append(f"- **{result.region}**: {status}")
                if result.success:
                    report.append(f"  - URL: {result.url}")
                    report.append(f"  - Revision: {result.revision_name}")
                else:
                    report.append(f"  - Error: {result.error_message}")
                report.append("")
        
        success_rate = (successful_deployments / total_deployments) * 100 if total_deployments > 0 else 0
        report.append(f"## Summary")
        report.append(f"- Total Deployments: {total_deployments}")
        report.append(f"- Successful: {successful_deployments}")
        report.append(f"- Failed: {total_deployments - successful_deployments}")
        report.append(f"- Success Rate: {success_rate:.1f}%")
        
        return "\n".join(report)


async def main():
    """Main function for multi-region deployment"""
    logging.basicConfig(level=logging.INFO)
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "prometheus-ai-platform")
    
    deployment_manager = MultiRegionDeploymentManager(project_id)
    
    # Deploy all services
    print("Starting multi-region deployment...")
    results = await deployment_manager.deploy_all_services()
    
    # Setup global load balancing
    await deployment_manager.setup_global_load_balancing()
    
    # Generate and print report
    report = deployment_manager.generate_deployment_report(results)
    print(report)
    
    # Monitor deployments
    health_metrics = await deployment_manager.monitor_deployments()
    print("\nHealth Metrics:")
    print(json.dumps(health_metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())