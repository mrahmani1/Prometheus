#!/usr/bin/env python3
"""
Cloud Run Serverless Scaling Optimizer

This module provides intelligent scaling and cost optimization for Cloud Run services
based on real-time data processing demands and resource utilization patterns.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import monitoring_v3, run_v2


class ScalingStrategy(Enum):
    """Scaling strategies for different workload patterns"""

    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predicted demand
    HYBRID = "hybrid"  # Combine reactive and predictive
    COST_OPTIMIZED = "cost_optimized"  # Prioritize cost over performance


@dataclass
class ServiceMetrics:
    """Metrics for a Cloud Run service"""

    service_name: str
    region: str
    current_instances: int
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_latency: float
    error_rate: float
    cold_starts: int
    timestamp: datetime


@dataclass
class ScalingDecision:
    """Scaling decision for a service"""

    service_name: str
    region: str
    target_instances: int
    max_instances: int
    cpu_limit: str
    memory_limit: str
    concurrency: int
    use_preemptible: bool
    reasoning: str
    confidence: float


@dataclass
class CostOptimization:
    """Cost optimization configuration"""

    use_spot_instances: bool = True
    preemptible_percentage: float = 0.7
    scale_to_zero_timeout: int = 300  # seconds
    min_instances_override: Dict[str, int] = None
    cost_threshold_per_hour: float = 100.0


class CloudRunOptimizer:
    """
    Intelligent Cloud Run scaling optimizer that handles:
    - Automatic scaling based on real-time metrics
    - Cost optimization with preemptible instances
    - Multi-region deployment management
    - Predictive scaling for geological data processing
    """

    def __init__(self, project_id: str, regions: List[str]):
        self.project_id = project_id
        self.regions = regions
        self.client = run_v2.ServicesClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.logger = logging.getLogger(__name__)

        # Service-specific configurations
        self.service_configs = {
            "helios-core": {
                "cpu_intensive": True,
                "memory_intensive": True,
                "max_concurrency": 10,
                "startup_time": 15,  # seconds
                "scaling_strategy": ScalingStrategy.PREDICTIVE,
            },
            "data-ingestion": {
                "cpu_intensive": False,
                "memory_intensive": True,
                "max_concurrency": 100,
                "startup_time": 5,
                "scaling_strategy": ScalingStrategy.REACTIVE,
            },
            "volcano-watchtower": {
                "cpu_intensive": True,
                "memory_intensive": False,
                "max_concurrency": 50,
                "startup_time": 8,
                "scaling_strategy": ScalingStrategy.HYBRID,
            },
            "api-gateway": {
                "cpu_intensive": False,
                "memory_intensive": False,
                "max_concurrency": 1000,
                "startup_time": 3,
                "scaling_strategy": ScalingStrategy.REACTIVE,
            },
        }

        self.cost_config = CostOptimization()

    async def get_service_metrics(
        self, service_name: str, region: str
    ) -> ServiceMetrics:
        """Retrieve current metrics for a Cloud Run service"""
        try:
            # Get service information
            service_path = (
                f"projects/{self.project_id}/locations/{region}/services/{service_name}"
            )
            service = self.client.get_service(name=service_path)

            # Get monitoring metrics
            metrics = await self._fetch_monitoring_metrics(service_name, region)

            return ServiceMetrics(
                service_name=service_name,
                region=region,
                current_instances=metrics.get("instance_count", 0),
                cpu_utilization=metrics.get("cpu_utilization", 0.0),
                memory_utilization=metrics.get("memory_utilization", 0.0),
                request_rate=metrics.get("request_rate", 0.0),
                response_latency=metrics.get("response_latency", 0.0),
                error_rate=metrics.get("error_rate", 0.0),
                cold_starts=metrics.get("cold_starts", 0),
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            self.logger.error(
                f"Error getting metrics for {service_name} in {region}: {e}"
            )
            return None

    async def _fetch_monitoring_metrics(self, service_name: str, region: str) -> Dict:
        """Fetch metrics from Cloud Monitoring"""
        try:
            project_name = f"projects/{self.project_id}"
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": int(datetime.utcnow().timestamp())},
                    "start_time": {
                        "seconds": int(
                            (datetime.utcnow() - timedelta(minutes=5)).timestamp()
                        )
                    },
                }
            )

            metrics = {}

            # CPU utilization
            cpu_filter = (
                'resource.type="cloud_run_revision" AND '
                f'resource.labels.service_name="{service_name}" AND '
                f'resource.labels.location="{region}"'
            )

            results = self.monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": cpu_filter,
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )

            # Process metrics (simplified for example)
            for result in results:
                if (
                    result.metric.type
                    == "run.googleapis.com/container/cpu/utilizations"
                ):
                    if result.points:
                        metrics["cpu_utilization"] = result.points[0].value.double_value
                elif (
                    result.metric.type
                    == "run.googleapis.com/container/memory/utilizations"
                ):
                    if result.points:
                        metrics["memory_utilization"] = result.points[
                            0
                        ].value.double_value
                elif result.metric.type == "run.googleapis.com/request_count":
                    if result.points:
                        metrics["request_rate"] = result.points[0].value.int64_value

            return metrics

        except Exception as e:
            self.logger.error(f"Error fetching monitoring metrics: {e}")
            return {}

    def calculate_scaling_decision(self, metrics: ServiceMetrics) -> ScalingDecision:
        """Calculate optimal scaling decision based on metrics and strategy"""
        config = self.service_configs.get(metrics.service_name, {})
        strategy = config.get("scaling_strategy", ScalingStrategy.REACTIVE)

        if strategy == ScalingStrategy.REACTIVE:
            return self._reactive_scaling(metrics, config)
        elif strategy == ScalingStrategy.PREDICTIVE:
            return self._predictive_scaling(metrics, config)
        elif strategy == ScalingStrategy.HYBRID:
            return self._hybrid_scaling(metrics, config)
        else:
            return self._cost_optimized_scaling(metrics, config)

    def _reactive_scaling(
        self, metrics: ServiceMetrics, config: Dict
    ) -> ScalingDecision:
        """Reactive scaling based on current metrics"""
        current_instances = metrics.current_instances
        cpu_util = metrics.cpu_utilization
        memory_util = metrics.memory_utilization
        request_rate = metrics.request_rate

        # Calculate target instances based on utilization
        cpu_target_instances = max(1, int(current_instances * cpu_util / 0.7))
        memory_target_instances = max(1, int(current_instances * memory_util / 0.8))
        request_target_instances = max(
            1, int(request_rate / config.get("max_concurrency", 100))
        )

        target_instances = max(
            cpu_target_instances, memory_target_instances, request_target_instances
        )

        # Apply cost optimization
        use_preemptible = self._should_use_preemptible(metrics, target_instances)

        return ScalingDecision(
            service_name=metrics.service_name,
            region=metrics.region,
            target_instances=min(target_instances, 100),  # Cap at 100 instances
            max_instances=100,
            cpu_limit="2000m" if config.get("cpu_intensive") else "1000m",
            memory_limit="4Gi" if config.get("memory_intensive") else "2Gi",
            concurrency=config.get("max_concurrency", 100),
            use_preemptible=use_preemptible,
            reasoning=f"Reactive scaling: CPU={cpu_util:.2f}, Memory={memory_util:.2f}, Requests={request_rate}",
            confidence=0.8,
        )

    def _predictive_scaling(
        self, metrics: ServiceMetrics, config: Dict
    ) -> ScalingDecision:
        """Predictive scaling for geological data processing patterns"""
        # Simulate predictive analysis (in real implementation, use ML models)
        current_hour = datetime.utcnow().hour

        # Geological data processing typically peaks during business hours
        if 8 <= current_hour <= 18:  # Business hours
            scale_factor = 1.5
        elif 19 <= current_hour <= 22:  # Evening processing
            scale_factor = 1.2
        else:  # Night/early morning
            scale_factor = 0.5

        base_instances = max(1, metrics.current_instances)
        predicted_instances = int(base_instances * scale_factor)

        # Consider geological event patterns (simplified)
        if metrics.service_name == "volcano-watchtower":
            # Scale up if high seismic activity expected
            predicted_instances = max(predicted_instances, 3)

        use_preemptible = self._should_use_preemptible(metrics, predicted_instances)

        return ScalingDecision(
            service_name=metrics.service_name,
            region=metrics.region,
            target_instances=min(predicted_instances, 50),
            max_instances=50,
            cpu_limit="4000m" if config.get("cpu_intensive") else "2000m",
            memory_limit="8Gi" if config.get("memory_intensive") else "4Gi",
            concurrency=config.get("max_concurrency", 50),
            use_preemptible=use_preemptible,
            reasoning=f"Predictive scaling: Hour={current_hour}, Factor={scale_factor}",
            confidence=0.7,
        )

    def _hybrid_scaling(self, metrics: ServiceMetrics, config: Dict) -> ScalingDecision:
        """Hybrid scaling combining reactive and predictive approaches"""
        reactive_decision = self._reactive_scaling(metrics, config)
        predictive_decision = self._predictive_scaling(metrics, config)

        # Weighted combination
        reactive_weight = 0.6
        predictive_weight = 0.4

        target_instances = int(
            reactive_decision.target_instances * reactive_weight
            + predictive_decision.target_instances * predictive_weight
        )

        return ScalingDecision(
            service_name=metrics.service_name,
            region=metrics.region,
            target_instances=target_instances,
            max_instances=max(
                reactive_decision.max_instances, predictive_decision.max_instances
            ),
            cpu_limit=predictive_decision.cpu_limit,  # Use higher resource limits
            memory_limit=predictive_decision.memory_limit,
            concurrency=config.get("max_concurrency", 75),
            use_preemptible=self._should_use_preemptible(metrics, target_instances),
            reasoning=f"Hybrid: Reactive({reactive_decision.target_instances}) + Predictive({predictive_decision.target_instances})",
            confidence=0.85,
        )

    def _cost_optimized_scaling(
        self, metrics: ServiceMetrics, config: Dict
    ) -> ScalingDecision:
        """Cost-optimized scaling prioritizing cost over performance"""
        # Minimal scaling approach
        current_instances = metrics.current_instances

        # Only scale up if absolutely necessary
        if metrics.cpu_utilization > 0.9 or metrics.memory_utilization > 0.9:
            target_instances = current_instances + 1
        elif metrics.cpu_utilization < 0.3 and metrics.memory_utilization < 0.3:
            target_instances = max(
                0, current_instances - 1
            )  # Scale to zero if possible
        else:
            target_instances = current_instances

        return ScalingDecision(
            service_name=metrics.service_name,
            region=metrics.region,
            target_instances=target_instances,
            max_instances=10,  # Low maximum for cost control
            cpu_limit="1000m",  # Minimal resources
            memory_limit="2Gi",
            concurrency=config.get(
                "max_concurrency", 200
            ),  # High concurrency to reduce instances
            use_preemptible=True,  # Always use preemptible for cost savings
            reasoning=f"Cost-optimized: Minimal scaling, CPU={metrics.cpu_utilization:.2f}",
            confidence=0.9,
        )

    def _should_use_preemptible(
        self, metrics: ServiceMetrics, target_instances: int
    ) -> bool:
        """Determine if preemptible instances should be used"""
        if not self.cost_config.use_spot_instances:
            return False

        # Don't use preemptible for critical real-time services during high activity
        if metrics.service_name == "volcano-watchtower" and metrics.request_rate > 100:
            return False

        # Use preemptible for batch processing and non-critical workloads
        if metrics.service_name in ["data-ingestion", "helios-core"]:
            return True

        # Use preemptible for a percentage of instances
        preemptible_count = int(
            target_instances * self.cost_config.preemptible_percentage
        )
        return preemptible_count > 0

    async def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply the scaling decision to the Cloud Run service"""
        try:
            service_path = f"projects/{self.project_id}/locations/{decision.region}/services/{decision.service_name}"

            # Get current service configuration
            service = self.client.get_service(name=service_path)

            # Update service configuration
            service.spec.template.spec.container_concurrency = decision.concurrency
            service.spec.template.metadata.annotations.update(
                {
                    "autoscaling.knative.dev/maxScale": str(decision.max_instances),
                    "autoscaling.knative.dev/minScale": (
                        "0" if decision.target_instances == 0 else "1"
                    ),
                    "run.googleapis.com/cpu-throttling": (
                        "false" if decision.service_name == "helios-core" else "true"
                    ),
                    "run.googleapis.com/execution-environment": "gen2",
                }
            )

            # Set resource limits
            if service.spec.template.spec.template.spec.containers:
                container = service.spec.template.spec.template.spec.containers[0]
                container.resources.limits = {
                    "cpu": decision.cpu_limit,
                    "memory": decision.memory_limit,
                }

            # Apply preemptible configuration if needed
            if decision.use_preemptible:
                service.spec.template.metadata.annotations[
                    "run.googleapis.com/execution-environment"
                ] = "gen1"
                service.spec.template.metadata.annotations[
                    "run.googleapis.com/cpu-throttling"
                ] = "true"

            # Update the service
            operation = self.client.update_service(service=service)

            self.logger.info(
                f"Applied scaling decision for {decision.service_name} in {decision.region}: "
                f"instances={decision.target_instances}, reasoning={decision.reasoning}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error applying scaling decision: {e}")
            return False

    async def optimize_all_services(self) -> Dict[str, List[ScalingDecision]]:
        """Optimize scaling for all services across all regions"""
        decisions = {}

        for service_name in self.service_configs.keys():
            decisions[service_name] = []

            for region in self.regions:
                metrics = await self.get_service_metrics(service_name, region)
                if metrics:
                    decision = self.calculate_scaling_decision(metrics)
                    success = await self.apply_scaling_decision(decision)

                    if success:
                        decisions[service_name].append(decision)

        return decisions

    async def monitor_and_optimize(self, interval_seconds: int = 60):
        """Continuous monitoring and optimization loop"""
        self.logger.info(
            f"Starting continuous optimization with {interval_seconds}s interval"
        )

        while True:
            try:
                decisions = await self.optimize_all_services()

                # Log optimization results
                total_decisions = sum(
                    len(service_decisions) for service_decisions in decisions.values()
                )
                self.logger.info(
                    f"Optimization cycle completed: {total_decisions} scaling decisions applied"
                )

                # Wait for next cycle
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in optimization cycle: {e}")
                await asyncio.sleep(interval_seconds)


async def main():
    """Main function for running the Cloud Run optimizer"""
    logging.basicConfig(level=logging.INFO)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "prometheus-ai-platform")
    regions = ["us-central1", "europe-west1", "asia-northeast1"]

    optimizer = CloudRunOptimizer(project_id, regions)

    # Run continuous optimization
    await optimizer.monitor_and_optimize(interval_seconds=30)


if __name__ == "__main__":
    asyncio.run(main())
