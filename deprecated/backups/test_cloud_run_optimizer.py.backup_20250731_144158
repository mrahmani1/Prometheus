#!/usr/bin/env python3
"""
Tests for Cloud Run Optimizer

This module contains comprehensive tests for the Cloud Run serverless scaling
and cost optimization functionality.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud_run_optimizer import (
    CloudRunOptimizer,
    ScalingStrategy,
    ServiceMetrics,
    ScalingDecision,
    CostOptimization
)
from multi_region_deployment import (
    MultiRegionDeploymentManager,
    DeploymentStrategy,
    TrafficRoutingStrategy,
    RegionConfig,
    ServiceDeployment,
    DeploymentResult
)
from cost_optimization import (
    CloudRunCostOptimizer,
    CostOptimizationStrategy,
    CostMetrics,
    OptimizationRecommendation,
    ResourceOptimization
)


class TestCloudRunOptimizer:
    """Test cases for CloudRunOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create a CloudRunOptimizer instance for testing"""
        return CloudRunOptimizer(
            project_id="test-project",
            regions=["us-central1", "europe-west1"]
        )
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample service metrics for testing"""
        return ServiceMetrics(
            service_name="helios-core",
            region="us-central1",
            current_instances=5,
            cpu_utilization=0.75,
            memory_utilization=0.60,
            request_rate=100.0,
            response_latency=2.5,
            error_rate=0.02,
            cold_starts=3,
            timestamp=datetime.utcnow()
        )
    
    def test_reactive_scaling_decision(self, optimizer, sample_metrics):
        """Test reactive scaling decision calculation"""
        decision = optimizer._reactive_scaling(
            sample_metrics, 
            optimizer.service_configs["helios-core"]
        )
        
        assert isinstance(decision, ScalingDecision)
        assert decision.service_name == "helios-core"
        assert decision.region == "us-central1"
        assert decision.target_instances > 0
        assert decision.confidence == 0.8
        assert "Reactive scaling" in decision.reasoning
    
    def test_predictive_scaling_decision(self, optimizer, sample_metrics):
        """Test predictive scaling decision calculation"""
        decision = optimizer._predictive_scaling(
            sample_metrics,
            optimizer.service_configs["helios-core"]
        )
        
        assert isinstance(decision, ScalingDecision)
        assert decision.service_name == "helios-core"
        assert decision.confidence == 0.7
        assert "Predictive scaling" in decision.reasoning
    
    def test_hybrid_scaling_decision(self, optimizer, sample_metrics):
        """Test hybrid scaling decision calculation"""
        decision = optimizer._hybrid_scaling(
            sample_metrics,
            optimizer.service_configs["helios-core"]
        )
        
        assert isinstance(decision, ScalingDecision)
        assert decision.confidence == 0.85
        assert "Hybrid" in decision.reasoning
    
    def test_cost_optimized_scaling_decision(self, optimizer, sample_metrics):
        """Test cost-optimized scaling decision calculation"""
        decision = optimizer._cost_optimized_scaling(
            sample_metrics,
            optimizer.service_configs["helios-core"]
        )
        
        assert isinstance(decision, ScalingDecision)
        assert decision.use_preemptible == True
        assert decision.max_instances <= 10  # Low maximum for cost control
        assert "Cost-optimized" in decision.reasoning
    
    def test_preemptible_instance_decision(self, optimizer, sample_metrics):
        """Test preemptible instance usage decision"""
        # Test for batch processing service
        sample_metrics.service_name = "data-ingestion"
        should_use = optimizer._should_use_preemptible(sample_metrics, 5)
        assert should_use == True
        
        # Test for critical real-time service with high activity
        sample_metrics.service_name = "volcano-watchtower"
        sample_metrics.request_rate = 150
        should_use = optimizer._should_use_preemptible(sample_metrics, 5)
        assert should_use == False
    
    @pytest.mark.asyncio
    async def test_get_service_metrics(self, optimizer):
        """Test service metrics retrieval"""
        with patch.object(optimizer, '_fetch_monitoring_metrics') as mock_fetch:
            mock_fetch.return_value = {
                "instance_count": 3,
                "cpu_utilization": 0.65,
                "memory_utilization": 0.45,
                "request_rate": 75.0
            }
            
            metrics = await optimizer.get_service_metrics("api-gateway", "us-central1")
            
            assert metrics is not None
            assert metrics.service_name == "api-gateway"
            assert metrics.region == "us-central1"
            assert metrics.current_instances == 3
            assert metrics.cpu_utilization == 0.65
    
    @pytest.mark.asyncio
    async def test_apply_scaling_decision(self, optimizer):
        """Test applying scaling decision to Cloud Run service"""
        decision = ScalingDecision(
            service_name="test-service",
            region="us-central1",
            target_instances=3,
            max_instances=10,
            cpu_limit="2000m",
            memory_limit="4Gi",
            concurrency=100,
            use_preemptible=True,
            reasoning="Test scaling",
            confidence=0.9
        )
        
        with patch.object(optimizer.client, 'get_service') as mock_get, \
             patch.object(optimizer.client, 'update_service') as mock_update:
            
            mock_service = Mock()
            mock_service.spec.template.spec.container_concurrency = 50
            mock_service.spec.template.metadata.annotations = {}
            mock_get.return_value = mock_service
            
            mock_operation = Mock()
            mock_operation.result.return_value = Mock()
            mock_update.return_value = mock_operation
            
            result = await optimizer.apply_scaling_decision(decision)
            
            assert result == True
            mock_update.assert_called_once()


class TestMultiRegionDeploymentManager:
    """Test cases for MultiRegionDeploymentManager"""
    
    @pytest.fixture
    def deployment_manager(self):
        """Create a MultiRegionDeploymentManager instance for testing"""
        return MultiRegionDeploymentManager(project_id="test-project")
    
    @pytest.fixture
    def sample_deployment(self, deployment_manager):
        """Create a sample service deployment configuration"""
        return deployment_manager.service_deployments["helios-core"]
    
    def test_resource_config_generation(self, deployment_manager):
        """Test resource configuration generation for different services"""
        helios_config = deployment_manager._get_resource_config("helios-core")
        assert helios_config["cpu_limit"] == "4000m"
        assert helios_config["memory_limit"] == "8Gi"
        assert helios_config["concurrency"] == 10
        
        api_config = deployment_manager._get_resource_config("api-gateway")
        assert api_config["cpu_limit"] == "2000m"
        assert api_config["concurrency"] == 1000
    
    @pytest.mark.asyncio
    async def test_deploy_service_to_region(self, deployment_manager, sample_deployment):
        """Test deploying a service to a specific region"""
        region = deployment_manager.region_configs["us-central1"]
        
        with patch.object(deployment_manager.run_client, 'get_service') as mock_get, \
             patch.object(deployment_manager.run_client, 'update_service') as mock_update:
            
            # Mock service not found (new deployment)
            mock_get.side_effect = Exception("Service not found")
            
            mock_operation = Mock()
            mock_result = Mock()
            mock_result.status.url = "https://test-service-12345.run.app"
            mock_operation.result.return_value = mock_result
            
            with patch.object(deployment_manager.run_client, 'create_service') as mock_create:
                mock_create.return_value = mock_operation
                
                result = await deployment_manager.deploy_service_to_region(
                    sample_deployment, region
                )
                
                assert isinstance(result, DeploymentResult)
                assert result.success == True
                assert result.service_name == "helios-core"
                assert result.region == "us-central1"
                assert "https://" in result.url
    
    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, deployment_manager, sample_deployment):
        """Test blue-green deployment strategy"""
        with patch.object(deployment_manager, 'deploy_service_to_region') as mock_deploy:
            mock_deploy.return_value = DeploymentResult(
                service_name="helios-core",
                region="us-central1",
                success=True,
                revision_name="helios-core-20231201-120000-green",
                url="https://test-service.run.app",
                deployment_time=datetime.utcnow()
            )
            
            results = await deployment_manager._blue_green_deployment(sample_deployment)
            
            assert len(results) == len(sample_deployment.regions)
            assert all(result.success for result in results)
            assert all("green" in result.revision_name for result in results)
    
    @pytest.mark.asyncio
    async def test_canary_deployment(self, deployment_manager, sample_deployment):
        """Test canary deployment strategy"""
        with patch.object(deployment_manager, 'deploy_service_to_region') as mock_deploy:
            mock_deploy.return_value = DeploymentResult(
                service_name="helios-core",
                region="us-central1",
                success=True,
                revision_name="helios-core-20231201-120000",
                url="https://test-service.run.app",
                deployment_time=datetime.utcnow()
            )
            
            # Mock asyncio.sleep to speed up test
            with patch('asyncio.sleep'):
                results = await deployment_manager._canary_deployment(sample_deployment)
                
                assert len(results) == len(sample_deployment.regions)
                # First deployment should be to primary region (canary)
                assert results[0].region == "us-central1"  # Primary region
    
    def test_deployment_report_generation(self, deployment_manager):
        """Test deployment report generation"""
        sample_results = {
            "helios-core": [
                DeploymentResult(
                    service_name="helios-core",
                    region="us-central1",
                    success=True,
                    revision_name="helios-core-123",
                    url="https://helios-123.run.app",
                    deployment_time=datetime.utcnow()
                ),
                DeploymentResult(
                    service_name="helios-core",
                    region="europe-west1",
                    success=False,
                    revision_name="",
                    url="",
                    error_message="Deployment failed",
                    deployment_time=datetime.utcnow()
                )
            ]
        }
        
        report = deployment_manager.generate_deployment_report(sample_results)
        
        assert "Multi-Region Deployment Report" in report
        assert "helios-core" in report
        assert "✅ SUCCESS" in report
        assert "❌ FAILED" in report
        assert "Success Rate: 50.0%" in report


class TestCloudRunCostOptimizer:
    """Test cases for CloudRunCostOptimizer"""
    
    @pytest.fixture
    def cost_optimizer(self):
        """Create a CloudRunCostOptimizer instance for testing"""
        return CloudRunCostOptimizer(
            project_id="test-project",
            billing_account_id="test-billing-account"
        )
    
    @pytest.fixture
    def sample_cost_metrics(self):
        """Create sample cost metrics for testing"""
        return CostMetrics(
            service_name="helios-core",
            region="us-central1",
            hourly_cost=5.50,
            monthly_projection=3960.0,
            cpu_cost=3.20,
            memory_cost=1.80,
            request_cost=0.50,
            instance_hours=24.0,
            preemptible_savings=1.20,
            timestamp=datetime.utcnow()
        )
    
    def test_optimization_recommendations_balanced(self, cost_optimizer, sample_cost_metrics):
        """Test cost optimization recommendations with balanced strategy"""
        recommendation = cost_optimizer.generate_optimization_recommendations(
            sample_cost_metrics,
            CostOptimizationStrategy.BALANCED
        )
        
        assert isinstance(recommendation, OptimizationRecommendation)
        assert recommendation.service_name == "helios-core"
        assert recommendation.region == "us-central1"
        assert recommendation.current_cost == 5.50
        assert recommendation.optimized_cost < recommendation.current_cost
        assert len(recommendation.recommendations) > 0
    
    def test_optimization_recommendations_aggressive(self, cost_optimizer, sample_cost_metrics):
        """Test cost optimization recommendations with aggressive strategy"""
        recommendation = cost_optimizer.generate_optimization_recommendations(
            sample_cost_metrics,
            CostOptimizationStrategy.AGGRESSIVE
        )
        
        assert recommendation.savings_percentage > 0
        assert any("preemptible" in rec.lower() for rec in recommendation.recommendations)
    
    def test_optimal_resource_calculation(self, cost_optimizer, sample_cost_metrics):
        """Test optimal resource configuration calculation"""
        recommendation = cost_optimizer.generate_optimization_recommendations(
            sample_cost_metrics,
            CostOptimizationStrategy.BALANCED
        )
        
        optimal_resources = cost_optimizer.calculate_optimal_resources(
            sample_cost_metrics,
            recommendation
        )
        
        assert isinstance(optimal_resources, ResourceOptimization)
        assert optimal_resources.cpu_limit in ["1000m", "2000m", "4000m"]
        assert optimal_resources.memory_limit in ["2Gi", "4Gi", "8Gi"]
        assert optimal_resources.concurrency > 0
        assert optimal_resources.max_instances > 0
    
    def test_regional_cost_factors(self, cost_optimizer):
        """Test regional cost factor calculations"""
        us_central_factor = cost_optimizer.regional_cost_factors["us-central1"]
        europe_factor = cost_optimizer.regional_cost_factors["europe-west1"]
        asia_factor = cost_optimizer.regional_cost_factors["asia-northeast1"]
        
        assert us_central_factor == 1.0  # Base pricing
        assert europe_factor > us_central_factor  # More expensive
        assert asia_factor > europe_factor  # Most expensive
    
    @pytest.mark.asyncio
    async def test_analyze_service_costs(self, cost_optimizer):
        """Test service cost analysis"""
        with patch.object(cost_optimizer, '_get_usage_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "cpu_hours": 48.0,
                "memory_gb_hours": 192.0,
                "total_requests": 50000,
                "instance_hours": 24.0,
                "preemptible_hours": 12.0
            }
            
            cost_metrics = await cost_optimizer.analyze_service_costs(
                "helios-core", "us-central1", days=7
            )
            
            assert cost_metrics is not None
            assert cost_metrics.service_name == "helios-core"
            assert cost_metrics.region == "us-central1"
            assert cost_metrics.hourly_cost > 0
            assert cost_metrics.monthly_projection > 0
    
    def test_cost_report_generation(self, cost_optimizer):
        """Test cost optimization report generation"""
        sample_results = {
            "helios-core": [
                OptimizationRecommendation(
                    service_name="helios-core",
                    region="us-central1",
                    current_cost=5.50,
                    optimized_cost=4.20,
                    savings_percentage=23.6,
                    recommendations=["Use more preemptible instances"],
                    risk_level="low",
                    implementation_priority=2
                )
            ]
        }
        
        report = cost_optimizer.generate_cost_report(sample_results)
        
        assert "Cloud Run Cost Optimization Report" in report
        assert "helios-core" in report
        assert "23.6%" in report
        assert "Summary" in report
        assert "Total Monthly Savings" in report


class TestIntegration:
    """Integration tests for the complete Cloud Run serverless scaling system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self):
        """Test the complete optimization workflow from metrics to deployment"""
        # This would be a comprehensive integration test
        # that tests the entire workflow from getting metrics
        # to applying optimizations
        
        project_id = "test-project"
        regions = ["us-central1", "europe-west1"]
        
        # Initialize components
        optimizer = CloudRunOptimizer(project_id, regions)
        deployment_manager = MultiRegionDeploymentManager(project_id)
        cost_optimizer = CloudRunCostOptimizer(project_id, "test-billing")
        
        # Mock external dependencies
        with patch.object(optimizer, 'get_service_metrics') as mock_metrics, \
             patch.object(optimizer, 'apply_scaling_decision') as mock_apply, \
             patch.object(deployment_manager, 'deploy_service_to_region') as mock_deploy, \
             patch.object(cost_optimizer, 'analyze_service_costs') as mock_cost_analysis:
            
            # Setup mocks
            mock_metrics.return_value = ServiceMetrics(
                service_name="helios-core",
                region="us-central1",
                current_instances=3,
                cpu_utilization=0.70,
                memory_utilization=0.60,
                request_rate=80.0,
                response_latency=2.0,
                error_rate=0.01,
                cold_starts=2,
                timestamp=datetime.utcnow()
            )
            
            mock_apply.return_value = True
            
            mock_deploy.return_value = DeploymentResult(
                service_name="helios-core",
                region="us-central1",
                success=True,
                revision_name="helios-core-optimized",
                url="https://optimized-service.run.app",
                deployment_time=datetime.utcnow()
            )
            
            mock_cost_analysis.return_value = CostMetrics(
                service_name="helios-core",
                region="us-central1",
                hourly_cost=4.50,
                monthly_projection=3240.0,
                cpu_cost=2.70,
                memory_cost=1.50,
                request_cost=0.30,
                instance_hours=20.0,
                preemptible_savings=0.90,
                timestamp=datetime.utcnow()
            )
            
            # Run optimization workflow
            scaling_decisions = await optimizer.optimize_all_services()
            cost_recommendations = await cost_optimizer.optimize_all_services()
            
            # Verify results
            assert len(scaling_decisions) > 0
            assert len(cost_recommendations) > 0
            
            # Verify that optimization decisions were applied
            mock_apply.assert_called()
            
            # Verify cost analysis was performed
            mock_cost_analysis.assert_called()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])