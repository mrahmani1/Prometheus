"""
Model Management Service Demo
Demonstrates the key features of the Model Management Service
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any

class ModelManagementDemo:
    """Demo client for Model Management Service"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def train_model(self, model_type: str = "eruption") -> Dict[str, Any]:
        """Train a new model"""
        training_request = {
            "model_type": model_type,
            "force_retrain": True,
            "cloud_providers": ["local"],
            "experiment_name": f"demo_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "hyperparameter_tuning": True,
            "cross_validation_folds": 3
        }
        
        async with self.session.post(
            f"{self.base_url}/models/train", 
            json=training_request
        ) as response:
            return await response.json()
    
    async def deploy_model(self, model_id: str) -> Dict[str, Any]:
        """Deploy a trained model"""
        deployment_request = {
            "model_id": model_id,
            "deployment_strategy": "rolling",
            "target_environment": "staging",
            "cloud_providers": ["local"],
            "traffic_percentage": 100.0,
            "rollback_threshold": 0.05,
            "monitoring_duration_minutes": 5
        }
        
        async with self.session.post(
            f"{self.base_url}/models/deploy",
            json=deployment_request
        ) as response:
            return await response.json()
    
    async def make_prediction(self, model_type: str = "eruption") -> Dict[str, Any]:
        """Make a prediction using a model"""
        # Sample volcanic monitoring features
        features = [
            2.5,    # Seismic activity (magnitude)
            15.2,   # Ground deformation (mm)
            450.0,  # SO2 emissions (tons/day)
            1.8,    # Thermal anomaly (°C above baseline)
            0.3,    # GPS displacement (cm)
            25.0,   # Number of earthquakes per day
            3.2,    # Dominant frequency (Hz)
            850.0,  # Temperature (°C)
            0.15,   # Tilt change (microradians)
            2.1     # Gas ratio (CO2/SO2)
        ]
        
        prediction_request = {
            "model_type": model_type,
            "features": features,
            "uncertainty_estimation": True,
            "n_samples": 50,
            "return_probabilities": True,
            "explain_prediction": True
        }
        
        async with self.session.post(
            f"{self.base_url}/models/predict",
            json=prediction_request
        ) as response:
            return await response.json()
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        async with self.session.get(
            f"{self.base_url}/models/{model_id}/performance"
        ) as response:
            return await response.json()
    
    async def list_models(self) -> Dict[str, Any]:
        """List all available models"""
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def list_training_jobs(self) -> Dict[str, Any]:
        """List all training jobs"""
        async with self.session.get(f"{self.base_url}/training-jobs") as response:
            return await response.json()
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        async with self.session.get(f"{self.base_url}/metrics") as response:
            return await response.json()

async def run_demo():
    """Run the complete demo"""
    print("🌋 Model Management Service Demo")
    print("=" * 50)
    
    async with ModelManagementDemo() as demo:
        try:
            # 1. Check service health
            print("\n1. Checking service health...")
            health = await demo.check_health()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Components: {health.get('components', {})}")
            print(f"   Model count: {health.get('model_count', 0)}")
            
            # 2. List existing models
            print("\n2. Listing existing models...")
            models = await demo.list_models()
            if models.get('models'):
                for model in models['models']:
                    print(f"   - {model.get('model_type', 'unknown')} "
                          f"(v{model.get('version', '?')}) - "
                          f"{model.get('deployment_status', 'unknown')}")
            else:
                print("   No models found")
            
            # 3. Train a new model
            print("\n3. Training new eruption prediction model...")
            training_result = await demo.train_model("eruption")
            job_id = training_result.get('job_id')
            print(f"   Training job started: {job_id}")
            print(f"   Status: {training_result.get('status')}")
            
            # Wait a bit for training to process
            print("   Waiting for training to process...")
            await asyncio.sleep(2)
            
            # 4. Check training jobs
            print("\n4. Checking training jobs...")
            jobs = await demo.list_training_jobs()
            if jobs.get('training_jobs'):
                for job in jobs['training_jobs']:
                    print(f"   - Job {job.get('job_id', 'unknown')}: "
                          f"{job.get('status', 'unknown')} "
                          f"({job.get('model_type', 'unknown')})")
            
            # 5. Make predictions (assuming we have a model)
            print("\n5. Making sample predictions...")
            try:
                prediction = await demo.make_prediction("eruption")
                print(f"   Prediction: {prediction.get('prediction')}")
                if 'uncertainty' in prediction:
                    print(f"   Uncertainty: {prediction.get('uncertainty')}")
                if 'confidence' in prediction:
                    print(f"   Confidence: {prediction.get('confidence'):.3f}")
                print(f"   Prediction time: {prediction.get('prediction_time_ms', 0):.2f}ms")
            except Exception as e:
                print(f"   Prediction failed (model may not be ready): {e}")
            
            # 6. Get system metrics
            print("\n6. Getting system metrics...")
            try:
                metrics = await demo.get_system_metrics()
                print(f"   Timestamp: {metrics.get('timestamp')}")
                print(f"   Model count: {metrics.get('model_count', 0)}")
                print(f"   Active training jobs: {metrics.get('active_training_jobs', 0)}")
                print(f"   Active deployments: {metrics.get('active_deployments', 0)}")
            except Exception as e:
                print(f"   Metrics retrieval failed: {e}")
            
            # 7. Demonstrate batch prediction
            print("\n7. Demonstrating batch prediction...")
            try:
                # Create batch of features for multiple volcanoes
                batch_features = [
                    [2.5, 15.2, 450.0, 1.8, 0.3, 25.0, 3.2, 850.0, 0.15, 2.1],  # High risk
                    [1.2, 5.1, 120.0, 0.5, 0.1, 8.0, 2.8, 650.0, 0.05, 1.5],    # Medium risk
                    [0.8, 2.3, 50.0, 0.2, 0.05, 3.0, 2.1, 400.0, 0.02, 1.1]     # Low risk
                ]
                
                batch_request = {
                    "model_type": "eruption",
                    "features_batch": batch_features,
                    "uncertainty_estimation": True,
                    "batch_size": 3
                }
                
                async with demo.session.post(
                    f"{demo.base_url}/models/predict/batch",
                    json=batch_request
                ) as response:
                    if response.status == 200:
                        batch_result = await response.json()
                        print(f"   Batch predictions: {batch_result.get('predictions')}")
                        print(f"   Batch size: {batch_result.get('batch_size')}")
                        print(f"   Total time: {batch_result.get('total_prediction_time_ms', 0):.2f}ms")
                    else:
                        print("   Batch prediction endpoint not available")
            except Exception as e:
                print(f"   Batch prediction failed: {e}")
            
        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
            print("   Make sure the Model Management Service is running on http://localhost:8003")
        except Exception as e:
            print(f"❌ Demo error: {e}")
    
    print("\n✅ Demo completed!")
    print("\nNext steps:")
    print("- Check the service logs for detailed information")
    print("- Use the API documentation at http://localhost:8003/docs")
    print("- Monitor model performance and drift detection")
    print("- Set up automated retraining pipelines")

def run_sync_demo():
    """Synchronous wrapper for the demo"""
    asyncio.run(run_demo())

if __name__ == "__main__":
    run_sync_demo()